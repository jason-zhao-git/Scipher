"""Resume Gene Transformer training from a checkpoint.

Loads model weights, optimizer state, scheduler state, and loss history
from a checkpoint and continues training from the next epoch.

Usage:
    uv run python scripts/resume_training.py --checkpoint data/checkpoint/transformer_2026-04-15_2026-01-29_CL0000988/epoch01.pt
    uv run python scripts/resume_training.py --checkpoint data/checkpoint/transformer_2026-04-15_2026-01-29_CL0000988/epoch01.pt --epochs 6

With nohup:
    nohup uv run python scripts/resume_training.py --checkpoint <path> > resume.log 2>&1 &
"""

import argparse
import sys
import logging
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score

import tiledbsoma as soma
from tiledbsoma_ml import ExperimentDataset, experiment_dataloader
from attr import evolve

# Find project root
_cwd = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _cwd
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

from scipher.hierarchy import (
    load_prebuilt_hierarchy, MarginalizationLoss, WideNN,
)
from scipher.embedders.gene_transformer import GeneTransformerEmbedder


# ============================================================
# Config — only things NOT in the checkpoint
# ============================================================
SOMA_URI = "/scratch/sigbio_project_root/sigbio_project25/jingqiao/mccell-single/soma_db_homo_sapiens"
MIN_CELL_COUNT = 50
SEED = 42
NUM_WORKERS = 3
GRAD_CLIP = 1.0


# ============================================================
# Model definition (must match the one used during training)
# ============================================================
class ScipherModel(nn.Module):
    def __init__(self, embed_dim, num_leaves, gene_embs,
                 d_model=512, n_layers=4, n_heads=8, n_cls=8,
                 d_ff=2048, output_dim=256, dropout=0.1):
        super().__init__()
        self.embedder = GeneTransformerEmbedder(
            gene_embed_dim=embed_dim, d_model=d_model, output_dim=output_dim,
            n_layers=n_layers, n_heads=n_heads, n_cls=n_cls, d_ff=d_ff,
            dropout=dropout,
        )
        self.classifier = WideNN(input_dim=output_dim, output_dim=num_leaves)
        self.register_buffer("gene_embs", gene_embs)

    def forward(self, expression, mask):
        cell_embedding, attn_weights = self.embedder(
            self.gene_embs, expression, mask,
        )
        logits = self.classifier(cell_embedding)
        return logits, cell_embedding, attn_weights


def main():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Total epochs to train (default: use checkpoint config)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    # --- Load checkpoint ---
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    start_epoch = ckpt["epoch"]  # already completed this many epochs

    # Read all config from checkpoint
    HIERARCHY_DATE = cfg["hierarchy_date"]
    ROOT_CL_ID = cfg["root_cl_id"]
    BATCH_SIZE = cfg["batch_size"]
    LEAF_WEIGHT = cfg["leaf_weight"]
    D_MODEL = cfg["d_model"]
    N_LAYERS = cfg["n_layers"]
    N_HEADS = cfg["n_heads"]
    N_CLS = cfg["n_cls"]
    D_FF = cfg["d_ff"]
    OUTPUT_DIM = cfg["output_dim"]
    DROPOUT = cfg["dropout"]
    PEAK_LR = cfg.get("peak_lr", cfg.get("lr", 1e-4))
    MIN_LR = cfg.get("min_lr", 1e-5)
    LR_WARMUP_STEPS = cfg.get("lr_warmup_steps", 500)
    WEIGHT_DECAY = cfg.get("weight_decay", 0.01)
    EPOCHS = args.epochs if args.epochs is not None else cfg.get("epochs", 4)

    effective_batch_size = BATCH_SIZE * max(num_gpus, 1)

    # Checkpoints go in the same directory as the source checkpoint
    checkpoint_dir = ckpt_path.parent
    log_file = checkpoint_dir / "resume.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Resuming from: {ckpt_path}")
    logger.info(f"Completed epochs: {start_epoch}, continuing to epoch {EPOCHS}")
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPUs: {num_gpus}x {torch.cuda.get_device_name(0)}")
        logger.info(f"Batch size: {BATCH_SIZE}/GPU x {num_gpus} GPUs = {effective_batch_size} effective")

    if start_epoch >= EPOCHS:
        logger.info(f"Already completed {start_epoch}/{EPOCHS} epochs. Nothing to do.")
        logger.info(f"Pass --epochs N (N > {start_epoch}) to train further.")
        return

    # --- Load hierarchy ---
    (
        mapping_dict, leaf_values, internal_values,
        marginalization_df, parent_child_df, exclusion_df,
    ) = load_prebuilt_hierarchy(HIERARCHY_DATE, ROOT_CL_ID)

    all_cell_values = list(leaf_values) + list(internal_values)
    logger.info(f"Leaf types: {len(leaf_values)}, Internal: {len(internal_values)}")

    # --- Load gene embeddings ---
    EMB_PATH = DATA_DIR / "embeddings" / "gene_to_embedding.pkl"
    with open(EMB_PATH, "rb") as f:
        gene_to_embedding = pickle.load(f)

    embed_dim = next(iter(gene_to_embedding.values())).shape[0]
    logger.info(f"Gene embeddings: {len(gene_to_embedding):,} genes x {embed_dim}-dim")

    # --- BioMart mapping ---
    BIOMART_FILE = DATA_DIR / "raw" / "mart_export.txt"
    df_biomart = pd.read_csv(BIOMART_FILE)
    df_pc = df_biomart[df_biomart["Gene type"] == "protein_coding"].dropna(
        subset=["Gene stable ID", "Gene name"]
    )
    gene_list = df_pc["Gene stable ID"].unique().tolist()
    ensembl_to_symbol = (
        df_pc.drop_duplicates("Gene stable ID")
        .set_index("Gene stable ID")["Gene name"]
        .to_dict()
    )

    # --- SOMA streaming dataset ---
    experiment = soma.open(SOMA_URI, mode="r")

    obs_df = (
        experiment.obs.read(
            value_filter='assay == "10x 3\' v3" and is_primary_data == True',
            column_names=["soma_joinid", "cell_type_ontology_term_id", "cell_type"],
        )
        .concat()
        .to_pandas()
    )

    obs_df = obs_df[obs_df["cell_type_ontology_term_id"].isin(all_cell_values)].copy()
    type_counts = obs_df["cell_type_ontology_term_id"].value_counts()
    keep_types = type_counts[type_counts >= MIN_CELL_COUNT].index
    obs_df = obs_df[obs_df["cell_type_ontology_term_id"].isin(keep_types)].copy()
    logger.info(f"Filtered: {len(obs_df):,} cells, {obs_df['cell_type_ontology_term_id'].nunique()} types")

    cl_to_name = (
        obs_df.drop_duplicates("cell_type_ontology_term_id")
        .set_index("cell_type_ontology_term_id")["cell_type"]
        .to_dict()
    )

    joinids = obs_df["soma_joinid"].values
    var_value_filter = f"feature_id in {gene_list}"

    with experiment.axis_query(
        measurement_name="RNA",
        obs_query=soma.AxisQuery(coords=(joinids,)),
        var_query=soma.AxisQuery(value_filter=var_value_filter),
    ) as query:
        var_df = query.var(column_names=["feature_id", "feature_name"]).concat().to_pandas()
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["cell_type_ontology_term_id"],
            batch_size=effective_batch_size,
            shuffle=True,
            seed=SEED,
        )

    train_ds, val_ds = ds.random_split(0.8, 0.2, seed=SEED)
    val_ds = evolve(val_ds, shuffle=False)

    train_loader = experiment_dataloader(
        train_ds, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = experiment_dataloader(
        val_ds, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    logger.info(f"Train: ~{train_ds.shape[0]:,} batches, Val: ~{val_ds.shape[0]:,} batches")

    # --- Build gene embedding index ---
    col_indices = []
    gene_names_mapped = []
    seen_symbols = set()
    for pos, (_, row) in enumerate(var_df.iterrows()):
        ensembl_id = row["feature_id"]
        symbol = ensembl_to_symbol.get(ensembl_id, row["feature_name"])
        if symbol in gene_to_embedding and symbol not in seen_symbols:
            col_indices.append(pos)
            gene_names_mapped.append(symbol)
            seen_symbols.add(symbol)

    col_indices = np.array(col_indices)
    gene_embs_tensor = torch.stack(
        [torch.from_numpy(gene_to_embedding[g]) for g in gene_names_mapped]
    ).to(device)

    logger.info(f"Genes with embeddings: {len(col_indices):,}/{len(var_df):,}")

    idx_to_cl = {v: k for k, v in mapping_dict.items()}
    leaf_idx_set = {mapping_dict[cl] for cl in leaf_values}

    # --- Reconstruct model and load weights ---
    model = ScipherModel(
        embed_dim=embed_dim, num_leaves=len(leaf_values),
        gene_embs=gene_embs_tensor,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        n_cls=N_CLS, d_ff=D_FF, output_dim=OUTPUT_DIM, dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using DataParallel across {num_gpus} GPUs")

    loss_fn = MarginalizationLoss(
        marginalization_df, parent_child_df, exclusion_df,
        leaf_values, internal_values, mapping_dict,
        leaf_weight=LEAF_WEIGHT, device=device,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ScipherModel: {n_params:,} trainable parameters")

    # --- Restore optimizer and scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Restored optimizer state")

    steps_per_epoch = train_ds.shape[0]
    total_steps = EPOCHS * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < LR_WARMUP_STEPS:
            return current_step / max(LR_WARMUP_STEPS, 1)
        progress = (current_step - LR_WARMUP_STEPS) / max(total_steps - LR_WARMUP_STEPS, 1)
        return max(MIN_LR / PEAK_LR, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info("Restored scheduler state")

    # --- Restore loss history ---
    loss_history = ckpt.get("loss_history", {"total": [], "leaf": [], "parent": []})
    warmup_losses = ckpt.get("warmup_losses", [])
    epoch_times = ckpt.get("epoch_times", [])

    # Helper: process one batch
    def process_batch(X_batch, obs_batch):
        X = torch.from_numpy(X_batch) if isinstance(X_batch, np.ndarray) else torch.from_numpy(X_batch.toarray())
        X = X.float()
        X_mapped = X[:, col_indices]
        mask = (X_mapped > 0).to(device)
        X_log = torch.log1p(X_mapped)
        expr_sum = X_log.sum(dim=1, keepdim=True).clamp(min=1e-10)
        expression = (X_log / expr_sum).to(device)
        labels = obs_batch["cell_type_ontology_term_id"].values
        y_batch = torch.tensor(
            [mapping_dict[t] for t in labels], device=device, dtype=torch.long,
        )
        return expression, mask, y_batch

    # --- Resume training ---
    logger.info("=" * 70)
    logger.info(f"Resuming training from epoch {start_epoch + 1} to {EPOCHS}")
    logger.info(f"Steps/epoch: {steps_per_epoch:,}, total steps: {total_steps:,}")
    current_lr = scheduler.get_last_lr()[0]
    logger.info(f"Current LR: {current_lr:.2e}")
    logger.info(f"Saving checkpoints to: {checkpoint_dir}")
    logger.info("=" * 70)

    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_ds.set_epoch(epoch)
        epoch_start = time.time()
        epoch_losses = []

        for i, (X_batch, obs_batch) in enumerate(train_loader):
            expression, mask, y_batch = process_batch(X_batch, obs_batch)

            optimizer.zero_grad()
            logits, _, _ = model(expression, mask)
            total_loss, loss_leafs, loss_parents = loss_fn(logits, y_batch)
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            loss_history["total"].append(total_loss.item())
            loss_history["leaf"].append(loss_leafs.item())
            loss_history["parent"].append(loss_parents.item())
            epoch_losses.append(total_loss.item())
            global_step += 1

            if (i + 1) % 50 == 0:
                avg_recent = np.mean(epoch_losses[-50:])
                current_lr = scheduler.get_last_lr()[0]
                mem_alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                logger.info(
                    f"  [Epoch {epoch+1}/{EPOCHS}, Batch {i+1}] "
                    f"Loss: {total_loss.item():.4f} (avg: {avg_recent:.4f}) "
                    f"lr: {current_lr:.2e} GPU mem: {mem_alloc:.1f}GB"
                )

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = np.mean(epoch_losses)

        # --- Save checkpoint ---
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        ckpt_save = {
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss_history": {k: list(v) for k, v in loss_history.items()},
            "warmup_losses": warmup_losses,
            "epoch_times": list(epoch_times),
            "config": cfg,
        }
        save_path = checkpoint_dir / f"epoch{epoch+1:02d}.pt"
        tmp_path = checkpoint_dir / f"epoch{epoch+1:02d}.pt.tmp"
        torch.save(ckpt_save, tmp_path)
        tmp_path.rename(save_path)

        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} -- avg loss: {avg_loss:.4f}, "
            f"lr: {current_lr:.2e}, time: {epoch_time:.1f}s, saved: {save_path.name}"
        )

    # --- Validation ---
    logger.info("Running validation...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, obs_batch in val_loader:
            expression, mask, y_batch = process_batch(X_batch, obs_batch)

            logits, _, _ = model(expression, mask)
            preds = torch.argmax(logits, dim=1)

            is_leaf = torch.tensor(
                [y.item() in leaf_idx_set for y in y_batch], device=device,
            )
            if is_leaf.sum() > 0:
                all_preds.extend(preds[is_leaf].cpu().numpy())
                all_labels.extend(y_batch[is_leaf].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Leaf accuracy: {acc:.4f}")
    logger.info(f"  Macro F1:      {macro_f1:.4f}")
    logger.info(f"  Weighted F1:   {weighted_f1:.4f}")
    logger.info(f"  N val samples: {len(all_labels):,}")
    logger.info(f"  Epochs:        {start_epoch + 1} to {EPOCHS}")
    logger.info(f"  Total time:    {sum(epoch_times):.1f}s ({sum(epoch_times)/3600:.1f}h)")
    logger.info(f"  Checkpoints:   {checkpoint_dir}")
    logger.info(f"  Log file:      {log_file}")


if __name__ == "__main__":
    main()
