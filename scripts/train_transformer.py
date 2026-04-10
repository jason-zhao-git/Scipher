"""Gene Transformer training script.

Standalone script (not notebook) so training survives Jupyter disconnects.
All output logged to file + stdout. Run with:

    nohup uv run python scripts/train_transformer.py > train.log 2>&1 &
    tail -f train.log

Or directly:
    uv run python scripts/train_transformer.py
"""

import sys
import logging
import pickle
import time
from pathlib import Path
from datetime import datetime

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
# Config — edit these
# ============================================================
HIERARCHY_DATE = "2026-01-29"
ROOT_CL_ID = "CL:0000988"  # blood cells
SOMA_URI = "/scratch/sigbio_project_root/sigbio_project25/jingqiao/mccell-single/soma_db_homo_sapiens"
MIN_CELL_COUNT = 50
BATCH_SIZE = 64       # per GPU
LR = 1e-4
LEAF_WEIGHT = 7.0
GRAD_CLIP = 1.0
EPOCHS = 10
SEED = 42
NUM_WORKERS = 3

# Model — scaled down from 768 to 512 for safe memory margins (~20GB/GPU)
D_MODEL = 512
N_LAYERS = 4
N_HEADS = 8
N_CLS = 8
D_FF = 2048
OUTPUT_DIM = 512
DROPOUT = 0.1


# ============================================================
# Logging to file + stdout
# ============================================================
cl_folder = ROOT_CL_ID.replace(":", "")
run_date = datetime.now().strftime("%Y-%m-%d")
checkpoint_dir = DATA_DIR / "checkpoint" / f"transformer_{run_date}_{HIERARCHY_DATE}_{cl_folder}"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

log_file = checkpoint_dir / "train.log"
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


# ============================================================
# Model definition
# ============================================================
class ScipherModel(nn.Module):
    def __init__(self, embed_dim, num_leaves, gene_embs,
                 d_model=512, n_layers=4, n_heads=8, n_cls=8,
                 d_ff=2048, output_dim=512, dropout=0.1):
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


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    effective_batch_size = BATCH_SIZE * max(num_gpus, 1)

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"Log file: {log_file}")
    if torch.cuda.is_available():
        logger.info(f"GPUs: {num_gpus}x {torch.cuda.get_device_name(0)}")
        logger.info(f"Batch size: {BATCH_SIZE}/GPU x {num_gpus} GPUs = {effective_batch_size} effective")
    logger.info(f"GeneTransformer: d_model={D_MODEL}, n_layers={N_LAYERS}, "
                f"n_heads={N_HEADS}, n_cls={N_CLS}, d_ff={D_FF}")

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
    logger.info(f"Protein-coding Ensembl IDs: {len(gene_list):,}")

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
    logger.info(f"Total 10x v3 primary cells: {len(obs_df):,}")

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

    n_obs, n_vars = ds.shape
    logger.info(f"Streaming dataset: {n_obs:,} cells x {n_vars:,} genes")
    logger.info(f"Train: ~{train_ds.shape[0]:,}, Val: ~{val_ds.shape[0]:,}")

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
    logger.info(f"Gene embedding tensor: {gene_embs_tensor.shape}")

    idx_to_cl = {v: k for k, v in mapping_dict.items()}
    leaf_idx_set = {mapping_dict[cl] for cl in leaf_values}

    # --- Model ---
    model = ScipherModel(
        embed_dim=embed_dim, num_leaves=len(leaf_values),
        gene_embs=gene_embs_tensor,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        n_cls=N_CLS, d_ff=D_FF, output_dim=OUTPUT_DIM, dropout=DROPOUT,
    ).to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using DataParallel across {num_gpus} GPUs")

    loss_fn = MarginalizationLoss(
        marginalization_df, parent_child_df, exclusion_df,
        leaf_values, internal_values, mapping_dict,
        leaf_weight=LEAF_WEIGHT, device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ScipherModel: {n_params:,} trainable parameters")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU memory after model load: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    # --- Training ---
    loss_history = {"total": [], "leaf": [], "parent": []}
    epoch_times = []

    logger.info(f"Training: {EPOCHS} epochs")
    logger.info(f"Saving checkpoints to: {checkpoint_dir}")
    logger.info("-" * 70)

    for epoch in range(EPOCHS):
        model.train()
        train_ds.set_epoch(epoch)
        epoch_start = time.time()
        epoch_losses = []

        for i, (X_batch, obs_batch) in enumerate(train_loader):
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

            optimizer.zero_grad()
            logits, _, _ = model(expression, mask)
            total_loss, loss_leafs, loss_parents = loss_fn(logits, y_batch)
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            loss_history["total"].append(total_loss.item())
            loss_history["leaf"].append(loss_leafs.item())
            loss_history["parent"].append(loss_parents.item())
            epoch_losses.append(total_loss.item())

            if (i + 1) % 50 == 0:
                avg_recent = np.mean(epoch_losses[-50:])
                mem_alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                logger.info(
                    f"  [Epoch {epoch+1}/{EPOCHS}, Batch {i+1}] "
                    f"Loss: {total_loss.item():.4f} (avg: {avg_recent:.4f}) "
                    f"GPU mem: {mem_alloc:.1f}GB"
                )

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = np.mean(epoch_losses)

        # --- Save checkpoint ---
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": {k: list(v) for k, v in loss_history.items()},
            "epoch_times": list(epoch_times),
            "config": {
                "input_dim": embed_dim, "output_dim": OUTPUT_DIM,
                "run_date": run_date, "hierarchy_date": HIERARCHY_DATE,
                "root_cl_id": ROOT_CL_ID, "model_class": "ScipherModel",
                "embedder_class": "GeneTransformerEmbedder",
                "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
                "n_cls": N_CLS, "d_ff": D_FF, "dropout": DROPOUT,
                "lr": LR, "batch_size": BATCH_SIZE, "leaf_weight": LEAF_WEIGHT,
                "num_gpus": num_gpus,
            },
        }
        ckpt_path = checkpoint_dir / f"epoch{epoch+1:02d}.pt"
        torch.save(ckpt, ckpt_path)

        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} -- avg loss: {avg_loss:.4f}, "
            f"time: {epoch_time:.1f}s, saved: {ckpt_path.name}"
        )

    # --- Validation ---
    logger.info("Running validation...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, obs_batch in val_loader:
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
    logger.info(f"  Epochs:        {EPOCHS}")
    logger.info(f"  Total time:    {sum(epoch_times):.1f}s ({sum(epoch_times)/3600:.1f}h)")
    logger.info(f"  Checkpoints:   {checkpoint_dir}")
    logger.info(f"  Log file:      {log_file}")


if __name__ == "__main__":
    main()
