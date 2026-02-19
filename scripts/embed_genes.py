"""End-to-end gene embedding pipeline.

BioMart gene list → MyGene/UniProt sequence mapping → ESM-2 embeddings

Produces:
    data/embeddings/gene_to_embedding.pkl   - {gene_symbol: np.ndarray} dict
    data/embeddings/embedding_report.txt    - detailed stats on the full pipeline
    data/sequences/gene_to_sequence.pkl     - {gene_symbol: sequence} intermediate artifact

Usage:
    python scripts/embed_genes.py --biomart data/raw/mart_export.txt
    python scripts/embed_genes.py --biomart data/raw/mart_export.txt --model esm2_t30_150M_UR50D --batch-size 16
    python scripts/embed_genes.py --biomart data/raw/mart_export.txt --skip-embedding  # sequence mapping only
"""

import argparse
import logging
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Add project root to path so we can import scipher
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipher.preprocess.gene_list import get_protein_coding_genes
from scipher.preprocess.gene_mapping import map_genes_to_sequences

logger = logging.getLogger(__name__)

# ---- Output directories ----

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SEQ_DIR = DATA_DIR / "sequences"
EMB_DIR = DATA_DIR / "embeddings"
REPORT_DIR = DATA_DIR / "reports"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dirs():
    for d in [RAW_DIR, SEQ_DIR, EMB_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---- Step 1: Load genes from BioMart ----

def load_genes(biomart_file: Path):
    """Load protein-coding genes and collect BioMart-level stats."""
    gene_symbols, gene_descriptions = get_protein_coding_genes(biomart_file)

    import pandas as pd
    df = pd.read_csv(biomart_file)

    stats = {
        "biomart_total_rows": len(df),
        "biomart_gene_types": dict(df["Gene type"].value_counts()),
        "protein_coding_unique": len(gene_symbols),
        "with_descriptions": len(gene_descriptions),
    }

    return gene_symbols, gene_descriptions, stats


# ---- Step 2: Map genes to protein sequences ----

def map_sequences(gene_symbols, gene_descriptions, cache_path: Path = None):
    """Map gene symbols to amino acid sequences, with caching."""
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached gene-to-sequence mapping from {cache_path}")
        with open(cache_path, "rb") as f:
            gene_to_sequence = pickle.load(f)
        missing = [g for g in gene_symbols if g not in gene_to_sequence]
        return gene_to_sequence, missing

    gene_to_sequence, missing = map_genes_to_sequences(
        gene_symbols, gene_descriptions
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(gene_to_sequence, f)
        logger.info(f"Cached gene-to-sequence mapping to {cache_path}")

    return gene_to_sequence, missing


def analyze_sequences(gene_to_sequence, missing, max_length: int):
    """Compute detailed stats about the sequence mapping results."""
    regular = {k: v for k, v in gene_to_sequence.items() if isinstance(v, str)}
    readthrough = {k: v for k, v in gene_to_sequence.items() if isinstance(v, tuple)}

    # Sequence lengths
    regular_lengths = [len(seq) for seq in regular.values()]
    readthrough_total_lengths = [
        sum(len(s) for s in seqs) for seqs in readthrough.values()
    ]
    all_lengths = regular_lengths + readthrough_total_lengths

    # Truncation analysis
    truncated = [g for g, seq in regular.items() if len(seq) > max_length]
    truncated_lengths = [len(regular[g]) for g in truncated]

    # Length buckets
    buckets = [
        (0, 100, "tiny (<100 aa)"),
        (100, 500, "small (100-500 aa)"),
        (500, 1000, "medium (500-1000 aa)"),
        (1000, 1022, "large (1000-1022 aa)"),
        (1022, 2000, "truncated (1022-2000 aa)"),
        (2000, float("inf"), "heavily truncated (>2000 aa)"),
    ]
    bucket_counts = {}
    for lo, hi, label in buckets:
        bucket_counts[label] = sum(1 for l in regular_lengths if lo <= l < hi)

    # Missing gene patterns
    missing_patterns = Counter()
    for g in missing:
        if "-" in g and not g.startswith("HLA-"):
            missing_patterns["hyphenated (possible readthrough)"] += 1
        elif g.startswith("LOC"):
            missing_patterns["LOC* (uncharacterized)"] += 1
        elif g.startswith("LINC"):
            missing_patterns["LINC* (lncRNA mislabeled?)"] += 1
        elif any(c.isdigit() for c in g) and g.startswith(("C", "FAM")):
            missing_patterns["C*/FAM* (uncharacterized family)"] += 1
        else:
            missing_patterns["other"] += 1

    stats = {
        "n_regular": len(regular),
        "n_readthrough": len(readthrough),
        "n_total_mapped": len(gene_to_sequence),
        "n_missing": len(missing),
        "regular_lengths": regular_lengths,
        "all_lengths": all_lengths,
        "length_mean": np.mean(all_lengths) if all_lengths else 0,
        "length_median": np.median(all_lengths) if all_lengths else 0,
        "length_min": min(all_lengths) if all_lengths else 0,
        "length_max": max(all_lengths) if all_lengths else 0,
        "length_std": np.std(all_lengths) if all_lengths else 0,
        "n_truncated": len(truncated),
        "truncated_genes": truncated,
        "truncated_lengths": truncated_lengths,
        "length_buckets": bucket_counts,
        "missing_patterns": dict(missing_patterns),
        "missing_genes": missing,
    }
    return stats


# ---- Step 3: Generate ESM-2 embeddings ----

def generate_embeddings(
    gene_to_sequence, model_name, batch_size, cache_path, max_length
):
    """Run ESM-2 on all sequences."""
    from scipher.embedders.gene_embeddings.esm2 import ESM2Embedder

    embedder = ESM2Embedder(model_name=model_name, max_length=max_length)
    gene_to_embedding = embedder.generate_all_embeddings(
        gene_to_sequence,
        readthrough_strategy="concat",
        cache_path=cache_path,
        batch_size=batch_size,
    )
    return gene_to_embedding


# ---- Report ----

def write_report(
    biomart_stats, seq_stats, model_name, max_length,
    embedding_path, elapsed_sec, report_path
):
    """Write a detailed human-readable report."""
    lines = []
    w = lines.append

    w("=" * 72)
    w("Gene Embedding Pipeline Report")
    w("=" * 72)
    w("")

    # BioMart
    w("1. BioMart Gene Loading")
    w("-" * 40)
    w(f"  BioMart file rows:           {biomart_stats['biomart_total_rows']:,}")
    w(f"  Gene types in file:")
    for gtype, count in sorted(
        biomart_stats["biomart_gene_types"].items(), key=lambda x: -x[1]
    ):
        w(f"    {gtype:30s} {count:>7,}")
    w(f"  Protein-coding (unique):     {biomart_stats['protein_coding_unique']:,}")
    w(f"  With descriptions:           {biomart_stats['with_descriptions']:,}")
    w("")

    # Sequence mapping
    w("2. Gene-to-Protein Sequence Mapping")
    w("-" * 40)
    w(f"  Successfully mapped:         {seq_stats['n_total_mapped']:,}")
    w(f"    Regular genes:             {seq_stats['n_regular']:,}")
    w(f"    Readthrough (fusion):      {seq_stats['n_readthrough']:,}")
    w(f"  Failed to map:               {seq_stats['n_missing']:,}")
    success_rate = (
        100 * seq_stats["n_total_mapped"]
        / (seq_stats["n_total_mapped"] + seq_stats["n_missing"])
    )
    w(f"  Success rate:                {success_rate:.1f}%")
    w("")

    # Sequence lengths
    w("3. Protein Sequence Length Distribution")
    w("-" * 40)
    w(f"  Mean:                        {seq_stats['length_mean']:.0f} aa")
    w(f"  Median:                      {seq_stats['length_median']:.0f} aa")
    w(f"  Std:                         {seq_stats['length_std']:.0f} aa")
    w(f"  Min:                         {seq_stats['length_min']} aa")
    w(f"  Max:                         {seq_stats['length_max']} aa")
    w("")
    w(f"  Length buckets (regular genes only):")
    for label, count in seq_stats["length_buckets"].items():
        pct = 100 * count / seq_stats["n_regular"] if seq_stats["n_regular"] else 0
        w(f"    {label:35s} {count:>6,}  ({pct:5.1f}%)")
    w("")

    # Truncation
    w(f"4. Truncation Analysis (max_length={max_length})")
    w("-" * 40)
    w(f"  Sequences truncated:         {seq_stats['n_truncated']:,}")
    if seq_stats["n_regular"]:
        pct = 100 * seq_stats["n_truncated"] / seq_stats["n_regular"]
        w(f"  Truncation rate:             {pct:.1f}%")
    if seq_stats["truncated_genes"]:
        w(f"  Longest truncated sequences:")
        sorted_trunc = sorted(
            zip(seq_stats["truncated_genes"], seq_stats["truncated_lengths"]),
            key=lambda x: -x[1],
        )
        for gene, length in sorted_trunc[:20]:
            lost = length - max_length
            w(f"    {gene:20s} {length:>6,} aa  (losing {lost:,} aa, {100*lost/length:.0f}%)")
        if len(sorted_trunc) > 20:
            w(f"    ... and {len(sorted_trunc) - 20} more")
    w("")

    # Missing genes
    w("5. Unmapped Genes Analysis")
    w("-" * 40)
    if seq_stats["missing_patterns"]:
        w(f"  Failure pattern breakdown:")
        for pattern, count in sorted(
            seq_stats["missing_patterns"].items(), key=lambda x: -x[1]
        ):
            w(f"    {pattern:40s} {count:>5,}")
    w("")
    if seq_stats["missing_genes"]:
        w(f"  Full list of unmapped genes ({len(seq_stats['missing_genes'])}):")
        for g in sorted(seq_stats["missing_genes"]):
            w(f"    {g}")
    w("")

    # Embedding
    w("6. ESM-2 Embedding Generation")
    w("-" * 40)
    w(f"  Model:                       {model_name}")
    w(f"  Output:                      {embedding_path}")
    if elapsed_sec is not None:
        m, s = divmod(int(elapsed_sec), 60)
        h, m = divmod(m, 60)
        w(f"  Wall time:                   {h}h {m}m {s}s")
        genes_per_sec = seq_stats["n_total_mapped"] / elapsed_sec if elapsed_sec > 0 else 0
        w(f"  Throughput:                  {genes_per_sec:.1f} genes/sec")
    w("")
    w("=" * 72)

    report_text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Report written to {report_path}")

    # Also print to stdout
    print(report_text)


# ---- Main ----

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ESM-2 gene embeddings from BioMart gene list."
    )
    p.add_argument(
        "--biomart",
        type=Path,
        default=RAW_DIR / "mart_export.txt",
        help="Path to BioMart CSV export (default: data/raw/mart_export.txt)",
    )
    p.add_argument(
        "--model",
        default="esm2_t33_650M_UR50D",
        help="ESM-2 model variant (default: esm2_t33_650M_UR50D, 1280-dim)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Sequences per ESM-2 forward pass (default: 8, reduce if OOM)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=1022,
        help="Max residues per sequence before truncation (default: 1022)",
    )
    p.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Only run BioMart + sequence mapping, skip ESM-2 (useful for debugging)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device (cuda/cpu). Default: auto-detect",
    )
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    ensure_dirs()

    seq_cache = SEQ_DIR / "gene_to_sequence.pkl"
    emb_cache = EMB_DIR / "gene_to_embedding.pkl"
    report_path = REPORT_DIR / "embedding_report.txt"

    # Step 1: BioMart
    logger.info("Step 1/3: Loading protein-coding genes from BioMart")
    gene_symbols, gene_descriptions, biomart_stats = load_genes(args.biomart)

    # Step 2: Sequence mapping
    logger.info("Step 2/3: Mapping genes to protein sequences")
    gene_to_sequence, missing = map_sequences(
        gene_symbols, gene_descriptions, cache_path=seq_cache
    )
    seq_stats = analyze_sequences(gene_to_sequence, missing, args.max_length)

    # Step 3: ESM-2 embeddings
    elapsed = None
    if not args.skip_embedding:
        logger.info("Step 3/3: Generating ESM-2 embeddings")
        t0 = time.time()
        generate_embeddings(
            gene_to_sequence,
            model_name=args.model,
            batch_size=args.batch_size,
            cache_path=emb_cache,
            max_length=args.max_length,
        )
        elapsed = time.time() - t0
    else:
        logger.info("Step 3/3: Skipped (--skip-embedding)")

    # Report
    write_report(
        biomart_stats=biomart_stats,
        seq_stats=seq_stats,
        model_name=args.model,
        max_length=args.max_length,
        embedding_path=emb_cache,
        elapsed_sec=elapsed,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()
