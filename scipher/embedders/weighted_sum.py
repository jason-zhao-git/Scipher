"""Weighted-sum cell embedder: cell_embedding = normalized_expression @ gene_embeddings.

Copied from funcCell/src/model/cell_embeddings.py
- Kept CellEmbedder class and load/save helpers intact
- This is the baseline (no trainable parameters)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


class WeightedSumEmbedder:
    """Aggregate gene embeddings via expression-weighted sum (baseline)."""

    def __init__(self, gene_to_embedding: Dict[str, np.ndarray]):
        self.gene_to_embedding = gene_to_embedding
        first_emb = next(iter(gene_to_embedding.values()))
        self.embedding_dim = first_emb.shape[0]
        logger.info(f"WeightedSumEmbedder: {len(gene_to_embedding)} genes, {self.embedding_dim}-dim")

    def embed(self, adata: AnnData, batch_size: int = 1000) -> np.ndarray:
        """Create cell embeddings via expression-weighted sum.

        Formula: cell_embedding = (expression / sum(expression)) @ gene_embeddings

        Args:
            adata: AnnData with expression data (cells x genes)
            batch_size: Cells per processing batch

        Returns:
            Array of shape (n_cells, embedding_dim)
        """
        common_genes = [g for g in adata.var_names if g in self.gene_to_embedding]
        logger.info(f"Gene coverage: {len(common_genes)}/{adata.n_vars} ({len(common_genes)/adata.n_vars*100:.1f}%)")

        if not common_genes:
            raise ValueError("No common genes found between data and embeddings")

        adata_filtered = adata[:, common_genes]
        expression = adata_filtered.X
        if issparse(expression):
            expression = expression.toarray()

        gene_embeddings = np.array([self.gene_to_embedding[g] for g in common_genes])

        n_cells = expression.shape[0]
        cell_embeddings = np.zeros((n_cells, self.embedding_dim), dtype=np.float32)

        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch_expr = expression[start:end]
            row_sums = batch_expr.sum(axis=1, keepdims=True)
            normalized = batch_expr / (row_sums + 1e-10)
            cell_embeddings[start:end] = normalized @ gene_embeddings

        logger.info(f"Generated cell embeddings: {cell_embeddings.shape}")
        return cell_embeddings


def load_gene_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """Load gene embeddings from pickle file."""
    with open(path, "rb") as f:
        gene_to_embedding = pickle.load(f)
    logger.info(f"Loaded {len(gene_to_embedding)} gene embeddings from {path}")
    return gene_to_embedding
