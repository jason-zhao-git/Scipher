"""Linear projection pooling: replaces MLP attention with matrix-weight aggregation.

Instead of MLP([embedding; expression]) → scalar weight (prone to shortcutting),
this module projects gene embeddings via a learned matrix and uses expression
values directly as mixing coefficients:

    P = G @ W          # project gene embeddings: (g, d) @ (d, K) → (g, K)
    c_i = e_i @ P      # aggregate by expression: (1, g) @ (g, K) → (1, K)

Expression and embeddings play structurally separated roles:
- W learns *what aspects* of gene embeddings matter (biological feature space)
- Expression determines *how much* each gene contributes (sample-specific mixture)

W never sees expression, so it cannot shortcut to just denoising abundance.
"""

import torch
import torch.nn as nn


class LinearProjectionPooling(nn.Module):
    """Project gene embeddings and aggregate by expression.

    Cell embedding: c_i = LayerNorm(e_i @ G @ W)

    Args:
        embed_dim: Dimension of gene embeddings (1280 for ESM-2)
        output_dim: Projection dimension K (cell embedding size)
        layer_norm: Whether to apply LayerNorm to the output
    """

    def __init__(self, embed_dim: int = 1280, output_dim: int = 256, layer_norm: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.projection = nn.Linear(embed_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim) if layer_norm else nn.Identity()

    def forward(self, gene_embeddings, expression, expression_mask):
        """Compute projection-pooled cell embeddings.

        Args:
            gene_embeddings: (n_genes, embed_dim) - fixed ESM-2 embeddings, NO batch dim
            expression: (batch, n_genes) - L1-normalized expression values
            expression_mask: (batch, n_genes) - boolean mask for expressed genes

        Returns:
            cell_embedding: (batch, output_dim)
            weights: (batch, n_genes) - masked expression (the effective gene weights)
        """
        # Zero out unexpressed genes
        weights = expression * expression_mask.float()

        # Project gene embeddings: (n_genes, embed_dim) → (n_genes, output_dim)
        projected = self.projection(gene_embeddings)

        # Expression-weighted aggregation: (batch, g) @ (g, K) → (batch, K)
        cell_embedding = torch.matmul(weights, projected)

        cell_embedding = self.norm(cell_embedding)

        return cell_embedding, weights
