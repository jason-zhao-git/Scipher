"""Matrix weight cell embedder: linear projection forcing expression-embedding interaction.

Cell embedding = expression @ G @ W
- G: frozen gene embeddings (n_genes, embed_dim), shared across batch
- W: learned projection (embed_dim, output_dim), no bias
- expression: log+L1-normalized expression (batch, n_genes)

W only sees G, never expression. Expression is purely mixing coefficients.
This makes it structurally impossible for the model to shortcut by ignoring
protein embeddings (unlike AttentionPooling where the MLP sees [embedding; expression]).
"""

import torch
import torch.nn as nn


class MatrixWeightEmbedder(nn.Module):
    """Linear projection embedder that forces expression-embedding interaction.

    Computes cell_embedding = W(expression @ gene_embeddings), where W is a
    learned linear projection. Expression determines the mixture of gene
    embeddings; W determines the feature space. Neither can override the other.

    Args:
        embed_dim: Dimension of gene embeddings (e.g. 1280 for ESM-2)
        output_dim: Dimension of output cell embeddings (e.g. 256)
    """

    def __init__(self, embed_dim: int = 1280, output_dim: int = 256):
        super().__init__()
        self.W = nn.Linear(embed_dim, output_dim, bias=False)

    def forward(self, gene_embeddings, expression, expression_mask=None):
        """Compute expression-weighted projected cell embeddings.

        Args:
            gene_embeddings: (n_genes, embed_dim) -- 2D, shared across batch
            expression: (batch, n_genes) -- log+L1-normalized
            expression_mask: ignored (zero expression = zero contribution)

        Returns:
            cell_embedding: (batch, output_dim)
            expression: (batch, n_genes) -- passthrough (expression IS the weight)
        """
        # expression @ gene_embeddings: (batch, embed_dim)
        # W projects to output_dim
        cell_embedding = self.W(expression @ gene_embeddings)
        return cell_embedding, expression
