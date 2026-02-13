"""Attention pooling cell embedder: learns gene importance from embedding + expression.

Adapted from funcCell/src/model/attention_pooling.py
- DECOUPLED the binary classifier head from the embedding mechanism
- AttentionPooling now returns cell embeddings only (no classification)
- Classifier head belongs downstream (e.g. with MarginalizationLoss from real_McCell)
- CellDataset kept for loading expression + gene embeddings together
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.sparse import issparse
import anndata


class AttentionPooling(nn.Module):
    """Attention-based pooling that learns gene importance from embedding + expression.

    Unlike weighted sum (where weight = expression), this learns
    weight = f(embedding, expression), discovering which genes matter
    regardless of expression level.

    Returns cell embeddings only. Classification head is separate.

    Args:
        embed_dim: Dimension of gene embeddings (e.g. 512 for ProteinBERT, 1280 for ESM-2)
        hidden_dim: Hidden dimension of attention network
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        # Attention network: gene embedding (embed_dim) + expression (1) -> score
        self.attention = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, gene_embeddings, expression, expression_mask):
        """Compute attention-weighted cell embeddings.

        Args:
            gene_embeddings: (batch, n_genes, embed_dim)
            expression: (batch, n_genes) - normalized expression values
            expression_mask: (batch, n_genes) - boolean mask for expressed genes

        Returns:
            cell_embedding: (batch, embed_dim)
            attn_weights: (batch, n_genes)
        """
        # Concatenate expression to embeddings: (batch, n_genes, embed_dim+1)
        gene_features = torch.cat([gene_embeddings, expression.unsqueeze(-1)], dim=-1)

        # Compute attention weights
        attn_scores = self.attention(gene_features).squeeze(-1)  # (batch, n_genes)
        attn_scores = attn_scores.masked_fill(~expression_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, n_genes)

        # Weighted pool: (batch, 1, n_genes) @ (batch, n_genes, embed_dim) -> (batch, embed_dim)
        cell_embedding = torch.bmm(attn_weights.unsqueeze(1), gene_embeddings).squeeze(1)

        return cell_embedding, attn_weights


class CellDataset(Dataset):
    """Dataset that pairs expression data with gene embeddings.

    Gene embeddings are stored once and accessed via get_gene_embeddings(),
    NOT returned per-sample (avoids copying ~40MB per item).

    Args:
        adata_list: List of AnnData objects to concatenate
        labels: Array of labels (int class indices for multi-class)
        gene_to_embedding: Dict mapping gene names to embedding vectors
    """

    def __init__(self, adata_list, labels, gene_to_embedding):
        self.adata = anndata.concat(adata_list, join='inner') if len(adata_list) > 1 else adata_list[0]
        self.labels = np.asarray(labels)

        # Filter to genes with embeddings
        self.gene_list = [g for g in self.adata.var_names if g in gene_to_embedding]
        print(f"CellDataset: {len(self.gene_list)} genes with embeddings, {len(self.labels)} cells")

        gene_embeddings_np = np.stack(
            [gene_to_embedding[g] for g in self.gene_list]
        ).astype(np.float32)
        self._gene_embeddings_tensor = torch.tensor(gene_embeddings_np, dtype=torch.float32)
        self._gene_embeddings_np = gene_embeddings_np

        self.adata = self.adata[:, self.gene_list]

    def get_gene_embeddings(self, device=None):
        """Get gene embeddings tensor (n_genes, embed_dim), optionally on device."""
        if device is not None:
            return self._gene_embeddings_tensor.to(device)
        return self._gene_embeddings_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        expr = self.adata[idx].X
        if issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()

        expr_norm = expr / (expr.sum() + 1e-10)
        mask = expr > 0

        return {
            'expression': torch.tensor(expr_norm, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
