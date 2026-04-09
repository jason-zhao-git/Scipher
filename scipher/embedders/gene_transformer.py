"""Gene Transformer cell embedder.

Expression-gated dense self-attention over ALL expressed genes per cell,
with multi-CLS token readout. Uses flash attention (F.scaled_dot_product_attention)
so variable-length sequences up to ~4K tokens are memory-efficient.

Architecture:
    1. Select all expressed genes per cell, pad to batch max
    2. Project frozen ESM-2 embeddings: (seq, 1280) -> Linear -> (seq, d_model)
    3. Gate by expression: token = proj(ESM2) * log1p_norm(expr)
    4. Prepend K learnable CLS tokens -> K+seq tokens
    5. Pre-norm dense self-attention (N layers, H heads)
       — no attn_mask so SDPA uses flash attention (O(n) memory)
       — padded positions zeroed after each layer to prevent leakage
    6. K CLS outputs -> concat -> Linear + LayerNorm -> cell embedding (output_dim)

Multiple CLS tokens let the model learn K different "readout heads" that
attend to different gene programs (e.g. kinases, TFs, surface markers).
Concatenating them preserves all K perspectives before final projection.

Inspired by UCE (Rosen & Roohani 2023) but supervised with MarginalizationLoss,
no gene subsampling, and expression-gated tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with flash attention.

    No attn_mask is passed to SDPA so it uses the flash attention kernel
    (O(n) memory). Padded positions must be zeroed externally after each layer.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.W_QKV = nn.Linear(d_model, 3 * d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq, d_model) — padded positions should be zero

        Returns:
            (batch, seq, d_model)
        """
        batch, seq, d_model = x.shape

        # Pre-norm attention
        h = self.norm1(x)
        qkv = self.W_QKV(h).reshape(batch, seq, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (batch, seq, n_heads, d_head)
        q = q.transpose(1, 2)  # (batch, n_heads, seq, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # No attn_mask — enables flash attention (O(n) memory instead of O(n^2))
        # Padded positions have zero vectors, so they contribute ~zero value.
        # CLS tokens waste a small amount of attention on padding, which is
        # an acceptable tradeoff for 8x memory reduction.
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p,
        )  # (batch, n_heads, seq, d_head)
        out = out.transpose(1, 2).reshape(batch, seq, d_model)
        out = self.W_O(out)
        x = x + out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))

        return x


class GeneTransformerEmbedder(nn.Module):
    """Gene Transformer cell embedder.

    Processes ALL expressed genes per cell through dense self-attention
    with expression gating and multi-CLS token readout.

    Args:
        gene_embed_dim: Dimension of input gene embeddings (e.g. 1280 for ESM-2).
        d_model: Internal dimension after projection.
        output_dim: Dimension of output cell embedding.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        n_cls: Number of CLS readout tokens.
        d_ff: FFN hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        gene_embed_dim=1280,
        d_model=512,
        output_dim=512,
        n_layers=4,
        n_heads=8,
        n_cls=8,
        d_ff=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.gene_embed_dim = gene_embed_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_cls = n_cls

        # Project gene embeddings: 1280 -> d_model
        self.input_proj = nn.Linear(gene_embed_dim, d_model, bias=False)

        # Learnable CLS tokens: K different readout heads
        self.cls_tokens = nn.Parameter(torch.randn(1, n_cls, d_model) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final norm (since we use pre-norm layers)
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection: K * d_model -> output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(n_cls * d_model, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, gene_embeddings, expression, expression_mask):
        """Compute cell embeddings via expression-gated transformer.

        Args:
            gene_embeddings: (n_genes, gene_embed_dim) - frozen, shared across batch
            expression: (batch, n_genes) - log1p + L1 normalized
            expression_mask: (batch, n_genes) - boolean, True where gene is expressed

        Returns:
            cell_embedding: (batch, output_dim)
            gene_attn: (batch, n_genes) - avg CLS->gene attention from last layer
        """
        batch_size = expression.shape[0]
        n_genes = gene_embeddings.shape[0]
        device = expression.device

        # Project all gene embeddings once: (n_genes, d_model)
        G_proj = self.input_proj(gene_embeddings)

        # For each cell, gather expressed genes and gate by expression
        # Find max expressed genes in batch for padding target
        n_expressed = expression_mask.sum(dim=1)  # (batch,)
        max_expressed = n_expressed.max().item()

        # Build packed sequences: vectorized (no Python loop)
        # Sort mask so True values come first per row, track original indices
        # argsort on ~mask puts True (0) before False (1), stable preserves order
        sorted_indices = (~expression_mask).long().argsort(dim=1, stable=True)
        gene_indices = sorted_indices[:, :max_expressed]  # (batch, max_expressed)

        # Build seq_mask: first n_expressed[i] positions are valid per row
        positions = torch.arange(max_expressed, device=device).unsqueeze(0)  # (1, max_expressed)
        seq_mask = positions < n_expressed.unsqueeze(1)  # (batch, max_expressed)

        # Gather gene embeddings and expression values
        # G_proj: (n_genes, d_model) -> gathered: (batch, max_expressed, d_model)
        gathered_embs = G_proj[gene_indices]  # (batch, max_expressed, d_model)
        gathered_expr = torch.gather(
            expression, 1, gene_indices,
        )  # (batch, max_expressed)

        # Gate: token = proj(ESM2) * expression
        tokens = gathered_embs * gathered_expr.unsqueeze(-1)

        # Zero out padded positions (critical: flash attention has no mask,
        # so padded positions must be zero to avoid contributing to attention)
        pad_mask = seq_mask.unsqueeze(-1)  # (batch, max_expressed, 1)
        tokens = tokens * pad_mask

        # Prepend CLS tokens
        cls = self.cls_tokens.expand(batch_size, -1, -1)  # (batch, n_cls, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_cls+max_expressed, d_model)

        # Build full mask for re-zeroing: CLS positions are always valid
        cls_valid = torch.ones(batch_size, self.n_cls, 1, dtype=torch.bool, device=device)
        full_pad_mask = torch.cat([cls_valid, pad_mask], dim=1)  # (batch, n_cls+max_expressed, 1)

        # Transformer layers — re-zero padded positions after each layer
        for layer in self.layers:
            tokens = layer(tokens)
            tokens = tokens * full_pad_mask

        # Final norm
        tokens = self.final_norm(tokens)

        # Multi-CLS readout: concat all K CLS outputs
        cls_out = tokens[:, :self.n_cls]  # (batch, n_cls, d_model)
        cls_concat = cls_out.reshape(batch_size, self.n_cls * self.d_model)
        cell_embedding = self.output_proj(cls_concat)  # (batch, output_dim)

        # Extract CLS->gene attention from last layer for interpretability
        gene_attn = self._extract_cls_attention(
            tokens, gene_indices, seq_mask, n_genes,
        )

        return cell_embedding, gene_attn

    @torch.no_grad()
    def _extract_cls_attention(self, tokens, gene_indices, seq_mask, n_genes):
        """Extract CLS->gene attention weights from last layer, scattered to full gene dim.

        Averages attention over all CLS tokens and all heads.
        """
        batch_size = tokens.shape[0]
        device = tokens.device
        layer = self.layers[-1]

        h = layer.norm1(tokens)
        qkv = layer.W_QKV(h).reshape(
            batch_size, tokens.shape[1], 3, layer.n_heads, layer.d_head,
        )
        q, k, _ = qkv.unbind(dim=2)

        # CLS queries: (batch, n_heads, n_cls, d_head)
        q_cls = q[:, :self.n_cls].transpose(1, 2)
        # All keys: (batch, n_heads, seq, d_head)
        k_all = k.transpose(1, 2)

        scores = torch.matmul(q_cls, k_all.transpose(-2, -1)) / (layer.d_head ** 0.5)
        # scores: (batch, n_heads, n_cls, n_cls+max_expressed)

        # Mask padded positions for accurate attention weights
        full_mask = torch.cat([
            torch.ones(batch_size, self.n_cls, dtype=torch.bool, device=device),
            seq_mask,
        ], dim=1)
        pad_mask = ~full_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Average over heads and CLS tokens, drop CLS->CLS positions
        attn_avg = attn.mean(dim=(1, 2))[:, self.n_cls:]  # (batch, max_expressed)

        # Scatter back to full n_genes dimension
        gene_attn = torch.zeros(batch_size, n_genes, device=device)
        gene_attn.scatter_(1, gene_indices, attn_avg * seq_mask.float())

        return gene_attn
