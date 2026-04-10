# Scipher Gene Transformer — Full Architecture Walkthrough

## What are we doing?

We classify **cell types** from single-cell RNA-seq data. Each cell has ~19,000 genes with expression counts. We want to predict what type of cell it is (e.g., T cell, B cell, monocyte) using a hierarchical classification that respects the Cell Ontology (a tree where "T cell" is a child of "lymphocyte" which is a child of "leukocyte").

The key idea: instead of using raw gene counts directly, we represent each gene by its **protein-level embedding** from ESM-2 (a protein language model). This encodes what the protein *does* (kinase, receptor, transcription factor) rather than just how much of it there is.

---

## The pipeline, end to end

```
Raw counts (19K genes per cell)
        |
        v
[1] Expression preprocessing (log1p + L1 norm)
        |
        v
[2] Gene selection (keep only expressed genes, ~2-4K per cell)
        |
        v
[3] Protein embedding projection (ESM-2 1280-dim → 512-dim)
        |
        v
[4] Expression gating (token = projected_embedding * normalized_expression)
        |
        v
[5] Prepend 8 CLS tokens
        |
        v
[6] Transformer (4 layers, 8 heads, flash attention in bf16)
        |
        v
[7] Multi-CLS readout (8 × 512 = 4096 → project to 512-dim cell embedding)
        |
        v
[8] WideNN classifier (512 → 2048 → 2048 → 512 → 61 leaf types)
        |
        v
[9] MarginalizationLoss (leaf CrossEntropy + parent BCE)
```

---

## Step-by-step detail

### [1] Expression preprocessing

**Input:** Raw integer counts per gene, e.g., `[0, 0, 5, 0, 1832, 0, 3, ...]` (19K values, mostly zeros)

**Operation:**
```python
X_log = torch.log1p(X)           # log(1 + count), compresses dynamic range
expression = X_log / X_log.sum()  # L1 normalize so values sum to 1
```

**Output:** `[0, 0, 0.0012, 0, 0.0054, 0, 0.0003, ...]` — a probability distribution in log-space

**Why log1p:** A gene with 10,000 counts isn't 100x more important than one with 100 counts. Log compresses this to ~9.2 vs ~4.6, a ~2x ratio. This prevents housekeeping genes (extremely high counts in every cell) from dominating.

**Why L1 normalize:** Makes cells comparable regardless of total sequencing depth. A cell sequenced deeply (1M total reads) and one sequenced shallowly (100K total reads) get normalized to the same scale.

**Why both together:** log1p compresses the dynamic range, L1 normalization removes library size effects. The combination gives a distribution where each value reflects relative importance of that gene in that cell.

---

### [2] Gene selection

**Input:** Expression vector (19K values) and boolean mask (True where count > 0)

**Operation:** Keep only genes where expression > 0. A typical cell expresses ~2,000-4,000 out of ~19,000 genes. Different cells express different genes.

```
Cell A expressed genes: [TP53, BRCA1, MYC, CD4, ...]     → 3,200 genes
Cell B expressed genes: [TP53, CD8A, FOXP3, IL2, ...]     → 2,800 genes
Cell C expressed genes: [HBA1, HBB, SLC4A1, ...]          → 1,900 genes
```

**Padding:** Since cells have different numbers of expressed genes, we pad to the maximum in the batch. If the batch max is 3,200, cell B gets 400 zero-padded positions and cell C gets 1,300.

**Why not use all 19K:** The ~15K unexpressed genes are zeros — they carry no information for that cell and would waste computation. Attention over 4K tokens is much cheaper than 19K.

**Why not top-K:** Top-K by expression would select housekeeping genes (ribosomal proteins, GAPDH) that are highly expressed in every cell type. These are the *least* informative for classification. Using all expressed genes avoids this bias.

**Implementation detail:** Vectorized via `argsort` (no Python loop). The expressed gene indices are gathered efficiently.

---

### [3] Protein embedding projection

**Input:** Frozen ESM-2 embeddings, shape `(n_genes, 1280)`

**Operation:**
```python
G_proj = Linear(1280 → 512, no bias)(gene_embeddings)  # (n_genes, 512)
gathered = G_proj[gene_indices]                          # (batch, seq, 512)
```

**What ESM-2 embeddings encode:** ESM-2 is a protein language model trained on 250M protein sequences via masked language modeling. The 1280-dim embedding for each gene captures:
- Protein structure (alpha-helix vs beta-sheet regions)
- Evolutionary conservation (which residues are critical)
- Functional domains (kinase domain, DNA-binding domain)
- Protein family relationships (all kinases cluster together)

ESM-2 was NOT trained on expression data or cell biology — it only sees amino acid sequences. But proteins with similar functions get similar embeddings because they have similar sequences due to evolution.

**Why project 1280 → 512:** The transformer runs at d_model=512. Running at 1280 would be ~6x more expensive (attention and FFN scale with d_model^2). The projection learns which dimensions of the ESM-2 space matter for cell type classification and discards the rest (e.g., detailed structural info that doesn't distinguish cell types).

**Why frozen, not fine-tuned:** ESM-2 has 650M parameters. Fine-tuning it would dominate training and overfit on our ~2M cells. Frozen embeddings are a fixed, high-quality representation of protein function.

**This is shared across the batch:** `input_proj(gene_embeddings)` runs once per forward pass, not per cell. All cells share the same projected gene embeddings — the per-cell variation comes from which genes are selected and how they're gated.

---

### [4] Expression gating

**Input:** Projected embeddings `(batch, seq, 512)` and expression values `(batch, seq)`

**Operation:**
```python
token = projected_embedding * expression_value
```

For gene TP53 in cell A with expression 0.0012:
```
token_TP53 = proj(ESM2_TP53) * 0.0012
```

**What this does:** Each token's **direction** is determined by the protein embedding (what the gene *is*). The **magnitude** is scaled by expression (how much of it there is). The transformer sees a "sentence" where each "word" encodes both identity and abundance.

**Why multiplicative, not additive:** If we concatenated expression as an extra feature (`[proj(ESM2), expr]`), the model could learn to ignore the 512-dim protein embedding and just read the scalar expression value — collapsing to an expression-only model. Multiplicative gating structurally prevents this: the only source of *directional* information is the protein embedding. The model cannot shortcut past it.

**Concern about small values:** With ~3000 expressed genes after L1 normalization, each value is ~0.0003. So tokens have tiny magnitude (~0.0003 × ||proj(ESM2)||). This is OK because:
1. LayerNorm in the first transformer layer normalizes each token independently, equalizing magnitudes
2. The *relative* differences between tokens still carry the signal

---

### [5] Prepend CLS tokens

**Input:** Gene tokens `(batch, seq, 512)`

**Operation:**
```python
cls_tokens = Parameter(randn(1, 8, 512) * 0.02)  # 8 learned vectors
tokens = cat([cls_tokens, gene_tokens], dim=1)     # (batch, 8+seq, 512)
```

**What CLS tokens are:** 8 learnable vectors prepended to every cell's gene sequence. They have no biological meaning at initialization — they learn to *query* the gene tokens through attention. After the transformer, each CLS token has aggregated information from the genes it attended to.

**Why 8, not 1:** A single CLS token must compress ~3000 gene interactions into one 512-dim vector. With 8 CLS tokens, each can specialize: one might learn to attend to kinases, another to transcription factors, another to surface markers. The 8 outputs are concatenated (8 × 512 = 4096) then projected down to 512, preserving all 8 perspectives.

**Why CLS tokens instead of mean-pooling:** Mean-pooling would average all gene tokens equally — a gene with expression 0.0001 gets the same weight as one with 0.005. CLS tokens learn *which* genes matter through attention, weighting them adaptively. This is the same readout mechanism as BERT and UCE.

---

### [6] Transformer (4 layers)

**Input:** `(batch, 8+seq, 512)` — CLS tokens + gene tokens

**Each layer does:**
```
1. LayerNorm(tokens)                          # normalize
2. MultiHeadAttention(Q, K, V)                # every token attends to every other
3. tokens = tokens + attention_output         # residual connection
4. LayerNorm(tokens)                          # normalize
5. FFN: Linear(512→2048) → GELU → Linear(2048→512)  # per-token MLP
6. tokens = tokens + ffn_output               # residual connection
7. Zero out padded positions                  # prevent information leakage
```

**What attention does here:** Each gene token looks at every other gene token and every CLS token. The attention scores are based on Q·K similarity of the *protein embeddings* (scaled by expression). So a kinase token will attend strongly to its substrate tokens — not because we told it to (no PPI graph), but because their ESM-2 embeddings are functionally related and the attention weights learn to exploit this.

CLS tokens attend to all genes, learning which gene programs define the cell. Genes also attend to CLS, getting a "global context" signal.

**Pre-norm (not post-norm):** LayerNorm before attention/FFN (not after). This is the modern standard — more stable training, especially with deep transformers. GPT-2+, LLaMA, etc. all use pre-norm.

**Flash attention:** `F.scaled_dot_product_attention` with bf16 autocast. Never materializes the full seq×seq attention matrix (which would be 4000×4000 = 16M entries per head). Instead processes in tiles on GPU SRAM. Memory: O(n) instead of O(n^2).

**bf16 autocast:** Flash attention only activates in half precision. `torch.autocast("cuda", dtype=torch.bfloat16)` casts Q/K/V to bf16, enabling the flash kernel. Also halves FFN memory. bfloat16 keeps the same exponent range as float32, so no overflow issues (unlike fp16).

**Gradient checkpointing:** During training, activations from each layer are NOT stored. Instead, they're recomputed during the backward pass. This trades ~30% more compute for ~4x less memory. Critical for fitting 4000-token sequences on A40 GPUs.

**Re-zeroing padded positions:** After each layer, padded positions are multiplied by zero. Without the attention mask (which we dropped for flash attention), padded positions could accumulate non-zero values through residual connections. Re-zeroing prevents this leakage.

**Dimensions:**
- d_model = 512 (token dimension)
- n_heads = 8 (each head has d_head = 64)
- d_ff = 2048 (FFN expands 4x then contracts back)
- 4 layers deep

---

### [7] Multi-CLS readout

**Input:** All tokens after 4 transformer layers, shape `(batch, 8+seq, 512)`

**Operation:**
```python
cls_outputs = tokens[:, :8]                          # (batch, 8, 512)
cls_concat = cls_outputs.reshape(batch, 8 * 512)     # (batch, 4096)
cell_embedding = Linear(4096 → 512)(cls_concat)      # (batch, 512)
cell_embedding = LayerNorm(cell_embedding)            # normalize
```

**What happened to the gene tokens:** They're discarded. All the information has been funneled into the 8 CLS tokens through 4 layers of attention. The gene tokens served as "memory" that the CLS tokens read from.

**Why concat then project (not mean):** Concatenation preserves each CLS token's distinct perspective. Mean-pooling would blur them together, defeating the purpose of having 8 specialized readout heads.

**Output:** A 512-dim vector that represents the cell's identity, informed by protein-level gene functions and expression levels.

---

### [8] WideNN classifier

**Input:** Cell embedding `(batch, 512)`

**Architecture:**
```
Linear(512 → 2048) → BatchNorm → ReLU → Dropout(0.2)
Linear(2048 → 2048) → BatchNorm → ReLU → Dropout(0.2)
Linear(2048 → 512) → BatchNorm → ReLU → Dropout(0.2)
Linear(512 → 61)    # 61 leaf cell types
```

**Output:** Raw logits `(batch, 61)` — one score per leaf cell type. NOT probabilities (no softmax here — the loss function handles that).

**Why a separate classifier (not just a linear head):** The transformer produces a general cell embedding. The WideNN has enough capacity (3 hidden layers, 2048 wide) to learn complex decision boundaries between 61 cell types. A single linear layer would be too simple for 61-way classification.

**Why WideNN, not the transformer output directly:** The transformer's job is to produce a good *representation*. The classifier's job is to map that representation to cell types. Separating them means the embedding is reusable (e.g., for UMAP visualization, kNN, clustering) without being tied to a specific classification head.

---

### [9] MarginalizationLoss

**Input:** Logits `(batch, 61)` and ground truth labels (integer indices into the cell type hierarchy)

**This is the key innovation from real_McCell.** The Cell Ontology is a tree:

```
hematopoietic cell (CL:0000988)
├── lymphocyte
│   ├── T cell         ← leaf (we predict this directly)
│   ├── B cell         ← leaf
│   └── NK cell        ← leaf
├── myeloid cell
│   ├── monocyte       ← leaf
│   └── neutrophil     ← leaf
└── ...
```

Some training cells are labeled as **leaves** (e.g., "T cell") and some are labeled as **internal nodes** (e.g., "lymphocyte" — we know it's a lymphocyte but not which specific type).

**Two loss components:**

**Leaf loss (weighted by 7.0):**
```python
CrossEntropyLoss(logits[leaf_cells], leaf_labels)
```
Standard classification loss for cells with specific leaf labels. Weighted 7x because these are the most informative training signals.

**Parent loss (for ALL cells):**
```python
# 1. Convert logits to leaf probabilities
leaf_probs = softmax(logits)                    # (batch, 61)

# 2. Marginalize: sum leaf probs under each internal node
#    e.g., P(lymphocyte) = P(T cell) + P(B cell) + P(NK cell)
internal_probs = marginalization_matrix @ leaf_probs  # (batch, 43)

# 3. BCE loss against true parent labels
#    A "T cell" should have P(lymphocyte)=1, P(myeloid)=0
#    A "lymphocyte" should have P(lymphocyte)=1
BCE(internal_probs, parent_labels, weight=exclusion_mask)
```

**Why this helps:** If a cell is labeled "lymphocyte" (internal node), standard CrossEntropy can't use it — there's no single leaf to target. MarginalizationLoss can: it says "the sum of probabilities for T cell + B cell + NK cell should be ~1." This lets us use all training data, including imprecisely labeled cells.

**Exclusion mask:** Some internal node relationships are irrelevant (e.g., a T cell shouldn't be penalized for low P(erythrocyte lineage)). The exclusion mask zeros out unrelated internal nodes.

---

## Data flow shapes (concrete example)

```
Batch of 64 cells, ~3000 expressed genes each, 19K total genes

Expression preprocessing:
  raw counts:       (64, 19K)  int
  log1p + L1:       (64, 19K)  float32, sums to 1 per row

Gene selection:
  mask:             (64, 19K)  bool, ~3000 True per row
  gene_indices:     (64, 3200) long (padded to batch max)
  seq_mask:         (64, 3200) bool

Embedding projection:
  G_proj:           (19K, 512) float32 (shared, computed once)
  gathered_embs:    (64, 3200, 512) float32

Expression gating:
  tokens:           (64, 3200, 512) float32

Prepend CLS:
  tokens:           (64, 3208, 512) float32  [8 CLS + 3200 genes]

Transformer (×4 layers, in bf16):
  tokens:           (64, 3208, 512) bf16 inside layers, fp32 between

CLS readout:
  cls_out:          (64, 8, 512)
  cls_concat:       (64, 4096)
  cell_embedding:   (64, 512)

Classifier:
  logits:           (64, 61)

Loss:
  leaf_loss:        scalar (CrossEntropy on leaf-labeled cells)
  parent_loss:      scalar (BCE on marginalized internal probs)
  total_loss:       leaf_loss * 7.0 + parent_loss
```

---

## Parameter count

```
GeneTransformerEmbedder:
  input_proj (1280 → 512, no bias):         655,360
  cls_tokens (8 × 512):                       4,096
  4 × TransformerLayer:
    per layer:
      W_QKV (512 → 1536):                   786,432
      W_O (512 → 512):                      262,656
      FFN Linear(512 → 2048):             1,049,088
      FFN Linear(2048 → 512):             1,049,088
      2 × LayerNorm(512):                     2,048
    subtotal per layer:                    3,149,312
    × 4 layers:                           12,597,248
  final_norm (512):                            1,024
  output_proj Linear(4096 → 512):          2,097,664
  output_proj LayerNorm(512):                  1,024
  Embedder total:                         ~15,356,416

WideNN classifier:
  Linear(512 → 2048) + BN:                1,052,672
  Linear(2048 → 2048) + BN:               4,198,400
  Linear(2048 → 512) + BN:                1,049,600
  Linear(512 → 61):                           31,293
  Classifier total:                        ~6,331,965

TOTAL TRAINABLE:                          ~21,688,381 (~22M)
```

---

## Memory budget per GPU (A40, 46GB)

With bf16 autocast + gradient checkpointing + flash attention:

```
Gene embeddings buffer (19K × 1280 × fp32):          ~94 MB
Model parameters (fp32 master copy):                  ~88 MB
Optimizer states (Adam: 2 copies × fp32):            ~176 MB
Gradients (fp32):                                     ~88 MB
Token tensor (64 × 3200 × 512 × fp32):              ~400 MB
Peak layer activation (FFN in bf16):                   ~1.6 GB
  — only 1 layer at a time due to gradient checkpointing
Flash attention workspace:                            ~200 MB
DataParallel overhead:                                ~500 MB
PyTorch allocator overhead:                            ~2 GB
                                                    --------
Estimated total:                                     ~5-8 GB
Headroom on 46GB A40:                                ~38 GB free
```

---

## Training configuration

```
Dataset:          CL:0000988 (blood cells), ~2M cells, 61 leaf types
Split:            80/20 train/val, streamed from SOMA database
Batch:            64 per GPU × 3-4 GPUs = 192-256 effective
Optimizer:        Adam, lr=1e-4, no schedule
Gradient clip:    1.0
Epochs:           10
Loss:             MarginalizationLoss (leaf_weight=7.0)
Checkpoints:      Every epoch (full model + optimizer + loss history)
```

---

## What makes this different from UCE

| | UCE | Scipher |
|---|---|---|
| Gene representation | ESM2-15B (5120-dim) | ESM2-650M (1280-dim) |
| Gene selection | Sample 1024 weighted by expression | ALL expressed genes (~2-4K) |
| Expression in tokens | Only used for sampling weights | Multiplicatively gates each token |
| Attention | Dense, 33 layers, 650M params | Dense, 4 layers, ~22M params |
| Readout | 1 CLS token | 8 CLS tokens |
| Training | Self-supervised (masked gene prediction), 36M cells, 40 days on 24 A100s | Supervised (hierarchical classification), ~2M cells, ~1 day on 4 A40s |

---

## Known tradeoffs and limitations

1. **No attention mask → padding tokens get some attention.** CLS tokens waste a small fraction of attention weight on zero-padded positions. Acceptable tradeoff for flash attention's O(n) memory.

2. **Expression gating produces tiny magnitudes.** After L1 normalization, each value is ~0.0003. LayerNorm in the first transformer layer equalizes this, but the first layer's attention patterns may be noisy before LayerNorm kicks in.

3. **No learning rate warmup or schedule.** Flat lr=1e-4 throughout. A cosine schedule with warmup would likely improve convergence.

4. **Gradient checkpointing adds ~30% compute overhead.** Each layer's activations are recomputed during backward. Necessary for fitting in GPU memory.

5. **DataParallel (not DistributedDataParallel).** DataParallel has a GIL bottleneck and uneven GPU memory usage. DDP would be faster but requires multi-process launch. Acceptable for 3-4 GPUs.

6. **SOMA streaming is the speed bottleneck.** GPU utilization is limited by how fast data arrives from disk. The transformer forward/backward is fast (~100ms per batch); data loading takes the rest.
