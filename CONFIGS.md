# Training Configurations

Tested configurations for GeneTransformerEmbedder + WideNN on blood cells (CL:0000988).
All runs on Great Lakes cluster with 4x NVIDIA A40 (46GB each).

## Config A: Large (v1, epoch 1 completed)

Run date: 2026-04-10. Checkpoint: `transformer_2026-04-10_2026-01-29_CL0000988/epoch01.pt`

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_layers | 6 |
| n_heads | 12 |
| n_cls | 12 |
| d_ff | 3072 |
| output_dim | 512 |
| dropout | 0.1 |
| **Total params** | **54.6M** (48.2M embedder + 6.3M classifier) |
| batch_size | 32/GPU (128 effective) |
| optimizer | Adam, lr=1e-4, no schedule |
| GPU memory | ~36GB/GPU (tight) |
| Time/epoch | ~83 hours |
| Epochs trained | 1 |

Results (epoch 1, 50K val cells):
- Micro F1: 0.7534, Macro F1: 0.2078, Weighted F1: 0.7923
- Train/val loss gap was 77% SGD measurement artifact (real gap only +0.47)

## Config B: Medium (v2, current)

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| n_layers | 4 |
| n_heads | 8 |
| n_cls | 8 |
| d_ff | 2048 |
| output_dim | 256 |
| dropout | 0.1 |
| **Total params** | **~21M** (~15M embedder + ~6M classifier) |
| batch_size | 64/GPU (256 effective) |
| optimizer | Phase 1: Adam lr=1e-3 (classifier warmup, 1000 steps) |
|  | Phase 2: AdamW, peak_lr=2e-4, min_lr=1e-5, weight_decay=0.01 |
| LR schedule | Linear warmup (500 steps) + cosine decay |
| GPU memory | ~24-28GB/GPU at batch_size=64 (~60%) |
| Est. time/epoch | ~47 hours |
| Target | 4 epochs in ~190 hours |

## Notes

- SOMA data streaming is the bottleneck (~2.1s of ~2.2s per batch), not GPU compute
- Larger batch size = fewer batches/epoch = primary speedup lever
- Model reduction mainly frees GPU memory to enable larger batch sizes
- **batch_size=96/GPU OOMs on peak batches** — variable sequence length (cells expressing 3-4K genes) causes memory spikes up to 44.9GB on A40 (46GB). 64/GPU is the safe max.
