# Next Optimization Steps for gfx906

## Current Performance (4x MI50, 64 GB VRAM)

| Model | Config | Gen tok/s | Context |
|-------|--------|-----------|---------|
| Qwen3-Coder-Next 80B Q4_1 | f16 KV + HIP graphs | 57 tok/s | 90K max |
| Qwen3-Coder-Next 80B Q4_1 | turbo3 KV + HIP graphs | 46 tok/s | 300K max |
| Devstral 24B Q4_K_M | f16 + spec (warm) | 42 tok/s | 131K |
| Devstral 24B Q4_K_M | f16 baseline | 28 tok/s | 131K |

Reference: vllm-gfx906 gets 56 tok/s on Qwen3.5-27B TP4 — we are at parity.

## Bottleneck Analysis

At batch=1 decode, gfx906 is at **10% memory bandwidth utilization**.
The bottleneck is NOT bandwidth — it's latency:
- Kernel launch overhead: ~3.6ms per token (720 launches)
- Pipeline bubble: ~8ms per token (5 graph splits, 4 GPUs)
- rocBLAS setup: ~2ms per token (1100+ GEMM calls)

## Implemented Optimizations

- [x] HIP Graphs (`-DGGML_HIP_GRAPHS=ON`): +8-10% gen speed
- [x] TurboQuant turbo3 KV cache: 3.3x more context, 18% speed cost
- [x] Shadow cache with stride fix: best turbo3 path
- [x] v_dot2_f32_f16 for turbo3 KQ: confirmed ALU-bound, no gain
- [x] Optimized V dequant (shift/mask, LUT cache)
- [x] Speculative decoding (ngram-mod): +47% on warm cache (pure transformers only)

## Not Yet Implemented

### 1. Fused MoE Kernels (High Impact)
The gfx906/fused/ directory has:
- `graph-fusion.cuh` — RMS+mul+MMQ fusion
- `mmq-prequantized.cuh` — prequantized MoE matmul
- `norm-fused-q8.cuh` — fused norm+quantize

These are in the codebase but may not be active for all model architectures.
Audit: check if fused paths fire for Qwen3-Coder-Next MoE layers.

### 2. Kernel Launch Reduction (High Impact)
720 kernel launches per token at ~5us each = 3.6ms overhead.
Options:
- HIP graphs already help (~10%)
- Fuse more ops (attention pre/post ops, layernorm+matmul)
- Persistent kernels (keep kernel running, feed work via queue)

### 3. Pipeline Parallelism Tuning (Medium Impact)
Currently 5 graph splits across 4 GPUs. At batch=1 this causes ~8ms bubble.
Options:
- Reduce splits by balancing layer assignment
- Use tensor parallelism instead (split each layer across GPUs)
- Overlap compute between pipeline stages

### 4. Speculative Decoding for Hybrid Models (Medium Impact)
Qwen3-Coder-Next (hybrid SSM+attention) doesn't support spec decoding
because SSM layers can't do partial sequence removal.
Options:
- Implement SSM state checkpointing for rollback
- Use a separate draft model instead of self-speculation

### 5. gfx906-Specific Kernel Tuning (Low-Medium Impact)
From vllm-gfx906-mobydick analysis:
- `num_stages=1` for all kernels (LDS stability)
- `num_warps=4` as default (register pressure sweet spot)
- Adaptive BLOCK_SIZE_M for small batches (16 for batch≤1)
- 512-token chunk size for attention computation
- LDS limit 64KB (not 160KB like newer chips)

### 6. Power Tuning (No Impact at Current Workload)
MI50 TDP is 300W, tested at 225W limit.
Under load, clocks already ramp to near-max (1725 MHz SCLK, 1000 MHz MCLK).
At batch=1, power is NOT the bottleneck — latency is.
Would help at larger batch sizes (batch>8).

## Key Insight

The fundamental limit at batch=1 on gfx906 is **kernel launch latency**, not compute or bandwidth. Both our fork and vllm-gfx906 hit the same ~55-57 tok/s ceiling on 80B MoE models. Breaking through requires either:
- Larger batch sizes (multi-user serving)
- Speculative decoding (amortize overhead over draft tokens)
- Drastically fewer kernel launches (mega-fused kernels)
