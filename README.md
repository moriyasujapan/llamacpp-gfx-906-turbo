# llama.cpp-gfx906-turbo

llama.cpp fork for **AMD MI50/MI60 (gfx906)** with **TurboQuant turbo3 KV cache compression**.

Combines upstream llama.cpp with [iacopPBK](https://github.com/iacopPBK/llama.cpp-gfx906) gfx906 Wave64 kernels and [Madreag](https://github.com/Madreag/turbo3-cuda) TurboQuant CUDA port, plus HIP-specific bug fixes for gfx906 correctness.

## Results

- **3.3x more context** than fp16 KV cache on the same hardware
- **4.6x KV cache compression** (turbo3: 3.5 bits/value vs 16 bits fp16)
- **300K context** with turbo3 vs **90K max** with f16 on Qwen3-Coder-Next 80B
- **1M context** on Qwen3.5-27B Q4_0 with 4x MI50

### Benchmark: Qwen3-Coder-Next Q4_1 (80B MoE) — 4x MI50 (64 GB)

| KV Cache | Max Context | Gen (tok/s) | Prompt (tok/s) | Peak VRAM | KV Size |
|----------|-------------|-------------|----------------|-----------|---------|
| **turbo3 K+V** | **300,000** | 37.2 | 116.1 | 97% | ~3.5 GB |
| turbo3 K+V | 256,000 | 39.7 | 130.8 | 93% | ~3.0 GB |
| turbo3 K+V | 65,536 | 42.4 | 127.7 | 69% | ~0.8 GB |
| f16 K+V | 90,000 | 51.6 | 138.8 | 96% | ~10.5 GB |
| f16 K+V | 65,536 | 51.6 | 153.2 | 96% | ~7.7 GB |

Tensor split: `7,8,8,8` — flags: `--no-mmap --no-warmup`

**Trade-off**: turbo3 gives 3.3x more context at 18% gen speed cost (at same context size).
The speed gap widens to 26% at max context due to larger KV cache read overhead.

## Build

```bash
cd llama-cpp-gfx906-turbo
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-server -j$(nproc)
```

Requires ROCm 7.1+.

## Usage

```bash
./build/bin/llama-server \
  -m model.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 999 -c 262144 -np 1 \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --no-warmup
```

For context beyond 262K, add YaRN:
```bash
  --rope-scaling yarn --rope-freq-scale 0.25 \
  --yarn-ext-factor 1.0 --yarn-attn-factor 1.0 \
  --yarn-beta-fast 32 --yarn-beta-slow 1
```

## Changes from Upstream

### New Files

**gfx906 Wave64 kernels** (from iacopPBK) — `ggml/src/ggml-cuda/gfx906/`
- 29 files: DPP warp reductions, Q8 FlashAttention tile kernel, optimized RoPE, warp-cooperative MMVQ, software-pipelined MMQ, custom SGEMM/MMF for medium batch sizes

**TurboQuant CUDA kernels** (from Madreag)
- `ggml/src/ggml-cuda/turbo-quant.cu` + `.cuh` — FWHT rotation, SET_ROWS quantize, dequant kernels for turbo2/3/4
- `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo*` — 5 FA template instances
- `ggml/src/ggml-turbo-quant.c` — CPU quant/dequant
- `src/turbo-rotation-data.h` — Precomputed 128x128 rotation matrix

### Modified Files — gfx906 Patches

| File | Change |
|------|--------|
| `ggml-cuda/common.cuh` | DPP shuffle XOR dispatch, `GGML_CUDA_CC_IS_GCN` |
| `ggml-cuda/fattn-common.cuh` | GCN tile sizes, thread counts |
| `ggml-cuda/fattn.cu` | Q8 tile kernel selection + turbo3 shadow cache + native vec path |
| `ggml-cuda/fattn-vec.cuh` | Turbo3 FA vec dispatch, sparse V skip, `__expf` |
| `ggml-cuda/mmq.cu` + `mmq.cuh` | Vectorized loads, software pipelining for GCN |
| `ggml-cuda/mmvq.cu` | Half-warp MoE dispatch, Wave64 warp ID fix |
| `ggml-cuda/quantize.cu` | DPP-based warp reductions for Q8 quantize |
| `ggml-cuda/rope.cu` | gfx906 RoPE kernel dispatch |
| `ggml-cuda/vecdotq.cuh` | gfx906 MXFP4 lookup, `get_int_b2_fast` |
| `ggml-cuda/add-id.cu` | gfx906 vectorized add_id |
| `ggml-cuda/mmid.cu` | gfx906 MoE dispatch |
| `ggml-cuda/ssm-scan.cu` | gfx906 SSM optimization |
| `ggml-cuda/ggml-cuda.cu` | gfx906 custom GEMM dispatch, SOLVE_TRI limits, turbo3 ops |
| `ggml-hip/CMakeLists.txt` | gfx906 sources, turbo3 FA instances, `-fPIC`, `-O1` for turbo-quant |

### Modified Files — TurboQuant Type System

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | `GGML_TYPE_TURBO2/3/4_0` enum, `GGML_OP_TURBO_WHT` |
| `ggml/src/ggml.c` | Type table entries, `ggml_turbo_wht()` op constructor |
| `ggml/src/ggml-common.h` | `block_turbo3_0` struct (16 bytes per 32 values) |
| `ggml/src/ggml-quants.h` | `quantize_row_turbo3_0_ref`, `dequantize_row_turbo3_0` |
| `ggml-cuda/convert.cu` | Turbo3 dequant dispatch (fp16/fp32/bf16) |
| `ggml-cuda/set-rows.cu` | Turbo3 quantize dispatch (KV cache write) |
| `ggml-cuda/getrows.cu` | Turbo3 get_rows dispatch |
| `ggml-cuda/cpy.cu` + `cpy-utils.cuh` | Turbo3 copy + quantize block function |
| `ggml-cuda/dequantize.cuh` | `dequantize_turbo3_0` for get_rows |
| `ggml-cuda/fattn-common.cuh` | `vec_dot_fattn_vec_KQ_turbo3_0`, `dequantize_V_turbo3_0` |

### Modified Files — llama.cpp Integration

| File | Change |
|------|--------|
| `src/llama-graph.cpp` | WHT rotation for Q (forward) and V output (inverse) in `build_attn` |
| `src/llama-kv-cache.cpp` + `.h` | Rotation matrix tensor init, layer-adaptive mode |
| `src/llama-context.cpp` | Force flash attention for turbo types |
| `src/llama-memory.h` + hybrid variants | Turbo rotation accessors |
| `common/arg.cpp` | `--cache-type-k turbo3` CLI flag |
| `tools/llama-bench/llama-bench.cpp` | Turbo type name strings |

### Modified Files — CPU + Metal Backends

| File | Change |
|------|--------|
| `ggml-cpu/ggml-cpu.c` | Turbo type traits, `GGML_OP_TURBO_WHT` dispatch |
| `ggml-cpu/ops.cpp` + `ops.h` | CPU FWHT butterfly implementation |
| `ggml-cpu/quants.h` | CPU quant function declarations |
| `ggml-metal/ggml-metal-device.cpp/.h/.m` | Turbo WHT Metal pipeline |
| `ggml-metal/ggml-metal-ops.cpp/.h` | Turbo WHT Metal dispatch |
| `ggml-metal/ggml-metal-impl.h` | Turbo WHT kernel args struct |

### HIP/gfx906 Bug Fixes

These bugs were found and fixed during integration:

| Bug | Fix | File |
|-----|-----|------|
| `TURBO_WHT` falls through to `SOLVE_TRI` dimension check in `supports_op` | Add explicit `return true` before `SOLVE_TRI` | `ggml-cuda.cu` |
| Shadow cache dequant reads wrong strides for 4D K view | Swap `nb[1]`/`nb[2]` (head vs KV position stride) | `fattn.cu` |
| `hipMalloc` in WHT dispatch deadlocks on multi-GPU | Use FWHT kernel (no device alloc) | `turbo-quant.cu` |
| HIP compiler misoptimizes FWHT butterfly at `-O3` | Compile `turbo-quant.cu` at `-O1` | `CMakeLists.txt` |
| `cuda_fp16.h` not found on HIP | Add `#ifdef GGML_USE_HIP` guard for `hip_fp16.h` | `turbo-quant.cu` |
| Missing WHT rotation in `build_attn(attn_kv_iswa)` | Add turbo WHT calls matching `attn_kv` overload | `llama-graph.cpp` |
| HIP objects not position-independent | Add `-fPIC` to HIP compile flags | `CMakeLists.txt` |
| Turbo3 FA template instances missing from HIP build | Add glob patterns for turbo FA instances | `CMakeLists.txt` |

## Known Limitations

- **Shadow cache quality**: The bulk turbo3→fp16 dequant has a stride issue on some tensor layouts. The native FA vec kernel path (default) works correctly.
- **Hybrid SSM models**: Models with `GGML_OP_SOLVE_TRI` (e.g., Qwen3-Coder-Next) may hang on gfx906 during warmup. Use `--no-warmup`.
- **turbo3 V-cache**: Works on single GPU. Multi-GPU V-cache needs the native FA vec kernel path.

## Credits

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — Base
- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) — gfx906 Wave64 kernels
- [Madreag/turbo3-cuda](https://github.com/Madreag/turbo3-cuda) — TurboQuant CUDA port
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — TurboQuant CPU + Metal
- [Aaryan-Kapoor/llama.cpp](https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0) — Clean CPU TurboQuant
- TurboQuant paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
