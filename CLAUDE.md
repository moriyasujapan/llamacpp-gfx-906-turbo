# Roadmap: Rust LLM Inference Stack for gfx906 + TurboQuant

**Goal:** Rust API via `llama-cpp-2` over a custom llama.cpp fork that combines:
- Upstream llama.cpp (current, maintainable base)
- iacopPBK gfx906 fixes (Wave64 correctness + performance)
- TurboQuant turbo3 KV cache (4.6x memory compression)

**Why upstream instead of iacopPBK as base:**
- TurboQuant branches (TheTom, Madreag, Aaryan-Kapoor) are all based on recent upstream
- iacopPBK is pinned at build 7924; upstream is ~b8200+ and growing
- gfx906 patches are well-documented and can be applied as a clean patch series
- Stays maintainable — you can pull upstream fixes without huge rebases

---

## Phase 0 — Plumbing (1-2 days)

### 0.1 Fork upstream llama.cpp to your GitLab
```bash
git clone https://github.com/ggml-org/llama.cpp
# push to gitlab.raptors.pizza/raptor-1/llama.cpp-gfx906-turbo.git
```

### 0.2 Verify llama-cpp-sys-2 source override mechanism
The `llama-cpp-sys-2` build.rs resolves the llama.cpp source via:
```rust
let llama_src = env::var("LLAMA_CPP_PATH")
    .map(PathBuf::from)
    .unwrap_or_else(|_| /* bundled submodule path */);
```

**VERIFIED:** build.rs has NO `LLAMA_CPP_PATH` override — source is hardcoded
to `{CARGO_MANIFEST_DIR}/llama.cpp` submodule. Option A is the only option.
Feature flag is `rocm` (not `hipblas`), which sets `GGML_HIP=ON` in CMake.

**Option A:** Fork `utilityai/llama-cpp-rs`, change the llama.cpp
submodule to point at your fork. One-time setup, fully self-contained.
Your `Cargo.toml`:
```toml
[dependencies]
llama-cpp-2 = { git = "https://gitlab.raptors.pizza/raptor-1/llama-cpp-rs.git" }
```

### 0.3 Baseline build test
Build upstream llama.cpp via llama-cpp-rs with hipblas feature, gfx906 target.
Expect it to crash on your hardware — this is the known upstream baseline.
```bash
HIPCXX=$(hipconfig -l)/clang HIP_PATH=$(hipconfig -R) \
  cargo build --release --features hipblas
```

---

## Phase 1 — gfx906 Fixes (3-5 days)

Apply iacopPBK's patches to your upstream fork. The patches fall into two categories:

### 1.1 New directory — zero conflict, copy verbatim
```
ggml/src/ggml-cuda/gfx906/          <- entire directory, ~15 files
├── gfx906-common.cuh               Wave64 DPP warp reductions
├── gfx906-config.h                 Feature toggles
├── attention/
│   ├── fattn-q8.cuh/.cu            Q8 FlashAttention kernel
│   ├── rope.cuh                    Optimized RoPE
│   └── instances/
├── fused/
│   ├── gather-q8.cuh/.cu
│   ├── graph-fusion.cuh
│   ├── norm-fused-q8.cuh/.cu
│   └── mmq-prequantized.cuh
├── matmul/
│   ├── mmvq-q4_0.cuh               Warp-cooperative MMVQ (MoE fix)
│   ├── mmvq-q4_1.cuh
│   ├── mmvq-q8_0.cuh
│   ├── mmq.cuh / mmq-prefetch.cuh  Software pipelining
│   ├── mmf.cuh / sgemm.cuh
└── quantize/
    ├── epilogue.cuh
    ├── q8-cache.cuh
    └── vecdotq.cuh
```
Action: `cp -r iacopPBK/ggml/src/ggml-cuda/gfx906 your-fork/ggml/src/ggml-cuda/`

### 1.2 Modified files — apply as diffs against upstream
These files have targeted changes. Use `git diff` between iacopPBK and its
upstream base commit to extract clean patches, then apply to current upstream.

| File | iacopPBK change | Conflict risk |
|------|----------------|---------------|
| `ggml-cuda/fattn-common.cuh` | GCN tile sizes, thread counts | Medium |
| `ggml-cuda/fattn.cu` | Q8 tile kernel selection | Medium |
| `ggml-cuda/mmq.cu` | Vectorized loads dispatch | Low |
| `ggml-cuda/mmvq.cu` | Half-warp MoE dispatch + Wave64 warp ID fix | Low |
| `ggml-cuda/common.cuh` | DPP shuffle XOR dispatch | Low |
| `CMakeLists.txt` | Add gfx906/ dir to HIP sources | Low |

### 1.3 Test gfx906 correctness
```bash
./llama-bench -m qwen3-coder.gguf -ngl 999 -fa 1 \
  --cache-type-k q8_0 --cache-type-v q8_0
# Must not crash, must match iacopPBK throughput (~30 tok/s)
```

---

## Phase 2 — TurboQuant (3-5 days)

Apply TurboQuant on top of Phase 1. Source: Madreag/turbo3-cuda for file list,
TheTom/turboquant_plus for CPU + block format, Aaryan-Kapoor for clean CPU impl.

### 2.1 New files — copy verbatim, HIP-translate the CUDA kernel

**Pure new files (no conflict):**
```
ggml/src/ggml-cuda/turbo-quant.cuh       Block format, codebook constants
ggml/src/ggml-cuda/turbo-quant.cu -> .hip FWHT + quant/dequant kernels
ggml/src/ggml-cuda/template-instances/
  fattn-vec-instance-turbo3_0-f16.hip
  fattn-vec-instance-turbo3_0-turbo3_0.hip
```

**HIP translation of turbo-quant.cu (mechanical):**
- `cudaMemcpy*` -> `hipMemcpy*`
- No warp shuffles in FWHT — butterfly ops are pure arithmetic, Wave64-neutral
- Any warp reduction: use `gfx906_warp_reduce()` from `gfx906-common.cuh`
- Block size: use 64 (one wavefront) for FWHT kernel

### 2.2 Additive edits — low conflict, extend switch/case statements

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | Add `GGML_TYPE_TQ3_0 = 36` enum value |
| `ggml/src/ggml.c` | Type table entry + CPU quant/dequant functions |
| `ggml-cuda/convert.cu` | Dequant dispatch for turbo3 |
| `ggml-cuda/set-rows.cu` | Quantize dispatch (KV cache write path) |
| `ggml-cuda/getrows.cu` | get_rows dispatch |
| `ggml-cuda/cpy.cu` | Same-type raw copy |
| `ggml-cuda/cpy-utils.cuh` | `quantize_f32_turbo3_0_block` device fn |
| `ggml-cuda/dequantize.cuh` | `dequantize_turbo3_0` for get_rows |
| `common/arg.cpp` | `--cache-type-k turbo3` CLI flag |

All additive, no overlap with gfx906 changes.

### 2.3 High-attention merges — overlap with Phase 1 files

| File | gfx906 change | TurboQuant change | Strategy |
|------|--------------|-------------------|----------|
| `fattn-common.cuh` | GCN tile sizes | turbo3 vec_dot/dequant fns | Add turbo3 functions after gfx906 block |
| `fattn-vec.cuh` | Wave64 thread config | turbo3 thread/block case | Add case alongside gfx906 cases |
| `fattn.cu` | Q8 tile selection | turbo3 dispatch entries | Add turbo3 cases in separate block |
| `ggml-cuda.cu` | MMVQ Wave64 fix, MoE dispatch | TURBO_WHT op, supports_op, MUL_MAT exclusion | All additive to different parts |

### 2.4 CMakeLists.txt update
```cmake
file(GLOB TURBO_FATTN_INSTANCES
  ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo3_0*.hip)
list(APPEND GGML_HIP_SOURCES ${TURBO_FATTN_INSTANCES})
list(APPEND GGML_HIP_SOURCES ggml/src/ggml-cuda/turbo-quant.hip)
```

### 2.5 Test TurboQuant
```bash
./llama-server -m qwen3-coder.gguf -ngl 0 \
  --cache-type-k turbo3 --cache-type-v turbo3 -c 32000
./llama-bench -m qwen3-coder.gguf -ngl 999 -fa 1 \
  --cache-type-k turbo3 --cache-type-v turbo3
```

---

## Phase 3 — Rust Integration (1-2 days)

### 3.1 Wire up llama-cpp-rs to your fork
In your fork of `utilityai/llama-cpp-rs`:
```bash
git submodule set-url llama.cpp https://gitlab.raptors.pizza/raptor-1/llama.cpp-gfx906-turbo.git
git submodule update --remote
```

### 3.2 Cargo.toml in your project
```toml
[dependencies]
llama-cpp-2 = {
  git = "https://gitlab.raptors.pizza/raptor-1/llama-cpp-rs.git",
  features = ["hipblas"]
}
```

### 3.3 Build and smoke test
```bash
HIPCXX=$(hipconfig -l)/clang \
HIP_PATH=$(hipconfig -R) \
AMDGPU_TARGETS=gfx906 \
  cargo build --release --features hipblas
```

---

## Phase 4 — Maintenance Strategy

Monthly upstream sync. gfx906 directory and turbo3 new files are purely additive.
Only `fattn-common.cuh` and `ggml-cuda.cu` need judgment each sync.
Once turbo3 merges upstream (likely Q2-Q3 2026), drop Phase 2 entirely.

---

## Key Repositories

| Repo | Clone URL | Purpose |
|------|-----------|---------|
| `ggml-org/llama.cpp` | `https://github.com/ggml-org/llama.cpp` | Base (current upstream) |
| `iacopPBK/llama.cpp-gfx906` | `https://github.com/iacopPBK/llama.cpp-gfx906.git` | gfx906 patch source |
| `TheTom/llama-cpp-turboquant` | `https://github.com/TheTom/llama-cpp-turboquant.git` branch `feature/turboquant-kv-cache` | TurboQuant CPU + block format spec |
| `Madreag/turbo3-cuda` | `https://github.com/Madreag/turbo3-cuda.git` | TurboQuant CUDA kernel reference |
| `Aaryan-Kapoor/llama.cpp` | `https://github.com/Aaryan-Kapoor/llama.cpp.git` branch `turboquant-tq3_0` | Clean minimal CPU TurboQuant |
| `utilityai/llama-cpp-rs` | `https://github.com/utilityai/llama-cpp-rs.git` | Rust bindings base |
