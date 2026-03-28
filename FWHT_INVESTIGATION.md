# FWHT Investigation: TurboQuant Rotation Bug on HIP/gfx906

## Status
turbo3 KV cache works WITHOUT rotation (plain 3-bit scalar quantization).
With FWHT rotation enabled, output is corrupted. All other components verified working.

## What the Paper Says (arXiv 2504.19874)

TurboQuant uses a **random orthogonal rotation** before scalar quantization:
1. Rotate: `y = Pi @ x` (Pi is a d×d random orthogonal matrix)
2. Quantize each coordinate of `y` using Lloyd-Max optimal codebook
3. Dequantize: `x_hat = Pi^T @ centroid_vector * norm`

The rotation makes coordinates approximately independent and N(0, 1/d) distributed,
which is the optimal distribution for scalar quantization.

**For inner products**: Since Pi is orthogonal, `<Pi@q, Pi@k> = <q, k>`.
So Q gets the same rotation as K, and dot products are preserved.

## What the PyTorch Reference Does (turboquant-pytorch)

```python
# Full random orthogonal matrix via QR decomposition — O(d²)
G = torch.randn(d, d)
Q, R = torch.linalg.qr(G)
Pi = Q  # d×d orthogonal matrix

# Rotate
y = x @ Pi.T  # = Pi^T @ x

# Dequantize: inverse rotate
x_hat = centroids[indices] @ Pi  # = Pi @ centroids
```

This is the gold standard but costs O(d²) per vector — too slow for inference.

## What Madreag's llama.cpp Implementation Does

Uses **SRHT (Signed Randomized Hadamard Transform)** as a fast O(d log d) approximation:

```
R = D2 * H * D1 / sqrt(d)
```

Where:
- D1, D2 = diagonal matrices of random ±1 signs (stored as `d_turbo_wht_signs1/2`)
- H = Walsh-Hadamard matrix (computed via butterfly in O(d log d))
- 1/sqrt(d) = normalization factor

The FWHT butterfly implements H:
```c
for (h = 1; h < 128; h *= 2) {
    for (i = 0; i < 128; i += h * 2) {
        for (j = i; j < i + h; j++) {
            a = x[j]; b = x[j+h];
            x[j]   = a + b;
            x[j+h] = a - b;
        }
    }
}
```

**Design**: K is stored in rotated space. Dequant does NOT inverse-rotate.
Instead, Q is forward-rotated before attention, and output is inverse-rotated after.
This is mathematically equivalent and saves the inverse rotation at dequant time.

## What's Broken on HIP/gfx906

### Verified Working
- `__constant__` memory signs are readable (confirmed via printf from kernel)
- Sign values match between GPU and CPU (-1, 1, 1, -1, ...)
- Shared memory butterfly produces same (wrong) results as register butterfly
- `-O1` compiler flag produces slightly better but still wrong results
- Block format, packing, unpacking, shadow cache dequant — all correct
- Without rotation, turbo3 produces coherent output

### The Bug
The FWHT butterfly + sign application produces numerically wrong output on HIP.
Both the SET_ROWS kernel and the kernel_turbo_wht graph op are affected.

### What We Tried
1. **Register-based FWHT** — garbage output
2. **`-O1` compile flag** — slightly less garbage (coherent words but wrong content)
3. **Shared memory FWHT** — same garbage as registers
4. **Inlined sign arrays** — garbage (likely copy error in signs)
5. **`__constant__` signs** — signs read correctly, but result is still wrong

### What This Rules Out
- NOT a register spill issue (shared memory gives same result)
- NOT a `__constant__` memory issue (values confirmed correct)
- NOT a compiler optimization issue alone (`-O1` only marginally helps)
- NOT a block format issue (works without rotation)

## Root Cause Hypothesis

The FWHT butterfly is correct in isolation (it's too simple to be wrong).
The issue is likely in how the **sign application interacts with the butterfly**
or in the **normalization factor**.

Possible causes:
1. **Sign arrays are applied in wrong order** — the SRHT formula is
   `R = D2 * H * D1 / sqrt(d)`, meaning D1 is applied FIRST (before H),
   D2 is applied SECOND (after H). If D1 and D2 are swapped, the rotation
   is still orthogonal but DIFFERENT from what the graph-level Q rotation uses.

2. **V signs vs K signs mismatch** — SET_ROWS applies different signs for K
   (`signs1/signs2`) vs V (`signs1_v/signs2_v`). The graph inverse WHT uses
   V signs for direction=1. If there's any mismatch, V reconstruction breaks.

3. **Normalization mismatch** — SET_ROWS normalizes by L2 norm of the 128-group
   BEFORE rotation, stores corrected_norm. Shadow dequant uses centroid * norm.
   If the norm correction isn't computed correctly, the scale is wrong.

4. **Butterfly is correct but computes a DIFFERENT Hadamard matrix** than expected.
   The standard FWHT computes the Walsh (sequency) ordered Hadamard matrix.
   If the sign arrays were computed assuming natural (Hadamard) ordering,
   the rotation would be wrong.

## Proposed Fix Plan

### Approach 1: Use Dense Rotation Matrix (Slow but Correct)
Replace FWHT with full 128×128 matrix multiply using the precomputed rotation
matrix from `turbo-rotation-data.h` (already loaded into KV cache).

```c
// In kernel_set_rows_turbo3, replace FWHT with:
for (int i = 0; i < 128; i++) {
    float sum = 0;
    for (int j = 0; j < 128; j++) {
        sum += rotation_matrix[i * 128 + j] * normalized[j];
    }
    x[i] = sum;
}
```

Pro: Definitely correct (matches PyTorch reference exactly)
Con: O(d²) = 16384 FMAs per group, ~10x slower than FWHT

### Approach 2: Verify FWHT Against Known Values
Write a standalone test that:
1. Creates a known input vector
2. Applies FWHT on CPU (Python/C)
3. Applies FWHT on GPU (HIP kernel)
4. Compares results

This isolates whether the butterfly itself is wrong or the sign interaction.

### Approach 3: Match PyTorch Reference Exactly
1. Run the PyTorch `TurboQuantMSE` on a test vector, capture Pi, centroids, indices
2. Replicate the same operations in CUDA/HIP step by step
3. Find where the outputs diverge

### Approach 4: Replace SRHT with Hadamard-Rademacher (Different Sign Structure)
The current implementation uses D2 * H * D1 (two sign matrices).
Some SRHT variants use only one sign matrix: `R = H * D / sqrt(d)`.
This is simpler and may avoid the sign ordering issue.

## Recommended Next Steps (Priority Order)

1. **Approach 2 first** — standalone FWHT correctness test on HIP.
   If the butterfly output matches CPU, the bug is in sign application.
   If it doesn't match, the bug is in the HIP butterfly compilation.

2. **Approach 1 as fallback** — dense rotation is slow but will make turbo3
   fully functional while the FWHT issue is debugged.

3. **Approach 3 for validation** — once rotation works, validate against
   PyTorch end-to-end to ensure quality matches the paper's claims.

## Key Files

| File | Purpose |
|------|---------|
| `fork-old/ggml/src/ggml-cuda/turbo-quant.cu` | GPU kernels: SET_ROWS, TURBO_WHT, dequant |
| `fork-old/ggml/src/ggml-cuda/fattn.cu` | FA shadow cache dequant |
| `fork-old/src/llama-graph.cpp` | Graph-level Q/V WHT rotation insertion |
| `fork-old/ggml/src/ggml-cpu/ops.cpp` | CPU FWHT implementation |
| `turboquant-pytorch/turboquant.py` | PyTorch reference (full rotation) |
| `turboquant-pytorch/lloyd_max.py` | Lloyd-Max codebook computation |

## Current Working Configuration

```bash
llama-server -m model.gguf --cache-type-k turbo3 --cache-type-v f16
```

- turbo3 K-cache: 3.5 bits/value (no rotation, plain scalar quantization)
- f16 V-cache: 16 bits/value (standard)
- Quality: usable on 27B+ models, degrades on long sequences
- Speed: ~21 tok/s on 4x MI50
