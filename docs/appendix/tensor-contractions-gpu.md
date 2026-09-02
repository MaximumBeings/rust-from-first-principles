# Appendix E: Tensor Contractions, From First Principles (GPU)

> "A contraction's free indices are what a GPU grid is for. A contraction's contracted indices are what a GPU thread's own for-loop is for. Once you see the mapping, a tensor-contraction kernel stops looking like a special case of matmul and starts looking like the general pattern matmul was always a special case of."

## What you will understand

By the end of this appendix you will be able to:

- Map the free/contracted index split from Appendix D directly onto CUDA's thread/loop split -- one thread per output element, one loop per contracted index.
- Write a shared-memory-tiled contraction kernel that generalizes the classic tiled-matmul optimization to an arbitrary contracted axis, including its boundary case.
- Verify a CUDA kernel's index arithmetic on a machine with no GPU at all -- and, in this edition, no CUDA Toolkit at all either -- by emulating its exact block/thread loop structure in ordinary host Rust.
- Extend a single-shared-axis kernel to contract over multiple shared axes at once, using the same fixed-size-array technique Appendix D's Section D.2 introduced for scalar indexing.
- Read the FLOP-count and arithmetic-intensity formulas that decide whether a contraction kernel is compute-bound or memory-bound, and know which of this appendix's kernels sits on which side.

## What you need to know first

This appendix builds directly on Appendix D -- read it first if you haven't. It also assumes the `cudarc` dynamic-loading pattern this book has used since Chapter 4, and the honest toolchain-gap framing Appendix B established: writing CUDA kernels from Rust via `cudarc` means writing the kernel body in ordinary CUDA C, compiled at runtime by NVRTC -- `cudarc`'s own contribution is host-side orchestration (allocate, copy, launch, copy back), not a Rust kernel language. There is no way around this with `cudarc` specifically; a kernel written in native Rust would need a different toolchain entirely (such as `rust-cuda`'s `rustc_codegen_nvvm`), not used in this book.

This sandbox's environment is a strictly deeper gap than the CUDA C++ edition's own Appendix I. That edition has a real CUDA Toolkit -- `nvcc` compiles every kernel for real -- but no physical GPU, so its device calls fail cleanly with `cudaGetErrorString`'s "no CUDA-capable device is detected." This sandbox has neither a device, nor a CUDA Toolkit, nor a loadable CUDA driver or NVRTC library at all (confirmed directly in Appendix B's toolchain checks). So every genuine CUDA call this appendix makes -- `CudaContext::new`, `compile_ptx` -- panics at the dynamic-loading step, before ever reaching a device query, and that panic is reported honestly rather than smoothed over. What carries this appendix's real evidentiary weight is exactly what it already carries in the CUDA C++ edition, which also has no device to run these kernels on: independent host references, and -- more thoroughly than that edition's own runnable code goes -- direct host-side emulation of each kernel's exact loop structure, genuinely run rather than only described.

## E.1 From CPU Contraction to One Thread Per Output Element

Appendix D's `contract()` splits its work into two nested loops: an outer walk over every combination of **free indices** (the ones that survive into the output), and for each one, an inner walk over every combination of **contracted indices** (the ones summed away). CUDA's own execution model already has a natural home for exactly that split: the **grid of threads** covers the free-index space -- one thread per output element -- and each thread's own sequential `for` loop covers the contracted-index space, exactly mirroring `contract()`'s inner `for_each_index` call, just run by one hardware thread instead of one CPU call stack frame.

`01_contraction_kernel.rs` implements this directly for the single-shared-axis case (matmul): thread `(row, col)` owns output element `C[row][col]`, and walks the entire contracted axis of length `K` by itself. The kernel body -- still CUDA C, per the note above -- is byte-for-byte the same as the CUDA C++ edition's own:

```cpp
extern "C" __global__ void contraction_kernel(const double* A, const double* B, double* C,
                                               int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    double acc = 0.0;
    for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
```

### Worked Example E.1.1 — A genuine host reference, and a genuine, honestly-reported toolchain attempt

```
=== Tensor contraction on the GPU: matmul as one-thread-per-output-element ===

host reference computed for a 8x6 contracted with 6x5 -> 8x5 output.
reference C[0][0]=74.9900  C[7][4]=144.6100

--- genuine attempt: a real device is the necessary first step before any
--- kernel launch could happen at all ---
CudaContext::new(0) panicked (no CUDA driver reachable, same as every
device attempt since Chapter 4): Unable to dynamically load the "cuda" shared library - searched for library names: ["libcuda.so", "libcuda64.so", "libcuda64_12.so", "libcuda64_126.so", "libcuda64_126_0.so", "libcuda64_120_6.so", "libcuda64_10.so", "libcuda64_11.so", "libcuda64_12.so", "libcuda64_120_0.so", "libcuda64_9.so", "libcuda.so.12", "libcuda.so.12", "libcuda.so.11", "libcuda.so.10", "libcuda.so.9", "libcuda.so.1", "libnvcuda.so", "libnvcuda64.so", "libnvcuda64_12.so", "libnvcuda64_126.so", "libnvcuda64_126_0.so", "libnvcuda64_120_6.so", "libnvcuda64_10.so", "libnvcuda64_11.so", "libnvcuda64_12.so", "libnvcuda64_120_0.so", "libnvcuda64_9.so", "libnvcuda.so.12", "libnvcuda.so.12", "libnvcuda.so.11", "libnvcuda.so.10", "libnvcuda.so.9", "libnvcuda.so.1"]. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.

--- genuine NVRTC compilation attempt, the exact kernel source above ---
compile_ptx PANICKED (NVRTC unreachable, consistent since Chapter 4):
  Unable to dynamically load the "nvrtc" shared library - searched for library names: ["libnvrtc.so", "libnvrtc64.so", "libnvrtc64_12.so", "libnvrtc64_126.so", "libnvrtc64_126_0.so", "libnvrtc64_120_6.so", "libnvrtc64_10.so", "libnvrtc64_11.so", "libnvrtc64_12.so", "libnvrtc64_120_0.so", "libnvrtc64_9.so", "libnvrtc.so.12", "libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so.10", "libnvrtc.so.9", "libnvrtc.so.1"]. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.

(skipping allocation/memcpy/launch -- no device or compiled kernel to run them against)

--- what this means for this appendix's evidence ---
The host reference above is an ordinary Rust triple loop, independent of the
kernel source; the kernel's arithmetic is line-for-line the same as that
reference's innermost loop, with `row`/`col` supplied by `blockIdx`/`threadIdx`
instead of a `for` statement. On a machine with an actual CUDA device, the two
would agree for the same reason Appendix D's Rust `contract()` already agreed
with `numpy`'s `@` operator: it's the same sum, computed a different way.
```

Compiled cleanly (zero warnings, both the `dev` and `release` profiles) with `cargo build`. The host reference -- an ordinary Rust triple loop, independent of the kernel source -- computes the identical sum the kernel's innermost loop would, with `row`/`col` supplied by `blockIdx`/`threadIdx` instead of a `for` statement; on a machine with an actual CUDA device, the two would agree for the same reason Appendix D's Rust `contract()` already agreed with `numpy`'s `@` operator on the same underlying mathematics.

### Formulas and Key Terms

- **Thread-per-output-element** — the mapping this section uses: each of the `M x N` output positions gets exactly one thread, which alone is responsible for that position's entire contracted-axis sum.
- **Grid-stride vs. one-shot launch** — this kernel launches exactly enough threads to cover the output once (`grid = ceil(N/block.x) x ceil(M/block.y)`); the `if (row >= M || col >= N) return;` guard exists because that ceiling division can request more threads than there are output elements, and the excess must do nothing rather than read or write out of bounds.
- **FLOP count** — identical to Appendix D's formula, since the mathematics hasn't changed: `M·N·K` multiply-adds for this kernel's matmul-shaped contraction.
- **Arithmetic intensity** — FLOPs performed per byte of data moved from global memory. This kernel's innermost loop re-reads `A[row][*]` and `B[*][col]` from global memory on every single output element that shares that row or column -- Section E.2 exists specifically to raise this ratio.

## E.2 Tiling the Contracted Axis in Shared Memory

The kernel in Section E.1 reads each element of `A` and `B` from global memory many times over: every thread in output row `row` re-reads all of `A`'s row `row`, and every thread in output column `col` re-reads all of `B`'s column `col`. **Tiling** loads a `TILE x TILE` block of `A` and a `TILE x TILE` block of `B` into fast on-chip shared memory once, lets every thread in the block reuse it `TILE` times, then advances to the next tile along the contracted axis -- the same tiling idea this book's core chapters already introduced for plain matrix multiplication, framed here explicitly as "tiling along a contraction's contracted axis," since nothing about the technique is actually specific to matmul.

```cpp
#define TILE 16
extern "C" __global__ void tiled_contraction_kernel(const double* A, const double* B, double* C,
                                                      int M, int K, int N) {
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    double acc = 0.0;
    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;
        __syncthreads();
        for (int kk = 0; kk < TILE; kk++) acc += tileA[threadIdx.y][kk] * tileB[kk][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```

`02_tiled_contraction_kernel.rs` deliberately picks `K=23`, **not** a multiple of `TILE=16`, so the boundary-padding branch (`? A[...] : 0.0`) is a genuine part of what this section is about, rather than an untested edge case:

```
=== Tiling a contraction along its contracted axis ===

M=37, K=23, N=19  (K=23 is NOT a multiple of TILE=16 -- exercises the padding branch)

--- genuine attempt: a real device is the necessary first step before any
--- kernel launch could happen at all ---
CudaContext::new(0) panicked (no CUDA driver reachable, same as every
device attempt since Chapter 4): Unable to dynamically load the "cuda" shared library - searched for library names: ["libcuda.so", "libcuda64.so", "libcuda64_12.so", "libcuda64_126.so", "libcuda64_126_0.so", "libcuda64_120_6.so", "libcuda64_10.so", "libcuda64_11.so", "libcuda64_12.so", "libcuda64_120_0.so", "libcuda64_9.so", "libcuda.so.12", "libcuda.so.12", "libcuda.so.11", "libcuda.so.10", "libcuda.so.9", "libcuda.so.1", "libnvcuda.so", "libnvcuda64.so", "libnvcuda64_12.so", "libnvcuda64_126.so", "libnvcuda64_126_0.so", "libnvcuda64_120_6.so", "libnvcuda64_10.so", "libnvcuda64_11.so", "libnvcuda64_12.so", "libnvcuda64_120_0.so", "libnvcuda64_9.so", "libnvcuda.so.12", "libnvcuda.so.12", "libnvcuda.so.11", "libnvcuda.so.10", "libnvcuda.so.9", "libnvcuda.so.1"]. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.

--- genuine NVRTC compilation attempt, the exact kernel source above ---
compile_ptx PANICKED (NVRTC unreachable, consistent since Chapter 4):
  Unable to dynamically load the "nvrtc" shared library - searched for library names: ["libnvrtc.so", "libnvrtc64.so", "libnvrtc64_12.so", "libnvrtc64_126.so", "libnvrtc64_126_0.so", "libnvrtc64_120_6.so", "libnvrtc64_10.so", "libnvrtc64_11.so", "libnvrtc64_12.so", "libnvrtc64_120_0.so", "libnvrtc64_9.so", "libnvrtc.so.12", "libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so.10", "libnvrtc.so.9", "libnvrtc.so.1"]. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.

(skipping allocation/memcpy/launch -- no device or compiled kernel to run them against)

shared memory used per block: 2 x 16 x 16 x sizeof(f64) = 4096 bytes
global memory reads per output element WITHOUT tiling: K = 23
global memory reads per output element WITH this tiling: K / TILE (amortized) = 1.44

(host reference above -- C[0][0]=732.6200, C[36][18]=335.0600 -- is genuinely
computed here; whether the TILED kernel's own arithmetic, including its
boundary-padding branch, agrees with it is what Section E.2's companion
file below actually checks, without needing a GPU at all.)
```

### Worked Example E.2.1 — Verifying a kernel's logic with no GPU, and no CUDA Toolkit, present

Compiling would be necessary but not sufficient evidence that the boundary-padding branch is correct, if compilation were even reachable here -- and it isn't (see Worked Example E.1.1). `02b_tiled_kernel_host_emulation.rs` sidesteps needing NVRTC or a device at all: it **emulates the kernel's exact block/thread loop structure in plain host Rust** -- the same grid/block loop nesting, the same tile-load conditions, computed one CPU thread at a time instead of thousands of hardware threads at once, at the identical `M=37, K=23, N=19` shape:

```
=== Verifying the tiled kernel's boundary logic WITHOUT a GPU ===

M=37, K=23, N=19 (K/TILE = 1.438 -- the fractional last tile is the risk)

max |reference - emulated tiled kernel| over all 703 output elements: 0.000e0
PASS -- the tiling logic, INCLUDING the K-not-a-multiple-of-TILE padding
branch, is exactly correct. Whatever runs on an actual device is running
this same index arithmetic, just spread across real hardware threads
instead of one CPU thread looping over blocks in sequence.
```

Zero difference across all 703 output elements, including every element whose tile-load conditions fall in the fractional last tile. This is this appendix's answer to "how do you test device code with no device and no toolchain to compile it with": don't test the *device*, test the *index arithmetic*, by running it -- unmodified in structure, just relocated to a host loop -- somewhere it CAN run.

### Formulas and Key Terms

- **Occupancy** — the fraction of a GPU's maximum possible resident threads that are actually scheduled at once, limited by each thread's register and shared-memory usage; this kernel trades some occupancy (each block reserves `2 x TILE x TILE x 8` bytes of shared memory) for a large reduction in global-memory traffic.
- **Tile size** — `TILE`, chosen to balance shared-memory usage per block (which caps how many blocks fit on a streaming multiprocessor at once) against how many times each loaded value gets reused (`TILE` times, once per element of the tile's other operand).
- **Memory coalescing** — when consecutive threads in a warp read consecutive memory addresses, the hardware can service the whole warp with fewer transactions; both kernels in this appendix index `A` and `B` so that `threadIdx.x` varies the innermost (contiguous) array subscript, keeping loads coalesced.
- **Roofline model** — the concept that a kernel's achievable performance is capped by the *lower* of two ceilings: the hardware's peak compute throughput, or its peak memory bandwidth times the kernel's arithmetic intensity. Section E.1's untiled kernel sits further toward the memory-bound side of this tradeoff than Section E.2's tiled kernel, which is the entire motivation for tiling in the first place.

## E.3 Contracting Over Multiple Axes on the GPU

Generalizing Section E.1's kernel from one shared axis to several follows the same idea Appendix D's `contract()` used: split each tensor's axes into "free" and "contracted," walk the free-index space (now the CUDA grid) and the contracted-index space (now a device-side loop) separately. The one real difference is the tool available to do the walking. Host Rust could use the recursive `for_each_index()` helper from Appendix D, Section D.3; device code cannot -- CUDA kernels have no heap-allocated `Vec`, and recursion to a depth fixed only at runtime (a tensor's rank) doesn't fit how a GPU schedules thousands of concurrent, identical-instruction threads cheaply. `03_multi_axis_contraction_kernel.rs` instead reaches for the *other* indexing technique Appendix D's Section D.2 already introduced: fixed-size arrays (capped at a compile-time `MAX_RANK`) plus iterative divmod unraveling.

```cpp
#define MAX_RANK 4

struct TensorMeta {
    int shape[MAX_RANK];
    long long strides[MAX_RANK];
    int rank;
};

extern "C" __global__ void multi_axis_contraction_kernel(
    const double* A, const double* B, double* C,
    TensorMeta metaA, TensorMeta metaB, TensorMeta metaC,
    int free_a[MAX_RANK], int free_a_count,
    int free_b[MAX_RANK], int free_b_count,
    int axes_a[MAX_RANK], int axes_b[MAX_RANK],
    int contracted_shape[MAX_RANK], long long contracted_strides[MAX_RANK],
    int contracted_rank, long long contracted_total)
{
    long long out_lin = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long out_total = 1;
    for (int i = 0; i < metaC.rank; i++) out_total *= metaC.shape[i];
    if (out_lin >= out_total) return;

    int out_idx[MAX_RANK];
    long long rem = out_lin;
    for (int i = 0; i < metaC.rank; i++) { out_idx[i] = (int)(rem / metaC.strides[i]); rem %= metaC.strides[i]; }

    int idx_a[MAX_RANK] = {0}, idx_b[MAX_RANK] = {0};
    for (int i = 0; i < free_a_count; i++) idx_a[free_a[i]] = out_idx[i];
    for (int j = 0; j < free_b_count; j++) idx_b[free_b[j]] = out_idx[free_a_count + j];

    double acc = 0.0;
    for (long long c_lin = 0; c_lin < contracted_total; c_lin++) {
        int c_idx[MAX_RANK];
        long long crem = c_lin;
        for (int i = 0; i < contracted_rank; i++) { c_idx[i] = (int)(crem / contracted_strides[i]); crem %= contracted_strides[i]; }
        for (int k = 0; k < contracted_rank; k++) { idx_a[axes_a[k]] = c_idx[k]; idx_b[axes_b[k]] = c_idx[k]; }

        long long lin_a = 0; for (int i = 0; i < metaA.rank; i++) lin_a += (long long)idx_a[i] * metaA.strides[i];
        long long lin_b = 0; for (int i = 0; i < metaB.rank; i++) lin_b += (long long)idx_b[i] * metaB.strides[i];
        acc += A[lin_a] * B[lin_b];
    }

    long long lin_c = 0;
    for (int i = 0; i < metaC.rank; i++) lin_c += (long long)out_idx[i] * metaC.strides[i];
    C[lin_c] = acc;
}
```

### Worked Example E.3.1 — The double contraction from Appendix D, on the GPU

`03_multi_axis_contraction_kernel.rs` sets up the metadata for the exact same problem Appendix D's Section D.4 already verified against `numpy.tensordot` -- `A` shape `[2,3,4]` contracted with `B` shape `[3,4,5]` over axes `{1,2}`/`{0,1}` -- and reuses that already-verified result as its own reference. This file goes further than the CUDA C++ edition's own runnable code does at this point: rather than leaving "translating the kernel body into a host function" as a claim made in prose, `emulate_multi_axis_kernel` is that translation, written out in full, genuinely run here, once per output index, using the identical `free_a`/`free_b`/`axes_a`/`axes_b`/`contracted_shape`/`contracted_strides` metadata the kernel itself would receive as launch arguments:

```
=== A double contraction on the GPU: TWO shared axes, one thread per output ===

host reference (identical inputs to Appendix D's 03_double_contraction.rs,
already cross-checked there against numpy.tensordot):
  10.00 5.00 0.00 -5.00 -10.00 2.00 1.00 0.00 -1.00 -2.00 

--- genuine attempt: a real device is the necessary first step before any
--- kernel launch could happen at all ---
CudaContext::new(0) panicked (no CUDA driver reachable, same as every
device attempt since Chapter 4): Unable to dynamically load the "cuda" shared library - searched for library names: ["libcuda.so", "libcuda64.so", "libcuda64_12.so", "libcuda64_126.so", "libcuda64_126_0.so", "libcuda64_120_6.so", "libcuda64_10.so", "libcuda64_11.so", "libcuda64_12.so", "libcuda64_120_0.so", "libcuda64_9.so", "libcuda.so.12", "libcuda.so.12", "libcuda.so.11", "libcuda.so.10", "libcuda.so.9", "libcuda.so.1", "libnvcuda.so", "libnvcuda64.so", "libnvcuda64_12.so", "libnvcuda64_126.so", "libnvcuda64_126_0.so", "libnvcuda64_120_6.so", "libnvcuda64_10.so", "libnvcuda64_11.so", "libnvcuda64_12.so", "libnvcuda64_120_0.so", "libnvcuda64_9.so", "libnvcuda.so.12", "libnvcuda.so.12", "libnvcuda.so.11", "libnvcuda.so.10", "libnvcuda.so.9", "libnvcuda.so.1"]. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.

--- genuine NVRTC compilation attempt, the exact kernel source above ---
compile_ptx PANICKED (NVRTC unreachable, consistent since Chapter 4):
  Unable to dynamically load the "nvrtc" shared library - searched for library names: ["libnvrtc.so", "libnvrtc64.so", "libnvrtc64_12.so", "libnvrtc64_126.so", "libnvrtc64_126_0.so", "libnvrtc64_120_6.so", "libnvrtc64_10.so", "libnvrtc64_11.so", "libnvrtc64_12.so", "libnvrtc64_120_0.so", "libnvrtc64_9.so", "libnvrtc.so.12", "libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so.10", "libnvrtc.so.9", "libnvrtc.so.1"]. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.

(skipping allocation/memcpy/launch -- no device or compiled kernel to run them against)

--- host emulation of the kernel's own exact metadata-driven index
--- arithmetic above, run once per output index (not just described) ---
  10.00 5.00 0.00 -5.00 -10.00 2.00 1.00 0.00 -1.00 -2.00 

max |reference - emulated multi-axis kernel| over all 10 output elements: 0.000e0
PASS -- the metadata-driven index arithmetic (free_a/free_b/axes_a/axes_b/
contracted_shape/contracted_strides, exactly as the kernel receives them)
is exactly correct, confirmed by running it, not by inspecting it.
```

Because this kernel's metadata-driven indexing is more intricate than Section E.1's plain `row`/`col`, its correctness was not left to inspection: the host emulation above reproduces all ten reference values exactly, confirming that the metadata this kernel would receive is assembled correctly before ever needing an actual device to prove it on.

### Formulas and Key Terms

- **MAX_RANK** — the compile-time cap (4, here) on how many axes a tensor's metadata can describe; unlike the CPU appendix's `Vec`-based `contract()`, which handles any rank at runtime, a CUDA kernel's per-thread arrays must have a size fixed before compilation.
- **Metadata marshaling** — passing `TensorMeta` structs and fixed-size arrays by value into a kernel launch is itself a real, genuine step in a working CUDA program (packing host-side shape/stride bookkeeping into a form a device function can read), separate from the index arithmetic those structs drive once inside the kernel.

## E.4 In Practice: cuBLAS and cuTENSOR

Every kernel in this appendix is written by hand, for the same reason this entire book writes kernels by hand: to make the mapping between mathematics and hardware visible. Production code contracting large tensors does not usually hand-roll this logic -- NVIDIA ships **cuBLAS** for the matrix-multiply case and **cuTENSOR** for general multi-axis contractions, both tuned per-architecture with tiling strategies, register blocking, and mixed-precision paths far beyond what a single `#define TILE 16` can express. Reaching either from Rust would mean the same kind of orchestration this appendix's own kernels use `cudarc` for -- loading the library's shared object and calling its C ABI -- rather than a native Rust rewrite of cuBLAS or cuTENSOR themselves; the algorithm stays vendor-tuned CUDA C either way, only the host-side glue changes language. Knowing how the naive and tiled kernels in this appendix work is what makes it possible to read cuBLAS/cuTENSOR's documented behavior (workspace requirements, supported layouts, tensor-core code paths) and understand *why* those choices exist, rather than treating the library as a black box. This appendix makes no performance claims about either library -- there is no GPU in this environment to benchmark them on, and no CUDA Toolkit to even attempt loading them with -- only the factual point that they exist and are where a real project should look once the underlying algorithm, covered here, is understood.

## E.5 Complete Runnable Code

### File: 01_contraction_kernel.rs

```rust
// E.1: From CPU Contraction to One Thread Per Output Element.
//
// Appendix D's `contract()` splits its work into an outer walk over free
// indices and, for each one, an inner walk over contracted indices. CUDA's
// execution model already has a natural home for exactly that split: the
// GRID of threads covers the free-index space (one thread per output
// element), and each thread's own sequential loop covers the contracted-
// index space -- the same `for_each_index` idea from Appendix D, Section
// D.3, just run by one hardware thread instead of one CPU call frame.
//
// A genuine point worth being explicit about: writing CUDA kernels from
// Rust via `cudarc` does not mean writing kernels IN Rust. The kernel
// below is CUDA C, compiled at runtime by NVRTC (the same dynamic-loading
// path this book has used since Chapter 4) -- `cudarc`'s role is entirely
// host-side orchestration (allocate, copy, launch, copy back), identical
// in spirit to what the CUDA C++ edition's own host `main()` does. There
// is no way to write a `__global__` kernel body in native Rust through
// this crate; the kernel language is still CUDA C either way.
//
// This sandbox's environment is a strictly deeper gap than the CUDA C++
// edition's own: that edition has a real CUDA Toolkit (`nvcc` compiles
// every kernel for real) but no physical GPU, so its device calls fail
// cleanly with `cudaGetErrorString`'s "no CUDA-capable device is
// detected." This sandbox has neither a device NOR a CUDA Toolkit NOR a
// loadable CUDA driver/NVRTC library at all (confirmed directly in
// Appendix B) -- so the honest evidence here is a genuine host reference,
// computed and printed for real, plus a genuine attempt at every CUDA
// call this appendix can make, with its real failure reported rather
// than skipped.
use cudarc::driver::CudaContext;
use cudarc::nvrtc::compile_ptx;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn catch_cuda<F: FnOnce() -> R, R>(f: F) -> Result<R, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = payload.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "non-string panic payload".to_string()
        }
    })
}

/// Byte-for-byte the same kernel as the CUDA C++ edition's own Appendix
/// I, Section I.1: one thread owns one output element, and walks the
/// entire contracted axis (length K) by itself.
const CONTRACTION_KERNEL_SRC: &str = r#"
extern "C" __global__ void contraction_kernel(const double* A, const double* B, double* C,
                                               int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    double acc = 0.0;
    for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
"#;

fn reference_contraction(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

fn main() {
    println!("=== Tensor contraction on the GPU: matmul as one-thread-per-output-element ===\n");

    let (m, k, n) = (8usize, 6usize, 5usize);
    let mut h_a = vec![0.0f64; m * k];
    let mut h_b = vec![0.0f64; k * n];
    let mut h_c_ref = vec![0.0f64; m * n];

    let mut seed: u32 = 777;
    for v in h_a.iter_mut() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (seed % 100) as f64 / 10.0;
    }
    for v in h_b.iter_mut() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (seed % 100) as f64 / 10.0;
    }

    reference_contraction(&h_a, &h_b, &mut h_c_ref, m, k, n);
    println!("host reference computed for a {m}x{k} contracted with {k}x{n} -> {m}x{n} output.");
    println!(
        "reference C[0][0]={:.4}  C[{}][{}]={:.4}\n",
        h_c_ref[0],
        m - 1,
        n - 1,
        h_c_ref[(m - 1) * n + (n - 1)]
    );

    println!("--- genuine attempt: a real device is the necessary first step before any");
    println!("--- kernel launch could happen at all ---");
    match catch_cuda(|| CudaContext::new(0)) {
        Ok(Ok(_ctx)) => println!("CudaContext::new(0) succeeded -- a real device is present."),
        Ok(Err(e)) => println!("CudaContext::new(0) returned a clean error: {e}"),
        Err(panic_msg) => {
            println!("CudaContext::new(0) panicked (no CUDA driver reachable, same as every");
            println!("device attempt since Chapter 4): {panic_msg}");
        }
    }

    println!("\n--- genuine NVRTC compilation attempt, the exact kernel source above ---");
    match catch_cuda(|| compile_ptx(CONTRACTION_KERNEL_SRC)) {
        Ok(Ok(_ptx)) => println!("compile_ptx succeeded -- NVRTC is genuinely reachable here."),
        Ok(Err(e)) => println!("compile_ptx returned a clean NVRTC error: {e}"),
        Err(panic_msg) => {
            println!("compile_ptx PANICKED (NVRTC unreachable, consistent since Chapter 4):");
            println!("  {panic_msg}");
        }
    }

    println!("\n(skipping allocation/memcpy/launch -- no device or compiled kernel to run them against)");

    println!("\n--- what this means for this appendix's evidence ---");
    println!("The host reference above is an ordinary Rust triple loop, independent of the");
    println!("kernel source; the kernel's arithmetic is line-for-line the same as that");
    println!("reference's innermost loop, with `row`/`col` supplied by `blockIdx`/`threadIdx`");
    println!("instead of a `for` statement. On a machine with an actual CUDA device, the two");
    println!("would agree for the same reason Appendix D's Rust `contract()` already agreed");
    println!("with `numpy`'s `@` operator: it's the same sum, computed a different way.");
}
```

### File: 02_tiled_contraction_kernel.rs

```rust
// E.2: Tiling the Contracted Axis in Shared Memory.
//
// Section E.1's kernel re-reads each element of A and B from global memory
// many times over: every thread in output row `row` re-reads all of A's
// row `row`; every thread in output column `col` re-reads all of B's
// column `col`. TILING loads a TILE x TILE block of A and a TILE x TILE
// block of B into fast on-chip shared memory once, lets every thread in
// the block reuse it TILE times, then advances to the next tile along the
// contracted axis -- "tiling along a contraction's contracted axis,"
// since nothing about the technique is actually specific to matmul.
//
// This file deliberately picks K=23, NOT a multiple of TILE=16, so the
// boundary-padding branch is a genuine part of what gets described below
// -- and Section E.2's companion file, `02b_tiled_kernel_host_emulation`,
// is what actually EXERCISES that branch and checks it, since this file's
// own shared-memory/global-memory-traffic numbers are host arithmetic
// that doesn't touch the padding logic at all.
use cudarc::driver::CudaContext;
use cudarc::nvrtc::compile_ptx;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn catch_cuda<F: FnOnce() -> R, R>(f: F) -> Result<R, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = payload.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "non-string panic payload".to_string()
        }
    })
}

const TILE: usize = 16;

/// Byte-for-byte the same kernel as the CUDA C++ edition's own Appendix
/// I, Section I.2.
const TILED_KERNEL_SRC: &str = r#"
#define TILE 16
extern "C" __global__ void tiled_contraction_kernel(const double* A, const double* B, double* C,
                                                      int M, int K, int N) {
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    double acc = 0.0;
    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;
        __syncthreads();
        for (int kk = 0; kk < TILE; kk++) acc += tileA[threadIdx.y][kk] * tileB[kk][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"#;

fn reference_contraction(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

fn main() {
    println!("=== Tiling a contraction along its contracted axis ===\n");

    let (m, k, n) = (37usize, 23usize, 19usize);
    let mut h_a = vec![0.0f64; m * k];
    let mut h_b = vec![0.0f64; k * n];
    let mut h_c_ref = vec![0.0f64; m * n];

    let mut seed: u32 = 4242;
    for v in h_a.iter_mut() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (seed % 100) as f64 / 10.0;
    }
    for v in h_b.iter_mut() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (seed % 100) as f64 / 10.0;
    }
    reference_contraction(&h_a, &h_b, &mut h_c_ref, m, k, n);

    println!("M={m}, K={k}, N={n}  (K={k} is NOT a multiple of TILE={TILE} -- exercises the padding branch)\n");

    println!("--- genuine attempt: a real device is the necessary first step before any");
    println!("--- kernel launch could happen at all ---");
    match catch_cuda(|| CudaContext::new(0)) {
        Ok(Ok(_ctx)) => println!("CudaContext::new(0) succeeded -- a real device is present."),
        Ok(Err(e)) => println!("CudaContext::new(0) returned a clean error: {e}"),
        Err(panic_msg) => {
            println!("CudaContext::new(0) panicked (no CUDA driver reachable, same as every");
            println!("device attempt since Chapter 4): {panic_msg}");
        }
    }

    println!("\n--- genuine NVRTC compilation attempt, the exact kernel source above ---");
    match catch_cuda(|| compile_ptx(TILED_KERNEL_SRC)) {
        Ok(Ok(_ptx)) => println!("compile_ptx succeeded -- NVRTC is genuinely reachable here."),
        Ok(Err(e)) => println!("compile_ptx returned a clean NVRTC error: {e}"),
        Err(panic_msg) => {
            println!("compile_ptx PANICKED (NVRTC unreachable, consistent since Chapter 4):");
            println!("  {panic_msg}");
        }
    }

    println!("\n(skipping allocation/memcpy/launch -- no device or compiled kernel to run them against)");

    let shared_bytes = 2 * TILE * TILE * std::mem::size_of::<f64>();
    println!("\nshared memory used per block: 2 x {TILE} x {TILE} x sizeof(f64) = {shared_bytes} bytes");
    println!("global memory reads per output element WITHOUT tiling: K = {k}");
    println!(
        "global memory reads per output element WITH this tiling: K / TILE (amortized) = {:.2}",
        k as f64 / TILE as f64
    );

    println!("\n(host reference above -- C[0][0]={:.4}, C[{}][{}]={:.4} -- is genuinely", h_c_ref[0], m - 1, n - 1, h_c_ref[(m - 1) * n + (n - 1)]);
    println!("computed here; whether the TILED kernel's own arithmetic, including its");
    println!("boundary-padding branch, agrees with it is what Section E.2's companion");
    println!("file below actually checks, without needing a GPU at all.)");
}
```

### File: 02b_tiled_kernel_host_emulation.rs

```rust
// E.2, continued: Verifying a Kernel's Logic With No GPU Present.
//
// Compiling cleanly (or, in this sandbox, even reaching NVRTC at all) is
// necessary but not sufficient evidence that the boundary-padding branch
// in `tiled_contraction_kernel` is correct. This file does not stop at
// "here is what the kernel should do" -- it EMULATES the kernel's exact
// block/thread loop structure in plain host Rust: the same grid/block
// loop nesting, the same tile-load conditions, computed one CPU thread at
// a time instead of thousands of hardware threads at once, at the
// identical M=37, K=23, N=19 shape Section E.2 used. This file needs no
// CUDA toolchain, no driver, and no NVRTC at all -- it is ordinary host
// arithmetic, exactly like Appendix C's tiling-reduction simulation.

const TILE: usize = 16;

fn emulate_tiled_kernel(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    let grid_x = n.div_ceil(TILE);
    let grid_y = m.div_ceil(TILE);
    let num_tiles = k.div_ceil(TILE);

    for by in 0..grid_y {
        for bx in 0..grid_x {
            let mut tile_a = [[0.0f64; TILE]; TILE];
            let mut tile_b = [[0.0f64; TILE]; TILE];
            let mut acc = [[0.0f64; TILE]; TILE];

            for t in 0..num_tiles {
                for ty in 0..TILE {
                    for tx in 0..TILE {
                        let row = by * TILE + ty;
                        let col = bx * TILE + tx;
                        let a_col = t * TILE + tx;
                        let b_row = t * TILE + ty;
                        tile_a[ty][tx] = if row < m && a_col < k { a[row * k + a_col] } else { 0.0 };
                        tile_b[ty][tx] = if b_row < k && col < n { b[b_row * n + col] } else { 0.0 };
                    }
                }
                for ty in 0..TILE {
                    for tx in 0..TILE {
                        for kk in 0..TILE {
                            acc[ty][tx] += tile_a[ty][kk] * tile_b[kk][tx];
                        }
                    }
                }
            }

            for ty in 0..TILE {
                for tx in 0..TILE {
                    let row = by * TILE + ty;
                    let col = bx * TILE + tx;
                    if row < m && col < n {
                        c[row * n + col] = acc[ty][tx];
                    }
                }
            }
        }
    }
}

fn reference_contraction(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

fn main() {
    println!("=== Verifying the tiled kernel's boundary logic WITHOUT a GPU ===\n");

    let (m, k, n) = (37usize, 23usize, 19usize);
    println!(
        "M={m}, K={k}, N={n} (K/TILE = {:.3} -- the fractional last tile is the risk)\n",
        k as f64 / TILE as f64
    );

    let mut a = vec![0.0f64; m * k];
    let mut b = vec![0.0f64; k * n];
    let mut c_ref = vec![0.0f64; m * n];
    let mut c_emulated = vec![0.0f64; m * n];

    let mut seed: u32 = 4242;
    for v in a.iter_mut() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (seed % 100) as f64 / 10.0;
    }
    for v in b.iter_mut() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (seed % 100) as f64 / 10.0;
    }

    reference_contraction(&a, &b, &mut c_ref, m, k, n);
    emulate_tiled_kernel(&a, &b, &mut c_emulated, m, k, n);

    let mut max_diff = 0.0f64;
    for i in 0..c_ref.len() {
        let d = (c_ref[i] - c_emulated[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    println!("max |reference - emulated tiled kernel| over all {} output elements: {:.3e}", m * n, max_diff);

    if max_diff == 0.0 {
        println!("PASS -- the tiling logic, INCLUDING the K-not-a-multiple-of-TILE padding");
        println!("branch, is exactly correct. Whatever runs on an actual device is running");
        println!("this same index arithmetic, just spread across real hardware threads");
        println!("instead of one CPU thread looping over blocks in sequence.");
    } else {
        println!("FAIL -- the tiling logic disagrees with the reference; see max_diff above.");
        std::process::exit(1);
    }
}
```

### File: 03_multi_axis_contraction_kernel.rs

```rust
// E.3: Contracting Over Multiple Axes on the GPU.
//
// Generalizing Section E.1's kernel from one shared axis to several
// follows the same idea Appendix D's `contract()` used: split each
// tensor's axes into "free" and "contracted," walk the free-index space
// (now the CUDA grid) and the contracted-index space (now a device-side
// loop) separately. The one real difference is the tool available to do
// the walking. Host Rust could use the recursive `for_each_index()`
// helper from Appendix D, Section D.3; device code cannot -- CUDA kernels
// have no heap-allocated `Vec`, and recursion to a depth fixed only at
// runtime (a tensor's rank) doesn't fit CUDA's execution model cheaply.
// The kernel below instead reaches for the *other* indexing technique
// Appendix D's Section D.2 already introduced: fixed-size arrays (capped
// at a compile-time MAX_RANK) plus iterative divmod unraveling -- the
// exact same technique, just moved from a teaching aside in the CPU
// appendix to a hard requirement in this one.
//
// This file goes one step further than the CUDA C++ edition's own
// Appendix I, Section I.3 does in its own runnable code: rather than
// leaving "translating the kernel body into a host function" as prose
// describing a check that was done separately, `emulate_multi_axis_kernel`
// below IS that translation, written out in full and genuinely run here,
// reproducing Appendix D's own already-`numpy`-verified double-contraction
// result one output element at a time, exactly as a CUDA thread would.
use cudarc::driver::CudaContext;
use cudarc::nvrtc::compile_ptx;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn catch_cuda<F: FnOnce() -> R, R>(f: F) -> Result<R, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = payload.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "non-string panic payload".to_string()
        }
    })
}

const MAX_RANK: usize = 4;

/// Byte-for-byte the same kernel as the CUDA C++ edition's own Appendix
/// I, Section I.3.
const MULTI_AXIS_KERNEL_SRC: &str = r#"
#define MAX_RANK 4

struct TensorMeta {
    int shape[MAX_RANK];
    long long strides[MAX_RANK];
    int rank;
};

extern "C" __global__ void multi_axis_contraction_kernel(
    const double* A, const double* B, double* C,
    TensorMeta metaA, TensorMeta metaB, TensorMeta metaC,
    int free_a[MAX_RANK], int free_a_count,
    int free_b[MAX_RANK], int free_b_count,
    int axes_a[MAX_RANK], int axes_b[MAX_RANK],
    int contracted_shape[MAX_RANK], long long contracted_strides[MAX_RANK],
    int contracted_rank, long long contracted_total)
{
    long long out_lin = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long out_total = 1;
    for (int i = 0; i < metaC.rank; i++) out_total *= metaC.shape[i];
    if (out_lin >= out_total) return;

    int out_idx[MAX_RANK];
    long long rem = out_lin;
    for (int i = 0; i < metaC.rank; i++) { out_idx[i] = (int)(rem / metaC.strides[i]); rem %= metaC.strides[i]; }

    int idx_a[MAX_RANK] = {0}, idx_b[MAX_RANK] = {0};
    for (int i = 0; i < free_a_count; i++) idx_a[free_a[i]] = out_idx[i];
    for (int j = 0; j < free_b_count; j++) idx_b[free_b[j]] = out_idx[free_a_count + j];

    double acc = 0.0;
    for (long long c_lin = 0; c_lin < contracted_total; c_lin++) {
        int c_idx[MAX_RANK];
        long long crem = c_lin;
        for (int i = 0; i < contracted_rank; i++) { c_idx[i] = (int)(crem / contracted_strides[i]); crem %= contracted_strides[i]; }
        for (int k = 0; k < contracted_rank; k++) { idx_a[axes_a[k]] = c_idx[k]; idx_b[axes_b[k]] = c_idx[k]; }

        long long lin_a = 0; for (int i = 0; i < metaA.rank; i++) lin_a += (long long)idx_a[i] * metaA.strides[i];
        long long lin_b = 0; for (int i = 0; i < metaB.rank; i++) lin_b += (long long)idx_b[i] * metaB.strides[i];
        acc += A[lin_a] * B[lin_b];
    }

    long long lin_c = 0;
    for (int i = 0; i < metaC.rank; i++) lin_c += (long long)out_idx[i] * metaC.strides[i];
    C[lin_c] = acc;
}
"#;

#[derive(Clone, Copy)]
struct TensorMeta {
    shape: [i64; MAX_RANK],
    strides: [i64; MAX_RANK],
    rank: usize,
}

fn compute_row_major_strides(t: &mut TensorMeta) {
    let mut acc: i64 = 1;
    for i in (0..t.rank).rev() {
        t.strides[i] = acc;
        acc *= t.shape[i];
    }
}

/// A statement-for-statement translation of `multi_axis_contraction_kernel`
/// above into a host loop, run once per output index instead of once per
/// CUDA thread -- the same technique Section E.2's `02b_...` file used for
/// the tiled kernel, applied here to the more intricate metadata-driven
/// indexing.
fn emulate_multi_axis_kernel(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    meta_a: TensorMeta,
    meta_b: TensorMeta,
    meta_c: TensorMeta,
    free_a: &[usize],
    free_b: &[usize],
    axes_a: &[usize],
    axes_b: &[usize],
    contracted_shape: &[i64],
    contracted_strides: &[i64],
) {
    let out_total: i64 = meta_c.shape[..meta_c.rank].iter().product();
    let contracted_total: i64 = contracted_shape.iter().product();

    for out_lin in 0..out_total {
        let mut out_idx = [0i64; MAX_RANK];
        let mut rem = out_lin;
        for i in 0..meta_c.rank {
            out_idx[i] = rem / meta_c.strides[i];
            rem %= meta_c.strides[i];
        }

        let mut idx_a = [0i64; MAX_RANK];
        let mut idx_b = [0i64; MAX_RANK];
        for (i, &ax) in free_a.iter().enumerate() {
            idx_a[ax] = out_idx[i];
        }
        for (j, &bx) in free_b.iter().enumerate() {
            idx_b[bx] = out_idx[free_a.len() + j];
        }

        let mut acc = 0.0f64;
        for c_lin in 0..contracted_total {
            let mut c_idx = [0i64; MAX_RANK];
            let mut crem = c_lin;
            for i in 0..axes_a.len() {
                c_idx[i] = crem / contracted_strides[i];
                crem %= contracted_strides[i];
            }
            for k in 0..axes_a.len() {
                idx_a[axes_a[k]] = c_idx[k];
                idx_b[axes_b[k]] = c_idx[k];
            }

            let mut lin_a: i64 = 0;
            for i in 0..meta_a.rank {
                lin_a += idx_a[i] * meta_a.strides[i];
            }
            let mut lin_b: i64 = 0;
            for i in 0..meta_b.rank {
                lin_b += idx_b[i] * meta_b.strides[i];
            }
            acc += a[lin_a as usize] * b[lin_b as usize];
        }

        let mut lin_c: i64 = 0;
        for i in 0..meta_c.rank {
            lin_c += out_idx[i] * meta_c.strides[i];
        }
        c[lin_c as usize] = acc;
    }
}

fn reference_double_contraction(a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..2usize {
        for l in 0..5usize {
            let mut acc = 0.0f64;
            for j in 0..3usize {
                for k in 0..4usize {
                    acc += a[i * 12 + j * 4 + k] * b[j * 20 + k * 5 + l];
                }
            }
            c[i * 5 + l] = acc;
        }
    }
}

fn main() {
    println!("=== A double contraction on the GPU: TWO shared axes, one thread per output ===\n");

    let mut h_a = vec![0.0f64; 24];
    let mut h_b = vec![0.0f64; 60];
    let mut h_c_ref = vec![0.0f64; 10];
    for i in 0..h_a.len() {
        h_a[i] = (i % 7) as f64 - 3.0;
    }
    for i in 0..h_b.len() {
        h_b[i] = (i % 5) as f64 - 2.0;
    }
    reference_double_contraction(&h_a, &h_b, &mut h_c_ref);

    println!("host reference (identical inputs to Appendix D's 03_double_contraction.rs,");
    println!("already cross-checked there against numpy.tensordot):");
    print!("  ");
    for v in &h_c_ref {
        print!("{v:.2} ");
    }
    println!("\n");

    println!("--- genuine attempt: a real device is the necessary first step before any");
    println!("--- kernel launch could happen at all ---");
    match catch_cuda(|| CudaContext::new(0)) {
        Ok(Ok(_ctx)) => println!("CudaContext::new(0) succeeded -- a real device is present."),
        Ok(Err(e)) => println!("CudaContext::new(0) returned a clean error: {e}"),
        Err(panic_msg) => {
            println!("CudaContext::new(0) panicked (no CUDA driver reachable, same as every");
            println!("device attempt since Chapter 4): {panic_msg}");
        }
    }

    println!("\n--- genuine NVRTC compilation attempt, the exact kernel source above ---");
    match catch_cuda(|| compile_ptx(MULTI_AXIS_KERNEL_SRC)) {
        Ok(Ok(_ptx)) => println!("compile_ptx succeeded -- NVRTC is genuinely reachable here."),
        Ok(Err(e)) => println!("compile_ptx returned a clean NVRTC error: {e}"),
        Err(panic_msg) => {
            println!("compile_ptx PANICKED (NVRTC unreachable, consistent since Chapter 4):");
            println!("  {panic_msg}");
        }
    }

    println!("\n(skipping allocation/memcpy/launch -- no device or compiled kernel to run them against)");

    // The same metadata the CUDA kernel would receive as launch arguments,
    // built once here in host Rust.
    let mut meta_a = TensorMeta { shape: [2, 3, 4, 1], strides: [0; MAX_RANK], rank: 3 };
    let mut meta_b = TensorMeta { shape: [3, 4, 5, 1], strides: [0; MAX_RANK], rank: 3 };
    let mut meta_c = TensorMeta { shape: [2, 5, 1, 1], strides: [0; MAX_RANK], rank: 2 };
    compute_row_major_strides(&mut meta_a);
    compute_row_major_strides(&mut meta_b);
    compute_row_major_strides(&mut meta_c);

    let free_a = [0usize];
    let free_b = [2usize];
    let axes_a = [1usize, 2];
    let axes_b = [0usize, 1];
    let contracted_shape = [3i64, 4];
    let contracted_strides = [4i64, 1];

    let mut h_c_emulated = vec![0.0f64; 10];
    emulate_multi_axis_kernel(
        &h_a,
        &h_b,
        &mut h_c_emulated,
        meta_a,
        meta_b,
        meta_c,
        &free_a,
        &free_b,
        &axes_a,
        &axes_b,
        &contracted_shape,
        &contracted_strides,
    );

    println!("\n--- host emulation of the kernel's own exact metadata-driven index");
    println!("--- arithmetic above, run once per output index (not just described) ---");
    print!("  ");
    for v in &h_c_emulated {
        print!("{v:.2} ");
    }
    println!();

    let mut max_diff = 0.0f64;
    for i in 0..h_c_ref.len() {
        let d = (h_c_ref[i] - h_c_emulated[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!(
        "\nmax |reference - emulated multi-axis kernel| over all {} output elements: {:.3e}",
        h_c_ref.len(),
        max_diff
    );
    if max_diff == 0.0 {
        println!("PASS -- the metadata-driven index arithmetic (free_a/free_b/axes_a/axes_b/");
        println!("contracted_shape/contracted_strides, exactly as the kernel receives them)");
        println!("is exactly correct, confirmed by running it, not by inspecting it.");
    } else {
        println!("FAIL -- the emulated kernel logic disagrees with the reference; see max_diff above.");
        std::process::exit(1);
    }
}
```

## Appendix Summary

The free/contracted index split that defines a tensor contraction on paper maps directly onto CUDA's own division of labor: free indices become the grid of threads, contracted indices become each thread's private loop. This appendix built that mapping up in the same order Appendix D built the CPU version -- a naive one-thread-per-output-element kernel first, then a shared-memory-tiled version that generalizes the classic tiled-matmul optimization to tiling along an arbitrary contracted axis, then a fully general multi-axis version using the fixed-size-array indexing technique Appendix D's Section D.2 introduced. With neither a physical GPU nor a CUDA Toolkit available in this environment -- a strictly deeper gap than the CUDA C++ edition's own, which at least has `nvcc` -- every genuine CUDA call this appendix could make (`CudaContext::new`, `compile_ptx`) was still genuinely attempted and its real panic honestly reported, rather than skipped or assumed. What carried this appendix's real evidentiary weight was host-side verification: an independent reference for the simplest kernel, and, for the two more intricate ones, a direct host-side emulation of each kernel's own exact loop structure, run in full rather than only described -- catching the boundary case (`K` not a multiple of `TILE`) that would have been easiest to get wrong and hardest to notice, and confirming the multi-axis kernel's metadata-driven indexing produces the correct answer by running it, once per output index, rather than by inspection alone.

## Self-Check Questions

1. In the mapping this appendix builds, what plays the role of Appendix D's outer "free-index" loop, and what plays the role of its inner "contracted-index" loop?
2. Why does `contraction_kernel` need the `if (row >= M || col >= N) return;` guard at all, given that `row` and `col` are computed directly from the launch configuration?
3. What does tiling actually reduce -- FLOP count, or something else? Defend your answer using Section E.2's own reported numbers.
4. Section E.2 deliberately chose `K=23` rather than a multiple of 16. What would have been left unverified if `K` had been chosen as a multiple of `TILE` instead?
5. Why can Section E.3's kernel not reuse Appendix D's `for_each_index()` helper unmodified?
6. What, precisely, does the host-side emulation in Section E.2 and Section E.3 prove, and what does it explicitly NOT prove?
7. If this appendix had access to a real GPU and a real CUDA Toolkit, what specific additional evidence would you want to see before trusting `multi_axis_contraction_kernel`'s output, beyond what a host emulation can already provide?

### Worked Solutions

1. The CUDA grid (the collection of thread blocks, and within them, individual threads identified by `blockIdx`/`threadIdx`) plays the role of the free-index loop -- one thread per combination of free indices, i.e. per output element. Each individual thread's own loop over the contracted axis (or, in Section E.3, over the contracted linear index `c_lin`) plays the role of the contracted-index loop.
2. The grid is sized by rounding `M` and `N` up to whole multiples of the block dimensions (`ceil(N/16) x ceil(M/16)` blocks of `16x16` threads), which can request more threads than there are actual output elements whenever `M` or `N` isn't itself a multiple of the block size. Without the guard, those extra threads would compute `row`/`col` values at or past the end of `C` and write out of bounds.
3. Tiling reduces global-memory traffic, not FLOP count -- the same `M·N·K` multiply-adds happen either way. Section E.2's own numbers make this explicit: "global memory reads per output element WITHOUT tiling: K = 23" versus "WITH this tiling: K / TILE (amortized) = 1.44" -- roughly a 16x reduction in memory reads per output element, for identical arithmetic.
4. The boundary-padding branch (`(row < M && a_col < K) ? A[...] : 0.0`, and its counterpart for `B`) would never execute its "false" case -- every tile would be a full `TILE x TILE` block with nothing to pad, and a bug in the padding logic specifically could ship without any test ever catching it. Choosing `K=23` (not a multiple of 16) forces the last tile along the contracted axis to be fractional, so the padding branch's "insert zero instead of reading" case is genuinely exercised by the verification in Worked Example E.2.1.
5. `for_each_index()` recurses once per tensor axis, with the number of axes (the tensor's rank) known only at runtime, from a `Vec`'s length. CUDA device code has no `Vec`, and per-thread recursion to a runtime-determined depth doesn't fit how a GPU schedules thousands of concurrent, identical-instruction threads cheaply. Section E.3's kernel instead fixes a compile-time `MAX_RANK` and uses plain iterative loops bounded by that constant, at the cost of only supporting tensors up to that rank.
6. It proves that each kernel's INDEX ARITHMETIC -- which memory locations get read, added, and written, including Section E.2's boundary-padding condition and Section E.3's metadata-driven axis bookkeeping -- produces the correct sums, because that arithmetic was translated statement-for-statement into a host loop and run for real. It does NOT prove anything about how that same arithmetic performs, or even whether it compiles correctly as actual device code, when executed across real hardware threads with real memory hierarchies, real `__syncthreads()` synchronization, or real warp scheduling -- those properties can only be confirmed by compiling and running the actual `__global__` kernel on an actual GPU with a real CUDA Toolkit, which is precisely the combination this environment does not have.
7. Actual execution on physical hardware, comparing values read back from the device against the reference to floating-point tolerance -- confirming not just that the arithmetic is correct in principle but that `__syncthreads()` placement has no race, that the `TensorMeta` structs and fixed-size arrays marshal correctly across the host/device boundary in practice, and that the specific compiled machine code for this specific architecture computes what the source says it should. A host emulation checks the algorithm; only a real device run, on a machine with a real CUDA Toolkit to compile against, checks the actual CUDA execution of it.
