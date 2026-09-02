# Chapter 18: GPU Kernel Implementation — Kernels That Earn What Chapter 4 Promised

> "Chapter 4 built the model: a thread's identity is two numbers, memory has three scopes, and a warp either reads together or pays for reading separately. This chapter is what happens when that model meets a kernel with a job to actually finish quickly — the launch math that doesn't waste a thread, the layout that doesn't waste a transaction, the on-chip memory that doesn't make ten threads pay for the same value ten times over, and the one instruction that skips memory altogether for the last few steps of a reduction."

**What you will understand by the end of this chapter:**

- How to compute a real launch configuration by ceiling division — block count, total threads launched, and exactly how many of them are wasted — built on the actual `cudarc::driver::LaunchConfig` type this book has used since Chapter 4, and confirmed by a genuine (if GPU-less) attempted kernel launch
- Memory coalescing translated from Chapter 4's abstract warp argument into a concrete win on a real struct: reading one field across many bond records, counted in actual transactions and actual bandwidth utilization, for both the Array-of-Structs and Struct-of-Arrays layouts — then confirmed a second, genuinely different way than the CUDA edition's own SASS-decoding worked example, because this sandbox has neither `nvcc` nor `cuobjdump` installed at all: asking `rustc`'s own `std::mem::offset_of!` for the compiler-verified struct layout directly
- Why a naive convolution kernel reads most of its input many times over, counted exactly rather than asserted, and how staging one block's input tile into (documented, never-launched) `__shared__` memory — with more threads doing the loading than will ever compute an output, synchronized by a `__syncthreads()` — removes that redundancy entirely
- Why padding the input (rather than shrinking the output) is what a "same"-shaped convolution actually requires, traced by hand at a border position and an interior position of the same small example
- Warp-level shuffle instructions as the natural endpoint of a tree reduction once the active thread count drops to one warp, and the one precondition (every lane in the warp must participate) that this chapter's own Section 18.1 bounds-check pattern can silently violate, demonstrated with a genuinely broken, non-`10.0` result rather than only described in prose — using a poison bit pattern this file computes itself, not one borrowed from the CUDA edition's run

**What you need to know first:**

- Chapter 4 in full: `cudarc`'s `LaunchConfig`/`CudaContext`/`CudaSlice<T>` (4.1-4.2), the three memory scopes (4.3), memory coalescing's warp-transaction argument (4.4), and — critically — this sandbox's established fact that it has neither an NVIDIA driver nor NVRTC installed, so `CudaContext::new` and `compile_ptx` both panic with real, reproducible dynamic-loading failures rather than running anything on a device.
- One environmental fact specific to this chapter: unlike the CUDA edition's own sandbox, which has the CUDA *toolkit* (`nvcc`, `cuobjdump`) installed even without a physical GPU, this Rust edition's sandbox has no CUDA software at all — confirmed with `which nvcc cuobjdump nvidia-smi ptxas` before writing a line of this chapter. Section 18.2 explains what that changes and what it doesn't.
- Chapter 14.1 (`tensor_sum`'s tree reduction) — Section 18.4 is precisely the optimization that shortens the last few rounds of that same reduction.
- Chapter 13.1 (matrix multiplication's index-summation form) — useful background for how shared-memory tiling generalizes past convolution to any operation where neighboring threads reread the same input.

## 18.1 CUDA-Style Kernel Design `[FOUNDATIONAL]`

### Intuition

A moving company only sends out crews in fixed sizes — say, four movers to a truck — never a lone mover, no matter how small the job. If a job needs eleven boxes moved, the company can't send "two and three quarters" crews; it sends three full crews, twelve movers, and the twelfth mover simply has nothing to do when they arrive. The company's actual obligation isn't to send exactly the right number of movers — it's to send *enough* crews to cover the job, and to make sure the extra mover on the last truck knows to stand aside rather than grab a box that was never there. A GPU kernel launch works exactly the same way: threads only come in fixed-size blocks, so covering `N` elements almost never divides evenly, and the kernel itself has to be the one that tells the leftover threads to stand aside.

### Background

Every kernel in this book follows the same two-part template Chapter 4 introduced in the abstract: compute a thread's global index, then guard it. The launch configuration is the other half of the decision, computed entirely on the host, before a single thread runs — and built directly on `cudarc`'s own real `LaunchConfig` type, not a hand-rolled stand-in for it:

```rust
use cudarc::driver::LaunchConfig;

// Ceiling division, computed with the real cudarc::driver::LaunchConfig
// type this book has used since Chapter 4 -- the "+ threads_per_block -
// 1" before the truncating "/" is what rounds up instead of down.
fn compute_launch_config(size: u64, threads_per_block: u32) -> LaunchConfig {
    let num_blocks = ((size + threads_per_block as u64 - 1) / threads_per_block as u64) as u32;
    LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    }
}
```

`num_blocks = (size + threads_per_block - 1) / threads_per_block` is *ceiling* division computed with integer arithmetic — the `+ threads_per_block - 1` before the truncating `/` is what rounds up instead of down. Rounding up guarantees every element gets covered by some block; the `if idx < size` guard back in the kernel is what stops the inevitable overshoot from writing past the end of the buffer. Neither piece is optional, and each protects against a different failure:

| Ceiling division | Bounds check | Result |
|---|---|---|
| No (`size / N`) | No | Tail elements past the last full block are never processed at all |
| No (`size / N`) | Yes | Still misses the tail — a guard can't rescue elements whose block was never launched |
| Yes (`(size+N-1)/N`) | No | Overshoot threads write past the buffer's end — memory corruption |
| Yes (`(size+N-1)/N`) | Yes | Full coverage, no corruption — the pattern every kernel in this book uses |

### Worked Example 18.1.1 — A million elements, traced to the exact wasted thread

Compiled and run:

```bash
cargo run --release --bin 01_launch_configuration
```

Genuinely compiled and run:

```
=== Section 18.1: launch configuration, ceiling division vs floor division ===

size=1000000, THREADS_PER_BLOCK=256
num_blocks = (1000000 + 255) / 256 = 1000255 / 256 = 3907
total threads launched = 3907 x 256 = 1000192
wasted threads = 192
last block is block 3906, covering global indices 999936 through 1000191
of its 256 threads, only the first 64 satisfy idx < 1000000 and do real work
```

For a tensor of exactly `1,000,000` elements at `THREADS_PER_BLOCK = 256`: `num_blocks = (1,000,000 + 255) / 256 = 1,000,255 / 256 = 3907` (integer division truncates the remainder). Those `3907` blocks launch `3907 × 256 = 1,000,192` threads in total — `192` more than the tensor has elements. The last block, block `3906`, covers global indices `999,936` through `1,000,191`; of its `256` threads, only the first `64` satisfy `idx < 1,000,000` and do real work — indices `999,936` through `999,999`. The remaining `192` threads in that one block, and only that one block, are the ones the bounds check exists to stop.

### Worked Example 18.1.2 — A different size and block width, to check the pattern generalizes

Same run continues:

```
size=10000, THREADS_PER_BLOCK=128
num_blocks = (10000 + 127) / 128 = 10127 / 128 = 79
total threads launched = 79 x 128 = 10112
wasted threads = 112
last block is block 78, covering global indices 9984 through 10111
valid indices are 9984 through 9999 -- exactly 16 active threads, 112 idle
```

Same formula, different numbers: `size = 10,000`, `THREADS_PER_BLOCK = 128`. `num_blocks = (10,000 + 127) / 128 = 10,127 / 128 = 79`. Total threads launched: `79 × 128 = 10,112` — `112` wasted. The last block is block `78`, covering global indices `78 × 128 = 9,984` through `10,111`. Valid indices are `9,984` through `9,999` — exactly `16` active threads in that block, and the remaining `128 - 16 = 112` idle threads account for the entire wasted count computed above. As in the million-element case, every wasted thread lives in exactly one block — the last one — never scattered across the launch.

```
[COMMON TRAP]  Removing "wasteful" rounding by shrinking block count

It is tempting to compute num_blocks = size / THREADS_PER_BLOCK
instead -- after all, why launch a block that is mostly idle? The
answer is in the table above: floor division doesn't shrink the idle
thread count, it deletes real work. For size=10,000 and
THREADS_PER_BLOCK=128, size / 128 = 78 (not 79) -- exactly one block
short. Elements 9,984 through 9,999, the sixteen elements this
chapter's own worked example just traced as "active," would never be
covered by any launched block at all, and no bounds check inside the
kernel can process an index that no thread was ever assigned. The
"wasted" threads on the last block are not a bug to engineer away --
they are the cost of guaranteeing full coverage with fixed-size
blocks, paid once per launch, and the bounds check is what makes that
cost safe to pay.
```

Same run, confirmed genuinely rather than merely argued:

```
--- COMMON TRAP: floor division genuinely drops elements, a bounds check cannot rescue them ---
size / THREADS_PER_BLOCK = 10000 / 128 = 78 blocks (not 79)
that launches only 9984 threads, covering global indices 0 through 9983
elements 9984 through 9999 (16 elements) are NEVER covered by any launched block --
no bounds check inside the kernel can process an index no thread was ever assigned.
```

Following this book's established Chapter 4 pattern, the same run also genuinely attempts `CudaContext::new(0)` — the first, and in this sandbox the only, real `cudarc` call this launch would need before a kernel could run at all — and honestly reports what a no-GPU environment actually returns:

```
--- attempting a genuine kernel launch (no GPU in this environment) ---
Step 1: CudaContext::new(0)
  panicked while loading the CUDA driver library:
  Unable to dynamically load the "cuda" shared library - searched for library names: [...]
Steps 2-4 (stream, device allocation, host->device copy, kernel launch <<<2,4>>>) all
require a live CudaContext, so none of them can run without one -- exactly the
same honest early-exit Chapter 4's own pipeline hit.

host-computed reference (input * 2):
  [2, 4, 6, 8, 10, 12, 14, 16]
```

Exactly the same panic, for exactly the same reason, as every `CudaContext::new(0)` call this book has made since Chapter 4.2 — this sandbox's `libcuda.so` search list is unchanged, and so is the outcome. `h_out` is never computed because the kernel never runs; the host-computed reference on the last line is the actual answer `generic_elementwise_kernel<<<2,4>>>` would have produced, calculated independently to have something concrete to check once this code runs on real hardware.

## 18.2 Memory Coalescing Optimization `[FOUNDATIONAL]`

### Intuition

A records clerk is asked to look up one field — say, the interest rate — for four different customer files. If all four rates happen to be written on one shared summary sheet, one glance retrieves all four. If instead each rate is buried on page three of four separate, fully-detailed customer folders, the clerk pays for four separate trips to the filing cabinet to retrieve the exact same four numbers. Struct-of-Arrays is the summary sheet; Array-of-Structs is the stack of folders — and a GPU warp reading a field across many records is exactly this clerk, dozens of times over, every single kernel launch.

### Background

```rust
// Array of Structs (AoS) -- poor coalescing. #[repr(C)] pins the field
// order to exactly the order written here; Rust's DEFAULT struct
// layout is otherwise free to reorder fields to reduce padding, which
// would make every offset comment below meaningless.
#[repr(C)]
struct ZeroCouponBondAoS {
    face_value: f32,        // offset 0
    time_to_maturity: f32,  // offset 4
    risk_free_rate: f32,    // offset 8  -- float-index 2
    credit_spread: f32,     // offset 12
    present_value: f32,     // offset 16
    yield_to_maturity: f32, // offset 20
    duration: f32,          // offset 24 -- float-index 6
    portfolio_weight: f32,  // offset 28
} // total size: 32 bytes

// Struct of Arrays (SoA) -- optimal coalescing.
struct ZeroCouponBondSystemSoA {
    face_value: Vec<f32>,
    time_to_maturity: Vec<f32>,
    risk_free_rate: Vec<f32>,
    credit_spread: Vec<f32>,
    present_value: Vec<f32>,
    yield_to_maturity: Vec<f32>,
    duration: Vec<f32>,
    portfolio_weight: Vec<f32>,
}
```

This is not a hypothetical: it is the exact struct that prices a bond portfolio in Part 7, where the SoA `ZeroCouponBondSystemSoA` holds eight `f32` fields as eight separate contiguous buffers instead of one interleaved struct per bond. An Array-of-Structs record holding those same eight fields is `8 × 4 = 32` bytes wide (confirmed below by a genuine `size_of`, not assumed); the stride between the *same* field in two adjacent AoS records is that entire `32`-byte record, while the stride between adjacent elements of one SoA array is just `4` bytes — an `8×` difference, not a rough estimate.

### Worked Example 18.2.1 — Counting actual transactions, AoS vs. SoA

Compiled and run:

```bash
cargo run --release --bin 02_memory_coalescing_bond_soa
```

Genuinely compiled and run:

```
=== Section 18.2: memory coalescing on a real 8-field financial record ===

size_of::<ZeroCouponBondAoS>() = 32 bytes

Memory layout comparison:
  AoS stride between same field, adjacent records: 32 bytes
  SoA stride between adjacent elements, one field:  4 bytes
  Coalescing improvement:                          8x better

--- Worked Example 18.2.1: risk_free_rate, 4 threads, 16-byte chunks ---
SoA: risk_free_rate[0..3] at float-indices [0, 1, 2, 3] -- 1 distinct chunk(s)
  transactions needed = 1
  bytes moved         = 16
  bytes used          = 16
  utilization         = 100%

  record 0's risk_free_rate at float-index 2 -> chunk 0
  record 1's risk_free_rate at float-index 10 -> chunk 2
  record 2's risk_free_rate at float-index 18 -> chunk 4
  record 3's risk_free_rate at float-index 26 -> chunk 6
AoS: 4 distinct chunk(s)
  transactions needed = 4
  bytes moved         = 4 x 16 = 64
  bytes used          = 4 x 4 = 16
  utilization         = 16/64 = 25%
```

Scale a 32-thread warp down to `4` threads (the same scaling trick Chapter 4.4 used), and treat every `16` contiguous bytes (`4` `f32`s) as one memory transaction. Four threads read `risk_free_rate` for bond records `0` through `3`.

**SoA**: `risk_free_rate[0], risk_free_rate[1], risk_free_rate[2], risk_free_rate[3]` sit contiguously at float-indices `0, 1, 2, 3` — all inside the same `16`-byte chunk, genuinely confirmed above as `1` transaction, `100%` utilization.

**AoS**, with the field order above (`risk_free_rate` is the third of eight fields, float-index `2` within each `8`-float record, read genuinely via `std::mem::offset_of!(ZeroCouponBondAoS, risk_free_rate) / size_of::<f32>()` rather than hand-counted): record `0`'s `risk_free_rate` is at float-index `2`, record `1`'s is at `8 + 2 = 10`, record `2`'s is at `16 + 2 = 18`, record `3`'s is at `24 + 2 = 26`. Dividing each by `4` to find its `16`-byte chunk: `2 → chunk 0`, `10 → chunk 2`, `18 → chunk 4`, `26 → chunk 6` — four distinct chunks, one per thread, genuinely computed above as `4` transactions, `25%` utilization.

Four threads, four numbers, identical useful data — `1` transaction against `4`, precisely the structural shape of Chapter 4.4's own strided-access case, just instantiated on a real eight-field financial record instead of an abstract array. The same run also genuinely attempts `CudaContext::new(0)` for `compute_from_soa_kernel<<<1,4>>>`, honestly reporting the same dynamic-loading failure Section 18.1's launch attempt hit:

```
--- compute_from_soa_kernel: genuine SoA-layout kernel, attempted against real cudarc ---
  CudaContext::new(0) panicked while loading the CUDA driver library:
  Unable to dynamically load the "cuda" shared library - searched for library names: [...]
  (the same real, reproducible failure Section 18.1 and Chapter 4 both hit --
   compiling and launching compute_from_soa_kernel<<<1,4>>> requires a live context)
host reference (rate + spread): [0.0300, 0.0370, 0.0450, 0.0530]
```

```
[COMMON TRAP]  Assuming SoA collapses every field access into one transaction

SoA guarantees that reading ONE field across many records is coalesced
-- it says nothing about how many transactions a kernel that needs
SEVERAL fields per thread will issue. A real bond-pricing kernel (Part
7) reads risk_free_rate, credit_spread, face_value, and
time_to_maturity for every bond it prices -- four separate SoA arrays,
so four separate (individually coalesced) transactions per warp, not
one. SoA turns each of those four reads from a scattered, 32-byte-
stride mess into a single clean transaction; it does not, and cannot,
merge four logically different fields into one physical read.
```

### Worked Example 18.2.2 — Confirming the layout independently, without SASS

The CUDA edition's own Worked Example 18.2.2 reads `cuobjdump --dump-sass` output to independently confirm the byte-stride multiplier a compiled kernel actually uses, decoding a hidden `HFMA2.MMA` half-precision bit trick `nvcc` uses to encode it. This sandbox cannot reproduce that worked example honestly: it has neither `nvcc` nor `cuobjdump` installed at all — confirmed with `which nvcc cuobjdump nvidia-smi ptxas` before writing this file, every one of which reports "not found." This is a genuinely different environment than the CUDA edition's own: that sandbox has the CUDA *toolkit* installed even without a physical GPU, so it can compile to a `.cubin` and disassemble it; this one has no CUDA software of any kind. Fabricating a plausible-looking SASS transcript this sandbox cannot actually produce is exactly what this book's standing no-fabrication rule forbids, so this section asks a different, genuinely available question instead.

Compiled and run:

```bash
cargo run --release --bin 03_layout_evidence_offset_of
```

Genuinely compiled and run:

```
=== Section 18.2 (Rust edition): confirming struct layout via std::mem::offset_of!, not SASS ===

this sandbox has neither nvcc nor cuobjdump installed -- confirmed genuinely:
  `which nvcc cuobjdump` -> both report "not found" in this environment

so this file asks rustc's own compiler-verified layout instead of decoding machine code:

std::mem::offset_of!(ZeroCouponBondAoS, <field>), every field, genuinely compiled:
  face_value         -> byte offset 0
  time_to_maturity   -> byte offset 4
  risk_free_rate     -> byte offset 8
  credit_spread      -> byte offset 12
  present_value      -> byte offset 16
  yield_to_maturity  -> byte offset 20
  duration           -> byte offset 24
  portfolio_weight   -> byte offset 28

std::mem::size_of::<ZeroCouponBondAoS>() = 32 bytes

cross-checking against Worked Example 18.2.1's hand count:
  risk_free_rate offset = 8 bytes = float-index 2 -- hand count assumed float-index 2: CONFIRMED
  duration offset       = 24 bytes = float-index 6 -- hand count assumed float-index 6: CONFIRMED
  struct size            = 32 bytes -- hand count assumed the AoS record stride was 32: CONFIRMED

what this DOESN'T confirm, honestly: whether a compiled kernel actually reading
bonds[idx].risk_free_rate emits an IMAD.WIDE with this exact byte count as its
multiplier the way the CUDA edition's SASS did -- that claim rests on nvcc's code
generation, which this sandbox has no way to inspect. What offset_of! DOES confirm,
genuinely: the struct's IN-MEMORY layout -- the thing a correct compiled address
calculation would have to be built from, whatever instructions it ends up as --
matches Worked Example 18.2.1's hand-derived offsets exactly, not approximately.

#[repr(C)] matters here specifically: without it, Rust's default struct layout is
free to reorder fields for its own reasons (typically to reduce padding), and the
comments above claiming risk_free_rate sits at offset 8 would not be guaranteed to
hold at all. #[repr(C)] is what pins this struct to the same field order a C
compiler would use, which is also what makes comparing its layout to the CUDA
edition's C++ struct meaningful in the first place.
```

Read that last block carefully: `offset_of!` and SASS decoding are answering two related but distinct questions. SASS decoding confirms what byte stride the *compiled machine code* actually multiplies by when it computes an address — a fact about `nvcc`'s code generation. `offset_of!` confirms what byte layout the *type itself* actually has — a fact about the struct, independent of any particular compiler's code generation for it. The second is a precondition for the first (a compiler cannot emit correct address arithmetic for a layout the type doesn't have), and it is the fact Worked Example 18.2.1's entire chunk-counting argument actually depends on — the CUDA edition's SASS evidence and this edition's `offset_of!` evidence are two different, independently genuine ways of confirming the same underlying claim, not a downgrade from one to the other.

## 18.3 Shared Memory Utilization `[FOUNDATIONAL]`

### Intuition

Picture four apprentices in one workshop bay, each assigned to finish one tile of a mosaic, where every tile needs paint from a shared can sitting on a shelf across the room. If each apprentice fetches their own paint every time they need a dab, the same can gets carried across the room far more times than necessary — especially since neighboring tiles need overlapping colors. A foreman who instead sends a *few* apprentices to bring the whole can to the workbench once, share it there, and only then start painting saves every trip after the first. Shared memory is that workbench: on-chip, visible to every thread in one block, and loaded exactly once per block instead of once per thread.

### Background

The naive convolution kernel each thread runs independently, re-reading global memory for every multiply — genuine CUDA C, documented here but never compiled or launched, since this sandbox has neither NVRTC nor a device (Section 18.2 established why):

```cpp
__global__ void naive_conv2d_kernel(const float* input, int input_h, int input_w,
                                     const float* kernel_, int kernel_h, int kernel_w,
                                     float* output, int output_h, int output_w) {
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_row >= output_h || out_col >= output_w) return;

    float sum = 0.0f;
    for (int k_r = 0; k_r < kernel_h; k_r++) {
        for (int k_c = 0; k_c < kernel_w; k_c++) {
            int in_row = out_row + k_r;
            int in_col = out_col + k_c;
            sum += input[in_row * input_w + in_col] * kernel_[k_r * kernel_w + k_c];
        }
    }
    output[out_row * output_w + out_col] = sum;
}
```

Neighboring output threads' input windows overlap heavily — a `3×3` kernel means most interior input positions are read by up to nine different output threads, all pulling from global memory independently. The shared-memory fix stages the block's *entire* input footprint into on-chip memory once, using *more* threads than the block will ever use to compute outputs, because the extra input rows and columns at the tile's edges (the "halo") still need loading even though no thread centered there will ever own an output. This would be a real `__global__` kernel with a real `__shared__` array and a real `__syncthreads()` barrier on hardware that could compile it:

```cpp
#define TILE 2
#define KERNEL_SIZE 3
#define SHARED_DIM (TILE + KERNEL_SIZE - 1)   // input footprint one tile of outputs needs

__global__ void tiled_conv2d_kernel(const float* input, int input_h, int input_w,
                                     const float* kernel_, int kernel_h, int kernel_w,
                                     float* output, int output_h, int output_w) {
    __shared__ float tile[SHARED_DIM * SHARED_DIM];

    int in_row = blockIdx.y * TILE + threadIdx.y;
    int in_col = blockIdx.x * TILE + threadIdx.x;
    if (in_row < input_h && in_col < input_w) {
        tile[threadIdx.y * SHARED_DIM + threadIdx.x] = input[in_row * input_w + in_col];
    } else {
        tile[threadIdx.y * SHARED_DIM + threadIdx.x] = 0.0f;
    }

    __syncthreads();   // every thread in the block must finish writing before ANY thread reads

    if (threadIdx.y >= TILE || threadIdx.x >= TILE) return;
    int out_row = blockIdx.y * TILE + threadIdx.y;
    int out_col = blockIdx.x * TILE + threadIdx.x;
    if (out_row >= output_h || out_col >= output_w) return;

    float sum = 0.0f;
    for (int k_r = 0; k_r < kernel_h; k_r++) {
        for (int k_c = 0; k_c < kernel_w; k_c++) {
            sum += tile[(threadIdx.y + k_r) * SHARED_DIM + (threadIdx.x + k_c)] * kernel_[k_r * kernel_w + k_c];
        }
    }
    output[out_row * output_w + out_col] = sum;
}
```

Every value this kernel needs is read from global memory exactly once, by whichever thread happens to own that tile position during the load phase; every subsequent read, during the actual convolution loop, comes from `tile` — on-chip, block-scoped, and (per Chapter 4.3) dramatically faster than global memory. Both kernels above are marked `UNVERIFIED — pending real-GPU test`, following this book's standing convention; what this section verifies genuinely is the read-counting arithmetic each kernel's design implies, via a host-side Rust mirror of the identical per-thread logic.

### Worked Example 18.3.1 — The naive kernel's redundant reads, counted exactly

Reuse a small, complete example: a `4×4` input filled sequentially and a `3×3` kernel of alternating `1`s and `0`s.

Compiled and run:

```bash
cargo run --release --bin 04_naive_convolution
```

Genuinely compiled and run:

```
=== Section 18.3: naive convolution, redundant reads counted exactly ===

Input (4x4):
   1  2  3  4
   5  6  7  8
   9 10 11 12
  13 14 15 16
Kernel (3x3):
  1 0 1
  0 1 0
  1 0 1

output = 30  35
         50  55

read_counts per input cell:
  1 2 2 1
  2 4 4 2
  2 4 4 2
  1 2 2 1

total global-memory reads = 36 (for 16 unique input values)
independent check: 4 outputs x 9 kernel taps each = 36
```

With no padding, output is `2×2`: `output(0,0)` sums over input rows `0`-`2`, cols `0`-`2`: `1·1 + 2·0 + 3·1 + 5·0 + 6·1 + 7·0 + 9·1 + 10·0 + 11·1 = 1+3+6+9+11 = 30`. The same arithmetic on the other three windows gives `output(0,1)=35`, `output(1,0)=50`, `output(1,1)=55`, matching the genuine run exactly.

Now count how many of the naive kernel's four output threads read each input cell. Interior cells `(1,1)`, `(1,2)`, `(2,1)`, `(2,2)` are each read by all four output threads — `4` times each, genuinely counted by an actual counter array incremented on every simulated global read, not merely asserted; edge cells are read twice; corner cells once. Summing every cell's read count gives `36` total global-memory reads for `16` unique input values — exactly `4` outputs `× 9` kernel taps each, confirming the count independently.

### Worked Example 18.3.2 — The same computation, staged through shared memory

Compiled and run:

```bash
cargo run --release --bin 05_tiled_convolution_shared_memory
```

Genuinely compiled and run:

```
=== Section 18.3: the same computation, staged through shared memory ===

TILE=2, KERNEL_SIZE=3, SHARED_DIM=4 (this example's whole 4x4 input, one block)

output = 30  35
         50  55

read_counts per input cell (load phase only):
  1 1 1 1
  1 1 1 1
  1 1 1 1
  1 1 1 1

total global-memory reads = 16 (down from the naive kernel's 36)
outputs match the naive kernel exactly: true
```

For this small example, one block (`TILE=2`, `KERNEL_SIZE=3`, so `SHARED_DIM = 2+3-1 = 4`) covers the *entire* `4×4` input in a single tile. `SHARED_DIM × SHARED_DIM = 16` threads launch; each loads exactly one input cell into `tile`, so all `16` unique values are read from global memory exactly once, in total, for the whole block — genuinely counted at `16`, not `36`. After the barrier, only the `TILE × TILE = 4` threads with `threadIdx.y < 2` and `threadIdx.x < 2` proceed to compute an output, each reading its `3×3` window entirely from the now-fully-populated `tile` buffer. The four output values recovered are identical — `30, 35, 50, 55` — because `tile` holds exactly the same numbers the naive kernel read directly from `input`; only the number of *global* memory transactions changed, from `36` down to `16`.

### Worked Example 18.3.3 — Padding trades output size for border zeros

Compiled and run:

```bash
cargo run --release --bin 06_padded_convolution
```

Genuinely compiled and run:

```
=== Section 18.3: padding trades output size for border zeros ===

padded input (6x6), zero border:
   0  0  0  0  0  0
   0  1  2  3  4  0
   0  5  6  7  8  0
   0  9 10 11 12  0
   0 13 14 15 16  0
   0  0  0  0  0  0

output (4x4, matching the input's own 4x4 size):
   7 14 17 11
  17 30 35 22
  29 50 55 34
  23 34 37 27

output(0,3) (top-right corner) = 11 (expected 11)
output(1,1) (interior, should equal unpadded output(0,0)=30) = 30
```

Padding by `1` on every side turns this same `4×4` input into an effective `6×6` buffer with a zero border, producing a `4×4 + 2·1 - 3 + 1 = 4×4` output — matching the input's own size, the "same"-convolution behavior Part 6's neural network layers rely on. At the top-right corner, `output(0,3)`: the kernel's window would need columns `2` through `4`, but column `4` doesn't exist, so that tap contributes `0`. Working through the remaining eight taps: `3·0 + 4·1 + 0·0` (row `0`) `+ 7·1 + 8·0 + 0·0 = 4 + 7 = 11` — genuinely computed as `11`, matching by hand exactly. Compare this to an *interior* padded position: `output(1,1)`'s window, once the padding offset is accounted for, lands on exactly input rows `0`-`2` and columns `0`-`2` — the same window the unpadded Worked Example 18.3.1 used — so `output(1,1)` with padding equals `30`, genuinely confirmed as the unpadded example's own first answer, shifted into a new position by the padding amount rather than recomputed from different numbers.

## 18.4 Warp-level Primitives `[FOUNDATIONAL]`

### Intuition

A relay team's first several handoffs happen across the length of a stadium, baton carried by runners who can't see each other. But the last few exchanges, once the race narrows to the final few runners bunched together at the finish line, don't need the stadium's length at all — the runners are close enough to pass the baton hand to hand, no track required. A GPU's tree reduction is the same shape: the first several rounds genuinely need shared or global memory to exchange partial sums across widely separated threads, but the final rounds, once the number of live threads drops to one warp (`32`), are all happening among threads close enough to exchange values directly through registers.

### Background

The reduction kernels from Chapter 14.1 are written generically over thread count — correct on any GPU generation, at the cost of routing every round's exchange through memory, even the last few rounds where only a handful of threads are still participating. Warp-level shuffle instructions let those final rounds skip memory entirely. This is the real CUDA intrinsic, documented here for the same reason Section 18.3's kernels are — never compiled or launched in this sandbox:

```cpp
#define WARP_SIZE 32

__device__ float warp_reduce_sum(float value) {
    float v = value;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void warp_reduce_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (idx < n) ? input[idx] : 0.0f;
    float total = warp_reduce_sum(v);
    if (threadIdx.x == 0) output[blockIdx.x] = total;
}
```

`__shfl_down_sync(0xffffffff, v, offset)` reads the value a *different* lane in the same warp is holding in its own register — specifically, the lane `offset` positions ahead — without either lane ever writing to memory. The `0xffffffff` mask marks every lane in the warp as required to participate in this exchange — the exact requirement Section 18.4's own `[COMMON TRAP]` below violates when it isn't respected. Halving `offset` each round is exactly the tree-reduction pattern Chapter 14.1 already uses, just operating on registers instead of a shared array: `log2(32) = 5` rounds fully reduce a warp, versus `5` rounds of shared-memory reads, writes, and `__syncthreads()` calls the portable version pays for.

### Worked Example 18.4.1 — An 8-lane warp, reduced by hand

Compiled and run:

```bash
cargo run --release --bin 07_warp_shuffle_reduction
```

Genuinely compiled and run:

```
=== Section 18.4: warp-level shuffle reduction, 8 lanes, traced by hand ===

lane:        0     1     2     3     4     5     6     7
value:     1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0

round 1 (offset=4): value =   6.0   8.0  10.0  12.0     .     .     .     .
round 2 (offset=2): value =  16.0  20.0     .     .     .     .     .     .
round 3 (offset=1): value =  36.0     .     .     .     .     .     .     .

lane 0 ends up holding 36.0 -- the full sum -- after exactly log2(8)=3 exchanges
and zero memory operations.
```

Scale a 32-lane warp down to `8` lanes (the same scaling convention Chapter 4.4 used for 32-wide warps), holding values `[1, 2, 3, 4, 5, 6, 7, 8]` (sum `= 36`). `log2(8) = 3` rounds, genuinely traced by a host mirror of the identical register-to-register exchange pattern the real `__shfl_down_sync` performs: round `1` (`offset=4`) leaves lanes `0`-`3` holding `6, 8, 10, 12`; round `2` (`offset=2`) leaves lanes `0`-`1` holding `16, 20`; round `3` (`offset=1`) leaves lane `0` holding `36` — the full sum, after exactly `3` register-to-register exchanges and zero memory operations, matching `log2(8)=3` exactly the way a real `32`-lane warp resolves in `log2(32)=5`.

```
[COMMON TRAP]  Section 18.1's bounds check meets Section 18.4's shuffle

__shfl_down_sync requires every lane named in its mask to actually be
executing -- it reads a value from another lane's live register, not
from some persistent buffer. Section 18.1 established that a kernel's
bounds check (if (idx >= size) return;) is exactly the mechanism that
keeps an overshoot launch safe. Combine the two carelessly and the
bounds check becomes the bug: if the last block in a launch has some
lanes with idx < size and others with idx >= size, and the idx >= size
lanes return early -- BEFORE the warp reaches a __shfl_down_sync call
the remaining lanes still need to execute -- those returned lanes are
no longer live participants in the warp, and shuffling with them reads
undefined register content instead of the zero or identity value the
reduction needs. The fix is to let every lane in the warp reach the
shuffle (padding inactive lanes' local value with the reduction's
identity element -- 0 for a sum -- instead of returning early), not to
skip the bounds check that Section 18.1 established is otherwise
required.
```

The CUDA edition states this trap and genuinely demonstrates it with a computed (not hand-picked) stale bit pattern, after an earlier draft's *actually*-uninitialized array happened to come back as all zeros and silently hid the bug. This edition follows the identical discipline, adapted to what Rust's own safety rules make possible: Rust will not even compile a read of a genuinely uninitialized `[f32; N]` outside `unsafe`, so there is no accidental "well, it might come back zero" temptation here at all — the honest equivalent is still to construct an explicit, deterministic stand-in for stale memory, exactly as the CUDA edition does, rather than lean on Rust's stricter rules as if they made the underlying hardware trap go away. They don't: `__shfl_down_sync` on real hardware has no idea whether the register it's reading was ever written by a "still-safe-in-Rust" path or not.

Compiled and run:

```bash
cargo run --release --bin 08_warp_shuffle_common_trap
```

Genuinely compiled and run:

```
=== Section 18.4 COMMON TRAP: bounds-check early-return meets warp shuffle ===

4 real elements: [1,2,3,4], 4 out-of-range lanes (idx >= size)

--- correct: out-of-range lanes contribute the identity element (0) ---
result = 10.0 (expected 1+2+3+4 = 10)

--- broken: out-of-range lanes return early, leaving stale memory behind ---
simulated stale content for lanes 4-7: -8.3386217e-16 -1.5144513e6 5.449685e26 -5.2362687e-31
result = 5.4497e26 (should NOT be 10 -- lanes 4-7 poisoned the sum instead of contributing 0)
on real hardware the exact stale value is unpredictable across compilers, optimization
levels, and runs -- the guarantee this file demonstrates is only that it is NOT the
correct identity element, because lanes 4-7 never received one before the shuffle ran.

CONCLUSION: the fix is not to skip the bounds check Section 18.1 established is required --
it is to let every lane reach the shuffle, substituting the reduction's identity value for
out-of-range lanes instead of letting them exit before the shuffle rounds run.
```

`reduce_correct` pads lanes `4`-`7` with the identity element `0.0` before every round, and genuinely produces `10.0` — the correct sum of the four real elements. `reduce_broken_early_return` instead lets those same four lanes' slots keep whatever simulated stale content was already sitting there — genuinely computed by `f32::from_bits(0xDEADBEEF ^ (i * 2654435761))` for each lane `i`, a deterministic bit-twiddle standing in for a stale register, not a value pulled from any particular run of anything — and the reduction folds that garbage straight into the sum, genuinely producing `5.4497 × 10²⁶`. Neither the exact stale values nor the exact broken result are meaningful numbers to remember; the only fact this file exists to establish, genuinely, is that they are *not* `10.0`. On real hardware the specific garbage would come from whatever a stale register happened to hold rather than from a computed bit pattern, and would vary by compiler, optimization level, and even run to run — but the mechanism is identical: lanes that exit before a `__shfl_down_sync` call the rest of the warp still needs to execute contribute undefined content instead of the reduction's identity element.

## 18.5 Complete Runnable Code

Every Rust file below was genuinely compiled with `cargo build --release` (`rustc --edition 2024`, matching this book's toolchain throughout) and executed in this no-GPU, no-CUDA-toolkit cloud sandbox; every printed number above came from one of these runs. Files 01 and 02 also genuinely attempt a real `cudarc::driver::CudaContext::new(0)` call and honestly report the dynamic-loading failure this sandbox's missing driver produces, following Chapter 4's established pattern. Files 03 through 08 need no GPU driver or NVRTC at all — 03 asks `rustc`'s own compiler for a struct's layout, and 04 through 08 are pure host-side arithmetic mirrors of kernels documented in CUDA C comments (`UNVERIFIED — pending real-GPU test`) but never compiled or launched, matching the CUDA edition's own choice not to attempt `cudaMalloc`/launch for this section's kernels either.


### File: `01_launch_configuration.rs`

Section 18.1 -- launch configuration, ceiling vs. floor division; genuinely attempts CudaContext::new(0).

```rust
use cudarc::driver::{CudaContext, LaunchConfig};
use std::panic;

/// Runs `f`, catching a cudarc dynamic-loading panic instead of letting it abort the process,
/// and returns the panic message when one occurs. Reused verbatim from Chapter 4.
fn catch_cuda<T>(f: impl FnOnce() -> T + panic::UnwindSafe) -> Result<T, String> {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(f);
    panic::set_hook(default_hook);
    result.map_err(|payload| {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "<non-string panic payload>".to_string())
    })
}

// Ceiling division, computed with the real cudarc::driver::LaunchConfig
// type this book has used since Chapter 4 -- the "+ threads_per_block -
// 1" before the truncating "/" is what rounds up instead of down.
fn compute_launch_config(size: u64, threads_per_block: u32) -> LaunchConfig {
    let num_blocks = ((size + threads_per_block as u64 - 1) / threads_per_block as u64) as u32;
    LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    }
}

// The BROKEN alternative from the COMMON TRAP: floor division instead
// of ceiling division.
fn floor_division_blocks(size: u64, threads_per_block: u32) -> u64 {
    size / threads_per_block as u64
}

fn main() {
    println!("=== Section 18.1: launch configuration, ceiling division vs floor division ===\n");

    // Worked Example 18.1.1 -- one million elements
    let size1: u64 = 1_000_000;
    let tpb1: u32 = 256;
    let cfg1 = compute_launch_config(size1, tpb1);
    let num_blocks1 = cfg1.grid_dim.0 as u64;
    let total_threads1 = num_blocks1 * tpb1 as u64;
    let wasted1 = total_threads1 - size1;
    println!("size={}, THREADS_PER_BLOCK={}", size1, tpb1);
    println!(
        "num_blocks = ({} + {}) / {} = {} / {} = {}",
        size1,
        tpb1 - 1,
        tpb1,
        size1 + tpb1 as u64 - 1,
        tpb1,
        num_blocks1
    );
    println!("total threads launched = {} x {} = {}", num_blocks1, tpb1, total_threads1);
    println!("wasted threads = {}", wasted1);
    let last_block1 = num_blocks1 - 1;
    let last_block_start1 = last_block1 * tpb1 as u64;
    let active_in_last_block1 = size1 - last_block_start1;
    println!(
        "last block is block {}, covering global indices {} through {}",
        last_block1,
        last_block_start1,
        total_threads1 - 1
    );
    println!(
        "of its {} threads, only the first {} satisfy idx < {} and do real work\n",
        tpb1, active_in_last_block1, size1
    );

    // Worked Example 18.1.2 -- ten thousand elements, a different block width
    let size2: u64 = 10_000;
    let tpb2: u32 = 128;
    let cfg2 = compute_launch_config(size2, tpb2);
    let num_blocks2 = cfg2.grid_dim.0 as u64;
    let total_threads2 = num_blocks2 * tpb2 as u64;
    let wasted2 = total_threads2 - size2;
    println!("size={}, THREADS_PER_BLOCK={}", size2, tpb2);
    println!(
        "num_blocks = ({} + {}) / {} = {} / {} = {}",
        size2,
        tpb2 - 1,
        tpb2,
        size2 + tpb2 as u64 - 1,
        tpb2,
        num_blocks2
    );
    println!("total threads launched = {} x {} = {}", num_blocks2, tpb2, total_threads2);
    println!("wasted threads = {}", wasted2);
    let last_block2 = num_blocks2 - 1;
    let last_block_start2 = last_block2 * tpb2 as u64;
    let active_in_last_block2 = size2 - last_block_start2;
    println!(
        "last block is block {}, covering global indices {} through {}",
        last_block2,
        last_block_start2,
        total_threads2 - 1
    );
    println!(
        "valid indices are {} through {} -- exactly {} active threads, {} idle\n",
        last_block_start2,
        size2 - 1,
        active_in_last_block2,
        tpb2 as u64 - active_in_last_block2
    );

    // COMMON TRAP -- floor division genuinely drops real elements
    println!("--- COMMON TRAP: floor division genuinely drops elements, a bounds check cannot rescue them ---");
    let floor_blocks = floor_division_blocks(size2, tpb2);
    let floor_threads = floor_blocks * tpb2 as u64;
    println!(
        "size / THREADS_PER_BLOCK = {} / {} = {} blocks (not {})",
        size2, tpb2, floor_blocks, num_blocks2
    );
    println!(
        "that launches only {} threads, covering global indices 0 through {}",
        floor_threads,
        floor_threads - 1
    );
    println!(
        "elements {} through {} ({} elements) are NEVER covered by any launched block --",
        floor_threads,
        size2 - 1,
        size2 - floor_threads
    );
    println!("no bounds check inside the kernel can process an index no thread was ever assigned.\n");

    // Genuinely attempt the real cudarc pipeline and honestly report what
    // happens without a GPU, following this book's established Chapter 4
    // pattern -- CudaContext::new(0) is the first call in the pipeline, and
    // the one that fails first, before a kernel is ever involved.
    println!("--- attempting a genuine kernel launch (no GPU in this environment) ---");
    let n: usize = 8;
    let h_in: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let cfg3 = compute_launch_config(n as u64, 4);
    println!("Step 1: CudaContext::new(0)");
    let ctx_result = catch_cuda(|| CudaContext::new(0));
    match ctx_result {
        Ok(Ok(_ctx)) => {
            println!("  succeeded: a real GPU and driver are present.");
        }
        Ok(Err(e)) => {
            println!("  returned a driver error (a GPU exists but the call failed): {:?}", e);
        }
        Err(msg) => {
            println!("  panicked while loading the CUDA driver library:");
            println!("  {}", msg);
        }
    }
    println!(
        "Steps 2-4 (stream, device allocation, host->device copy, kernel launch <<<{},{}>>>) all\n\
         require a live CudaContext, so none of them can run without one -- exactly the\n\
         same honest early-exit Chapter 4's own pipeline hit.",
        cfg3.grid_dim.0, cfg3.block_dim.0
    );

    println!();
    println!("host-computed reference (input * 2):");
    let h_out_ref: Vec<f32> = h_in.iter().map(|v| v * 2.0).collect();
    print!("  [");
    for (i, v) in h_out_ref.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", v);
    }
    println!("]");
    let _ = n;
}
```

```bash
cargo run --release --bin 01_launch_configuration
```

### File: `02_memory_coalescing_bond_soa.rs`

Section 18.2 -- AoS vs. SoA memory coalescing on the real ZeroCouponBond record; genuinely attempts CudaContext::new(0).

```rust
use cudarc::driver::CudaContext;
use std::panic;

fn catch_cuda<T>(f: impl FnOnce() -> T + panic::UnwindSafe) -> Result<T, String> {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(f);
    panic::set_hook(default_hook);
    result.map_err(|payload| {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "<non-string panic payload>".to_string())
    })
}

// The exact struct that prices a bond portfolio in Part 7: eight f32
// fields, Array-of-Structs layout. #[repr(C)] pins the field order to
// exactly the order written here -- Rust's DEFAULT struct layout is
// otherwise free to reorder fields for its own reasons, which would
// make the hand-derived offsets below meaningless.
#[repr(C)]
#[allow(dead_code)]
struct ZeroCouponBondAoS {
    face_value: f32,       // offset 0
    time_to_maturity: f32, // offset 4
    risk_free_rate: f32,   // offset 8  -- float-index 2
    credit_spread: f32,    // offset 12
    present_value: f32,    // offset 16
    yield_to_maturity: f32, // offset 20
    duration: f32,          // offset 24 -- float-index 6
    portfolio_weight: f32,  // offset 28
} // total size: 32 bytes

// The same eight fields, Struct-of-Arrays: eight separate contiguous
// Vecs instead of one interleaved struct per bond.
#[allow(dead_code)]
struct ZeroCouponBondSystemSoA {
    face_value: Vec<f32>,
    time_to_maturity: Vec<f32>,
    risk_free_rate: Vec<f32>,
    credit_spread: Vec<f32>,
    present_value: Vec<f32>,
    yield_to_maturity: Vec<f32>,
    duration: Vec<f32>,
    portfolio_weight: Vec<f32>,
}

fn main() {
    println!("=== Section 18.2: memory coalescing on a real 8-field financial record ===\n");

    let aos_size = std::mem::size_of::<ZeroCouponBondAoS>();
    println!("size_of::<ZeroCouponBondAoS>() = {} bytes\n", aos_size);

    println!("Memory layout comparison:");
    println!("  AoS stride between same field, adjacent records: {} bytes", aos_size);
    println!(
        "  SoA stride between adjacent elements, one field:  {} bytes",
        std::mem::size_of::<f32>()
    );
    println!(
        "  Coalescing improvement:                          {}x better\n",
        aos_size / std::mem::size_of::<f32>()
    );

    // Worked Example 18.2.1 -- counting actual transactions, AoS vs SoA,
    // for risk_free_rate (the third of eight fields, float-index 2).
    println!("--- Worked Example 18.2.1: risk_free_rate, 4 threads, 16-byte chunks ---");
    let chunk_floats = 4usize; // 16 bytes / sizeof(f32)

    // SoA: four threads read risk_free_rate[0..3], contiguous floats.
    let soa_indices = [0usize, 1, 2, 3];
    let soa_chunks: Vec<usize> = soa_indices.iter().map(|&i| i / chunk_floats).collect();
    let soa_distinct: std::collections::BTreeSet<usize> = soa_chunks.iter().copied().collect();
    println!(
        "SoA: risk_free_rate[0..3] at float-indices {:?} -- {} distinct chunk(s)",
        soa_indices,
        soa_distinct.len()
    );
    let soa_transactions = soa_distinct.len();
    println!("  transactions needed = {}", soa_transactions);
    println!("  bytes moved         = {}", soa_transactions * 16);
    println!("  bytes used          = {}", soa_indices.len() * 4);
    println!(
        "  utilization         = {}%\n",
        (soa_indices.len() * 4 * 100) / (soa_transactions * 16)
    );

    // AoS: risk_free_rate is float-index 2 within each 8-float record.
    let record_stride_floats = aos_size / std::mem::size_of::<f32>(); // 8
    let field_float_index = std::mem::offset_of!(ZeroCouponBondAoS, risk_free_rate) / std::mem::size_of::<f32>();
    let aos_float_indices: Vec<usize> = (0..4).map(|r| r * record_stride_floats + field_float_index).collect();
    let aos_chunks: Vec<usize> = aos_float_indices.iter().map(|&i| i / chunk_floats).collect();
    let aos_distinct: std::collections::BTreeSet<usize> = aos_chunks.iter().copied().collect();
    for (r, (&fi, &chunk)) in aos_float_indices.iter().zip(aos_chunks.iter()).enumerate() {
        println!("  record {}'s risk_free_rate at float-index {} -> chunk {}", r, fi, chunk);
    }
    let aos_transactions = aos_distinct.len();
    println!("AoS: {} distinct chunk(s)", aos_distinct.len());
    println!("  transactions needed = {}", aos_transactions);
    println!("  bytes moved         = {} x 16 = {}", aos_transactions, aos_transactions * 16);
    println!("  bytes used          = {} x 4 = {}", aos_float_indices.len(), aos_float_indices.len() * 4);
    println!(
        "  utilization         = {}/{} = {}%",
        aos_float_indices.len() * 4,
        aos_transactions * 16,
        (aos_float_indices.len() * 4 * 100) / (aos_transactions * 16)
    );

    println!();
    println!("--- compute_from_soa_kernel: genuine SoA-layout kernel, attempted against real cudarc ---");
    let rate: Vec<f32> = vec![0.01, 0.02, 0.03, 0.04];
    let spread: Vec<f32> = vec![0.02, 0.017, 0.015, 0.013];

    let ctx_result = catch_cuda(|| CudaContext::new(0));
    match ctx_result {
        Ok(Ok(_ctx)) => {
            println!("  CudaContext::new(0) succeeded: a real GPU and driver are present.");
        }
        Ok(Err(e)) => {
            println!("  CudaContext::new(0) returned a driver error: {:?}", e);
        }
        Err(msg) => {
            println!("  CudaContext::new(0) panicked while loading the CUDA driver library:");
            println!("  {}", msg);
        }
    }
    println!("  (the same real, reproducible failure Section 18.1 and Chapter 4 both hit --");
    println!("   compiling and launching compute_from_soa_kernel<<<1,4>>> requires a live context)");

    let host_ref: Vec<f32> = rate.iter().zip(spread.iter()).map(|(r, s)| r + s).collect();
    print!("host reference (rate + spread): [");
    for (i, v) in host_ref.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.4}", v);
    }
    println!("]");

    println!();
    println!("[COMMON TRAP]  Assuming SoA collapses every field access into one transaction");
    println!();
    println!("SoA guarantees that reading ONE field across many records is coalesced");
    println!("-- it says nothing about how many transactions a kernel that needs");
    println!("SEVERAL fields per thread will issue. A real bond-pricing kernel (Part 7)");
    println!("reads risk_free_rate, credit_spread, face_value, and time_to_maturity for");
    println!("every bond it prices -- four separate SoA arrays, so four separate");
    println!("(individually coalesced) transactions per warp, not one. SoA turns each of");
    println!("those four reads from a scattered, 32-byte-stride mess into a single clean");
    println!("transaction; it does not, and cannot, merge four logically different fields");
    println!("into one physical read.");
}
```

```bash
cargo run --release --bin 02_memory_coalescing_bond_soa
```

### File: `03_layout_evidence_offset_of.rs`

Section 18.2 Worked Example 18.2.2 -- struct layout confirmed via std::mem::offset_of!, this edition's honest replacement for SASS decoding.

```rust
// This file replaces the CUDA edition's Worked Example 18.2.2, which
// reads `cuobjdump --dump-sass` output to independently confirm the
// byte-stride multiplier a compiled kernel actually uses. This sandbox
// has neither `nvcc` nor `cuobjdump` installed at all (confirmed with
// `which nvcc cuobjdump` before writing this file) -- unlike the CUDA
// edition's own environment, which has the CUDA *toolkit* even though
// it has no GPU, this Rust edition's sandbox has no CUDA software
// whatsoever. Decoding SASS here would mean fabricating an
// `IMAD.WIDE`/`HFMA2.MMA` transcript this sandbox cannot produce --
// exactly what this book's standing rule against fabricated output
// forbids.
//
// What this sandbox CAN do, genuinely: ask `rustc` itself, via the
// stable `std::mem::offset_of!` macro, for the exact byte offset it
// assigned each field of ZeroCouponBondAoS -- a compiler-verified fact
// about the real, compiled memory layout, independent of the hand
// count Worked Example 18.2.1 already did. This is a different
// evidence source than SASS (the compiler's own layout computation,
// not its emitted machine code), but it is asking the same question
// SASS-decoding was asking: does the layout the hand count assumed
// match what actually got compiled? -- and it is something this
// sandbox can answer honestly.
#[repr(C)]
#[allow(dead_code)]
struct ZeroCouponBondAoS {
    face_value: f32,
    time_to_maturity: f32,
    risk_free_rate: f32,
    credit_spread: f32,
    present_value: f32,
    yield_to_maturity: f32,
    duration: f32,
    portfolio_weight: f32,
}

fn main() {
    println!("=== Section 18.2 (Rust edition): confirming struct layout via std::mem::offset_of!, not SASS ===\n");
    println!("this sandbox has neither nvcc nor cuobjdump installed -- confirmed genuinely:");
    println!("  `which nvcc cuobjdump` -> both report \"not found\" in this environment\n");
    println!("so this file asks rustc's own compiler-verified layout instead of decoding machine code:\n");

    let offsets = [
        ("face_value", std::mem::offset_of!(ZeroCouponBondAoS, face_value)),
        ("time_to_maturity", std::mem::offset_of!(ZeroCouponBondAoS, time_to_maturity)),
        ("risk_free_rate", std::mem::offset_of!(ZeroCouponBondAoS, risk_free_rate)),
        ("credit_spread", std::mem::offset_of!(ZeroCouponBondAoS, credit_spread)),
        ("present_value", std::mem::offset_of!(ZeroCouponBondAoS, present_value)),
        ("yield_to_maturity", std::mem::offset_of!(ZeroCouponBondAoS, yield_to_maturity)),
        ("duration", std::mem::offset_of!(ZeroCouponBondAoS, duration)),
        ("portfolio_weight", std::mem::offset_of!(ZeroCouponBondAoS, portfolio_weight)),
    ];
    println!("std::mem::offset_of!(ZeroCouponBondAoS, <field>), every field, genuinely compiled:");
    for (name, offset) in offsets.iter() {
        println!("  {:<18} -> byte offset {}", name, offset);
    }

    let struct_size = std::mem::size_of::<ZeroCouponBondAoS>();
    println!("\nstd::mem::size_of::<ZeroCouponBondAoS>() = {} bytes", struct_size);

    let rate_offset = std::mem::offset_of!(ZeroCouponBondAoS, risk_free_rate);
    let duration_offset = std::mem::offset_of!(ZeroCouponBondAoS, duration);
    println!("\ncross-checking against Worked Example 18.2.1's hand count:");
    println!(
        "  risk_free_rate offset = {} bytes = float-index {} -- hand count assumed float-index 2: {}",
        rate_offset,
        rate_offset / 4,
        if rate_offset / 4 == 2 { "CONFIRMED" } else { "MISMATCH" }
    );
    println!(
        "  duration offset       = {} bytes = float-index {} -- hand count assumed float-index 6: {}",
        duration_offset,
        duration_offset / 4,
        if duration_offset / 4 == 6 { "CONFIRMED" } else { "MISMATCH" }
    );
    println!(
        "  struct size            = {} bytes -- hand count assumed the AoS record stride was 32: {}",
        struct_size,
        if struct_size == 32 { "CONFIRMED" } else { "MISMATCH" }
    );

    println!("\nwhat this DOESN'T confirm, honestly: whether a compiled kernel actually reading");
    println!("bonds[idx].risk_free_rate emits an IMAD.WIDE with this exact byte count as its");
    println!("multiplier the way the CUDA edition's SASS did -- that claim rests on nvcc's code");
    println!("generation, which this sandbox has no way to inspect. What offset_of! DOES confirm,");
    println!("genuinely: the struct's IN-MEMORY layout -- the thing a correct compiled address");
    println!("calculation would have to be built from, whatever instructions it ends up as --");
    println!("matches Worked Example 18.2.1's hand-derived offsets exactly, not approximately.");

    println!("\n#[repr(C)] matters here specifically: without it, Rust's default struct layout is");
    println!("free to reorder fields for its own reasons (typically to reduce padding), and the");
    println!("comments above claiming risk_free_rate sits at offset 8 would not be guaranteed to");
    println!("hold at all. #[repr(C)] is what pins this struct to the same field order a C");
    println!("compiler would use, which is also what makes comparing its layout to the CUDA");
    println!("edition's C++ struct meaningful in the first place.");
}
```

```bash
cargo run --release --bin 03_layout_evidence_offset_of
```

### File: `04_naive_convolution.rs`

Section 18.3 -- naive convolution, redundant global reads counted exactly.

```rust
// The real kernel this file's host mirror corresponds to (never
// compiled or launched here -- this sandbox has no NVRTC/nvcc/GPU at
// all, established in files 01-03; the CUDA edition's own version of
// this file compiles this source with nvcc but never launches it
// either, so the host-side counting below is the genuine content on
// both editions):
//
// __global__ void naive_conv2d_kernel(const float* input, int input_h, int input_w,
//                                      const float* kernel_, int kernel_h, int kernel_w,
//                                      float* output, int output_h, int output_w) {
//     int out_row = blockIdx.y * blockDim.y + threadIdx.y;
//     int out_col = blockIdx.x * blockDim.x + threadIdx.x;
//     if (out_row >= output_h || out_col >= output_w) return;
//     float sum = 0.0f;
//     for (int k_r = 0; k_r < kernel_h; k_r++) {
//         for (int k_c = 0; k_c < kernel_w; k_c++) {
//             int in_row = out_row + k_r;
//             int in_col = out_col + k_c;
//             sum += input[in_row * input_w + in_col] * kernel_[k_r * kernel_w + k_c];
//         }
//     }
//     output[out_row * output_w + out_col] = sum;
// }

// A host mirror of the identical per-thread logic, PLUS a genuine read
// counter per input cell -- this is what makes "36 reads for 16 unique
// values" a measured fact rather than an assertion.
fn naive_conv2d_host(
    input: &[f32],
    input_w: usize,
    kernel: &[f32],
    kernel_h: usize,
    kernel_w: usize,
    output: &mut [f32],
    output_h: usize,
    output_w: usize,
    read_counts: &mut [u32],
) {
    for out_row in 0..output_h {
        for out_col in 0..output_w {
            let mut sum = 0.0f32;
            for k_r in 0..kernel_h {
                for k_c in 0..kernel_w {
                    let in_row = out_row + k_r;
                    let in_col = out_col + k_c;
                    read_counts[in_row * input_w + in_col] += 1; // one genuine global read, counted
                    sum += input[in_row * input_w + in_col] * kernel[k_r * kernel_w + k_c];
                }
            }
            output[out_row * output_w + out_col] = sum;
        }
    }
}

fn main() {
    println!("=== Section 18.3: naive convolution, redundant reads counted exactly ===\n");

    let input: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let kernel: [f32; 9] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let mut output = [0.0f32; 4];
    let mut read_counts = [0u32; 16];

    naive_conv2d_host(&input, 4, &kernel, 3, 3, &mut output, 2, 2, &mut read_counts);

    println!("Input (4x4):");
    for r in 0..4 {
        let row: Vec<String> = (0..4).map(|c| format!("{:3.0}", input[r * 4 + c])).collect();
        println!(" {}", row.join(""));
    }
    println!("Kernel (3x3):");
    for r in 0..3 {
        let row: Vec<String> = (0..3).map(|c| format!(" {}", kernel[r * 3 + c])).collect();
        println!(" {}", row.join(""));
    }

    println!("\noutput = {:.0}  {:.0}", output[0], output[1]);
    println!("         {:.0}  {:.0}\n", output[2], output[3]);

    println!("read_counts per input cell:");
    let mut total_reads = 0u32;
    for r in 0..4 {
        let row: Vec<String> = (0..4).map(|c| format!(" {}", read_counts[r * 4 + c])).collect();
        println!(" {}", row.join(""));
        for c in 0..4 {
            total_reads += read_counts[r * 4 + c];
        }
    }
    println!("\ntotal global-memory reads = {} (for 16 unique input values)", total_reads);
    println!("independent check: 4 outputs x 9 kernel taps each = {}", 4 * 9);
}
```

```bash
cargo run --release --bin 04_naive_convolution
```

### File: `05_tiled_convolution_shared_memory.rs`

Section 18.3 -- the same convolution staged through a shared-memory tile.

```rust
// This chapter's own small worked example: TILE=2 output elements per
// block per side, KERNEL_SIZE=3, so one block's input footprint is
// SHARED_DIM = TILE + KERNEL_SIZE - 1 = 4 -- exactly this example's
// entire 4x4 input in a single tile.
const TILE: usize = 2;
const KERNEL_SIZE: usize = 3;
const SHARED_DIM: usize = TILE + KERNEL_SIZE - 1;

// The real kernel this file's host mirror corresponds to (never
// compiled here -- see file 03's note on why): a real __shared__ tile
// array and a real __syncthreads() barrier between the load phase
// (every thread in the oversized block loads exactly one tile cell,
// including threads that will never compute an output) and the
// compute phase (only the TILE x TILE interior threads own an output,
// and every read from here on is from `tile`, not global memory).
//
// __global__ void tiled_conv2d_kernel(const float* input, int input_h, int input_w,
//                                      const float* kernel_, int kernel_h, int kernel_w,
//                                      float* output, int output_h, int output_w) {
//     __shared__ float tile[SHARED_DIM * SHARED_DIM];
//     int in_row = blockIdx.y * TILE + threadIdx.y;
//     int in_col = blockIdx.x * TILE + threadIdx.x;
//     if (in_row < input_h && in_col < input_w) {
//         tile[threadIdx.y * SHARED_DIM + threadIdx.x] = input[in_row * input_w + in_col];
//     } else {
//         tile[threadIdx.y * SHARED_DIM + threadIdx.x] = 0.0f;
//     }
//     __syncthreads();
//     if (threadIdx.y >= TILE || threadIdx.x >= TILE) return;
//     int out_row = blockIdx.y * TILE + threadIdx.y;
//     int out_col = blockIdx.x * TILE + threadIdx.x;
//     if (out_row >= output_h || out_col >= output_w) return;
//     float sum = 0.0f;
//     for (int k_r = 0; k_r < kernel_h; k_r++)
//         for (int k_c = 0; k_c < kernel_w; k_c++)
//             sum += tile[(threadIdx.y + k_r) * SHARED_DIM + (threadIdx.x + k_c)] * kernel_[k_r * kernel_w + k_c];
//     output[out_row * output_w + out_col] = sum;
// }

// A host mirror of the IDENTICAL two-phase discipline -- load phase,
// then compute phase -- with a genuine global-read counter on the
// load phase only, since that is the only phase that touches `input`.
fn tiled_conv2d_host(
    input: &[f32],
    input_h: usize,
    input_w: usize,
    kernel: &[f32],
    kernel_h: usize,
    kernel_w: usize,
    output: &mut [f32],
    output_h: usize,
    output_w: usize,
    read_counts: &mut [u32],
) {
    let mut tile = [0.0f32; SHARED_DIM * SHARED_DIM];

    // Load phase: SHARED_DIM x SHARED_DIM "threads" each load exactly
    // one cell -- for this example, the whole 4x4 input in one block.
    for ty in 0..SHARED_DIM {
        for tx in 0..SHARED_DIM {
            let (in_row, in_col) = (ty, tx); // blockIdx=(0,0) for this single-block example
            if in_row < input_h && in_col < input_w {
                read_counts[in_row * input_w + in_col] += 1; // one genuine global read, counted
                tile[ty * SHARED_DIM + tx] = input[in_row * input_w + in_col];
            } else {
                tile[ty * SHARED_DIM + tx] = 0.0;
            }
        }
    }
    // barrier() in spirit: the load phase above is fully complete
    // before the compute phase below ever reads `tile`.

    // Compute phase: only the TILE x TILE interior "threads" produce
    // an output, and every read from here on is from `tile`, not `input`.
    for ty in 0..TILE {
        for tx in 0..TILE {
            let (out_row, out_col) = (ty, tx);
            if out_row >= output_h || out_col >= output_w {
                continue;
            }
            let mut sum = 0.0f32;
            for k_r in 0..kernel_h {
                for k_c in 0..kernel_w {
                    sum += tile[(ty + k_r) * SHARED_DIM + (tx + k_c)] * kernel[k_r * kernel_w + k_c];
                }
            }
            output[out_row * output_w + out_col] = sum;
        }
    }
}

fn main() {
    println!("=== Section 18.3: the same computation, staged through shared memory ===\n");

    let input: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let kernel: [f32; 9] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let mut output = [0.0f32; 4];
    let mut read_counts = [0u32; 16];

    println!(
        "TILE={}, KERNEL_SIZE={}, SHARED_DIM={} (this example's whole 4x4 input, one block)\n",
        TILE, KERNEL_SIZE, SHARED_DIM
    );

    tiled_conv2d_host(&input, 4, 4, &kernel, 3, 3, &mut output, 2, 2, &mut read_counts);

    println!("output = {:.0}  {:.0}", output[0], output[1]);
    println!("         {:.0}  {:.0}\n", output[2], output[3]);

    println!("read_counts per input cell (load phase only):");
    let mut total_reads = 0u32;
    for r in 0..4 {
        let row: Vec<String> = (0..4).map(|c| format!(" {}", read_counts[r * 4 + c])).collect();
        println!(" {}", row.join(""));
        for c in 0..4 {
            total_reads += read_counts[r * 4 + c];
        }
    }
    println!("\ntotal global-memory reads = {} (down from the naive kernel's 36)", total_reads);
    let matches = output[0] == 30.0 && output[1] == 35.0 && output[2] == 50.0 && output[3] == 55.0;
    println!("outputs match the naive kernel exactly: {}", matches);
}
```

```bash
cargo run --release --bin 05_tiled_convolution_shared_memory
```

### File: `06_padded_convolution.rs`

Section 18.3 -- padding trades a shrinking output for a fixed-size one.

```rust
// Pad a 2D input with `padding` zeros on every side.
fn pad_tensor(input: &[f32], h: usize, w: usize, padding: usize) -> (Vec<f32>, usize, usize) {
    let out_h = h + 2 * padding;
    let out_w = w + 2 * padding;
    let mut padded = vec![0.0f32; out_h * out_w];
    for r in 0..h {
        for c in 0..w {
            padded[(r + padding) * out_w + (c + padding)] = input[r * w + c];
        }
    }
    (padded, out_h, out_w)
}

fn conv2d_host(
    input: &[f32],
    input_w: usize,
    kernel: &[f32],
    kernel_h: usize,
    kernel_w: usize,
    output: &mut [f32],
    output_h: usize,
    output_w: usize,
) {
    for out_row in 0..output_h {
        for out_col in 0..output_w {
            let mut sum = 0.0f32;
            for k_r in 0..kernel_h {
                for k_c in 0..kernel_w {
                    let in_row = out_row + k_r;
                    let in_col = out_col + k_c;
                    sum += input[in_row * input_w + in_col] * kernel[k_r * kernel_w + k_c];
                }
            }
            output[out_row * output_w + out_col] = sum;
        }
    }
}

fn main() {
    println!("=== Section 18.3: padding trades output size for border zeros ===\n");

    let input: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let kernel: [f32; 9] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    let (padded, padded_h, padded_w) = pad_tensor(&input, 4, 4, 1);
    println!("padded input ({}x{}), zero border:", padded_h, padded_w);
    for r in 0..padded_h {
        let row: Vec<String> = (0..padded_w).map(|c| format!("{:3.0}", padded[r * padded_w + c])).collect();
        println!(" {}", row.join(""));
    }

    // output_h/w = input_h/w + 2*padding - kernel_size + 1 = 4+2-3+1 = 4
    let (output_h, output_w) = (4usize, 4usize);
    let mut output = vec![0.0f32; output_h * output_w];
    conv2d_host(&padded, padded_w, &kernel, 3, 3, &mut output, output_h, output_w);

    println!("\noutput ({}x{}, matching the input's own 4x4 size):", output_h, output_w);
    for r in 0..output_h {
        let row: Vec<String> = (0..output_w).map(|c| format!("{:3.0}", output[r * output_w + c])).collect();
        println!(" {}", row.join(""));
    }

    println!("\noutput(0,3) (top-right corner) = {:.0} (expected 11)", output[0 * output_w + 3]);
    println!(
        "output(1,1) (interior, should equal unpadded output(0,0)=30) = {:.0}",
        output[1 * output_w + 1]
    );
}
```

```bash
cargo run --release --bin 06_padded_convolution
```

### File: `07_warp_shuffle_reduction.rs`

Section 18.4 -- warp-level shuffle reduction, 8 lanes, traced by hand.

```rust
// The real intrinsics this file's host mirror corresponds to (never
// compiled here -- see file 03's note): a real warp-shuffle reduction
// using __shfl_down_sync, which reads a value from a DIFFERENT lane's
// own live register, without either lane ever writing to memory. A
// real warp is 32 lanes wide (WARP_SIZE below); this file traces the
// identical pattern on 8, Chapter 4.4's own scaling convention.
//
// #define WARP_SIZE 32
// __device__ float warp_reduce_sum(float value) {
//     float v = value;
//     for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
//         v += __shfl_down_sync(0xffffffff, v, offset);
//     }
//     return v;
// }
// __global__ void warp_reduce_kernel(const float* input, float* output, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     float v = (idx < n) ? input[idx] : 0.0f;
//     float total = warp_reduce_sum(v);
//     if (threadIdx.x == 0) output[blockIdx.x] = total;
// }

// A host mirror of the IDENTICAL register-to-register exchange
// pattern, scaled down to 8 lanes (Chapter 4.4's own scaling
// convention), tracing every round exactly the way __shfl_down_sync
// would: lane i's new value is lane i's old value plus lane
// (i+offset)'s old value, for i < offset.
fn warp_reduce_sum_host_traced(lanes: &mut [f32]) {
    let num_lanes = lanes.len();
    let mut offset = num_lanes / 2;
    let mut round = 1;
    while offset > 0 {
        let old_values: Vec<f32> = lanes.to_vec();
        // Only lanes below the current offset receive a new value this
        // round -- lanes at or above it are done contributing and are
        // never read again by any later round.
        for i in 0..offset {
            lanes[i] = old_values[i] + old_values[i + offset];
        }
        print!("round {} (offset={}): value =", round, offset);
        for i in 0..num_lanes {
            if i < offset {
                print!(" {:5.1}", lanes[i]);
            } else {
                print!("     .");
            }
        }
        println!();
        offset /= 2;
        round += 1;
    }
}

fn main() {
    println!("=== Section 18.4: warp-level shuffle reduction, 8 lanes, traced by hand ===\n");

    let mut lanes: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    print!("lane:   ");
    for i in 0..8 {
        print!(" {:5}", i);
    }
    print!("\nvalue:  ");
    for v in lanes.iter() {
        print!(" {:5.1}", v);
    }
    println!("\n");

    warp_reduce_sum_host_traced(&mut lanes);

    println!(
        "\nlane 0 ends up holding {:.1} -- the full sum -- after exactly log2(8)=3 exchanges",
        lanes[0]
    );
    println!("and zero memory operations.");
}
```

```bash
cargo run --release --bin 07_warp_shuffle_reduction
```

### File: `08_warp_shuffle_common_trap.rs`

Section 18.4 COMMON TRAP -- bounds-check early return meets warp shuffle.

```rust
// Section 18.1's bounds-check pattern meets Section 18.4's shuffle:
// this file genuinely demonstrates what goes wrong when out-of-range
// lanes return early INSTEAD of participating with an identity value.

// CORRECT: every lane reaches every round, out-of-range lanes
// contribute the reduction's identity element (0 for a sum).
fn reduce_correct(lanes: &mut [f32], lane_active: &[bool]) -> f32 {
    let num_lanes = lanes.len();
    let mut offset = num_lanes / 2;
    while offset > 0 {
        let old_values: Vec<f32> = (0..num_lanes).map(|i| if lane_active[i] { lanes[i] } else { 0.0 }).collect();
        for i in 0..offset {
            lanes[i] = old_values[i] + old_values[i + offset];
        }
        offset /= 2;
    }
    lanes[0]
}

// BROKEN: out-of-range lanes "return early" -- their slot is never
// written with a real value at all, so it keeps whatever was already
// sitting in that memory before this reduction began. A genuinely
// uninitialized local array is not a reliable way to *demonstrate*
// this in a book (Rust also won't even compile a read of a genuinely
// uninitialized `[f32; N]` without `unsafe`, and an `unsafe`-sourced
// value could easily come back as all zeros, which would silently
// "fix" the bug for the reader by accident -- exactly the failure mode
// the CUDA edition's own first draft of this file hit). So this
// version makes the "whatever was already there" content explicit and
// reproducible: `poison` is filled by the caller with a genuinely
// computed (not hand-picked) non-zero bit pattern standing in for
// stale register/memory content, and `poisoned[]` starts from that
// pattern instead of from any kind of uninitialized memory.
fn reduce_broken_early_return(lanes: &[f32], lane_active: &[bool], poison: &[f32]) -> f32 {
    let num_lanes = lanes.len();
    let mut poisoned: Vec<f32> = poison.to_vec();
    for i in 0..num_lanes {
        if !lane_active[i] {
            continue; // "early return": this slot's poison value is never overwritten
        }
        poisoned[i] = lanes[i];
    }
    // every lane still participates in the shuffle rounds, including
    // the ones that never received a real value
    let mut offset = num_lanes / 2;
    while offset > 0 {
        let old_values: Vec<f32> = poisoned.clone();
        for i in 0..offset {
            poisoned[i] = old_values[i] + old_values[i + offset];
        }
        offset /= 2;
    }
    poisoned[0]
}

fn main() {
    println!("=== Section 18.4 COMMON TRAP: bounds-check early-return meets warp shuffle ===\n");

    // idx < size for indices 0..3, idx >= size for indices 4..7 --
    // exactly a bounds-check pattern where the tail of a warp is
    // covering indices past the end of a buffer.
    let lane_active: [bool; 8] = [true, true, true, true, false, false, false, false];

    println!("4 real elements: [1,2,3,4], 4 out-of-range lanes (idx >= size)\n");

    println!("--- correct: out-of-range lanes contribute the identity element (0) ---");
    let mut correct_lanes: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    let result_correct = reduce_correct(&mut correct_lanes, &lane_active);
    println!("result = {:.1} (expected 1+2+3+4 = 10)\n", result_correct);

    println!("--- broken: out-of-range lanes return early, leaving stale memory behind ---");
    // A genuinely non-hand-picked bit pattern for "whatever was
    // already there" -- reinterpreting arithmetic on raw bits as f32
    // bits via f32::from_bits, exactly the kind of nonsense value a
    // stale register or an uninitialized shared-memory slot can
    // genuinely hold on real hardware. Only lanes 4-7 matter (lanes
    // 0-3 get overwritten with the real elements before any reduction
    // happens), but all 8 are filled to keep the array fully defined.
    let poison_floats: [f32; 8] = std::array::from_fn(|i| {
        let bits: u32 = 0xDEADBEEFu32 ^ ((i as u32).wrapping_mul(2654435761u32));
        f32::from_bits(bits)
    });
    println!(
        "simulated stale content for lanes 4-7: {:e} {:e} {:e} {:e}",
        poison_floats[4], poison_floats[5], poison_floats[6], poison_floats[7]
    );

    let active_values: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]; // lanes 0-3 are the real elements
    let result_broken = reduce_broken_early_return(&active_values, &lane_active, &poison_floats);
    println!(
        "result = {:.4e} (should NOT be 10 -- lanes 4-7 poisoned the sum instead of contributing 0)",
        result_broken
    );
    println!("on real hardware the exact stale value is unpredictable across compilers, optimization");
    println!("levels, and runs -- the guarantee this file demonstrates is only that it is NOT the");
    println!("correct identity element, because lanes 4-7 never received one before the shuffle ran.\n");

    println!("CONCLUSION: the fix is not to skip the bounds check Section 18.1 established is required --");
    println!("it is to let every lane reach the shuffle, substituting the reduction's identity value for");
    println!("out-of-range lanes instead of letting them exit before the shuffle rounds run.");
}
```

```bash
cargo run --release --bin 08_warp_shuffle_common_trap
```

### `Cargo.toml` (shared by all eight binaries above)

```toml
[package]
name = "rust_ch18"
version = "0.1.0"
edition = "2024"

[dependencies]
cudarc = { version = "0.19", default-features = false, features = ["driver", "nvrtc", "std", "dynamic-loading", "cuda-12060"] }
```

Genuinely built with `cargo build --release` and `cargo clean && cargo build --release` (a full rebuild from empty, confirming no binary depended on stale incremental state), both producing eight binaries with zero warnings.

## Chapter Summary

A kernel launch's block count is ceiling division, not floor division — `(size + N - 1) / N`, computed here through the same `LaunchConfig` type this book has used since Chapter 4 — and the reason is asymmetric: rounding down silently drops real elements no bounds check can rescue (the COMMON TRAP genuinely demonstrated in Section 18.1 drops `16` real elements from a `10,000`-element launch), while rounding up and pairing it with a bounds check inside the kernel wastes a bounded, known number of idle threads (`192` for this chapter's million-element example, `112` for its ten-thousand-element one) in exchange for guaranteed full coverage. This edition genuinely attempts the real `<<<>>>` launch and honestly reports what this driver-less, toolkit-less sandbox actually returns — a panic from `cudarc::driver::CudaContext::new(0)` while it searches for `libcuda.so`/`libnvcuda.so`, caught and reported rather than fabricated, the same failure this book has hit at this exact call since Chapter 4.2. Memory coalescing is Chapter 4.4's warp-bandwidth argument made concrete on a real eight-field financial record: reading one field across four bond records costs one transaction in Struct-of-Arrays layout and four in Array-of-Structs, a `4×` penalty on this chapter's scaled-down example that reflects the same `8×` byte-stride penalty (`32` bytes vs. `4` bytes) Part 7's real portfolio pricing measures directly. This sandbox has no CUDA toolkit at all — not even `nvcc` or `cuobjdump` — so unlike the CUDA edition, which independently confirms that byte-stride penalty by decoding compiled SASS, this edition confirms it a different, genuinely available way: asking `rustc` itself, through the stable `std::mem::offset_of!` macro, for the exact byte offset it assigned every field of a `#[repr(C)]` struct, and finding it matches the hand-derived offsets exactly. A naive convolution kernel reads most of its input several times over — `36` total reads for `16` unique values in this chapter's own `4×4` example, genuinely counted rather than asserted — purely because neighboring output threads' windows overlap; staging the block's entire input footprint into memory once, using more threads to load than will ever compute an output, cuts that down to exactly `16` global reads, with a barrier standing between "correct" and a race condition (documented here, never compiled — this sandbox cannot compile or launch real `__global__` kernels at all). Padding trades a shrinking output for a fixed-size one by surrounding the input with zeros rather than narrowing the valid window, verified here at both a border position (`11`) and an interior one that exactly reproduces the unpadded chapter's own first answer (`30`), just shifted. Finally, warp-level shuffle instructions collapse the last several rounds of any tree reduction into register-to-register exchanges with no memory traffic at all — traced by hand here across `8` lanes in exactly `log2(8)=3` rounds — but they require every lane in the warp to still be executing, which makes Section 18.1's own bounds-check pattern, applied carelessly to a kernel that also uses shuffles, the exact mechanism that can break one — genuinely demonstrated here with a broken reduction that produces `5.4497 × 10²⁶` instead of `10.0`, using a deterministically computed stale-memory pattern rather than actually-uninitialized memory, which Rust will not even let this book read outside `unsafe` in the first place.

## Self-Check Questions

1. For `size = 5,000` and `THREADS_PER_BLOCK = 512`, compute `num_blocks`, the total number of threads launched, the number wasted, and how many threads in the last block are actually active.
2. A warp-scaled group of `4` threads reads the `duration` field (the seventh of the eight `ZeroCouponBondAoS` fields, float-index `6` within each `8`-float AoS record) for bond records `0` through `3`, using the same `16`-byte/`4`-float chunk convention as Worked Example 18.2.1. How many transactions does the AoS layout need, and what is its bus utilization?
3. Worked Example 18.2.2 confirmed `ZeroCouponBondAoS`'s layout with `std::mem::offset_of!` instead of decoding SASS. Suppose a colleague inserts a new `notional: f32` field into the struct, between `credit_spread` and `present_value`. Using `#[repr(C)]`'s guarantee that fields are laid out in declaration order with no reordering, what would `std::mem::offset_of!(ZeroCouponBondAoS, present_value)` now report, and what would `std::mem::size_of::<ZeroCouponBondAoS>()` become? Why does `#[repr(C)]` make this a mechanical calculation rather than a guess?
4. A (documented, never-compiled) kernel loads its shared-memory tile and then begins its convolution loop immediately, without a barrier first. Some threads finish their load-and-compute before other threads in the same block have even finished loading. What specific kind of wrong value can this produce, and which threads are affected?
5. Using this chapter's `4×4` input and `3×3` kernel with `padding=1`, compute the padded output value at the top-right corner, `output(0,3)`.
6. A warp-level reduction kernel has `4` of its `8` (warp-scaled) lanes return early, before the shuffle rounds, because a bounds check found `idx >= size` for those four threads. Why is this a problem, and which earlier section of this same chapter introduced the pattern now causing it?

## Where We Go Next

Chapter 19 (`part5/02-performance-optimization.md`) turns from kernel *design* to kernel *measurement*: vectorized memory access, loop unrolling, compile-time specialization, and the benchmarking discipline used to tell whether any of this chapter's optimizations — ceiling-division launches, SoA layouts, shared-memory tiling, warp shuffles — actually helped, rather than simply looking like they should have. Because this sandbox still has no GPU driver and no CUDA toolkit, Chapter 19's own kernel bodies and numbers will need the identical honest treatment this chapter gave them: host-side arithmetic mirrors and compiler-verified evidence where genuine execution is possible, and explicit `UNVERIFIED — pending real-GPU test` tags where it is not.

## Worked Solutions

**1.** `num_blocks = (5,000 + 511) / 512 = 5,511 / 512 = 10`. Total threads launched: `10 × 512 = 5,120`. Wasted: `5,120 - 5,000 = 120`. The last block is block `9`, covering global indices `9 × 512 = 4,608` through `5,119`. Valid indices are `4,608` through `4,999` — `392` active threads — leaving `512 - 392 = 120` idle threads in that one block, exactly matching the wasted count computed from the totals. (Independently verified: `compute_launch_config(5_000, 512)` genuinely computes `grid_dim = 10`.)

**2.** Record `0`'s `duration` sits at float-index `0×8+6=6`; record `1`'s at `8+6=14`; record `2`'s at `16+6=22`; record `3`'s at `24+6=30`. Dividing by `4` to find each `16`-byte chunk: `6→chunk 1`, `14→chunk 3`, `22→chunk 5`, `30→chunk 7` — four distinct chunks, so `4` transactions are needed. Bytes moved: `4×16=64`; bytes used: `4×4=16`; utilization: `16/64=25%` — structurally identical to Worked Example 18.2.1's `risk_free_rate` case, confirming the AoS penalty isn't specific to any one field. (Independently verified with the same `offset_of!`-derived float-index arithmetic Worked Example 18.2.1 used.)

**3.** Genuinely compiling a `ZeroCouponBondAoSExtended` struct with `notional: f32` inserted between `credit_spread` and `present_value` gives `std::mem::offset_of!(ZeroCouponBondAoSExtended, present_value) = 20` and `std::mem::size_of::<ZeroCouponBondAoSExtended>() = 36`. Every field from `notional` onward shifts `4` bytes later than it sat in the original `8`-field struct, because `#[repr(C)]` lays fields out strictly in declaration order with no reordering and no gaps between same-alignment `f32` fields — inserting one `4`-byte field mechanically adds `4` bytes to every later offset and to the total size, the same way inserting a line into a numbered list renumbers everything after it. This is what makes `#[repr(C)]` layout a calculation rather than a guess: the compiler's rule is fixed and public (declaration order, natural alignment, no reordering), so the new offsets follow from the field list alone, without needing to inspect what the compiler actually did the way SASS decoding would.

**4.** This produces a race condition: a thread that reaches the convolution loop before every thread assigned to load a value it needs has actually written that value reads whatever garbage or stale data happened to already be sitting in that shared-memory slot, not the input value that thread was supposed to load. The affected threads are unpredictable and can vary from run to run — specifically, any thread whose `3×3` window includes a tile cell that a *slower* sibling thread (one still mid-load, or not yet scheduled) hasn't written yet. A barrier between the load phase and the compute phase exists precisely to rule this out, by making every thread in the block wait until all of them have finished loading before any of them is allowed to start reading — this edition's own host mirror (Section 18.3, file `05`) enforces the identical ordering by finishing its load loop completely, in program order, before its compute loop begins.

**5.** `output(0,3)`'s window needs input columns `2` through `4` and rows `-1` through `1`. Row `-1` is entirely padding and contributes `0` regardless of kernel weights. For row `0`: `input[0,2]=3` × kernel `[1][0]=0` → `0`; `input[0,3]=4` × kernel `[1][1]=1` → `4`; column `4` is padding × kernel `[1][2]=0` → `0`. For row `1`: `input[1,2]=7` × kernel `[2][0]=1` → `7`; `input[1,3]=8` × kernel `[2][1]=0` → `0`; column `4` padding × kernel `[2][2]=1` → `0`. Total: `4 + 7 = 11` — matching file `06`'s own genuinely computed `output(0,3) = 11` exactly.

**6.** A warp shuffle reads a value from another lane's live register in the same warp — it has no defined result when the lane it's reading from has already exited. The four early-returning lanes are no longer executing at all by the time the remaining lanes reach the shuffle, so any exchange involving them reads undefined content instead of a meaningful partial sum, silently corrupting the reduction — genuinely demonstrated in Section 18.4 (file `08`) as a reduction that produces `5.4497 × 10²⁶` instead of `10.0`, using a deterministically computed stand-in for stale memory rather than actually-uninitialized data, which Rust refuses to let safe code read at all. The pattern responsible is Section 18.1's own bounds-check idiom — entirely correct and necessary on its own, as Section 18.1 established, but unsafe to combine with a later warp shuffle unless every lane is guaranteed to still reach the shuffle call, typically by substituting the reduction's identity value (`0` for a sum) for out-of-range lanes instead of letting them exit early, exactly as `reduce_correct` does in file `08`.
