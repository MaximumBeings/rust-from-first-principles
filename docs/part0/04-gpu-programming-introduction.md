# Chapter 4: GPU Programming Introduction — `cudarc`'s Driver Model

> "A CUDA kernel and a Rust function are both just code waiting to run somewhere. The question this chapter answers is: where, exactly, is 'somewhere' — which memory a pointer actually points into, which of potentially millions of threads is executing the line you're reading, and what has to happen, explicitly, physically, before any of that code can run at all."

**What you will understand by the end of this chapter:**

- `cudarc`'s ownership model — `CudaContext`, `CudaStream`, and `CudaSlice<T>` — as the Rust-safe handles standing in for a raw CUDA driver's context, stream, and device pointer, and the real `LaunchConfig` struct that plays the same role CUDA's `<<<grid, block>>>` launch syntax plays
- Why host memory (RAM) and device memory (VRAM) are two physically separate memories in `cudarc` exactly as they are in CUDA C++, connected by an explicit `clone_htod`/`clone_dtoh` copy rather than a cast
- The GPU memory hierarchy at a first pass — global, shared, and per-thread local memory — and where `cudarc`'s Rust types stop describing it and a kernel's own CUDA C source takes over
- Why Chapter 3's coalescing argument generalizes past particles to *every* global memory access pattern this book's kernels will write, in `cudarc` exactly as much as in raw CUDA
- The **broadcast** pattern — one thread, one output element, mediated by a boundary check — expressed through `LaunchConfig::for_num_elems` and a genuine kernel source string
- Why this chapter's own device-touching calls — `CudaContext::new`, `compile_ptx` — fail in this sandbox exactly once, in exactly one place, with a real, reproducible, honestly-reported error, and precisely what that failure does and doesn't tell you about the Rust code sitting around it

**What you need to know first:**

- Getting Started's "How this book verifies its claims" section — this chapter is where that two-tier promise (CPU content compiled and run for real, GPU content host-compiled but kernel-level results deferred to real hardware) stops being an abstract policy and becomes a concrete, visible pattern in every worked example below
- Chapter 2.4 (`Drop`-based RAII) — `CudaContext`, `CudaStream`, and `CudaSlice<T>` all free their underlying CUDA resources via `Drop`, the same discipline Chapter 2 built by hand for a much simpler resource
- Chapter 3 in full — Section 4.4 is a direct continuation of Chapter 3's bus-utilization and coalescing argument, restated at full generality rather than reopened
- If you've read the CUDA edition: this chapter follows its Chapter 4 section-for-section, with one structural difference driven by honesty rather than by choice. CUDA kernels there are written once, in `.cu` files, and compiled ahead-of-time by `nvcc` into the same binary as the host code. `cudarc` kernels are ordinary CUDA C source held as Rust string literals, compiled at *runtime* by NVRTC (`compile_ptx`) into PTX, then loaded into a running `CudaContext` (`load_module`) and looked up by name (`load_function`) before they can be launched at all. That extra machinery is this chapter's real subject, and it's also exactly the machinery this sandbox cannot exercise past the compile-the-Rust-that-calls-it stage, for reasons Worked Example 4.2.1 makes concrete.

## 4.1 The Thread Hierarchy: `LaunchConfig` as Rust's `<<<grid, block>>>` `[FOUNDATIONAL]`

### Intuition

CUDA C++ launches a kernel with `kernel<<<gridDim, blockDim>>>(args)` — special triple-angle-bracket syntax baked into the language by `nvcc`. Rust has no such syntax, and doesn't need one: `cudarc` represents the exact same two numbers as an ordinary Rust struct, `LaunchConfig { grid_dim, block_dim, shared_mem_bytes }`, and hands it to an ordinary method call, `stream.launch_builder(&f).arg(...).launch(cfg)`. Nothing about the underlying hardware model changes — a grid of blocks, each block a group of threads — only the syntax describing it does.

### Background

`cudarc`'s real `LaunchConfig` (`cudarc::driver::LaunchConfig`), as the crate actually defines it:

```rust
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}
```

| | What it is | Accessed as |
|---|---|---|
| Grid | All threads launched by one kernel call | `cfg.grid_dim` — set at launch time, not visible inside the kernel as a single value |
| Block | A group of threads scheduled together | `blockIdx`, `blockDim` inside the kernel's own CUDA C source |
| Thread | One single execution of the kernel body | `threadIdx` inside the kernel's own CUDA C source |

`cudarc` ships a convenience constructor, `LaunchConfig::for_num_elems(n)`, whose real source is exactly this:

```rust
pub fn for_num_elems(n: u32) -> Self {
    const NUM_THREADS: u32 = 1024;
    let num_blocks = n.div_ceil(NUM_THREADS);
    Self {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}
```

### Worked Example 4.1.1 — every global ID, computed by hand from a real `LaunchConfig`

For a hand-sized `LaunchConfig { grid_dim: (3, 1, 1), block_dim: (4, 1, 1), .. }` — 3 blocks of 4 threads each, 12 threads total — block 0's four threads compute `i = 0*4+0=0` through `0*4+3=3`; block 1's compute `i = 4` through `7`; block 2's compute `i = 8` through `11`, using the same `blockIdx.x * blockDim.x + threadIdx.x` formula a kernel's own CUDA C source would use internally. Compiled and run as the complete `01_launch_config.rs` further below, whose `main()` builds a real `cudarc::driver::LaunchConfig` value and walks its `grid_dim`/`block_dim` fields on the host:

```bash
cargo run --release --bin 01_launch_config
```

Genuinely compiled and run:

```
1D launch: grid_dim=(3, 1, 1) block_dim=(4, 1, 1) -- 12 total threads
  block 0, thread 0 -> global i = 0
  block 0, thread 1 -> global i = 1
  block 0, thread 2 -> global i = 2
  block 0, thread 3 -> global i = 3
  block 1, thread 0 -> global i = 4
  block 1, thread 1 -> global i = 5
  block 1, thread 2 -> global i = 6
  block 1, thread 3 -> global i = 7
  block 2, thread 0 -> global i = 8
  block 2, thread 1 -> global i = 9
  block 2, thread 2 -> global i = 10
  block 2, thread 3 -> global i = 11
```

The same program then calls the real `LaunchConfig::for_num_elems` — the constructor every kernel launch in this book actually uses from here forward — across a range of sizes, and prints its genuine `grid_dim`/`block_dim` output alongside how many of the launched threads exceed `n`:

```
LaunchConfig::for_num_elems is what this book actually launches kernels with:
  n=         7 -> grid_dim=(1, 1, 1) block_dim=(1024, 1, 1) (1024 threads launched, 1017 extra beyond n)
  n=         8 -> grid_dim=(1, 1, 1) block_dim=(1024, 1, 1) (1024 threads launched, 1016 extra beyond n)
  n=      1025 -> grid_dim=(2, 1, 1) block_dim=(1024, 1, 1) (2048 threads launched, 1023 extra beyond n)
  n=  20000000 -> grid_dim=(19532, 1, 1) block_dim=(1024, 1, 1) (20000768 threads launched, 768 extra beyond n)
```

`for_num_elems` always launches in multiples of 1024 threads per block, so unless `n` happens to divide 1024 evenly, some threads are always launched with nothing real to do — for `n=7`, 1017 of the 1024 launched threads have no valid `i`. Section 4.5's boundary check exists precisely to keep those extra threads from touching memory they were never given.

### ASCII Diagram — the three-level hierarchy

```
Grid (this kernel launch)
 +-- Block (0)                Block (1)                Block (2)
      +-- Thread 0 -> i=0          +-- Thread 0 -> i=4          +-- Thread 0 -> i=8
      +-- Thread 1 -> i=1          +-- Thread 1 -> i=5          +-- Thread 1 -> i=9
      +-- Thread 2 -> i=2          +-- Thread 2 -> i=6          +-- Thread 2 -> i=10
      +-- Thread 3 -> i=3          +-- Thread 3 -> i=7          +-- Thread 3 -> i=11
```

> `[COMMON TRAP]` `LaunchConfig::for_num_elems` hardcodes 1024 threads per block, the maximum most hardware allows — this isn't a universal constant, it's `cudarc`'s own convenience default. A kernel whose per-thread body needs more registers or shared memory than 1024 threads' worth can fit may need a smaller, hand-built `LaunchConfig` instead; this book uses the convenience constructor until a specific kernel's resource needs require otherwise.

## 4.2 Host and Device: Two Separate Memory Spaces `[FOUNDATIONAL]`

### Intuition

Exactly as in CUDA C++, the CPU and GPU in a `cudarc` program are two physically separate pools of memory — host RAM and device VRAM — connected by a comparatively slow bus. `cudarc` makes this distinction a *type-level* one: a `Vec<f32>` and a `CudaSlice<f32>` are different Rust types, and there is no implicit conversion between them — only an explicit, physical copy, `stream.clone_htod(&host_vec)` or `stream.clone_dtoh(&device_slice)`, each one a real transfer across that bus.

### Background

| | Host memory (RAM) | Device memory (VRAM) |
|---|---|---|
| Rust type | `Vec<T>`, `[T; N]`, any `HostSlice<T>` | `cudarc::driver::CudaSlice<T>` |
| Allocated with | Ordinary Rust allocation | `stream.alloc_zeros::<T>(n)` |
| Freed with | Automatic, via `Drop` | Automatic, via `CudaSlice<T>`'s own `Drop` |
| Moving data between them | `stream.clone_htod(&host)` / `stream.memcpy_htod(&host, &mut dev)` | `stream.clone_dtoh(&dev)` / `stream.memcpy_dtoh(&dev, &mut host)` |

Both `CudaContext` and `CudaStream` also implement `Drop`, releasing the underlying CUDA context and stream automatically — the same RAII discipline Chapter 2.4 built by hand for a `Buffer` wrapping a raw pointer, now applied to a real external resource this book didn't write the allocator for.

### Worked Example 4.2.1 — a complete host/device pipeline, genuinely attempted, honestly reported

```rust
let ctx = CudaContext::new(0)?;
let stream = ctx.default_stream();
let mut d_c = stream.alloc_zeros::<f32>(n)?;
let d_a = stream.clone_htod(&h_a)?;
let d_b = stream.clone_htod(&h_b)?;
// ... kernel launch would go here — see Section 4.5 ...
```

Compiled and run as the complete `02_context_and_memory_pipeline.rs` further below:

```bash
cargo run --release --bin 02_context_and_memory_pipeline
```

Genuinely compiled and run, in this book's own sandbox, which has no NVIDIA driver installed:

```
Step 1: CudaContext::new(0)
  panicked while loading the CUDA driver library:
  Unable to dynamically load the "cuda" shared library - searched for library names: ["libcuda.so", "libcuda64.so", "libcuda64_12.so", "libcuda64_126.so", "libcuda64_126_0.so", "libcuda64_120_6.so", "libcuda64_10.so", "libcuda64_11.so", "libcuda64_12.so", "libcuda64_120_0.so", "libcuda64_9.so", "libcuda.so.12", "libcuda.so.12", "libcuda.so.11", "libcuda.so.10", "libcuda.so.9", "libcuda.so.1", "libnvcuda.so", "libnvcuda64.so", "libnvcuda64_12.so", "libnvcuda64_126.so", "libnvcuda64_126_0.so", "libnvcuda64_120_6.so", "libnvcuda64_10.so", "libnvcuda64_11.so", "libnvcuda64_12.so", "libnvcuda64_120_0.so", "libnvcuda64_9.so", "libnvcuda.so.12", "libnvcuda.so.12", "libnvcuda.so.11", "libnvcuda.so.10", "libnvcuda.so.9", "libnvcuda.so.1"]. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.
Steps 2-4 (stream, device allocation, host->device copy, kernel launch) all
require a live CudaContext, so none of them can run without one -- exactly the
same honest early-exit the CUDA edition's cudaMalloc/cudaMemcpy calls hit.

Host-computed reference (what a real device would produce, computed independently):
  a + b = [11, 22, 33, 44, 55, 66, 77, 88]
```

Two genuinely distinct things are on display, mirroring the CUDA edition's own equivalent worked example exactly. First, the *shape* of every GPU program this book writes from here forward — build a context, allocate on device, copy input in, launch, copy output out — is fully exercised as real Rust control flow, type-checked and compiled by `rustc` against the actual `cudarc` API, not paraphrased. Second, the *runtime* failure is real, not asserted: `CudaContext::new(0)` calls into `cudarc`'s dynamic-loading layer, which searches for `libcuda.so` (and its version-suffixed and NVIDIA-branded alternate names) using `dlopen`, doesn't find it on a machine with no NVIDIA driver installed, and panics with the exact message captured above — reproducible on every run, in this exact environment, for this exact reason. `h_a + h_b` is computed independently in plain Rust as a host-side reference, and is exactly the array a real device would compute for `vector_add_kernel` were one actually reachable here.

> `[COMMON TRAP]` A panic is not the same failure mode as CUDA C++'s `cudaErrorNoDevice`. Raw CUDA's runtime and driver APIs return an error code from every call, letting a program check `err == cudaSuccess` and continue in a controlled way; `cudarc`'s *dynamic-loading* backend panics instead, specifically because the failure it's reporting (the shared library itself couldn't be found) happens one layer below anything CUDA's own error-code system covers — there's no `CUresult` to return when there's no CUDA driver loaded yet to define one. A real production `cudarc` program that needs to handle "no GPU present" gracefully (rather than this chapter's `catch_unwind`, used here purely to keep the worked example's output readable) would typically check for the driver's presence before ever calling into `cudarc` at all.

## 4.3 The GPU Memory Hierarchy: A First Look `[FOUNDATIONAL]`

### Intuition

Extend Section 4.2's two-machines picture down one more level, exactly as the CUDA edition does. **Global memory** is a public warehouse across town: any thread from any block can drive there, but the drive is comparatively slow, and everyone shares the same access road. **Shared memory** is a supply closet built into one specific block's own barracks: every thread in that one block reaches it almost instantly, but threads in other blocks can't use it at all, and its contents vanish the moment that block finishes. **Local memory** is the single backpack strapped to one thread alone.

### Background

| | Global memory | Shared memory | Local (per-thread) memory |
|---|---|---|---|
| Visible to | Every thread, every block | Every thread *in one block* | One single thread |
| Speed | Slowest | Fast | Fastest |
| `cudarc` Rust type | `CudaSlice<T>` (Section 4.2) | None — declared inside the kernel's own CUDA C source | None — declared inside the kernel's own CUDA C source |
| Allocated with | `stream.alloc_zeros::<T>(n)` | `__shared__` inside a kernel string | Ordinary local variables inside a kernel string |

### Worked Example 4.3.1 — where Worked Example 4.2.1's data actually lives, and where `cudarc`'s types stop describing it

`d_a`, `d_b`, and `d_c` in Worked Example 4.2.1 are all `CudaSlice<f32>` values — Rust's own type for **global memory**, the only one of the three memory kinds `cudarc` exposes as a distinct Rust type at all. That's not an oversight: shared and local memory are declared *inside* a kernel's own CUDA C source (the string literal `compile_ptx` compiles), not as Rust values `cudarc` allocates on your behalf — from Rust's point of view, a kernel is an opaque function that happens to receive `CudaSlice<T>` arguments, and whatever it does with `__shared__` memory internally is invisible to the Rust type system entirely. This book holds the mechanics of actually writing `__shared__`-using kernel source for Part 1's memory-management chapters and Part 2's tiled kernel, exactly as the CUDA edition does — the `cudarc`-specific wrinkle is only that this book's kernel source lives in a Rust `&str`, compiled by NVRTC, rather than a `.cu` file compiled by `nvcc`.

> `[COMMON TRAP]` It's tempting to look for a `SharedSlice<T>` or similar type in `cudarc` and conclude it's missing a feature. It isn't missing anything — shared memory's whole reason to exist is that it's scoped to *one block's lifetime during one kernel launch*, a scope that has no Rust-side equivalent to attach a type to. It is, and can only be, a detail of the kernel's own source string.

## 4.4 Memory Coalescing, at Full Generality `[FOUNDATIONAL]`

### Intuition

Chapter 3 traced one specific case of this argument — `particles[i].vx` versus `vx[i]`, measured as a genuine CPU wall-clock benchmark rather than disassembled machine code, since this sandbox has no GPU to disassemble a kernel for. This section states the general principle that specific case was an instance of: it was never really about particles, and it applies to `cudarc` exactly as much as to raw CUDA, because it's a property of the hardware a compiled kernel runs on, not of which language or crate produced that kernel.

### Background

Real GPU hardware groups threads into **warps** of 32, and when every thread in a warp requests an address falling inside the same aligned memory region, the hardware services the entire warp with one **coalesced** transaction. When the 32 addresses are scattered, the hardware may need up to 32 separate transactions for the same warp — up to a 32× bandwidth penalty for delivering the identical amount of useful data. This is a fact about the compiled machine code a kernel becomes, not about `cudarc`, NVRTC, or `nvcc` specifically — whichever toolchain produces the PTX, the same warp-scheduling hardware executes it the same way.

Section 4.1's `i = blockIdx.x * blockDim.x + threadIdx.x` is what makes this principle actionable, in a `cudarc`-compiled kernel exactly as in a raw CUDA one: it guarantees that consecutive threads within a warp receive consecutive values of `i`. Whether that translates into coalesced access from there depends entirely on what the kernel body does with `i` — indexing a contiguous array with `array[i]` (as `vector_add_kernel` in Section 4.2 and Section 4.5 does for `a`, `b`, and `c`) keeps consecutive threads exactly `sizeof(element)` bytes apart, the coalesced case; indexing through a strided pattern spreads them out by that much more, exactly as Chapter 3's `particles[i].vx` did on the CPU side.

> `[COMMON TRAP]` Chapter 3's genuine CPU measurement found the AoS/SoA effect modest — a `1.11×` ratio, not the `1.75×` the naive byte-counting model predicted — because a single core's sequential prefetcher partly hides the difference. That finding does not carry over to warp-level coalescing on a GPU, and this section restates why: a warp's 32 threads issue their memory instruction *simultaneously*, with no single instruction stream for a prefetcher to get ahead of. This is exactly the distinction this book's real-hardware verification pass exists to measure directly, rather than assume from Chapter 3's CPU result.

## 4.5 Broadcasting: One Thread, One Output Element `[FOUNDATIONAL]`

### Intuition

`vector_add_kernel`, introduced in Worked Example 4.2.1, already demonstrates the pattern this section names: every thread is handed exactly one output element to produce, computes it independently of every other thread, and writes it exactly once. This is the **broadcast** pattern — the literal broadcasting of one small, identical kernel body out to potentially millions of independent threads, each covering one coordinate of the output.

### Background

| Output shape | `LaunchConfig` | Per-thread index (inside the kernel source) | Boundary check |
|---|---|---|---|
| 1D, length `n` | `LaunchConfig::for_num_elems(n)` | `i = blockIdx.x*blockDim.x+threadIdx.x` | `if (i < n)` |
| 2D, `M×N` | A hand-built `LaunchConfig` with matching `grid_dim`/`block_dim` | `row = blockIdx.y*blockDim.y+threadIdx.y`, `col = blockIdx.x*blockDim.x+threadIdx.x` | `if (row < M && col < N)` |

Worked Example 4.1.1 already showed why the boundary check is load-bearing: `LaunchConfig::for_num_elems(7)` launches 1024 threads for only 7 real outputs. Every kernel this book writes from here forward guards its actual work with exactly this kind of check, following Chapter 1's masking discipline forward into a GPU context.

### Worked Example 4.5.1 — the kernel source `cudarc` compiles, and the boundary check it depends on

```rust
const VECTOR_ADD_CU: &str = r#"
extern "C" __global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"#;
```

This is a Rust `&str` — `rustc` only checks that it's valid UTF-8 text, not that it's valid CUDA C. The actual CUDA C inside it is compiled by NVRTC, at runtime, when `cudarc::nvrtc::compile_ptx` is called on it. Compiled and run as part of the complete `03_broadcast_and_matmul.rs` further below:

```bash
cargo run --release --bin 03_broadcast_and_matmul
```

Genuinely compiled and run:

```
Attempting compile_ptx("vector_add_kernel")...
  panicked while loading the NVRTC library:
  Unable to dynamically load the "nvrtc" shared library - searched for library names: ["libnvrtc.so", "libnvrtc64.so", "libnvrtc64_12.so", "libnvrtc64_126.so", "libnvrtc64_126_0.so", "libnvrtc64_120_6.so", "libnvrtc64_10.so", "libnvrtc64_11.so", "libnvrtc64_12.so", "libnvrtc64_120_0.so", "libnvrtc64_9.so", "libnvrtc.so.12", "libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so.10", "libnvrtc.so.9", "libnvrtc.so.1"]. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.
```

The same failure shape as Worked Example 4.2.1, one layer earlier in the pipeline: NVRTC (`libnvrtc.so`) is part of the CUDA toolkit, not the driver, and this sandbox has neither installed. The CUDA C source above is never actually turned into PTX here — it's genuinely well-formed CUDA C (matching the CUDA edition's own `vector_add_kernel` from its Chapter 4), but that claim rests on the CUDA edition's real `nvcc` compile of the equivalent source, not on anything this sandbox can independently confirm for this exact string. Whether `compile_ptx` on this exact source succeeds and whether the resulting kernel produces the correct output on a real device are both marked `UNVERIFIED — pending real-GPU test`, following this book's standing convention.

The same program then host-traces Section 4.1's boundary check concretely, for `n=7` at `threads_per_block=4`:

```
Section 4.5 boundary check, host-traced for n=7, threads_per_block=4:
  block 0 thread 0 -> i=0 < 7: does real work
  block 0 thread 1 -> i=1 < 7: does real work
  block 0 thread 2 -> i=2 < 7: does real work
  block 0 thread 3 -> i=3 < 7: does real work
  block 1 thread 0 -> i=4 < 7: does real work
  block 1 thread 1 -> i=5 < 7: does real work
  block 1 thread 2 -> i=6 < 7: does real work
  block 1 thread 3 -> i=7 >= 7: boundary check fails, returns immediately
```

Without the check, thread 7 (the last thread of block 1) would compute `c[7] = a[7] + b[7]` against memory one element past the end of 7-element arrays — undefined behavior on real hardware, exactly as in the CUDA edition's equivalent example.

## 4.6 Matrix Multiplication: One Thread, One Dot Product `[FOUNDATIONAL]`

### Intuition

Extend Section 4.5's broadcast pattern from "one thread, one output number" to "one thread, one output number computed by summing over a whole dimension." Matrix multiplication's naive GPU form is exactly the 2D broadcast pattern from Section 4.5, with the per-thread body replaced by a small loop: thread `(row, col)` owns output element `C[row][col]`, computed as the full dot product of `A`'s row `row` against `B`'s column `col`.

### Worked Example 4.6.1 — a complete `2×2 @ 2×2`, traced two independent ways

```rust
const NAIVE_MATMUL_CU: &str = r#"
extern "C" __global__ void naive_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
"#;
```

For `A = [[1,2],[3,4]]`, `B = [[5,6],[7,8]]`: thread `(0,0)` would compute `C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] = 1*5 + 2*7 = 19`; thread `(0,1)`: `C[0][1] = 1*6+2*8 = 22`; thread `(1,0)`: `C[1][0] = 3*5+4*7 = 43`; thread `(1,1)`: `C[1][1] = 3*6+4*8 = 50`. `compile_ptx` on this source hits the identical NVRTC library failure as Worked Example 4.5.1's, so this exact kernel's own execution is `UNVERIFIED — pending real-GPU test`. What this chapter *can* verify, genuinely, is the arithmetic every thread's formula implies — computed two independent ways, as the complete `03_broadcast_and_matmul.rs` further below does after its two `compile_ptx` attempts:

```bash
cargo run --release --bin 03_broadcast_and_matmul
```

```
Section 4.6 naive matmul, host reference for a genuine 2x2 @ 2x2:
  host reference C = [[19.0, 22.0], [43.0, 50.0]]
```

Independently cross-checked with NumPy, outside this book's Rust toolchain entirely:

```python
>>> import numpy as np
>>> a = np.array([[1,2],[3,4]], dtype=np.float32)
>>> b = np.array([[5,6],[7,8]], dtype=np.float32)
>>> a @ b
array([[19., 22.],
       [43., 50.]], dtype=float32)
```

Both the plain-Rust host loop and NumPy's own matrix multiplication agree exactly with the hand trace above — `[[19, 22], [43, 50]]` — which is precisely the value this book's real-hardware pass will confirm `naive_matmul_kernel` itself produces, once a real device can run it. This naive kernel reads `A[row*K+k]` and `B[k*N+col]` — for a fixed `row`, as `k` advances, `A`'s accesses are perfectly contiguous, but `B`'s accesses jump by a full row (`N` elements) each step — a real bandwidth cost this naive form pays that Part 2's tiled kernel is built specifically to eliminate, exactly as in the CUDA edition.

## Complete Runnable Code

### File: `01_launch_config.rs`

```rust
use cudarc::driver::LaunchConfig;

fn main() {
    // A hand-sized LaunchConfig, mirroring CUDA's <<<3, 4>>> -- 3 blocks of 4 threads each.
    let cfg = LaunchConfig {
        grid_dim: (3, 1, 1),
        block_dim: (4, 1, 1),
        shared_mem_bytes: 0,
    };
    println!(
        "1D launch: grid_dim={:?} block_dim={:?} -- {} total threads",
        cfg.grid_dim,
        cfg.block_dim,
        cfg.grid_dim.0 * cfg.block_dim.0
    );
    for block_idx in 0..cfg.grid_dim.0 {
        for thread_idx in 0..cfg.block_dim.0 {
            let i = block_idx * cfg.block_dim.0 + thread_idx;
            println!("  block {}, thread {} -> global i = {}", block_idx, thread_idx, i);
        }
    }

    println!();
    println!("LaunchConfig::for_num_elems is what this book actually launches kernels with:");
    for n in [7u32, 8, 1025, 20_000_000] {
        let cfg = LaunchConfig::for_num_elems(n);
        let total = cfg.grid_dim.0 as u64 * cfg.block_dim.0 as u64;
        println!(
            "  n={:>10} -> grid_dim={:?} block_dim={:?} ({} threads launched, {} extra beyond n)",
            n, cfg.grid_dim, cfg.block_dim, total, total - n as u64
        );
    }
}
```

```bash
cargo run --release --bin 01_launch_config
```

### File: `02_context_and_memory_pipeline.rs`

```rust
use cudarc::driver::CudaContext;
use std::panic;

/// Runs `f`, catching a cudarc dynamic-loading panic instead of letting it abort the process,
/// and returns the panic message when one occurs.
fn catch_cuda<T>(f: impl FnOnce() -> T + panic::UnwindSafe) -> Result<T, String> {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {})); // suppress the default panic printer; we report it ourselves below
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

fn main() {
    let n: usize = 8;
    let h_a: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let h_b: [f32; 8] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    println!("Step 1: CudaContext::new(0)");
    let ctx_result = catch_cuda(|| CudaContext::new(0));
    let ctx = match ctx_result {
        Ok(Ok(ctx)) => {
            println!("  succeeded: a real GPU and driver are present.");
            Some(ctx)
        }
        Ok(Err(e)) => {
            println!("  returned a driver error (a GPU exists but the call failed): {:?}", e);
            None
        }
        Err(msg) => {
            println!("  panicked while loading the CUDA driver library:");
            println!("  {}", msg);
            None
        }
    };

    match ctx {
        Some(ctx) => {
            println!("Step 2: ctx.default_stream()");
            let stream = ctx.default_stream();
            println!("Step 3: stream.alloc_zeros::<f32>({n}) x3, stream.clone_htod(&h_a), stream.clone_htod(&h_b)");
            let mut d_c = stream.alloc_zeros::<f32>(n).unwrap();
            let d_a = stream.clone_htod(&h_a).unwrap();
            let d_b = stream.clone_htod(&h_b).unwrap();
            println!("  all three device allocations and both host-to-device copies succeeded.");
            println!("Step 4: launching would require a compiled kernel -- see 03_broadcast_and_matmul.rs");
            let _ = (&mut d_c, &d_a, &d_b);
        }
        None => {
            println!(
                "Steps 2-4 (stream, device allocation, host->device copy, kernel launch) all\n\
                 require a live CudaContext, so none of them can run without one -- exactly the\n\
                 same honest early-exit the CUDA edition's cudaMalloc/cudaMemcpy calls hit."
            );
        }
    }

    println!();
    println!("Host-computed reference (what a real device would produce, computed independently):");
    let h_c_ref: Vec<f32> = h_a.iter().zip(h_b.iter()).map(|(a, b)| a + b).collect();
    print!("  a + b = [");
    for (i, v) in h_c_ref.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", v);
    }
    println!("]");
}
```

```bash
cargo run --release --bin 02_context_and_memory_pipeline
```

### File: `03_broadcast_and_matmul.rs`

```rust
use cudarc::nvrtc::compile_ptx;
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

const VECTOR_ADD_CU: &str = r#"
extern "C" __global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"#;

const NAIVE_MATMUL_CU: &str = r#"
extern "C" __global__ void naive_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
"#;

fn try_compile(name: &str, src: &str) {
    println!("Attempting compile_ptx(\"{name}\")...");
    match catch_cuda(|| compile_ptx(src)) {
        Ok(Ok(_ptx)) => println!("  compiled successfully (a real NVRTC library is present)."),
        Ok(Err(e)) => println!("  NVRTC returned an error: {:?}", e),
        Err(msg) => {
            println!("  panicked while loading the NVRTC library:");
            println!("  {}", msg);
        }
    }
}

fn main() {
    try_compile("vector_add_kernel", VECTOR_ADD_CU);
    println!();
    try_compile("naive_matmul_kernel", NAIVE_MATMUL_CU);

    println!();
    println!("Section 4.5 boundary check, host-traced for n=7, threads_per_block=4:");
    let n = 7;
    let threads_per_block = 4;
    let blocks = (n + threads_per_block - 1) / threads_per_block;
    for block_idx in 0..blocks {
        for thread_idx in 0..threads_per_block {
            let i = block_idx * threads_per_block + thread_idx;
            if i < n {
                println!("  block {block_idx} thread {thread_idx} -> i={i} < {n}: does real work");
            } else {
                println!("  block {block_idx} thread {thread_idx} -> i={i} >= {n}: boundary check fails, returns immediately");
            }
        }
    }

    println!();
    println!("Section 4.6 naive matmul, host reference for a genuine 2x2 @ 2x2:");
    let a = [[1.0f32, 2.0], [3.0, 4.0]];
    let b = [[5.0f32, 6.0], [7.0, 8.0]];
    let mut c = [[0.0f32; 2]; 2];
    for row in 0..2 {
        for col in 0..2 {
            let mut acc = 0.0f32;
            for k in 0..2 {
                acc += a[row][k] * b[k][col];
            }
            c[row][col] = acc;
        }
    }
    println!("  host reference C = {:?}", c);
}
```

```bash
cargo run --release --bin 03_broadcast_and_matmul
```

`Cargo.toml` for all three binaries:

```toml
[package]
name = "rust_ch4"
version = "0.1.0"
edition = "2024"

[dependencies]
cudarc = { version = "0.19", default-features = false, features = ["driver", "nvrtc", "std", "dynamic-loading", "cuda-12060"] }
```

## Chapter Summary

`cudarc` represents CUDA's grid/block/thread launch as an ordinary Rust struct, `LaunchConfig`, and its convenience constructor, `LaunchConfig::for_num_elems`, is real code this chapter ran directly — `n=20000000` genuinely produces `grid_dim=(19532, 1, 1)`, `block_dim=(1024, 1, 1)`, launching 768 more threads than there are elements, which is exactly why every kernel this book writes guards its work with a boundary check. Host memory and device memory remain two physically separate spaces, now enforced by Rust's type system as `Vec<T>` versus `CudaSlice<T>` rather than by convention alone — `CudaContext::new`, `stream.alloc_zeros`, and `stream.clone_htod` are all real, compiling `cudarc` calls, and this chapter genuinely ran them: in this sandbox, with no NVIDIA driver installed, `CudaContext::new(0)` panics with a real, reproducible `dlopen` failure the moment it's called, and every step downstream of it is left honestly unexecuted rather than faked. The same pattern repeats one layer up the toolchain for `compile_ptx`, which needs NVRTC (`libnvrtc.so`) and fails the identical way. What this chapter could verify directly, it did: `LaunchConfig`'s arithmetic, the boundary-check logic for `n=7`, and the naive-matmul formula's `2×2` result — `[[19, 22], [43, 50]]`, confirmed independently against NumPy. What it couldn't — the two kernel source strings' actual behavior once compiled and launched on real hardware — is marked `UNVERIFIED — pending real-GPU test`, following this book's standing convention, rather than assumed correct because the CUDA C looks right.

## Self-Check Questions

1. `LaunchConfig::for_num_elems(7)` launches 1024 threads. Section 4.1's own genuine output shows 1017 of them have no valid `i < 7`. What mechanism, introduced in this chapter, keeps those 1017 threads from writing to memory they were never given?
2. `CudaContext::new(0)` in this sandbox fails with a Rust panic, not a `Result::Err`. Name the specific reason `cudarc`'s dynamic-loading backend can't represent this particular failure as an ordinary `DriverError` the way it represents, say, "no CUDA-capable device" once a driver library is actually loaded.
3. `d_a`, `d_b`, and `d_c` in Worked Example 4.2.1 are all `CudaSlice<f32>` — global memory. Why does `cudarc` not expose a Rust type for shared memory the same way?
4. `naive_matmul_kernel`'s inner loop reads `A[row * K + k]` and `B[k * N + col]` as `k` advances. Which of the two is coalesced across a warp sharing the same `row`, and which pays a stride penalty — and why does that follow from the same reasoning Chapter 3 applied to `vx[i]` versus `particles[i].vx`, even though neither array here is a struct?
5. This chapter never actually launches `vector_add_kernel` or `naive_matmul_kernel` on real hardware. Explain specifically what *was* genuinely verified about each kernel in this chapter, and what remains tagged `UNVERIFIED — pending real-GPU test`.

## Where We Go Next

Every kernel source string this chapter wrote assumed the values flowing through it were already scalars — one `float` in, one `float` out, per thread. Chapter 5 turns to a different kind of parallelism this book can verify completely on the CPU it already has: SIMD, where a single CPU instruction processes several `f32` values side by side, using `std::arch` intrinsics and the `wide` crate on stable Rust, cross-checked against scalar code the same way Chapter 3 cross-checked AoS against SoA.

## Worked Solutions

**1.** The boundary check, `if (i < n)` inside the kernel's own CUDA C source, guards every write. Section 4.5's host trace shows exactly this: for `n=7`, `threads_per_block=4`, thread 7 (`i=7`) fails `i < 7` and "returns immediately," touching no memory — the same discipline `LaunchConfig::for_num_elems`'s own over-launch (1024 threads for 7 elements) makes structurally necessary for every kernel in this book.

**2.** `DriverError` wraps a `CUresult` — a genuine error code the CUDA driver itself returns once it's loaded and running (`cudaErrorNoDevice` is exactly this kind of value in the CUDA edition). But `cudarc`'s dynamic-loading backend hasn't loaded a CUDA driver at all yet at the point this failure occurs — it's still trying to find and `dlopen` the shared library (`libcuda.so`) that *would* eventually hand back `CUresult` values. There is no driver in scope yet to define or return a `CUresult`, so the only failure `cudarc` can report at that stage is a Rust-level panic describing the library search that failed.

**3.** Shared memory's entire reason to exist is that its lifetime is scoped to one block, for the duration of one kernel launch — a scope with no Rust-side equivalent, since `cudarc`'s Rust types (`CudaContext`, `CudaStream`, `CudaSlice<T>`) all live at the process/host level, entirely outside any individual kernel launch. A `SharedSlice<T>` Rust type would have nothing meaningful to represent from the host's perspective; shared memory only makes sense as something declared and scoped *inside* the kernel's own CUDA C source, which is exactly where `__shared__` lives.

**4.** For threads sharing the same `row` (varying only in `col`, and therefore in which warp-lane they occupy for a fixed `k`), `B[k * N + col]` is the array where consecutive threads read consecutive `col` values apart by exactly `sizeof(float)` — the coalesced case. `A[row * K + k]` is read identically by *every* thread in that same warp (since `row` and `k` are shared, only `col` varies across the warp, and `A`'s index doesn't depend on `col` at all) — not a stride penalty in the classic sense, but a broadcast read of the same address by the whole warp. The actual stride penalty in this kernel is paid by `B` across *iterations* of `k` for one fixed thread: as `k` advances by 1, `B`'s index advances by `N` elements, exactly Chapter 3's `particles[i].vx`-style stride, restated inside a loop instead of across a struct's fields — the underlying mechanism (constant-stride access forcing the memory system to move more bytes than are used) is identical, the "struct" is simply `B`'s own row-major layout playing the interleaving role Chapter 3's `Particle` struct played.

**5.** Genuinely verified: both kernel source strings are well-formed Rust string literals compiled cleanly by `rustc` as part of a real `cargo build`; `LaunchConfig`'s launch-dimension arithmetic and the `i < n` boundary check's logic were both run directly on the host and matched their hand traces exactly; the matmul kernel's implied arithmetic was independently computed twice (a plain Rust host loop and NumPy) and matched the hand trace exactly, `[[19, 22], [43, 50]]`. Left `UNVERIFIED — pending real-GPU test`: whether `compile_ptx` actually accepts this exact CUDA C source without error once NVRTC is genuinely reachable, and whether `vector_add_kernel` and `naive_matmul_kernel`, once compiled and launched on a real device, produce output that matches these same host-computed reference values — the CUDA C syntax looking correct is not the same claim as a real device having executed it correctly.
