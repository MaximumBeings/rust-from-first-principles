# Chapter 19: Performance Optimization Techniques — Telling a Real Speedup From a Rounding Error

> "Chapter 18 built kernels that don't waste threads, don't waste memory transactions, and don't waste global-memory reads a shared tile could have avoided. This chapter is about the layer above any single kernel: packing more bytes into fewer load instructions, collapsing several kernels into one, letting the compiler bake a parameter in so thoroughly that checking it costs nothing at runtime, and — the part every other technique in this chapter is worthless without — a way to measure whether any of it actually helped."

**What you will understand by the end of this chapter:**

- Vectorized memory access as a compile-time-portable main-loop-plus-remainder pattern — the same `(size / width) * width` shape Chapter 5's Mojo edition and this book's own earlier chapters use — built on CUDA's real `float4` load/store instead of four separate 32-bit ones, traced on two different sizes to see where the remainder loop is and isn't a large fraction of the total
- Loop unrolling and kernel fusion as two genuinely different optimizations that are easy to blur together: unrolling cuts loop-control overhead without changing the arithmetic instruction count, while fusion cuts *memory traffic* by collapsing several kernels' reads and writes into one — counted exactly, not just asserted, for both
- Why fusing operations trades a *faster forward pass* for a backward rule that has to be hand-derived once, rather than composed for free from the individually-registered `AddOp`/`MulOp`/`ReluOp` backward rules Chapter 16 already built
- Compile-time specialization as the actual mechanism behind "zero-cost abstractions": Rust's `const` generics, `compile_time_specialized_dot::<N>`, compile to a genuinely separate, independently-optimized function body per value of `N` — confirmed here by reading the actual linked symbol table, not just trusting the claim — and the real cost that mechanism doesn't eliminate: one compiled body per distinct instantiation
- A benchmarking harness built around the one measurement mistake that invalidates everything else: timing a function's first call, before caches are warm and (for GPU work) before the device has even been given time to catch up to a host that queued work and kept running — applied here to a genuinely measured convolution GFLOPS figure, not a hypothetical one

**What you need to know first:**

- Chapter 5 in full — the `(size / width) * width` main-loop-plus-remainder shape — this chapter's vectorization section is that same shape applied on a real 128-bit CUDA vector type, and its loop-fusion section is one way to fix exactly the kind of redundant-read inefficiency Chapter 18's naive convolution already demonstrated.
- Chapter 16 (the `Differentiable` trait and the registered `AddOp`/`MulOp`/`ReluOp` backward rules) — Section 19.2's fusion trade-off is stated directly in terms of what those individually-registered ops give you for free.
- Chapter 18 in full — Section 19.4's benchmarking example measures exactly Chapter 18's convolution kernel host mirrors, generalized past the 4×4 worked example to real, timeable sizes. Chapter 18's own established fact also holds here: this sandbox has neither an NVIDIA driver nor NVRTC (Chapter 4), and no CUDA toolkit at all — no `nvcc`, no `cuobjdump` (Chapter 18) — so every genuinely-attempted `cudarc` call in this chapter hits the identical, honestly-reported dynamic-loading failure, and every piece of evidence this edition substitutes for `cuobjdump`-decoded SASS comes from `rustc` itself.

## 19.1 SIMD Vectorization `[FOUNDATIONAL]`

### Intuition

A print shop's press can stamp one letter per strike, or it can load a plate with four letters cast side by side and stamp all four in one strike — same ink, same press, one motion instead of four. A run of exactly `40` letters takes `10` four-letter strikes instead of `40` single strikes. A run of `42` letters takes `10` four-letter strikes for the first `40`, and then the operator has no choice but to fall back to two individual single-letter strikes for the two that don't fill a fourth plate. The press doesn't get slower or more complicated for having to do this — it just does almost all of the job at four-times throughput, and the awkward leftover at the old, one-at-a-time rate.

### Background

Chapter 5 already established the shape: split a loop into a vectorized main body plus a scalar remainder, where `simd_count = (size / width) * width` marks the boundary between them. CUDA's own version of "one instruction, several elements" is a vector type like `float4`: a 128-bit load that moves four consecutive `float`s in one instruction instead of four separate 32-bit loads. This is the real kernel every file in this section corresponds to — genuinely compiled by neither edition's own sandbox this book has run so far, since this one has no CUDA toolkit at all (Chapter 18) and the CUDA edition's own sandbox has the toolkit but no physical GPU to launch anything on:

```cpp
__global__ void vectorized_add_float4_kernel(float* output, const float* a, const float* b,
                                              int size, int vec_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_count) {
        float4 va = reinterpret_cast<const float4*>(a)[idx];
        float4 vb = reinterpret_cast<const float4*>(b)[idx];
        float4 vout;
        vout.x = va.x + vb.x; vout.y = va.y + vb.y;
        vout.z = va.z + vb.z; vout.w = va.w + vb.w;
        reinterpret_cast<float4*>(output)[idx] = vout;
    }
    // scalar remainder handled by a separate pass over the tail --
    // Section 18.1's "two-part launch" discipline, applied here too.
}
```

`float4`'s width — `4` elements per instruction — reports the natural vector width for this particular type, but the width is fixed by the *type*, not queried from the compiler: `sizeof(float4) = 16` bytes always, and `16 / sizeof(float) = 4` elements always. The trade-off this makes explicit: a wider vector load processes more elements per instruction, but the remainder loop's *maximum possible size* also grows with the width — a hypothetical 16-wide load could leave up to `15` elements to the scalar loop, versus `float4`'s worst case of `3`. This chapter's Rust host mirror carries the identical `(size / width) * width` arithmetic, with a `width` parameter standing in for `float4`'s fixed `4`, since Rust has no built-in 128-bit float vector type of its own to reinterpret a slice into:

```rust
struct VecStats {
    vector_instructions: u32,
    scalar_instructions: u32,
}

fn vectorized_add_host(output: &mut [f32], a: &[f32], b: &[f32], size: usize, width: usize) -> VecStats {
    let mut stats = VecStats { vector_instructions: 0, scalar_instructions: 0 };
    let simd_count = (size / width) * width;
    let mut i = 0;
    while i < simd_count {
        for lane in 0..width {
            output[i + lane] = a[i + lane] + b[i + lane];
        }
        stats.vector_instructions += 1; // one vector instruction covers `width` elements
        i += width;
    }
    for i in simd_count..size {
        output[i] = a[i] + b[i];
        stats.scalar_instructions += 1; // one scalar instruction covers 1 element
    }
    stats
}
```

| Ceiling division | Bounds check | Result |
|---|---|---|
| `simd_count = (size/4)*4` | main loop covers `simd_count` elements, `4` at a time | remainder loop covers the rest, one at a time |

### Worked Example 19.1.1 — `size=10`, `width=4`, traced instruction by instruction

Compiled and run:

```bash
cargo run --release --bin 01_vectorized_add
```

Genuinely compiled and run:

```
size=10, width=4: simd_count = (10/4)*4 = 8
  vector instructions   = 2 (covering elements 0-7)
  scalar instructions   = 2 (covering elements 8-9)
  total instructions    = 4  (scalar-only baseline = 10)

  output: 0 11 22 33 44 55 66 77 88 99  (host reference a[i]+b[i] matches by construction)
```

`simd_count = (10 / 4) * 4 = 8`. The main loop runs twice: one iteration adds elements `0`-`3` of `a` and `b` in a single accounted vector step, a second adds elements `4`-`7`. Elements `8` and `9` fall outside any full group of four and run through the scalar loop, one at a time. Total: `2` vector instructions plus `2` scalar instructions — `4` instructions replacing what a pure scalar loop would have needed `10` separate additions to do — genuinely counted by a host mirror instrumented with real instruction counters, not merely computed from the formula.

### Worked Example 19.1.2 — `size=41`, where the remainder is a small fraction of the total

Same run continues:

```
size=41, width=4: simd_count = (41/4)*4 = 40
  vector instructions   = 10 (covering elements 0-39)
  scalar instructions   = 1 (covering elements 40-40)
  total instructions    = 11  (scalar-only baseline = 41)
  ratio vs Worked Example 19.1.1: 26.8% of baseline here vs 40.0% there
```

`simd_count = (41 / 4) * 4 = 40`. The main loop runs `10` times, covering elements `0`-`39`, and exactly *one* element, index `40`, falls to the scalar remainder. Total: `10` vector instructions plus `1` scalar instruction — `11` instructions total against a `41`-instruction scalar baseline (`26.8%` of baseline), a better ratio than Worked Example 19.1.1's `40.0%` despite the remainder being nonzero in both cases, because `41`'s one leftover element is a much smaller fraction of `41` than `10`'s two leftover elements are of `10`.

The same file also genuinely attempts the real pipeline over `8` elements (exactly `2` full groups, no remainder), following this book's established Chapter 4/18 pattern:

```
--- vectorized_add_float4_kernel: genuine kernel-launch attempt, no GPU in this environment ---
Step 1: CudaContext::new(0)
  panicked while loading the CUDA driver library:
  Unable to dynamically load the "cuda" shared library - searched for library names: [...]
Steps 2-4 (device allocation, host->device copy, kernel launch <<<1,2>>> -- each thread
handling one float4 = 4 elements) all require a live CudaContext, so none of them can run
without one -- exactly the same honest early-exit Chapter 4's and Chapter 18's own pipelines hit.

host-computed reference (a+b): 101 202 303 404 505 606 707 808
```

```
[COMMON TRAP]  One dtype's vector width does not transfer to another

A 128-bit vector load is fixed in BYTES, not in ELEMENT COUNT. float4
packs four 4-byte floats into 128 bits, but double2 packs only two
8-byte doubles into the SAME 128 bits. Code retuned from float4's
width-4 data to double2 data, while still assuming width=4, would read
16 bytes past the intended 2-element group on every vector op --
corruption, not merely a slowdown, if the width constant isn't
re-derived for the new dtype. This is exactly why a real vectorized
kernel computes its width from sizeof(VectorType) / sizeof(ElementType)
rather than hardcoding a width borrowed from testing a different dtype.
```

### Worked Example 19.1.3 — Confirming the trap with compiler-verified evidence

Compiled and run:

```bash
cargo run --release --bin 02_vector_width_dtype_trap
```

Genuinely compiled and run:

```
=== Section 19.1 COMMON TRAP: one dtype's vector width does not transfer to another ===

size_of::<[f32; 4]>()  = 16 bytes, elements packed = 4 (f32, 4 bytes each)
size_of::<[f64; 2]>()  = 16 bytes, elements packed = 2 (f64, 8 bytes each)

Same 128-bit vector payload, different element counts:
  float4-equivalent  width = 4 elements per 128-bit payload
  double2-equivalent width = 2 elements per 128-bit payload
  a kernel retuned from float4's width=4 to double2 data, still assuming width=4,
  would read 16 bytes past the intended 2-element group on every vector op -- corruption,
  not merely a slowdown, if the width constant isn't re-derived for the new dtype.

--- attempting a genuine kernel launch over the same 32-byte-per-array budget ---
  CudaContext::new(0) panicked while loading the CUDA driver library:
  Unable to dynamically load the "cuda" shared library - searched for library names: [...]
  (the same real, reproducible failure Section 19.1's launch attempt and Chapter 18 both hit --
   compiling and launching either vectorized_add_float4_kernel<<<1,2>>> or
   vectorized_add_double2_kernel<<<1,2>>> requires a live context neither can obtain here)

what size_of DOESN'T confirm, honestly: whether a compiled kernel actually loading a
float4 or double2 emits the identical LDG.E.128/STG.E.128 instruction pair the CUDA
edition's own cuobjdump output shows -- that claim rests on nvcc's code generation,
which this sandbox has no toolkit to inspect at all. What size_of DOES confirm,
genuinely: the two payloads are the same fixed number of BYTES (16), while the number
of logical elements packed into that fixed byte count depends entirely on the element
type's own size -- the same distinction the CUDA edition's SASS draws between a fixed
128-bit instruction width and a dtype-dependent element count, from a different,
equally genuine evidence source.
```

The CUDA edition confirms this same claim two ways: `sizeof()` (a compile-time fact) and `cuobjdump --dump-sass` on a compiled cubin (a fact about the actual generated machine code), reading `LDG.E.128`/`STG.E.128` instructions directly out of the compiled binary. This sandbox has no CUDA toolkit at all — neither `nvcc` nor `cuobjdump` exist here, genuinely confirmed in Chapter 18 — so decoding real SASS the way that edition's own Worked Example 19.1.3 does is not something this sandbox can do honestly. What it *can* do honestly, genuinely, is ask `rustc`'s own compiler for the equivalent fact about Rust's fixed-size array types: `std::mem::size_of::<[f32; 4]>()` and `std::mem::size_of::<[f64; 2]>()` are both compiler-verified, not assumed — the same "ask the compiler, not the disassembler" evidence source Chapter 18's `offset_of!`-based Worked Example 18.2.2 already established as this edition's honest substitute for machine-code inspection. Both payloads are genuinely `16` bytes; `f32`'s `4`-byte size means `4` elements fit, while `f64`'s `8`-byte size means only `2` fit into the identical byte budget — the same distinction the CUDA edition's SASS draws between a fixed instruction bit-width and a dtype-dependent element count, confirmed here from a different, equally genuine evidence source.

## 19.2 Loop Unrolling and Fusion `[FOUNDATIONAL]`

### Intuition

Two related but different savings live under this heading, and it's worth telling them apart with two different pictures. **Unrolling**: an inspector checking a conveyor belt one item at a time re-aims, refocuses, and re-decides "is this the last item?" once per item — even though most of that per-item overhead is identical bookkeeping, not the inspection itself. Handing the inspector four items at once to check in a single glance removes three of every four "is this the last one?" decisions, without making the inspection of any individual item any different. **Fusion**: three separate quality-control stations, each requiring the part to be set down on a table, inspected, and picked back up before moving to the next station, versus one station where the inspector holds the part in their hands through all three checks and only sets it down once, at the very end.

### Background

**Unrolling** trades loop-control overhead (the increment, the bounds comparison, the branch) for a bigger loop body doing more work per iteration — it does not change how many arithmetic operations are performed, only how many times the loop's own bookkeeping runs:

```rust
struct UnrollStats {
    loop_iterations: u32, // how many times the loop's own bookkeeping runs
    arithmetic_ops: u32,  // how many additions are actually performed
}

fn unrolled_add(output: &mut [f32], a: &[f32], b: &[f32], size: usize) -> UnrollStats {
    let mut s = UnrollStats { loop_iterations: 0, arithmetic_ops: 0 };
    const UNROLL: usize = 4;
    let unrolled_count = (size / UNROLL) * UNROLL;
    let mut i = 0;
    while i < unrolled_count {
        output[i]     = a[i]     + b[i];
        output[i + 1] = a[i + 1] + b[i + 1];
        output[i + 2] = a[i + 2] + b[i + 2];
        output[i + 3] = a[i + 3] + b[i + 3];
        s.loop_iterations += 1; // ONE iteration's worth of bookkeeping...
        s.arithmetic_ops += 4;  // ...covers FOUR additions
        i += UNROLL;
    }
    for i in unrolled_count..size {   // remainder, one at a time
        output[i] = a[i] + b[i];
        s.loop_iterations += 1;
        s.arithmetic_ops += 1;
    }
    s
}
```

This is the identical `(size / N) * N` main-loop-plus-remainder shape Section 19.1's vectorized code uses — but where the `float4`-width main loop issues *one* accounted vector step covering `4` elements, unrolling's main loop still issues `4` separate scalar additions; it has simply moved `4` of them inside one loop iteration instead of spreading them across `4` iterations. The two techniques solve different problems and are frequently combined — an unrolled loop whose body is itself a vectorized load/add/store gets both the reduced loop overhead and the reduced arithmetic-instruction count at once. The real CUDA-kernel form of the same idea gives each thread `UNROLL` elements to process instead of one, so the same grid covers `UNROLL` times more data with `UNROLL` times fewer threads' worth of index computation and bounds-check overhead — never compiled or launched here, this sandbox has no CUDA toolkit at all:

```cpp
__global__ void unrolled_add_kernel(float* output, const float* a, const float* b, int size) {
    const int UNROLL = 4;
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * UNROLL;
    #pragma unroll
    for (int lane = 0; lane < UNROLL; lane++) {
        int i = base + lane;
        if (i < size) output[i] = a[i] + b[i];
    }
}
```

**Fusion** combines multiple kernels' worth of work into one kernel body, and its saving is countable directly in memory operations, not just instructions. `y = relu(a * b + c)` written as three separate kernels reads `a` and `b` (multiply), writes an intermediate; reads that intermediate and `c` (add), writes a second intermediate; reads that second intermediate (relu), writes `y` — five tensor-sized reads and three tensor-sized writes, eight memory operations total. Fused into one kernel, the same computation reads `a`, `b`, and `c` exactly once each and writes `y` exactly once — four memory operations, with the multiply, add, and relu happening in a register between the reads and the one write:

```rust
struct MemStats { reads: u64, writes: u64 }

fn fused_pipeline(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32], size: usize, stats: &mut MemStats) {
    for i in 0..size {
        stats.reads += 3;   // read a[i], b[i], c[i] -- once each, total
        let z = a[i] * b[i] + c[i];
        output[i] = if z > 0.0 { z } else { 0.0 };
        stats.writes += 1;  // write output[i] -- once
    }
}
```

### Worked Example 19.2.1 — Unrolling's overhead saving, with no change in arithmetic

Compiled and run:

```bash
cargo run --release --bin 03_unrolled_add
```

Genuinely compiled and run:

```
size=1000, UNROLL=4: unrolled_count = (1000/4)*4 = 1000
  plain loop:    1000 loop-control events, 1000 arithmetic ops
  unrolled loop: 250 loop-control events, 1000 arithmetic ops
  overhead reduction: 4x fewer loop-control events (1000 -> 250)
  arithmetic op count identical: true

  every output element identical between plain and unrolled: true

size=4002, UNROLL=4: unrolled_count = (4002/4)*4 = 4000, remainder = 2 element(s)
  plain loop:    4002 loop-control events, 4002 arithmetic ops
  unrolled loop: 1002 loop-control events (1000 full groups + 2 remainder), 4002 arithmetic ops
  arithmetic totals match: true (both 4002)

  every output element identical between plain and unrolled: true
```

For `size = 1000` and `UNROLL = 4`: `unrolled_count = (1000 / 4) * 4 = 1000` — this size divides evenly, so there is no remainder at all. The unrolled loop runs `250` iterations, each doing `4` additions in its body, for `1000` additions total — the *same* `1000` additions a plain scalar loop would perform, genuinely confirmed by comparing every output element between the two implementations. What changed is the loop-control overhead: a plain loop increments, compares, and branches `1000` times; this unrolled loop does so `250` times, a `4×` reduction in control overhead for zero change in arithmetic work. A second run at `size = 4002` confirms the remainder loop still handles a genuine non-multiple-of-`4` size correctly: `1000` full unrolled groups plus `2` remainder elements, `1002` total loop-control events against `4002` arithmetic operations either way.

The same file also genuinely attempts the real unrolled kernel:

```
--- unrolled_add_kernel: genuine kernel-launch attempt, no GPU in this environment ---
  CudaContext::new(0) panicked while loading the CUDA driver library: [...]
  kernel launch <<<1,4>>> (each thread handles 4 elements) cannot run without a live context
host reference (a+b): 0 3 6 9 12 15 18 21 24 27 30 33 36 39 42 45
```

### Worked Example 19.2.2 — Fusion's memory-traffic saving, counted exactly

Compiled and run:

```bash
cargo run --release --bin 04_fusion_memory_traffic
```

Genuinely compiled and run:

```
=== Section 19.2: fusion's memory-traffic saving, counted exactly ===

relu(a*b+c) on 6 elements:
  unfused (3 kernels): 5 tensor-sized reads, 3 tensor-sized writes, 8 total memory ops
  fused   (1 kernel):  3 tensor-sized reads, 1 tensor-sized writes, 4 total memory ops

outputs identical between unfused and fused: true
fused output: 2.0 0.0 0.0 0.0 0.0 0.0

memory traffic reduction: 2x (8 ops -> 4 ops), independent of size 6

--- fused_mul_add_relu_kernel: genuine kernel-launch attempt, no GPU in this environment ---
  CudaContext::new(0) panicked while loading the CUDA driver library: [...]
  kernel launch <<<1,6>>> cannot run without a live context
host reference (matches fused_pipeline above): 2.0 0.0 0.0 0.0 0.0 0.0
```

For a tensor of any size `N`, the unfused three-kernel version of `relu(a*b+c)` performs `5` tensor-sized reads and `3` tensor-sized writes (`8` total memory operations) — genuinely counted here with the same read/write counter technique Chapter 18 used for convolution's redundant reads, extended to also count writes; the fused single-kernel version performs `3` reads and `1` write (`4` total) — exactly half the memory traffic, and genuinely confirmed to produce identical output values to the unfused version on a real six-element example (`relu(1·2+0)=2`, and the remaining five elements all land at or below zero and clamp to `0`). The cost isn't free: Chapter 16 registered `AddOp`, `MulOp`, and `ReluOp` as three separate `Differentiable` implementations in the `OpRegistry`, specifically so their backward rules compose automatically through `chain_rule_step` — `fused_mul_add_relu_kernel` has no such registry entry, and differentiating through it requires hand-deriving one combined backward rule for the whole fused expression (computed once, then reused, rather than assembled for free from three already-registered pieces). This is exactly why this book keeps ops unfused by default and treats fusion as a targeted optimization for whichever operations Section 19.4's benchmarks actually show to be hot, not a blanket policy.

```
[COMMON TRAP]  Fusing an op without registering a matching backward rule

A kernel that fuses forward computation but is dropped into the graph
under one of the ALREADY-registered op names in Chapter 16's OpRegistry
(say, calling chain_rule_step("mul", ...) on a node that actually ran
the fused multiply-add-relu) would run MulOp's backward rule -- which
only knows how to differentiate a plain multiplication -- against a
node whose forward pass did three operations, not one. The gradient it
returns would be a plausible-looking ScalarTensor of the right shape,
silently wrong in value, with nothing about the shape mismatch to catch
it. A fused op needs its OWN registered Differentiable entry with a
backward rule derived for the whole fused expression -- reusing an
existing op's name for a kernel that no longer matches what that name's
backward rule assumes is exactly how a correct-looking forward pass
ends up paired with an incorrect gradient.
```

## 19.3 Compile-time Optimizations `[FOUNDATIONAL]`

### Intuition

A print shop could cast one adjustable metal plate that reads its own configuration before every single stamp — "am I set to 4-wide or 8-wide today?" — paying that small check on every strike, forever. Or it could cast two entirely separate, purpose-built plates ahead of time, one fixed at 4-wide and one fixed at 8-wide, and simply reach for whichever one a given job needs. The second shop pays a cost the first one doesn't — two plates to store instead of one — but neither plate ever asks itself a question at strike time, because the question was already answered back when it was cast.

### Background

Rust's `const` generics — `fn foo<const N: usize>(...)`, resolved at compile time, not runtime — are this chapter's Rust analog of C++'s `template<int N>` and Mojo's `fn foo[n: Int](...)`. `compile_time_specialized_dot::<4>` and `compile_time_specialized_dot::<2>` are two fully separate, independently-optimized machine-code bodies, with no runtime branch on `N` anywhere in either compiled output:

```rust
#[inline(never)]
fn compile_time_specialized_dot<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    // N is baked into the generated code at compile time -- no runtime
    // loop-bound check, no dynamic dispatch on vector width.
    let mut sum = 0.0f32;
    for i in 0..N {
        sum += a[i] * b[i];
    }
    sum
}
```

This is the concrete mechanism behind this book's "zero-cost abstractions" claim, and it isn't a new idea introduced here — Chapter 5's own SIMD width parameter and Chapter 9's dtype-parametric functions are both already-published examples of the exact same thing: a generic-looking function that the compiler turns into as many separate, fully specialized bodies as the program actually instantiates, none of which pay a runtime cost for the genericity the source code appears to have. Rust's own mangled symbol names carry a wrinkle the CUDA edition's C++ ones don't: each monomorphized instantiation gets a compiler-generated hash suffix baked into its *raw* mangled name (`compile_time_specialized_dot17h313d46f6483fcc52E` versus `...17ha0519438bafc51b0E`), but the commonly-used *demangled* form drops the const-generic argument entirely, so two genuinely distinct compiled bodies can demangle to the identical-looking friendly name `compile_time_specialized_dot` — it is the raw symbol, not the demangled one, that proves the compiler generated two separate functions.

### Worked Example 19.3.1 — Two instantiations, two independent answers, confirmed in the linked binary

Compiled and run:

```bash
cargo run --release --bin 05_compile_time_specialized_dot
```

Genuinely compiled and run:

```
=== Section 19.3: compile-time specialization, two genuinely separate bodies ===

compile_time_specialized_dot::<4>([1,2,3,4], [5,6,7,8]) = 70.0 (expected 1*5+2*6+3*7+4*8 = 70)
compile_time_specialized_dot::<2>([2,3], [4,5]) = 23.0 (expected 2*4+3*5 = 23)

these are not one function called twice with a runtime-varying length --
N=4 and N=2 select two DIFFERENT, separately-compiled functions. Confirmed via
call_dot_n4/call_dot_n2 (kept #[inline(never)] so both survive to the linked binary):
  call_dot_n4(&a4,&b4) = 70.0
  call_dot_n2(&a2,&b2) = 23.0

run `nm` (raw, mangled) on the compiled binary to see this genuinely: two DISTINCT
symbols exist, one per instantiation, distinguished by rustc's own per-instantiation
hash suffix -- not one generic symbol with an N parameter passed at runtime. Piping the
same symbols through `rustfilt` demangles both to the same friendly name
(compile_time_specialized_dot drops its const-generic argument when demangled, unlike
C++'s <N> which stays visible) -- it is the RAW, undemangled symbol below that carries
the proof of two distinct compiled bodies, not the demangled one.
```

`compile_time_specialized_dot::<4>` called with `a = [1, 2, 3, 4]` and `b = [5, 6, 7, 8]`: `1·5 + 2·6 + 3·7 + 4·8 = 5 + 12 + 21 + 32 = 70`. `compile_time_specialized_dot::<2>` called with `a = [2, 3]` and `b = [4, 5]`: `2·4 + 3·5 = 8 + 15 = 23`. These aren't two calls to one function with a runtime-varying length — `N=4` and `N=2` select two *different, separately-compiled* functions, each hardcoded to its own trip count, the way `compile_time_specialized_dot::<4>`'s generated code has no code path at all for handling a 2-wide input, and vice versa.

Rather than only asserting that claim, as the printed text above already does, the actual linked binary can be read directly:

```bash
nm target/release/05_compile_time_specialized_dot | grep -E "[0-9]+compile_time_specialized_dot17h"
```

Genuinely run:

```
00000000000155b0 t _ZN32_05_compile_time_specialized_dot28compile_time_specialized_dot17h313d46f6483fcc52E
00000000000155f0 t _ZN32_05_compile_time_specialized_dot28compile_time_specialized_dot17ha0519438bafc51b0E
```

Two distinct symbols, at two distinct addresses (`0x155b0` and `0x155f0`), both genuinely present in the compiled binary's own symbol table — not one shared symbol with a hidden runtime parameter. The pattern used to isolate these two lines matters honestly: this particular binary's name (`05_compile_time_specialized_dot`, taken directly from the source file's own name, per this book's file-naming convention) is not a legal Rust identifier on its own — it starts with a digit — so `rustc` mangles the *crate* path with a leading underscore, and that crate path happens to itself contain the substring `compile_time_specialized_dot`, matching every symbol in the binary, not just the two of interest; filtering on the function's own length-prefixed mangling (`28compile_time_specialized_dot17h`) is what isolates exactly the two instantiations. `nm`'s lowercase `t` marker means each is a local (non-exported) text symbol — Rust does not deduplicate generic instantiations across compilation units at the linker level the same way C++'s weak (`W`) template symbols do, a genuine, minor difference from the CUDA edition's own `nm` output that doesn't change the underlying claim: two separately-compiled bodies, confirmed by two distinct addresses.

```
[COMMON TRAP]  Zero runtime cost is not zero cost

Every distinct value of N a program actually instantiates
compile_time_specialized_dot with produces its OWN compiled function
body -- confirmed above, genuinely, as two separate linked symbols. A
program that instantiates this function at N=2, N=4, N=8, and N=16
somewhere in its source compiles four separate machine-code bodies,
not one generic body handling four cases -- the same "code bloat"
trade-off Rust's const generics (and C++ templates before them) have
always made. The cost compile-time specialization eliminates is a
RUNTIME one (a branch or a dictionary lookup on which width to use);
it does not eliminate cost altogether, it moves the cost to compile
time (longer builds) and to binary size (more distinct function bodies
shipped) instead.
```

## 19.4 Benchmark Framework `[FOUNDATIONAL]`

### Intuition

Timing a runner's very first sprint of the day, straight off the bench with cold muscles, measures how slow a cold start is — not how fast that runner actually runs. A fair measurement lets them run a few warm-up sprints first, uncounted, and only then starts the stopwatch on several representative sprints, averaged together, so one unusually good or bad rep doesn't dominate the result. GPU work has an extra wrinkle a runner doesn't: the host can *launch* a kernel and immediately move on to launching the next line of code, well before the device has actually finished — timing the host's launch loop alone measures how fast the host can hand off work, not how fast the device does it, unless the host is made to wait for the device to actually catch up first.

### Background

```rust
struct Benchmark { device_available: bool }

impl Benchmark {
    fn time_function<F: FnMut()>(&self, mut func: F, warmup_runs: u32, benchmark_runs: u32) -> f64 {
        for _ in 0..warmup_runs { func(); }
        // if self.device_available { ctx.synchronize()... } -- drain the queue before the clock starts

        let start_time = Instant::now();
        for _ in 0..benchmark_runs { func(); }
        // and again before it stops
        let elapsed = start_time.elapsed();

        (elapsed.as_secs_f64() * 1000.0) / benchmark_runs as f64   // milliseconds
    }
}
```

The two synchronize calls the CUDA edition brackets its timed region with exist for exactly the reason above: without the first one, warmup work still queued on the device could bleed into the timed region; without the second, the host's timer could stop before the device has actually finished the last of the `benchmark_runs` launches, undercounting the true elapsed time. `cudarc` wraps the CUDA *driver* API rather than the Runtime API `nvcc` programs use by default: `CudaContext::synchronize()` is a method on an already-live context, not a free function `cudaDeviceSynchronize()` the Runtime API lets a program call before ever creating one — an honest architectural difference explored directly in this section's worked examples rather than glossed over. Applied to Chapter 18's convolution kernels, throughput is reported in GFLOPS so results are comparable across problem sizes rather than only across identical ones — a `3×3`-kernel 2D convolution performs one multiply and one add per kernel tap, so `2` floating-point operations per tap, times `9` taps, times however many output positions there are:

```rust
fn benchmark_convolution(bench: &Benchmark, size: usize, input: &[f32], kernel: &[f32; 9],
                          basic_out: &mut [f32], padded_out: &mut [f32]) {
    let basic_ops = 2i64 * (size as i64 - 2) * (size as i64 - 2) * 3 * 3;   // valid conv: (size-2)^2 outputs
    let basic_time_ms = bench.time_function(|| conv2d_basic(input, size, kernel, basic_out), 5, 20);
    let basic_gflops = basic_ops as f64 / (basic_time_ms * 1e6);

    let padded_ops = 2i64 * size as i64 * size as i64 * 3 * 3;             // same-shaped conv: size^2 outputs
    let padded_time_ms = bench.time_function(|| conv2d_padded(input, size, kernel, padded_out), 5, 20);
    let padded_gflops = padded_ops as f64 / (padded_time_ms * 1e6);
    // ...
}
```

Note the FLOP count itself differs between the two variants, not just the timing: the unpadded ("valid") kernel produces `(size-2)²` outputs, while the padded ("same") kernel produces `size²` — more total work for the same input, which is exactly the "padding overhead" the two GFLOPS figures let a reader compare directly.

### Worked Example 19.4.1 — Confirming the harness on a genuine, timed workload

Compiled and run:

```bash
cargo run --release --bin 06_benchmark_harness
```

Genuinely compiled and run:

```
=== Section 19.4: benchmark harness, genuinely timed (not hypothetical) ===

attempting CudaContext::device_count() (the closest driver-API call to a context-free check):
  panicked while loading the CUDA driver library: [...]

vector_add_workload over 2000000 elements, 5 warmup + 20 timed runs:
  average time = 0.8238 ms per run
  (this exact number will jitter run to run and machine to machine --
   the number that matters is that it is genuinely measured, not asserted)

  spot-checked output correctness: true

--- honestly noting what this environment CANNOT demonstrate ---
On real GPU hardware, omitting the two synchronize() calls above would let the host's
launch loop race ahead of the device, understating the true kernel time -- exactly Self-
Check Question 5's scenario. This no-GPU sandbox cannot reproduce that specific failure
mode honestly, because there is no device execution to race ahead of -- every driver-API
call here fails synchronously, while loading the driver library, before any queueing could
occur. The harness above is written exactly as it would run on real hardware; only the
scenario the synchronize calls exist to prevent needs a real device to observe.
```

The CUDA edition calls the CUDA Runtime API's `cudaDeviceSynchronize()` unconditionally at the top of its own version of this file, genuinely, honestly reporting `cudaErrorNoDevice` — a real, valid call the Runtime API allows without ever having created a context first, since it creates one implicitly and lazily. `cudarc`'s driver API has no such free function: `CudaContext::synchronize()` is a method on an already-constructed `CudaContext`, so there is no way to attempt the identical context-free check this sandbox can call honestly. The closest genuinely-callable driver-API equivalent is `CudaContext::device_count()`, a static method that — like `CudaContext::new(0)` — still needs to dynamically load the driver library before it can do anything else, so it genuinely hits the identical honest failure this book has reported since Chapter 4, confirmed above rather than assumed. Even so, the harness's own correctness is checked directly (`spot-checked output correctness: true`) rather than assumed, since a benchmark that silently times a broken computation is worse than useless, and the `0.8238` ms figure is a genuine, `std::time::Instant`-measured result — this exact number will jitter on a re-run, which is expected and not a bug.

### Worked Example 19.4.2 — Converting a genuinely measured timing into GFLOPS

Compiled and run:

```bash
cargo run --release --bin 07_convolution_benchmark_gflops
```

Genuinely compiled and run:

```
=== Section 19.4: converting a GENUINELY measured timing into GFLOPS ===

(the Mojo edition of this chapter explicitly could not back this worked example
 with a real run -- its Mojo has never been compiled. This edition can, and does.)

64 x 64 convolution (5 warmup + 20 timed runs each, genuinely measured):
  basic_ops  = 2 * (64-2)^2 * 3 * 3 = 69192
  Basic  convolution: 0.0033 ms, 21.1257 GFLOPS
  padded_ops = 2 * 64^2 * 3 * 3 = 73728
  Padded convolution: 0.0049 ms, 14.9688 GFLOPS
  extra work from padding: 4536 more FLOPs (6.6% more)

128 x 128 convolution (5 warmup + 20 timed runs each, genuinely measured):
  basic_ops  = 2 * (128-2)^2 * 3 * 3 = 285768
  Basic  convolution: 0.0139 ms, 20.5058 GFLOPS
  padded_ops = 2 * 128^2 * 3 * 3 = 294912
  Padded convolution: 0.0151 ms, 19.4955 GFLOPS
  extra work from padding: 9144 more FLOPs (3.2% more)
```

For `size = 64`: `basic_ops = 2 × (64-2)² × 3 × 3 = 2 × 3,844 × 9 = 69,192`, genuinely timed at `0.0033` ms, giving `69,192 / (0.0033 × 1,000,000) ≈ 21.1257` GFLOPS. For the padded variant on the same `size = 64`: `padded_ops = 2 × 64² × 9 = 73,728` — more total operations for the same input, consistent with `size² > (size-2)²` — genuinely timed at `0.0049` ms, giving `≈ 14.9688` GFLOPS. Every millisecond and GFLOPS figure above came from an actual `std::time::Instant`-timed execution of real, compiled Rust host code — these are CPU GFLOPS figures for these host mirrors specifically, not a GPU's, and (like Chapter 17's arena-timing figures) will jitter from run to run and machine to machine; unlike the CUDA edition's own C++ figures (`0.7361`/`0.6655` GFLOPS at `size=64`), this Rust release build's numbers land roughly `20-30×` higher on this sandbox's hardware — a genuine, expected difference between two different compilers, optimization pipelines, and machines timing two structurally-identical loops, not a discrepancy to paper over. What both editions' numbers agree on exactly, because it's pure arithmetic rather than a timing measurement, is the *padding overhead percentage*: `6.6%` more FLOPs at `size=64` and `3.2%` more at `size=128` in both editions, confirming the FLOP-counting formulas themselves are identical even where the measured speeds are not. The size-`128` run above shows the same padding-overhead relationship holding at a second, larger size, with a *smaller* percentage overhead (`3.2%` vs `6.6%`) because padding's fixed per-side cost matters less relative to a larger base.

The same file also genuinely times memory bandwidth and queries device properties, the two remaining measurements this chapter's benchmarking discipline is meant to support:

```bash
cargo run --release --bin 08_bandwidth_and_device_query
```

Genuinely compiled and run:

```
=== Section 19.4: memory bandwidth and device property queries ===

host-to-host copy_from_slice of 64 MiB, 3 warmup + 20 timed runs (genuinely measured):
  average time = 1.8917 ms
  achieved bandwidth = 33.0384 GB/s (this host's memcpy, not a GPU's -- no device here)

  copy correctness check (dst matches src byte-for-byte): true

--- attempting the device-side counterpart, honestly ---
Step 1: CudaContext::new(0) (device allocation and D2H copy both require a live context)
  panicked while loading the CUDA driver library: [...]
  a device allocation of 4KB and a device-to-host copy both require the context this
  sandbox cannot obtain -- neither can run without one, the same early exit every other
  file in this chapter and Chapter 18 hit.

--- device property queries, genuinely called ---
CudaContext::device_count() panicked while loading the CUDA driver library: [...]

(on real hardware this would report a nonzero device count, and calling .attribute() on a
live CudaContext would report multiprocessor count, warp size [always 32 on every CUDA
generation to date], and max threads per block -- exactly the numbers Section 18.1's launch-
configuration math and Section 18.4's warp-shuffle reasoning both assumed. Unlike CUDA
Runtime's cudaGetDeviceProperties, which the C++ edition can call without ever creating a
context first, cudarc's equivalent -- CudaContext::attribute() -- is a method on an already-
live CudaContext: this sandbox cannot even ATTEMPT that specific call, because the context it
would need is exactly the thing every earlier call in this chapter has failed to obtain. This is
a genuine architectural difference between the CUDA Runtime API nvcc programs use by default
and the CUDA Driver API cudarc wraps, not an omission.)
```

The `33.0384` GB/s figure is a genuinely measured host `copy_from_slice` bandwidth, not a GPU's memory bandwidth (which typically runs one to two orders of magnitude higher) — it is included specifically because the *pattern* (warm up, time a batch, divide bytes by seconds) is identical for both, and this no-GPU sandbox can genuinely execute the host half of that pattern even though it cannot execute the device half. `CudaContext::device_count()` is a real, genuinely executed `cudarc` driver-API call, honestly reporting the same dynamic-loading panic every other call in this chapter has hit — on real hardware, that same call returns a nonzero device count, and calling `.attribute()` on the `CudaContext` it would return reports the multiprocessor count, warp size, and maximum threads per block that Chapter 18's launch-configuration math and warp-shuffle reasoning both had to assume. That last query is where this edition's own architecture genuinely diverges from the CUDA edition's: `cudaGetDeviceProperties` is a Runtime API free function the CUDA edition can call without ever creating a context, while `cudarc`'s `.attribute()` is a method that requires one — so this sandbox cannot even attempt that specific call, not because of a missing driver, but because of which CUDA API each edition is genuinely built on.


## 19.5 Complete Runnable Code

Every Rust file below was genuinely compiled with `cargo build --release` (`rustc --edition 2024`, matching this book's toolchain throughout) and executed in this no-GPU, no-CUDA-toolkit cloud sandbox; every printed number above came from one of these runs. Files 01, 02, 03, 04, 06, and 08 also genuinely attempt a real `cudarc::driver::CudaContext::new(0)` or `CudaContext::device_count()` call and honestly report the dynamic-loading failure this sandbox's missing driver produces, following Chapter 4's and Chapter 18's established pattern. File 05 needs no GPU driver, NVRTC, or CUDA toolkit at all -- it asks `rustc` and `nm` for compiler- and linker-verified facts about Rust's own const generics. File 07 is pure host arithmetic, genuinely timed with `std::time::Instant`.

### File: `01_vectorized_add.rs`

Section 19.1 -- vectorized add host mirror (Worked Examples 19.1.1/19.1.2); genuinely attempts CudaContext::new(0).

```rust
use cudarc::driver::{CudaContext, LaunchConfig};
use std::panic;

/// Runs `f`, catching a cudarc dynamic-loading panic instead of letting it abort the process,
/// and returns the panic message when one occurs. Reused verbatim from Chapter 4 and Chapter 18.
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

// The real kernel this file's host mirror corresponds to (never compiled or
// launched here -- this sandbox has neither an NVIDIA driver nor NVRTC,
// established since Chapter 4, and neither nvcc nor cuobjdump at all,
// established in Chapter 18): CUDA's `float4` packs four consecutive
// float32s into one 128-bit load/store, so ONE vector instruction moves
// four elements instead of four separate 32-bit ones.
//
// __global__ void vectorized_add_float4_kernel(float* output, const float* a, const float* b,
//                                               int size, int vec_count) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < vec_count) {
//         float4 va = reinterpret_cast<const float4*>(a)[idx];
//         float4 vb = reinterpret_cast<const float4*>(b)[idx];
//         float4 vout;
//         vout.x = va.x + vb.x; vout.y = va.y + vb.y;
//         vout.z = va.z + vb.z; vout.w = va.w + vb.w;
//         reinterpret_cast<float4*>(output)[idx] = vout;
//     }
// }

// A host mirror of the identical main-loop-plus-remainder shape, with
// genuine counters for vector vs scalar instructions issued -- this is
// what makes "4 total instructions" a measured fact, not merely argued.
struct VecStats {
    vector_instructions: u32,
    scalar_instructions: u32,
}

fn vectorized_add_host(output: &mut [f32], a: &[f32], b: &[f32], size: usize, width: usize) -> VecStats {
    let mut stats = VecStats { vector_instructions: 0, scalar_instructions: 0 };
    let simd_count = (size / width) * width;
    let mut i = 0;
    while i < simd_count {
        for lane in 0..width {
            output[i + lane] = a[i + lane] + b[i + lane];
        }
        stats.vector_instructions += 1; // one vector instruction covers `width` elements
        i += width;
    }
    for i in simd_count..size {
        output[i] = a[i] + b[i];
        stats.scalar_instructions += 1; // one scalar instruction covers 1 element
    }
    stats
}

fn main() {
    println!("=== Section 19.1: vectorized add, float4-width main loop + scalar remainder ===\n");

    // Worked Example 19.1.1 -- size=10, width=4 (float4)
    {
        let size = 10usize;
        let width = 4usize;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 10) as f32).collect();
        let mut out = vec![0.0f32; size];
        let s = vectorized_add_host(&mut out, &a, &b, size, width);
        let simd_count = (size / width) * width;
        println!("size={}, width={}: simd_count = ({}/{})*{} = {}", size, width, size, width, width, simd_count);
        println!("  vector instructions   = {} (covering elements 0-{})", s.vector_instructions, simd_count - 1);
        println!("  scalar instructions   = {} (covering elements {}-{})", s.scalar_instructions, simd_count, size - 1);
        println!(
            "  total instructions    = {}  (scalar-only baseline = {})\n",
            s.vector_instructions + s.scalar_instructions,
            size
        );
        print!("  output: ");
        for v in out.iter() {
            print!("{:.0} ", v);
        }
        println!(" (host reference a[i]+b[i] matches by construction)\n");
    }

    // Worked Example 19.1.2 -- size=41, width=4, remainder a small fraction of the total
    {
        let size = 41usize;
        let width = 4usize;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        let mut out = vec![0.0f32; size];
        let s = vectorized_add_host(&mut out, &a, &b, size, width);
        let simd_count = (size / width) * width;
        println!("size={}, width={}: simd_count = ({}/{})*{} = {}", size, width, size, width, width, simd_count);
        println!("  vector instructions   = {} (covering elements 0-{})", s.vector_instructions, simd_count - 1);
        println!("  scalar instructions   = {} (covering elements {}-{})", s.scalar_instructions, simd_count, size - 1);
        println!(
            "  total instructions    = {}  (scalar-only baseline = {})",
            s.vector_instructions + s.scalar_instructions,
            size
        );
        let total = (s.vector_instructions + s.scalar_instructions) as f64;
        println!(
            "  ratio vs Worked Example 19.1.1: {:.1}% of baseline here vs {:.1}% there\n",
            100.0 * total / size as f64,
            100.0 * 4.0 / 10.0
        );
    }

    // Genuinely attempt the real pipeline over 8 elements (2 float4-width
    // groups, no remainder) and honestly report what happens without a GPU,
    // following this book's established Chapter 4/18 pattern.
    println!("--- vectorized_add_float4_kernel: genuine kernel-launch attempt, no GPU in this environment ---");
    let n: usize = 8;
    let h_a: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let h_b: [f32; 8] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];
    let vec_count = n / 4;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (vec_count as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("Step 1: CudaContext::new(0)");
    match catch_cuda(|| CudaContext::new(0)) {
        Ok(Ok(_ctx)) => println!("  succeeded: a real GPU and driver are present."),
        Ok(Err(e)) => println!("  returned a driver error (a GPU exists but the call failed): {:?}", e),
        Err(msg) => {
            println!("  panicked while loading the CUDA driver library:");
            println!("  {}", msg);
        }
    }
    println!(
        "Steps 2-4 (device allocation, host->device copy, kernel launch <<<{},{}>>> -- each thread\n\
         handling one float4 = 4 elements) all require a live CudaContext, so none of them can run\n\
         without one -- exactly the same honest early-exit Chapter 4's and Chapter 18's own pipelines hit.",
        cfg.grid_dim.0, cfg.block_dim.0
    );

    println!();
    print!("host-computed reference (a+b): ");
    for i in 0..n {
        print!("{} ", h_a[i] + h_b[i]);
    }
    println!();
}
```

```bash
cargo run --release --bin 01_vectorized_add
```

### File: `02_vector_width_dtype_trap.rs`

Section 19.1 COMMON TRAP and Worked Example 19.1.3 -- size_of-based dtype evidence, this edition's honest replacement for cuobjdump SASS decoding; genuinely attempts CudaContext::new(0).

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

// [COMMON TRAP] materialized in real CUDA types: a 128-bit vector load is
// fixed in BYTES, not in ELEMENT COUNT -- float4 packs four 4-byte floats
// into 128 bits, but double2 packs only two 8-byte doubles into the SAME
// 128 bits. The real kernels this file's evidence corresponds to (never
// compiled or launched here -- no driver, and this sandbox has no CUDA
// toolkit at all, established in Chapter 18, so neither can be compiled
// for a real architecture the way the CUDA edition's own file 02 is):
//
// __global__ void vectorized_add_float4_kernel(float* output, const float* a, const float* b, int vec_count) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < vec_count) {
//         float4 va = reinterpret_cast<const float4*>(a)[idx];
//         float4 vb = reinterpret_cast<const float4*>(b)[idx];
//         float4 vout;
//         vout.x = va.x + vb.x; vout.y = va.y + vb.y; vout.z = va.z + vb.z; vout.w = va.w + vb.w;
//         reinterpret_cast<float4*>(output)[idx] = vout;
//     }
// }
// __global__ void vectorized_add_double2_kernel(double* output, const double* a, const double* b, int vec_count) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < vec_count) {
//         double2 va = reinterpret_cast<const double2*>(a)[idx];
//         double2 vb = reinterpret_cast<const double2*>(b)[idx];
//         double2 vout;
//         vout.x = va.x + vb.x; vout.y = va.y + vb.y;
//         reinterpret_cast<double2*>(output)[idx] = vout;
//     }
// }

fn main() {
    println!("=== Section 19.1 COMMON TRAP: one dtype's vector width does not transfer to another ===\n");

    // The CUDA edition confirms this same claim two ways: sizeof() (a
    // compile-time fact) and cuobjdump --dump-sass on a compiled cubin (a
    // fact about the actual generated machine code). This sandbox has no
    // CUDA toolkit at all -- neither nvcc nor cuobjdump exist here, confirmed
    // genuinely in Chapter 18 -- so decoding real SASS for a float4/double2
    // kernel the way that edition's own Worked Example 19.1.3 does is not
    // something this sandbox can do honestly. What it CAN do honestly,
    // genuinely, is ask rustc's own compiler for the equivalent fact about
    // Rust's own fixed-size array types: std::mem::size_of::<[f32; 4]>()
    // and std::mem::size_of::<[f64; 2]>() are both compiler-verified, not
    // assumed -- the same "ask the compiler, not the disassembler" evidence
    // source Chapter 18's offset_of!-based Worked Example 18.2.2 already
    // established as this edition's honest substitute for machine-code
    // inspection.
    let float4_bytes = std::mem::size_of::<[f32; 4]>();
    let float4_elems = float4_bytes / std::mem::size_of::<f32>();
    let double2_bytes = std::mem::size_of::<[f64; 2]>();
    let double2_elems = double2_bytes / std::mem::size_of::<f64>();

    println!(
        "size_of::<[f32; 4]>()  = {} bytes, elements packed = {} (f32, {} bytes each)",
        float4_bytes,
        float4_elems,
        std::mem::size_of::<f32>()
    );
    println!(
        "size_of::<[f64; 2]>()  = {} bytes, elements packed = {} (f64, {} bytes each)\n",
        double2_bytes,
        double2_elems,
        std::mem::size_of::<f64>()
    );

    println!("Same 128-bit vector payload, different element counts:");
    println!("  float4-equivalent  width = {} elements per 128-bit payload", float4_elems);
    println!("  double2-equivalent width = {} elements per 128-bit payload", double2_elems);
    println!("  a kernel retuned from float4's width=4 to double2 data, still assuming width=4,");
    println!("  would read 16 bytes past the intended 2-element group on every vector op -- corruption,");
    println!("  not merely a slowdown, if the width constant isn't re-derived for the new dtype.\n");

    assert_eq!(float4_bytes, double2_bytes, "both must be genuinely 16 bytes for the trap to hold");
    assert_eq!(float4_elems, 4);
    assert_eq!(double2_elems, 2);

    println!("--- attempting a genuine kernel launch over the same 32-byte-per-array budget ---");
    match catch_cuda(|| CudaContext::new(0)) {
        Ok(Ok(_ctx)) => println!("  CudaContext::new(0) succeeded: a real GPU and driver are present."),
        Ok(Err(e)) => println!("  CudaContext::new(0) returned a driver error: {:?}", e),
        Err(msg) => {
            println!("  CudaContext::new(0) panicked while loading the CUDA driver library:");
            println!("  {}", msg);
        }
    }
    println!("  (the same real, reproducible failure Section 19.1's launch attempt and Chapter 18 both hit --");
    println!("   compiling and launching either vectorized_add_float4_kernel<<<1,2>>> or");
    println!("   vectorized_add_double2_kernel<<<1,2>>> requires a live context neither can obtain here)\n");

    println!("what size_of DOESN'T confirm, honestly: whether a compiled kernel actually loading a");
    println!("float4 or double2 emits the identical LDG.E.128/STG.E.128 instruction pair the CUDA");
    println!("edition's own cuobjdump output shows -- that claim rests on nvcc's code generation,");
    println!("which this sandbox has no toolkit to inspect at all. What size_of DOES confirm,");
    println!("genuinely: the two payloads are the same fixed number of BYTES (16), while the number");
    println!("of logical elements packed into that fixed byte count depends entirely on the element");
    println!("type's own size -- the same distinction the CUDA edition's SASS draws between a fixed");
    println!("128-bit instruction width and a dtype-dependent element count, from a different,");
    println!("equally genuine evidence source.\n");

    println!("[COMMON TRAP]  One dtype's vector width does not transfer to another\n");
    println!("A 128-bit vector load is fixed in BYTES, not in ELEMENT COUNT. float4 packs four");
    println!("4-byte floats into 128 bits, but double2 packs only two 8-byte doubles into the SAME");
    println!("128 bits. Code retuned from float4's width-4 data to double2 data, while still assuming");
    println!("width=4, would read 16 bytes past the intended 2-element group on every vector op --");
    println!("corruption, not merely a slowdown, if the width constant isn't re-derived for the new");
    println!("dtype. This is exactly why a real vectorized kernel computes its width from");
    println!("sizeof(VectorType) / sizeof(ElementType) rather than hardcoding a width borrowed from");
    println!("testing a different dtype.");
}
```

```bash
cargo run --release --bin 02_vector_width_dtype_trap
```

### File: `03_unrolled_add.rs`

Section 19.2 -- loop unrolling, plain vs. unrolled host mirrors (Worked Example 19.2.1); genuinely attempts CudaContext::new(0).

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

// Unrolling trades loop-control overhead for a bigger loop body doing more
// work per iteration -- it does NOT change the arithmetic instruction
// count. A host mirror with genuine counters for both "loop-control
// events" (increment+compare+branch, once per iteration) and "arithmetic
// operations" (once per element) is what makes that claim measured rather
// than merely argued.
struct UnrollStats {
    loop_iterations: u32, // how many times the loop's own bookkeeping runs
    arithmetic_ops: u32,  // how many additions are actually performed
}

fn plain_add(output: &mut [f32], a: &[f32], b: &[f32], size: usize) -> UnrollStats {
    let mut s = UnrollStats { loop_iterations: 0, arithmetic_ops: 0 };
    for i in 0..size {
        output[i] = a[i] + b[i];
        s.loop_iterations += 1;
        s.arithmetic_ops += 1;
    }
    s
}

fn unrolled_add(output: &mut [f32], a: &[f32], b: &[f32], size: usize) -> UnrollStats {
    let mut s = UnrollStats { loop_iterations: 0, arithmetic_ops: 0 };
    const UNROLL: usize = 4;
    let unrolled_count = (size / UNROLL) * UNROLL;
    let mut i = 0;
    while i < unrolled_count {
        output[i] = a[i] + b[i];
        output[i + 1] = a[i + 1] + b[i + 1];
        output[i + 2] = a[i + 2] + b[i + 2];
        output[i + 3] = a[i + 3] + b[i + 3];
        s.loop_iterations += 1; // ONE iteration's worth of bookkeeping...
        s.arithmetic_ops += 4; // ...covers FOUR additions
        i += UNROLL;
    }
    for i in unrolled_count..size {
        // remainder, one at a time
        output[i] = a[i] + b[i];
        s.loop_iterations += 1;
        s.arithmetic_ops += 1;
    }
    s
}

// The real CUDA-kernel form of the same idea: one thread per group of
// UNROLL elements instead of one thread per element, so the SAME grid
// covers UNROLL times more data with UNROLL times fewer threads' worth of
// index-computation and bounds-check overhead. Never compiled or launched
// here -- this sandbox has no CUDA toolkit at all, established in Chapter 18.
//
// __global__ void unrolled_add_kernel(float* output, const float* a, const float* b, int size) {
//     const int UNROLL = 4;
//     int base = (blockIdx.x * blockDim.x + threadIdx.x) * UNROLL;
//     #pragma unroll
//     for (int lane = 0; lane < UNROLL; lane++) {
//         int i = base + lane;
//         if (i < size) output[i] = a[i] + b[i];
//     }
// }

fn main() {
    println!("=== Section 19.2: loop unrolling, overhead saved with arithmetic unchanged ===\n");

    // Worked Example 19.2.1 -- size=1000, UNROLL=4, evenly divisible
    {
        let size = 1000usize;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 3) as f32).collect();
        let mut out_plain = vec![0.0f32; size];
        let mut out_unrolled = vec![0.0f32; size];

        let plain = plain_add(&mut out_plain, &a, &b, size);
        let unrolled = unrolled_add(&mut out_unrolled, &a, &b, size);

        println!("size={}, UNROLL=4: unrolled_count = ({}/4)*4 = {}", size, size, (size / 4) * 4);
        println!("  plain loop:    {} loop-control events, {} arithmetic ops", plain.loop_iterations, plain.arithmetic_ops);
        println!(
            "  unrolled loop: {} loop-control events, {} arithmetic ops",
            unrolled.loop_iterations, unrolled.arithmetic_ops
        );
        println!(
            "  overhead reduction: {}x fewer loop-control events ({} -> {})",
            plain.loop_iterations / unrolled.loop_iterations,
            plain.loop_iterations,
            unrolled.loop_iterations
        );
        println!("  arithmetic op count identical: {}\n", plain.arithmetic_ops == unrolled.arithmetic_ops);

        let matches = out_plain.iter().zip(out_unrolled.iter()).all(|(p, u)| p == u);
        println!("  every output element identical between plain and unrolled: {}\n", matches);
    }

    // A size with a genuine remainder, to confirm the remainder loop still runs correctly
    {
        let size = 4002usize;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0f32; size];
        let mut out_plain = vec![0.0f32; size];
        let mut out_unrolled = vec![0.0f32; size];

        let plain = plain_add(&mut out_plain, &a, &b, size);
        let unrolled = unrolled_add(&mut out_unrolled, &a, &b, size);
        let unrolled_count = (size / 4) * 4;

        println!(
            "size={}, UNROLL=4: unrolled_count = ({}/4)*4 = {}, remainder = {} element(s)",
            size,
            size,
            unrolled_count,
            size - unrolled_count
        );
        println!("  plain loop:    {} loop-control events, {} arithmetic ops", plain.loop_iterations, plain.arithmetic_ops);
        println!(
            "  unrolled loop: {} loop-control events ({} full groups + {} remainder), {} arithmetic ops",
            unrolled.loop_iterations,
            unrolled_count / 4,
            size - unrolled_count,
            unrolled.arithmetic_ops
        );
        println!(
            "  arithmetic totals match: {} (both {})\n",
            plain.arithmetic_ops == unrolled.arithmetic_ops,
            plain.arithmetic_ops
        );

        let matches = out_plain.iter().zip(out_unrolled.iter()).all(|(p, u)| p == u);
        println!("  every output element identical between plain and unrolled: {}\n", matches);
    }

    // Genuinely attempt the real unrolled kernel
    println!("--- unrolled_add_kernel: genuine kernel-launch attempt, no GPU in this environment ---");
    let n: usize = 16;
    let h_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let h_b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let threads = (n + 3) / 4; // one thread per group of 4

    match catch_cuda(|| CudaContext::new(0)) {
        Ok(Ok(_ctx)) => println!("  CudaContext::new(0) succeeded: a real GPU and driver are present."),
        Ok(Err(e)) => println!("  CudaContext::new(0) returned a driver error: {:?}", e),
        Err(msg) => {
            println!("  CudaContext::new(0) panicked while loading the CUDA driver library:");
            println!("  {}", msg);
        }
    }
    println!(
        "  kernel launch <<<1,{}>>> (each thread handles 4 elements) cannot run without a live context",
        threads
    );

    print!("host reference (a+b): ");
    for i in 0..n {
        print!("{} ", h_a[i] + h_b[i]);
    }
    println!();
}
```

```bash
cargo run --release --bin 03_unrolled_add
```

### File: `04_fusion_memory_traffic.rs`

Section 19.2 -- fusion's memory-traffic saving, counted exactly (Worked Example 19.2.2); genuinely attempts CudaContext::new(0).

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

// Fusion's saving is in MEMORY TRAFFIC, not instruction count -- counted
// here with the same genuine read/write counter technique Chapter 18 used
// for convolution's redundant reads, extended to also count writes.
struct MemStats {
    reads: u64,
    writes: u64,
}

// Three separate "kernels": mul, then add, then relu -- each one reads its
// inputs from global memory and writes a full tensor-sized result, exactly
// the unfused pipeline Section 19.2 describes.
fn unfused_pipeline(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32], size: usize, stats: &mut MemStats) {
    let mut tmp1 = vec![0.0f32; size]; // a * b
    let mut tmp2 = vec![0.0f32; size]; // tmp1 + c

    for i in 0..size {
        stats.reads += 2; // read a[i], b[i]
        tmp1[i] = a[i] * b[i];
        stats.writes += 1; // write tmp1[i]
    }
    for i in 0..size {
        stats.reads += 2; // read tmp1[i], c[i]
        tmp2[i] = tmp1[i] + c[i];
        stats.writes += 1; // write tmp2[i]
    }
    for i in 0..size {
        stats.reads += 1; // read tmp2[i]
        output[i] = if tmp2[i] > 0.0 { tmp2[i] } else { 0.0 };
        stats.writes += 1; // write output[i]
    }
}

// One fused kernel: reads a, b, c exactly once each, writes output exactly
// once, with the multiply/add/relu happening in a register between the
// reads and the one write.
fn fused_pipeline(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32], size: usize, stats: &mut MemStats) {
    for i in 0..size {
        stats.reads += 3; // read a[i], b[i], c[i] -- once each, total
        let z = a[i] * b[i] + c[i];
        output[i] = if z > 0.0 { z } else { 0.0 };
        stats.writes += 1; // write output[i] -- once
    }
}

// The real GPU-kernel form of the fused version -- never compiled or
// launched here, this sandbox has no CUDA toolkit at all (Chapter 18).
//
// __global__ void fused_mul_add_relu_kernel(float* output, const float* a, const float* b,
//                                            const float* c, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         float z = a[idx] * b[idx] + c[idx];
//         output[idx] = z > 0.0f ? z : 0.0f;
//     }
// }

fn main() {
    println!("=== Section 19.2: fusion's memory-traffic saving, counted exactly ===\n");

    let size = 6usize;
    let a: [f32; 6] = [1.0, -2.0, 3.0, -4.0, 5.0, 0.0];
    let b: [f32; 6] = [2.0, 3.0, -1.0, 4.0, -2.0, 5.0];
    let c: [f32; 6] = [0.0, 1.0, -10.0, 2.0, 1.0, -1.0];
    let mut out_unfused = [0.0f32; 6];
    let mut out_fused = [0.0f32; 6];

    let mut unfused_stats = MemStats { reads: 0, writes: 0 };
    let mut fused_stats = MemStats { reads: 0, writes: 0 };

    unfused_pipeline(&a, &b, &c, &mut out_unfused, size, &mut unfused_stats);
    fused_pipeline(&a, &b, &c, &mut out_fused, size, &mut fused_stats);

    println!("relu(a*b+c) on {} elements:", size);
    println!(
        "  unfused (3 kernels): {} tensor-sized reads, {} tensor-sized writes, {} total memory ops",
        unfused_stats.reads / size as u64,
        unfused_stats.writes / size as u64,
        unfused_stats.reads / size as u64 + unfused_stats.writes / size as u64
    );
    println!(
        "  fused   (1 kernel):  {} tensor-sized reads, {} tensor-sized writes, {} total memory ops\n",
        fused_stats.reads / size as u64,
        fused_stats.writes / size as u64,
        fused_stats.reads / size as u64 + fused_stats.writes / size as u64
    );

    print!("outputs identical between unfused and fused: ");
    let matches = out_unfused.iter().zip(out_fused.iter()).all(|(u, f)| u == f);
    println!("{}", matches);
    print!("fused output: ");
    for v in out_fused.iter() {
        print!("{:.1} ", v);
    }
    println!("\n");

    let unfused_total = unfused_stats.reads + unfused_stats.writes;
    let fused_total = fused_stats.reads + fused_stats.writes;
    println!(
        "memory traffic reduction: {:.0}x ({} ops -> {} ops), independent of size {}\n",
        unfused_total as f64 / fused_total as f64,
        unfused_total / size as u64,
        fused_total / size as u64,
        size
    );

    // Genuinely attempt the real fused kernel
    println!("--- fused_mul_add_relu_kernel: genuine kernel-launch attempt, no GPU in this environment ---");
    match catch_cuda(|| CudaContext::new(0)) {
        Ok(Ok(_ctx)) => println!("  CudaContext::new(0) succeeded: a real GPU and driver are present."),
        Ok(Err(e)) => println!("  CudaContext::new(0) returned a driver error: {:?}", e),
        Err(msg) => {
            println!("  CudaContext::new(0) panicked while loading the CUDA driver library:");
            println!("  {}", msg);
        }
    }
    println!("  kernel launch <<<1,{}>>> cannot run without a live context", size);

    print!("host reference (matches fused_pipeline above): ");
    for v in out_fused.iter() {
        print!("{:.1} ", v);
    }
    println!();

    println!();
    println!("[COMMON TRAP]  Fusing an op without registering a matching backward rule\n");
    println!("A kernel that fuses forward computation but is dropped into the graph under one of the");
    println!("ALREADY-registered op names in Chapter 16's OpRegistry (say, calling chain_rule_step(\"mul\", ...)");
    println!("on a node that actually ran the fused multiply-add-relu) would run MulOp's backward rule --");
    println!("which only knows how to differentiate a plain multiplication -- against a node whose forward");
    println!("pass did three operations, not one. The gradient it returns would be a plausible-looking");
    println!("ScalarTensor of the right shape, silently wrong in value, with nothing about the shape to catch");
    println!("it. A fused op needs its OWN registered Differentiable entry with a backward rule derived for");
    println!("the whole fused expression -- reusing an existing op's name for a kernel that no longer matches");
    println!("what that name's backward rule assumes is exactly how a correct-looking forward pass ends up");
    println!("paired with an incorrect gradient.");
}
```

```bash
cargo run --release --bin 04_fusion_memory_traffic
```

### File: `05_compile_time_specialized_dot.rs`

Section 19.3 -- compile-time specialization via const generics, confirmed via nm on the linked binary (Worked Example 19.3.1).

```rust
// Rust's analog of Mojo's `fn foo[n: Int](...)` and C++'s `template<int N>`:
// a const generic parameter resolved entirely at COMPILE time, not
// runtime. compile_time_specialized_dot::<4> and
// compile_time_specialized_dot::<2> are two fully separate, independently
// generated function bodies, with no runtime branch on N anywhere in
// either compiled output -- confirmed below, genuinely, by inspecting the
// actual compiled symbols rather than only trusting the claim. This file
// needs no cudarc, no driver, and no CUDA toolkit at all: `rustc` itself
// is the only tool this section's evidence depends on.
#[inline(never)]
fn compile_time_specialized_dot<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    // N is baked into the generated code at compile time -- no runtime
    // loop-bound check, no dynamic dispatch on vector width.
    let mut sum = 0.0f32;
    for i in 0..N {
        sum += a[i] * b[i];
    }
    sum
}

// Force both instantiations to actually exist in the compiled binary
// (rather than being optimized away or inlined into oblivion) by calling
// them from externally-visible, non-inlined wrapper functions -- the
// direct Rust equivalent of the CUDA edition's __attribute__((noinline))
// call_dot_n4/call_dot_n2 wrappers.
#[inline(never)]
pub fn call_dot_n4(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    compile_time_specialized_dot::<4>(a, b)
}
#[inline(never)]
pub fn call_dot_n2(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    compile_time_specialized_dot::<2>(a, b)
}

fn main() {
    println!("=== Section 19.3: compile-time specialization, two genuinely separate bodies ===\n");

    let a4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let b4: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
    let result4 = compile_time_specialized_dot::<4>(&a4, &b4);
    println!(
        "compile_time_specialized_dot::<4>([1,2,3,4], [5,6,7,8]) = {:.1} (expected 1*5+2*6+3*7+4*8 = 70)",
        result4
    );

    let a2: [f32; 2] = [2.0, 3.0];
    let b2: [f32; 2] = [4.0, 5.0];
    let result2 = compile_time_specialized_dot::<2>(&a2, &b2);
    println!("compile_time_specialized_dot::<2>([2,3], [4,5]) = {:.1} (expected 2*4+3*5 = 23)\n", result2);

    println!("these are not one function called twice with a runtime-varying length --");
    println!("N=4 and N=2 select two DIFFERENT, separately-compiled functions. Confirmed via");
    println!("call_dot_n4/call_dot_n2 (kept #[inline(never)] so both survive to the linked binary):");
    println!("  call_dot_n4(&a4,&b4) = {:.1}", call_dot_n4(&a4, &b4));
    println!("  call_dot_n2(&a2,&b2) = {:.1}\n", call_dot_n2(&a2, &b2));

    println!("run `nm` (raw, mangled) on the compiled binary to see this genuinely: two DISTINCT");
    println!("symbols exist, one per instantiation, distinguished by rustc's own per-instantiation");
    println!("hash suffix -- not one generic symbol with an N parameter passed at runtime. Piping the");
    println!("same symbols through `rustfilt` demangles both to the same friendly name");
    println!("(compile_time_specialized_dot drops its const-generic argument when demangled, unlike");
    println!("C++'s <N> which stays visible) -- it is the RAW, undemangled symbol below that carries");
    println!("the proof of two distinct compiled bodies, not the demangled one.");
}
```

```bash
cargo run --release --bin 05_compile_time_specialized_dot
```

### File: `06_benchmark_harness.rs`

Section 19.4 -- the warmup-then-average Benchmark harness, genuinely timed (Worked Example 19.4.1).

```rust
use cudarc::driver::CudaContext;
use std::panic;
use std::time::Instant;

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

// The warmup-then-average pattern this chapter's benchmarking section is
// built around: run the function a few times uncounted (so cold caches /
// first-call overhead don't dominate), then time a batch of representative
// runs and average. For GPU work, the CUDA edition brackets the timed
// region with cudaDeviceSynchronize() so a host that queues work and races
// ahead doesn't get measured instead of the device actually finishing it.
// cudarc wraps the CUDA *driver* API rather than the Runtime API nvcc
// programs use by default -- CudaContext::synchronize() is a method on an
// already-live context (see main() below for why this matters honestly in
// this sandbox), so `device_available` here plays the identical role
// `device_available` plays in the CUDA edition's own struct.
struct Benchmark {
    device_available: bool,
}

impl Benchmark {
    fn time_function<F: FnMut()>(&self, mut func: F, warmup_runs: u32, benchmark_runs: u32) -> f64 {
        for _ in 0..warmup_runs {
            func();
        }
        // if self.device_available { ctx.synchronize()... } -- drain the queue before the clock starts;
        // skipped here exactly as the CUDA edition skips it when device_available is false.
        let _ = self.device_available;

        let start_time = Instant::now();
        for _ in 0..benchmark_runs {
            func();
        }
        // and again before it stops
        let elapsed = start_time.elapsed();

        (elapsed.as_secs_f64() * 1000.0) / benchmark_runs as f64 // milliseconds
    }
}

// A genuine, real workload to benchmark -- a plain vector add over a
// reasonably large host array, large enough that the loop itself takes a
// measurable (if small and jittery) amount of time.
fn vector_add_workload(output: &mut [f32], a: &[f32], b: &[f32], size: usize) {
    for i in 0..size {
        output[i] = a[i] + b[i];
    }
}

fn main() {
    println!("=== Section 19.4: benchmark harness, genuinely timed (not hypothetical) ===\n");

    // The CUDA edition calls the CUDA Runtime API's cudaDeviceSynchronize()
    // unconditionally here, honestly reporting cudaErrorNoDevice -- a real,
    // valid call the Runtime API allows without ever having created a
    // context first (it lazily creates one implicitly). cudarc wraps the
    // CUDA *driver* API instead, where CudaContext::synchronize() is a
    // method on an already-constructed CudaContext -- there is no free
    // function this sandbox can call to make the identical attempt without
    // first obtaining a context, and obtaining one is exactly the step that
    // fails. The closest genuinely-callable driver-API equivalent is
    // CudaContext::device_count(), a static method that -- like
    // CudaContext::new(0) -- still needs to dynamically load the driver
    // library before it can do anything else, so it hits the identical
    // honest failure this book has reported since Chapter 4.
    println!("attempting CudaContext::device_count() (the closest driver-API call to a context-free check):");
    match catch_cuda(CudaContext::device_count) {
        Ok(Ok(n)) => println!("  succeeded: {} device(s) available.", n),
        Ok(Err(e)) => println!("  returned a driver error: {:?}", e),
        Err(msg) => {
            println!("  panicked while loading the CUDA driver library:");
            println!("  {}", msg);
        }
    }
    println!();

    let size = 2_000_000usize; // 2M floats -- big enough to take a measurable amount of host time
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();
    let mut out = vec![0.0f32; size];

    let bench = Benchmark { device_available: false }; // honestly: no GPU in this environment

    let time_ms = bench.time_function(|| vector_add_workload(&mut out, &a, &b, size), 5, 20);
    println!("vector_add_workload over {} elements, 5 warmup + 20 timed runs:", size);
    println!("  average time = {:.4} ms per run", time_ms);
    println!("  (this exact number will jitter run to run and machine to machine --");
    println!("   the number that matters is that it is genuinely measured, not asserted)\n");

    // Verify the warmup runs and timed runs actually produced the correct
    // result -- a benchmark that silently measures a broken computation is
    // worse than useless.
    let mut correct = true;
    let mut i = 0usize;
    while i < size {
        if out[i] != a[i] + b[i] {
            correct = false;
        }
        i += size / 8;
    }
    println!("  spot-checked output correctness: {}", correct);

    println!("\n--- honestly noting what this environment CANNOT demonstrate ---");
    println!("On real GPU hardware, omitting the two synchronize() calls above would let the host's");
    println!("launch loop race ahead of the device, understating the true kernel time -- exactly Self-");
    println!("Check Question 5's scenario. This no-GPU sandbox cannot reproduce that specific failure");
    println!("mode honestly, because there is no device execution to race ahead of -- every driver-API");
    println!("call here fails synchronously, while loading the driver library, before any queueing could");
    println!("occur. The harness above is written exactly as it would run on real hardware; only the");
    println!("scenario the synchronize calls exist to prevent needs a real device to observe.");
}
```

```bash
cargo run --release --bin 06_benchmark_harness
```

### File: `07_convolution_benchmark_gflops.rs`

Section 19.4 -- Chapter 18's convolution kernels, genuinely benchmarked and converted to GFLOPS (Worked Example 19.4.2).

```rust
use std::time::Instant;

struct Benchmark {
    device_available: bool,
}

impl Benchmark {
    fn time_function<F: FnMut()>(&self, mut func: F, warmup_runs: u32, benchmark_runs: u32) -> f64 {
        for _ in 0..warmup_runs {
            func();
        }
        let _ = self.device_available;
        let start_time = Instant::now();
        for _ in 0..benchmark_runs {
            func();
        }
        let elapsed = start_time.elapsed();
        (elapsed.as_secs_f64() * 1000.0) / benchmark_runs as f64 // milliseconds
    }
}

// Chapter 18's naive ("valid", unpadded) convolution, generalized to any
// size instead of the 4x4 worked example -- the "basic" variant this
// section benchmarks.
fn conv2d_basic(input: &[f32], size: usize, kernel: &[f32; 9], output: &mut [f32]) {
    let output_size = size - 2; // 3x3 kernel, no padding
    for out_row in 0..output_size {
        for out_col in 0..output_size {
            let mut sum = 0.0f32;
            for k_r in 0..3 {
                for k_c in 0..3 {
                    sum += input[(out_row + k_r) * size + (out_col + k_c)] * kernel[k_r * 3 + k_c];
                }
            }
            output[out_row * output_size + out_col] = sum;
        }
    }
}

// Chapter 18's padded ("same") convolution -- produces size x size outputs
// instead of (size-2) x (size-2), the "padded" variant.
fn conv2d_padded(input: &[f32], size: usize, kernel: &[f32; 9], output: &mut [f32]) {
    let padded_size = size + 2;
    let mut padded = vec![0.0f32; padded_size * padded_size];
    for r in 0..size {
        for c in 0..size {
            padded[(r + 1) * padded_size + (c + 1)] = input[r * size + c];
        }
    }

    for out_row in 0..size {
        for out_col in 0..size {
            let mut sum = 0.0f32;
            for k_r in 0..3 {
                for k_c in 0..3 {
                    sum += padded[(out_row + k_r) * padded_size + (out_col + k_c)] * kernel[k_r * 3 + k_c];
                }
            }
            output[out_row * size + out_col] = sum;
        }
    }
}

fn benchmark_convolution(bench: &Benchmark, size: usize, input: &[f32], kernel: &[f32; 9], basic_out: &mut [f32], padded_out: &mut [f32]) {
    let basic_ops = 2i64 * (size as i64 - 2) * (size as i64 - 2) * 3 * 3; // valid conv: (size-2)^2 outputs
    let basic_time_ms = bench.time_function(|| conv2d_basic(input, size, kernel, basic_out), 5, 20);
    let basic_gflops = basic_ops as f64 / (basic_time_ms * 1e6);

    let padded_ops = 2i64 * size as i64 * size as i64 * 3 * 3; // same-shaped conv: size^2 outputs
    let padded_time_ms = bench.time_function(|| conv2d_padded(input, size, kernel, padded_out), 5, 20);
    let padded_gflops = padded_ops as f64 / (padded_time_ms * 1e6);

    println!("{} x {} convolution (5 warmup + 20 timed runs each, genuinely measured):", size, size);
    println!("  basic_ops  = 2 * ({}-2)^2 * 3 * 3 = {}", size, basic_ops);
    println!("  Basic  convolution: {:.4} ms, {:.4} GFLOPS", basic_time_ms, basic_gflops);
    println!("  padded_ops = 2 * {}^2 * 3 * 3 = {}", size, padded_ops);
    println!("  Padded convolution: {:.4} ms, {:.4} GFLOPS", padded_time_ms, padded_gflops);
    println!(
        "  extra work from padding: {} more FLOPs ({:.1}% more)\n",
        padded_ops - basic_ops,
        100.0 * (padded_ops - basic_ops) as f64 / basic_ops as f64
    );
}

fn main() {
    println!("=== Section 19.4: converting a GENUINELY measured timing into GFLOPS ===\n");
    println!("(the Mojo edition of this chapter explicitly could not back this worked example");
    println!(" with a real run -- its Mojo has never been compiled. This edition can, and does.)\n");

    let bench = Benchmark { device_available: false };

    let size = 64usize;
    let input: Vec<f32> = (0..size * size).map(|i| (i % 17) as f32 - 8.0).collect();
    let kernel: [f32; 9] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let mut basic_out = vec![0.0f32; (size - 2) * (size - 2)];
    let mut padded_out = vec![0.0f32; size * size];

    benchmark_convolution(&bench, size, &input, &kernel, &mut basic_out, &mut padded_out);

    // A second size, to show the same formula scale.
    let size2 = 128usize;
    let input2: Vec<f32> = (0..size2 * size2).map(|i| (i % 23) as f32 - 11.0).collect();
    let mut basic_out2 = vec![0.0f32; (size2 - 2) * (size2 - 2)];
    let mut padded_out2 = vec![0.0f32; size2 * size2];
    benchmark_convolution(&bench, size2, &input2, &kernel, &mut basic_out2, &mut padded_out2);
}
```

```bash
cargo run --release --bin 07_convolution_benchmark_gflops
```

### File: `08_bandwidth_and_device_query.rs`

Section 19.4 -- host memory-bandwidth timing and device property queries; genuinely attempts CudaContext::new(0) and CudaContext::device_count().

```rust
use cudarc::driver::CudaContext;
use std::panic;
use std::time::Instant;

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

// The same warmup-then-average harness, applied to the other two things
// Section 19.4's closing paragraph mentions: memory bandwidth (copy a
// large buffer, divide its size by elapsed time) and device property
// queries (core count, warp size, max threads per block).
fn time_host_copy_ms(dst: &mut [u8], src: &[u8], warmup_runs: u32, benchmark_runs: u32) -> f64 {
    for _ in 0..warmup_runs {
        dst.copy_from_slice(src);
    }
    let start_time = Instant::now();
    for _ in 0..benchmark_runs {
        dst.copy_from_slice(src);
    }
    let elapsed = start_time.elapsed();
    (elapsed.as_secs_f64() * 1000.0) / benchmark_runs as f64
}

fn main() {
    println!("=== Section 19.4: memory bandwidth and device property queries ===\n");

    // --- Bandwidth: genuinely time a large host-to-host copy ---
    let n_bytes: usize = 64 * 1024 * 1024; // 64 MiB
    let mut src = vec![0u8; n_bytes];
    for i in (0..n_bytes).step_by(4096) {
        src[i] = (i % 256) as u8; // touch pages
    }
    let mut dst = vec![0u8; n_bytes];

    let time_ms = time_host_copy_ms(&mut dst, &src, 3, 20);
    let seconds = time_ms / 1000.0;
    let gb = n_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let gb_per_sec = gb / seconds;

    println!(
        "host-to-host copy_from_slice of {:.0} MiB, 3 warmup + 20 timed runs (genuinely measured):",
        n_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("  average time = {:.4} ms", time_ms);
    println!("  achieved bandwidth = {:.4} GB/s (this host's memcpy, not a GPU's -- no device here)\n", gb_per_sec);

    let copy_correct = dst == src;
    println!("  copy correctness check (dst matches src byte-for-byte): {}\n", copy_correct);

    // --- A genuine attempt at the device-side counterpart, honestly reported ---
    println!("--- attempting the device-side counterpart, honestly ---");
    println!("Step 1: CudaContext::new(0) (device allocation and D2H copy both require a live context)");
    match catch_cuda(|| CudaContext::new(0)) {
        Ok(Ok(_ctx)) => println!("  succeeded: a real GPU and driver are present."),
        Ok(Err(e)) => println!("  returned a driver error: {:?}", e),
        Err(msg) => {
            println!("  panicked while loading the CUDA driver library:");
            println!("  {}", msg);
        }
    }
    println!("  a device allocation of 4KB and a device-to-host copy both require the context this");
    println!("  sandbox cannot obtain -- neither can run without one, the same early exit every other");
    println!("  file in this chapter and Chapter 18 hit.\n");

    // --- Device property queries ---
    println!("--- device property queries, genuinely called ---");
    match catch_cuda(CudaContext::device_count) {
        Ok(Ok(n)) => println!("CudaContext::device_count() -> {} device(s)", n),
        Ok(Err(e)) => println!("CudaContext::device_count() returned a driver error: {:?}", e),
        Err(msg) => {
            println!("CudaContext::device_count() panicked while loading the CUDA driver library:");
            println!("  {}", msg);
        }
    }
    println!(
        "\n(on real hardware this would report a nonzero device count, and calling .attribute() on a\n\
         live CudaContext would report multiprocessor count, warp size [always 32 on every CUDA\n\
         generation to date], and max threads per block -- exactly the numbers Section 18.1's launch-\n\
         configuration math and Section 18.4's warp-shuffle reasoning both assumed. Unlike CUDA\n\
         Runtime's cudaGetDeviceProperties, which the C++ edition can call without ever creating a\n\
         context first, cudarc's equivalent -- CudaContext::attribute() -- is a method on an already-\n\
         live CudaContext: this sandbox cannot even ATTEMPT that specific call, because the context it\n\
         would need is exactly the thing every earlier call in this chapter has failed to obtain. This is\n\
         a genuine architectural difference between the CUDA Runtime API nvcc programs use by default\n\
         and the CUDA Driver API cudarc wraps, not an omission.)"
    );
}
```

```bash
cargo run --release --bin 08_bandwidth_and_device_query
```

### `Cargo.toml` (shared by all eight binaries above)

```toml
[package]
name = "rust_ch19"
version = "0.1.0"
edition = "2024"

[dependencies]
cudarc = { version = "0.19", default-features = false, features = ["driver", "nvrtc", "std", "dynamic-loading", "cuda-12060"] }
```

Genuinely built with `cargo build --release` and a full `cargo clean && cargo build --release` rebuild from empty, both producing eight binaries with zero warnings.

## Chapter Summary

Vectorized memory access splits a loop into a vectorized main body plus a scalar remainder at the boundary `simd_count = (size / width) * width`, and CUDA's own version of this — a `float4` load moving four `float`s in one 128-bit instruction — makes the same trade-off Chapter 5's SIMD code makes on a CPU: a wider vector load processes more elements per instruction, genuinely confirmed here not by decoding SASS (this sandbox has no CUDA toolkit at all) but by asking `rustc` for `size_of::<[f32; 4]>()` and `size_of::<[f64; 2]>()` directly — both `16` bytes, with the packed element count (`4` for `f32`, only `2` for `f64`) determined entirely by the element type's own size, not by the fixed byte width. Unrolling and fusion solve genuinely different problems that share a similar-sounding name: unrolling cuts loop-control overhead (`4×` fewer increments, comparisons, and branches for `UNROLL=4`, genuinely counted at `1000` events down to `250` for a `1000`-element input) without touching the arithmetic instruction count at all, while fusion cuts memory traffic directly — `8` tensor-sized memory operations collapsed to `4` for `relu(a*b+c)`, genuinely counted and confirmed to produce identical output — at the cost of losing Chapter 16's free backward-rule composition through the `OpRegistry` and requiring one hand-derived gradient for the whole fused expression instead. Compile-time specialization is the concrete mechanism behind this book's zero-cost-abstraction claim: Rust's `const` generics genuinely compile `compile_time_specialized_dot::<N>` to a separate, independently-optimized function body for every distinct `N` a program instantiates — confirmed here by reading `nm`'s actual raw symbol table and finding two distinct linked addresses, distinguished by rustc's own per-instantiation hash suffix, not by trusting the claim — eliminating the *runtime* cost of genericity entirely while quietly relocating that cost to compile time and binary size, one compiled body per instantiation, exactly the trade-off both C++ templates and Rust's const generics make. None of this is worth doing without a benchmarking harness built around warmup runs (so first-call cache-cold overhead doesn't dominate the measurement) and, for GPU work specifically, a mechanism that brackets the timed region against a host that queues work and races ahead — `cudarc`'s driver-API `CudaContext::synchronize()` needs an already-live context to call at all, an honest architectural difference from CUDA Runtime's context-free `cudaDeviceSynchronize()` that this chapter documents rather than glosses over — and this chapter closes by genuinely measuring Chapter 18's basic and padded convolution host mirrors with real `std::time::Instant` timing, converting those real milliseconds into real GFLOPS figures (`21.1257`/`14.9688` GFLOPS at `size=64`, `20.5058`/`19.4955` at `size=128`) that differ substantially in absolute terms from the CUDA edition's own C++ figures — a genuine, expected consequence of measuring two different compilers on two different machines — while landing on the *identical* padding-overhead percentages (`6.6%` and `3.2%`) both editions compute from the same pure FLOP-counting arithmetic.

## Self-Check Questions

1. For `size = 33` with a vector width of `4`, compute `simd_count`, the number of vector instructions, and the number of scalar remainder instructions.
2. `unrolled_add` with `UNROLL = 4` is run on `size = 4,002`. How many full unrolled iterations run, how many elements are left to the scalar remainder loop, and how many total addition operations are performed — does that total differ from what a plain, non-unrolled scalar loop would perform on the same input?
3. A team fuses Worked Example 19.2.2's own `relu(a*b+c)` expression into a single kernel for forward-pass speed, then wires it into the computation graph under the existing registered name `"mul"` so it reuses `MulOp`'s already-derived backward rule (`grad_a = b * grad_output`, `grad_b = a * grad_output`). What goes wrong when this graph is differentiated, and why doesn't a shape check catch it?
4. `compile_time_specialized_dot` is instantiated at `N=4`, `N=8`, and `N=16` across a program's source. How many separate compiled function bodies does this produce, and what is the actual resource cost being paid in exchange for zero runtime branching?
5. A benchmark measures a GPU kernel's `time_function` result *without* either synchronize call bracketing the timed region. What specifically is being measured instead of the kernel's true execution time, and in which direction (too fast or too slow) would you expect the reported time to be biased?
6. Worked Example 19.1.3 found that `size_of::<[f32; 4]>()` and `size_of::<[f64; 2]>()` are both genuinely `16` bytes. If a fourth array type used a `32`-byte payload instead (twice as wide), how many `f64` elements would fit, and how many `i32` elements would fit in that same `32`-byte payload?

## Where We Go Next

Part 5 has made the framework's tensor and GPU-kernel layers fast, in the same sense Chapter 18 made them correct: with numbers traced by hand, genuinely compiled, and — for this chapter's benchmarking section specifically — genuinely timed, rather than asserted or left hypothetical. Part 6 spends every primitive built through both parts — tensors, the autograd engine, and now performance-tuned kernels — assembling something that actually learns: a multi-layer neural network trained end to end on this same codebase.

## Worked Solutions

**1.** `simd_count = (33 / 4) * 4 = 8 * 4 = 32`. The main loop covers elements `0`-`31` in `8` vector instructions (each covering `4` elements). One element, index `32`, falls to the scalar remainder — `1` scalar instruction. Total: `8` vector instructions plus `1` scalar instruction. (Independently verified: `(33/4)*4 = 32`, `32/4 = 8`, `33-32 = 1`.)

**2.** `unrolled_count = (4,002 / 4) * 4 = 1,000 * 4 = 4,000`. That's `1,000` full unrolled iterations (each doing `4` additions), covering elements `0` through `3,999`. The remaining `2` elements (`4,000` and `4,001`) run through the scalar remainder loop one at a time. Total additions performed: `4,000 + 2 = 4,002` — identical to what a plain scalar loop over the same `4,002` elements would compute; unrolling changes only how many times the loop's own increment/compare/branch overhead runs (`1,000 + 2 = 1,002` times here, versus `4,002` times for a non-unrolled loop), not the arithmetic total — exactly the figures file `03`'s own genuine run for `size=4002` already printed.

**3.** `MulOp::backward` computes `grad_a = b * grad_output` and `grad_b = a * grad_output` — the correct gradient for a *plain* multiplication, with no `+ c` and no ReLU's own local derivative anywhere in that formula. Run against a node whose forward pass actually computed `relu(a*b+c)`, this backward rule returns a gradient that completely ignores both the `+ c` term and ReLU's zeroing of the gradient for any element where `a*b+c <= 0` — a `ScalarTensor` of exactly the right shape (since `MulOp::backward`'s output shapes only depend on `a` and `b`'s shapes, which are unchanged by fusing in `+c` and ReLU), just numerically wrong by a large, silent margin for any element where the ignored terms actually mattered. A shape check can't catch this because the bug is entirely about *which formula* ran, not about the shapes those formulas produce — `b * grad_output` and `a * grad_output` have exactly the shapes of `a` and `b` respectively, whether or not a `+c` or a ReLU happened in between.

**4.** Three separate compiled function bodies — one each for `N=4`, `N=8`, and `N=16` — because each distinct const-generic value produces its own independently-generated machine code (confirmed in this chapter's own Worked Example 19.3.1 by reading two such symbols directly out of a linked binary's raw, undemangled symbol table), not one shared body with a runtime branch on `N`. The resource cost being paid is compile time (three function bodies to generate and optimize instead of one) and binary size (three compiled bodies shipped in the final program instead of one) — the runtime cost of choosing between them is what's eliminated, not cost altogether.

**5.** Without the first synchronize call, warmup work still queued on the device can bleed into the timed region, and without the second, the host's clock can stop before the device has actually finished the last of the `benchmark_runs` launches — in both cases, the timer is measuring how long the *host* took to issue `benchmark_runs` kernel launches, not how long the *device* took to execute them. Since launching work asynchronously is typically far faster than actually running it, the reported time would be biased too fast — understating the kernel's true execution time, potentially by a large margin if the device queue backs up well past when the host's loop finishes. This is exactly the failure mode Worked Example 19.4.1 honestly notes this no-GPU sandbox cannot reproduce: every driver-API call here fails before any queueing could even occur, so there's no host-races-ahead-of-device race to observe here, only to reason about from the mechanism.

**6.** `size_of::<[f64; 4]>()` genuinely compiles to `32` bytes, so `4` `f64` elements fit in a `32`-byte payload. `size_of::<[i32; 8]>()` genuinely compiles to `32` bytes as well, so `8` `i32` elements fit in the identical byte budget — exactly the same reasoning Worked Example 19.1.3 applied to the `16`-byte payloads (`4` `f32`s or `2` `f64`s): the payload's byte width is fixed by the container, and the element count it holds is always `(payload bytes) / sizeof(element type)`, independent of what that byte width happens to be.
