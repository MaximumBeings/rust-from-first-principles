# Chapter 12: Element-wise Operations — One Thread, One Position, No Dependencies

> "Chapter 4's whole argument was about how efficiently a warp's threads fetch the cargo they need to compute something. Element-wise operations are the simplest possible thing to do once that cargo has arrived: one thread, one output position, and nothing computed by any other thread that this one needs to wait for. It's almost too simple to dwell on — until the one place in this chapter where the launch configuration and the kernel body quietly stop agreeing with each other, verified here not by reading compiled assembly (this sandbox has no CUDA toolchain to produce any) but by actually running both formulas across a simulated grid and counting, position by position, what got written and what didn't."

**What you will understand by the end of this chapter:**

- `vector_add_kernel`'s indexing formula — `thread_idx` alone — and precisely what breaks if you scale its launch configuration up to a production-sized vector without also updating that formula to match what every later kernel in this chapter actually uses; confirmed not by argument but by genuinely executing both formulas across a simulated multi-block grid and directly counting which output positions get written how many times
- Local derivatives for `+`, `-`, `×`, `÷`, `pow`, and `exp`, each checked against a real finite-difference nudge on the chapter's own numbers, as a first, concrete preview of the chain rule
- Why `exp` gets a dedicated kernel with a self-referencing derivative (`d/dx[eˣ] = eˣ`) that a later backward pass can exploit by reusing the cached forward output instead of recomputing anything
- The stride-0 broadcasting trick from Chapter 7.4, now applied literally inside a kernel-shaped function body rather than as tensor metadata — and exactly which broadcasting shapes `broadcast_add_kernel` handles versus the fully general rule Chapter 7.4 built

**What you need to know first:**

- Chapter 4 and Chapter 10 (the host/device split, and why this sandbox has no device to launch a real kernel on — every "kernel" in this chapter is a plain Rust function receiving its grid coordinates as ordinary arguments, run through a small host-side launch simulator rather than a real GPU scheduler)
- Chapter 7.4 (broadcasting via zero-stride dimensions, computed as pure shape logic — this chapter puts that exact mechanism to work inside a kernel-shaped function for the first time)
- Chapter 9's habit of testing a claim by running code rather than arguing it in prose — Section 12.1 applies that same habit to a question the CUDA edition answers with compiled SASS and this edition answers with genuine, counted, runtime coverage instead

## 12.1 Addition and Subtraction: One Thread Per Output Position `[FOUNDATIONAL]`

### Intuition

Element-wise addition is the purest case of "no thread needs anything any other thread computed": `output[i] = a[i] + b[i]` for every `i`, independently, all at once. It's worth treating as the baseline every other kernel in this chapter (and the chain-rule machinery several chapters from now) gets compared against.

### Background

```rust
const SIZE: usize = 4;

// output[i] = a[i] + b[i] for all i -- one thread per element.
fn vector_add_kernel_naive(output: &mut [f32], a: &[f32], b: &[f32], _block_idx: usize, thread_idx: usize, _threads_per_block: usize) {
    let i = thread_idx;
    if i < SIZE {
        output[i] = a[i] + b[i];
    }
}
```

A real GPU kernel receives `blockIdx`/`threadIdx` directly from the hardware scheduler launching it. This sandbox has no CUDA toolchain at all — no `nvcc`, no `cuobjdump`, confirmed absent by checking directly rather than assumed — so every kernel-shaped function in this chapter takes its coordinates as ordinary parameters instead, supplied by a small `simulate_launch` helper that stands in for the hardware grid: it calls the given closure once per `(block_idx, thread_idx)` pair across a configurable number of simulated blocks and threads-per-block, the same nested-loop shape a real launch configuration describes.

### Worked Example 12.1.1 — Four threads, four sums

`a = [1, 2, 3, 4]`, `b = [10, 20, 30, 40]`. With one simulated block of `SIZE = 4` threads, thread `0` computes `output[0] = a[0] + b[0] = 1 + 10 = 11`; thread `1` computes `2 + 20 = 22`; thread `2` computes `3 + 30 = 33`; thread `3` computes `4 + 40 = 44` — nothing here depends on execution order, since none of the four calls reads anything another one writes.

Compiled and run exactly as described:

```bash
rustc --edition 2024 -O 01_vector_add_and_indexing_bug.rs -o 01_vector_add_and_indexing_bug
./01_vector_add_and_indexing_bug
```

Genuinely compiled and run:

```
output (a + b): [11.0, 22.0, 33.0, 44.0]
```

matching the hand trace exactly.

### Worked Example 12.1.2 — Subtraction, same launch, flipped operator

The same launch configuration with `output[i] = b[i] - a[i]` on the same vectors gives `[10-1, 20-2, 30-3, 40-4] = [9, 18, 27, 36]` — no new kernel structure, just a different one-character arithmetic operator inside the identical per-thread body, genuinely computed alongside the addition above in the same run:

```
output (b - a): [9.0, 18.0, 27.0, 36.0]
```

### Worked Example 12.1.3 — Scaling the launch, without scaling the kernel

For a production-sized, one-million-element vector, the natural move is to retile: pick `threads_per_block = 256` and compute `blocks = (1,000,000 + 255) / 256 = 3907` blocks, with an `if i < size` bounds check protecting the tail block (covering global indices `999,936` through `1,000,191`, of which only `64` fall inside real data). That reasoning is correct — *for a kernel that computes its index as* `block_idx * threads_per_block + thread_idx`, the pattern every kernel from Section 12.2 onward in this chapter actually uses.

### Worked Example 12.1.4 — The bug, confirmed by genuinely counting writes across a simulated grid

Rather than just describe what `vector_add_kernel_naive`'s missing `block_idx` term implies, this section actually launches both `vector_add_kernel_naive`'s indexing formula and the general `block_idx * threads_per_block + thread_idx` formula across an identical simulated grid — `3` blocks of `4` threads each, for a `12`-element output — and counts, for every output position, how many times some thread wrote to it. The CUDA edition answers this same question by disassembling compiled SASS and confirming which hardware registers the kernel reads; that tool doesn't exist in this sandbox, so this edition answers it by direct execution instead, which if anything demonstrates the *consequence* more concretely: not just that the naive formula ignores block identity, but exactly which output positions that leaves silently wrong.

```bash
rustc --edition 2024 -O 01_vector_add_and_indexing_bug.rs -o 01_vector_add_and_indexing_bug
./01_vector_add_and_indexing_bug
```

Genuinely compiled and run:

```
--- Worked Example 12.1.4: 3 blocks of 4 threads each, SIZE=12 ---
naive (thread_idx alone) write counts per position:   [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]
general (block*tpb+thread) write counts per position: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
naive: positions never written by any thread: [4, 5, 6, 7, 8, 9, 10, 11]
naive: positions written by more than one thread (redundant/racing): [0, 1, 2, 3]
general: every position written by exactly one thread? true
```

The naive formula's write counts spell out the bug directly: positions `0` through `3` are each written *three times* — once by every one of the three simulated blocks, since `thread_idx` alone never depends on which block is running — and positions `4` through `11`, two-thirds of the output buffer, are never written at all. The general formula's counts are uniformly `1`: every output position gets written by exactly one thread, once, regardless of how many blocks the launch uses. A final run of `vector_add_kernel_general` across that same `3×4` grid, on real (not just zero/one) input data, confirms the formula also produces the *correct* sums, not just full coverage:

```
--- vector_add_kernel_general, actually launched across the same 3x4 grid ---
a12 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
b12 = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0]
output (general formula, 3 blocks of 4 threads) = [0.0, 101.0, 202.0, 303.0, 404.0, 505.0, 606.0, 707.0, 808.0, 909.0, 1010.0, 1111.0]
```

Every position matches `a12[i] + b12[i]` by hand.

> `[COMMON TRAP]` Retile the launch to `3907` blocks of `256` threads each, as Worked Example 12.1.3 describes, *without* also rewriting the kernel body to the combined formula, and every one of those `3907` blocks still computes `i` as `thread_idx` alone — a value in `[0, 255]` in every single block, regardless of which block it is. All `3907` blocks would redundantly (and, on real hardware, racily) write to `output[0]` through `output[255]` only; the other `999,744` positions in the buffer are never touched by any thread, in any block, at any point. The bounds check `if i < size` doesn't catch this either, since `i` never grows large enough to trip it — the bug is that `i` never reflects which block is running at all. Every kernel from `elementwise_mul_kernel` onward in this chapter uses the combined formula correctly; `vector_add_kernel_naive` is the one exception, and Worked Example 12.1.1 and 12.1.2 only got away with it because they never actually launch more than one simulated block.

**Local derivative:** `∂(a+b)/∂a = 1` and `∂(a+b)/∂b = 1` — a change to either input passes straight through to the output, unscaled.

## 12.2 Multiplication and Division: The Same Pattern, Sharper Arithmetic `[FOUNDATIONAL]`

### Intuition

Multiplication and division run the identical one-thread-per-element pattern Section 12.1 established — the only thing that changes is which operator sits inside the `if i < size` body, and, this time, the indexing formula that actually generalizes past a single block.

### Background

```rust
fn elementwise_mul_kernel(output: &mut [f32], a: &[f32], b: &[f32], size: usize, block_idx: usize, thread_idx: usize, threads_per_block: usize) {
    let i = block_idx * threads_per_block + thread_idx;
    if i < size {
        output[i] = a[i] * b[i];
    }
}

fn elementwise_div_kernel(output: &mut [f32], a: &[f32], b: &[f32], size: usize, block_idx: usize, thread_idx: usize, threads_per_block: usize) {
    let i = block_idx * threads_per_block + thread_idx;
    if i < size {
        // Division by zero produces IEEE-754 inf/NaN rather than a panic --
        // f32 division, unlike integer division, never traps -- a later
        // chapter's autograd engine checks for both before accepting a
        // gradient into an optimizer step.
        output[i] = a[i] / b[i];
    }
}
```

Both kernel-shaped functions compute `i` as `block_idx * threads_per_block + thread_idx` — the general form Worked Example 12.1.4's write-count evidence showed `vector_add_kernel_naive` is missing, and the form that correctly identifies a thread's global position across however many blocks a launch actually uses.

### Worked Example 12.2.1 — Forward values and a finite-difference check

`a = [2, 3, 4]`, `b = [5, 6, 7]`. Compiled and run:

```bash
rustc --edition 2024 -O 02_mul_div_kernels.rs -o 02_mul_div_kernels
./02_mul_div_kernels
```

Genuinely compiled and run:

```
multiply: [10.0, 18.0, 28.0]
divide:   [0.400000, 0.500000, 0.571429]

multiply finite diff at a=2,b=5: c=10.0000, c(a+0.01)=10.0500, slope=5.0000 (expect b=5)
```

Element-wise multiply gives `[10, 18, 28]`; element-wise divide gives `[0.4, 0.5, 0.571...]`, matching `2×5`, `3×6`, `4×7` and `2/5`, `3/6`, `4/7` by hand. Checking the multiplication derivative numerically at position `0` (`a=2, b=5, c=10`): nudging `a` from `2` to `2.01` moves `c` from `10` to `2.01 × 5 = 10.05`, a change of `0.05` over a nudge of `0.01` — a slope of `5`, matching `∂c/∂a = b = 5` exactly, since `c = a × b`'s partial derivative with respect to `a` is just `b`.

### Worked Example 12.2.2 — Division's two derivatives

For `c = a / b`: `∂c/∂a = 1/b` and `∂c/∂b = -a/b²`. At position `0` (`a=2, b=5`), genuinely computed in the same run:

```
division derivatives at a=2,b=5: dc/da=0.2000 (expect 0.2), dc/db=-0.0800 (expect -0.08)
```

`∂c/∂a = 1/5 = 0.2` and `∂c/∂b = -2/25 = -0.08`. Both backward kernels a later chapter builds from these two rules are one more elementwise pass over the same-shaped buffers — no new infrastructure beyond what Section 12.1 already established, just a kernel body multiplying by whichever of these local derivatives applies to that operand.

## 12.3 Power and Exponential Functions: Dedicated Kernels for the Two That Recur Everywhere `[FOUNDATIONAL]`

### Intuition

`pow` and `exp` show up in essentially every loss function and activation function later in this book, which is why they get their own dedicated kernel-shaped functions here rather than being routed through a generic "apply this scalar function" abstraction — on real hardware, a dedicated kernel lets the compiler specialize and inline the underlying `powf`/`expf` intrinsic directly, instead of dispatching through a function pointer or a branch on which operation was requested; in Rust, `f32::powf` and `f32::exp` are the equivalent intrinsics, and each gets its own function here for the same reason.

### Background

```rust
fn elementwise_pow_kernel(output: &mut [f32], base: &[f32], exponent: f32, size: usize, block_idx: usize, thread_idx: usize, threads_per_block: usize) {
    let i = block_idx * threads_per_block + thread_idx;
    if i < size {
        output[i] = base[i].powf(exponent);
    }
}

fn elementwise_exp_kernel(output: &mut [f32], input: &[f32], size: usize, block_idx: usize, thread_idx: usize, threads_per_block: usize) {
    let i = block_idx * threads_per_block + thread_idx;
    if i < size {
        output[i] = input[i].exp();
    }
}
```

### Worked Example 12.3.1 — Squaring three values, and checking the derivative by hand

`x = [1, 2, 3]`. Compiled and run:

```bash
rustc --edition 2024 -O 03_pow_exp_kernels.rs -o 03_pow_exp_kernels
./03_pow_exp_kernels
```

Genuinely compiled and run:

```
pow(x, 2): [1.0, 4.0, 9.0]
exp(x):    [2.71828, 7.38906, 20.08554]

pow finite diff at x=2,n=2: pow(2,2)=4.0000, pow(2.01,2)=4.0401, slope=4.0100 (expect n*x^(n-1)=4)
```

`pow(x, 2)` gives `[1, 4, 9]`. Checking the derivative at `x=2`: `d/dx[xⁿ] = n·x^(n-1)`, so with `n=2` that's `2×2 = 4`. The finite-difference check confirms it: `pow(2.01, 2) = 4.0401`, and `(4.0401 - 4.0) / 0.01 = 4.01 ≈ 4` — the small residual above `4` is exactly the expected error of a finite forward-difference approximation, not a discrepancy in the calculus.

### Worked Example 12.3.2 — The derivative that is the function

`exp(x)` on the same input gives `[e¹, e², e³] ≈ [2.71828, 7.38906, 20.08554]`, genuinely computed above. Its derivative, `d/dx[eˣ] = eˣ`, is the function itself:

```
exp derivative at x=2: d/dx[e^x] = e^x = 7.38906, same as forward value exp_out[1]=7.38906
```

at `x=2` that's `e² ≈ 7.38906`, exactly the forward value already computed for that position, not a separately-derived number. This self-referencing property is exactly why a later chapter's backward pass for `exp` can read the *cached* forward output during the backward pass rather than recomputing `exp(x)` a second time — the forward pass already computed the one number the backward pass needs.

## 12.4 Broadcasting: The Stride-0 Trick, Now Inside a Kernel-Shaped Function `[FOUNDATIONAL]`

### Intuition

Every operation so far in this chapter assumed both inputs were the same shape. Broadcasting is what lets a `[2,3]` tensor add to a `[1,3]` tensor anyway, by treating the smaller tensor's missing dimension as if it were silently repeated — and Chapter 7.4 already built the shape-level machinery for deciding *when* that's legal and what the resulting strides should be. This section is where that machinery gets consumed: not as tensor metadata computed once ahead of time, but as a literal stride value read by every single simulated thread, every time it computes its own input address.

### Background

```
A = 1  2  3        B = 10  20  30
    4  5  6
```

```rust
fn broadcast_add_kernel(
    output: &mut [f32],
    a: &[f32],
    b: &[f32],
    a_stride_row: usize,   // 0 if this operand is broadcast along rows
    b_stride_row: usize,   // 0 if this operand is broadcast along rows
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
) {
    if row < rows && col < cols {
        let a_val = a[row * a_stride_row + col];
        let b_val = b[row * b_stride_row + col];
        output[row * cols + col] = a_val + b_val;
    }
}
```

Setting a tensor's stride to `0` along a dimension — Chapter 7.4's broadcast-stride computation producing exactly this value ahead of the kernel launch — means "don't advance the memory address at all as this dimension's index increases," which is precisely "keep re-reading the same values." This function is a deliberately narrower instrument than Chapter 7.4's general rule, though: it only ever broadcasts along the row dimension, via one `_stride_row` parameter per operand, rather than handling an arbitrary number of dimensions each independently aligned from the right the way the general rule does. It's the 2-D specialization of that rule, not a reimplementation of the whole thing.

### Worked Example 12.4.1 — Tracing `b_stride_row = 0`

`A` is `2×3`, `B` is a single row of 3 values meant to add to every row of `A`. Compiled and run:

```bash
rustc --edition 2024 -O 04_broadcast_add_kernel.rs -o 04_broadcast_add_kernel
./04_broadcast_add_kernel
```

Genuinely compiled and run:

```
b_stride_row = 0 (B broadcast down every row of A):
output = [[11, 22, 33], [14, 25, 36]]
```

With `b_stride_row = 0`: for row `0`, the function reads `b[0×0 + col] = b[col]`; for row `1`, it reads `b[1×0 + col] = b[col]` — the *same* address both times, `B`'s one and only row, read twice. `A`'s stride is unmodified (a normal, non-zero row stride), so each row of `A` still reads its own distinct data. The result, `[[11, 22, 33], [14, 25, 36]]`, matches `[1+10, 2+20, 3+30]` and `[4+10, 5+20, 6+30]` by hand, with `B` never copied — no extra memory traffic beyond what a same-shaped add would already cost, which is the entire point of doing broadcasting at the stride level instead of the copy level.

### Worked Example 12.4.2 — The symmetric case, also genuinely traced

`a_stride_row = 0` (with `b_stride_row` left at its normal nonzero value) produces the mirror image: every row of the output grid re-reads row `0` of `A` while `B` supplies a genuinely different row each time. Traced with `A = [1, 2, 3]` (one row, broadcast) against `B = [[10, 20, 30], [40, 50, 60]]`, genuinely computed in the same run:

```
a_stride_row = 0 (A broadcast down every row of B):
output = [[11, 22, 33], [41, 52, 63]]
```

Row `0` reads `A`'s only row plus `B`'s row `0`: `[1+10, 2+20, 3+30] = [11, 22, 33]`. Row `1` re-reads that *same* row of `A` — `a[1×0 + col] = a[col]`, identical to row `0`'s read — plus `B`'s genuinely different row `1`: `[1+40, 2+50, 3+60] = [41, 52, 63]`. Nothing about the function body needs to know in advance which operand is the one being broadcast; both `a_stride_row` and `b_stride_row` are ordinary parameters, and whichever one the caller sets to `0` is the one that gets silently repeated.

## 12.5 Complete Runnable Code

### File: `01_vector_add_and_indexing_bug.rs`

```rust
const SIZE: usize = 4;

// Section 12.1's kernel exactly as written: valid only for a single block.
// A real GPU kernel would receive block_idx/thread_idx from the hardware
// directly (SR_CTAID.X / SR_TID.X in CUDA terms); this sandbox has no CUDA
// toolchain at all (no nvcc, no cuobjdump -- confirmed absent, unlike
// Chapter 4/10's device-driver-only gap), so every "kernel" in this chapter
// is written as a plain function that receives its coordinates as ordinary
// arguments, and `simulate_launch` below stands in for the hardware grid
// that would normally supply them.
fn vector_add_kernel_naive(output: &mut [f32], a: &[f32], b: &[f32], _block_idx: usize, thread_idx: usize, _threads_per_block: usize) {
    let i = thread_idx;
    if i < SIZE {
        output[i] = a[i] + b[i];
    }
}

// The general form every later kernel in this chapter actually uses.
fn vector_add_kernel_general(output: &mut [f32], a: &[f32], b: &[f32], size: usize, block_idx: usize, thread_idx: usize, threads_per_block: usize) {
    let i = block_idx * threads_per_block + thread_idx;
    if i < size {
        output[i] = a[i] + b[i];
    }
}

fn vector_sub_kernel_naive(output: &mut [f32], a: &[f32], b: &[f32], _block_idx: usize, thread_idx: usize, _threads_per_block: usize) {
    let i = thread_idx;
    if i < SIZE {
        output[i] = b[i] - a[i];
    }
}

// Runs `thread_fn` once per (block_idx, thread_idx) pair across a simulated
// grid of `blocks` blocks of `threads_per_block` threads each -- the host
// stand-in, in this sandbox, for a hardware kernel launch.
fn simulate_launch(blocks: usize, threads_per_block: usize, mut thread_fn: impl FnMut(usize, usize, usize)) {
    for block_idx in 0..blocks {
        for thread_idx in 0..threads_per_block {
            thread_fn(block_idx, thread_idx, threads_per_block);
        }
    }
}

// Runs a simulated launch and records, for every output position, how many
// times some thread wrote to it -- 0 means "never touched by any thread",
// more than 1 means "written redundantly by more than one thread".
fn count_writes(blocks: usize, threads_per_block: usize, size: usize, indexing: impl Fn(usize, usize, usize) -> usize) -> Vec<u32> {
    let mut write_counts = vec![0u32; size];
    simulate_launch(blocks, threads_per_block, |block_idx, thread_idx, tpb| {
        let i = indexing(block_idx, thread_idx, tpb);
        if i < size {
            write_counts[i] += 1;
        }
    });
    write_counts
}

fn main() {
    println!("=== Section 12.1: vector_add_kernel, both indexing formulas, genuinely executed ===");
    println!("(this sandbox has no CUDA toolchain -- no nvcc, no cuobjdump -- so unlike the CUDA");
    println!(" edition's SASS disassembly, the indexing bug below is demonstrated by actually");
    println!(" running both formulas across a simulated multi-block grid and counting writes)\n");

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let mut out_add = [0.0f32; SIZE];
    let mut out_sub = [0.0f32; SIZE];

    simulate_launch(1, SIZE, |block_idx, thread_idx, tpb| {
        vector_add_kernel_naive(&mut out_add, &a, &b, block_idx, thread_idx, tpb);
    });
    simulate_launch(1, SIZE, |block_idx, thread_idx, tpb| {
        vector_sub_kernel_naive(&mut out_sub, &a, &b, block_idx, thread_idx, tpb);
    });

    println!("output (a + b): [{:.1}, {:.1}, {:.1}, {:.1}]", out_add[0], out_add[1], out_add[2], out_add[3]);
    println!("output (b - a): [{:.1}, {:.1}, {:.1}, {:.1}]", out_sub[0], out_sub[1], out_sub[2], out_sub[3]);

    println!("\n--- Worked Example 12.1.4: 3 blocks of 4 threads each, SIZE=12 ---");
    let blocks = 3;
    let threads_per_block = 4;
    let size = 12;

    let naive_counts = count_writes(blocks, threads_per_block, size, |_block_idx, thread_idx, _tpb| thread_idx);
    let general_counts = count_writes(blocks, threads_per_block, size, |block_idx, thread_idx, tpb| block_idx * tpb + thread_idx);

    println!("naive (thread_idx alone) write counts per position:   {:?}", naive_counts);
    println!("general (block*tpb+thread) write counts per position: {:?}", general_counts);

    let naive_untouched: Vec<usize> = naive_counts.iter().enumerate().filter(|&(_, &c)| c == 0).map(|(i, _)| i).collect();
    let naive_redundant: Vec<usize> = naive_counts.iter().enumerate().filter(|&(_, &c)| c > 1).map(|(i, _)| i).collect();
    let general_all_once = general_counts.iter().all(|&c| c == 1);

    println!("naive: positions never written by any thread: {:?}", naive_untouched);
    println!("naive: positions written by more than one thread (redundant/racing): {:?}", naive_redundant);
    println!("general: every position written by exactly one thread? {}", general_all_once);

    println!("\n--- vector_add_kernel_general, actually launched across the same 3x4 grid ---");
    let a12: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let b12: Vec<f32> = (0..12).map(|i| (i * 100) as f32).collect();
    let mut out12 = vec![0.0f32; 12];
    simulate_launch(blocks, threads_per_block, |block_idx, thread_idx, tpb| {
        vector_add_kernel_general(&mut out12, &a12, &b12, size, block_idx, thread_idx, tpb);
    });
    println!("a12 = {:?}", a12);
    println!("b12 = {:?}", b12);
    println!("output (general formula, 3 blocks of 4 threads) = {:?}", out12);
}
```

```bash
rustc --edition 2024 -O 01_vector_add_and_indexing_bug.rs -o 01_vector_add_and_indexing_bug
./01_vector_add_and_indexing_bug
```

### File: `02_mul_div_kernels.rs`

```rust
fn elementwise_mul_kernel(output: &mut [f32], a: &[f32], b: &[f32], size: usize, block_idx: usize, thread_idx: usize, threads_per_block: usize) {
    let i = block_idx * threads_per_block + thread_idx;
    if i < size {
        output[i] = a[i] * b[i];
    }
}

fn elementwise_div_kernel(output: &mut [f32], a: &[f32], b: &[f32], size: usize, block_idx: usize, thread_idx: usize, threads_per_block: usize) {
    let i = block_idx * threads_per_block + thread_idx;
    if i < size {
        // Division by zero produces IEEE-754 inf/NaN rather than a panic --
        // f32 division, unlike integer division, never traps -- a later
        // chapter's autograd engine checks for both before accepting a
        // gradient into an optimizer step.
        output[i] = a[i] / b[i];
    }
}

fn simulate_launch(blocks: usize, threads_per_block: usize, mut thread_fn: impl FnMut(usize, usize, usize)) {
    for block_idx in 0..blocks {
        for thread_idx in 0..threads_per_block {
            thread_fn(block_idx, thread_idx, threads_per_block);
        }
    }
}

fn main() {
    println!("=== Section 12.2: multiplication and division, forward values and finite differences ===\n");

    let a = [2.0f32, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0];
    let size = 3;
    let mut mul_out = [0.0f32; 3];
    let mut div_out = [0.0f32; 3];

    simulate_launch(1, size, |block_idx, thread_idx, tpb| {
        elementwise_mul_kernel(&mut mul_out, &a, &b, size, block_idx, thread_idx, tpb);
    });
    simulate_launch(1, size, |block_idx, thread_idx, tpb| {
        elementwise_div_kernel(&mut div_out, &a, &b, size, block_idx, thread_idx, tpb);
    });

    println!("multiply: [{:.1}, {:.1}, {:.1}]", mul_out[0], mul_out[1], mul_out[2]);
    println!("divide:   [{:.6}, {:.6}, {:.6}]", div_out[0], div_out[1], div_out[2]);

    // Finite-difference check of d(a*b)/da at position 0 (a=2, b=5)
    let c0 = a[0] * b[0];
    let c0_nudged = 2.01f32 * b[0];
    println!(
        "\nmultiply finite diff at a=2,b=5: c={:.4}, c(a+0.01)={:.4}, slope={:.4} (expect b=5)",
        c0,
        c0_nudged,
        (c0_nudged - c0) / 0.01f32
    );

    // Division derivatives at a=2, b=5
    let d_da = 1.0f32 / b[0];
    let d_db = -a[0] / (b[0] * b[0]);
    println!(
        "division derivatives at a=2,b=5: dc/da={:.4} (expect 0.2), dc/db={:.4} (expect -0.08)",
        d_da, d_db
    );
}
```

```bash
rustc --edition 2024 -O 02_mul_div_kernels.rs -o 02_mul_div_kernels
./02_mul_div_kernels
```

### File: `03_pow_exp_kernels.rs`

```rust
fn elementwise_pow_kernel(output: &mut [f32], base: &[f32], exponent: f32, size: usize, block_idx: usize, thread_idx: usize, threads_per_block: usize) {
    let i = block_idx * threads_per_block + thread_idx;
    if i < size {
        output[i] = base[i].powf(exponent);
    }
}

fn elementwise_exp_kernel(output: &mut [f32], input: &[f32], size: usize, block_idx: usize, thread_idx: usize, threads_per_block: usize) {
    let i = block_idx * threads_per_block + thread_idx;
    if i < size {
        output[i] = input[i].exp();
    }
}

fn simulate_launch(blocks: usize, threads_per_block: usize, mut thread_fn: impl FnMut(usize, usize, usize)) {
    for block_idx in 0..blocks {
        for thread_idx in 0..threads_per_block {
            thread_fn(block_idx, thread_idx, threads_per_block);
        }
    }
}

fn main() {
    println!("=== Section 12.3: pow and exp, dedicated kernels with a derivative check each ===\n");

    let x = [1.0f32, 2.0, 3.0];
    let size = 3;
    let mut pow_out = [0.0f32; 3];
    let mut exp_out = [0.0f32; 3];

    simulate_launch(1, size, |block_idx, thread_idx, tpb| {
        elementwise_pow_kernel(&mut pow_out, &x, 2.0, size, block_idx, thread_idx, tpb);
    });
    simulate_launch(1, size, |block_idx, thread_idx, tpb| {
        elementwise_exp_kernel(&mut exp_out, &x, size, block_idx, thread_idx, tpb);
    });

    println!("pow(x, 2): [{:.1}, {:.1}, {:.1}]", pow_out[0], pow_out[1], pow_out[2]);
    println!("exp(x):    [{:.5}, {:.5}, {:.5}]", exp_out[0], exp_out[1], exp_out[2]);

    // finite difference for pow derivative at x=2, n=2
    let p0 = 2.0f32.powf(2.0);
    let p0_nudged = 2.01f32.powf(2.0);
    println!(
        "\npow finite diff at x=2,n=2: pow(2,2)={:.4}, pow(2.01,2)={:.4}, slope={:.4} (expect n*x^(n-1)=4)",
        p0,
        p0_nudged,
        (p0_nudged - p0) / 0.01f32
    );

    println!(
        "\nexp derivative at x=2: d/dx[e^x] = e^x = {:.5}, same as forward value exp_out[1]={:.5}",
        2.0f32.exp(),
        exp_out[1]
    );
}
```

```bash
rustc --edition 2024 -O 03_pow_exp_kernels.rs -o 03_pow_exp_kernels
./03_pow_exp_kernels
```

### File: `04_broadcast_add_kernel.rs`

```rust
fn broadcast_add_kernel(
    output: &mut [f32],
    a: &[f32],
    b: &[f32],
    a_stride_row: usize,
    b_stride_row: usize,
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
) {
    if row < rows && col < cols {
        let a_val = a[row * a_stride_row + col];
        let b_val = b[row * b_stride_row + col];
        output[row * cols + col] = a_val + b_val;
    }
}

fn simulate_2d_launch(rows: usize, cols: usize, mut thread_fn: impl FnMut(usize, usize)) {
    for row in 0..rows {
        for col in 0..cols {
            thread_fn(row, col);
        }
    }
}

fn main() {
    println!("=== Section 12.4: broadcast_add_kernel, stride-0 traced by hand ===\n");

    // A is 2x3, B is a single row of 3 values broadcast down both rows.
    let a_mat = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // row-major 2x3
    let b_row = [10.0f32, 20.0, 30.0];
    let rows = 2;
    let cols = 3;
    let a_stride_row = 3; // normal stride
    let b_stride_row = 0; // broadcast: never advance

    let mut output = vec![0.0f32; rows * cols];
    simulate_2d_launch(rows, cols, |row, col| {
        broadcast_add_kernel(&mut output, &a_mat, &b_row, a_stride_row, b_stride_row, rows, cols, row, col);
    });
    println!("b_stride_row = 0 (B broadcast down every row of A):");
    println!(
        "output = [[{:.0}, {:.0}, {:.0}], [{:.0}, {:.0}, {:.0}]]",
        output[0], output[1], output[2], output[3], output[4], output[5]
    );

    // Symmetric case: A broadcast, B varies.
    let a_row = [1.0f32, 2.0, 3.0];
    let b_mat = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]; // row-major 2x3
    let a_stride_row2 = 0;
    let b_stride_row2 = 3;

    let mut output2 = vec![0.0f32; rows * cols];
    simulate_2d_launch(rows, cols, |row, col| {
        broadcast_add_kernel(&mut output2, &a_row, &b_mat, a_stride_row2, b_stride_row2, rows, cols, row, col);
    });
    println!("\na_stride_row = 0 (A broadcast down every row of B):");
    println!(
        "output = [[{:.0}, {:.0}, {:.0}], [{:.0}, {:.0}, {:.0}]]",
        output2[0], output2[1], output2[2], output2[3], output2[4], output2[5]
    );
}
```

```bash
rustc --edition 2024 -O 04_broadcast_add_kernel.rs -o 04_broadcast_add_kernel
./04_broadcast_add_kernel
```

## Chapter Summary

Every kernel-shaped function in this chapter shares one structure: one thread, one output position, no dependency on any other thread's result — the property that makes element-wise operations both the highest-frequency operation in a training loop and the easiest to parallelize. `vector_add_kernel_naive` computes that structure correctly for a single block (`thread_idx` alone is a valid global index when there's only one block), but Worked Example 12.1.4 went past argument into genuine execution evidence: with no CUDA toolchain available in this sandbox to disassemble, a small `simulate_launch` grid harness ran both the naive and general indexing formulas across an identical 3-block, 4-thread grid and counted writes directly, confirming that the naive formula leaves two-thirds of a 12-element output buffer completely untouched while redundantly overwriting the same four positions three times each — the literal, measured consequence of a formula that never reads which block it's running in. Multiplication, division, power, and exponential all follow the same one-thread-per-element shape, each paired with a local derivative checked against a real finite-difference nudge on the chapter's own numbers — `∂c/∂a = b` for multiplication, `∂c/∂a = 1/b` and `∂c/∂b = -a/b²` for division, `n·x^(n-1)` for power, and the self-referencing `eˣ` for exponential, whose backward pass can reuse a cached forward value instead of recomputing anything. Broadcasting closed the chapter by taking Chapter 7.4's stride-0 mechanism — computed there as pure shape metadata — and applying it literally inside a kernel-shaped function body for the first time: `b_stride_row = 0` makes every row of the output grid re-read the exact same row of `B`, genuinely traced in both directions (`B` broadcasting over `A`, and the symmetric case of `A` broadcasting over `B`), with zero extra memory traffic and zero copying — though `broadcast_add_kernel` only ever broadcasts along rows, a narrower case than Chapter 7.4's fully general, arbitrary-dimension rule.

## Self-Check Questions

1. `vector_add_kernel_naive` is launched via `simulate_launch` with `blocks = 2` and `threads_per_block = 4` for an 8-element vector (`SIZE` updated to `8`), but the function body is left completely unchanged. Which output positions actually get written, and which ones never get touched by any simulated thread?
2. Using `∂c/∂b = -a/b²` for `c = a/b`, compute the local derivative with respect to `b` at `a=4, b=2`, and verify it with a finite-difference nudge (`b` from `2` to `2.01`) the way Worked Example 12.2.1 checked multiplication.
3. `pow(x, 3)` is applied to `x = 2`. Using `d/dx[xⁿ] = n·x^(n-1)`, what is the exact derivative at that point, and what finite-difference nudge and result would confirm it the way Worked Example 12.3.1 did for `n=2`?
4. `broadcast_add_kernel` is called with `a_stride_row` set to `A`'s normal (nonzero) row stride and `b_stride_row` set to `0`, on a `3×4` `A` and a `1×4` `B`. Write out, for each of the three rows, which literal element of `B` gets added to `A[row][2]`.
5. If you ran `elementwise_mul_kernel` through the same `count_writes` harness Worked Example 12.1.4 used for `vector_add_kernel_general`, across the same 3-block, 4-thread, 12-element grid, what write-count pattern would you expect to see, and why? What specific pattern in that output would tell you, without looking at the source at all, that a kernel had the same bug `vector_add_kernel_naive` has?

## Where We Go Next

Chapter 13 moves from per-element operations to operations that mix elements across a whole row or column — matrix multiplication and transpose both generalize from — the first place in this book where one output position genuinely depends on more than one input position at once.

## Worked Solutions

**1.** With `blocks = 2` and `threads_per_block = 4`, `thread_idx` ranges over `0, 1, 2, 3` in *both* simulated blocks (block index isn't part of the formula at all). Block `0`'s four threads compute `output[0]` through `output[3]`; block `1`'s four threads also compute `i = 0, 1, 2, 3` and write to the exact same `output[0]` through `output[3]` — a redundant rewrite of the same four positions (a genuine data race on real hardware, where both blocks could run concurrently). `output[4]` through `output[7]` are never written by any thread in either block, even though `SIZE = 8` and the bounds check `if i < SIZE` would happily allow indices up to `7` — nothing in the function ever produces an `i` value of `4` or higher, because `thread_idx` alone tops out at `3` regardless of which block is running, exactly as Worked Example 12.1.4's `count_writes` harness demonstrated for the 3-block, 12-element case.

**2.** `∂c/∂b = -a/b²` at `a=4, b=2`: `-4/4 = -1`. Finite-difference check, genuinely computed: `c = a/b = 4/2 = 2.0` at `b=2`; at `b=2.01`, `c = 4/2.01 = 1.990050`. The change in `c` is `1.990050 - 2.0 = -0.009950` over a nudge of `0.01`, giving a slope of `-0.995028 ≈ -1` — matching `∂c/∂b = -1` (the small residual is the expected finite-difference approximation error, the same pattern Worked Example 12.3.1 saw for `pow`).

**3.** `d/dx[x³] = 3x²`; at `x=2`, that's `3×4 = 12`. A finite-difference check nudges `x` from `2` to `2.01`, genuinely computed: `pow(2.01, 3) = 8.120601`, and `pow(2, 3) = 8.0`, so `(8.120601 - 8.0)/0.01 = 12.0601 ≈ 12` — confirming the derivative, with the small residual above `12` again being the expected forward-difference approximation error rather than a discrepancy.

**4.** With `b_stride_row = 0`, every row reads `b[row × 0 + col] = b[col]` — the same single row of `B` regardless of `row`. For column `2` specifically, every one of the three rows (`row = 0`, `row = 1`, `row = 2`) adds the exact same element, `B[0][2]` — the third value in `B`'s one and only row — to `A[row][2]`. Which row of `A` is involved changes; which element of `B` is read does not.

**5.** `elementwise_mul_kernel` computes `i` with the exact same `block_idx * threads_per_block + thread_idx` formula as `vector_add_kernel_general`, so running it through `count_writes` across the identical 3-block, 4-thread, 12-element grid would produce the identical all-`1`s pattern: every one of the 12 output positions written by exactly one thread, once. The tell-tale sign of a `vector_add_kernel_naive`-style bug, visible purely from the write-count output with no source code in view at all, is exactly what Worked Example 12.1.4 found: a block of low positions (as many as `threads_per_block`) each reporting a write count equal to the number of blocks in the launch, paired with every higher position reporting a write count of `0`. That specific shape — a short prefix over-written by exactly `blocks` and a long suffix touched by nobody — can only arise from an indexing formula that caps out at `threads_per_block - 1` no matter how many blocks are actually launched; a correct formula covering the full range would never produce it, on this data or any other.
