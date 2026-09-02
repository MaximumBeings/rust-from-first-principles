# Appendix D: Tensor Contractions, From First Principles (CPU)

> "Every tensor contraction you will ever write is either a matrix multiply in disguise, or a small collection of matrix multiplies stapled together. The disguise is worth removing once, carefully, so that every later appearance is instantly recognizable."

## What you will understand

By the end of this appendix you will be able to:

- State the general definition of a tensor contraction, and recognize matrix multiplication as its simplest non-trivial case.
- Implement a fully general N-dimensional contraction in Rust, from a shape/stride representation up through a working `contract()` function.
- Contract two tensors over more than one shared axis at once, and independently verify the result against `numpy.tensordot`.
- Recognize and guard against the specific silent-corruption trap that an unvalidated axis pair opens up -- and see, by actually building and running the unchecked version, exactly what it does instead of only being told.
- Explain, with genuine wall-clock measurements rather than an assertion, why loop *order* changes a contraction's running time without changing its answer, and why an optimizing compiler can change that story further still.

## What you need to know first

This appendix assumes you are comfortable with the struct and memory-layout material from Chapters 2-3, and with the generic, closure-based function style this book has used since Part 3. Nothing here requires a GPU -- every example in this appendix is ordinary, portable Rust compiled with `cargo` and run on the CPU. The companion appendix, Appendix E, takes the exact same mathematics onto CUDA (to the extent this sandbox's toolchain permits it -- see Appendix B's own notes on that).

## D.1 What Is a Tensor Contraction?

A **tensor** here is nothing more exotic than a struct earlier chapters have already used many times: a flat buffer of numbers, plus a **shape** describing how many axes it has and how large each one is. A vector is a rank-1 tensor (one axis); a matrix is a rank-2 tensor (two axes); a rank-3 tensor is a "stack" of matrices, and so on.

A **contraction** takes two tensors, picks out one or more axes on *each* that have matching sizes, and produces a new tensor by summing the elementwise product of the two inputs over every combination of values those matched axes can take -- while keeping every other axis untouched. The matched axes are called **contracted axes** (or, in the physics convention this appendix borrows a term from, *dummy indices*, because their specific value never appears in the output -- only every value they range over, summed away). The untouched axes are the **free indices**, and they simply concatenate to form the output's shape.

Matrix multiplication is the contraction everyone already knows, they just haven't always seen it named that way. Given `A` with shape `[M, K]` and `B` with shape `[K, N]`:

```
C[i][j] = Σ_k  A[i][k] · B[k][j]
```

`i` and `j` are free indices (they survive into the output, shape `[M, N]`); `k` is the contracted index (it's summed away). Nothing about this definition requires `A` and `B` to be rank-2. If `A` has shape `[2, 3, 4]` and `B` has shape `[3, 4, 5]`, contracting over *both* of `A`'s last two axes against *both* of `B`'s first two axes is exactly as well-defined -- it just sums over a pair of indices instead of one, and the output shape is `[2, 5]` (A's one remaining axis, then B's one remaining axis). Section D.4 builds and verifies exactly this case.

### Formulas and Key Terms

```
General contraction, axis sets S_A (on A) and S_B (on B), |S_A| = |S_B|:

  C[free_A indices, free_B indices]  =  Σ (over all values of the S_A/S_B axes)
                                            A[free_A indices, S_A indices]
                                          · B[S_B indices, free_B indices]

Matrix multiply, the rank-2, single-shared-axis case:

  C[i][j] = Σ_k  A[i][k] · B[k][j]        (A: [M,K], B: [K,N], C: [M,N])
```

- **Rank** — the number of axes a tensor has; a scalar is rank 0, a vector rank 1, a matrix rank 2.
- **Shape** — the size of each axis, e.g. `[2, 3, 4]` for a rank-3 tensor whose axes hold 2, 3, and 4 values respectively.
- **Free index** — an axis that is *not* contracted; it survives into the output tensor's shape unchanged.
- **Contracted index** ("dummy index") — an axis that *is* contracted; it is summed over and does not appear in the output shape at all.
- **Einstein summation convention** — the notational shortcut this appendix's formulas rely on informally: any index letter repeated between two tensors being multiplied together is implicitly summed over. `C[i][j] = A[i][k]·B[k][j]` already means "sum over k" the moment `k` appears on both sides of the product.
- **Output rank** — for a contraction over `p` shared axis pairs, `rank(C) = rank(A) + rank(B) - 2p`. Matrix multiply: `2 + 2 - 2(1) = 2`, a matrix, as expected.

## D.2 Shapes, Strides, and Indexing

Before any contraction logic can be written, there is one piece of bookkeeping every later section depends on: converting between a **multi-index** (one integer per axis) and the single flat integer offset into a tensor's underlying buffer.

This appendix uses **row-major** layout throughout (the last axis varies fastest in memory -- the same convention this book has used since Chapter 2). The **stride** of an axis is how many flat positions a single step along that axis is worth; in row-major order, the stride of the *last* axis is always 1, and each stride moving left is the product of everything to its right.

### Worked Example D.2.1 — Round-tripping every index of a small tensor

`01_tensor_indexing.rs` computes strides for a `[2, 3, 4]` tensor, unravels one specific linear offset by hand-checked arithmetic, and then exhaustively round-trips *every* one of its 24 offsets through unravel-then-ravel to confirm none of them get corrupted:

```
shape = [2, 3, 4]
row-major strides = [12, 4, 1]
total elements = 24

unravel_index(17) = (1, 1, 1)  [hand-computed expected: (1, 1, 1)]
ravel_index(1,1,1) = 17  [must round-trip back to 17]

exhaustive round-trip check over all 24 offsets: 0 failures

All checks passed.
```

Genuinely compiled with `cargo build` (dev profile) and `cargo build --release`, both producing zero warnings, and run; the hand-worked check on linear offset 17 (`17 / 12 = 1` remainder `5`, `5 / 4 = 1` remainder `1`, `1 / 1 = 1` remainder `0`, giving multi-index `(1, 1, 1)`) matches the code's own output exactly, and the exhaustive loop confirms the same holds for every other offset, not just the one checked by hand.

## D.3 A Generic Contraction Function

With indexing settled, `contract()` in `02_generic_contraction.rs` implements the general definition from Section D.1 directly: given two tensors and a list of axes to contract on each, it works out which axes are free, builds the output shape, and fills it in by walking every combination of free indices, and for each one, summing over every combination of contracted indices.

The walk itself uses a small recursive helper, `for_each_index()`, which visits every multi-index of a given shape in row-major order -- one level of recursion per axis. This is a different (and, for a teaching example, more directly readable) technique than the iterative divisor/modulo unraveling of Section D.2; both compute the same thing, and Appendix E.3 deliberately returns to the *iterative* style, because a GPU kernel cannot recurse over an unknown number of axes the way host code can.

**Two genuine, worth-noting Rust-specific design choices** show up in this port, neither of which is a workaround for anything broken -- they are simply how the same idea is expressed idiomatically in this language:

- **No self-referencing closure.** C++'s version of `for_each_index()` captures itself inside a `std::function` closure so the lambda can call itself recursively. Rust's closures cannot straightforwardly reference themselves this way -- a `FnMut` cannot hold a reference to itself without extra indirection (a `RefCell`, or boxing into a trait object with a workaround for the chicken-and-egg borrow). Rather than reach for that machinery, this port simply writes `for_each_index_recurse()` as an ordinary named function taking the current `axis` explicitly as a parameter. Same recursion, same result, no self-reference required.
- **No exceptions.** C++'s `contract()` throws `std::invalid_argument` on a validation failure. Rust has no exceptions at all; `contract()` here returns `Result<Tensor, String>` instead. This is not a lesser substitute -- an unused `Result` triggers a compiler warning, so the type system itself makes it structurally awkward to silently ignore a validation failure the way a caller *can* silently ignore a C++ exception by simply not wrapping the call in a `try` block. Section D.5 uses this difference directly.

### Worked Example D.3.1 — Matrix multiply, recovered as `contract(A, {1}, B, {0})`

```
=== Matrix multiply as a rank-2 contraction over ONE shared axis ===

A [3 x 2]:
     1.00    2.00 
     3.00    4.00 
     5.00    6.00 
B [2 x 4]:
     1.00    0.00    2.00    1.00 
     0.00    1.00    1.00    2.00 
C = contract(A, axis 1, B, axis 0) [3 x 4]:
     1.00    2.00    4.00    5.00 
     3.00    4.00   10.00   11.00 
     5.00    6.00   16.00   17.00 

hand-computed cross-check:
  C[0][0] expected 1*1 + 2*0 = 1.00   -> got 1.00
  C[2][3] expected 5*1 + 6*2 = 17.00  -> got 17.00

All checks passed. (Full cross-check against numpy in the next step.)
```

The two spot-checked entries above were worked out by hand before running the program, not after. A separate, independent check goes further and compares *every* entry of `C` against `numpy`'s own `@` matrix-multiply operator, computed by a completely different implementation in a different language:

```python
>>> import numpy as np
>>> A = np.array([[1,2],[3,4],[5,6]], dtype=float)
>>> B = np.array([[1,0,2,1],[0,1,1,2]], dtype=float)
>>> np.allclose(A @ B, [[1,2,4,5],[3,4,10,11],[5,6,16,17]])
True
```

## D.4 Contracting Over Multiple Axes at Once

The whole point of writing `contract()` generically in Section D.3 -- rather than hard-coding "one shared axis" -- is that it already handles more than one shared axis, with no changes at all. `03_double_contraction.rs` contracts a `[2, 3, 4]` tensor against a `[3, 4, 5]` tensor over *two* axis pairs simultaneously (A's axes `{1, 2}` against B's axes `{0, 1}`), producing a `[2, 5]` output -- exactly the case introduced at the end of Section D.1.

### Worked Example D.4.1 — A double contraction, cross-checked against `numpy.tensordot`

```
=== A double contraction: TWO shared axes at once ===

output shape: [2, 5]  (expected [2, 5])
C =
     10.00     5.00     0.00    -5.00   -10.00 
      2.00     1.00     0.00    -1.00    -2.00 
```

`numpy.tensordot` is numpy's own general-purpose contraction function, built independently of anything in this appendix, and it takes the identical axis specification (`axes=([1,2],[0,1])`):

```python
>>> import numpy as np
>>> C = np.tensordot(A, B, axes=([1,2],[0,1]))
>>> C
array([[ 10.,   5.,   0.,  -5., -10.],
       [  2.,   1.,   0.,  -1.,  -2.]])
>>> np.allclose(C, rust_output)
True
```

Two structurally different implementations -- this appendix's recursive, hand-rolled `contract()`, and numpy's own compiled tensordot -- agree exactly on every one of the ten output values.

## D.5 [COMMON TRAP] Validating Axis Pairs

It is easy to pass the wrong axis, or an axis pair whose sizes don't actually match, and have the mistake produce a plausible-looking but *wrong* tensor instead of an error. `contract()` guards against this up front: before doing any arithmetic, it checks that every contracted axis pair has matching sizes, and returns `Err(String)` if not -- Rust's own idiom for the same job C++'s `throw std::invalid_argument` does, since Rust has no exceptions at all (see Section D.3's language note on this).

`04_shape_mismatch_trap.rs` genuinely triggers this check, rather than only describing it -- and then goes one step further than the CUDA C++ edition of this appendix does. There, what an unvalidated contraction would do was described in prose and illustrated with numbers computed separately, outside the appendix's own runnable file. Here, `contract_unchecked()` is `contract()` with the validation removed and nothing else changed, genuinely built and run on real, nonzero data in the *same* program, with its output independently cross-checked against a manually-sliced-B matmul computed by a completely separate loop, right there in `main()`:

```
=== [COMMON TRAP] contracting a mismatched axis pair ===

A.shape = [3, 2], B.shape = [4, 5]
attempting contract(A, axis 1 [size 2], B, axis 0 [size 4]) ...

caught Err, as expected:
  "contract: mismatched dimension on contracted axis pair 0 (A.shape[1]=2 vs B.shape[0]=4)"

Without that check, contract_unchecked() runs the identical walk on the
same A and B -- and does not panic, does not error, and produces a fully
shaped, plausible-looking result:

contract_unchecked(A, axis 1, B, axis 0) [3 x 5]:
    13.0   16.0   19.0   22.0   25.0 
    27.0   34.0   41.0   48.0   55.0 
    41.0   52.0   63.0   74.0   85.0 

independent cross-check -- A multiplied by B sliced down to its first
2 rows (A's axis-1 size), computed by a separate loop in this program:
    13.0   16.0   19.0   22.0   25.0 
    27.0   34.0   41.0   48.0   55.0 
    41.0   52.0   63.0   74.0   85.0 

max |contract_unchecked - manual sliced matmul| = 0.00e0  (bit-for-bit match, as the mechanism predicts)

No panic, no error, no obviously-wrong shape -- just a confidently wrong
answer (B's rows 2 and 3 were never read) built from half the intended
data. This is precisely the failure mode contract()'s up-front axis-size
validation exists to convert into an immediate, loud Err instead.
```

The mechanism is visible directly in the code, not just in the numbers: `contract_core()`'s `contracted_shape` is built from `A`'s axis sizes only (the loop `for &ax in &axes_a { contracted_shape.push(a.shape[ax as usize]); } `, marked with a comment in the source below). With validation in place, that's provably safe -- a mismatch would already have returned `Err` before this line runs. Without validation, this exact same line is the entire mechanism: the contracted-index loop is sized from `A`'s axis-1 (size 2), so it iterates `k` over `0..2`, and `B`'s rows 2 and 3 are simply never visited. No out-of-bounds access, no crash -- just data that is never read. The independent cross-check above (a hand-written loop that explicitly slices `B` down to its first 2 rows and multiplies) reproduces `contract_unchecked()`'s output bit-for-bit, confirming the mechanism is exactly what the code predicts, not merely what the prose claims.

No panic, no error, no obviously-wrong shape -- just a confidently wrong answer built from half the intended data. This is precisely the failure mode the up-front validation in Section D.3 exists to convert into an immediate, loud `Err` instead.

## D.6 Performance: Loop Order and Cache Locality

The mathematical definition of a contraction says nothing about the order its loops run in. On real hardware, that order decides whether the innermost loop reads memory contiguously or with a large stride -- and `05_loop_order_performance.rs` *measures* the difference this makes on matrix multiply, rather than asserting one loop order is faster.

Two loop orders compute the identical sum:

- **`ijk`** (the order `contract()` itself uses): for each output row `i`, for each column `j`, sum over `k`. `B[k][j]` is read with stride `N` (a whole row skipped per step) in the innermost loop.
- **`ikj`**: for each row `i`, for each `k`, sweep `j` across a full row of both `C` and `B`. Every innermost-loop access is now stride-1.

```
=== Loop order and cache locality, measured (not asserted) ===

     N       ijk (ms)       ikj (ms) ikj speedup
    64          0.476          0.083      5.71x   (sum match: rel diff 0.00e0)
   128          4.563          0.785      5.82x   (sum match: rel diff 0.00e0)
   256         35.564          5.703      6.24x   (sum match: rel diff 0.00e0)
   384        124.785         22.209      5.62x   (sum match: rel diff 0.00e0)

Both loop orders compute the identical sum at every size (confirmed above);
only the WALL-CLOCK cost differs, purely from how memory is walked.

(Ran inside a shared cloud container, not dedicated hardware -- see the
prose discussion of these numbers for why the TREND, not the absolute
milliseconds, is the portable claim here.)
```

**On these specific numbers:** this ran inside a shared cloud container, not dedicated hardware, so the exact millisecond values above are not portable to your own machine and should not be quoted as general facts about "matrix multiply performance." The `(sum match: rel diff 0.00e0)` column on every row is unambiguous, though: reordering loops for speed changed nothing about the answer.

**A genuine divergence from the CUDA C++ edition's own measurement, worth stating honestly rather than smoothing over:** the C++ edition's `g++`-compiled run showed the `ikj` speedup *growing* with `N` (from roughly 1.2x at `N=64` to roughly 2.1x at `N=384`), and explained this as the working set outgrowing cache. This Rust port's `--release` numbers above show a much *larger* speedup (roughly 5.6x-6.2x) that does **not** grow cleanly with `N` -- it's already almost entirely present at the smallest size tested. Re-running the identical Rust source in the unoptimized `dev` profile changes the picture substantially:

```
=== Loop order and cache locality, measured (not asserted) ===

     N       ijk (ms)       ikj (ms) ikj speedup
    64          3.316          3.228      1.03x   (sum match: rel diff 0.00e0)
   128         27.926         25.032      1.12x   (sum match: rel diff 0.00e0)
   256        247.053        209.487      1.18x   (sum match: rel diff 0.00e0)
   384        917.843        726.136      1.26x   (sum match: rel diff 0.00e0)

Both loop orders compute the identical sum at every size (confirmed above);
only the WALL-CLOCK cost differs, purely from how memory is walked.

(Ran inside a shared cloud container, not dedicated hardware -- see the
prose discussion of these numbers for why the TREND, not the absolute
milliseconds, is the portable claim here.)
```

The `dev`-profile numbers (a roughly 1.03x-1.26x speedup, growing with `N`) look much more like the CUDA C++ edition's own shape. The likely explanation: LLVM's `--release` optimizer can auto-vectorize `ikj`'s innermost loop, because every access in it (`C[i*N+j]`, `B[k*N+j]`) is stride-1 across a fixed range of `j` -- a textbook target for SIMD. `ijk`'s innermost loop, reading `B[k*N+j]` with stride `N`, offers the optimizer no such opportunity. So in the optimized build, most of `ikj`'s advantage comes from vectorization that is available *regardless of problem size*, and the additional cache-locality effect that grows with `N` is still present but is a comparatively small contribution on top of it. In the unoptimized build, neither loop is vectorized, so the *only* effect being measured is the cache-locality one -- which is exactly the effect the CUDA C++ edition's own (unoptimized, `-fsanitize=address`-instrumented) build was measuring too. Both loop orders computing the identical sum, at every size, in both profiles, is the one part of this finding that holds unconditionally.

### Formulas and Key Terms

- **Cache line** — the unit of memory a CPU actually fetches from main memory on a miss (commonly 64 bytes, or 8 `f64`s); a stride-1 access pattern uses every value fetched, a large-stride pattern discards most of each line fetched.
- **Arithmetic intensity** — the ratio of floating-point operations performed to bytes of memory moved; a contraction's *total* FLOP count doesn't change with loop order, but how many bytes must actually be re-fetched from slow memory (versus reused from cache) does, which is exactly what Section D.6's timings are measuring the consequence of.
- **FLOP count** — for a contraction summing over `p` shared axes each of size `K_1..K_p`, producing an output of `T` total elements, the total multiply-add count is `T · Π K_i`; for matrix multiply (`M×K` by `K×N`), that's `M·N·K` multiply-adds, or `2MNK` FLOPs counting the multiply and add separately.
- **Auto-vectorization** — an optimizing compiler's ability to rewrite a scalar loop into SIMD instructions that process several elements per CPU instruction, when the loop's memory-access pattern makes that transformation provably safe; a stride-1 inner loop is the common case this applies to, which is exactly the structural difference between `ikj`'s and `ijk`'s innermost loops here.

## D.7 Complete Runnable Code

### File: 01_tensor_indexing.rs

```rust
// D.2: Shapes, Strides, and Indexing.
//
// Before any contraction logic can be written, there is one piece of
// bookkeeping every later section depends on: converting between a
// multi-index (one integer per axis) and the single flat integer offset
// into a tensor's underlying buffer. This file uses row-major layout
// throughout -- the last axis varies fastest in memory, the same
// convention this book has used since Chapter 2.

fn row_major_strides(shape: &[i32]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut acc: i64 = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i] as i64;
    }
    strides
}

fn ravel_index(idx: &[i32], strides: &[i64]) -> i64 {
    idx.iter().zip(strides.iter()).map(|(&i, &s)| i as i64 * s).sum()
}

fn unravel_index(mut lin: i64, shape: &[i32], strides: &[i64]) -> Vec<i32> {
    let mut idx = vec![0i32; shape.len()];
    for i in 0..shape.len() {
        idx[i] = (lin / strides[i]) as i32;
        lin %= strides[i];
    }
    idx
}

fn main() {
    let shape = vec![2, 3, 4];
    let strides = row_major_strides(&shape);

    println!("shape = [{}, {}, {}]", shape[0], shape[1], shape[2]);
    println!("row-major strides = [{}, {}, {}]", strides[0], strides[1], strides[2]);

    let total: i64 = shape.iter().map(|&s| s as i64).product();
    println!("total elements = {total}\n");

    let idx17 = unravel_index(17, &shape, &strides);
    println!(
        "unravel_index(17) = ({}, {}, {})  [hand-computed expected: (1, 1, 1)]",
        idx17[0], idx17[1], idx17[2]
    );
    assert_eq!(idx17, vec![1, 1, 1]);

    let back = ravel_index(&idx17, &strides);
    println!("ravel_index(1,1,1) = {back}  [must round-trip back to 17]\n");
    assert_eq!(back, 17);

    let mut failures = 0;
    for lin in 0..total {
        let idx = unravel_index(lin, &shape, &strides);
        let round_trip = ravel_index(&idx, &strides);
        if round_trip != lin {
            failures += 1;
        }
    }
    println!("exhaustive round-trip check over all {total} offsets: {failures} failures");
    assert_eq!(failures, 0);

    println!("\nAll checks passed.");
}
```

### File: 02_generic_contraction.rs

```rust
// D.3: A Generic Contraction Function.
//
// With indexing settled, `contract()` implements the general definition of
// a tensor contraction directly: given two tensors and a list of axes to
// contract on each, it works out which axes are free, builds the output
// shape, and fills it in by walking every combination of free indices,
// and for each one, summing over every combination of contracted indices.
//
// The walk uses `for_each_index()`, which visits every multi-index of a
// given shape in row-major order -- one level of recursion per axis. C++'s
// version captures itself in a `std::function` closure; Rust's closures
// cannot straightforwardly reference themselves this way (a `FnMut`
// cannot hold a reference to itself without extra indirection), so this
// is written as an ordinary named function taking the current axis
// explicitly -- a small, genuine language difference with an identical
// result, not a workaround for anything broken.
//
// `contract()` itself returns `Result<Tensor, String>` rather than
// throwing a C++ exception -- Rust has no exceptions, and a `Result` the
// caller must explicitly handle (an unused `Result` is a compiler warning)
// is this language's own way of making a validation failure impossible to
// silently ignore.
struct Tensor {
    shape: Vec<i32>,
    data: Vec<f64>,
}

impl Tensor {
    fn new(shape: Vec<i32>) -> Self {
        let total: i64 = shape.iter().map(|&s| s as i64).product();
        Tensor { shape, data: vec![0.0; total as usize] }
    }
}

fn row_major_strides(shape: &[i32]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut acc: i64 = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i] as i64;
    }
    strides
}

fn ravel_index(idx: &[i32], strides: &[i64]) -> i64 {
    idx.iter().zip(strides.iter()).map(|(&i, &s)| i as i64 * s).sum()
}

fn for_each_index_recurse<F: FnMut(&[i32])>(axis: usize, shape: &[i32], idx: &mut Vec<i32>, visit: &mut F) {
    if axis == shape.len() {
        visit(idx);
        return;
    }
    for i in 0..shape[axis] {
        idx[axis] = i;
        for_each_index_recurse(axis + 1, shape, idx, visit);
    }
}

fn for_each_index<F: FnMut(&[i32])>(shape: &[i32], visit: &mut F) {
    let mut idx = vec![0i32; shape.len()];
    if shape.is_empty() {
        visit(&idx);
        return;
    }
    for_each_index_recurse(0, shape, &mut idx, visit);
}

fn contract(a: &Tensor, axes_a: Vec<i32>, b: &Tensor, axes_b: Vec<i32>) -> Result<Tensor, String> {
    if axes_a.len() != axes_b.len() {
        return Err("contract: axes_a and axes_b must have the same length".to_string());
    }

    for k in 0..axes_a.len() {
        let da = a.shape[axes_a[k] as usize];
        let db = b.shape[axes_b[k] as usize];
        if da != db {
            return Err(format!(
                "contract: mismatched dimension on contracted axis pair {k} (A.shape[{}]={da} vs B.shape[{}]={db})",
                axes_a[k], axes_b[k]
            ));
        }
    }

    let free_a: Vec<i32> = (0..a.shape.len() as i32).filter(|x| !axes_a.contains(x)).collect();
    let free_b: Vec<i32> = (0..b.shape.len() as i32).filter(|x| !axes_b.contains(x)).collect();

    let mut out_shape = Vec::new();
    for &ax in &free_a {
        out_shape.push(a.shape[ax as usize]);
    }
    for &bx in &free_b {
        out_shape.push(b.shape[bx as usize]);
    }

    let mut contracted_shape = Vec::new();
    for &ax in &axes_a {
        contracted_shape.push(a.shape[ax as usize]);
    }

    let mut c = Tensor::new(out_shape.clone());
    let s_a = row_major_strides(&a.shape);
    let s_b = row_major_strides(&b.shape);
    let s_c = row_major_strides(&out_shape);

    for_each_index(&out_shape, &mut |out_idx: &[i32]| {
        let mut idx_a = vec![0i32; a.shape.len()];
        let mut idx_b = vec![0i32; b.shape.len()];
        for i in 0..free_a.len() {
            idx_a[free_a[i] as usize] = out_idx[i];
        }
        for j in 0..free_b.len() {
            idx_b[free_b[j] as usize] = out_idx[free_a.len() + j];
        }

        let mut acc = 0.0f64;
        for_each_index(&contracted_shape, &mut |c_idx: &[i32]| {
            for k in 0..axes_a.len() {
                idx_a[axes_a[k] as usize] = c_idx[k];
                idx_b[axes_b[k] as usize] = c_idx[k];
            }
            acc += a.data[ravel_index(&idx_a, &s_a) as usize] * b.data[ravel_index(&idx_b, &s_b) as usize];
        });
        c.data[ravel_index(out_idx, &s_c) as usize] = acc;
    });

    Ok(c)
}

fn print_matrix(label: &str, m: &Tensor) {
    println!("{label} [{} x {}]:", m.shape[0], m.shape[1]);
    for i in 0..m.shape[0] {
        print!("  ");
        for j in 0..m.shape[1] {
            print!("{:7.2} ", m.data[(i * m.shape[1] + j) as usize]);
        }
        println!();
    }
}

fn main() {
    println!("=== Matrix multiply as a rank-2 contraction over ONE shared axis ===\n");

    let mut a = Tensor::new(vec![3, 2]);
    let mut b = Tensor::new(vec![2, 4]);
    let a_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vals = [1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0];
    a.data.copy_from_slice(&a_vals);
    b.data.copy_from_slice(&b_vals);

    print_matrix("A", &a);
    print_matrix("B", &b);

    let c = contract(&a, vec![1], &b, vec![0]).expect("valid axis pair");
    print_matrix("C = contract(A, axis 1, B, axis 0)", &c);

    println!("\nhand-computed cross-check:");
    println!("  C[0][0] expected 1*1 + 2*0 = 1.00   -> got {:.2}", c.data[0 * 4 + 0]);
    println!("  C[2][3] expected 5*1 + 6*2 = 17.00  -> got {:.2}", c.data[2 * 4 + 3]);
    assert!((c.data[0 * 4 + 0] - 1.0).abs() < 1e-9);
    assert!((c.data[2 * 4 + 3] - 17.0).abs() < 1e-9);

    println!("\nAll checks passed. (Full cross-check against numpy in the next step.)");
}
```

### File: 03_double_contraction.rs

```rust
// D.4: Contracting Over Multiple Axes at Once.
//
// The whole point of writing `contract()` generically in D.3 -- rather than
// hard-coding "one shared axis" -- is that it already handles more than one
// shared axis, with no changes at all. This file contracts a [2, 3, 4]
// tensor against a [3, 4, 5] tensor over TWO axis pairs simultaneously
// (A's axes {1, 2} against B's axes {0, 1}), producing a [2, 5] output --
// exactly the general case introduced in D.1.

struct Tensor {
    shape: Vec<i32>,
    data: Vec<f64>,
}

impl Tensor {
    fn new(shape: Vec<i32>) -> Self {
        let total: i64 = shape.iter().map(|&s| s as i64).product();
        Tensor { shape, data: vec![0.0; total as usize] }
    }
}

fn row_major_strides(shape: &[i32]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut acc: i64 = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i] as i64;
    }
    strides
}

fn ravel_index(idx: &[i32], strides: &[i64]) -> i64 {
    idx.iter().zip(strides.iter()).map(|(&i, &s)| i as i64 * s).sum()
}

fn for_each_index_recurse<F: FnMut(&[i32])>(axis: usize, shape: &[i32], idx: &mut Vec<i32>, visit: &mut F) {
    if axis == shape.len() {
        visit(idx);
        return;
    }
    for i in 0..shape[axis] {
        idx[axis] = i;
        for_each_index_recurse(axis + 1, shape, idx, visit);
    }
}

fn for_each_index<F: FnMut(&[i32])>(shape: &[i32], visit: &mut F) {
    let mut idx = vec![0i32; shape.len()];
    if shape.is_empty() {
        visit(&idx);
        return;
    }
    for_each_index_recurse(0, shape, &mut idx, visit);
}

fn contract(a: &Tensor, axes_a: Vec<i32>, b: &Tensor, axes_b: Vec<i32>) -> Result<Tensor, String> {
    if axes_a.len() != axes_b.len() {
        return Err("contract: axes_a and axes_b must have the same length".to_string());
    }

    for k in 0..axes_a.len() {
        let da = a.shape[axes_a[k] as usize];
        let db = b.shape[axes_b[k] as usize];
        if da != db {
            return Err(format!("contract: mismatched dimension on axis pair {k}"));
        }
    }

    let free_a: Vec<i32> = (0..a.shape.len() as i32).filter(|x| !axes_a.contains(x)).collect();
    let free_b: Vec<i32> = (0..b.shape.len() as i32).filter(|x| !axes_b.contains(x)).collect();

    let mut out_shape = Vec::new();
    for &ax in &free_a {
        out_shape.push(a.shape[ax as usize]);
    }
    for &bx in &free_b {
        out_shape.push(b.shape[bx as usize]);
    }

    let mut contracted_shape = Vec::new();
    for &ax in &axes_a {
        contracted_shape.push(a.shape[ax as usize]);
    }

    let mut c = Tensor::new(out_shape.clone());
    let s_a = row_major_strides(&a.shape);
    let s_b = row_major_strides(&b.shape);
    let s_c = row_major_strides(&out_shape);

    for_each_index(&out_shape, &mut |out_idx: &[i32]| {
        let mut idx_a = vec![0i32; a.shape.len()];
        let mut idx_b = vec![0i32; b.shape.len()];
        for i in 0..free_a.len() {
            idx_a[free_a[i] as usize] = out_idx[i];
        }
        for j in 0..free_b.len() {
            idx_b[free_b[j] as usize] = out_idx[free_a.len() + j];
        }

        let mut acc = 0.0f64;
        for_each_index(&contracted_shape, &mut |c_idx: &[i32]| {
            for k in 0..axes_a.len() {
                idx_a[axes_a[k] as usize] = c_idx[k];
                idx_b[axes_b[k] as usize] = c_idx[k];
            }
            acc += a.data[ravel_index(&idx_a, &s_a) as usize] * b.data[ravel_index(&idx_b, &s_b) as usize];
        });
        c.data[ravel_index(out_idx, &s_c) as usize] = acc;
    });

    Ok(c)
}

fn main() {
    println!("=== A double contraction: TWO shared axes at once ===\n");

    let mut a = Tensor::new(vec![2, 3, 4]);
    let mut b = Tensor::new(vec![3, 4, 5]);
    for i in 0..a.data.len() {
        a.data[i] = (i % 7) as f64 - 3.0;
    }
    for i in 0..b.data.len() {
        b.data[i] = (i % 5) as f64 - 2.0;
    }

    let c = contract(&a, vec![1, 2], &b, vec![0, 1]).expect("valid axis pair");
    println!("output shape: [{}, {}]  (expected [2, 5])", c.shape[0], c.shape[1]);

    println!("C =");
    for i in 0..c.shape[0] {
        print!("  ");
        for j in 0..c.shape[1] {
            print!("{:8.2} ", c.data[(i * c.shape[1] + j) as usize]);
        }
        println!();
    }

    assert_eq!(c.shape, vec![2, 5]);
}
```

### File: 04_shape_mismatch_trap.rs

```rust
// D.5: [COMMON TRAP] Validating Axis Pairs.
//
// It is easy to pass the wrong axis, or an axis pair whose sizes don't
// actually match, and have the mistake produce a plausible-looking but
// WRONG tensor instead of an error. `contract()` guards against this up
// front: before doing any arithmetic, it checks that every contracted axis
// pair has matching sizes, and returns `Err(...)` if not -- Rust's own
// idiom for the same job C++'s `throw std::invalid_argument` does, since
// Rust has no exceptions.
//
// This file genuinely triggers that check (the validated path), and then
// goes one step further than only describing what would happen without
// it: `contract_unchecked()` below is `contract()` with the validation
// removed -- nothing else changed -- run on real, nonzero data, with its
// output independently cross-checked in this same program against a
// manually-sliced-B matmul computed by a completely separate loop.

struct Tensor {
    shape: Vec<i32>,
    data: Vec<f64>,
}

impl Tensor {
    fn new(shape: Vec<i32>) -> Self {
        let total: i64 = shape.iter().map(|&s| s as i64).product();
        Tensor { shape, data: vec![0.0; total as usize] }
    }
}

fn row_major_strides(shape: &[i32]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut acc: i64 = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i] as i64;
    }
    strides
}

fn ravel_index(idx: &[i32], strides: &[i64]) -> i64 {
    idx.iter().zip(strides.iter()).map(|(&i, &s)| i as i64 * s).sum()
}

fn for_each_index_recurse<F: FnMut(&[i32])>(axis: usize, shape: &[i32], idx: &mut Vec<i32>, visit: &mut F) {
    if axis == shape.len() {
        visit(idx);
        return;
    }
    for i in 0..shape[axis] {
        idx[axis] = i;
        for_each_index_recurse(axis + 1, shape, idx, visit);
    }
}

fn for_each_index<F: FnMut(&[i32])>(shape: &[i32], visit: &mut F) {
    let mut idx = vec![0i32; shape.len()];
    if shape.is_empty() {
        visit(&idx);
        return;
    }
    for_each_index_recurse(0, shape, &mut idx, visit);
}

// The validated path: checks axis sizes before touching any memory.
fn contract(a: &Tensor, axes_a: Vec<i32>, b: &Tensor, axes_b: Vec<i32>) -> Result<Tensor, String> {
    if axes_a.len() != axes_b.len() {
        return Err("contract: axes_a and axes_b must have the same length".to_string());
    }

    for k in 0..axes_a.len() {
        let da = a.shape[axes_a[k] as usize];
        let db = b.shape[axes_b[k] as usize];
        if da != db {
            return Err(format!(
                "contract: mismatched dimension on contracted axis pair {k} (A.shape[{}]={da} vs B.shape[{}]={db})",
                axes_a[k], axes_b[k]
            ));
        }
    }

    contract_core(a, axes_a, b, axes_b)
}

// The SAME walk, with the up-front validation removed. Nothing else about
// this function differs from `contract()` -- which is exactly the point:
// the bug is an omission, not a different algorithm.
fn contract_unchecked(a: &Tensor, axes_a: Vec<i32>, b: &Tensor, axes_b: Vec<i32>) -> Tensor {
    contract_core(a, axes_a, b, axes_b).expect("contract_core cannot fail once lengths already match")
}

// Shared walk used by both the checked and unchecked entry points above.
// `contracted_shape` is built from A's axes ONLY (line marked below) --
// with validation in place that's provably safe, because a mismatch would
// already have returned `Err` before this ran. Without validation, this
// same line is exactly the mechanism that silently discards part of B.
fn contract_core(a: &Tensor, axes_a: Vec<i32>, b: &Tensor, axes_b: Vec<i32>) -> Result<Tensor, String> {
    let free_a: Vec<i32> = (0..a.shape.len() as i32).filter(|x| !axes_a.contains(x)).collect();
    let free_b: Vec<i32> = (0..b.shape.len() as i32).filter(|x| !axes_b.contains(x)).collect();

    let mut out_shape = Vec::new();
    for &ax in &free_a {
        out_shape.push(a.shape[ax as usize]);
    }
    for &bx in &free_b {
        out_shape.push(b.shape[bx as usize]);
    }

    // <-- built from A's axis sizes only; B's matching axis size is never consulted here.
    let mut contracted_shape = Vec::new();
    for &ax in &axes_a {
        contracted_shape.push(a.shape[ax as usize]);
    }

    let mut c = Tensor::new(out_shape.clone());
    let s_a = row_major_strides(&a.shape);
    let s_b = row_major_strides(&b.shape);
    let s_c = row_major_strides(&out_shape);

    for_each_index(&out_shape, &mut |out_idx: &[i32]| {
        let mut idx_a = vec![0i32; a.shape.len()];
        let mut idx_b = vec![0i32; b.shape.len()];
        for i in 0..free_a.len() {
            idx_a[free_a[i] as usize] = out_idx[i];
        }
        for j in 0..free_b.len() {
            idx_b[free_b[j] as usize] = out_idx[free_a.len() + j];
        }

        let mut acc = 0.0f64;
        for_each_index(&contracted_shape, &mut |c_idx: &[i32]| {
            for k in 0..axes_a.len() {
                idx_a[axes_a[k] as usize] = c_idx[k];
                idx_b[axes_b[k] as usize] = c_idx[k];
            }
            acc += a.data[ravel_index(&idx_a, &s_a) as usize] * b.data[ravel_index(&idx_b, &s_b) as usize];
        });
        c.data[ravel_index(out_idx, &s_c) as usize] = acc;
    });

    Ok(c)
}

fn print_matrix(label: &str, m: &Tensor) {
    println!("{label} [{} x {}]:", m.shape[0], m.shape[1]);
    for i in 0..m.shape[0] {
        print!("  ");
        for j in 0..m.shape[1] {
            print!("{:6.1} ", m.data[(i * m.shape[1] + j) as usize]);
        }
        println!();
    }
}

fn main() {
    println!("=== [COMMON TRAP] contracting a mismatched axis pair ===\n");

    let mut a = Tensor::new(vec![3, 2]);
    let mut b = Tensor::new(vec![4, 5]);
    let a_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    a.data.copy_from_slice(&a_vals);
    for i in 0..b.data.len() {
        b.data[i] = (i + 1) as f64;
    }

    println!("A.shape = [3, 2], B.shape = [4, 5]");
    println!("attempting contract(A, axis 1 [size 2], B, axis 0 [size 4]) ...\n");

    match contract(&a, vec![1], &b, vec![0]) {
        Ok(_) => {
            println!("(unreached -- this should have returned Err)");
            std::process::exit(1);
        }
        Err(msg) => {
            println!("caught Err, as expected:");
            println!("  \"{msg}\"");
        }
    }

    println!("\nWithout that check, contract_unchecked() runs the identical walk on the");
    println!("same A and B -- and does not panic, does not error, and produces a fully");
    println!("shaped, plausible-looking result:\n");

    let unchecked = contract_unchecked(&a, vec![1], &b, vec![0]);
    print_matrix("contract_unchecked(A, axis 1, B, axis 0)", &unchecked);

    // Independent cross-check: manually slice B down to its first 2 rows
    // (A's axis-1 size) and multiply, via a completely separate loop that
    // shares no code with contract_core().
    let b_sliced_rows = 2usize;
    let mut manual = vec![0.0f64; 3 * 5];
    for i in 0..3usize {
        for j in 0..5usize {
            let mut acc = 0.0f64;
            for k in 0..b_sliced_rows {
                acc += a.data[i * 2 + k] * b.data[k * 5 + j];
            }
            manual[i * 5 + j] = acc;
        }
    }

    println!("\nindependent cross-check -- A multiplied by B sliced down to its first");
    println!("2 rows (A's axis-1 size), computed by a separate loop in this program:");
    for i in 0..3usize {
        print!("  ");
        for j in 0..5usize {
            print!("{:6.1} ", manual[i * 5 + j]);
        }
        println!();
    }

    let mut max_abs_diff = 0.0f64;
    for i in 0..unchecked.data.len() {
        let d = (unchecked.data[i] - manual[i]).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    println!(
        "\nmax |contract_unchecked - manual sliced matmul| = {:.2e}  ({})",
        max_abs_diff,
        if max_abs_diff < 1e-9 { "bit-for-bit match, as the mechanism predicts" } else { "MISMATCH" }
    );
    assert!(max_abs_diff < 1e-9);

    println!("\nNo panic, no error, no obviously-wrong shape -- just a confidently wrong");
    println!("answer (B's rows 2 and 3 were never read) built from half the intended");
    println!("data. This is precisely the failure mode contract()'s up-front axis-size");
    println!("validation exists to convert into an immediate, loud Err instead.");
}
```

### File: 05_loop_order_performance.rs

```rust
// D.6: Performance -- Loop Order and Cache Locality.
//
// The mathematical definition of a contraction says nothing about the
// order its loops run in. On real hardware, that order decides whether the
// innermost loop reads memory contiguously or with a large stride -- and
// this file MEASURES the difference that makes on matrix multiply, rather
// than asserting one loop order is faster.
//
// Two loop orders compute the identical sum:
//   ijk -- for each output row i, for each column j, sum over k. B[k][j]
//          is read with stride N (a whole row skipped per step) in the
//          innermost loop.
//   ikj -- for each row i, for each k, sweep j across a full row of both
//          C and B. Every innermost-loop access is now stride-1.

fn matmul_ijk(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

fn matmul_ikj(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    c.iter_mut().for_each(|v| *v = 0.0);
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
}

fn main() {
    println!("=== Loop order and cache locality, measured (not asserted) ===\n");
    println!("{:>6} {:>14} {:>14} {:>10}", "N", "ijk (ms)", "ikj (ms)", "ikj speedup");

    let sizes = [64usize, 128, 256, 384];
    for &n in &sizes {
        let mut a = vec![0.0f64; n * n];
        let mut b = vec![0.0f64; n * n];
        let mut c = vec![0.0f64; n * n];
        let mut seed: u32 = 12345;
        for v in a.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = (seed % 1000) as f64 / 100.0;
        }
        for v in b.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = (seed % 1000) as f64 / 100.0;
        }

        let t0 = std::time::Instant::now();
        matmul_ijk(&a, &b, &mut c, n);
        let ms_ijk = t0.elapsed().as_secs_f64() * 1000.0;
        let sum_ijk: f64 = c.iter().sum();

        let t2 = std::time::Instant::now();
        matmul_ikj(&a, &b, &mut c, n);
        let ms_ikj = t2.elapsed().as_secs_f64() * 1000.0;
        let sum_ikj: f64 = c.iter().sum();

        print!(
            "{:>6} {:>14.3} {:>14.3} {:>9.2}x",
            n,
            ms_ijk,
            ms_ikj,
            ms_ijk / ms_ikj
        );
        let rel_diff = (sum_ijk - sum_ikj).abs() / sum_ijk.abs();
        println!("   (sum match: rel diff {:.2e})", rel_diff);
        if rel_diff > 1e-9 {
            println!("MISMATCH -- loop reorder changed the answer!");
            std::process::exit(1);
        }
    }

    println!("\nBoth loop orders compute the identical sum at every size (confirmed above);");
    println!("only the WALL-CLOCK cost differs, purely from how memory is walked.");
    println!("\n(Ran inside a shared cloud container, not dedicated hardware -- see the");
    println!("prose discussion of these numbers for why the TREND, not the absolute");
    println!("milliseconds, is the portable claim here.)");
}
```

## Appendix Summary

A tensor contraction generalizes matrix multiplication by allowing more than one shared axis, and more than two axes total, while keeping the same core operation: sum the elementwise product of two tensors over their matched axes, keep everything else. This appendix built that generality from the ground up -- shapes and strides, a generic `contract()` function, matrix multiply and a genuine double contraction recovered as special cases of it -- and checked every numeric claim against an independent computation: hand-worked arithmetic for the smallest cases, and `numpy`'s own, differently-implemented `@` and `tensordot` for the larger ones. Two further lessons came from *running* code rather than reasoning about it. First, an unvalidated axis pair doesn't crash, it silently discards data -- and this port went further than only describing that, by building and running an unchecked variant in the same file and independently reproducing its exact wrong answer with a separate hand-written loop, confirming the failure mode is exactly the mechanism the text claims, not merely a plausible story. Second, loop order changes a contraction's wall-clock cost without changing a single digit of its answer -- and *how much* it changes turned out to depend on something this appendix's CUDA C++ sibling didn't have to reckon with: whether an optimizing compiler can auto-vectorize the faster loop order's inner loop, which dominates the picture in a release build and disappears entirely in an unoptimized one. Appendix E carries this same mathematics onto the GPU, where the free-index loop that this appendix's `for_each_index()` walks in software becomes the grid of threads a CUDA kernel launches in hardware.

## Self-Check Questions

1. Given tensors of rank 3 and rank 4, contracted over exactly 2 shared axis pairs, what is the rank of the output?
2. Why does row-major layout give the *last* axis a stride of 1, rather than the first?
3. In `contract(A, {1}, B, {0})` for matrix multiply, which index (`i`, `j`, or `k`) is the contracted index, and which two are free?
4. What specifically goes wrong -- not just "it's wrong," but the actual mechanism -- when `contract()`'s axis-size validation is skipped and a mismatched pair is contracted anyway?
5. Section D.6 found the `ikj` speedup over `ijk` to be much larger, but much less dependent on `N`, in a `--release` build than in a `dev` build. What is the compiler doing differently between the two that explains this?
6. `for_each_index()` in Section D.3 is recursive, with one level of recursion per axis. Why can't the exact same technique be reused, unmodified, inside a CUDA `__global__` kernel?
7. Two independent implementations agreeing (this appendix's `contract()` and `numpy.tensordot`) is stronger evidence of correctness than either one alone. What kind of bug could still slip past *both* of them agreeing?
8. If a contraction's total FLOP count doesn't change with loop order, what quantity *does* change, and why does that quantity affect wall-clock time on real hardware?

### Worked Solutions

1. `rank(C) = rank(A) + rank(B) - 2p = 3 + 4 - 2(2) = 3`.
2. Because the last axis is defined to be the one that varies fastest in memory in row-major order -- "stride 1" and "varies fastest" are the same statement. Column-major layout (used by, e.g., Fortran and classic BLAS) makes the same choice for the *first* axis instead; row-major and column-major differ only in which axis gets stride 1, not in the underlying idea.
3. `k` is contracted (it appears on both `A[i][k]` and `B[k][j]`, and is summed away); `i` and `j` are free (each appears on exactly one input, and both survive into `C[i][j]`).
4. The contracted-index loop gets its size from only ONE of the two tensors (`A`'s axis, in `contract_core()`'s implementation) rather than validating that both agree. With `A`'s axis smaller than `B`'s matching axis, the loop simply never reaches most of `B`'s data along that axis -- it isn't read out of bounds, it's just never read at all, silently discarding it. The output has the right *shape* and looks entirely plausible; only its *values* are wrong, computed from a truncated slice of one input. Section D.5 verified this exact mechanism by building and running `contract_unchecked()` and comparing its output, bit-for-bit, against a value independently computed by deliberately slicing `B` down to match.
5. In `--release`, LLVM's optimizer can auto-vectorize `ikj`'s innermost loop, because every one of its memory accesses (`C[i*N+j]`, `B[k*N+j]`) is stride-1 across a fixed range of `j` -- exactly the pattern SIMD instructions exploit. `ijk`'s innermost loop reads `B[k*N+j]` with stride `N`, which the optimizer cannot vectorize the same way. That vectorization advantage is available at every problem size, including the smallest one tested, which is why the release-build speedup is both large and roughly flat across `N`. In `dev`, neither loop is vectorized, so the only advantage left is the cache-locality one from Section D.6's stride-1-versus-stride-N discussion -- and that advantage genuinely does grow with `N`, as the working set outgrows cache, which is exactly the shape the `dev`-profile measurement showed.
6. Recursion depth in `for_each_index()` equals the tensor's rank, decided at runtime from a `Vec`'s length -- ordinary function-call recursion, which device code can do in principle, but which needs a rank known at compile time or a call stack CUDA threads don't cheaply have thousands of copies of. Appendix E.3 solves this the way Section D.2 already solved indexing before `contract()` existed: fixed-size arrays and iterative divmod unraveling, capped at a compile-time `MAX_RANK`, rather than recursion.
7. A bug present in the *mathematical specification itself*, shared by construction -- for example, if this appendix had mis-stated which axes to contract in the double-contraction example, both `contract()` and `numpy.tensordot` would faithfully compute what was actually asked for and agree with each other, while still answering a different question than the one intended. Independent implementations catch implementation bugs; they do not catch a shared misunderstanding of the problem, which is exactly why Section D.4's hand-picked axis pairs were chosen to match a case worth checking by inspection, not just by agreement.
8. The number of bytes actually moved between main memory and the CPU's cache (as opposed to reused from cache) changes with loop order, even though the FLOP count is fixed. Real hardware spends time waiting on memory transfers, not only on arithmetic -- so a loop order that reuses more of what it fetches, before that data is evicted, spends less wall-clock time waiting, for the identical number of multiply-adds. Section D.6's finding adds a second such quantity in an optimized build: how much of the loop the compiler can express as SIMD instructions, which is a property of the access pattern, not of the FLOP count either.
