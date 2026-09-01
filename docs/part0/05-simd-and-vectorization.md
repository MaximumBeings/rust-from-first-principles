# Chapter 5: SIMD and Vectorization — `std::arch` and the `wide` Crate on Stable Rust

> "A scalar loop asks the same question a million times, one element at a time. A SIMD loop asks the same question a million times too — it just asks it eight at a time, and only has to ask it a hundred and twenty-five thousand times to get through the same million answers."

**What you will understand by the end of this chapter:**

- Why this book's SIMD code is built on `std::arch::x86_64` intrinsics and the `wide` crate rather than Rust's own `std::simd` — a real constraint of this sandbox, not a stylistic choice, and exactly what `wide` turns out to be once you look inside it
- **SIMD registers and lane width**: genuine, runtime CPU feature detection: (`is_x86_feature_detected!`), `size_of`-confirmed register widths, and what a lane-wise comparison actually produces
- The canonical **vectorized-loop shape** — a main loop over full-width chunks plus a scalar remainder loop for whatever's left — computed by hand and cross-checked element-by-element against a plain scalar loop
- **SIMD reduction**: accumulating into a width-wide vector and reducing it once, at the end, rather than after every chunk — and a genuine, reproducible finding this chapter did not go looking for: that reduction pattern isn't just faster, it's measurably *more numerically accurate* than naive scalar summation, independently confirmed against an `f64` reference
- Why Chapter 3's Struct-of-Arrays argument is a precondition for everything in this chapter, not a separate concern — contiguous SIMD loads need contiguous memory to load from, which is exactly what SoA was built to guarantee

**What you need to know first:**

- Chapter 1 (fixed-size arrays, `[T; N]`) and Chapter 3 in full — this chapter's vectorized loads assume the same contiguous, SoA-style layout Chapter 3 built and measured
- Getting Started's toolchain note that `nvptx64-nvidia-cuda` and nightly Rust are both unreachable in this sandbox — the same network restriction is why `std::simd` (`#![feature(portable_simd)]`) is off-limits here too, and Section 5.1 confirms this directly rather than assuming it
- If you've read the CUDA or Mojo editions: both of those chapters are titled "SIMD and Vectorization," but they cover genuinely different hardware. CUDA's Chapter 5 is about the warp as *implicit* hardware SIMD — 32 threads the hardware groups together, never named as a type in source. Mojo's Chapter 5 is about `SIMD[DType, width]`, a first-class, compiler-native vector type. Rust's standard library has an equivalent to Mojo's type, `std::simd::Simd<T, N>` — but it's nightly-only, and nightly is unreachable in this sandbox, so this chapter follows Mojo's *loop shape* (main loop plus remainder, wide-then-narrow reduction) using the closest thing stable Rust actually offers.

## 5.1 Why `std::arch` and `wide`, Not `std::simd` `[FOUNDATIONAL]`

### Intuition

Rust has a first-class portable SIMD type, `std::simd::Simd<T, N>`, that looks almost exactly like Mojo's `SIMD[DType, width]` — but it lives behind a nightly-only feature flag, `#![feature(portable_simd)]`, and this book's toolchain is stable-only, confirmed back in Getting Started. Two things fill the gap on stable Rust: `std::arch::x86_64`, raw, unsafe, hardware-specific intrinsics that compile directly to one CPU instruction each, and `wide`, a crate providing safe, portable vector types (`f32x4`, `f32x8`, ...) that are — this chapter confirms directly, not by assertion — thin wrappers around exactly those same intrinsics.

### Background

| | `std::simd::Simd<T, N>` | `std::arch::x86_64` | `wide` |
|---|---|---|---|
| Stability | Nightly-only (`#![feature(portable_simd)]`) | Stable | Stable |
| Safety | Safe | `unsafe` — no bounds or feature checking | Safe |
| Portability | Portable across architectures | x86_64-specific, one function per instruction | Portable — dispatches to the right intrinsics internally |
| Available in this sandbox | No (network-blocked `rust-src`, confirmed in Getting Started) | Yes | Yes |

### Worked Example 5.1.1 — confirming `wide` is a real wrapper around real intrinsics, not a reimplementation

```rust
#[target_feature(enable = "avx")]
unsafe fn add_via_raw_intrinsic(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let vc = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), vc);
        out
    }
}
```

Compiled and run as part of the complete `01_lanes_and_intrinsics.rs` further below:

```bash
cargo run --release --bin 01_lanes_and_intrinsics
```

Genuinely compiled and run, in this book's own sandbox:

```
wide::f32x8 result:        [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]
raw _mm256_add_ps result:  [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]
identical? true
```

`wide::f32x8`'s own `Add` implementation (its real, published source) reads, on a machine with the AVX target feature enabled: `Self { avx: add_m256(self.avx, rhs.avx) }`, and `add_m256`'s own real source is one line: `m256(unsafe { _mm256_add_ps(a.0, b.0) })` — the exact same intrinsic this chapter's own raw version calls directly. The two results above matching exactly isn't a coincidence this chapter is reporting on faith; it's the same CPU instruction executing twice, once reached through 40-some lines of `wide`'s safe API and once reached by hand.

> `[COMMON TRAP]` It's tempting to read "safe wrapper around raw intrinsics" as "safe wrapper around raw intrinsics, so no `unsafe` needed anywhere in this chapter." That's true for `wide`'s own API, which this chapter uses for everything except this one confirming example — but Worked Example 5.1.1's raw version still needs `unsafe` twice over: once because `_mm256_add_ps` and friends are all individually `unsafe fn`s (nothing stops you from calling one on a CPU that doesn't support AVX and getting an illegal-instruction crash), and once because the function itself carries `#[target_feature(enable = "avx")]`, which requires the caller to have already confirmed AVX is actually present — exactly what `is_x86_feature_detected!("avx")` does below before this chapter ever calls it.

## 5.2 SIMD Registers, Lane Width, and Lane-Wise Comparison `[FOUNDATIONAL]`

### Intuition

A SIMD register is a fixed-width piece of hardware — 128 bits for SSE, 256 bits for AVX/AVX2 on this book's own test machine — split into equal-sized lanes. `wide::f32x4` claims 4 lanes of 32 bits (128 bits total); `wide::f32x8` claims 8 lanes of 32 bits (256 bits total). Neither claim needs to be taken on faith: `std::mem::size_of` reports a type's real size in bytes, genuinely computed by the compiler from the type's actual layout.

### Background

| Rust type | Lanes | Bits per lane | Total register width | `size_of` (bytes) |
|---|---|---|---|---|
| `wide::f32x4` | 4 | 32 | 128 | 16 |
| `wide::f32x8` | 8 | 32 | 256 | 32 |

### Worked Example 5.2.1 — feature detection and register width, genuinely queried

Compiled and run as part of `01_lanes_and_intrinsics.rs`:

```
CPU feature detection (genuinely queried at runtime):
  sse2  = true
  sse4.1= true
  avx   = true
  avx2  = true
  fma   = true

size_of::<wide::f32x4>() = 16 bytes (a 128-bit register)
size_of::<wide::f32x8>() = 32 bytes (a 256-bit register)
```

`is_x86_feature_detected!` is a real `std` macro — it queries the CPU at runtime (via the `cpuid` instruction under the hood), not at compile time, so the same compiled binary correctly reports different results on different machines. This book's own sandbox genuinely has AVX2 and FMA available, which is why Worked Example 5.1.1's `#[target_feature(enable = "avx")]` function can run here at all.

### Worked Example 5.2.2 — lane-wise comparison, and what "true" actually looks like

```rust
let a = f32x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let b = f32x8::new([2.0, 2.0, 1.0, 5.0, 5.0, 3.0, 9.0, 8.0]);
let lt = a.cmp_lt(b);
```

Genuinely computed:

```
Lane-wise comparison, a < b:
  a      = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  b      = [2.0, 2.0, 1.0, 5.0, 5.0, 3.0, 9.0, 8.0]
  a<b raw lanes (bit pattern reinterpreted as f32) = [NaN, 0.0, 0.0, NaN, 0.0, 0.0, NaN, 0.0]
```

Hand-traced lane by lane: `1.0<2.0` true, `2.0<2.0` false, `3.0<1.0` false, `4.0<5.0` true, `5.0<5.0` false, `6.0<3.0` false, `7.0<9.0` true, `8.0<8.0` false — matching the `NaN`/`0.0` pattern above exactly once "true" and "false" are decoded correctly. `a.cmp_lt(b)` does not return one `bool`, or even eight `bool`s — it returns another `f32x8`, where SIMD hardware's actual boolean convention is "all bits set" for true and "all bits clear" for false. Reinterpreted as `f32` for printing, all-bits-set happens to be one of the bit patterns IEEE 754 defines as `NaN`, and all-bits-clear is exactly `0.0` — so `NaN` above means *true*, not "not a number gone wrong." `wide` provides a `.blend()` method (used correctly, lane-wise select) precisely so user code almost never needs to look at this bit pattern directly.

> `[COMMON TRAP]` Seeing `NaN` in a comparison's output and assuming something went wrong is a completely reasonable first reaction — and precisely backwards here. This is SIMD hardware's real boolean representation leaking through `wide`'s otherwise safe API in the one place it's hard to fully hide: a lane-wise comparison has to return *something* of the same vector type it compared, and "all bits set" is what "true, in every bit" looks like once you commit to reusing the same 32-bit lane you'd otherwise store a float in.

## 5.3 The Vectorized-Loop Shape: Main Loop Plus Remainder `[FOUNDATIONAL]`

### Intuition

A delivery truck that only accepts full pallets of 8 boxes at a time: the crew stacks as many complete pallets as the boxes allow, and whatever handful is left over — fewer than 8, by definition, or there'd be one more full pallet — gets carried on by hand, individually. The pallets are the fast, SIMD path; the leftover boxes are the unavoidable scalar path for whatever doesn't divide evenly by the register width.

### Background

Every vectorized loop this book writes from here forward is built from the same two pieces:

```
simd_count = (n / width) * width     # how many elements the main loop covers
remainder  = n - simd_count          # always in [0, width - 1]
```

### Worked Example 5.3.1 — `n = 1003`, `width = 8`, a real remainder

```rust
const WIDTH: usize = 8;

fn vector_add_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let simd_count = (n / WIDTH) * WIDTH;
    let mut i = 0;
    while i < simd_count {
        let va = f32x8::new(a[i..i + WIDTH].try_into().unwrap());
        let vb = f32x8::new(b[i..i + WIDTH].try_into().unwrap());
        out[i..i + WIDTH].copy_from_slice(&(va + vb).to_array());
        i += WIDTH;
    }
    while i < n {
        out[i] = a[i] + b[i];  // remainder loop: whatever's left, scalar
        i += 1;
    }
    out
}
```

Compiled and run as the complete `02_vectorized_loop_shape.rs` further below:

```bash
cargo run --release --bin 02_vectorized_loop_shape
```

Genuinely compiled and run:

```
n = 1003, width = 8
simd_count = (n / width) * width = 1000
remainder  = n - simd_count       = 3
elements checked: 1003, mismatches: 0
first main-loop chunk (i=0..8):
  a[0..8] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  b[0..8] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
  simd_result[0..8]   = [0.0, 0.3, 0.6, 0.90000004, 1.2, 1.5, 1.8000001, 2.1]
  scalar_result[0..8] = [0.0, 0.3, 0.6, 0.90000004, 1.2, 1.5, 1.8000001, 2.1]
remainder tail (i=1000..1003):
  simd_result[1000..1003]   = [300.0, 300.3, 300.6]
  scalar_result[1000..1003] = [300.0, 300.3, 300.6]
```

`1003 / 8 = 125` (integer division truncates), so `simd_count = 125*8 = 1000` and `remainder = 3` — the main loop covers indices `0` through `999` in 125 chunks of 8, and a 3-element scalar tail covers indices `1000`, `1001`, `1002`. All 1003 elements were compared, in full, between the SIMD path and an independent plain scalar loop, and `mismatches: 0` is a genuine count, not an assumption — including the first chunk's `0.90000004` and `1.8000001`, real `f32` rounding artifacts that both paths reproduce identically rather than one path "cleaning up."

> `[COMMON TRAP]` It's tempting to test only with sizes that divide evenly by the width — everything passes, because the remainder loop's body never executes, exactly as Mojo's edition of this same trap warns. This chapter deliberately chose `n = 1003`, not a multiple of 8, specifically so the remainder loop's three real iterations show up in the verified output above rather than staying untested.

## 5.4 SIMD Reduction: Accumulate Wide, Reduce Narrow `[FOUNDATIONAL]`

### Intuition

Eight cash registers ring up sales all day, completely independently, each keeping its own running subtotal. Only once, at closing time, does the manager walk to all eight registers and add their eight subtotals into one final total. The registers never coordinate during business hours — coordination happens exactly once, briefly, at the end.

### Background

```rust
fn sum_simd(data: &[f32]) -> f32 {
    let mut acc = f32x8::ZERO;               // 8 independent running subtotals
    let mut i = 0;
    while i < simd_count {
        acc = acc + f32x8::new(data[i..i+8].try_into().unwrap());  // lane-wise, per chunk
        i += 8;
    }
    let mut total = acc.reduce_add();         // ONE horizontal reduction, at the very end
    // ... remainder loop ...
    total
}
```

### Worked Example 5.4.1 — the first two chunks, and a closed-form check

With `data[i] = i+1` for `i` in `1..=16` and `width = 8`, compiled and run as part of `03_reduction_and_benchmark.rs` further below:

```
chunk 0: data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  running vector_sum = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
chunk 1: data = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
  running vector_sum = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
horizontal reduction of vector_sum = 136
closed-form sum 1..=16 (n(n+1)/2)   = 136
match? true
```

Lane 0 is quietly accumulating `1, 9` (sum `10`); lane 1, `2, 10` (sum `12`); and so on — eight separate partial sums, each covering a different residue class mod 8, computed simultaneously. `reduce_add()` folds those eight partial sums into one, `136`, matching the closed-form `n(n+1)/2 = 16×17/2 = 136` exactly.

### Worked Example 5.4.2 — a genuine benchmark, and an accuracy result this chapter did not go looking for

```bash
cargo run --release --bin 03_reduction_and_benchmark
```

Genuinely compiled and run, on 20 million `f32` values, timing the best of 5 runs each, with an independent `f64`-accumulated reference computed for comparison (not timed — it exists purely to check correctness):

```
n = 20000000 elements, best of 5 runs each
scalar sum:    0.0261 s, result = 452170620
SIMD sum:      0.0096 s, result = 479999420
f64 reference (not timed):  result = 479999420
scalar/SIMD time ratio: 2.71x
scalar result matches f64 reference? false
SIMD result matches f64 reference?   true
scalar error vs reference: 27828800 (5.7977%)
SIMD error vs reference:   0 (0.0000%)
```

This is worth stopping on, because it's not the result this chapter set out to demonstrate — it's a real numerical phenomenon this book's own benchmark surfaced, verified rather than smoothed over. The SIMD sum is genuinely `2.71×` faster (this book measured a `2.66×`–`2.97×` range across repeated runs), which is the expected story: 8 lanes doing the additions in parallel. But the *scalar* sum is also genuinely, measurably wrong — off from the `f64`-accumulated reference by nearly `5.8%`, while the SIMD sum matches that reference bit-for-bit. Neither number was hand-picked; both came directly out of the same benchmark run.

The reason is `f32`'s limited precision (24 bits of mantissa, exact only up to `2^24 ≈ 16.78` million) meeting a 20-million-element sum head-on. A single scalar accumulator summing all 20 million values sequentially grows past `16.78` million well before the loop finishes, and every addition after that point is adding a small value (at most `48.0` here) to an accumulator too large for `f32` to represent the result exactly — the small addend is partially or entirely rounded away, over and over, for millions of iterations, compounding into a real, measurable error. The SIMD version's 8 lane-accumulators each only ever sum roughly `2.5` million values before the single, final `reduce_add()` combines them — each individual lane total stays much smaller for much longer, so each one loses far less precision before the eight are finally combined into one number at the very end.

> `[COMMON TRAP]` It's tempting to treat "SIMD is faster" and "SIMD gave a more accurate answer" as two lucky, unrelated facts about the same benchmark. They're not unrelated — both come from the identical structural change, splitting one long sequential accumulation into eight shorter, independent ones. The 8-way split is what buys the speed (eight adds genuinely execute per instruction), and it's *also* what buys the accuracy (eight partial sums, each accumulating roughly an eighth as many terms, each staying numerically well-behaved roughly eight times longer before the one combining step at the end). Neither a plain scalar loop unrolled by hand nor a naive SIMD loop that reduces after every single chunk (rather than accumulating across all chunks first) would reproduce this same accuracy benefit — it specifically depends on keeping the reduction wide for as long as possible, exactly as this section's own "accumulate wide, reduce narrow" name says.

## 5.5 Why This Chapter Depends on Chapter 3's SoA Argument `[FOUNDATIONAL]`

### Intuition

Every SIMD load this chapter wrote — `f32x8::new(data[i..i+8].try_into().unwrap())` — assumes eight contiguous, tightly-packed `f32` values sit at `data[i]` through `data[i+7]`, ready to be pulled into one register with a single instruction. That assumption isn't automatic; it's exactly the property Chapter 3 spent an entire chapter arguing for.

### Background

Chapter 3.1's `Particle` struct, under Array-of-Structs, packs `vx` between `y`/`z` and `vy`/`vz`/`mass` — a single field, `vx`, is never eight contiguous `f32`s anywhere in that layout; it's one `f32` every 28 bytes. A SIMD load of eight consecutive `vx` values from an AoS array would need eight separate, scattered loads — gather, not vectorized load — defeating the entire premise of this chapter's `f32x8::new(slice[i..i+8].try_into().unwrap())`. Struct-of-Arrays, by contrast, is precisely "every element of one field, contiguous" — which is precisely "exactly what a SIMD load needs, already true by construction."

> `[COMMON TRAP]` It's tempting to read this chapter and Chapter 3 as covering unrelated topics — one about CPU vector instructions, one about struct layout. They're the same argument at two different levels: Chapter 3 established *why* SoA is the right layout for bulk numerical operations (bus utilization, cache-line usage); this chapter shows *one concrete mechanism* — a genuinely 2.7×–3.0× faster, and here, more accurate, vectorized sum — that SoA's contiguity directly enables and AoS directly forecloses. This book's `Tensor` type, SoA since Part 1 by Chapter 3's argument, is exactly what makes every reduction and elementwise kernel from Part 2 onward eligible for the loop shape this chapter just built.

## Complete Runnable Code

### File: `01_lanes_and_intrinsics.rs`

```rust
use wide::{f32x8, CmpLt};

#[target_feature(enable = "avx")]
unsafe fn add_via_raw_intrinsic(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let vc = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), vc);
        out
    }
}

fn main() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    let wide_result = (f32x8::new(a) + f32x8::new(b)).to_array();
    let raw_result = if is_x86_feature_detected!("avx") {
        unsafe { add_via_raw_intrinsic(a, b) }
    } else {
        [0.0; 8]
    };
    println!("wide::f32x8 result:        {:?}", wide_result);
    println!("raw _mm256_add_ps result:  {:?}", raw_result);
    println!("identical? {}", wide_result == raw_result);

    println!();
    println!("CPU feature detection (genuinely queried at runtime):");
    println!("  sse2  = {}", is_x86_feature_detected!("sse2"));
    println!("  sse4.1= {}", is_x86_feature_detected!("sse4.1"));
    println!("  avx   = {}", is_x86_feature_detected!("avx"));
    println!("  avx2  = {}", is_x86_feature_detected!("avx2"));
    println!("  fma   = {}", is_x86_feature_detected!("fma"));
    println!();
    println!("size_of::<wide::f32x4>() = {} bytes (a 128-bit register)", std::mem::size_of::<wide::f32x4>());
    println!("size_of::<wide::f32x8>() = {} bytes (a 256-bit register)", std::mem::size_of::<wide::f32x8>());

    println!();
    println!("Lane-wise comparison, a < b:");
    let cmp_a = f32x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let cmp_b = f32x8::new([2.0, 2.0, 1.0, 5.0, 5.0, 3.0, 9.0, 8.0]);
    let lt = cmp_a.cmp_lt(cmp_b);
    println!("  a      = {:?}", cmp_a.to_array());
    println!("  b      = {:?}", cmp_b.to_array());
    println!("  a<b raw lanes (bit pattern reinterpreted as f32) = {:?}", lt.to_array());
}
```

```bash
cargo run --release --bin 01_lanes_and_intrinsics
```

### File: `02_vectorized_loop_shape.rs`

```rust
use wide::f32x8;

const WIDTH: usize = 8;

fn vector_add_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    assert_eq!(n, b.len());
    let mut out = vec![0.0f32; n];

    let simd_count = (n / WIDTH) * WIDTH;
    let mut i = 0;
    while i < simd_count {
        let va = f32x8::new(a[i..i + WIDTH].try_into().unwrap());
        let vb = f32x8::new(b[i..i + WIDTH].try_into().unwrap());
        let vc = va + vb;
        out[i..i + WIDTH].copy_from_slice(&vc.to_array());
        i += WIDTH;
    }
    // remainder loop: whatever's left, scalar
    while i < n {
        out[i] = a[i] + b[i];
        i += 1;
    }
    out
}

fn vector_add_scalar(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn main() {
    let n: usize = 1003; // deliberately NOT a multiple of 8
    let simd_count = (n / WIDTH) * WIDTH;
    let remainder = n - simd_count;
    println!("n = {n}, width = {WIDTH}");
    println!("simd_count = (n / width) * width = {simd_count}");
    println!("remainder  = n - simd_count       = {remainder}");

    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();

    let simd_result = vector_add_simd(&a, &b);
    let scalar_result = vector_add_scalar(&a, &b);

    let mut mismatches = 0;
    for i in 0..n {
        if simd_result[i] != scalar_result[i] {
            mismatches += 1;
        }
    }
    println!("elements checked: {n}, mismatches: {mismatches}");
    println!("first main-loop chunk (i=0..8):");
    println!("  a[0..8] = {:?}", &a[0..8]);
    println!("  b[0..8] = {:?}", &b[0..8]);
    println!("  simd_result[0..8]   = {:?}", &simd_result[0..8]);
    println!("  scalar_result[0..8] = {:?}", &scalar_result[0..8]);
    println!("remainder tail (i=1000..1003):");
    println!("  simd_result[1000..1003]   = {:?}", &simd_result[1000..1003]);
    println!("  scalar_result[1000..1003] = {:?}", &scalar_result[1000..1003]);
}
```

```bash
cargo run --release --bin 02_vectorized_loop_shape
```

### File: `03_reduction_and_benchmark.rs`

```rust
use std::hint::black_box;
use std::time::Instant;
use wide::f32x8;

const WIDTH: usize = 8;

fn sum_simd(data: &[f32]) -> f32 {
    let n = data.len();
    let simd_count = (n / WIDTH) * WIDTH;

    let mut acc = f32x8::ZERO; // width-wide accumulator: 8 independent running subtotals
    let mut i = 0;
    while i < simd_count {
        let v = f32x8::new(data[i..i + WIDTH].try_into().unwrap());
        acc = acc + v; // lane-wise add, once per chunk -- no reduction yet
        i += WIDTH;
    }
    let mut total = acc.reduce_add(); // ONE horizontal reduction, at the very end
    while i < n {
        total += data[i];
        i += 1;
    }
    total
}

fn sum_scalar(data: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for &v in data {
        total += v;
    }
    total
}

/// A high-precision reference sum, accumulated in f64 and rounded back to f32 only at
/// the very end. Used purely to check which of the two f32 sums above is closer to correct.
fn sum_f64_reference(data: &[f32]) -> f32 {
    let mut total = 0.0f64;
    for &v in data {
        total += v as f64;
    }
    total as f32
}

fn main() {
    // Worked Example: first two chunks of a real sum, data[i] = i+1, width=8
    let small: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let mut acc = f32x8::ZERO;
    for (chunk_idx, chunk) in small.chunks(WIDTH).enumerate() {
        let v = f32x8::new(chunk.try_into().unwrap());
        acc = acc + v;
        println!("chunk {chunk_idx}: data = {:?}", chunk);
        println!("  running vector_sum = {:?}", acc.to_array());
    }
    let reduced = acc.reduce_add();
    let closed_form: f32 = (1..=16).sum::<i32>() as f32; // n(n+1)/2 = 16*17/2 = 136
    println!("horizontal reduction of vector_sum = {}", reduced);
    println!("closed-form sum 1..=16 (n(n+1)/2)   = {}", closed_form);
    println!("match? {}", reduced == closed_form);

    println!();
    // Genuine benchmark: scalar vs SIMD sum, large array, best of 5 runs each
    let n: usize = 20_000_000;
    let data: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.5).collect();

    const RUNS: u32 = 5;
    let mut scalar_best = f64::MAX;
    let mut scalar_result = 0.0f32;
    for _ in 0..RUNS {
        let start = Instant::now();
        let r = sum_scalar(black_box(&data));
        let elapsed = start.elapsed().as_secs_f64();
        black_box(r);
        scalar_result = r;
        scalar_best = scalar_best.min(elapsed);
    }

    let mut simd_best = f64::MAX;
    let mut simd_result = 0.0f32;
    for _ in 0..RUNS {
        let start = Instant::now();
        let r = sum_simd(black_box(&data));
        let elapsed = start.elapsed().as_secs_f64();
        black_box(r);
        simd_result = r;
        simd_best = simd_best.min(elapsed);
    }

    let reference = sum_f64_reference(&data);

    println!("n = {n} elements, best of {RUNS} runs each");
    println!("scalar sum:    {:.4} s, result = {}", scalar_best, scalar_result);
    println!("SIMD sum:      {:.4} s, result = {}", simd_best, simd_result);
    println!("f64 reference (not timed):  result = {}", reference);
    println!("scalar/SIMD time ratio: {:.2}x", scalar_best / simd_best);
    println!("scalar result matches f64 reference? {}", scalar_result == reference);
    println!("SIMD result matches f64 reference?   {}", simd_result == reference);
    println!(
        "scalar error vs reference: {} ({:.4}%)",
        reference - scalar_result,
        100.0 * (reference - scalar_result).abs() as f64 / reference as f64
    );
    println!(
        "SIMD error vs reference:   {} ({:.4}%)",
        reference - simd_result,
        100.0 * (reference - simd_result).abs() as f64 / reference as f64
    );
}
```

```bash
cargo run --release --bin 03_reduction_and_benchmark
```

`Cargo.toml` for all three binaries:

```toml
[package]
name = "rust_ch5"
version = "0.1.0"
edition = "2024"

[dependencies]
wide = "0.7"
```

## Chapter Summary

Stable Rust has no first-class portable SIMD type — `std::simd` is nightly-only, unreachable in this sandbox exactly as `nvptx64-nvidia-cuda` was in Chapter 4 — so this book builds on `std::arch::x86_64`'s raw intrinsics and the `wide` crate's safe wrapper around them, confirmed identical on a real `_mm256_add_ps` call this chapter ran both ways. A SIMD register's width is a genuine, `size_of`-confirmed hardware fact — 16 bytes for `f32x4`, 32 for `f32x8` — and a lane-wise comparison returns another vector, not a scalar `bool`, its true/false lanes stored as SIMD hardware's real all-bits/no-bits convention, which prints as `NaN`/`0.0` when reinterpreted as `f32`. Every vectorized loop in this book follows the same shape: `simd_count = (n/width)*width` elements through a main SIMD loop, and a scalar remainder loop for whatever's left — verified here against `n=1003`, a size chosen deliberately to exercise a genuine 3-element remainder, with zero mismatches across all 1003 elements against an independently-written scalar version. SIMD reduction — accumulating into a wide vector and reducing once at the end — turned out to matter for more than speed alone: this chapter's own 20-million-element benchmark measured a genuine `2.71×` speedup, and, unplanned but independently confirmed against an `f64` reference, a naive scalar sum that was measurably wrong by nearly `5.8%` from `f32` precision loss, against a SIMD sum that matched the reference exactly. None of this chapter's vectorized loads work without Chapter 3's Struct-of-Arrays argument holding first — a contiguous `f32x8` load needs contiguous `f32`s to load, which is exactly what SoA guarantees and AoS forecloses.

## Self-Check Questions

1. Why does this book use `std::arch::x86_64` and `wide` instead of `std::simd::Simd<T, N>`, and what specific, previously-confirmed sandbox limitation is this the same root cause as?
2. `is_x86_feature_detected!("avx")` genuinely returns `true` in this book's own sandbox. What would change about Worked Example 5.1.1's raw-intrinsic function if this book's sandbox lacked AVX, and why would simply removing the `if is_x86_feature_detected!(...)` check around it be unsafe rather than merely wrong?
3. `cmp_a.cmp_lt(cmp_b)` in Worked Example 5.2.2 returns an `f32x8`, not eight `bool`s. Explain what "true" and "false" actually look like as bit patterns in that returned value, and why printing them as `f32` shows `NaN` for true.
4. For `n = 1003` and `width = 8`, compute `simd_count` and `remainder` by hand, and state which indices the main loop covers and which the remainder loop covers.
5. This chapter's SIMD sum was both faster and more accurate than the scalar sum on the same 20-million-element input. Explain why those two outcomes share the same root cause rather than being two unrelated pieces of good luck.

## Where We Go Next

Part 0 has now built every foundational tool this book's `Tensor` type needs before it can exist as more than a struct: ownership and move semantics (Chapter 1), struct layout and dispatch (Chapter 2), the Array-of-Structs versus Struct-of-Arrays choice (Chapter 3) that Chapter 4's coalescing argument and this chapter's vectorized loads both depended on, `cudarc`'s host-side GPU driver model (Chapter 4), and now CPU-side SIMD (Chapter 5) — the two forms of "many lanes, one instruction" this book's `Tensor` operations will draw on, one on the CPU path, one on the GPU path, from Part 1 onward. Part 1 begins there: the `Tensor` struct itself, its SoA-backed `.data` buffer, and the first operations built on top of it.

## Worked Solutions

**1.** `std::simd::Simd<T, N>` requires `#![feature(portable_simd)]`, a nightly-only feature. Getting Started already confirmed this sandbox's toolchain has no reachable nightly channel — `static.rust-lang.org`, where `rustup` would fetch nightly's `rust-std`/`rust-src` components, returns a blocked-network 403 — the exact same root cause Chapter 4's `nvptx64-nvidia-cuda` target compilation hit (that target's kernel ABI also requires nightly). Both are instances of one underlying constraint: this sandbox's stable-only toolchain, not a preference for `std::arch`/`wide` on their own merits.

**2.** Without AVX, `_mm256_add_ps` and the other `_mm256_*` intrinsics are simply not valid instructions for the CPU to execute — the function would need to be rewritten around `_mm_add_ps` (SSE, 128-bit, `f32x4`-sized) or a scalar fallback instead. Removing the `is_x86_feature_detected!("avx")` check would not just produce a wrong *answer* — Rust's `#[target_feature(enable = "avx")]` functions compile to code that assumes AVX instructions are legal on whatever CPU eventually runs them; calling one on a CPU that doesn't support AVX is undefined behavior (in practice, typically an illegal-instruction crash), which is exactly the class of problem runtime feature detection exists to prevent, as opposed to an ordinary logic bug that would merely compute the wrong number.

**3.** SIMD comparison instructions produce, per lane, either all bits set to `1` (true) or all bits set to `0` (false) — not a single bit, and not a separate "boolean lane type." When those all-`1` or all-`0` 32-bit patterns are read back as IEEE 754 `f32` values (which is what `to_array()` and `{:?}` do, since `f32x8`'s lanes are typed as `f32`), an all-`1`-bits pattern happens to fall inside the range of bit patterns IEEE 754 reserves for `NaN`, while all-`0`-bits is exactly the bit pattern for `0.0`. So `NaN` in this printed output means "true," and `0.0` means "false" — a fact about how the comparison's result bits happen to decode as floats, not a computation that produced an actual not-a-number value.

**4.** `simd_count = (1003 / 8) * 8`. Integer division truncates: `1003 / 8 = 125` (since `125 × 8 = 1000 ≤ 1003 < 1008 = 126 × 8`), so `simd_count = 125 × 8 = 1000`. `remainder = 1003 - 1000 = 3`. The main SIMD loop covers indices `0` through `999` (125 chunks of 8 elements each), and the scalar remainder loop covers indices `1000`, `1001`, and `1002` — exactly the three indices this chapter's own verified run traced explicitly.

**5.** Both outcomes come from the same structural change: replacing one long sequential `f32` accumulation with eight shorter, independent ones that are only combined once, at the very end. The speed benefit follows because eight lane-additions genuinely execute per SIMD instruction rather than one addition per scalar instruction. The accuracy benefit follows from a different but related consequence of the same split: each of the eight lane-accumulators only ever sums roughly an eighth as many values (here, about 2.5 million rather than 20 million) before the single `reduce_add()` combines them, so each individual running total stays numerically well-behaved — well under the `2^24`-ish threshold where `f32` starts silently dropping small additions — for roughly eight times longer than the one scalar accumulator does. Splitting the work eight ways is simultaneously what makes it fast (parallel lanes) and what makes it accurate (each lane's own accumulator error grows more slowly), rather than being two independent properties of the same code.
