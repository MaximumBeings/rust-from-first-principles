# Chapter 8: Tensor Creation Functions — Factories, Random Generation, and I/O

> "A `Tensor` with an allocated, uninitialized buffer is a promise, not a value — a fresh `Vec<f32>` hands back memory, not necessarily zeros, unless you specifically ask it to. Every one of this chapter's functions exists to keep that promise a specific way: a predictable ramp of numbers, a reproducible stream of randomness, or the exact bytes some other program already wrote to disk. Two of the three genuine bugs this chapter finds along the way were sitting in code that looked completely reasonable until the moment a small seed or a missing trailing newline exposed them — and a fourth, smaller discovery turned up in nothing more exotic than the order four print arguments get evaluated in."

**What you will understand by the end of this chapter:**

- `arange` and `linspace`, both plain host functions following Chapter 4.5's one-index-one-element broadcast pattern, and the genuine distinction between them — `arange` never reaches its stop value, `linspace` always includes both ends
- `eye`, generalized with a diagonal offset, and why "the diagonal" is really just one specific case of "every position where `col - row` equals some fixed constant"
- A small, deterministic pseudorandom generator, chosen specifically because this book's GPU verification is still on hold pending real hardware — and two genuine, discovered flaws in the naive version of it: identical seeds silently producing identical streams, and a small seed producing a measurably weak first draw
- Fisher-Yates shuffling and inverse-CDF sampling, both built on top of that same generator and checked against real, computable invariants — a permutation's element sum, and a large sample's empirical frequency
- A genuine off-by-one bug in a CSV row-counting function, caught by testing it against a file that happens not to end in a newline — and why trusting a `Tensor`'s own recorded shape is more robust than re-deriving it from a text file's bytes
- A small, genuinely discovered difference in how Rust and C++ order side-effecting function-call arguments, and why Rust's guarantee here is a real safety property, not a stylistic footnote

**What you need to know first:**

- Chapter 4.5 (the broadcast kernel pattern: one thread, one output element, guarded by a boundary check) — every function in Section 8.1 is a direct host-side instance of it
- Chapter 6.1–6.3 (`TensorShape`, `TensorStrides`, and `Tensor`'s constructor) — this chapter fills the buffers Chapter 6 learned to allocate
- Chapter 3.4's cross-check discipline (compute the same result two independent ways and confirm they agree) — Sections 8.1 through 8.3 all lean on it
- If you've read the Mojo edition: this chapter follows its Chapter 8 section-for-section (factories, random generation, data import/export), including its habit of finding a genuine bug in Section 8.3's parsing code rather than describing one hypothetically. The random-number section differs in mechanism from both siblings: this chapter builds the same small custom generator the CUDA edition does, for the same reason — its own GPU verification discipline (Chapter 4's `cudarc` panics) rules out relying on a device-only library here too.

## 8.1 Factory Functions: `arange`, `linspace`, and `eye` `[FOUNDATIONAL]`

### Intuition

Chapter 6's `Tensor` constructor decides a shape and allocates a `Vec<f32>` for it; it says nothing about what ends up in that memory. The three functions this section builds are all the same shape of thing Chapter 4.5 already named the broadcast pattern — one index, one output element — with the element's value computed from nothing but its own position, no input tensor required at all.

### Background

| Function | Per-element formula | Includes the stop value? |
|---|---|---|
| `arange(start, step, n)` | `data[i] = start + i * step` | No — `n` elements starting at `start`, stopping one step short of `start + n*step` |
| `linspace(start, stop, n)` | `data[i] = start + i * (stop - start)/(n-1)` | Yes, by construction — `i=0` gives exactly `start`, `i=n-1` gives exactly `stop` |
| `eye(n, k)` | `data[i*n+j] = 1` if `j - i == k`, else `0` | N/A — `k=0` is the main diagonal, `k=1`/`k=-1` shift it one column right/left |

### Worked Example 8.1.1 — `arange` and `linspace`, side by side

```rust
// data[i] = start + i*step, for i in [0, n) -- one index, one element, the
// exact broadcast pattern Chapter 4.5 established.
fn arange(start: f32, step: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| start + (i as f32) * step).collect()
}

// n points from start to stop INCLUSIVE of both ends -- step = (stop-start)/(n-1).
fn linspace(start: f32, stop: f32, n: usize) -> Vec<f32> {
    let step = (stop - start) / ((n - 1) as f32);
    (0..n).map(|i| start + (i as f32) * step).collect()
}
```

Compiled and run as part of the complete `01_factory_functions.rs` further below:

```bash
rustc --edition 2024 -O 01_factory_functions.rs -o 01_factory_functions
./01_factory_functions
```

Genuinely compiled and run:

```
arange(start=0, step=2, n=5) = [0.0, 2.0, 4.0, 6.0, 8.0]
linspace(0, 1, n=5) = [0.00, 0.25, 0.50, 0.75, 1.00]  (both ends included)
```

`arange(0.0, 2.0, 5)` produces `0, 2, 4, 6, 8` — five values, the fifth and last one being `8`, one full `step` short of where a sixth value (`10`) would land. `linspace(0.0, 1.0, 5)` produces `0.00, 0.25, 0.50, 0.75, 1.00` — the same five-element count, but computed from a step size derived specifically to land exactly on `1.0` at `i=4`: `step = (1-0)/(5-1) = 0.25`.

> `[COMMON TRAP]` `linspace`'s formula divides by `n - 1`, which is a genuine problem for `n = 1` in two ways at once: `usize` subtraction underflows (`0usize - 1` panics in a debug build and wraps to a huge value in `--release`, rather than cleanly producing a negative number the way C++'s signed `int - 1` would), and even setting that aside, "1 point from 0 to 10" has no well-defined step size — only a single point that could reasonably be `start`, `stop`, or their midpoint depending on convention. It's a genuinely different failure mode from `arange`, which has no such restriction — `arange(0.0, 1.0, 1)` is a perfectly well-defined single-element `[0.0]`, because `arange`'s formula never subtracts from or divides by `n` at all. A real `linspace` implementation checks for `n == 1` (or `n == 0`) before ever reaching the general formula; this chapter's version deliberately leaves that check out to keep the core formula visible, the same choice the CUDA edition made.

### Worked Example 8.1.2 — `eye`, with a diagonal offset

```rust
// Identity matrix, flattened row-major, with an optional diagonal offset k:
// k=0 is the main diagonal, k=1 is the superdiagonal (one column right),
// k=-1 is the subdiagonal (one column left).
fn eye(n: usize, k: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if (j as i32) - (i as i32) == k {
                out[i * n + j] = 1.0;
            }
        }
    }
    out
}
```

Compiled and run as part of the complete `01_factory_functions.rs` further below:

```bash
rustc --edition 2024 -O 01_factory_functions.rs -o 01_factory_functions
./01_factory_functions
```

Genuinely compiled and run:

```
eye(4, k=0) (main diagonal):
  [1 0 0 0]
  [0 1 0 0]
  [0 0 1 0]
  [0 0 0 1]

eye(4, k=1) (superdiagonal):
  [0 1 0 0]
  [0 0 1 0]
  [0 0 0 1]
  [0 0 0 0]
```

`k=0` reproduces the ordinary identity matrix — every position where `col == row` gets `1`. `k=1` shifts that condition to `col - row == 1`, which is satisfied by `(0,1)`, `(1,2)`, `(2,3)` — one column right of the main diagonal — and by nothing in the last row, since row `3`'s corresponding column, `4`, doesn't exist in a `4×4` matrix. `(j as i32) - (i as i32)` deliberately casts both loop indices to a signed type before subtracting — `i` and `j` are `usize`, and computing `j - i` directly would panic the moment `j < i` (an unsigned subtraction going negative), exactly the kind of trap `eye`'s own subdiagonal case (`k=-1`) would trigger immediately if the cast were left out.

## 8.2 Random Tensors: A Small Generator, and Two Genuine Bugs in It `[FOUNDATIONAL]`

### Intuition

A production PRNG library would be the obvious choice here, but this book's own GPU-verification discipline (Chapter 4's genuine `cudarc` panics from a missing `libcuda.so`/`libnvrtc.so`) means any GPU-side random generation still can't be genuinely run and checked in this sandbox. This section instead builds a small xorshift-style generator by hand, deliberately simple enough to trace, and then genuinely discovers two real weaknesses in it worth knowing about in *any* PRNG, not just this one.

### Background

| | A production RNG library | `SimpleRng` (this section) |
|---|---|---|
| Quality | Production-grade, extensively tested | A minimal xorshift64 core, built for this chapter's tracing, not for real statistical work |
| Verifiable in this sandbox? | Depends — GPU-backed generation still cannot be genuinely run here | Yes — every number in this section's worked examples is a real, computed value |

### Worked Example 8.2.1 — the reproducibility trap

```rust
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed ^ 0x9E3779B97F4A7C15u64 }
    }
    fn next_uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 40) & 0xFFFFFF) as f32 / (1u32 << 24) as f32
    }
}
```

Compiled and run as part of the complete `02_random_generation.rs` further below:

```bash
rustc --edition 2024 -O 02_random_generation.rs -o 02_random_generation
./02_random_generation
```

Genuinely compiled and run:

```
rng_a(seed=42) first 4 draws: 0.8598 0.7694 0.9378 0.0953
rng_b(seed=42) first 4 draws: 0.8598 0.7694 0.9378 0.0953  (identical -- same seed)
```

Two entirely separate `SimpleRng` instances, constructed from the identical seed `42`, produce byte-for-byte identical sequences — not a bug in isolation (a deterministic generator is *supposed* to be reproducible from a fixed seed, and that's exactly what makes Worked Example 8.2.4's frequency check trustworthy), but a genuine trap the moment two different tensors in the same program are seeded from the same fixed constant by accident, expecting them to hold independent random values and silently getting identical ones instead.

> `[COMMON TRAP]` This exact same generator, same seed, same four draws, produces its output in the *opposite* printed order in this book's CUDA edition: `0.0953 0.9378 0.7694 0.8598` there, versus `0.8598 0.7694 0.9378 0.0953` here — genuinely re-verified directly (see the box below), not a typo in either book. C++ leaves the evaluation order of a function call's arguments unspecified, and `printf("%.4f %.4f %.4f %.4f\n", rng.next_uniform(), rng.next_uniform(), rng.next_uniform(), rng.next_uniform())` calls `next_uniform()` four times as a *side effect* of evaluating those arguments — this book's CUDA toolchain evidently evaluates them right-to-left, so the argument printed *first* is actually the *last* draw taken, and vice versa. Rust's language specification, by contrast, guarantees left-to-right evaluation for every function call and macro invocation, `println!` included — confirmed directly by instrumenting four argument expressions with a shared counter and observing them fire in call order, 1 through 4, with no exceptions. This is a small difference, but a real one: the same side-effecting logic embedded in an argument list is a source of genuinely unspecified, compiler-dependent behavior in C++, and a source of exactly one guaranteed behavior in Rust.

### Worked Example 8.2.2 — a genuinely discovered small-seed weakness, and its fix

```rust
// NAIVE version: the seed is used directly as the initial state.
struct NaiveRng {
    state: u64,
}

impl NaiveRng {
    fn new(seed: u64) -> Self {
        NaiveRng { state: seed }
    }
    fn next_uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 40) & 0xFFFFFF) as f32 / (1u32 << 24) as f32
    }
}
```

Compiled and run as part of the complete `02_random_generation.rs` further below:

```bash
rustc --edition 2024 -O 02_random_generation.rs -o 02_random_generation
./02_random_generation
```

Genuinely compiled and run:

```
small-seed quality, naive vs. pre-mixed constructor, seed=42:
NaiveRng(42) first draw:  0.000000  (measurably weak -- see explanation below)
SimpleRng(42) first draw: 0.859794  (pre-mixed seed, high-quality from draw 1)
```

This is not a hypothetical bug — it's exactly what this book's own testing pass turned up while verifying this chapter's code (matching the identical, independently-rediscovered weakness the CUDA edition found in the same algorithm). `NaiveRng`'s constructor stores the seed directly as `state`; a small integer like `42` occupies only `state`'s lowest 6 bits (`42 = 0b101010`), leaving every bit above that at `0`, and `next_uniform()` reads its output from bits `40`–`63` — bits that a *single* round of `xorshift64` hasn't yet had the chance to spread any of the seed's original entropy into. The result: the very first draw from a small, freshly-seeded `NaiveRng` is `0.0`, not a rare or unlucky value but a direct, mechanical consequence of where the seed's few bits started out. `SimpleRng`'s fix is one extra step in the constructor — XOR the seed against a fixed, odd 64-bit constant before it becomes the initial state — a standard technique (sometimes called splitmix-style seed mixing) that immediately spreads a small seed's few bits across the entire 64-bit word, before `next_uniform()` is ever called at all.

> `[COMMON TRAP]` It's tempting to treat "the numbers look random after a few draws" as sufficient testing for a PRNG, since `NaiveRng`'s *second* and later draws in an actual run are statistically unremarkable — this chapter caught the weakness specifically by checking the *first* draw from a *small* seed, a combination easy to skip if a test suite only ever checks generators seeded from, say, the current time (a large, effectively random 64-bit value already, where this particular weakness is invisible). The lesson generalizes past random number generation: a bug that only shows up for a specific, easy-to-overlook input (a small seed, Section 8.3's missing trailing newline) is not the same as a rare bug — it's a common bug hiding behind an untested case.

### Worked Example 8.2.3 — Fisher-Yates shuffle, traced step by step

```rust
let mut arr = [10, 20, 30, 40, 50];
let mut shuffle_rng = SimpleRng::new(7);
for i in (1..5).rev() {
    let mut j = (shuffle_rng.next_uniform() * (i as f32 + 1.0)) as usize; // j drawn from [0, i], NOT [0, n)
    if j > i {
        j = i;
    }
    arr.swap(i, j);
}
```

Compiled and run as part of the complete `02_random_generation.rs` further below:

```bash
rustc --edition 2024 -O 02_random_generation.rs -o 02_random_generation
./02_random_generation
```

Genuinely compiled and run:

```
Fisher-Yates shuffle, traced (using the fixed SimpleRng):
before: [10,20,30,40,50]
  i=4, j=4 -> [10,20,30,40,50]
  i=3, j=0 -> [40,20,30,10,50]
  i=2, j=2 -> [40,20,30,10,50]
  i=1, j=0 -> [20,40,30,10,50]
after:  [20,40,30,10,50]
same 5 elements, just reordered? sum before=150, sum after=150, match=true
```

Five iterations shrink the "still needs shuffling" range one element at a time: `i=4` draws `j` from `[0,4]` (all five positions still eligible), swaps position `4` with itself (a no-op this time, but a legal outcome), then `i=3` draws from the now-*smaller* range `[0,3]` — position `4` is excluded, because it's already been finalized. `sum before = sum after = 150` confirms the array still holds the identical five elements, just reordered — a real, checkable invariant for *any* correct shuffle, though not a proof of uniform randomness by itself. `arr.swap(i, j)` replaces the CUDA edition's manual three-line temp-variable swap with a single call into `[T]`'s own standard library method — a genuine simplification available because Rust's slice type ships this operation directly, not a change in what the algorithm does.

> `[COMMON TRAP]` Drawing `j` from the *full* range `[0, n)` on every iteration instead of the *shrinking* range `[0, i]` is a well-known, genuinely common way to implement Fisher-Yates incorrectly — it still produces *a* permutation (the sum-invariant check above would still pass), but not a *uniformly random* one; some final orderings become measurably more likely than others. The bug is invisible to a check that only verifies "still the same elements," which is exactly why real test suites for shuffle algorithms check the *distribution* of many repeated shuffles, not just one run's output.

### Worked Example 8.2.4 — inverse-CDF sampling, checked against a large-sample frequency

```rust
let cumulative = [0.2f32, 0.5, 1.0]; // running sum of target probabilities [0.2, 0.3, 0.5]

fn sample_category(u: f32, cumulative: &[f32]) -> usize {
    for (c, &bound) in cumulative.iter().enumerate() {
        if u < bound {
            return c;
        }
    }
    cumulative.len() - 1
}
```

Compiled and run as the complete `03_inverse_cdf_sampling.rs` further below:

```bash
rustc --edition 2024 -O 03_inverse_cdf_sampling.rs -o 03_inverse_cdf_sampling
./03_inverse_cdf_sampling
```

Genuinely compiled and run:

```
target probabilities: [0.2, 0.3, 0.5]
cumulative:           [0.2, 0.5, 1.0]

10000 draws, empirical frequency vs. target:
  category 0: count=1974, frequency=0.1974, target=0.2000, |diff|=0.0026
  category 1: count=2994, frequency=0.2994, target=0.3000, |diff|=0.0006
  category 2: count=5032, frequency=0.5032, target=0.5000, |diff|=0.0032
```

`sample_category` finds the first cumulative bucket a uniform draw `u` falls under — `u < 0.2` lands in category 0, `0.2 ≤ u < 0.5` lands in category 1, and everything else lands in category 2 — so a category's *share* of the `[0,1)` interval it owns directly determines how often it gets drawn over many samples. `10000` draws from `SimpleRng::new(2024)` land within `0.0032` of every target probability, a real, deterministic, exactly-reproducible result (this exact seed always produces this exact count) rather than a claim about randomness in the abstract — and, since `sample_category` takes `cumulative` as an ordinary function argument evaluated once per call rather than four side-effecting arguments in one call, Worked Example 8.2.1's evaluation-order question never arises here at all.

## 8.3 CSV Import/Export: A Genuine Off-by-One, Found by Testing `[FOUNDATIONAL]`

### Intuition

Getting a `Tensor`'s values to and from a text file — model weights saved from another framework, a small dataset, a debugging dump — sounds like it should be one of this book's simplest operations. Section 8.2 already showed one plausible-looking function harboring a real bug; this section finds a second one, in code that passes every test you'd think to write until you specifically try a file that doesn't end the way most editors' files do.

### Background

| | A file ending in `\n` (the common case) | A file with no trailing `\n` on its last line |
|---|---|---|
| Newline bytes present | One per row | One *fewer* than the number of rows |
| Naive "count the `\n` bytes" row count | Correct | Wrong — undercounts by exactly 1 |
| Why this happens in practice | — | Some tools, and many hand-edited files, simply don't add a final newline |

### Worked Example 8.3.1 — the bug, genuinely triggered

```rust
fn count_rows_naive(filename: &str) -> i32 {
    let mut contents = Vec::new();
    File::open(filename).expect("open failed").read_to_end(&mut contents).expect("read failed");
    contents.iter().filter(|&&b| b == b'\n').count() as i32
}
```

Compiled and run as part of the complete `04_tensor_csv_io.rs` further below:

```bash
rustc --edition 2024 -O 04_tensor_csv_io.rs -o 04_tensor_csv_io
./04_tensor_csv_io
```

Genuinely compiled and run:

```
file WITH trailing newline (3 real rows):
  count_rows_naive  = 3
  count_rows_fixed  = 3

file WITHOUT trailing newline (still 3 real rows):
  count_rows_naive  = 2  <- BUG: undercounts by 1
  count_rows_fixed  = 3  <- correct
```

Two files, both genuinely written by this program, holding the identical 3 rows of data — one written with a trailing `\n` after every row (Section 8.1's convention, and this section's own `write_csv` function), one deliberately written as the raw bytes `b"1.0,2.0\n3.0,4.0\n5.0,6.0"` — no trailing `\n` after the last row. `count_rows_naive` reports `3` for the first file and `2` for the second, an undercount caused by exactly the mechanism the Background table names: three rows means two internal newlines plus, ordinarily, a final one — and the second file simply doesn't have that final one.

### Worked Example 8.3.2 — the fix, and a full round trip

```rust
fn count_rows_fixed(filename: &str) -> i32 {
    let mut contents = Vec::new();
    File::open(filename).expect("open failed").read_to_end(&mut contents).expect("read failed");
    let mut count = contents.iter().filter(|&&b| b == b'\n').count() as i32;
    if !contents.is_empty() && *contents.last().unwrap() != b'\n' {
        count += 1; // the missing final newline's row
    }
    count
}
```

Compiled and run as part of the complete `04_tensor_csv_io.rs` further below:

```bash
rustc --edition 2024 -O 04_tensor_csv_io.rs -o 04_tensor_csv_io
./04_tensor_csv_io
```

Genuinely compiled and run:

```
full round trip: write 6 known values, read them back, compare:
  wrote:      [1.0,2.0,3.0,4.0,5.0,6.0]
  read back:  [1.0,2.0,3.0,4.0,5.0,6.0]  (n_read=6)
  round trip exact match? true
```

`count_rows_fixed` checks the last byte in the buffer directly rather than tracking state one character at a time through a stream; if the file has any content at all and that last byte isn't a newline, it counts one more row for the unterminated final line — genuinely correct on both test files above. The round trip — `write_csv` followed immediately by `read_csv` on the same 6 known values — confirms the actual data survives the trip exactly, parsed-float for written-float, which is the check that actually matters for a `Tensor`'s values; row-counting is only ever a means to that end.

> `[COMMON TRAP]` This entire bug exists because `count_rows_naive` tries to *rediscover* a file's row count by scanning its bytes — information a `Tensor` already has, unambiguously, in its own `shape`. Every real import path this book's later chapters use passes the expected shape in *before* reading (exactly `read_csv`'s `max_values` parameter above), rather than asking the file to reveal its own dimensions through a heuristic that a missing trailing newline, an extra blank line, or a stray trailing comma can each independently break. Row-counting from raw bytes is occasionally unavoidable — the very first time an entirely unfamiliar file is opened — but it should never be trusted as an ongoing substitute for a shape the program is already supposed to know.

## 8.4 Complete Runnable Code

### File: `01_factory_functions.rs`

```rust
// data[i] = start + i*step, for i in [0, n) -- one index, one element, the
// exact broadcast pattern Chapter 4.5 established (here run as an ordinary
// host loop, since this book's Tensor is Vec<f32>-backed until Chapter 10).
fn arange(start: f32, step: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| start + (i as f32) * step).collect()
}

// n points from start to stop INCLUSIVE of both ends -- step = (stop-start)/(n-1).
fn linspace(start: f32, stop: f32, n: usize) -> Vec<f32> {
    let step = (stop - start) / ((n - 1) as f32);
    (0..n).map(|i| start + (i as f32) * step).collect()
}

// Identity matrix, flattened row-major, with an optional diagonal offset k:
// k=0 is the main diagonal, k=1 is the superdiagonal (one column right),
// k=-1 is the subdiagonal (one column left).
fn eye(n: usize, k: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if (j as i32) - (i as i32) == k {
                out[i * n + j] = 1.0;
            }
        }
    }
    out
}

fn main() {
    let ar = arange(0.0, 2.0, 5);
    println!(
        "arange(start=0, step=2, n=5) = [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
        ar[0], ar[1], ar[2], ar[3], ar[4]
    );

    let ls = linspace(0.0, 1.0, 5);
    println!(
        "linspace(0, 1, n=5) = [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}]  (both ends included)",
        ls[0], ls[1], ls[2], ls[3], ls[4]
    );

    let id = eye(4, 0);
    println!("\neye(4, k=0) (main diagonal):");
    for i in 0..4 {
        println!(
            "  [{:.0} {:.0} {:.0} {:.0}]",
            id[i * 4], id[i * 4 + 1], id[i * 4 + 2], id[i * 4 + 3]
        );
    }

    let sup = eye(4, 1);
    println!("\neye(4, k=1) (superdiagonal):");
    for i in 0..4 {
        println!(
            "  [{:.0} {:.0} {:.0} {:.0}]",
            sup[i * 4], sup[i * 4 + 1], sup[i * 4 + 2], sup[i * 4 + 3]
        );
    }
}
```

```bash
rustc --edition 2024 -O 01_factory_functions.rs -o 01_factory_functions
./01_factory_functions
```

Produces exactly the output shown in Worked Examples 8.1.1 and 8.1.2 above.

### File: `02_random_generation.rs`

```rust
// NAIVE version: the seed is used directly as the initial state. Kept here
// only to demonstrate a real, discovered weakness -- see main() below.
struct NaiveRng {
    state: u64,
}

impl NaiveRng {
    fn new(seed: u64) -> Self {
        NaiveRng { state: seed }
    }
    fn next_uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 40) & 0xFFFFFF) as f32 / (1u32 << 24) as f32
    }
}

// FIXED version: the seed is pre-mixed (XORed with a fixed odd constant,
// the standard splitmix-style trick) before becoming the initial state.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed ^ 0x9E3779B97F4A7C15u64 }
    }
    fn next_uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 40) & 0xFFFFFF) as f32 / (1u32 << 24) as f32
    }
}

fn main() {
    // The REPRODUCIBILITY trap: two RNG instances constructed with the SAME
    // seed produce the IDENTICAL sequence -- easy to trigger by accident if
    // every tensor in a batch is seeded from the same fixed constant instead
    // of a per-tensor seed.
    let mut rng_a = SimpleRng::new(42);
    let mut rng_b = SimpleRng::new(42);
    println!(
        "rng_a(seed=42) first 4 draws: {:.4} {:.4} {:.4} {:.4}",
        rng_a.next_uniform(), rng_a.next_uniform(), rng_a.next_uniform(), rng_a.next_uniform()
    );
    println!(
        "rng_b(seed=42) first 4 draws: {:.4} {:.4} {:.4} {:.4}  (identical -- same seed)",
        rng_b.next_uniform(), rng_b.next_uniform(), rng_b.next_uniform(), rng_b.next_uniform()
    );

    // The SMALL-SEED trap, genuinely discovered while testing this chapter's
    // code: a tiny numeric seed like 42 occupies only the low bits of a
    // 64-bit state, and xorshift needs a few iterations to spread that
    // entropy into the HIGH bits next_uniform() actually reads.
    println!("\nsmall-seed quality, naive vs. pre-mixed constructor, seed=42:");
    let mut naive = NaiveRng::new(42);
    println!(
        "NaiveRng(42) first draw:  {:.6}  (measurably weak -- see explanation below)",
        naive.next_uniform()
    );
    let mut fixed = SimpleRng::new(42);
    println!(
        "SimpleRng(42) first draw: {:.6}  (pre-mixed seed, high-quality from draw 1)",
        fixed.next_uniform()
    );

    // Fisher-Yates shuffle, traced: walk i from the LAST index down to 1,
    // swap element i with a uniformly random element in [0, i].
    println!("\nFisher-Yates shuffle, traced (using the fixed SimpleRng):");
    let mut arr = [10, 20, 30, 40, 50];
    let mut shuffle_rng = SimpleRng::new(7);
    println!("before: [{},{},{},{},{}]", arr[0], arr[1], arr[2], arr[3], arr[4]);
    for i in (1..5).rev() {
        let mut j = (shuffle_rng.next_uniform() * (i as f32 + 1.0)) as usize;
        if j > i {
            j = i; // guard the rare rounding case where next_uniform() rounds up to exactly 1.0
        }
        arr.swap(i, j);
        println!("  i={}, j={} -> [{},{},{},{},{}]", i, j, arr[0], arr[1], arr[2], arr[3], arr[4]);
    }
    println!("after:  [{},{},{},{},{}]", arr[0], arr[1], arr[2], arr[3], arr[4]);

    let sum_before = 10 + 20 + 30 + 40 + 50;
    let sum_after: i32 = arr.iter().sum();
    println!(
        "same 5 elements, just reordered? sum before={}, sum after={}, match={}",
        sum_before, sum_after, sum_before == sum_after
    );
}
```

```bash
rustc --edition 2024 -O 02_random_generation.rs -o 02_random_generation
./02_random_generation
```

Produces exactly the output shown in Worked Examples 8.2.1, 8.2.2, and 8.2.3 above.

### File: `03_inverse_cdf_sampling.rs`

```rust
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed ^ 0x9E3779B97F4A7C15u64 }
    }
    fn next_uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 40) & 0xFFFFFF) as f32 / (1u32 << 24) as f32
    }
}

// Sample one of 3 categories with target probabilities [0.2, 0.3, 0.5] by
// drawing u in [0,1) and finding which cumulative bucket it falls into --
// inverse-CDF sampling from a discrete distribution.
fn sample_category(u: f32, cumulative: &[f32]) -> usize {
    for (c, &bound) in cumulative.iter().enumerate() {
        if u < bound {
            return c;
        }
    }
    cumulative.len() - 1 // guards float rounding landing exactly at 1.0
}

fn main() {
    let target_probs = [0.2f32, 0.3, 0.5];
    let cumulative = [0.2f32, 0.5, 1.0]; // running sum of target_probs

    println!(
        "target probabilities: [{:.1}, {:.1}, {:.1}]",
        target_probs[0], target_probs[1], target_probs[2]
    );
    println!(
        "cumulative:           [{:.1}, {:.1}, {:.1}]",
        cumulative[0], cumulative[1], cumulative[2]
    );

    let n = 10000;
    let mut counts = [0i32; 3];
    let mut rng = SimpleRng::new(2024);
    for _ in 0..n {
        let u = rng.next_uniform();
        let c = sample_category(u, &cumulative);
        counts[c] += 1;
    }

    println!("\n{} draws, empirical frequency vs. target:", n);
    for c in 0..3 {
        let freq = counts[c] as f32 / n as f32;
        println!(
            "  category {}: count={}, frequency={:.4}, target={:.4}, |diff|={:.4}",
            c, counts[c], freq, target_probs[c], (freq - target_probs[c]).abs()
        );
    }
}
```

```bash
rustc --edition 2024 -O 03_inverse_cdf_sampling.rs -o 03_inverse_cdf_sampling
./03_inverse_cdf_sampling
```

Produces exactly the output shown in Worked Example 8.2.4 above.

### File: `04_tensor_csv_io.rs`

```rust
use std::fs::File;
use std::io::{Read, Write};

// Write a row-major host buffer out as comma-separated text, one row per line.
fn write_csv(filename: &str, data: &[f32], rows: usize, cols: usize) {
    let mut f = File::create(filename).expect("create failed");
    for r in 0..rows {
        let mut line = String::new();
        for c in 0..cols {
            line.push_str(&format!("{:.1}", data[r * cols + c]));
            if c < cols - 1 {
                line.push(',');
            }
        }
        line.push('\n');
        f.write_all(line.as_bytes()).expect("write failed");
    }
}

// Read comma-separated floats back into a flat, row-major buffer. Assumes
// the caller already knows rows*cols (Chapter 6's Tensor always does --
// its shape is never a mystery the way an arbitrary external file's is).
fn read_csv(filename: &str, max_values: usize) -> (Vec<f32>, usize) {
    let mut contents = String::new();
    File::open(filename).expect("open failed").read_to_string(&mut contents).expect("read failed");
    let mut out = Vec::new();
    for tok in contents.split(|ch: char| ch == ',' || ch == '\n' || ch == '\r') {
        if out.len() >= max_values {
            break;
        }
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if let Ok(v) = tok.parse::<f32>() {
            out.push(v);
        }
    }
    let n = out.len();
    (out, n)
}

// The NAIVE row counter: count newline characters. This has a real bug --
// see main() below for the discovery.
fn count_rows_naive(filename: &str) -> i32 {
    let mut contents = Vec::new();
    File::open(filename).expect("open failed").read_to_end(&mut contents).expect("read failed");
    contents.iter().filter(|&&b| b == b'\n').count() as i32
}

// The FIXED row counter: count newlines, but if the file has any content at
// all and does NOT end with a newline, that last, unterminated line is a
// real row too -- count it.
fn count_rows_fixed(filename: &str) -> i32 {
    let mut contents = Vec::new();
    File::open(filename).expect("open failed").read_to_end(&mut contents).expect("read failed");
    let mut count = contents.iter().filter(|&&b| b == b'\n').count() as i32;
    if !contents.is_empty() && *contents.last().unwrap() != b'\n' {
        count += 1; // the missing final newline's row
    }
    count
}

fn main() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows, 2 cols

    write_csv("with_trailing_newline.csv", &data, 3, 2);

    // Deliberately write a version with NO trailing newline on the last
    // line, exactly the way some tools (and some hand-edited files) do.
    let mut f = File::create("no_trailing_newline.csv").expect("create failed");
    f.write_all(b"1.0,2.0\n3.0,4.0\n5.0,6.0").expect("write failed"); // note: no trailing \n
    drop(f);

    println!("file WITH trailing newline (3 real rows):");
    println!("  count_rows_naive  = {}", count_rows_naive("with_trailing_newline.csv"));
    println!("  count_rows_fixed  = {}", count_rows_fixed("with_trailing_newline.csv"));

    println!("\nfile WITHOUT trailing newline (still 3 real rows):");
    println!("  count_rows_naive  = {}  <- BUG: undercounts by 1", count_rows_naive("no_trailing_newline.csv"));
    println!("  count_rows_fixed  = {}  <- correct", count_rows_fixed("no_trailing_newline.csv"));

    println!("\nfull round trip: write 6 known values, read them back, compare:");
    let (roundtrip, n_read) = read_csv("with_trailing_newline.csv", 6);
    println!(
        "  wrote:      [{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}]",
        data[0], data[1], data[2], data[3], data[4], data[5]
    );
    println!(
        "  read back:  [{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}]  (n_read={})",
        roundtrip[0], roundtrip[1], roundtrip[2], roundtrip[3], roundtrip[4], roundtrip[5], n_read
    );
    let all_match = (0..6).all(|i| (data[i] - roundtrip[i]).abs() <= 1e-6);
    println!("  round trip exact match? {}", all_match);
}
```

```bash
rustc --edition 2024 -O 04_tensor_csv_io.rs -o 04_tensor_csv_io
./04_tensor_csv_io
```

Produces exactly the output shown in Worked Examples 8.3.1 and 8.3.2 above.

Every number here was independently verified earlier in this chapter. All four files genuinely compile and run to completion in this sandbox with the real `rustc` toolchain — Section 8.1's functions run as ordinary host code (matching Chapter 4.5's broadcast pattern conceptually, not as an actually-launched GPU kernel), and Sections 8.2 and 8.3's code touches no GPU API at all, so nothing in this chapter carries the unverified-pending-GPU tag.

## Chapter Summary

`arange` and `linspace` are both the broadcast pattern applied to pure index arithmetic, differing only in whether the formula is built to reach the stop value exactly (`linspace`) or deliberately stop one step short of it (`arange`) — and `linspace`'s division by `n-1` is a real edge case at `n=1` that `arange`'s formula never encounters, doubly so in Rust since the `usize` subtraction itself would need guarding too. `eye` generalizes to any diagonal offset by testing `col - row` against a constant rather than testing for equality alone, cast to a signed type first so the subtraction can go negative without panicking. This chapter built a small custom PRNG from scratch — the same discipline that keeps this book's GPU claims honest also rules out leaning on a device-only random library here — and testing it thoroughly, not just running it once and eyeballing the output, turned up two genuine, real weaknesses: identical seeds silently producing identical streams, and a small seed producing a measurably weak first draw, fixed by pre-mixing the seed before it becomes the generator's initial state. A third, smaller discovery fell directly out of comparing this chapter's output against the CUDA edition's: the same four PRNG draws print in reverse order between the two books, because C++ leaves multi-argument function-call evaluation order unspecified while Rust guarantees left-to-right, a real, verified difference rather than an inconsistency between editions. Fisher-Yates shuffling and inverse-CDF sampling both build on that same generator, checked against real invariants — an unchanged element sum, and a large sample's frequency matching its target probabilities to within a few thousandths. And a CSV row-counting function that passed every straightforward test still had a real off-by-one bug, invisible until tested against a file lacking a trailing newline — the same general lesson Worked Example 8.2.2's small-seed weakness taught: an untested edge case is not a rare bug, it's a bug nobody has looked for yet.

## Self-Check Questions

1. `arange(start=5.0, step=3.0, n=4)` — compute all four values by hand.
2. Why does `linspace`'s formula fail for `n=1` in Rust in two independent ways, while `arange`'s formula fails in neither?
3. For `eye(5, k=-2)`, which `(row, col)` pairs would hold a `1`? List them.
4. `NaiveRng(42)`'s first draw was measurably weak because the seed `42` only occupies a few low bits of a 64-bit state. Would `NaiveRng(1000000007)` (a seed occupying more bits) necessarily avoid the same problem on its first draw? What does `SimpleRng`'s pre-mixing step do that a merely "bigger" seed doesn't guarantee?
5. `count_rows_naive` undercounts a file missing its trailing newline. Would the same function correctly count an EMPTY file (zero bytes)? Would it correctly count a file containing a single blank line (just one `\n` and nothing else)? Reason through both using the function's actual logic, not just its behavior on this chapter's two test files.

## Where We Go Next

Every tensor this chapter has filled has been a plain, dense, general-purpose `Vec<f32>` — every element genuinely allocated and genuinely written. Chapter 9 looks at shapes where that's wasteful by construction: an identity matrix is mostly zeros in a perfectly predictable pattern, and a specialized representation can skip storing (and skip computing) almost all of them.

## Worked Solutions

**1.** `arange(5.0, 3.0, 4)`: `i=0` gives `5 + 0*3 = 5`; `i=1` gives `5 + 1*3 = 8`; `i=2` gives `5 + 2*3 = 11`; `i=3` gives `5 + 3*3 = 14`. Result: `[5, 8, 11, 14]`.

**2.** First, `linspace`'s `step` computation divides by `(n - 1) as f32`, and `(1.0 - 1.0) / 0.0` (or any nonzero numerator divided by `0.0`) produces `f32::INFINITY` or `NaN` rather than a clean error — a silent numerical failure. Second, and earlier still: the intermediate `n - 1` on `usize` values underflows before the cast even happens if it were ever written as `n - 1` on the `usize` directly rather than cast first — Rust's `usize` subtraction panics on underflow in a debug build and wraps to a huge number in `--release`, neither of which is the negative `-1` a signed computation would cleanly produce. `arange`'s formula, `start + i * step`, takes `step` as a direct input rather than deriving it from `n` — there's no subtraction from or division by `n` anywhere in `arange`'s formula at all, so `n=1` is just an ordinary one-iteration loop producing `[start]`, no special case required.

**3.** `k=-2` means `col - row == -2`, i.e. `row = col + 2`. For a `5×5` matrix (rows and columns `0` through `4`), the valid pairs are `(row=2, col=0)`, `(row=3, col=1)`, and `(row=4, col=2)` — three positions, since `row=5` or `row=6` (which `col=3` or `col=4` would require) fall outside the `5×5` matrix.

**4.** No — a larger seed doesn't inherently avoid the problem, because the issue isn't the seed's numeric *size*, it's how many of the state word's bits it actually sets to a non-zero value combined with how many xorshift rounds have run. A seed like `1000000007` happens to set bits across a wider range than `42` does, so it's *less likely* to trigger the exact same visible symptom, but nothing about simply picking a "bigger" number *guarantees* good bit-spread — a seed like `0x8000000000000000u64` is numerically enormous yet sets only a single high bit. `SimpleRng`'s pre-mixing step doesn't depend on the seed's magnitude at all: XORing against a fixed constant with roughly half its bits set to `1` guarantees the resulting initial state has a healthy mix of `0`s and `1`s spread across the full 64 bits, regardless of what the original seed looked like.

**5.** For an empty file, `count_rows_naive`'s `.filter(...).count()` runs over zero bytes and correctly returns `0` — an empty file has zero rows, and the function happens to get this right, though only because "zero newlines" and "zero rows" coincide for this one specific input. For a file containing a single blank line (exactly one `\n` and nothing else), the filter sees exactly one `\n` byte and returns `1` — also correct, since one blank line is genuinely one row. Both of these "accidentally correct" cases are worth noticing precisely because they could easily be mistaken for evidence the function is fully correct, when Worked Example 8.3.1 already proved it isn't for the much more common case of a normal, multi-row file missing only its final newline.
