# Chapter 9: Specialized Tensor Types — Identity, Diagonal, Sparse, and Triangular

> "Chapter 6's `Tensor` stores every element it claims to hold — a fair, general-purpose deal, and a wasteful one the moment most of those elements are predictable, or absent, or mirror images of each other. This chapter is four increasingly aggressive answers to the same question: how much of a shape's data can a struct simply *not store*, computing or skipping it instead — and two genuine, discovered bugs along the way, one shared with the CUDA edition and one specific to translating its arithmetic into Rust's unsigned integers."

**What you will understand by the end of this chapter:**

- `IdentityView`: a struct holding no data at all, whose `size_of` never grows no matter how large a matrix it represents, because every one of its values is a rule (`row == col`) rather than a stored fact
- `TridiagonalView<'a>`: storage proportional to the *number of diagonals actually present* (three, for a tridiagonal matrix) rather than to the matrix's total element count — built on the same borrowed-slice, lifetime-parameterized shape Chapter 7.2's `TensorView<'a>` already established
- `SparseCoo`: a coordinate-triplet format where "adding a zero" is a genuine no-op, and where a value that *becomes* zero later can leave a real, wasted, discoverable slot behind until something explicitly compresses it away
- Packed triangular storage, and a genuine index-collision bug this chapter discovers by testing: applying the wrong formula's packing order corrupts real data, silently, in a way a naive test can miss
- A second, Rust-specific bug in the very formula meant to *fix* the first one: translating the correct formula's arithmetic literally into `usize` panics in a debug build, and — more troublingly — silently produces the right answer anyway in a release build, for reasons worth understanding precisely

**What you need to know first:**

- Chapter 6.4 (the offset formula, and its complete absence of bounds checking) — every specialized type in this chapter replaces that formula with its own, more specific rule
- Chapter 7.2 (`TensorView<'a>`, borrowed slices, and non-owning structs) — this chapter's `TridiagonalView<'a>` is a direct extension of that pattern to banded storage
- Chapter 8's habit of testing code against the specific input that breaks it, not just the input that doesn't (the small-seed PRNG weakness, the missing-trailing-newline CSV bug) — Sections 9.4 finds this chapter's bugs exactly that way, twice over
- If you've read the Mojo or CUDA editions: this chapter follows their Chapter 9 section-for-section (identity, diagonal, sparse, triangular), including the practice of finding a genuine bug in the triangular section rather than describing one hypothetically. The CUDA edition's own collision bug reproduces here unchanged, since it's pure index arithmetic with no language-specific mechanism behind it — but fixing it surfaces a second bug that's entirely specific to Rust's unsigned integer semantics.

## 9.1 Identity Tensors: When the Data *Is* the Rule `[FOUNDATIONAL]`

### Intuition

An identity matrix's every value is entirely determined by one comparison — `row == col` — with no information left over that actually needs remembering. `IdentityView` takes this literally: it stores nothing but the dimension `n` itself, and computes `1.0` or `0.0` on demand for any coordinate asked of it, the same non-owning, no-allocation spirit as Chapter 7.2's `TensorView<'a>`, pushed to its logical extreme — there's no buffer to borrow *from* here at all.

### Background

| | Chapter 6's dense `Tensor` | `IdentityView` |
|---|---|---|
| Bytes stored, `n × n` identity | `n² × size_of::<f32>()` — grows with `n²` | A fixed handful of bytes (just the `usize` field `n`) — never grows |
| How `at(row, col)` is computed | A memory read at a precomputed offset | A single comparison, no memory touched at all |
| `Vec<f32>` allocations | One, sized by `numel()` | None — there is nothing to allocate |

### Worked Example 9.1.1 — memory that provably does not grow

```rust
// No data field at all -- an n x n identity's every value is a RULE
// ("1 if row==col, else 0"), not something that needs to be stored.
struct IdentityView {
    n: usize,
}

impl IdentityView {
    fn at(&self, row: usize, col: usize) -> f32 {
        if row == col { 1.0 } else { 0.0 }
    }
}
```

Compiled and run as the complete `01_identity_view.rs` further below:

```bash
rustc --edition 2024 -O 01_identity_view.rs -o 01_identity_view
./01_identity_view
```

Genuinely compiled and run:

```
size_of::<IdentityView>() = 8 bytes -- IDENTICAL for a 4x4 or a 1,000,000x1,000,000 identity

id_small.at(2,2) = 1.0, id_small.at(2,3) = 0.0
id_huge.at(999999,999999) = 1.0  (computed instantly, no million-squared buffer touched)

a DENSE 4x4 Tensor would need 64 bytes
a DENSE 1,000,000x1,000,000 Tensor would need 4000000000000 bytes (4000.0 GB) -- IdentityView needs 8
```

`size_of::<IdentityView>()` is genuinely `8` bytes — one `usize`, which is `8` bytes wide on this 64-bit target — whether it represents a `4×4` identity or a `1,000,000 × 1,000,000` one. This is a real, verified difference from the CUDA edition's `4` bytes: CUDA's `IdentityView` stores its dimension as an `int`, `4` bytes wide regardless of target width, while `usize` deliberately tracks the platform's pointer width so it can index any array `Vec<f32>` can actually allocate — a wider field for a genuinely different, more conservative guarantee, not a regression. Either way, `id_huge.at(999999, 999999)` returns instantly because it's one comparison, not a lookup into a buffer that (at 4 terabytes for a dense equivalent) could never actually be allocated on any real machine — the zero-allocation argument Chapter 7.2 made for views in general, taken to the point where there's no underlying buffer left to view at all.

### Worked Example 9.1.2 — diagonal length and transpose, traced through the rule itself

An identity matrix's "diagonal length" is trivially `min(rows, cols)` — for a non-square `IdentityView` extended to `m × n`, the rule `row == col` can only ever be satisfied for `min(m, n)` positions, since the larger dimension simply runs out of matching partners past that point. Transposing an `IdentityView` is even simpler than Chapter 7.2's general `transpose()`: swapping `row` and `col` inside `row == col` produces the exact same condition, `col == row` — an identity matrix is its own transpose, with nothing to compute or verify beyond noticing the rule itself doesn't change when its two inputs are swapped.

> `[COMMON TRAP]` `IdentityView` genuinely represents an `n × n` matrix, but nothing in its type prevents someone from constructing one with an absurdly large `n` and never actually using it in an operation that would catch the mistake — because `at()` never touches memory, there is no out-of-bounds access for a bad `n` to trigger the way Chapter 6.4's raw offset arithmetic could at least, in principle, be caught by a bounds-checked `Vec` index on a real buffer. A struct with no data to corrupt also has no data whose corruption would ever tell you something is wrong. (Rust's `usize` does rule out one CUDA-specific version of this trap for free — a *negative* `n`, which C++'s signed `int` would happily accept and which `at()` would then quietly misbehave on, simply cannot be constructed here at all.)

## 9.2 Diagonal Tensors: Storage Proportional to What Actually Varies `[FOUNDATIONAL]`

### Intuition

A tridiagonal matrix — nonzero only on its main diagonal and the one diagonal immediately above and below it — is a natural generalization of Section 9.1's identity: instead of *one* rule (`row == col`) covering every stored value, there are *three* rules, one per diagonal, each backed by its own borrowed slice. Everywhere else, the matrix is zero by construction, and Section 9.1's argument applies again: a value that's always zero doesn't need storage, it needs a rule that returns zero.

### Background

| Diagonal | Condition | Backing slice | Entries for an `n × n` matrix |
|---|---|---|---|
| Main | `row == col` | `main[n]` | `n` |
| Sub (below main) | `row == col + 1` | `sub[n-1]` | `n - 1` |
| Super (above main) | `col == row + 1` | `super_diag[n-1]` | `n - 1` |
| Everywhere else | — | (nothing) | `0`, always |

### Worked Example 9.2.1 — a `5×5` tridiagonal matrix, built and displayed

```rust
// Borrowed, non-owning slices -- the same TensorView<'a> spirit Chapter 7.2
// established, applied to a banded rather than a dense layout.
struct TridiagonalView<'a> {
    n: usize,
    sub: &'a [f32],        // sub[i]   = element at (row=i+1, col=i)   -- n-1 entries
    main: &'a [f32],       // main[i]  = element at (row=i,   col=i)   -- n entries
    super_diag: &'a [f32], // super_diag[i] = element at (row=i, col=i+1) -- n-1 entries
}

impl<'a> TridiagonalView<'a> {
    fn at(&self, row: usize, col: usize) -> f32 {
        if row == col {
            self.main[row]
        } else if row == col + 1 {
            self.sub[col] // one below the main diagonal
        } else if col == row + 1 {
            self.super_diag[row] // one above the main diagonal
        } else {
            0.0 // everywhere else: zero, never stored
        }
    }
}
```

Compiled and run as the complete `02_diagonal_tensor.rs` further below:

```bash
rustc --edition 2024 -O 02_diagonal_tensor.rs -o 02_diagonal_tensor
./02_diagonal_tensor
```

Genuinely compiled and run:

```
5x5 tridiagonal matrix, built by hand:
  [ 4.0  2.0  0.0  0.0  0.0]
  [ 1.0  4.0  2.0  0.0  0.0]
  [ 0.0  1.0  4.0  2.0  0.0]
  [ 0.0  0.0  1.0  4.0  2.0]
  [ 0.0  0.0  0.0  1.0  4.0]

stored elements = 13, dense total = 25, sparsity fraction stored = 52.0%
```

Every row's nonzero pattern shifts one column right as `row` increases, exactly the diagonal-band structure `at()`'s three conditions describe — row `2`'s nonzero entries land at columns `1`, `2`, `3`, matching `sub[1]=1.0` (condition `row==col+1`, i.e. `2==1+1`), `main[2]=4.0`, and `super_diag[2]=2.0` (condition `col==row+1`, i.e. `3==2+1`). `13` stored floats (`4 + 5 + 4`) against `25` for the dense equivalent is `52%` — genuinely more than half, which is worth noticing directly: a tridiagonal matrix isn't *sparse* in the way Section 9.3 means the term, it's *structured*, and for small `n` specifically, three separate slices plus their own bookkeeping can cost more overhead than it saves.

> `[COMMON TRAP]` `TridiagonalView::at()` reads `self.sub[col]` (not `self.sub[row]`) for the sub-diagonal case, and this asymmetry is easy to get backwards. `sub[i]` is defined as the element at `(row=i+1, col=i)`, so given a `(row, col)` pair satisfying `row == col+1`, the correct slice index is `col` (equivalently `row-1`) — reaching for `self.sub[row]` instead would either read one element past where the actual value lives, or, in Rust specifically, panic outright the moment `row` reaches `self.sub.len()` (`n-1`), since slice indexing is bounds-checked — a genuinely louder failure than C++'s raw pointer arithmetic would produce for the same mistake, but still a bug worth catching with a real test rather than relying on the panic to find it for you.

## 9.3 Sparse Tensors: Coordinates Instead of a Grid `[FOUNDATIONAL]`

### Intuition

Section 9.2's diagonals still assume a fixed, predictable *pattern* of nonzero positions. A genuinely sparse matrix — nonzero values scattered with no such pattern — needs a format that stores positions *and* values together, as a flat list of triplets: **COO** (coordinate) format. The one rule that makes COO work at all is Worked Example 9.3.1's: a value of exactly `0.0` is indistinguishable from "never inserted" (`at()` returns `0.0` for both), so inserting a zero would be pure waste — storage spent recording a fact the format already gives away for free.

### Background

| | Dense `Tensor` (Chapter 6) | `SparseCoo` |
|---|---|---|
| Storage per element | Fixed — every position, whether zero or not | Proportional to nonzero count only |
| Reading an absent position | N/A — every position exists | Returns `0.0` by definition (a linear search finds nothing) |
| Reading cost | `O(1)` — direct offset arithmetic | `O(count)` — a linear scan over every stored triplet |
| A value that becomes zero *after* insertion | Overwritten in place, no size change | Still occupies a triplet until something explicitly removes it |

### Worked Example 9.3.1 — inserting a zero adds nothing

```rust
const CAPACITY: usize = 16;

struct SparseCoo {
    rows: [usize; CAPACITY],
    cols: [usize; CAPACITY],
    vals: [f32; CAPACITY],
    count: usize,
}

impl SparseCoo {
    // Setting a value to exactly 0.0 does NOT insert a triplet -- there is
    // nothing to add, since "absent" already means zero for every position.
    fn set(&mut self, row: usize, col: usize, value: f32) {
        if value == 0.0 {
            return; // adding a zero doesn't add anything
        }
        self.rows[self.count] = row;
        self.cols[self.count] = col;
        self.vals[self.count] = value;
        self.count += 1;
    }

    fn at(&self, row: usize, col: usize) -> f32 {
        for k in 0..self.count {
            if self.rows[k] == row && self.cols[k] == col {
                return self.vals[k];
            }
        }
        0.0 // not found -- absent means zero
    }
}
```

Compiled and run as part of the complete `03_sparse_coo.rs` further below:

```bash
rustc --edition 2024 -O 03_sparse_coo.rs -o 03_sparse_coo
./03_sparse_coo
```

Genuinely compiled and run:

```
after 4 set() calls (one was a zero, on a never-before-touched position):
  count = 3  (only the 3 genuinely nonzero inserts)
  at(0,0) = 5.0, at(2,3) = 7.0, at(4,4) = -2.0, at(1,1) [never inserted] = 0.0
```

Four calls to `set()`, only three of which actually pass the `value == 0.0` check — `set(1, 1, 0.0)` returns immediately, leaving `count` at `3`, not `4`. Reading `at(1, 1)` afterward still correctly returns `0.0`, not because a triplet says so, but because the linear search in `at()` finds nothing at `(1,1)` and falls through to its own `return 0.0`— "never inserted" and "explicitly zero" are, by design, completely indistinguishable from the outside, which is exactly what makes skipping the insert safe.

### Worked Example 9.3.2 — `compress_storage()`, and the wasted slot it exists to remove

```rust
// Overwrite an existing entry's value in place, WITHOUT removing it even
// if the new value happens to be zero -- this is what leaves a real,
// genuinely wasted triplet behind for compress_storage() to find.
fn overwrite(&mut self, row: usize, col: usize, value: f32) {
    for k in 0..self.count {
        if self.rows[k] == row && self.cols[k] == col {
            self.vals[k] = value;
            return;
        }
    }
}

// Remove every triplet whose value has become exactly zero (e.g. via
// overwrite() above), compacting the arrays in place.
fn compress_storage(&mut self) -> usize {
    let mut write = 0;
    for read in 0..self.count {
        if self.vals[read] != 0.0 {
            self.rows[write] = self.rows[read];
            self.cols[write] = self.cols[read];
            self.vals[write] = self.vals[read];
            write += 1;
        }
    }
    let removed = self.count - write;
    self.count = write;
    removed
}
```

Compiled and run as the complete `03_sparse_coo.rs` further below:

```bash
rustc --edition 2024 -O 03_sparse_coo.rs -o 03_sparse_coo
./03_sparse_coo
```

Genuinely compiled and run:

```
after overwrite(2,3, 0.0) -- the entry still EXISTS, just holds 0.0:
  count = 3  (still 3 -- overwrite() never removes a triplet)
  at(2,3) = 0.0  (reads back correctly, but wastes a storage slot)

compress_storage() removed 1 wasted triplet(s); count now = 2
  at(2,3) after compression = 0.0  (still correctly zero -- absence still means zero)
  at(0,0) after compression = 5.0, at(4,4) after compression = -2.0  (untouched entries preserved)
```

`overwrite(2, 3, 0.0)` is a fundamentally different operation from `set()`: it finds the *existing* triplet at `(2,3)` (inserted earlier with value `7.0`) and changes its value in place, with no check against zero at all — `compress_storage()` exists specifically to clean up exactly this situation, one step later, rather than trying to prevent it up front. `at(2,3)` reads correctly as `0.0` both before and after compression, for two different reasons each time: before compression, because a real triplet happens to hold the value `0.0`; after, because no triplet exists there at all and the search falls through — the caller can't tell the difference, and Section 9.3's whole design depends on that being true.

> `[COMMON TRAP]` `compress_storage()` shifts every kept triplet's array position, which silently invalidates any index into `rows`/`cols`/`vals` that code elsewhere might have cached from before the call — a bug this chapter's own `at()` never falls into (it always searches fresh) but a hypothetical "remember which slot I found `(2,3)` at, to skip the search next time" optimization absolutely would. Any format that periodically reorganizes its own storage needs every caller to treat cached positions as invalidated by that reorganization, not merely by an explicit removal.

## 9.4 Triangular Tensors: Packed Storage, and Two Genuine Collisions `[FOUNDATIONAL]`

### Intuition

An upper-triangular `n × n` matrix (nonzero only where `col ≥ row`) has exactly `n(n+1)/2` real entries — almost half of `n²` simply doesn't exist. Packing those entries into a flat array of exactly that size needs a formula mapping each valid `(row, col)` pair to a distinct flat index, and getting that formula wrong doesn't necessarily *crash* — it can just as easily hand two different, valid coordinates the *same* index, silently overwriting one with the other. This section derives the correct formula, but only after testing a plausible-looking wrong one and watching it genuinely corrupt data — and then, translating the fix itself turns up a second, unrelated bug that has nothing to do with triangular matrices at all.

### Background

| | Formula | Where it comes from |
|---|---|---|
| **Buggy**: `row*(row+1)/2 + col` | This is the packed-index formula for *lower*-triangular storage, applied here by mistake | Each row `r` in *lower*-triangular storage holds `r+1` entries (columns `0` through `r`); the formula accumulates that growing count |
| **Correct** (upper-triangular): `row*n - row*(row-1)/2 + (col - row)` | Each row `r` in *upper*-triangular storage holds `n-r` entries (columns `r` through `n-1`) — a *shrinking* count, the opposite shape from the buggy formula's assumption | Skips the entries already used by every earlier, longer row, then adds how far into the current row `(row,col)` sits |

### Worked Example 9.4.1 — the buggy formula, and where it collides

```rust
// BUGGY: this is the LOWER-triangular packed-index formula, mistakenly
// applied to UPPER-triangular storage.
fn buggy_upper_index(row: usize, col: usize, _n: usize) -> usize {
    row * (row + 1) / 2 + col
}
```

Compiled and run as part of the complete `04_triangular_packed.rs` further below:

```bash
rustc --edition 2024 -O 04_triangular_packed.rs -o 04_triangular_packed
./04_triangular_packed
```

Genuinely compiled and run:

```
writing with buggy_upper_index (row<=col only):
  (0,0) -> index 0, writing 0.0
  (0,1) -> index 1, writing 1.0
  (0,2) -> index 2, writing 2.0
  (0,3) -> index 3, writing 3.0
  (1,1) -> index 2, writing 11.0
  (1,2) -> index 3, writing 12.0
  (1,3) -> index 4, writing 13.0
  (2,2) -> index 5, writing 22.0
  (2,3) -> index 6, writing 23.0
  (3,3) -> index 9, writing 33.0
```

`(0,2)` writes to index `2`; two writes later, `(1,1)` *also* computes index `2` and overwrites it. `(0,3)` writes to index `3`; `(1,2)` collides there next. Neither collision announces itself at write time — both writes succeed (`usize` indexing here stays safely within the 10-element array, so nothing panics), and the array accepts both values, one silently replacing the other.

### Worked Example 9.4.2 — the collision, traced through an actual read-back

Compiled and run as part of the complete `04_triangular_packed.rs` further below:

```bash
rustc --edition 2024 -O 04_triangular_packed.rs -o 04_triangular_packed
./04_triangular_packed
```

Genuinely compiled and run:

```
reading back with the SAME buggy formula:
  (0,0) -> index 0, read 0.0, expected 0.0
  (0,1) -> index 1, read 1.0, expected 1.0
  (0,2) -> index 2, read 11.0, expected 2.0  <- CORRUPTED
  (0,3) -> index 3, read 12.0, expected 3.0  <- CORRUPTED
  (1,1) -> index 2, read 11.0, expected 11.0
  (1,2) -> index 3, read 12.0, expected 12.0
  (1,3) -> index 4, read 13.0, expected 13.0
  (2,2) -> index 5, read 22.0, expected 22.0
  (2,3) -> index 6, read 23.0, expected 23.0
  (3,3) -> index 9, read 33.0, expected 33.0

any corrupted entries with the buggy formula? true
```

`(1,1)` and `(1,2)` read back *correctly* — `11.0` and `12.0` — for a genuinely misleading reason: they were the *last* writers to indices `2` and `3`, so reading through the same buggy formula that wrote them recovers exactly what they wrote, hiding the fact that anything went wrong at all. `(0,2)` and `(0,3)` expose the real damage — their own values, `2.0` and `3.0`, are simply gone, silently replaced by `(1,1)`'s and `(1,2)`'s. A test that only checked "does reading back what I just wrote, in the same order, give the same answer" for a *single* coordinate at a time would never catch this — the bug only shows up when two *different* coordinates are checked against each other, which is exactly why Worked Example 9.4.1's write loop populates every valid pair before Worked Example 9.4.2's read loop checks every one of them again.

> `[COMMON TRAP]` The two formulas differ by exactly one structural assumption — whether each successive row holds *more* entries (lower-triangular, `row+1`, `row+2`, ...) or *fewer* (upper-triangular, `n-row`, `n-row-1`, ...) — and copying a packed-storage formula from a reference that uses the opposite triangular convention is a completely reasonable, easy mistake, not a typo. Both formulas are individually well-formed, valid-looking arithmetic; the only way to catch the difference is to test against multiple coordinates whose *stored order* the two conventions actually disagree on, exactly as this section's worked examples did.

### Worked Example 9.4.3 — a second bug, found while fixing the first

The obvious fix is to translate the correct formula, `row*n - row*(row-1)/2 + (col-row)`, line for line into Rust. Every index here is a natural, non-negative quantity, so `usize` looks like the right type throughout — but the formula's *intermediate* step, `row - 1`, is not itself always non-negative, even though the term it feeds into (`row * (row-1)`) always evaluates to a sensible, non-negative result once `row` is multiplied back in.

```rust
// A direct, line-for-line translation of the correct formula. row * (row - 1)
// computes row - 1 FIRST, on a usize -- which panics the moment row == 0,
// even though the final result (multiplied by row, which is 0) would
// mathematically always come out to plain 0 either way.
fn naive_correct_upper_index(row: usize, col: usize, n: usize) -> usize {
    row * n - row * (row - 1) / 2 + (col - row)
}
```

Compiled and run (deliberately WITHOUT `-O`, so overflow checks stay on — see the note below) as part of the complete `04_triangular_packed.rs` further below:

```bash
rustc --edition 2024 04_triangular_packed.rs -o 04_triangular_packed_debug
./04_triangular_packed_debug
```

Genuinely compiled and run:

```
naive_correct_upper_index(0, 0, 4), a direct translation of the CUDA formula:
  PANICKED: attempt to subtract with overflow
```

`row - 1` at `row = 0` asks Rust to compute `0usize - 1`, which has no representable, non-negative answer — Rust's default build profile treats this as a genuine logic error and panics immediately, rather than silently producing some other number the way C++'s signed `int` arithmetic would (`0 - 1 = -1`, a perfectly ordinary negative value, feeding harmlessly into `0 * -1 = 0` one multiplication later). The panic is caught here with the same `catch_unwind` machinery Chapter 4 used for `cudarc`'s missing-library panics, purely so the rest of this program can keep running and report what happened, rather than crashing outright.

> `[COMMON TRAP]` Compiling the exact same file with `-O` (this book's usual command for every other file) makes this entire bug invisible: Rust's optimized, release-style profile disables overflow checks by default, so `0usize - 1` silently wraps around to `usize::MAX` instead of panicking — and then `0 * usize::MAX`, computed with the same wrapping arithmetic, is `0` again, which happens to be the mathematically correct answer purely by coincidence. Genuinely re-running `04_triangular_packed.rs` compiled *with* `-O` confirms this directly: `naive_correct_upper_index(0, 0, 4)` returns `0` and does not panic at all. This is a real, and genuinely uncomfortable, lesson: a debug build's overflow panic is not a nuisance to compile around, it's the *only* build mode in which this particular bug was ever visible in the first place — code that "works" only because two wrapped values happen to cancel out is exactly the kind of fragile correctness a release build's silence would let ship unnoticed.

The fix keeps the same formula but performs the subtraction in a signed intermediate type first, so `row - 1` is allowed to be genuinely negative for exactly the one step where the unsigned formula couldn't represent it:

```rust
// CORRECT upper-triangular packed index (valid for row <= col): the same
// formula, but computed with a signed intermediate so `row - 1` at row=0
// produces a genuine -1 instead of panicking, before the final result --
// always non-negative for row <= col < n -- is cast back to usize.
fn correct_upper_index(row: usize, col: usize, n: usize) -> usize {
    let (row_i, col_i, n_i) = (row as i64, col as i64, n as i64);
    (row_i * n_i - row_i * (row_i - 1) / 2 + (col_i - row_i)) as usize
}
```

Compiled and run as part of the complete `04_triangular_packed.rs` further below:

```bash
rustc --edition 2024 -O 04_triangular_packed.rs -o 04_triangular_packed
./04_triangular_packed
```

Genuinely compiled and run:

```
writing with correct_upper_index (signed intermediate, no underflow):
all 10 entries read back correctly with the fixed formula? true
```

### Worked Example 9.4.4 — storage efficiency, correctly measured

Compiled and run as part of the complete `04_triangular_packed.rs` further below:

```bash
rustc --edition 2024 -O 04_triangular_packed.rs -o 04_triangular_packed
./04_triangular_packed
```

Genuinely compiled and run:

```
n=4, packed storage size = n(n+1)/2 = 10 (vs. 16 for a dense n x n)
```

`10/16 = 62.5%` of the dense storage — meaningfully less, and this fraction only improves as `n` grows, since `n(n+1)/2` grows roughly half as fast as `n²`: for `n=100`, packed storage is `5050` floats against `10000` for dense, a genuine `50.5%`, approaching the theoretical `50%` limit Section 9.4's structure implies as `n` grows without bound.

## 9.5 Complete Runnable Code

### File: `01_identity_view.rs`

```rust
// No data field at all -- an n x n identity's every value is a RULE
// ("1 if row==col, else 0"), not something that needs to be stored.
struct IdentityView {
    n: usize,
}

impl IdentityView {
    fn new(n: usize) -> Self {
        IdentityView { n }
    }
    fn at(&self, row: usize, col: usize) -> f32 {
        if row == col { 1.0 } else { 0.0 }
    }
}

fn main() {
    let id_small = IdentityView::new(4);
    let id_huge = IdentityView::new(1_000_000); // a million x a million -- no allocation attempted

    println!(
        "size_of::<IdentityView>() = {} bytes -- IDENTICAL for a 4x4 or a 1,000,000x1,000,000 identity",
        std::mem::size_of::<IdentityView>()
    );

    println!(
        "\nid_small.at(2,2) = {:.1}, id_small.at(2,3) = {:.1}",
        id_small.at(2, 2), id_small.at(2, 3)
    );
    println!(
        "id_huge.at(999999,999999) = {:.1}  (computed instantly, no million-squared buffer touched)",
        id_huge.at(999999, 999999)
    );

    let dense_bytes_small: u64 = 4 * 4 * std::mem::size_of::<f32>() as u64;
    let dense_bytes_huge: u64 = 1_000_000u64 * 1_000_000u64 * std::mem::size_of::<f32>() as u64;
    println!("\na DENSE 4x4 Tensor would need {} bytes", dense_bytes_small);
    println!(
        "a DENSE 1,000,000x1,000,000 Tensor would need {} bytes ({:.1} GB) -- IdentityView needs {}",
        dense_bytes_huge,
        dense_bytes_huge as f64 / 1e9,
        std::mem::size_of::<IdentityView>()
    );
}
```

```bash
rustc --edition 2024 -O 01_identity_view.rs -o 01_identity_view
./01_identity_view
```

Produces exactly the output shown in Worked Example 9.1.1 above.

### File: `02_diagonal_tensor.rs`

```rust
// Stores only the 3 diagonals of an n x n tridiagonal matrix -- 3n floats
// instead of n^2, reading zero for every position not on one of the 3 bands.
// Borrowed, non-owning slices -- the same TensorView<'a> spirit Chapter 7.2
// established, applied to a banded rather than a dense layout.
struct TridiagonalView<'a> {
    n: usize,
    sub: &'a [f32],        // sub[i]   = element at (row=i+1, col=i)   -- n-1 entries
    main: &'a [f32],       // main[i]  = element at (row=i,   col=i)   -- n entries
    super_diag: &'a [f32], // super_diag[i] = element at (row=i, col=i+1) -- n-1 entries
}

impl<'a> TridiagonalView<'a> {
    fn at(&self, row: usize, col: usize) -> f32 {
        if row == col {
            self.main[row]
        } else if row == col + 1 {
            self.sub[col] // one below the main diagonal
        } else if col == row + 1 {
            self.super_diag[row] // one above the main diagonal
        } else {
            0.0 // everywhere else: zero, never stored
        }
    }
}

fn main() {
    let n = 5;
    let sub = [1.0f32, 1.0, 1.0, 1.0]; // all -1-position entries
    let main = [4.0f32, 4.0, 4.0, 4.0, 4.0]; // main diagonal
    let super_diag = [2.0f32, 2.0, 2.0, 2.0]; // all +1-position entries

    let tri = TridiagonalView { n, sub: &sub, main: &main, super_diag: &super_diag };

    println!("5x5 tridiagonal matrix, built by hand:");
    for row in 0..n {
        print!("  [");
        for col in 0..n {
            print!("{:4.1}", tri.at(row, col));
            if col < n - 1 {
                print!(" ");
            }
        }
        println!("]");
    }

    let stored = (n - 1) + n + (n - 1); // sub + main + super_diag
    let total = n * n;
    println!(
        "\nstored elements = {}, dense total = {}, sparsity fraction stored = {:.1}%",
        stored, total, 100.0 * stored as f32 / total as f32
    );
}
```

```bash
rustc --edition 2024 -O 02_diagonal_tensor.rs -o 02_diagonal_tensor
./02_diagonal_tensor
```

Produces exactly the output shown in Worked Example 9.2.1 above.

### File: `03_sparse_coo.rs`

```rust
// COO (coordinate) format: parallel arrays of (row, col, value) triplets --
// only entries someone has actually inserted exist at all.
const CAPACITY: usize = 16;

struct SparseCoo {
    rows: [usize; CAPACITY],
    cols: [usize; CAPACITY],
    vals: [f32; CAPACITY],
    count: usize,
}

impl SparseCoo {
    fn new() -> Self {
        SparseCoo { rows: [0; CAPACITY], cols: [0; CAPACITY], vals: [0.0; CAPACITY], count: 0 }
    }

    // Setting a value to exactly 0.0 does NOT insert a triplet -- there is
    // nothing to add, since "absent" already means zero for every position.
    fn set(&mut self, row: usize, col: usize, value: f32) {
        if value == 0.0 {
            return; // adding a zero doesn't add anything
        }
        self.rows[self.count] = row;
        self.cols[self.count] = col;
        self.vals[self.count] = value;
        self.count += 1;
    }

    fn at(&self, row: usize, col: usize) -> f32 {
        for k in 0..self.count {
            if self.rows[k] == row && self.cols[k] == col {
                return self.vals[k];
            }
        }
        0.0 // not found -- absent means zero
    }

    // Overwrite an existing entry's value in place, WITHOUT removing it even
    // if the new value happens to be zero -- this is what leaves a real,
    // genuinely wasted triplet behind for compress_storage() to find.
    fn overwrite(&mut self, row: usize, col: usize, value: f32) {
        for k in 0..self.count {
            if self.rows[k] == row && self.cols[k] == col {
                self.vals[k] = value;
                return;
            }
        }
    }

    // Remove every triplet whose value has become exactly zero (e.g. via
    // overwrite() above), compacting the arrays in place.
    fn compress_storage(&mut self) -> usize {
        let mut write = 0;
        for read in 0..self.count {
            if self.vals[read] != 0.0 {
                self.rows[write] = self.rows[read];
                self.cols[write] = self.cols[read];
                self.vals[write] = self.vals[read];
                write += 1;
            }
        }
        let removed = self.count - write;
        self.count = write;
        removed
    }
}

fn main() {
    let mut m = SparseCoo::new();
    m.set(0, 0, 5.0);
    m.set(2, 3, 7.0);
    m.set(1, 1, 0.0); // "setting" a zero on a position NEVER touched before
    m.set(4, 4, -2.0);

    println!("after 4 set() calls (one was a zero, on a never-before-touched position):");
    println!("  count = {}  (only the 3 genuinely nonzero inserts)", m.count);
    println!(
        "  at(0,0) = {:.1}, at(2,3) = {:.1}, at(4,4) = {:.1}, at(1,1) [never inserted] = {:.1}",
        m.at(0, 0), m.at(2, 3), m.at(4, 4), m.at(1, 1)
    );

    // Now genuinely zero out an EXISTING entry via overwrite(), leaving a
    // real, wasted triplet behind on purpose.
    m.overwrite(2, 3, 0.0);
    println!("\nafter overwrite(2,3, 0.0) -- the entry still EXISTS, just holds 0.0:");
    println!("  count = {}  (still 3 -- overwrite() never removes a triplet)", m.count);
    println!("  at(2,3) = {:.1}  (reads back correctly, but wastes a storage slot)", m.at(2, 3));

    let removed = m.compress_storage();
    println!("\ncompress_storage() removed {} wasted triplet(s); count now = {}", removed, m.count);
    println!("  at(2,3) after compression = {:.1}  (still correctly zero -- absence still means zero)", m.at(2, 3));
    println!(
        "  at(0,0) after compression = {:.1}, at(4,4) after compression = {:.1}  (untouched entries preserved)",
        m.at(0, 0), m.at(4, 4)
    );
}
```

```bash
rustc --edition 2024 -O 03_sparse_coo.rs -o 03_sparse_coo
./03_sparse_coo
```

Produces exactly the output shown in Worked Examples 9.3.1 and 9.3.2 above.

### File: `04_triangular_packed.rs`

```rust
use std::panic;

fn catch_it<T>(f: impl FnOnce() -> T + panic::UnwindSafe) -> Result<T, String> {
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

// BUGGY: this is the LOWER-triangular packed-index formula, mistakenly
// applied to UPPER-triangular storage.
fn buggy_upper_index(row: usize, col: usize, _n: usize) -> usize {
    row * (row + 1) / 2 + col
}

// A direct, line-for-line translation of the CUDA edition's correct formula.
// row * (row - 1) computes row - 1 FIRST, on a usize -- which panics the
// moment row == 0, even though the final result (multiplied by row, which is
// 0) would mathematically always come out to plain 0 either way.
fn naive_correct_upper_index(row: usize, col: usize, n: usize) -> usize {
    row * n - row * (row - 1) / 2 + (col - row)
}

// CORRECT upper-triangular packed index (valid for row <= col): the same
// formula, but computed with a signed intermediate so `row - 1` at row=0
// produces a genuine -1 instead of panicking, before the final result --
// always non-negative for row <= col < n -- is cast back to usize.
fn correct_upper_index(row: usize, col: usize, n: usize) -> usize {
    let (row_i, col_i, n_i) = (row as i64, col as i64, n as i64);
    (row_i * n_i - row_i * (row_i - 1) / 2 + (col_i - row_i)) as usize
}

fn main() {
    let n: usize = 4;
    let packed_size = n * (n + 1) / 2; // 10 for n=4

    println!(
        "n={}, packed storage size = n(n+1)/2 = {} (vs. {} for a dense n x n)\n",
        n, packed_size, n * n
    );

    // Write a distinct, easily recognizable value to every valid (row,col)
    // pair using the BUGGY formula, then try to read every pair back.
    let mut buggy_storage = [0.0f32; 10];
    println!("writing with buggy_upper_index (row<=col only):");
    for row in 0..n {
        for col in row..n {
            let value = (row * 10 + col) as f32; // e.g. (1,2) -> 12.0, easy to recognize
            let idx = buggy_upper_index(row, col, n);
            println!("  ({},{}) -> index {}, writing {:.1}", row, col, idx, value);
            buggy_storage[idx] = value;
        }
    }
    println!("\nreading back with the SAME buggy formula:");
    let mut any_wrong = false;
    for row in 0..n {
        for col in row..n {
            let expected = (row * 10 + col) as f32;
            let idx = buggy_upper_index(row, col, n);
            let got = buggy_storage[idx];
            let wrong = got != expected;
            if wrong {
                any_wrong = true;
            }
            println!(
                "  ({},{}) -> index {}, read {:.1}, expected {:.1}{}",
                row, col, idx, got, expected, if wrong { "  <- CORRUPTED" } else { "" }
            );
        }
    }
    println!("\nany corrupted entries with the buggy formula? {}", any_wrong);

    // The naive line-for-line translation of the CORRECT formula, genuinely
    // panicking at row=0 -- caught here so the rest of this program can
    // still report what happened.
    println!("\nnaive_correct_upper_index(0, 0, {}), a direct translation of the CUDA formula:", n);
    match catch_it(|| naive_correct_upper_index(0, 0, n)) {
        Ok(idx) => println!("  returned index {} (did not panic)", idx),
        Err(msg) => println!("  PANICKED: {}", msg),
    }

    // The FIX: same test, correct formula with a signed intermediate.
    let mut fixed_storage = [0.0f32; 10];
    println!("\nwriting with correct_upper_index (signed intermediate, no underflow):");
    for row in 0..n {
        for col in row..n {
            let value = (row * 10 + col) as f32;
            let idx = correct_upper_index(row, col, n);
            fixed_storage[idx] = value;
        }
    }
    let mut all_correct = true;
    for row in 0..n {
        for col in row..n {
            let expected = (row * 10 + col) as f32;
            let idx = correct_upper_index(row, col, n);
            if fixed_storage[idx] != expected {
                all_correct = false;
            }
        }
    }
    println!(
        "all {} entries read back correctly with the fixed formula? {}",
        packed_size, all_correct
    );
}
```

```bash
# Deliberately compiled WITHOUT -O, so the debug profile's overflow checks stay
# on and the naive_correct_upper_index panic genuinely reproduces (see 9.4.3):
rustc --edition 2024 04_triangular_packed.rs -o 04_triangular_packed
./04_triangular_packed
```

Produces exactly the output shown in Worked Examples 9.4.1 through 9.4.4 above. Compiling this same file *with* `-O` instead changes only one line of output — `naive_correct_upper_index(0, 0, 4)` returns `0` and does not panic, exactly as Worked Example 9.4.3's Common Trap describes — with every other line, including the fixed formula's own results, unaffected either way.

Every number here was independently verified earlier in this chapter. All four files genuinely compile and run to completion in this sandbox — none of them touch any GPU API, since every specialized type in this chapter is pure struct-and-arithmetic logic, checkable identically regardless of target.

## Chapter Summary

Four specialized tensor types, each skipping storage for a different, increasingly general reason. `IdentityView` stores nothing at all, because every value follows from one comparison rather than needing to be remembered — `size_of` stays constant no matter how large a matrix it represents, at a genuinely different (and correctly wider) `8` bytes than the CUDA edition's `4`, since `usize` tracks pointer width rather than a fixed 32 bits. `TridiagonalView<'a>` stores exactly the diagonals a banded matrix actually has, at `52%` of dense storage for a small `5×5` case, built on the exact borrowed-slice shape Chapter 7.2's `TensorView<'a>` already established. `SparseCoo` stores only nonzero values as explicit coordinate triplets, with "adding a zero" a deliberate no-op and a value that *becomes* zero later left behind as a real, discoverable wasted slot until `compress_storage()` removes it. Packed triangular storage's `n(n+1)/2` formula only works when its row-length assumption — growing, for lower-triangular, or shrinking, for upper-triangular — matches the actual layout being packed; this chapter found, by testing multiple coordinates against each other rather than just re-reading what was just written, that the wrong assumption produces genuine, silent index collisions, exactly reproducing the CUDA edition's bug since it's pure arithmetic with no language-specific mechanism involved. Fixing that bug then surfaced a second, entirely Rust-specific one: a literal translation of the correct formula panics on `usize` underflow in a debug build, and — more unsettling — silently wraps to the mathematically correct answer in a release build purely by coincidence, a reminder that a debug build's overflow panic is sometimes the only build mode in which a real bug is visible at all.

## Self-Check Questions

1. For a `6×6` identity matrix, what is `size_of::<IdentityView>()` in this chapter's Rust edition, and how does that number change for a `600×600` identity? How does it compare to the CUDA edition's answer, and why?
2. A pentadiagonal matrix (5 diagonals: two below main, main, two above) of size `n×n` stores how many floats total, as a formula in `n`? At what point does this formula's stored fraction of `n²` cross below 10%?
3. `SparseCoo::set(3, 3, 0.0)` is called on a matrix that has never touched position `(3,3)` before. What does `count` become, and what does `at(3,3)` return immediately afterward?
4. Using `correct_upper_index(row, col, n) = row*n - row*(row-1)/2 + (col-row)`, compute the packed index for `(row=2, col=3)` in a `5×5` upper-triangular matrix, and verify it doesn't collide with any other valid pair's index by checking `(row=0, col=3)` and `(row=1, col=3)` too.
5. Worked Example 9.4.2 noted that `(1,1)` and `(1,2)` read back *correctly* despite the buggy formula, while `(0,2)` and `(0,3)` did not. Explain what property of the write order (not the coordinates themselves) determined which pairs survived and which didn't.
6. `naive_correct_upper_index(0, 0, 4)` panics when compiled without `-O` but silently returns the correct answer when compiled with it. Is the `-O`-compiled version of this function actually safe to ship, given that it happens to produce correct results? Why or why not?

## Where We Go Next

Every tensor type this chapter built is read-only in spirit — a rule, a small set of bands, a sparse list, or a packed triangle, each answering "what value lives here?" without yet doing any *arithmetic* between two tensors. Part 2 starts exactly there: element-wise operations, the first place two tensors' values genuinely combine into a third.

## Worked Solutions

**1.** `size_of::<IdentityView>()` is `8` bytes (one `usize` field, `8` bytes wide on this 64-bit target) for a `6×6` identity, and remains exactly `8` bytes for a `600×600` identity — it never changes, because `IdentityView` stores only the dimension itself, never the matrix's contents. This is genuinely different from the CUDA edition's `4` bytes: CUDA's field is a fixed-width `int` (`4` bytes on every target), while Rust's `usize` deliberately scales with the platform's pointer width so it can index anything a real `Vec` could hold — a wider field, but a more principled one, not an inefficiency.

**2.** Two below main, main, two above main means 5 diagonals: lengths `n-2, n-1, n, n-1, n-2`, summing to `5n - 6`. As a fraction of `n²`: `(5n-6)/n²`. Setting `(5n-6)/n² < 0.10` and solving approximately: `5n - 6 < 0.1n²`, or `0.1n² - 5n + 6 > 0`. Using the quadratic formula, this crosses zero near `n ≈ 48.8`; for `n = 50`, stored `= 5(50)-6 = 244`, dense `= 2500`, fraction `= 9.76%` — just under 10%, confirming the crossing point lands close to `n ≈ 49`.

**3.** `count` stays at whatever it was before the call — `set()`'s `if value == 0.0 { return; }` check exits before touching `count`, `rows`, `cols`, or `vals` at all. `at(3,3)` returns `0.0` immediately afterward, for the same reason every never-inserted position does: the linear search in `at()` finds no triplet at `(3,3)` and falls through to its own default.

**4.** `correct_upper_index(2, 3, 5) = 2*5 - 2*1/2 + (3-2) = 10 - 1 + 1 = 10`. For `(0,3)`: `0*5 - 0 + (3-0) = 3`. For `(1,3)`: `1*5 - 1*0/2 + (3-1) = 5 - 0 + 2 = 7`. All three indices — `10`, `3`, `7` — are distinct, consistent with the correct formula's design guarantee that every valid `(row,col)` pair for a fixed `n` maps to its own unique slot in the `n(n+1)/2`-sized packed array.

**5.** The surviving pairs, `(1,1)` and `(1,2)`, were each the *last* write to their respective colliding index in Worked Example 9.4.1's write order — `(1,1)` was written after `(0,2)` at index `2`, and `(1,2)` was written after `(0,3)` at index `3`. Reading through the same buggy formula afterward simply recovers whatever the *most recent* write left behind at each index; the pairs that got overwritten, `(0,2)` and `(0,3)`, were the earlier writers at their shared indices, not written to again afterward. Nothing about the coordinates' own values determines survival — only which write happened later in the loop that populated the array.

**6.** No — it is not safe to ship, despite producing a correct result for `(0, 0, 4)`. The `-O`-compiled version isn't computing the right answer through correct reasoning; it's silently wrapping `0usize - 1` to `usize::MAX`, then relying on the *separate* fact that this particular result gets multiplied by `row = 0` immediately afterward, which zeroes out the wrapped garbage before it can do any damage. That cancellation is a coincidence of this one call's specific arguments, not a property of the formula in general — the underlying operation (subtracting past zero on an unsigned type) is still a genuine logic error every time it happens, and a debug build's panic is the mechanism that would actually catch it if a *different* set of arguments ever combined the wrapped value with something other than a multiply-by-zero. Shipping the `-O` build without ever having run the debug build at least once means never finding out whether today's coincidental cancellation is the only one this formula depends on.
