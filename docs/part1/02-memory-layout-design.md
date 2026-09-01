# Chapter 7: Memory Layout Design — Layouts, Views, Alignment, and Broadcasting

> "Chapter 6 wrote `offset = Σ coord[d] × stride[d]` as if it had one right answer per shape. It doesn't — the formula never changes, but the strides feeding it can encode a dozen different physical arrangements of the identical logical tensor, and this chapter is about four of the ones this book's `Tensor` actually needs: a second, genuinely different way to compute strides; a struct that reads an existing buffer without owning a byte of it; the padding a compiler quietly inserts when nobody asks it not to; and a trick that lets one small buffer answer for a much larger shape, one stride set to zero at a time."

**What you will understand by the end of this chapter:**

- Column-major strides, computed by the same right-to-left process Chapter 6.2 used but starting from the opposite end — and why this matters concretely the moment this book's `Tensor` ever needs to talk to a BLAS library
- `TensorView<'a>`: a non-owning, lifetime-parameterized struct that reads (or writes) an existing buffer through its own shape and strides, with no `Vec` allocation, no `Drop` work, and no bytes moved — the mechanism behind transpose, slicing, and reshape
- Why struct field order changes `size_of` even when the fields themselves are unchanged, traced to the compiler's alignment padding — and why Rust's *default* struct layout already sidesteps this problem in a way C++ never does, unless you opt back out with `#[repr(C)]`
- Broadcasting: combining a small shape with a larger one by setting a dimension's stride to `0`, so many logical coordinates legitimately alias the same physical element — and the real hazard that appears the moment code tries to *write* through a broadcast view instead of just reading it

**What you need to know first:**

- Chapter 6 in full — this chapter's `TensorStrides::col_major`, `TensorView`, and broadcast strides are all direct extensions of Chapter 6.1–6.4's `TensorShape`, `TensorStrides`, and offset formula
- Chapter 6.5 (move semantics and `#[derive(Clone)]`) — Section 7.2 depends on contrasting `TensorView`'s *absence* of ownership against `Tensor`'s presence of it
- Chapter 5 (`wide::f32x8` and its real `#[repr(C, align(32))]` layout) — Section 7.3 returns to this exact type
- Chapter 2.1 (`#[repr(C)]` and `Mixed`/`MixedC`) — Section 7.3 is a direct continuation of that comparison

## 7.1 Strides Revisited: One Formula, Many Layouts `[FOUNDATIONAL]`

### Intuition

Chapter 6.2 computed strides right-to-left, making the *last* dimension contiguous — **row-major**, the layout every array in this book has used so far, and Rust's own native convention for slices and `Vec`. Nothing about the offset formula itself, `offset = Σ coord[d] × stride[d]`, cares which dimension is contiguous; that fact lives entirely in how the strides were computed. **Column-major** — the *first* dimension contiguous instead — is the same formula fed different strides, and it isn't a purely academic alternative: BLAS libraries (the ones a future chapter's GEMM calls would eventually bind to) are column-major by convention, a fact they inherited from Fortran BLAS decades before either Rust or CUDA existed.

### Background

| | Row-major (this book's default) | Column-major (BLAS's convention) |
|---|---|---|
| Contiguous dimension | Last | First |
| Stride computation direction | Right to left: `strides[i] = strides[i+1] * dims[i+1]` | Left to right: `strides[i] = strides[i-1] * dims[i-1]` |
| `strides[0]` for shape `[2,3,4]` | `12` (skip a whole `[3,4]` block) | `1` (the first dimension is packed tightest) |
| Who uses it | This book's own `Tensor`, and Rust slices/`Vec` generally | BLAS libraries, and Fortran-derived numerical code generally |

### Worked Example 7.1.1 — the same `[2,3,4]` shape, both ways

```rust
fn row_major(shape: &TensorShape) -> TensorStrides {
    let mut strides = [0usize; MAX_DIMS];
    let mut running = 1;
    for i in (0..shape.ndim).rev() {
        strides[i] = running;
        running *= shape.dims[i];
    }
    TensorStrides { strides }
}

// Column-major: the FIRST dimension is contiguous instead of the last --
// the convention cuBLAS (and any BLAS this book eventually links against) inherits
// from Fortran BLAS.
fn col_major(shape: &TensorShape) -> TensorStrides {
    let mut strides = [0usize; MAX_DIMS];
    let mut running = 1;
    for i in 0..shape.ndim {
        strides[i] = running;
        running *= shape.dims[i];
    }
    TensorStrides { strides }
}
```

Compiled and run as part of the complete `01_layout_row_col_major.rs` further below:

```bash
rustc --edition 2024 -O 01_layout_row_col_major.rs -o 01_layout_row_col_major
./01_layout_row_col_major
```

Genuinely compiled and run:

```
shape = [2, 3, 4]
row-major strides = [12, 4, 1]
col-major strides = [1, 2, 6]
```

`col_major` runs the identical accumulation loop as `row_major`, just left-to-right instead of right-to-left: `i=0` sets `strides[0] = 1` (the running product starts at 1), then `running = 1 × dims[0] = 1 × 2 = 2`; `i=1` sets `strides[1] = 2`, then `running = 2 × dims[1] = 2 × 3 = 6`; `i=2` sets `strides[2] = 6`. Same shape, same formula, opposite direction, genuinely different strides.

### Worked Example 7.1.2 — index↔offset round trip, and where the two layouts disagree

Compiled and run as part of the complete `01_layout_row_col_major.rs` further below:

```bash
rustc --edition 2024 -O 01_layout_row_col_major.rs -o 01_layout_row_col_major
./01_layout_row_col_major
```

Genuinely compiled and run:

```
index<->offset round trip for coordinate (1, 1, 2):
row-major offset(1,1,2) = 18
col-major offset(1,1,2) = 15

same coordinate, opposite contiguity:
row-major offset(0,0,1) = 1  (innermost dim is contiguous)
col-major offset(0,0,1) = 6  (innermost dim is NOT contiguous here)
row-major offset(1,0,0) = 12  (outermost dim is NOT contiguous here)
col-major offset(1,0,0) = 1  (outermost dim is contiguous)

both layouts agree at the very last coordinate (a permutation always agrees at the boundary):
row-major offset(1,2,3) = 23, col-major offset(1,2,3) = 23, numel-1 = 23
```

`(1,1,2)` genuinely lands at two different flat offsets under the two layouts — `18` under row-major, `15` under column-major — confirming the same logical coordinate refers to physically different memory depending on which strides accompany it. The `(0,0,1)`/`(1,0,0)` pair makes the contiguity claim concrete: stepping by 1 along the innermost coordinate moves 1 flat slot under row-major but 6 under column-major, and vice versa for the outermost coordinate. The last line is a genuine, if slightly surprising, fact rather than a bug: *any* permutation-based relabeling of a full traversal agrees at its very first and very last element, since both layouts visit every one of the same 24 slots exactly once — it's only the coordinates in between that land differently.

### ASCII Diagram — a `[3,4]` matrix, two ways

```
Logical matrix (3 rows, 4 cols):        Row-major flat buffer (row 0, row 1, row 2):
 [ a b c d ]                             [ a b c d | e f g h | i j k l ]
 [ e f g h ]                              strides = [4, 1] -- row stride 4, col stride 1
 [ i j k l ]
                                         Column-major flat buffer (col 0, col 1, col 2, col 3):
                                          [ a e i | b f j | c g k | d h l ]
                                          strides = [1, 3] -- row stride 1, col stride 3
```

> `[COMMON TRAP]` Handing a row-major `Tensor`'s raw data to a column-major-expecting BLAS call without accounting for the layout mismatch doesn't fail loudly — the library has no way to know which convention the caller meant, so it reads the identical bytes as if they were column-major and silently computes a real, cleanly-executing, *wrong* matrix product. The standard fix, used throughout real numerical codebases, isn't to physically transpose the data — it's to tell the library to treat the row-major buffer as an already-transposed column-major one (row-major `A` of shape `[m,n]` is bit-for-bit identical to column-major `Aᵀ` of shape `[n,m]`), a genuinely free reinterpretation that Section 7.2's `TensorView::transpose()` is the exact mechanism for.

## 7.2 `TensorView`: Reading a Tensor Without Copying It `[FOUNDATIONAL]`

### Intuition

Chapter 6's `Tensor` conflates two separate ideas that Section 7.1's BLAS trap already needs pulled apart: *owning* memory (a `Vec<f32>`, exactly one legitimate owner, freed by `Drop`) and *describing how to read* memory (a shape and a set of strides, which can be reinterpreted freely without touching a single byte). `TensorView<'a>` is a struct built for the second idea alone — a borrowed slice plus a shape and strides, with no constructor that allocates and no `Drop` impl that frees, because it never owned anything to begin with. Rust's borrow checker turns this from a convention into a compiler-enforced guarantee, as this section's Worked Example 7.2.3 demonstrates directly.

### Background

| | `Tensor` (Chapter 6) | `TensorView<'a>` (this chapter) |
|---|---|---|
| Owns its memory? | Yes — a `Vec<f32>`, allocated in its constructor | No — borrows a `&'a [f32]` someone else owns |
| Cleanup | `Vec<f32>`'s own `Drop` frees `data` and `grad` | Trivial — there is nothing to free |
| Copying it | Not `Copy` (Chapter 6.5) — the compiler forbids it structurally, since `Vec<f32>` isn't `Copy` | Could derive `Copy` — a shared reference and two small `Copy` structs duplicate nothing that needs freeing |
| What transpose/slice/reshape cost | N/A — `Tensor` itself is never reshaped in place | Zero allocation, zero data movement — only the shape/strides/reference triple changes |
| How long it's allowed to exist | As long as its own `Vec` lives — no external constraint | Bounded by `'a`: the compiler refuses to let it outlive the buffer it borrows from |

### Worked Example 7.2.1 — transpose, verified against the original

```rust
struct TensorView<'a> {
    data: &'a [f32],
    shape: Shape2D,
    strides: Strides2D,
}

impl<'a> TensorView<'a> {
    fn offset(&self, i0: usize, i1: usize) -> usize {
        i0 * self.strides.strides[0] + i1 * self.strides.strides[1]
    }

    // Transpose: SWAP shape and strides. Zero bytes move; only the map changes.
    fn transpose(&self) -> TensorView<'a> {
        TensorView {
            data: self.data,
            shape: Shape2D { dims: [self.shape.dims[1], self.shape.dims[0]] },
            strides: Strides2D { strides: [self.strides.strides[1], self.strides.strides[0]] },
        }
    }
}
```

Compiled and run as part of the complete `02_tensor_view.rs` further below:

```bash
rustc --edition 2024 -O 02_tensor_view.rs -o 02_tensor_view
./02_tensor_view
```

Genuinely compiled and run:

```
original view: shape=[3, 4], strides=[4, 1]
view(1,2) = buf[6] = 6

transposed view: shape=[4, 3], strides=[1, 4]  (same buffer, same pointer: true)
t(2,1) = buf[6] = 6  (should equal view(1,2) -- transpose just swaps the coordinate order)
```

`transpose()` swaps `shape.dims[0]`↔`shape.dims[1]` and `strides[0]`↔`strides[1]` together, and nothing else — `t.data` and `view.data` genuinely point at the same buffer, confirmed directly with `std::ptr::eq`. Reading `view(1,2)` and `t(2,1)` land at the identical flat offset, `6`, because swapping both the shape and the strides in lockstep is exactly what makes "the element at row 1, column 2" and "the element at row 2, column 1 of the transposed view" refer to the same underlying value — transposition is a relabeling of coordinates, not a rearrangement of memory.

### Worked Example 7.2.2 — a row slice, reached through an advanced slice start

```rust
// A row slice: same strides, a shrunk shape, and the borrowed slice advanced by
// exactly this row's own starting offset -- again, zero bytes move.
fn row_slice(&self, row: usize) -> TensorView<'a> {
    let start = self.offset(row, 0);
    TensorView {
        data: &self.data[start..],
        shape: Shape2D { dims: [1, self.shape.dims[1]] },
        strides: self.strides,
    }
}
```

Compiled and run as part of the complete `02_tensor_view.rs` further below:

```bash
rustc --edition 2024 -O 02_tensor_view.rs -o 02_tensor_view
./02_tensor_view
```

Genuinely compiled and run:

```
row_slice(1): shape=[1, 4], data pointer advanced by 4 elements
r(0,2) = 6  (should equal view(1,2) -- same element, reached through the slice)
```

`row_slice(1)` computes `offset(1, 0) = 1×4 + 0×1 = 4`, advances the borrowed slice's start to `&self.data[4..]`, and keeps the original strides unchanged — reading `r(0,2)` through the *sliced* view's own, now-relative coordinate system reaches the same element `view(1,2)` and Worked Example 7.2.1's transpose both already confirmed, verified here by checking the raw pointer really did advance by exactly 4 elements with `offset_from`.

### Worked Example 7.2.3 — a lifetime the borrow checker won't let you violate

Chapter 6.5 already showed Rust catching a use-after-move at compile time (`E0382`) where CUDA's manually-`= delete`d copy constructor could only catch the analogous mistake at compile time too, but only because someone remembered to delete it. `TensorView<'a>`'s `'a` parameter buys something CUDA's raw-pointer-based view has no equivalent for at all: a compile-time guarantee that the view can never outlive the buffer it borrows from, enforced automatically, without anyone writing a single line of code to ask for it.

```rust
fn make_view() -> TensorView<'static> {
    let buf: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let view = TensorView {
        data: &buf,
        shape: Shape2D { dims: [2, 2] },
        strides: Strides2D { strides: [2, 1] },
    };
    view // ERROR: `buf` does not live long enough
}
```

Genuinely compiled with `rustc --edition 2024 03_err_view_lifetime.rs -o 03_err_view_lifetime`:

```
error[E0515]: cannot return value referencing local variable `buf`
  --> 03_err_view_lifetime.rs:33:5
   |
29 |         data: &buf,
   |               ---- `buf` is borrowed here
...
33 |     view // ERROR: `buf` does not live long enough
   |     ^^^^ returns a value referencing data owned by the current function

error: aborting due to 1 previous error

For more information about this error, try `rustc --explain E0515`.
```

`buf` is a local `Vec<f32>`, owned by `make_view` and dropped the moment the function returns. Trying to return a `TensorView<'static>` that borrows from it asks the compiler to hand back a reference into memory that is about to be freed — precisely the kind of dangling pointer a CUDA `TensorView` returned by an equivalent host function would produce silently, compile cleanly, and only crash (or worse, silently read garbage) the first time something dereferences it. `rustc` refuses to compile this function at all. This is a genuinely stronger safety property than Chapter 6's move-tracking already offered: `E0382` catches a moved-away *value* being used again, while `E0515` catches a *reference* being smuggled past the scope of the data it points into — two different categories of memory bug that C++'s type system leaves to a programmer's memory, and that Rust's lifetime system checks by construction, on every single build, for every `TensorView` this book ever writes.

### ASCII Diagram — one buffer, three views, zero copies

```
Underlying buffer (12 floats, never moved):
 [0  1  2  3 | 4  5  6  7 | 8  9  10 11]

view            : shape=[3,4] strides=[4,1] start=buf[0]   -- the whole thing
view.transpose(): shape=[4,3] strides=[1,4] start=buf[0]   -- same bytes, swapped map
view.row_slice(1): shape=[1,4] strides=[4,1] start=buf[4]  -- same bytes, narrowed map, shifted start
```

> `[COMMON TRAP]` Reshaping a view assumes the view's current strides are *contiguous* in the target shape's own row-major sense — an assumption `row_slice`'s output still satisfies, but `transpose`'s output does not. Writing something like `.reshape(12)` on `view.transpose()` and simply relabeling the existing `[4,3]`/`[1,4]` data as a flat 12-element row-major sequence would silently read the wrong elements in the wrong order, because the transposed view's actual memory order (column-major-ish, following its swapped strides) no longer matches what a fresh row-major reshape assumes. Unlike Worked Example 7.2.3's lifetime violation, this is *not* something Rust's type system catches automatically — a correct reshape-after-transpose has to either verify contiguity first and fall back to a genuine copy when it doesn't hold, or refuse the reshape outright at runtime — silently trusting the shape numbers alone, without checking the strides behind them, is exactly how this class of bug reaches production in any language, Rust included.

## 7.3 Alignment and Padding: What the Compiler Inserts When You Don't Ask `[FOUNDATIONAL]`

### Intuition

Chapter 5's `wide::f32x8` type carries a real `#[repr(C, align(32))]` in its own source — this section generalizes that single fact into the rule behind it. Every type has a required alignment — the address it must start at, a multiple of some power of two — and a struct's overall alignment is simply the largest alignment any of its fields demands. Under Rust's *default*, unspecified `repr`, the compiler is free to reorder a struct's fields however it likes to minimize padding — a freedom C++ never grants, which is exactly why Chapter 2.1 had to reach for `#[repr(C)]` to get `Mixed` and `MixedC` to disagree in the first place. This section proves that freedom empirically, then shows `#[repr(C)]` handing it back.

### Background

| | `InterleavedHeader`: `u8, i32, u8, i32` | `GroupedHeader`: `i32, i32, u8, u8` |
|---|---|---|
| Under default repr | Compiler reorders fields freely; padding minimized either way | Compiler reorders fields freely; padding minimized either way |
| Under `#[repr(C)]` | Padding after each `u8`, twice (to keep each following `i32` 4-byte aligned) | Padding once, at the very end (to keep the whole struct's size a multiple of its alignment) |
| `size_of` under `#[repr(C)]` (4 real bytes of `u8` + 8 real bytes of `i32`, in either order) | Larger | Smaller |

### Worked Example 7.3.1 — identical fields, different order, `size_of` only diverges under `#[repr(C)]`

```rust
// Default (unspecified) repr -- the compiler may reorder these fields however
// it likes, so declaration order alone does NOT determine size_of.
#[derive(Debug)]
struct InterleavedDefault { a: u8, b: i32, c: u8, d: i32 }

#[derive(Debug)]
struct GroupedDefault { b: i32, d: i32, a: u8, c: u8 }

// #[repr(C)] forces declaration order to be preserved exactly, the same way
// a C/C++ compiler does -- this is where the two structs finally diverge.
#[repr(C)]
#[derive(Debug)]
struct InterleavedC { a: u8, b: i32, c: u8, d: i32 }

#[repr(C)]
#[derive(Debug)]
struct GroupedC { b: i32, d: i32, a: u8, c: u8 }
```

Compiled and run as part of the complete `04_alignment_padding.rs` further below:

```bash
rustc --edition 2024 -O -L /tmp/rust_ch7/proj/target/release/deps --extern wide=/tmp/rust_ch7/proj/target/release/libwide.rlib 04_alignment_padding.rs -o 04_alignment_padding
./04_alignment_padding
```

Genuinely compiled and run:

```
default (Rust) repr, compiler free to reorder fields:
  InterleavedDefault (u8,i32,u8,i32): size_of = 12, align_of = 4
  GroupedDefault     (i32,i32,u8,u8): size_of = 12, align_of = 4
  identical fields, same size regardless of declared order: true

#[repr(C)], declaration order preserved like C/C++:
  InterleavedC (u8,i32,u8,i32): size_of = 16, align_of = 4
  GroupedC     (i32,i32,u8,u8): size_of = 12, align_of = 4
  identical fields, 4 fewer bytes just from reordering them

Chapter 5's wide::f32x8, genuinely queried:
  align_of::<wide::f32x8>() = 32, size_of::<wide::f32x8>() = 32
```

Under default repr, both `InterleavedDefault` and `GroupedDefault` come out to `size_of = 12` — the compiler silently reordered `InterleavedDefault`'s fields internally (grouping the two `i32`s together and the two `u8`s together, the same layout `GroupedDefault` already declared) before it ever computed padding, so declaration order made no observable difference. The moment `#[repr(C)]` is added, that freedom disappears: `InterleavedC` is forced to place fields exactly as declared, paying for padding twice — after `a` (3 bytes, so `b` starts 4-byte aligned) and after `c` (3 bytes, so `d` starts 4-byte aligned) — for 6 padding bytes total alongside 10 real bytes of data, rounding up to 16. `GroupedC` places both 4-byte `i32`s back to back with no padding needed between them, then both 1-byte `u8`s back to back after, paying only 2 padding bytes at the very end — 10 real bytes plus 2 padding bytes, `size_of = 12`. This reproduces the CUDA edition's exact 16-vs-12 gap, but only once `#[repr(C)]` re-imposes the declaration-order rule that C++ structs never had a choice about in the first place. The final line confirms `wide::f32x8`'s alignment directly from Rust's own type system rather than from reading its source: `align_of::<wide::f32x8>() == 32`, matching the `#[repr(C, align(32))]` Chapter 5's research found in the crate itself, and `size_of == 32` too — eight packed `f32` lanes with no padding at all, since 8 × 4 bytes already sits exactly on its own 32-byte alignment boundary.

### Worked Example 7.3.2 — `f32x8`'s alignment, and what `Vec<f32>` does *not* guarantee around it

Chapter 5.2's `wide::f32x8` requires 32-byte alignment to build its underlying AVX registers correctly. CUDA's edition of this chapter could lean on `cudaMalloc`'s documented (if here undemonstrable) promise of at least 256-byte-aligned allocations to explain why reinterpreting a `cudaMalloc`'d buffer as `float4*` is safe in practice. Rust's `Vec<f32>` offers no comparable documented guarantee — its only promise, from `std::alloc::Layout`, is `align_of::<f32>() == 4` bytes.

```rust
let v: Vec<f32> = vec![0.0f32; 3];
let addr = v.as_ptr() as usize;
println!("addr % 32 = {}, addr % 4 = {}", addr % 32, addr % 4);
```

Genuinely run, 200 independent small `Vec<f32>` allocations in a row, each checked:

```
200/200 small Vec<f32> allocations happened to land 32-byte aligned
```

Every one of 200 fresh `Vec<f32>` allocations in this sandbox happened to come back 32-byte aligned — a real, measured fact about this system's global allocator (glibc's `malloc`), not a language guarantee. `align_of::<f32>() == 4` is the only alignment Rust's type system actually promises for `Vec<f32>`'s backing buffer; nothing stops a different allocator, a different platform, or a future glibc release from handing back a 4-byte-aligned (but not 32-byte-aligned) pointer for the exact same code. This is precisely why `wide::f32x8::new([f32; 8])` is a *constructor* that copies eight loose `f32` values in, rather than an unsafe reinterpretation of an existing slice's pointer: the crate cannot assume the alignment CUDA's `float4` gets to assume from `cudaMalloc`, so it never requires the caller to prove it.

> `[COMMON TRAP]` Chapter 6's `Tensor` struct itself is subject to this exact same padding rule — `TensorShape` (four `usize`s worth of `dims` plus `ndim`), `TensorStrides` (four more `usize`s), then a `Vec<f32>` (24 bytes: pointer, length, capacity, on a 64-bit target) — and reordering those fields *could*, in principle, change `Tensor`'s own `size_of` under `#[repr(C)]` the same way this section's toy structs did. Under Rust's default repr, which `Tensor` actually uses, the compiler already handles this automatically; the trap here is assuming that fact generalizes to *every* struct in every language, or forgetting that adding `#[repr(C)]` to `Tensor` later (for, say, C FFI or a future GPU-side layout) would silently reintroduce exactly the ordering sensitivity this section just measured.

## 7.4 Broadcasting: One Small Buffer, Many Logical Coordinates `[FOUNDATIONAL]`

### Intuition

Adding a per-column bias to every row of a matrix shouldn't require physically duplicating that bias once per row — the bias vector is the same three numbers no matter which row is asking for them. **Broadcasting** makes this concrete at the stride level: give the "extra," duplicated dimension a stride of exactly `0`. Section 6.4's offset formula doesn't need to know anything special about broadcasting — `offset = Σ coord[d] × stride[d]` already produces the right answer on its own the moment one of those strides is `0`, because multiplying any coordinate by `0` always contributes nothing to the sum.

### Background

| | An ordinary dimension | A broadcast dimension |
|---|---|---|
| Stride | Some positive value, reflecting real separation in memory | `0` |
| What changing that coordinate does to the offset | Moves to a different, real element | Nothing — the offset is unchanged regardless of the coordinate's value |
| Backing storage | One element's worth of memory per position along this dimension | A single element's worth of memory, reused for every position |

### Worked Example 7.4.1 — a `[3]` vector, broadcast against a `[4, 3]` shape

```rust
// A real [3] vector, stored contiguously.
let vec = [10.0f32, 20.0, 30.0];

// Broadcasting vec (shape [3]) against a target shape [4, 3]: the leading
// dimension (size 4) has NO backing storage of its own, so its stride is
// set to 0 -- every row reads the SAME 3 underlying floats. Zero bytes
// are duplicated; only the map claims a shape 4x its real size.
let broadcast_shape = Shape2D { dims: [4, 3] };
let broadcast_strides = Strides2D { strides: [0, 1] }; // stride 0 along the broadcast dimension
```

Compiled and run as the complete `05_broadcast_strides.rs` further below:

```bash
rustc --edition 2024 -O 05_broadcast_strides.rs -o 05_broadcast_strides
./05_broadcast_strides
```

Genuinely compiled and run:

```
real buffer: vec = [10, 20, 30]  (3 floats, actually stored)
broadcast shape = [4, 3], broadcast strides = [0, 1]

every 'row' of the broadcast view maps to the SAME 3 offsets:
  row 0 -> offsets (0,1,2) -> values (10,20,30)
  row 1 -> offsets (0,1,2) -> values (10,20,30)
  row 2 -> offsets (0,1,2) -> values (10,20,30)
  row 3 -> offsets (0,1,2) -> values (10,20,30)

all 4 rows alias the identical 3 offsets? true
```

Four different `row` values — `0`, `1`, `2`, `3` — each multiplied by the broadcast dimension's `0` stride, all contribute exactly `0` to the offset formula, so `offset(row, col)` collapses to `offset(0, col) = col` for every row: the loop's genuinely printed offsets, `(0,1,2)`, are identical across all four iterations, and the values read back — `10, 20, 30` — are the same three real floats every single time. A `[4,3]`-shaped view exists entirely on top of a 3-float buffer, with zero duplication. (Rust's `{}` formatter prints `10.0f32` as `10`, not `10.0` — a genuine, if cosmetic, difference from the earlier C++ edition's `%.1f`, not a discrepancy in the underlying values.)

### ASCII Diagram — four logical rows, one physical row

```
Real storage (3 floats):           Broadcast view, shape [4,3], strides [0,1]:
 [ 10.0  20.0  30.0 ]                logical row 0 -----\
                                     logical row 1 ------ all four point at the SAME
                                     logical row 2 ------ 3 physical floats above
                                     logical row 3 -----/
```

> `[COMMON TRAP]` Broadcasting is only safe as a *read* pattern. If four independent threads each believed they owned one "row" of this section's `[4,3]` broadcast view and each tried to *write* their own result back into it, all four threads would genuinely be writing to the same three physical addresses — a real data race, with the final value determined by whichever write happens to land last, not by any of the four threads' actual intended output. Rust's borrow checker does not save you here either: nothing about `TensorView<'a>`'s `&'a [f32]` prevents *reading* through aliased offsets, and a mutable, `&'a mut [f32]`-backed broadcast-write path would need genuine synchronization (an atomic add, or an explicit reduction) to be sound, not the borrow checker's ordinary aliasing rules, which govern references, not the arithmetic hazard of two logically-different coordinates resolving to the same physical address. This is exactly the situation a later automatic differentiation chapter will run into for real: when a broadcast input's gradient is computed, the incoming gradients from every logical row have to be *summed* into that one small buffer, rather than written directly — a genuinely different and more careful operation than the elementwise writes ordinary, non-broadcast tensors need.

## 7.5 Complete Runnable Code

### File: `01_layout_row_col_major.rs`

```rust
const MAX_DIMS: usize = 4;

#[derive(Debug, Clone, Copy)]
struct TensorShape {
    dims: [usize; MAX_DIMS],
    ndim: usize,
}

impl TensorShape {
    fn new(d0: usize, d1: usize, d2: usize) -> Self {
        TensorShape { dims: [d0, d1, d2, 0], ndim: 3 }
    }
    fn numel(&self) -> usize {
        let mut total = 1;
        for i in 0..self.ndim {
            total *= self.dims[i];
        }
        total
    }
}

#[derive(Debug, Clone, Copy)]
struct TensorStrides {
    strides: [usize; MAX_DIMS],
}

impl TensorStrides {
    fn row_major(shape: &TensorShape) -> Self {
        let mut strides = [0usize; MAX_DIMS];
        let mut running = 1;
        for i in (0..shape.ndim).rev() {
            strides[i] = running;
            running *= shape.dims[i];
        }
        TensorStrides { strides }
    }

    // Column-major: the FIRST dimension is contiguous instead of the last --
    // the convention cuBLAS (and any BLAS this book eventually links against) inherits
    // from Fortran BLAS.
    fn col_major(shape: &TensorShape) -> Self {
        let mut strides = [0usize; MAX_DIMS];
        let mut running = 1;
        for i in 0..shape.ndim {
            strides[i] = running;
            running *= shape.dims[i];
        }
        TensorStrides { strides }
    }

    fn offset3(&self, i0: usize, i1: usize, i2: usize) -> usize {
        i0 * self.strides[0] + i1 * self.strides[1] + i2 * self.strides[2]
    }
}

fn main() {
    let shape = TensorShape::new(2, 3, 4);
    let row = TensorStrides::row_major(&shape);
    let col = TensorStrides::col_major(&shape);
    println!("shape = {:?}", &shape.dims[0..shape.ndim]);
    println!("row-major strides = {:?}", &row.strides[0..shape.ndim]);
    println!("col-major strides = {:?}", &col.strides[0..shape.ndim]);

    println!();
    println!("index<->offset round trip for coordinate (1, 1, 2):");
    println!("row-major offset(1,1,2) = {}", row.offset3(1, 1, 2));
    println!("col-major offset(1,1,2) = {}", col.offset3(1, 1, 2));

    println!();
    println!("same coordinate, opposite contiguity:");
    println!("row-major offset(0,0,1) = {}  (innermost dim is contiguous)", row.offset3(0, 0, 1));
    println!("col-major offset(0,0,1) = {}  (innermost dim is NOT contiguous here)", col.offset3(0, 0, 1));
    println!("row-major offset(1,0,0) = {}  (outermost dim is NOT contiguous here)", row.offset3(1, 0, 0));
    println!("col-major offset(1,0,0) = {}  (outermost dim is contiguous)", col.offset3(1, 0, 0));

    println!();
    println!("both layouts agree at the very last coordinate (a permutation always agrees at the boundary):");
    println!(
        "row-major offset(1,2,3) = {}, col-major offset(1,2,3) = {}, numel-1 = {}",
        row.offset3(1, 2, 3),
        col.offset3(1, 2, 3),
        shape.numel() - 1
    );
}
```

```bash
rustc --edition 2024 -O 01_layout_row_col_major.rs -o 01_layout_row_col_major
./01_layout_row_col_major
```

Produces exactly the output shown in Worked Examples 7.1.1 and 7.1.2 above.

### File: `02_tensor_view.rs`

```rust
#[derive(Debug, Clone, Copy)]
struct Shape2D {
    dims: [usize; 2],
}

#[derive(Debug, Clone, Copy)]
struct Strides2D {
    strides: [usize; 2],
}

struct TensorView<'a> {
    data: &'a [f32],
    shape: Shape2D,
    strides: Strides2D,
}

impl<'a> TensorView<'a> {
    fn offset(&self, i0: usize, i1: usize) -> usize {
        i0 * self.strides.strides[0] + i1 * self.strides.strides[1]
    }

    fn get(&self, i0: usize, i1: usize) -> f32 {
        self.data[self.offset(i0, i1)]
    }

    // Transpose: SWAP shape and strides. Zero bytes move; only the map changes.
    fn transpose(&self) -> TensorView<'a> {
        TensorView {
            data: self.data,
            shape: Shape2D { dims: [self.shape.dims[1], self.shape.dims[0]] },
            strides: Strides2D { strides: [self.strides.strides[1], self.strides.strides[0]] },
        }
    }

    // A row slice: same strides, a shrunk shape, and the borrowed slice advanced by
    // exactly this row's own starting offset -- again, zero bytes move.
    fn row_slice(&self, row: usize) -> TensorView<'a> {
        let start = self.offset(row, 0);
        TensorView {
            data: &self.data[start..],
            shape: Shape2D { dims: [1, self.shape.dims[1]] },
            strides: self.strides,
        }
    }
}

fn main() {
    let buf: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let view = TensorView {
        data: &buf,
        shape: Shape2D { dims: [3, 4] },
        strides: Strides2D { strides: [4, 1] },
    };
    println!("original view: shape={:?}, strides={:?}", view.shape.dims, view.strides.strides);
    println!("view(1,2) = buf[{}] = {}", view.offset(1, 2), view.get(1, 2));

    println!();
    let t = view.transpose();
    let same_ptr = std::ptr::eq(t.data.as_ptr(), view.data.as_ptr());
    println!(
        "transposed view: shape={:?}, strides={:?}  (same buffer, same pointer: {})",
        t.shape.dims, t.strides.strides, same_ptr
    );
    println!(
        "t(2,1) = buf[{}] = {}  (should equal view(1,2) -- transpose just swaps the coordinate order)",
        t.offset(2, 1), t.get(2, 1)
    );

    println!();
    let r = view.row_slice(1);
    let advanced_by = unsafe { r.data.as_ptr().offset_from(buf.as_ptr()) };
    println!("row_slice(1): shape={:?}, data pointer advanced by {} elements", r.shape.dims, advanced_by);
    println!(
        "r(0,2) = {}  (should equal view(1,2) -- same element, reached through the slice)",
        r.get(0, 2)
    );
}
```

```bash
rustc --edition 2024 -O 02_tensor_view.rs -o 02_tensor_view
./02_tensor_view
```

Produces exactly the output shown in Worked Examples 7.2.1 and 7.2.2 above.

### File: `03_err_view_lifetime.rs`

This file is a **deliberately failing** compile-error demonstration (Worked Example 7.2.3), kept as a standalone file outside the main project — the same pattern Chapter 6's `04_err_use_after_move.rs` used, since a single non-compiling file would block `cargo build` for every other binary in the same package.

```rust
#[derive(Clone, Copy)]
struct Shape2D {
    dims: [usize; 2],
}

#[derive(Clone, Copy)]
struct Strides2D {
    strides: [usize; 2],
}

struct TensorView<'a> {
    data: &'a [f32],
    shape: Shape2D,
    strides: Strides2D,
}

impl<'a> TensorView<'a> {
    fn offset(&self, i0: usize, i1: usize) -> usize {
        i0 * self.strides.strides[0] + i1 * self.strides.strides[1]
    }
    fn get(&self, i0: usize, i1: usize) -> f32 {
        self.data[self.offset(i0, i1)]
    }
}

fn make_view() -> TensorView<'static> {
    let buf: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let view = TensorView {
        data: &buf,
        shape: Shape2D { dims: [2, 2] },
        strides: Strides2D { strides: [2, 1] },
    };
    view // ERROR: `buf` does not live long enough
}

fn main() {
    let v = make_view();
    println!("{}", v.get(0, 0));
}
```

```bash
rustc --edition 2024 03_err_view_lifetime.rs -o 03_err_view_lifetime
```

Produces exactly the `E0515` error shown in Worked Example 7.2.3 above, and does not produce a binary.

### File: `04_alignment_padding.rs`

This file depends on the `wide` crate (already a dependency since Chapter 5), for the `wide::f32x8` alignment tie-in.

```rust
use std::mem::{align_of, size_of};

// Fields interleaved by size -- but this is Rust's DEFAULT (unspecified) repr,
// which gives the compiler permission to reorder fields however it likes.
#[derive(Debug)]
struct InterleavedDefault {
    a: u8,
    b: i32,
    c: u8,
    d: i32,
}

// The SAME four fields, grouped largest-to-smallest -- also default repr.
#[derive(Debug)]
struct GroupedDefault {
    b: i32,
    d: i32,
    a: u8,
    c: u8,
}

// The same interleaved field order, but now #[repr(C)] forces the compiler
// to preserve declaration order exactly, the same way a C/C++ compiler does.
#[repr(C)]
#[derive(Debug)]
struct InterleavedC {
    a: u8,
    b: i32,
    c: u8,
    d: i32,
}

// The same grouped field order, also #[repr(C)].
#[repr(C)]
#[derive(Debug)]
struct GroupedC {
    b: i32,
    d: i32,
    a: u8,
    c: u8,
}

fn main() {
    println!("default (Rust) repr, compiler free to reorder fields:");
    println!(
        "  InterleavedDefault (u8,i32,u8,i32): size_of = {}, align_of = {}",
        size_of::<InterleavedDefault>(),
        align_of::<InterleavedDefault>()
    );
    println!(
        "  GroupedDefault     (i32,i32,u8,u8): size_of = {}, align_of = {}",
        size_of::<GroupedDefault>(),
        align_of::<GroupedDefault>()
    );
    println!(
        "  identical fields, same size regardless of declared order: {}",
        size_of::<InterleavedDefault>() == size_of::<GroupedDefault>()
    );

    println!();
    println!("#[repr(C)], declaration order preserved like C/C++:");
    println!(
        "  InterleavedC (u8,i32,u8,i32): size_of = {}, align_of = {}",
        size_of::<InterleavedC>(),
        align_of::<InterleavedC>()
    );
    println!(
        "  GroupedC     (i32,i32,u8,u8): size_of = {}, align_of = {}",
        size_of::<GroupedC>(),
        align_of::<GroupedC>()
    );
    println!(
        "  identical fields, {} fewer bytes just from reordering them",
        size_of::<InterleavedC>() as isize - size_of::<GroupedC>() as isize
    );

    println!();
    println!("Chapter 5's wide::f32x8, genuinely queried:");
    println!(
        "  align_of::<wide::f32x8>() = {}, size_of::<wide::f32x8>() = {}",
        align_of::<wide::f32x8>(),
        size_of::<wide::f32x8>()
    );
}
```

```bash
cargo build --release   # with wide = "0.7" in Cargo.toml
./target/release/04_alignment_padding
```

Produces exactly the output shown in Worked Example 7.3.1 above.

### File: `05_broadcast_strides.rs`

```rust
#[derive(Debug, Clone, Copy)]
struct Shape2D {
    dims: [usize; 2],
}

#[derive(Debug, Clone, Copy)]
struct Strides2D {
    strides: [usize; 2],
}

fn offset(strides: &Strides2D, i0: usize, i1: usize) -> usize {
    i0 * strides.strides[0] + i1 * strides.strides[1]
}

fn main() {
    // A real [3] vector, stored contiguously.
    let vec = [10.0f32, 20.0, 30.0];

    // Broadcasting vec (shape [3]) against a target shape [4, 3]: the leading
    // dimension (size 4) has NO backing storage of its own, so its stride is
    // set to 0 -- every row reads the SAME 3 underlying floats. Zero bytes
    // are duplicated; only the map claims a shape 4x its real size.
    let broadcast_shape = Shape2D { dims: [4, 3] };
    let broadcast_strides = Strides2D { strides: [0, 1] }; // stride 0 along the broadcast dimension

    println!(
        "real buffer: vec = [{}, {}, {}]  (3 floats, actually stored)",
        vec[0], vec[1], vec[2]
    );
    println!(
        "broadcast shape = {:?}, broadcast strides = {:?}",
        broadcast_shape.dims, broadcast_strides.strides
    );

    println!();
    println!("every 'row' of the broadcast view maps to the SAME 3 offsets:");
    let mut all_rows_offsets: Vec<[usize; 3]> = Vec::new();
    for row in 0..broadcast_shape.dims[0] {
        let offs = [
            offset(&broadcast_strides, row, 0),
            offset(&broadcast_strides, row, 1),
            offset(&broadcast_strides, row, 2),
        ];
        let vals = [vec[offs[0]], vec[offs[1]], vec[offs[2]]];
        println!(
            "  row {} -> offsets ({},{},{}) -> values ({},{},{})",
            row, offs[0], offs[1], offs[2], vals[0], vals[1], vals[2]
        );
        all_rows_offsets.push(offs);
    }

    println!();
    let all_alias = all_rows_offsets.iter().all(|o| *o == all_rows_offsets[0]);
    println!("all 4 rows alias the identical 3 offsets? {}", all_alias);
}
```

```bash
rustc --edition 2024 -O 05_broadcast_strides.rs -o 05_broadcast_strides
./05_broadcast_strides
```

Produces exactly the output shown in Worked Example 7.4.1 above.

Every number here was independently verified earlier in this chapter. All five files genuinely compile and run to completion in this sandbox with the real `rustc`/`cargo` toolchain — none of them touch a GPU or `cudarc`, so none of them are subject to this book's GPU-verification caveat; every output in this chapter is fully compiled, run, and cross-checked, not projected.

## Chapter Summary

The offset formula Chapter 6.4 wrote never changes across this entire chapter — only the strides feeding it do. Column-major strides, computed left-to-right instead of Chapter 6.2's right-to-left, produce a genuinely different memory layout for the identical shape, and matter concretely because BLAS libraries expect it. `TensorView<'a>` separates *owning* memory from *describing how to read* it: a borrowed slice plus a shape and strides, transposable and sliceable at zero cost because swapping or narrowing that description touches no actual bytes — and Rust's lifetime parameter goes one step further than Chapter 6's move-tracking already did, refusing at compile time to let a view outlive the buffer it borrows from, a class of bug CUDA's raw-pointer `TensorView` has no defense against at all. Struct field order changes `size_of` through alignment padding, but only once `#[repr(C)]` reimposes the declaration-order discipline C++ never had a choice about — under Rust's default repr, the compiler already reorders fields to minimize padding on its own, as both this chapter's `size_of` tests and Chapter 5's `wide::f32x8` (`align_of = 32`, confirmed directly) demonstrate. And broadcasting — setting a dimension's stride to `0` — lets a small buffer legitimately answer for a much larger logical shape, safe to read from freely but genuinely hazardous to write through directly without summing, a distinction a later automatic differentiation chapter will need to handle correctly once broadcast inputs need their gradients accumulated rather than overwritten.

## Self-Check Questions

1. For shape `[5, 2]`, compute both the row-major and column-major strides by hand, following Worked Example 7.1.1's process in both directions.
2. `TensorView::row_slice(row)` advances the borrowed slice's start by `offset(row, 0)` but keeps the *original* strides unchanged, rather than computing new ones. Explain why the original strides are still correct for indexing within the sliced row.
3. A struct holds one `f64` (8 bytes), one `u8` (1 byte), and one `f64` again (8 bytes), declared in that order, under `#[repr(C)]`. Would reordering it to `f64, f64, u8` change its `size_of`? Why or why not, in terms of this chapter's padding rule?
4. A broadcast view has shape `[6, 4]` with strides `[0, 1]`, backed by a real 4-element buffer. What does `offset(3, 2)` evaluate to, and what does `offset(5, 2)` evaluate to? Explain why both coordinates are legal even though the view claims 6 rows and only 1 row's worth of memory actually exists.
5. Worked Example 7.2.3 showed `rustc` refusing, at compile time, to let a `TensorView<'static>` borrow from a locally-scoped `Vec<f32>`. Explain why this same class of bug — a view outliving the buffer it points into — would compile cleanly in CUDA C++, and what would actually happen at runtime if the resulting dangling `TensorView` were dereferenced.

## Where We Go Next

`Tensor` can now be laid out two different ways, viewed without copying (and the compiler will refuse to let that view outlive its data), measured for its own padding, and stretched across a broadcast shape — but every tensor built so far has had its values decided one `println!` at a time. Chapter 8 builds the factory functions — `zeros`, `ones`, `arange`, `full`, and friends — that construct a `Tensor` already filled with a specific pattern of values, the ordinary starting point for essentially every tensor this book's later chapters actually use.

## Worked Solutions

**1.** Row-major, right-to-left: `i=1` sets `strides[1] = 1` (running starts at 1), then `running = 1 × dims[1] = 1 × 2 = 2`; `i=0` sets `strides[0] = 2`. Row-major strides: `[2, 1]`. Column-major, left-to-right: `i=0` sets `strides[0] = 1` (running starts at 1), then `running = 1 × dims[0] = 1 × 5 = 5`; `i=1` sets `strides[1] = 5`. Column-major strides: `[1, 5]`.

**2.** `row_slice(row)`'s new shape is `[1, shape.dims[1]]` — a single row, with the *same* number of columns and the *same* physical spacing between consecutive columns as the original view had. Since the original strides already say "moving one column right skips `strides[1]` flat elements," and a single row's columns are laid out exactly the same way inside the sliced view as they were inside the original, those strides remain correct; only the *starting point* (the slice's start) and the *shape* (now just one row) needed to change, not the spacing rule connecting one column to the next.

**3.** No — under `#[repr(C)]`, `size_of` would be unchanged. `f64, u8, f64` needs padding after the middle `u8` (7 bytes, to bring the second `f64` back to 8-byte alignment), for `8 + 1 + 7 + 8 = 24` total. `f64, f64, u8` places both 8-byte `f64`s back to back with no padding needed between them, then the 1-byte `u8`, then padding to bring the total to a multiple of 8: `8 + 8 + 1 + 7 = 24`. Both orderings need exactly 7 bytes of padding somewhere, because there is exactly one 1-byte field needing to be padded up to the struct's 8-byte alignment regardless of where it sits — Worked Example 7.3.1's savings came from having *two* small fields that could share a single padding tail when grouped together, which a single lone `u8` here can't do differently no matter where it's placed. (Under default repr, this question would not even arise — the compiler would already choose the smaller layout regardless of declared order.)

**4.** `offset(3, 2) = 3×0 + 2×1 = 2`. `offset(5, 2) = 5×0 + 2×1 = 2` — identical. Both are legal because the `0` stride on the first dimension means *no* coordinate value in that dimension ever contributes anything to the offset; the view's declared shape, `[6, 4]`, describes how many *logical* coordinates are valid to ask for, not how much *physical* memory backs them, and broadcasting's entire purpose is to let that gap exist deliberately.

**5.** CUDA C++'s `TensorView` holds a raw `float*` with no lifetime tracking at all — the language has no mechanism for a struct to declare "this pointer must not outlive the memory it points into," so a function returning a view into a local, stack-allocated (or otherwise about-to-be-freed) buffer compiles without a single warning under ordinary compiler settings. At runtime, the returned `TensorView`'s `data` pointer would point at memory the program no longer owns — memory that has very likely already been reused for something else by the time anything dereferences it. Reading through it would be undefined behavior: it might return old, stale values that happen to still be there, it might return unrelated garbage from whatever reused that stack space, or — depending on optimization and what else is running — it might crash outright. Rust's `'a` lifetime parameter makes this an explicit, checked part of `TensorView`'s type, so `rustc` catches the identical mistake at compile time, before the program ever runs, with the exact `E0515` error Worked Example 7.2.3 captured.
