# Chapter 6: The Tensor — Shape, Strides, and Ownership

> "Every worked example so far in this book has used a struct built for exactly one demonstration — a `Particle`, a `Buffer` that exists only to prove a destructor fires in the right order, a `Mixed`/`MixedC` pair that exists only to prove `#[repr(C)]` changes layout. This chapter builds the struct the rest of this book actually runs on: one type, `Tensor`, that carries its own shape, knows how to turn a coordinate into a flat offset, owns its memory the way Chapter 2.4 taught, and lets Rust's own ownership rules — not a hand-written guard — make copying it by accident a compile error instead of a silent bug."

**What you will understand by the end of this chapter:**

- `TensorShape`: a small, fixed-size array of dimension sizes and the one arithmetic fact — `numel = product of dims` — every later chapter's allocation size comes from
- `TensorStrides`: how row-major strides are computed from a shape alone, and why they're the one piece of information that turns a multi-dimensional coordinate into a single flat array offset
- How `Tensor` composes `TensorShape`, `TensorStrides`, and two owned buffers (`data` and `grad`) into one struct, directly building on Chapter 2.4's RAII discipline and Chapter 3.5's reason for keeping value and gradient in separate SoA-style arrays
- Indexing: the exact formula, `offset = Σ coord[d] × stride[d]`, that every later chapter's element access compiles down to, and why Rust never checks it for you inside this specific function
- Why `Tensor` is *not* `Copy` — and never could be, the moment it owns a `Vec<f32>` — so that `let t2 = t1;` genuinely moves ownership rather than duplicating it, with a real, compiler-caught `E0382` error the moment code tries to use `t1` afterward, and an explicit `.clone()` for the one legitimate way to get an independent copy

**What you need to know first:**

- Chapter 2.2 (constructor patterns and value semantics) and Chapter 2.4 (RAII via `Drop`) — this chapter is where those two chapters combine into the struct this book is actually building toward
- Chapter 3.5 (why this book's `Tensor` stores `data` and `grad` as two separate contiguous buffers rather than one array of `{value, grad}` pairs) and Chapter 5.5 (the same SoA argument, restated for SIMD)
- Chapter 1's genuine, compiler-caught errors (`E0502`, `E0382` already appeared there in a smaller example) — this chapter captures a real `E0382` of its own, on the exact struct this book uses from here forward
- A design decision this chapter states outright rather than leaving implicit: `Tensor`'s `data` and `grad` fields are `Vec<f32>` — ordinary, host-resident memory — not a `cudarc::driver::CudaSlice<f32>` (Chapter 4). This book's core `Tensor` type is CPU-resident by construction, fully genuine and fully verified in this sandbox with no `UNVERIFIED` tags anywhere in this chapter; a later chapter in this Part, Device Abstraction Layer, is where a second, `cudarc`-backed storage option gets layered in as an alternative, following this book's standing CPU/GPU verification split from Getting Started
- If you've read the CUDA or Mojo editions: both build the same three pieces — `TensorShape`, `TensorStrides`, and a `Tensor` type combining them with owned buffers — in the same order this chapter does. CUDA's edition enforces move-only ownership by hand, `= delete`-ing the copy constructor and writing an explicit move constructor; this chapter's Section 6.5 shows why Rust needs neither: a struct containing a `Vec<f32>` is automatically *not* `Copy`, and Rust's default assignment semantics are already a move, so the "delete the dangerous default, provide an explicit safe alternative" design CUDA's chapter builds by hand is simply how Rust structs already behave.

## 6.1 `TensorShape`: A Fixed-Size List of Dimension Sizes `[FOUNDATIONAL]`

### Intuition

A tensor's shape is nothing more than a short list of integers, one per dimension — `[2, 3, 4]` means "2 of the next thing, each containing 3 of the next thing, each containing 4 numbers." The one arithmetic fact this section builds everything else on: the total number of elements a shape describes, `numel`, is simply the product of every dimension size, because each dimension's count multiplies the number of things the dimension "inside" it repeats.

### Background

| | This chapter's `TensorShape` | Chapter 1's primitive types |
|---|---|---|
| Size known at | Compile time — a fixed-capacity array, `MAX_DIMS = 4` | Compile time |
| Storage | `dims: [usize; 4]` plus a `ndim: usize` recording how many are actually used | A handful of named fields |
| Computed property | `numel()` — the product of the first `ndim` entries of `dims` | N/A |
| `Copy`? | Yes — a fixed-size array of `usize` plus a `usize`, no heap allocation anywhere | Yes |

### Worked Example 6.1.1 — `[2, 3, 4]`, and its element count

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
```

Compiled and run as part of the complete `01_shape_and_strides.rs` further below:

```bash
rustc -O 01_shape_and_strides.rs -o 01_shape_and_strides
./01_shape_and_strides
```

Genuinely compiled and run:

```
shape = [2, 3, 4], ndim = 3
numel() = 24
```

`numel() = 2 × 3 × 4 = 24`, computed by the loop multiplying `dims[0]`, `dims[1]`, and `dims[2]` together — no shortcut, no hardcoded formula, just the general product-of-dims loop that works identically whether `ndim` is 1, 2, or the full `MAX_DIMS = 4`. This is the exact number Section 6.3's constructor uses to decide how large a `Vec<f32>` to allocate.

> `[COMMON TRAP]` `MAX_DIMS = 4` is a real, compile-time capacity limit, not a suggestion — `TensorShape` has no `Vec` underneath it to grow, by design, so that a `Tensor`'s shape lives entirely inline, as part of the struct itself, with no separate heap allocation of its own. A 5-dimensional tensor genuinely cannot be represented by this exact struct; a real framework would either raise `MAX_DIMS` or fall back to a heap-allocated shape (`Vec<usize>`) for the rare higher-rank case, a tradeoff this book does not need to make for the tensors Parts 2 through 7 actually build. `#[derive(Clone, Copy)]` here is not a formality either — it's only legal because every field is itself `Copy`, and it's precisely this fact, restated for `Tensor` itself in Section 6.5, that determines whether `#[derive(Copy)]` is even available to write.

## 6.2 `TensorStrides`: From a Coordinate to a Flat Offset `[FOUNDATIONAL]`

### Intuition

Underneath its nested, multi-dimensional shape, a tensor is still just one flat, contiguous buffer — Chapter 3 never stopped being true. **Strides** are the missing piece connecting the two pictures: `strides[d]` is exactly how many flat array slots you skip when you move one step along dimension `d`, holding every other dimension fixed. For row-major layout — the only layout this book's `Tensor` uses — the last dimension is the most tightly packed (`stride = 1`), and each dimension moving outward multiplies the previous dimension's stride by that previous dimension's own size.

### Background

| | Computing `strides[ndim-1]` | Computing any earlier `strides[d]` |
|---|---|---|
| Row-major rule | Always `1` — the last dimension is contiguous | `strides[d] = strides[d+1] × dims[d+1]` |
| Direction computed | N/A (base case) | Right to left — each stride depends on the one after it |

### Worked Example 6.2.1 — strides for the same `[2, 3, 4]` shape

```rust
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
}
```

Compiled and run as part of `01_shape_and_strides.rs`:

```
strides = [12, 4, 1]
```

Traced right to left: `i=2` (the last dimension) sets `strides[2] = 1` (the running product starts at 1), then updates `running = 1 × dims[2] = 1 × 4 = 4`. `i=1` sets `strides[1] = running = 4`, then updates `running = 4 × dims[1] = 4 × 3 = 12`. `i=0` sets `strides[0] = running = 12`. Result: `[12, 4, 1]` — moving one step along dimension 0 (the outermost) skips 12 flat elements, exactly the size of one whole `[3,4]` sub-tensor; moving one step along dimension 2 (the innermost) skips exactly 1, confirming the last dimension really is the contiguous one.

`(0..shape.ndim).rev()` is Rust's way of writing CUDA's `for (int i = ndim-1; i >= 0; i--)` — a `Range` counting up, reversed, over `usize` indices that can never go negative in the first place, sidestepping the underflow risk a raw `i >= 0` loop over an unsigned type would otherwise carry.

### ASCII Diagram — nested shape, one flat buffer underneath

```
Logical shape [2, 3, 4]:                Flat buffer, 24 elements, strides=[12,4,1]:
 [ [ [.,.,.,.], [.,.,.,.], [.,.,.,.] ],   [ 0  1  2  3 | 4  5  6  7 | 8  9  10 11 |
   [ [.,.,.,.], [.,.,.,.], [.,.,.,.] ] ]   12 13 14 15 | 16 17 18 19 | 20 21 22 23 ]
                                            \___________________________________/
 dim 0 (size 2): jump by 12 to move to    \___________/
 the second [3,4] block                  dim 1 (size 3): jump by 4 to move
                                          to the next [4]-row within a block
```

> `[COMMON TRAP]` Reading a multi-dimensional coordinate's flat offset is a *sum* across every dimension, not a single stride lookup: `offset = i0*strides[0] + i1*strides[1] + i2*strides[2]`, all three terms added together. It's a common mistake to reach for just the innermost dimension's stride (`strides[2] = 1`) and treat the coordinate's last index as "the" offset, forgetting that `i0` and `i1` each contribute their own, much larger jump — Section 6.4 makes this formula concrete and checks it against hand-computed values.

## 6.3 Composing `Tensor`: Shape, Strides, and Two Owned Buffers `[FOUNDATIONAL]`

### Intuition

This book's actual `Tensor` needs Chapter 2.4's RAII discipline twice over: once for the tensor's own values (`data`), and once for the gradient Part 4's autodiff engine will eventually accumulate into (`grad`) — kept as a second, separate buffer rather than interleaved with `data`, for precisely the reason Chapter 3.5 and Chapter 5.5 already argued in full. `Tensor` is nothing conceptually new at this point; it's `TensorShape` plus `TensorStrides` plus two owned `Vec<f32>` allocations, composed into one struct.

### Background

| Field | Type | Role |
|---|---|---|
| `shape` | `TensorShape` | How many elements, along how many dimensions (Section 6.1) |
| `strides` | `TensorStrides` | How to turn a coordinate into a flat offset (Section 6.2) |
| `data` | `Vec<f32>`, owned | The tensor's own values — `shape.numel()` floats, allocated in the constructor |
| `grad` | `Vec<f32>`, owned | Space for a gradient of the same shape — allocated alongside `data`, unused until Part 4 |

### Worked Example 6.3.1 — one constructor call, two allocations

```rust
struct Tensor {
    shape: TensorShape,
    strides: TensorStrides,
    data: Vec<f32>,
    grad: Vec<f32>,
}

impl Tensor {
    fn new(shape: TensorShape) -> Self {
        let n = shape.numel();
        let strides = TensorStrides::row_major(&shape);
        let data = vec![0.0f32; n];
        let grad = vec![0.0f32; n];
        println!(
            "Tensor(numel={}) constructed -> data.len()={}, grad.len()={}",
            n, data.len(), grad.len()
        );
        Tensor { shape, strides, data, grad }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        println!(
            "~Tensor(numel={}) destructor firing, freeing data (len={}), grad (len={})",
            self.shape.numel(), self.data.len(), self.grad.len()
        );
    }
}
```

Compiled and run as the complete `02_tensor_struct.rs` further below:

```bash
rustc -O 02_tensor_struct.rs -o 02_tensor_struct
./02_tensor_struct
```

Genuinely compiled and run:

```
constructing t
Tensor(numel=24) constructed -> data.len()=24, grad.len()=24
t.shape.numel() = 24
t.strides = [12, 4, 1]
leaving main, destructor about to fire
~Tensor(numel=24) destructor firing, freeing data (len=24), grad (len=24)
```

One constructor call, `Tensor::new(TensorShape::new(2, 3, 4))`, genuinely triggers two independent `vec![0.0f32; n]` allocations — `numel() = 24` computed once (Section 6.1's formula) and reused for both allocation sizes. Unlike the CUDA edition's equivalent worked example, there is nothing to honestly report as a device error here: `Vec::new`-style allocation is ordinary host memory, and it genuinely succeeds every time in this sandbox — this chapter's `Tensor` is fully, unconditionally verified, with the GPU-backed alternative deferred to a later chapter by design, not by necessity.

> `[COMMON TRAP]` `Tensor` now owns *two* heap allocations instead of one — which is exactly why Section 6.5 matters. If `Tensor` could be duplicated by ordinary assignment the way `TensorShape` can (`#[derive(Copy)]`), that duplication would need to either deep-copy two `Vec<f32>`s silently (surprising, expensive, easy to write by accident) or copy just the two internal pointers (a double-free waiting to happen, the exact bug CUDA's chapter deletes a copy constructor to prevent). Rust doesn't need a hand-written guard against the second failure mode, because `Vec<f32>` itself is never `Copy` — and a struct containing a non-`Copy` field can never derive `Copy` either, full stop.

## 6.4 Indexing: The Offset Formula, Made Concrete `[FOUNDATIONAL]`

### Intuition

Section 6.2's Common Trap named the formula; this section runs it. `offset(i0, i1, i2)` takes a three-dimensional coordinate and returns the single flat index Section 6.3's `data` should be read or written at — pure arithmetic on `usize`s, touching no memory at all, which makes it one of the few `Tensor` operations this book can genuinely run to completion and check by hand.

### Background

| Coordinate | Formula | For shape `[2,3,4]`, strides `[12,4,1]` |
|---|---|---|
| `(0, 0, 0)` | `0*12 + 0*4 + 0*1` | `0` — the very first flat element |
| `(0, 0, 1)` | `0*12 + 0*4 + 1*1` | `1` — one step along the innermost, contiguous dimension |
| `(1, 2, 3)` | `1*12 + 2*4 + 3*1` | `12+8+3 = 23` — the *last* valid coordinate for this shape |

### Worked Example 6.4.1 — five coordinates, hand-traced and confirmed

```rust
fn offset(strides: &TensorStrides, i0: usize, i1: usize, i2: usize) -> usize {
    i0 * strides.strides[0] + i1 * strides.strides[1] + i2 * strides.strides[2]
}
```

Compiled and run as part of `01_shape_and_strides.rs`:

```
shape = [2, 3, 4], strides = [12, 4, 1], numel = 24
offset(0,0,0) = 0
offset(0,0,1) = 1
offset(0,1,0) = 4
offset(1,0,0) = 12
offset(1,2,3) = 23  (the last valid coordinate; numel-1 = 23)
```

Every value matches the Background table's hand trace exactly, and `offset(1,2,3) = 23 = numel() - 1` is not a coincidence: `(1,2,3)` is the largest legal coordinate for a `[2,3,4]` shape (indices `0`–`1` for dimension 0, `0`–`2` for dimension 1, `0`–`3` for dimension 2), and a correct row-major layout guarantees the largest coordinate always maps to the very last flat slot.

> `[COMMON TRAP]` `offset()` performs no bounds checking whatsoever — `offset(&strides, 5, 0, 0)` for this same `[2,3,4]` shape compiles cleanly and returns `5*12 = 60`, a flat index 36 slots past the end of a 24-element buffer, with nothing in `offset()`'s own type signature to stop it. This is a genuinely different concern from indexing `data` itself: `data[60]` on a 24-element `Vec<f32>` *would* panic at runtime (`Vec`'s own `Index` bounds-checks every access, unlike a raw pointer), but `offset()` computing a nonsensical number in the first place is still a real bug, just one Rust catches one step later than you might expect — at the point of use, not at the point of computing the bad offset. A production `Tensor` would typically add an opt-in, debug-only assertion (`debug_assert!(i0 < shape.dims[0])`, compiled out entirely in a release build) rather than pay for one on every access unconditionally — a decision this book leaves as-is, exactly as the CUDA edition does, since Parts 2 through 7 write their own indexing carefully enough that a separately checked `Tensor` variant isn't worth this book's added complexity.

## 6.5 Move, Clone, and Rust's Ownership Rules `[FOUNDATIONAL]`

### Intuition

Section 6.3's Common Trap named the danger; this section shows why Rust closes it automatically rather than by hand. `Tensor` contains two `Vec<f32>` fields, and `Vec<T>` is never `Copy`, for exactly the reason CUDA's chapter deletes its copy constructor: a `Vec` owns a heap allocation, and duplicating the struct that contains it by blindly copying bytes would duplicate the pointer, not the data — two owners, one buffer, a double-free waiting to happen. Rust's rule is simple and structural: a struct can derive `Copy` only if *every* field is `Copy`. `Tensor` has two fields that aren't, so `Tensor` isn't `Copy` — not because this chapter chose to forbid it, but because the type system can prove, from `Vec`'s own definition, that it would be unsound.

### Background

| | Ordinary assignment (`let t2 = t1;`) | `.clone()` |
|---|---|---|
| What happens to `data`/`grad` | Ownership of the existing `Vec<f32>`s transfers to `t2` — no allocation, no copy of contents | Two new `Vec<f32>` allocations, contents copied into them |
| Is `t1` still usable afterward? | No — this is a **move**, and using `t1` again is a compile error | Yes — `t1` (or `t2`, whichever is cloned) is untouched |
| Cost | Free — a few `usize`s and pointers copied, the underlying heap memory doesn't move | A real `O(n)` copy, proportional to `data.len() + grad.len()` |

### Worked Example 6.5.1 — a move, genuinely compiled and run

```rust
let t1 = Tensor::new("t1", TensorShape::new(2, 3, 4));
let t2 = t1; // MOVE: t1's fields transferred to t2
println!("t2.shape.numel() = {}", t2.shape.numel());
```

Compiled and run as part of the complete `03_ownership_move_and_clone.rs` further below:

```bash
rustc -O 03_ownership_move_and_clone.rs -o 03_ownership_move_and_clone
./03_ownership_move_and_clone
```

Genuinely compiled and run:

```
--- move: t1 -> t2, no copy, t1 is no longer usable ---
Tensor(t1, numel=24) constructed -> data.len()=24, grad.len()=24
t2.shape.numel() = 24 (t2 now owns what used to be t1's buffers)
```

`let t2 = t1;` moved every field of `t1` into `t2` — including a `name` field this worked example's `Tensor` carries purely to trace identity — which is worth stating precisely: `t2` isn't a *new* tensor that happens to have the same values as `t1`; it *is* what used to be called `t1`, now reachable under a different variable name. Section 6.6's destructor trace below confirms this directly: when `t2` is eventually dropped, it prints `~Tensor(t1) destructor firing` — its `name` field, moved unchanged from `t1`, never got updated, because a move doesn't construct a new value, it relocates the existing one.

### Worked Example 6.5.2 — the same move, misused, and the genuine compile error

```rust
let t1 = Tensor::new(TensorShape::new(2, 3, 4));
let t2 = t1;
println!("t2.data.len() = {}", t2.data.len());
println!("t1.data.len() = {}", t1.data.len()); // ERROR: use of moved value
```

Compiled as the complete `04_err_use_after_move.rs` further below:

```bash
rustc --edition 2024 04_err_use_after_move.rs -o /tmp/err_use_after_move_bin
```

Genuinely compiled, and genuinely refused:

```
error[E0382]: borrow of moved value: `t1`
  --> 04_err_use_after_move.rs:38:36
   |
35 |     let t1 = Tensor::new(TensorShape::new(2, 3, 4));
   |         -- move occurs because `t1` has type `Tensor`, which does not implement the `Copy` trait
36 |     let t2 = t1; // MOVE: t1's fields transferred to t2
   |              -- value moved here
37 |     println!("t2.data.len() = {}", t2.data.len());
38 |     println!("t1.data.len() = {}", t1.data.len()); // ERROR: use of moved value
   |                                    ^^^^^^^ value borrowed here after move
   |
note: if `Tensor` implemented `Clone`, you could clone the value
  --> 04_err_use_after_move.rs:22:1
   |
22 | struct Tensor {
   | ^^^^^^^^^^^^^ consider implementing `Clone` for this type
...
36 |     let t2 = t1; // MOVE: t1's fields transferred to t2
   |              -- you could clone this value

error: aborting due to 1 previous error

For more information about this error, try `rustc --explain E0382`.
```

This is the exact same class of enforcement CUDA's edition gets by manually writing `Tensor(const Tensor&) = delete;` — a mistake caught during compilation, before the program ever runs — except Rust's compiler produces this error *automatically*, from `Tensor` simply containing a `Vec<f32>`, with no explicit `= delete` anywhere in this chapter's code. The compiler's own suggestion in the note above — "if `Tensor` implemented `Clone`, you could clone the value" — is precisely Section 6.5's next worked example.

### Worked Example 6.5.3 — `.clone()`, and genuine value independence

```rust
#[derive(Clone)]
struct Tensor { /* ... */ }

let mut t3 = t2.clone();
t3.data[0] = 99.0;
```

Compiled and run as part of `03_ownership_move_and_clone.rs`:

```
--- clone(): t2 -> t3, a genuine deep copy, independently owned ---
t2.data.len()=24, t3.data.len()=24 (two separate Vec<f32> allocations)

--- proving value independence: mutate t3, check t2 is untouched ---
before mutation: t2.data[0]=0, t3.data[0]=0
after t3.data[0]=99.0: t2.data[0]=0, t3.data[0]=99

leaving main -- t3 then t2 destructors fire, LIFO:
~Tensor(t3) destructor firing
~Tensor(t1) destructor firing
```

`#[derive(Clone)]` generates exactly the deep-copy `clone()` this chapter needs — for a struct made of `TensorShape` (itself `Copy`), `TensorStrides` (`Copy`), and two `Vec<f32>`s, the derived implementation clones each field, and `Vec<T>::clone()` allocates a genuinely new buffer and copies every element into it. Because this is real, host-resident memory, this chapter can prove independence directly — no host-side stand-in needed the way the CUDA edition requires for its device-only buffers: `t3.data[0] = 99.0` changes only `t3`'s own allocation, leaving `t2.data[0]` at `0` exactly as constructed. `t3` and `t2`'s destructors fire in the reverse of their construction order — `t3` first, then `t2` (still printing its original `t1` name, per Worked Example 6.5.1) — the same LIFO discipline Chapter 2.4's `Buffer` first demonstrated.

> `[COMMON TRAP]` It's tempting to read "moves are free, clones cost a real copy" and conclude a function should always take `Tensor` by move when it doesn't need to keep the original around. That's usually backwards for the same reason Chapter 2's discipline already established: a function that only needs to *read* a `Tensor` should take `&Tensor`, not consume it by value — taking ownership needlessly forces every caller to either give up their own `Tensor` or add an explicit `.clone()` just to keep using it afterward. Every function this book writes from here forward takes a `Tensor` by shared reference (`&Tensor`) to read it, by mutable reference (`&mut Tensor`) to mutate it in place, and by value only when it genuinely intends to consume or construct one — exactly the discipline CUDA's Common Trap states for `const Tensor&`/`Tensor&`, arrived at here as Rust's ordinary default rather than a rule this book has to remember to apply.

## Complete Runnable Code

### File: `01_shape_and_strides.rs`

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
}

fn offset(strides: &TensorStrides, i0: usize, i1: usize, i2: usize) -> usize {
    i0 * strides.strides[0] + i1 * strides.strides[1] + i2 * strides.strides[2]
}

fn main() {
    let shape = TensorShape::new(2, 3, 4);
    println!("shape = {:?}, ndim = {}", &shape.dims[0..shape.ndim], shape.ndim);
    println!("numel() = {}", shape.numel());

    println!();
    let strides = TensorStrides::row_major(&shape);
    println!("strides = {:?}", &strides.strides[0..shape.ndim]);

    println!();
    println!("shape = {:?}, strides = {:?}, numel = {}", &shape.dims[0..shape.ndim], &strides.strides[0..shape.ndim], shape.numel());
    println!("offset(0,0,0) = {}", offset(&strides, 0, 0, 0));
    println!("offset(0,0,1) = {}", offset(&strides, 0, 0, 1));
    println!("offset(0,1,0) = {}", offset(&strides, 0, 1, 0));
    println!("offset(1,0,0) = {}", offset(&strides, 1, 0, 0));
    println!(
        "offset(1,2,3) = {}  (the last valid coordinate; numel-1 = {})",
        offset(&strides, 1, 2, 3),
        shape.numel() - 1
    );
}
```

```bash
rustc -O 01_shape_and_strides.rs -o 01_shape_and_strides
./01_shape_and_strides
```

### File: `02_tensor_struct.rs`

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
}

#[derive(Debug)]
struct Tensor {
    shape: TensorShape,
    strides: TensorStrides,
    data: Vec<f32>,
    grad: Vec<f32>,
}

impl Tensor {
    fn new(shape: TensorShape) -> Self {
        let n = shape.numel();
        let strides = TensorStrides::row_major(&shape);
        let data = vec![0.0f32; n];
        let grad = vec![0.0f32; n];
        println!(
            "Tensor(numel={}) constructed -> data.len()={}, grad.len()={}",
            n,
            data.len(),
            grad.len()
        );
        Tensor { shape, strides, data, grad }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        println!(
            "~Tensor(numel={}) destructor firing, freeing data (len={}), grad (len={})",
            self.shape.numel(),
            self.data.len(),
            self.grad.len()
        );
    }
}

fn main() {
    println!("constructing t");
    let t = Tensor::new(TensorShape::new(2, 3, 4));
    println!("t.shape.numel() = {}", t.shape.numel());
    println!("t.strides = {:?}", &t.strides.strides[0..t.shape.ndim]);
    println!("leaving main, destructor about to fire");
}
```

```bash
rustc -O 02_tensor_struct.rs -o 02_tensor_struct
./02_tensor_struct
```

### File: `03_ownership_move_and_clone.rs`

```rust
const MAX_DIMS: usize = 4;

#[allow(dead_code)]
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

#[allow(dead_code)]
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
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Tensor {
    shape: TensorShape,
    strides: TensorStrides,
    data: Vec<f32>,
    grad: Vec<f32>,
    name: &'static str, // just for these worked examples, to trace which Tensor is which
}

impl Tensor {
    fn new(name: &'static str, shape: TensorShape) -> Self {
        let n = shape.numel();
        let strides = TensorStrides::row_major(&shape);
        println!("Tensor({name}, numel={n}) constructed -> data.len()={n}, grad.len()={n}");
        Tensor {
            shape,
            strides,
            data: vec![0.0f32; n],
            grad: vec![0.0f32; n],
            name,
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        println!("~Tensor({}) destructor firing", self.name);
    }
}

fn main() {
    println!("--- move: t1 -> t2, no copy, t1 is no longer usable ---");
    let t1 = Tensor::new("t1", TensorShape::new(2, 3, 4));
    let t2 = t1; // MOVE: t1's fields transferred to t2, t1 can no longer be used
    println!("t2.shape.numel() = {} (t2 now owns what used to be t1's buffers)", t2.shape.numel());
    // t1.data.len() here would be a compile error -- see 04_err_use_after_move.rs

    println!();
    println!("--- clone(): t2 -> t3, a genuine deep copy, independently owned ---");
    let mut t3 = t2.clone();
    t3.name = "t3";
    println!("t2.data.len()={}, t3.data.len()={} (two separate Vec<f32> allocations)", t2.data.len(), t3.data.len());

    println!();
    println!("--- proving value independence: mutate t3, check t2 is untouched ---");
    println!("before mutation: t2.data[0]={}, t3.data[0]={}", t2.data[0], t3.data[0]);
    t3.data[0] = 99.0;
    println!("after t3.data[0]=99.0: t2.data[0]={}, t3.data[0]={}", t2.data[0], t3.data[0]);

    println!();
    println!("leaving main -- t3 then t2 destructors fire, LIFO:");
}
```

```bash
rustc -O 03_ownership_move_and_clone.rs -o 03_ownership_move_and_clone
./03_ownership_move_and_clone
```

### File: `04_err_use_after_move.rs`

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

struct Tensor {
    shape: TensorShape,
    data: Vec<f32>,
}

impl Tensor {
    fn new(shape: TensorShape) -> Self {
        let n = shape.numel();
        Tensor { shape, data: vec![0.0f32; n] }
    }
}

fn main() {
    let t1 = Tensor::new(TensorShape::new(2, 3, 4));
    let t2 = t1; // MOVE: t1's fields transferred to t2
    println!("t2.data.len() = {}", t2.data.len());
    println!("t1.data.len() = {}", t1.data.len()); // ERROR: use of moved value
}
```

```bash
rustc --edition 2024 04_err_use_after_move.rs -o /tmp/err_use_after_move_bin
```

This file is expected to fail to compile — that failure, captured verbatim in Worked Example 6.5.2, is the point.

## Chapter Summary

`TensorShape` is a fixed-size, `Copy` array of dimension sizes plus a `ndim`, whose `numel()` — genuinely computed as `2×3×4=24` for this chapter's running example — is the one number every later allocation size in this book comes from. `TensorStrides::row_major` computes `[12, 4, 1]` for that same shape by walking dimensions right to left, the last dimension always contiguous (`stride=1`) and each earlier stride the product of everything to its right. `Tensor` composes both with two owned `Vec<f32>` buffers, `data` and `grad`, kept separate for the SoA reasons Chapters 3.5 and 5.5 already established — and because this chapter's `Tensor` is CPU-resident by design, its constructor is unconditionally, genuinely verified in this sandbox, with no device errors to honestly report and no `UNVERIFIED` tags anywhere in this chapter. `offset(i0,i1,i2)` turns a coordinate into a flat index via `Σ coord[d]×stride[d]`, genuinely matching every hand-traced value from `0` through `23` for this chapter's shape, with no bounds checking of its own. Most centrally: `Tensor` contains two `Vec<f32>` fields, and because `Vec<T>` is never `Copy`, `Tensor` cannot be `Copy` either — a structural fact the compiler enforces, not a rule this book had to add by hand. `let t2 = t1;` is therefore a genuine move, confirmed both by a successful run (Worked Example 6.5.1) and by a real, compiler-refused attempt to use `t1` afterward (`E0382`, captured verbatim in Worked Example 6.5.2) — and `.clone()`, derived automatically, provides the one legitimate way to get an independent copy, its value independence proven directly against real host memory rather than a host-side stand-in.

## Self-Check Questions

1. For a shape `[4, 5, 6]`, compute `numel()` and `strides` (row-major) by hand, following Section 6.1 and 6.2's exact method.
2. `TensorShape` and `TensorStrides` both derive `Copy`, but `Tensor` cannot. State the specific structural reason, referring to `Tensor`'s actual fields.
3. Worked Example 6.5.1 shows `t2`'s destructor printing `~Tensor(t1) destructor firing`, not `~Tensor(t2)`. Explain why, in terms of what a move actually does to a struct's fields.
4. `offset(&strides, 5, 0, 0)` for a `[2,3,4]`-shaped tensor compiles and returns `60`. Explain the two separate reasons this doesn't immediately crash the program, and where a bounds violation stemming from this bad offset would actually be caught.
5. Explain why a function that only needs to read a `Tensor`'s values should take `&Tensor` rather than `Tensor`, in terms of what taking `Tensor` by value would force every caller to do.

## Where We Go Next

`Tensor` now has a shape, strides, an indexing formula, and correct, compiler-enforced ownership — but every `Tensor` this chapter constructed started out full of zeros, and every field this chapter accessed was reached by hand-writing struct literals in `main()`. Chapter 7 turns to memory layout design: how `Tensor`'s row-major assumption compares to alternatives, and the layout-level decisions this book commits to before Chapter 8 gives `Tensor` real, ergonomic constructors — `zeros()`, `ones()`, `from_vec()` — in place of this chapter's bare struct literals.

## Worked Solutions

**1.** `numel() = 4 × 5 × 6 = 120`. Strides, computed right to left: `strides[2] = 1` (base case), `running = 1 × 6 = 6`; `strides[1] = 6`, `running = 6 × 5 = 30`; `strides[0] = 30`. Result: `strides = [30, 6, 1]` — moving one step along dimension 0 skips 30 elements (one whole `[5,6]` sub-tensor), matching `numel()/dims[0] = 120/4 = 30` exactly, the same cross-check Section 6.2's diagram makes visually.

**2.** `Tensor` has two `Vec<f32>` fields, `data` and `grad`. `Vec<T>` owns a heap allocation and is never `Copy` for any `T` — copying a `Vec` by bit-for-bit duplication would duplicate its pointer without duplicating its contents, producing two owners of the same heap memory, exactly the double-free risk CUDA's edition deletes a copy constructor to prevent. Rust's derive rule is structural: `#[derive(Copy)]` is only valid when every field of a struct is itself `Copy`. Since `Vec<f32>` isn't, `Tensor` isn't either — regardless of how many other fields (`shape`, `strides`) are individually `Copy`.

**3.** A move transfers a struct's fields exactly as they were, byte for byte — it does not run any constructor or re-initialize anything. `t1`'s `name` field held the string `"t1"`; `let t2 = t1;` moved that exact field, unchanged, into `t2`. `t2` is not a fresh `Tensor` that happens to resemble `t1` — it is the same value, reachable under a new variable name, and its `name` field says so because nothing about a move touches field values, only which variable is allowed to use them.

**4.** First: `offset()` itself performs pure arithmetic on `usize`s — `5 * 12 + 0 * 4 + 0 * 1 = 60` is a completely well-defined computation with no memory access involved, so there's nothing for it to crash on. Second: `60` merely being an out-of-range number doesn't cause a problem until something actually uses it to touch memory — specifically, `data[60]` on a 24-element `Vec<f32>`, which is where `Vec`'s own bounds-checked `Index` implementation would panic at runtime. The bad value can be computed, stored, and passed around freely; the actual failure is deferred to the first genuine out-of-bounds access.

**5.** Taking `Tensor` by value means the function *takes ownership* of the caller's `Tensor` — after the call, the caller no longer has it, exactly as Worked Example 6.5.2's `t1` became unusable after `let t2 = t1;`. A caller who still needs their `Tensor` afterward would be forced to either restructure their code around no longer having it, or add an explicit `.clone()` before every such call just to keep a usable copy — paying Worked Example 6.5.3's real `O(n)` copy cost for what was only ever meant to be a read. Taking `&Tensor` instead borrows access without taking ownership, so the caller keeps their `Tensor`, no allocation happens, and the function's signature honestly documents that it only reads what's passed in.
