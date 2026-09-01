# Chapter 2: Struct Design Patterns

> "Chapter 1 established that a value has exactly one owner. A struct doesn't get a special exemption from that rule — its ownership is just the composition of its fields' ownership, decided once at the type's definition and enforced identically at every call site. A struct that owns a heap allocation is exactly as strict, and exactly as free of accidents, as the single field it wraps — provided you build it the way this chapter shows, and not the way its one deliberately dangerous counter-example does."

**What you will understand by the end of this chapter:**

- What a `struct` actually is in memory — a fixed, compile-time-known layout of named fields — and the one guarantee CUDA C++ gives you here that Rust genuinely doesn't: declaration order is *not* memory order unless you ask for it explicitly with `#[repr(C)]`
- Why Rust has no constructor overloading at all — no two functions or methods can share a name in the same scope — and how associated functions (`Point2D::new(...)`) and the `Default` trait fill that role instead
- Why Rust structs, unlike C++ structs, do *not* get an automatically-generated copy — assignment moves by default, exactly as Chapter 1.3 established for `String`, even when every field is a plain `f32`
- How Rust's generics get compiled into one genuinely separate, fully-specialized piece of machine code per concrete type — **monomorphization** — verified directly as distinct symbols in a compiled binary, the same evidence CUDA's chapter used for C++ templates
- How Rust's `Drop` trait gives RAII without the double-free hazard C++'s copy constructor leaves open by default — and, when you deliberately reach for `unsafe` and a raw pointer instead of an owned field, exactly how that hazard reappears

**What you need to know first:**

- Chapter 1 in full — this chapter reuses its stack-layout reasoning (Section 1.1), its ownership and move semantics (Section 1.3), and its borrow rules (Section 1.4) without re-explaining them
- If you've read the CUDA edition: this chapter follows its Chapter 2 section-for-section (struct layout, multiple constructors, generics, RAII, compile-time interfaces) — but several of the substitutions flip the story rather than just renaming it. CUDA C++ has no first-class trait keyword and reaches for the CRTP workaround; Rust's `trait` is first-class, so Section 2.5 here is a genuine choice between two real dispatch mechanisms, not a workaround for a missing one. C++ auto-generates a struct's copy constructor; Rust generates nothing unless you ask, which Section 2.2 demonstrates directly.

## 2.1 What a Struct Actually Is `[FOUNDATIONAL]`

### Intuition

An architectural blueprint doesn't describe *a* house — it describes the layout every house built from it will share: the front door on the north wall, the kitchen four meters from it. A `struct` is that blueprint applied to memory instead of a plot of land: `struct Point2D { x: f32, y: f32 }` fixes that every `Point2D` has an `x` field and a `y` field, both `f32`. What it does *not* fix, unlike CUDA C++, is which byte offset each field lands at — Rust's compiler is explicitly permitted to reorder a struct's fields in memory to reduce padding, and by default, it will.

### Background

| | C / C++ `struct` | Rust `struct` (default layout) |
|---|---|---|
| Field layout | Fixed offsets, in declaration order, guaranteed by the language standard | Fixed offsets, but *not* guaranteed to match declaration order — the compiler may reorder fields to minimize padding |
| Getting C-compatible, declaration-order layout | The default | Opt-in only, via `#[repr(C)]` |
| Methods | Not supported directly (free functions instead) | Attached via a separate `impl` block, not inside the struct body itself |
| Constructor | A special member function with the type's own name | No special syntax — an ordinary associated function, conventionally named `new` |

### Worked Example 2.1.1 — the exact byte layout, genuinely measured with `offset_of!`

```rust
struct Point2D {
    x: f32,
    y: f32,
}

impl Point2D {
    fn new(x: f32, y: f32) -> Self {
        Point2D { x, y }
    }

    fn distance_from_origin(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn distance_to(&self, other: &Point2D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}
```

Compiled and run as the complete `01_basic_structs.rs` further below:

```bash
rustc -O 01_basic_structs.rs -o 01_basic_structs
./01_basic_structs
```

Genuinely compiled and run:

```
size_of::<Point2D>() = 8
point1.distance_from_origin() = 5
point1.distance_to(&point2) = 3.6055512
offset_of(Point2D, x) = 0
offset_of(Point2D, y) = 4
```

`size_of::<Point2D>() == 8` — two 4-byte `f32` fields, no padding, since both are already 4-byte aligned. `point1 = Point2D::new(3.0, 4.0)`: `distance_from_origin()` computes `sqrt(3² + 4²) = sqrt(25) = 5.0`, the same 3-4-5 triangle Chapter 1's sibling used. `point2 = Point2D::new(1.0, 1.0)`: `distance_to` computes `dx=2, dy=3, sqrt(4+9) = sqrt(13) ≈ 3.6055512`. `std::mem::offset_of!(Point2D, x)` and `..., y)` are not a guess — they're the standard library asking the compiler directly for each field's real byte offset, and for this particular two-`f32` struct, the compiler happened to keep declaration order (offsets `0` and `4`). Section 2.1's Common Trap below shows a struct where it genuinely doesn't.

### ASCII Diagram — one struct, two instances, layout confirmed by `offset_of!`

```
point1 (Point2D):              point2 (Point2D):
 +--------------------+         +--------------------+
 | x (offset+0): 3.0  |         | x (offset+0): 1.0  |
 | y (offset+4): 4.0  |         | y (offset+4): 1.0  |
 +--------------------+         +--------------------+
       8 bytes total                  8 bytes total

distance_to reads other.x at (other's base address + 0) and other.y at
(other's base address + 4) -- confirmed above by offset_of!, not assumed.
```

> `[COMMON TRAP]` It's tempting to assume Rust structs lay their fields out in declaration order the way C structs always do — Worked Example 2.1.1's `Point2D` even encourages that assumption, since its two same-sized fields happen to keep their declared order. They don't have to. Genuinely measured with `offset_of!` on a struct mixing field sizes:
>
> ```rust
> struct Mixed { a: u8, b: f64, c: u8 }
>
> #[repr(C)]
> struct MixedC { a: u8, b: f64, c: u8 }
> ```
>
> ```bash
> rustc -O 02_layout_reorder.rs -o 02_layout_reorder
> ./02_layout_reorder
> ```
>
> ```
> size_of::<Mixed>()  = 16
> offset_of(Mixed, a) = 8
> offset_of(Mixed, b) = 0
> offset_of(Mixed, c) = 9
> size_of::<MixedC>()  = 24
> offset_of(MixedC, a) = 0
> offset_of(MixedC, b) = 8
> offset_of(MixedC, c) = 16
> ```
>
> `Mixed` — Rust's default layout — genuinely moves `b` (the 8-byte-aligned `f64`) to offset `0`, ahead of the two `u8` fields declared before it, and ends up `16` bytes total. `MixedC`, with `#[repr(C)]` forcing C's declaration-order rule, keeps `a` at `0`, but now has to pad `b` out to the next 8-byte boundary (offset `8`) and pad the whole struct out to a multiple of 8 afterward, ending up `24` bytes — *larger*, not smaller, than the version the compiler was free to reorder. This isn't a hidden cost of Rust's flexibility; it's the opposite: the default layout is free to be smaller precisely because it isn't required to preserve an order nothing in safe Rust actually depends on. The one place this matters is `unsafe` code doing raw pointer arithmetic into a struct, or FFI layout shared with C — both need `#[repr(C)]` explicitly, exactly as `MixedC` does here.

## 2.2 Constructors: Associated Functions, Not Overloading `[FOUNDATIONAL]`

### Intuition

C++ lets several different `Point2D(...)` signatures share one name, resolved by the compiler purely from the arguments at each call site. Rust doesn't have this mechanism at all — no two functions or methods, anywhere in the same scope, may share a name, regardless of how their parameter lists differ. A "constructor" in Rust is just an ordinary associated function, by convention named `new`, and if a type needs several different ways to be built, each one gets its own distinct name (`from_polar`, `zero`, `midpoint`) rather than an overload of `new`.

### Background

| | C++ constructor overloading | Rust |
|---|---|---|
| Multiple ways to build a value | Several `Point2D(...)` overloads, resolved by argument types | Several distinctly-named associated functions (`new`, `from_x`, ...) |
| A no-argument "default" constructor | `Point2D()`, an overload like any other | The `Default` trait's `default()` method — a real trait, not special syntax |
| Does `let p2 = p1;` copy `p1`, for a plain-data struct? | Yes — the compiler generates a copy constructor automatically | **No** — it moves `p1`, unless the struct explicitly derives `Copy` |

### Worked Example 2.2.1 — a real, genuinely captured move, even for an all-`f32` struct

```rust
struct Point2D { x: f32, y: f32 }

fn main() {
    let point1 = Point2D { x: 3.0, y: 4.0 };
    let point2 = point1;
    println!("point1=({},{}) point2=({},{})", point1.x, point1.y, point2.x, point2.y);
}
```

```bash
rustc 03_no_derive_move.rs
```

Genuinely compiled (this file is not included in the Complete Runnable Code section below, since it deliberately does not compile):

```
error[E0382]: borrow of moved value: `point1`
 --> 03_no_derive_move.rs:6:57
  |
4 |     let point1 = Point2D { x: 3.0, y: 4.0 };
  |         ------ move occurs because `point1` has type `Point2D`, which does not implement the `Copy` trait
5 |     let point2 = point1;
  |                  ------ value moved here
6 |     println!("point1=({},{}) point2=({},{})", point1.x, point1.y, point2.x, point2.y);
  |                                                         ^^^^^^^^ value borrowed here after move
  |
note: if `Point2D` implemented `Clone`, you could clone the value
 --> 03_no_derive_move.rs:1:1
  |
1 | struct Point2D { x: f32, y: f32 }
  | ^^^^^^^^^^^^^^ consider implementing `Clone` for this type
...
5 |     let point2 = point1;
  |                  ------ you could clone this value

error: aborting due to 1 previous error

For more information about this error, try `rustc --explain E0382`.
```

This is genuinely surprising the first time you see it: both fields are `f32`, individually `Copy`, and C++'s equivalent code compiles and copies without a second thought. Rust's rule doesn't inspect the fields to decide — a struct is `Copy` only if it explicitly opts in with `#[derive(Copy)]`, regardless of what its fields are made of. Without that opt-in, `let point2 = point1;` is exactly Chapter 1.3's move, applied to a struct instead of a `String`.

### Worked Example 2.2.2 — the fix: `derive(Clone)`, an explicit `new`, and `Default`

```rust
#[derive(Clone)]
struct Point2D {
    x: f32,
    y: f32,
}

impl Point2D {
    fn new(x: f32, y: f32) -> Self {
        Point2D { x, y }
    }
}

impl Default for Point2D {
    fn default() -> Self {
        Point2D { x: 0.0, y: 0.0 }
    }
}
```

Compiled and run as the complete `04_constructors_value_semantics.rs` further below:

```bash
rustc -O 04_constructors_value_semantics.rs -o 04_constructors_value_semantics
./04_constructors_value_semantics
```

Genuinely compiled and run:

```
point1 = (3, 4)
point2 (clone) = (3, 4)
after point2.x = 99.0: point1 = (3, 4), point2 = (99, 4)
Point2D::default() = (0, 0)
```

`point1.clone()` — not `let point2 = point1;` — is what genuinely produces an independent second `Point2D`, because `.clone()` borrows `point1` rather than moving it, and `#[derive(Clone)]` generates a real, correct, byte-for-byte copy for a struct this simple. Mutating `point2.x` afterward leaves `point1.x` at its original `3.0`, confirming two genuinely independent instances, the same value-semantics guarantee CUDA's chapter demonstrates — just requiring one visible word (`derive` or `.clone()`) that C++ never asks for.

## 2.3 Generics and Monomorphization `[FOUNDATIONAL]`

### Intuition

A cookie cutter shaped like a star doesn't produce *one* cookie — it produces a whole batch, each one from different dough, but every one exactly star-shaped. A Rust **generic** struct is a cookie cutter for code: `struct FixedVec<T, const N: usize> { data: [T; N] }` isn't compiled code by itself — it's a pattern the compiler stamps out into genuine, independent, fully compiled code the moment you actually use it with concrete type and value arguments, once per distinct combination — the process is called **monomorphization**, and it's the direct Rust counterpart to C++ template instantiation.

### Background

| | Un-generic struct | Generic struct |
|---|---|---|
| Compiled how many times? | Once | Once *per distinct instantiation* (`FixedVec<f32,4>` and `FixedVec<i32,3>` are two separate compiled types) |
| Where are `T` and `N` resolved? | N/A | At compile time, from the type/value arguments at each use site |
| Runtime cost of genericity | N/A | Zero — by the time the program runs, every instantiation is already ordinary, concrete, fully-typed machine code |

### Worked Example 2.3.1 — two specializations, genuinely proven distinct

```rust
struct FixedVec<T, const N: usize> {
    data: [T; N],
}

impl<T: Copy + Default + std::ops::AddAssign, const N: usize> FixedVec<T, N> {
    fn sum(&self) -> T {
        let mut total = T::default();
        for i in 0..N {
            total += self.data[i];
        }
        total
    }
}
```

Compiled and run as the complete `05_generics.rs` further below:

```bash
rustc -O 05_generics.rs -o 05_generics
./05_generics
```

Genuinely compiled and run:

```
size_of::<FixedVec<f32,4>>() = 16, sum = 10
size_of::<FixedVec<i32,3>>() = 12, sum = 60
```

`FixedVec<f32,4>` is 16 bytes (four 4-byte `f32`s) and its sum is `1+2+3+4 = 10`. `FixedVec<i32,3>` is 12 bytes (three 4-byte `i32`s) and its sum is `10+20+30 = 60`. These aren't two runs of the same code with different data — recompiled without optimization (so the compiler doesn't inline the evidence away) and inspected with `nm`, demangled with `rustfilt` (a Rust-symbol demangler, installed with `cargo install rustfilt` — the direct counterpart to `nm -C`'s built-in C++ demangling):

```bash
rustc -C debuginfo=0 05_generics.rs -o 05_generics_noopt
nm 05_generics_noopt | grep sum | rustfilt
```

```
00000000000154b0 t _05_generics::FixedVec<T,_>::sum
0000000000015550 t _05_generics::FixedVec<T,_>::sum
```

Two distinct symbols, at two distinct addresses (`0x154b0` and `0x15550`), sharing one demangled name — `rustfilt` collapses the const-generic parameter `N` in its pretty-printed output, so both instantiations print identically, but the raw mangled names underneath them (visible with plain `nm`, before piping through `rustfilt`) carry different hash suffixes, and the two separate addresses are themselves unambiguous proof the compiler emitted separate machine code for each. Nothing in either compiled body branches on "am I holding floats or ints" at runtime — that question was answered once, at compile time, per instantiation, the same zero-runtime-cost guarantee Chapter 1.2 established for type inference.

## 2.4 `Drop`: RAII Without the Double-Free Hazard `[FOUNDATIONAL]`

### Intuition

A hotel room key that locks the door automatically the moment you drop it in the checkout slot doesn't rely on you remembering to lock up — the *scope* of your stay is tied directly to the resource's lifetime. This pattern, RAII, is exactly what CUDA's chapter builds with a constructor/destructor pair; Rust's `Drop` trait is the same idea under a different name. What's different is the default: a Rust struct built entirely from owned fields (a `Vec<f32>`, another struct, anything that isn't a raw pointer) gets correct, automatic, exactly-once cleanup with no way to double-free it — Chapter 1's ownership rules already forbid two live owners of the same value, and `Drop::drop` only ever runs once per value, when its one owner goes out of scope.

### Background

| | Manual allocation, no `Drop` | `Drop`-wrapped, owned field (`Vec<f32>`) | `Drop`-wrapped, raw pointer field (`unsafe`) |
|---|---|---|
| Who frees the memory? | You, explicitly, at every exit path | `Drop::drop`, automatically, at scope exit | `Drop::drop`, automatically — but only correct if nothing else holds a copy of the pointer |
| Forgetting to free | A silent leak | Not possible — the compiler forces a `Drop` impl to exist if you write one at all | Possible to write incorrectly, same as C++ |
| Can a naive `#[derive(Clone)]` cause a double-free? | N/A | **No** — `Vec<f32>`'s own `Clone` genuinely deep-copies the buffer | **Yes** — a raw pointer's `Clone` is a bitwise copy of the address, same hazard as C++ |

### Worked Example 2.4.1 — construction and destruction order, safely and automatically

```rust
struct DeviceBuffer {
    data: Vec<f32>,
    size: usize,
}

impl DeviceBuffer {
    fn new(n: usize) -> Self {
        let data = vec![0.0f32; n];
        println!("  DeviceBuffer::new({}) constructed -> {} f32s allocated", n, data.len());
        DeviceBuffer { data, size: n }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        println!("  DeviceBuffer(size={}) dropped, freeing {} f32s", self.size, self.data.len());
    }
}

fn scoped_demo() {
    println!("entering scoped_demo");
    let _a = DeviceBuffer::new(100);
    let _b = DeviceBuffer::new(200);
    println!("both buffers constructed, about to leave scope");
}
```

Compiled and run as the complete `06_drop_raii.rs` further below:

```bash
rustc -O 06_drop_raii.rs -o 06_drop_raii
./06_drop_raii
```

Genuinely compiled and run:

```
entering scoped_demo
  DeviceBuffer::new(100) constructed -> 100 f32s allocated
  DeviceBuffer::new(200) constructed -> 200 f32s allocated
both buffers constructed, about to leave scope
  DeviceBuffer(size=200) dropped, freeing 200 f32s
  DeviceBuffer(size=100) dropped, freeing 100 f32s
back in main, both drops already ran
```

`_b` (constructed second) drops first — LIFO order, exactly matching CUDA's chapter, and for the identical reason: Rust drops local variables in the reverse of their declaration order within a scope. Nothing here required `unsafe`, a manual `alloc`/`dealloc` call, or a hand-written copy constructor to get right — `Vec<f32>` already owns its buffer correctly, so `DeviceBuffer` composing it inherits that correctness for free, exactly as this chapter's opening epigraph claims.

> `[COMMON TRAP]` The safety in Worked Example 2.4.1 is not automatic in *all* Rust code — it's automatic in *safe* Rust, because `Vec`'s own internals are the ones doing the real allocation work correctly. Reach for a raw pointer directly, and the exact hazard CUDA's chapter describes reappears, because a raw pointer's `Clone` is just a bitwise copy of the address:
>
> ```rust
> use std::alloc::{alloc, dealloc, Layout};
>
> #[derive(Clone)] // derives a bitwise copy of `ptr`, not a deep copy of the allocation
> struct RawBuffer {
>     ptr: *mut f32,
>     len: usize,
> }
>
> impl RawBuffer {
>     fn new(n: usize) -> Self {
>         let layout = Layout::array::<f32>(n).unwrap();
>         let ptr = unsafe { alloc(layout) as *mut f32 };
>         RawBuffer { ptr, len: n }
>     }
> }
>
> impl Drop for RawBuffer {
>     fn drop(&mut self) {
>         let layout = Layout::array::<f32>(self.len).unwrap();
>         unsafe { dealloc(self.ptr as *mut u8, layout) };
>     }
> }
> ```
>
> ```bash
> rustc -O 07_raw_ptr_double_free_risk.rs -o 07_raw_ptr_double_free_risk
> ./07_raw_ptr_double_free_risk
> ```
>
> Genuinely compiled and run — the full file, `07_raw_ptr_double_free_risk.rs`, further below, deliberately calls `std::mem::forget` on both `a` and `b` before either can drop, specifically so this book can show the shared-pointer evidence without triggering a real double-free:
>
> ```
> a.ptr = 0x5614ceec4d60, b.ptr = 0x5614ceec4d60 (same address)
> ```
>
> That's the hazard, genuinely measured: `a.clone()` compiled cleanly (raw pointers are `Copy`, so `#[derive(Clone)]` accepts them without complaint) and produced `b` with the *identical* address as `a`. Deleting the two `std::mem::forget` calls from this file would let both `a` and `b` drop normally at the end of `main`, and both drops would call `dealloc` on that same address — a genuine double-free, undefined behavior on real hardware, exactly CUDA's chapter's `DeviceBuffer` hazard. The difference from C++ isn't that Rust makes this impossible; it's that Rust makes it require `unsafe` and a raw pointer to even set up — every safe Rust type this book builds from here on, starting with Part 1's `Tensor`, is deliberately built from owned fields like `Vec<f32>` for exactly this reason.

## 2.5 Static vs. Dynamic Dispatch: Generics vs. `dyn Trait`

### Intuition

CUDA C++ has no first-class trait keyword, so its chapter reaches for CRTP as a workaround to get a compile-time-checked interface. Rust doesn't need a workaround — `trait` is a real, first-class language feature — so the interesting question isn't "how do I fake an interface," it's "which of Rust's two genuine dispatch mechanisms do I want." A generic function bound by a trait (`fn f<S: Shape>(s: &S)`) gets monomorphized per concrete type, exactly like Section 2.3's `FixedVec` — no runtime lookup at all. A `dyn Trait` reference is Rust's vtable-based mechanism: one compiled function body, shared across every type that implements the trait, dispatched through a **fat pointer** carrying both a data address and a vtable address.

### Background

| | Generic (`<S: Shape>`), static dispatch | `dyn Shape`, dynamic dispatch |
|---|---|---|
| How the right `area()` is found | Resolved directly at compile time, once per instantiation | A vtable pointer, carried alongside the data pointer, followed at runtime |
| `size_of::<&Circle>()` vs. `size_of::<&dyn Shape>()` | `8` (an ordinary thin reference) | `16` (a fat pointer: 8-byte data address + 8-byte vtable address) |
| Compiled bodies for two shape types | Two, one per instantiation (Section 2.3's evidence) | One — the same `dyn Shape` function serves every implementor |
| Where does the vtable pointer live? | N/A — no vtable exists | At the *reference* or `Box`, never inside the struct itself |

That last row is a genuine, precise difference from CUDA's virtual dispatch, not just a renaming: C++'s vtable pointer lives *inside every instance* of a class with virtual functions, which is exactly why CUDA's `sizeof(Circle)` grew to include it. Rust's `dyn Trait` vtable pointer lives at the *pointer*, never in the struct — `size_of::<Circle>()` stays `4` bytes (just `radius`) whether or not `Circle` is ever used behind a `dyn Shape` reference anywhere in the program.

### Worked Example 2.5.1 — both mechanisms, genuinely measured and run

```rust
trait Shape {
    fn area(&self) -> f32;
}

struct Circle { radius: f32 }
impl Shape for Circle {
    fn area(&self) -> f32 { 3.14159 * self.radius * self.radius }
}

struct Square { side: f32 }
impl Shape for Square {
    fn area(&self) -> f32 { self.side * self.side }
}

fn print_area_generic<S: Shape>(s: &S) -> f32 {
    s.area()
}

fn print_area_dyn(s: &dyn Shape) -> f32 {
    s.area()
}
```

Compiled and run as the complete `08_dyn_vs_generic.rs` further below:

```bash
rustc -O 08_dyn_vs_generic.rs -o 08_dyn_vs_generic
./08_dyn_vs_generic
```

Genuinely compiled and run:

```
size_of::<&Circle>()      = 8
size_of::<&dyn Shape>()   = 16
size_of::<Box<Circle>>()  = 8
size_of::<Box<dyn Shape>>() = 16
circle area (generic, static dispatch) = 12.56636
square area (generic, static dispatch) = 9
dyn Shape area (dynamic dispatch) = 12.56636
dyn Shape area (dynamic dispatch) = 9
```

`3.14159 × 2.0² = 12.56636` and `3.0² = 9`, computed identically by both dispatch paths — the two mechanisms are only different in *how* `area()` gets found, never in what it computes. The fat-pointer doubling (`8` bytes to `16`) is genuinely measured, not asserted. And recompiled without optimization to check the symbol table:

```bash
rustc -C debuginfo=0 08_dyn_vs_generic.rs -o 08_dyn_vs_generic_noopt
nm 08_dyn_vs_generic_noopt | grep print_area | rustfilt
```

```
0000000000015d70 t _08_dyn_vs_generic::print_area_dyn
0000000000015d80 t _08_dyn_vs_generic::print_area_generic
0000000000015d90 t _08_dyn_vs_generic::print_area_generic
```

One `print_area_dyn` symbol total, genuinely shared by both `Circle` and `Square` at runtime through their respective vtables — but *two* `print_area_generic` symbols, one fully specialized per shape type, the same monomorphization evidence Section 2.3 already established. The trade-off is exact: static dispatch costs compiled code size (one body per type) in exchange for zero runtime indirection; dynamic dispatch costs one indirect call per invocation in exchange for a single shared body regardless of how many types implement `Shape`.

## Complete Runnable Code

### File: `01_basic_structs.rs`

```rust
struct Point2D {
    x: f32,
    y: f32,
}

impl Point2D {
    fn new(x: f32, y: f32) -> Self {
        Point2D { x, y }
    }

    fn distance_from_origin(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn distance_to(&self, other: &Point2D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

fn main() {
    let point1 = Point2D::new(3.0, 4.0);
    let point2 = Point2D::new(1.0, 1.0);
    println!("size_of::<Point2D>() = {}", std::mem::size_of::<Point2D>());
    println!("point1.distance_from_origin() = {}", point1.distance_from_origin());
    println!("point1.distance_to(&point2) = {}", point1.distance_to(&point2));
    println!("offset_of(Point2D, x) = {}", std::mem::offset_of!(Point2D, x));
    println!("offset_of(Point2D, y) = {}", std::mem::offset_of!(Point2D, y));
}
```

```bash
rustc -O 01_basic_structs.rs -o 01_basic_structs
./01_basic_structs
```

### File: `02_layout_reorder.rs`

```rust
struct Mixed {
    a: u8,
    b: f64,
    c: u8,
}

#[repr(C)]
struct MixedC {
    a: u8,
    b: f64,
    c: u8,
}

fn main() {
    println!("size_of::<Mixed>()  = {}", std::mem::size_of::<Mixed>());
    println!("offset_of(Mixed, a) = {}", std::mem::offset_of!(Mixed, a));
    println!("offset_of(Mixed, b) = {}", std::mem::offset_of!(Mixed, b));
    println!("offset_of(Mixed, c) = {}", std::mem::offset_of!(Mixed, c));

    println!("size_of::<MixedC>()  = {}", std::mem::size_of::<MixedC>());
    println!("offset_of(MixedC, a) = {}", std::mem::offset_of!(MixedC, a));
    println!("offset_of(MixedC, b) = {}", std::mem::offset_of!(MixedC, b));
    println!("offset_of(MixedC, c) = {}", std::mem::offset_of!(MixedC, c));
}
```

```bash
rustc -O 02_layout_reorder.rs -o 02_layout_reorder
./02_layout_reorder
```

### File: `04_constructors_value_semantics.rs`

```rust
#[derive(Clone)]
struct Point2D {
    x: f32,
    y: f32,
}

impl Point2D {
    fn new(x: f32, y: f32) -> Self {
        Point2D { x, y }
    }
}

impl Default for Point2D {
    fn default() -> Self {
        Point2D { x: 0.0, y: 0.0 }
    }
}

fn main() {
    let point1 = Point2D::new(3.0, 4.0);
    let mut point2 = point1.clone();
    println!("point1 = ({}, {})", point1.x, point1.y);
    println!("point2 (clone) = ({}, {})", point2.x, point2.y);
    point2.x = 99.0;
    println!(
        "after point2.x = 99.0: point1 = ({}, {}), point2 = ({}, {})",
        point1.x, point1.y, point2.x, point2.y
    );
    let origin = Point2D::default();
    println!("Point2D::default() = ({}, {})", origin.x, origin.y);
}
```

```bash
rustc -O 04_constructors_value_semantics.rs -o 04_constructors_value_semantics
./04_constructors_value_semantics
```

### File: `05_generics.rs`

```rust
struct FixedVec<T, const N: usize> {
    data: [T; N],
}

impl<T: Copy + Default + std::ops::AddAssign, const N: usize> FixedVec<T, N> {
    fn sum(&self) -> T {
        let mut total = T::default();
        for i in 0..N {
            total += self.data[i];
        }
        total
    }
}

fn main() {
    let vf: FixedVec<f32, 4> = FixedVec { data: [1.0, 2.0, 3.0, 4.0] };
    let vi: FixedVec<i32, 3> = FixedVec { data: [10, 20, 30] };

    println!("size_of::<FixedVec<f32,4>>() = {}, sum = {}", std::mem::size_of::<FixedVec<f32,4>>(), vf.sum());
    println!("size_of::<FixedVec<i32,3>>() = {}, sum = {}", std::mem::size_of::<FixedVec<i32,3>>(), vi.sum());
}
```

```bash
rustc -O 05_generics.rs -o 05_generics
./05_generics
```

### File: `06_drop_raii.rs`

```rust
struct DeviceBuffer {
    data: Vec<f32>,
    size: usize,
}

impl DeviceBuffer {
    fn new(n: usize) -> Self {
        let data = vec![0.0f32; n];
        println!("  DeviceBuffer::new({}) constructed -> {} f32s allocated", n, data.len());
        DeviceBuffer { data, size: n }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        println!("  DeviceBuffer(size={}) dropped, freeing {} f32s", self.size, self.data.len());
    }
}

fn scoped_demo() {
    println!("entering scoped_demo");
    let _a = DeviceBuffer::new(100);
    let _b = DeviceBuffer::new(200);
    println!("both buffers constructed, about to leave scope");
}

fn main() {
    scoped_demo();
    println!("back in main, both drops already ran");
}
```

```bash
rustc -O 06_drop_raii.rs -o 06_drop_raii
./06_drop_raii
```

### File: `07_raw_ptr_double_free_risk.rs`

```rust
use std::alloc::{alloc, dealloc, Layout};

#[derive(Clone)] // <-- derives a bitwise copy of `ptr`, not a deep copy of the allocation
struct RawBuffer {
    ptr: *mut f32,
    len: usize,
}

impl RawBuffer {
    fn new(n: usize) -> Self {
        let layout = Layout::array::<f32>(n).unwrap();
        let ptr = unsafe { alloc(layout) as *mut f32 };
        RawBuffer { ptr, len: n }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        let layout = Layout::array::<f32>(self.len).unwrap();
        unsafe { dealloc(self.ptr as *mut u8, layout) };
    }
}

fn main() {
    let a = RawBuffer::new(64);
    let b = a.clone(); // compiles cleanly: b.ptr == a.ptr, bit-for-bit
    println!("a.ptr = {:?}, b.ptr = {:?} (same address)", a.ptr, b.ptr);
    // Deliberately not letting both `a` and `b` drop here -- see chapter text.
    std::mem::forget(a);
    std::mem::forget(b);
}
```

```bash
rustc -O 07_raw_ptr_double_free_risk.rs -o 07_raw_ptr_double_free_risk
./07_raw_ptr_double_free_risk
```

### File: `08_dyn_vs_generic.rs`

```rust
trait Shape {
    fn area(&self) -> f32;
}

struct Circle {
    radius: f32,
}

impl Shape for Circle {
    fn area(&self) -> f32 {
        3.14159 * self.radius * self.radius
    }
}

struct Square {
    side: f32,
}

impl Shape for Square {
    fn area(&self) -> f32 {
        self.side * self.side
    }
}

fn print_area_generic<S: Shape>(s: &S) -> f32 {
    s.area()
}

fn print_area_dyn(s: &dyn Shape) -> f32 {
    s.area()
}

fn main() {
    let c = Circle { radius: 2.0 };
    let sq = Square { side: 3.0 };

    println!("size_of::<&Circle>()      = {}", std::mem::size_of::<&Circle>());
    println!("size_of::<&dyn Shape>()   = {}", std::mem::size_of::<&dyn Shape>());
    println!("size_of::<Box<Circle>>()  = {}", std::mem::size_of::<Box<Circle>>());
    println!("size_of::<Box<dyn Shape>>() = {}", std::mem::size_of::<Box<dyn Shape>>());

    println!("circle area (generic, static dispatch) = {}", print_area_generic(&c));
    println!("square area (generic, static dispatch) = {}", print_area_generic(&sq));

    let shapes: Vec<Box<dyn Shape>> = vec![Box::new(Circle { radius: 2.0 }), Box::new(Square { side: 3.0 })];
    for s in &shapes {
        println!("dyn Shape area (dynamic dispatch) = {}", print_area_dyn(s.as_ref()));
    }
}
```

```bash
rustc -O 08_dyn_vs_generic.rs -o 08_dyn_vs_generic
./08_dyn_vs_generic
```

## Chapter Summary

A Rust `struct` is a fixed, compile-time-known layout of named fields, extending Chapter 1's primitive-type reasoning to types you name yourself — but unlike C++, declaration order is not a layout guarantee unless you opt into it explicitly with `#[repr(C)]`, genuinely demonstrated in this chapter as a *smaller* default-layout struct that reorders its fields to reduce padding. Rust has no constructor overloading at all — "constructors" are ordinary, distinctly-named associated functions, with the `Default` trait filling the no-argument case — and, more surprisingly, a struct built entirely from `Copy` fields still doesn't get an automatically-generated copy the way a C++ struct does: assignment moves it, exactly like Chapter 1.3's `String`, unless the struct explicitly derives `Clone` or `Copy`. Generics compile into one genuinely separate, fully-specialized piece of machine code per concrete instantiation — monomorphization — proven directly in this chapter as distinct symbols in a compiled binary, the same evidence CUDA's chapter used for C++ templates. `Drop` gives RAII with no double-free hazard by default, because Rust's ownership rules (Chapter 1.3) already forbid two live owners of the same value — that safety only breaks down where this chapter deliberately broke it, with a raw pointer and `unsafe` code standing in for what C++ does by default. And where CUDA C++ has to fake a compile-time interface with CRTP, Rust's `trait` is first-class: a generic function bound by a trait monomorphizes per type with zero runtime cost, while `dyn Trait` gives real vtable-based dynamic dispatch through a fat pointer — genuinely measured here as double the size of a plain reference, with the vtable pointer living at the reference itself, never embedded inside the struct the way a C++ virtual class's vtable pointer is.

## Self-Check Questions

1. `struct Mixed { a: u8, b: f64, c: u8 }` reports `size_of` of `16` bytes with Rust's default layout, but `24` bytes under `#[repr(C)]`. Explain what the compiler is doing differently in each case, and why the C-compatible version ends up larger rather than smaller.
2. `Point2D { x: f32, y: f32 }`, with no derives at all, fails to compile when you write `let point2 = point1;` followed by a use of `point1`. Both fields are individually `Copy`. Why doesn't that make `Point2D` itself `Copy`?
3. `FixedVec<f32, 4>` and `FixedVec<i32, 3>` are both instantiated in the same program. What compiled-binary evidence from this chapter proves the compiler emitted two genuinely separate pieces of machine code, rather than one generic function branching on type at runtime?
4. `DeviceBuffer` (owning a `Vec<f32>`) and `RawBuffer` (owning a `*mut f32`) both implement `Drop`. Explain precisely why deriving `Clone` is safe for one and a genuine double-free hazard for the other.
5. `size_of::<&Circle>()` is `8`; `size_of::<&dyn Shape>()` is `16`. Where does the extra 8 bytes live, and why does `size_of::<Circle>()` itself — not a reference to it — stay `4` bytes regardless of whether `Circle` is ever used behind a `dyn Shape` reference?

## Where We Go Next

Every struct this chapter built held at most a handful of scalar fields or a single `Vec` — the layout and ownership questions stayed simple because there was only ever a little data to account for. Chapter 3 tackles what happens once a struct needs to describe *multi-dimensional* data laid out in memory — rows, columns, strides between them — the memory-layout groundwork this book's actual `Tensor` type, starting in Part 1, is built directly on top of.

## Worked Solutions

**1.** With no `#[repr]` attribute, Rust's compiler is free to reorder `Mixed`'s fields to minimize padding, and genuinely does: it places the 8-byte-aligned `f64` field `b` first (at offset 0, needing no padding before it), then packs both 1-byte `u8` fields (`a` and `c`) into the remaining space, for a total of 16 bytes. `#[repr(C)]` forces C's rule instead — fields stay in declaration order, so `a` sits at offset 0, `b` must be padded out to the next 8-byte boundary (offset 8), and the whole struct is padded to a multiple of 8 afterward to keep repeated instances (as in an array) properly aligned, landing at 24 bytes. The C-compatible layout is larger specifically because it's *not* allowed to reorder fields to close the gaps the default layout is free to close.

**2.** Rust decides whether a struct is `Copy` by an explicit opt-in — `#[derive(Copy)]` — never by inspecting whether its fields happen to all be `Copy` themselves. This is a deliberate design choice, not an oversight: a struct's author may add a non-`Copy` field (a `String`, a `Vec`, eventually a `Drop` impl) in a later version, and if plain field-composition silently granted `Copy`, that change could silently alter every call site's move-vs-copy behavior without any compile error to flag it. Requiring an explicit `derive` means a struct's `Copy`-ness is a decision its author makes and states, not an accident of its current field types.

**3.** Recompiling without optimization and inspecting the symbol table with `nm | rustfilt` shows two distinct symbols for `FixedVec<T,_>::sum`, at two different addresses, with two different underlying mangled hashes — one per concrete instantiation (`FixedVec<f32,4>` and `FixedVec<i32,3>`). A single generic function branching on type at runtime would show as exactly one symbol regardless of how many types used it, the way `print_area_dyn` in Section 2.5 genuinely does show as one shared symbol; two symbols is direct evidence of monomorphization, not a design assumption.

**4.** `DeviceBuffer` owns a `Vec<f32>`, and `Vec`'s own `Clone` implementation performs a real, independent heap allocation and copies every element — so `DeviceBuffer`'s derived (or hand-written) `Clone` inherits a genuinely correct deep copy for free, and each of the two resulting `Vec`s frees its own independent buffer when dropped. `RawBuffer` owns a `*mut f32`, and a raw pointer's `Clone` (inherited from its `Copy` implementation) is a bitwise copy of the address itself, not the memory it points to — cloning a `RawBuffer` produces two structs whose `ptr` fields hold the identical address, and both of their `Drop` implementations will eventually call `dealloc` on it, a double-free, unless something (like this chapter's deliberate `std::mem::forget` calls) prevents both from actually dropping.

**5.** The extra 8 bytes live at the *reference* (or `Box`), not inside `Circle` itself — a `&dyn Shape` is a fat pointer carrying both an 8-byte data address and an 8-byte vtable address, where a `&Circle` is an ordinary 8-byte thin pointer to just the data. `size_of::<Circle>()` stays `4` bytes (just its `radius` field) regardless of `dyn Shape` usage because Rust's vtable pointer is never stored inside the object — unlike C++, where a class with virtual functions embeds its vtable pointer in every instance, permanently growing `sizeof` for that type everywhere it's used, whether or not any particular instance is ever accessed through a base-class pointer.
