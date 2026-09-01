# Chapter 1: Variables, Ownership, and Types

> "A type is a promise you make to the compiler about what a piece of memory means. Every type in this book's Rust code carries a second, unavoidable promise as well: who owns it, and therefore who is responsible for freeing it — a promise the compiler checks with the same rigor it checks whether a byte means a signed integer or an IEEE-754 float, and one no C++ compiler enforces at all."

**What you will understand by the end of this chapter:**

- What actually happens in memory when you write `let x: i32 = 42` — the stack slot, the byte width, and why the compiler needs to know the type before that line runs, exactly as in any statically typed language
- Why Rust's type inference (`let a = 10`) is not the same thing as Python's dynamic typing, even though neither one has a visible type annotation
- What ownership actually is: a compile-time-tracked answer to "who is responsible for freeing this value," enforced by the borrow checker with the same rigor as any other type rule — Rust's genuine second axis, the way `__host__`/`__device__` is CUDA C++'s
- Why moving a value out from under a live reference, or using a value after it has been moved, is caught at compile time in Rust — where the equivalent pattern in C++ either doesn't compile for unrelated reasons or, worse, compiles cleanly into undefined behavior
- Why a plain Rust array like `[f32; 4]` does *not* automatically get the same 16-byte alignment CUDA's `float4` gets for free — and how `#[repr(align(16))]` gets it explicitly, genuinely measured with `size_of`/`align_of`

**What you need to know first:**

- General programming experience — this chapter assumes familiarity with basic programming concepts, not prior Rust or GPU exposure
- No prior Rust exposure is assumed — this is the first technical chapter of the book
- If you've read the CUDA, Mojo, or Triton editions of this book, this chapter opens differently on purpose. CUDA C++'s (and, in its own way, Mojo's) earliest distinguishing idea is a second compiler-enforced axis about *where* code runs — host versus device. Rust's earliest distinguishing idea is a second compiler-enforced axis about *who owns* a value. That axis has no equivalent in any of the three sibling books: nothing in C++, Mojo, or the Python host Triton relies on tracks ownership at compile time the way Rust does. No GPU content appears in this chapter at all — that begins in Part 0, Chapter 4, once the ownership foundation this chapter builds is in place. It isn't a detour: the `cudarc` buffer handles Chapter 4 introduces are themselves ordinary owned Rust values, freed automatically through the exact mechanism this chapter covers.

## 1.1 What a Type Actually Is `[FOUNDATIONAL]`

### Intuition

Think about shipping a package through a courier service. Before the courier will take it, you declare what's inside and how much it weighs — not because the courier is nosy, but because that declaration determines everything downstream: which truck it goes on, how it's stacked, whether it needs a fragile sticker. A type is that same declaration, attached not to a package but to a region of computer memory. `i32` means "this is 4 bytes, interpret those bytes as a signed whole number." `f32` means "this is 4 bytes, interpret them according to the IEEE-754 rules for a 32-bit floating-point number." Once the compiler knows the label, every later instruction touching that memory already knows exactly how many bytes to read and how to interpret them — no inspection required at the moment of use.

### Background

A **statically typed** language fixes the type of every variable before the program runs — either because you wrote it explicitly or because the compiler deduced it unambiguously. Rust, like C, C++, and CUDA C++, is statically typed. A **dynamically typed** language, like Python, attaches the type to the *value* at runtime instead of the variable: the name has no type of its own, it just currently points at some object, and that object carries its own type tag checked on every operation.

| Language | Type known at | Where the value lives | Type check per use |
|---|---|---|---|
| Rust / C / C++ | Compile time (mandatory) | Stack (usually, for these types) | None |
| Python | Runtime (attached to the object) | Heap (always) | Every operation |

### Worked Example 1.1.1 — One declaration, traced to the bit

```rust
let x: i32 = 42;
let y: f32 = 3.14159;
let z: f64 = 2.71828;
```

The complete, compiled, and genuinely run version — wrapped in a `main` function with `println!` calls that print not just the values but their exact bit patterns — is `01_stack_types.rs`, shown in full further below.

```bash
rustc -O 01_stack_types.rs -o 01_stack_types
./01_stack_types
```

Genuinely compiled and run:

```
x=42 y=3.14159 z=2.71828
size_of(i32)=4 size_of(f32)=4 size_of(f64)=8
x bits (two's complement, 32-bit): 00000000000000000000000000101010
y bits (IEEE-754 binary32):        01000000010010010000111111010000
z bits (IEEE-754 binary64):        0100000000000101101111110000100110010101101010101111011110010000
```

The compiler reserves 4 bytes for `x` and writes the two's-complement bit pattern for `42`. It reserves another 4 bytes for `y` (a Rust `f32` is IEEE-754 single precision, exactly like C++'s `float`) and writes the bit pattern for `3.14159`. It reserves 8 bytes for `z` (`f64` is IEEE-754 double precision) and writes the bit pattern for `2.71828`. `y.to_bits()` and `z.to_bits()` aren't a trick — they're ordinary standard-library methods that reinterpret the float's existing bytes as an unsigned integer of the same width, so the binary strings above are the actual bytes the first `println!` line already used to produce `3.14159` and `2.71828`, not a separately-computed approximation.

### ASCII Diagram — one stack frame, three variables

```
Stack frame for a function containing:
    let x: i32 = 42;
    let y: f32 = 3.14159;
    let z: f64 = 2.71828;

High address
 +--------------------------+
 | ... caller's frame ...   |
 +--------------------------+
 | z  (8 bytes, f64)        |  <- 0100000000000101101111110000100110010101101010101111011110010000
 +--------------------------+
 | y  (4 bytes, f32)        |  <- 01000000010010010000111111010000  (padded to 32 bits shown)
 +--------------------------+
 | x  (4 bytes, i32)        |  <- 00000000000000000000000000101010  (two's-complement 42)
 +--------------------------+
Low address                    <- stack pointer sits here
```

> `[COMMON TRAP]` C++ silently converts between `float` and `double` wherever a narrowing conversion is contextually allowed — `float x = 3.14159265358979;` compiles cleanly and quietly rounds, discarding precision with no warning by default. Rust closes this off entirely: there is no implicit numeric conversion between *any* two numeric types, narrowing or widening. Genuinely captured:
>
> ```rust
> let precise: f64 = 3.14159265358979;
> let narrowed: f32 = precise;  // no implicit f64 -> f32 conversion in Rust
> ```
>
> ```bash
> rustc 06_no_silent_narrowing.rs
> ```
>
> ```
> error[E0308]: mismatched types
>  --> 06_no_silent_narrowing.rs:3:25
>   |
> 3 |     let narrowed: f32 = precise;
>   |                   ---   ^^^^^^^ expected `f32`, found `f64`
>   |                   |
>   |                   expected due to this
>
> error: aborting due to 1 previous error
>
> For more information about this error, try `rustc --explain E0308`.
> ```
>
> The fix has to be spelled out explicitly with `as`, which makes the precision loss visible at the call site instead of silent:
>
> ```rust
> let narrowed: f32 = precise as f32;
> ```
>
> Genuinely compiled and run — `06_explicit_cast.rs` further below:
>
> ```
> precise=3.14159265358979 narrowed=3.1415927
> ```

## 1.2 Type Inference vs. Dynamic Typing `[FOUNDATIONAL]`

### Intuition

Imagine two tailors. The first takes your measurements once, at your very first fitting, and cuts a suit that is permanently your size — it never adjusts itself later. The second doesn't measure you up front at all; every time you put the suit on, they re-measure and refit it to whatever you are *that day*. Rust's `let a = 10;` is the first tailor: the compiler looks at `10` once, at compile time, deduces `i32` (Rust's default integer type), and `a` is an `i32` for the rest of its scope, exactly as if you had written `let a: i32 = 10;` yourself. Python's `a = 10` is the second tailor: `a` has no fixed size at all, and nothing stops the next line from repointing it at a string.

### Background

**Type inference** is a compile-time process — the compiler looks at the initializer (and, in Rust, sometimes at how the variable is used later in the function too) and locks a concrete type in for the variable's whole scope. No inference happens while the program runs; by execution time, `let`-inferred variables are indistinguishable from explicitly-typed ones. **Dynamic typing** has no such compile-time fixing step, because the type belongs to the object a name currently references, not to the name itself.

| | Rust `let a = 10;` | Python `a = 10` |
|---|---|---|
| When is the type decided? | Once, at compile time | Never fixed — checked fresh on every use |
| Can `a` later hold a `String`? | No — compile error | Yes — `a = "ten"` is legal |
| Cost of inference at runtime | Zero (already resolved) | N/A — no inference, only per-use checking |

### Worked Example 1.2.1 — the same-looking line, two different outcomes

```rust
let a = 10;       // compiler infers i32 here, at compile time -- permanent
let b = 5.5;       // compiler infers f64 here -- permanent
let c = true;      // compiler infers bool here -- permanent
```

Compiled and run as `02_type_inference.rs` (the complete file appears further below):

```bash
rustc -O 02_type_inference.rs -o 02_type_inference
./02_type_inference
```

Genuinely compiled and run:

```
a=10 (size 4), b=5.5 (size 8), c=true (size 1)
```

`size_of_val(&a)` is `4` because the compiler resolved `a`'s type to `i32` at compile time, before this program ever ran — the same size a hand-written `let a: i32 = 10;` would report. `b` resolves to `f64`, Rust's default floating-point type (not `f32` — Rust's defaults for untyped integer and float literals are `i32` and `f64` respectively, chosen for range and precision, not speed). `c` is `bool`, and unlike C's `printf("%d", ...)`, Rust's `{}` formatter knows `bool` is its own type and prints the word `true`, not `1` — `size_of_val(&c)` is still `1` byte, the same single-byte representation C++'s `bool` uses.

## 1.3 Ownership: The Type System's Third Axis `[FOUNDATIONAL]`

### Intuition

In C++, assigning one `std::string` to another (`std::string s2 = s1;`) copies the string's contents by default — both `s1` and `s2` end up as independent, fully-formed strings, and both destructors will eventually run. In Python, `s2 = s1` doesn't copy anything at all; it makes `s2` a second name for the *same* underlying string object, and Python's reference counter keeps track of how many names point at it so it knows when it's finally safe to free. Rust picks neither default. `let s2 = s1;` for a `String` **moves** the value: `s1`'s bytes on the heap now belong to `s2`, `s1` is left in a compiler-tracked "no longer valid" state, and the compiler will refuse to compile any later line that tries to use `s1` again. This is not a runtime check and not a convention — it's enforced the same way a type mismatch is enforced, at compile time, before the program ever runs.

### Background

Every value in Rust has exactly one owner at a time. When the owner goes out of scope, Rust automatically calls that value's `Drop` implementation (freeing heap memory, closing a file handle, releasing a `cudarc` buffer — whatever cleanup the type defines), with no garbage collector and no manual `free`/`delete` call required anywhere in the program. Assignment (`let s2 = s1;`), passing by value into a function, and returning a value all transfer ownership by default for types that manage a resource (like `String` or `Vec<T>`) — this is called a **move**. Simple, fixed-size types that are cheap to duplicate (`i32`, `f32`, `bool`, and other types marked `Copy`) are the one exception: assigning them duplicates the bits instead of moving, because there's no heap resource to worry about owning twice.

| | C++ (`std::string s2 = s1;`) | Python (`s2 = s1`) | Rust (`let s2 = s1;`, `String`) |
|---|---|---|---|
| What happens to the data | Deep-copied by default | Not copied — `s2` is a second reference | Moved — ownership transfers to `s2` |
| Is `s1` still valid afterward? | Yes, independent copy | Yes, same object as `s2` | **No** — compile error to use it |
| Who frees the memory, and when | Both `s1` and `s2`'s destructors, independently | Reference count reaches zero (runtime-tracked) | `s2`'s single `Drop`, when `s2` goes out of scope |
| When is this checked | Not checked — it's just what assignment does | Not checked — reference counting is automatic | **Compile time** — the borrow checker |

### Worked Example 1.3.1 — a real, genuinely captured move violation

```rust
let s1 = String::from("hello");
let s2 = s1;
println!("s1 = {}, s2 = {}", s1, s2);
```

```bash
rustc 03_move_error.rs
```

Genuinely compiled (the complete file, `03_move_error.rs`, is not included in this chapter's Complete Runnable Code section below, since it deliberately does not compile):

```
error[E0382]: borrow of moved value: `s1`
 --> 03_move_error.rs:4:34
  |
2 |     let s1 = String::from("hello");
  |         -- move occurs because `s1` has type `String`, which does not implement the `Copy` trait
3 |     let s2 = s1;
  |              -- value moved here
4 |     println!("s1 = {}, s2 = {}", s1, s2);
  |                                  ^^ value borrowed here after move
  |
help: consider cloning the value if the performance cost is acceptable
  |
3 |     let s2 = s1.clone();
  |                ++++++++

error: aborting due to 1 previous error

For more information about this error, try `rustc --explain E0382`.
```

This is not a warning and not a lint — `rustc` refuses to produce a binary at all. Note what the compiler is actually tracking: not the *value* `"hello"`, but the *binding* `s1` — after line 3, `s1` is a name the compiler considers dead, regardless of what bytes are still sitting in memory underneath it.

### Worked Example 1.3.2 — the fix: an explicit, independent copy

```rust
let s1 = String::from("hello");
let s2 = s1.clone();
println!("s1 = {}, s2 = {}", s1, s2);
```

```bash
rustc -O 03b_move_fixed.rs -o 03b_move_fixed
./03b_move_fixed
```

Genuinely compiled and run — `03b_move_fixed.rs` further below:

```
s1 = hello, s2 = hello
```

`.clone()` is the explicit, visible opt-in to what C++ does *implicitly* on every `std::string` assignment: a real heap allocation and a real byte-for-byte copy, so that `s1` and `s2` become two fully independent owners. The compiler's own suggested fix in Worked Example 1.3.1's error message is exactly this line — Rust doesn't hide the cost of a deep copy behind ordinary-looking assignment syntax the way C++ does; if you want the copy, you write the word that means "copy."

## 1.4 Ownership at a Function Call Boundary: Move vs. Borrow `[FOUNDATIONAL]`

### Intuition

Handing a physical package to a courier and handing them a photograph of it are different transactions with different consequences. Hand over the package itself, and it's the courier's now — you can't go retrieve it from your own shelf afterward, because it isn't there anymore. Hand over a photograph, and you both still have full use of the original; the courier can look at it, but the package never left your possession. Passing a value into a Rust function by its type (`fn f(s: String)`) is the first transaction — it's a move, exactly like Worked Example 1.3.1's assignment, just across a function call instead of a `let`. Passing a reference (`fn f(s: &String)`) is the second — the function can read (or, with `&mut String`, modify) the value, but ownership never leaves the caller.

### Background

| | Owned parameter (`s: String`) | Shared reference (`s: &String`) | Exclusive reference (`s: &mut String`) |
|---|---|---|---|
| Ownership transfers to the function? | Yes — a move | No | No |
| Caller can still use the original after the call? | No | Yes | Only after the borrow ends |
| Can the function modify the value? | Yes, it owns it now | No (compile error to try) | Yes |
| How many of these can exist on the same value at once | N/A (ownership is exclusive by definition) | Any number, simultaneously | Exactly one, and no shared references at the same time |

The last row is the rule the Rust community calls "aliasing XOR mutability," and it's enforced by the same compiler pass — the **borrow checker** — that caught Worked Example 1.3.1's use-after-move. It isn't a separate feature bolted on top of ownership; a `&mut` reference is really a temporary, exclusive loan of the value's single ownership slot, checked with the same rigor.

### Worked Example 1.4.1 — a real, genuinely captured call-boundary violation

```rust
fn takes_ownership(s: String) {
    println!("takes_ownership got: {}", s);
}

fn main() {
    let s = String::from("hello");
    takes_ownership(s);
    println!("s = {}", s);
}
```

```bash
rustc 04b_use_after_move.rs
```

Genuinely compiled (this file is not included in the Complete Runnable Code section below, since it deliberately does not compile):

```
error[E0382]: borrow of moved value: `s`
 --> 04b_use_after_move.rs:8:24
  |
6 |     let s = String::from("hello");
  |         - move occurs because `s` has type `String`, which does not implement the `Copy` trait
7 |     takes_ownership(s);
  |                     - value moved here
8 |     println!("s = {}", s);
  |                        ^ value borrowed here after move
  |
note: consider changing this parameter type in function `takes_ownership` to borrow instead if owning the value isn't necessary
 --> 04b_use_after_move.rs:1:23
  |
1 | fn takes_ownership(s: String) {
  |    ---------------    ^^^^^^ this parameter takes ownership of the value
  |    |
  |    in this function
help: consider cloning the value if the performance cost is acceptable
  |
7 |     takes_ownership(s.clone());
  |                      ++++++++

error: aborting due to 1 previous error

For more information about this error, try `rustc --explain E0382`.
```

This is caught during the same compilation pass that resolves ordinary type-checking, exactly as Worked Example 1.3.1's error was — `takes_ownership(s)` genuinely consumes `s`, and the compiler's own suggested fix (a note pointing at the function signature, distinct from the `.clone()` `help`) shows it understands *why*: change the parameter to borrow instead, if ownership was never actually needed.

### Worked Example 1.4.2 — the fix: borrow first, move last

```rust
fn takes_ownership(s: String) {
    println!("takes_ownership got: {}", s);
}

fn borrows(s: &String) {
    println!("borrows got: {}", s);
}

fn main() {
    let s = String::from("hello");
    borrows(&s);
    println!("still valid after borrow: {}", s);
    takes_ownership(s);
    println!("s moved into takes_ownership; still usable here");
}
```

```bash
rustc -O 04_ownership_functions.rs -o 04_ownership_functions
./04_ownership_functions
```

Genuinely compiled and run — `04_ownership_functions.rs` further below:

```
borrows got: hello
still valid after borrow: hello
takes_ownership got: hello
s moved into takes_ownership; still usable here
```

`borrows(&s)` takes a reference, so `s` is still valid on the next line — genuinely printed as `"still valid after borrow: hello"`. Only the final `takes_ownership(s)` actually moves `s`, and by that point in the function nothing needs `s` afterward, so the compiler has nothing to object to. The rule from Worked Example 1.4.1 hasn't changed; this version simply never violates it.

> `[COMMON TRAP]` In C++, growing a `std::vector` that has a live pointer or iterator into it — say, by calling `.push_back()` after taking `&v[0]` — compiles without complaint and is **undefined behavior** if the vector happens to reallocate its backing storage, since the old pointer now dangles into freed memory. Rust's borrow checker rejects this pattern categorically, at compile time, every time, regardless of whether that particular `push` would have actually triggered a reallocation:
>
> ```rust
> let mut v = vec![1, 2, 3];
> let first = &v[0];
> v.push(4);
> println!("first = {}", first);
> ```
>
> ```bash
> rustc 07_borrow_conflict.rs
> ```
>
> ```
> error[E0502]: cannot borrow `v` as mutable because it is also borrowed as immutable
>  --> 07_borrow_conflict.rs:4:5
>   |
> 3 |     let first = &v[0];
>   |                  - immutable borrow occurs here
> 4 |     v.push(4);
>   |     ^^^^^^^^^ mutable borrow occurs here
> 5 |     println!("first = {}", first);
>   |                            ----- immutable borrow later used here
>
> error: aborting due to 1 previous error
>
> For more information about this error, try `rustc --explain E0502`.
> ```
>
> `v.push(4)` needs `&mut v` (it might reallocate, which touches every element), and `first` is a live `&v[0]` at that point — the exact "one mutable reference, XOR any number of shared references" rule from this section's Background table, caught here for a `Vec` instead of a `String`. This entire bug class — a reference silently outliving the reallocation that invalidated it — is a real, well-known category of memory-safety bug in C++; Rust doesn't detect it at runtime, it makes the program that could produce it fail to compile in the first place.

## 1.5 Fixed-Size Arrays and Compiler-Enforced Alignment

### Intuition

A Rust `[f32; 4]` is, like CUDA's `float4`, one type the compiler knows the exact size of — 16 bytes, four `f32`s laid out contiguously, with a genuine `size_of`/`align_of` the compiler will report on request. Where it *doesn't* match `float4` is alignment: CUDA's built-in vector types carry a compiler-enforced 16-byte alignment specifically so hardware can fetch all four floats in one aligned memory transaction. A plain Rust array gets no such upgrade automatically — its alignment defaults to its element type's alignment, 4 bytes for `f32`, whether the array holds one element or a hundred. Getting `float4`'s actual alignment guarantee in Rust means asking for it explicitly, with `#[repr(align(16))]` on a wrapper type.

### Background

| Type | `size_of` | `align_of` |
|---|---|---|
| `f32` | 4 | 4 |
| `[f32; 2]` | 8 | 4 |
| `[f32; 4]` | 16 | **4** — not automatically upgraded |
| `#[repr(C, align(16))] struct F32x4 { x: f32, y: f32, z: f32, w: f32 }` | 16 | **16** — genuinely 16, once asked for explicitly |

### Worked Example 1.5.1 — genuinely measured, not assumed

```rust
#[repr(C, align(16))]
struct F32x4 { x: f32, y: f32, z: f32, w: f32 }

fn main() {
    println!("size_of::<f32>()  = {}, align_of::<f32>()  = {}", std::mem::size_of::<f32>(), std::mem::align_of::<f32>());
    println!("size_of::<[f32;4]>() = {}, align_of::<[f32;4]>() = {}", std::mem::size_of::<[f32;4]>(), std::mem::align_of::<[f32;4]>());
    println!("size_of::<F32x4>() = {}, align_of::<F32x4>() = {}", std::mem::size_of::<F32x4>(), std::mem::align_of::<F32x4>());
    let v4 = F32x4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
    println!("F32x4 fields: x={} y={} z={} w={}", v4.x, v4.y, v4.z, v4.w);
}
```

```bash
rustc -O 05_layout_types.rs -o 05_layout_types
./05_layout_types
```

Genuinely compiled and run — `05_layout_types.rs` further below:

```
size_of::<f32>()  = 4, align_of::<f32>()  = 4
size_of::<[f32;2]>() = 8, align_of::<[f32;2]>() = 4
size_of::<[f32;4]>() = 16, align_of::<[f32;4]>() = 4
size_of::<F32x2>() = 8, align_of::<F32x2>() = 4
size_of::<F32x4>() = 16, align_of::<F32x4>() = 16
plain array: [1.0, 2.0, 3.0, 4.0]
F32x4 fields: x=1 y=2 z=3 w=4
```

`align_of::<[f32;4]>()` is genuinely `4`, not `16` — the same result you'd get from a bare `f32`, because a plain array's alignment is always its element type's alignment, with no size-based upgrade. `F32x4`, with `#[repr(align(16))]` spelled out explicitly, genuinely reports `16`. This isn't a cosmetic difference: Part 0 Chapter 5's SIMD intrinsics and, much later, Part 1's `Tensor` buffers both need a real 16-byte (or wider) alignment guarantee to issue aligned vector loads — and in Rust, unlike CUDA, that guarantee is something you ask for by name, not something a `[f32; 4]` gives you for free.

## Complete Runnable Code

### File: `01_stack_types.rs`

```rust
fn main() {
    let x: i32 = 42;
    let y: f32 = 3.14159;
    let z: f64 = 2.71828;
    println!("x={} y={} z={}", x, y, z);
    println!(
        "size_of(i32)={} size_of(f32)={} size_of(f64)={}",
        std::mem::size_of::<i32>(),
        std::mem::size_of::<f32>(),
        std::mem::size_of::<f64>()
    );
    println!("x bits (two's complement, 32-bit): {:032b}", x as u32);
    println!("y bits (IEEE-754 binary32):        {:032b}", y.to_bits());
    println!("z bits (IEEE-754 binary64):        {:064b}", z.to_bits());
}
```

```bash
rustc -O 01_stack_types.rs -o 01_stack_types
./01_stack_types
```

### File: `02_type_inference.rs`

```rust
fn main() {
    let a = 10;
    let b = 5.5;
    let c = true;
    println!(
        "a={} (size {}), b={} (size {}), c={} (size {})",
        a, std::mem::size_of_val(&a),
        b, std::mem::size_of_val(&b),
        c, std::mem::size_of_val(&c)
    );
}
```

```bash
rustc -O 02_type_inference.rs -o 02_type_inference
./02_type_inference
```

### File: `03b_move_fixed.rs`

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();
    println!("s1 = {}, s2 = {}", s1, s2);
}
```

```bash
rustc -O 03b_move_fixed.rs -o 03b_move_fixed
./03b_move_fixed
```

### File: `04_ownership_functions.rs`

```rust
fn takes_ownership(s: String) {
    println!("takes_ownership got: {}", s);
}

fn borrows(s: &String) {
    println!("borrows got: {}", s);
}

fn main() {
    let s = String::from("hello");
    borrows(&s);
    println!("still valid after borrow: {}", s);
    takes_ownership(s);
    println!("s moved into takes_ownership; still usable here");
}
```

```bash
rustc -O 04_ownership_functions.rs -o 04_ownership_functions
./04_ownership_functions
```

### File: `05_layout_types.rs`

```rust
#[repr(C)]
struct F32x2 { x: f32, y: f32 }

#[repr(C, align(16))]
struct F32x4 { x: f32, y: f32, z: f32, w: f32 }

fn main() {
    println!("size_of::<f32>()  = {}, align_of::<f32>()  = {}", std::mem::size_of::<f32>(), std::mem::align_of::<f32>());
    println!("size_of::<[f32;2]>() = {}, align_of::<[f32;2]>() = {}", std::mem::size_of::<[f32;2]>(), std::mem::align_of::<[f32;2]>());
    println!("size_of::<[f32;4]>() = {}, align_of::<[f32;4]>() = {}", std::mem::size_of::<[f32;4]>(), std::mem::align_of::<[f32;4]>());

    println!("size_of::<F32x2>() = {}, align_of::<F32x2>() = {}", std::mem::size_of::<F32x2>(), std::mem::align_of::<F32x2>());
    println!("size_of::<F32x4>() = {}, align_of::<F32x4>() = {}", std::mem::size_of::<F32x4>(), std::mem::align_of::<F32x4>());

    let v: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    println!("plain array: {:?}", v);

    let v4 = F32x4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
    println!("F32x4 fields: x={} y={} z={} w={}", v4.x, v4.y, v4.z, v4.w);
}
```

```bash
rustc -O 05_layout_types.rs -o 05_layout_types
./05_layout_types
```

### File: `06_explicit_cast.rs`

```rust
fn main() {
    let precise: f64 = 3.14159265358979;
    let narrowed: f32 = precise as f32;
    println!("precise={} narrowed={}", precise, narrowed);
}
```

```bash
rustc -O 06_explicit_cast.rs -o 06_explicit_cast
./06_explicit_cast
```

### File: `07_borrow_conflict.rs`

This file is intentionally not included here in runnable form — it exists only to demonstrate a compile-time rejection, and is shown in full inside this chapter's `[COMMON TRAP]` callout in Section 1.4.

## Chapter Summary

A type in Rust is exactly the compile-time promise it is in any statically typed language — `let x: i32 = 42;` reserves 4 bytes and fixes their meaning before the program runs, whether you write the type explicitly or let Rust's inference deduce it from the initializer, in sharp contrast to Python, where a name has no type of its own and simply points at whichever object it was last assigned. What Rust adds on top of ordinary static typing, with no equivalent in C++ or Python, is a second compile-time-enforced axis: **ownership**. Every value has exactly one owner; assigning, passing by value, or returning a non-`Copy` value moves ownership rather than copying it, and the compiler — not a runtime check, not a convention — refuses to compile any later use of a binding whose value has moved away, the same rigor it applies to an ordinary type mismatch. Passing `&T` or `&mut T` instead of `T` borrows a value without transferring ownership, governed by a single rule enforced everywhere in the language at once: any number of shared (`&`) borrows, or exactly one exclusive (`&mut`) borrow, never both at the same time — the rule that turns C++'s classic "pointer outlived a reallocation" bug class into a compile error instead of undefined behavior. Fixed-size arrays like `[f32; 4]` are, like CUDA's `float4`, single types with a compiler-known size — but unlike `float4`, their alignment is *not* automatically upgraded for SIMD-friendly access; that guarantee has to be requested explicitly with `#[repr(align(16))]`, genuinely confirmed here with `size_of`/`align_of` rather than assumed.

## Self-Check Questions

1. `let a = 10;` in Rust and `a = 10` in Python both omit an explicit type annotation. What question distinguishes the two cases, and why does that question have a different answer for each?
2. `let s2 = s1;` for a `String` behaves differently in Rust than the equivalent-looking assignment does in C++ or Python. Describe what happens to `s1`'s data in all three languages, and specifically what happens to the *name* `s1` in each.
3. Why does `takes_ownership(s)` followed by `println!("{}", s)` fail to compile, while `borrows(&s)` followed by the same `println!("{}", s)` compiles cleanly — even though both function calls look similar from the call site?
4. In C++, taking a pointer to a `std::vector`'s first element and then calling `.push_back()` on the same vector compiles without any error and is undefined behavior if the vector reallocates. What does the equivalent Rust pattern do instead, and at what point does it fail?
5. `align_of::<[f32; 4]>()` reports `4`, not `16`, even though `size_of::<[f32; 4]>()` is `16`. Explain, in one or two sentences, why Rust doesn't upgrade a plain array's alignment automatically the way CUDA's `float4` does, and what has to be written to get that guarantee explicitly.

## Where We Go Next

Every type this chapter introduced — `i32`, `f32`, `[f32; 4]` — was a primitive or a fixed-size aggregate of primitives, whose ownership and layout the compiler already knows how to handle by default. Chapter 2 introduces designing your own Rust `struct`s and the traits that give them behavior, and traces exactly how ownership composes when a struct owns heap-allocated fields of its own — the foundation this book's `Tensor` type, starting in Part 1, is built directly on top of.

## Worked Solutions

**1.** The distinguishing question is: *is the type fixed before the program runs, and can the same name later refer to a value of a different type?* For `let a = 10;`, the compiler examines the literal `10` during compilation, resolves `a`'s type to `i32`, and rejects any later attempt in the same scope to assign a non-`i32` value to `a` — fixed permanently, at compile time. For `a = 10` in Python, there is no compile-time type-fixing step; `a` is a name bound to whatever object it was last assigned, and `a = "ten"` on the next line is completely legal, because the type lives on the object, not the name.

**2.** In C++, `std::string s2 = s1;` performs a full deep copy — `s1` and `s2` become two independent strings, and both remain valid, each freed by its own destructor when it goes out of scope. In Python, `s2 = s1` copies nothing; `s2` becomes a second name for the exact same string object, and both names remain valid — the object is only freed once its reference count drops to zero. In Rust, `let s2 = s1;` moves the value: the underlying heap data now belongs solely to `s2`, and the name `s1` is marked by the compiler as no longer valid — not because the data was destroyed, but because the compiler will not permit two live names claiming ownership of the same resource at once, and any attempt to use `s1` afterward is a compile error, not a runtime condition.

**3.** `takes_ownership(s: String)` takes its parameter by value, which moves `s` into the function — after that call, the compiler considers the caller's `s` binding dead, exactly as Section 1.3's `let s2 = s1;` killed `s1`, so the later `println!` referencing `s` is a use of a moved value and fails to compile. `borrows(s: &String)` takes a reference instead: ownership of `s` never leaves the caller, the function only receives temporary read access, and the caller's `s` binding remains fully valid for the `println!` that follows.

**4.** Rust's borrow checker rejects the pattern at compile time, categorically, regardless of whether that specific `push` call would have actually triggered a reallocation. `&v[0]` creates a shared (`&`) borrow of `v`; `v.push(4)` requires an exclusive (`&mut`) borrow, because a push might reallocate the entire backing buffer, which would invalidate every existing reference into it. Rust's "any number of shared borrows, or exactly one exclusive borrow, never both" rule makes holding `first` across the `push` call a compile error — `error[E0502]: cannot borrow v as mutable because it is also borrowed as immutable` — rather than allowing the program to compile and potentially dereference a dangling pointer at runtime, which is exactly what C++'s equivalent pattern risks.

**5.** A plain array's alignment in Rust is defined as its element type's alignment, with no automatic upgrade based on the array's total size — `[f32; 4]` is 16 bytes of data, but the compiler only guarantees it starts on a 4-byte boundary, the same guarantee a lone `f32` gets. CUDA's `float4` is a distinct built-in type whose 16-byte alignment is part of its own definition, baked in by NVIDIA's headers, not derived from `float`'s alignment. Getting the equivalent guarantee in Rust means defining a wrapper type and asking for it explicitly with `#[repr(align(16))]`, as `F32x4` does in Worked Example 1.5.1 — after which `align_of` genuinely reports `16`, confirmed rather than assumed.
