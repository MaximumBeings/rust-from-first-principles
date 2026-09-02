# Chapter 11: Memory Management System — Shared Ownership, Bump Allocation, and Pooling

> "Chapter 6 gave every tensor exactly one owner, and Rust's own compiler enforces that rule for free — a moved-from binding simply cannot be used again, full stop. That's simpler and stricter than Chapter 6's C++ counterpart, and it's still too rigid for what's coming: a computational graph that needs a tensor's buffer visible from two places at once. This chapter is where 'exactly one owner, enforced by the type system' becomes 'as many owners as the graph needs, opted into on purpose' — and where two of the three new tricks that make that possible turn out to have a real gap of their own, genuinely demonstrated rather than merely claimed. One of those gaps isn't even the one the CUDA edition found."

**What you will understand by the end of this chapter:**

- `RefCountedBuffer<T>`: genuine shared ownership, where *any* of several live clones can validly be the one whose `Drop` impl actually frees the memory — and a direct, compiled demonstration of why Rust won't hand you this behavior by accident the way C++'s implicit copy constructor does
- The bump-pointer `Arena`: why its 256-byte alignment arithmetic is correct as written, and a genuinely compiled, genuinely run comparison of Rust's two "cheap runtime check" macros — `assert!` and `debug_assert!` — only one of which turns out to behave anything like C's `NDEBUG`-gated `assert()`, discovered here by testing both rather than assuming Rust's `assert!` is a drop-in translation
- `DeviceMemoryPool`'s size-classed free list, traced through three acquire/release cycles to a `0.667` hit rate, plus a `Drop` impl this struct never got — genuinely measured, via a tracked-allocation counter, to leak exactly one outstanding buffer for as long as the pool lives
- A genuine double-release bug: releasing the same raw pointer twice compiles without a single warning and silently hands the *same* address back out to two unrelated callers — demonstrated with real, printed, identical pointer values, and traced to the one place in this chapter where Rust's ownership rules simply don't apply

**What you need to know first:**

- Chapter 6 (`Tensor`'s ownership: construction allocates, `Drop` frees, exactly once, enforced by the borrow checker)
- Chapter 7.2 (`TensorView<'a>`, this chapter's direct sibling — a view that never owns, versus `RefCountedBuffer<T>`, a buffer that any of several clones might own)
- Chapter 7.3 and Chapter 10.2 (the 256-byte alignment a real device allocation guarantees, and the alignment gap already found in `DeviceAwareAllocator`'s host fallback — `Arena`'s alignment choice in this chapter is that same guarantee, applied deliberately rather than lost by accident)
- Chapter 9.4's habit of testing a debug/release divergence rather than assuming one (the `usize` underflow that panics in a debug build and silently wraps in a release build) — Section 11.2 runs that exact playbook again, on a completely different mechanism

## 11.1 Reference Counting: When More Than One Owner Needs the Same Buffer `[FOUNDATIONAL]`

### Intuition

Two roommates share a one-bedroom apartment. Neither individually decides when the lease ends — the landlord only reclaims the apartment once *both* have moved out, and it doesn't matter which one happens to hand back the last key. Chapter 6's `Tensor` doesn't work this way: Rust's ownership rules give it exactly one owner for the whole of its lifetime, and the compiler enforces that directly — there is no way to accidentally end up with two. `RefCountedBuffer<T>` is the roommate arrangement instead: a small shared counter, allocated alongside the data, that every clone increments on arrival and decrements on departure, so that *whichever* clone happens to go out of scope last is correctly the one that frees the memory, without any clone needing to know in advance which one that will be.

### Background

```rust
struct RefCountedBuffer<T> {
    data: *mut T,
    refcount: *mut usize,
    count: usize,
    data_layout: std::alloc::Layout,
}
```

| | Chapter 6 `Tensor` | Chapter 7.2 `TensorView<'a>` | `RefCountedBuffer<T>` (this chapter) |
|---|---|---|---|
| How many owners at once | exactly one, enforced by the borrow checker | zero — a view never owns | any number, tracked live |
| Who frees the memory | the one owning `Tensor`, via `Drop` | nobody — the viewed `Tensor` still owns it | whichever clone's `Drop` sees the count hit zero |
| Cost per share | not shareable — moving is the only option, and the moved-from binding becomes unusable | one pointer/shape/stride copy, no allocation | one shared-memory increment, via an explicit `.clone()` call |
| Cost per destruction | one `dealloc` call | none — nothing to free | one shared-memory decrement plus a zero check |

`RefCountedBuffer<T>`'s own data buffer is allocated here with `std::alloc::alloc`, not through a device allocator — for a real device-resident `Tensor` buffer this mechanism would wrap a device `alloc`/`free` pair exactly as Chapter 6 does, but Chapter 10 already established that a genuine device allocation fails immediately in this no-GPU sandbox. Using the plain global allocator here lets the actual refcounting arithmetic — the part this section is about — genuinely construct, clone, and drop end to end, rather than being simulated.

There's a structural difference from the CUDA edition's `RefCountedBuffer<T>` worth stating up front, not just as a footnote: C++'s copy constructor runs *implicitly*, the moment you write `RefCountedBuffer<float> v(t);` — nothing marks that line as doing anything unusual. Rust has no implicit copy for a type that owns a `Drop` impl. `let v = t;` **moves** `t`'s fields into `v`; it does not share them, and `t` becomes unusable afterward — the compiler rejects any later use of `t` at compile time, not at runtime. To get the CUDA chapter's actual behavior — two live values sharing one buffer, both eventually running their destructor — this chapter has to opt in explicitly, with a hand-written `Clone` impl that does exactly what the C++ copy constructor did: copy the pointers, not the data, and bump the shared count. Section 11.1's second worked example demonstrates the alternative directly, rather than asserting it.

### Worked Example 11.1.1 — The count, traced through three lifetimes

`RefCountedBuffer::<f32>::new(4)` allocates the buffer and sets `*refcount = 1`. Entering an inner scope and calling `t.clone()` to produce `v` gives `v` the *same* `data` and `refcount` pointers as `t` (not copies of the data itself) and increments the shared count to `2`. When `v` goes out of scope first, its `Drop` impl decrements the count to `1`; since `1 != 0`, the buffer survives — `t`'s own copy of that same `refcount` pointer reads the identical `1`. Only when `t` itself later goes out of scope does its `Drop` impl decrement the count to `0` and actually free both `data` and `refcount`. Neither `t` nor `v` ever checked whether the other was still alive — the shared `usize` they both point at carried that information for both of them.

Compiled and run exactly as described:

```bash
rustc --edition 2024 -O 01_refcounted_buffer.rs -o 01_refcounted_buffer
./01_refcounted_buffer
```

Genuinely compiled and run:

```
=== Section 11.1: RefCountedBuffer, traced through three lifetimes ===
  RefCountedBuffer::new(count=4) constructed -> refcount=1
t.get_refcount() = 1
  clone() ran (shares buffer) -> refcount=2
v.get_refcount() = 2, t.get_refcount() = 2
  drop() ran -> refcount=1
after v goes out of scope, t.get_refcount() = 1
  drop() ran -> refcount=0
  refcount hit 0 -> freeing data and refcount
after t goes out of scope, buffer has been freed
```

The trace lands exactly on `1 -> 2 -> 1 -> 0`, with the "freeing" message printed from inside the `drop()` call that actually observes the zero — genuinely the *second* of the two live clones' `Drop` impls to run, which in this scope happens to be `t`'s, precisely because `v`'s inner scope ends first.

### Worked Example 11.1.2 — What happens if you skip `.clone()` and just write `let v = t;`

A direct, line-for-line port of the C++ chapter's `RefCountedBuffer<float> v(t);` — the version of this file with the `Clone` impl deleted and `let v = t.clone();` changed to `let v = t;` — was genuinely compiled to see what the compiler does with it, rather than assuming the answer:

```bash
rustc --edition 2024 02_move_vs_clone_demo.rs -o 02_move_vs_clone_demo
```

```
error[E0382]: borrow of moved value: `t`
  --> 02_move_vs_clone_demo.rs:50:50
   |
44 |     let t: RefCountedBuffer<f32> = RefCountedBuffer::new(4);
   |         - move occurs because `t` has type `RefCountedBuffer<f32>`, which does not implement the `Copy` trait
45 |     println!("t.get_refcount() before move = {}", t.get_refcount());
46 |     let v = t;
   |             - value moved here
...
50 |     println!("t.get_refcount() after move = {}", t.get_refcount());
   |                                                  ^ value borrowed here after move
   |
note: if `RefCountedBuffer<f32>` implemented `Clone`, you could clone the value
  --> 02_move_vs_clone_demo.rs:3:1
   |
 3 | struct RefCountedBuffer<T> {
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^ consider implementing `Clone` for this type
...
46 |     let v = t;
   |             - you could clone this value

error: aborting due to 1 previous error
```

This is a genuine `E0382`, not a hypothetical one — the compiler refuses to build a program that reads `t` after it has been moved into `v`. Stepping back, this is worth sitting with for a moment: the entire bug class Section 11.4 is about to spend real effort demonstrating for `DeviceMemoryPool` — a raw resource handed to two places that both think they own it — is exactly what Rust's default (moving, not copying) ownership model prevents *by construction*, for any type that doesn't explicitly opt out of it. `RefCountedBuffer<T>` only regains the "two owners" behavior at all because its `Clone` impl deliberately hands back a second handle to the same shared state. The compiler isn't weaker here than C++ — it's stricter by default, and this struct spends real, explicit code (the `Clone` impl, the manual refcount) buying back a capability Rust otherwise declines to give away for free.

> `[COMMON TRAP]` Chapter 7.2's `TensorView<'a>` is the direct consumer this struct exists for — a view built from a `RefCountedBuffer`-backed tensor can report the shared refcount for exactly as long as any view is alive, and the underlying allocation only frees once every view and the owning tensor have each run their own `Drop` impl. But the `Clone` impl above copies `other.count` directly, meaning every clone of a `RefCountedBuffer<T>` reports the *same* element count as whatever it was cloned from. This struct can express "two things looking at the exact same whole buffer," but has no field at all for "a view of just the first ten elements of a hundred-element buffer, sharing that one allocation." That finer-grained slicing — an independent length and offset layered on top of a shared buffer — is exactly what Chapter 7.2's `TensorView<'a>` (shape, strides, and a borrowed pointer, no ownership of its own) adds on top; `RefCountedBuffer<T>` on its own only ever hands out whole-buffer aliases.

## 11.2 Arena-Based Allocation: Trading Individual Frees for One Bulk Reset `[FOUNDATIONAL]`

### Intuition

A coat check that issues a numbered tag for every coat, and requires that exact tag back before releasing it, does real bookkeeping on every single coat, all night. A coat check that instead says "we close at midnight, and every coat gets returned in one pass, whatever order they show up in" does none of that per-coat bookkeeping at all — it only needs to know where the coats end and the empty space begins. `Arena` is the second kind of coat check applied to memory: instead of tracking each allocation's lifetime individually (as `RefCountedBuffer<T>` does), it hands out increasing offsets into one big pre-allocated slab with nothing more than addition, and reclaims *everything* at once with a single `reset()` — appropriate exactly when every allocation in a batch is known to die together, the way a whole training step's intermediate activations do.

### Background

```rust
struct Arena {
    base: *mut u8,
    capacity: usize,
    offset: usize,
    layout: std::alloc::Layout,
}
```

`alloc::<T>(count)` computes `bytes_needed = count * size_of::<T>()`, rounds `offset` up to the next 256-byte boundary with `(offset + 255) & !255usize`, checks the result still fits inside `capacity`, and advances `offset` past the new allocation. `reset()` is one line: `offset = 0`. Nothing here does a per-allocation free; everything is reclaimed at once when the arena resets, or when its own `Drop` impl runs.

The 256-byte figure is not an arbitrary round number: it is the exact alignment a real device allocation guarantees (Chapter 7.3), the same guarantee Chapter 10.2 found `DeviceAwareAllocator`'s host fallback silently drops. An `Arena`-backed buffer handed to a kernel expecting that alignment — say, one of Chapter 7.4's vectorized `wide::f32x8` loads — genuinely gets it, by construction, rather than by accident.

### Worked Example 11.2.1 — The alignment arithmetic, bit by bit

`(offset + 255) & !255usize` is the standard round-up-to-a-power-of-two-boundary idiom: adding `255` (one less than the 256-byte alignment) pushes any offset that isn't already a multiple of 256 up into the next block, and `& !255usize` (clearing the low 8 bits) rounds back down to that block's start — landing exactly on the next multiple of 256 at or above the original offset. Traced against this chapter's own numbers: first request, `offset = 0`, wants `100` `f32`s (`400` bytes) — `(0 + 255) & !255 = 255 & !255`. `255` has only its low 8 bits set, so clearing those 8 bits leaves `0`. Already aligned; the allocation runs from byte `0` to byte `400`, and `offset` becomes `400`. Second request, `50` more `f32`s (`200` bytes): `(400 + 255) & !255 = 655 & !255`. `655 = 2×256 + 143`, so clearing the low 8 bits drops the `143` and leaves `512` — the allocation runs from `512` to `712`, having quietly skipped `512 - 400 = 112` bytes of padding to land on the boundary.

Compiled and run exactly as described:

```bash
rustc --edition 2024 -O 03_arena_allocator.rs -o 03_arena_allocator
./03_arena_allocator
```

Genuinely compiled and run:

```
=== Section 11.2: Arena, alignment arithmetic traced by hand ===
after alloc::<f32>(100): offset=400 (expected 400)
after alloc::<f32>(50):  offset=712 (expected 712)
second allocation's aligned start = 512, padding skipped = 112
a[0] written through arena pointer = 1.5
b[0] written through arena pointer = 2.5
after reset(): offset=0
```

Both returned pointers are also genuinely writable — `a[0]` and `b[0]` are written and read back through the raw pointers `alloc::<T>` returns, not merely computed as offsets and left untouched, confirming the arena's bump allocation hands back real, usable memory rather than just correct-looking arithmetic.

### Worked Example 11.2.2 — A real over-capacity request, and a genuinely different divergence than the CUDA edition's

That `112`-byte gap is the price of alignment, and it's bounded: rounding up to a 256-byte boundary can never waste more than `255` bytes on any single allocation, regardless of how many allocations came before it. Compare that fixed, small, per-allocation ceiling to `RefCountedBuffer<T>`'s cost model from Section 11.1 — a shared-memory increment and decrement on every clone and every drop — and `Arena::alloc` touches none of that: one addition, one bitwise mask, one bounds check, and pointer arithmetic. That's the entire reason a training step with thousands of small per-layer allocations reaches for this structure instead of `RefCountedBuffer<T>` or a general-purpose allocator.

The CUDA edition's bounds check is `assert(aligned_offset + bytes_needed <= capacity && "Arena exhausted")`, and its whole point in that chapter is that `NDEBUG` (the flag a release build configuration typically sets) strips it to nothing. The most direct Rust translation of that line is `assert!(...)` — and testing that translation, rather than assuming it carries the same behavior, is exactly what this chapter did:

```bash
# "debug" build: plain rustc, debug-assertions on by default
rustc --edition 2024 04a_arena_overrun_assert.rs -o 04a_debug
./04a_debug
echo "exit code: $?"

# "release" build: rustc -O, debug-assertions off by default
rustc --edition 2024 -O 04a_arena_overrun_assert.rs -o 04a_release
./04a_release
echo "exit code: $?"
```

Genuinely compiled and run, both ways:

```
--- debug build (plain rustc) ---
thread 'main' panicked at 04a_arena_overrun_assert.rs:23:9:
Arena exhausted
exit code: 101

--- release build (rustc -O) ---
thread 'main' panicked at 04a_arena_overrun_assert.rs:23:9:
Arena exhausted
exit code: 101
```

Both builds panic, identically, at the identical line. Rust's `assert!` macro is **not** gated by any build-profile flag the way C's `assert()` is by `NDEBUG` — it is compiled into the binary and checked at runtime unconditionally, in a plain build and an `-O` build alike. A literal, line-for-line translation of the CUDA edition's `assert` into Rust's `assert!` does not reproduce the divergence that chapter demonstrates; it produces a *stronger* guarantee than the C++ original, one that survives into whatever build configuration actually ships.

The macro that does behave like C's `NDEBUG`-gated `assert()` is `debug_assert!` — and Chapter 9.4 already established the exact mechanism that governs it: `-O` implies `debug-assertions=off` by default, the same codegen flag that silently turned off the overflow checking behind that chapter's `usize` underflow finding. Swapping only the macro, keeping everything else about `Arena::alloc` identical, and testing both builds again:

```bash
rustc --edition 2024 04b_arena_overrun_debug_assert.rs -o 04b_debug
./04b_debug
echo "exit code: $?"

rustc --edition 2024 -O 04b_arena_overrun_debug_assert.rs -o 04b_release
./04b_release
echo "exit code: $?"
```

Genuinely compiled and run, both ways:

```
--- debug build (plain rustc, debug-assertions on) ---
thread 'main' panicked at 04b_arena_overrun_debug_assert.rs:25:9:
Arena exhausted
exit code: 101

--- release build (rustc -O, debug-assertions off) ---
=== requesting far more than an 800-byte arena's capacity (debug_assert!) ===
alloc did not abort. pointer offset from base = 0 bytes (arena capacity = 800)
writing through the returned pointer now (this touches memory past the arena)...
wrote p[0] = 3.14 without any detected error
exit code: 0
```

This is the CUDA edition's exact divergence, reproduced faithfully: the debug build aborts with a clear diagnostic naming the failed condition (exit code `101`, Rust's panic exit code, playing the same role as C's `SIGABRT`/`134`); the `-O` build — the configuration a real training run would actually compile with, since a bounds check on a hot allocation path is exactly the overhead a release build is built to shed — proceeds silently, hands back a pointer whose bookkeeping already exceeds the arena's declared capacity, and lets a write through it succeed with `wrote p[0] = 3.14` and no detected error at all. The chapter's own `Arena::alloc`, shown earlier in Section 11.2's Background, uses `assert!` rather than `debug_assert!` specifically *because* `assert!`'s always-on behavior is the more defensible default for a teaching codebase — this worked example exists to make the trade-off, and the fact that it *is* a choice rather than an automatic consequence of the C++-to-Rust port, explicit and verified rather than assumed.

> `[COMMON TRAP]` "It compiles like C's `assert`, so it behaves like C's `assert`" is the trap here, and it runs in the opposite direction from most of this book's C++-to-Rust naming collisions: Rust's `assert!` is not the weaker, strippable macro its name suggests to anyone coming from C or C++ — `debug_assert!` is. Reaching for `assert!` out of habit, expecting a release build to shed its cost, silently gets *more* safety than expected (the check survives); reaching for `debug_assert!` out of habit, expecting the same always-on behavior C's `NDEBUG`-agnostic code sometimes assumes, silently gets *less* (the check, and the expression inside it, vanish in `-O`, exactly as Worked Example 11.2.2 just measured). Neither mistake announces itself — the only way to know which one a given line of code makes is to compile it both ways and check, the way this section just did.

## 11.3 Device Memory Pooling: Reuse Instead of Re-Asking the Allocator `[FOUNDATIONAL]`

### Intuition

A construction site that rents a specific size of scaffolding for every job, returns it when the job's done, and rents an identical size again the very next week, is paying a rental company's overhead twice for equipment it could have simply kept on-site between jobs. `DeviceMemoryPool` is the "keep it on-site" version for device buffers: instead of returning memory to the allocator the moment a step finishes (a real device free/allocate round trip is one to three orders of magnitude slower than a host allocation, when a device actually exists to round-trip to), it holds released buffers in a free list, bucketed by exact element count, ready to hand the *same* buffer straight back out the next time something asks for that exact size — which, across steps that repeat the same layer shapes every iteration, is most of the time.

### Background

```rust
struct DeviceMemoryPool {
    free_lists: HashMap<usize, Vec<*mut f32>>,
    bytes_allocated: i64,
    bytes_reused: i64,
}
```

`acquire(count)` checks whether `free_lists[&count]` has anything in it; if so, it pops and returns an existing buffer (`bytes_reused += count * 4`) instead of asking the allocator for new memory. Otherwise it allocates fresh (`bytes_allocated += count * 4`). `release(ptr, count)` never frees anything — it appends `ptr` to the free list for that size, keeping it available for the next `acquire` of the same count. In this sandbox, "allocates fresh" genuinely routes through a tracked stand-in for a real device allocation — a small wrapper that increments a global outstanding-allocation counter on every call and decrements it on every genuine free — precisely so that this section's coming leak isn't only asserted, but actually counted. That counter is an `AtomicIsize`, not a `static mut`: edition 2024 denies taking a reference to a `static mut` outright, and an atomic is exactly as simple here, single-threaded program or not.

### Worked Example 11.3.1 — Three steps, one buffer, climbing toward a hit rate of `0.667`

Step 1 requests a `256`-element buffer: the free list for size `256` is empty, so the pool allocates fresh (`bytes_allocated = 256 × 4 = 1024`) and hands it out; at the end of the step it's released back, not freed, leaving one entry in that size's free list. Step 2 requests another `256`-element buffer: the free list isn't empty this time, so the pool pops that *same* buffer and reuses it (`bytes_reused = 1024`) instead of touching the allocator again. Step 3: identical story, `bytes_reused = 2048`. After three steps, `hit_rate() = bytes_reused / (bytes_allocated + bytes_reused) = 2048 / 3072 = 0.667` — and every further step that requests the same `256`-element size reuses the same buffer yet again, pushing the ratio toward `1.0` as training continues, since real training loops request the same handful of activation shapes on every iteration.

Compiled and run exactly as described:

```bash
rustc --edition 2024 -O 05_device_memory_pool.rs -o 05_device_memory_pool
./05_device_memory_pool
```

Genuinely compiled and run:

```
=== Section 11.3: DeviceMemoryPool, three steps toward a hit rate ===
step 1: acquire(256) -> fresh allocation. bytes_allocated=1024 bytes_reused=0
step 2: acquire(256) -> reused? true (same pointer). bytes_allocated=1024 bytes_reused=1024
step 3: acquire(256) -> reused? true (same pointer). bytes_allocated=1024 bytes_reused=2048
hit_rate() = 2048 / 3072 = 0.667
G_OUTSTANDING_ALLOCATIONS while pool is still alive = 1
G_OUTSTANDING_ALLOCATIONS after pool went out of scope = 1 (expected: still 1, not 0)
```

Every pointer comparison across the three steps genuinely reports `true` — `acquire` really does hand back the identical address each time — and the hit rate lands exactly on `0.667`, matching the hand trace above.

> `[COMMON TRAP]` `DeviceMemoryPool` has no `Drop` impl — it leaks every buffer it ever allocates, for the pool's entire lifetime, and this run just measured it rather than merely claiming it. Look for a `Drop` impl on `DeviceMemoryPool` above and there isn't one — the compiler synthesizes ordinary, field-by-field drop glue for a struct that implements no `Drop` of its own, meaning when a `DeviceMemoryPool` goes out of scope, its `free_lists: HashMap<usize, Vec<*mut f32>>` and its two `i64` counters are each torn down using their own types' `Drop` behavior. `HashMap` and `Vec`'s own drop glue correctly frees the *bookkeeping* structures — the hash table, the array of pointer values sitting inside each free list — but `*mut f32` is a raw pointer with no `Drop` impl at all, exactly like a raw `float*` in C++ (exactly why every other allocating struct in this book, `Arena` and `RefCountedBuffer<T>` included, defines its own explicit `Drop` impl that calls `dealloc`). Dropping a `Vec<*mut f32>` frees the vector's own backing array — the slots that held the addresses — never the memory each address actually points to. The single buffer this pool ever created and never handed back out again before the pool itself went out of scope is gone, in the sense that nothing can reach it anymore, while `G_OUTSTANDING_ALLOCATIONS` above shows the memory it occupies was never actually freed: it reads `1` before the pool's scope ends and *still* reads `1` afterward — a textbook leak, measured rather than assumed, and one that directly contradicts the claim Section 11.4 is about to make one paragraph later in this very chapter.

## 11.4 Ownership Patterns and the Claim This Chapter's Own Code Doesn't Quite Earn `[FOUNDATIONAL]`

### Intuition

Every allocating struct so far in this book has followed one rule: acquire the resource in the constructor, release it in `Drop`, and let Rust's ordinary value-lifetime rules make sure that pairing actually happens — a rule Rust's own compiler enforces more strictly than C++'s ever did, as Worked Example 11.1.2 already demonstrated directly. This section states that rule as the chapter's closing claim — and is the right place to check whether every struct in the chapter actually lives up to it, the way this book has checked every "compiled once, cached, reused"-style claim before it.

### Background

```rust
fn training_step(pool: &mut DeviceMemoryPool, count: usize) {
    let activations = pool.acquire(count);
    // ... forward + backward pass using `activations` ...
    pool.release(activations, count);
    // activations is not read again after this line -- but nothing in
    // release()'s signature actually stops that from compiling.
}
```

The natural claim to make about this pattern is: "every allocator in this chapter follows the same rule Chapter 6's `Tensor` established — acquisition happens in the constructor, release happens in `Drop`, and there is no code path that can leak or alias a resource, because Rust's ownership rules won't let one compile." Section 11.3 already found the first half of that false for `DeviceMemoryPool`. This section checks the second half directly, against `release()`'s actual signature: `fn release(&mut self, ptr: *mut f32, count: usize)` takes `ptr` as an ordinary parameter of a raw pointer type. Unlike `RefCountedBuffer<T>` from Section 11.1 — a type whose `Drop` impl makes Rust track it strictly, refusing even a plain `let v = t;` without an explicit `.clone()` — a raw pointer type is `Copy` unconditionally, by design, no matter what it points at. Rust's ownership tracking exists entirely at the type level; a `*mut f32` is, to the borrow checker, indistinguishable from a `usize` that happens to look like an address. Nothing at the type-system level prevents `activations` from being read again after `release()`, prevents calling `release()` a second time on the same pointer, or prevents `acquire()` handing that same address back out to two unrelated callers.

### Worked Example 11.4.1 — Releasing the same pointer twice, genuinely followed through

```rust
let activations = pool.acquire(128);
pool.release(activations, 128);
pool.release(activations, 128);   // bug: same pointer, released twice -- compiles clean

let a = pool.acquire(128);
let b = pool.acquire(128);
// is a == b ?
```

Compiled and run exactly as described:

```bash
rustc --edition 2024 -O 06_raii_and_pointer_safety.rs -o 06_raii_and_pointer_safety
./06_raii_and_pointer_safety
```

Genuinely compiled and run (the specific hex addresses will differ between runs — ASLR randomizes where the allocator places a fresh heap allocation each time the program starts; two separate runs of this exact binary produced `0x557747996d60` and `0x55e949a95d60` respectively, confirming the values genuinely vary while what they reveal about each other does not):

```
=== Section 11.4: what release()'s raw-pointer signature does not prevent ===
acquired activations = 0x557747996d60
released the SAME pointer twice -- compiled and ran without complaint
acquire(128) #1 -> 0x557747996d60
acquire(128) #2 -> 0x557747996d60
are they the same address? true -- ALIASED
writing through 'a' now silently corrupts whatever 'b' is used for,
and vice versa -- two live tensors sharing one buffer, neither aware of it.
after a[0]=1.0 then b[0]=2.0: a[0]=2 b[0]=2 (both read back the same memory)
```

The double `release()` call inserts the identical address into the size-`128` free list twice — `Vec::push` has no way to know it's already there, because a `Vec<*mut f32>` is just a list of addresses, not a set of owned resources. The next two `acquire(128)` calls each pop one of those two (identical) entries and hand it out, and the printed addresses confirm it directly: `a` and `b` are the same pointer, genuinely observed, not asserted. Writing `*a = 1.0` followed by `*b = 2.0` and reading both back confirms the consequence exactly as predicted — `a[0]` reads `2`, the value written through `b`, because there was never two buffers to begin with. Rust's compiler did not warn about any of this: `pool.release(activations, 128)` twice in a row is, to the type system, no different from calling `println!` twice in a row with the same integer — a `*mut f32` carries none of the "used exactly once" tracking that a genuinely owning Rust type, like `RefCountedBuffer<T>`, gets for free.

> `[COMMON TRAP]` Two claims sit side by side in this chapter, and neither one fully survives contact with the chapter's own code. First: "every allocator in this chapter follows the acquire-in-constructor, release-in-`Drop` rule, and Rust's compiler enforces it." Section 11.3's `DeviceMemoryPool` doesn't — it has no `Drop` impl at all, as directly measured. The rule holds for `RefCountedBuffer<T>` and `Arena`, both of which define an explicit `Drop` impl; it does not hold for the third struct this same chapter builds, one section earlier. Second, narrower claim: that nothing can alias or double-release through this interface, *because this is Rust*. `release()` takes a plain `*mut f32`, and Worked Example 11.4.1 just showed, with real printed pointer values, that the same address genuinely comes back out of `acquire()` twice after a double `release()` — the safety Rust seems to promise by reputation is real only where a type's ownership is actually tracked by the compiler, and a bare pointer type opts out of that tracking entirely, on purpose, because `unsafe` code needs an escape hatch that isn't fighting the borrow checker on every line. Fixing this for real would mean giving up on a bare `*mut f32` at the `acquire`/`release` boundary the same way this book gave up on a bare pointer inside `Tensor` back in Chapter 6 — wrapping it in a small owning type (much like `RefCountedBuffer<T>` itself, or the kind of move-only handle Worked Example 11.1.2 showed Rust hands out automatically to anything that doesn't `.clone()`) whose value is *consumed* by `release()` — taking `self` rather than `&self`, so a second attempt to release the same handle is rejected at compile time as a use of an already-moved value, rather than silently succeeding at run time the way a bare pointer allows.

Taken together with `Arena`'s macro choice from Section 11.2, this chapter's three structures don't earn the blanket safety claim a first pass at this section's prose would like to make for them: `RefCountedBuffer<T>`'s shared-ownership discipline genuinely works as described, and is genuinely *stronger* than its C++ counterpart because the compiler refuses the accidental-sharing mistake outright (Worked Example 11.1.2) — but `Arena`'s protection against overrun depends on which of two similarly-named macros its author reached for, and `DeviceMemoryPool` — the very struct Worked Example 11.4.1 just misused — has no cleanup path and no protection against being handed the same pointer twice, because both of its problems live at exactly the boundary where Rust's ownership guarantees stop applying: a raw pointer with no `Drop` impl watching it. Combined, Part 1 now has a real memory story: deterministic single ownership from Chapter 6 that the compiler itself enforces, shared ownership where a graph needs it and is willing to pay for the `unsafe` code to buy it back (Section 11.1), bump allocation where per-step churn dominates and the safety-versus-speed trade-off is a genuine choice between two macros rather than an automatic consequence of the language (Section 11.2), and a device-side pool where the *allocation itself*, not the memory, is the expensive resource (Section 11.3) — with two specific, genuinely demonstrated gaps in that story now on the record rather than taken on faith.

## 11.5 Complete Runnable Code

### File: `01_refcounted_buffer.rs`

```rust
use std::alloc::{alloc, dealloc, Layout};

struct RefCountedBuffer<T> {
    data: *mut T,
    refcount: *mut usize,
    count: usize,
    data_layout: Layout,
}

impl<T> RefCountedBuffer<T> {
    fn new(count: usize) -> Self {
        // A real device Tensor buffer would allocate through cudarc here (Chapter 6);
        // this uses the plain global allocator so the refcounting mechanics below can
        // genuinely execute end to end on the host in this no-GPU sandbox -- Chapter 10
        // already established why a real device allocation can't succeed here.
        let data_layout = Layout::array::<T>(count).unwrap();
        let data = unsafe { alloc(data_layout) as *mut T };
        let refcount = unsafe { alloc(Layout::new::<usize>()) as *mut usize };
        unsafe {
            *refcount = 1;
        }
        println!(
            "  RefCountedBuffer::new(count={}) constructed -> refcount={}",
            count,
            unsafe { *refcount }
        );
        RefCountedBuffer { data, refcount, count, data_layout }
    }

    fn get_refcount(&self) -> usize {
        unsafe { *self.refcount }
    }
}

// Unlike C++'s implicit copy constructor -- invoked just by writing
// `RefCountedBuffer<float> v(t);` -- Rust never copies a type that owns a
// Drop impl unless something explicitly asks it to. `let v = t;` below would
// MOVE t's fields into v, not share them, and using `t` afterward would be a
// compile error (see 02_move_vs_clone_demo.rs). To get C++'s "both t and v
// point at the same buffer, both destructors eventually run" behavior, this
// chapter has to opt in on purpose, with a hand-written Clone impl that does
// exactly what the C++ copy constructor did: copy the pointers, not the data,
// and bump the shared count.
impl<T> Clone for RefCountedBuffer<T> {
    fn clone(&self) -> Self {
        unsafe {
            *self.refcount += 1;
        }
        println!("  clone() ran (shares buffer) -> refcount={}", self.get_refcount());
        RefCountedBuffer {
            data: self.data,
            refcount: self.refcount,
            count: self.count,
            data_layout: self.data_layout,
        }
    }
}

impl<T> Drop for RefCountedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            *self.refcount -= 1;
        }
        println!("  drop() ran -> refcount={}", self.get_refcount());
        if self.get_refcount() == 0 {
            println!("  refcount hit 0 -> freeing data and refcount");
            unsafe {
                dealloc(self.data as *mut u8, self.data_layout);
                dealloc(self.refcount as *mut u8, Layout::new::<usize>());
            }
        }
    }
}

fn main() {
    println!("=== Section 11.1: RefCountedBuffer, traced through three lifetimes ===");
    {
        let t: RefCountedBuffer<f32> = RefCountedBuffer::new(4);
        println!("t.get_refcount() = {}", t.get_refcount());
        {
            let v = t.clone();
            println!(
                "v.get_refcount() = {}, t.get_refcount() = {}",
                v.get_refcount(),
                t.get_refcount()
            );
        }
        println!("after v goes out of scope, t.get_refcount() = {}", t.get_refcount());
    }
    println!("after t goes out of scope, buffer has been freed");
}
```

```bash
rustc --edition 2024 -O 01_refcounted_buffer.rs -o 01_refcounted_buffer
./01_refcounted_buffer
```

### File: `02_move_vs_clone_demo.rs` — the compile error from Worked Example 11.1.2

```rust
use std::alloc::{alloc, dealloc, Layout};

struct RefCountedBuffer<T> {
    data: *mut T,
    refcount: *mut usize,
    count: usize,
    data_layout: Layout,
}

impl<T> RefCountedBuffer<T> {
    fn new(count: usize) -> Self {
        let data_layout = Layout::array::<T>(count).unwrap();
        let data = unsafe { alloc(data_layout) as *mut T };
        let refcount = unsafe { alloc(Layout::new::<usize>()) as *mut usize };
        unsafe {
            *refcount = 1;
        }
        RefCountedBuffer { data, refcount, count, data_layout }
    }

    fn get_refcount(&self) -> usize {
        unsafe { *self.refcount }
    }
}

impl<T> Drop for RefCountedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            *self.refcount -= 1;
        }
        if self.get_refcount() == 0 {
            unsafe {
                dealloc(self.data as *mut u8, self.data_layout);
                dealloc(self.refcount as *mut u8, Layout::new::<usize>());
            }
        }
    }
}

// Deliberately NOT calling .clone() here -- this is the direct, unguarded
// translation of C++'s `RefCountedBuffer<float> v(t);`, written the way
// someone porting the C++ chapter line-for-line might first try it.
fn main() {
    let t: RefCountedBuffer<f32> = RefCountedBuffer::new(4);
    println!("t.get_refcount() before move = {}", t.get_refcount());
    let v = t;
    println!("v.get_refcount() = {}", v.get_refcount());
    // t was moved into v on the line above -- t is no longer a valid value
    // to use. This line is expected to fail to compile.
    println!("t.get_refcount() after move = {}", t.get_refcount());
}
```

```bash
rustc --edition 2024 02_move_vs_clone_demo.rs -o 02_move_vs_clone_demo
```

*(This file is expected to fail to compile — see Worked Example 11.1.2 for the exact `E0382` it produces. It is kept here, in the Complete Runnable Code listing, precisely because it never produces a runnable binary; the failure itself is this file's entire content.)*

### File: `03_arena_allocator.rs`

```rust
use std::alloc::{alloc, dealloc, Layout};

struct Arena {
    base: *mut u8,
    capacity: usize,
    offset: usize,
    layout: Layout,
}

impl Arena {
    fn new(capacity_bytes: usize) -> Self {
        let layout = Layout::from_size_align(capacity_bytes, 1).unwrap();
        let base = unsafe { alloc(layout) };
        Arena { base, capacity: capacity_bytes, offset: 0, layout }
    }

    fn alloc<T>(&mut self, count: usize) -> *mut T {
        let bytes_needed = count * std::mem::size_of::<T>();
        // 256-byte alignment matches the alignment guarantee a real device
        // allocation gives (Chapter 7.3, exercised for real in Chapter 10.2)
        // -- so a buffer this arena hands out is as usable for a vectorized
        // wide::f32x8 load (Chapter 7.4) as one from the device path would be.
        let aligned_offset = (self.offset + 255) & !255usize;
        assert!(aligned_offset + bytes_needed <= self.capacity, "Arena exhausted");
        let ptr = unsafe { self.base.add(aligned_offset) as *mut T };
        self.offset = aligned_offset + bytes_needed;
        ptr
    }

    fn reset(&mut self) {
        self.offset = 0;
    }

    fn bytes_used(&self) -> usize {
        self.offset
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.base, self.layout);
        }
    }
}

fn main() {
    println!("=== Section 11.2: Arena, alignment arithmetic traced by hand ===");
    let mut arena = Arena::new(4096);

    let a = arena.alloc::<f32>(100);
    println!("after alloc::<f32>(100): offset={} (expected 400)", arena.bytes_used());

    let b = arena.alloc::<f32>(50);
    println!("after alloc::<f32>(50):  offset={} (expected 712)", arena.bytes_used());
    let b_start = unsafe { (b as *mut u8).offset_from(arena.base) as usize };
    println!(
        "second allocation's aligned start = {}, padding skipped = {}",
        b_start,
        b_start - 400
    );

    // touch both pointers so nothing above is optimized away as dead code,
    // and so the returned addresses are genuinely usable, not just computed
    unsafe {
        *a = 1.5;
        *b = 2.5;
    }
    println!("a[0] written through arena pointer = {}", unsafe { *a });
    println!("b[0] written through arena pointer = {}", unsafe { *b });

    arena.reset();
    println!("after reset(): offset={}", arena.bytes_used());
}
```

```bash
rustc --edition 2024 -O 03_arena_allocator.rs -o 03_arena_allocator
./03_arena_allocator
```

### File: `04a_arena_overrun_assert.rs` — Rust's `assert!`, tested in both build profiles

```rust
struct Arena {
    base: *mut u8,
    capacity: usize,
    offset: usize,
}

impl Arena {
    fn new(capacity_bytes: usize) -> Self {
        use std::alloc::{alloc, Layout};
        let layout = Layout::from_size_align(capacity_bytes, 1).unwrap();
        let base = unsafe { alloc(layout) };
        Arena { base, capacity: capacity_bytes, offset: 0 }
    }

    fn alloc<T>(&mut self, count: usize) -> *mut T {
        let bytes_needed = count * std::mem::size_of::<T>();
        let aligned_offset = (self.offset + 255) & !255usize;
        // A direct, unguarded translation of C++'s `assert(... && "Arena exhausted")`:
        // Rust's assert! macro, unlike C's assert(), is NOT gated by any
        // build-profile flag -- it is compiled into the binary and checked
        // at runtime regardless of debug or release. This file exists to
        // test that claim directly rather than assume it.
        assert!(aligned_offset + bytes_needed <= self.capacity, "Arena exhausted");
        let ptr = unsafe { self.base.add(aligned_offset) as *mut T };
        self.offset = aligned_offset + bytes_needed;
        ptr
    }
}

fn main() {
    println!("=== requesting far more than an 800-byte arena's capacity (assert!) ===");
    let mut arena = Arena::new(800);
    // 1000 f32s = 4000 bytes, into an 800-byte arena
    let p = arena.alloc::<f32>(1000);
    println!(
        "alloc did not abort. pointer offset from base = {} bytes (arena capacity = {})",
        unsafe { (p as *mut u8).offset_from(arena.base) },
        arena.capacity
    );
    println!("writing through the returned pointer now (this touches memory past the arena)...");
    unsafe {
        *p = 3.14;
    }
    println!("wrote p[0] = {} without any detected error", unsafe { *p });
}
```

```bash
# "debug" build: plain rustc, debug-assertions on by default
rustc --edition 2024 04a_arena_overrun_assert.rs -o 04a_debug
./04a_debug
echo "exit code: $?"

# "release" build: rustc -O, debug-assertions off by default
rustc --edition 2024 -O 04a_arena_overrun_assert.rs -o 04a_release
./04a_release
echo "exit code: $?"
```

### File: `04b_arena_overrun_debug_assert.rs` — Rust's `debug_assert!`, the actual `NDEBUG` analog

```rust
struct Arena {
    base: *mut u8,
    capacity: usize,
    offset: usize,
}

impl Arena {
    fn new(capacity_bytes: usize) -> Self {
        use std::alloc::{alloc, Layout};
        let layout = Layout::from_size_align(capacity_bytes, 1).unwrap();
        let base = unsafe { alloc(layout) };
        Arena { base, capacity: capacity_bytes, offset: 0 }
    }

    fn alloc<T>(&mut self, count: usize) -> *mut T {
        let bytes_needed = count * std::mem::size_of::<T>();
        let aligned_offset = (self.offset + 255) & !255usize;
        // debug_assert! -- not assert! -- is Rust's actual analog of C's
        // NDEBUG-gated assert(): the check (and the condition expression
        // itself) is compiled in only when debug-assertions are enabled,
        // and compiled out entirely otherwise. -O implies
        // debug-assertions=off by default, the same codegen flag that
        // already turned off overflow-checks for Chapter 9's usize
        // underflow -- both are governed by the same on/off switch.
        debug_assert!(aligned_offset + bytes_needed <= self.capacity, "Arena exhausted");
        let ptr = unsafe { self.base.add(aligned_offset) as *mut T };
        self.offset = aligned_offset + bytes_needed;
        ptr
    }
}

fn main() {
    println!("=== requesting far more than an 800-byte arena's capacity (debug_assert!) ===");
    let mut arena = Arena::new(800);
    // 1000 f32s = 4000 bytes, into an 800-byte arena
    let p = arena.alloc::<f32>(1000);
    println!(
        "alloc did not abort. pointer offset from base = {} bytes (arena capacity = {})",
        unsafe { (p as *mut u8).offset_from(arena.base) },
        arena.capacity
    );
    println!("writing through the returned pointer now (this touches memory past the arena)...");
    unsafe {
        *p = 3.14;
    }
    println!("wrote p[0] = {} without any detected error", unsafe { *p });
}
```

```bash
rustc --edition 2024 04b_arena_overrun_debug_assert.rs -o 04b_debug
./04b_debug
echo "exit code: $?"

rustc --edition 2024 -O 04b_arena_overrun_debug_assert.rs -o 04b_release
./04b_release
echo "exit code: $?"
```

### File: `05_device_memory_pool.rs`

```rust
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::sync::atomic::{AtomicIsize, Ordering};

// A tracked allocator standing in for a real device allocation: Chapter 10
// already established that a genuine device allocation fails immediately in
// this no-GPU sandbox, so DeviceMemoryPool below routes through this tracked
// alloc/free pair instead -- purely so that the leak this section
// demonstrates is one we can genuinely count, not merely narrate. An
// AtomicIsize, not a `static mut`, is the idiomatic way to hold mutable
// global state in Rust -- edition 2024 denies taking a reference to a
// `static mut` outright, and the atomic is just as simple here since this
// program is single-threaded.
static G_OUTSTANDING_ALLOCATIONS: AtomicIsize = AtomicIsize::new(0);

fn tracked_alloc(count: usize) -> *mut f32 {
    G_OUTSTANDING_ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
    let layout = Layout::array::<f32>(count).unwrap();
    unsafe { alloc(layout) as *mut f32 }
}

#[allow(dead_code)]
fn tracked_free(ptr: *mut f32, count: usize) {
    G_OUTSTANDING_ALLOCATIONS.fetch_sub(1, Ordering::SeqCst);
    let layout = Layout::array::<f32>(count).unwrap();
    unsafe {
        dealloc(ptr as *mut u8, layout);
    }
}

struct DeviceMemoryPool {
    free_lists: HashMap<usize, Vec<*mut f32>>,
    bytes_allocated: i64,
    bytes_reused: i64,
    // Deliberately no Drop impl -- Section 11.3's point.
}

impl DeviceMemoryPool {
    fn new() -> Self {
        DeviceMemoryPool { free_lists: HashMap::new(), bytes_allocated: 0, bytes_reused: 0 }
    }

    fn acquire(&mut self, count: usize) -> *mut f32 {
        if let Some(list) = self.free_lists.get_mut(&count) {
            if let Some(ptr) = list.pop() {
                self.bytes_reused += (count * 4) as i64;
                return ptr;
            }
        }
        self.bytes_allocated += (count * 4) as i64;
        tracked_alloc(count)
    }

    fn release(&mut self, ptr: *mut f32, count: usize) {
        self.free_lists.entry(count).or_insert_with(Vec::new).push(ptr);
    }

    fn hit_rate(&self) -> f64 {
        let total = self.bytes_allocated + self.bytes_reused;
        if total > 0 {
            self.bytes_reused as f64 / total as f64
        } else {
            0.0
        }
    }
}

fn main() {
    println!("=== Section 11.3: DeviceMemoryPool, three steps toward a hit rate ===");
    {
        let mut pool = DeviceMemoryPool::new();

        let step1 = pool.acquire(256);
        println!(
            "step 1: acquire(256) -> fresh allocation. bytes_allocated={} bytes_reused={}",
            pool.bytes_allocated, pool.bytes_reused
        );
        pool.release(step1, 256);

        let step2 = pool.acquire(256);
        println!(
            "step 2: acquire(256) -> reused? {}. bytes_allocated={} bytes_reused={}",
            if step2 == step1 { "true (same pointer)" } else { "false" },
            pool.bytes_allocated,
            pool.bytes_reused
        );
        pool.release(step2, 256);

        let step3 = pool.acquire(256);
        println!(
            "step 3: acquire(256) -> reused? {}. bytes_allocated={} bytes_reused={}",
            if step3 == step1 { "true (same pointer)" } else { "false" },
            pool.bytes_allocated,
            pool.bytes_reused
        );
        pool.release(step3, 256);

        println!(
            "hit_rate() = {} / {} = {:.3}",
            pool.bytes_reused,
            pool.bytes_allocated + pool.bytes_reused,
            pool.hit_rate()
        );

        println!(
            "G_OUTSTANDING_ALLOCATIONS while pool is still alive = {}",
            G_OUTSTANDING_ALLOCATIONS.load(Ordering::SeqCst)
        );
    } // pool goes out of scope here -- the compiler-generated Drop glue for
      // HashMap<usize, Vec<*mut f32>> runs, since DeviceMemoryPool itself
      // defines no Drop impl of its own.
    println!(
        "G_OUTSTANDING_ALLOCATIONS after pool went out of scope = {} (expected: still 1, not 0)",
        G_OUTSTANDING_ALLOCATIONS.load(Ordering::SeqCst)
    );
}
```

```bash
rustc --edition 2024 -O 05_device_memory_pool.rs -o 05_device_memory_pool
./05_device_memory_pool
```

### File: `06_raii_and_pointer_safety.rs`

```rust
use std::alloc::{alloc, Layout};
use std::collections::HashMap;

struct DeviceMemoryPool {
    free_lists: HashMap<usize, Vec<*mut f32>>,
    bytes_allocated: i64,
    bytes_reused: i64,
}

impl DeviceMemoryPool {
    fn new() -> Self {
        DeviceMemoryPool { free_lists: HashMap::new(), bytes_allocated: 0, bytes_reused: 0 }
    }

    fn acquire(&mut self, count: usize) -> *mut f32 {
        if let Some(list) = self.free_lists.get_mut(&count) {
            if let Some(ptr) = list.pop() {
                self.bytes_reused += (count * 4) as i64;
                return ptr;
            }
        }
        self.bytes_allocated += (count * 4) as i64;
        let layout = Layout::array::<f32>(count).unwrap();
        unsafe { alloc(layout) as *mut f32 }
    }

    // release() takes a plain *mut f32 -- a raw pointer, not an owned value.
    // A raw pointer in Rust is Copy (unconditionally, regardless of what it
    // points at), meaning this call takes a copy of the address, not
    // something borrowed or moved out of the caller. Nothing about this
    // signature stops the same pointer from being passed in twice, or read
    // from again afterward -- the exact same gap Section 11.4's C++ edition
    // found in its own release(float*, int), for the same underlying reason:
    // a raw pointer opts out of the ownership tracking that would otherwise
    // be Rust's whole pitch.
    fn release(&mut self, ptr: *mut f32, count: usize) {
        self.free_lists.entry(count).or_insert_with(Vec::new).push(ptr);
    }
}

fn main() {
    println!("=== Section 11.4: what release()'s raw-pointer signature does not prevent ===");
    let mut pool = DeviceMemoryPool::new();

    let activations = pool.acquire(128);
    println!("acquired activations = {:?}", activations);

    // A caller bug: release the same pointer twice. Nothing in release()'s
    // signature -- a plain *mut f32, not an owned/consumed value -- rejects
    // this at compile time or catches it at run time. Because *mut f32 is
    // Copy, this compiles without so much as a warning.
    pool.release(activations, 128);
    pool.release(activations, 128);
    println!("released the SAME pointer twice -- compiled and ran without complaint");

    // The free list for size 128 now holds the identical address twice.
    let a = pool.acquire(128);
    let b = pool.acquire(128);
    println!("acquire(128) #1 -> {:?}", a);
    println!("acquire(128) #2 -> {:?}", b);
    println!("are they the same address? {}", if a == b { "true -- ALIASED" } else { "false" });

    if a == b {
        println!("writing through 'a' now silently corrupts whatever 'b' is used for,");
        println!("and vice versa -- two live tensors sharing one buffer, neither aware of it.");
        unsafe {
            *a = 1.0;
            *b = 2.0;
        }
        println!(
            "after a[0]=1.0 then b[0]=2.0: a[0]={} b[0]={} (both read back the same memory)",
            unsafe { *a },
            unsafe { *b }
        );
    }
}
```

```bash
rustc --edition 2024 -O 06_raii_and_pointer_safety.rs -o 06_raii_and_pointer_safety
./06_raii_and_pointer_safety
```

## Chapter Summary

`RefCountedBuffer<T>` generalizes Chapter 6's single-owner `Tensor` and Chapter 7.2's never-owning `TensorView<'a>` into genuine shared ownership: a counter incremented on every `.clone()` and decremented on every `Drop`, so that whichever live clone happens to see the count reach zero is correctly the one that frees the buffer, traced end to end through a three-step lifetime (`1 → 2 → 1 → 0`) and genuinely compiled and run to confirm it — with a second, Rust-specific worked example showing that the C++ edition's implicit-sharing bug class doesn't even compile here without an explicit `Clone` impl opting into it (`E0382`, genuinely produced). `Arena` trades that per-clone bookkeeping for a bump-pointer scheme — one addition and a 256-byte alignment mask per allocation, chosen to match a real device allocation's own guarantee rather than an arbitrary round number, verified by hand and by execution against this chapter's own `0 → 400 → 512 → 712` offset sequence — with individual frees replaced entirely by one bulk `reset()`, at the cost of a bounds check whose behavior turned out to hinge on which of two near-identical-looking macros its author reached for: `assert!`, genuinely tested in both a plain and an `-O` build, panics identically both times (exit code `101` twice), while `debug_assert!` — Rust's actual `NDEBUG` analog — panics in the plain build and vanishes entirely in the `-O` build, reproducing the CUDA edition's abort-versus-silent-corruption divergence exactly, once the right macro is used. `DeviceMemoryPool` trades the *allocation itself* for reuse, bucketing released buffers by exact element count and reaching a hit rate of exactly `0.667` after three steps — but the struct defines no `Drop` impl at all, and a tracked-allocation counter (an `AtomicIsize`, not a `static mut`) directly measured the resulting leak at exactly one outstanding buffer for the pool's entire lifetime, rather than merely asserting one exists. Section 11.4 then took the chapter's own closing safety claim at face value and broke it on purpose: releasing the same raw `*mut f32` twice compiles without a warning, because a raw pointer is unconditionally `Copy` and carries none of the ownership tracking that made Worked Example 11.1.2's move-only `RefCountedBuffer<T>` refuse to compile a comparable mistake — and the next two `acquire()` calls genuinely return the identical address to two unrelated callers, a real, printed, observed aliasing bug arising directly from `release()`'s bare-pointer signature.

## Self-Check Questions

1. `let a = RefCountedBuffer::<f32>::new(10)`, then `let b = a.clone()`, then `let c = b.clone()`. In what order would the refcount need to reach `0` for the underlying buffer to be freed, and does it matter which of `a`, `b`, or `c` goes out of scope last? Separately: what would happen if the second line had been written `let b = a;` instead of `let b = a.clone();`, and why?
2. An `Arena` with `offset = 300` receives a request for `20` `f32`s (`80` bytes). Compute `(300 + 255) & !255usize` by hand the way Worked Example 11.2.1 did, state the resulting aligned offset, and state how many bytes of padding were skipped.
3. Worked Example 11.2.2 compiled the identical over-capacity request behind two different macros, in two different build profiles each. Explain concretely why `assert!` produced identical behavior (exit code `101`) in both the plain and the `-O` build, while `debug_assert!` diverged between them — and name the specific compiler flag responsible for the divergence.
4. A `DeviceMemoryPool` is created, used for several `acquire`/`release` cycles across two distinct sizes, and then goes out of scope. Trace what happens to (a) the `HashMap`/`Vec` bookkeeping structures inside `free_lists`, and (b) the actual buffers those structures were holding pointers to. Are both reclaimed?
5. Worked Example 11.4.1 released the same pointer twice and then observed two `acquire()` calls return the identical address. Explain, in terms of what kind of Rust type a `*mut f32` is, why nothing at the type-system level catches this — and describe concretely what would have to change about `acquire`/`release`'s signatures for a double-release like this to fail to compile instead.

## Where We Go Next

Chapter 12 is the first chapter of Part 2, and builds the first real tensor operations directly on top of the memory story this chapter completed: single ownership from Chapter 6 for the common case, `RefCountedBuffer<T>` from Section 11.1 wherever an operation's result needs to alias rather than copy, and the `Arena`/`DeviceMemoryPool` machinery from Sections 11.2–11.3 wherever an operation's intermediate results are cheap to churn through and expensive to allocate one at a time — with Section 11.4's aliasing bug now a named, understood risk rather than a surprise waiting in Part 2's first kernel launch, and with `assert!`-versus-`debug_assert!` now a deliberate choice this book will keep making explicitly rather than by habit.

## Worked Solutions

**1.** The refcount starts at `1` when `a` is constructed, becomes `2` when `a.clone()` produces `b`, and becomes `3` when `b.clone()` produces `c`. It needs to count down from `3` to `0` — through `2`, then `1`, then `0` — for the buffer to free, and each step happens whenever any one of the three currently-live clones has its `Drop` impl run, regardless of which variable that clone is bound to. It does *not* matter which of `a`, `b`, or `c` happens to go out of scope last — the shared counter, not the specific variable, determines when the buffer is actually freed. Had the second line been `let b = a;` instead, `a`'s fields would have been **moved** into `b` rather than shared: `a` would become invalid to use for the rest of the program (any later reference to `a` is a compile-time `E0382`, exactly as Worked Example 11.1.2 demonstrated), the refcount would never be incremented by that line at all, and there would only ever be one live value (first named `a`, then `b`) at any given point — not two clones sharing one buffer.

**2.** `(300 + 255) & !255usize = 555 & !255usize`. `555 = 2×256 + 43`, so clearing the low 8 bits removes the `43` and leaves `512`. The aligned offset is `512`, and the padding skipped is `512 - 300 = 212` bytes — within the `0`-to-`255`-byte range Worked Example 11.2.2 established as the maximum possible waste per allocation.

**3.** `assert!` is not conditionally compiled at all in Rust — the macro always expands to a runtime check, in every build profile, with no flag that strips it. That's why both the plain build and the `-O` build panicked identically at the same line, with the same exit code `101`: there was never a difference in what code got compiled, only (irrelevantly, for this particular check) a difference in optimization level. `debug_assert!`, by contrast, expands to nothing at all — not even the condition expression is evaluated — whenever the crate is compiled with `debug-assertions` off, which `-O` sets by default alongside `overflow-checks` (the same flag pairing Chapter 9.4 already found governing the `usize` underflow panic). The specific flag is `-C debug-assertions=off`, implied automatically by `-O` unless overridden; passing `rustc -O -C debug-assertions=on` would make even the `-O` build panic, and conversely `rustc -C debug-assertions=off` (without `-O`) would make even the unoptimized build skip the check.

**4.** The `HashMap<usize, Vec<*mut f32>>` and its inner `Vec`s are reclaimed: the compiler's synthesized drop glue for `DeviceMemoryPool` tears down `free_lists` using `HashMap`'s own `Drop` behavior, which in turn drops each `Vec<*mut f32>`, freeing the array that held the pointer *values* (addresses) themselves. The actual buffers those addresses pointed to are not reclaimed: a raw `*mut f32` has no `Drop` impl that calls `dealloc` on itself, so tearing down the vector of addresses simply discards the addresses — it never issues the deallocation call those addresses would need for the underlying allocations to actually be released. Only (a) happens; (b) is the leak Section 11.3 measured directly via `G_OUTSTANDING_ALLOCATIONS`.

**5.** A raw `*mut f32` in Rust is `Copy` unconditionally — copying it, passing it by value, or handing it to a function like `release(&mut self, ptr: *mut f32, count: usize)` is indistinguishable, at the type-system level, from copying a `usize`. `release()` takes `ptr` by value as an ordinary `Copy` parameter, meaning the call passes a *copy* of the address, not a value the caller gives up — nothing about calling `pool.release(activations, 128)` a second time invalidates `activations` or prevents the identical call from compiling and running again, in exactly the same way nothing prevented `let b = a;` from compiling for a `Copy` type. For a double-release like Worked Example 11.4.1's to fail to compile instead, `acquire`/`release` would need to work with a type the compiler actually tracks ownership of — for instance, `acquire` returning a move-only handle (a small non-`Copy` wrapper, in the spirit of this chapter's own `RefCountedBuffer<T>`) and `release` taking that handle by value (`self`, not `&self`) so the value is *consumed* rather than merely read, the same mechanism that made Worked Example 11.1.2's unwrapped `let v = t;` a compile error in the first place — so that a second attempt to release the same handle would be rejected at compile time as a use of an already-moved value, rather than silently succeeding at run time the way a bare pointer allows.
