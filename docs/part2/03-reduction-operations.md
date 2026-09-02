# Chapter 14: Reduction Operations — Collapsing Many Values Into One

> "Every loss function in this book ends the same way: a tensor of per-example errors collapses into a single scalar a training loop can actually act on. Every operation before this chapter preserved shape — the same number of positions went in as came out. A reduction is the first place that stops being true on purpose, and it's also the first place where doing it the parallel-obvious way — every thread adds its value into one shared accumulator — is a race condition, not an optimization. It's also the first place in this book where a faithfully-ported bug's *symptom* changes because the language changed underneath it: the same missing kernel-launch call that returns different garbage on every C++ run returns the exact same wrong answer on every safe-Rust run, for a reason worth understanding precisely rather than glossing over."

**What you will understand by the end of this chapter:**

- The tree-reduction pattern — pair, combine, halve, repeat — traced by hand on `[1, 4, 9, 16]` round by round until one value remains, and why every pair *within* a round is independent while rounds themselves must run in strict sequence
- Why `max_reduce_kernel` carries an index buffer alongside its value buffer, and how that surviving index becomes the one position a backward pass's gradient flows through, with every other position receiving exactly zero — plus a genuine, safe-Rust demonstration of what its missing `else` branch actually leaves behind
- The L2 norm as sum-of-squares-then-square-root, and gradient-norm clipping as the one place in this book where a norm's *value* changes what happens next rather than just measuring something
- Variance and standard deviation as sum-and-mean applied twice — once for the mean itself, once for the mean of squared deviations from it — traced on a full eight-value dataset down to an exact variance of `4` and standard deviation of `2`
- Why a faithfully-ported `tensor_sum` driver's own per-round kernel launch is written as a comment rather than working code, genuinely run to show what it returns — and why that returned value is a completely different *kind* of wrong in idiomatic Rust than in C++, purely because of what each language's default memory allocation actually guarantees

**What you need to know first:**

- Chapter 4 and Chapter 12 (kernel launch mechanics and the one-thread-per-position pattern — `sum_reduce_kernel` and `max_reduce_kernel` both modify that pattern so that one thread now reads *two* input positions and writes one output position, instead of one-to-one)
- Chapter 11.2 and 11.3 (the allocate-then-free discipline this chapter's scratch buffers follow — allocate once up front, free exactly once at the end, the same shape as the arena and pool patterns those sections built)
- Chapter 10 (no GPU exists in this sandbox to launch a kernel on at all — the reason every reduction in this chapter is verified through a host-executable mirror of each kernel's exact per-thread logic, not a genuine device launch)
- Chapter 13.1's shape-check-gap finding (the same "same bug, different consequence depending on what the buffer's real bounds are" shape reappears in this chapter's own `tensor_sum_incomplete`, this time driven by initialization rather than bounds)

## 14.1 Sum and Mean Reductions `[FOUNDATIONAL]`

### Intuition

A straight-line loop that adds every element into one running total is easy to write and easy to get wrong in parallel: if a thousand GPU threads all try to add into that same total simultaneously, most of those additions are lost — two threads reading the current total, both adding their own value, and both writing back overwrite each other, so only one addition out of every colliding pair actually survives. **Tree reduction** avoids this by never letting two threads touch the same memory at the same time: pair up elements and combine each pair, then pair up *those* results and combine again, halving the array's length each round until one value remains. Every combine within a single round reads and writes entirely disjoint memory, so a round's worth of pairs really can run at once — the discipline is only that one round has to finish completely before the next round starts, since round `N+1`'s inputs are round `N`'s outputs.

### Background

```rust
// One round of pairwise reduction: size elements -> ceil(size/2). A genuine
// GPU kernel would take this exact per-thread shape; this sandbox has no
// device to launch one on at all (Chapter 10), which is exactly why a host
// mirror -- run for every simulated tid on the CPU -- stands in for it here.
fn sum_reduce_round_host(output: &mut [f32], input: &[f32], size: usize) {
    let next_size = (size + 1) / 2;
    for tid in 0..next_size {
        if tid * 2 + 1 < size {
            output[tid] = input[tid * 2] + input[tid * 2 + 1];
        } else if tid * 2 < size {
            output[tid] = input[tid * 2]; // odd leftover, pass through
        } else {
            output[tid] = 0.0; // no work this round: zero-filled
        }
    }
}
```

`sum_reduce_round_host` is one round: simulated thread `tid` combines input positions `2·tid` and `2·tid+1`, unless `2·tid` is the very last element of an odd-length input, in which case it passes that lone value through unchanged (the `else if` branch) — an explicit `else` branch even zero-fills any tid that has no work at all, so every output position this round writes gets a defined value, never leftover memory from a previous round.

A faithful port of the driver that repeatedly calls this per-round logic looks like this:

```rust
fn tensor_sum_incomplete(input: &[f32]) -> f32 {
    let size = input.len();
    let mut current: &[f32] = input;
    let scratch: Vec<f32> = vec![0.0; size]; // zero-initialized, unlike malloc
    let mut current_size = size;
    while current_size > 1 {
        let next_size = (current_size + 1) / 2;
        // TODO: launch sum_reduce_kernel with current_size threads,
        // writing into scratch
        current = &scratch;
        current_size = next_size;
    }
    current[0]
}
```

That function is deliberately preserved with the same gap the design it's ported from has: the per-round kernel launch is a comment, not code. Rather than quietly fix it, this chapter compiles and runs it exactly as shown, then verifies the *algorithm* is correct by running each kernel's per-thread logic on the host instead:

```rust
// The corrected, genuinely-executable host mirror: one round of
// sum_reduce_kernel's exact per-thread logic, run for every tid on the CPU.
fn tensor_sum_host(input: &[f32]) -> f32 {
    let size = input.len();
    let mut current: Vec<f32> = input.to_vec();
    let mut scratch: Vec<f32> = vec![0.0; size];
    let mut current_size = size;
    while current_size > 1 {
        let next_size = (current_size + 1) / 2;
        sum_reduce_round_host(&mut scratch, &current, current_size);
        std::mem::swap(&mut current, &mut scratch);
        current_size = next_size;
    }
    current[0]
}

fn tensor_mean_host(input: &[f32]) -> f32 {
    tensor_sum_host(input) / input.len() as f32
}
```

`std::mem::swap` plays the role of the CUDA edition's manual pointer-swap idiom (`float* tmp = current; current = scratch; scratch = tmp;`) — swapping which `Vec` the `current`/`scratch` bindings refer to, without moving or copying any of the actual float data, the same ping-pong-buffer discipline the C++ version uses.

### Worked Example 14.1.1 — Tree reduction on `[1, 4, 9, 16]`, round by round

```
Round 0: [ 1,  4,  9, 16]
           \  /    \  /
Round 1: [  5   ,   25  ]
             \       /
Round 2: [     30      ]
```

Compiled and run:

```bash
rustc --edition 2024 -O 01_sum_mean_reduction.rs -o 01_sum_mean_reduction
./01_sum_mean_reduction
```

Genuinely compiled and run:

```
tensor_sum_host([1,4,9,16]) = 30.0
tensor_mean_host([1,4,9,16]) = 7.5
```

Round `1` pairs `(1,4)→5` and `(9,16)→25`. Round `2` pairs `(5,25)→30`. Three additions total — the same count a straight-line loop would perform — but the two additions inside round `1` are fully independent (thread `0` touches only `input[0]` and `input[1]`; thread `1` touches only `input[2]` and `input[3]`), so a GPU can run both at once, then run round `2`'s single addition once both of round `1`'s results exist. `tensor_mean_host` divides the final sum by the count: `30 / 4 = 7.5`.

### Worked Example 14.1.2 — An odd-length array across three rounds

`[2, 6, 3, 8, 5]` (five elements, forcing the odd-leftover branch). Genuinely computed in the same run:

```
tensor_sum_host([2,6,3,8,5]) = 24.0 (expect 24)
```

Round `0`: `next_size = (5+1)/2 = 3`. Thread `0`: `2+6=8`. Thread `1`: `3+8=11`. Thread `2`: `tid*2=4`, which is `< 5` but `tid*2+1=5` is not — the leftover branch fires, passing `input[4]=5` straight through. Round `0` output: `[8, 11, 5]`.

Round `1`: `current_size=3`, `next_size=(3+1)/2=2`. Thread `0`: `8+11=19`. Thread `1`: `tid*2=2 < 3` but `tid*2+1=3` is not — leftover branch again, passing `5` straight through. Round `1` output: `[19, 5]`.

Round `2`: `current_size=2`, `next_size=1`. Thread `0`: `19+5=24`. Final result: `24` — matching a direct sum of the original five values, `2+6+3+8+5=24`, exactly, and matching the genuinely executed result above.

### Worked Example 14.1.3 — What the incomplete driver actually returns, genuinely run — and why the returned value's own *character* is a Rust-specific finding

```bash
rustc --edition 2024 -O 01_sum_mean_reduction.rs -o 01_sum_mean_reduction
./01_sum_mean_reduction
```

Genuinely compiled and run (three separate runs of the same binary):

```
tensor_sum_incomplete([1,4,9,16]) = 0 (idiomatic Rust: deterministic 0.0, NOT 30, and NOT random garbage)
tensor_sum_incomplete([1,4,9,16]) = 0
tensor_sum_incomplete([1,4,9,16]) = 0
```

Genuinely run three separate times, `tensor_sum_incomplete([1,4,9,16])` returns `0.0` — the exact same wrong answer, every single time. This is not a weaker demonstration of the CUDA edition's bug; it's a *different* bug, and the difference is worth tracing precisely rather than waved past. `scratch` here is built with `vec![0.0; size]`, the ordinary, idiomatic way to allocate a `Vec<f32>` in Rust — and unlike C's `malloc`, which reserves memory without writing anything into it, `vec![0.0; size]` genuinely, deterministically zero-initializes every element before the function ever sees it. The per-round kernel launch is still a comment, still not code, still never touches `scratch` — but because `scratch` started at `[0.0, 0.0, 0.0, 0.0]` rather than arbitrary bytes, `current` ends the loop pointing at a buffer that is reliably, provably all zeros, and `current[0]` reliably returns `0.0`.

That determinism is itself worth testing rather than trusting on reputation, and reproducing the CUDA edition's actual nondeterministic-garbage behavior in Rust turns out to require deliberately opting out of the language's safety guarantees:

```bash
rustc --edition 2024 -O 01b_sum_incomplete_uninit.rs -o 01b_sum_incomplete_uninit
./01b_sum_incomplete_uninit
```

Genuinely compiled and run (three separate runs of the same binary):

```
tensor_sum_incomplete_uninit([1,4,9,16]) = 0.000000000000000000000000000000000000000045914 (uninitialized memory via unsafe set_len, NOT 30)
tensor_sum_incomplete_uninit([1,4,9,16]) = 0.000000000000000000000000000000000000000045912
tensor_sum_incomplete_uninit([1,4,9,16]) = 0.000000000000000000000000000000000000000045912
```

`Vec::with_capacity(size)` followed by an `unsafe { v.set_len(size) }` call is the closest Rust equivalent to what `malloc` actually does: it reserves the memory without initializing it, and then the `set_len` call tells the compiler "trust me, all `size` elements are already valid data" — a promise that is, right at that line, false. Reading the buffer afterward is genuinely undefined behavior by Rust's own rules, in exactly the sense that reading a `malloc`'d-but-unwritten buffer is undefined behavior in C — it isn't a softer or "more checked" version of the same read. What comes back is real leftover bits from whatever this process's allocator had previously placed at that address, and it genuinely varies from run to run, confirmed here across three separate executions of the identical binary. It happens to vary in a narrower range in this sandbox than the CUDA edition's own demonstration (`0.0`, `nan`, and a large negative float across three runs) — a difference in *how much* the two runs' leftover bytes happen to differ, which depends on this process's own allocator history, not on anything more fundamental — but it is genuinely different every time, which is the entire point: nondeterministic-garbage behavior in Rust is not something ordinary, safe code produces by accident. It has to be built on purpose, with `unsafe`, and the moment that happens, Rust stops offering any of its usual guarantees about what gets read back.

```
[COMMON TRAP]  tensor_sum's own kernel launch is a comment, not code -- and
                what "wrong" means depends entirely on how scratch was built

Read the while loop in tensor_sum_incomplete literally: each iteration
computes next_size, then the very next line is a comment --
"// TODO: launch sum_reduce_kernel with current_size threads,
writing into scratch" -- describing what should happen, not code that makes
it happen. No kernel launch, nothing writes a single value into scratch at
any point. The very next line, `current = &scratch`, executes
unconditionally regardless, and by the time the loop exits, `current`
points at scratch -- whose contents were fixed once, at allocation time,
and never touched again.

Two completely different failure modes fall out of that single missing
line, and which one you get depends entirely on how scratch was allocated:
`vec![0.0; size]` -- ordinary, safe, idiomatic Rust -- means scratch was
genuinely all zeros before the loop ever ran, so the wrong answer is
`0.0`, every time, on every machine, forever. `Vec::with_capacity(size)`
plus `unsafe { set_len(size) }` -- reaching, on purpose, for C's exact
"reserve but don't initialize" behavior -- means scratch holds real
leftover bytes from the allocator's history, so the wrong answer is
different every run, and reading it at all is technically undefined
behavior by Rust's own rules, whether or not it happens to look
"reasonable" on a given run.

Neither version is more correct than the other -- both are missing the
same kernel launch, and both would need the exact same fix: an actual
call to sum_reduce_round_host (or a real kernel launch on hardware that
has a GPU to launch onto) inside that while loop, each round, before
current gets reassigned. What differs is only the SYMPTOM, and it differs
because Rust's ordinary, safe way of allocating memory happens to zero it,
while C's ordinary, unsafe-by-default way of allocating memory does not --
a language-level difference with real consequences for how a missing-code
bug like this one actually presents itself to whoever runs into it.

The 30 this chapter reports for tensor_sum_host([1,4,9,16]) describes
what sum_reduce_round_host's per-round logic actually computes when it IS
run each round -- verified independently in Worked Examples 14.1.1 and
14.1.2 above, both by hand and by tensor_sum_host's genuine execution
of that same per-thread logic on the host -- not what tensor_sum_incomplete
literally does as written, in either its safe or its unsafe form. And even
a corrected version, with a real kernel launch call in place of the
comment, would still fail to run in this specific sandbox: Chapter 10
already established that no CUDA-capable device exists here at all, so a
genuine kernel launch here would have nothing to launch onto. Both facts
are true at once and are not the same bug: tensor_sum_incomplete is broken
by a missing line of code that would need fixing on any machine, GPU or
not; tensor_sum_host's approach -- running the identical per-thread
arithmetic as an ordinary host loop -- is how this book verifies the
algorithm anyway, regardless of which of those two problems is the one
actually blocking a given run.
```

Reduction is also the identical primitive Part 7's bond-pricing chapter uses to total a portfolio's present value — the same "sum a buffer of floats via pairwise halving" operation that sums squared errors for a loss function also sums discounted cash flows for a bond book, one pairwise round at a time.

## 14.2 Min/Max Operations `[FOUNDATIONAL]`

### Intuition

Min and max reductions run the identical tree pattern Section 14.1 established, with the combine step swapped from addition to a comparison — but a comparison-based reduction needs to remember more than the winning *value*, because a backward pass needs to know exactly which input position produced it. `d(max(x))/dx_i` is `1` for the single index that held the maximum and `0` everywhere else — a gradient that flows through exactly one position, not a share distributed across all of them the way sum's gradient does.

### Background

```rust
// Host mirror of one round's exact per-thread logic. No else branch for the
// "no work this round" case -- Section 14.2's COMMON TRAP: an over-launched
// tid leaves out_val[tid]/out_idx[tid] completely untouched by this call,
// unlike sum_reduce_round_host's explicit zero-fill.
fn max_reduce_round_host(out_val: &mut [f32], out_idx: &mut [i32], in_val: &[f32], in_idx: &[i32], size: usize) {
    let next_size = (size + 1) / 2;
    for tid in 0..next_size {
        if tid * 2 + 1 < size {
            let left = in_val[tid * 2];
            let right = in_val[tid * 2 + 1];
            if left >= right {
                out_val[tid] = left;
                out_idx[tid] = in_idx[tid * 2];
            } else {
                out_val[tid] = right;
                out_idx[tid] = in_idx[tid * 2 + 1];
            }
        } else if tid * 2 < size {
            out_val[tid] = in_val[tid * 2];
            out_idx[tid] = in_idx[tid * 2];
        }
        // no else: an over-launched tid leaves out_val[tid]/out_idx[tid]
        // completely unwritten -- this section's COMMON TRAP.
    }
}
```

Every round now carries two parallel buffers instead of one: `out_val`/`in_val` hold the running maximum, and `out_idx`/`in_idx` hold *which original position* that value came from — seeded, on the very first round, with `in_idx[i] = i` for every `i`, and thereafter simply forwarding whichever index won each comparison.

### Worked Example 14.2.1 — Tracking both the value and its origin

`[3, 7, 2, 9]`, with `in_idx = [0, 1, 2, 3]` seeding the first round. Compiled and run:

```bash
rustc --edition 2024 -O 02_minmax_reduction.rs -o 02_minmax_reduction
./02_minmax_reduction
```

Genuinely compiled and run:

```
argmax([3,7,2,9]) = value 9.0 at index 3 (expect 9 at index 3)
```

Round `1`: thread `0` compares `left=3` (`in_idx[0]=0`) against `right=7` (`in_idx[1]=1`) — `right` wins: `out_val[0]=7`, `out_idx[0]=1`. Thread `1` compares `left=2` (`in_idx[2]=2`) against `right=9` (`in_idx[3]=3`) — `9` wins: `out_val[1]=9`, `out_idx[1]=3`. Round `1` output: values `[7, 9]`, indices `[1, 3]`. Round `2`: thread `0` compares `7` (index `1`) against `9` (index `3`) — `9` wins again: final value `9`, final index `3`. The maximum, `9`, and the position it came from, index `3` in the *original* array, survive together through every round — exactly the pair a later `argmax`-based backward pass needs, since it has to route a gradient back to index `3` specifically and nowhere else.

### Worked Example 14.2.2 — An odd length, and the branch that goes silent

`[5, 2, 9, 1, 7]` with `in_idx = [0,1,2,3,4]`. Genuinely computed in the same run:

```
argmax([5,2,9,1,7]) = value 9.0 at index 2 (expect 9 at index 2)
```

Round `0`, `next_size = 3`: thread `0` compares `5` vs `2` → `5` wins, index `0`. Thread `1` compares `9` vs `1` → `9` wins, index `2`. Thread `2`: `tid*2=4 < 5` but `tid*2+1=5` is not — the `else if` fires, passing `input[4]=7` and `in_idx[4]=4` straight through unchanged. Round `0` output: values `[5, 9, 7]`, indices `[0, 2, 4]`. Round `1`, `next_size=2`: thread `0` compares `5` vs `9` → `9` wins, index `2`. Thread `1`: `tid*2=2 < 3` but `tid*2+1=3` is not — leftover branch, passing `7`/index `4` through. Round `1` output: values `[9, 7]`, indices `[2, 4]`. Round `2`: thread `0` compares `9` vs `7` → `9` wins, final index `2`. The maximum is `9`, at original position `2` — matching a direct scan of `[5, 2, 9, 1, 7]` by eye, and matching the genuinely executed result above.

### Worked Example 14.2.3 — The COMMON TRAP, genuinely demonstrated with stale sentinels

Unlike `sum_reduce_round_host`, `max_reduce_round_host` never needed `unsafe` or uninitialized memory to demonstrate its own trap — the bug is a plain, safe-Rust-representable one: an output buffer that gets reused across rounds (the same ping-pong `scratch_val`/`scratch_idx` pattern `tensor_argmax_host` uses) but isn't fully overwritten by an over-launched round simply keeps whatever was already sitting in the slots the round's `next_size` never reached. Genuinely computed:

```
--- COMMON TRAP: over-launching a round leaves some slots unwritten ---
out_val before the call (buffer as reused from a previous round): [111.0, 222.0, 999.0]
out_idx before the call: [-1, -1, -777]
out_val after the (over-launched) call: [6.0, 8.0, 999.0]
out_idx after the (over-launched) call: [0, 2, -777]
out_val[2] is still the stale 999.0 sentinel, out_idx[2] is still the stale -777 sentinel: true
```

`d = [6, 2, 8]` has `size = 3`, so a correct driver would launch `next_size = (3+1)/2 = 2` threads: `tid = 0` (comparing `6` vs `2`) and `tid = 1` (the leftover branch, passing `8` through). This demonstration deliberately over-launches — looping `tid` over `0..3` instead of `0..2`, mirroring a driver bug that (incorrectly) reuses the *current* round's size as next round's thread count. `tid = 2` satisfies neither `tid*2+1 < 3` (`5 < 3` is false) nor `tid*2 < 3` (`4 < 3` is false), so the loop body's `if`/`else if` both fail to match, and — with no `else` arm at all — `out_val[2]` and `out_idx[2]` are never assigned. `out_val` and `out_idx` were deliberately pre-seeded with sentinel values (`999.0` and `-777`) standing in for whatever a real previous round left behind, and the genuinely-executed output confirms those exact two sentinels survive the call completely untouched, sitting right alongside two genuinely-computed, correct results (`out_val[0]=6.0`, `out_val[1]=8.0`) — a silently stale value hiding in plain sight next to real ones, exactly the trap the CUDA edition describes, reproduced here in ordinary safe Rust with no memory-safety violation at all: the bug is entirely in the driver's logic (which count to loop over), not in anything Rust's own guarantees could have caught.

```
[COMMON TRAP]  No else branch means unwritten, not zero-filled

Compare max_reduce_round_host's structure to sum_reduce_round_host's
directly: sum_reduce_round_host has three branches -- pair, leftover, and
an explicit else that zero-fills any tid with no real work. max_reduce_
round_host has only the first two. A tid that satisfies neither
tid*2+1 < size nor tid*2 < size -- which happens whenever a round is
called with more tids than ceil(size/2) actually calls for -- writes
nothing to out_val[tid] or out_idx[tid] at all. Nothing panics, because
the slice is still perfectly valid Rust memory; the position simply keeps
whatever value was already sitting there, silently, whether that's a
stale result left over from a previous round's reuse of the same buffer
or (only if the buffer itself were built with the same unsafe uninitialized
trick Section 14.1 used) genuinely uninitialized memory. A caller who
assumes max_reduce_round_host's output buffer is fully, freshly written
every round -- the way sum_reduce_round_host's explicit else branch
guarantees -- can end up folding a leftover value from round N-2 into
round N's comparison, silently, with no signal that anything went wrong.
```

The `argmax` this produces feeds directly into classification metrics and into the sparse gradient a later backward pass scatters back through only the winning path, leaving every other position's gradient at exactly zero.

## 14.3 Norm Calculations `[FOUNDATIONAL]`

### Intuition

The L2 norm collapses an entire vector into one number measuring its overall size — useful both as a loss function in its own right (regression error is often exactly this) and as a safety valve during training: if a single gradient vector grows unexpectedly large, rescaling it by its own norm can shrink it back to a sane magnitude before it ever reaches an optimizer step.

### Background

```rust
fn l2_norm(input: &[f32]) -> f32 {
    let squared: Vec<f32> = input.iter().map(|x| x * x).collect();
    let sum_sq = tensor_sum_host(&squared);
    sum_sq.sqrt()
}

fn clip_grad_norm(grad: &mut [f32], max_norm: f32) {
    let norm = l2_norm(grad);
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}
```

`l2_norm` squares every entry into a scratch buffer, reduces that buffer with `tensor_sum_host` (Section 14.1's genuinely-executable version, not the incomplete one), and takes one square root of the total. `clip_grad_norm` only ever *shrinks* a gradient, never grows one: when the norm already fits under `max_norm`, the function does nothing at all. `grad.iter_mut()` borrows `grad` mutably to rescale it in place — the same "write through the caller's own buffer" behavior as the C++ edition's `float* grad`, but with the borrow checker (rather than programmer discipline) guaranteeing nothing else can read or write `grad` while this loop holds its mutable borrow.

### Worked Example 14.3.1 — The 3-4-5 triangle

`[3, 4]`. Compiled and run:

```bash
rustc --edition 2024 -O 03_norm_calculations.rs -o 03_norm_calculations
./03_norm_calculations
```

Genuinely compiled and run:

```
l2_norm([3,4]) = 5.0 (the 3-4-5 triangle)
```

Squares are `[9, 16]`, sum is `25`, square root is `5` — the familiar 3-4-5 right triangle, and a useful sanity check that `l2_norm` is wired correctly before trusting it on a real, high-dimensional gradient vector.

### Worked Example 14.3.2 — Clipping a gradient back to a safe magnitude

Take that same `[3, 4]` vector as a gradient with `max_norm = 2.0`. Genuinely computed in the same run:

```
clip_grad_norm([3,4], max_norm=2.0) -> [1.2, 1.6]
clipped norm = 2.0000 (expect exactly 2.0)
```

`l2_norm` returns `5.0`, which exceeds `2.0`, so clipping fires: `scale = 2.0 / 5.0 = 0.4`. Rescaling every entry: `[3 × 0.4, 4 × 0.4] = [1.2, 1.6]`. Checking the result's own norm confirms the clip worked exactly as intended: `√(1.2² + 1.6²) = √(1.44 + 2.56) = √4.0 = 2.0` — landing precisely on `max_norm`, not merely under it, because clipping always rescales to *exactly* the limit rather than to some smaller, arbitrary value. Without this safety valve, one unusually large gradient — from a numerically unstable loss spike, for instance — could otherwise take a single, enormous, destabilizing optimizer step.

## 14.4 Statistical Functions `[FOUNDATIONAL]`

### Intuition

Variance and standard deviation are sum-and-mean applied twice in sequence — first to find the mean itself, then again to find the mean of how far every value strays from that mean, squared. Placing this section last in the chapter is deliberate: there is no new reduction primitive here, only Section 14.1's `tensor_sum_host`/`tensor_mean_host` reused twice over.

### Background

```rust
fn tensor_variance(input: &[f32]) -> f32 {
    let mean = tensor_mean_host(input);
    let sq_dev: Vec<f32> = input.iter().map(|x| (x - mean) * (x - mean)).collect();
    tensor_mean_host(&sq_dev)
}

fn tensor_std(input: &[f32]) -> f32 {
    tensor_variance(input).sqrt()
}
```

This is the numerically safer of the two common variance formulas: computing the mean first and then the mean of *actual* squared deviations from it, rather than the single-pass shortcut `mean(x²) − mean(x)²`, which can lose precision to catastrophic cancellation when two large, nearly-equal numbers are subtracted. `tensor_variance` pays for that safety with a second full pass over the data (a second `Vec` allocation via `.collect()`), which is a fine trade for a quantity computed once at the end of an epoch, not once per training step.

### Worked Example 14.4.1 — Variance and standard deviation on eight values

`[2, 4, 4, 4, 5, 5, 7, 9]` — a textbook example precisely because its answer comes out round. Compiled and run:

```bash
rustc --edition 2024 -O 04_statistical_functions.rs -o 04_statistical_functions
./04_statistical_functions
```

Genuinely compiled and run:

```
mean = 5.0
variance = 4.0
std = 2.0
```

Mean: `(2+4+4+4+5+5+7+9)/8 = 40/8 = 5`. Squared deviations from that mean: `(2-5)²=9`, `(4-5)²=1` (three times), `(5-5)²=0` (twice), `(7-5)²=4`, `(9-5)²=16` — giving `[9, 1, 1, 1, 0, 0, 4, 16]`. Mean of those: `(9+1+1+1+0+0+4+16)/8 = 32/8 = 4` — the variance, matching the genuinely computed value exactly. Standard deviation is `√4 = 2`.

Batch normalization layers in a later neural-network chapter call exactly this pair, per-feature, to normalize activations before every hidden layer, and Part 7's risk analytics call the same pair across a bond portfolio's yields to size a Value-at-Risk estimate.

## 14.5 Complete Runnable Code

### File: `01_sum_mean_reduction.rs`

```rust
// One round of pairwise reduction: size elements -> ceil(size/2). A genuine
// GPU kernel would take this exact per-thread shape; this sandbox has no
// device to launch one on at all (Chapter 10), which is exactly why a host
// mirror -- run for every simulated tid on the CPU -- stands in for it here.
fn sum_reduce_round_host(output: &mut [f32], input: &[f32], size: usize) {
    let next_size = (size + 1) / 2;
    for tid in 0..next_size {
        if tid * 2 + 1 < size {
            output[tid] = input[tid * 2] + input[tid * 2 + 1];
        } else if tid * 2 < size {
            output[tid] = input[tid * 2]; // odd leftover, pass through
        } else {
            output[tid] = 0.0; // no work this round: zero-filled
        }
    }
}

fn tensor_sum_host(input: &[f32]) -> f32 {
    let size = input.len();
    let mut current: Vec<f32> = input.to_vec();
    let mut scratch: Vec<f32> = vec![0.0; size];
    let mut current_size = size;
    while current_size > 1 {
        let next_size = (current_size + 1) / 2;
        sum_reduce_round_host(&mut scratch, &current, current_size);
        std::mem::swap(&mut current, &mut scratch);
        current_size = next_size;
    }
    current[0]
}

fn tensor_mean_host(input: &[f32]) -> f32 {
    tensor_sum_host(input) / input.len() as f32
}

// Ported faithfully, bug and all: the per-round launch is a comment, not
// code -- exactly the gap Section 14.1's COMMON TRAP describes. This
// compiles cleanly. Note the scratch buffer here is built the ordinary,
// idiomatic Rust way (`vec![0.0; size]`), which -- unlike C's malloc --
// genuinely, deterministically zero-initializes every element. See the
// chapter text and 01b_sum_incomplete_uninit.rs for what changes when that
// idiomatic choice is deliberately replaced with C-style uninitialized
// memory instead.
fn tensor_sum_incomplete(input: &[f32]) -> f32 {
    let size = input.len();
    let mut current: &[f32] = input;
    let scratch: Vec<f32> = vec![0.0; size]; // zero-initialized, unlike malloc
    let mut current_size = size;
    while current_size > 1 {
        let next_size = (current_size + 1) / 2;
        // TODO: launch sum_reduce_kernel with current_size threads,
        // writing into scratch
        current = &scratch;
        current_size = next_size;
    }
    current[0]
}

fn main() {
    println!("=== Section 14.1: sum/mean reduction, tree pattern ===\n");

    // Worked Example 14.1.1
    let a = [1.0f32, 4.0, 9.0, 16.0];
    let sum_a = tensor_sum_host(&a);
    let mean_a = tensor_mean_host(&a);
    println!("tensor_sum_host([1,4,9,16]) = {:.1}", sum_a);
    println!("tensor_mean_host([1,4,9,16]) = {:.1}", mean_a);

    // Worked Example 14.1.2
    let b = [2.0f32, 6.0, 3.0, 8.0, 5.0];
    let sum_b = tensor_sum_host(&b);
    println!("tensor_sum_host([2,6,3,8,5]) = {:.1} (expect 24)", sum_b);

    // Worked Example 14.1.3 / COMMON TRAP: the incomplete driver, run
    // genuinely to show it's broken -- but, unlike the CUDA edition,
    // deterministically so, since `vec![0.0; size]` always zero-fills.
    println!("\n--- COMMON TRAP: tensor_sum_incomplete's launch is a comment ---");
    let result_incomplete = tensor_sum_incomplete(&a);
    println!(
        "tensor_sum_incomplete([1,4,9,16]) = {} (idiomatic Rust: deterministic 0.0, NOT 30, and NOT random garbage)",
        result_incomplete
    );
}
```

```bash
rustc --edition 2024 -O 01_sum_mean_reduction.rs -o 01_sum_mean_reduction
./01_sum_mean_reduction
```

### File: `01b_sum_incomplete_uninit.rs` — the Rust-specific companion from Worked Example 14.1.3

```rust
// Deliberately kept as its own binary, outside the "Complete Runnable Code"
// set: this file exists purely to show what it actually takes, in Rust, to
// reproduce the CUDA edition's nondeterministic-garbage version of
// tensor_sum_incomplete's bug. 01_sum_mean_reduction.rs's idiomatic port
// used `vec![0.0; size]`, which deterministically zero-fills -- Rust's safe
// Vec allocation simply doesn't have an uninitialized-memory mode. Getting
// C's malloc-style "reserve space, initialize nothing" behavior back
// requires reaching for Vec::with_capacity plus an unsafe set_len call that
// tells the compiler the buffer is initialized when it is not -- and once
// that promise is made, reading the buffer is genuinely undefined behavior
// by Rust's own rules, exactly as reading a malloc'd-not-yet-written buffer
// is undefined behavior in C. It usually "works" in the sense of not
// crashing (f32 has almost no invalid bit patterns), but it is not sound
// code, and it is written this way on purpose, once, to make the point.
fn tensor_sum_incomplete_uninit(input: &[f32]) -> f32 {
    let size = input.len();
    let mut current: &[f32] = input;
    let scratch: Vec<f32> = {
        let mut v: Vec<f32> = Vec::with_capacity(size);
        unsafe {
            v.set_len(size); // UNSAFE: claims `size` initialized elements that were never written
        }
        v
    };
    let mut current_size = size;
    while current_size > 1 {
        let next_size = (current_size + 1) / 2;
        // TODO: launch sum_reduce_kernel with current_size threads,
        // writing into scratch
        current = &scratch;
        current_size = next_size;
    }
    current[0]
}

fn main() {
    println!("=== Rust-specific companion to Worked Example 14.1.3: reproducing malloc-style garbage ===");
    let a = [1.0f32, 4.0, 9.0, 16.0];
    let result = tensor_sum_incomplete_uninit(&a);
    println!("tensor_sum_incomplete_uninit([1,4,9,16]) = {} (uninitialized memory via unsafe set_len, NOT 30)", result);
}
```

```bash
rustc --edition 2024 -O 01b_sum_incomplete_uninit.rs -o 01b_sum_incomplete_uninit
./01b_sum_incomplete_uninit
```

### File: `02_minmax_reduction.rs`

```rust
// Host mirror of one round's exact per-thread logic. No else branch for the
// "no work this round" case -- Section 14.2's COMMON TRAP: an over-launched
// tid leaves out_val[tid]/out_idx[tid] completely untouched by this call,
// unlike sum_reduce_round_host's explicit zero-fill.
fn max_reduce_round_host(out_val: &mut [f32], out_idx: &mut [i32], in_val: &[f32], in_idx: &[i32], size: usize) {
    let next_size = (size + 1) / 2;
    for tid in 0..next_size {
        if tid * 2 + 1 < size {
            let left = in_val[tid * 2];
            let right = in_val[tid * 2 + 1];
            if left >= right {
                out_val[tid] = left;
                out_idx[tid] = in_idx[tid * 2];
            } else {
                out_val[tid] = right;
                out_idx[tid] = in_idx[tid * 2 + 1];
            }
        } else if tid * 2 < size {
            out_val[tid] = in_val[tid * 2];
            out_idx[tid] = in_idx[tid * 2];
        }
        // no else: an over-launched tid leaves out_val[tid]/out_idx[tid]
        // completely unwritten -- this section's COMMON TRAP.
    }
}

fn tensor_argmax_host(input: &[f32]) -> (f32, i32) {
    let size = input.len();
    let mut cur_val: Vec<f32> = input.to_vec();
    let mut cur_idx: Vec<i32> = (0..size as i32).collect();
    let mut scratch_val: Vec<f32> = vec![0.0; size];
    let mut scratch_idx: Vec<i32> = vec![0; size];
    let mut current_size = size;
    while current_size > 1 {
        let next_size = (current_size + 1) / 2;
        max_reduce_round_host(&mut scratch_val, &mut scratch_idx, &cur_val, &cur_idx, current_size);
        std::mem::swap(&mut cur_val, &mut scratch_val);
        std::mem::swap(&mut cur_idx, &mut scratch_idx);
        current_size = next_size;
    }
    (cur_val[0], cur_idx[0])
}

fn main() {
    println!("=== Section 14.2: min/max reduction, value carried with its origin ===\n");

    let a = [3.0f32, 7.0, 2.0, 9.0];
    let (max_a, idx_a) = tensor_argmax_host(&a);
    println!("argmax([3,7,2,9]) = value {:.1} at index {} (expect 9 at index 3)", max_a, idx_a);

    let b = [5.0f32, 2.0, 9.0, 1.0, 7.0];
    let (max_b, idx_b) = tensor_argmax_host(&b);
    println!("argmax([5,2,9,1,7]) = value {:.1} at index {} (expect 9 at index 2)", max_b, idx_b);

    // Self-check 2 preview: tie-breaking with left >= right
    let c = [4.0f32, 12.0, 7.0, 12.0, 3.0];
    let (max_c, idx_c) = tensor_argmax_host(&c);
    println!("argmax([4,12,7,12,3]) (tie at value 12) = value {:.1} at index {}", max_c, idx_c);

    // Worked Example 14.2.3: the COMMON TRAP, genuinely demonstrated -- an
    // over-launched round (called with more tids than next_size actually
    // calls for) leaves the extra output slots holding whatever was already
    // in that reused buffer, not a fresh, defined value.
    println!("\n--- COMMON TRAP: over-launching a round leaves some slots unwritten ---");
    let d = [6.0f32, 2.0, 8.0]; // size 3, next_size = 2
    let mut out_val = [111.0f32, 222.0, 999.0]; // 999.0 is a stale sentinel sitting in slot 2
    let mut out_idx = [-1i32, -1, -777]; // -777 is the matching stale sentinel index
    let in_idx = [0i32, 1, 2];
    println!("out_val before the call (buffer as reused from a previous round): {:?}", out_val);
    println!("out_idx before the call: {:?}", out_idx);
    // Genuinely over-launch: loop the round's OWN thread count (`size`, 3)
    // instead of the correct `next_size` (2) a non-buggy driver would pass.
    let over_launched_threads = d.len(); // BUG: should be (d.len()+1)/2 = 2
    for tid in 0..over_launched_threads {
        if tid * 2 + 1 < d.len() {
            let left = d[tid * 2];
            let right = d[tid * 2 + 1];
            if left >= right {
                out_val[tid] = left;
                out_idx[tid] = in_idx[tid * 2];
            } else {
                out_val[tid] = right;
                out_idx[tid] = in_idx[tid * 2 + 1];
            }
        } else if tid * 2 < d.len() {
            out_val[tid] = d[tid * 2];
            out_idx[tid] = in_idx[tid * 2];
        }
        // no else -- tid=2 here satisfies neither branch (2*2+1=5 not < 3,
        // 2*2=4 not < 3), so out_val[2]/out_idx[2] are never touched.
    }
    println!("out_val after the (over-launched) call: {:?}", out_val);
    println!("out_idx after the (over-launched) call: {:?}", out_idx);
    println!(
        "out_val[2] is still the stale 999.0 sentinel, out_idx[2] is still the stale -777 sentinel: {}",
        out_val[2] == 999.0 && out_idx[2] == -777
    );
}
```

```bash
rustc --edition 2024 -O 02_minmax_reduction.rs -o 02_minmax_reduction
./02_minmax_reduction
```

### File: `03_norm_calculations.rs`

```rust
fn sum_reduce_round_host(output: &mut [f32], input: &[f32], size: usize) {
    let next_size = (size + 1) / 2;
    for tid in 0..next_size {
        if tid * 2 + 1 < size {
            output[tid] = input[tid * 2] + input[tid * 2 + 1];
        } else if tid * 2 < size {
            output[tid] = input[tid * 2];
        } else {
            output[tid] = 0.0;
        }
    }
}

fn tensor_sum_host(input: &[f32]) -> f32 {
    let size = input.len();
    let mut current: Vec<f32> = input.to_vec();
    let mut scratch: Vec<f32> = vec![0.0; size];
    let mut current_size = size;
    while current_size > 1 {
        let next_size = (current_size + 1) / 2;
        sum_reduce_round_host(&mut scratch, &current, current_size);
        std::mem::swap(&mut current, &mut scratch);
        current_size = next_size;
    }
    current[0]
}

fn l2_norm(input: &[f32]) -> f32 {
    let squared: Vec<f32> = input.iter().map(|x| x * x).collect();
    let sum_sq = tensor_sum_host(&squared);
    sum_sq.sqrt()
}

fn clip_grad_norm(grad: &mut [f32], max_norm: f32) {
    let norm = l2_norm(grad);
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}

fn main() {
    println!("=== Section 14.3: L2 norm and gradient clipping ===\n");

    let v = [3.0f32, 4.0];
    println!("l2_norm([3,4]) = {:.1} (the 3-4-5 triangle)", l2_norm(&v));

    let mut grad = [3.0f32, 4.0];
    clip_grad_norm(&mut grad, 2.0);
    println!("clip_grad_norm([3,4], max_norm=2.0) -> [{:.1}, {:.1}]", grad[0], grad[1]);
    println!("clipped norm = {:.4} (expect exactly 2.0)", l2_norm(&grad));
}
```

```bash
rustc --edition 2024 -O 03_norm_calculations.rs -o 03_norm_calculations
./03_norm_calculations
```

### File: `04_statistical_functions.rs`

```rust
fn sum_reduce_round_host(output: &mut [f32], input: &[f32], size: usize) {
    let next_size = (size + 1) / 2;
    for tid in 0..next_size {
        if tid * 2 + 1 < size {
            output[tid] = input[tid * 2] + input[tid * 2 + 1];
        } else if tid * 2 < size {
            output[tid] = input[tid * 2];
        } else {
            output[tid] = 0.0;
        }
    }
}

fn tensor_sum_host(input: &[f32]) -> f32 {
    let size = input.len();
    let mut current: Vec<f32> = input.to_vec();
    let mut scratch: Vec<f32> = vec![0.0; size];
    let mut current_size = size;
    while current_size > 1 {
        let next_size = (current_size + 1) / 2;
        sum_reduce_round_host(&mut scratch, &current, current_size);
        std::mem::swap(&mut current, &mut scratch);
        current_size = next_size;
    }
    current[0]
}

fn tensor_mean_host(input: &[f32]) -> f32 {
    tensor_sum_host(input) / input.len() as f32
}

fn tensor_variance(input: &[f32]) -> f32 {
    let mean = tensor_mean_host(input);
    let sq_dev: Vec<f32> = input.iter().map(|x| (x - mean) * (x - mean)).collect();
    tensor_mean_host(&sq_dev)
}

fn tensor_std(input: &[f32]) -> f32 {
    tensor_variance(input).sqrt()
}

fn main() {
    println!("=== Section 14.4: variance and standard deviation, sum-and-mean twice ===");
    let data = [2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    println!("mean = {:.1}", tensor_mean_host(&data));
    println!("variance = {:.1}", tensor_variance(&data));
    println!("std = {:.1}", tensor_std(&data));
}
```

```bash
rustc --edition 2024 -O 04_statistical_functions.rs -o 04_statistical_functions
./04_statistical_functions
```

## Chapter Summary

Every reduction in this chapter shares the same tree shape Section 14.1 established: pair up values, combine each pair, halve the array, repeat until one value remains — a discipline that exists because letting many threads write into one shared accumulator is a race condition, and because floating-point addition isn't perfectly associative, so *how* values get combined can change the answer, not just how fast it arrives. `sum_reduce_round_host`'s per-round logic is correct and was genuinely verified on both an even-length array (`[1,4,9,16] → 30`) and an odd-length one (`[2,6,3,8,5] → 24`, across three full rounds) through a host-executable mirror of its exact per-thread arithmetic — necessary because no GPU exists in this sandbox to launch a real kernel on (Chapter 10). A faithfully-ported driver with the per-round kernel launch left as a comment was genuinely compiled and run, and here the Rust port produced a genuinely different *kind* of wrong answer than its CUDA counterpart: idiomatic `vec![0.0; size]` zero-initializes deterministically, so `tensor_sum_incomplete` returns exactly `0.0` on every run, while a second binary built with `unsafe` `Vec::with_capacity`/`set_len` — deliberately reproducing C's uninitialized-memory behavior — returned genuinely varying garbage across runs, just as the CUDA edition's does. `max_reduce_round_host` extends the same pattern with a parallel index buffer, so the position of the maximum survives every round alongside its value — precisely what a sparse backward pass needs to route a gradient through one winning index and zero everywhere else — though unlike `sum_reduce_round_host`, it has no `else` branch, so an over-launched round leaves some output positions silently unwritten rather than zero-filled, genuinely demonstrated here with planted sentinel values that survive a call completely untouched, no `unsafe` required. The L2 norm reduces a vector to one measure of size via sum-of-squares-then-square-root, checked against the exact 3-4-5 triangle, and gradient-norm clipping uses that same norm to rescale an overlarge gradient down to precisely `max_norm`, never below it. Variance and standard deviation close the chapter by reusing sum-and-mean twice over — once for the mean, once for the mean of squared deviations from it — the numerically safer two-pass approach, verified end to end on an eight-value dataset down to an exact variance of `4` and standard deviation of `2`.

## Self-Check Questions

1. Trace `sum_reduce_round_host`'s per-round logic over `[10, 20, 30, 40, 50]` (five elements) round by round the way Worked Example 14.1.2 traced `[2,6,3,8,5]`, and report the final sum along with a direct check against adding all five values.
2. Trace `max_reduce_round_host`'s per-round logic over `[4, 12, 7, 12, 3]` (note the tie at value `12`) with `in_idx = [0,1,2,3,4]`. Which original index does the final result report, and why does the `left >= right` comparison (rather than a strict `>`) determine the answer in the case of a tie?
3. A gradient vector `[6, 8]` is passed to `clip_grad_norm` with `max_norm = 3.0`. What does `l2_norm` return, what is `scale`, and what is the clipped vector? Verify the clipped vector's own norm equals `max_norm` exactly.
4. Compute the variance and standard deviation of `[1, 1, 1, 1, 5, 5, 5, 5]` by hand, following the same two-pass method Worked Example 14.4.1 used (mean first, then mean of squared deviations from it).
5. A colleague reads `tensor_sum_incomplete` and assumes it must return `0.0` on any platform, since that's what it returned when they ran it. What specifically about how `scratch` is allocated would you need to point them to in order to explain why that assumption is platform/implementation-specific rather than a property of the algorithm itself — and what would `01b_sum_incomplete_uninit.rs`'s existence tell them about how to get a genuinely different (and genuinely worse) answer out of the identical missing-kernel-launch bug?

## Where We Go Next

Chapter 15 turns to *recording* the compositions this chapter and the two before it made possible — element-wise, matrix, and reduction operations now cover every arithmetic primitive the rest of the book composes — as a graph, so the framework knows how to run any of them backward.

## Worked Solutions

**1.** Round `0` (`current_size=5`, `next_size=3`): thread `0`: `10+20=30`. Thread `1`: `30+40=70`. Thread `2`: leftover, passes `50` through. Round `0` output: `[30, 70, 50]`. Round `1` (`current_size=3`, `next_size=2`): thread `0`: `30+70=100`. Thread `1`: leftover, passes `50` through. Round `1` output: `[100, 50]`. Round `2` (`next_size=1`): thread `0`: `100+50=150`. Final sum: `150`, genuinely confirmed by compiling and running this exact trace. Direct check: `10+20+30+40+50=150` — matches exactly.

**2.** Round `0`: thread `0` compares `4` (index `0`) vs `12` (index `1`) — `12` wins, index `1`. Thread `1` compares `7` (index `2`) vs `12` (index `3`) — `12` wins, index `3`. Thread `2`: leftover, passes `3`/index `4` through. Round `0`: values `[12, 12, 3]`, indices `[1, 3, 4]`. Round `1`: thread `0` compares `12` (index `1`) vs `12` (index `3`) — a genuine tie. `left >= right` evaluates `12 >= 12`, which is `true`, so `left` wins: the result is index `1`, not index `3`. Thread `1`: leftover, passes `3`/index `4` through. Round `1`: values `[12, 3]`, indices `[1, 4]`. Round `2`: `12` (index `1`) beats `3` — final index `1`. The `>=` (rather than strict `>`) means ties are broken in favor of the *earlier*-encountered operand in each individual comparison — here, the value that started at index `1` — not the later one, exactly matching the genuinely computed `argmax([4,12,7,12,3])` result above.

**3.** `l2_norm([6,8]) = √(36+64) = √100 = 10.0`. Since `10.0 > 3.0`, clipping fires: `scale = 3.0/10.0 = 0.3`. Clipped vector: `[6×0.3, 8×0.3] = [1.8, 2.4]`. Verification: `√(1.8² + 2.4²) = √(3.24 + 5.76) = √9.0 = 3.0` — exactly `max_norm`, all genuinely confirmed by compiling and running this exact trace.

**4.** Mean: `(1+1+1+1+5+5+5+5)/8 = 24/8 = 3`. Squared deviations: `(1-3)²=4` (four times), `(5-3)²=4` (four times) — every single deviation is `4`. Mean of those: `32/8 = 4` — the variance. Standard deviation: `√4 = 2`.

**5.** The colleague's platform-specific assumption traces directly to `let scratch: Vec<f32> = vec![0.0; size];` inside `tensor_sum_incomplete` — `vec![0.0; size]` is a request for `size` elements each explicitly initialized to `0.0`, a guarantee Rust's standard library actually makes, which is why the result is `0.0` on every run, every platform, forever, as long as that exact allocation is used. That guarantee is a property of *how the buffer was built*, not of the reduction algorithm — the missing kernel launch is identical either way. `01b_sum_incomplete_uninit.rs` shows what changes if that guarantee is deliberately given up: replacing `vec![0.0; size]` with `Vec::with_capacity(size)` plus an `unsafe { v.set_len(size) }` call reproduces C's actual "reserve but don't initialize" behavior, and reading the result back is genuinely undefined behavior in Rust's own terms — which is exactly why it returns different leftover bits on different runs, the same nondeterministic-garbage behavior the CUDA edition's `malloc`-based version shows, rather than the safe version's reliable `0.0`.
