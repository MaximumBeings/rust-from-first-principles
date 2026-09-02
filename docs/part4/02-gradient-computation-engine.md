# Chapter 17: Gradient Computation Engine — Running Every Backward Rule to Completion

> "Every rule Chapter 16 derived by hand — `AddOp` passes a gradient through, `MulOp` scales it by the other operand, `MatMulOp` transposes and re-multiplies — is inert until something actually walks the graph and calls each one in the right order, on the right numbers, adding up whatever needs adding. This chapter is that something."

**What you will understand by the end of this chapter:**

- The full backward pass for `w = x*y + x`, traced start to finish as one table — and a precise, genuinely reproduced gap between `GraphNode::grad` (Chapter 15.2) and the tensor-level `.grad` state `accumulate_gradient` actually writes to, plus the exact field, already built in Chapter 15.3, that closes it
- A second gap this chapter's own port uncovers on top of that one — and a genuine correction of how the CUDA/Mojo edition frames it: it is not a C++-specific quirk. `ScalarTensor` here derives `Copy` (Chapter 15's explicit scoping choice), so Rust bitwise-copies the whole struct into `GraphNode::inputs` exactly the way C++'s copy-by-value `ScalarTensor` does — the same gap, reproduced with the same numbers, for the same structural reason, in a language the CUDA edition's own prose claims doesn't have to confront it
- Why `accumulate_gradient` branches on "does this tensor have a gradient yet" rather than always adding — and the definitive answer, finally, to the buffer-aliasing question Chapter 16 raised about `AddOp::backward` returning one value to two different callers, plus a genuinely stronger answer than the CUDA edition's own: `Rc<T>`'s reference counting doesn't just make the CUDA edition's *third* trap (freeing a still-aliased buffer) easy to avoid by discipline, it makes the naive mistake impossible to even attempt in safe code — proven by compiling and running the exact call that would have to succeed for the bug to exist, and watching it return `None` instead
- A gap this chapter's own `accumulate_gradient` doesn't handle at all: what has to happen when an operand was broadcast during forward (Chapter 12.4), traced with the exact numbers that section already established
- Why this framework's computation graph is single-use — discarded after every backward pass and rebuilt from scratch on the next forward call — and how an arena-style bump allocator makes that a genuinely free design choice rather than a wasteful one
- Which saved inputs a node can safely drop the instant forward finishes, and which it must keep alive until backward actually visits it — a distinction that turns into real, countable memory on anything larger than a two-node example

**What you need to know first:**

- Chapter 16 (every backward rule this chapter's traversal actually calls: `AddOp`, `MulOp` — and the open aliasing question this chapter resolves)
- Chapter 15.2 and 15.3 (`GraphNode`'s `grad` field and `output.grad_fn_index` — both turn out to be exactly the machinery this chapter's central finding depends on)
- Rust's `Rc<T>` (reference-counted shared ownership) and `Option<T>` — this chapter's `Rc::get_mut` demonstration is the direct payoff of understanding what reference counting actually tracks

## The full backward pass, worked by hand, start to finish

Everything is now in place to run the running example — `w = x*y + x`, `x=3, y=4, z=12, w=15` — backward completely, one node at a time, using nothing but the rules Chapter 16 already derived. Walk it as a table, exactly the steps Section 17.1's code automates:

| Step | Node visited | Incoming gradient | Local rule (Chapter 16) | Result | Running totals |
|---|---|---|---|---|---|
| 0 | — (seed) | — | `∂w/∂w = 1` by definition | `w.grad = 1.0` | `w: 1.0` |
| 1 | `add(z, x) → w` | `w.grad = 1.0` | `AddOp::backward` passes the gradient through unchanged to both inputs | `z` gets `1.0`; `x` gets `1.0` | `z: 1.0`, `x: 1.0` |
| 2 | `mul(x, y) → z` | `z.grad = 1.0` | `MulOp::backward`: `x` gets `grad_z × y`, `y` gets `grad_z × x` | `x` gets `1.0 × 4 = 4.0`; `y` gets `1.0 × 3 = 3.0` | `x: 1.0 + 4.0 = 5.0`, `y: 3.0` |

Final answer: **`x.grad = 5.0`, `y.grad = 3.0`** — matching the calculus in Chapter 15 (`∂w/∂x = y+1 = 5`, `∂w/∂y = x = 3`) exactly, and arrived at without a single symbolic derivative, only local multiplications and one addition, applied mechanically in the reverse order Chapter 15.4 established. The one place a "sum" happened rather than a plain pass-through is `x` in Step 2, because `x` was used twice in the forward pass — once directly in the addition, once inside the multiply — precisely Section 16.1's "sum over paths" chain rule, now happening inside a traversal instead of on paper.

## 17.1 Reverse-mode AD Implementation `[FOUNDATIONAL]`

### Intuition

The worked table above is a hand simulation of one function, conventionally called `backward()`, that turns a scalar loss into gradients on every parameter that fed into it. Writing it correctly means getting three things right at once: seeding the very first gradient, visiting nodes in an order where every dependency is already resolved, and skipping nodes that genuinely have nothing to contribute.

### Background

Ported as literally as possible from Chapters 15 and 16's own machinery:

```rust
// backward_naive(), ported LITERALLY -- including a bug this section's
// COMMON TRAP identifies before this file ever fixes it.
fn backward_naive(registry: &OpRegistry, graph: &mut ComputationGraph, loss: &mut ScalarTensor) {
    // Seed: dL/dL = 1
    loss.grad = 1.0;
    loss.has_grad = true;

    let order = topological_backward_order(graph);
    for node_idx in order {
        let node = &graph.nodes[node_idx];
        if node.grad == 0.0 {
            continue; // this output was "never used downstream" -- or so this reads
        }
        let input_grads = chain_rule_step(registry, &node.op_name, node.grad, &node.inputs, &node.output);
        let n = node.inputs.len();
        for i in 0..n {
            accumulate_gradient_naive(&mut graph.nodes[node_idx].inputs[i], input_grads[i]);
        }
    }
}
```

`ScalarTensor` here gains a `has_grad`/`grad` pair on top of Chapter 15's fields — Rust's stand-in for Mojo's `Tensor.grad` being `.is_none()` before anything has ever been assigned to it.

### Worked Example 17.1.1 — Tracing the loop literally, and watching it fail

Compiled and run:

```bash
rustc --edition 2024 -O 01_naive_backward_bug.rs -o 01_naive_backward_bug
./01_naive_backward_bug
```

Genuinely compiled and run:

```
=== Section 17.1: backward(), ported literally -- including its own bug ===
w = 15.0, graph.nodes.len() = 2
graph.nodes[0].grad (mul node, at construction) = 0.0000
graph.nodes[1].grad (add node, at construction) = 0.0000

backward_naive: order = [1, 0]
  visiting graph.nodes[1] ("add"): node.grad = 0.0000 -> is_zero(), SKIPPING
  visiting graph.nodes[0] ("mul"): node.grad = 0.0000 -> is_zero(), SKIPPING

after backward_naive:
  w.has_grad = true, w.grad = 1.0000
  x.has_grad = false (x.grad never touched by this run)
  y.has_grad = false (y.grad never touched by this run)

expected from the hand-worked table: x.grad=5.0, y.grad=3.0 -- NOT what this run produced.
root cause: loss.grad was seeded on `w` (a ScalarTensor), but the loop reads
graph.nodes[node_idx].grad -- a SEPARATE field on GraphNode that this function
never writes to at all. graph.nodes[1].grad is still 0.0000 when the loop reaches it,
so the add node's backward is skipped, and neither x.grad nor y.grad is ever set.
```

This is a genuine, compiled, running reproduction of the bug — not a hypothetical, and not one the borrow checker has any reason to catch, since every line here type-checks: `loss.grad = 1.0` really does write a `f32` field, `graph.nodes[node_idx].grad` really does read a different `f32` field, and nothing about Rust's ownership rules distinguishes "the right field" from "a field." `loss.grad = 1.0` sets `w`'s own `.grad` field, a completely separate piece of storage from `graph.nodes[1].grad`, since `ScalarTensor` and `GraphNode` are separate structs each carrying their own `grad` member. The very first thing the loop does is check `graph.nodes[1].grad == 0.0`, which is `true`, and `continue`s — the add node's backward never runs, `accumulate_gradient` is never called for `z` or `x`, and the loop returns having touched nothing but `w` itself.

```
[COMMON TRAP]  node.grad is read every iteration, but nothing here ever writes it

Look at what backward_naive actually touches. It reads node.grad
(feeding chain_rule_step) and it calls accumulate_gradient_naive(...),
writing into each INPUT's own .grad field. loss.grad = 1.0 sets
loss's own .grad field too. Not one of these three writes ever
touches graph.nodes[node_idx].grad -- the separate field GraphNode
itself carries, zero-initialized in Chapter 15.2's constructor and
never assigned anywhere in this function.

The fix is a piece of machinery this book already built, two chapters
ago, and simply hasn't wired up yet: Chapter 15.3's
`out.grad_fn_index = self.nodes.len() as i32`, which exists
specifically so "the output remembers who made it." Seeding needs to
set `graph.nodes[loss.grad_fn_index].grad = loss.grad`, not just
loss.grad itself; and accumulate_gradient needs to mirror every
update into graph.nodes[tensor.grad_fn_index].grad as well as
tensor.grad, whenever tensor is itself the output of some earlier
node. Without it, grad_fn_index is a field this book built and never
actually used -- confirmed above, literally, by compiling and running
the code with the bug still in it.
```

### Worked Example 17.1.2 — Applying the fix, and finding a second one hiding behind it

Compiled and run:

```bash
rustc --edition 2024 -O 02_fixed_backward_and_accumulation.rs -o 02_fixed_backward_and_accumulation
./02_fixed_backward_and_accumulation
```

Genuinely compiled and run:

```
=== Section 17.1/17.2: fixed backward(), full trace for w = x*y + x ===

--- Part A: Section 17.1's grad_fn_index fix, applied literally ---
graph.nodes[1] ("add").grad = 1.0000, graph.nodes[0] ("mul").grad = 1.0000  (mirroring works)
but xA.grad (the ORIGINAL variable declared in main()) = 0.0000, has_grad = false
expected 5.0 -- MISMATCH. Cause: node.inputs[i] inside each GraphNode is an
INDEPENDENT COPY of xA (ScalarTensor derives Copy, Chapter 15 -- Rust bitwise-copies
the whole struct into the Vec, same as Chapter 15 flagged for a value-copying design).
accumulate_gradient_part_a writes into that COPY's .grad field, not into xA itself --
the mirroring fix genuinely works for graph.nodes[].grad, but is not enough on its own.

--- Part B: a grad_table keyed by tensor_id, closing the value-copy gap ---
xB.tensor_id = 4 (every copy of xB inside graphB.nodes carries this same id)
read_grad(graphB, xB) = 5.0000, read_grad(graphB, yB) = 3.0000
matches the hand-worked table: x.grad=5.0, y.grad=3.0 -- CONFIRMED

--- Worked Example 17.2.1: the two branches, walked explicitly ---
Step 1 (add node): xB's first contribution -- table has no entry yet, insert 1.0
Step 2 (mul node): xB's second contribution -- entry exists, add: 1.0 + 4.0 = 5.0
final read_grad(graphB, xB) = 5.0

--- zero_grad ---
before zero_grad: read_grad(xB)=5.0, read_grad(yB)=3.0
after zero_grad: read_grad(xB)=0.0, read_grad(yB)=0.0
```

Applying exactly the fix the previous section derived — mirroring the seed and every accumulation into `graph.nodes[tensor.grad_fn_index].grad` — genuinely does fix `graph.nodes[].grad` (Part A confirms both node grads read `1.0`, correctly). But `xA.grad`, the *original* `ScalarTensor` variable declared in `main()`, still reads `0.0`.

Here is where this Rust edition has to correct the CUDA/Mojo edition's own explanation rather than just translate it. The CUDA text frames this second gap as something C++-specific — "Mojo's `Tensor` wraps a shared `UnsafePointer`, so every 'copy' of a tensor still refers to the same underlying data, but Chapter 15 built `ScalarTensor` to copy by value instead." Read that as a claim about *languages* and it's wrong for this edition: Rust's `ScalarTensor` (Chapter 15.2) derives `Copy`, and `#[derive(Copy)]` means exactly "bitwise-copy this struct on every assignment, function argument, and `vec![...]` element" — there is no way to opt out of that once the derive is present, and no borrow-checker rule that would catch `xA`'s independent copy sitting inside `graph.nodes[0].inputs[0]` as a problem, because from the type system's point of view it *is* a completely ordinary, valid, independent `ScalarTensor`. The gap here isn't a C++ tax; it's a tax on *value-semantics tensors* specifically, and this book's Rust edition chose exactly the same value-semantics design Chapter 15 chose for the C++ edition (both diverging from Mojo's pointer-based `Tensor`) — so both editions owe the same bill, for the same structural reason, confirmed above with the same `xA.grad = 0.0000` a fresh compile and run actually produces.

```rust
// Part B's fix: a graph-owned table keyed by tensor_id, which every
// COPY of a ScalarTensor still carries (Rust's derived Copy impl
// copies every field, tensor_id included) -- closing the gap Part A
// leaves open.
fn accumulate_gradient(graph: &mut ComputationGraph, tensor: &ScalarTensor, incoming_grad: f32) {
    let entry = graph.grad_table.entry(tensor.tensor_id).or_insert(0.0);
    *entry += incoming_grad;
    if tensor.grad_fn_index >= 0 {
        let updated = graph.grad_table[&tensor.tensor_id];
        graph.nodes[tensor.grad_fn_index as usize].grad = updated;
    }
}
```

Every `ScalarTensor` is given a `tensor_id` at construction (a simple `AtomicI32` counter, `NEXT_TENSOR_ID.fetch_add(1, Ordering::SeqCst)`), and — critically — the derived `Copy` implementation copies that id along with everything else, so every independent value-copy of `xB` scattered across `graphB.nodes` still carries the *same* `tensor_id`. Routing accumulation through a `HashMap<i32, f32>` keyed by that id, owned by the graph rather than by any one copy, is what finally lets `read_grad(graphB, xB)` report `5.0` — genuinely confirmed above, `xB.tensor_id = 4` and all. This is not a divergence Mojo's own `Tensor` has to deal with, since its pointer-sharing design never splits into independent copies in the first place; it is a direct, traceable consequence of the value-semantics choice Chapter 15 made explicitly for *both* the C++ and Rust editions of this book, and flagged for exactly this moment.

## 17.2 Gradient Accumulation Strategies `[FOUNDATIONAL]`

### Intuition

Step 2 of the worked table is where `x.grad` became `1.0 + 4.0 = 5.0` rather than being overwritten to `4.0`. That single addition is the entire content of this section, and it is one of the two or three most common autograd bugs in every framework that implements it — get it wrong, and any input used more than once (a shared weight, a residual connection `y = f(x) + x`) silently receives only its *last* gradient contribution instead of the sum of all of them.

### Background

```rust
fn accumulate_gradient(graph: &mut ComputationGraph, tensor: &ScalarTensor, incoming_grad: f32) {
    let entry = graph.grad_table.entry(tensor.tensor_id).or_insert(0.0);
    *entry += incoming_grad;   // accumulate, don't replace
    if tensor.grad_fn_index >= 0 {
        let updated = graph.grad_table[&tensor.tensor_id];
        graph.nodes[tensor.grad_fn_index as usize].grad = updated;
    }
}

fn zero_grad(graph: &mut ComputationGraph, params: &[&ScalarTensor]) {
    for p in params {
        graph.grad_table.remove(&p.tensor_id);
    }
}
```

`HashMap::entry(...).or_insert(0.0)` is doing exactly the "not found → insert, found → add" branch the CUDA edition writes as an explicit `if`/`else` — Rust's entry API just folds both branches into one expression, the same idiom Chapter 15's own `OpRegistry` used for `register_op`.

### Worked Example 17.2.1 — The two branches, walked against the running example

At Step 1, `x` has no entry in `graph.grad_table` yet, so `or_insert(0.0)` inserts a fresh `0.0` and the `+=` brings it to `1.0` — the "not found" branch. At Step 2, `x`'s entry already holds `1.0`, so the new contribution `4.0` is *added*: `1.0 + 4.0 = 5.0`. Replacing instead of adding at Step 2 would have silently produced `x.grad = 4.0` — plausible-looking, still a number, and wrong. This is exactly why the residual-connection case in a real network is dangerous: `x` feeding both the shortcut and the transformed branch is structurally identical to `x` feeding both `AddOp` and `MulOp` above.

### Worked Example 17.2.2 — Resolving Chapter 16's open aliasing question, with real addresses

Chapter 16.2 flagged something left unresolved: `AddOp::backward` returns `[grad_output, grad_output]` — if `grad_output` were buffer-backed rather than a plain `f32`, that would be the *same* underlying allocation, handed to both `z`'s incoming gradient and `x`'s first contribution. Trace what actually happens to that aliasing, step by step, with `Rc<Vec<f32>>` standing in for a real gradient buffer — the exact tool Chapter 16.2's own COMMON TRAP identified as what reproducing this kind of aliasing on purpose requires:

Compiled and run:

```bash
rustc --edition 2024 -O 03_aliasing_resolution.rs -o 03_aliasing_resolution
./03_aliasing_resolution
```

Genuinely compiled and run:

```
=== Section 17.2: resolving Chapter 16's AddOp aliasing question with real addresses ===

add_backward_output (the ALIASED buffer AddOp::backward returns twice):
  address = 0x5632ca607af0, value = 1.0, strong_count = 1

--- Step 1: accumulate_gradient(z, add_backward_output) and accumulate_gradient(x, add_backward_output) ---
z_grad: address = 0x5632ca607af0, value = 1.0
x_grad: address = 0x5632ca607af0, value = 1.0
z_grad and x_grad share an address? true -- both hit the None branch, both ALIASED
strong_count on that shared allocation now = 4

--- Step 2: x receives its SECOND contribution (4.0, from MulOp::backward) ---
x_grad: address = 0x5632ca607dd0 (was 0x5632ca607af0), value = 5.0
z_grad: address = 0x5632ca607af0 (was 0x5632ca607af0), value = 1.0 -- UNTOUCHED by x's reassignment

x_grad's address changed?   true (elementwise_add_buffer allocated a fresh Rc)
z_grad's address unchanged? true (nothing ever mutated the allocation z_grad still points at)
z_grad still holds the correct value? true (1.0)

CONCLUSION: the aliasing between z_grad and x_grad after Step 1 was real, but it was
harmless to VALUES, because accumulate_gradient's Some-branch never mutates a buffer
that might be shared -- it allocates a brand-new one and reassigns, the exact same
first-assign-then-add-a-fresh-buffer discipline production autograd engines rely on.

--- what the CUDA edition's third trap (freeing the OLD buffer) looks like in Rust ---
The CUDA edition's own drafting process hit a real bug here: an earlier version of its
C++ file called free() on the old buffer right before reassignment, on the assumption
that nothing else could still be referencing it -- and that assumption is exactly what
aliasing breaks, corrupting z_grad's memory the first time that file was actually run.
Rust's Rc<T> makes the naive equivalent of that mistake IMPOSSIBLE to even attempt in
safe code -- there is no free() to call. The nearest analogue, Rc::get_mut, only ever
hands back a mutable reference when the reference count is exactly 1:

shared_for_demo strong_count (after cloning an alias) = 2
Rc::get_mut(&mut one_ref) while aliased (strong_count=2) = None
Rc::get_mut(&mut one_ref) after the alias is dropped (strong_count=1) = Some(..)

Rc::get_mut returned None while aliased -- the compiler and runtime jointly refuse to
hand out mutable access (which freeing-then-reusing would require) while a second owner
might still be reading the same allocation, and once that second owner (alias_for_demo)
is dropped, strong_count falls to 1 and get_mut succeeds. What Chapter 11.1's
RefCountedBuffer<T> gives C++ as a disciplined convention the programmer must uphold by
hand, Rc<T> gives Rust as a property the type system enforces: the actual free happens
automatically, exactly once, when the LAST Rc pointing at an allocation is dropped --
never before, and never manually.
```

The exact hex digits after `0x5632ca607` are genuinely ASLR-dependent and will differ on a different run or machine; what is reproducible is that the two "before" addresses match exactly, that `x_grad`'s address changes at Step 2 while `z_grad`'s does not, and that `Rc::get_mut` returns `None` while aliased and `Some(..)` once the alias count drops to one. `add_backward_output`'s buffer is cloned once (`Rc::clone`) to get an owned handle passed to both `accumulate_gradient` calls, so by the time both `z_grad` and `x_grad` are pointing at it, `strong_count` reads `4` — the original tensor's own handle, the cloned handle passed in, and the two aliases `z_grad` and `x_grad` now hold.

The CUDA edition's own drafting process hit a real, compiled bug here: an earlier version of its `03_aliasing_resolution.cu` called `free()` on the old buffer immediately after reassigning `x_grad`, on the assumption that nothing else could still be referencing it — an assumption that aliasing breaks exactly once, corrupting `z_grad`'s memory the first time that file actually ran. Porting *that* mistake into Rust isn't possible without leaving safe code altogether: there is no `free()` function to call on an `Rc<Vec<f32>>`, only `Rc::get_mut`, which hands back `Option<&mut Vec<f32>>` and returns `None` whenever `strong_count() > 1` — genuinely demonstrated above, twice, once returning `None` while `shared_for_demo` is still aliased and once returning `Some(..)` the instant the alias is dropped. Where Chapter 11.1's `RefCountedBuffer<T>` is a convention C++ gives the programmer to *manually* uphold correctly, `Rc<T>` is the same idea enforced by the type system: the actual deallocation happens automatically, exactly once, when the strong count reaches zero — not the instant any single alias stops using it, and never by a line of code a programmer could get wrong.

### Worked Example 17.2.3 — Verifying `x.grad = 5.0` without any calculus at all

The whole point of a gradient is that it predicts how much the output moves for a tiny nudge to the input — so test that prediction directly, by nudging `x` by `±0.001` and reading `w` both times:

Compiled and run:

```bash
rustc --edition 2024 -O 04_finite_difference_verification.rs -o 04_finite_difference_verification
./04_finite_difference_verification
```

Genuinely compiled and run:

```
=== Section 17.2: verifying x.grad=5.0, y.grad=3.0 with finite differences, no calculus ===

w(x=3.001, y=4.0) = 15.005
w(x=2.999, y=4.0) = 14.995
slope ~= (15.005 - 14.995) / 0.002 = 4.9992  (backward()'s x.grad was 5.0)

w(x=3.0, y=4.001) = 15.003
w(x=3.0, y=3.999) = 14.997
slope ~= (15.003 - 14.997) / 0.002 = 3.0003  (backward()'s y.grad was 3.0)

both slopes match backward()'s computed gradients: CONFIRMED
```

Both slopes land almost exactly on `backward()`'s computed gradients — `4.9992` against `5.0` and `3.0003` against `3.0` — with the small residual coming from `f32`'s roughly 7-digit precision rather than from any error in the finite-difference method itself (`w` is linear in both `x` and `y`, so a centered difference like this one is mathematically exact for it; the deviation here is purely `f32` rounding in the intermediate multiplication). These are the same figures, digit for digit, the CUDA edition's own `float` arithmetic produces — a case where genuinely independent compilations of the same formula, in two different languages, land on identical rounding, unlike Chapter 16.7's bisection example.

```
[COMMON TRAP]  accumulate_gradient assumes every operand already has the output's shape

Neither AddOp::backward (Chapter 16.2) nor accumulate_gradient above
ever compares shapes. That's fine for the running example -- every
tensor involved is a scalar -- but Chapter 12.4 already built a
kernel, broadcast_add, specifically for the case where one operand is
smaller and gets silently repeated. Reuse that section's own numbers:
A is 2x3, B is a single row of 3 values, and broadcast_add produces
C = [[11,22,33],[14,25,36]], a 2x3 result, from A (2x3) and B (1x3).

Suppose this addition is graph-tracked and the upstream gradient
arriving at C is grad_C = [[1,1,1],[1,1,1]] (a 2x3 matrix of ones, the
same all-ones convention Chapter 16.3 used for grad_output). A
already has the full 2x3 output shape, so grad_A = grad_C unchanged is
correct. But B's ORIGINAL shape was 1x3, not 2x3 -- it was repeated
down both rows by the broadcast, not actually duplicated in memory.
Handing B the full 2x3 grad_C and calling accumulate_gradient(B,
grad_C) would either fail outright (if elementwise_add requires
matching shapes) or -- if it doesn't check at all -- silently store a
2x3 gradient on a tensor whose own data is 1x3, a shape mismatch that
corrupts every later use of B's gradient.

What actually has to happen is a reduction: every row of grad_C that
came from the SAME repeated row of B needs to be summed back into one
row before it can be B's gradient. Column by column: grad_B[0,0] =
grad_C[0,0] + grad_C[1,0] = 1 + 1 = 2; the same sum applies to columns
1 and 2, giving grad_B = [2, 2, 2] -- a 1x3 result, matching B's real,
pre-broadcast shape, genuinely computed below.
```

Compiled and run:

```bash
rustc --edition 2024 -O 05_broadcast_gradient_trap.rs -o 05_broadcast_gradient_trap
./05_broadcast_gradient_trap
```

Genuinely compiled and run:

```
=== Section 17.2 COMMON TRAP: accumulate_gradient and broadcasting ===

A (2x3):
  [  1.0,  2.0,  3.0]
  [  4.0,  5.0,  6.0]
B (1x3):
  [ 10.0, 20.0, 30.0]
C = broadcast_add(A, B) (2x3):
  [ 11.0, 22.0, 33.0]
  [ 14.0, 25.0, 36.0]
(Chapter 12.4's own numbers: C = [[11,22,33],[14,25,36]] -- match: true)

grad_C (upstream, all ones) (2x3):
  [  1.0,  1.0,  1.0]
  [  1.0,  1.0,  1.0]

--- A's gradient: already output-shaped, no reduction needed ---
grad_A = grad_C unchanged, shape [2,3] matches A's shape [2,3] -- correct

--- B's gradient: the BROKEN approach ---
broken_grad_b(grad_C) (2x3):
  [  1.0,  1.0,  1.0]
  [  1.0,  1.0,  1.0]
shape is [2,3], but B's real shape is [1,3] -- MISMATCH, would corrupt B.grad

--- B's gradient: unbroadcast_gradient, correctly reducing rows ---
unbroadcast_gradient(grad_C, target=[1,3]) (1x3):
  [  2.0,  2.0,  2.0]
shape [1,3] matches B's real shape [1,3] -- correct
values: [2.0, 2.0, 2.0] (expected [2,2,2] -- each column summed down both rows)
matches the chapter's hand-derived grad_B = [2,2,2]: CONFIRMED
```

Neither `AddOp::backward` nor `accumulate_gradient` as built in this chapter performs this reduction anywhere; a broadcasting-aware version would need an explicit `unbroadcast_gradient` step — genuinely implemented and checked above, not left as an illustration — that sums a gradient back down to an operand's original shape before `accumulate_gradient` ever sees it. Every axis the forward pass was allowed to repeat a value across, backward has to sum back into one slot — the exact mirror image of what made the forward broadcast cheap in the first place.

## 17.3 Graph Traversal and Execution `[FOUNDATIONAL]`

### Intuition

Section 17.1's `backward()` already showed the traversal; what's worth calling out separately is what happens to the graph *afterward*.

### Background

In this framework, as in eager-mode PyTorch, the graph is single-use: `graph.nodes` for the running example held exactly two entries, was consumed once by the loop in Section 17.1, and would be discarded before the next forward pass builds a fresh one from scratch. This is a deliberate simplicity trade-off — a persistent, reusable graph enables graph-level optimization passes, but a rebuild-every-step graph is dramatically easier to reason about, and combined with an arena-style bump allocator, just as cheap in practice: resetting the arena at the top of the next `forward()` call reclaims every node from the discarded graph in `O(1)`, no matter how many nodes it held.

### Worked Example 17.3.1 — What "discard and rebuild" actually costs

Compiled and run:

```bash
rustc --edition 2024 -O 06_arena_single_use_graph.rs -o 06_arena_single_use_graph
./06_arena_single_use_graph
```

Genuinely compiled and run:

```
=== Section 17.3: the computation graph is single-use -- discard and rebuild is O(1) ===

--- building a 2-node graph (the running w = x*y + x example) ---
after building: arena.offset = 352 bytes
after arena.reset(): arena.offset = 0 bytes
reset() took 0.1830 microseconds

--- building a 2000-node graph ---
after building: arena.offset = 511840 bytes
after arena.reset(): arena.offset = 0 bytes
reset() took 0.0430 microseconds

reset() cost for 2 nodes vs 2000 nodes: 0.1830 us vs 0.0430 us
(both figures are single measurements of a sub-microsecond operation and will jitter
 run to run -- what is reproducible is that reset() does not scale with node count,
 since it only ever writes one field: offset = 0, regardless of how many allocations
 preceded it. Chapter 11.2 already established the same fact on raw byte allocations;
 this just confirms it holds for graph-sized allocations too.)

--- for contrast: individually dropping 2000 heap-allocated nodes instead ---
dropping 2000 individually heap-allocated nodes took 11.3830 microseconds -- scales with
node count, unlike arena.reset()'s single field write, even though both end up
reclaiming the same amount of memory.
```

The two `reset()` timings are both sub-microsecond, single-measurement figures that will jitter on any rerun — in this particular run, the *smaller* graph's reset happened to measure slower than the *larger* one, which is itself the point: at this timescale, the measurement noise dwarfs any dependence on node count, because there genuinely isn't one. `reset()` only ever writes a single field (`offset = 0`); no matter how many `allocate()` calls preceded it, undoing all of them costs exactly the same one write. The byte counts above, by contrast, are exactly reproducible: `352` bytes for the 2-node graph and `511840` for the 2000-node one, both direct consequences of the same 256-byte-aligned bump arithmetic every run performs identically. The contrast at the bottom makes the alternative's real cost visible: individually dropping 2,000 separately heap-allocated `Vec<u8>` buffers — Rust's per-value `Drop` running once per element as `individual.drain(..)` yields each one — took `11.38` microseconds in this run, genuinely proportional to node count, unlike the arena's reset, even though both approaches reclaim the exact same amount of memory in the end. The arena trades a small amount of memory (its peak size has to accommodate the largest graph any single forward pass ever builds) for making that per-step cost disappear entirely.

## 17.4 Memory Optimization for Gradients `[FOUNDATIONAL]`

### Intuition

Two optimizations matter once this engine runs on real workloads rather than a two-node example.

### Background

First, **gradient-only-where-needed**: a node whose entire input subtree has `requires_grad = false` was already excluded from the graph in Chapter 15.3 (`record`'s early `return output` when `needs_grad` is `false`), so no gradient memory is ever allocated for it. Second, **saved-tensor pruning** — and the running example already demonstrates exactly which inputs are safe to drop. `AddOp::backward` (Chapter 16.2) never reads `inputs` at all — its two return values are just copies of `grad_output`. That means a node recording an addition can drop its references to both inputs immediately after forward runs, relying purely on the `grad_output` handed to it later, while `MulOp` and `MatMulOp` cannot — their backward rules read the *other* input, so both must stay alive until backward visits that node.

```rust
// AddOp: grad(a) = grad(b) = grad_output, no input needed.
// MulOp, MatMulOp: backward reads the *other* input (Chapter 16.2, 16.3).
fn needs_input_for_backward(op_name: &str) -> bool {
    op_name != "add"
}
```

### Worked Example 17.4.1 — Putting a number on the saving

A `[500, 8]` `f32` activation tensor is `500 × 8 × 4 bytes = 16,000 bytes`. An `AddOp` node in the middle of a training loop that drops both of its saved inputs the instant forward passes it frees `32,000` bytes it would otherwise have to keep alive for the entire duration of the backward pass.

Compiled and run:

```bash
rustc --edition 2024 -O 07_saved_tensor_pruning.rs -o 07_saved_tensor_pruning
./07_saved_tensor_pruning
```

Genuinely compiled and run:

```
=== Section 17.4: saved-tensor pruning -- which inputs backward actually needs ===

needs_input_for_backward("add")    = false
needs_input_for_backward("mul")    = true
needs_input_for_backward("matmul") = true

a [500, 8] f32 activation tensor: 500 x 8 x 4 bytes = 16000 bytes
an AddOp node dropping BOTH of its saved inputs the instant forward finishes frees:
  16000 bytes x 2 = 32000 bytes

--- tracked allocations: AddOp-style pruning vs MulOp-style retention ---
6 AddOp nodes, 32000 bytes/node saved: pruned-path live bytes = 0, retained-path live bytes = 192000
difference: 192000 bytes NOT kept alive across the backward pass when AddOp prunes its inputs

over 10000 training steps, at 0.1831 MB saved per step (this one node type alone):
  cumulative memory that never had to stay resident at once: not simply additive across
  steps (each step's activations are freed before the next begins regardless) -- the real
  benefit is PEAK memory per step, 0.1831 MB lower every single step, which is what actually
  determines whether a model fits in GPU memory, not a running total across steps.
```

Multiply that `32,000`-byte saving by every `add_bias` call in a network with several layers, and by thousands of training steps, and "don't pay memory for a value nothing will read again" stops being a micro-optimization and starts being the difference between a model that fits in GPU memory and one that doesn't — the genuinely tracked `192,000`-byte figure above is for just six `AddOp` nodes in one forward pass, before scaling to a real network's layer count at all.

## 17.5 Complete Runnable Code
### File: `01_naive_backward_bug.rs`

```rust
use std::collections::HashMap;

// ScalarTensor gains a `has_grad`/`grad` pair on top of Chapter 15's
// fields -- Rust's stand-in for Mojo's Tensor.grad being an Optional
// that starts as `.is_none()`.
#[derive(Clone, Copy, Debug)]
struct ScalarTensor {
    value: f32,
    requires_grad: bool,
    grad_fn_index: i32,
    has_grad: bool,
    grad: f32,
}

impl ScalarTensor {
    fn new(value: f32, requires_grad: bool) -> Self {
        ScalarTensor { value, requires_grad, grad_fn_index: -1, has_grad: false, grad: 0.0 }
    }
}

trait Differentiable {
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32>;
}

struct AddOp;
impl Differentiable for AddOp {
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output, grad_output]
    }
}

struct MulOp;
impl Differentiable for MulOp {
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output * inputs[1].value, grad_output * inputs[0].value]
    }
}

struct OpRegistry {
    ops: HashMap<String, Box<dyn Differentiable>>,
}
impl OpRegistry {
    fn new() -> Self {
        OpRegistry { ops: HashMap::new() }
    }
    fn register_op(&mut self, name: &str, op: Box<dyn Differentiable>) {
        self.ops.insert(name.to_string(), op);
    }
    fn get(&self, name: &str) -> &dyn Differentiable {
        self.ops.get(name).expect("Unregistered op").as_ref()
    }
}

fn chain_rule_step(
    registry: &OpRegistry,
    op_name: &str,
    grad_output: f32,
    inputs: &[ScalarTensor],
    output: &ScalarTensor,
) -> Vec<f32> {
    registry.get(op_name).backward(grad_output, inputs, output)
}

// GraphNode, reused verbatim from Chapter 15.2 -- its own `grad` field
// is a SEPARATE piece of storage from any ScalarTensor's `.grad`.
struct GraphNode {
    op_name: String,
    inputs: Vec<ScalarTensor>,
    output: ScalarTensor,
    grad: f32, // zero-initialized, Chapter 15.2 -- and, as this file
               // demonstrates, never written by the naive backward() below.
    #[allow(dead_code)]
    requires_grad: bool,
}
impl GraphNode {
    fn new(op_name: String, inputs: Vec<ScalarTensor>, output: ScalarTensor) -> Self {
        GraphNode { op_name, inputs, output, requires_grad: true, grad: 0.0 }
    }
}

struct ComputationGraph {
    nodes: Vec<GraphNode>,
}
impl ComputationGraph {
    fn new() -> Self {
        ComputationGraph { nodes: Vec::new() }
    }
    fn record(&mut self, op_name: &str, inputs: Vec<ScalarTensor>, output: ScalarTensor) -> ScalarTensor {
        let needs_grad = inputs.iter().any(|t| t.requires_grad);
        if !needs_grad {
            return output;
        }
        let mut out = output;
        out.requires_grad = true;
        out.grad_fn_index = self.nodes.len() as i32;
        let node = GraphNode::new(op_name.to_string(), inputs, output);
        self.nodes.push(node);
        out
    }
}

fn mul(graph: &mut ComputationGraph, a: ScalarTensor, b: ScalarTensor) -> ScalarTensor {
    let result = ScalarTensor::new(a.value * b.value, false);
    graph.record("mul", vec![a, b], result)
}
fn add(graph: &mut ComputationGraph, a: ScalarTensor, b: ScalarTensor) -> ScalarTensor {
    let result = ScalarTensor::new(a.value + b.value, false);
    graph.record("add", vec![a, b], result)
}

fn topological_backward_order(graph: &ComputationGraph) -> Vec<usize> {
    (0..graph.nodes.len()).rev().collect()
}

// Naive accumulate_gradient -- the "has_grad vs elementwise_add" branch
// from Section 17.2, ported literally, with no mirroring back into
// GraphNode.grad (that gap is exactly this file's point).
fn accumulate_gradient_naive(tensor: &mut ScalarTensor, incoming_grad: f32) {
    if !tensor.has_grad {
        tensor.grad = incoming_grad;
        tensor.has_grad = true;
    } else {
        tensor.grad += incoming_grad;
    }
}

// backward(), ported LITERALLY from Section 17.1's design -- including
// the bug this section's COMMON TRAP identifies. Never called with the
// fix applied; see file 02 for the corrected version.
fn backward_naive(registry: &OpRegistry, graph: &mut ComputationGraph, loss: &mut ScalarTensor) {
    // Seed: dL/dL = 1
    loss.grad = 1.0;
    loss.has_grad = true;

    let order = topological_backward_order(graph);
    let order_strs: Vec<String> = order.iter().map(|i| i.to_string()).collect();
    println!("backward_naive: order = [{}]", order_strs.join(", "));

    for node_idx in order {
        let node = &graph.nodes[node_idx];
        println!(
            "  visiting graph.nodes[{}] (\"{}\"): node.grad = {:.4} -> {}",
            node_idx,
            node.op_name,
            node.grad,
            if node.grad == 0.0 { "is_zero(), SKIPPING" } else { "proceeding" }
        );
        if node.grad == 0.0 {
            continue; // this output was "never used downstream" -- or so this reads
        }
        let input_grads = chain_rule_step(registry, &node.op_name, node.grad, &node.inputs, &node.output);
        let n = node.inputs.len();
        for i in 0..n {
            accumulate_gradient_naive(&mut graph.nodes[node_idx].inputs[i], input_grads[i]);
        }
    }
}

fn main() {
    println!("=== Section 17.1: backward(), ported literally -- including its own bug ===");

    let mut registry = OpRegistry::new();
    registry.register_op("add", Box::new(AddOp));
    registry.register_op("mul", Box::new(MulOp));

    let mut graph = ComputationGraph::new();
    let x = ScalarTensor::new(3.0, true);
    let y = ScalarTensor::new(4.0, true);
    let z = mul(&mut graph, x, y);
    let mut w = add(&mut graph, z, x);

    println!("w = {:.1}, graph.nodes.len() = {}", w.value, graph.nodes.len());
    println!("graph.nodes[0].grad (mul node, at construction) = {:.4}", graph.nodes[0].grad);
    println!("graph.nodes[1].grad (add node, at construction) = {:.4}\n", graph.nodes[1].grad);

    backward_naive(&registry, &mut graph, &mut w);

    println!("\nafter backward_naive:");
    println!("  w.has_grad = {}, w.grad = {:.4}", w.has_grad, w.grad);
    println!("  x.has_grad = {} (x.grad never touched by this run)", x.has_grad);
    println!("  y.has_grad = {} (y.grad never touched by this run)", y.has_grad);
    println!("\nexpected from the hand-worked table: x.grad=5.0, y.grad=3.0 -- NOT what this run produced.");
    println!("root cause: loss.grad was seeded on `w` (a ScalarTensor), but the loop reads");
    println!("graph.nodes[node_idx].grad -- a SEPARATE field on GraphNode that this function");
    println!(
        "never writes to at all. graph.nodes[1].grad is still {:.4} when the loop reaches it,",
        graph.nodes[1].grad
    );
    println!("so the add node's backward is skipped, and neither x.grad nor y.grad is ever set.");
}
```

```bash
rustc --edition 2024 -O 01_naive_backward_bug.rs -o 01_naive_backward_bug
./01_naive_backward_bug
```

### File: `02_fixed_backward_and_accumulation.rs`

```rust
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};

// ScalarTensor gains a `tensor_id` on top of Chapter 15's fields -- see
// the note below Part A for why this file needs it.
static NEXT_TENSOR_ID: AtomicI32 = AtomicI32::new(0);

#[derive(Clone, Copy, Debug)]
struct ScalarTensor {
    value: f32,
    requires_grad: bool,
    grad_fn_index: i32,
    has_grad: bool,
    grad: f32,
    tensor_id: i32,
}

impl ScalarTensor {
    fn new(value: f32, requires_grad: bool) -> Self {
        let id = NEXT_TENSOR_ID.fetch_add(1, Ordering::SeqCst);
        ScalarTensor { value, requires_grad, grad_fn_index: -1, has_grad: false, grad: 0.0, tensor_id: id }
    }
}

trait Differentiable {
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32>;
}

struct AddOp;
impl Differentiable for AddOp {
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output, grad_output]
    }
}

struct MulOp;
impl Differentiable for MulOp {
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output * inputs[1].value, grad_output * inputs[0].value]
    }
}

struct OpRegistry {
    ops: HashMap<String, Box<dyn Differentiable>>,
}
impl OpRegistry {
    fn new() -> Self {
        OpRegistry { ops: HashMap::new() }
    }
    fn register_op(&mut self, name: &str, op: Box<dyn Differentiable>) {
        self.ops.insert(name.to_string(), op);
    }
    fn get(&self, name: &str) -> &dyn Differentiable {
        self.ops.get(name).expect("Unregistered op").as_ref()
    }
}

fn chain_rule_step(
    registry: &OpRegistry,
    op_name: &str,
    grad_output: f32,
    inputs: &[ScalarTensor],
    output: &ScalarTensor,
) -> Vec<f32> {
    registry.get(op_name).backward(grad_output, inputs, output)
}

struct GraphNode {
    op_name: String,
    inputs: Vec<ScalarTensor>,
    #[allow(dead_code)]
    output: ScalarTensor,
    grad: f32,
    #[allow(dead_code)]
    requires_grad: bool,
}
impl GraphNode {
    fn new(op_name: String, inputs: Vec<ScalarTensor>, output: ScalarTensor) -> Self {
        GraphNode { op_name, inputs, output, requires_grad: true, grad: 0.0 }
    }
}

struct ComputationGraph {
    nodes: Vec<GraphNode>,
    // A shared table keyed by tensor_id, not by GraphNode -- see Part B.
    grad_table: HashMap<i32, f32>,
}
impl ComputationGraph {
    fn new() -> Self {
        ComputationGraph { nodes: Vec::new(), grad_table: HashMap::new() }
    }
    fn record(&mut self, op_name: &str, inputs: Vec<ScalarTensor>, output: ScalarTensor) -> ScalarTensor {
        let needs_grad = inputs.iter().any(|t| t.requires_grad);
        if !needs_grad {
            return output;
        }
        let mut out = output;
        out.requires_grad = true;
        out.grad_fn_index = self.nodes.len() as i32;
        let node = GraphNode::new(op_name.to_string(), inputs, output);
        self.nodes.push(node);
        out
    }
}

fn mul(graph: &mut ComputationGraph, a: ScalarTensor, b: ScalarTensor) -> ScalarTensor {
    let result = ScalarTensor::new(a.value * b.value, false);
    graph.record("mul", vec![a, b], result)
}
fn add(graph: &mut ComputationGraph, a: ScalarTensor, b: ScalarTensor) -> ScalarTensor {
    let result = ScalarTensor::new(a.value + b.value, false);
    graph.record("add", vec![a, b], result)
}

fn topological_backward_order(graph: &ComputationGraph) -> Vec<usize> {
    (0..graph.nodes.len()).rev().collect()
}

// ---- Part A: Section 17.1's grad_fn_index fix, applied literally ----
// accumulate_gradient writes into tensor.grad (the local copy it was
// handed) and mirrors into graph.nodes[tensor.grad_fn_index].grad.
fn accumulate_gradient_part_a(graph: &mut ComputationGraph, tensor: &mut ScalarTensor, incoming_grad: f32) {
    if !tensor.has_grad {
        tensor.grad = incoming_grad;
        tensor.has_grad = true;
    } else {
        tensor.grad += incoming_grad;
    }
    if tensor.grad_fn_index >= 0 {
        graph.nodes[tensor.grad_fn_index as usize].grad = tensor.grad;
    }
}

fn backward_part_a(registry: &OpRegistry, graph: &mut ComputationGraph, loss: &mut ScalarTensor) {
    loss.grad = 1.0;
    loss.has_grad = true;
    if loss.grad_fn_index >= 0 {
        graph.nodes[loss.grad_fn_index as usize].grad = loss.grad;
    }

    for node_idx in topological_backward_order(graph) {
        let node = &graph.nodes[node_idx];
        if node.grad == 0.0 {
            continue;
        }
        let input_grads = chain_rule_step(registry, &node.op_name, node.grad, &node.inputs, &node.output);
        let n = node.inputs.len();
        for i in 0..n {
            let mut input_copy = graph.nodes[node_idx].inputs[i];
            accumulate_gradient_part_a(graph, &mut input_copy, input_grads[i]);
            graph.nodes[node_idx].inputs[i] = input_copy;
        }
    }
}

// ---- Part B: the SECOND, genuinely reproduced gap ----
// accumulate_gradient now writes into a graph-owned table keyed by
// tensor_id, which every COPY of a ScalarTensor still carries (Rust's
// derived Copy impl copies every field, tensor_id included) -- closing
// the gap Part A's trace exposes.
fn accumulate_gradient(graph: &mut ComputationGraph, tensor: &ScalarTensor, incoming_grad: f32) {
    let entry = graph.grad_table.entry(tensor.tensor_id).or_insert(0.0);
    *entry += incoming_grad;
    if tensor.grad_fn_index >= 0 {
        let updated = graph.grad_table[&tensor.tensor_id];
        graph.nodes[tensor.grad_fn_index as usize].grad = updated;
    }
}

fn read_grad(graph: &ComputationGraph, tensor: &ScalarTensor) -> f32 {
    *graph.grad_table.get(&tensor.tensor_id).unwrap_or(&0.0)
}

fn backward(registry: &OpRegistry, graph: &mut ComputationGraph, loss: &ScalarTensor) {
    graph.grad_table.insert(loss.tensor_id, 1.0);
    if loss.grad_fn_index >= 0 {
        graph.nodes[loss.grad_fn_index as usize].grad = 1.0;
    }

    for node_idx in topological_backward_order(graph) {
        let node = &graph.nodes[node_idx];
        if node.grad == 0.0 {
            continue;
        }
        let input_grads = chain_rule_step(registry, &node.op_name, node.grad, &node.inputs, &node.output);
        let inputs_snapshot = graph.nodes[node_idx].inputs.clone();
        for (i, inp) in inputs_snapshot.iter().enumerate() {
            accumulate_gradient(graph, inp, input_grads[i]);
        }
    }
}

fn zero_grad(graph: &mut ComputationGraph, params: &[&ScalarTensor]) {
    for p in params {
        graph.grad_table.remove(&p.tensor_id);
    }
}

fn main() {
    println!("=== Section 17.1/17.2: fixed backward(), full trace for w = x*y + x ===\n");

    let mut registry = OpRegistry::new();
    registry.register_op("add", Box::new(AddOp));
    registry.register_op("mul", Box::new(MulOp));

    // --- Part A: apply ONLY the grad_fn_index mirroring fix ---
    println!("--- Part A: Section 17.1's grad_fn_index fix, applied literally ---");
    let mut graph_a = ComputationGraph::new();
    let x_a = ScalarTensor::new(3.0, true);
    let y_a = ScalarTensor::new(4.0, true);
    let z_a = mul(&mut graph_a, x_a, y_a);
    let mut w_a = add(&mut graph_a, z_a, x_a);
    backward_part_a(&registry, &mut graph_a, &mut w_a);
    println!(
        "graph.nodes[1] (\"add\").grad = {:.4}, graph.nodes[0] (\"mul\").grad = {:.4}  (mirroring works)",
        graph_a.nodes[1].grad, graph_a.nodes[0].grad
    );
    println!(
        "but xA.grad (the ORIGINAL variable declared in main()) = {:.4}, has_grad = {}",
        x_a.grad, x_a.has_grad
    );
    println!("expected 5.0 -- MISMATCH. Cause: node.inputs[i] inside each GraphNode is an");
    println!("INDEPENDENT COPY of xA (ScalarTensor derives Copy, Chapter 15 -- Rust bitwise-copies");
    println!("the whole struct into the Vec, same as Chapter 15 flagged for a value-copying design).");
    println!("accumulate_gradient_part_a writes into that COPY's .grad field, not into xA itself --");
    println!("the mirroring fix genuinely works for graph.nodes[].grad, but is not enough on its own.\n");

    // --- Part B: fix the value-copy gap with a graph-owned grad table ---
    println!("--- Part B: a grad_table keyed by tensor_id, closing the value-copy gap ---");
    let mut graph_b = ComputationGraph::new();
    let x_b = ScalarTensor::new(3.0, true);
    let y_b = ScalarTensor::new(4.0, true);
    let z_b = mul(&mut graph_b, x_b, y_b);
    let w_b = add(&mut graph_b, z_b, x_b);
    println!("xB.tensor_id = {} (every copy of xB inside graphB.nodes carries this same id)", x_b.tensor_id);
    backward(&registry, &mut graph_b, &w_b);
    let x_b_grad = read_grad(&graph_b, &x_b);
    let y_b_grad = read_grad(&graph_b, &y_b);
    println!("read_grad(graphB, xB) = {:.4}, read_grad(graphB, yB) = {:.4}", x_b_grad, y_b_grad);
    println!(
        "matches the hand-worked table: x.grad=5.0, y.grad=3.0 -- {}\n",
        if x_b_grad == 5.0 && y_b_grad == 3.0 { "CONFIRMED" } else { "MISMATCH" }
    );

    println!("--- Worked Example 17.2.1: the two branches, walked explicitly ---");
    println!("Step 1 (add node): xB's first contribution -- table has no entry yet, insert 1.0");
    println!("Step 2 (mul node): xB's second contribution -- entry exists, add: 1.0 + 4.0 = 5.0");
    println!("final read_grad(graphB, xB) = {:.1}\n", x_b_grad);

    println!("--- zero_grad ---");
    println!(
        "before zero_grad: read_grad(xB)={:.1}, read_grad(yB)={:.1}",
        read_grad(&graph_b, &x_b),
        read_grad(&graph_b, &y_b)
    );
    zero_grad(&mut graph_b, &[&x_b, &y_b]);
    println!(
        "after zero_grad: read_grad(xB)={:.1}, read_grad(yB)={:.1}",
        read_grad(&graph_b, &x_b),
        read_grad(&graph_b, &y_b)
    );
}
```

```bash
rustc --edition 2024 -O 02_fixed_backward_and_accumulation.rs -o 02_fixed_backward_and_accumulation
./02_fixed_backward_and_accumulation
```

### File: `03_aliasing_resolution.rs`

```rust
use std::rc::Rc;

// A buffer-backed tensor, the Rust analogue of Chapter 16.2's COMMON
// TRAP tool for making the aliasing question concrete with real
// addresses. `Option<Rc<Vec<f32>>>` is the idiomatic Rust shape for
// "no gradient yet" (None, Mojo's `.is_none()`) plus "a SHARED,
// reference-counted buffer once one exists" -- Rc is exactly the
// mechanism Chapter 16.2's own COMMON TRAP identified as what
// reproducing the CUDA edition's aliasing on purpose requires.
struct BufferTensor {
    grad_data: Option<Rc<Vec<f32>>>,
}

fn make_grad(v: f32) -> BufferTensor {
    BufferTensor { grad_data: Some(Rc::new(vec![v])) }
}
fn no_grad() -> BufferTensor {
    BufferTensor { grad_data: None }
}

// elementwise_add_buffer: ALWAYS allocates a fresh output buffer,
// exactly like every kernel since Chapter 12 -- it never mutates
// either input's memory in place.
fn elementwise_add_buffer(a: &Rc<Vec<f32>>, b: &Rc<Vec<f32>>) -> Rc<Vec<f32>> {
    let out: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    Rc::new(out)
}

// accumulate_gradient, buffer version: the None branch ALIASES
// (Rc::clone bumps the reference count and shares the same
// allocation -- it does not copy the underlying Vec<f32> at all); the
// Some branch calls elementwise_add_buffer, which allocates something
// new and reassigns.
fn accumulate_gradient(tensor: &mut BufferTensor, incoming_grad: &Rc<Vec<f32>>) {
    match &tensor.grad_data {
        None => {
            tensor.grad_data = Some(Rc::clone(incoming_grad)); // ALIASES
        }
        Some(existing) => {
            let fresh = elementwise_add_buffer(existing, incoming_grad);
            tensor.grad_data = Some(fresh); // reassign to a freshly allocated buffer
        }
    }
}

fn main() {
    println!("=== Section 17.2: resolving Chapter 16's AddOp aliasing question with real addresses ===\n");

    // AddOp::backward hands out the SAME buffer to both z and x --
    // exactly Chapter 16.2's COMMON TRAP, reproduced here as the
    // upstream gradient arriving from the add node.
    let add_backward_output = make_grad(1.0);
    let add_output_rc = add_backward_output.grad_data.as_ref().unwrap();
    println!("add_backward_output (the ALIASED buffer AddOp::backward returns twice):");
    println!(
        "  address = {:p}, value = {:.1}, strong_count = {}\n",
        Rc::as_ptr(add_output_rc),
        add_output_rc[0],
        Rc::strong_count(add_output_rc)
    );

    let mut z_grad = no_grad();
    let mut x_grad = no_grad();

    println!("--- Step 1: accumulate_gradient(z, add_backward_output) and accumulate_gradient(x, add_backward_output) ---");
    let add_output_rc_owned = Rc::clone(add_output_rc);
    accumulate_gradient(&mut z_grad, &add_output_rc_owned);
    accumulate_gradient(&mut x_grad, &add_output_rc_owned);
    let z_ptr_1 = Rc::as_ptr(z_grad.grad_data.as_ref().unwrap());
    let x_ptr_1 = Rc::as_ptr(x_grad.grad_data.as_ref().unwrap());
    println!("z_grad: address = {:p}, value = {:.1}", z_ptr_1, z_grad.grad_data.as_ref().unwrap()[0]);
    println!("x_grad: address = {:p}, value = {:.1}", x_ptr_1, x_grad.grad_data.as_ref().unwrap()[0]);
    println!(
        "z_grad and x_grad share an address? {} -- both hit the None branch, both ALIASED",
        z_ptr_1 == x_ptr_1
    );
    println!(
        "strong_count on that shared allocation now = {}\n",
        Rc::strong_count(z_grad.grad_data.as_ref().unwrap())
    );

    let x_grad_address_before = x_ptr_1;
    let z_grad_address_before = z_ptr_1;

    println!("--- Step 2: x receives its SECOND contribution (4.0, from MulOp::backward) ---");
    let mul_contribution = Rc::new(vec![4.0f32]);
    accumulate_gradient(&mut x_grad, &mul_contribution);
    let x_ptr_2 = Rc::as_ptr(x_grad.grad_data.as_ref().unwrap());
    let z_ptr_2 = Rc::as_ptr(z_grad.grad_data.as_ref().unwrap());
    println!(
        "x_grad: address = {:p} (was {:p}), value = {:.1}",
        x_ptr_2, x_grad_address_before, x_grad.grad_data.as_ref().unwrap()[0]
    );
    println!(
        "z_grad: address = {:p} (was {:p}), value = {:.1} -- UNTOUCHED by x's reassignment\n",
        z_ptr_2, z_grad_address_before, z_grad.grad_data.as_ref().unwrap()[0]
    );

    println!(
        "x_grad's address changed?   {} (elementwise_add_buffer allocated a fresh Rc)",
        x_ptr_2 != x_grad_address_before
    );
    println!(
        "z_grad's address unchanged? {} (nothing ever mutated the allocation z_grad still points at)",
        z_ptr_2 == z_grad_address_before
    );
    println!(
        "z_grad still holds the correct value? {} ({:.1})\n",
        z_grad.grad_data.as_ref().unwrap()[0] == 1.0,
        z_grad.grad_data.as_ref().unwrap()[0]
    );

    println!("CONCLUSION: the aliasing between z_grad and x_grad after Step 1 was real, but it was");
    println!("harmless to VALUES, because accumulate_gradient's Some-branch never mutates a buffer");
    println!("that might be shared -- it allocates a brand-new one and reassigns, the exact same");
    println!("first-assign-then-add-a-fresh-buffer discipline production autograd engines rely on.\n");

    println!("--- what the CUDA edition's third trap (freeing the OLD buffer) looks like in Rust ---");
    println!("The CUDA edition's own drafting process hit a real bug here: an earlier version of its");
    println!("C++ file called free() on the old buffer right before reassignment, on the assumption");
    println!("that nothing else could still be referencing it -- and that assumption is exactly what");
    println!("aliasing breaks, corrupting z_grad's memory the first time that file was actually run.");
    println!("Rust's Rc<T> makes the naive equivalent of that mistake IMPOSSIBLE to even attempt in");
    println!("safe code -- there is no free() to call. The nearest analogue, Rc::get_mut, only ever");
    println!("hands back a mutable reference when the reference count is exactly 1:\n");

    let shared_for_demo = Rc::new(vec![1.0f32]);
    let alias_for_demo = Rc::clone(&shared_for_demo);
    println!(
        "shared_for_demo strong_count (after cloning an alias) = {}",
        Rc::strong_count(&shared_for_demo)
    );
    let mut one_ref = shared_for_demo;
    let get_mut_result = Rc::get_mut(&mut one_ref);
    println!(
        "Rc::get_mut(&mut one_ref) while aliased (strong_count=2) = {}",
        if get_mut_result.is_some() { "Some(..)" } else { "None" }
    );
    drop(alias_for_demo);
    let get_mut_after_drop = Rc::get_mut(&mut one_ref);
    println!(
        "Rc::get_mut(&mut one_ref) after the alias is dropped (strong_count=1) = {}",
        if get_mut_after_drop.is_some() { "Some(..)" } else { "None" }
    );
    println!("\nRc::get_mut returned None while aliased -- the compiler and runtime jointly refuse to");
    println!("hand out mutable access (which freeing-then-reusing would require) while a second owner");
    println!("might still be reading the same allocation, and once that second owner (alias_for_demo)");
    println!("is dropped, strong_count falls to 1 and get_mut succeeds. What Chapter 11.1's");
    println!("RefCountedBuffer<T> gives C++ as a disciplined convention the programmer must uphold by");
    println!("hand, Rc<T> gives Rust as a property the type system enforces: the actual free happens");
    println!("automatically, exactly once, when the LAST Rc pointing at an allocation is dropped --");
    println!("never before, and never manually.");
}
```

```bash
rustc --edition 2024 -O 03_aliasing_resolution.rs -o 03_aliasing_resolution
./03_aliasing_resolution
```

### File: `04_finite_difference_verification.rs`

```rust
// w = x*y + x, the running example -- verify backward()'s computed
// gradients against nothing but repeated forward evaluations.
fn w_forward(x: f32, y: f32) -> f32 {
    x * y + x
}

fn main() {
    println!("=== Section 17.2: verifying x.grad=5.0, y.grad=3.0 with finite differences, no calculus ===\n");

    let x = 3.0f32;
    let y = 4.0f32;
    let h = 0.001f32;

    let w_x_plus = w_forward(x + h, y);
    let w_x_minus = w_forward(x - h, y);
    let slope_x = (w_x_plus - w_x_minus) / (2.0 * h);
    println!("w(x={:.3}, y={:.1}) = {:.3}", x + h, y, w_x_plus);
    println!("w(x={:.3}, y={:.1}) = {:.3}", x - h, y, w_x_minus);
    println!(
        "slope ~= ({:.3} - {:.3}) / {:.3} = {:.4}  (backward()'s x.grad was 5.0)\n",
        w_x_plus, w_x_minus, 2.0 * h, slope_x
    );

    let w_y_plus = w_forward(x, y + h);
    let w_y_minus = w_forward(x, y - h);
    let slope_y = (w_y_plus - w_y_minus) / (2.0 * h);
    println!("w(x={:.1}, y={:.3}) = {:.3}", x, y + h, w_y_plus);
    println!("w(x={:.1}, y={:.3}) = {:.3}", x, y - h, w_y_minus);
    println!(
        "slope ~= ({:.3} - {:.3}) / {:.3} = {:.4}  (backward()'s y.grad was 3.0)\n",
        w_y_plus, w_y_minus, 2.0 * h, slope_y
    );

    let ok = slope_x > 4.99 && slope_x < 5.01 && slope_y > 2.99 && slope_y < 3.01;
    println!("both slopes match backward()'s computed gradients: {}", if ok { "CONFIRMED" } else { "MISMATCH" });
}
```

```bash
rustc --edition 2024 -O 04_finite_difference_verification.rs -o 04_finite_difference_verification
./04_finite_difference_verification
```

### File: `05_broadcast_gradient_trap.rs`

```rust
// Chapter 12.4's exact running example: A is 2x3, B is a single row
// of 3 values broadcast down both rows.
#[derive(Clone)]
struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        Matrix { data, rows, cols }
    }
    fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

// broadcast_add_kernel's host mirror, from Chapter 12.4: C[i,j] = A[i,j] + B[0,j].
fn broadcast_add(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = vec![0.0f32; a.rows * a.cols];
    for i in 0..a.rows {
        for j in 0..a.cols {
            out[i * a.cols + j] = a.at(i, j) + b.at(0, j);
        }
    }
    Matrix::new(a.rows, a.cols, out)
}

fn print_matrix(name: &str, m: &Matrix) {
    println!("{} ({}x{}):", name, m.rows, m.cols);
    for i in 0..m.rows {
        let row: Vec<String> = (0..m.cols).map(|j| format!("{:5.1}", m.at(i, j))).collect();
        println!("  [{}]", row.join(","));
    }
}

// The BROKEN approach: hand B the full output-shaped gradient
// unchanged, the same way accumulate_gradient does for every operand
// so far in this book -- none of which has ever needed a shape check.
fn broken_grad_b(grad_c: &Matrix) -> Matrix {
    grad_c.clone() // wrong shape: 2x3, but B's real shape is 1x3
}

// The CORRECT approach: sum every row of grad_C that came from the
// SAME repeated row of B back into one row before calling it B's
// gradient -- the mirror image of what made the forward broadcast cheap.
fn unbroadcast_gradient_2d(grad: &Matrix, target_rows: usize, target_cols: usize) -> Matrix {
    let mut result = vec![0.0f32; target_rows * target_cols];
    for i in 0..grad.rows {
        let target_row = if target_rows == 1 { 0 } else { i };
        for j in 0..grad.cols {
            let target_col = if target_cols == 1 { 0 } else { j };
            result[target_row * target_cols + target_col] += grad.at(i, j);
        }
    }
    Matrix::new(target_rows, target_cols, result)
}

fn main() {
    println!("=== Section 17.2 COMMON TRAP: accumulate_gradient and broadcasting ===\n");

    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::new(1, 3, vec![10.0, 20.0, 30.0]);
    let c = broadcast_add(&a, &b);
    print_matrix("A", &a);
    print_matrix("B", &b);
    print_matrix("C = broadcast_add(A, B)", &c);
    let matches_ch12 = c.at(0, 0) == 11.0
        && c.at(0, 1) == 22.0
        && c.at(0, 2) == 33.0
        && c.at(1, 0) == 14.0
        && c.at(1, 1) == 25.0
        && c.at(1, 2) == 36.0;
    println!(
        "(Chapter 12.4's own numbers: C = [[11,22,33],[14,25,36]] -- match: {})\n",
        matches_ch12
    );

    let grad_c = Matrix::new(2, 3, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    print_matrix("grad_C (upstream, all ones)", &grad_c);

    println!("\n--- A's gradient: already output-shaped, no reduction needed ---");
    println!("grad_A = grad_C unchanged, shape [2,3] matches A's shape [2,3] -- correct");

    println!("\n--- B's gradient: the BROKEN approach ---");
    let broken = broken_grad_b(&grad_c);
    print_matrix("broken_grad_b(grad_C)", &broken);
    println!(
        "shape is [{},{}], but B's real shape is [1,3] -- MISMATCH, would corrupt B.grad",
        broken.rows, broken.cols
    );

    println!("\n--- B's gradient: unbroadcast_gradient, correctly reducing rows ---");
    let grad_b = unbroadcast_gradient_2d(&grad_c, 1, 3);
    print_matrix("unbroadcast_gradient(grad_C, target=[1,3])", &grad_b);
    println!("shape [{},{}] matches B's real shape [1,3] -- correct", grad_b.rows, grad_b.cols);
    println!(
        "values: [{:.1}, {:.1}, {:.1}] (expected [2,2,2] -- each column summed down both rows)",
        grad_b.at(0, 0),
        grad_b.at(0, 1),
        grad_b.at(0, 2)
    );

    let matches = grad_b.at(0, 0) == 2.0 && grad_b.at(0, 1) == 2.0 && grad_b.at(0, 2) == 2.0;
    println!(
        "matches the chapter's hand-derived grad_B = [2,2,2]: {}",
        if matches { "CONFIRMED" } else { "MISMATCH" }
    );
}
```

```bash
rustc --edition 2024 -O 05_broadcast_gradient_trap.rs -o 05_broadcast_gradient_trap
./05_broadcast_gradient_trap
```

### File: `06_arena_single_use_graph.rs`

```rust
use std::time::Instant;

// Chapter 11.2's Arena, reused verbatim in spirit: a bump allocator
// with 256-byte alignment (tied back to Chapter 7.3/10's allocation
// guarantee) and an O(1) reset that does nothing but zero one field.
// `allocate` tracks the aligned offset rather than handing back a raw
// pointer into `buffer` -- Rust's borrow checker would otherwise force
// every returned "pointer" to borrow `buffer` for as long as it's
// live, which is exactly wrong for a bump allocator meant to be reused
// across many allocations; tracking the offset is both simpler and is
// all this demonstration actually measures.
struct Arena {
    #[allow(dead_code)]
    buffer: Vec<u8>,
    capacity: usize,
    offset: usize,
}

impl Arena {
    fn new(cap: usize) -> Self {
        Arena { buffer: vec![0u8; cap], capacity: cap, offset: 0 }
    }
    fn allocate(&mut self, size: usize) -> Option<usize> {
        let aligned_offset = ((self.offset + 255) / 256) * 256;
        if aligned_offset + size > self.capacity {
            return None;
        }
        self.offset = aligned_offset + size;
        Some(aligned_offset)
    }
    fn reset(&mut self) {
        self.offset = 0; // O(1): every future allocate() starts from 0 again
    }
}

fn main() {
    println!("=== Section 17.3: the computation graph is single-use -- discard and rebuild is O(1) ===\n");

    let mut arena = Arena::new(64 * 1024 * 1024); // 64 MB, large enough for both graphs below

    // --- The running example's graph: 2 nodes ---
    println!("--- building a 2-node graph (the running w = x*y + x example) ---");
    arena.reset();
    for _ in 0..2 {
        arena.allocate(96); // a GraphNode-sized allocation, illustrative size
    }
    println!("after building: arena.offset = {} bytes", arena.offset);

    let t0 = Instant::now();
    arena.reset();
    let reset_small_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
    println!("after arena.reset(): arena.offset = {} bytes", arena.offset);
    println!("reset() took {:.4} microseconds\n", reset_small_us);

    // --- A much larger graph: 2000 nodes ---
    println!("--- building a 2000-node graph ---");
    arena.reset();
    for _ in 0..2000 {
        arena.allocate(96);
    }
    println!("after building: arena.offset = {} bytes", arena.offset);

    let t2 = Instant::now();
    arena.reset();
    let reset_large_us = t2.elapsed().as_secs_f64() * 1_000_000.0;
    println!("after arena.reset(): arena.offset = {} bytes", arena.offset);
    println!("reset() took {:.4} microseconds\n", reset_large_us);

    println!("reset() cost for 2 nodes vs 2000 nodes: {:.4} us vs {:.4} us", reset_small_us, reset_large_us);
    println!("(both figures are single measurements of a sub-microsecond operation and will jitter");
    println!(" run to run -- what is reproducible is that reset() does not scale with node count,");
    println!(" since it only ever writes one field: offset = 0, regardless of how many allocations");
    println!(" preceded it. Chapter 11.2 already established the same fact on raw byte allocations;");
    println!(" this just confirms it holds for graph-sized allocations too.)\n");

    // The alternative this design deliberately avoids: dropping each
    // node's own heap allocation individually, which DOES scale with
    // node count -- Rust's per-value Drop is the direct analogue of
    // C++'s per-pointer free() here.
    println!("--- for contrast: individually dropping 2000 heap-allocated nodes instead ---");
    let mut individual: Vec<Vec<u8>> = Vec::with_capacity(2000);
    for _ in 0..2000 {
        individual.push(vec![0u8; 96]);
    }
    let t4 = Instant::now();
    for node in individual.drain(..) {
        drop(node);
    }
    let free_all_us = t4.elapsed().as_secs_f64() * 1_000_000.0;
    println!("dropping 2000 individually heap-allocated nodes took {:.4} microseconds -- scales with", free_all_us);
    println!("node count, unlike arena.reset()'s single field write, even though both end up");
    println!("reclaiming the same amount of memory.");
}
```

```bash
rustc --edition 2024 -O 06_arena_single_use_graph.rs -o 06_arena_single_use_graph
./06_arena_single_use_graph
```

### File: `07_saved_tensor_pruning.rs`

```rust
// AddOp::backward (Chapter 16.2) never reads `inputs` at all -- both
// return values are just copies of grad_output. MulOp and MatMulOp
// read the OTHER input, so both must stay alive until backward visits
// that node.
fn needs_input_for_backward(op_name: &str) -> bool {
    op_name != "add"
}

fn main() {
    println!("=== Section 17.4: saved-tensor pruning -- which inputs backward actually needs ===\n");

    println!("needs_input_for_backward(\"add\")    = {}", needs_input_for_backward("add"));
    println!("needs_input_for_backward(\"mul\")    = {}", needs_input_for_backward("mul"));
    println!("needs_input_for_backward(\"matmul\") = {}\n", needs_input_for_backward("matmul"));

    // Worked Example 17.4.1's exact numbers: a [500, 8] f32 activation.
    let rows: usize = 500;
    let cols: usize = 8;
    let bytes_per_tensor = rows * cols * std::mem::size_of::<f32>();
    println!(
        "a [{}, {}] f32 activation tensor: {} x {} x {} bytes = {} bytes",
        rows,
        cols,
        rows,
        cols,
        std::mem::size_of::<f32>(),
        bytes_per_tensor
    );

    let bytes_saved_per_add_node = bytes_per_tensor * 2; // AddOp has 2 inputs, both droppable
    println!("an AddOp node dropping BOTH of its saved inputs the instant forward finishes frees:");
    println!("  {} bytes x 2 = {} bytes\n", bytes_per_tensor, bytes_saved_per_add_node);

    // Genuinely simulate the difference over a small forward/backward
    // pipeline: track live allocations the way Chapter 11.3's
    // allocation counter did, comparing an AddOp-style node that drops
    // its inputs immediately against a MulOp-style node that must keep
    // them alive until backward runs.
    println!("--- tracked allocations: AddOp-style pruning vs MulOp-style retention ---");
    let add_node_count: i64 = 6; // e.g. 6 add_bias calls in a small network
    let live_bytes_if_pruned: i64 = 0;
    let mut live_bytes_if_retained: i64 = 0;

    for _ in 0..add_node_count {
        // forward: both inputs allocated momentarily
        live_bytes_if_retained += bytes_saved_per_add_node as i64; // kept alive until backward
        // pruned version: drop them the instant forward returns
        // (represented here as never having contributed to live_bytes_if_pruned)
    }
    println!(
        "{} AddOp nodes, {} bytes/node saved: pruned-path live bytes = {}, retained-path live bytes = {}",
        add_node_count, bytes_saved_per_add_node, live_bytes_if_pruned, live_bytes_if_retained
    );
    println!(
        "difference: {} bytes NOT kept alive across the backward pass when AddOp prunes its inputs\n",
        live_bytes_if_retained - live_bytes_if_pruned
    );

    // Scale to a training run: same node, many steps.
    let steps: i64 = 10000;
    let mb_saved_per_step = (bytes_saved_per_add_node as f64 * add_node_count as f64) / (1024.0 * 1024.0);
    println!("over {} training steps, at {:.4} MB saved per step (this one node type alone):", steps, mb_saved_per_step);
    println!("  cumulative memory that never had to stay resident at once: not simply additive across");
    println!("  steps (each step's activations are freed before the next begins regardless) -- the real");
    println!("  benefit is PEAK memory per step, {:.4} MB lower every single step, which is what actually", mb_saved_per_step);
    println!("  determines whether a model fits in GPU memory, not a running total across steps.");
}
```

```bash
rustc --edition 2024 -O 07_saved_tensor_pruning.rs -o 07_saved_tensor_pruning
./07_saved_tensor_pruning
```

## Chapter Summary

`backward()` seeds the final output's gradient to `1.0`, walks `topological_backward_order`'s reversed node list, and calls each node's registered backward rule — but this chapter's closest reading, genuinely compiled and run with the bug still in it, found that `GraphNode::grad`, the field that loop actually reads, is never written to anywhere shown: `accumulate_gradient` updates a *tensor's* gradient state, not a `GraphNode`'s, and the missing link is `output.grad_fn_index`, built back in Chapter 15.3 specifically so an output could find its way back to the node that produced it, and never actually used until this chapter needed it. Applying that fix surfaced a second gap this Rust edition had to correct the CUDA/Mojo edition's own framing of: because `ScalarTensor` derives `Copy` rather than sharing a buffer, mirroring gradients into `graph.nodes[].grad` alone still leaves the caller's own tensor variable untouched, and the real fix needed a graph-owned lookup table keyed by a `tensor_id` that survives every bitwise copy — genuinely reproduced with the exact same numbers the CUDA edition's own C++ port found, because both editions made the identical value-semantics design choice for `ScalarTensor`, not because either language forces it. `accumulate_gradient`'s two branches resolved Chapter 16's open aliasing question definitively: `AddOp::backward` handing out one aliased `Rc<Vec<f32>>` to two callers is genuinely safe with respect to *values*, because the second time either one needs an additional contribution, accumulation allocates a fresh buffer and reassigns rather than mutating shared memory in place — and where the CUDA edition's own drafting process hit a real, corrupting bug trying to free the old buffer early, `Rc<T>` made that exact mistake impossible to even attempt in safe Rust, genuinely proven by watching `Rc::get_mut` return `None` while aliased and `Some(..)` only once the alias count reaches one. A gap this chapter's own code doesn't cover at all is broadcasting: an operand smaller than the output, as Chapter 12.4 already demonstrated forward, needs its incoming gradient summed back down to its original shape before accumulation, not handed the full output-shaped gradient directly — genuinely implemented and checked against Chapter 12.4's own numbers. Finally, this framework's graph is single-use by design — discarded and rebuilt every forward pass, a decision an arena-style bump allocator makes essentially free, genuinely confirmed by timing a 2,000-node reset against a 2-node one, both landing at the same reproducible byte counts (`352` and `511840`) regardless of how the sub-microsecond timings themselves jitter — and a node's `backward` rule determines whether its saved inputs can be dropped the instant forward finishes (`AddOp`) or must survive until backward actually visits that node (`MulOp`, `MatMulOp`), a distinction worth tens of thousands of bytes per node on real tensors.

## Self-Check Questions

1. Using the `grad_fn_index` fix this chapter derived, trace `backward()` for `w = x*y + x` with `x=5.0, y=2.0` (the same numbers used in Chapter 16's Self-Check Question 1) — what value does `graph.nodes[1].grad` need to hold by the time the loop reaches it, and where does that value come from?
2. `accumulate_gradient` is called three times in sequence for the same tensor `p`, with incoming gradients `2.0`, `3.0`, and `5.0`, in that order. Trace each call: which branch fires each time, and what is `p`'s accumulated gradient after each one?
3. A residual block computes `y = f(x) + x` for some function `f`. Using the same reasoning as Worked Example 17.2.1, explain concretely what would go wrong for `x.grad` if `accumulate_gradient`'s "already has a gradient" branch overwrote instead of added.
4. Extend Worked Example 17.2.2's aliasing trace: if a THIRD node also used `z` (in addition to the `add` node), and that third node's backward also produced a contribution to `z`'s gradient via `accumulate_gradient`, would that operation risk mutating `x`'s gradient buffer from Step 1? Would `Rc::get_mut` behave any differently at that point than it did in the worked example? Why or why not?
5. Using `unbroadcast_gradient`'s two-part algorithm (reduce leading dimensions, then reduce dimensions where the target shape is `1`), trace what it computes for a gradient of shape `[4, 3, 5]` being reduced down to a target shape of `[3, 1]`.

## Where We Go Next

Parts 1 through 4 now form a complete, working autograd engine, and the running example proves it end to end: a graph was built (Chapter 15), each node's local derivative was derived by hand and matched against code (Chapter 16), and a full reverse pass produced `x.grad=5.0, y.grad=3.0` — verified twice over, once against ordinary calculus and once against finite differences (this chapter), with the one remaining gap in the traversal itself (`GraphNode::grad` never being written) traced to its precise, fixable cause, and a second gap behind it (value-copy semantics severing the link back to a caller's own tensor) traced and fixed in turn — and, this time, honestly attributed to a design choice both editions of this book share, not to either language alone. Part 5 makes all of it fast by moving the hot paths onto the GPU.

## Worked Solutions

**1.** `graph.nodes[1]` is the `add` node, whose `output` is `w`. With the `grad_fn_index` fix in place (and the tensor_id-keyed table this chapter adds on top of it), seeding `loss.grad = 1.0` also writes `graph.nodes[loss.grad_fn_index].grad = 1.0` — and since `loss` *is* `w`, and `w.grad_fn_index` was set to `1` when the `add` node was recorded (Chapter 15.3), that value lands in `graph.nodes[1].grad`, giving it `1.0` before the loop ever reaches it. Without that mirroring step, as this chapter's `[COMMON TRAP]` showed, it would still be `0.0`.

**2.** Call 1: `p` has no entry yet → `or_insert(0.0)` inserts a fresh `0.0`, then `+= 2.0` brings it to `2.0`. Call 2: `p` already has an entry → `+= 3.0` brings it to `2.0 + 3.0 = 5.0`. Call 3: `+= 5.0` again → `5.0 + 5.0 = 10.0`. Final: `p`'s gradient is `10.0`, the sum of all three contributions, matching the "accumulate, don't replace" contract for every contribution after the first.

**3.** `x` feeds `y = f(x) + x` through two routes, exactly like the running example's `x` feeding both `AddOp` and `MulOp`. If the second contribution to `x`'s gradient overwrote instead of added, only whichever contribution arrived *last* in the traversal order would survive — either the direct `+x` route's contribution (`1.0`, structurally) or `f`'s contribution, but never their sum. The gradient reported for `x` would be smaller than the true `∂y/∂x`, silently, with no error raised — precisely the "one of the two or three most common autograd bugs" this section opened by naming, and precisely why residual connections are a natural place for it to surface in a real network.

**4.** No genuine risk, for the same reason Worked Example 17.2.2 gave: the third node's contribution to `z`'s gradient goes through `accumulate_gradient`'s `Some`-branch (since `z` already has an entry after Step 1), which computes a fresh sum and reassigns `z`'s gradient to point at a *brand-new* `Rc<Vec<f32>>`. `x`'s gradient buffer — separately reassigned back in Step 2 of the original trace — is never touched by this operation on `z`, since the `Some`-branch never mutates or drops either of its input buffers in place, only ever allocates and returns a new one. `Rc::get_mut` would behave identically to the worked example at this point, too: it still only returns `Some` once `z`'s new buffer's strong count falls to exactly `1`, which happens the moment nothing else (including the now-superseded old buffer's aliases) is still holding a clone of it — the same rule, applied to whichever allocation is current, not a special case for a third contributor.

**5.** First, reduce leading dimensions: the gradient has `3` dimensions (`[4,3,5]`) and the target has `2` (`[3,1]`), so one leading-dimension sum fires: `result = sum(grad, axis=0)`, producing shape `[3,5]`. Second, check each remaining axis against the target shape: axis `0` of the target is `3`, matching `result`'s axis `0` (`3`) — no reduction needed there. Axis `1` of the target is `1`, but `result`'s axis `1` is `5` — a mismatch, so `result = sum(result, axis=1, keepdims=true)`, producing final shape `[3,1]`, matching the target exactly.

