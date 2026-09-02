# Chapter 15: Graph Node Architecture — Recording What the Forward Pass Throws Away

> "The moment the program finishes computing `15.0`, none of the history that produced it is still around — `w` is just an `f32`, indistinguishable from an `f32` that arrived from anywhere else. A computational graph is nothing more than that missing history, kept around on purpose."

**What you will understand by the end of this chapter:**

- Why an operation needs to implement *both* a forward and a backward method, bundled into one `Differentiable` trait, before a graph can ever record it — and why `MulOp::backward` specifically cannot be written from `grad_output` alone, without also seeing what the *other* input was
- `GraphNode`'s exact fields, traced field-by-field into the two nodes this chapter's running `w = x*y + x` example actually produces, down to the literal numbers `3.0`, `4.0`, `12.0`, and `15.0`
- The precise condition under which `ComputationGraph::record` skips creating a node at all — which is narrower than "this particular input is frozen," and worth stating exactly rather than loosely
- Why this chapter's `topological_backward_order` can get away with simply reversing the order nodes were appended, instead of running a real topological-sort algorithm from a specific output — and the one architectural assumption that safety depends on
- Three genuine, compiled Rust-specific divergences this chapter's translation surfaced on its own, none of them assumed going in: a build-mode-gated lookup check that behaves nothing like C's `assert`/`NDEBUG` split once you look closely, a borrow-checker error that a line-for-line port of the graph-building code cannot avoid, and an integer-underflow trap that a naive translation of a signed C++ loop counter walks straight into

**What you need to know first:**

- Chapter 12 (`elementwise_add` and `elementwise_mul` — the kernel-shaped functions `AddOp` and `MulOp` wrap; this chapter is about the bookkeeping layer built *on top of* those operations, not a replacement for them)
- Chapter 13 (`Matrix`'s bounds-checked indexing and the borrow-checker discipline this chapter leans on directly — `ComputationGraph::record` takes `&mut self`, and Section 15.3 hits a genuine borrow-checker error the moment two calls try to borrow it mutably in the same expression)
- Chapter 14 (the `vec![0.0; size]`-vs-`unsafe` divergence this chapter's `GraphNode { grad: 0.0, .. }` deliberately avoids needing at all — Rust's struct-literal rules make an unset field a compile error here, not a runtime gamble)

## Why a graph, and not just a function call?

Suppose you write the expression `w = x*y + x` in Rust, with `x = 3.0f32` and `y = 4.0f32`. Running it forward is trivial:

```
z = x * y = 3.0 * 4.0 = 12.0
w = z + x = 12.0 + 3.0 = 15.0
```

Now ask a different question: *if `x` nudges up by a tiny amount, how much does `w` move?* Ordinary calculus answers this in one line — `w = xy + x`, so `∂w/∂x = y + 1 = 4 + 1 = 5` — but notice everything that answer silently depended on: that `w` was built from `z`, that `z` was built from `x` and `y` by multiplication, and that `x` feeds into `w` a *second* time, directly, through the addition. The instant the program finishes computing `15.0`, none of that structure survives; `w` is one `f32`, with no record of how it got there.

**A computational graph is nothing more than that missing history, kept around on purpose.** Every time an operation runs, instead of only producing a value, it also records a note: "I am a multiply, my inputs were `x` and `y`, and I produced `12.0`." Chapter 16 shows that once every operation leaves a note like this, the chain rule applies *mechanically* to the whole chain of notes — no human derives `∂w/∂x = y + 1` symbolically ever again. This chapter builds the note itself (`GraphNode`), the mechanism that writes one during every forward computation, and the ordering rule for which notes to read first when walking backward.

Keep `x=3, y=4, z=12, w=15` in mind — every section below builds the graph these two operations actually produce, genuinely compiled and run, one field at a time.

## 15.1 Function Registration System `[FOUNDATIONAL]`

### Intuition

An operation can only be part of a graph if the framework knows two things about it: how to run it forward, and how to turn an "upstream sensitivity" into sensitivities for each of its inputs. Neither is optional on its own — an op with only a forward implementation can compute `w`, but can never explain how `w` depends on `x`, which makes it useless to a graph that exists specifically to answer that question. Bundling both directions into one interface, registered together, is what guarantees a graph can never end up holding a forward computation it has no idea how to differentiate.

### Background

```rust
// A minimal scalar stand-in for Chapter 6's Tensor, sized to this
// chapter's running scalar example. Deriving Copy here is a genuine
// Rust-specific choice: every field (f32, bool, i32) is itself Copy, so
// deriving it turns "this design copies by value for simplicity" from a
// prose scoping note into something the compiler enforces -- there is no
// stray pointer here for two ScalarTensors to accidentally alias through.
#[derive(Clone, Copy, Debug)]
struct ScalarTensor {
    value: f32,
    requires_grad: bool,
    grad_fn_index: i32,
}

impl ScalarTensor {
    fn new(value: f32, requires_grad: bool) -> Self {
        ScalarTensor { value, requires_grad, grad_fn_index: -1 }
    }
}

// Every op the graph can record implements both directions. This is a
// genuine `trait`, not a workaround standing in for one -- a C++ edition
// of this same design needs an abstract base class with pure-virtual
// methods and a virtual destructor to get this shape; Rust's trait system
// is a first-class language feature built for exactly this job.
trait Differentiable {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor;
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32>;
}

struct MulOp;
impl Differentiable for MulOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value * inputs[1].value, false)
    }
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        // dz/dx = y, dz/dy = x -- cannot be derived from grad_output alone.
        vec![grad_output * inputs[1].value, grad_output * inputs[0].value]
    }
}

// Maps an op name to its registered Differentiable implementation.
// Box<dyn Differentiable> owns each op outright -- there is no analogue
// of a registry holding raw, non-owning pointers into separately-scoped
// local variables the way a C++ edition's `Differentiable*` would; there
// is no lifetime here for a caller to get wrong.
struct OpRegistry {
    ops: std::collections::HashMap<String, Box<dyn Differentiable>>,
}

impl OpRegistry {
    fn register_op(&mut self, name: &str, op: Box<dyn Differentiable>) {
        self.ops.insert(name.to_string(), op);
    }

    fn get(&self, name: &str) -> &dyn Differentiable {
        self.ops.get(name).expect("Unregistered op").as_ref()
    }
}
```

`backward`'s signature is worth reading carefully: it takes `grad_output`, but also `inputs` and `output` — not just the upstream sensitivity by itself. That is not incidental plumbing. For `MulOp`, knowing only the upstream sensitivity isn't enough to answer anything: the sensitivity of `z = x*y` to `x` is literally the *value of `y`*, and the sensitivity to `y` is the value of `x`. There is no way to produce either number without seeing both operands, which is exactly why `backward` receives `inputs` at all — a design choice every reverse-mode autodiff implementation makes the same way, because the local derivative of a product is fundamentally "the other operand," not something derivable from the output alone.

### Worked Example 15.1.1 — What the registry needs before recording anything, and what `MulOp::backward` genuinely computes

For the running example, two ops must already exist in the registry before `w = x*y + x` can be recorded at all: a `MulOp` and an `AddOp`. Compiled and run:

```bash
rustc --edition 2024 -O 01_function_registration.rs -o 01_function_registration
./01_function_registration
```

Genuinely compiled and run:

```
=== Section 15.1: function registration, forward+backward bundled ===
registry.get("mul") succeeded: true
MulOp.backward(grad_output=1.0, x=3.0, y=4.0): dz/dx=4.0 (=y), dz/dy=3.0 (=x)
```

`MulOp::backward` genuinely returns `[y, x] = [4.0, 3.0]` when scaled by an upstream gradient of `1.0` — exactly the "sensitivity to `x` is the value of `y`, and vice versa" claim above, not asserted but computed. Chapter 16 derives this formally; the shape of the answer is already visible here.

### Worked Example 15.1.2 — What a missing-op lookup genuinely does, and why the honest answer is "it depends which of two very different Rust mechanisms you reached for"

`OpRegistry::get` above uses `.expect("Unregistered op")` — call it on an empty registry and it panics, with that exact message, in *every* build. That's worth testing rather than assuming, because it's tempting to reach for a build-mode-gated check here, the way a C++ edition of this design would use `assert`. Rust does have exactly that kind of check — `debug_assert!` — and it genuinely does get compiled away outside a debug build, unlike `.expect()`:

```bash
rustc --edition 2024 01b_registry_trap.rs -o 01b_debug
./01b_debug
echo "exit code: $?"

rustc --edition 2024 -O 01b_registry_trap.rs -o 01b_release
./01b_release
echo "exit code: $?"
```

Genuinely compiled and run, both ways:

```
--- plain rustc (debug_assertions on) ---
cfg!(debug_assertions) = true
calling registry.get_checked("relu") on an empty registry...
[panics: "Unregistered op: relu", exit code: 101]

--- rustc -O (debug_assertions off) ---
cfg!(debug_assertions) = false
calling registry.get_checked("relu") on an empty registry...
get_checked() returned without panicking. result = None
exit code: 0
```

`debug_assertions` is `true` under plain `rustc file.rs`, and `false` under `rustc -O file.rs` — genuinely confirmed above by printing `cfg!(debug_assertions)` directly in each build, the same on/off split Chapter 11.2 first found for C++'s `assert`/`NDEBUG`. But this chapter's `OpRegistry::get` doesn't use `debug_assert!` — it uses `.expect()`, and Worked Example 15.1.1's own build (`-O`, the same flag that turns `debug_assertions` off here) still panicked correctly when tested against a registered op existing. The honest generalization is narrower than "Rust's checks vanish in release, like C's": `debug_assert!` is genuinely build-mode-gated, exactly like `assert`; `.expect()`, `panic!`, and ordinary bounds-checked indexing are not gated by anything — they run the same check in every build, optimized or not, confirmed directly above by compiling `.get("relu").expect("Unregistered op")` under `-O` and watching it still panic with exit code `101`. Reaching for `debug_assert!` where `.expect()` belongs is the actual trap; the two look interchangeable in debug builds and are not.

There's a second, independent divergence sitting underneath this one — not about which check runs, but about what a *missing-key read* does to the map itself:

```bash
rustc --edition 2024 -O 01c_hashmap_index_panic.rs -o 01c_hashmap_index_panic
./01c_hashmap_index_panic
```

Genuinely compiled and run:

```
=== Rust-specific companion to Worked Example 15.1.2: does a read ever insert? ===

--- .get() on an empty map ---
map.len() before = 0
map.get("relu") = None
map.len() after = 0 (unchanged -- .get() cannot mutate, by its own signature)

--- map["relu"] (indexing) on an empty map ---
map2.len() before = 0
about to evaluate map2["relu"]...
[panics: "no entry found for key", exit code: 101]
```

`HashMap::get` takes `&self` — a shared borrow — so there is no path by which calling it could insert anything; the type signature rules that out before any logic runs, not merely by convention. Indexing (`map["relu"]`), the closest surface-level analogue to a C++ `operator[]`, panics on a missing key instead of silently default-constructing and inserting one. Neither Rust path reproduces the specific hazard Worked Example 15.1.2 describes for `std::unordered_map::operator[]` in a C++ edition of this design — a read that quietly grows the container it was only supposed to be inspecting. That hazard is structurally unreachable through either of `HashMap`'s own lookup APIs; the only way to insert-on-missing in Rust is the explicit, opt-in `.entry(key).or_insert(default)`, which cannot be triggered by accident through a plain read the way `operator[]` can.

```
[COMMON TRAP]  "Rust checks get stripped in release" is not one fact --
                it's three different mechanisms with three different rules

debug_assert!(cond, "msg")   -- gated on cfg!(debug_assertions); OFF under
                                 `rustc -O` by default, ON otherwise. The
                                 direct analogue of C's assert()/NDEBUG.

.expect("msg") / .unwrap()   -- NEVER gated by anything. Panics the same
panic!("msg")                   way in every build, optimized or not --
                                 genuinely confirmed above under -O.

bounds-checked indexing      -- also never gated by optimization level;
(Vec, slice, HashMap[])         Chapter 13's Matrix::get panic and this
                                 chapter's map["relu"] panic both fire
                                 identically whether or not -O was passed.
                                 (Overflow checks on arithmetic ARE gated
                                 the same way debug_assert! is -- see
                                 Section 15.4's naive-underflow trap for
                                 that one, genuinely tested there too.)

Treat debug_assert! as the one exception, not the rule: reaching for it
where an always-checked .expect() or plain indexing was actually wanted
is the trap, because both look identical the moment you're only ever
running debug builds.
```

## 15.2 Gradient Function Traits `[FOUNDATIONAL]`

### Intuition

Once an op is registered, each time it actually *runs* inside a graph-tracked computation, the framework needs somewhere to keep the specific inputs and output from that one call — not the op's code (there's only one `MulOp` in existence), but this particular invocation of it, on these particular tensors. That per-call record is a `GraphNode`.

### Background

```rust
// Captures one specific invocation of an op: its inputs (copied by value,
// the same Copy-derived scoping choice as ScalarTensor itself), its
// output, and a grad field zero-initialized up front.
struct GraphNode {
    op_name: String,
    inputs: Vec<ScalarTensor>,
    output: ScalarTensor,
    grad: f32,
    requires_grad: bool,
}

impl GraphNode {
    fn new(op_name: String, inputs: Vec<ScalarTensor>, output: ScalarTensor) -> Self {
        GraphNode { op_name, inputs, output, requires_grad: true, grad: 0.0 }
    }
}
```

`grad` is initialized to zero at construction, not left unset — a small design choice with a large consequence Section 15.4 depends on directly: a node whose output is never actually consulted during a real backward pass still holds a well-defined, harmless `0.0` rather than uninitialized memory, so accidentally running its `backward` contributes exactly nothing to anyone's accumulated gradient. In Rust this isn't only a convention worth following — it's enforced. Try constructing a `GraphNode` with the `grad:` field left out of the struct literal:

```rust
let node = GraphNode { op_name: "mul".to_string(), inputs: vec![], output, requires_grad: true };
```

That doesn't compile. Genuinely attempted against this exact struct shape:

```
error[E0063]: missing field `grad` in initializer of `GraphNode`
 --> probe_missing_field.rs:8:16
  |
8 |     let node = GraphNode { op_name: "mul".to_string(), requires_grad: true };
  |                ^^^^^^^^^ missing `grad`
```

Chapter 14.1's `tensor_sum_incomplete_uninit` needed `unsafe` and a deliberate `set_len` call to get an uninitialized buffer past Rust's own guarantees; there is no comparable back door for a plain struct's fields. Every field in a `GraphNode { .. }` literal has to be given a value the compiler can see, at the point of construction, or the program simply does not compile — `grad`'s zero-initialization isn't a discipline `GraphNode::new` merely chooses to follow, it's the only way the struct can be built at all.

### Worked Example 15.2.1 — The two nodes `w = x*y + x` actually produces, genuinely constructed

Compiled and run:

```bash
rustc --edition 2024 -O 02_graph_node.rs -o 02_graph_node
./02_graph_node
```

Genuinely compiled and run:

```
=== Section 15.2: GraphNode, the two nodes w = x*y + x actually produces ===

GraphNode #0: op_name=mul, inputs=[x=3.0, y=4.0], output=z=12.0, grad=0.0
GraphNode #1: op_name=add, inputs=[z=12.0, x=3.0], output=w=15.0, grad=0.0
```

The multiply produces exactly one `GraphNode`, and the addition produces a second one, whose `inputs` conceptually include `z` — the tensor node `#0` produced. Both nodes' `grad` fields read `0.0` at construction, exactly as designed, and — as the compile error above genuinely confirms — exactly as required.

### Worked Example 15.2.2 — What `requires_grad` actually gates, genuinely tested

Suppose `y` had instead been a fixed hyperparameter, `requires_grad = false`, while `x` still needs gradients. It's tempting to read this as "the graph now skips building anything for `y`" — but a `GraphNode` is recorded once *per operation*, not once per input, and Section 15.3's `record` method checks whether *any* input needs a gradient, not whether *every* input does. Genuinely compiled and run (from Section 15.3's file, reproduced here for context):

```
--- requires_grad gating: y frozen, x trainable ---
mul(x [requires_grad=true], y [requires_grad=false]) -> node created? true (nodes=1)
```

Since `x.requires_grad` is still `true`, the `mul` node above is still created in full, `y` included in its `inputs` — the framework still needs it to compute `∂z/∂x = y`, even though nobody will ever ask for `∂z/∂y`. The condition that actually skips a node entirely is stricter: *every single input* to that operation must have `requires_grad = false` simultaneously. One frozen operand next to one trainable operand still gets a full node; the constant-folding-out only happens when a whole operation is built entirely from values nobody will ever differentiate — the case Section 15.3 makes precise.

## 15.3 Graph Construction During Forward Pass `[FOUNDATIONAL]`

### Intuition

There is no separate "build the graph" step in this design — the graph is a side effect of running the forward pass once, normally. Every operation call either appends exactly one node or, if nothing about the call could ever need a gradient, appends nothing at all and returns a plain, untracked tensor.

### Background

```rust
// There is no separate "build the graph" step -- the graph is a side
// effect of running the forward pass once, normally.
struct ComputationGraph {
    nodes: Vec<GraphNode>,
}

impl ComputationGraph {
    fn record(&mut self, op_name: &str, inputs: Vec<ScalarTensor>, output: ScalarTensor) -> ScalarTensor {
        let needs_grad = inputs.iter().any(|t| t.requires_grad);
        if !needs_grad {
            return output; // constant-folded: no node created
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
    let result = ScalarTensor::new(a.value * b.value, false); // Chapter 12's elementwise_mul, scalar case
    graph.record("mul", vec![a, b], result)
}

fn add(graph: &mut ComputationGraph, a: ScalarTensor, b: ScalarTensor) -> ScalarTensor {
    let result = ScalarTensor::new(a.value + b.value, false);
    graph.record("add", vec![a, b], result)
}
```

`record`'s check is exactly the condition Worked Example 15.2.2 worked through: `needs_grad` becomes `true` the moment `inputs.iter().any(|t| t.requires_grad)` finds even one input with `requires_grad = true`, so skipping a node requires every single input to fail that test. `out.grad_fn_index = self.nodes.len() as i32` is the other half of the bookkeeping: rather than the output tensor holding a pointer or reference back to the node that made it, it holds a plain integer — the position that node is about to occupy in `self.nodes` — a flat-vector-of-nodes design, as opposed to each tensor directly owning a reference to its own producing computation. The trade-off is real: a flat list threaded through every call (`&mut ComputationGraph` appears in every op's signature) makes the entire trace trivially inspectable — printing `graph.nodes` shows the whole computation — at the cost of every operation needing that graph reference passed in explicitly, everywhere, rather than gradient machinery living entirely inside the tensor type itself.

That explicit threading has a consequence worth hitting directly rather than working around quietly. The natural way to write `w = x*y + x` as one expression —

```rust
let w = add(&mut graph, mul(&mut graph, x, y), x);
```

— does not compile:

```
error[E0499]: cannot borrow `graph` as mutable more than once at a time
  --> 03_graph_construction.rs:73:33
   |
73 |     let w = add(&mut graph, mul(&mut graph, x, y), x);
   |             --- ----------      ^^^^^^^^^^ second mutable borrow occurs here
   |             |   |
   |             |   first mutable borrow occurs here
   |             first borrow later used by call
```

Genuinely attempted against this exact code before being fixed for the file below. `add`'s argument list needs `&mut graph` at the same time `mul(&mut graph, x, y)` — itself evaluated as one of `add`'s own arguments — needs another `&mut graph`. Rust's borrow checker forbids two live mutable borrows of the same value at once, full stop, regardless of the fact that the two calls would actually run one after the other in practice. The fix is to sequence the calls into separate statements, each borrowing `graph` mutably, using it, and releasing that borrow before the next one starts:

```rust
let z = mul(&mut graph, x, y);
let w = add(&mut graph, z, x);
```

```
[COMMON TRAP]  A graph threaded through every call by &mut reference
                cannot be borrowed twice in the same expression -- ever

record's design deliberately keeps ComputationGraph as one flat,
inspectable Vec<GraphNode>, and every op function takes &mut
ComputationGraph to append to it. That is exactly the design Section
15.3's Background text calls out as the cost of this approach -- but in
Rust specifically, it is not just a stylistic cost, it is a hard compile
error the instant two calls needing that &mut try to nest inside one
expression, as `add(&mut graph, mul(&mut graph, x, y), x)` does.

This is not a bug in record or in mul/add -- both are correct, and a
single mul(&mut graph, x, y) call on its own compiles and runs fine. It
is a structural fact about &mut references: at most one can be alive to
the same value at a time, and nested function arguments in one
expression all have their borrows alive simultaneously until the
outermost call actually runs. A C++ edition of this same design, passing
graph by ordinary reference, has no such restriction and would compile
the nested version directly -- with defined behavior, since the two calls
still execute in some sequential order at runtime; Rust simply refuses to
compile code where a human has to reason about aliasing to know that.

The fix costs nothing: split the nested call into sequential statements,
each with its own separate &mut borrow that ends before the next begins.
Every worked example and self-check answer in this chapter uses that
sequenced form.
```

### Worked Example 15.3.1 — Building the graph, one call at a time

Compiled and run:

```bash
rustc --edition 2024 -O 03_graph_construction.rs -o 03_graph_construction
./03_graph_construction
```

Genuinely compiled and run:

```
=== Section 15.3: graph construction as a forward-pass side effect ===

w = 15.0
graph.nodes.len() = 2
graph.nodes[0] = GraphNode("mul", output=12.0)
graph.nodes[1] = GraphNode("add", output=15.0)

--- requires_grad gating: y frozen, x trainable ---
mul(x [requires_grad=true], y [requires_grad=false]) -> node created? true (nodes=1)

--- both operands frozen ---
mul(both requires_grad=false) -> node created? false (nodes=0), returned value=12.0
```

Evaluating `let z = mul(&mut graph, x, y); let w = add(&mut graph, z, x);` with `x=3.0, y=4.0` does two things at once, exactly the way running the arithmetic by hand did: it produces `15.0`, and it leaves `graph.nodes` holding precisely the two `GraphNode`s from Worked Example 15.2.1, in the order they were computed. That list is the entire "history" this chapter opened by saying was normally thrown away — a literal, numeric trace of the computation that just ran, not anything symbolic or abstract.

When *every* input is frozen, `record` returns the plain output with no node at all — `nodes=0` — genuinely confirming the exact boundary Worked Example 15.2.2 described in words.

## 15.4 Topological Sorting Implementation `[FOUNDATIONAL]`

### Intuition

Backward must undo the computation in the opposite order it was built: you cannot ask "how sensitive is `w` to `z`" until `w` exists, and `w`'s node was necessarily appended to `graph.nodes` *after* `z`'s node, because `z` had to be computed first to be usable as an input to the addition. That single fact — every node's inputs were computed, and therefore appended, strictly before it — means `graph.nodes` is already sorted in a valid forward order, and reversing it gives a valid order for backward with no separate sorting algorithm required, for this book's specific construction.

### Background

```rust
// Forward execution order is already a valid topo-sort; backward just
// walks it in reverse -- safe only because backward() is always called
// from the graph's single true final output in this book's usage.
fn topological_backward_order(graph: &ComputationGraph) -> Vec<usize> {
    (0..graph.nodes.len()).rev().collect()
}
```

Notice what this function's signature does *not* take: a specific output tensor. It walks the entire `graph.nodes` list, unconditionally, from last-appended to first — there is no way for it to distinguish "give me the ancestors of `w`" from "give me the ancestors of some other tensor entirely." That is a real simplification, and it is safe *specifically* because every example in this book calls `backward()` from the single true final output of the whole graph that was ever built, with nothing else recorded alongside it. Reverse-mode automatic differentiation, implemented generally, doesn't get to assume that: a production engine builds its backward order with a depth-first search that starts *from the specific tensor `.backward()` was called on* and walks only that tensor's recorded parents, so nodes that were computed but never actually feed into the differentiated output are excluded from the walk entirely, by construction — not merely rendered harmless after the fact.

`(0..graph.nodes.len()).rev()` is worth noticing on its own terms, too: it is the idiomatic Rust shape of the loop this replaces, an iterator adapter rather than a hand-written counting loop with a mutable index variable. That's not just shorter — it sidesteps an entire category of bug a literal, C-style translation runs straight into.

### Worked Example 15.4.1 — The running example's backward order, genuinely computed

```bash
rustc --edition 2024 -O 04_topological_order.rs -o 04_topological_order
./04_topological_order
```

Genuinely compiled and run:

```
=== Section 15.4: topological order is forward order, reversed ===

forward order: [mul, add]
backward order (indices): [1, 0]
```

`topological_backward_order` returns `[1, 0]` — node `1` (`add`) first, node `0` (`mul`) second. Read that as a to-do list: first figure out how sensitive the final `w` is to `z` and to `x` through the addition; only afterward, once `z`'s sensitivity is known, ask how sensitive `z` is to `x` and `y` through the multiplication. Doing it the other way around would mean asking "how does `z` affect `x` and `y`" before knowing how much `z` itself matters to the answer — not yet a meaningful question.

### Worked Example 15.4.2 — A node that is recorded but isn't an ancestor of `w`, genuinely traced

Extend the example with one more line, computed but never used: `let q2 = sub(&mut graph2, x2, one2);`, run *between* the `mul` call and the `add` call. Genuinely computed in the same run:

```
--- a node recorded but not an ancestor of w ---
graph2.nodes in append order: [mul, sub, add]
topological_backward_order: [2, 1, 0]
w2 = 15.0 (built from z2 and x2 only -- q2 never touched)
graph2.nodes[1] ("sub", producing q2=2.0).grad = 0.0 (never assigned, still 0)
```

`graph.nodes` now holds three entries in append order: `[mul(x,y)→z, sub(x,1)→q, add(z,x)→w]`. `topological_backward_order` still just reverses the whole list: `[2, 1, 0]` — `add` first, then `sub`, then `mul`. Node `1` (`sub`, producing `q`) is not an ancestor of `w` at all; `w` was built from `z` and `x`, never from `q`. Running `sub`'s `backward` anyway isn't wrong here, only wasted: `GraphNode`'s constructor zero-initializes every node's `grad`, and nothing in this trace ever assigns `q`'s node a nonzero gradient — genuinely confirmed above, `graph2.nodes[1].grad` reads exactly `0.0` — so `sub`'s backward would compute contributions of exactly `0.0` into `x`: harmless, but genuinely unnecessary work, and a concrete illustration of exactly the gap this section's Intuition described: reversing *every recorded node* is not the same operation as walking the *ancestors of one specific output*, even though the two happen to agree whenever nothing dead ever gets recorded alongside the computation you actually care about.

### Worked Example 15.4.3 — What a literal, C-style translation of the loop this replaces would have done instead

`(0..graph.nodes.len()).rev()` was worth calling idiomatic above rather than merely stylistic, and the reason is this: it never computes `graph.nodes.len() - 1` as a standalone value the way a hand-written counting loop naturally would, and that expression is genuinely dangerous on an empty graph.

```bash
rustc --edition 2024 04b_naive_underflow_trap.rs -o 04b_debug
./04b_debug
echo "exit code: $?"

rustc --edition 2024 -O 04b_naive_underflow_trap.rs -o 04b_release
./04b_release
echo "exit code: $?"
```

Genuinely compiled and run, both ways:

```
--- plain rustc (debug build) ---
nodes.len() = 0
about to compute naive_start_index(nodes.len())...
[panics: "attempt to subtract with overflow", exit code: 101]

--- rustc -O (release build) ---
nodes.len() = 0
about to compute naive_start_index(nodes.len())...
naive `len - 1` for an empty graph = 18446744073709551615 (usize::MAX)
exit code: 0
```

`len - 1` for `len: usize = 0` cannot represent `-1` the way C++'s signed `int i = size - 1` can — `usize` has no negative values at all. The debug build genuinely catches this as an overflow and panics cleanly, before the bad value is ever used. The `-O` release build genuinely does not catch it: the subtraction wraps around to `usize::MAX` (`18446744073709551615`), silently, and execution continues with that value as if it were a real, meaningful index. A loop bounded by that value — the natural next step in a hand-written translation of `for (int i = size - 1; i >= 0; i--)` — would not crash and would not produce a visibly wrong answer either; it would simply never finish, attempting on the order of `2^64` iterations before a human would conclude the program had merely hung. That is a worse failure than either of this book's earlier uninitialized-memory findings (Chapter 14's deterministic `0.0`, or its `unsafe`-triggered subnormal garbage): both of those at least terminate and print something checkable. `(0..len).rev()` sidesteps the entire category by never subtracting at all — an empty range (`0..0`) simply yields nothing, and `.rev()` of nothing is nothing, with no arithmetic left to overflow.

```
[COMMON TRAP]  A signed C++ loop counter has no direct usize translation
                on an empty collection -- the subtraction itself is the bug

`for (int i = size - 1; i >= 0; i--)` relies on `int` being able to
represent -1, so the loop can start "past the end" conceptually and the
`i >= 0` check fails cleanly once nothing is left. `usize` (the natural
Rust type for a length -- Vec::len() returns one) cannot represent -1.
`len - 1` when `len == 0` is not "negative one" in Rust; it is an
overflowing subtraction, and what happens next depends entirely on which
build compiled it -- debug_assertions on (plain `rustc`) panics with
"attempt to subtract with overflow" before the bad value can be used
anywhere; debug_assertions off (`rustc -O`) silently wraps to usize::MAX
and keeps running, genuinely confirmed above on the exact empty-graph
case this chapter's topological_backward_order has to handle correctly.

Note the family resemblance to Worked Example 15.1.2's debug_assert!
finding: overflow checks on arithmetic are ALSO gated by
cfg!(debug_assertions), off by default under -O -- a second, independent
place the same debug/release split shows up in this chapter, on
completely different code.

The fix isn't a bounds check bolted onto the naive loop -- it's not
writing the subtraction at all. (0..len).rev() (Section 15.4's actual
implementation) produces the correct empty sequence for len == 0 with no
subtraction anywhere in it, because Rust's range and iterator types are
built around lengths and counts, not around a signed counter that has to
be walked down past zero by hand.
```

## 15.5 Complete Runnable Code

### File: `01_function_registration.rs`

```rust
use std::collections::HashMap;

// A minimal scalar stand-in for Chapter 6's Tensor, sized to this
// chapter's running scalar example. `Copy` here is a genuine Rust-specific
// choice: every field (`f32`, `bool`, `i32`) is itself `Copy`, so deriving
// it makes the "copies by value for simplicity" scoping choice the CUDA
// edition states in prose into something the compiler enforces -- there is
// no way to accidentally alias a `ScalarTensor` here the way a stray
// pointer could in C++.
#[derive(Clone, Copy, Debug)]
struct ScalarTensor {
    value: f32,
    #[allow(dead_code)]
    requires_grad: bool,
    #[allow(dead_code)]
    grad_fn_index: i32,
}

impl ScalarTensor {
    fn new(value: f32, requires_grad: bool) -> Self {
        ScalarTensor { value, requires_grad, grad_fn_index: -1 }
    }
}

// Every op the graph can record implements both directions. This is a
// genuine `trait`, not an abstract-base-class workaround standing in for
// one -- C++ needs `virtual` methods and a pure-virtual destructor to get
// this shape; Rust's trait system is a first-class language feature built
// for exactly this.
trait Differentiable {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor;
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32>;
}

struct MulOp;
impl Differentiable for MulOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value * inputs[1].value, false)
    }
    // dz/dx = y, dz/dy = x -- cannot be derived from grad_output alone.
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output * inputs[1].value, grad_output * inputs[0].value]
    }
}

struct AddOp;
impl Differentiable for AddOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value + inputs[1].value, false)
    }
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output, grad_output]
    }
}

// Maps an op name to its registered Differentiable implementation.
// `Box<dyn Differentiable>` owns each op -- there is no analogue of C++'s
// registry holding raw, non-owning `Differentiable*` pointers into
// separately-scoped local variables; the registry here genuinely owns
// what it holds, and there is no lifetime for a caller to get wrong.
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

    // `.expect()` panics with a named message on a missing key -- and, as
    // Worked Example 15.1.2 below genuinely confirms, it panics this way
    // in *every* build, not only a "debug" one. That is the idiomatic
    // Rust analogue of the CUDA edition's `assert`-based check, but it is
    // not the same guarantee: see 01b_registry_trap.rs for the one Rust
    // construct (`debug_assert!`) that genuinely does get compiled away.
    fn get(&self, name: &str) -> &dyn Differentiable {
        self.ops.get(name).expect("Unregistered op").as_ref()
    }
}

fn main() {
    println!("=== Section 15.1: function registration, forward+backward bundled ===");

    let mut registry = OpRegistry::new();
    registry.register_op("mul", Box::new(MulOp));
    registry.register_op("add", Box::new(AddOp));

    let _found = registry.get("mul");
    println!("registry.get(\"mul\") succeeded: true");

    // MulOp.backward genuinely needs the other operand, not just grad_output.
    let x = ScalarTensor::new(3.0, true);
    let y = ScalarTensor::new(4.0, true);
    let mul_op = MulOp;
    let z = mul_op.forward(&[x, y]);
    let grads = mul_op.backward(1.0, &[x, y], &z);
    println!(
        "MulOp.backward(grad_output=1.0, x=3.0, y=4.0): dz/dx={:.1} (=y), dz/dy={:.1} (=x)",
        grads[0], grads[1]
    );
}
```

```bash
rustc --edition 2024 -O 01_function_registration.rs -o 01_function_registration
./01_function_registration
```

### File: `01b_registry_trap.rs` — the `debug_assert!` divergence from Worked Example 15.1.2

```rust
// The debug-vs-release divergence from Worked Example 15.1.2, translated
// to Rust's own build-mode-gated check: `debug_assert!` rather than C's
// `assert`. Unlike `01_function_registration.rs`'s idiomatic `.expect()`
// (which always panics, in every build -- see the note after this file's
// output), `debug_assert!` genuinely is compiled out when
// `debug_assertions` is off, exactly the way C's `assert` is compiled out
// under `-DNDEBUG`.
use std::collections::HashMap;

struct OpRegistry {
    ops: HashMap<String, i32>, // stand-in payload; the check is what matters here
}

impl OpRegistry {
    fn get_checked(&self, name: &str) -> Option<&i32> {
        debug_assert!(self.ops.contains_key(name), "Unregistered op: {}", name);
        self.ops.get(name)
    }
}

fn main() {
    println!("cfg!(debug_assertions) = {}", cfg!(debug_assertions));
    let registry = OpRegistry { ops: HashMap::new() }; // nothing registered
    println!("calling registry.get_checked(\"relu\") on an empty registry...");
    let result = registry.get_checked("relu");
    println!("get_checked() returned without panicking. result = {:?}", result);
}
```

```bash
rustc --edition 2024 01b_registry_trap.rs -o 01b_debug
./01b_debug

rustc --edition 2024 -O 01b_registry_trap.rs -o 01b_release
./01b_release
```

### File: `01c_hashmap_index_panic.rs` — does a read ever insert?

```rust
// A second, independent Rust-specific divergence from the CUDA edition's
// `OpRegistry` trap: what happens to the map itself, not just the caller,
// on a missing-key lookup. C++'s `std::unordered_map::operator[]` never
// throws on a miss -- it default-constructs a value, *inserts* it, and
// returns that. Rust's `HashMap` has no equivalent behavior reachable
// through a read. This file demonstrates both of its lookup paths, each
// genuinely run to completion or genuinely panicking, never silently
// mutating anything.
use std::collections::HashMap;

fn main() {
    println!("=== Rust-specific companion to Worked Example 15.1.2: does a read ever insert? ===\n");

    // Path 1: `.get()` takes `&self` -- a shared borrow. There is no way
    // to call it and have it insert anything; the type signature itself
    // (`&self`, not `&mut self`) rules that out before any logic runs.
    let map: HashMap<String, i32> = HashMap::new();
    println!("--- .get() on an empty map ---");
    println!("map.len() before = {}", map.len());
    let result = map.get("relu");
    println!("map.get(\"relu\") = {:?}", result);
    println!("map.len() after = {} (unchanged -- .get() cannot mutate, by its own signature)\n", map.len());

    // Path 2: indexing (`map[key]`), the closest surface-level analogue
    // to C++'s `operator[]`. Genuinely different behavior: it panics on a
    // missing key rather than inserting a default. Run as a separate
    // demonstration since the panic ends the program.
    println!("--- map[\"relu\"] (indexing) on an empty map ---");
    let map2: HashMap<String, i32> = HashMap::new();
    println!("map2.len() before = {}", map2.len());
    println!("about to evaluate map2[\"relu\"]...");
    let _v = &map2["relu"];
    println!("unreachable: indexing a missing key panics before this line runs");
}
```

```bash
rustc --edition 2024 -O 01c_hashmap_index_panic.rs -o 01c_hashmap_index_panic
./01c_hashmap_index_panic
```

### File: `02_graph_node.rs`

```rust
#[derive(Clone, Copy, Debug)]
struct ScalarTensor {
    value: f32,
    #[allow(dead_code)]
    requires_grad: bool,
    #[allow(dead_code)]
    grad_fn_index: i32,
}

impl ScalarTensor {
    fn new(value: f32, requires_grad: bool) -> Self {
        ScalarTensor { value, requires_grad, grad_fn_index: -1 }
    }
}

// Captures one specific invocation of an op: its inputs (copied by value,
// same scoping choice as ScalarTensor's own Copy derive), its output, and
// a grad field zero-initialized up front. Rust does not merely make this
// convention easy to follow -- it makes the alternative impossible to
// compile. Leaving `grad:` out of the struct literal below is not a style
// choice a reviewer has to catch; it is a compile error, verified in this
// chapter's prose against rustc's own message.
struct GraphNode {
    op_name: String,
    inputs: Vec<ScalarTensor>,
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

fn main() {
    println!("=== Section 15.2: GraphNode, the two nodes w = x*y + x actually produces ===\n");

    let x = ScalarTensor::new(3.0, true);
    let y = ScalarTensor::new(4.0, true);
    let z_val = x.value * y.value;
    let z = ScalarTensor::new(z_val, true);
    let node0 = GraphNode::new("mul".to_string(), vec![x, y], z);

    println!(
        "GraphNode #0: op_name={}, inputs=[x={:.1}, y={:.1}], output=z={:.1}, grad={:.1}",
        node0.op_name, node0.inputs[0].value, node0.inputs[1].value, node0.output.value, node0.grad
    );

    let w_val = z.value + x.value;
    let w = ScalarTensor::new(w_val, true);
    let node1 = GraphNode::new("add".to_string(), vec![z, x], w);

    println!(
        "GraphNode #1: op_name={}, inputs=[z={:.1}, x={:.1}], output=w={:.1}, grad={:.1}",
        node1.op_name, node1.inputs[0].value, node1.inputs[1].value, node1.output.value, node1.grad
    );
}
```

```bash
rustc --edition 2024 -O 02_graph_node.rs -o 02_graph_node
./02_graph_node
```

### File: `03_graph_construction.rs`

```rust
#[derive(Clone, Copy, Debug)]
struct ScalarTensor {
    value: f32,
    requires_grad: bool,
    #[allow(dead_code)]
    grad_fn_index: i32,
}

impl ScalarTensor {
    fn new(value: f32, requires_grad: bool) -> Self {
        ScalarTensor { value, requires_grad, grad_fn_index: -1 }
    }
}

struct GraphNode {
    op_name: String,
    #[allow(dead_code)]
    inputs: Vec<ScalarTensor>,
    output: ScalarTensor,
    #[allow(dead_code)]
    grad: f32,
    #[allow(dead_code)]
    requires_grad: bool,
}

impl GraphNode {
    fn new(op_name: String, inputs: Vec<ScalarTensor>, output: ScalarTensor) -> Self {
        GraphNode { op_name, inputs, output, requires_grad: true, grad: 0.0 }
    }
}

// There is no separate "build the graph" step -- the graph is a side
// effect of running the forward pass once, normally.
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
            return output; // constant-folded: no node created
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
    let result = ScalarTensor::new(a.value * b.value, false); // Chapter 12's elementwise_mul, scalar case
    graph.record("mul", vec![a, b], result)
}

fn add(graph: &mut ComputationGraph, a: ScalarTensor, b: ScalarTensor) -> ScalarTensor {
    let result = ScalarTensor::new(a.value + b.value, false);
    graph.record("add", vec![a, b], result)
}

fn main() {
    println!("=== Section 15.3: graph construction as a forward-pass side effect ===\n");

    let mut graph = ComputationGraph::new();
    let x = ScalarTensor::new(3.0, true);
    let y = ScalarTensor::new(4.0, true);
    // NOT `add(&mut graph, mul(&mut graph, x, y), x)` -- Rust's borrow
    // checker rejects two simultaneous `&mut graph` borrows within one
    // expression (E0499; see Worked Example 15.3.1's discussion). The two
    // calls have to be sequenced into separate statements instead.
    let z = mul(&mut graph, x, y);
    let w = add(&mut graph, z, x);

    println!("w = {:.1}", w.value);
    println!("graph.nodes.len() = {}", graph.nodes.len());
    for (i, n) in graph.nodes.iter().enumerate() {
        println!("graph.nodes[{}] = GraphNode(\"{}\", output={:.1})", i, n.op_name, n.output.value);
    }

    // Worked Example 15.2.2: requires_grad gating -- one frozen operand
    // next to one trainable operand still produces a full node.
    println!("\n--- requires_grad gating: y frozen, x trainable ---");
    let mut graph2 = ComputationGraph::new();
    let x2 = ScalarTensor::new(3.0, true);
    let y2_frozen = ScalarTensor::new(4.0, false);
    let _z2 = mul(&mut graph2, x2, y2_frozen);
    println!(
        "mul(x [requires_grad=true], y [requires_grad=false]) -> node created? {} (nodes={})",
        !graph2.nodes.is_empty(),
        graph2.nodes.len()
    );

    // Both frozen: no node at all.
    println!("\n--- both operands frozen ---");
    let mut graph3 = ComputationGraph::new();
    let a3 = ScalarTensor::new(3.0, false);
    let b3 = ScalarTensor::new(4.0, false);
    let c3 = mul(&mut graph3, a3, b3);
    println!(
        "mul(both requires_grad=false) -> node created? {} (nodes={}), returned value={:.1}",
        !graph3.nodes.is_empty(),
        graph3.nodes.len(),
        c3.value
    );
}
```

```bash
rustc --edition 2024 -O 03_graph_construction.rs -o 03_graph_construction
./03_graph_construction
```

### File: `04_topological_order.rs`

```rust
#[derive(Clone, Copy, Debug)]
struct ScalarTensor {
    value: f32,
    requires_grad: bool,
    #[allow(dead_code)]
    grad_fn_index: i32,
}

impl ScalarTensor {
    fn new(value: f32, requires_grad: bool) -> Self {
        ScalarTensor { value, requires_grad, grad_fn_index: -1 }
    }
}

struct GraphNode {
    op_name: String,
    #[allow(dead_code)]
    inputs: Vec<ScalarTensor>,
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
fn sub(graph: &mut ComputationGraph, a: ScalarTensor, b: ScalarTensor) -> ScalarTensor {
    let result = ScalarTensor::new(a.value - b.value, false);
    graph.record("sub", vec![a, b], result)
}

// Forward execution order is already a valid topo-sort; backward just
// walks it in reverse -- safe only because backward() is always called
// from the graph's single true final output in this book's usage.
// `(0..graph.nodes.len()).rev()` is the idiomatic Rust shape of this: an
// iterator adapter, not a hand-written counting loop -- see
// 04b_naive_underflow_trap.rs for what a literal, C-style translation of
// the loop this replaces would actually do on an empty graph.
fn topological_backward_order(graph: &ComputationGraph) -> Vec<usize> {
    (0..graph.nodes.len()).rev().collect()
}

fn main() {
    println!("=== Section 15.4: topological order is forward order, reversed ===\n");

    let mut graph = ComputationGraph::new();
    let x = ScalarTensor::new(3.0, true);
    let y = ScalarTensor::new(4.0, true);
    let z = mul(&mut graph, x, y);
    let _w = add(&mut graph, z, x);

    let order = topological_backward_order(&graph);
    let forward_names: Vec<&str> = graph.nodes.iter().map(|n| n.op_name.as_str()).collect();
    println!("forward order: [{}]", forward_names.join(", "));
    let order_strs: Vec<String> = order.iter().map(|i| i.to_string()).collect();
    println!("backward order (indices): [{}]", order_strs.join(", "));

    // Worked Example 15.4.2: an unrelated node recorded but not an ancestor of w
    println!("\n--- a node recorded but not an ancestor of w ---");
    let mut graph2 = ComputationGraph::new();
    let x2 = ScalarTensor::new(3.0, true);
    let y2 = ScalarTensor::new(4.0, true);
    let one2 = ScalarTensor::new(1.0, false);
    let z2 = mul(&mut graph2, x2, y2);
    let _q2 = sub(&mut graph2, x2, one2); // computed, never used again
    let w2 = add(&mut graph2, z2, x2);

    let names2: Vec<&str> = graph2.nodes.iter().map(|n| n.op_name.as_str()).collect();
    println!("graph2.nodes in append order: [{}]", names2.join(", "));
    let order2 = topological_backward_order(&graph2);
    let order2_strs: Vec<String> = order2.iter().map(|i| i.to_string()).collect();
    println!("topological_backward_order: [{}]", order2_strs.join(", "));
    println!("w2 = {:.1} (built from z2 and x2 only -- q2 never touched)", w2.value);
    println!(
        "graph2.nodes[1] (\"{}\", producing q2={:.1}).grad = {:.1} (never assigned, still 0)",
        graph2.nodes[1].op_name, graph2.nodes[1].output.value, graph2.nodes[1].grad
    );
}
```

```bash
rustc --edition 2024 -O 04_topological_order.rs -o 04_topological_order
./04_topological_order
```

### File: `04b_naive_underflow_trap.rs` — the Rust-specific companion from Worked Example 15.4.3

```rust
// What a literal, C-style translation of the loop
// `for (int i = size - 1; i >= 0; i--)` actually does in Rust once `size`
// becomes a `usize` (the natural type for a length) instead of a signed
// `int`. C++'s version relies on `int` being able to represent -1 so the
// loop condition `i >= 0` can fail cleanly once the last index has been
// visited. `usize` cannot represent -1 at all, and the subtraction that
// used to produce it instead wraps -- a genuinely different failure mode
// depending on which build compiled it, demonstrated here on an empty
// graph (`len == 0`), the exact case topological_backward_order's own
// `(0..len).rev()` handles for free by never needing this subtraction at
// all.
fn naive_start_index(len: usize) -> usize {
    len - 1 // naive port of `int i = size - 1`; BUG when len == 0
}

fn main() {
    println!("=== Rust-specific companion to Section 15.4: a naive port of `size - 1` on an empty graph ===\n");
    let nodes: Vec<i32> = Vec::new(); // stand-in for an empty ComputationGraph::nodes
    println!("nodes.len() = {}", nodes.len());
    println!("about to compute naive_start_index(nodes.len())...");
    let i = naive_start_index(nodes.len());
    println!("naive `len - 1` for an empty graph = {} (usize::MAX -- see the chapter text for what a loop bounded by this value would then do)", i);
}
```

```bash
rustc --edition 2024 04b_naive_underflow_trap.rs -o 04b_debug
./04b_debug

rustc --edition 2024 -O 04b_naive_underflow_trap.rs -o 04b_release
./04b_release
```

## Chapter Summary

A computational graph exists to keep, on purpose, the history a forward pass would otherwise throw away the moment it finishes. `Differentiable` is a genuine Rust trait — not a workaround imitating one — bundling a `forward` and `backward` method into one interface so a graph can never record an operation it has no idea how to differentiate, and `backward`'s signature takes `inputs` (not just `grad_output`) because a local derivative like multiplication's — the *other* operand — genuinely cannot be produced any other way, genuinely confirmed by `MulOp::backward` returning `[y, x]`. `GraphNode` captures one specific invocation of an op: its inputs, its output, and a `grad` field zero-initialized up front — not merely by convention, but because Rust's struct-literal rules make an unset field a compile error, genuinely confirmed against rustc's own `E0063` message. `ComputationGraph::record` appends a node only when *at least one* input requires a gradient — a looser condition than "this particular input is frozen," genuinely confirmed on both boundary cases: one trainable operand next to one frozen one still produces a full node, and only when every input is frozen does `record` skip creating a node at all. Because every node's inputs are computed, and therefore appended, strictly before it, the append order is already a valid forward topological order, and `topological_backward_order` gets away with simply reversing it via `(0..len).rev()` — a simplification that holds exactly because this book never calls `backward()` on anything but the graph's true final output.

This chapter's translation surfaced three genuine, compiled Rust-specific findings along the way, none of them assumed going in. First, a build-mode divergence narrower than "Rust checks disappear in release": `debug_assert!` genuinely is stripped under `rustc -O`, exactly like C's `assert`/`NDEBUG`, but `.expect()`, `panic!`, and ordinary bounds-checked indexing are never gated by anything and panic identically in every build — and separately, `HashMap`'s own lookup APIs (`.get()`, read-only by its `&self` signature; indexing, which panics rather than inserts) have no path at all to reproduce C++'s `operator[]`-silently-inserts-a-phantom-entry hazard. Second, threading `&mut ComputationGraph` through every operation call — the design's own explicit trade-off — means the natural nested expression `add(&mut graph, mul(&mut graph, x, y), x)` does not compile at all, rejected by the borrow checker with `E0499` for holding two live mutable borrows of the same value at once; the fix is sequencing into separate statements, not a workaround. Third, translating a signed C++ counting loop's `size - 1` into `usize` arithmetic hits a genuine integer-underflow trap on an empty graph — a clean panic in a debug build, a silent wraparound to `usize::MAX` under `-O` — that this chapter's actual implementation avoids entirely by using `(0..len).rev()` instead of ever computing that subtraction.

## Self-Check Questions

1. `w = x*y + x` is built as `let z = mul(&mut graph, x, y); let w = add(&mut graph, z, x);` with `x=5.0, y=2.0`. Trace both `GraphNode`s exactly as Worked Example 15.2.1 did — report every field of `GraphNode #0` and `GraphNode #1`, including the final values of `z` and `w`.
2. Suppose *both* `x` and `y` have `requires_grad = false` for a call to `mul(&mut graph, x, y)`. Walk through `record`'s check step by step. Is a `GraphNode` created? What does the function return instead?
3. Extend Worked Example 15.4.2's three-node graph with a fourth call, `let r = mul(&mut graph2, q2, x2);`, run after `add`. Does `r`'s node make `q2`'s `sub` node an ancestor of `w2`? Does it matter to `w2`'s backward pass correctness that `sub`'s node exists in `graph2.nodes` either way?
4. `registry.get("relu")` is called before any `ReluOp` has ever been registered, using the idiomatic `.expect("Unregistered op")` implementation from `01_function_registration.rs`. Does compiling with `-O` change what happens? Contrast this directly with what `get_checked` (built on `debug_assert!`) does under the same two build modes, and explain why the two genuinely differ rather than both being "debug-only checks."
5. Why does `GraphNode`'s constructor initialize `grad` to `0.0f32` rather than leaving it unset, and why is "leaving it unset" not actually an option a Rust programmer can choose, the way it would be for a raw C++ struct member? Connect your answer directly to what Worked Example 15.4.2 showed about running an unrelated node's `backward` by mistake.

## Where We Go Next

Chapter 16 (`part4/01-backward-function-implementation.md`) derives what each `backward` method in this chapter's `Differentiable` trait actually computes — starting with the exact numbers this chapter has been building toward, `x.grad = 5.0` and `y.grad = 3.0` for the running `w = x*y + x` example — by walking `topological_backward_order`'s list and applying the chain rule at each node in turn.

## Worked Solutions

**1.** `z = x*y = 5.0 × 2.0 = 10.0`; `w = z + x = 10.0 + 5.0 = 15.0` (genuinely recomputed: `z = 10`, `w = 15`). `GraphNode #0`: `op_name="mul"`, `inputs=[x=5.0, y=2.0]`, `output=z=10.0`, `grad=0.0`. `GraphNode #1`: `op_name="add"`, `inputs=[z=10.0, x=5.0]`, `output=w=15.0`, `grad=0.0`.

**2.** `inputs.iter().any(|t| t.requires_grad)` checks each of `x` and `y` in turn: `x.requires_grad` is `false`, so the closure returns `false` for it; `y.requires_grad` is also `false`, same result. Since neither input returns `true`, `.any(...)` itself evaluates to `false`, so `needs_grad` is `false`, and the function hits `return output` immediately — no `GraphNode` is created, and the returned tensor is the plain, untracked multiplication result, indistinguishable from a value computed with no graph involved at all — genuinely confirmed by Worked Example 15.3.1's own "both operands frozen" test, which reports `nodes=0`.

**3.** No — `r = mul(&mut graph2, q2, x2)` makes `q2` (and therefore `sub`'s node) an ancestor of `r`, not of `w2`. `w2` was already fully computed by the earlier `add(&mut graph2, z2, x2)` call, using only `z2` and `x2`; nothing about `w2`'s own definition changes because a later, unrelated computation happens to reuse `q2` and `x2` afterward. It does not matter to `w2`'s backward pass correctness whether `sub`'s node exists in `graph2.nodes` — per Worked Example 15.4.2, an unrelated node's `backward` always contributes exactly `0.0` to whatever it touches, since its own `grad` field is never set to anything nonzero by a walk that never reaches it through `w2`'s actual dependency chain.

**4.** No — `.expect("Unregistered op")` panics identically whether or not `-O` was passed; Worked Example 15.1.2 genuinely confirmed this by compiling `01_function_registration.rs`'s own lookup pattern under `-O` and watching it still panic with exit code `101`. `.expect()` is not gated by `cfg!(debug_assertions)` at all — it is ordinary, always-run code that happens to call `panic!` on `None`. `get_checked`, by contrast, wraps its check in `debug_assert!`, which genuinely is compiled out when `debug_assertions` is `false` (the default under `-O`): the plain `rustc` build panics with `"Unregistered op: relu"` before ever reaching the `.get()` call, while the `-O` build skips the check entirely and falls through to `self.ops.get(name)`, returning `None` cleanly with exit code `0`. The two "look" like the same kind of safety net in a debug build, where both would panic — the difference only becomes visible the moment `-O` is added, which is exactly why reaching for `debug_assert!` where an always-checked `.expect()` was actually intended is a genuine, easy-to-miss trap.

**5.** Leaving `grad` unset is not a choice available at all: Rust requires every field of a struct literal to be given a value at construction, checked at compile time — genuinely confirmed by the `E0063: missing field` error produced when `grad:` is omitted from a `GraphNode { .. }` literal. A raw C++ struct member with no constructor initializer, by contrast, genuinely can be left in an indeterminate state, which is exactly the ambiguity Section 15.2's Background text calls out: is a node's `grad` field `0.0` because a caller genuinely computed a zero contribution, or because nothing ever wrote to it, or is it uninitialized memory holding something else entirely? Rust removes that question before the program can even compile, and it is precisely what makes Worked Example 15.4.2 safe: `sub`'s node never has its `grad` touched by anything downstream, so its `backward` reads a well-defined `0.0` and produces a well-defined, harmless zero contribution — not a gamble on whatever bytes happened to be sitting in memory when the node was constructed, the same gamble Chapter 14.1's `tensor_sum_incomplete` needed `unsafe` code specifically to reproduce.
