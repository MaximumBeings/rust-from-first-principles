# Chapter 16: Backward Function Implementation — The Chain Rule, One Node at a Time

> "Chapter 15 ended with a graph for `w = x*y + x` and a to-do list for backward: visit `add(z,x)→w` first, then `mul(x,y)→z`. This chapter works out, by hand, exactly what happens at each stop on that list — and only afterward writes the Rust that automates it."

**What you will understand by the end of this chapter:**

- The multivariable chain rule as literally "sum the contribution from every path" — traced on `x`'s two separate routes into `w`, reaching the same `∂w/∂x = 5` Chapter 15 got from plain calculus
- `AddOp` and `MulOp`'s exact backward rules, why `MulOp`'s local derivative is fundamentally *the other operand's value*, and a genuinely Rust-specific answer to the buffer-aliasing question this chapter's own `AddOp::backward` raises: idiomatic `Vec<f32>` cloning can't reproduce the hazard at all, and reproducing it on purpose takes an explicit `Rc<RefCell<...>>`
- Why `ExpOp` reads the cached forward `output` instead of recomputing `eˣ` — and why `GraphNode` had to store `output` at all, back in Chapter 15, for that shortcut to even be possible
- `MatMulOp`'s backward rule, `grad_output @ Bᵀ` and `Aᵀ @ grad_output`, derived from index-summation first principles rather than only asserted, and verified with real numbers on *both* gradients this chapter's running matrix example produces
- The rest of the registry a working framework actually needs: `SubOp`/`DivOp`/`PowOp`/`LogOp`/`SqrtOp` alongside `Add`/`Mul`, five activation and trigonometric gradients (`ReluOp`, `SigmoidOp`, `TanhOp`, `SinOp`, `CosOp`) each derived from its own local derivative, and backward rules for the two shapes of operation Part 2 hasn't differentiated yet — reductions (`SumOp`, tying back into Chapter 14.2's argmax tracking for `MaxOp`) and shape changes (`ReshapeOp`, `TransposeOp`)
- The implicit function theorem as an escape hatch for differentiating through an iterative numerical solver — a bisection search — without ever unrolling and differentiating a single one of its steps, including a genuine, honestly-reported instance of the bisection's own floating-point rounding drifting slightly between this edition and the CUDA edition's, purely from arithmetic order, not from the calculus

**What you need to know first:**

- Chapter 15 (the `Differentiable` trait's `backward` signature, `GraphNode`'s `inputs`/`output` fields, and `topological_backward_order`'s `[1, 0]` to-do list — this chapter is entirely about what happens at each stop on that list)
- Chapter 13.1 (matrix multiplication and the `X (2×3) @ M (3×2)` running example this chapter's `MatMulOp` backward reuses directly)
- Chapter 12 (`elementwise_add`, `elementwise_mul`, `elementwise_exp` — the forward operations every op in this chapter wraps)
- Chapter 13.2 and 13.3 (transpose and reshape's forward behavior — this chapter derives their backward rules directly from how those forward operations move, or don't move, data)
- Chapter 14.1 and 14.2 (`tensor_sum`'s tree reduction and `max_reduce_kernel`'s argmax tracking — this chapter derives the backward rule each one needs)

## Why a graph, and not just a function call?

Chapter 15 posed the question by pure calculus: if `x` moves by a tiny amount, how much does `w` move? `w = xy + x`, so `∂w/∂x = y + 1 = 5`. What that one-line answer hides is that `x` doesn't reach `w` by one route — it reaches it by two, and the multivariable chain rule's actual content is nothing more sophisticated than "trace every route separately, then add up what each one contributes."

```
        ┌── (Route 1: through the multiply) ──┐
        │                                      ▼
   x ───┼────────────────────────────────▶ [ mul ] ──▶ z ──▶ [ add ] ──▶ w
        │                                                       ▲
        └── (Route 2: directly) ───────────────────────────────┘
```

## 16.1 Chain Rule Implementation `[FOUNDATIONAL]`

### Intuition

Section by section, Chapters 15 and 16 have been building toward one question: given a sensitivity flowing *into* a node, how does the graph turn that into a sensitivity for each of the node's *inputs*? That is exactly what `Differentiable::backward` computes, and "dispatch to whichever op a node actually holds, without caring which one it is" is the entire content of `chain_rule_step`.

### Background

**Route 1 — through the multiply.** `x` feeds into `z = x*y`. If `x` increases by a small `Δx`, `z` increases by approximately `Δx · y`, since `∂z/∂x = y = 4`. That change in `z` then feeds into `w = z + x`, where `∂w/∂z = 1`, so it passes straight through unchanged: `w` moves by `Δx · y · 1 = 4·Δx`.

**Route 2 — directly.** `x` *also* feeds straight into the addition `w = z + x`, independent of `z`. That contributes an additional `∂w/∂x|_{direct} = 1`, so `w` moves by another `1·Δx`.

**Total.** Both routes act on `w` simultaneously, so their contributions add: `w` moves by `(4 + 1)·Δx = 5·Δx` — exactly `∂w/∂x = 5`, now arrived at by summing contributions along every path from `x` to `w` rather than by symbolic differentiation of the whole expression at once. This *sum-over-paths* rule is the entire content of the multivariable chain rule, and it is also, not coincidentally, exactly what the reverse graph traversal in Chapter 17 computes — one path's contribution per visit to a node that uses `x`, summed as they arrive.

In code, "the sensitivity flowing into a node, converted into a sensitivity for each of its inputs" is one dispatch call:

```rust
// Dispatch to the registered backward for this op. The caller
// (Chapter 17) adds each result into the corresponding input's
// accumulated .grad.
fn chain_rule_step(
    registry: &OpRegistry,
    op_name: &str,
    grad_output: f32,
    inputs: &[ScalarTensor],
    output: &ScalarTensor,
) -> Vec<f32> {
    let op = registry.get(op_name);
    op.backward(grad_output, inputs, output)
}
```

### Worked Example 16.1.1 — Reconciling two routes into one number, genuinely dispatched

Section 15.4's `topological_backward_order` says: visit `add(z,x)→w` first, then `mul(x,y)→z`. Visiting `add` first is exactly what makes Route 2's contribution (the *direct* `1·Δx`) available immediately, and visiting `mul` second is what makes Route 1's contribution (the `4·Δx` that had to flow *through* `z` first) available only after `z`'s own sensitivity is already known. Neither route can be skipped, and neither can run before the node that produces it — which is precisely why the visiting order from Chapter 15 isn't optional bookkeeping, it's a dependency the arithmetic itself imposes.

Compiled and run:

```bash
rustc --edition 2024 -O 01_chain_rule_dispatch.rs -o 01_chain_rule_dispatch
./01_chain_rule_dispatch
```

Genuinely compiled and run:

```
=== Section 16.1: chain rule as sum-over-paths, dispatched through chain_rule_step ===
x=3.0, y=4.0, z=x*y=12.0, w=z+x=15.0

chain_rule_step("add", grad_output=1.0, inputs=[z=12.0, x=3.0]) -> [1.0, 1.0]
  z.grad (from add) = 1.0
  x's Route 2 contribution (direct edge into add) = 1.0

chain_rule_step("mul", grad_output=1.0, inputs=[x=3.0, y=4.0]) -> [4.0, 3.0]
  x's Route 1 contribution (through the multiply) = 4.0
  y.grad (only one route) = 3.0

x.grad = Route2 + Route1 = 1.0 + 4.0 = 5.0
y.grad = 3.0
cross-check against Chapter 15's plain calculus: dw/dx = y+1 = 5.0, dw/dy = x = 3.0
```

`chain_rule_step` never inspects which `Differentiable` implementation `registry.get("add")` returns — it just calls `.backward(...)` through the trait object. That is what lets Chapter 17's reverse pass stay a fixed, short loop no matter how many ops Sections 16.2 through 16.6 add to the registry.

## 16.2 Element-wise Operation Gradients `[FOUNDATIONAL]`

### Intuition

`AddOp` and `MulOp` are Section 16.1's two routes made concrete. `Add`'s local derivative is `1` for both inputs — a sensitivity that passes straight through unchanged, which is exactly what "direct" meant in Route 2. `Mul`'s local derivative is *the other operand's value* — which is exactly why `Differentiable::backward` was written, back in Chapter 15, to receive `inputs` and not merely `grad_output`.

### Background

```rust
struct AddOp;
impl Differentiable for AddOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value + inputs[1].value, false)
    }
    // d(a+b)/da = 1, d(a+b)/db = 1 -- gradient passes through unchanged
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output, grad_output]
    }
}

struct MulOp;
impl Differentiable for MulOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value * inputs[1].value, false)
    }
    // d(a*b)/da = b, d(a*b)/db = a
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output * inputs[1].value, grad_output * inputs[0].value]
    }
}
```

### Worked Example 16.2.1 — `AddOp`, by hand

`w = z + x`. Backward starts by being handed the sensitivity of the *final* output with respect to `w` itself — the *seed* — which is `1.0` by definition (`w` is exactly as sensitive to itself as it is to itself). `∂w/∂z = 1` and `∂w/∂x = 1`, so both of `AddOp`'s inputs simply receive a copy of whatever sensitivity arrived: with a seed of `1.0`, `AddOp::backward` returns `[1.0, 1.0]` — the first `1.0` is `z`'s incoming gradient, the second is `x`'s *first* contribution, Route 2 from Section 16.1. Hold onto that `x: 1.0` — Chapter 17 adds a second contribution to it.

```
[COMMON TRAP]  AddOp::backward and the question of whether both
               returned gradients are truly independent

`vec![grad_output, grad_output]` in Chapter 15's ScalarTensor world is
completely safe: ScalarTensor's `value` is a plain `f32` field, and
`f32` is `Copy`, so returning it twice returns two genuinely
independent numbers, full stop. But a real framework built past this
book represents a gradient as a buffer, not a bare float, the way
Chapter 6.3's real Tensor or Chapter 11.1's RefCountedBuffer<T> would.
The honest question, tested rather than assumed: once grad_output is
buffer-backed, does "return two copies" stop meaning "two independent
buffers" and start meaning "two handles to the same one"?

The answer genuinely depends on which Rust type stands in for that
buffer -- and this is a case where the two natural choices give
opposite answers, both confirmed below rather than argued from
principle. `Vec<f32>` is owned data: `grad_output.to_vec()` (or
`.clone()`) allocates a fresh buffer and copies every byte into it,
confirmed by comparing `.as_ptr()` on the two results and finding
DIFFERENT addresses, and by mutating one and finding the other
genuinely unaffected. There is no way to call `.clone()` on a `Vec`
and get aliasing by accident -- ownership rules it out structurally,
the same way Chapter 15's E0063 finding ruled out an unset struct
field.

Reproducing the CUDA edition's actual hazard on purpose needs a
different type entirely: `Rc<RefCell<Vec<f32>>>`, Rust's explicit,
opt-in mechanism for two owners sharing one allocation with interior
mutability. `Rc::clone` genuinely does return a second HANDLE to the
identical buffer -- confirmed below by matching `.as_ptr()` addresses,
a strong count that climbs to 3, and a mutation through one handle
showing up in the other. The trap this section's title promises is
real, but only for a Rust program that reached for `Rc<RefCell<...>>`
specifically; the idiomatic default (`Vec<f32>`, `Clone`) is immune to
it by construction, which is a stronger guarantee than the CUDA
edition's own C++ code gets from `BufferTensor`'s raw pointer, where
copying the struct copies the pointer with no opt-in required at all.
```

### Worked Example 16.2.2 — `MulOp`, by hand

`z = x*y`. `∂z/∂x = y = 4` and `∂z/∂y = x = 3` — each input's local derivative is *the other* input's value, which is exactly why `backward` is passed `inputs` and not just `grad_output`. `MulOp` receives `z`'s gradient from the step above (`1.0`, since `∂w/∂z=1`), and multiplies it by each local derivative: `x` receives `1.0 × 4 = 4.0`, `y` receives `1.0 × 3 = 3.0`.

Add `x`'s two contributions from the two nodes — `1.0` from `AddOp` and `4.0` from `MulOp` — and the total is `5.0`. That is `x.grad`, and it is exactly `∂w/∂x = 5`, computed three different ways now: by plain calculus (Chapter 15), by tracing paths (Section 16.1), and now by literally running the two registered backward functions in order and adding what they hand back. `y` only appears in one node, so `y.grad = 3.0` directly, with no accumulation needed at all.

### Worked Example 16.2.3 — `ExpOp`, and the case that needs `output`

Neither `AddOp` nor `MulOp` needs the node's *output*, only its inputs — but some ops do, and it's worth seeing one before Chapter 17 explains why that matters for memory. Take `u = exp(x)` at `x = 1.0`. Forward: `u = e¹ ≈ 2.71828`. The derivative of `exp` is famously itself: `du/dx = eˣ = u ≈ 2.71828`. With an upstream seed of `1.0`, `ExpOp::backward` returns `1.0 × 2.71828 = 2.71828` — and it gets that `2.71828` by reading `output`, the value forward already computed, rather than recomputing `exp(1.0)` a second time:

```rust
struct ExpOp;
impl Differentiable for ExpOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value.exp(), false)
    }
    // d(e^x)/dx = e^x = output -- reuse the cached forward result
    // instead of recomputing inputs[0].value.exp() a second time.
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output * output.value]
    }
}
```

This is exactly why `GraphNode` in Chapter 15 stores `output` alongside `inputs` — `ExpOp` is proof that a backward rule can legitimately need the forward answer, not just the forward arguments, and a `GraphNode` that only kept `inputs` would make `ExpOp`'s shortcut impossible to write at all.

Compiled and run, all three worked examples above plus the genuine buffer-aliasing demonstration from the Common Trap:

```bash
rustc --edition 2024 -O 02_element_wise_gradients.rs -o 02_element_wise_gradients
./02_element_wise_gradients
```

Genuinely compiled and run:

```
=== Section 16.2: AddOp, MulOp, ExpOp backward, by hand ===
w = z + x = 15.0
AddOp.backward(grad_output=1.0, [z=12.0, x=3.0]) -> [1.0, 1.0]
  z's incoming gradient = 1.0, x's Route 2 contribution = 1.0

z = x * y = 12.0
MulOp.backward(grad_output=1.0, [x=3.0, y=4.0]) -> [4.0, 3.0]
  x.grad = 1.0 (AddOp) + 4.0 (MulOp) = 5.0
  y.grad = 3.0 (only one route)

--- ExpOp: reusing the cached forward output instead of recomputing exp(x) ---
u = exp(x) at x=1.0: u = 2.71828
ExpOp.backward(grad_output=1.0, output=2.71828) -> 2.71828
  (du/dx = e^x = output, read directly rather than recomputing exp(1.0) again)

--- COMMON TRAP: does AddOp.backward hand out the SAME buffer twice? ---
(idiomatic Rust: Vec<f32>, .clone()-based "return two copies")
grad_output_vec.as_ptr() = 0x5647d83a1dc0
returned z_grad.as_ptr() = 0x5647d83a1de0, value: 1
returned x_grad.as_ptr() = 0x5647d83a1e00, value: 1
same address? false -- NOT aliased: Vec::clone deep-copies the buffer
mutating x_grad[0] = 999.0 leaves z_grad[0] = 1 untouched

(reproducing the CUDA edition's aliasing on purpose: Rc<RefCell<Vec<f32>>>)
grad_output_rc data ptr = 0x5647d83a1e20
returned z_grad_rc data ptr = 0x5647d83a1e20, value: 1
returned x_grad_rc data ptr = 0x5647d83a1e20, value: 1
same address? true -- ALIASED (Rc::strong_count = 3)
mutating z_grad_rc[0] = 999.0 through one handle corrupts x_grad_rc[0] too: 999
```

(The printed pointer values are genuinely ASLR-dependent and will differ on a different run or machine; what is reproducible is the pattern each block shows — different addresses and no corruption for `Vec::clone`, identical addresses and genuine corruption for `Rc::clone`.)

## 16.3 Matrix Operation Gradients `[FOUNDATIONAL]`

### Intuition

Matrix multiplication's backward rule isn't a restatement of a forward formula the way `Add` and `Mul`'s were — every output entry of `Y = X @ M` depends on an entire row of `X` and an entire column of `M` (Chapter 13.1), so a single output's sensitivity has to be redistributed back across many input entries at once, not just one. The rule turns out to have a clean closed form, and it's worth deriving *why* before simply trusting it.

### Background

Write the forward pass in index form, the same way Chapter 13.1 did: `Y[i,j] = Σ_k X[i,k]·M[k,j]`. The chain rule says `∂L/∂X[i,k]` sums the contribution of `X[i,k]` through *every* output entry it participates in — and `X[i,k]` appears in `Y[i,j]` for every `j`, since it's row `i` of `X` that feeds every column of the output:

```
∂L/∂X[i,k] = Σ_j (∂L/∂Y[i,j] · ∂Y[i,j]/∂X[i,k])
           = Σ_j (∂L/∂Y[i,j] · M[k,j])
           = (∂L/∂Y @ Mᵀ)[i,k]
```

The middle step uses `∂Y[i,j]/∂X[i,k] = M[k,j]` directly from the forward formula — `X[i,k]` only ever multiplies `M[k,j]` inside the sum that produces `Y[i,j]`. The final step is just recognizing that "sum over `j` of `∂L/∂Y[i,j]` times `M[k,j]`" is precisely what a matrix product against `Mᵀ` computes, entry by entry. The symmetric derivation for `M` gives `∂L/∂M[k,j] = Σ_i (∂L/∂Y[i,j] · X[i,k]) = (Xᵀ @ ∂L/∂Y)[k,j]`. In code, reusing Chapter 13's `Matrix` struct and `matrix_multiply`/`transpose` functions:

```rust
struct MatMulBackwardResult {
    grad_a: Matrix,
    grad_b: Matrix,
}

fn matmul_backward(grad_output: &Matrix, a: &Matrix, b: &Matrix) -> MatMulBackwardResult {
    let bt = transpose(b);
    let at = transpose(a);
    MatMulBackwardResult { grad_a: matrix_multiply(grad_output, &bt), grad_b: matrix_multiply(&at, grad_output) }
}
```

`MatMulOp` operates on whole matrices, not single floats, so it is scoped here as a plain function rather than forced through `ScalarTensor`'s `Differentiable` trait — the same scoping choice Chapter 13 itself made for matrix operations generally.

### Worked Example 16.3.1 — `dL/dX`, with real numbers

Reuse Chapter 13.1's exact running example: `X (2×3) @ M (3×2) = Y`, where

```
Y = X @ M =  22  28
             49  64
```

Suppose `Y` feeds into a loss whose gradient with respect to every entry of `Y` happens to be `1` — `grad_output` is a `2×2` matrix of ones, the simplest possible upstream signal. `Mᵀ` (transposing the `3×2` `M`) is:

```
Mᵀ = 1  3  5
     2  4  6
```

`grad_output @ Mᵀ` — a `2×2` matrix of ones times the `2×3` `Mᵀ` above — gives a `2×3` result where every entry is just the *column sum* of `Mᵀ`, because multiplying by a row of ones sums the column it's dotted against:

```
dL/dX = (1·1+1·2)  (1·3+1·4)  (1·5+1·6)     3   7  11
        (1·1+1·2)  (1·3+1·4)  (1·5+1·6)  =  3   7  11
```

Check the shape before trusting the arithmetic: `X` was `[2,3]`, and `dL/dX` is also `[2,3]` — a gradient with the wrong shape is a wrong gradient before a single number is even checked, and this shape-matching test is the cheapest sanity check for any new backward rule added to the registry.

### Worked Example 16.3.2 — `dL/dM`, completing the pair

The same `grad_output` also has to produce a gradient for `M`, using `∂L/∂M = Xᵀ @ ∂L/∂Y`. `Xᵀ` (transposing the `2×3` `X`) is:

```
Xᵀ = 1  4
     2  5
     3  6
```

`Xᵀ @ grad_output` — the `3×2` `Xᵀ` above times a `2×2` matrix of ones — gives a `3×2` result where every entry is the *row sum* of `Xᵀ`, since dotting any row of `Xᵀ` against a column of all ones just adds that row's two entries together:

```
dL/dM = (1+4)  (1+4)     5  5
        (2+5)  (2+5)  =  7  7
        (3+6)  (3+6)     9  9
```

Shape check: `M` was `[3,2]`, and `dL/dM` is also `[3,2]` — both gradients pass the cheap sanity check `MatMulOp::backward` needs to satisfy for *any* shapes, not just this chapter's specific `2×3` and `3×2`.

Compiled and run:

```bash
rustc --edition 2024 -O 03_matmul_gradients.rs -o 03_matmul_gradients
./03_matmul_gradients
```

Genuinely compiled and run:

```
=== Section 16.3: MatMulOp backward, dL/dX and dL/dM ===
X (2x3):
  [   1.0,    2.0,    3.0]
  [   4.0,    5.0,    6.0]
M (3x2):
  [   1.0,    2.0]
  [   3.0,    4.0]
  [   5.0,    6.0]
Y = X @ M (2x2):
  [  22.0,   28.0]
  [  49.0,   64.0]

M^T (2x3):
  [   1.0,    3.0,    5.0]
  [   2.0,    4.0,    6.0]
X^T (3x2):
  [   1.0,    4.0]
  [   2.0,    5.0]
  [   3.0,    6.0]

dL/dX = grad_output @ M^T (2x3):
  [   3.0,    7.0,   11.0]
  [   3.0,    7.0,   11.0]
  shape check: X is [2,3], dL/dX is [2,3] -- match: true

dL/dM = X^T @ grad_output (3x2):
  [   5.0,    5.0]
  [   7.0,    7.0]
  [   9.0,    9.0]
  shape check: M is [3,2], dL/dM is [3,2] -- match: true
```

## 16.4 Additional Element-wise Gradients `[FOUNDATIONAL]`

### Intuition

The registry so far only holds `AddOp`, `MulOp`, `ExpOp` — enough to run the running example and one matrix product, but nowhere near enough to build anything else in this book. `SubOp`, `DivOp`, `PowOp`, `LogOp`, and `SqrtOp` complete the basic arithmetic vocabulary the same way `Add` and `Mul` did: derive the local derivative from the forward formula, write it as a `Differentiable`, and check it against real numbers.

### Background

`SubOp`'s local derivative is almost `AddOp`'s, with one sign flipped: `∂(a-b)/∂a = 1`, but `∂(a-b)/∂b = -1`, since increasing `b` decreases `a-b`. `DivOp` and `PowOp` need both operands, the same way `MulOp` did:

```rust
struct SubOp;
impl Differentiable for SubOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value - inputs[1].value, false)
    }
    // d(a-b)/da = 1, d(a-b)/db = -1
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output, grad_output * -1.0]
    }
}

struct DivOp;
impl Differentiable for DivOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value / inputs[1].value, false)
    }
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        let a = inputs[0].value;
        let b = inputs[1].value;
        let grad_a = grad_output / b;
        let grad_b = grad_output * ((a * -1.0) / (b * b));
        vec![grad_a, grad_b]
    }
}

struct PowOp;
impl Differentiable for PowOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value.powf(inputs[1].value), false)
    }
    // d(a^b)/da = b * a^(b-1); d(a^b)/db = a^b * ln(a) = output * ln(a)
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32> {
        let a = inputs[0].value;
        let b = inputs[1].value;
        let grad_a = grad_output * (b * a.powf(b - 1.0));
        let grad_b = grad_output * (output.value * a.ln());
        vec![grad_a, grad_b]
    }
}

struct LogOp;
impl Differentiable for LogOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value.ln(), false)
    }
    // d(ln(x))/dx = 1/x
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output / inputs[0].value]
    }
}

struct SqrtOp;
impl Differentiable for SqrtOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value.sqrt(), false)
    }
    // d(sqrt(x))/dx = 1 / (2*sqrt(x)) = 1 / (2*output) -- reuse the
    // cached forward result rather than recomputing x.sqrt().
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32> {
        let denom = output.value * 2.0;
        vec![grad_output / denom]
    }
}
```

`SqrtOp` is written the same way `ExpOp` was in Section 16.2: it reads `output` (the already-computed `√x`) rather than recomputing a square root inside `backward`, for exactly the same reason — the forward pass already paid for that value once.

### Worked Example 16.4.1 — `SubOp` and `DivOp`, by hand

`a = 8.0, b = 5.0`. `c = a - b = 3.0`. With an upstream seed of `1.0`: `SubOp::backward` returns `[1.0, -1.0]` — `a` receives the seed unchanged, `b` receives its negation. Now `c = a / b = 1.6`. `DivOp::backward`: `grad_a = 1.0 / b = 1.0 / 5.0 = 0.2`; `grad_b = -1.0 × a / b² = -8.0 / 25.0 = -0.32`. Checked directly against a genuine finite-difference nudge, `a/(b+0.001)`, below.

### Worked Example 16.4.2 — `PowOp` and `LogOp`, by hand

`a = 2.0, b = 3.0` (i.e. `2³ = 8`). `∂(a^b)/∂a = b·a^{b-1} = 3 × 2² = 12`; `∂(a^b)/∂b = a^b·ln(a) = 8 × ln(2) ≈ 8 × 0.6931 ≈ 5.545`. With a seed of `1.0`, `PowOp::backward` returns `[12.0, 5.545]`. Separately, `LogOp` at `x = 2.0`: `ln(2.0) ≈ 0.6931`, and `d(ln x)/dx = 1/x = 0.5` — with a seed of `1.0`, `LogOp::backward` returns `0.5`, checked below against a genuine finite-difference nudge.

Compiled and run:

```bash
rustc --edition 2024 -O 04_additional_arithmetic_gradients.rs -o 04_additional_arithmetic_gradients
./04_additional_arithmetic_gradients
```

Genuinely compiled and run:

```
=== Section 16.4: SubOp, DivOp, PowOp, LogOp, SqrtOp, checked against finite differences ===
a=8.0, b=5.0, c=a-b=3.0
SubOp.backward(seed=1.0) -> [1.0, -1.0]

c=a/b=1.6
DivOp.backward(seed=1.0) -> grad_a=0.2000, grad_b=-0.3200
  finite-diff check: a/(b+0.001) = 1.5996801, slope = (1.5996801 - 1.6000000)/0.001 = -0.31996 (analytic: -0.32000)

a=2.0, b=3.0, a^b=8.0
PowOp.backward(seed=1.0) -> [12.000, 5.545]

x=2.0, ln(x)=0.6931
LogOp.backward(seed=1.0) -> 0.5000
  finite-diff check: ln(2.001) = 0.69365, slope = (0.69365 - 0.69315)/0.001 = 0.4998 (analytic: 0.5000)

x=4.0, sqrt(x)=2.0000
SqrtOp.backward(seed=1.0) -> 0.2500 (= 1/(2*sqrt(x)) = 1/(2*2.0000))
```

Both finite-difference checks land within the ordinary one-sided finite-difference error of their analytic answers — `-0.31996` against `-0.32000`, and `0.4998` against `0.5000` — the same small, expected gap this book's earlier finite-difference verifications have consistently shown, not a sign anything is wrong with either derivative.

## 16.5 Activation and Trigonometric Gradients `[FOUNDATIONAL]`

### Intuition

Every activation function and trigonometric function in this book eventually needs a backward rule, and each one follows the same recipe Section 16.4 established: find the local derivative, decide whether it's cheaper to express in terms of the input or the already-computed output, and write it as a `Differentiable`. `ReluOp` is the sharpest case — its derivative isn't a smooth formula at all, but a hard on/off switch.

### Background

These operate on whole vectors, not single floats — `ReluOp`'s own worked example below needs a four-entry mixed-sign input — so this section uses `VecTensor`, a vector-shaped analogue of Chapter 15's `ScalarTensor` sized for exactly that, and a matching `VecDifferentiable` trait:

```rust
#[derive(Clone)]
struct VecTensor {
    data: Vec<f32>,
}

trait VecDifferentiable {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor;
    fn backward(&self, grad_output: &VecTensor, inputs: &[VecTensor], output: &VecTensor) -> VecTensor;
}

struct ReluOp;
impl VecDifferentiable for ReluOp {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor {
        VecTensor { data: inputs[0].data.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect() }
    }
    // d(relu(x))/dx = 1 if x > 0 else 0 -- a hard mask, not a smooth derivative
    fn backward(&self, grad_output: &VecTensor, inputs: &[VecTensor], _output: &VecTensor) -> VecTensor {
        let out = (0..inputs[0].data.len())
            .map(|i| {
                let mask = if inputs[0].data[i] > 0.0 { 1.0 } else { 0.0 };
                grad_output.data[i] * mask
            })
            .collect();
        VecTensor { data: out }
    }
}

struct SigmoidOp;
impl VecDifferentiable for SigmoidOp {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor {
        VecTensor { data: inputs[0].data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect() }
    }
    // d(sigma(x))/dx = sigma(x) * (1 - sigma(x)) = output * (1 - output)
    fn backward(&self, grad_output: &VecTensor, _inputs: &[VecTensor], output: &VecTensor) -> VecTensor {
        let out = (0..output.data.len())
            .map(|i| {
                let o = output.data[i];
                grad_output.data[i] * (o * (1.0 - o))
            })
            .collect();
        VecTensor { data: out }
    }
}

struct TanhOp;
impl VecDifferentiable for TanhOp {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor {
        VecTensor { data: inputs[0].data.iter().map(|&v| v.tanh()).collect() }
    }
    // d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - output^2
    fn backward(&self, grad_output: &VecTensor, _inputs: &[VecTensor], output: &VecTensor) -> VecTensor {
        let out = (0..output.data.len())
            .map(|i| {
                let o = output.data[i];
                grad_output.data[i] * (1.0 - o * o)
            })
            .collect();
        VecTensor { data: out }
    }
}

struct SinOp;
impl VecDifferentiable for SinOp {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor {
        VecTensor { data: inputs[0].data.iter().map(|&v| v.sin()).collect() }
    }
    // d(sin(x))/dx = cos(x) -- needs the INPUT, not the output
    fn backward(&self, grad_output: &VecTensor, inputs: &[VecTensor], _output: &VecTensor) -> VecTensor {
        let out = (0..inputs[0].data.len()).map(|i| grad_output.data[i] * inputs[0].data[i].cos()).collect();
        VecTensor { data: out }
    }
}

struct CosOp;
impl VecDifferentiable for CosOp {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor {
        VecTensor { data: inputs[0].data.iter().map(|&v| v.cos()).collect() }
    }
    // d(cos(x))/dx = -sin(x)
    fn backward(&self, grad_output: &VecTensor, inputs: &[VecTensor], _output: &VecTensor) -> VecTensor {
        let out = (0..inputs[0].data.len()).map(|i| grad_output.data[i] * (-inputs[0].data[i].sin())).collect();
        VecTensor { data: out }
    }
}
```

`SigmoidOp` and `TanhOp` both read `output`, the same `ExpOp`/`SqrtOp` pattern from Sections 16.2 and 16.4 — `σ(x)(1-σ(x))` and `1-tanh²(x)` are both cheaper to compute from the already-known output than by re-evaluating `sigmoid`/`tanh` from `x` a second time. `SinOp` and `CosOp` are the opposite case: `cos(x)` isn't recoverable from `sin(x)`'s output alone (the sign of `cos` isn't determined by the value of `sin` at a single point), so both read `inputs[0]` instead.

### Worked Example 16.5.1 — `ReluOp` on a mixed-sign vector

`x = [-2.0, 3.0, -1.0, 5.0]`. Forward: `relu(x) = [0.0, 3.0, 0.0, 5.0]`. With an upstream gradient of `grad_output = [1.0, 1.0, 1.0, 1.0]`, the mask is `[0, 1, 0, 1]` — `1` exactly where `x > 0`. `ReluOp::backward` returns `[1.0×0, 1.0×1, 1.0×0, 1.0×1] = [0.0, 1.0, 0.0, 1.0]`: the negative positions get *zero* gradient, not a small or shrinking one — the same input value that got zeroed out on the forward pass gets zeroed out again on the backward pass, for a different reason each time (forward: `max(0,x)` clips it; backward: the local derivative genuinely is `0` there).

### Worked Example 16.5.2 — `SigmoidOp` and `TanhOp` at `x = 0`

At `x = 0`: `sigmoid(0) = 1/(1+e⁰) = 1/2 = 0.5`. Its derivative: `0.5 × (1 - 0.5) = 0.25` — the steepest point on the sigmoid curve, which is exactly why `x=0` is where a sigmoid-activated unit is most sensitive to its input. `tanh(0) = 0`. Its derivative: `1 - 0² = 1` — the steepest point on the tanh curve, and notably four times steeper than sigmoid's steepest point, a fact that shows up directly in how much faster tanh-activated gradients can grow or shrink layer to layer compared to sigmoid.

### Worked Example 16.5.3 — `SinOp`/`CosOp`, checked against each other

`x = 0.0`: `sin(0) = 0`, `cos(0) = 1`. `SinOp::backward` at this point returns `grad_output × cos(0) = grad_output × 1` — the gradient passes straight through unchanged, because `sin` is at its steepest exactly where its own value is zero. `CosOp::backward` at the same point returns `grad_output × (-sin(0)) = grad_output × 0` — zero, because `cos` is at a peak (flattest point, zero slope) exactly where `sin` is zero. The two functions' derivatives are `90°` out of phase with each other in exactly the way their own values are, which is a useful sanity check for any point picked to test either one.

Compiled and run:

```bash
rustc --edition 2024 -O 05_activation_trig_gradients.rs -o 05_activation_trig_gradients
./05_activation_trig_gradients
```

Genuinely compiled and run:

```
=== Section 16.5: ReluOp, SigmoidOp, TanhOp, SinOp, CosOp ===
x = [-2.0, 3.0, -1.0, 5.0]
relu(x) = [0.0, 3.0, 0.0, 5.0]
ReluOp.backward(grad_output=ones) = [0.0, 1.0, 0.0, 1.0]

sigmoid(0) = 0.5000, SigmoidOp.backward(seed=1.0) = 0.2500
tanh(0) = 0.0000, TanhOp.backward(seed=1.0) = 1.0000
tanh's local slope is 4.0x steeper than sigmoid's at the origin

sin(0) = 0.0000, SinOp.backward(seed=1.0) = 1.0000 (= seed * cos(0))
cos(0) = 1.0000, CosOp.backward(seed=1.0) = -0.0000 (= seed * -sin(0))

--- with grad_output=2.0 instead of 1.0 ---
SigmoidOp.backward(seed=2.0) = 0.5000
TanhOp.backward(seed=2.0) = 2.0000
```

`CosOp::backward` prints `-0.0000` rather than `0.0000` — `(-(0.0f32)).sin()`-derived `-sin(0.0)` genuinely evaluates to IEEE-754 negative zero in Rust exactly as it does in C++, since both languages implement the same floating-point standard; negative zero compares equal to positive zero and behaves identically in every calculation that follows, but prints with its sign bit intact. It is the expected floating-point value, not a bug, and not a coincidence that both editions show it — the standard, not the language, is what produces it.

## 16.6 Reduction and Shape Gradients `[FOUNDATIONAL]`

### Intuition

Every operation so far preserves the number of elements going in. Chapter 14's reductions and Chapter 13's shape operations don't — `SumOp` collapses many values into one, `MaxOp` collapses many into one *and* discards all but one index, and `ReshapeOp`/`TransposeOp` keep every value but rearrange where it lives. Each needs a backward rule shaped around exactly what its forward pass threw away.

### Background

```rust
// d(sum(x))/dx_i = 1 for every i -- the incoming scalar gradient gets
// broadcast back out to every position that was summed.
fn sum_backward(grad_output: f32, n: usize) -> Vec<f32> {
    vec![grad_output; n]
}

// Mirrors Chapter 14.2's max_reduce_kernel comparison exactly: a strict
// `>` means the earlier index wins any tie.
fn tensor_argmax_host(x: &[f32]) -> (f32, usize) {
    let mut best = x[0];
    let mut best_idx = 0;
    for i in 1..x.len() {
        if x[i] > best {
            best = x[i];
            best_idx = i;
        }
    }
    (best, best_idx)
}

// d(max(x))/dx_i = 1 for the winning index, 0 everywhere else -- requires
// the SAME index Chapter 14.2's kernel tracked.
fn max_backward_indexed(grad_output: f32, n: usize, winning_index: usize) -> Vec<f32> {
    let mut grad_x = vec![0.0f32; n];
    grad_x[winning_index] = grad_output;
    grad_x
}
```

`ReshapeOp` and `TransposeOp`'s backward rules don't need new formulas at all — reshape moved no data on the forward pass, so backward just reinterprets the same flat gradient buffer under the *original* shape; transpose is its own inverse for a 2-D matrix, so backward transposes the gradient right back.

`MaxOp`'s backward rule is the one genuinely new shape in this chapter: every other backward rule so far touches *every* input position. `MaxOp`'s touches exactly one — the winning index, the very value `max_reduce_kernel` was built in Chapter 14.2 specifically to carry alongside the maximum, for exactly this moment. Without that index, there would be no way to know which of the input positions deserves the incoming gradient at all. This is also the first place the fixed `(grad_output, inputs, output)` signature stops being quite enough on its own: `MaxOp::backward` needs a value — the winning index — that the forward reduction has to carry alongside its output, not derive from `inputs` or `grad_output` alone. `ReshapeOp` has the same problem one step earlier: its *forward* needs a target shape that isn't derivable from `inputs` at all, which is why it's the only op in this chapter's registry that carries a field of its own. Everything else here is stateless — one instance can serve every call.

### Worked Example 16.6.1 — `SumOp`, broadcasting the gradient back out

`x = [1, 4, 9, 16]` (Chapter 14.1's own running example). Forward: `sum(x) = 30`. With an upstream gradient of `grad_output = 1.0` (this sum feeding directly into a scalar loss), `SumOp::backward` broadcasts that single `1.0` back out to every position: `grad_x = [1.0, 1.0, 1.0, 1.0]`. This is the exact mirror image of what made the forward reduction lossy: every input position contributed equally to the sum, so every input position receives an equal share of the gradient flowing back.

### Worked Example 16.6.2 — `MaxOp`, routing gradient through one index only

`x = [3, 7, 2, 9]` (Chapter 14.2's own running example, where the maximum `9` was traced to original index `3`). With `grad_output = 1.0`, `MaxOp::backward` produces `grad_x = [0.0, 0.0, 0.0, 1.0]` — every position *except* index `3` receives exactly zero, and index `3` receives the full incoming gradient unchanged. Compare this to `SumOp`'s result on a same-length input: sum spreads gradient everywhere equally; max routes all of it through a single winner, precisely because only that one input position actually determined the output's value.

```
[COMMON TRAP]  What "the winning index" means when two inputs tie

`x = [3, 7, 2, 9]` has exactly one maximum, which is what makes Worked
Example 16.6.2 tidy. `x = [1, 5, 3, 5]` does not. Chapter 14.2's
tensor_argmax_host logic still returns a single index for it -- its
comparison is a strict `x[i] > best`, so the EARLIER of two equal
values wins every round, genuinely confirmed below -- and
max_backward_indexed therefore hands the entire incoming gradient to
index 1 and exactly zero to index 3, even though the two positions are
indistinguishable to the forward pass.

That is a defensible choice, not a correct one, because max has no
derivative at a tie in the first place: nudging either 5 upward raises
the maximum, so both one-sided derivatives are 1, and any convex
combination of the two is a valid subgradient. Picking one winner
gives (1, 0); splitting evenly gives (0.5, 0.5); both sum to 1, which
is the property that actually matters -- the total gradient leaving
the node has to equal the total arriving. The failure mode worth
watching for is an implementation that builds its mask by comparing
every element against the maximum VALUE rather than carrying an index
forward -- genuinely demonstrated below to hand out 2.0 total from an
input of 1.0, gradient invented out of a tie. Deriving the index
during the reduction, the way Chapter 14.2 does, avoids the question
by construction.
```

### Worked Example 16.6.3 — `ReshapeOp` and `TransposeOp`, undoing exactly what forward did

Reuse Chapter 13.3's `[2,6]`-to-`[3,4]` reshape: twelve values, `[0,1,...,11]`, reshaped from a `2×6` view to a `3×4` view with zero data movement. If a `grad_output` of shape `[3,4]` arrives at this node during backward, `ReshapeOp::backward` reshapes it right back to `[2,6]` — the same twelve gradient values, just re-sliced into the original grid, since reshape never moved a single value in the first place. Reuse Chapter 13.2's transpose example: `A = [[1,2,3],[4,5,6]]` (2×3) transposes to `Aᵀ` (3×2). A `grad_output` of shape `[3,2]` arriving at this node gets transposed back to `[2,3]` by `TransposeOp::backward` — transpose applied twice returns every value to its original position, which is exactly why "transpose the gradient" is the correct and complete backward rule, with no further correction needed.

Compiled and run:

```bash
rustc --edition 2024 -O 06_reduction_shape_gradients.rs -o 06_reduction_shape_gradients
./06_reduction_shape_gradients
```

Genuinely compiled and run:

```
=== Section 16.6: SumOp, MaxOp, ReshapeOp, TransposeOp ===
x = [1.0, 4.0, 9.0, 16.0], sum(x) = 30.0
SumOp.backward(grad_output=1.0) -> [1.0, 1.0, 1.0, 1.0]

x = [3.0, 7.0, 2.0, 9.0], max(x) = 9.0 at index 3
MaxOp.backward(grad_output=1.0) -> [0.0, 0.0, 0.0, 1.0]

--- COMMON TRAP: MaxOp.backward on a tie, [1, 5, 3, 5] ---
x = [1.0, 5.0, 3.0, 5.0], max = 5.0, winning index (earlier of the tie) = 1
indexed backward (correct): [0.0, 1.0, 0.0, 0.0], sum = 1.0
value-mask backward (broken): [0.0, 1.0, 0.0, 1.0], sum = 2.0 -- gradient invented out of a tie

original flat buffer, viewed as [2,6]:
  [  0.0,   1.0,   2.0,   3.0,   4.0,   5.0]
  [  6.0,   7.0,   8.0,   9.0,  10.0,  11.0]
reshaped to [3,4] (forward), viewed as [3,4]:
  [  0.0,   1.0,   2.0,   3.0]
  [  4.0,   5.0,   6.0,   7.0]
  [  8.0,   9.0,  10.0,  11.0]
grad_output reshaped back to [2,6] (backward), viewed as [2,6]:
  [  0.0,   1.0,   2.0,   3.0,   4.0,   5.0]
  [  6.0,   7.0,   8.0,   9.0,  10.0,  11.0]

A (2x3):
  [  1.0,   2.0,   3.0]
  [  4.0,   5.0,   6.0]
A^T (forward) (3x2):
  [  1.0,   4.0]
  [  2.0,   5.0]
  [  3.0,   6.0]
grad_output (3x2):
  [  1.0,   2.0]
  [  3.0,   4.0]
  [  5.0,   6.0]
TransposeOp.backward(grad_output) = transpose(grad_output) (2x3):
  [  1.0,   3.0,   5.0]
  [  2.0,   4.0,   6.0]
```

## 16.7 Custom Function Framework `[FOUNDATIONAL]`

### Intuition

Not every operation belongs in the core registry as a hand-differentiated forward/backward pair over a closed-form expression. Some values come from an *iterative* numerical procedure — a solver that runs a loop and converges toward an answer rather than computing one in a single formula — and differentiating through every step of that loop would be both wasteful and numerically noisy. The **implicit function theorem** is the escape hatch: treat the solver's output as *defined implicitly* by an equation it satisfies at convergence, and differentiate that equation instead of the solver's control flow.

### Background

Consider solving `x² = 2` for `x` with a numerical bisection search rather than `sqrt` — a stand-in, at small scale, for the bond-pricing bisection Part 7 actually differentiates through. Bisection between `a=1` and `b=2`: midpoint `1.5² = 2.25` (too big, move `b` to `1.5`); midpoint `1.25² = 1.5625` (too small, move `a` to `1.25`); midpoint `1.375² = 1.890625`; continuing this halves the bracket every step and converges toward `x ≈ √2`. The solver defines `x` implicitly by `f(x, c) = x² - c = 0`, where `c` is the constant being solved against (`c=2` here). The implicit function theorem says:

```
dx/dc = -(∂f/∂c) / (∂f/∂x) = -(-1) / (2x) = 1 / (2x)
```

The framework captures this as one opaque node with a closed-form backward, instead of an unrolled, differentiated bisection loop. `CustomFunction` holds a forward closure and a backward closure — Rust closures fill the role the CUDA edition's `std::function` does, boxed as trait objects so `CustomFunction` can hold any pair of them:

```rust
struct CustomFunction {
    forward_fn: Box<dyn Fn(&[f32]) -> f32>,
    backward_fn: Box<dyn Fn(f32, &[f32], f32) -> Vec<f32>>,
}

impl CustomFunction {
    fn forward(&self, inputs: &[f32]) -> f32 {
        (self.forward_fn)(inputs)
    }
    fn backward(&self, grad_output: f32, inputs: &[f32], output: f32) -> Vec<f32> {
        (self.backward_fn)(grad_output, inputs, output)
    }
}

// output holds the converged x; dx/dc = 1 / (2x) from the implicit function theorem
fn sqrt_via_bisection_backward(grad_output: f32, _inputs: &[f32], output: f32) -> Vec<f32> {
    let local_grad = 1.0 / (2.0 * output);
    vec![grad_output * local_grad]
}
```

### Worked Example 16.7.1 — Checking the implicit-function gradient against finite differences

Genuinely bisecting `x² = 2` to convergence and reading off `dx/dc = 1/(2x)` from the implicit function theorem, then checking it against a finite-difference nudge of `c` — bisecting `x² = 2.001` instead and comparing the resulting slope — is the same style of check earlier sections used for `DivOp` and `LogOp`. It's worth doing genuinely rather than trusting the formula, and it's worth reporting whatever numbers the bisection actually converges to in this compiler and this floating-point environment, rather than assuming they match another edition's run bit-for-bit — the last handful of bisection steps are exactly where two independently-compiled implementations of the identical algorithm can, and here genuinely do, drift by a rounding unit or two.

The digit count in a check like this is not incidental, and it is the easiest way to talk yourself out of a gradient rule that was right all along. Both bisection results have to be carried to seven significant figures to get three or four good ones out of the finite-difference slope, because the difference being measured is small relative to the values it's computed from, and every digit of agreement between those values is a digit the subtraction cancels away. Round `√2` to five digits first and the subtraction gives a slope over a percent off the analytic answer — genuinely demonstrated below — with every bit of that error coming from the rounding, none of it from the finite difference. A gradient check that disagrees in the third digit is far more often a truncated intermediate than a wrong derivative.

Compiled and run:

```bash
rustc --edition 2024 -O 07_custom_function_bisection.rs -o 07_custom_function_bisection
./07_custom_function_bisection
```

Genuinely compiled and run:

```
=== Section 16.7: CustomFunction, the implicit function theorem, bisection for x^2=2 ===
bisecting x^2 = 2, bracket [1, 2]:
  iter 0: mid=1.5000000, mid^2=2.2500000, too big, move b
  iter 1: mid=1.2500000, mid^2=1.5625000, too small (or exact), move a
  iter 2: mid=1.3750000, mid^2=1.8906250, too small (or exact), move a
  iter 3: mid=1.4375000, mid^2=2.0664062, too big, move b
  iter 4: mid=1.4062500, mid^2=1.9775391, too small (or exact), move a
  iter 5: mid=1.4218750, mid^2=2.0217285, too big, move b
  iter 6: mid=1.4140625, mid^2=1.9995728, too small (or exact), move a
  iter 7: mid=1.4179688, mid^2=2.0106354, too big, move b
  iter 8: mid=1.4160156, mid^2=2.0051003, too big, move b
  iter 9: mid=1.4150391, mid^2=2.0023355, too big, move b
  iter 10: mid=1.4145508, mid^2=2.0009539, too big, move b
  iter 11: mid=1.4143066, mid^2=2.0002632, too big, move b
  iter 12: mid=1.4141846, mid^2=1.9999180, too small (or exact), move a
  iter 13: mid=1.4142456, mid^2=2.0000906, too big, move b
  iter 14: mid=1.4142151, mid^2=2.0000043, too big, move b
  iter 15: mid=1.4141998, mid^2=1.9999611, too small (or exact), move a
  iter 16: mid=1.4142075, mid^2=1.9999827, too small (or exact), move a
  iter 17: mid=1.4142113, mid^2=1.9999936, too small (or exact), move a
  iter 18: mid=1.4142132, mid^2=1.9999989, too small (or exact), move a
  iter 19: mid=1.4142141, mid^2=2.0000017, too big, move b
  iter 20: mid=1.4142137, mid^2=2.0000002, too big, move b
  iter 21: mid=1.4142134, mid^2=1.9999996, too small (or exact), move a
  iter 22: mid=1.4142135, mid^2=1.9999999, too small (or exact), move a
  iter 23: mid=1.4142137, mid^2=2.0000002, too big, move b
  iter 24: mid=1.4142137, mid^2=2.0000002, too big, move b
converged x = 1.4142137 (true sqrt(2) = 1.4142135)

CustomFunction.forward(c=2.0) = 1.4142137
dx/dc = 1/(2x) = 1/(2*1.4142137) = 0.3535534

--- finite-difference check, bisecting c=2.001 too ---
bisection(c=2.001) converges to x = 1.4145670
finite-diff slope = (1.4145670 - 1.4142137) / 0.001 = 0.3533363
analytic dx/dc                              = 0.3535534

--- what happens if sqrt(2) is rounded to 6 digits before the subtraction ---
rounded x(c=2)     = 1.41421
rounded x(c=2.001) = 1.41457
slope from rounded values = (1.41457 - 1.41421) / 0.001 = 0.3600
that is 1.8% off the analytic answer of 0.3535534 -- purely from rounding, not from the calculus
```

Two things are worth stating plainly about this run. First, the bisection genuinely converges to `1.4142137` here, one floating-point unit away from the CUDA edition's own `1.4142136` — both are `√2` to the same six significant figures, and the difference is a real, reproducible artifact of the exact order operations happen in across two independently-compiled binaries built from the same algorithm, not a bug in either one; rerunning this binary twice produces the identical `1.4142137` both times, so the drift is deterministic per-build, not a symptom of nondeterminism at runtime. This chapter reports the number this Rust binary actually printed rather than adjusting it to match the CUDA edition's — the standing rule this whole book follows is to correct the *text* to match a genuine result, never the other way around. Second, and more importantly, the finite-difference slope (`0.3533363`) still agrees with the analytic answer (`0.3535534`) to three decimal places, the expected residual for a `0.001` one-sided nudge propagated through a bisection search that itself only converges to about seven digits — and the rounding demonstration still produces a genuine `1.8%`-off result, the same order of magnitude the CUDA edition found, because at five decimal digits of rounding, both editions' `x(c=2)` and `x(c=2.001)` collapse to the identical rounded values regardless of that one-ULP difference upstream.

The entire multi-step bisection loop collapses, for gradient purposes, into one multiplication by `1/(2x)` — exactly the pattern the z-spread bisection in Part 7 reuses: the forward pass runs the numerical solver as ordinary control flow, and one hand-derived `backward_fn` plugs the whole iterative procedure into the graph as a single differentiable op.

## 16.8 Reference Implementations — Every Op, One Registry, Genuinely Run

Sections 16.2 through 16.6 derived and checked seventeen backward rules one at a time, each next to the forward operation and worked example it belongs with, and Section 16.7 added an eighteenth `Differentiable`-shaped value that isn't a fixed rule at all — `CustomFunction`, which carries whatever backward its construction was handed. This section consolidates all of it into one registry, genuinely compiled and executed, with every result checked against the exact numbers Sections 16.2 through 16.6 already derived by hand.

Doing that exposed one real wrinkle worth stating plainly before the listing: this chapter's earlier sections each reached for a different, narrowly-typed Rust stand-in as needed — `ScalarTensor` for the purely scalar ops (16.2, 16.4), `VecTensor` for the vector-shaped activation and trig ops (16.5), and `Matrix` for `MatMulOp` and the reshape/transpose pair (16.3, 16.6). None of those three types can hold each other's data, so a single registry that dispatches on a name string and returns a common trait object needs exactly one shared representation, not three. This section introduces that one new type — a flat buffer plus a row/column shape, general enough to stand in for a scalar (`1×1`), a vector (`1×N`), or a matrix (`R×C`) uniformly — and reimplements all eighteen named ops against it, plus a small `winning_index` field carried alongside `MaxOp`'s scalar output, addressing the exact gap Section 16.6 already flagged: the fixed `(grad_output, inputs, output)` signature isn't quite enough on its own for that one op.

```rust
#[derive(Clone)]
struct Tensor {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    winning_index: i32,   // used only by MaxOp
}

impl Tensor {
    fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        Tensor { data, rows, cols, winning_index: -1 }
    }
    fn scalar(v: f32) -> Self { Tensor::new(1, 1, vec![v]) }
    fn vec(v: Vec<f32>) -> Self { let n = v.len(); Tensor::new(1, n, v) }
    fn size(&self) -> usize { self.rows * self.cols }
}

trait Differentiable {
    fn forward(&self, inputs: &[Tensor]) -> Tensor;
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], output: &Tensor) -> Vec<Tensor>;
}
```

`AddOp` through `SqrtOp`, `ReluOp` through `CosOp`, `MatMulOp`, `SumOp`, `MaxOp`, `ReshapeOp`, and `TransposeOp` are each reimplemented against this one `Tensor` type in the file below — the same formulas Sections 16.2 through 16.6 already derived and checked, just no longer split across three incompatible Rust types. `OpRegistry` and `chain_rule_step` are otherwise exactly Chapter 15's and Section 16.1's design. `build_op_registry`, though, gets to be genuinely simpler in Rust than a literal port of the CUDA edition's own version would be: that version takes eighteen separate reference parameters, one per op, because C++'s registry holds non-owning pointers into locals the *caller* has to keep alive in `main`'s own scope for as long as the registry exists. `Box<dyn Differentiable>` owns each op outright, so this edition's `build_op_registry` needs no parameters at all — it constructs and boxes every op itself:

```rust
fn build_op_registry() -> OpRegistry {
    let mut registry = OpRegistry::new();
    registry.register_op("add", Box::new(AddOp));
    registry.register_op("sub", Box::new(SubOp));
    registry.register_op("mul", Box::new(MulOp));
    registry.register_op("div", Box::new(DivOp));
    registry.register_op("pow", Box::new(PowOp));
    registry.register_op("exp", Box::new(ExpOp));
    registry.register_op("log", Box::new(LogOp));
    registry.register_op("sqrt", Box::new(SqrtOp));
    registry.register_op("relu", Box::new(ReluOp));
    registry.register_op("sigmoid", Box::new(SigmoidOp));
    registry.register_op("tanh", Box::new(TanhOp));
    registry.register_op("sin", Box::new(SinOp));
    registry.register_op("cos", Box::new(CosOp));
    registry.register_op("matmul", Box::new(MatMulOp));
    registry.register_op("sum", Box::new(SumOp));
    registry.register_op("max", Box::new(MaxOp));
    registry.register_op("reshape", Box::new(ReshapeOp { target_rows: 0, target_cols: 0 }));
    registry.register_op("transpose", Box::new(TransposeOp));
    registry
}
```

The eighteen named ops registered — `add`, `sub`, `mul`, `div`, `pow`, `exp`, `log`, `sqrt`, `relu`, `sigmoid`, `tanh`, `sin`, `cos`, `matmul`, `sum`, `max`, `reshape`, `transpose` — are the same list this chapter derived them in. `CustomFunction` is deliberately absent from it: it isn't one fixed rule, it's a slot for a caller-supplied pair of closures, so it gets constructed per use rather than registered once under a fixed name, exactly as Section 16.7 built it.

### Worked Example 16.8.1 — Every worked example in this chapter, re-run through one registry

Compiled and run:

```bash
rustc --edition 2024 -O 08_full_op_registry.rs -o 08_full_op_registry
./08_full_op_registry
```

Genuinely compiled and run:

```
=== Section 16.8: every op wired into ONE registry, genuinely compiled and run ===
(all eighteen ops below are registered under one Differentiable trait object and
 exercised through chain_rule_step, with every result checked against the exact
 numbers Sections 16.2-16.6 derived by hand.)

registry.ops.len() = 18 (18 named ops; CustomFunction stays unregistered, per Section 16.7)

--- add/mul (w=x*y+x) ---
  [x.grad] got=5.00000 expected=5.00000 -> MATCH
  [y.grad] got=3.00000 expected=3.00000 -> MATCH
--- sub/div (a=8, b=5) ---
  [SubOp grad_a] got=1.00000 expected=1.00000 -> MATCH
  [SubOp grad_b] got=-1.00000 expected=-1.00000 -> MATCH
  [DivOp grad_a] got=0.20000 expected=0.20000 -> MATCH
  [DivOp grad_b] got=-0.32000 expected=-0.32000 -> MATCH
--- pow/log (a=2, b=3) ---
  [PowOp grad_a] got=12.00000 expected=12.00000 -> MATCH
  [PowOp grad_b] got=5.54518 expected=5.54500 -> MATCH
  [LogOp grad] got=0.50000 expected=0.50000 -> MATCH
--- exp/sqrt ---
  [ExpOp grad (x=1)] got=2.71828 expected=2.71828 -> MATCH
  [SqrtOp grad (x=4)] got=0.25000 expected=0.25000 -> MATCH
--- relu ---
ReluOp.backward = [0.0000, 1.0000, 0.0000, 1.0000]
  [relu[0]] got=0.00000 expected=0.00000 -> MATCH
  [relu[1]] got=1.00000 expected=1.00000 -> MATCH
  [relu[2]] got=0.00000 expected=0.00000 -> MATCH
  [relu[3]] got=1.00000 expected=1.00000 -> MATCH
--- sigmoid/tanh/sin/cos at x=0 ---
  [SigmoidOp grad] got=0.25000 expected=0.25000 -> MATCH
  [TanhOp grad] got=1.00000 expected=1.00000 -> MATCH
  [SinOp grad] got=1.00000 expected=1.00000 -> MATCH
  [CosOp grad] got=-0.00000 expected=0.00000 -> MATCH
--- matmul ---
dL/dX = [3.0000, 7.0000, 11.0000, 3.0000, 7.0000, 11.0000]
dL/dM = [5.0000, 5.0000, 7.0000, 7.0000, 9.0000, 9.0000]
  [dL/dX[0]] got=3.00000 expected=3.00000 -> MATCH
  [dL/dX[2]] got=11.00000 expected=11.00000 -> MATCH
  [dL/dM[0]] got=5.00000 expected=5.00000 -> MATCH
  [dL/dM[5]] got=9.00000 expected=9.00000 -> MATCH
--- sum/max ---
SumOp.backward = [1.0000, 1.0000, 1.0000, 1.0000]
MaxOp.backward = [0.0000, 0.0000, 0.0000, 1.0000]
  [sum_grad[0]] got=1.00000 expected=1.00000 -> MATCH
  [max_grad winning index value] got=1.00000 expected=1.00000 -> MATCH
  [max_grad non-winner] got=0.00000 expected=0.00000 -> MATCH
--- reshape/transpose ---
reshape_g[0] shape = [2,6] (original shape restored)
  [reshape roundtrip shape rows] got=2.00000 expected=2.00000 -> MATCH
  [reshape roundtrip shape cols] got=6.00000 expected=6.00000 -> MATCH
TransposeOp.backward = [1.0000, 3.0000, 5.0000, 2.0000, 4.0000, 6.0000]
  [transpose_grad[0]] got=1.00000 expected=1.00000 -> MATCH
  [transpose_grad[3]] got=2.00000 expected=2.00000 -> MATCH

30 / 30 checks passed against the exact numbers Sections 16.2-16.6 derived by hand.
```

All 30 checks pass, and the process exit code is genuinely `0` (confirmed directly, not asserted) — confirming, by actually running it rather than asserting it, that a single registry dispatching on a bare op-name string reproduces every number this chapter derived section by section, and that `PowOp`'s `grad_b` (`5.54518` here versus `5.545` in Section 16.4's hand calculation) differs only in the number of digits `{:.3}` versus `{:.5}` formatting happened to show, not in the underlying value. `AddOp::backward`'s `vec![grad_output.clone(), grad_output.clone()]` line, reused verbatim from Section 16.2's Common Trap, is worth a second look here too: every `.clone()` in this file genuinely allocates a fresh `Vec<f32>`, so wiring all eighteen ops through one shared `Tensor` type never reintroduces the aliasing question Section 16.2 spent a worked example ruling out.

## 16.9 Complete Runnable Code

### File: `01_chain_rule_dispatch.rs`

```rust
use std::collections::HashMap;

// Reused verbatim from Chapter 15.
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

trait Differentiable {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor;
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32>;
}

struct AddOp;
impl Differentiable for AddOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value + inputs[1].value, false)
    }
    // d(a+b)/da = 1, d(a+b)/db = 1 -- gradient passes through unchanged
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output, grad_output]
    }
}

struct MulOp;
impl Differentiable for MulOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value * inputs[1].value, false)
    }
    // d(a*b)/da = b, d(a*b)/db = a
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

// Section 16.1: dispatch to the registered backward for an op. The
// caller (Chapter 17's GradientEngine) is the one that will eventually
// add each result into the corresponding input's accumulated .grad.
fn chain_rule_step(
    registry: &OpRegistry,
    op_name: &str,
    grad_output: f32,
    inputs: &[ScalarTensor],
    output: &ScalarTensor,
) -> Vec<f32> {
    let op = registry.get(op_name);
    op.backward(grad_output, inputs, output)
}

fn main() {
    println!("=== Section 16.1: chain rule as sum-over-paths, dispatched through chain_rule_step ===");

    let mut registry = OpRegistry::new();
    registry.register_op("add", Box::new(AddOp));
    registry.register_op("mul", Box::new(MulOp));

    let x = ScalarTensor::new(3.0, true);
    let y = ScalarTensor::new(4.0, true);
    let mul_op = MulOp;
    let z = mul_op.forward(&[x, y]);
    let add_op = AddOp;
    let w = add_op.forward(&[z, x]);
    println!("x={:.1}, y={:.1}, z=x*y={:.1}, w=z+x={:.1}\n", x.value, y.value, z.value, w.value);

    let add_grads = chain_rule_step(&registry, "add", 1.0, &[z, x], &w);
    println!(
        "chain_rule_step(\"add\", grad_output=1.0, inputs=[z={:.1}, x={:.1}]) -> [{:.1}, {:.1}]",
        z.value, x.value, add_grads[0], add_grads[1]
    );
    println!("  z.grad (from add) = {:.1}", add_grads[0]);
    println!("  x's Route 2 contribution (direct edge into add) = {:.1}\n", add_grads[1]);

    let mul_grads = chain_rule_step(&registry, "mul", 1.0, &[x, y], &z);
    println!(
        "chain_rule_step(\"mul\", grad_output=1.0, inputs=[x={:.1}, y={:.1}]) -> [{:.1}, {:.1}]",
        x.value, y.value, mul_grads[0], mul_grads[1]
    );
    println!("  x's Route 1 contribution (through the multiply) = {:.1}", mul_grads[0]);
    println!("  y.grad (only one route) = {:.1}\n", mul_grads[1]);

    let x_grad = add_grads[1] + mul_grads[0];
    println!("x.grad = Route2 + Route1 = {:.1} + {:.1} = {:.1}", add_grads[1], mul_grads[0], x_grad);
    println!("y.grad = {:.1}", mul_grads[1]);
    println!(
        "cross-check against Chapter 15's plain calculus: dw/dx = y+1 = {:.1}, dw/dy = x = {:.1}",
        y.value + 1.0,
        x.value
    );
}
```

```bash
rustc --edition 2024 -O 01_chain_rule_dispatch.rs -o 01_chain_rule_dispatch
./01_chain_rule_dispatch
```

### File: `02_element_wise_gradients.rs`

```rust
use std::cell::RefCell;
use std::rc::Rc;

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

trait Differentiable {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor;
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32>;
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

struct MulOp;
impl Differentiable for MulOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value * inputs[1].value, false)
    }
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output * inputs[1].value, grad_output * inputs[0].value]
    }
}

struct ExpOp;
impl Differentiable for ExpOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value.exp(), false)
    }
    // d(e^x)/dx = e^x = output -- reuse the cached forward result
    // instead of recomputing inputs[0].value.exp() a second time.
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output * output.value]
    }
}

// Idiomatic Rust: Vec<f32> is owned data. "Return two copies of
// grad_output" naturally means .clone() -- and Vec::clone deep-copies the
// buffer, so the two results are NOT aliased, structurally, with no extra
// discipline required to make that true.
fn add_backward_vec(grad_output: &[f32]) -> (Vec<f32>, Vec<f32>) {
    (grad_output.to_vec(), grad_output.to_vec())
}

// Reproducing the CUDA edition's aliasing bug on purpose requires reaching
// for shared, reference-counted, interior-mutable state explicitly --
// Rc<RefCell<Vec<f32>>>, Rust's opt-in analogue of two structs sharing one
// raw pointer. Rc::clone bumps a refcount and returns a second HANDLE to
// the SAME allocation, not a second allocation.
fn add_backward_aliased(
    grad_output: &Rc<RefCell<Vec<f32>>>,
) -> (Rc<RefCell<Vec<f32>>>, Rc<RefCell<Vec<f32>>>) {
    (Rc::clone(grad_output), Rc::clone(grad_output))
}

fn main() {
    println!("=== Section 16.2: AddOp, MulOp, ExpOp backward, by hand ===");

    // Worked Example 16.2.1 -- AddOp
    let z = ScalarTensor::new(12.0, true);
    let x = ScalarTensor::new(3.0, true);
    let add_op = AddOp;
    let w = add_op.forward(&[z, x]);
    let add_grads = add_op.backward(1.0, &[z, x], &w);
    println!("w = z + x = {:.1}", w.value);
    println!(
        "AddOp.backward(grad_output=1.0, [z={:.1}, x={:.1}]) -> [{:.1}, {:.1}]",
        z.value, x.value, add_grads[0], add_grads[1]
    );
    println!("  z's incoming gradient = {:.1}, x's Route 2 contribution = {:.1}\n", add_grads[0], add_grads[1]);

    // Worked Example 16.2.2 -- MulOp
    let y = ScalarTensor::new(4.0, true);
    let mul_op = MulOp;
    let zz = mul_op.forward(&[x, y]);
    let mul_grads = mul_op.backward(add_grads[0], &[x, y], &zz);
    println!("z = x * y = {:.1}", zz.value);
    println!(
        "MulOp.backward(grad_output={:.1}, [x={:.1}, y={:.1}]) -> [{:.1}, {:.1}]",
        add_grads[0], x.value, y.value, mul_grads[0], mul_grads[1]
    );
    let x_grad_total = add_grads[1] + mul_grads[0];
    println!("  x.grad = {:.1} (AddOp) + {:.1} (MulOp) = {:.1}", add_grads[1], mul_grads[0], x_grad_total);
    println!("  y.grad = {:.1} (only one route)\n", mul_grads[1]);

    // Worked Example 16.2.3 -- ExpOp, the case that needs `output`
    println!("--- ExpOp: reusing the cached forward output instead of recomputing exp(x) ---");
    let xe = ScalarTensor::new(1.0, true);
    let exp_op = ExpOp;
    let u = exp_op.forward(&[xe]);
    let exp_grads = exp_op.backward(1.0, &[xe], &u);
    println!("u = exp(x) at x=1.0: u = {:.5}", u.value);
    println!("ExpOp.backward(grad_output=1.0, output={:.5}) -> {:.5}", u.value, exp_grads[0]);
    println!("  (du/dx = e^x = output, read directly rather than recomputing exp(1.0) again)\n");

    // COMMON TRAP, Rust edition: does .clone()-based "return two copies"
    // alias the way a raw-pointer-backed C++ BufferTensor's struct copy
    // does? Tested both ways rather than assumed either way.
    println!("--- COMMON TRAP: does AddOp.backward hand out the SAME buffer twice? ---");
    println!("(idiomatic Rust: Vec<f32>, .clone()-based \"return two copies\")");
    let grad_output_vec = vec![1.0f32];
    println!("grad_output_vec.as_ptr() = {:p}", grad_output_vec.as_ptr());
    let (z_grad, mut x_grad) = add_backward_vec(&grad_output_vec);
    println!("returned z_grad.as_ptr() = {:p}, value: {}", z_grad.as_ptr(), z_grad[0]);
    println!("returned x_grad.as_ptr() = {:p}, value: {}", x_grad.as_ptr(), x_grad[0]);
    println!("same address? {} -- NOT aliased: Vec::clone deep-copies the buffer", z_grad.as_ptr() == x_grad.as_ptr());
    x_grad[0] = 999.0;
    println!("mutating x_grad[0] = 999.0 leaves z_grad[0] = {} untouched\n", z_grad[0]);

    println!("(reproducing the CUDA edition's aliasing on purpose: Rc<RefCell<Vec<f32>>>)");
    let grad_output_rc = Rc::new(RefCell::new(vec![1.0f32]));
    println!("grad_output_rc data ptr = {:p}", grad_output_rc.borrow().as_ptr());
    let (z_grad_rc, x_grad_rc) = add_backward_aliased(&grad_output_rc);
    println!("returned z_grad_rc data ptr = {:p}, value: {}", z_grad_rc.borrow().as_ptr(), z_grad_rc.borrow()[0]);
    println!("returned x_grad_rc data ptr = {:p}, value: {}", x_grad_rc.borrow().as_ptr(), x_grad_rc.borrow()[0]);
    println!(
        "same address? {} -- ALIASED (Rc::strong_count = {})",
        z_grad_rc.borrow().as_ptr() == x_grad_rc.borrow().as_ptr(),
        Rc::strong_count(&grad_output_rc)
    );
    z_grad_rc.borrow_mut()[0] = 999.0;
    println!(
        "mutating z_grad_rc[0] = 999.0 through one handle corrupts x_grad_rc[0] too: {}",
        x_grad_rc.borrow()[0]
    );
}
```

```bash
rustc --edition 2024 -O 02_element_wise_gradients.rs -o 02_element_wise_gradients
./02_element_wise_gradients
```

### File: `03_matmul_gradients.rs`

```rust
// Matrix reused verbatim from Chapter 13.
#[derive(Clone)]
struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

fn matrix_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let mut data = vec![0.0f32; a.rows * b.cols];
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.get(i, k) * b.get(k, j);
            }
            data[i * b.cols + j] = sum;
        }
    }
    Matrix { data, rows: a.rows, cols: b.cols }
}

fn transpose(a: &Matrix) -> Matrix {
    let mut data = vec![0.0f32; a.rows * a.cols];
    for i in 0..a.rows {
        for j in 0..a.cols {
            data[j * a.rows + i] = a.get(i, j);
        }
    }
    Matrix { data, rows: a.cols, cols: a.rows }
}

fn print_matrix(label: &str, m: &Matrix) {
    println!("{} ({}x{}):", label, m.rows, m.cols);
    for i in 0..m.rows {
        let row: Vec<String> = (0..m.cols).map(|j| format!("{:6.1}", m.get(i, j))).collect();
        println!("  [{}]", row.join(", "));
    }
}

// Section 16.3: dL/dX = grad_output @ B^T, dL/dB = A^T @ grad_output --
// MatMulOp operates on whole matrices, not single floats, so it is scoped
// as a plain function here rather than forced through ScalarTensor's
// Differentiable trait, the same scoping choice Chapter 13 made for
// matrix operations generally.
struct MatMulBackwardResult {
    grad_a: Matrix,
    grad_b: Matrix,
}

fn matmul_backward(grad_output: &Matrix, a: &Matrix, b: &Matrix) -> MatMulBackwardResult {
    let bt = transpose(b);
    let at = transpose(a);
    MatMulBackwardResult { grad_a: matrix_multiply(grad_output, &bt), grad_b: matrix_multiply(&at, grad_output) }
}

fn main() {
    println!("=== Section 16.3: MatMulOp backward, dL/dX and dL/dM ===");

    let x = Matrix { data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rows: 2, cols: 3 };
    let m = Matrix { data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rows: 3, cols: 2 };
    let y = matrix_multiply(&x, &m);

    print_matrix("X", &x);
    print_matrix("M", &m);
    print_matrix("Y = X @ M", &y);
    println!();

    let mt = transpose(&m);
    let xt = transpose(&x);
    print_matrix("M^T", &mt);
    print_matrix("X^T", &xt);
    println!();

    let grad_output = Matrix { data: vec![1.0, 1.0, 1.0, 1.0], rows: 2, cols: 2 };
    let result = matmul_backward(&grad_output, &x, &m);

    print_matrix("dL/dX = grad_output @ M^T", &result.grad_a);
    println!(
        "  shape check: X is [{},{}], dL/dX is [{},{}] -- match: {}",
        x.rows,
        x.cols,
        result.grad_a.rows,
        result.grad_a.cols,
        x.rows == result.grad_a.rows && x.cols == result.grad_a.cols
    );
    println!();

    print_matrix("dL/dM = X^T @ grad_output", &result.grad_b);
    println!(
        "  shape check: M is [{},{}], dL/dM is [{},{}] -- match: {}",
        m.rows,
        m.cols,
        result.grad_b.rows,
        result.grad_b.cols,
        m.rows == result.grad_b.rows && m.cols == result.grad_b.cols
    );
}
```

```bash
rustc --edition 2024 -O 03_matmul_gradients.rs -o 03_matmul_gradients
./03_matmul_gradients
```

### File: `04_additional_arithmetic_gradients.rs`

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

trait Differentiable {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor;
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32>;
}

struct SubOp;
impl Differentiable for SubOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value - inputs[1].value, false)
    }
    // d(a-b)/da = 1, d(a-b)/db = -1
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output, grad_output * -1.0]
    }
}

struct DivOp;
impl Differentiable for DivOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value / inputs[1].value, false)
    }
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        let a = inputs[0].value;
        let b = inputs[1].value;
        let grad_a = grad_output / b;
        let grad_b = grad_output * ((a * -1.0) / (b * b));
        vec![grad_a, grad_b]
    }
}

struct PowOp;
impl Differentiable for PowOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value.powf(inputs[1].value), false)
    }
    // d(a^b)/da = b * a^(b-1); d(a^b)/db = a^b * ln(a) = output * ln(a)
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32> {
        let a = inputs[0].value;
        let b = inputs[1].value;
        let grad_a = grad_output * (b * a.powf(b - 1.0));
        let grad_b = grad_output * (output.value * a.ln());
        vec![grad_a, grad_b]
    }
}

struct LogOp;
impl Differentiable for LogOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value.ln(), false)
    }
    // d(ln(x))/dx = 1/x
    fn backward(&self, grad_output: f32, inputs: &[ScalarTensor], _output: &ScalarTensor) -> Vec<f32> {
        vec![grad_output / inputs[0].value]
    }
}

struct SqrtOp;
impl Differentiable for SqrtOp {
    fn forward(&self, inputs: &[ScalarTensor]) -> ScalarTensor {
        ScalarTensor::new(inputs[0].value.sqrt(), false)
    }
    // d(sqrt(x))/dx = 1 / (2*sqrt(x)) = 1 / (2*output) -- reuse the
    // cached forward result rather than recomputing x.sqrt().
    fn backward(&self, grad_output: f32, _inputs: &[ScalarTensor], output: &ScalarTensor) -> Vec<f32> {
        let denom = output.value * 2.0;
        vec![grad_output / denom]
    }
}

fn main() {
    println!("=== Section 16.4: SubOp, DivOp, PowOp, LogOp, SqrtOp, checked against finite differences ===");

    let a48 = ScalarTensor::new(8.0, true);
    let b48 = ScalarTensor::new(5.0, true);
    let sub_op = SubOp;
    let c_sub = sub_op.forward(&[a48, b48]);
    let sub_grads = sub_op.backward(1.0, &[a48, b48], &c_sub);
    println!("a={:.1}, b={:.1}, c=a-b={:.1}", a48.value, b48.value, c_sub.value);
    println!("SubOp.backward(seed=1.0) -> [{:.1}, {:.1}]\n", sub_grads[0], sub_grads[1]);

    let div_op = DivOp;
    let c_div = div_op.forward(&[a48, b48]);
    let div_grads = div_op.backward(1.0, &[a48, b48], &c_div);
    println!("c=a/b={:.1}", c_div.value);
    println!("DivOp.backward(seed=1.0) -> grad_a={:.4}, grad_b={:.4}", div_grads[0], div_grads[1]);
    let nudged_div = a48.value / (b48.value + 0.001);
    let slope_div = (nudged_div - c_div.value) / 0.001;
    println!(
        "  finite-diff check: a/(b+0.001) = {:.7}, slope = ({:.7} - {:.7})/0.001 = {:.5} (analytic: {:.5})\n",
        nudged_div, nudged_div, c_div.value, slope_div, div_grads[1]
    );

    let a23 = ScalarTensor::new(2.0, true);
    let b23 = ScalarTensor::new(3.0, true);
    let pow_op = PowOp;
    let c_pow = pow_op.forward(&[a23, b23]);
    let pow_grads = pow_op.backward(1.0, &[a23, b23], &c_pow);
    println!("a={:.1}, b={:.1}, a^b={:.1}", a23.value, b23.value, c_pow.value);
    println!("PowOp.backward(seed=1.0) -> [{:.3}, {:.3}]\n", pow_grads[0], pow_grads[1]);

    let log_op = LogOp;
    let c_log = log_op.forward(&[a23]);
    let log_grads = log_op.backward(1.0, &[a23], &c_log);
    println!("x={:.1}, ln(x)={:.4}", a23.value, c_log.value);
    println!("LogOp.backward(seed=1.0) -> {:.4}", log_grads[0]);
    let nudged_log = (a23.value + 0.001f32).ln();
    let slope_log = (nudged_log - c_log.value) / 0.001;
    println!(
        "  finite-diff check: ln(2.001) = {:.5}, slope = ({:.5} - {:.5})/0.001 = {:.4} (analytic: {:.4})\n",
        nudged_log, nudged_log, c_log.value, slope_log, log_grads[0]
    );

    let x4 = ScalarTensor::new(4.0, true);
    let sqrt_op = SqrtOp;
    let c_sqrt = sqrt_op.forward(&[x4]);
    let sqrt_grads = sqrt_op.backward(1.0, &[x4], &c_sqrt);
    println!("x={:.1}, sqrt(x)={:.4}", x4.value, c_sqrt.value);
    println!(
        "SqrtOp.backward(seed=1.0) -> {:.4} (= 1/(2*sqrt(x)) = 1/(2*{:.4}))",
        sqrt_grads[0], c_sqrt.value
    );
}
```

```bash
rustc --edition 2024 -O 04_additional_arithmetic_gradients.rs -o 04_additional_arithmetic_gradients
./04_additional_arithmetic_gradients
```

### File: `05_activation_trig_gradients.rs`

```rust
// A vector-shaped analogue of Chapter 15's ScalarTensor, sized for the
// activation/trig ops in this section -- these operate on whole vectors,
// not single floats.
#[derive(Clone)]
struct VecTensor {
    data: Vec<f32>,
}

impl VecTensor {
    fn new(data: Vec<f32>) -> Self {
        VecTensor { data }
    }
    fn size(&self) -> usize {
        self.data.len()
    }
}

trait VecDifferentiable {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor;
    fn backward(&self, grad_output: &VecTensor, inputs: &[VecTensor], output: &VecTensor) -> VecTensor;
}

struct ReluOp;
impl VecDifferentiable for ReluOp {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor {
        VecTensor::new(inputs[0].data.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect())
    }
    // d(relu(x))/dx = 1 if x > 0 else 0 -- a hard mask, not a smooth derivative
    fn backward(&self, grad_output: &VecTensor, inputs: &[VecTensor], _output: &VecTensor) -> VecTensor {
        let out = (0..inputs[0].size())
            .map(|i| {
                let mask = if inputs[0].data[i] > 0.0 { 1.0 } else { 0.0 };
                grad_output.data[i] * mask
            })
            .collect();
        VecTensor::new(out)
    }
}

struct SigmoidOp;
impl VecDifferentiable for SigmoidOp {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor {
        VecTensor::new(inputs[0].data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect())
    }
    // d(sigma(x))/dx = sigma(x) * (1 - sigma(x)) = output * (1 - output)
    fn backward(&self, grad_output: &VecTensor, _inputs: &[VecTensor], output: &VecTensor) -> VecTensor {
        let out = (0..output.size())
            .map(|i| {
                let o = output.data[i];
                grad_output.data[i] * (o * (1.0 - o))
            })
            .collect();
        VecTensor::new(out)
    }
}

struct TanhOp;
impl VecDifferentiable for TanhOp {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor {
        VecTensor::new(inputs[0].data.iter().map(|&v| v.tanh()).collect())
    }
    // d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - output^2
    fn backward(&self, grad_output: &VecTensor, _inputs: &[VecTensor], output: &VecTensor) -> VecTensor {
        let out = (0..output.size())
            .map(|i| {
                let o = output.data[i];
                grad_output.data[i] * (1.0 - o * o)
            })
            .collect();
        VecTensor::new(out)
    }
}

struct SinOp;
impl VecDifferentiable for SinOp {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor {
        VecTensor::new(inputs[0].data.iter().map(|&v| v.sin()).collect())
    }
    // d(sin(x))/dx = cos(x) -- needs the INPUT, not the output
    fn backward(&self, grad_output: &VecTensor, inputs: &[VecTensor], _output: &VecTensor) -> VecTensor {
        let out = (0..inputs[0].size()).map(|i| grad_output.data[i] * inputs[0].data[i].cos()).collect();
        VecTensor::new(out)
    }
}

struct CosOp;
impl VecDifferentiable for CosOp {
    fn forward(&self, inputs: &[VecTensor]) -> VecTensor {
        VecTensor::new(inputs[0].data.iter().map(|&v| v.cos()).collect())
    }
    // d(cos(x))/dx = -sin(x)
    fn backward(&self, grad_output: &VecTensor, inputs: &[VecTensor], _output: &VecTensor) -> VecTensor {
        let out = (0..inputs[0].size()).map(|i| grad_output.data[i] * (-inputs[0].data[i].sin())).collect();
        VecTensor::new(out)
    }
}

fn fmt_vec(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{:.1}", x)).collect();
    format!("[{}]", parts.join(", "))
}

fn main() {
    println!("=== Section 16.5: ReluOp, SigmoidOp, TanhOp, SinOp, CosOp ===");

    let x = VecTensor::new(vec![-2.0, 3.0, -1.0, 5.0]);
    println!("x = {}", fmt_vec(&x.data));
    let relu_op = ReluOp;
    let relu_out = relu_op.forward(&[x.clone()]);
    println!("relu(x) = {}", fmt_vec(&relu_out.data));
    let ones4 = VecTensor::new(vec![1.0, 1.0, 1.0, 1.0]);
    let relu_grads = relu_op.backward(&ones4, &[x.clone()], &relu_out);
    println!("ReluOp.backward(grad_output=ones) = {}\n", fmt_vec(&relu_grads.data));

    let x0 = VecTensor::new(vec![0.0]);
    let sigmoid_op = SigmoidOp;
    let sig_out = sigmoid_op.forward(&[x0.clone()]);
    let one = VecTensor::new(vec![1.0]);
    let sig_grads = sigmoid_op.backward(&one, &[x0.clone()], &sig_out);
    println!("sigmoid(0) = {:.4}, SigmoidOp.backward(seed=1.0) = {:.4}", sig_out.data[0], sig_grads.data[0]);

    let tanh_op = TanhOp;
    let tanh_out = tanh_op.forward(&[x0.clone()]);
    let tanh_grads = tanh_op.backward(&one, &[x0.clone()], &tanh_out);
    println!("tanh(0) = {:.4}, TanhOp.backward(seed=1.0) = {:.4}", tanh_out.data[0], tanh_grads.data[0]);
    println!("tanh's local slope is {:.1}x steeper than sigmoid's at the origin\n", tanh_grads.data[0] / sig_grads.data[0]);

    let sin_op = SinOp;
    let sin_out = sin_op.forward(&[x0.clone()]);
    let sin_grads = sin_op.backward(&one, &[x0.clone()], &sin_out);
    println!("sin(0) = {:.4}, SinOp.backward(seed=1.0) = {:.4} (= seed * cos(0))", sin_out.data[0], sin_grads.data[0]);

    let cos_op = CosOp;
    let cos_out = cos_op.forward(&[x0.clone()]);
    let cos_grads = cos_op.backward(&one, &[x0.clone()], &cos_out);
    println!("cos(0) = {:.4}, CosOp.backward(seed=1.0) = {:.4} (= seed * -sin(0))\n", cos_out.data[0], cos_grads.data[0]);

    println!("--- with grad_output=2.0 instead of 1.0 ---");
    let two = VecTensor::new(vec![2.0]);
    let sig_grads2 = sigmoid_op.backward(&two, &[x0.clone()], &sig_out);
    println!("SigmoidOp.backward(seed=2.0) = {:.4}", sig_grads2.data[0]);
    let tanh_grads2 = tanh_op.backward(&two, &[x0.clone()], &tanh_out);
    println!("TanhOp.backward(seed=2.0) = {:.4}", tanh_grads2.data[0]);
}
```

```bash
rustc --edition 2024 -O 05_activation_trig_gradients.rs -o 05_activation_trig_gradients
./05_activation_trig_gradients
```

### File: `06_reduction_shape_gradients.rs`

```rust
// d(sum(x))/dx_i = 1 for every i -- the incoming scalar gradient gets
// broadcast back out to every position that was summed.
fn sum_backward(grad_output: f32, n: usize) -> Vec<f32> {
    vec![grad_output; n]
}

// Mirrors Chapter 14.2's max_reduce_kernel comparison exactly: a strict
// `>` means the earlier index wins any tie.
fn tensor_argmax_host(x: &[f32]) -> (f32, usize) {
    let mut best = x[0];
    let mut best_idx = 0;
    for i in 1..x.len() {
        if x[i] > best {
            best = x[i];
            best_idx = i;
        }
    }
    (best, best_idx)
}

// d(max(x))/dx_i = 1 for the winning index, 0 everywhere else -- requires
// the SAME index Chapter 14.2's kernel tracked.
fn max_backward_indexed(grad_output: f32, n: usize, winning_index: usize) -> Vec<f32> {
    let mut grad_x = vec![0.0f32; n];
    grad_x[winning_index] = grad_output;
    grad_x
}

// The broken alternative the COMMON TRAP below demonstrates: building the
// mask by comparing every element against the maximum VALUE, rather than
// carrying an index forward from the reduction.
fn max_backward_value_mask(grad_output: f32, x: &[f32], max_val: f32) -> Vec<f32> {
    x.iter().map(|&v| if v == max_val { grad_output } else { 0.0 }).collect()
}

#[derive(Clone)]
struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

fn transpose(a: &Matrix) -> Matrix {
    let mut data = vec![0.0f32; a.rows * a.cols];
    for i in 0..a.rows {
        for j in 0..a.cols {
            data[j * a.rows + i] = a.get(i, j);
        }
    }
    Matrix { data, rows: a.cols, cols: a.rows }
}

fn print_matrix(label: &str, m: &Matrix) {
    println!("{} ({}x{}):", label, m.rows, m.cols);
    for i in 0..m.rows {
        let row: Vec<String> = (0..m.cols).map(|j| format!("{:5.1}", m.get(i, j))).collect();
        println!("  [{}]", row.join(", "));
    }
}

fn fmt_vec(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{:.1}", x)).collect();
    format!("[{}]", parts.join(", "))
}

fn main() {
    println!("=== Section 16.6: SumOp, MaxOp, ReshapeOp, TransposeOp ===");

    let x = [1.0f32, 4.0, 9.0, 16.0];
    let sum: f32 = x.iter().sum();
    println!("x = {:?}, sum(x) = {:.1}", x, sum);
    let sum_grad = sum_backward(1.0, x.len());
    println!("SumOp.backward(grad_output=1.0) -> {}\n", fmt_vec(&sum_grad));

    let xm = [3.0f32, 7.0, 2.0, 9.0];
    let (max_val, max_idx) = tensor_argmax_host(&xm);
    println!("x = {:?}, max(x) = {:.1} at index {}", xm, max_val, max_idx);
    let max_grad = max_backward_indexed(1.0, xm.len(), max_idx);
    println!("MaxOp.backward(grad_output=1.0) -> {}\n", fmt_vec(&max_grad));

    println!("--- COMMON TRAP: MaxOp.backward on a tie, [1, 5, 3, 5] ---");
    let xt = [1.0f32, 5.0, 3.0, 5.0];
    let (tie_val, tie_idx) = tensor_argmax_host(&xt);
    println!("x = {:?}, max = {:.1}, winning index (earlier of the tie) = {}", xt, tie_val, tie_idx);
    let indexed = max_backward_indexed(1.0, xt.len(), tie_idx);
    let indexed_sum: f32 = indexed.iter().sum();
    println!("indexed backward (correct): {}, sum = {:.1}", fmt_vec(&indexed), indexed_sum);
    let value_mask = max_backward_value_mask(1.0, &xt, tie_val);
    let value_mask_sum: f32 = value_mask.iter().sum();
    println!(
        "value-mask backward (broken): {}, sum = {:.1} -- gradient invented out of a tie\n",
        fmt_vec(&value_mask),
        value_mask_sum
    );

    let flat: Vec<f32> = (0..12).map(|i| i as f32).collect();
    println!("original flat buffer, viewed as [2,6]:");
    for r in 0..2 {
        let row: Vec<String> = (0..6).map(|c| format!("{:5.1}", flat[r * 6 + c])).collect();
        println!("  [{}]", row.join(", "));
    }
    println!("reshaped to [3,4] (forward), viewed as [3,4]:");
    for r in 0..3 {
        let row: Vec<String> = (0..4).map(|c| format!("{:5.1}", flat[r * 4 + c])).collect();
        println!("  [{}]", row.join(", "));
    }
    // ReshapeOp::backward reshapes the gradient back to the ORIGINAL
    // shape -- no data movement, since reshape never moved a single value.
    println!("grad_output reshaped back to [2,6] (backward), viewed as [2,6]:");
    for r in 0..2 {
        let row: Vec<String> = (0..6).map(|c| format!("{:5.1}", flat[r * 6 + c])).collect();
        println!("  [{}]", row.join(", "));
    }
    println!();

    let a = Matrix { data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rows: 2, cols: 3 };
    print_matrix("A", &a);
    let at = transpose(&a);
    print_matrix("A^T (forward)", &at);
    let grad_output = Matrix { data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rows: 3, cols: 2 };
    print_matrix("grad_output", &grad_output);
    // TransposeOp::backward(grad_output) = transpose(grad_output) --
    // transpose applied twice returns every value to its original position.
    let transpose_grad = transpose(&grad_output);
    print_matrix("TransposeOp.backward(grad_output) = transpose(grad_output)", &transpose_grad);
}
```

```bash
rustc --edition 2024 -O 06_reduction_shape_gradients.rs -o 06_reduction_shape_gradients
./06_reduction_shape_gradients
```

### File: `07_custom_function_bisection.rs`

```rust
// The implicit function theorem as an escape hatch: differentiate the
// EQUATION a converged iterative solver satisfies, rather than unrolling
// and differentiating the solver's own loop.
struct CustomFunction {
    forward_fn: Box<dyn Fn(&[f32]) -> f32>,
    backward_fn: Box<dyn Fn(f32, &[f32], f32) -> Vec<f32>>,
}

impl CustomFunction {
    fn forward(&self, inputs: &[f32]) -> f32 {
        (self.forward_fn)(inputs)
    }
    fn backward(&self, grad_output: f32, inputs: &[f32], output: f32) -> Vec<f32> {
        (self.backward_fn)(grad_output, inputs, output)
    }
}

// output holds the converged x; dx/dc = 1 / (2x) from the implicit
// function theorem applied to f(x, c) = x^2 - c = 0.
fn sqrt_via_bisection_backward(grad_output: f32, _inputs: &[f32], output: f32) -> Vec<f32> {
    let local_grad = 1.0 / (2.0 * output);
    vec![grad_output * local_grad]
}

fn bisect_sqrt(c: f32, verbose: bool) -> f32 {
    let mut a = 1.0f32;
    let mut b = 2.0f32;
    for iter in 0..25 {
        let mid = (a + b) / 2.0;
        let mid_sq = mid * mid;
        if verbose {
            let status = if mid_sq > c { "too big, move b" } else { "too small (or exact), move a" };
            println!("  iter {}: mid={:.7}, mid^2={:.7}, {}", iter, mid, mid_sq, status);
        }
        if mid_sq > c {
            b = mid;
        } else {
            a = mid;
        }
    }
    (a + b) / 2.0
}

fn main() {
    println!("=== Section 16.7: CustomFunction, the implicit function theorem, bisection for x^2=2 ===");
    println!("bisecting x^2 = 2, bracket [1, 2]:");
    let converged = bisect_sqrt(2.0, true);
    println!("converged x = {:.7} (true sqrt(2) = {:.7})\n", converged, 2.0f32.sqrt());

    let sqrt_bisection = CustomFunction {
        forward_fn: Box::new(|inputs: &[f32]| bisect_sqrt(inputs[0], false)),
        backward_fn: Box::new(sqrt_via_bisection_backward),
    };

    let c = 2.0f32;
    let x = sqrt_bisection.forward(&[c]);
    println!("CustomFunction.forward(c={:.1}) = {:.7}", c, x);
    let grads = sqrt_bisection.backward(1.0, &[c], x);
    println!("dx/dc = 1/(2x) = 1/(2*{:.7}) = {:.7}\n", x, grads[0]);

    println!("--- finite-difference check, bisecting c=2.001 too ---");
    let x_nudged = bisect_sqrt(2.001, false);
    println!("bisection(c=2.001) converges to x = {:.7}", x_nudged);
    let fd_slope = (x_nudged - x) / 0.001;
    println!("finite-diff slope = ({:.7} - {:.7}) / 0.001 = {:.7}", x_nudged, x, fd_slope);
    println!("analytic dx/dc                              = {:.7}\n", grads[0]);

    println!("--- what happens if sqrt(2) is rounded to 6 digits before the subtraction ---");
    let rounded_x = (x * 100000.0).round() / 100000.0;
    let rounded_x_nudged = (x_nudged * 100000.0).round() / 100000.0;
    println!("rounded x(c=2)     = {:.5}", rounded_x);
    println!("rounded x(c=2.001) = {:.5}", rounded_x_nudged);
    let rounded_slope = (rounded_x_nudged - rounded_x) / 0.001;
    println!("slope from rounded values = ({:.5} - {:.5}) / 0.001 = {:.4}", rounded_x_nudged, rounded_x, rounded_slope);
    let pct_off = ((rounded_slope - grads[0]) / grads[0]).abs() * 100.0;
    println!("that is {:.1}% off the analytic answer of {:.7} -- purely from rounding, not from the calculus", pct_off, grads[0]);
}
```

```bash
rustc --edition 2024 -O 07_custom_function_bisection.rs -o 07_custom_function_bisection
./07_custom_function_bisection
```

### File: `08_full_op_registry.rs`

```rust
use std::collections::HashMap;

// Section 16.8: unlike the CUDA edition, which reached for a different,
// narrowly-typed stand-in per section -- ScalarTensor (16.2, 16.4),
// VecTensor (16.5), Matrix (16.3, 16.6) -- because a single f32, a
// same-shaped elementwise vector, and a 2-D matrix are different Rust
// types too, this file introduces ONE shared representation general
// enough to stand in for all three uniformly: a flat buffer plus a
// row/col shape, so all eighteen ops can be registered against one
// Differentiable trait object.
#[derive(Clone)]
struct Tensor {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    winning_index: i32, // used only by MaxOp -- see Section 16.6's own note
                         // that this op needs a value the fixed
                         // (grad_output, inputs, output) signature doesn't carry.
}

impl Tensor {
    fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        Tensor { data, rows, cols, winning_index: -1 }
    }
    fn scalar(v: f32) -> Self {
        Tensor::new(1, 1, vec![v])
    }
    fn vec(v: Vec<f32>) -> Self {
        let n = v.len();
        Tensor::new(1, n, v)
    }
    fn size(&self) -> usize {
        self.rows * self.cols
    }
}

trait Differentiable {
    fn forward(&self, inputs: &[Tensor]) -> Tensor;
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], output: &Tensor) -> Vec<Tensor>;
}

// ---- elementwise arithmetic (16.2 / 16.4) ----

struct AddOp;
impl Differentiable for AddOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for i in 0..out.size() {
            out.data[i] += inputs[1].data[i];
        }
        out
    }
    // grad_output.clone() genuinely allocates a NEW Vec each time --
    // Worked Example 16.2's finding, holding here too: returning the same
    // owned Tensor twice never aliases in idiomatic Rust.
    fn backward(&self, grad_output: &Tensor, _inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.clone(), grad_output.clone()]
    }
}
struct SubOp;
impl Differentiable for SubOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for i in 0..out.size() {
            out.data[i] -= inputs[1].data[i];
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, _inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        let mut grad_b = grad_output.clone();
        for v in grad_b.data.iter_mut() {
            *v = -*v;
        }
        vec![grad_output.clone(), grad_b]
    }
}
struct MulOp;
impl Differentiable for MulOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for i in 0..out.size() {
            out.data[i] *= inputs[1].data[i];
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        let mut grad_a = grad_output.clone();
        let mut grad_b = grad_output.clone();
        for i in 0..grad_output.size() {
            grad_a.data[i] = grad_output.data[i] * inputs[1].data[i];
            grad_b.data[i] = grad_output.data[i] * inputs[0].data[i];
        }
        vec![grad_a, grad_b]
    }
}
struct DivOp;
impl Differentiable for DivOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for i in 0..out.size() {
            out.data[i] /= inputs[1].data[i];
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        let mut grad_a = grad_output.clone();
        let mut grad_b = grad_output.clone();
        for i in 0..grad_output.size() {
            let a = inputs[0].data[i];
            let b = inputs[1].data[i];
            grad_a.data[i] = grad_output.data[i] / b;
            grad_b.data[i] = grad_output.data[i] * ((-a) / (b * b));
        }
        vec![grad_a, grad_b]
    }
}
struct PowOp;
impl Differentiable for PowOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for i in 0..out.size() {
            out.data[i] = inputs[0].data[i].powf(inputs[1].data[i]);
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], output: &Tensor) -> Vec<Tensor> {
        let mut grad_a = grad_output.clone();
        let mut grad_b = grad_output.clone();
        for i in 0..grad_output.size() {
            let a = inputs[0].data[i];
            let b = inputs[1].data[i];
            grad_a.data[i] = grad_output.data[i] * (b * a.powf(b - 1.0));
            grad_b.data[i] = grad_output.data[i] * (output.data[i] * a.ln());
        }
        vec![grad_a, grad_b]
    }
}
struct ExpOp;
impl Differentiable for ExpOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for v in out.data.iter_mut() {
            *v = v.exp();
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, _inputs: &[Tensor], output: &Tensor) -> Vec<Tensor> {
        let mut grad = grad_output.clone();
        for i in 0..grad_output.size() {
            grad.data[i] = grad_output.data[i] * output.data[i];
        }
        vec![grad]
    }
}
struct LogOp;
impl Differentiable for LogOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for v in out.data.iter_mut() {
            *v = v.ln();
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        let mut grad = grad_output.clone();
        for i in 0..grad_output.size() {
            grad.data[i] = grad_output.data[i] / inputs[0].data[i];
        }
        vec![grad]
    }
}
struct SqrtOp;
impl Differentiable for SqrtOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for v in out.data.iter_mut() {
            *v = v.sqrt();
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, _inputs: &[Tensor], output: &Tensor) -> Vec<Tensor> {
        let mut grad = grad_output.clone();
        for i in 0..grad_output.size() {
            grad.data[i] = grad_output.data[i] / (2.0 * output.data[i]);
        }
        vec![grad]
    }
}

// ---- activations and trig (16.5) ----

struct ReluOp;
impl Differentiable for ReluOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for v in out.data.iter_mut() {
            *v = if *v > 0.0 { *v } else { 0.0 };
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        let mut grad = grad_output.clone();
        for i in 0..grad_output.size() {
            grad.data[i] = grad_output.data[i] * if inputs[0].data[i] > 0.0 { 1.0 } else { 0.0 };
        }
        vec![grad]
    }
}
struct SigmoidOp;
impl Differentiable for SigmoidOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for v in out.data.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, _inputs: &[Tensor], output: &Tensor) -> Vec<Tensor> {
        let mut grad = grad_output.clone();
        for i in 0..grad_output.size() {
            let o = output.data[i];
            grad.data[i] = grad_output.data[i] * (o * (1.0 - o));
        }
        vec![grad]
    }
}
struct TanhOp;
impl Differentiable for TanhOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for v in out.data.iter_mut() {
            *v = v.tanh();
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, _inputs: &[Tensor], output: &Tensor) -> Vec<Tensor> {
        let mut grad = grad_output.clone();
        for i in 0..grad_output.size() {
            let o = output.data[i];
            grad.data[i] = grad_output.data[i] * (1.0 - o * o);
        }
        vec![grad]
    }
}
struct SinOp;
impl Differentiable for SinOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for v in out.data.iter_mut() {
            *v = v.sin();
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        let mut grad = grad_output.clone();
        for i in 0..grad_output.size() {
            grad.data[i] = grad_output.data[i] * inputs[0].data[i].cos();
        }
        vec![grad]
    }
}
struct CosOp;
impl Differentiable for CosOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut out = Tensor::new(inputs[0].rows, inputs[0].cols, inputs[0].data.clone());
        for v in out.data.iter_mut() {
            *v = v.cos();
        }
        out
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        let mut grad = grad_output.clone();
        for i in 0..grad_output.size() {
            grad.data[i] = grad_output.data[i] * (-inputs[0].data[i].sin());
        }
        vec![grad]
    }
}

// ---- matrix multiplication (16.3) ----

fn matmul_raw(a: &Tensor, b: &Tensor) -> Tensor {
    let mut data = vec![0.0f32; a.rows * b.cols];
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            data[i * b.cols + j] = sum;
        }
    }
    Tensor::new(a.rows, b.cols, data)
}
fn transpose_raw(a: &Tensor) -> Tensor {
    let mut data = vec![0.0f32; a.rows * a.cols];
    for i in 0..a.rows {
        for j in 0..a.cols {
            data[j * a.rows + i] = a.data[i * a.cols + j];
        }
    }
    Tensor::new(a.cols, a.rows, data)
}
struct MatMulOp;
impl Differentiable for MatMulOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        matmul_raw(&inputs[0], &inputs[1])
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        let grad_a = matmul_raw(grad_output, &transpose_raw(&inputs[1]));
        let grad_b = matmul_raw(&transpose_raw(&inputs[0]), grad_output);
        vec![grad_a, grad_b]
    }
}

// ---- reductions and shape ops (16.6) ----

struct SumOp;
impl Differentiable for SumOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let total: f32 = inputs[0].data.iter().sum();
        Tensor::scalar(total)
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        vec![Tensor::new(inputs[0].rows, inputs[0].cols, vec![grad_output.data[0]; inputs[0].size()])]
    }
}
struct MaxOp;
impl Differentiable for MaxOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut best = inputs[0].data[0];
        let mut idx = 0usize;
        for i in 1..inputs[0].size() {
            if inputs[0].data[i] > best {
                best = inputs[0].data[i];
                idx = i;
            }
        }
        let mut out = Tensor::scalar(best);
        out.winning_index = idx as i32;
        out
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], output: &Tensor) -> Vec<Tensor> {
        let mut grad = Tensor::new(inputs[0].rows, inputs[0].cols, vec![0.0f32; inputs[0].size()]);
        grad.data[output.winning_index as usize] = grad_output.data[0];
        vec![grad]
    }
}
struct ReshapeOp {
    target_rows: usize,
    target_cols: usize, // the only per-call state any op in this registry carries
}
impl Differentiable for ReshapeOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        Tensor::new(self.target_rows, self.target_cols, inputs[0].data.clone()) // no data movement, only shape changes
    }
    fn backward(&self, grad_output: &Tensor, inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        vec![Tensor::new(inputs[0].rows, inputs[0].cols, grad_output.data.clone())] // reshape the gradient back to the ORIGINAL shape
    }
}
struct TransposeOp;
impl Differentiable for TransposeOp {
    fn forward(&self, inputs: &[Tensor]) -> Tensor {
        transpose_raw(&inputs[0])
    }
    fn backward(&self, grad_output: &Tensor, _inputs: &[Tensor], _output: &Tensor) -> Vec<Tensor> {
        vec![transpose_raw(grad_output)]
    }
}

// ---- OpRegistry, chain_rule_step, build_op_registry (16.1 / 16.8) ----

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
    grad_output: &Tensor,
    inputs: &[Tensor],
    output: &Tensor,
) -> Vec<Tensor> {
    let op = registry.get(op_name);
    op.backward(grad_output, inputs, output)
}

// Box<dyn Differentiable> owns each op outright, so unlike the CUDA
// edition's build_op_registry (which takes eighteen separate reference
// parameters into locals a caller has to keep alive in main()'s own
// scope), this version needs no parameters at all -- it constructs and
// boxes every op itself, and the registry owns all eighteen for as long
// as it exists. There is no lifetime for a caller to manage.
fn build_op_registry() -> OpRegistry {
    let mut registry = OpRegistry::new();
    registry.register_op("add", Box::new(AddOp));
    registry.register_op("sub", Box::new(SubOp));
    registry.register_op("mul", Box::new(MulOp));
    registry.register_op("div", Box::new(DivOp));
    registry.register_op("pow", Box::new(PowOp));
    registry.register_op("exp", Box::new(ExpOp));
    registry.register_op("log", Box::new(LogOp));
    registry.register_op("sqrt", Box::new(SqrtOp));
    registry.register_op("relu", Box::new(ReluOp));
    registry.register_op("sigmoid", Box::new(SigmoidOp));
    registry.register_op("tanh", Box::new(TanhOp));
    registry.register_op("sin", Box::new(SinOp));
    registry.register_op("cos", Box::new(CosOp));
    registry.register_op("matmul", Box::new(MatMulOp));
    registry.register_op("sum", Box::new(SumOp));
    registry.register_op("max", Box::new(MaxOp));
    registry.register_op("reshape", Box::new(ReshapeOp { target_rows: 0, target_cols: 0 }));
    registry.register_op("transpose", Box::new(TransposeOp));
    registry
}

fn print_flat(label: &str, t: &Tensor) {
    let parts: Vec<String> = t.data.iter().map(|v| format!("{:.4}", v)).collect();
    println!("{} = [{}]", label, parts.join(", "));
}

fn main() {
    println!("=== Section 16.8: every op wired into ONE registry, genuinely compiled and run ===");
    println!("(all eighteen ops below are registered under one Differentiable trait object and");
    println!(" exercised through chain_rule_step, with every result checked against the exact");
    println!(" numbers Sections 16.2-16.6 derived by hand.)\n");

    let registry = build_op_registry();
    println!("registry.ops.len() = {} (18 named ops; CustomFunction stays unregistered, per Section 16.7)\n", registry.ops.len());

    let mut checks_passed = 0;
    let mut checks_total = 0;
    let mut check = |name: &str, got: f32, expected: f32, tol: f32| {
        checks_total += 1;
        let ok = (got - expected).abs() < tol;
        if ok {
            checks_passed += 1;
        }
        println!("  [{}] got={:.5} expected={:.5} -> {}", name, got, expected, if ok { "MATCH" } else { "MISMATCH" });
    };

    // add/mul: w = x*y + x, x=3, y=4 (Sections 16.1/16.2)
    let x = Tensor::scalar(3.0);
    let y = Tensor::scalar(4.0);
    let z = registry.get("mul").forward(&[x.clone(), y.clone()]);
    let w = registry.get("add").forward(&[z.clone(), x.clone()]);
    let add_g = chain_rule_step(&registry, "add", &Tensor::scalar(1.0), &[z.clone(), x.clone()], &w);
    let mul_g = chain_rule_step(&registry, "mul", &add_g[0], &[x.clone(), y.clone()], &z);
    println!("--- add/mul (w=x*y+x) ---");
    check("x.grad", add_g[1].data[0] + mul_g[0].data[0], 5.0, 1e-4);
    check("y.grad", mul_g[1].data[0], 3.0, 1e-4);

    // sub/div: a=8, b=5 (Section 16.4.1)
    let a48 = Tensor::scalar(8.0);
    let b48 = Tensor::scalar(5.0);
    let sub_out = registry.get("sub").forward(&[a48.clone(), b48.clone()]);
    let sub_g = chain_rule_step(&registry, "sub", &Tensor::scalar(1.0), &[a48.clone(), b48.clone()], &sub_out);
    let div_out = registry.get("div").forward(&[a48.clone(), b48.clone()]);
    let div_g = chain_rule_step(&registry, "div", &Tensor::scalar(1.0), &[a48.clone(), b48.clone()], &div_out);
    println!("--- sub/div (a=8, b=5) ---");
    check("SubOp grad_a", sub_g[0].data[0], 1.0, 1e-4);
    check("SubOp grad_b", sub_g[1].data[0], -1.0, 1e-4);
    check("DivOp grad_a", div_g[0].data[0], 0.2, 1e-4);
    check("DivOp grad_b", div_g[1].data[0], -0.32, 1e-4);

    // pow/log: a=2, b=3 (Section 16.4.2)
    let a23 = Tensor::scalar(2.0);
    let b23 = Tensor::scalar(3.0);
    let pow_out = registry.get("pow").forward(&[a23.clone(), b23.clone()]);
    let pow_g = chain_rule_step(&registry, "pow", &Tensor::scalar(1.0), &[a23.clone(), b23.clone()], &pow_out);
    let log_out = registry.get("log").forward(&[a23.clone()]);
    let log_g = chain_rule_step(&registry, "log", &Tensor::scalar(1.0), &[a23.clone()], &log_out);
    println!("--- pow/log (a=2, b=3) ---");
    check("PowOp grad_a", pow_g[0].data[0], 12.0, 1e-3);
    check("PowOp grad_b", pow_g[1].data[0], 5.545, 2e-3);
    check("LogOp grad", log_g[0].data[0], 0.5, 1e-4);

    // exp/sqrt (Sections 16.2.3, 16.4)
    let x1 = Tensor::scalar(1.0);
    let exp_out = registry.get("exp").forward(&[x1.clone()]);
    let exp_g = chain_rule_step(&registry, "exp", &Tensor::scalar(1.0), &[x1.clone()], &exp_out);
    let x4 = Tensor::scalar(4.0);
    let sqrt_out = registry.get("sqrt").forward(&[x4.clone()]);
    let sqrt_g = chain_rule_step(&registry, "sqrt", &Tensor::scalar(1.0), &[x4.clone()], &sqrt_out);
    println!("--- exp/sqrt ---");
    check("ExpOp grad (x=1)", exp_g[0].data[0], 2.71828, 1e-3);
    check("SqrtOp grad (x=4)", sqrt_g[0].data[0], 0.25, 1e-4);

    // relu on [-2,3,-1,5] (Section 16.5.1)
    let xr = Tensor::vec(vec![-2.0, 3.0, -1.0, 5.0]);
    let relu_out = registry.get("relu").forward(&[xr.clone()]);
    let relu_g = chain_rule_step(&registry, "relu", &Tensor::vec(vec![1.0, 1.0, 1.0, 1.0]), &[xr.clone()], &relu_out);
    println!("--- relu ---");
    print_flat("ReluOp.backward", &relu_g[0]);
    check("relu[0]", relu_g[0].data[0], 0.0, 1e-4);
    check("relu[1]", relu_g[0].data[1], 1.0, 1e-4);
    check("relu[2]", relu_g[0].data[2], 0.0, 1e-4);
    check("relu[3]", relu_g[0].data[3], 1.0, 1e-4);

    // sigmoid/tanh/sin/cos at x=0 (Sections 16.5.2, 16.5.3)
    let x0 = Tensor::scalar(0.0);
    let sig_out = registry.get("sigmoid").forward(&[x0.clone()]);
    let sig_g = chain_rule_step(&registry, "sigmoid", &Tensor::scalar(1.0), &[x0.clone()], &sig_out);
    let tanh_out = registry.get("tanh").forward(&[x0.clone()]);
    let tanh_g = chain_rule_step(&registry, "tanh", &Tensor::scalar(1.0), &[x0.clone()], &tanh_out);
    let sin_out = registry.get("sin").forward(&[x0.clone()]);
    let sin_g = chain_rule_step(&registry, "sin", &Tensor::scalar(1.0), &[x0.clone()], &sin_out);
    let cos_out = registry.get("cos").forward(&[x0.clone()]);
    let cos_g = chain_rule_step(&registry, "cos", &Tensor::scalar(1.0), &[x0.clone()], &cos_out);
    println!("--- sigmoid/tanh/sin/cos at x=0 ---");
    check("SigmoidOp grad", sig_g[0].data[0], 0.25, 1e-4);
    check("TanhOp grad", tanh_g[0].data[0], 1.0, 1e-4);
    check("SinOp grad", sin_g[0].data[0], 1.0, 1e-4);
    check("CosOp grad", cos_g[0].data[0], 0.0, 1e-4);

    // matmul: X(2x3) @ M(3x2) (Section 16.3)
    let x_mat = Tensor::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let m_mat = Tensor::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y_mat = registry.get("matmul").forward(&[x_mat.clone(), m_mat.clone()]);
    let ones22 = Tensor::new(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
    let mm_g = chain_rule_step(&registry, "matmul", &ones22, &[x_mat.clone(), m_mat.clone()], &y_mat);
    println!("--- matmul ---");
    print_flat("dL/dX", &mm_g[0]);
    print_flat("dL/dM", &mm_g[1]);
    check("dL/dX[0]", mm_g[0].data[0], 3.0, 1e-4);
    check("dL/dX[2]", mm_g[0].data[2], 11.0, 1e-4);
    check("dL/dM[0]", mm_g[1].data[0], 5.0, 1e-4);
    check("dL/dM[5]", mm_g[1].data[5], 9.0, 1e-4);

    // sum/max (Section 16.6)
    let xs = Tensor::vec(vec![1.0, 4.0, 9.0, 16.0]);
    let sum_out = registry.get("sum").forward(&[xs.clone()]);
    let sum_g = chain_rule_step(&registry, "sum", &Tensor::scalar(1.0), &[xs.clone()], &sum_out);
    let xm = Tensor::vec(vec![3.0, 7.0, 2.0, 9.0]);
    let max_out = registry.get("max").forward(&[xm.clone()]);
    let max_g = chain_rule_step(&registry, "max", &Tensor::scalar(1.0), &[xm.clone()], &max_out);
    println!("--- sum/max ---");
    print_flat("SumOp.backward", &sum_g[0]);
    print_flat("MaxOp.backward", &max_g[0]);
    check("sum_grad[0]", sum_g[0].data[0], 1.0, 1e-4);
    check("max_grad winning index value", max_g[0].data[3], 1.0, 1e-4);
    check("max_grad non-winner", max_g[0].data[0], 0.0, 1e-4);

    // reshape/transpose (Section 16.6.3)
    let flat = Tensor::new(2, 6, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    let reshape34 = ReshapeOp { target_rows: 3, target_cols: 4 };
    let reshaped = reshape34.forward(&[flat.clone()]);
    let reshape_g = reshape34.backward(&Tensor::new(3, 4, reshaped.data.clone()), &[flat.clone()], &reshaped);
    println!("--- reshape/transpose ---");
    println!("reshape_g[0] shape = [{},{}] (original shape restored)", reshape_g[0].rows, reshape_g[0].cols);
    check("reshape roundtrip shape rows", reshape_g[0].rows as f32, 2.0, 1e-4);
    check("reshape roundtrip shape cols", reshape_g[0].cols as f32, 6.0, 1e-4);

    let a_mat = Tensor::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let at_mat = registry.get("transpose").forward(&[a_mat.clone()]);
    let grad_out_t = Tensor::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t_g = chain_rule_step(&registry, "transpose", &grad_out_t, &[a_mat.clone()], &at_mat);
    print_flat("TransposeOp.backward", &t_g[0]);
    check("transpose_grad[0]", t_g[0].data[0], 1.0, 1e-4);
    check("transpose_grad[3]", t_g[0].data[3], 2.0, 1e-4);

    println!("\n{} / {} checks passed against the exact numbers Sections 16.2-16.6 derived by hand.", checks_passed, checks_total);
    std::process::exit(if checks_passed == checks_total { 0 } else { 1 });
}
```

```bash
rustc --edition 2024 -O 08_full_op_registry.rs -o 08_full_op_registry
./08_full_op_registry
```

## Chapter Summary

The multivariable chain rule is nothing more than summing a value's contribution along every path it takes to reach the output — traced concretely on `x`'s two routes into `w`, matching `∂w/∂x = 5` three separate ways and genuinely dispatched through one `chain_rule_step` function. `AddOp` passes its incoming gradient through unchanged to both inputs, and this chapter got a genuinely Rust-specific answer to the aliasing question it raised rather than leaving it open: idiomatic `Vec<f32>` cloning allocates two independent buffers every time, confirmed by comparing pointers and by mutating one without disturbing the other, and reproducing the CUDA edition's aliased-pointer hazard on purpose takes an explicit, opt-in `Rc<RefCell<Vec<f32>>>`. `MulOp` scales the incoming gradient by whichever input it *isn't* computing the gradient for, which is exactly why `backward` needs `inputs` at all. `ExpOp` demonstrated the other half of `GraphNode`'s design: some backward rules need the forward `output`, not just the forward arguments, to avoid recomputing an identical value a second time. `MatMulOp`'s backward rule, `grad_output @ Bᵀ` and `Aᵀ @ grad_output`, isn't just asserted in this chapter — it's derived from the same index-summation form Chapter 13.1 used for the forward pass, then verified with real numbers on both `dL/dX` (`[[3,7,11],[3,7,11]]`) and `dL/dM` (`[[5,5],[7,7],[9,9]]`), each checked against its own operand's shape.

The registry doesn't stop at those four operations, and neither does this chapter. Section 16.4 filled in the rest of the element-wise arithmetic a real framework needs — `SubOp` (`da=1, db=-1`), `DivOp` (`da=1/b, db=-a/b²`), `PowOp` (`da=b·a^(b-1)`, reusing `output` for `db=output·ln(a)` the same way `ExpOp` does), `LogOp` (`da=1/x`), and `SqrtOp` (`da=1/(2·output)`, another `output`-reusing rule) — with `DivOp`'s and `LogOp`'s checked against a genuine finite-difference nudge, landing within the ordinary one-sided finite-difference error of their analytic answers. Section 16.5 derived five activation and trigonometric gradients from their own local derivatives, on a `VecTensor` stand-in general enough to hold `ReluOp`'s four-entry mixed-sign example: `ReluOp`'s gradient is a hard `0`/`1` mask on where the input was positive, `SigmoidOp` and `TanhOp` both reuse their cached `output` (`output·(1-output)` and `1-output²` respectively) the way `ExpOp` first modeled, and `SinOp`/`CosOp` differentiate into each other with a 90° phase relationship, genuinely confirmed down to IEEE-754 negative zero — the same bit pattern C++ produces, because both languages implement the same floating-point standard. Section 16.6 covered the two shapes of operation Part 2 computes but doesn't yet differentiate: `SumOp` broadcasts its scalar gradient back out to every element that was summed, `MaxOp` routes the entire incoming gradient through the single winning index Chapter 14.2 already tracks and zeros every other entry (a choice, not a fact, once two inputs tie for the maximum — genuinely demonstrated to invent a spurious extra `1.0` of gradient when a mask is built from the maximum *value* instead), and `ReshapeOp`/`TransposeOp` simply undo, on the gradient, exactly the shape operation they applied on the forward pass. Section 16.7's implicit function theorem showed that a value produced by an iterative solver — bisection, standing in for the bond-pricing solver Part 7 differentiates through — doesn't need its loop unrolled and differentiated step by step; treating the converged answer as implicitly defined by the equation it satisfies collapses the entire gradient into one closed-form expression, genuinely verified against a finite-difference check that also demonstrated, with a real ~1.8%-off result, exactly how premature rounding can make a correct gradient look wrong — and that same check honestly reported a one-ULP difference from the CUDA edition's own converged value, a real artifact of independently-compiled floating-point arithmetic rather than either edition being wrong. Finally, this chapter's consolidated registry was genuinely compiled and run: all eighteen named ops, reimplemented against one shared `Tensor` type general enough to unify the three narrower stand-ins earlier sections needed, dispatched through `chain_rule_step`, and checked against all thirty of this chapter's hand-derived numbers at once — every one a match, and `build_op_registry` itself came out simpler than a literal port of the CUDA edition's version would have, since `Box<dyn Differentiable>` needs no caller-managed lifetime the way a registry of raw pointers does.

## Self-Check Questions

1. For `w = x*y + x` with `x=5.0, y=2.0` (the numbers from Chapter 15's Self-Check Question 1), trace both backward steps: what does `AddOp::backward` return, what does `MulOp::backward` return, and what is the final `x.grad`?
2. `MulOp::backward` computes `grad_a = grad_output * inputs[1].value`. If `inputs[1]` (i.e. `y`) were `0.0` instead of `4.0`, what would `x`'s contribution from this node be, and does that match what `∂z/∂x = y` predicts when `y = 0`?
3. Using the same index-summation derivation Section 16.3 used for `∂L/∂X`, and given `grad_output` is *not* a matrix of all ones but instead `[[1, 0], [0, 1]]` (the 2×2 identity), compute `dL/dX = grad_output @ Mᵀ` for this chapter's running `M`. (Recall `Mᵀ = [[1,3,5],[2,4,6]]`.)
4. `ReluOp::backward` builds its mask from `inputs[0]`, not `output` — a strict `inputs[0].data[i] > 0.0`. For `x = [-3.0, 2.0, 0.0, -1.0, 5.0]` and `grad_output = [1.0, 1.0, 1.0, 1.0, 1.0]`, what is `grad_x`? What does the mask do with the `x=0.0` entry specifically, and does that match the mathematical fact that ReLU has no defined derivative at exactly `0`?
5. `SigmoidOp::backward` computes `grad_a = grad_output * output * (1-output)`; `TanhOp::backward` computes `grad_a = grad_output * (1-output²)`. At `x=0`, `sigmoid(0)=0.5` and `tanh(0)=0`. If both ops receive the same `grad_output = 2.0` at `x=0`, what does each pass back to its input, and which activation has the steeper local slope at the origin?

## Where We Go Next

Chapter 17 (`part4/02-gradient-computation-engine.md`) is where every backward rule this chapter derived actually gets *run* by a general engine rather than by hand-written `main` code calling each `backward` in sequence: seeding `w.grad = 1.0`, walking `[add, mul]` — the reverse order Chapter 15 established — calling `chain_rule_step` at each stop, and accumulating `x`'s two contributions (`1.0` from `AddOp`, `4.0` from `MulOp`) into a final `x.grad = 5.0`. It's also where this chapter's own answer to the `AddOp::backward` aliasing question gets put to use: since idiomatic `Vec<f32>` cloning never aliases, Chapter 17's `accumulate_gradient` is free to mutate a gradient buffer in place without the corruption risk the CUDA edition's raw-pointer version has to design around — and it's where the same reverse pass runs unmodified over any of the fourteen additional ops Sections 16.4 through 16.6 added to the registry, since a `GradientEngine` never needs to inspect which `Differentiable` implementation a node holds.

## Worked Solutions

**1.** `AddOp::backward` (seed `1.0`) returns `[1.0, 1.0]` — `z`'s gradient and `x`'s first contribution. `MulOp::backward`, receiving `z`'s gradient of `1.0` and `inputs = [x=5.0, y=2.0]`, computes `grad_x = 1.0 × y = 1.0 × 2.0 = 2.0` and `grad_y = 1.0 × x = 1.0 × 5.0 = 5.0`. Final `x.grad = 1.0 (from AddOp) + 2.0 (from MulOp) = 3.0`; `y.grad = 5.0` directly. Cross-check with calculus: `∂w/∂x = y+1 = 2+1 = 3` and `∂w/∂y = x = 5` — both match, and both were genuinely recomputed rather than assumed.

**2.** With `y = 0.0`, `grad_a = grad_output * 0.0 = 0.0` — `x`'s contribution from the `mul` node would be exactly `0.0`. This matches `∂z/∂x = y` directly: when `y = 0`, nudging `x` doesn't move `z = x·y` at all, since anything times `0` is `0`, so a local derivative of `0` is exactly correct, not a sign of anything broken.

**3.** `grad_output @ Mᵀ` with `grad_output = [[1,0],[0,1]]` and `Mᵀ = [[1,3,5],[2,4,6]]`: row `0` of the identity picks out row `0` of `Mᵀ` unchanged: `[1,3,5]`. Row `1` of the identity picks out row `1` of `Mᵀ` unchanged: `[2,4,6]`. So `dL/dX = [[1,3,5],[2,4,6]]` — `Mᵀ` back unchanged, and the right `[2,3]` shape for a gradient on `X`, genuinely confirmed by running the same nested-loop matmul this chapter uses throughout. This is the identity fact Chapter 13.4 verified for forward matrix multiplication (`A @ I = A`), arriving here from the other side as `I @ A = A`, since it's the *upstream gradient* that happens to be the identity this time rather than the operand.

**4.** `grad_x = [0.0, 1.0, 0.0, 0.0, 1.0]` — the mask is `1` where `x > 0` (indices `1` and `4`, values `2.0` and `5.0`) and `0` everywhere else, including at `x = 0.0`. The mask's strict `>` comparison treats the `x=0.0` entry as failing the test, giving it a gradient of `0`, not `1`. This matches reality only by convention: ReLU's true derivative is undefined at exactly `x=0` (the function has a corner there, not a well-defined slope), so any autograd engine has to pick one of the two one-sided derivatives — `0` or `1` — as a *subgradient*, and this implementation's strict inequality is what fixes its choice at `0`.

**5.** `SigmoidOp` passes back `2.0 × 0.5 × (1-0.5) = 2.0 × 0.25 = 0.5`. `TanhOp` passes back `2.0 × (1-0²) = 2.0 × 1 = 2.0`. Tanh has the steeper local slope at the origin (local derivative `1` versus sigmoid's `0.25`) — the same four-times-steeper relationship Worked Example 16.5.2 traced directly from the two derivative formulas, now confirmed by pushing an actual gradient value through both — genuinely reproduced in this chapter's own file as `SigmoidOp.backward(seed=2.0) = 0.5000` and `TanhOp.backward(seed=2.0) = 2.0000`.
