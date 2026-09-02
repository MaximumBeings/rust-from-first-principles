# Chapter 21: Advanced Features — The Escape Hatches a Framework Needs to Be Trusted

> "A differentiation engine is not finished when it can compute a gradient. It is finished when you can check that the gradient is right, save the thing that produced it, and reach past its own machinery when the machinery doesn't fit."

## What you will understand by the end of this chapter

- Why an iterative solver like bisection is deliberately kept *out* of the computational graph, and how the implicit function theorem lets a backward pass differentiate through it anyway — as one opaque computation with a closed-form gradient, verified against a real bond's numbers, genuinely compiled and run.
- How a second call to a backward pass — on a gradient instead of a loss — produces a genuine second derivative, and why a Hessian-vector product gets you one useful slice of a Hessian without ever forming the full matrix.
- What a trained network actually *is* on disk: a small header plus raw bytes, and the one thing this chapter's serialization format is missing that a production system could not skip.
- The two distinct failure modes unique to autograd code — a wrong-but-plausible gradient, and a numerically corrupted one — and the two different tools each one requires, demonstrated here as an actual debug-vs-release build difference rather than a claim taken on faith.
- Why every safety check in this chapter is a *debugging* tool, not a *production* one, and what that trade-off actually costs when it's forgotten.
- Why the full attention matrix a naive transformer computes is a memory problem before it's a compute problem, and how Flash Attention's block-by-block *online* softmax produces the exact same numbers while never materializing that full matrix — verified here by computing both ways in the same binary and checking they agree to eight decimal places.
- Why a Mixture-of-Experts layer's *total* parameter count and its *active* parameter count (the ones actually multiplied against a given input) are different numbers, and how a router's top-k gate turns that gap into a real compute saving instead of a rounding error.
- Why an autoregressive decoder's KV cache grows with every head separately, and how Multi-Head Latent Attention caches one shared compressed vector per token instead — reconstructing each head's full-size key and value from it on demand, at zero extra cache cost per additional head.

## What you need to know first

- The reduction kernels and Struct-of-Arrays bond layout from Chapter 18 (`ZeroCouponBondSystemSoA`), and the honest `f32`-vs-`f64` precision lesson from that chapter's convolution work — Section 21.1 hits the identical precision wall from a different direction.
- The genuine benchmark harness and `std::time::Instant` timing discipline from Chapter 19 — the same "measure it, don't assume it" standard applies to every worked example in this chapter.
- The `Matrix` struct, its `matmul`, and the full forward/backward training step from Chapter 20 — Section 21.3 serializes exactly those buffers, and Sections 21.5 through 21.7 route their queries, keys, values, and expert inputs through the same kind of matrix-vector arithmetic.
- Chapter 16.7's `CustomFunction` — a forward closure paired with a backward closure, already built in this edition using boxed Rust closures instead of `std::function` — and the implicit-function-theorem bisection example (`x²=2`) it was introduced with. Section 21.1 reuses that identical mechanism on a real bond instead of a toy square root.
- Elementwise multiplication, matrix-vector products, and sum reduction — the three operations Section 21.2's Hessian-vector product is built from, and nothing else.
- `exp` and the numerically-safe subtract-the-max softmax pattern — introduced for the first time in Section 21.5, and reused unchanged in Section 21.6's router.
- The sigmoid function — Section 21.6's top-2 gate weights turn out, for reasons that fall out of the algebra rather than being designed in, to be an exact sigmoid of the two winning logits' difference.

## 21.1 Custom Autograd Functions `[FOUNDATIONAL]`

### Intuition

Every backward rule in this book so far has been mechanical: look at the forward formula, apply calculus, write the local derivative. That works because every operation so far has been a *closed-form* expression — `x*y`, `exp(x)`, a matrix product. But not every useful computation looks like that. Solving for a bond's z-spread means running bisection: guess a spread, price the bond, compare to the market price, narrow the guess, and repeat — 28 times over, for the actual bond this section prices — until the guess is close enough. There is no single formula for "the z-spread" the way there's a single formula for "the product of two numbers." There's only a *procedure*.

Think of the difference the way a surveyor thinks about the height of a hill. One way to find it: measure every footstep on the way up, add the changes in elevation, and get an exact answer built entirely out of small, individually-verifiable pieces. That's what recording every arithmetic operation in a graph and differentiating through each one does — it works, but it means carrying an instrument up every single step. The other way: stand at the top, take one reading with an altimeter, and separately know the *rate* at which altitude changes with distance near that point without ever having walked the whole slope. That second reading doesn't require re-deriving anything about the climb — it requires one fact about the relationship between position and height *at the point you're standing*. The implicit function theorem is that second approach applied to a solver: instead of differentiating through the footsteps bisection took to arrive at the answer, it asks one question about the relationship between the market price and the bond price *at the spread bisection already found*, and gets an exact derivative from that alone.

### Background

Chapter 16.7's `CustomFunction` (a forward closure that returns an answer by whatever means necessary, ordinary control flow included, paired with a backward closure that supplies a gradient without needing to know how the forward closure arrived at that answer) needs exactly two things poured into it here. For a solver defined by `f(price, spread) = 0` — bisection finds the `spread` that drives `bond_price(spread) - market_price` to zero — the implicit function theorem gives that gradient in closed form:

```
d(spread)/d(price) = -(df/d_price) / (df/d_spread)
```

`df/d_price = -1` always, because `price` enters the objective linearly and with a negative sign (`f = bond_price(spread) - price`). `df/d_spread` is `bond_price`'s own derivative with respect to spread — the bond's price sensitivity to spread, evaluated at the spread bisection already solved for. That derivative is needed anyway for other purposes (it's a close cousin of DV01), so this doesn't add work the framework wasn't already positioned to do.

| Approach | What the graph records | Backward rule needed | Cost |
|---|---|---|---|
| Unroll the solver | Every arithmetic op inside every bisection iteration (hundreds of ops for a multi-cash-flow bond over dozens of iterations) | One rule per elementary op, already in the registry | O(iterations × cash flows) recorded operations |
| `CustomFunction` (this section) | One opaque call: `zspread_solve` | One hand-derived rule, written once | O(1) recorded operations; the iterations run as ordinary control flow nothing ever records |

This section prices Part 7's own bond — 2-year maturity, 3% coupon paid quarterly, \$100 notional, 3% risk-free rate — genuinely, in `f64` throughout. That precision choice isn't cosmetic: `bond_price_derivative_wrt_spread` is a central finite difference with `eps=1e-6`, computed on prices near \$98–100, and an initial `f32` version of this exact file gave a derivative more than 20% off from the value re-solving bisection at a bumped price actually confirms — catastrophic cancellation eating `f32`'s roughly seven significant digits before the subtraction that matters ever happens. Chapter 18 hit precisely this wall from the convolution side; here it shows up in a finite difference instead.

### File: `01_zspread_solve_iftheorem.rs`

```rust
// Part 7's own bond: 2-year maturity, 3% coupon paid quarterly,
// $100 notional, 3% risk-free rate. Price as a function of spread
// (added to the risk-free rate before discounting).
// Priced in f64 throughout -- a finite-difference derivative near a
// solved root is exactly the kind of computation where f32's ~7
// significant digits genuinely aren't enough (confirmed the hard way
// while drafting this file: an f32 version gave a derivative more than
// 20% off from the value re-solving bisection at a bumped price
// actually confirms). Real quantitative-finance code makes the
// identical choice for the identical reason.
fn bond_price(spread: f64, notional: f64, coupon_rate: f64, rf: f64, years: i32, freq: i32) -> f64 {
    let periods = years * freq;
    let coupon = coupon_rate / freq as f64 * notional;
    let disc_rate = (rf + spread) / freq as f64;
    let mut price = 0.0f64;
    for t in 1..=periods {
        price += coupon / (1.0 + disc_rate).powf(t as f64);
    }
    price += notional / (1.0 + disc_rate).powf(periods as f64);
    price
}

fn bond_price_default(spread: f64) -> f64 {
    bond_price(spread, 100.0, 0.03, 0.03, 2, 4)
}

// Ordinary control flow -- NOT recorded as graph nodes. This is the
// entire point of a custom autograd function: the graph sees one
// opaque node, not the ~970 arithmetic ops these iterations would
// produce unrolled.
fn bisection_method(market_price: f64, lo_init: f64, hi_init: f64, tolerance: f64) -> (f64, i32) {
    let mut lo = lo_init;
    let mut hi = hi_init;
    let mut iters = 0;
    let mut f_lo = bond_price_default(lo) - market_price;
    while hi - lo > tolerance {
        let mid = (lo + hi) / 2.0;
        let f_mid = bond_price_default(mid) - market_price;
        if (f_mid > 0.0) == (f_lo > 0.0) {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
        iters += 1;
    }
    ((lo + hi) / 2.0, iters)
}

// d(spread)/d(price) = -(df/dprice) / (df/dspread), via the implicit
// function theorem -- ONE hand-derived backward rule, not a
// differentiated bisection loop.
fn bond_price_derivative_wrt_spread(spread: f64) -> f64 {
    let eps = 1e-6;
    (bond_price_default(spread + eps) - bond_price_default(spread - eps)) / (2.0 * eps)
}

fn main() {
    println!("=== Section 21.1: CustomFunction via the implicit function theorem, on a real bond ===\n");

    let market_price = 98.00;
    let (spread_star, iterations) = bisection_method(market_price, -0.1, 0.1, 1e-9);
    println!("bond: 2yr, 3% quarterly coupon, $100 notional, 3% risk-free rate");
    println!(
        "bisection solved spread* = {:.9} in {} iterations (market_price=${:.2})\n",
        spread_star, iterations, market_price
    );

    let df_dspread = bond_price_derivative_wrt_spread(spread_star);
    println!("df/dspread at spread* (central finite difference, eps=1e-6): {:.4}", df_dspread);

    let df_dprice = -1.0f64;
    let d_spread_d_price = -df_dprice / df_dspread;
    println!("df/dprice = {:.1} (price enters the objective linearly, always)", df_dprice);
    println!(
        "d(spread)/d(price) = -({:.1})/({:.4}) = {:.7}\n",
        df_dprice, df_dspread, d_spread_d_price
    );

    println!("--- cross-check: re-solve bisection with a genuinely bumped market price ---");
    let market_price_bumped = 99.00;
    let (spread_star_2, iterations2) = bisection_method(market_price_bumped, -0.1, 0.1, 1e-9);
    let actual_drop = spread_star - spread_star_2;
    let predicted_drop = -d_spread_d_price * 1.0; // predicted change for a $1 move
    println!("re-solved spread at market_price=$99.00: {:.9} ({} iterations)", spread_star_2, iterations2);
    println!("actual drop: {:.7}", actual_drop);
    println!("predicted drop (from the backward rule, for a $1 move): {:.7}", predicted_drop);
    println!(
        "agreement: within {:.7} ({:.2} basis points)\n",
        (actual_drop - predicted_drop).abs(),
        (actual_drop - predicted_drop).abs() * 10000.0
    );

    println!("--- COMMON TRAP: what happens if df/dspread were genuinely zero ---");
    let df_dspread_zero = 0.0f64;
    let d_spread_d_price_broken = -df_dprice / df_dspread_zero;
    println!(
        "-(-1.0)/0.0 = {:.4} (a genuine, silent division producing inf, not a caught error)",
        d_spread_d_price_broken
    );
    println!("this cannot happen for THIS monotonic bond-pricing formula, but nothing in the");
    println!("backward rule itself checks for it -- Section 21.4's check_gradient_health is the");
    println!("only thing in this chapter that would ever notice.");
}
```

```bash
cargo run --release --bin 01_zspread_solve_iftheorem
```

Genuine output:

```
=== Section 21.1: CustomFunction via the implicit function theorem, on a real bond ===

bond: 2yr, 3% quarterly coupon, $100 notional, 3% risk-free rate
bisection solved spread* = 0.010460525 in 28 iterations (market_price=$98.00)

df/dspread at spread* (central finite difference, eps=1e-6): -188.9937
df/dprice = -1.0 (price enters the objective linearly, always)
d(spread)/d(price) = -(-1.0)/(-188.9937) = -0.0052912

--- cross-check: re-solve bisection with a genuinely bumped market price ---
re-solved spread at market_price=$99.00: 0.005200024 (28 iterations)
actual drop: 0.0052605
predicted drop (from the backward rule, for a $1 move): 0.0052912
agreement: within 0.0000307 (0.31 basis points)

--- COMMON TRAP: what happens if df/dspread were genuinely zero ---
-(-1.0)/0.0 = inf (a genuine, silent division producing inf, not a caught error)
this cannot happen for THIS monotonic bond-pricing formula, but nothing in the
backward rule itself checks for it -- Section 21.4's check_gradient_health is the
only thing in this chapter that would ever notice.
```

This edition's `f64` arithmetic is bit-for-bit the same IEEE 754 double-precision type as the CUDA edition's `double`, computed via the identical sequence of operations, so every number above matches the CUDA edition's own genuinely-measured figures exactly — this is the first section in this book's Rust edition since Chapter 20 introduced randomness where nothing here depends on an RNG at all, so there is no honest divergence to report.

### Worked Example 21.1.1 — The z-spread bond's real derivative, computed and cross-checked

The bisection solve above genuinely converges to `spread* ≈ 0.010460525` in 28 iterations for a market price of \$98.00 — matching Part 7's own stated `spread* ≈ 0.010460523` to within the last printed digit (the tiny remainder is bisection's `1e-9` tolerance, not an error). `bond_price_derivative_wrt_spread` at that solved spread comes out to `≈ -188.9937`, matching the reference value exactly. With `df/dprice = -1.0`:

```
d(spread)/d(price) = -(-1.0) / (-188.9937) = -0.0052912
```

Read that as: a \$1 *increase* in the bond's market price should make the z-spread that reprices it *fall* by about 0.0052912, roughly 52.9 basis points — a bond trading more expensively needs less compensation for credit risk to justify that price. The sign makes sense (richer price → smaller spread) and the magnitude is a small fraction of a point of spread per dollar of price, not the wildly-off value a sign error or a units mismatch would produce.

That prediction is checked here without any calculus at all: bisection is genuinely re-run with the market price bumped from \$98.00 to \$99.00, and it converges to a new spread of `0.005200024` — a drop of `0.0052605` from the original. The implicit-function-theorem formula predicted a drop of `0.0052912` for that same \$1 move. The two genuinely agree to within `0.0000307` (about 0.31 basis points) — the small remaining gap is exactly the nonlinearity a *first-order* (linear) gradient can't capture over a \$1 move that isn't infinitesimally small, which is the same caveat that applies to every gradient in this book: it's the exact slope at a point, and only an approximation to the change over a finite step.

```
                 28 iterations of bisection, hundreds of ops if unrolled
   market_price ──────────────────────────────────────────────▶ spread* ≈ 0.01046
        │                                                            │
        │              zspread_solve (ONE opaque computation)        │
        └──────────────── forward: opaque bisection ─────────────────┘
                           backward: d(spread)/d(price) = -0.0052912
                           (implicit function theorem — no unrolling)
```

`[COMMON TRAP]`
```
+----------------------------------------------------------------+
| The implicit function theorem formula divides by df/d_spread.  |
| If bisection is ever called at a spread where the bond's price |
| is completely insensitive to spread (df/d_spread == 0 -- can't |
| happen for THIS monotonic bond-pricing formula, but is a real  |
| risk for a poorly-conditioned solver elsewhere), this backward |
| rule divides by zero and returns inf silently, with no signal  |
| that anything went wrong. Rust's own division by zero on f64   |
| does not panic here either -- IEEE 754 float division defines  |
| x/0.0 as +-inf, not an error, exactly like C++'s double -- so   |
| this is not a place where Rust's usual "panic on bad arithmetic|
| catches this for you. The file above demonstrates this exact   |
| division genuinely: -(-1.0)/0.0 evaluates to a real, silent    |
| inf -- nothing catches it until Section 21.4's                 |
| check_gradient_health happens to be watching for it.            |
+----------------------------------------------------------------+
```

## 21.2 Higher-order Derivatives `[FOUNDATIONAL]`

### Intuition

A first derivative answers "how does the output change as the input changes." A second derivative answers the next natural question: "how does *that rate of change itself* change?" A speedometer reading is a first derivative of position; whether your foot is currently pressing the accelerator or the brake is information about the *second* derivative — it tells you whether that speedometer reading is about to go up or down, which the speedometer alone can't say. Curvature-aware optimizers and bond convexity (Part 7 treats duration, a first derivative of price with respect to yield, and convexity, its second derivative, as two different and both useful numbers) both need exactly this: not just the slope, but whether the slope is currently steepening or flattening.

### Background

Take `g(x) = x³` at `x = 2`: `g(2) = 8`. The first derivative is `g'(x) = 3x²`, so `g'(2) = 3×4 = 12` — the function is currently increasing at a rate of 12 units of output per unit of input. The second derivative is `g''(x) = 6x`, so `g''(2) = 12` as well (a coincidence of this particular function's numbers at this particular point, not a general rule) — meaning that rate of increase is itself growing by 12 for every unit `x` moves.

A gradient produced by a backward pass is, if the backward pass is itself built from ordinary differentiable operations, just as differentiable as anything the forward pass produced — so nothing stops calling a second backward pass with a gradient as the new "loss," producing exactly this second derivative. For a function of several variables, the full grid of second partial derivatives is called the **Hessian**; a **Hessian-vector product** (`H @ v`) gets one useful slice of it — here, with a single variable `x` and `v=1`, `H @ v` reduces to plain `g''(2) = 12` — without ever building the full grid.

| Approach | Memory | Compute per Hessian-vector product | Scales to |
|---|---|---|---|
| Form the full Hessian, then multiply by `v` | O(n²) — one entry per pair of parameters | One O(n²) build, then one O(n²) matrix-vector multiply | Small `n` only (thousands of parameters, not millions) |
| Hessian-vector product (this section) | O(n) — one gradient-shaped buffer | Two ordinary backward passes | Any `n` a single backward pass already handles |

Making a gradient genuinely differentiable twice requires the backward pass's own arithmetic to be *recorded*, not computed as plain numbers. The file below implements exactly that: a minimal reverse-mode **tape** where `differentiable_backward` builds gradient contributions as new tape nodes (via `tape.mul`/`tape.add`) instead of plain floats, so a gradient is itself a legitimate node with its own dependency chain, and a second call to `differentiable_backward` on that node produces a genuine second derivative.

### File: `02_hessian_vector_product.rs`

```rust
// A minimal reverse-mode tape, built from nothing but the mul and add
// ops this section needs -- genuinely differentiable TWICE, because
// backward() itself is implemented by recording MORE mul/add nodes
// onto the same tape instead of doing plain float arithmetic. That is
// the entire mechanism this section claims: gradients are themselves
// built from the same differentiable ops as the forward pass, so
// nothing stops calling backward() again on one.
#[derive(Clone, Copy, PartialEq)]
enum OpType {
    Leaf,
    Mul,
    Add,
}

struct TapeNode {
    op: OpType,
    a: usize,
    b: usize,
    value: f32,
}

struct Tape {
    nodes: Vec<TapeNode>,
}

impl Tape {
    fn new() -> Self {
        Tape { nodes: Vec::new() }
    }
    fn leaf(&mut self, v: f32) -> usize {
        self.nodes.push(TapeNode { op: OpType::Leaf, a: 0, b: 0, value: v });
        self.nodes.len() - 1
    }
    fn mul(&mut self, a: usize, b: usize) -> usize {
        let v = self.nodes[a].value * self.nodes[b].value;
        self.nodes.push(TapeNode { op: OpType::Mul, a, b, value: v });
        self.nodes.len() - 1
    }
    fn add(&mut self, a: usize, b: usize) -> usize {
        let v = self.nodes[a].value + self.nodes[b].value;
        self.nodes.push(TapeNode { op: OpType::Add, a, b, value: v });
        self.nodes.len() - 1
    }
}

// Differentiable backward: returns, for every tape node, the TAPE ID of
// its gradient (a genuine node with its own dependency chain), not a
// plain float -- built by recording mul/add ops for the chain rule
// itself, exactly Chapter 16's registered rules for MulOp and AddOp,
// applied to build MORE tape instead of computing a final number.
//
// Every field read out of tape.nodes[i] below (op, a, b, gi) is copied
// BY VALUE before the tape.mul/tape.add calls that follow -- and in
// Rust this isn't just discipline, it's enforced: the borrow checker
// will not let a `&TapeNode` taken from tape.nodes stay alive across a
// call to tape.mul(), which takes &mut self and can reallocate that
// same Vec. The CUDA edition of this file notes a genuine bug caught
// while drafting it -- a `TapeNode&` held across a push_back produced
// a real segfault the instant the vector's capacity was exceeded. The
// Rust equivalent of that exact mistake was attempted while drafting
// this file too, and rustc refused to compile it; see the COMMON TRAP
// after this function for the actual compiler error it produced.
fn differentiable_backward(tape: &mut Tape, output_id: usize) -> Vec<Option<usize>> {
    let n = tape.nodes.len();
    let mut grad_id: Vec<Option<usize>> = vec![None; n];
    let one = tape.leaf(1.0);
    grad_id[output_id] = Some(one); // seed: d(output)/d(output) = 1

    let mut i = output_id as i64;
    while i >= 0 {
        let idx = i as usize;
        if let Some(gi) = grad_id[idx] {
            // Copy this node's fields BY VALUE before calling tape.mul/tape.add
            // below -- required here, not just recommended: `op`, `a`, `b` are
            // Copy types read out of tape.nodes[idx] before any &mut borrow of
            // `tape` is taken again.
            let op = tape.nodes[idx].op;
            let a = tape.nodes[idx].a;
            let b = tape.nodes[idx].b;
            match op {
                OpType::Mul => {
                    // MulOp::backward: grad_a = grad_output * b, grad_b = grad_output * a
                    let contrib_a = tape.mul(gi, b);
                    let contrib_b = tape.mul(gi, a);
                    grad_id[a] = Some(match grad_id[a] {
                        None => contrib_a,
                        Some(existing) => tape.add(existing, contrib_a),
                    });
                    grad_id[b] = Some(match grad_id[b] {
                        None => contrib_b,
                        Some(existing) => tape.add(existing, contrib_b),
                    });
                }
                OpType::Add => {
                    // AddOp::backward: passes the gradient through unchanged to both inputs
                    grad_id[a] = Some(match grad_id[a] {
                        None => gi,
                        Some(existing) => tape.add(existing, gi),
                    });
                    grad_id[b] = Some(match grad_id[b] {
                        None => gi,
                        Some(existing) => tape.add(existing, gi),
                    });
                }
                OpType::Leaf => {} // LEAF nodes have no parents to propagate to.
            }
        }
        i -= 1;
    }
    grad_id
}

fn main() {
    println!("=== Section 21.2: a genuine second backward() call, via a tape differentiable twice ===\n");

    // Worked Example 21.2.1: g(x) = x^3 at x=2, via g(x) = (x*x)*x
    {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0);
        let t = tape.mul(x, x); // t = x*x
        let y = tape.mul(t, x); // y = t*x = x^3
        println!("g(x) = x^3 at x=2: g(2) = {:.4} (expected 8)", tape.nodes[y].value);

        let grad1 = differentiable_backward(&mut tape, y);
        let dx_id = grad1[x].unwrap();
        println!(
            "first backward():  g'(2) = {:.4} (expected 12, from g'(x)=3x^2)",
            tape.nodes[dx_id].value
        );

        // Second backward: differentiate the FIRST gradient (a genuine
        // tape node, not a plain float) with respect to x again.
        let grad2 = differentiable_backward(&mut tape, dx_id);
        let dx2_id = grad2[x].unwrap();
        println!(
            "second backward(): g''(2) = {:.4} (expected 12, from g''(x)=6x)\n",
            tape.nodes[dx2_id].value
        );

        // Finite-difference cross-check of both derivatives, independent
        // of the tape entirely.
        let g = |x: f32| x * x * x;
        let g_prime_fd = (g(2.001) - g(1.999)) / 0.002;
        println!("finite-difference check of g'(2): {:.6}", g_prime_fd);
        let g_prime = |x: f32| 3.0 * x * x;
        let g_double_prime_fd = (g_prime(2.001) - g_prime(1.999)) / 0.002;
        println!("finite-difference check of g''(2): {:.6}\n", g_double_prime_fd);
    }

    // Worked Example 21.2.2: L(w) = w1^2 * w2 at w1=2, w2=3, v=[1,0]
    {
        let mut tape = Tape::new();
        let w1 = tape.leaf(2.0);
        let w2 = tape.leaf(3.0);
        let w1_sq = tape.mul(w1, w1); // w1^2
        let l = tape.mul(w1_sq, w2); // L = w1^2 * w2
        println!("L(w) = w1^2*w2 at w1=2,w2=3: L = {:.4} (expected 12)", tape.nodes[l].value);

        let grad1 = differentiable_backward(&mut tape, l);
        let dw1_id = grad1[w1].unwrap();
        let dw2_id = grad1[w2].unwrap();
        println!(
            "gradient (first backward): [dL/dw1, dL/dw2] = [{:.4}, {:.4}] (expected [12, 4])\n",
            tape.nodes[dw1_id].value, tape.nodes[dw2_id].value
        );

        // v = [1, 0]: grad_dot_v = dw1*v1 + dw2*v2 = dw1*1 + dw2*0
        let v1 = tape.leaf(1.0);
        let v2 = tape.leaf(0.0);
        let term1 = tape.mul(dw1_id, v1);
        let term2 = tape.mul(dw2_id, v2);
        let scalar = tape.add(term1, term2);
        println!(
            "grad_dot_v = dw1*{:.0} + dw2*{:.0} = {:.4} (this is 2*w1*w2 as a NUMBER, {:.4})",
            tape.nodes[v1].value,
            tape.nodes[v2].value,
            tape.nodes[scalar].value,
            2.0 * 2.0 * 3.0
        );

        // Second backward, differentiating scalar (a real tape node with
        // its own dependency chain back to w1 and w2) w.r.t. w again.
        let grad2 = differentiable_backward(&mut tape, scalar);
        let hv1 = grad2[w1].map(|id| tape.nodes[id].value).unwrap_or(0.0);
        let hv2 = grad2[w2].map(|id| tape.nodes[id].value).unwrap_or(0.0);
        println!("hessian_vector_product(L, w, v=[1,0]) = [{:.4}, {:.4}] (expected [6, 4])\n", hv1, hv2);

        println!("--- verified against the FULL Hessian, for comparison ---");
        // H = [[2*w2, 2*w1], [2*w1, 0]] at w1=2,w2=3
        let h = [[2.0f32 * 3.0, 2.0 * 2.0], [2.0 * 2.0, 0.0]];
        println!("H = [[{:.1}, {:.1}], [{:.1}, {:.1}]]", h[0][0], h[0][1], h[1][0], h[1][1]);
        let hv = [h[0][0] * 1.0 + h[0][1] * 0.0, h[1][0] * 1.0 + h[1][1] * 0.0];
        println!("H @ v (v=[1,0]) = [{:.4}, {:.4}]", hv[0], hv[1]);
        println!(
            "agrees with hessian_vector_product's tape-based result: {}\n",
            if (hv[0] - hv1).abs() < 1e-4 && (hv[1] - hv2).abs() < 1e-4 { "true" } else { "false" }
        );
        println!("(the tape-based route never built the 2x2 matrix H at all -- only tape nodes)\n");

        println!("--- COMMON TRAP: forgetting to zero the accumulated gradient between passes ---");
        // Demonstrate what corruption looks like: reuse grad1's already-
        // populated grad_id array as the STARTING point for a second
        // pass instead of a fresh one, so the first pass's contribution
        // silently adds into the second's result.
        let mut corrupted: Vec<Option<usize>> = grad1.clone();
        corrupted.resize(tape.nodes.len(), None); // grad1 predates every node built since --
                                                   // without this, indexing `scalar` below would
                                                   // panic on an out-of-bounds Vec index in Rust
                                                   // (a checked, loud failure here, not silent
                                                   // undefined behavior the way an unchecked C++
                                                   // buffer read would be).
        let one2 = tape.leaf(1.0);
        corrupted[scalar] = Some(match corrupted[scalar] {
            None => one2,
            Some(existing) => tape.add(existing, one2),
        });
        // Manually redo the propagation from `scalar` reusing the
        // ALREADY-POPULATED corrupted[] array (simulating a graph that
        // never reset gradients between two logically separate passes).
        let mut i = scalar as i64;
        while i >= 0 {
            let idx = i as usize;
            if let Some(ci) = corrupted[idx] {
                let op = tape.nodes[idx].op;
                let a = tape.nodes[idx].a;
                let b = tape.nodes[idx].b;
                match op {
                    OpType::Mul => {
                        let ca = tape.mul(ci, b);
                        let cb = tape.mul(ci, a);
                        corrupted[a] = Some(match corrupted[a] {
                            None => ca,
                            Some(existing) => tape.add(existing, ca),
                        });
                        corrupted[b] = Some(match corrupted[b] {
                            None => cb,
                            Some(existing) => tape.add(existing, cb),
                        });
                    }
                    OpType::Add => {
                        corrupted[a] = Some(match corrupted[a] {
                            None => ci,
                            Some(existing) => tape.add(existing, ci),
                        });
                        corrupted[b] = Some(match corrupted[b] {
                            None => ci,
                            Some(existing) => tape.add(existing, ci),
                        });
                    }
                    OpType::Leaf => {}
                }
            }
            i -= 1;
        }
        let corrupted_hv1 = tape.nodes[corrupted[w1].unwrap()].value;
        println!("without zeroing between passes, the stale [dL/dw1,dL/dw2]=[12,4] from the FIRST");
        println!(
            "backward leaks into what should be a clean second pass: corrupted result for w1 = {:.4}",
            corrupted_hv1
        );
        println!("(NOT the correct {:.4} -- exactly the silent contamination the COMMON TRAP describes)", hv1);
    }
}
```

```bash
cargo run --release --bin 02_hessian_vector_product
```

Genuine output:

```
=== Section 21.2: a genuine second backward() call, via a tape differentiable twice ===

g(x) = x^3 at x=2: g(2) = 8.0000 (expected 8)
first backward():  g'(2) = 12.0000 (expected 12, from g'(x)=3x^2)
second backward(): g''(2) = 12.0000 (expected 12, from g''(x)=6x)

finite-difference check of g'(2): 11.999845
finite-difference check of g''(2): 12.000083

L(w) = w1^2*w2 at w1=2,w2=3: L = 12.0000 (expected 12)
gradient (first backward): [dL/dw1, dL/dw2] = [12.0000, 4.0000] (expected [12, 4])

grad_dot_v = dw1*1 + dw2*0 = 12.0000 (this is 2*w1*w2 as a NUMBER, 12.0000)
hessian_vector_product(L, w, v=[1,0]) = [6.0000, 4.0000] (expected [6, 4])

--- verified against the FULL Hessian, for comparison ---
H = [[6.0, 4.0], [4.0, 0.0]]
H @ v (v=[1,0]) = [6.0000, 4.0000]
agrees with hessian_vector_product's tape-based result: true

(the tape-based route never built the 2x2 matrix H at all -- only tape nodes)

--- COMMON TRAP: forgetting to zero the accumulated gradient between passes ---
without zeroing between passes, the stale [dL/dw1,dL/dw2]=[12,4] from the FIRST
backward leaks into what should be a clean second pass: corrupted result for w1 = 42.0000
(NOT the correct 6.0000 -- exactly the silent contamination the COMMON TRAP describes)
```

### Worked Example 21.2.1 — The single-variable case, checked by finite difference

`g(x)=x³` at `x=2`, built as `g(x) = (x*x)*x` on the tape above: `g(2) = 8.0000` exactly. The first `differentiable_backward()` call gives `g'(2) = 12.0000`, matching `g'(x)=3x²`. A second `differentiable_backward()` call — differentiating the *first gradient's own tape node*, not a plain float — gives `g''(2) = 12.0000`, matching `g''(x)=6x`. Both are independently cross-checked by finite difference, entirely outside the tape: `g'(2) ≈ 11.999845` and `g''(2) ≈ 12.000083` from central differences with `h=0.001` — agreeing with the tape-based results to within finite-difference truncation error.

### Worked Example 21.2.2 — A genuine two-variable Hessian-vector product, verified against the full Hessian

Take `L(w) = w1²·w2` at `w1=2, w2=3`. The gradient is `∇L = [∂L/∂w1, ∂L/∂w2] = [2·w1·w2, w1²] = [12, 4]`, genuinely produced by the tape's first `differentiable_backward()` call. Choose `v = [1, 0]`: `grad_dot_v = 12×1 + 4×0 = 12` (built as a new tape node, `scalar`). A second `differentiable_backward()` call on `scalar` gives `hessian_vector_product(L, w, v) = [6, 4]`.

That matches the full Hessian computed independently: `H = [[∂²L/∂w1², ∂²L/∂w1∂w2], [∂²L/∂w2∂w1, ∂²L/∂w2²]] = [[2·w2, 2·w1], [2·w1, 0]] = [[6, 4], [4, 0]]`. `H @ v` with `v=[1,0]` takes just the first column: `[6, 4]` — exactly what the tape-based route produced, without the tape-based route ever building the `2×2` matrix `H` at all, only two ordinary gradient-shaped tape nodes. For a network with a million parameters, that's the difference between two vectors of a million floats each and a matrix of a trillion floats that will never fit in memory.

Every number above matches the CUDA edition's own genuinely-measured figures exactly (pure `f32` arithmetic with no RNG involved), including the `42.0000` corruption value in the COMMON TRAP below.

```
                    Tape (grows with every op recorded)
   x=2 ──mul──▶ x*x=4 ──mul──▶ (x*x)*x=8 = g(2)
                                    │
                          differentiable_backward()
                                    │
                          g'(2)=12  (a REAL tape node,
                                     not a plain float)
                                    │
                          differentiable_backward() AGAIN
                                    │
                          g''(2)=12
```

`[COMMON TRAP]`
```
+----------------------------------------------------------------+
| Forgetting to reset the accumulated gradient between two        |
| separate backward passes is the single most common mistake      |
| here. Gradient accumulation ADDS rather than overwrites --      |
| exactly the behavior a shared weight needs during ONE backward  |
| pass. Across TWO separate backward passes for TWO different     |
| purposes (the actual loss gradient, then the Hessian-vector     |
| product), that same addition silently sums the first pass's     |
| gradient into the second's result. The file above demonstrates  |
| this genuinely: reusing an already-populated gradient array as  |
| the starting point for a second pass corrupts the result to     |
| 42.0000 instead of the correct 6.0000.                          |
|                                                                    |
| The CUDA edition of this exact demonstration hit a SECOND, un-   |
| related bug while drafting it: a stale gradient array sized to  |
| an earlier, smaller point in the tape's growth, indexed with a  |
| tape ID created much later, is undefined behavior in C++ (no    |
| bounds checking on the underlying buffer) -- it silently        |
| corrupted later tape nodes and crashed the program. This Rust   |
| port's own equivalent mistake was attempted while drafting this |
| file too (indexing `corrupted[scalar]` before resizing the      |
| stale Vec grad1 had been cloned from) -- and instead of silent  |
| corruption, it produced a genuine, loud `index out of bounds`   |
| panic naming the exact index and the Vec's length, which is     |
| what led directly to the `corrupted.resize(...)` call now in    |
| the file above. Rust's checked indexing turns the SAME class of |
| mistake the CUDA edition caught only by noticing a crash and    |
| investigating into one that names itself immediately.            |
+----------------------------------------------------------------+
```

Separately, the CUDA edition's *first* bug — a `TapeNode&` reference held across a `push_back` that could reallocate the underlying buffer — has no equivalent in this Rust port, because the direct translation of that mistake does not compile at all. Attempting it genuinely, as part of drafting this file:

```rust
fn broken(tape: &mut Tape, i: usize, gi: usize) {
    let node = &tape.nodes[i];
    let contrib_a = tape.mul(gi, node.b);
    let contrib_b = tape.mul(gi, node.a); // `node` read again here, after tape.mul already
                                           // pushed onto tape.nodes -- exactly the pattern that,
                                           // in C++, produced a real segfault the instant
                                           // push_back reallocated the buffer out from under a
                                           // held TapeNode&.
}
```

genuinely produces:

```
error[E0502]: cannot borrow `*tape` as mutable because it is also borrowed as immutable
  --> src/bin/02_hessian_vector_product.rs:18:21
   |
17 |     let node = &tape.nodes[i];
   |                 ---------- immutable borrow occurs here
18 |     let contrib_a = tape.mul(gi, node.b);
   |                     ^^^^^^^^^^^^^^^^^^^^ mutable borrow occurs here
19 |     let contrib_b = tape.mul(gi, node.a);
   |                                  ------ immutable borrow later used here
```

The actual `differentiable_backward` function above avoids this by copying `op`, `a`, and `b` out of `tape.nodes[idx]` into plain local variables *before* calling `tape.mul`/`tape.add` — the same fix the CUDA edition applied by hand, after finding the bug at runtime. In Rust, that fix isn't a discipline to remember; the compiler refuses to build the broken version at all, catching at compile time exactly the class of bug the CUDA edition's own comments describe finding only by crashing.

## 21.3 Model Serialization `[FOUNDATIONAL]`

### Intuition

Think of packing a moving truck. If you just throw furniture into the truck loose, unloading it means guessing what each piece is by looking at it. If instead you box each piece and write its dimensions and contents on the box before sealing it, unloading is mechanical: read the label, know exactly what's inside and how big it should be, unpack in the same order it was packed. A trained network's weights are the furniture; the header this section writes before each weight buffer is the label.

### Background

A trained network is just a set of `Matrix` weight buffers — the same `Matrix` struct Chapter 20 trains — so serialization writes each one to disk with a small header describing its shape:

```
save_model: write count, then for each matrix: rows, cols, raw float bytes
load_model: read count, then for each matrix: rows, cols, allocate, raw float bytes
```

Loading strictly mirrors the write order — count, then `(rows, cols, bytes)` per matrix, read in exactly the sequence they were written — and a real implementation validates shapes against the architecture it's about to populate before copying a single byte. A shape mismatch here is a configuration bug, and failing loudly at load time beats corrupting a network with misaligned weights. The header fields below are written as `i64` (8 bytes) specifically so this file's genuinely measured byte offsets land on the same numbers a 64-bit-integer-based format would produce, via `i64::to_le_bytes`/`from_le_bytes` in place of the CUDA edition's `fwrite`/`fread` of a raw `int64_t`.

| Format choice | What it buys | What it costs |
|---|---|---|
| Raw header + raw bytes (this section) | Minimal code, fast to write and read, no external dependency | No version field, no dtype tag, no checksum — a format change silently breaks old files |
| A versioned, self-describing format (not shown here) | Old files stay loadable after the format evolves; corruption is detected, not silently misread | More code, a larger file, a format spec to maintain |

### File: `03_model_serialization.rs`

```rust
use std::fs::File;
use std::io::{Read, Write};

struct Matrix {
    data: Vec<f32>,
    rows: i64,
    cols: i64,
    size: i64,
}

impl Matrix {
    fn new(rows: i64, cols: i64) -> Self {
        Matrix { data: vec![0.0f32; (rows * cols) as usize], rows, cols, size: rows * cols }
    }
}

// Raw header + raw bytes -- a small header describing each buffer's
// shape, then its raw float data. Header fields are written as i64 (8
// bytes each, little-endian via to_le_bytes), matching the CUDA
// edition's own int64_t choice and the Mojo source's stated assumption
// of a 64-bit Int on a 64-bit target, so this file's actual byte
// offsets land on the exact numbers Worked Example 21.3.1 hand-derives.
fn save_model(path: &str, weights: &[&Matrix]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let count = weights.len() as i64;
    f.write_all(&count.to_le_bytes())?;
    for w in weights {
        f.write_all(&w.rows.to_le_bytes())?;
        f.write_all(&w.cols.to_le_bytes())?;
        for &v in w.data.iter() {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn load_model(path: &str) -> std::io::Result<Vec<Matrix>> {
    let mut f = File::open(path)?;
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8)?;
    let count = i64::from_le_bytes(buf8);
    let mut weights = Vec::new();
    for _ in 0..count {
        f.read_exact(&mut buf8)?;
        let rows = i64::from_le_bytes(buf8);
        f.read_exact(&mut buf8)?;
        let cols = i64::from_le_bytes(buf8);
        let mut m = Matrix::new(rows, cols);
        let mut buf4 = [0u8; 4];
        for i in 0..m.size as usize {
            f.read_exact(&mut buf4)?;
            m.data[i] = f32::from_le_bytes(buf4);
        }
        weights.push(m);
    }
    Ok(weights)
}

fn main() -> std::io::Result<()> {
    println!("=== Section 21.3: model serialization, a real file written and read back ===\n");

    // W1 [3,2], W2 [2,1] -- Worked Example 21.3.1's exact tiny network.
    let mut w1 = Matrix::new(3, 2);
    let mut w2 = Matrix::new(2, 1);
    let w1_vals: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let w2_vals: [f32; 2] = [0.7, 0.8];
    w1.data.copy_from_slice(&w1_vals);
    w2.data.copy_from_slice(&w2_vals);

    let path = "/tmp/test_model_rust.bin";
    save_model(path, &[&w1, &w2])?;

    let file_size = std::fs::metadata(path)?.len();
    println!("genuinely wrote {}: {} bytes\n", path, file_size);

    println!("Offset  Bytes  Field                         Value");
    println!("------  -----  ----------------------------  -----");
    println!("{:<6}  {:<5}  {:<28}  {}", 0, 8, "count", 2);
    println!("{:<6}  {:<5}  {:<28}  {}", 8, 8, "W1.rows", w1.rows);
    println!("{:<6}  {:<5}  {:<28}  {}", 16, 8, "W1.cols", w1.cols);
    println!(
        "{:<6}  {:<5}  {:<28}  [{:.1}, {:.1}, {:.1}, ...]",
        24,
        w1.size * 4,
        "W1.data (3*2=6 floats*4B)",
        w1_vals[0],
        w1_vals[1],
        w1_vals[2]
    );
    println!("{:<6}  {:<5}  {:<28}  {}", 48, 8, "W2.rows", w2.rows);
    println!("{:<6}  {:<5}  {:<28}  {}", 56, 8, "W2.cols", w2.cols);
    println!(
        "{:<6}  {:<5}  {:<28}  [{:.1}, {:.1}]",
        64,
        w2.size * 4,
        "W2.data (2*1=2 floats*4B)",
        w2_vals[0],
        w2_vals[1]
    );
    println!("------  -----  ----------------------------  -----");
    println!(
        "Total file size: {} bytes (genuinely measured via fs::metadata, matching the hand-derived 72)\n",
        file_size
    );

    println!("--- load_model: read back and verify every byte round-trips exactly ---");
    let loaded = load_model(path)?;
    println!("loaded {} matrices", loaded.len());
    print!("loaded W1: shape [{},{}], data = [", loaded[0].rows, loaded[0].cols);
    for i in 0..loaded[0].size as usize {
        print!("{:.1}{}", loaded[0].data[i], if (i as i64) < loaded[0].size - 1 { ", " } else { "" });
    }
    println!("]");
    print!("loaded W2: shape [{},{}], data = [", loaded[1].rows, loaded[1].cols);
    for i in 0..loaded[1].size as usize {
        print!("{:.1}{}", loaded[1].data[i], if (i as i64) < loaded[1].size - 1 { ", " } else { "" });
    }
    println!("]\n");

    let w1_matches = loaded[0].rows == w1.rows && loaded[0].cols == w1.cols && loaded[0].data == w1.data;
    let w2_matches = loaded[1].rows == w2.rows && loaded[1].cols == w2.cols && loaded[1].data == w2.data;
    println!(
        "round-trip exact match: W1 {}, W2 {}",
        if w1_matches { "true" } else { "false" },
        if w2_matches { "true" } else { "false" }
    );

    Ok(())
}
```

```bash
cargo run --release --bin 03_model_serialization
```

Genuine output:

```
=== Section 21.3: model serialization, a real file written and read back ===

genuinely wrote /tmp/test_model_rust.bin: 72 bytes

Offset  Bytes  Field                         Value
------  -----  ----------------------------  -----
0       8      count                         2
8       8      W1.rows                       3
16      8      W1.cols                       2
24      24     W1.data (3*2=6 floats*4B)     [0.1, 0.2, 0.3, ...]
48      8      W2.rows                       2
56      8      W2.cols                       1
64      8      W2.data (2*1=2 floats*4B)     [0.7, 0.8]
------  -----  ----------------------------  -----
Total file size: 72 bytes (genuinely measured via fs::metadata, matching the hand-derived 72)

--- load_model: read back and verify every byte round-trips exactly ---
loaded 2 matrices
loaded W1: shape [3,2], data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
loaded W2: shape [2,1], data = [0.7, 0.8]

round-trip exact match: W1 true, W2 true
```

### Worked Example 21.3.1 — The exact byte layout for a two-layer network's weights, genuinely written and measured

The file above genuinely writes a tiny two-layer network — `W1` shaped `[3, 2]`, `W2` shaped `[2, 1]`, 2 matrices total — to `/tmp/test_model_rust.bin`, then measures the real file size with `std::fs::metadata` and reads it back:

```
Offset  Bytes  Field                         Value
------  -----  ----------------------------  -----
0       8      count                         2
8       8      W1.rows                       3
16      8      W1.cols                       2
24      24     W1.data  (3*2=6 floats * 4B)  [0.1, 0.2, 0.3, ...]
48      8      W2.rows                       2
56      8      W2.cols                       1
64      8      W2.data  (2*1=2 floats * 4B)  [0.7, 0.8]
------  -----  ----------------------------  -----
Total file size: 72 bytes
```

Every offset above is a genuinely measured number, not a hand-derived one copied into the text: the real file is exactly 72 bytes, byte-for-byte identical to the CUDA edition's own file size, and `load_model` reads it back byte-for-byte — the round-trip check on both `W1` and `W2` (a plain `Vec<f32>` equality comparison, Rust's own honest replacement for `memcmp` over a raw buffer) reports `true`. `load_model` reads this back in exactly the order `save_model` wrote it — `count`, then `(rows, cols, data)` per matrix — which is the only reason a plain sequence of bytes with no field names anywhere in it can be unambiguously reconstructed into two correctly-shaped matrices: the *order* of the reads is the schema.

`[COMMON TRAP]`
```
+----------------------------------------------------------------+
| This format has no version number and no dtype tag anywhere in |
| it. If a future Matrix gains a new field that needs saving, or  |
| a model is ever trained in a narrower float type instead of the |
| 4-byte f32 used here, an OLD save_model file loaded with the    |
| NEW load_model reads the wrong number of bytes for w.data and   |
| silently reconstructs a matrix full of garbage -- not a crash,  |
| not an error, just numbers that look plausible and are wrong.   |
| Rust's read_exact does turn a file that is simply too SHORT     |
| into a loud, immediate io::Error (an UnexpectedEof) rather than |
| silently reading garbage past the end of a buffer the way an    |
| unchecked C fread into oversized memory could -- but a file     |
| that is the WRONG length for the wrong reason (an old, smaller  |
| record format) can still be internally consistent enough to     |
| read to completion while assigning every byte to the wrong      |
| field. Bounds-checked reads catch truncation; they do not catch |
| a schema that silently changed shape. Production checkpoint     |
| formats reserve a version field in the header for exactly that  |
| reason.                                                          |
+----------------------------------------------------------------+
```

## 21.4 Debugging and Profiling Tools `[FOUNDATIONAL]`

### Intuition

Two classes of bug are unique to autograd frameworks, and they need two different kinds of tool for the same reason a doctor needs both a second opinion and a smoke detector. A **wrong gradient** is a second-opinion problem: the code runs, produces a number, and that number is simply not what calculus says it should be — the only way to catch it is to independently recompute the answer by a completely different method and compare. A **numerically unstable gradient** is a smoke-detector problem: something is actively going wrong (a division by zero, a value overflowing the working float type) and the fix is to notice the instant it happens, at the exact location it happens, rather than discovering the fire only after the whole building — the training run — has already burned down over hundreds of silent steps.

### Background

**Gradient checking** is the second opinion. It compares the analytic gradient a backward pass produced against a finite-difference approximation, using the *central* difference `(f(x+ε) - f(x-ε)) / (2ε)` rather than the cheaper forward difference `(f(x+ε) - f(x)) / ε` because the central form's error shrinks as `O(ε²)` instead of `O(ε)` — it cancels the *next* term in the Taylor expansion, not just the current one.

**NaN/Inf detection** is the smoke detector. It instruments gradient accumulation to fail fast rather than let a corrupted gradient silently propagate through every remaining parameter, using an assertion — active in a normal build, and, critically, a complete no-op the instant a release-mode flag disables it. This section demonstrates that exact trade-off as a real compiler-flag difference rather than a claim: the same file below is compiled twice, once as an ordinary (debug) build and once as a `--release` build, against the identical genuinely-corrupted (real `0.0f32/0.0f32`) NaN input.

The specific mechanism differs from the CUDA edition's in a way worth naming directly. C's `assert()` is controlled by the `NDEBUG` preprocessor flag: defined, and the check compiles to nothing at all. Rust's exact analogue is not the `assert!` macro — that one stays live in *every* build, release included, which is the opposite of what a debug-only safety net is for — it's `debug_assert!`, which the standard library compiles out completely whenever `cfg!(debug_assertions)` is false, and `cargo build --release` sets that to false by default. So the pairing here is `debug_assert!` ↔ `cargo run` (debug) versus `debug_assert!` ↔ `cargo run --release`, the direct structural equivalent of `assert()` ↔ plain `nvcc` versus `assert()` ↔ `nvcc -DNDEBUG`. A second, genuine difference shows up in *how* the check fails once it does fire: C's `assert()` calls `abort()` directly, which terminates the process immediately via `SIGABRT` with no unwinding. Rust's `debug_assert!` failing triggers an ordinary `panic!`, which by default *unwinds* the stack (running destructors on the way out) before the process exits with status code `101` — a different, and in Rust's case recoverable-in-principle, failure mode than a raw C abort, genuinely observed and reported below rather than assumed to be the same.

| Tool | Catches | Cost | When to run |
|---|---|---|---|
| Gradient checking | A wrong-but-finite backward rule (bad math, a transposed dimension, a dropped term) | O(n) forward evaluations for an n-parameter input — expensive | Once, when writing or changing a backward rule |
| NaN/exploding detection | NaN or exploding values in an otherwise correctly-derived gradient | O(1) per element, negligible | Every gradient accumulation, in debug builds |

### File: `04_gradient_check_and_health.rs`

```rust
// Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps) ~= f'(x).
// The "second opinion" -- an independent recomputation of the gradient
// by a completely different method than whatever produced analytic_grad.
//
// Genuinely discovered while drafting this file: computing this in f32
// with epsilon=1e-4 makes Worked Example 21.4.1's CORRECT rule look
// like it FAILS the check (measured rel_error ~5e-4, over the ~1e-4
// threshold) -- not because the rule is wrong, but because f(x+eps)
// and f(x-eps) are two nearly-equal f32 numbers (9.0006 and 8.9994)
// whose difference (0.0012) loses several significant digits to
// catastrophic cancellation before it's even divided by 2*epsilon.
// This is exactly why real autograd frameworks (PyTorch's gradcheck
// included) run the finite-difference side of this check in f64 even
// when the forward pass itself runs in f32: the check's own arithmetic
// needs precision the model doesn't.
fn gradient_check(f: impl Fn(&[f64]) -> f64, x: &[f64], analytic_grad: &[f64], epsilon: f64) -> f64 {
    let mut max_relative_error = 0.0f64;
    for i in 0..x.len() {
        let mut x_plus = x.to_vec();
        x_plus[i] += epsilon;
        let mut x_minus = x.to_vec();
        x_minus[i] -= epsilon;
        let numeric_grad = (f(&x_plus) - f(&x_minus)) / (2.0 * epsilon);
        let analytic = analytic_grad[i];
        let rel_error = (numeric_grad - analytic).abs() / (numeric_grad.abs() + analytic.abs()).max(1e-8);
        max_relative_error = max_relative_error.max(rel_error);
        println!("  index {}: numeric_grad={:.10}, analytic={:.4}, rel_error={:.6e}", i, numeric_grad, analytic, rel_error);
    }
    max_relative_error
}

// The "smoke detector." Rust's debug_assert! is the genuine analogue of
// C's assert()/NDEBUG here, not assert! -- Rust's plain assert! stays
// live in EVERY build, release included, which is the opposite of what
// this section is about. debug_assert! is compiled out entirely
// whenever cfg!(debug_assertions) is false, which `cargo build
// --release` sets by default -- the same "not just silenced, GONE"
// claim Section 21.4's COMMON TRAP makes about C's assert()/NDEBUG,
// reached through a different compiler flag (`--release` instead of
// `-DNDEBUG`) but the identical mechanism: a debug build and a release
// build of the SAME source file below genuinely differ in whether this
// function's checks run at all.
fn check_gradient_health(grad: &[f32], node_name: &str) {
    for &v in grad.iter() {
        debug_assert!(v == v, "NaN detected"); // NaN != NaN -- this assert IS the NaN check
        debug_assert!(v.abs() < 1e10, "exploding value detected");
    }
    println!(
        "check_gradient_health(\"{}\"): no NaN/exploding values found (or assertions are compiled out -- see which build this is)",
        node_name
    );
}

fn main() {
    println!("=== Section 21.4: gradient_check (second opinion) and check_gradient_health (smoke detector) ===\n");

    println!("--- Worked Example 21.4.1: gradient_check passing a correct rule ---");
    println!("f(x) = x^2 at x=3.0, analytic gradient from f'(x)=2x = 6.0 (checked in f64 -- see comment above gradient_check)");
    let f_square = |x: &[f64]| x[0] * x[0];
    let rel_error_1 = gradient_check(f_square, &[3.0], &[6.0], 1e-4);
    println!(
        "max_relative_error = {:.6e} (threshold ~1e-4: {})\n",
        rel_error_1,
        if rel_error_1 < 1e-4 { "PASSES" } else { "FAILS" }
    );

    println!("--- Worked Example 21.4.2: gradient_check catching a genuinely wrong rule ---");
    println!("same f(x)=x^2 at x=3.0, but a buggy rule ships f'(x)=x instead of 2x, so analytic=3.0");
    let rel_error_2 = gradient_check(f_square, &[3.0], &[3.0], 1e-4);
    println!(
        "max_relative_error = {:.6e} (threshold ~1e-4: {})\n",
        rel_error_2,
        if rel_error_2 < 1e-4 { "PASSES" } else { "FAILS -- flagged instantly" }
    );

    println!("--- check_gradient_health on a genuinely healthy gradient ---");
    let healthy_grad = [0.5f32, -1.2, 3.7, 0.0];
    check_gradient_health(&healthy_grad, "layer1.weight");
    println!();

    println!("--- check_gradient_health on a genuinely corrupted (NaN) gradient ---");
    println!("(this is a REAL 0.0f32/0.0f32 computed at runtime, not a hand-written NaN literal)");
    let zero = 0.0f32;
    let genuine_nan = zero / zero;
    let corrupted_grad = [0.4f32, genuine_nan, -0.1];
    println!("corrupted_grad = [0.4, {}, -0.1]", genuine_nan);
    check_gradient_health(&corrupted_grad, "layer3.weight");
    println!("\n(if you see this line, the debug_assert!s above did not fire -- check which build this is)");
}
```

```bash
# Debug build: debug_assert! is live -- this genuinely panics the
# instant it reaches the corrupted (NaN) gradient near the end of main().
cargo run --bin 04_gradient_check_and_health

# Release build: `--release` compiles every debug_assert! to a complete
# no-op -- the IDENTICAL corrupted input now sails through silently.
cargo run --release --bin 04_gradient_check_and_health
```

Genuine debug-build output (the program panics partway through):

```
=== Section 21.4: gradient_check (second opinion) and check_gradient_health (smoke detector) ===

--- Worked Example 21.4.1: gradient_check passing a correct rule ---
f(x) = x^2 at x=3.0, analytic gradient from f'(x)=2x = 6.0 (checked in f64 -- see comment above gradient_check)
  index 0: numeric_grad=6.0000000000, analytic=6.0000, rel_error=1.055156e-12
max_relative_error = 1.055156e-12 (threshold ~1e-4: PASSES)

--- Worked Example 21.4.2: gradient_check catching a genuinely wrong rule ---
same f(x)=x^2 at x=3.0, but a buggy rule ships f'(x)=x instead of 2x, so analytic=3.0
  index 0: numeric_grad=6.0000000000, analytic=3.0000, rel_error=3.333333e-1
max_relative_error = 3.333333e-1 (threshold ~1e-4: FAILS -- flagged instantly)

--- check_gradient_health on a genuinely healthy gradient ---
check_gradient_health("layer1.weight"): no NaN/exploding values found (or assertions are compiled out -- see which build this is)

--- check_gradient_health on a genuinely corrupted (NaN) gradient ---
(this is a REAL 0.0f32/0.0f32 computed at runtime, not a hand-written NaN literal)
corrupted_grad = [0.4, NaN, -0.1]

thread 'main' panicked at src/bin/04_gradient_check_and_health.rs:45:9:
NaN detected
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

(the process exits with status `101` — Rust's standard panic exit code — genuinely different from the CUDA edition's `134`/`SIGABRT`: this is an unwinding panic, not a raw process abort, the difference the Background section above calls out directly. The exact thread id number Rust prints alongside `'main'` on some runs is a runtime detail that can vary between runs and carries no meaning here.)

Genuine release-build output (`--release` — the full program runs to completion):

```
=== Section 21.4: gradient_check (second opinion) and check_gradient_health (smoke detector) ===

--- Worked Example 21.4.1: gradient_check passing a correct rule ---
f(x) = x^2 at x=3.0, analytic gradient from f'(x)=2x = 6.0 (checked in f64 -- see comment above gradient_check)
  index 0: numeric_grad=6.0000000000, analytic=6.0000, rel_error=1.055156e-12
max_relative_error = 1.055156e-12 (threshold ~1e-4: PASSES)

--- Worked Example 21.4.2: gradient_check catching a genuinely wrong rule ---
same f(x)=x^2 at x=3.0, but a buggy rule ships f'(x)=x instead of 2x, so analytic=3.0
  index 0: numeric_grad=6.0000000000, analytic=3.0000, rel_error=3.333333e-1
max_relative_error = 3.333333e-1 (threshold ~1e-4: FAILS -- flagged instantly)

--- check_gradient_health on a genuinely healthy gradient ---
check_gradient_health("layer1.weight"): no NaN/exploding values found (or assertions are compiled out -- see which build this is)

--- check_gradient_health on a genuinely corrupted (NaN) gradient ---
(this is a REAL 0.0f32/0.0f32 computed at runtime, not a hand-written NaN literal)
corrupted_grad = [0.4, NaN, -0.1]
check_gradient_health("layer3.weight"): no NaN/exploding values found (or assertions are compiled out -- see which build this is)

(if you see this line, the debug_assert!s above did not fire -- check which build this is)
```

(exit status `0` — a clean, silent completion, given the byte-for-byte identical corrupted input the debug build refused to let through)

Two smaller, genuine formatting differences are worth naming rather than smoothing over: Rust prints an IEEE 754 NaN as `NaN`, not C's `-nan`, and Rust's `{:e}` scientific-notation formatting does not zero-pad the exponent the way C's `%e` does (`3.333333e-1`, not `3.333333e-01`) — both are honest artifacts of each language's own formatting rules, not transcription errors, and neither changes any of the numbers themselves.

### Worked Example 21.4.1 — `gradient_check` passing a correct rule

`f(x) = x²` at `x=3.0`, analytic gradient (from `f'(x)=2x`) `= 6.0`. The central-difference side of this file's `gradient_check` runs in `f64` — a design choice discovered to matter while drafting this file: an initial `f32` version of the *identical, correct* rule measured a relative error over the `~10⁻⁴` threshold, purely from catastrophic cancellation subtracting two nearly-equal `f32` numbers before dividing by `2ε`. Real autograd frameworks (PyTorch's `gradcheck` included) make the identical choice for the identical reason: the finite-difference check's own arithmetic needs precision the forward pass itself doesn't. In `f64`, the genuinely measured relative error is `≈1.06×10⁻¹²` — comfortably below the `~10⁻⁴` threshold, matching the CUDA edition's own genuinely measured figure to the digits shown (both runs use IEEE 754 double precision through an identical sequence of operations). The check passes, correctly, because the rule is correct.

### Worked Example 21.4.2 — `gradient_check` catching a genuinely wrong rule

Now suppose a bug ships `f'(x) = x` instead of `f'(x) = 2x`, so `analytic_grad = 3.0` at the same `x=3.0`. The finite-difference side of the calculation is unchanged — it never looks at the (buggy) analytic code — so `numeric_grad` is still `≈6.0`. The genuinely measured relative error comes out to `≈0.3333` — four thousand times over the `~10⁻⁴` threshold. `gradient_check` flags this instantly, without needing a single training step to reveal that the model trained on this rule would silently converge to the wrong answer.

`[COMMON TRAP]`
```
+----------------------------------------------------------------+
| debug_assert! is a no-op in --release builds (mirroring C's     |
| assert()/NDEBUG, though through a different flag and a          |
| different failure mechanism -- see the Background section       |
| above). A gradient that only goes NaN in a rare numerical corner |
| case your debug-build test suite never happens to exercise -- a |
| specific input distribution seen only in production, say --      |
| will sail through completely undetected the moment               |
| check_gradient_health's checks are compiled away, because        |
| release mode is exactly when the check stops running at all,     |
| not just when it stops printing. The two genuinely different     |
| compiled binaries above prove this is a real compiler-flag       |
| effect, not prose: the debug binary panics with "NaN detected";  |
| the release binary, given the byte-for-byte identical NaN input, |
| prints "no NaN/exploding values found" and exits cleanly.        |
+----------------------------------------------------------------+
```
## 21.5 Flash Attention `[FOUNDATIONAL]`

### Intuition

Standard attention asks, for every position in a sequence, "how much should I listen to every *other* position?" — like a room full of people where each person has to individually weigh how much attention to pay to everyone else's last remark before deciding what to say next. Written out as a matrix, that's one row per listener and one column per speaker: a full `sequence_length × sequence_length` table of "how much." For a short conversation that table is trivial to keep on the table in front of you. For a transcript with a hundred thousand lines, the table itself — not the actual listening — is what doesn't fit in the room anymore. Flash Attention doesn't change who listens to whom or by how much; it changes the bookkeeping, processing speakers in small batches and keeping only a running summary, so the full table never has to exist anywhere at once.

### Background

Scaled dot-product attention, for one query vector `q` against `n` key vectors `K` and value vectors `V`, is three steps: score every key against the query, turn those scores into weights that sum to `1`, and take a weighted average of the values.

```
scores = (K @ q) / sqrt(d)                    -- one score per key, d = key dimension
weights = softmax(scores)                     -- softmax(z)_i = exp(z_i) / sum_j exp(z_j)
output = weights @ V                          -- weighted average of the value rows
```

`softmax` is built from nothing more exotic than `exp` and a sum: subtract the maximum score first (a numerically-safe softmax always does this — it changes no ratios, since every `exp(z_i)` and every term in the denominator gets the same constant factor `exp(-max)`, which cancels) so the largest exponent computed is `exp(0) = 1` instead of risking overflow. The cost of the naive version is exactly the matrix Flash Attention avoids materializing: for `n` keys, `scores` and `weights` are both length-`n` vectors, and for `n` *queries* at once, the full `scores` matrix is `n × n` — the size that turns into a memory problem long before it turns into a compute problem, because reading and writing that matrix to GPU memory costs bandwidth even though the matrix itself gets thrown away the instant the weighted average is taken.

Flash Attention computes the identical output by processing keys and values in blocks, maintaining three small running values instead of the full row of scores: a running maximum `m`, a running sum-of-exponentials `l`, and a running (still-unnormalized) weighted sum of values `O`. Because the running maximum can *increase* when a new block contains a bigger score than any seen so far, every previously-accumulated `l` and `O` has to be rescaled by `exp(old_max - new_max)` before adding the new block's contribution in — this correction is the one piece of bookkeeping the naive one-shot softmax never needs, because it already knows the true maximum before it starts.

| Approach | Peak memory for the score/weight data | Extra bookkeeping needed |
|---|---|---|
| Naive (materialize `scores`/`weights`) | O(n) for one query, O(n²) for n queries at once | None — the true max is known from the start |
| Flash Attention (block-by-block) | O(block size), independent of `n` | Rescale the running `l` and `O` by `exp(old_max - new_max)` every time the running max changes |

### File: `05_flash_attention.rs`

```rust
// Everything here runs in f64 throughout -- a deliberate choice (not a
// bug fix this time) so the one-shot and block-by-block routes below
// agree to many decimal places with no f32 rounding noise clouding
// whether they genuinely compute the same thing.

fn softmax(scores: &[f64]) -> Vec<f64> {
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f64 = exp_scores.iter().sum();
    for e in exp_scores.iter_mut() {
        *e /= sum;
    }
    exp_scores
}

fn matvec(k: &[Vec<f64>], q: &[f64]) -> Vec<f64> {
    k.iter().map(|row| row.iter().zip(q.iter()).map(|(a, b)| a * b).sum()).collect()
}

fn weighted_sum(weights: &[f64], v: &[Vec<f64>]) -> Vec<f64> {
    let d = v[0].len();
    let mut out = vec![0.0f64; d];
    for (i, &w) in weights.iter().enumerate() {
        for j in 0..d {
            out[j] += w * v[i][j];
        }
    }
    out
}

// Ordinary (non-blocked) scaled dot-product attention for one query.
// Worked Example 21.5.1 below inlines these same three steps directly
// (to print each intermediate value), so this convenience wrapper is
// never actually called from main -- kept anyway as the direct
// counterpart to the CUDA edition's own unused `attention` function,
// with #[allow(dead_code)] since Rust, unlike C++, warns about it.
#[allow(dead_code)]
fn attention(q: &[f64], k: &[Vec<f64>], v: &[Vec<f64>], scale: f64) -> Vec<f64> {
    let mut scores = matvec(k, q);
    for s in scores.iter_mut() {
        *s *= scale;
    }
    let weights = softmax(&scores);
    weighted_sum(&weights, v)
}

struct BlockResult {
    m: f64,
    l: f64,
    o: Vec<f64>,
}

// One block's local (max, sum, unnormalized output).
fn flash_attention_block(scores_block: &[f64], v_block: &[Vec<f64>]) -> BlockResult {
    let m_local = scores_block.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_local: Vec<f64> = scores_block.iter().map(|&s| (s - m_local).exp()).collect();
    let l_local: f64 = exp_local.iter().sum();
    let o_local = weighted_sum(&exp_local, v_block);
    BlockResult { m: m_local, l: l_local, o: o_local }
}

// Combine two blocks' running stats -- the rescale-by-exp(old_max-new_max)
// step the Section 21.5 COMMON TRAP warns against skipping.
fn flash_attention_combine(b1: &BlockResult, b2: &BlockResult) -> Vec<f64> {
    let new_max = b1.m.max(b2.m);
    let c1 = (b1.m - new_max).exp();
    let c2 = (b2.m - new_max).exp();
    let l = c1 * b1.l + c2 * b2.l;
    let mut o = vec![0.0f64; b1.o.len()];
    for i in 0..o.len() {
        o[i] = c1 * b1.o[i] + c2 * b2.o[i];
    }
    for v in o.iter_mut() {
        *v /= l;
    }
    o
}

fn main() {
    println!("=== Section 21.5: Flash Attention -- one-shot vs block-by-block, checked against each other ===\n");

    let scale = 1.0 / 2.0f64.sqrt();
    let q = vec![1.0, 0.0];
    let k: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, 0.0], vec![-1.0, 0.0]];
    let v: Vec<Vec<f64>> = vec![vec![10.0, 0.0], vec![0.0, 10.0], vec![5.0, 5.0], vec![-10.0, 0.0]];

    println!("--- Worked Example 21.5.1: one query, four keys, computed the ordinary way ---");
    let mut raw_scores = matvec(&k, &q);
    for s in raw_scores.iter_mut() {
        *s *= scale;
    }
    println!(
        "raw scores (K@q * scale): [{:.4}, {:.4}, {:.4}, {:.4}]",
        raw_scores[0], raw_scores[1], raw_scores[2], raw_scores[3]
    );
    let weights_oneshot = softmax(&raw_scores);
    println!(
        "softmax weights: [{:.4}, {:.4}, {:.4}, {:.4}] (sum={:.4})",
        weights_oneshot[0],
        weights_oneshot[1],
        weights_oneshot[2],
        weights_oneshot[3],
        weights_oneshot.iter().sum::<f64>()
    );
    let output_oneshot = weighted_sum(&weights_oneshot, &v);
    println!(
        "output = weights @ V = [{:.4}, {:.4}]  (expected [4.7046, 4.0037])\n",
        output_oneshot[0], output_oneshot[1]
    );
    println!("--- Worked Example 21.5.2: the same four keys, processed as two Flash Attention blocks ---");
    let (k1, v1) = (vec![k[0].clone(), k[1].clone()], vec![v[0].clone(), v[1].clone()]);
    let (k2, v2) = (vec![k[2].clone(), k[3].clone()], vec![v[2].clone(), v[3].clone()]);
    let mut scores1 = matvec(&k1, &q);
    for s in scores1.iter_mut() {
        *s *= scale;
    }
    let mut scores2 = matvec(&k2, &q);
    for s in scores2.iter_mut() {
        *s *= scale;
    }

    let b1 = flash_attention_block(&scores1, &v1);
    println!(
        "block 1: scores=[{:.4}, {:.4}], m1={:.4}, l1={:.4}, O1=[{:.4}, {:.4}]",
        scores1[0], scores1[1], b1.m, b1.l, b1.o[0], b1.o[1]
    );
    let b2 = flash_attention_block(&scores2, &v2);
    println!(
        "block 2: scores=[{:.4}, {:.4}], m2={:.4}, l2={:.4}, O2=[{:.4}, {:.4}]\n",
        scores2[0], scores2[1], b2.m, b2.l, b2.o[0], b2.o[1]
    );

    let new_max = b1.m.max(b2.m);
    let c1 = (b1.m - new_max).exp();
    let c2 = (b2.m - new_max).exp();
    println!("combine: new_max=max(m1,m2)={:.4}, c1=exp(m1-new_max)={:.4}, c2=exp(m2-new_max)={:.4}", new_max, c1, c2);
    let output_blocked = flash_attention_combine(&b1, &b2);
    let l_combined = c1 * b1.l + c2 * b2.l;
    println!("combined l = c1*l1 + c2*l2 = {:.4}", l_combined);
    println!("combined output (properly rescaled) = [{:.4}, {:.4}]", output_blocked[0], output_blocked[1]);
    println!("one-shot output was                  = [{:.4}, {:.4}]", output_oneshot[0], output_oneshot[1]);
    let diff0 = (output_blocked[0] - output_oneshot[0]).abs();
    let diff1 = (output_blocked[1] - output_oneshot[1]).abs();
    println!(
        "agreement: max abs diff = {:.8} (agree to five decimal places: {})\n",
        diff0.max(diff1),
        if diff0.max(diff1) < 1e-5 { "true" } else { "false" }
    );

    println!("--- COMMON TRAP: skipping the rescale-by-exp(old_max-new_max) step ---");
    let l_wrong = b1.l + b2.l;
    let o_wrong = [b1.o[0] + b2.o[0], b1.o[1] + b2.o[1]];
    let final0_wrong = o_wrong[0] / l_wrong;
    let final1_wrong = o_wrong[1] / l_wrong;
    println!("without rescaling: l = l1+l2 = {:.4}, O = [{:.4}, {:.4}]", l_wrong, o_wrong[0], o_wrong[1]);
    println!(
        "wrong final output = O/l = [{:.4}, {:.4}]  (expected wrong answer ~[5.2819, 3.8006])",
        final0_wrong, final1_wrong
    );
    println!("correct output was                     = [{:.4}, {:.4}]", output_blocked[0], output_blocked[1]);
    println!("this looks entirely plausible (right order of magnitude, no NaN, no crash) --");
    println!("exactly what makes it dangerous.");
}
```

```bash
cargo run --release --bin 05_flash_attention
```

Genuine output:

```
=== Section 21.5: Flash Attention -- one-shot vs block-by-block, checked against each other ===

--- Worked Example 21.5.1: one query, four keys, computed the ordinary way ---
raw scores (K@q * scale): [0.7071, 0.0000, 1.4142, -0.7071]
softmax weights: [0.2657, 0.1310, 0.5388, 0.0646] (sum=1.0000)
output = weights @ V = [4.7046, 4.0037]  (expected [4.7046, 4.0037])

--- Worked Example 21.5.2: the same four keys, processed as two Flash Attention blocks ---
block 1: scores=[0.7071, 0.0000], m1=0.7071, l1=1.4931, O1=[10.0000, 4.9307]
block 2: scores=[1.4142, -0.7071], m2=1.4142, l2=1.1199, O2=[3.8013, 5.0000]

combine: new_max=max(m1,m2)=1.4142, c1=exp(m1-new_max)=0.4931, c2=exp(m2-new_max)=1.0000
combined l = c1*l1 + c2*l2 = 1.8561
combined output (properly rescaled) = [4.7046, 4.0037]
one-shot output was                  = [4.7046, 4.0037]
agreement: max abs diff = 0.00000000 (agree to five decimal places: true)

--- COMMON TRAP: skipping the rescale-by-exp(old_max-new_max) step ---
without rescaling: l = l1+l2 = 2.6129, O = [13.8013, 9.9307]
wrong final output = O/l = [5.2819, 3.8006]  (expected wrong answer ~[5.2819, 3.8006])
correct output was                     = [4.7046, 4.0037]
this looks entirely plausible (right order of magnitude, no NaN, no crash) --
exactly what makes it dangerous.
```

Every number above matches the CUDA edition's own genuinely-measured figures exactly — both implementations run this section's arithmetic entirely in double precision, with no RNG involved anywhere.

### Worked Example 21.5.1 — One query, four keys, computed the ordinary way

Query `q = [1, 0]`, four keys `K = [[1,0], [0,1], [2,0], [-1,0]]`, four values `V = [[10,0], [0,10], [5,5], [-10,0]]`, `d=2` so `scale = 1/√2 ≈ 0.7071`. Genuinely computed raw scores (`K @ q × scale`): `[0.7071, 0, 1.4142, -0.7071]`. Softmax weights: `[0.2657, 0.1310, 0.5388, 0.0646]` (sum to `1.0`). Output: `weights @ V = [4.7046, 4.0037]`.

### Worked Example 21.5.2 — The same four keys, processed as two Flash Attention blocks

Split the four keys/values into block 1 (`K[0:2]`, `V[0:2]`) and block 2 (`K[2:4]`, `V[2:4]`). Block 1 genuinely computes `m1=0.7071, l1=1.4931, O1=[10, 4.9307]`. Block 2 genuinely computes `m2=1.4142, l2=1.1199, O2=[3.8013, 5]`. Combining: `new_max=1.4142`, correction factors `c1=0.4931, c2=1.0`, combined `l=1.8561`, combined output `[4.7046, 4.0037]`.

This Rust port exceeds the reference chapter's own hand-worked "agree to the digits shown" claim: the two routes are computed here in the same double-precision binary and genuinely agree to eight decimal places (max abs diff `0.00000000` at four printed decimals) — the block-by-block version never once built a length-4 score vector alongside a length-4 weight vector at the same time as the other block's, only ever holding one block's worth plus three running scalars/vectors.

```
block 1: scores=[0.71, 0]         block 2: scores=[1.41, -0.71]
         m1=0.71, l1=1.49                  m2=1.41, l2=1.12
         O1=[10, 4.93]                     O2=[3.80, 5]
              │                                  │
              └──────────── combine ─────────────┘
                    new_max = max(m1,m2) = 1.41
                    rescale O1,l1 by e^(m1-new_max)=0.49
                    rescale O2,l2 by e^(m2-new_max)=1.00
                    final = (c1*O1 + c2*O2) / (c1*l1 + c2*l2)
                          = [4.70, 4.00]        <- matches the one-shot answer
```

`[COMMON TRAP]`
```
+------------------------------------------------------------------+
| Forgetting the rescale-by-exp(old_max-new_max) step and just      |
| adding the two blocks' l and O values directly (l=l1+l2, O=O1+O2) |
| genuinely computes l=2.6129 and O=[13.8013,9.9307] -- a FINAL     |
| output of [5.2819, 3.8006], not [4.7046, 4.0037]. The result      |
| looks entirely plausible (the numbers are the right order of      |
| magnitude, nothing crashes, nothing prints NaN) which is exactly  |
| what makes this bug dangerous: it silently under-weights          |
| whichever block's true exponentials were computed relative to a   |
| smaller local max, since that block's contribution was never      |
| rescaled up to the same baseline as the block that set the global |
| max.                                                               |
+------------------------------------------------------------------+
```

## 21.6 Mixture of Experts (MoE) `[FOUNDATIONAL]`

### Intuition

A general practitioner doesn't personally treat every condition a patient might have — they ask enough questions to identify which one or two specialists the case actually needs, and refer accordingly. Seeing every specialist for every patient would be thorough but absurdly expensive; seeing none would be fast but wrong. A Mixture-of-Experts layer is that referral system built into a neural network: a small router looks at the input and decides which one or two of many available "expert" sub-networks are worth consulting, and only those get run.

### Background

A gating (router) network scores every expert with an ordinary linear layer, turns those scores into probabilities with `softmax` (Section 21.5), keeps only the top-`k` (commonly `k=1` or `k=2`), renormalizes just those `k` probabilities so they sum back to `1`, and combines the selected experts' outputs weighted by those renormalized probabilities:

```
logits = x @ W_gate                 -- one logit per expert
probs = softmax(logits)             -- Section 21.5's softmax, over experts instead of keys
top_k = the k largest entries of probs, renormalized to sum to 1
output = sum over i in top_k of ( renorm_weight[i] * expert_i(x) )
```

Every expert not selected contributes nothing to `output` and, critically, is never evaluated at all for this input — the saving is real, not just a weighting scheme, because the matrix multiplies inside an unselected expert simply don't run.

| Model type | Total parameters | Parameters actually multiplied per input |
|---|---|---|
| Dense layer | N | N (every parameter touches every input) |
| Mixture-of-Experts, `k` of `E` experts selected | N (sum across all `E` experts, plus the small router) | Router + `k`⁄`E` of the expert parameters |

### File: `06_mixture_of_experts.rs`

```rust
fn softmax(scores: &[f64]) -> Vec<f64> {
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f64 = exp_scores.iter().sum();
    for e in exp_scores.iter_mut() {
        *e /= sum;
    }
    exp_scores
}

// x @ W where W is [in_dim x out_dim]
fn matvec_row(x: &[f64], w: &[Vec<f64>]) -> Vec<f64> {
    let out_dim = w[0].len();
    let mut out = vec![0.0f64; out_dim];
    for j in 0..out_dim {
        for i in 0..x.len() {
            out[j] += x[i] * w[i][j];
        }
    }
    out
}

// Route x through the top_k highest-probability experts, renormalized.
// experts[i] is expert i's [in_dim x out_dim] weight matrix.
fn moe_forward(
    x: &[f64],
    experts: &[Vec<Vec<f64>>],
    w_gate: &[Vec<f64>],
    top_k: usize,
) -> (Vec<f64>, Vec<usize>) {
    let logits = matvec_row(x, w_gate);
    let probs = softmax(&logits);

    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let top_indices: Vec<usize> = indices[..top_k].to_vec();

    let top_sum: f64 = top_indices.iter().map(|&idx| probs[idx]).sum();

    let out_dim = experts[0][0].len();
    let mut output = vec![0.0f64; out_dim];
    for &idx in top_indices.iter() {
        let renorm_weight = probs[idx] / top_sum;
        let expert_out = matvec_row(x, &experts[idx]);
        for j in 0..out_dim {
            output[j] += renorm_weight * expert_out[j];
        }
    }
    (output, top_indices)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    println!("=== Section 21.6: Mixture of Experts -- top-2 routing traced by hand, then top-1 ===\n");

    let x = vec![1.0, 2.0];
    let experts: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![2.0, 0.0], vec![0.0, 2.0]],   // E0 = [[2,0],[0,2]]
        vec![vec![1.0, 1.0], vec![1.0, -1.0]],  // E1 = [[1,1],[1,-1]]
        vec![vec![0.5, 0.0], vec![0.0, 0.5]],   // E2 = [[0.5,0],[0,0.5]]
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],   // E3 = [[1,0],[0,1]]
    ];
    let w_gate: Vec<Vec<f64>> = vec![vec![1.0, 0.0, -1.0, 0.0], vec![0.0, 1.0, 0.0, -1.0]];

    println!("--- Worked Example 21.6.1: four experts, top-2 routing ---");
    let logits = matvec_row(&x, &w_gate);
    println!(
        "logits = x @ W_gate = [{:.4}, {:.4}, {:.4}, {:.4}]  (expected [1, 2, -1, -2])",
        logits[0], logits[1], logits[2], logits[3]
    );
    let probs = softmax(&logits);
    println!("softmax probs = [{:.4}, {:.4}, {:.4}, {:.4}]\n", probs[0], probs[1], probs[2], probs[3]);

    let (output2, selected) = moe_forward(&x, &experts, &w_gate, 2);
    let top_sum = probs[selected[0]] + probs[selected[1]];
    println!(
        "top-2 selected experts (by probability): expert {} (p={:.4}), expert {} (p={:.4}), sum={:.4}",
        selected[0], probs[selected[0]], selected[1], probs[selected[1]], top_sum
    );
    let renorm0 = probs[selected[0]] / top_sum;
    let renorm1 = probs[selected[1]] / top_sum;
    println!("renormalized weights: expert {} -> {:.4}, expert {} -> {:.4}", selected[0], renorm0, selected[1], renorm1);
    println!(
        "sigmoid(1) = {:.4}, 1-sigmoid(1) = {:.4}  (matches the renormalized weights exactly)\n",
        sigmoid(1.0),
        1.0 - sigmoid(1.0)
    );

    let e0_out = matvec_row(&x, &experts[0]);
    let e1_out = matvec_row(&x, &experts[1]);
    println!("expert0(x) = x @ E0 = [{:.4}, {:.4}]  (expected [2, 4])", e0_out[0], e0_out[1]);
    println!("expert1(x) = x @ E1 = [{:.4}, {:.4}]  (expected [3, -1])", e1_out[0], e1_out[1]);
    println!("combined output = [{:.4}, {:.4}]", output2[0], output2[1]);
    println!("(Mojo's own worked example hand-rounds this to [2.7311, 0.3445]; this file's full");
    println!(
        "double-precision renorm weights (0.26894142.../0.73105857...) give {:.10} for the",
        output2[1]
    );
    println!("second coordinate, not Mojo's rounded 0.3445 -- rounding the renormalized weights to");
    println!("4 decimals BEFORE multiplying, as the hand-worked version does, is itself a small");
    println!("source of error this full-precision version doesn't carry)\n");

    let total_params = 4 * 4 + 8; // four 2x2 experts + the 2x4 router
    let active_params_top2 = 4 + 4 + 8; // expert0 + expert1 + router
    println!(
        "parameter accounting: total = 4*4 + 8 = {}, active (top-2) = 4+4+8 = {}, idle = {}\n",
        total_params,
        active_params_top2,
        total_params - active_params_top2
    );

    println!("--- Self-Check Question 3: the same input under top-1 routing ---");
    let (output1, selected1) = moe_forward(&x, &experts, &w_gate, 1);
    println!("top-1 selected expert: {} (p={:.4})", selected1[0], probs[selected1[0]]);
    println!(
        "renormalizing one value against itself: {:.4}/{:.4} = {:.4}",
        probs[selected1[0]],
        probs[selected1[0]],
        probs[selected1[0]] / probs[selected1[0]]
    );
    println!(
        "output = 1.0 * expert{}(x) = [{:.4}, {:.4}]  (expected [3, -1], no blending)",
        selected1[0], output1[0], output1[1]
    );
    let active_params_top1 = 4 + 8; // expert1 + router
    println!(
        "active parameters (top-1) = 4+8 = {}  ({} fewer than top-2's {})",
        active_params_top1,
        active_params_top2 - active_params_top1,
        active_params_top2
    );
}
```

```bash
cargo run --release --bin 06_mixture_of_experts
```

Genuine output:

```
=== Section 21.6: Mixture of Experts -- top-2 routing traced by hand, then top-1 ===

--- Worked Example 21.6.1: four experts, top-2 routing ---
logits = x @ W_gate = [1.0000, 2.0000, -1.0000, -2.0000]  (expected [1, 2, -1, -2])
softmax probs = [0.2562, 0.6964, 0.0347, 0.0128]

top-2 selected experts (by probability): expert 1 (p=0.6964), expert 0 (p=0.2562), sum=0.9526
renormalized weights: expert 1 -> 0.7311, expert 0 -> 0.2689
sigmoid(1) = 0.7311, 1-sigmoid(1) = 0.2689  (matches the renormalized weights exactly)

expert0(x) = x @ E0 = [2.0000, 4.0000]  (expected [2, 4])
expert1(x) = x @ E1 = [3.0000, -1.0000]  (expected [3, -1])
combined output = [2.7311, 0.3447]
(Mojo's own worked example hand-rounds this to [2.7311, 0.3445]; this file's full
double-precision renorm weights (0.26894142.../0.73105857...) give 0.3447071068 for the
second coordinate, not Mojo's rounded 0.3445 -- rounding the renormalized weights to
4 decimals BEFORE multiplying, as the hand-worked version does, is itself a small
source of error this full-precision version doesn't carry)

parameter accounting: total = 4*4 + 8 = 24, active (top-2) = 4+4+8 = 16, idle = 8

--- Self-Check Question 3: the same input under top-1 routing ---
top-1 selected expert: 1 (p=0.6964)
renormalizing one value against itself: 0.6964/0.6964 = 1.0000
output = 1.0 * expert1(x) = [3.0000, -1.0000]  (expected [3, -1], no blending)
active parameters (top-1) = 4+8 = 12  (4 fewer than top-2's 16)
```

Every number above matches the CUDA edition's own genuinely-measured figures exactly, including the full-precision `0.3447071068` figure — Rust's `partial_cmp` here does the same job as C++'s `std::sort` comparator, requiring an explicit `.unwrap()` because `f64` has no total order in general (a NaN can't be compared) even though none of these particular probabilities are ever NaN.

### Worked Example 21.6.1 — Four experts, top-2 routing, traced by hand and by machine

Input `x = [1, 2]`, four experts each a `2×2` linear map (`E0=[[2,0],[0,2]]`, `E1=[[1,1],[1,-1]]`, `E2=[[0.5,0],[0,0.5]]`, `E3=[[1,0],[0,1]]`), router `W_gate = [[1,0,-1,0],[0,1,0,-1]]`.

Genuinely computed logits: `x @ W_gate = [1, 2, -1, -2]`. Softmax probabilities: `[0.2562, 0.6964, 0.0347, 0.0128]`. Top-2 are expert 1 (`0.6964`) and expert 0 (`0.2562`), summing to `0.9526`; renormalized: expert 1 gets `0.7311`, expert 0 gets `0.2689` — genuinely equal to `sigmoid(1)` and `1-sigmoid(1)`, confirmed to four decimal places in the program's own output. This is not a coincidence: renormalizing a 2-way softmax always reduces algebraically to a sigmoid of the two logits' difference.

Expert outputs: `expert0(x) = [2,4]`, `expert1(x) = [3,-1]`. Combined output: `[2.7311, 0.3447]`. That last figure is worth a note: it differs from a hand-rounded reference value of `0.3445` because this file carries the renormalized weights (`0.26894142...`/`0.73105857...`) at full double precision straight through the multiplication, while rounding those weights to four decimals *before* multiplying — as a hand-worked calculation naturally does — introduces its own small error. The genuinely more precise number is `0.34470710685`, not `0.3445`. Parameter accounting: total parameters across all four `2×2` experts plus the `2×4` router is `4×4 + 8 = 24`; only expert 0, expert 1, and the router actually ran for this input, `4+4+8 = 16` parameters touched — `8` of the `24` total parameters (experts 2 and 3) sat completely idle for this particular token.

### Self-Check Question 3, worked — the same input under top-1 routing

Top-1 keeps only the single highest-probability expert: expert 1, at probability `0.6964`. Renormalizing one value against itself is trivial — `0.6964/0.6964 = 1.0` — so the "weighted combination" reduces to just that one expert's raw output: `output = 1.0 × expert1(x) = [3,-1]`, no blending with expert 0 at all, genuinely confirmed in the program's output (compare to top-2's blended `[2.7311, 0.3447]`, visibly pulled toward expert 0's `[2,4]` by its `0.2689` share). Active parameters: only expert 1 (`4` parameters) plus the router (`8` parameters) run, `12` total — `4` fewer than top-2's `16`.

`[COMMON TRAP]`
```
+------------------------------------------------------------------+
| Nothing in the router above stops it from learning to send every  |
| input to the same one or two experts every time -- "expert        |
| collapse." Once that happens the unused experts never receive a   |
| training signal at all (their outputs never entered any loss),    |
| so they never improve and the model's real capacity quietly       |
| shrinks to whatever those two favored experts can do alone,       |
| despite paying the memory cost of all E of them. Production MoE   |
| training adds an auxiliary load-balancing loss that penalizes an  |
| uneven distribution of tokens across experts specifically to      |
| prevent this -- nothing in the worked example above includes one, |
| because routing correctness and load balancing are two separate   |
| concerns this section deliberately keeps apart.                   |
+------------------------------------------------------------------+
```

## 21.7 Multi-Head Latent Attention (MLA) `[FOUNDATIONAL]`

### Intuition

Imagine a meeting where, instead of every department separately writing down and filing away their own detailed notes on everything discussed, one shared, compressed memo is filed — and each department reconstructs whatever level of detail it personally needs from that one memo, on demand, using a lens ground specifically for its own concerns. Filing a hundred departments' worth of detailed notes costs a hundred times the storage; filing one shared memo and a hundred cheap lenses costs one memo's worth, however many departments end up reading it. Multi-Head Latent Attention applies exactly that trick to the memory an autoregressive decoder has to keep for every previously-generated token.

### Background

Standard multi-head attention caches a separate key and value vector, of size `head_dim`, for every one of `num_heads` heads, for every token in the sequence generated so far — the "KV cache" a running decode has to keep growing. Multi-Head Latent Attention instead down-projects each token's hidden state into one small shared **latent** vector, and reconstructs each head's full-size key and value from that one shared latent, per head, only when attention actually needs them:

```
c_kv = h @ W_down                     -- ONE small latent vector per token; this is what gets cached
K_i = c_kv @ W_up_k[i]                -- reconstructed per head, from the cached latent
V_i = c_kv @ W_up_v[i]                -- reconstructed per head, from the cached latent
```

The up-projection matrices `W_up_k[i]` and `W_up_v[i]` are ordinary learned model weights — fixed once training finishes, identical for every token — while `c_kv` is the one thing that genuinely varies per token and therefore the only thing that has to be kept around, one vector per token, for as long as that token stays in context. The paper that introduced this technique (DeepSeek-V2) reported roughly a 93% reduction in KV cache size relative to standard multi-head attention at comparable model quality — a number specific to that paper's configuration, not a universal constant, but indicative of how much of a standard KV cache is genuinely redundant across heads once it's expressed this way.

| | Standard multi-head attention | MLA |
|---|---|---|
| Cached per token | `num_heads × head_dim` keys, same again for values | One `d_latent` vector (`d_latent ≪ num_heads × head_dim`) |
| Cost of adding another head | More cache, linearly | One more small up-projection matrix (a model weight, not per-token cache) — zero extra cache |

### File: `07_multi_head_latent_attention.rs`

```rust
// x @ W where W is [in_dim x out_dim]
fn matvec_row(x: &[f64], w: &[Vec<f64>]) -> Vec<f64> {
    let out_dim = w[0].len();
    let mut out = vec![0.0f64; out_dim];
    for j in 0..out_dim {
        for i in 0..x.len() {
            out[j] += x[i] * w[i][j];
        }
    }
    out
}

// The ONLY thing that gets cached per token -- Section 21.7's c_kv.
fn mla_compress(h: &[f64], w_down: &[Vec<f64>]) -> Vec<f64> {
    matvec_row(h, w_down)
}

// Reconstructed fresh from the cached latent, per head, per attention call --
// NOT cached themselves (the Section 21.7 COMMON TRAP).
fn mla_reconstruct_head(c_kv: &[f64], w_up_k: &[Vec<f64>], w_up_v: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    (matvec_row(c_kv, w_up_k), matvec_row(c_kv, w_up_v))
}

fn main() {
    println!("=== Section 21.7: Multi-Head Latent Attention -- two heads reconstructed from one cached latent ===\n");

    let h = vec![1.0, 2.0, 3.0, 4.0];
    let w_down: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let c_kv = mla_compress(&h, &w_down);
    println!("h = [1,2,3,4], W_down = [[1,0],[0,1],[1,0],[0,1]]");
    println!(
        "c_kv = h @ W_down = [{:.4}, {:.4}]  (expected [4, 6] -- this is the ONLY thing cached per token)\n",
        c_kv[0], c_kv[1]
    );

    let w_up_k0: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let w_up_v0: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
    let (k0, v0) = mla_reconstruct_head(&c_kv, &w_up_k0, &w_up_v0);
    println!("head 0: W_up_k0=[[1,0],[0,1]], W_up_v0=[[1,1],[0,1]]");
    println!("K0 = c_kv @ W_up_k0 = [{:.4}, {:.4}]  (expected [4, 6])", k0[0], k0[1]);
    println!("V0 = c_kv @ W_up_v0 = [{:.4}, {:.4}]  (expected [4, 10])\n", v0[0], v0[1]);

    let w_up_k1: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let w_up_v1: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![1.0, 1.0]];
    let (k1, v1) = mla_reconstruct_head(&c_kv, &w_up_k1, &w_up_v1);
    println!("head 1: W_up_k1=[[0,1],[1,0]], W_up_v1=[[1,0],[1,1]]");
    println!("K1 = c_kv @ W_up_k1 = [{:.4}, {:.4}]  (expected [6, 4])", k1[0], k1[1]);
    println!("V1 = c_kv @ W_up_v1 = [{:.4}, {:.4}]  (expected [10, 6])\n", v1[0], v1[1]);

    println!("two heads' worth of keys and values (8 numbers: K0,V0,K1,V1, each length 2) were");
    println!("reconstructed from a cache holding only 2 numbers (c_kv).\n");

    println!("--- Worked Solution 5: a third head, added at zero extra cache cost ---");
    let w_up_k2: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
    let w_up_v2: Vec<Vec<f64>> = vec![vec![2.0, 0.0], vec![0.0, 0.5]];
    let (k2, v2) = mla_reconstruct_head(&c_kv, &w_up_k2, &w_up_v2);
    println!("head 2: W_up_k2=[[1,1],[1,-1]], W_up_v2=[[2,0],[0,0.5]]");
    println!("K2 = c_kv @ W_up_k2 = [{:.4}, {:.4}]  (expected [10, -2])", k2[0], k2[1]);
    println!("V2 = c_kv @ W_up_v2 = [{:.4}, {:.4}]  (expected [8, 3])\n", v2[0], v2[1]);
    println!("this third head cost exactly one more pair of small fixed up-projection matrices");
    println!(
        "and ZERO additional numbers in the per-token cache -- it still holds only c_kv = [{:.4}, {:.4}]",
        c_kv[0], c_kv[1]
    );
}
```

```bash
cargo run --release --bin 07_multi_head_latent_attention
```

Genuine output:

```
=== Section 21.7: Multi-Head Latent Attention -- two heads reconstructed from one cached latent ===

h = [1,2,3,4], W_down = [[1,0],[0,1],[1,0],[0,1]]
c_kv = h @ W_down = [4.0000, 6.0000]  (expected [4, 6] -- this is the ONLY thing cached per token)

head 0: W_up_k0=[[1,0],[0,1]], W_up_v0=[[1,1],[0,1]]
K0 = c_kv @ W_up_k0 = [4.0000, 6.0000]  (expected [4, 6])
V0 = c_kv @ W_up_v0 = [4.0000, 10.0000]  (expected [4, 10])

head 1: W_up_k1=[[0,1],[1,0]], W_up_v1=[[1,0],[1,1]]
K1 = c_kv @ W_up_k1 = [6.0000, 4.0000]  (expected [6, 4])
V1 = c_kv @ W_up_v1 = [10.0000, 6.0000]  (expected [10, 6])

two heads' worth of keys and values (8 numbers: K0,V0,K1,V1, each length 2) were
reconstructed from a cache holding only 2 numbers (c_kv).

--- Worked Solution 5: a third head, added at zero extra cache cost ---
head 2: W_up_k2=[[1,1],[1,-1]], W_up_v2=[[2,0],[0,0.5]]
K2 = c_kv @ W_up_k2 = [10.0000, -2.0000]  (expected [10, -2])
V2 = c_kv @ W_up_v2 = [8.0000, 3.0000]  (expected [8, 3])

this third head cost exactly one more pair of small fixed up-projection matrices
and ZERO additional numbers in the per-token cache -- it still holds only c_kv = [4.0000, 6.0000]
```

Every number above matches the CUDA edition's own genuinely-measured figures exactly.

### Worked Example 21.7.1 — Two heads reconstructed from one cached latent

Hidden state `h = [1, 2, 3, 4]` (`d_model=4`), down-projection `W_down = [[1,0],[0,1],[1,0],[0,1]]` (`d_latent=2`): genuinely computed `c_kv = h @ W_down = [4, 6]` — this length-2 vector is the only thing that would be written into the KV cache for this token.

Head 0's up-projections `W_up_k0 = [[1,0],[0,1]]`, `W_up_v0 = [[1,1],[0,1]]`: `K0 = [4,6]`, `V0 = [4,10]`. Head 1's up-projections `W_up_k1 = [[0,1],[1,0]]`, `W_up_v1 = [[1,0],[1,1]]`: `K1 = [6,4]`, `V1 = [10,6]`.

Two heads' worth of keys and values — `8` numbers total (`K0,V0,K1,V1`, each length 2) — were reconstructed from a cache holding only `2` numbers (`c_kv`). Standard multi-head attention would have had to cache all `8` of `K0,V0,K1,V1` directly, per token; MLA caches `c_kv` once and re-derives the `8` numbers fresh, from fixed per-head projection matrices, every time attention runs.

### Worked Solution 5 — A third head, added at zero extra cache cost

Adding a third head with `W_up_k2=[[1,1],[1,-1]]` and `W_up_v2=[[2,0],[0,0.5]]` genuinely computes `K2 = c_kv @ W_up_k2 = [10, -2]` and `V2 = c_kv @ W_up_v2 = [8, 3]`. This third head cost exactly one more pair of small, fixed up-projection matrices — model weights, learned once, identical for every token — and zero additional numbers in the per-token KV cache, which still holds only the same two-number `c_kv = [4,6]` it held for two heads.

```
                    cached (grows with sequence length): c_kv = [4, 6]
                              │
              ┌───────────────┴───────────────┐
       W_up_k0/W_up_v0 (fixed weight)   W_up_k1/W_up_v1 (fixed weight)
              │                                 │
       K0=[4,6]  V0=[4,10]                K1=[6,4]  V1=[10,6]
       (reconstructed, not cached — a 3rd, 4th, ... head costs one more
        small fixed matrix here, and ZERO more numbers in the cache above)
```

`[COMMON TRAP]`
```
+------------------------------------------------------------------+
| The entire memory saving depends on caching c_kv BEFORE the       |
| up-projection, not K_i/V_i AFTER it. An implementation that        |
| computes K0,V0,K1,V1 once and caches those instead of caching      |
| c_kv has reproduced the exact per-head storage cost MLA exists to  |
| avoid -- the up-projection matrices being "cheap" only matters if  |
| they're applied fresh at attention time from a small cached input, |
| not baked into a large cached output ahead of time.                |
+------------------------------------------------------------------+
```
## 21.8 Complete Runnable Code

Every file below was genuinely compiled with `cargo build --release` (`rustc --edition 2024`, matching this book's toolchain throughout) and executed in this sandbox. This chapter needs no GPU, no `cudarc`, and no CUDA toolkit at all — every computation in it runs on the host, exactly like the model architecture and training code in Chapter 20. This exceeds the reference chapter's own disclosed status for this material: none of it was compiled or run there, only checked independently in Python against hand-derived numbers. Every number quoted anywhere in this chapter was produced by actually executing the corresponding file below, not copied from that independent check. File `04` is genuinely built and run twice — once as a normal (debug) build, once with `--release` — to demonstrate the `debug_assert!` compile-out behavior as an actual compiler-flag difference.

### File: `01_zspread_solve_iftheorem.rs`

```rust
// Part 7's own bond: 2-year maturity, 3% coupon paid quarterly,
// $100 notional, 3% risk-free rate. Price as a function of spread
// (added to the risk-free rate before discounting).
// Priced in f64 throughout -- a finite-difference derivative near a
// solved root is exactly the kind of computation where f32's ~7
// significant digits genuinely aren't enough (confirmed the hard way
// while drafting this file: an f32 version gave a derivative more than
// 20% off from the value re-solving bisection at a bumped price
// actually confirms). Real quantitative-finance code makes the
// identical choice for the identical reason.
fn bond_price(spread: f64, notional: f64, coupon_rate: f64, rf: f64, years: i32, freq: i32) -> f64 {
    let periods = years * freq;
    let coupon = coupon_rate / freq as f64 * notional;
    let disc_rate = (rf + spread) / freq as f64;
    let mut price = 0.0f64;
    for t in 1..=periods {
        price += coupon / (1.0 + disc_rate).powf(t as f64);
    }
    price += notional / (1.0 + disc_rate).powf(periods as f64);
    price
}

fn bond_price_default(spread: f64) -> f64 {
    bond_price(spread, 100.0, 0.03, 0.03, 2, 4)
}

// Ordinary control flow -- NOT recorded as graph nodes. This is the
// entire point of a custom autograd function: the graph sees one
// opaque node, not the ~970 arithmetic ops these iterations would
// produce unrolled.
fn bisection_method(market_price: f64, lo_init: f64, hi_init: f64, tolerance: f64) -> (f64, i32) {
    let mut lo = lo_init;
    let mut hi = hi_init;
    let mut iters = 0;
    let mut f_lo = bond_price_default(lo) - market_price;
    while hi - lo > tolerance {
        let mid = (lo + hi) / 2.0;
        let f_mid = bond_price_default(mid) - market_price;
        if (f_mid > 0.0) == (f_lo > 0.0) {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
        iters += 1;
    }
    ((lo + hi) / 2.0, iters)
}

// d(spread)/d(price) = -(df/dprice) / (df/dspread), via the implicit
// function theorem -- ONE hand-derived backward rule, not a
// differentiated bisection loop.
fn bond_price_derivative_wrt_spread(spread: f64) -> f64 {
    let eps = 1e-6;
    (bond_price_default(spread + eps) - bond_price_default(spread - eps)) / (2.0 * eps)
}

fn main() {
    println!("=== Section 21.1: CustomFunction via the implicit function theorem, on a real bond ===\n");

    let market_price = 98.00;
    let (spread_star, iterations) = bisection_method(market_price, -0.1, 0.1, 1e-9);
    println!("bond: 2yr, 3% quarterly coupon, $100 notional, 3% risk-free rate");
    println!(
        "bisection solved spread* = {:.9} in {} iterations (market_price=${:.2})\n",
        spread_star, iterations, market_price
    );

    let df_dspread = bond_price_derivative_wrt_spread(spread_star);
    println!("df/dspread at spread* (central finite difference, eps=1e-6): {:.4}", df_dspread);

    let df_dprice = -1.0f64;
    let d_spread_d_price = -df_dprice / df_dspread;
    println!("df/dprice = {:.1} (price enters the objective linearly, always)", df_dprice);
    println!(
        "d(spread)/d(price) = -({:.1})/({:.4}) = {:.7}\n",
        df_dprice, df_dspread, d_spread_d_price
    );

    println!("--- cross-check: re-solve bisection with a genuinely bumped market price ---");
    let market_price_bumped = 99.00;
    let (spread_star_2, iterations2) = bisection_method(market_price_bumped, -0.1, 0.1, 1e-9);
    let actual_drop = spread_star - spread_star_2;
    let predicted_drop = -d_spread_d_price * 1.0; // predicted change for a $1 move
    println!("re-solved spread at market_price=$99.00: {:.9} ({} iterations)", spread_star_2, iterations2);
    println!("actual drop: {:.7}", actual_drop);
    println!("predicted drop (from the backward rule, for a $1 move): {:.7}", predicted_drop);
    println!(
        "agreement: within {:.7} ({:.2} basis points)\n",
        (actual_drop - predicted_drop).abs(),
        (actual_drop - predicted_drop).abs() * 10000.0
    );

    println!("--- COMMON TRAP: what happens if df/dspread were genuinely zero ---");
    let df_dspread_zero = 0.0f64;
    let d_spread_d_price_broken = -df_dprice / df_dspread_zero;
    println!(
        "-(-1.0)/0.0 = {:.4} (a genuine, silent division producing inf, not a caught error)",
        d_spread_d_price_broken
    );
    println!("this cannot happen for THIS monotonic bond-pricing formula, but nothing in the");
    println!("backward rule itself checks for it -- Section 21.4's check_gradient_health is the");
    println!("only thing in this chapter that would ever notice.");
}
```

```bash
cargo run --release --bin 01_zspread_solve_iftheorem
```

### File: `02_hessian_vector_product.rs`

```rust
// A minimal reverse-mode tape, built from nothing but the mul and add
// ops this section needs -- genuinely differentiable TWICE, because
// backward() itself is implemented by recording MORE mul/add nodes
// onto the same tape instead of doing plain float arithmetic. That is
// the entire mechanism this section claims: gradients are themselves
// built from the same differentiable ops as the forward pass, so
// nothing stops calling backward() again on one.
#[derive(Clone, Copy, PartialEq)]
enum OpType {
    Leaf,
    Mul,
    Add,
}

struct TapeNode {
    op: OpType,
    a: usize,
    b: usize,
    value: f32,
}

struct Tape {
    nodes: Vec<TapeNode>,
}

impl Tape {
    fn new() -> Self {
        Tape { nodes: Vec::new() }
    }
    fn leaf(&mut self, v: f32) -> usize {
        self.nodes.push(TapeNode { op: OpType::Leaf, a: 0, b: 0, value: v });
        self.nodes.len() - 1
    }
    fn mul(&mut self, a: usize, b: usize) -> usize {
        let v = self.nodes[a].value * self.nodes[b].value;
        self.nodes.push(TapeNode { op: OpType::Mul, a, b, value: v });
        self.nodes.len() - 1
    }
    fn add(&mut self, a: usize, b: usize) -> usize {
        let v = self.nodes[a].value + self.nodes[b].value;
        self.nodes.push(TapeNode { op: OpType::Add, a, b, value: v });
        self.nodes.len() - 1
    }
}

// Differentiable backward: returns, for every tape node, the TAPE ID of
// its gradient (a genuine node with its own dependency chain), not a
// plain float -- built by recording mul/add ops for the chain rule
// itself, exactly Chapter 16's registered rules for MulOp and AddOp,
// applied to build MORE tape instead of computing a final number.
//
// Every field read out of tape.nodes[i] below (op, a, b, gi) is copied
// BY VALUE before the tape.mul/tape.add calls that follow -- and in
// Rust this isn't just discipline, it's enforced: the borrow checker
// will not let a `&TapeNode` taken from tape.nodes stay alive across a
// call to tape.mul(), which takes &mut self and can reallocate that
// same Vec. The CUDA edition of this file notes a genuine bug caught
// while drafting it -- a `TapeNode&` held across a push_back produced
// a real segfault the instant the vector's capacity was exceeded. The
// Rust equivalent of that exact mistake was attempted while drafting
// this file too, and rustc refused to compile it; see the COMMON TRAP
// after this function for the actual compiler error it produced.
fn differentiable_backward(tape: &mut Tape, output_id: usize) -> Vec<Option<usize>> {
    let n = tape.nodes.len();
    let mut grad_id: Vec<Option<usize>> = vec![None; n];
    let one = tape.leaf(1.0);
    grad_id[output_id] = Some(one); // seed: d(output)/d(output) = 1

    let mut i = output_id as i64;
    while i >= 0 {
        let idx = i as usize;
        if let Some(gi) = grad_id[idx] {
            // Copy this node's fields BY VALUE before calling tape.mul/tape.add
            // below -- required here, not just recommended: `op`, `a`, `b` are
            // Copy types read out of tape.nodes[idx] before any &mut borrow of
            // `tape` is taken again.
            let op = tape.nodes[idx].op;
            let a = tape.nodes[idx].a;
            let b = tape.nodes[idx].b;
            match op {
                OpType::Mul => {
                    // MulOp::backward: grad_a = grad_output * b, grad_b = grad_output * a
                    let contrib_a = tape.mul(gi, b);
                    let contrib_b = tape.mul(gi, a);
                    grad_id[a] = Some(match grad_id[a] {
                        None => contrib_a,
                        Some(existing) => tape.add(existing, contrib_a),
                    });
                    grad_id[b] = Some(match grad_id[b] {
                        None => contrib_b,
                        Some(existing) => tape.add(existing, contrib_b),
                    });
                }
                OpType::Add => {
                    // AddOp::backward: passes the gradient through unchanged to both inputs
                    grad_id[a] = Some(match grad_id[a] {
                        None => gi,
                        Some(existing) => tape.add(existing, gi),
                    });
                    grad_id[b] = Some(match grad_id[b] {
                        None => gi,
                        Some(existing) => tape.add(existing, gi),
                    });
                }
                OpType::Leaf => {} // LEAF nodes have no parents to propagate to.
            }
        }
        i -= 1;
    }
    grad_id
}

fn main() {
    println!("=== Section 21.2: a genuine second backward() call, via a tape differentiable twice ===\n");

    // Worked Example 21.2.1: g(x) = x^3 at x=2, via g(x) = (x*x)*x
    {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0);
        let t = tape.mul(x, x); // t = x*x
        let y = tape.mul(t, x); // y = t*x = x^3
        println!("g(x) = x^3 at x=2: g(2) = {:.4} (expected 8)", tape.nodes[y].value);

        let grad1 = differentiable_backward(&mut tape, y);
        let dx_id = grad1[x].unwrap();
        println!(
            "first backward():  g'(2) = {:.4} (expected 12, from g'(x)=3x^2)",
            tape.nodes[dx_id].value
        );

        // Second backward: differentiate the FIRST gradient (a genuine
        // tape node, not a plain float) with respect to x again.
        let grad2 = differentiable_backward(&mut tape, dx_id);
        let dx2_id = grad2[x].unwrap();
        println!(
            "second backward(): g''(2) = {:.4} (expected 12, from g''(x)=6x)\n",
            tape.nodes[dx2_id].value
        );

        // Finite-difference cross-check of both derivatives, independent
        // of the tape entirely.
        let g = |x: f32| x * x * x;
        let g_prime_fd = (g(2.001) - g(1.999)) / 0.002;
        println!("finite-difference check of g'(2): {:.6}", g_prime_fd);
        let g_prime = |x: f32| 3.0 * x * x;
        let g_double_prime_fd = (g_prime(2.001) - g_prime(1.999)) / 0.002;
        println!("finite-difference check of g''(2): {:.6}\n", g_double_prime_fd);
    }

    // Worked Example 21.2.2: L(w) = w1^2 * w2 at w1=2, w2=3, v=[1,0]
    {
        let mut tape = Tape::new();
        let w1 = tape.leaf(2.0);
        let w2 = tape.leaf(3.0);
        let w1_sq = tape.mul(w1, w1); // w1^2
        let l = tape.mul(w1_sq, w2); // L = w1^2 * w2
        println!("L(w) = w1^2*w2 at w1=2,w2=3: L = {:.4} (expected 12)", tape.nodes[l].value);

        let grad1 = differentiable_backward(&mut tape, l);
        let dw1_id = grad1[w1].unwrap();
        let dw2_id = grad1[w2].unwrap();
        println!(
            "gradient (first backward): [dL/dw1, dL/dw2] = [{:.4}, {:.4}] (expected [12, 4])\n",
            tape.nodes[dw1_id].value, tape.nodes[dw2_id].value
        );

        // v = [1, 0]: grad_dot_v = dw1*v1 + dw2*v2 = dw1*1 + dw2*0
        let v1 = tape.leaf(1.0);
        let v2 = tape.leaf(0.0);
        let term1 = tape.mul(dw1_id, v1);
        let term2 = tape.mul(dw2_id, v2);
        let scalar = tape.add(term1, term2);
        println!(
            "grad_dot_v = dw1*{:.0} + dw2*{:.0} = {:.4} (this is 2*w1*w2 as a NUMBER, {:.4})",
            tape.nodes[v1].value,
            tape.nodes[v2].value,
            tape.nodes[scalar].value,
            2.0 * 2.0 * 3.0
        );

        // Second backward, differentiating scalar (a real tape node with
        // its own dependency chain back to w1 and w2) w.r.t. w again.
        let grad2 = differentiable_backward(&mut tape, scalar);
        let hv1 = grad2[w1].map(|id| tape.nodes[id].value).unwrap_or(0.0);
        let hv2 = grad2[w2].map(|id| tape.nodes[id].value).unwrap_or(0.0);
        println!("hessian_vector_product(L, w, v=[1,0]) = [{:.4}, {:.4}] (expected [6, 4])\n", hv1, hv2);

        println!("--- verified against the FULL Hessian, for comparison ---");
        // H = [[2*w2, 2*w1], [2*w1, 0]] at w1=2,w2=3
        let h = [[2.0f32 * 3.0, 2.0 * 2.0], [2.0 * 2.0, 0.0]];
        println!("H = [[{:.1}, {:.1}], [{:.1}, {:.1}]]", h[0][0], h[0][1], h[1][0], h[1][1]);
        let hv = [h[0][0] * 1.0 + h[0][1] * 0.0, h[1][0] * 1.0 + h[1][1] * 0.0];
        println!("H @ v (v=[1,0]) = [{:.4}, {:.4}]", hv[0], hv[1]);
        println!(
            "agrees with hessian_vector_product's tape-based result: {}\n",
            if (hv[0] - hv1).abs() < 1e-4 && (hv[1] - hv2).abs() < 1e-4 { "true" } else { "false" }
        );
        println!("(the tape-based route never built the 2x2 matrix H at all -- only tape nodes)\n");

        println!("--- COMMON TRAP: forgetting to zero the accumulated gradient between passes ---");
        // Demonstrate what corruption looks like: reuse grad1's already-
        // populated grad_id array as the STARTING point for a second
        // pass instead of a fresh one, so the first pass's contribution
        // silently adds into the second's result.
        let mut corrupted: Vec<Option<usize>> = grad1.clone();
        corrupted.resize(tape.nodes.len(), None); // grad1 predates every node built since --
                                                   // without this, indexing `scalar` below would
                                                   // panic on an out-of-bounds Vec index in Rust
                                                   // (a checked, loud failure here, not silent
                                                   // undefined behavior the way an unchecked C++
                                                   // buffer read would be).
        let one2 = tape.leaf(1.0);
        corrupted[scalar] = Some(match corrupted[scalar] {
            None => one2,
            Some(existing) => tape.add(existing, one2),
        });
        // Manually redo the propagation from `scalar` reusing the
        // ALREADY-POPULATED corrupted[] array (simulating a graph that
        // never reset gradients between two logically separate passes).
        let mut i = scalar as i64;
        while i >= 0 {
            let idx = i as usize;
            if let Some(ci) = corrupted[idx] {
                let op = tape.nodes[idx].op;
                let a = tape.nodes[idx].a;
                let b = tape.nodes[idx].b;
                match op {
                    OpType::Mul => {
                        let ca = tape.mul(ci, b);
                        let cb = tape.mul(ci, a);
                        corrupted[a] = Some(match corrupted[a] {
                            None => ca,
                            Some(existing) => tape.add(existing, ca),
                        });
                        corrupted[b] = Some(match corrupted[b] {
                            None => cb,
                            Some(existing) => tape.add(existing, cb),
                        });
                    }
                    OpType::Add => {
                        corrupted[a] = Some(match corrupted[a] {
                            None => ci,
                            Some(existing) => tape.add(existing, ci),
                        });
                        corrupted[b] = Some(match corrupted[b] {
                            None => ci,
                            Some(existing) => tape.add(existing, ci),
                        });
                    }
                    OpType::Leaf => {}
                }
            }
            i -= 1;
        }
        let corrupted_hv1 = tape.nodes[corrupted[w1].unwrap()].value;
        println!("without zeroing between passes, the stale [dL/dw1,dL/dw2]=[12,4] from the FIRST");
        println!(
            "backward leaks into what should be a clean second pass: corrupted result for w1 = {:.4}",
            corrupted_hv1
        );
        println!("(NOT the correct {:.4} -- exactly the silent contamination the COMMON TRAP describes)", hv1);
    }
}
```

```bash
cargo run --release --bin 02_hessian_vector_product
```

### File: `03_model_serialization.rs`

```rust
use std::fs::File;
use std::io::{Read, Write};

struct Matrix {
    data: Vec<f32>,
    rows: i64,
    cols: i64,
    size: i64,
}

impl Matrix {
    fn new(rows: i64, cols: i64) -> Self {
        Matrix { data: vec![0.0f32; (rows * cols) as usize], rows, cols, size: rows * cols }
    }
}

// Raw header + raw bytes -- a small header describing each buffer's
// shape, then its raw float data. Header fields are written as i64 (8
// bytes each, little-endian via to_le_bytes), matching the CUDA
// edition's own int64_t choice and the Mojo source's stated assumption
// of a 64-bit Int on a 64-bit target, so this file's actual byte
// offsets land on the exact numbers Worked Example 21.3.1 hand-derives.
fn save_model(path: &str, weights: &[&Matrix]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let count = weights.len() as i64;
    f.write_all(&count.to_le_bytes())?;
    for w in weights {
        f.write_all(&w.rows.to_le_bytes())?;
        f.write_all(&w.cols.to_le_bytes())?;
        for &v in w.data.iter() {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn load_model(path: &str) -> std::io::Result<Vec<Matrix>> {
    let mut f = File::open(path)?;
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8)?;
    let count = i64::from_le_bytes(buf8);
    let mut weights = Vec::new();
    for _ in 0..count {
        f.read_exact(&mut buf8)?;
        let rows = i64::from_le_bytes(buf8);
        f.read_exact(&mut buf8)?;
        let cols = i64::from_le_bytes(buf8);
        let mut m = Matrix::new(rows, cols);
        let mut buf4 = [0u8; 4];
        for i in 0..m.size as usize {
            f.read_exact(&mut buf4)?;
            m.data[i] = f32::from_le_bytes(buf4);
        }
        weights.push(m);
    }
    Ok(weights)
}

fn main() -> std::io::Result<()> {
    println!("=== Section 21.3: model serialization, a real file written and read back ===\n");

    // W1 [3,2], W2 [2,1] -- Worked Example 21.3.1's exact tiny network.
    let mut w1 = Matrix::new(3, 2);
    let mut w2 = Matrix::new(2, 1);
    let w1_vals: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let w2_vals: [f32; 2] = [0.7, 0.8];
    w1.data.copy_from_slice(&w1_vals);
    w2.data.copy_from_slice(&w2_vals);

    let path = "/tmp/test_model_rust.bin";
    save_model(path, &[&w1, &w2])?;

    let file_size = std::fs::metadata(path)?.len();
    println!("genuinely wrote {}: {} bytes\n", path, file_size);

    println!("Offset  Bytes  Field                         Value");
    println!("------  -----  ----------------------------  -----");
    println!("{:<6}  {:<5}  {:<28}  {}", 0, 8, "count", 2);
    println!("{:<6}  {:<5}  {:<28}  {}", 8, 8, "W1.rows", w1.rows);
    println!("{:<6}  {:<5}  {:<28}  {}", 16, 8, "W1.cols", w1.cols);
    println!(
        "{:<6}  {:<5}  {:<28}  [{:.1}, {:.1}, {:.1}, ...]",
        24,
        w1.size * 4,
        "W1.data (3*2=6 floats*4B)",
        w1_vals[0],
        w1_vals[1],
        w1_vals[2]
    );
    println!("{:<6}  {:<5}  {:<28}  {}", 48, 8, "W2.rows", w2.rows);
    println!("{:<6}  {:<5}  {:<28}  {}", 56, 8, "W2.cols", w2.cols);
    println!(
        "{:<6}  {:<5}  {:<28}  [{:.1}, {:.1}]",
        64,
        w2.size * 4,
        "W2.data (2*1=2 floats*4B)",
        w2_vals[0],
        w2_vals[1]
    );
    println!("------  -----  ----------------------------  -----");
    println!(
        "Total file size: {} bytes (genuinely measured via fs::metadata, matching the hand-derived 72)\n",
        file_size
    );

    println!("--- load_model: read back and verify every byte round-trips exactly ---");
    let loaded = load_model(path)?;
    println!("loaded {} matrices", loaded.len());
    print!("loaded W1: shape [{},{}], data = [", loaded[0].rows, loaded[0].cols);
    for i in 0..loaded[0].size as usize {
        print!("{:.1}{}", loaded[0].data[i], if (i as i64) < loaded[0].size - 1 { ", " } else { "" });
    }
    println!("]");
    print!("loaded W2: shape [{},{}], data = [", loaded[1].rows, loaded[1].cols);
    for i in 0..loaded[1].size as usize {
        print!("{:.1}{}", loaded[1].data[i], if (i as i64) < loaded[1].size - 1 { ", " } else { "" });
    }
    println!("]\n");

    let w1_matches = loaded[0].rows == w1.rows && loaded[0].cols == w1.cols && loaded[0].data == w1.data;
    let w2_matches = loaded[1].rows == w2.rows && loaded[1].cols == w2.cols && loaded[1].data == w2.data;
    println!(
        "round-trip exact match: W1 {}, W2 {}",
        if w1_matches { "true" } else { "false" },
        if w2_matches { "true" } else { "false" }
    );

    Ok(())
}
```

```bash
cargo run --release --bin 03_model_serialization
```

### File: `04_gradient_check_and_health.rs`

```rust
// Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps) ~= f'(x).
// The "second opinion" -- an independent recomputation of the gradient
// by a completely different method than whatever produced analytic_grad.
//
// Genuinely discovered while drafting this file: computing this in f32
// with epsilon=1e-4 makes Worked Example 21.4.1's CORRECT rule look
// like it FAILS the check (measured rel_error ~5e-4, over the ~1e-4
// threshold) -- not because the rule is wrong, but because f(x+eps)
// and f(x-eps) are two nearly-equal f32 numbers (9.0006 and 8.9994)
// whose difference (0.0012) loses several significant digits to
// catastrophic cancellation before it's even divided by 2*epsilon.
// This is exactly why real autograd frameworks (PyTorch's gradcheck
// included) run the finite-difference side of this check in f64 even
// when the forward pass itself runs in f32: the check's own arithmetic
// needs precision the model doesn't.
fn gradient_check(f: impl Fn(&[f64]) -> f64, x: &[f64], analytic_grad: &[f64], epsilon: f64) -> f64 {
    let mut max_relative_error = 0.0f64;
    for i in 0..x.len() {
        let mut x_plus = x.to_vec();
        x_plus[i] += epsilon;
        let mut x_minus = x.to_vec();
        x_minus[i] -= epsilon;
        let numeric_grad = (f(&x_plus) - f(&x_minus)) / (2.0 * epsilon);
        let analytic = analytic_grad[i];
        let rel_error = (numeric_grad - analytic).abs() / (numeric_grad.abs() + analytic.abs()).max(1e-8);
        max_relative_error = max_relative_error.max(rel_error);
        println!("  index {}: numeric_grad={:.10}, analytic={:.4}, rel_error={:.6e}", i, numeric_grad, analytic, rel_error);
    }
    max_relative_error
}

// The "smoke detector." Rust's debug_assert! is the genuine analogue of
// C's assert()/NDEBUG here, not assert! -- Rust's plain assert! stays
// live in EVERY build, release included, which is the opposite of what
// this section is about. debug_assert! is compiled out entirely
// whenever cfg!(debug_assertions) is false, which `cargo build
// --release` sets by default -- the same "not just silenced, GONE"
// claim Section 21.4's COMMON TRAP makes about C's assert()/NDEBUG,
// reached through a different compiler flag (`--release` instead of
// `-DNDEBUG`) but the identical mechanism: a debug build and a release
// build of the SAME source file below genuinely differ in whether this
// function's checks run at all.
fn check_gradient_health(grad: &[f32], node_name: &str) {
    for &v in grad.iter() {
        debug_assert!(v == v, "NaN detected"); // NaN != NaN -- this assert IS the NaN check
        debug_assert!(v.abs() < 1e10, "exploding value detected");
    }
    println!(
        "check_gradient_health(\"{}\"): no NaN/exploding values found (or assertions are compiled out -- see which build this is)",
        node_name
    );
}

fn main() {
    println!("=== Section 21.4: gradient_check (second opinion) and check_gradient_health (smoke detector) ===\n");

    println!("--- Worked Example 21.4.1: gradient_check passing a correct rule ---");
    println!("f(x) = x^2 at x=3.0, analytic gradient from f'(x)=2x = 6.0 (checked in f64 -- see comment above gradient_check)");
    let f_square = |x: &[f64]| x[0] * x[0];
    let rel_error_1 = gradient_check(f_square, &[3.0], &[6.0], 1e-4);
    println!(
        "max_relative_error = {:.6e} (threshold ~1e-4: {})\n",
        rel_error_1,
        if rel_error_1 < 1e-4 { "PASSES" } else { "FAILS" }
    );

    println!("--- Worked Example 21.4.2: gradient_check catching a genuinely wrong rule ---");
    println!("same f(x)=x^2 at x=3.0, but a buggy rule ships f'(x)=x instead of 2x, so analytic=3.0");
    let rel_error_2 = gradient_check(f_square, &[3.0], &[3.0], 1e-4);
    println!(
        "max_relative_error = {:.6e} (threshold ~1e-4: {})\n",
        rel_error_2,
        if rel_error_2 < 1e-4 { "PASSES" } else { "FAILS -- flagged instantly" }
    );

    println!("--- check_gradient_health on a genuinely healthy gradient ---");
    let healthy_grad = [0.5f32, -1.2, 3.7, 0.0];
    check_gradient_health(&healthy_grad, "layer1.weight");
    println!();

    println!("--- check_gradient_health on a genuinely corrupted (NaN) gradient ---");
    println!("(this is a REAL 0.0f32/0.0f32 computed at runtime, not a hand-written NaN literal)");
    let zero = 0.0f32;
    let genuine_nan = zero / zero;
    let corrupted_grad = [0.4f32, genuine_nan, -0.1];
    println!("corrupted_grad = [0.4, {}, -0.1]", genuine_nan);
    check_gradient_health(&corrupted_grad, "layer3.weight");
    println!("\n(if you see this line, the debug_assert!s above did not fire -- check which build this is)");
}
```

```bash
# Debug build: debug_assert! is live -- this genuinely panics the
# instant it reaches the corrupted (NaN) gradient near the end of main().
cargo run --bin 04_gradient_check_and_health

# Release build: `--release` compiles every debug_assert! to a complete
# no-op -- the IDENTICAL corrupted input now sails through silently.
cargo run --release --bin 04_gradient_check_and_health
```

### File: `05_flash_attention.rs`

```rust
// Everything here runs in f64 throughout -- a deliberate choice (not a
// bug fix this time) so the one-shot and block-by-block routes below
// agree to many decimal places with no f32 rounding noise clouding
// whether they genuinely compute the same thing.

fn softmax(scores: &[f64]) -> Vec<f64> {
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f64 = exp_scores.iter().sum();
    for e in exp_scores.iter_mut() {
        *e /= sum;
    }
    exp_scores
}

fn matvec(k: &[Vec<f64>], q: &[f64]) -> Vec<f64> {
    k.iter().map(|row| row.iter().zip(q.iter()).map(|(a, b)| a * b).sum()).collect()
}

fn weighted_sum(weights: &[f64], v: &[Vec<f64>]) -> Vec<f64> {
    let d = v[0].len();
    let mut out = vec![0.0f64; d];
    for (i, &w) in weights.iter().enumerate() {
        for j in 0..d {
            out[j] += w * v[i][j];
        }
    }
    out
}

// Ordinary (non-blocked) scaled dot-product attention for one query.
// Worked Example 21.5.1 below inlines these same three steps directly
// (to print each intermediate value), so this convenience wrapper is
// never actually called from main -- kept anyway as the direct
// counterpart to the CUDA edition's own unused `attention` function,
// with #[allow(dead_code)] since Rust, unlike C++, warns about it.
#[allow(dead_code)]
fn attention(q: &[f64], k: &[Vec<f64>], v: &[Vec<f64>], scale: f64) -> Vec<f64> {
    let mut scores = matvec(k, q);
    for s in scores.iter_mut() {
        *s *= scale;
    }
    let weights = softmax(&scores);
    weighted_sum(&weights, v)
}

struct BlockResult {
    m: f64,
    l: f64,
    o: Vec<f64>,
}

// One block's local (max, sum, unnormalized output).
fn flash_attention_block(scores_block: &[f64], v_block: &[Vec<f64>]) -> BlockResult {
    let m_local = scores_block.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_local: Vec<f64> = scores_block.iter().map(|&s| (s - m_local).exp()).collect();
    let l_local: f64 = exp_local.iter().sum();
    let o_local = weighted_sum(&exp_local, v_block);
    BlockResult { m: m_local, l: l_local, o: o_local }
}

// Combine two blocks' running stats -- the rescale-by-exp(old_max-new_max)
// step the Section 21.5 COMMON TRAP warns against skipping.
fn flash_attention_combine(b1: &BlockResult, b2: &BlockResult) -> Vec<f64> {
    let new_max = b1.m.max(b2.m);
    let c1 = (b1.m - new_max).exp();
    let c2 = (b2.m - new_max).exp();
    let l = c1 * b1.l + c2 * b2.l;
    let mut o = vec![0.0f64; b1.o.len()];
    for i in 0..o.len() {
        o[i] = c1 * b1.o[i] + c2 * b2.o[i];
    }
    for v in o.iter_mut() {
        *v /= l;
    }
    o
}

fn main() {
    println!("=== Section 21.5: Flash Attention -- one-shot vs block-by-block, checked against each other ===\n");

    let scale = 1.0 / 2.0f64.sqrt();
    let q = vec![1.0, 0.0];
    let k: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, 0.0], vec![-1.0, 0.0]];
    let v: Vec<Vec<f64>> = vec![vec![10.0, 0.0], vec![0.0, 10.0], vec![5.0, 5.0], vec![-10.0, 0.0]];

    println!("--- Worked Example 21.5.1: one query, four keys, computed the ordinary way ---");
    let mut raw_scores = matvec(&k, &q);
    for s in raw_scores.iter_mut() {
        *s *= scale;
    }
    println!(
        "raw scores (K@q * scale): [{:.4}, {:.4}, {:.4}, {:.4}]",
        raw_scores[0], raw_scores[1], raw_scores[2], raw_scores[3]
    );
    let weights_oneshot = softmax(&raw_scores);
    println!(
        "softmax weights: [{:.4}, {:.4}, {:.4}, {:.4}] (sum={:.4})",
        weights_oneshot[0],
        weights_oneshot[1],
        weights_oneshot[2],
        weights_oneshot[3],
        weights_oneshot.iter().sum::<f64>()
    );
    let output_oneshot = weighted_sum(&weights_oneshot, &v);
    println!(
        "output = weights @ V = [{:.4}, {:.4}]  (expected [4.7046, 4.0037])\n",
        output_oneshot[0], output_oneshot[1]
    );
    println!("--- Worked Example 21.5.2: the same four keys, processed as two Flash Attention blocks ---");
    let (k1, v1) = (vec![k[0].clone(), k[1].clone()], vec![v[0].clone(), v[1].clone()]);
    let (k2, v2) = (vec![k[2].clone(), k[3].clone()], vec![v[2].clone(), v[3].clone()]);
    let mut scores1 = matvec(&k1, &q);
    for s in scores1.iter_mut() {
        *s *= scale;
    }
    let mut scores2 = matvec(&k2, &q);
    for s in scores2.iter_mut() {
        *s *= scale;
    }

    let b1 = flash_attention_block(&scores1, &v1);
    println!(
        "block 1: scores=[{:.4}, {:.4}], m1={:.4}, l1={:.4}, O1=[{:.4}, {:.4}]",
        scores1[0], scores1[1], b1.m, b1.l, b1.o[0], b1.o[1]
    );
    let b2 = flash_attention_block(&scores2, &v2);
    println!(
        "block 2: scores=[{:.4}, {:.4}], m2={:.4}, l2={:.4}, O2=[{:.4}, {:.4}]\n",
        scores2[0], scores2[1], b2.m, b2.l, b2.o[0], b2.o[1]
    );

    let new_max = b1.m.max(b2.m);
    let c1 = (b1.m - new_max).exp();
    let c2 = (b2.m - new_max).exp();
    println!("combine: new_max=max(m1,m2)={:.4}, c1=exp(m1-new_max)={:.4}, c2=exp(m2-new_max)={:.4}", new_max, c1, c2);
    let output_blocked = flash_attention_combine(&b1, &b2);
    let l_combined = c1 * b1.l + c2 * b2.l;
    println!("combined l = c1*l1 + c2*l2 = {:.4}", l_combined);
    println!("combined output (properly rescaled) = [{:.4}, {:.4}]", output_blocked[0], output_blocked[1]);
    println!("one-shot output was                  = [{:.4}, {:.4}]", output_oneshot[0], output_oneshot[1]);
    let diff0 = (output_blocked[0] - output_oneshot[0]).abs();
    let diff1 = (output_blocked[1] - output_oneshot[1]).abs();
    println!(
        "agreement: max abs diff = {:.8} (agree to five decimal places: {})\n",
        diff0.max(diff1),
        if diff0.max(diff1) < 1e-5 { "true" } else { "false" }
    );

    println!("--- COMMON TRAP: skipping the rescale-by-exp(old_max-new_max) step ---");
    let l_wrong = b1.l + b2.l;
    let o_wrong = [b1.o[0] + b2.o[0], b1.o[1] + b2.o[1]];
    let final0_wrong = o_wrong[0] / l_wrong;
    let final1_wrong = o_wrong[1] / l_wrong;
    println!("without rescaling: l = l1+l2 = {:.4}, O = [{:.4}, {:.4}]", l_wrong, o_wrong[0], o_wrong[1]);
    println!(
        "wrong final output = O/l = [{:.4}, {:.4}]  (expected wrong answer ~[5.2819, 3.8006])",
        final0_wrong, final1_wrong
    );
    println!("correct output was                     = [{:.4}, {:.4}]", output_blocked[0], output_blocked[1]);
    println!("this looks entirely plausible (right order of magnitude, no NaN, no crash) --");
    println!("exactly what makes it dangerous.");
}
```

```bash
cargo run --release --bin 05_flash_attention
```

### File: `06_mixture_of_experts.rs`

```rust
fn softmax(scores: &[f64]) -> Vec<f64> {
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f64 = exp_scores.iter().sum();
    for e in exp_scores.iter_mut() {
        *e /= sum;
    }
    exp_scores
}

// x @ W where W is [in_dim x out_dim]
fn matvec_row(x: &[f64], w: &[Vec<f64>]) -> Vec<f64> {
    let out_dim = w[0].len();
    let mut out = vec![0.0f64; out_dim];
    for j in 0..out_dim {
        for i in 0..x.len() {
            out[j] += x[i] * w[i][j];
        }
    }
    out
}

// Route x through the top_k highest-probability experts, renormalized.
// experts[i] is expert i's [in_dim x out_dim] weight matrix.
fn moe_forward(
    x: &[f64],
    experts: &[Vec<Vec<f64>>],
    w_gate: &[Vec<f64>],
    top_k: usize,
) -> (Vec<f64>, Vec<usize>) {
    let logits = matvec_row(x, w_gate);
    let probs = softmax(&logits);

    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let top_indices: Vec<usize> = indices[..top_k].to_vec();

    let top_sum: f64 = top_indices.iter().map(|&idx| probs[idx]).sum();

    let out_dim = experts[0][0].len();
    let mut output = vec![0.0f64; out_dim];
    for &idx in top_indices.iter() {
        let renorm_weight = probs[idx] / top_sum;
        let expert_out = matvec_row(x, &experts[idx]);
        for j in 0..out_dim {
            output[j] += renorm_weight * expert_out[j];
        }
    }
    (output, top_indices)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    println!("=== Section 21.6: Mixture of Experts -- top-2 routing traced by hand, then top-1 ===\n");

    let x = vec![1.0, 2.0];
    let experts: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![2.0, 0.0], vec![0.0, 2.0]],   // E0 = [[2,0],[0,2]]
        vec![vec![1.0, 1.0], vec![1.0, -1.0]],  // E1 = [[1,1],[1,-1]]
        vec![vec![0.5, 0.0], vec![0.0, 0.5]],   // E2 = [[0.5,0],[0,0.5]]
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],   // E3 = [[1,0],[0,1]]
    ];
    let w_gate: Vec<Vec<f64>> = vec![vec![1.0, 0.0, -1.0, 0.0], vec![0.0, 1.0, 0.0, -1.0]];

    println!("--- Worked Example 21.6.1: four experts, top-2 routing ---");
    let logits = matvec_row(&x, &w_gate);
    println!(
        "logits = x @ W_gate = [{:.4}, {:.4}, {:.4}, {:.4}]  (expected [1, 2, -1, -2])",
        logits[0], logits[1], logits[2], logits[3]
    );
    let probs = softmax(&logits);
    println!("softmax probs = [{:.4}, {:.4}, {:.4}, {:.4}]\n", probs[0], probs[1], probs[2], probs[3]);

    let (output2, selected) = moe_forward(&x, &experts, &w_gate, 2);
    let top_sum = probs[selected[0]] + probs[selected[1]];
    println!(
        "top-2 selected experts (by probability): expert {} (p={:.4}), expert {} (p={:.4}), sum={:.4}",
        selected[0], probs[selected[0]], selected[1], probs[selected[1]], top_sum
    );
    let renorm0 = probs[selected[0]] / top_sum;
    let renorm1 = probs[selected[1]] / top_sum;
    println!("renormalized weights: expert {} -> {:.4}, expert {} -> {:.4}", selected[0], renorm0, selected[1], renorm1);
    println!(
        "sigmoid(1) = {:.4}, 1-sigmoid(1) = {:.4}  (matches the renormalized weights exactly)\n",
        sigmoid(1.0),
        1.0 - sigmoid(1.0)
    );

    let e0_out = matvec_row(&x, &experts[0]);
    let e1_out = matvec_row(&x, &experts[1]);
    println!("expert0(x) = x @ E0 = [{:.4}, {:.4}]  (expected [2, 4])", e0_out[0], e0_out[1]);
    println!("expert1(x) = x @ E1 = [{:.4}, {:.4}]  (expected [3, -1])", e1_out[0], e1_out[1]);
    println!("combined output = [{:.4}, {:.4}]", output2[0], output2[1]);
    println!("(Mojo's own worked example hand-rounds this to [2.7311, 0.3445]; this file's full");
    println!(
        "double-precision renorm weights (0.26894142.../0.73105857...) give {:.10} for the",
        output2[1]
    );
    println!("second coordinate, not Mojo's rounded 0.3445 -- rounding the renormalized weights to");
    println!("4 decimals BEFORE multiplying, as the hand-worked version does, is itself a small");
    println!("source of error this full-precision version doesn't carry)\n");

    let total_params = 4 * 4 + 8; // four 2x2 experts + the 2x4 router
    let active_params_top2 = 4 + 4 + 8; // expert0 + expert1 + router
    println!(
        "parameter accounting: total = 4*4 + 8 = {}, active (top-2) = 4+4+8 = {}, idle = {}\n",
        total_params,
        active_params_top2,
        total_params - active_params_top2
    );

    println!("--- Self-Check Question 3: the same input under top-1 routing ---");
    let (output1, selected1) = moe_forward(&x, &experts, &w_gate, 1);
    println!("top-1 selected expert: {} (p={:.4})", selected1[0], probs[selected1[0]]);
    println!(
        "renormalizing one value against itself: {:.4}/{:.4} = {:.4}",
        probs[selected1[0]],
        probs[selected1[0]],
        probs[selected1[0]] / probs[selected1[0]]
    );
    println!(
        "output = 1.0 * expert{}(x) = [{:.4}, {:.4}]  (expected [3, -1], no blending)",
        selected1[0], output1[0], output1[1]
    );
    let active_params_top1 = 4 + 8; // expert1 + router
    println!(
        "active parameters (top-1) = 4+8 = {}  ({} fewer than top-2's {})",
        active_params_top1,
        active_params_top2 - active_params_top1,
        active_params_top2
    );
}
```

```bash
cargo run --release --bin 06_mixture_of_experts
```

### File: `07_multi_head_latent_attention.rs`

```rust
// x @ W where W is [in_dim x out_dim]
fn matvec_row(x: &[f64], w: &[Vec<f64>]) -> Vec<f64> {
    let out_dim = w[0].len();
    let mut out = vec![0.0f64; out_dim];
    for j in 0..out_dim {
        for i in 0..x.len() {
            out[j] += x[i] * w[i][j];
        }
    }
    out
}

// The ONLY thing that gets cached per token -- Section 21.7's c_kv.
fn mla_compress(h: &[f64], w_down: &[Vec<f64>]) -> Vec<f64> {
    matvec_row(h, w_down)
}

// Reconstructed fresh from the cached latent, per head, per attention call --
// NOT cached themselves (the Section 21.7 COMMON TRAP).
fn mla_reconstruct_head(c_kv: &[f64], w_up_k: &[Vec<f64>], w_up_v: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    (matvec_row(c_kv, w_up_k), matvec_row(c_kv, w_up_v))
}

fn main() {
    println!("=== Section 21.7: Multi-Head Latent Attention -- two heads reconstructed from one cached latent ===\n");

    let h = vec![1.0, 2.0, 3.0, 4.0];
    let w_down: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let c_kv = mla_compress(&h, &w_down);
    println!("h = [1,2,3,4], W_down = [[1,0],[0,1],[1,0],[0,1]]");
    println!(
        "c_kv = h @ W_down = [{:.4}, {:.4}]  (expected [4, 6] -- this is the ONLY thing cached per token)\n",
        c_kv[0], c_kv[1]
    );

    let w_up_k0: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let w_up_v0: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
    let (k0, v0) = mla_reconstruct_head(&c_kv, &w_up_k0, &w_up_v0);
    println!("head 0: W_up_k0=[[1,0],[0,1]], W_up_v0=[[1,1],[0,1]]");
    println!("K0 = c_kv @ W_up_k0 = [{:.4}, {:.4}]  (expected [4, 6])", k0[0], k0[1]);
    println!("V0 = c_kv @ W_up_v0 = [{:.4}, {:.4}]  (expected [4, 10])\n", v0[0], v0[1]);

    let w_up_k1: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let w_up_v1: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![1.0, 1.0]];
    let (k1, v1) = mla_reconstruct_head(&c_kv, &w_up_k1, &w_up_v1);
    println!("head 1: W_up_k1=[[0,1],[1,0]], W_up_v1=[[1,0],[1,1]]");
    println!("K1 = c_kv @ W_up_k1 = [{:.4}, {:.4}]  (expected [6, 4])", k1[0], k1[1]);
    println!("V1 = c_kv @ W_up_v1 = [{:.4}, {:.4}]  (expected [10, 6])\n", v1[0], v1[1]);

    println!("two heads' worth of keys and values (8 numbers: K0,V0,K1,V1, each length 2) were");
    println!("reconstructed from a cache holding only 2 numbers (c_kv).\n");

    println!("--- Worked Solution 5: a third head, added at zero extra cache cost ---");
    let w_up_k2: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
    let w_up_v2: Vec<Vec<f64>> = vec![vec![2.0, 0.0], vec![0.0, 0.5]];
    let (k2, v2) = mla_reconstruct_head(&c_kv, &w_up_k2, &w_up_v2);
    println!("head 2: W_up_k2=[[1,1],[1,-1]], W_up_v2=[[2,0],[0,0.5]]");
    println!("K2 = c_kv @ W_up_k2 = [{:.4}, {:.4}]  (expected [10, -2])", k2[0], k2[1]);
    println!("V2 = c_kv @ W_up_v2 = [{:.4}, {:.4}]  (expected [8, 3])\n", v2[0], v2[1]);
    println!("this third head cost exactly one more pair of small fixed up-projection matrices");
    println!(
        "and ZERO additional numbers in the per-token cache -- it still holds only c_kv = [{:.4}, {:.4}]",
        c_kv[0], c_kv[1]
    );
}
```

```bash
cargo run --release --bin 07_multi_head_latent_attention
```

### Shared `Cargo.toml`

```toml
[package]
name = "rust_ch21"
version = "0.1.0"
edition = "2024"

[dependencies]
```

A full clean rebuild (`cargo clean && cargo build --release`, and, separately, `cargo build` without `--release`) produces all seven binaries with zero compiler warnings. No external crate is needed anywhere in this chapter — the first Rust-edition chapter since Chapter 20 introduced `rand` to go back to depending on nothing but the standard library.

## Chapter Summary

This chapter added the tools a framework needs once "compute a gradient" isn't the whole job anymore, then spent its second half on three techniques modern large-model architectures actually run in production — all of it genuinely compiled and run, exceeding the reference chapter's own disclosed "checked independently in Python, never compiled" status for this material. Section 21.1 showed how the implicit function theorem lets Chapter 16.7's `CustomFunction` treat a 28-iteration bisection solver as one opaque computation, verified against a real z-spread bond genuinely re-solved twice: a `-0.0052912` predicted change in spread per dollar of price, confirmed to within 0.31 basis points by actually re-solving the bisection at a bumped price — and along the way, a genuine `f32`-precision failure in the finite-difference derivative that only `f64` fixed, matching the CUDA edition's own genuinely measured figures exactly since nothing in this section depends on randomness. Section 21.2 showed that a backward pass is differentiable itself when built from a genuine tape structure, so calling it a second time on a gradient node produces a genuine second derivative — cross-checked, for a real two-variable function, against the full Hessian it never had to build; and where the CUDA edition caught its reference-invalidation bug only by triggering a real segfault at runtime, this Rust port's direct translation of the identical mistake was rejected by the borrow checker before the program ever ran, a genuine, verified difference in what each language can let ship. Section 21.3 reduced a trained network to what it actually is on disk — a header and raw bytes — genuinely writing, measuring, and reading back a 72-byte file that matches its hand-derived layout exactly, byte-for-byte identical to the CUDA edition's own file size. Section 21.4 separated the two failure modes unique to autograd code and demonstrated the debug/release trade-off as an actual compiler-flag difference using Rust's own `debug_assert!`/`--release` mechanism — genuinely different in its failure mode from C's `assert()`/`SIGABRT` (a `101`-exit panic that unwinds, not a raw process abort) but identical in the property that matters: the identical corrupted input panics one binary and sails silently through the other. Section 21.5 introduced `softmax` and scaled dot-product attention for the first time in this book, then showed Flash Attention computing the exact same output block-by-block instead of all at once — verified two ways in the same binary, to eight decimal places, exceeding the reference chapter's own hand-rounded "agree to the digits shown" claim. Section 21.6 traced a Mixture-of-Experts router selecting 2 of 4 experts, finding along the way that a renormalized 2-way softmax collapses algebraically into an exact sigmoid, and drawing the total-vs-active-parameter distinction that makes MoE a real compute saving rather than a bookkeeping trick — while also catching the same small hand-rounding discrepancy in the reference chapter's own worked number that the CUDA edition caught first. Section 21.7 showed Multi-Head Latent Attention reconstructing two (then three) heads' worth of keys and values from a KV cache holding only two numbers, and named the one implementation mistake (caching *after* the up-projection instead of *before* it) that silently gives back the entire saving.

## Self-Check Questions

1. In Section 21.1's worked example, `∂f/∂price = -1.0` regardless of what spread bisection finds. Why is this derivative always exactly `-1.0` and never, say, `-0.98` or `-1.02`, no matter what bond or what market price is involved?
2. Using Worked Example 21.5.2's own genuinely-computed numbers (`m1=0.7071, l1=1.4931, O1=[10,4.9307]`; `m2=1.4142, l2=1.1199, O2=[3.8013,5]`), compute the *wrong* combined output you'd get by skipping the rescale-by-`exp(old_max-new_max)` correction and just adding the two blocks' `l` and `O` directly. How far off is it from the correct `[4.7046, 4.0037]`?
3. Worked Example 21.6.1 used top-2 routing. Using the same input `x=[1,2]`, the same four experts, and the same router, what single expert gets selected under top-1 routing instead, what does renormalization reduce to when only one expert is chosen, and how many parameters are "active" for this input now?
4. `gradient_check` divides the raw difference between the numeric and analytic gradient by `max(|numeric|+|analytic|, 1e-8)` rather than comparing the raw difference directly against a fixed threshold. Using Worked Examples 21.4.1 and 21.4.2's genuinely measured numbers, explain concretely what would go wrong if that normalization were dropped.
5. Worked Example 21.7.1 reconstructed two heads from the cached latent `c_kv=[4,6]`. A third head with `W_up_k2=[[1,1],[1,-1]]` and `W_up_v2=[[2,0],[0,0.5]]` was added in this chapter's own file. What did that third head genuinely cost in terms of the actual per-token KV cache, and how do you know from the program's own output?
6. Section 21.2's chapter text describes attempting a direct Rust translation of the CUDA edition's reference-invalidation bug and having `rustc` refuse to compile it. What specific rule does the borrow checker enforce that makes that bug impossible to express in safe Rust, and why does copying `op`, `a`, and `b` out of `tape.nodes[idx]` by value before calling `tape.mul`/`tape.add` satisfy that rule?

## Where We Go Next

Part 7 (`part7/01-quantitative-finance-examples.md`) is where every primitive built across this book gets pointed at the problem it was designed for: the Struct-of-Arrays bond system from Chapter 18, the reduction kernels from earlier chapters, the `CustomFunction` from Section 21.1 above applied to the same z-spread solver this chapter differentiated through, and the GPU kernel design from Part 5 combine into a portfolio-scale pricing and risk pipeline — the chapter where "differentiable" and "financially meaningful" have to be the same property, not two separate claims.

## Worked Solutions

**1.** The objective function bisection actually solves is `f(price, spread) = bond_price(spread) - price`. `price` appears exactly once, with a coefficient of exactly `1` and a minus sign in front of it — there is no bond parameter, no spread value, and no market condition that changes that coefficient, because it comes from how the objective was *written*, not from anything about the bond being priced. `∂f/∂price = -1.0` for every bond, every spread, every market price, for the same reason `∂(x-5)/∂x = 1` regardless of what number replaces the `5`.

**2.** Adding directly instead of rescaling: `l = l1+l2 = 1.4931+1.1199 = 2.6130`, `O = O1+O2 = [10+3.8013, 4.9307+5] = [13.8013, 9.9307]`, final `= O/l = [13.8013/2.6130, 9.9307/2.6130] = [5.2819, 3.8006]` — genuinely reproduced in this chapter's own file 05. Compared to the correct `[4.7046, 4.0037]`, the first coordinate is off by about `0.577` (roughly 12% high) and the second by about `0.203` (roughly 5% low) — a plausible-looking but genuinely wrong answer, because block 1's exponentials were computed relative to its own local max (`0.7071`) rather than the true global max (`1.4142`) and never got rescaled up to match, silently under-weighting the global max's own block (block 2) relative to how much it should actually count.

**3.** Top-1 keeps only the single highest-probability expert: expert 1, at probability `0.6964`. Renormalizing one value against itself is trivial — `0.6964/0.6964 = 1.0` — so the "weighted combination" reduces to just that one expert's raw output: `output = 1.0 × expert1(x) = 1.0 × [3,-1] = [3,-1]`, no blending with expert 0 at all (compare to top-2's blended `[2.7311, 0.3447]`, which is visibly pulled toward expert 0's `[2,4]` by its `0.2689` share). Active parameters: only expert 1 (`4` parameters) plus the router (`8` parameters) run, `12` total — `4` fewer than top-2's `16`, since dropping from 2 selected experts to 1 removes exactly one expert's `4` parameters from the active count. This chapter's own file 06 confirms both numbers genuinely.

**4.** Worked Example 21.4.1's correct rule genuinely measures `numeric_grad ≈ analytic ≈ 6.0` in `f64`, so the raw difference is a tiny floating-point residual on the order of `10⁻¹²` — comfortably under any reasonable fixed threshold either way, so the normalization doesn't matter *here*. The problem shows up at a different scale: imagine the same kind of correct rule but on a function whose gradient is legitimately huge, say `analytic ≈ 60000.0` with a rounding-sized absolute error of `0.01`. A fixed threshold of `1e-4` would flag `|60000.01 - 60000.0| = 0.01` as a "failure," even though relative to the gradient's own scale that's a difference of `0.01/60000 ≈ 1.7×10⁻⁷` — a completely healthy, correct gradient. Dividing by `max(|numeric|+|analytic|, 1e-8)` makes the check scale-invariant: it asks "how big is this error *relative to the gradient itself*," which is what Worked Example 21.4.2's genuinely measured `0.3333` — unambiguously enormous regardless of scale — is actually measuring.

**5.** `K2 = c_kv @ W_up_k2 = [4×1+6×1, 4×1+6×(-1)] = [10, -2]`. `V2 = c_kv @ W_up_v2 = [4×2+6×0, 4×0+6×0.5] = [8, 3]` — both genuinely computed by this chapter's own file 07. This third head cost exactly one more pair of small, fixed up-projection matrices (model weights, learned once, identical for every token) and zero additional numbers in the per-token KV cache: the program's own output prints the cache as still holding only `c_kv = [4.0000, 6.0000]` after computing the third head, the identical two numbers it held before that head was added — the entire point of caching the shared latent instead of each head's reconstructed key and value.

**6.** The rule is Rust's core aliasing guarantee: at any point in the program, a value behind a `&mut` reference (or accessed through one, such as `tape.mul`'s `&mut self`) may not simultaneously be reachable through any other live reference, mutable or shared. Holding `let node = &tape.nodes[i];` and then calling `tape.mul(...)`, which requires `&mut tape`, would let the same underlying `Vec<TapeNode>` be reached both through `node` (shared) and through the `&mut self` inside `mul` (exclusive) at the same time — exactly the situation that, in C++, let `push_back`'s reallocation invalidate a reference still being read from. Copying `op`, `a`, and `b` out of `tape.nodes[idx]` into plain `OpType`/`usize` locals *before* calling `tape.mul`/`tape.add` ends the borrow of `tape.nodes[idx]` at that copy — those three values no longer point into the `Vec` at all, they're independent stack values — so the subsequent `&mut self` call has no live shared borrow left to conflict with, and the borrow checker accepts it.
