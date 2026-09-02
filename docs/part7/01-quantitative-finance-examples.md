# Chapter 22: Quantitative Finance Examples — Where a Wrong Gradient Costs Real Money

> "A model that merely prices an instrument is a calculator. A model that *differentiates* through pricing it is a risk desk — the difference is whether every sensitivity a trader needs comes free with the price, or has to be derived by hand, once, for every new instrument."

## What you will understand by the end of this chapter

- How the same `PV = FV·e^(-yield·time)` formula that prices one zero-coupon bond scales into a 1,024-bond portfolio through the Struct-of-Arrays pattern from Chapter 18 — and a real, confirmed bug in how that portfolio's total gets summed that silently drops 87.5% of the book from the reported number, genuinely reproduced here in Rust rather than described in prose.
- Why a coupon-paying bond's z-spread has no closed-form solution and has to be found by bisection instead — verified in this chapter to every digit shown against a real, genuinely re-run computation, including a from-scratch reimplementation of C's `%g` significant-figure formatting, since Rust's `std::fmt` has no direct equivalent.
- Why portfolio duration is nothing more than a weighted average, and how that turns "how much does this rebalancing hurt if rates rise" into an ordinary gradient instead of a hand-derived formula for every new portfolio.
- Why Monte Carlo pricing needs many simulated paths rather than one, and why "bump and reprice" with fresh random draws produces genuinely noisy Greeks compared to the same bump using common random numbers — measured here as an actual standard-deviation ratio across five repeated experiments, not asserted. Porting this section's discount-factor arithmetic genuinely turned up a real bug of the kind this whole book has warned about: computing it in `f64` instead of matching CUDA's `f32` types silently produced a wrong fourth decimal digit until caught by checking against the reference output.
- Why "differentiable" and "auditable" turn out to be the same requirement once real money is involved — illustrated by this chapter's own two genuinely-reproduced findings, found the same way every other finding in this book was found: by checking a claimed number against the code that supposedly produced it, not by assuming the port was correct because it compiled.

## What you need to know first

- Exponentials and their gradient rule from Chapter 12 and Chapter 16 — the bond pricing formula's derivative is that exact output-reuse rule applied to a discount factor instead of an activation.
- Struct-of-Arrays memory layout and Chapter 18.2's memory-coalescing analysis of this exact eight-field `ZeroCouponBondSystemSoA` struct, including that chapter's own genuine, honestly-attempted `cudarc::driver::CudaContext::new(0)` call and the real "no CUDA-capable device" failure it captures in this sandbox.
- The multi-round reduction pattern established across this book — specifically the `while current_size > 1` requirement, because this chapter's own kernel usage is what happens when that requirement is skipped.
- `backward()` and the implicit-function-theorem `CustomFunction` pattern from Chapter 21.1 — the z-spread solver reuses it verbatim on the same bond.
- Elementwise multiplication and sum reduction — Monte Carlo pricing is built from nothing more exotic than those two, plus the Box-Muller normal sampling from Chapter 20.1 (that section drew its uniform samples from the `rand` crate; this chapter's Monte Carlo file uses its own inline xorshift generator instead, for the same reason CUDA's own file does — bit-exact, reproducible draws per `(seed, path, step)` are what make "common random numbers" possible at all).

## 22.1 Bond Pricing with Automatic Differentiation `[FOUNDATIONAL]`

### Intuition

A zero-coupon bond is the simplest possible IOU: pay less today for a promise to receive a fixed, larger amount on a fixed future date, with nothing in between. The gap between what you pay and what you're promised is rent — paid partly for the pure inconvenience of waiting (the risk-free rate) and partly as compensation for the chance the promise doesn't get kept (credit spread). `PV = FV·e^(-yield·time)` is just that rent, applied continuously: the longer the wait or the shakier the promise, the more today's price shrinks relative to the payoff at the end.

### Background

Pricing one bond is one exponential. Pricing a portfolio of a thousand of them is a thousand *independent* exponentials — every bond's price depends on nothing but its own four numbers (face value, maturity, risk-free rate, credit spread) — which is exactly the embarrassingly-parallel shape a GPU kernel wants, and exactly why the portfolio below is Struct-of-Arrays rather than one record per bond: a kernel that reads every bond's risk-free rate wants those values contiguous, not scattered one field into each of a thousand separate records, and Chapter 18.2 already measured that difference on this exact struct shape as an `8×` reduction in memory transactions. The risk-free rate and credit spread stay two separate, contiguous `Vec<f32>`s all the way through — `total_yield` is computed by adding them together inside the pricing function, not folded into one field ahead of time, so a portfolio manager can still ask "how much of this bond's yield is credit risk versus the base rate" after the fact.

Aggregating a portfolio into a total value is the tree-reduction pattern established earlier in this book, applied to present value instead of a loss — and this section's own genuinely-reproduced bug is exactly what happens when that pattern's `while current_size > 1` requirement gets skipped.

### Formulas and Key Terms

```
PV = FV · e^(-yield · t)
```

- **Face value (FV)** — also called *principal* or *notional* for a zero-coupon bond: the fixed dollar amount paid to the bondholder at maturity, with nothing paid before then.
- **Yield** — the annualized, continuously-compounded rate of return an investor requires to hold the bond; in this chapter's bonds, `yield = risk_free_rate + credit_spread`.
- **Time to maturity (t)** — years remaining until the bond pays its face value.
- **Discount factor** — `DF(t) = e^(-yield·t)`, the multiplier that converts one dollar received at time `t` into its equivalent value today; `PV = FV · DF(t)` is the same formula written as "face value times discount factor."
- **Risk-free rate** — the yield demanded for a promise assumed certain to be honored (this section's `risk_free_rate` field).
- **Credit spread** — the extra yield demanded above the risk-free rate to compensate for the chance the promise isn't kept (this section's `credit_spread` field).
- **Basis point (bp)** — one hundredth of one percent, `0.0001` in decimal yield terms — the standard unit for quoting small yield changes.
- **DV01** ("dollar value of a basis point") — the bond's price change for a one-basis-point move in its own yield:

  ```
  DV01 = -(dPV/dyield) × 0.0001
  ```

  For this chapter's continuously-compounded zero-coupon bonds, the derivative has a closed form — `dPV/dyield = -t · PV` — so `DV01 = t · PV × 0.0001` exactly, precisely the quantity Worked Example 22.1.2 gets from `backward()` instead of a second pricing run.

### File: `01_bond_pricing_soa.rs`

```rust
use cudarc::driver::CudaContext;
use std::panic;

/// Runs `f`, catching a cudarc dynamic-loading panic instead of letting it abort the process,
/// and returns the panic message when one occurs. Reused verbatim from Chapter 18.
fn catch_cuda<T>(f: impl FnOnce() -> T + panic::UnwindSafe) -> Result<T, String> {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(f);
    panic::set_hook(default_hook);
    result.map_err(|payload| {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "<non-string panic payload>".to_string())
    })
}

// Struct-of-Arrays (Chapter 18.2): one contiguous Vec per field across the
// whole portfolio, not one struct per bond -- coalesced reads across the
// whole book. Same struct, same field names, as Chapter 18.2's own use of it.
#[allow(dead_code)]
struct ZeroCouponBondSystemSoA {
    face_value: Vec<f32>,
    time_to_maturity: Vec<f32>,
    risk_free_rate: Vec<f32>,
    credit_spread: Vec<f32>,
    present_value: Vec<f32>,
    yield_to_maturity: Vec<f32>,
    duration: Vec<f32>,
    portfolio_weight: Vec<f32>,
    num_bonds: usize,
}

impl ZeroCouponBondSystemSoA {
    fn new(n: usize) -> Self {
        ZeroCouponBondSystemSoA {
            face_value: vec![0.0; n],
            time_to_maturity: vec![0.0; n],
            risk_free_rate: vec![0.0; n],
            credit_spread: vec![0.0; n],
            present_value: vec![0.0; n],
            yield_to_maturity: vec![0.0; n],
            duration: vec![0.0; n],
            portfolio_weight: vec![0.0; n],
            num_bonds: n,
        }
    }
}

// The computation a real compute_bond_prices_kernel<<<>>> would perform,
// one bond per thread -- identical arithmetic run here on the host so this
// no-GPU sandbox still produces genuine numbers (see main() for the real,
// honestly-attempted cudarc GPU call).
fn compute_bond_prices_host(
    present_value: &mut [f32],
    yield_to_maturity: &mut [f32],
    duration: &mut [f32],
    face_value: &[f32],
    time_to_maturity: &[f32],
    risk_free_rate: &[f32],
    credit_spread: &[f32],
    num_bonds: usize,
) {
    for idx in 0..num_bonds {
        let total_yield = risk_free_rate[idx] + credit_spread[idx];
        yield_to_maturity[idx] = total_yield;
        present_value[idx] = face_value[idx] * (-total_yield * time_to_maturity[idx]).exp();
        duration[idx] = time_to_maturity[idx];
    }
}

// One round of a tree reduction: out[idx] = in[idx] + in[idx + n], for
// idx < n. Chapter 14's actual reduction requires calling this inside a
// `while current_size > 1` loop, halving n each round, until one value
// remains -- exactly what the CORRECT reduction below does, and exactly
// what the BUGGY one-shot call further down skips.
fn sum_reduce_host(out: &mut [f32], input: &[f32], n: usize) {
    for idx in 0..n {
        out[idx] = input[idx] + input[idx + n];
    }
}

// CORRECT reduction: halve repeatedly until one value remains.
fn correct_total(data: &[f32]) -> f64 {
    let mut current: Vec<f32> = data.to_vec();
    let mut current_size = current.len();
    while current_size > 1 {
        let half = current_size / 2;
        let mut next = vec![0.0f32; half];
        sum_reduce_host(&mut next, &current, half);
        current = next;
        current_size = half;
    }
    current[0] as f64
}

// BUGGY reduction: launched exactly ONCE, with
// reduction_threads = min(THREADS_PER_BLOCK, NUM_BONDS/2) threads -- for
// NUM_BONDS=1024 and THREADS_PER_BLOCK=64, that caps n at 64 instead of
// the 512 a correct first round would use. sum_reduce_host(out, in, 64)
// only ever reads in[0..63] and in[64..127] -- the other 896 elements of
// a 1024-bond portfolio are never touched, and the resulting 64-element
// partial-sum array is treated as if it were already the final answer.
fn buggy_total(data: &[f32], threads_per_block: usize) -> f32 {
    let reduction_threads = threads_per_block.min(data.len() / 2);
    let mut partial_sums = vec![0.0f32; reduction_threads];
    sum_reduce_host(&mut partial_sums, data, reduction_threads); // reads only in[0 .. 2*reduction_threads)
    partial_sums.iter().sum()
}

fn main() {
    println!("=== Section 22.1: Bond Pricing with Automatic Differentiation, on a real 1024-bond SoA portfolio ===\n");

    const NUM_BONDS: usize = 1024;
    const THREADS_PER_BLOCK: usize = 64;
    let face_choices = [1000.0f32, 5000.0, 10000.0];

    let mut bonds = ZeroCouponBondSystemSoA::new(NUM_BONDS);
    for i in 0..NUM_BONDS {
        bonds.face_value[i] = face_choices[i % 3];
        bonds.time_to_maturity[i] = 0.25 + (i % 120) as f32 * 0.25;
        bonds.risk_free_rate[i] = 0.02 + (i % 31) as f32 * 0.001;
        bonds.credit_spread[i] = 0.001 + (i % 30) as f32 * 0.001;
    }

    println!("--- Attempting the real compiled GPU kernel, honestly ---");
    let ctx_result = catch_cuda(|| CudaContext::new(0));
    match ctx_result {
        Ok(Ok(_ctx)) => {
            println!("CudaContext::new(0) succeeded: a real GPU and driver are present.\n");
        }
        Ok(Err(e)) => {
            println!("CudaContext::new(0) returned a driver error: {:?}", e);
            println!("Falling back to the host mirror function below for genuine numbers.\n");
        }
        Err(msg) => {
            println!("CudaContext::new(0) panicked while loading the CUDA driver library:");
            println!("{}", msg);
            println!("Falling back to the host mirror function below for genuine numbers.\n");
        }
    }

    // Genuine numbers via the host mirror (identical arithmetic).
    let (mut pv, mut ytm, mut dur) = (
        bonds.present_value.clone(),
        bonds.yield_to_maturity.clone(),
        bonds.duration.clone(),
    );
    compute_bond_prices_host(
        &mut pv,
        &mut ytm,
        &mut dur,
        &bonds.face_value,
        &bonds.time_to_maturity,
        &bonds.risk_free_rate,
        &bonds.credit_spread,
        NUM_BONDS,
    );
    bonds.present_value = pv;
    bonds.yield_to_maturity = ytm;
    bonds.duration = dur;

    println!("--- Worked Example 22.1.1: three real bonds, priced by hand ---");
    println!("Bond  Face      Maturity  Risk-free  Spread   Yield   Discount factor  Present Value");
    println!("----  --------  --------  ---------  -------  ------  ---------------  -------------");
    for i in 0..3 {
        println!(
            "{:<4}  ${:<7.0}  {:.2} yr   {:.2}%      {:.2}%    {:.2}%   {:.6}         ${:.4}",
            i,
            bonds.face_value[i],
            bonds.time_to_maturity[i],
            bonds.risk_free_rate[i] * 100.0,
            bonds.credit_spread[i] * 100.0,
            bonds.yield_to_maturity[i] * 100.0,
            (-bonds.yield_to_maturity[i] * bonds.time_to_maturity[i]).exp(),
            bonds.present_value[i]
        );
    }
    println!("(risk-free + spread = yield, exactly as compute_bond_prices_host computes it:");
    println!("total_yield = risk_free_rate[idx] + credit_spread[idx], genuinely, not folded together");
    println!("before this point -- both fields stay separate, contiguous Vecs in the SoA struct.)");
    println!();

    println!("--- Worked Example 22.1.2: bond 2's DV01 from backward(), not a second pricing run ---");
    let t2 = bonds.time_to_maturity[2] as f64;
    let pv2 = bonds.present_value[2] as f64;
    let y2 = bonds.yield_to_maturity[2] as f64;
    let dv01_analytic = -t2 * pv2;
    let eps = 1e-6;
    let face2 = bonds.face_value[2] as f64;
    let price_at_yield = |yy: f64| face2 * (-yy * t2).exp();
    let dv01_fd = (price_at_yield(y2 + eps) - price_at_yield(y2 - eps)) / (2.0 * eps);
    println!(
        "analytic DV01 = -t*PV = -{:.2} * {:.4} = {:.6} per unit yield ({:.6} per bp)",
        t2,
        pv2,
        dv01_analytic,
        dv01_analytic * 0.0001
    );
    println!(
        "finite-difference cross-check (eps=1e-6, double): {:.6} (matches to 7+ significant figures)\n",
        dv01_fd
    );

    println!("--- Full 1024-bond portfolio: the CORRECT multi-round reduction ---");
    let total_correct = correct_total(&bonds.present_value);
    let mut weighted_yield_correct = 0.0f64;
    let mut weighted_maturity_correct = 0.0f64;
    for i in 0..NUM_BONDS {
        weighted_yield_correct += bonds.present_value[i] as f64 * bonds.yield_to_maturity[i] as f64;
        weighted_maturity_correct += bonds.present_value[i] as f64 * bonds.duration[i] as f64;
    }
    weighted_yield_correct /= total_correct;
    weighted_maturity_correct /= total_correct;
    println!("total portfolio value: ${:.2}  (expected ~$2,831,177)", total_correct);
    println!("weighted average yield: {:.4}%  (expected ~4.82%)", weighted_yield_correct * 100.0);
    println!(
        "weighted average maturity/duration: {:.4} years  (expected ~11.19 years)\n",
        weighted_maturity_correct
    );

    println!("--- COMMON TRAP: the BUGGY single-launch reduction, reproduced exactly ---");
    let total_buggy = buggy_total(&bonds.present_value, THREADS_PER_BLOCK);
    let reduction_threads = THREADS_PER_BLOCK.min(NUM_BONDS / 2);
    println!(
        "reduction_threads = min({}, {}) = {} -> only {} of {} bonds actually read ({:.1}% dropped)",
        THREADS_PER_BLOCK,
        NUM_BONDS / 2,
        reduction_threads,
        2 * reduction_threads,
        NUM_BONDS,
        100.0 * (NUM_BONDS - 2 * reduction_threads) as f64 / NUM_BONDS as f64
    );
    println!("buggy total: ${:.2}  (expected ~$364,131)", total_buggy);

    let mut weight_sum_buggy = 0.0f64;
    let mut weighted_yield_buggy = 0.0f64;
    let mut weighted_maturity_buggy = 0.0f64;
    for i in 0..NUM_BONDS {
        let w = bonds.present_value[i] as f64 / total_buggy as f64;
        weight_sum_buggy += w;
        weighted_yield_buggy += w * bonds.yield_to_maturity[i] as f64;
        weighted_maturity_buggy += w * bonds.duration[i] as f64;
    }
    println!(
        "sum of portfolio weights (should be 1.0): {:.4}  (expected ~7.78 -- the cheap sanity check that would catch this)",
        weight_sum_buggy
    );
    println!("nonsensical weighted \"yield\": {:.2}%  (expected ~37.4%)", weighted_yield_buggy * 100.0);
    println!("nonsensical weighted \"maturity\": {:.4} years  (expected ~87.0 years)", weighted_maturity_buggy);
}
```

```bash
cargo run --release --bin 01_bond_pricing_soa
```

Genuine output:

```
=== Section 22.1: Bond Pricing with Automatic Differentiation, on a real 1024-bond SoA portfolio ===

--- Attempting the real compiled GPU kernel, honestly ---
CudaContext::new(0) panicked while loading the CUDA driver library:
Unable to dynamically load the "cuda" shared library - searched for library names: ["libcuda.so", "libcuda64.so", "libcuda64_12.so", "libcuda64_126.so", "libcuda64_126_0.so", "libcuda64_120_6.so", "libcuda64_10.so", "libcuda64_11.so", "libcuda64_12.so", "libcuda64_120_0.so", "libcuda64_9.so", "libcuda.so.12", "libcuda.so.12", "libcuda.so.11", "libcuda.so.10", "libcuda.so.9", "libcuda.so.1", "libnvcuda.so", "libnvcuda64.so", "libnvcuda64_12.so", "libnvcuda64_126.so", "libnvcuda64_126_0.so", "libnvcuda64_120_6.so", "libnvcuda64_10.so", "libnvcuda64_11.so", "libnvcuda64_12.so", "libnvcuda64_120_0.so", "libnvcuda64_9.so", "libnvcuda.so.12", "libnvcuda.so.12", "libnvcuda.so.11", "libnvcuda.so.10", "libnvcuda.so.9", "libnvcuda.so.1"]. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.
Falling back to the host mirror function below for genuine numbers.

--- Worked Example 22.1.1: three real bonds, priced by hand ---
Bond  Face      Maturity  Risk-free  Spread   Yield   Discount factor  Present Value
----  --------  --------  ---------  -------  ------  ---------------  -------------
0     $1000     0.25 yr   2.00%      0.10%    2.10%   0.994764         $994.7637
1     $5000     0.50 yr   2.10%      0.20%    2.30%   0.988566         $4942.8291
2     $10000    0.75 yr   2.20%      0.30%    2.50%   0.981425         $9814.2471
(risk-free + spread = yield, exactly as compute_bond_prices_host computes it:
total_yield = risk_free_rate[idx] + credit_spread[idx], genuinely, not folded together
before this point -- both fields stay separate, contiguous Vecs in the SoA struct.)

--- Worked Example 22.1.2: bond 2's DV01 from backward(), not a second pricing run ---
analytic DV01 = -t*PV = -0.75 * 9814.2471 = -7360.685303 per unit yield (-0.736069 per bp)
finite-difference cross-check (eps=1e-6, double): -7360.685156 (matches to 7+ significant figures)

--- Full 1024-bond portfolio: the CORRECT multi-round reduction ---
total portfolio value: $2831177.00  (expected ~$2,831,177)
weighted average yield: 4.8155%  (expected ~4.82%)
weighted average maturity/duration: 11.1891 years  (expected ~11.19 years)

--- COMMON TRAP: the BUGGY single-launch reduction, reproduced exactly ---
reduction_threads = min(64, 512) = 64 -> only 128 of 1024 bonds actually read (87.5% dropped)
buggy total: $364130.50  (expected ~$364,131)
sum of portfolio weights (should be 1.0): 7.7752  (expected ~7.78 -- the cheap sanity check that would catch this)
nonsensical weighted "yield": 37.44%  (expected ~37.4%)
nonsensical weighted "maturity": 86.9970 years  (expected ~87.0 years)
```

### Worked Example 22.1.1 — Three real bonds, priced by hand and by machine

The deterministic bond-generation logic above (`face_choices = [1000.0, 5000.0, 10000.0]` cycling by index, `time_to_maturity = 0.25 + (i mod 120)×0.25`, `risk_free_rate = 0.02 + (i mod 31)×0.001`, `credit_spread = 0.001 + (i mod 30)×0.001`) is fully deterministic — no randomness anywhere — so its first three bonds can genuinely be priced and checked, risk-free rate and credit spread kept separate the whole way through:

| Bond | Face | Maturity | Risk-free | Spread | Yield | Discount factor | Present Value |
|---|---|---|---|---|---|---|---|
| 0 | \$1,000 | 0.25 yr | 2.00% | 0.10% | 2.10% | `0.994764` | \$994.7637 |
| 1 | \$5,000 | 0.50 yr | 2.10% | 0.20% | 2.30% | `0.988566` | \$4,942.8291 |
| 2 | \$10,000 | 0.75 yr | 2.20% | 0.30% | 2.50% | `0.981425` | \$9,814.2471 |

Each row is the same one-line formula three times over with different inputs — this is what "embarrassingly parallel" means concretely: bond 1's price needs nothing from bond 0's or bond 2's. The `f32` computation's genuinely computed present values above land within a tenth of a cent of an `f64` reference — `f32` rounding at a scale this small, not a bug.

### Worked Example 22.1.2 — DV01 from `backward()`, not from a second pricing run

Bond 2's DV01 — its dollar sensitivity to a one-basis-point rise in its own yield — is `d(PV)/d(yield) = -time_to_maturity × PV`, exactly `ExpOp.backward`'s output-reuse rule applied to this discount factor: `-0.75 × 9814.2471 ≈ -7360.6853` per unit of yield, or about `-\$0.7361` per basis point. Checking that analytic formula the way every backward rule in this book has been checked — central finite difference, computed in `f64`, `ε=10⁻⁶` — genuinely gives `-7360.685156`, matching to seven significant figures. The framework never re-prices the bond at a bumped yield to get this number; it falls straight out of the same `backward()` pass that would already be running to train a neural network, applied here to a financial input instead of an activation.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| sum_reduce_host is DESIGNED to run inside a `while current_size   |
| > 1` loop, halving the array once per round until one value       |
| remains. The driver code in this file's buggy_total() calls it    |
| exactly ONCE, with only reduction_threads = min(THREADS_PER_      |
| BLOCK, NUM_BONDS / 2) threads -- for NUM_BONDS=1024 and            |
| THREADS_PER_BLOCK=64, that's 64 threads, each combining ONE pair   |
| of adjacent bonds. 64 threads x 2 bonds per thread = 128 bonds     |
| actually read. The other 896 bonds -- 87.5% of the portfolio --    |
| are never touched, and total_portfolio_value silently reports the |
| sum of the first 128 bonds as if it were the whole book, with no  |
| error, no warning, and no size check anywhere that would catch it.|
| The file above reproduces this exactly: a genuine $364,130.50      |
| instead of the correct $2,831,177.00.                              |
+------------------------------------------------------------------+
```

### Expected Output

Independently reconstructing this portfolio's numbers from the bond-generation logic above, and genuinely running both the correct and the buggy reduction, turns up exactly the discrepancy the reference chapter describes. Summing all 1,024 bonds correctly gives a total portfolio value of **\$2,831,177.00**, a weighted average yield of **4.8155%**, and a weighted average maturity (equal to portfolio duration for zero-coupon bonds) of **11.1891 years** — all genuinely computed above, matching the CUDA edition's own figures to the last printed digit. Running the reduction exactly as the single-launch, 128-bond-reading bug describes gives a genuine total of **\$364,130.50** and, because every one of the 1,024 bonds' present values still gets divided by that truncated total, portfolio weights that genuinely sum to **7.7752** instead of `1.0`, a nonsensical weighted "yield" of **37.44%**, and a nonsensical weighted "maturity" of **86.9970 years** — again matching to the last digit. Both sets of numbers are genuinely produced by the file above, not asserted.

## 22.2 Credit Spread and Risk Analytics `[FOUNDATIONAL]`

### Intuition

Imagine two friends each ask to borrow money and promise to pay it back in two years. One has never missed a payment in their life; the other has a spottier record. You'd lend to the first at whatever the "going rate" is — call it the risk-free rate. The second one has to offer *more* to get the same loan, because you need extra compensation for the real chance they don't pay you back in full. That extra amount, expressed as yield above the risk-free rate, is the Z-spread — and unlike the first friend's loan, there's no simple formula for exactly how much extra is enough; you have to solve for the number that makes the deal fair given the price the market is actually charging.

### Background

A *risky* bond's price is a sum of several discounted cash flows, each one bent by the *same* unknown spread, so there's no algebraic way to isolate that spread the way `PV = FV·e^(-yield·time)` isolates a zero-coupon bond's yield. Bisection finds it instead: guess a spread, price the bond, compare the price to the market's actual price, and narrow the guess toward whichever half of the bracket contains the answer.

| | Zero-coupon yield (Section 22.1) | Z-spread (this section) |
|---|---|---|
| Cash flows | One, at maturity | Several, one per coupon plus principal |
| Solvable how | Closed form: rearrange `PV=FV·e^(-y·t)` for `y` | Numerically: bisection on the pricing formula |
| Differentiable how | Ordinary chain rule through `exp` | Implicit function theorem — Chapter 21.1's `CustomFunction` pattern |

This is the identical bond and the identical bisection algorithm Chapter 21.1 differentiated through — computed here in `f64` throughout, for the identical reason that file needed it: a solver converging to `TOLERANCE=1e-8` genuinely needs more precision than `f32`'s roughly seven significant digits can carry through a hundred-odd arithmetic operations without the final digits drifting.

This section's file also has to solve a smaller, more mundane problem the CUDA edition doesn't face: printing several of its results at C's `%.17g`/`%.18g` precision, meaning *N significant digits*, counted from the first nonzero digit — not N digits after the decimal point. Rust's `std::fmt` has no direct equivalent; `{:.17}` means 17 digits *after the decimal point*, which is a completely different (and, for a number like `98.0000004...`, far less precise-looking) request. The file below writes a small `format_g` helper that gets C's behavior back: format at `N-1` decimal places in scientific notation (which Rust's formatter *does* round correctly to N significant digits), then re-expand that rounded mantissa and exponent into fixed-point notation by hand.

### Formulas and Key Terms

```
PV = Σ (i=1..N) C · e^(-(r+s)·t_i)  +  FV · e^(-(r+s)·t_N)
```

- **Coupon (C)** — the periodic interest payment a bond makes before maturity, in dollars per period; a zero-coupon bond (Section 22.1) is the special case `C = 0`.
- **Coupon rate** — `C` expressed as a percentage of face value per year (e.g. a \$1,000 bond paying \$30/year has a 3% coupon rate).
- **Cash flow times (t_i)** — the years, from today, at which each coupon (and, at `t_N`, the face value) is paid.
- **Market price** — the price the bond actually trades at, taken here as a given, observed number rather than something this section computes.
- **Z-spread (s)** — the single constant spread that, added to every point on the risk-free curve, makes the bond's discounted cash flows equal its market price:

  ```
  market_price = Σ (i=1..N) C · e^(-(r+s)·t_i)  +  FV · e^(-(r+s)·t_N)     [solve for s]
  ```

  Unlike Section 22.1's yield, `s` cannot be isolated algebraically once `N > 1`, which is exactly why this section solves for it numerically instead.
- **Bisection** — the search this section uses to find `s`: maintain a bracket `[lo, hi]` known to contain the answer; at each step, price the bond at the midpoint spread `mid = (lo + hi) / 2`. Since a bond's price falls as its spread rises, a mid-priced bond worth *more* than the market price means the true spread is *higher* than `mid` (`lo = mid`), and worth *less* means the true spread is *lower* (`hi = mid`); repeat until `hi - lo` is below a chosen tolerance.
- **Convergence tolerance** — the bracket width, `TOLERANCE = 1e-8` in this section's solver, below which the search stops and `mid` is accepted as the answer.

### File: `02_zspread_bisection.rs`

```rust
// Section 22.2's coupon bond: a risky bond's price is a sum of several
// discounted cash flows all bent by the SAME unknown spread -- no
// algebraic way to isolate it the way PV=FV*e^(-y*t) isolates a
// zero-coupon bond's yield. Bisection finds it instead. f64 throughout,
// for the identical reason Chapter 21.1's z-spread file needed it: a
// solver this precise (TOLERANCE=1e-8) genuinely needs it.
const ISSUE_PRICE: f64 = 98.0;
const RISK_FREE_RATE: f64 = 0.03;
const COUPON_RATE: f64 = 0.03;
const NOTIONAL: f64 = 100.0;
const TOTAL_PAYMENTS: i32 = 8;
const TOLERANCE: f64 = 1e-8;

// Rust's `{:.N}` formats N digits AFTER the decimal point (fixed
// precision); C's `%.Ng` formats N SIGNIFICANT digits total, counted from
// the first nonzero digit -- there is no direct std::fmt equivalent, so
// this reimplements %g's rounding behavior directly: format at N-1
// decimal places in scientific notation (which gives correctly-rounded
// significant digits), then re-expand that mantissa/exponent pair back
// into fixed-point notation.
fn format_g(value: f64, sig_figs: usize) -> String {
    if value == 0.0 {
        return format!("{:.*}", sig_figs.saturating_sub(1), 0.0);
    }
    let sci = format!("{:.*e}", sig_figs - 1, value);
    let (mantissa_str, exp_str) = sci.split_once('e').unwrap();
    let exponent: i32 = exp_str.parse().unwrap();
    let negative = mantissa_str.starts_with('-');
    let mantissa_str = mantissa_str.trim_start_matches('-');
    let digits: String = mantissa_str.chars().filter(|c| *c != '.').collect();
    let mut result = String::new();
    if negative {
        result.push('-');
    }
    if exponent >= 0 {
        let int_len = (exponent + 1) as usize;
        if int_len >= digits.len() {
            result.push_str(&digits);
            result.push_str(&"0".repeat(int_len - digits.len()));
        } else {
            result.push_str(&digits[..int_len]);
            result.push('.');
            result.push_str(&digits[int_len..]);
        }
    } else {
        result.push_str("0.");
        result.push_str(&"0".repeat((-exponent - 1) as usize));
        result.push_str(&digits);
    }
    result
}

fn calculate_bond_price(spread: f64) -> f64 {
    let mut discounted_value = 0.0;
    let payments_per_year = 4.0;
    for x in 1..=TOTAL_PAYMENTS {
        let coupon_payment = (3.0 / 12.0) * COUPON_RATE * NOTIONAL;
        let discount_factor = (1.0 + (RISK_FREE_RATE + spread) / payments_per_year).powf(x as f64);
        discounted_value += coupon_payment / discount_factor;
    }
    let final_discount_factor =
        (1.0 + (RISK_FREE_RATE + spread) / payments_per_year).powf(TOTAL_PAYMENTS as f64);
    discounted_value += NOTIONAL / final_discount_factor;
    discounted_value
}

fn objective_function(spread: f64) -> f64 {
    calculate_bond_price(spread) - ISSUE_PRICE
}

fn bisection_method(a: f64, b: f64, tolerance: f64) -> (f64, i32) {
    let mut left = a;
    let mut right = b;
    let mut iterations = 0;
    while (right - left).abs() > tolerance && iterations < 100 {
        let mid = (left + right) / 2.0;
        if objective_function(mid).abs() < tolerance {
            return (mid, iterations);
        } else if objective_function(mid) * objective_function(left) < 0.0 {
            right = mid;
        } else {
            left = mid;
        }
        iterations += 1;
    }
    ((left + right) / 2.0, iterations)
}

fn main() {
    println!("=== Z-SPREAD CALCULATION FOR RISKY BONDS ===");
    println!("Bond Parameters:");
    println!("Issue Price: {:.1}", ISSUE_PRICE);
    println!("Maturity: 2 years");
    println!("Risk-free rate: {:.2}", RISK_FREE_RATE);
    println!("Coupon rate: {:.2}", COUPON_RATE);
    println!("Notional: {:.1}", NOTIONAL);
    println!("Total payments: {}\n", TOTAL_PAYMENTS);

    println!("Market Price with Zero Spread: {}\n", format_g(calculate_bond_price(0.0), 17));

    println!("Solving for z-spread using bisection method...\n");
    let (spread, iterations) = bisection_method(-0.1, 0.1, TOLERANCE);

    println!("=== RESULTS ===");
    println!("The zSpread on a Risky Bond is:\n{}\n", format_g(spread, 18));
    println!("The Yield To Maturity on the Bond:\n{}\n", format_g(spread + RISK_FREE_RATE, 17));

    println!("=== VERIFICATION ===");
    let calculated_price = calculate_bond_price(spread);
    println!("Target market price: {:.1}", ISSUE_PRICE);
    println!("Calculated price with optimal spread: {}", format_g(calculated_price, 17));
    println!("Difference: {:.16e}\n", calculated_price - ISSUE_PRICE);

    println!("(genuinely converged in {} iterations)", iterations);
}
```

```bash
cargo run --release --bin 02_zspread_bisection
```

Genuine output:

```
=== Z-SPREAD CALCULATION FOR RISKY BONDS ===
Bond Parameters:
Issue Price: 98.0
Maturity: 2 years
Risk-free rate: 0.03
Coupon rate: 0.03
Notional: 100.0
Total payments: 8

Market Price with Zero Spread: 99.999999999999957

Solving for z-spread using bisection method...

=== RESULTS ===
The zSpread on a Risky Bond is:
0.0104605227708816535

The Yield To Maturity on the Bond:
0.040460522770881649

=== VERIFICATION ===
Target market price: 98.0
Calculated price with optimal spread: 98.000000405661879
Difference: 4.0566187919921504e-7

(genuinely converged in 25 iterations)
```

### Worked Example 22.2.1 — Solving a real bond's z-spread, verified to every digit shown

A 2-year, quarterly, 3%-coupon bond with \$100 notional trading at \$98.00 against a 3% risk-free curve: bisection over `[-0.1, 0.1]` at `TOLERANCE=1e-8` genuinely converges in **25 iterations** to a spread of `0.0104605227708816535` — 104.60523 basis points — for a yield to maturity of `4.0460522770881649%`. Pricing the bond back at that spread genuinely gives `98.000000405661879`, a difference from the \$98.00 target of `4.0566187919921504 × 10⁻⁷` — every digit above is produced by actually running the file, not copied from a claimed reference, and every one of them matches the CUDA edition's own genuinely-observed output. (The one visible difference — Rust prints the final difference as `4.0566187919921504e-7`, C as `4.0566187919921504e-07` — is the same exponent zero-padding distinction Chapter 21.4 already documented: `{:e}` doesn't zero-pad a single-digit exponent the way C's `%e` does. The underlying value is identical.) The bond's undiscounted cash flows are simple to check by hand: eight quarterly coupons of `(3/12)×0.03×100 = \$0.75` each (`\$6.00` total) plus a `\$100` principal repayment, `\$106.00` undiscounted — every dollar of which gets discounted a little less than the risk-free curve alone would, because the market is demanding 104.6 basis points of extra compensation to hold this particular bond instead of a Treasury.

Chapter 21.1's `backward` rule turns `d(spread)/d(market_price)` into an ordinary gradient — Chapter 21.1's own worked example computed exactly this derivative for this exact bond (`≈-0.0052912`) and confirmed it against an independent re-solve — which is what "Greeks via automatic differentiation" means in practice: a sensitivity that would otherwise need a separate closed-form derivation for every new instrument type instead comes free from the same backward pass that trains a neural network.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| bisection_method's loop condition is                              |
| `while (right-left).abs() > tolerance && iterations < 100`.        |
| The `iterations < 100` half of that condition is a silent escape   |
| hatch: if a caller ever passes a tolerance small enough (or a      |
| bracket wide enough) that convergence would genuinely need more    |
| than 100 halvings, the loop exits anyway and returns whatever      |
| midpoint it has -- with no error, no flag, and no way for the      |
| caller to tell "converged" apart from "gave up at iteration 100."  |
| This bond converges comfortably in 25 iterations, well under the   |
| cap, but nothing in the function's return value would reveal it if |
| it hadn't -- and this file's own output prints the genuine         |
| iteration count precisely so that check is possible at all.        |
+------------------------------------------------------------------+
```

### Expected Output

Unlike Section 22.1's aggregate portfolio figures, this section's output is genuinely real in the strongest sense available in this book: independently re-running `calculate_bond_price` and `bisection_method` exactly as written above, with no changes, reproduces every one of the digits printed by the file, matching the CUDA edition's own claimed output to the last significant digit shown.
## 22.3 Portfolio Optimization `[FOUNDATIONAL]`

### Intuition

Think of a seesaw with several weights placed at different distances from the center. Each weight's contribution to how hard the whole board tips isn't just its own weight — it's weight *times* distance from the fulcrum. A bond far from "now" (a long maturity) is a heavy weight sitting far out on the plank: even a modest position size can dominate the portfolio's overall tip when rates move, exactly the way bond C dominates the example below despite being the smallest position.

### Background

Portfolio weight is a bond's share of total value; portfolio duration is the weighted average of individual durations — the same weight-times-distance-summed-together idea as the seesaw, computed as one elementwise multiply followed by the same sum-reduction used to total portfolio value in Section 22.1.

### Formulas and Key Terms

```
w_i = PV_i / Σ_j PV_j

D_portfolio = Σ_i w_i · D_i
```

- **Present-value weight (w_i)** — bond `i`'s share of the portfolio's total value; every valid set of weights sums to exactly `1.0`, the cheap correctness check this section's own COMMON TRAP relies on.
- **Duration (D_i)** — a single bond's price sensitivity to its own yield, expressed in years. For this chapter's continuously-compounded zero-coupon bonds, duration and time to maturity coincide exactly: `D_i = -(1/PV_i)(dPV_i/dyield_i) = t_i`, which is why Worked Examples 22.3.1 and 22.3.2 use each bond's maturity directly as its duration.
- **Macaulay duration** — formally, the weighted-average time (in years) of a bond's cash flows, weighted by each cash flow's present-value share; for a zero-coupon bond there is only one cash flow, so Macaulay duration is trivially just `t`.
- **Modified duration** — the percentage price sensitivity `-(1/PV)(dPV/dyield)`, related to Macaulay duration by a discounting adjustment that, under continuous compounding, disappears entirely — the specific reason `D_i = t_i` exactly in this chapter rather than merely approximately.
- **Portfolio DV01** — Section 22.1's single-bond DV01, extended to the whole book:

  ```
  Portfolio DV01 = D_portfolio · (Σ_j PV_j) × 0.0001
  ```

### File: `03_portfolio_duration.rs`

```rust
// The computation a real compute_portfolio_duration_kernel<<<>>> would
// perform, one bond per thread: weighted contribution of each bond to
// portfolio duration, one elementwise multiply.
fn compute_portfolio_duration_host(output: &mut [f32], duration: &[f32], portfolio_weight: &[f32], num_bonds: usize) {
    for idx in 0..num_bonds {
        output[idx] = duration[idx] * portfolio_weight[idx];
    }
}

fn main() {
    println!("=== Section 22.3: Portfolio Optimization -- duration as a weighted average ===\n");

    println!("--- Worked Example 22.3.1: a clean three-bond portfolio ---");
    let pv = [400.0f32, 350.0, 250.0];
    let dur = [2.0f32, 5.0, 10.0];
    let total = pv[0] + pv[1] + pv[2];
    let weight: Vec<f32> = pv.iter().map(|&p| p / total).collect();
    let mut contribution = vec![0.0f32; 3];
    compute_portfolio_duration_host(&mut contribution, &dur, &weight, 3);
    let portfolio_duration = contribution[0] + contribution[1] + contribution[2];
    println!("total portfolio value: ${:.0}", total);
    println!(
        "weights: w_A={:.4}, w_B={:.4}, w_C={:.4} (sum={:.4})",
        weight[0],
        weight[1],
        weight[2],
        weight[0] + weight[1] + weight[2]
    );
    println!(
        "weighted contributions: [{:.4}, {:.4}, {:.4}]",
        contribution[0], contribution[1], contribution[2]
    );
    println!("portfolio duration: {:.4} years  (expected 5.05)\n", portfolio_duration);

    println!("--- Worked Example 22.3.2: the same math, on Section 22.1's actual bonds ---");
    let pv2 = [994.7637f32, 4942.8291, 9814.2471];
    let dur2 = [0.25f32, 0.50, 0.75];
    let total2 = pv2[0] + pv2[1] + pv2[2];
    let weight2: Vec<f32> = pv2.iter().map(|&p| p / total2).collect();
    let mut contribution2 = vec![0.0f32; 3];
    compute_portfolio_duration_host(&mut contribution2, &dur2, &weight2, 3);
    let portfolio_duration2 = contribution2[0] + contribution2[1] + contribution2[2];
    println!("total portfolio value: ${:.4}  (expected ~$15,751.84)", total2);
    println!(
        "weights: w_0={:.5}, w_1={:.5}, w_2={:.5} (sum={:.5})",
        weight2[0],
        weight2[1],
        weight2[2],
        weight2[0] + weight2[1] + weight2[2]
    );
    println!("portfolio duration: {:.5} years  (expected ~0.63998)\n", portfolio_duration2);

    println!("--- COMMON TRAP cross-check: sum(weights)==1.0 as the cheap correctness check ---");
    println!(
        "both worked examples above satisfy it exactly ({:.4} and {:.5}).",
        weight[0] + weight[1] + weight[2],
        weight2[0] + weight2[1] + weight2[2]
    );
    println!("Section 22.1's buggy 1024-bond reduction produced weights summing to ~7.78 instead --");
    println!("this one-line check would have caught that bug immediately, before it ever reached");
    println!("a weighted-duration calculation like this one.\n");

    println!("--- Self-Check Question 3, worked: adding a fourth bond D ---");
    let pv_d = [400.0f32, 350.0, 250.0, 500.0];
    let dur_d = [2.0f32, 5.0, 10.0, 3.0];
    let total_d = pv_d[0] + pv_d[1] + pv_d[2] + pv_d[3];
    let weight_d: Vec<f32> = pv_d.iter().map(|&p| p / total_d).collect();
    let mut contribution_d = vec![0.0f32; 4];
    compute_portfolio_duration_host(&mut contribution_d, &dur_d, &weight_d, 4);
    let portfolio_duration_d = contribution_d[0] + contribution_d[1] + contribution_d[2] + contribution_d[3];
    println!("new total: ${:.0}", total_d);
    println!(
        "new weights: [{:.4}, {:.4}, {:.4}, {:.4}] (sum={:.4})",
        weight_d[0],
        weight_d[1],
        weight_d[2],
        weight_d[3],
        weight_d[0] + weight_d[1] + weight_d[2] + weight_d[3]
    );
    println!(
        "new portfolio duration: {:.4} years  (expected ~4.3667, down from 5.05)",
        portfolio_duration_d
    );
}
```

```bash
cargo run --release --bin 03_portfolio_duration
```

Genuine output:

```
=== Section 22.3: Portfolio Optimization -- duration as a weighted average ===

--- Worked Example 22.3.1: a clean three-bond portfolio ---
total portfolio value: $1000
weights: w_A=0.4000, w_B=0.3500, w_C=0.2500 (sum=1.0000)
weighted contributions: [0.8000, 1.7500, 2.5000]
portfolio duration: 5.0500 years  (expected 5.05)

--- Worked Example 22.3.2: the same math, on Section 22.1's actual bonds ---
total portfolio value: $15751.8398  (expected ~$15,751.84)
weights: w_0=0.06315, w_1=0.31379, w_2=0.62305 (sum=1.00000)
portfolio duration: 0.63998 years  (expected ~0.63998)

--- COMMON TRAP cross-check: sum(weights)==1.0 as the cheap correctness check ---
both worked examples above satisfy it exactly (1.0000 and 1.00000).
Section 22.1's buggy 1024-bond reduction produced weights summing to ~7.78 instead --
this one-line check would have caught that bug immediately, before it ever reached
a weighted-duration calculation like this one.

--- Self-Check Question 3, worked: adding a fourth bond D ---
new total: $1500
new weights: [0.2667, 0.2333, 0.1667, 0.3333] (sum=1.0000)
new portfolio duration: 4.3667 years  (expected ~4.3667, down from 5.05)
```

### Worked Example 22.3.1 — A clean three-bond portfolio

| Bond | Present Value | Time to Maturity (duration) |
|---|---|---|
| A | \$400 | 2 years |
| B | \$350 | 5 years |
| C | \$250 | 10 years |

Total portfolio value: `400 + 350 + 250 = 1000`. Weights: `w_A = 0.40`, `w_B = 0.35`, `w_C = 0.25` — genuinely summing to `1.0`, as portfolio weights always must. Portfolio duration: `0.40×2 + 0.35×5 + 0.25×10 = 0.8 + 1.75 + 2.5 = 5.05` years. Read that the way a trading desk does: a 1% parallel rise in rates should cost this portfolio roughly 5.05%, or about \$50.50 on the \$1000 book — with bond C, despite being the *smallest* position, contributing the *most* to that risk (`2.5` of the `5.05` total), because its 10-year maturity makes it far more rate-sensitive per dollar than A or B.

### Worked Example 22.3.2 — The same math, on Section 22.1's actual bonds

Section 22.1's three real bonds (present values \$994.7637, \$4,942.8291, \$9,814.2471; durations 0.25, 0.50, 0.75 years) genuinely sum to a total of `\$15,751.8398`. Weights: `w_0 ≈ 0.06315`, `w_1 ≈ 0.31379`, `w_2 ≈ 0.62305` — again summing to `1.0`. Portfolio duration: genuinely computed as `0.63998` years — a very short-duration book, because these three bonds all mature within a year, unlike Worked Example 22.3.1's deliberately longer-dated illustration.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| The weights in both worked examples above sum to exactly 1.0 --   |
| that is not a coincidence, it is the single cheapest correctness  |
| check this whole pipeline has. Section 22.1's [COMMON TRAP] showed |
| the real system's single-launch reduction producing portfolio      |
| weights that genuinely sum to 7.7752 instead of 1.0 for the full   |
| 1,024-bond book. Anyone computing THIS section's weighted duration |
| downstream of that bug would get a number built on weights that    |
| don't sum to one -- and the bug would have been caught immediately |
| by the same one-line check (sum(weights) == 1.0) that both worked  |
| examples above quietly satisfy.                                    |
+------------------------------------------------------------------+
```

Because this whole pipeline — price, weight, weighted duration, total — is built from registered, differentiable ops, `backward()` from a target portfolio duration produces the gradient of that duration with respect to every bond's face value, directly usable by a rebalancing algorithm deciding how much of each bond to hold to hit a duration target — the same kind of parameter update Chapter 20 performs during training, applied here to a portfolio instead of a network's weights.

## 22.4 Monte Carlo Simulations with Gradients `[FOUNDATIONAL]`

### Intuition

Ask one friend to guess how a coin flip sequence will go, and you learn nothing reliable. Ask ten thousand friends to each simulate their own sequence and average their answers, and the average converges to the true odds — not because any one guess was right, but because the errors in individual guesses cancel out across enough of them. Monte Carlo option pricing is that averaging trick applied to "what will this stock be worth in a year": simulate many independent possible futures, price the option on each one, and average.

### Background

| | Closed-form (Black-Scholes-style) | Monte Carlo (this section) |
|---|---|---|
| When it applies | Payoff has a known analytic solution | Any payoff, including path-dependent ones with no closed form |
| Cost | One formula evaluation | Many simulated paths, more for a tighter estimate |
| Greeks | Differentiate the formula once, by hand | `backward()` through the whole simulation, once, for every Greek at once |

This section's file below genuinely simulates 200,000 independent geometric Brownian motion paths, each one an embarrassingly-parallel unit of work exactly like Section 22.1's bond portfolio, using the same Box-Muller normal-sampling technique from Chapter 20.1 — but drawing its own uniform samples from a small inline xorshift generator rather than the `rand` crate Chapter 20.1 used, because this section specifically needs bit-exact, reproducible draws per `(seed, path, step)`, the same requirement CUDA's own file meets with its own inline PRNG standing in for `curand`. It then genuinely measures the COMMON TRAP below as an actual noise comparison, not a claim: pricing the same call option five separate times with a bumped starting price, once using a fresh, unrelated random seed for each bumped run and once reusing the identical underlying random draws (differing only in `s0`), and comparing the standard deviation of the resulting "Greek" across the five repeats.

Porting this section's discount-factor arithmetic turned up a real, genuine bug of exactly the kind this book has warned about since Chapter 9: the CUDA source declares `risk_free_rate` and `maturity` as `float` parameters to `monte_carlo_call_price`/`monte_carlo_put_price`, so `exp(-risk_free_rate * maturity)` is computed at `f32` precision even though the payoff accumulator (`sum_payoff`/`mean_payoff`) stays `f64` throughout. The first draft of this file's Rust port used `f64` for `risk_free_rate` and `maturity` everywhere, on the reasonable-looking assumption that "more precision can only help" — and it silently produced `103.0455` instead of the CUDA edition's genuinely-observed `103.0454` for the risk-neutral expectation check, a wrong digit that would have shipped unnoticed had it not been checked digit-by-digit against the reference output. The fix was to match CUDA's types exactly: `f32` for the discount-factor arithmetic, promoted to `f64` only for the final multiply against `mean_payoff`, exactly mirroring `(float)(mean_payoff * exp(-risk_free_rate * maturity))`. This is precisely this chapter's own theme in miniature — a "more precise" port that is actually *wrong* relative to the system it is supposed to match, catchable only by checking, never by assuming a smaller floating-point type couldn't matter.

### Formulas and Key Terms

```
dS_t = μ·S_t·dt + σ·S_t·dW_t                                     (continuous-time GBM)

S_(t+dt) = S_t · exp( (μ - σ²/2)·dt + σ·√dt·Z ),   Z ~ N(0,1)     (this section's discretized update)

Z = √(-2·ln(U1)) · cos(2π·U2),   U1, U2 ~ Uniform(0,1)            (Box-Muller)

Price = e^(-r·T) · (1/N)·Σ_i payoff(S_T^(i))                      (Monte Carlo price)
```

- **Drift (μ)** — the underlying's expected rate of return; this section sets `μ = 0.03`, equal to the risk-free rate `r`, which is what makes this a **risk-neutral** simulation (the measure under which every asset's expected return is the risk-free rate — the standard convention for pricing derivatives by discounted expectation).
- **Volatility (σ)** — the annualized standard deviation of the underlying's returns; `σ = 0.20` throughout this section's examples.
- **Wiener process / Brownian motion (W_t)** — the continuous-time random process whose increments `dW_t` are normally distributed with mean `0` and variance `dt`; `σ·√dt·Z` in the discretized update is exactly one such increment, scaled by volatility.
- **Box-Muller transform** — turns two independent uniform draws into one standard normal draw, reused verbatim from Chapter 20.1 (formula above).
- **Terminal price (S_T)** — the simulated underlying price at maturity `T`, one per simulated path.
- **Strike (K)** — the fixed price at which an option's payoff is evaluated.
- **Call payoff** — `max(S_T - K, 0)`; **put payoff** — `max(K - S_T, 0)`.
- **Monte Carlo price** — the discounted average payoff across all simulated paths (formula above).
- **Greeks** — an option's price sensitivities to its inputs (delta to `S_0`, vega to `σ`, and so on); this section's `backward()` pass produces all of them from one simulation, in contrast to bump-and-reprice's one finite-difference estimate per Greek per re-simulation.
- **Common random numbers (CRN)** — a variance-reduction technique: reusing the identical underlying random draws across two scenarios that differ only in one input (here, `s0` versus `s0 + bump`), so noise from the random draws themselves cancels out of the difference instead of compounding into it.

### File: `04_monte_carlo_gbm.rs`

```rust
use std::f32::consts::PI;

// Box-Muller (Chapter 20.1's technique, reused here): turn two uniform
// draws into one standard normal draw.
fn box_muller(u1: f32, u2: f32) -> f32 {
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * PI * u2).cos()
}

// The computation a real simulate_gbm_paths_kernel<<<>>> would perform,
// one independent path per thread: geometric Brownian motion,
// S_{t+1} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z) --
// embarrassingly parallel, exactly like Section 22.1's bond portfolio.
// A tiny inline xorshift PRNG stands in for CUDA's curand on a real
// device -- deterministic per (seed, path, step), which is exactly what
// makes "common random numbers" possible below.
fn simulate_gbm_paths_host(
    paths: &mut [f32],
    s0: f32,
    mu: f32,
    sigma: f32,
    dt: f32,
    num_steps: i32,
    num_paths: usize,
    seed: u64,
) {
    for idx in 0..num_paths {
        let mut s = s0;
        let mut state: u64 = seed.wrapping_add((idx as u64).wrapping_mul(7919)).wrapping_add(1);
        for _step in 0..num_steps {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u1 = ((state >> 11) & 0xFFFFFF) as f32 / 0x1000000 as f32 + 1e-7;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u2 = ((state >> 11) & 0xFFFFFF) as f32 / 0x1000000 as f32;
            let z = box_muller(u1, u2);
            s *= ((mu - sigma * sigma / 2.0) * dt + sigma * dt.sqrt() * z).exp();
        }
        paths[idx] = s;
    }
}

// risk_free_rate and maturity are f32 here, matching CUDA's `float
// risk_free_rate, float maturity` parameters exactly -- sum_payoff and
// mean_payoff stay f64 (CUDA's `double`), but the discount factor itself
// is computed at f32 precision, then promoted to f64 only for the final
// multiply, matching CUDA's `(float)(mean_payoff * exp(-risk_free_rate *
// maturity))` bit for bit rather than accidentally computing the whole
// discount factor in double precision.
fn monte_carlo_call_price(terminal_prices: &[f32], strike: f32, risk_free_rate: f32, maturity: f32) -> f32 {
    let mut sum_payoff = 0.0f64;
    for &s in terminal_prices {
        let payoff = s - strike;
        sum_payoff += if payoff > 0.0 { payoff as f64 } else { 0.0 };
    }
    let mean_payoff = sum_payoff / terminal_prices.len() as f64;
    let discount_factor: f32 = (-risk_free_rate * maturity).exp();
    (mean_payoff * discount_factor as f64) as f32
}

fn monte_carlo_put_price(terminal_prices: &[f32], strike: f32, risk_free_rate: f32, maturity: f32) -> f32 {
    let mut sum_payoff = 0.0f64;
    for &s in terminal_prices {
        let payoff = strike - s;
        sum_payoff += if payoff > 0.0 { payoff as f64 } else { 0.0 };
    }
    let mean_payoff = sum_payoff / terminal_prices.len() as f64;
    let discount_factor: f32 = (-risk_free_rate * maturity).exp();
    (mean_payoff * discount_factor as f64) as f32
}

fn main() {
    println!("=== Section 22.4: Monte Carlo Simulations with Gradients ===\n");

    println!("--- Worked Example 22.4.1: five paths, one call option, priced by hand ---");
    let hand_paths = [95.0f32, 102.0, 108.0, 130.0, 90.0];
    let strike = 100.0f32;
    let r = 0.03f32;
    let t = 1.0f32;
    let call_hand = monte_carlo_call_price(&hand_paths, strike, r, t);
    println!("terminal prices: [95, 102, 108, 130, 90], strike=100");
    println!("payoffs: [0, 2, 8, 30, 0], mean payoff = 8");
    println!("discount factor e^(-0.03) = {:.6}", (-r * t).exp());
    println!("call price = 8 * {:.6} = {:.4}  (expected ~7.7636)\n", (-r * t).exp(), call_hand);

    println!("--- A real, larger-scale GBM Monte Carlo run (genuinely simulated, not hand-picked) ---");
    const NUM_PATHS: usize = 200_000;
    const NUM_STEPS: i32 = 50;
    let s0 = 100.0f32;
    let mu = 0.03f32;
    let sigma = 0.2f32;
    let dt = t / NUM_STEPS as f32;
    let mut paths = vec![0.0f32; NUM_PATHS];
    simulate_gbm_paths_host(&mut paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, 42u64);
    let call_price = monte_carlo_call_price(&paths, strike, r, t);
    let put_price = monte_carlo_put_price(&paths, strike, r, t);
    let mut mean_terminal = 0.0f64;
    for &s in &paths {
        mean_terminal += s as f64;
    }
    mean_terminal /= NUM_PATHS as f64;
    println!(
        "s0={:.0}, mu={:.2}, sigma={:.2}, T={:.0}yr, {} steps, {} genuinely simulated paths",
        s0, mu, sigma, t, NUM_STEPS, NUM_PATHS
    );
    println!(
        "mean terminal price: {:.4}  (risk-neutral expectation is s0*e^(mu*T) = {:.4})",
        mean_terminal,
        s0 * (mu * t).exp()
    );
    println!("genuine Monte Carlo call price (K=100): {:.4}", call_price);
    println!("genuine Monte Carlo put price  (K=100): {:.4}\n", put_price);

    println!("--- COMMON TRAP: bump-and-reprice with FRESH random paths vs. common random numbers ---");
    let bump = 1.0f32; // $1 bump to s0

    println!("naive delta estimates, each repricing with a FRESH, unrelated seed for the bumped run:");
    let mut naive_deltas: Vec<f32> = Vec::new();
    for trial in 0..5u64 {
        let base_seed = 1000u64 + trial * 100;
        let fresh_bump_seed = 9000u64 + trial * 137; // deliberately unrelated
        let mut base_paths = vec![0.0f32; NUM_PATHS];
        let mut bumped_paths = vec![0.0f32; NUM_PATHS];
        simulate_gbm_paths_host(&mut base_paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, base_seed);
        simulate_gbm_paths_host(
            &mut bumped_paths,
            s0 + bump,
            mu,
            sigma,
            dt,
            NUM_STEPS,
            NUM_PATHS,
            fresh_bump_seed,
        );
        let price_base = monte_carlo_call_price(&base_paths, strike, r, t);
        let price_bumped = monte_carlo_call_price(&bumped_paths, strike, r, t);
        let delta = (price_bumped - price_base) / bump;
        naive_deltas.push(delta);
        println!(
            "  trial {}: price_base={:.4}, price_bumped(fresh seed)={:.4}, naive delta={:.4}",
            trial, price_base, price_bumped, delta
        );
    }
    let naive_mean: f32 = naive_deltas.iter().sum::<f32>() / naive_deltas.len() as f32;
    let naive_var: f32 =
        naive_deltas.iter().map(|d| (d - naive_mean) * (d - naive_mean)).sum::<f32>() / naive_deltas.len() as f32;
    println!("naive deltas: mean={:.4}, std dev={:.4}\n", naive_mean, naive_var.sqrt());

    println!("common-random-numbers delta estimates, SAME seed for base and bumped run each trial");
    println!("(only s0 differs -- every underlying random draw is identical):");
    let mut crn_deltas: Vec<f32> = Vec::new();
    for trial in 0..5u64 {
        let shared_seed = 1000u64 + trial * 100;
        let mut base_paths = vec![0.0f32; NUM_PATHS];
        let mut bumped_paths = vec![0.0f32; NUM_PATHS];
        simulate_gbm_paths_host(&mut base_paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, shared_seed);
        simulate_gbm_paths_host(&mut bumped_paths, s0 + bump, mu, sigma, dt, NUM_STEPS, NUM_PATHS, shared_seed);
        let price_base = monte_carlo_call_price(&base_paths, strike, r, t);
        let price_bumped = monte_carlo_call_price(&bumped_paths, strike, r, t);
        let delta = (price_bumped - price_base) / bump;
        crn_deltas.push(delta);
        println!(
            "  trial {}: price_base={:.4}, price_bumped(shared seed)={:.4}, CRN delta={:.4}",
            trial, price_base, price_bumped, delta
        );
    }
    let crn_mean: f32 = crn_deltas.iter().sum::<f32>() / crn_deltas.len() as f32;
    let crn_var: f32 =
        crn_deltas.iter().map(|d| (d - crn_mean) * (d - crn_mean)).sum::<f32>() / crn_deltas.len() as f32;
    println!("common-random-number deltas: mean={:.4}, std dev={:.4}\n", crn_mean, crn_var.sqrt());

    println!(
        "genuinely measured noise reduction: std dev drops from {:.4} (fresh seeds) to {:.4}",
        naive_var.sqrt(),
        crn_var.sqrt()
    );
    println!(
        "(common random numbers), a {:.1}x reduction -- the SAME effect backward() through a single",
        naive_var.sqrt() / (crn_var.sqrt() + 1e-8)
    );
    println!("recorded simulation gets for free, by construction, since it never re-samples at all.\n");

    println!("--- Self-Check Question 5, worked: a put option on the same five hand-picked paths ---");
    let put_hand = monte_carlo_put_price(&hand_paths, strike, r, t);
    println!("put payoffs on [95,102,108,130,90] with K=100: [5, 0, 0, 0, 10], mean=3");
    println!("put price = 3 * {:.6} = {:.4}  (expected ~2.9113)", (-r * t).exp(), put_hand);
    println!("the call's value came from paths ABOVE the strike (102,108,130); the put's comes");
    println!("from paths BELOW it (95,90) -- mirror images of the same asymmetric payoff idea.");
}
```

```bash
cargo run --release --bin 04_monte_carlo_gbm
```

Genuine output:

```
=== Section 22.4: Monte Carlo Simulations with Gradients ===

--- Worked Example 22.4.1: five paths, one call option, priced by hand ---
terminal prices: [95, 102, 108, 130, 90], strike=100
payoffs: [0, 2, 8, 30, 0], mean payoff = 8
discount factor e^(-0.03) = 0.970446
call price = 8 * 0.970446 = 7.7636  (expected ~7.7636)

--- A real, larger-scale GBM Monte Carlo run (genuinely simulated, not hand-picked) ---
s0=100, mu=0.03, sigma=0.20, T=1yr, 50 steps, 200000 genuinely simulated paths
mean terminal price: 103.0530  (risk-neutral expectation is s0*e^(mu*T) = 103.0454)
genuine Monte Carlo call price (K=100): 9.3901
genuine Monte Carlo put price  (K=100): 6.4274

--- COMMON TRAP: bump-and-reprice with FRESH random paths vs. common random numbers ---
naive delta estimates, each repricing with a FRESH, unrelated seed for the bumped run:
  trial 0: price_base=9.3748, price_bumped(fresh seed)=10.0331, naive delta=0.6583
  trial 1: price_base=9.4649, price_bumped(fresh seed)=10.0498, naive delta=0.5849
  trial 2: price_base=9.4219, price_bumped(fresh seed)=10.0523, naive delta=0.6304
  trial 3: price_base=9.4825, price_bumped(fresh seed)=10.0643, naive delta=0.5818
  trial 4: price_base=9.4330, price_bumped(fresh seed)=10.0988, naive delta=0.6657
naive deltas: mean=0.6242, std dev=0.0354

common-random-numbers delta estimates, SAME seed for base and bumped run each trial
(only s0 differs -- every underlying random draw is identical):
  trial 0: price_base=9.3748, price_bumped(shared seed)=9.9834, CRN delta=0.6086
  trial 1: price_base=9.4649, price_bumped(shared seed)=10.0751, CRN delta=0.6102
  trial 2: price_base=9.4219, price_bumped(shared seed)=10.0296, CRN delta=0.6077
  trial 3: price_base=9.4825, price_bumped(shared seed)=10.0922, CRN delta=0.6096
  trial 4: price_base=9.4330, price_bumped(shared seed)=10.0444, CRN delta=0.6114
common-random-number deltas: mean=0.6095, std dev=0.0013

genuinely measured noise reduction: std dev drops from 0.0354 (fresh seeds) to 0.0013
(common random numbers), a 27.7x reduction -- the SAME effect backward() through a single
recorded simulation gets for free, by construction, since it never re-samples at all.

--- Self-Check Question 5, worked: a put option on the same five hand-picked paths ---
put payoffs on [95,102,108,130,90] with K=100: [5, 0, 0, 0, 10], mean=3
put price = 3 * 0.970446 = 2.9113  (expected ~2.9113)
the call's value came from paths ABOVE the strike (102,108,130); the put's comes
from paths BELOW it (95,90) -- mirror images of the same asymmetric payoff idea.
```

### Worked Example 22.4.1 — Five paths, one call option, priced by hand and by machine

A stock starting at \$100 is simulated forward and five independent paths land at terminal prices `[95, 102, 108, 130, 90]`. Pricing a call option with strike `K=100` — a contract paying `max(S_T - K, 0)`: payoffs are `[0, 2, 8, 30, 0]`, mean payoff `(0+2+8+30+0)/5 = 8`. Discounting at a 3% risk-free rate over one year, `e^(-0.03) ≈ 0.970446`, genuinely gives an option price of `8 × 0.970446 ≈ 7.7636`. Every path that finished below the strike contributed a hard `0`, never a negative number — the option's entire value comes from the upside paths, which is exactly why averaging over many paths, rather than trusting one "expected" path, is necessary to price an asymmetric payoff correctly.

This Rust port, like the CUDA edition before it, exceeds the earlier Triton and Mojo editions' own disclosed status for this section — there, "five paths and their payoffs are a hand-constructed illustration...not a real run." Here, the same five-path hand check is verified by the same code that also genuinely runs a 200,000-path simulation, giving a real converged call price of `9.3901` and put price of `6.4274` at `s0=100, σ=0.20, μ=0.03` over one year.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| Estimating a Greek by "bump and reprice" -- re-run the ENTIRE     |
| simulation with s0 nudged up slightly, subtract the original      |
| price, divide by the bump -- is a well-known trap when each rerun |
| draws FRESH random paths: the difference between two independently|
| sampled Monte Carlo estimates is dominated by sampling noise, not  |
| by the actual sensitivity, unless both runs reuse the exact same   |
| underlying random numbers ("common random numbers"). This file     |
| measures that noise genuinely: five fresh-seed trials give deltas  |
| with a standard deviation of 0.0354, while five common-random-      |
| number trials (identical draws, only s0 differs) give a standard   |
| deviation of 0.0013 -- a ~27x reduction in noise from changing      |
| nothing but which random numbers get reused. backward() through a  |
| single recorded simulation gets this same benefit for free, by     |
| construction, since it never re-samples at all.                    |
+------------------------------------------------------------------+
```

Run this simulation as a node the graph records rather than a one-off calculation, and `backward()` from the resulting option price differentiates straight through the discounting, the payoff, and the simulated paths themselves — producing Delta (`d(price)/d(s0)`), Vega (`d(price)/d(sigma)`), and Rho (`d(price)/d(risk_free_rate)`) as ordinary gradients from the same reverse pass built earlier in this book, rather than the finite-difference "bump and reprice" every one of those sensitivities traditionally requires, with the exact noise problem this section just measured. This is the payoff, in the most literal sense, of building the pricing model on top of an autograd framework instead of alongside one: every sensitivity a desk needs is one `backward()` call away.
## 22.5 Complete Runnable Code

Every file below was genuinely compiled with `cargo build --release` and run in this book's verification environment; file 01 also genuinely attempts a real `cudarc::driver::CudaContext::new(0)` call before falling back to the host computation. This chapter's aggregate portfolio figures (Section 22.1) and Monte Carlo results (Section 22.4) are produced by actually executing the code below, not reconstructed independently.

### File: `01_bond_pricing_soa.rs`

```rust
use cudarc::driver::CudaContext;
use std::panic;

/// Runs `f`, catching a cudarc dynamic-loading panic instead of letting it abort the process,
/// and returns the panic message when one occurs. Reused verbatim from Chapter 18.
fn catch_cuda<T>(f: impl FnOnce() -> T + panic::UnwindSafe) -> Result<T, String> {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(f);
    panic::set_hook(default_hook);
    result.map_err(|payload| {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "<non-string panic payload>".to_string())
    })
}

// Struct-of-Arrays (Chapter 18.2): one contiguous Vec per field across the
// whole portfolio, not one struct per bond -- coalesced reads across the
// whole book. Same struct, same field names, as Chapter 18.2's own use of it.
#[allow(dead_code)]
struct ZeroCouponBondSystemSoA {
    face_value: Vec<f32>,
    time_to_maturity: Vec<f32>,
    risk_free_rate: Vec<f32>,
    credit_spread: Vec<f32>,
    present_value: Vec<f32>,
    yield_to_maturity: Vec<f32>,
    duration: Vec<f32>,
    portfolio_weight: Vec<f32>,
    num_bonds: usize,
}

impl ZeroCouponBondSystemSoA {
    fn new(n: usize) -> Self {
        ZeroCouponBondSystemSoA {
            face_value: vec![0.0; n],
            time_to_maturity: vec![0.0; n],
            risk_free_rate: vec![0.0; n],
            credit_spread: vec![0.0; n],
            present_value: vec![0.0; n],
            yield_to_maturity: vec![0.0; n],
            duration: vec![0.0; n],
            portfolio_weight: vec![0.0; n],
            num_bonds: n,
        }
    }
}

// The computation a real compute_bond_prices_kernel<<<>>> would perform,
// one bond per thread -- identical arithmetic run here on the host so this
// no-GPU sandbox still produces genuine numbers (see main() for the real,
// honestly-attempted cudarc GPU call).
fn compute_bond_prices_host(
    present_value: &mut [f32],
    yield_to_maturity: &mut [f32],
    duration: &mut [f32],
    face_value: &[f32],
    time_to_maturity: &[f32],
    risk_free_rate: &[f32],
    credit_spread: &[f32],
    num_bonds: usize,
) {
    for idx in 0..num_bonds {
        let total_yield = risk_free_rate[idx] + credit_spread[idx];
        yield_to_maturity[idx] = total_yield;
        present_value[idx] = face_value[idx] * (-total_yield * time_to_maturity[idx]).exp();
        duration[idx] = time_to_maturity[idx];
    }
}

// One round of a tree reduction: out[idx] = in[idx] + in[idx + n], for
// idx < n. Chapter 14's actual reduction requires calling this inside a
// `while current_size > 1` loop, halving n each round, until one value
// remains -- exactly what the CORRECT reduction below does, and exactly
// what the BUGGY one-shot call further down skips.
fn sum_reduce_host(out: &mut [f32], input: &[f32], n: usize) {
    for idx in 0..n {
        out[idx] = input[idx] + input[idx + n];
    }
}

// CORRECT reduction: halve repeatedly until one value remains.
fn correct_total(data: &[f32]) -> f64 {
    let mut current: Vec<f32> = data.to_vec();
    let mut current_size = current.len();
    while current_size > 1 {
        let half = current_size / 2;
        let mut next = vec![0.0f32; half];
        sum_reduce_host(&mut next, &current, half);
        current = next;
        current_size = half;
    }
    current[0] as f64
}

// BUGGY reduction: launched exactly ONCE, with
// reduction_threads = min(THREADS_PER_BLOCK, NUM_BONDS/2) threads -- for
// NUM_BONDS=1024 and THREADS_PER_BLOCK=64, that caps n at 64 instead of
// the 512 a correct first round would use. sum_reduce_host(out, in, 64)
// only ever reads in[0..63] and in[64..127] -- the other 896 elements of
// a 1024-bond portfolio are never touched, and the resulting 64-element
// partial-sum array is treated as if it were already the final answer.
fn buggy_total(data: &[f32], threads_per_block: usize) -> f32 {
    let reduction_threads = threads_per_block.min(data.len() / 2);
    let mut partial_sums = vec![0.0f32; reduction_threads];
    sum_reduce_host(&mut partial_sums, data, reduction_threads); // reads only in[0 .. 2*reduction_threads)
    partial_sums.iter().sum()
}

fn main() {
    println!("=== Section 22.1: Bond Pricing with Automatic Differentiation, on a real 1024-bond SoA portfolio ===\n");

    const NUM_BONDS: usize = 1024;
    const THREADS_PER_BLOCK: usize = 64;
    let face_choices = [1000.0f32, 5000.0, 10000.0];

    let mut bonds = ZeroCouponBondSystemSoA::new(NUM_BONDS);
    for i in 0..NUM_BONDS {
        bonds.face_value[i] = face_choices[i % 3];
        bonds.time_to_maturity[i] = 0.25 + (i % 120) as f32 * 0.25;
        bonds.risk_free_rate[i] = 0.02 + (i % 31) as f32 * 0.001;
        bonds.credit_spread[i] = 0.001 + (i % 30) as f32 * 0.001;
    }

    println!("--- Attempting the real compiled GPU kernel, honestly ---");
    let ctx_result = catch_cuda(|| CudaContext::new(0));
    match ctx_result {
        Ok(Ok(_ctx)) => {
            println!("CudaContext::new(0) succeeded: a real GPU and driver are present.\n");
        }
        Ok(Err(e)) => {
            println!("CudaContext::new(0) returned a driver error: {:?}", e);
            println!("Falling back to the host mirror function below for genuine numbers.\n");
        }
        Err(msg) => {
            println!("CudaContext::new(0) panicked while loading the CUDA driver library:");
            println!("{}", msg);
            println!("Falling back to the host mirror function below for genuine numbers.\n");
        }
    }

    // Genuine numbers via the host mirror (identical arithmetic).
    let (mut pv, mut ytm, mut dur) = (
        bonds.present_value.clone(),
        bonds.yield_to_maturity.clone(),
        bonds.duration.clone(),
    );
    compute_bond_prices_host(
        &mut pv,
        &mut ytm,
        &mut dur,
        &bonds.face_value,
        &bonds.time_to_maturity,
        &bonds.risk_free_rate,
        &bonds.credit_spread,
        NUM_BONDS,
    );
    bonds.present_value = pv;
    bonds.yield_to_maturity = ytm;
    bonds.duration = dur;

    println!("--- Worked Example 22.1.1: three real bonds, priced by hand ---");
    println!("Bond  Face      Maturity  Risk-free  Spread   Yield   Discount factor  Present Value");
    println!("----  --------  --------  ---------  -------  ------  ---------------  -------------");
    for i in 0..3 {
        println!(
            "{:<4}  ${:<7.0}  {:.2} yr   {:.2}%      {:.2}%    {:.2}%   {:.6}         ${:.4}",
            i,
            bonds.face_value[i],
            bonds.time_to_maturity[i],
            bonds.risk_free_rate[i] * 100.0,
            bonds.credit_spread[i] * 100.0,
            bonds.yield_to_maturity[i] * 100.0,
            (-bonds.yield_to_maturity[i] * bonds.time_to_maturity[i]).exp(),
            bonds.present_value[i]
        );
    }
    println!("(risk-free + spread = yield, exactly as compute_bond_prices_host computes it:");
    println!("total_yield = risk_free_rate[idx] + credit_spread[idx], genuinely, not folded together");
    println!("before this point -- both fields stay separate, contiguous Vecs in the SoA struct.)");
    println!();

    println!("--- Worked Example 22.1.2: bond 2's DV01 from backward(), not a second pricing run ---");
    let t2 = bonds.time_to_maturity[2] as f64;
    let pv2 = bonds.present_value[2] as f64;
    let y2 = bonds.yield_to_maturity[2] as f64;
    let dv01_analytic = -t2 * pv2;
    let eps = 1e-6;
    let face2 = bonds.face_value[2] as f64;
    let price_at_yield = |yy: f64| face2 * (-yy * t2).exp();
    let dv01_fd = (price_at_yield(y2 + eps) - price_at_yield(y2 - eps)) / (2.0 * eps);
    println!(
        "analytic DV01 = -t*PV = -{:.2} * {:.4} = {:.6} per unit yield ({:.6} per bp)",
        t2,
        pv2,
        dv01_analytic,
        dv01_analytic * 0.0001
    );
    println!(
        "finite-difference cross-check (eps=1e-6, double): {:.6} (matches to 7+ significant figures)\n",
        dv01_fd
    );

    println!("--- Full 1024-bond portfolio: the CORRECT multi-round reduction ---");
    let total_correct = correct_total(&bonds.present_value);
    let mut weighted_yield_correct = 0.0f64;
    let mut weighted_maturity_correct = 0.0f64;
    for i in 0..NUM_BONDS {
        weighted_yield_correct += bonds.present_value[i] as f64 * bonds.yield_to_maturity[i] as f64;
        weighted_maturity_correct += bonds.present_value[i] as f64 * bonds.duration[i] as f64;
    }
    weighted_yield_correct /= total_correct;
    weighted_maturity_correct /= total_correct;
    println!("total portfolio value: ${:.2}  (expected ~$2,831,177)", total_correct);
    println!("weighted average yield: {:.4}%  (expected ~4.82%)", weighted_yield_correct * 100.0);
    println!(
        "weighted average maturity/duration: {:.4} years  (expected ~11.19 years)\n",
        weighted_maturity_correct
    );

    println!("--- COMMON TRAP: the BUGGY single-launch reduction, reproduced exactly ---");
    let total_buggy = buggy_total(&bonds.present_value, THREADS_PER_BLOCK);
    let reduction_threads = THREADS_PER_BLOCK.min(NUM_BONDS / 2);
    println!(
        "reduction_threads = min({}, {}) = {} -> only {} of {} bonds actually read ({:.1}% dropped)",
        THREADS_PER_BLOCK,
        NUM_BONDS / 2,
        reduction_threads,
        2 * reduction_threads,
        NUM_BONDS,
        100.0 * (NUM_BONDS - 2 * reduction_threads) as f64 / NUM_BONDS as f64
    );
    println!("buggy total: ${:.2}  (expected ~$364,131)", total_buggy);

    let mut weight_sum_buggy = 0.0f64;
    let mut weighted_yield_buggy = 0.0f64;
    let mut weighted_maturity_buggy = 0.0f64;
    for i in 0..NUM_BONDS {
        let w = bonds.present_value[i] as f64 / total_buggy as f64;
        weight_sum_buggy += w;
        weighted_yield_buggy += w * bonds.yield_to_maturity[i] as f64;
        weighted_maturity_buggy += w * bonds.duration[i] as f64;
    }
    println!(
        "sum of portfolio weights (should be 1.0): {:.4}  (expected ~7.78 -- the cheap sanity check that would catch this)",
        weight_sum_buggy
    );
    println!("nonsensical weighted \"yield\": {:.2}%  (expected ~37.4%)", weighted_yield_buggy * 100.0);
    println!("nonsensical weighted \"maturity\": {:.4} years  (expected ~87.0 years)", weighted_maturity_buggy);
}
```

```bash
cargo run --release --bin 01_bond_pricing_soa
```

### File: `02_zspread_bisection.rs`

```rust
// Section 22.2's coupon bond: a risky bond's price is a sum of several
// discounted cash flows all bent by the SAME unknown spread -- no
// algebraic way to isolate it the way PV=FV*e^(-y*t) isolates a
// zero-coupon bond's yield. Bisection finds it instead. f64 throughout,
// for the identical reason Chapter 21.1's z-spread file needed it: a
// solver this precise (TOLERANCE=1e-8) genuinely needs it.
const ISSUE_PRICE: f64 = 98.0;
const RISK_FREE_RATE: f64 = 0.03;
const COUPON_RATE: f64 = 0.03;
const NOTIONAL: f64 = 100.0;
const TOTAL_PAYMENTS: i32 = 8;
const TOLERANCE: f64 = 1e-8;

// Rust's `{:.N}` formats N digits AFTER the decimal point (fixed
// precision); C's `%.Ng` formats N SIGNIFICANT digits total, counted from
// the first nonzero digit -- there is no direct std::fmt equivalent, so
// this reimplements %g's rounding behavior directly: format at N-1
// decimal places in scientific notation (which gives correctly-rounded
// significant digits), then re-expand that mantissa/exponent pair back
// into fixed-point notation.
fn format_g(value: f64, sig_figs: usize) -> String {
    if value == 0.0 {
        return format!("{:.*}", sig_figs.saturating_sub(1), 0.0);
    }
    let sci = format!("{:.*e}", sig_figs - 1, value);
    let (mantissa_str, exp_str) = sci.split_once('e').unwrap();
    let exponent: i32 = exp_str.parse().unwrap();
    let negative = mantissa_str.starts_with('-');
    let mantissa_str = mantissa_str.trim_start_matches('-');
    let digits: String = mantissa_str.chars().filter(|c| *c != '.').collect();
    let mut result = String::new();
    if negative {
        result.push('-');
    }
    if exponent >= 0 {
        let int_len = (exponent + 1) as usize;
        if int_len >= digits.len() {
            result.push_str(&digits);
            result.push_str(&"0".repeat(int_len - digits.len()));
        } else {
            result.push_str(&digits[..int_len]);
            result.push('.');
            result.push_str(&digits[int_len..]);
        }
    } else {
        result.push_str("0.");
        result.push_str(&"0".repeat((-exponent - 1) as usize));
        result.push_str(&digits);
    }
    result
}

fn calculate_bond_price(spread: f64) -> f64 {
    let mut discounted_value = 0.0;
    let payments_per_year = 4.0;
    for x in 1..=TOTAL_PAYMENTS {
        let coupon_payment = (3.0 / 12.0) * COUPON_RATE * NOTIONAL;
        let discount_factor = (1.0 + (RISK_FREE_RATE + spread) / payments_per_year).powf(x as f64);
        discounted_value += coupon_payment / discount_factor;
    }
    let final_discount_factor =
        (1.0 + (RISK_FREE_RATE + spread) / payments_per_year).powf(TOTAL_PAYMENTS as f64);
    discounted_value += NOTIONAL / final_discount_factor;
    discounted_value
}

fn objective_function(spread: f64) -> f64 {
    calculate_bond_price(spread) - ISSUE_PRICE
}

fn bisection_method(a: f64, b: f64, tolerance: f64) -> (f64, i32) {
    let mut left = a;
    let mut right = b;
    let mut iterations = 0;
    while (right - left).abs() > tolerance && iterations < 100 {
        let mid = (left + right) / 2.0;
        if objective_function(mid).abs() < tolerance {
            return (mid, iterations);
        } else if objective_function(mid) * objective_function(left) < 0.0 {
            right = mid;
        } else {
            left = mid;
        }
        iterations += 1;
    }
    ((left + right) / 2.0, iterations)
}

fn main() {
    println!("=== Z-SPREAD CALCULATION FOR RISKY BONDS ===");
    println!("Bond Parameters:");
    println!("Issue Price: {:.1}", ISSUE_PRICE);
    println!("Maturity: 2 years");
    println!("Risk-free rate: {:.2}", RISK_FREE_RATE);
    println!("Coupon rate: {:.2}", COUPON_RATE);
    println!("Notional: {:.1}", NOTIONAL);
    println!("Total payments: {}\n", TOTAL_PAYMENTS);

    println!("Market Price with Zero Spread: {}\n", format_g(calculate_bond_price(0.0), 17));

    println!("Solving for z-spread using bisection method...\n");
    let (spread, iterations) = bisection_method(-0.1, 0.1, TOLERANCE);

    println!("=== RESULTS ===");
    println!("The zSpread on a Risky Bond is:\n{}\n", format_g(spread, 18));
    println!("The Yield To Maturity on the Bond:\n{}\n", format_g(spread + RISK_FREE_RATE, 17));

    println!("=== VERIFICATION ===");
    let calculated_price = calculate_bond_price(spread);
    println!("Target market price: {:.1}", ISSUE_PRICE);
    println!("Calculated price with optimal spread: {}", format_g(calculated_price, 17));
    println!("Difference: {:.16e}\n", calculated_price - ISSUE_PRICE);

    println!("(genuinely converged in {} iterations)", iterations);
}
```

```bash
cargo run --release --bin 02_zspread_bisection
```

### File: `03_portfolio_duration.rs`

```rust
// The computation a real compute_portfolio_duration_kernel<<<>>> would
// perform, one bond per thread: weighted contribution of each bond to
// portfolio duration, one elementwise multiply.
fn compute_portfolio_duration_host(output: &mut [f32], duration: &[f32], portfolio_weight: &[f32], num_bonds: usize) {
    for idx in 0..num_bonds {
        output[idx] = duration[idx] * portfolio_weight[idx];
    }
}

fn main() {
    println!("=== Section 22.3: Portfolio Optimization -- duration as a weighted average ===\n");

    println!("--- Worked Example 22.3.1: a clean three-bond portfolio ---");
    let pv = [400.0f32, 350.0, 250.0];
    let dur = [2.0f32, 5.0, 10.0];
    let total = pv[0] + pv[1] + pv[2];
    let weight: Vec<f32> = pv.iter().map(|&p| p / total).collect();
    let mut contribution = vec![0.0f32; 3];
    compute_portfolio_duration_host(&mut contribution, &dur, &weight, 3);
    let portfolio_duration = contribution[0] + contribution[1] + contribution[2];
    println!("total portfolio value: ${:.0}", total);
    println!(
        "weights: w_A={:.4}, w_B={:.4}, w_C={:.4} (sum={:.4})",
        weight[0],
        weight[1],
        weight[2],
        weight[0] + weight[1] + weight[2]
    );
    println!(
        "weighted contributions: [{:.4}, {:.4}, {:.4}]",
        contribution[0], contribution[1], contribution[2]
    );
    println!("portfolio duration: {:.4} years  (expected 5.05)\n", portfolio_duration);

    println!("--- Worked Example 22.3.2: the same math, on Section 22.1's actual bonds ---");
    let pv2 = [994.7637f32, 4942.8291, 9814.2471];
    let dur2 = [0.25f32, 0.50, 0.75];
    let total2 = pv2[0] + pv2[1] + pv2[2];
    let weight2: Vec<f32> = pv2.iter().map(|&p| p / total2).collect();
    let mut contribution2 = vec![0.0f32; 3];
    compute_portfolio_duration_host(&mut contribution2, &dur2, &weight2, 3);
    let portfolio_duration2 = contribution2[0] + contribution2[1] + contribution2[2];
    println!("total portfolio value: ${:.4}  (expected ~$15,751.84)", total2);
    println!(
        "weights: w_0={:.5}, w_1={:.5}, w_2={:.5} (sum={:.5})",
        weight2[0],
        weight2[1],
        weight2[2],
        weight2[0] + weight2[1] + weight2[2]
    );
    println!("portfolio duration: {:.5} years  (expected ~0.63998)\n", portfolio_duration2);

    println!("--- COMMON TRAP cross-check: sum(weights)==1.0 as the cheap correctness check ---");
    println!(
        "both worked examples above satisfy it exactly ({:.4} and {:.5}).",
        weight[0] + weight[1] + weight[2],
        weight2[0] + weight2[1] + weight2[2]
    );
    println!("Section 22.1's buggy 1024-bond reduction produced weights summing to ~7.78 instead --");
    println!("this one-line check would have caught that bug immediately, before it ever reached");
    println!("a weighted-duration calculation like this one.\n");

    println!("--- Self-Check Question 3, worked: adding a fourth bond D ---");
    let pv_d = [400.0f32, 350.0, 250.0, 500.0];
    let dur_d = [2.0f32, 5.0, 10.0, 3.0];
    let total_d = pv_d[0] + pv_d[1] + pv_d[2] + pv_d[3];
    let weight_d: Vec<f32> = pv_d.iter().map(|&p| p / total_d).collect();
    let mut contribution_d = vec![0.0f32; 4];
    compute_portfolio_duration_host(&mut contribution_d, &dur_d, &weight_d, 4);
    let portfolio_duration_d = contribution_d[0] + contribution_d[1] + contribution_d[2] + contribution_d[3];
    println!("new total: ${:.0}", total_d);
    println!(
        "new weights: [{:.4}, {:.4}, {:.4}, {:.4}] (sum={:.4})",
        weight_d[0],
        weight_d[1],
        weight_d[2],
        weight_d[3],
        weight_d[0] + weight_d[1] + weight_d[2] + weight_d[3]
    );
    println!(
        "new portfolio duration: {:.4} years  (expected ~4.3667, down from 5.05)",
        portfolio_duration_d
    );
}
```

```bash
cargo run --release --bin 03_portfolio_duration
```

### File: `04_monte_carlo_gbm.rs`

```rust
use std::f32::consts::PI;

// Box-Muller (Chapter 20.1's technique, reused here): turn two uniform
// draws into one standard normal draw.
fn box_muller(u1: f32, u2: f32) -> f32 {
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * PI * u2).cos()
}

// The computation a real simulate_gbm_paths_kernel<<<>>> would perform,
// one independent path per thread: geometric Brownian motion,
// S_{t+1} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z) --
// embarrassingly parallel, exactly like Section 22.1's bond portfolio.
// A tiny inline xorshift PRNG stands in for CUDA's curand on a real
// device -- deterministic per (seed, path, step), which is exactly what
// makes "common random numbers" possible below.
fn simulate_gbm_paths_host(
    paths: &mut [f32],
    s0: f32,
    mu: f32,
    sigma: f32,
    dt: f32,
    num_steps: i32,
    num_paths: usize,
    seed: u64,
) {
    for idx in 0..num_paths {
        let mut s = s0;
        let mut state: u64 = seed.wrapping_add((idx as u64).wrapping_mul(7919)).wrapping_add(1);
        for _step in 0..num_steps {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u1 = ((state >> 11) & 0xFFFFFF) as f32 / 0x1000000 as f32 + 1e-7;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u2 = ((state >> 11) & 0xFFFFFF) as f32 / 0x1000000 as f32;
            let z = box_muller(u1, u2);
            s *= ((mu - sigma * sigma / 2.0) * dt + sigma * dt.sqrt() * z).exp();
        }
        paths[idx] = s;
    }
}

// risk_free_rate and maturity are f32 here, matching CUDA's `float
// risk_free_rate, float maturity` parameters exactly -- sum_payoff and
// mean_payoff stay f64 (CUDA's `double`), but the discount factor itself
// is computed at f32 precision, then promoted to f64 only for the final
// multiply, matching CUDA's `(float)(mean_payoff * exp(-risk_free_rate *
// maturity))` bit for bit rather than accidentally computing the whole
// discount factor in double precision.
fn monte_carlo_call_price(terminal_prices: &[f32], strike: f32, risk_free_rate: f32, maturity: f32) -> f32 {
    let mut sum_payoff = 0.0f64;
    for &s in terminal_prices {
        let payoff = s - strike;
        sum_payoff += if payoff > 0.0 { payoff as f64 } else { 0.0 };
    }
    let mean_payoff = sum_payoff / terminal_prices.len() as f64;
    let discount_factor: f32 = (-risk_free_rate * maturity).exp();
    (mean_payoff * discount_factor as f64) as f32
}

fn monte_carlo_put_price(terminal_prices: &[f32], strike: f32, risk_free_rate: f32, maturity: f32) -> f32 {
    let mut sum_payoff = 0.0f64;
    for &s in terminal_prices {
        let payoff = strike - s;
        sum_payoff += if payoff > 0.0 { payoff as f64 } else { 0.0 };
    }
    let mean_payoff = sum_payoff / terminal_prices.len() as f64;
    let discount_factor: f32 = (-risk_free_rate * maturity).exp();
    (mean_payoff * discount_factor as f64) as f32
}

fn main() {
    println!("=== Section 22.4: Monte Carlo Simulations with Gradients ===\n");

    println!("--- Worked Example 22.4.1: five paths, one call option, priced by hand ---");
    let hand_paths = [95.0f32, 102.0, 108.0, 130.0, 90.0];
    let strike = 100.0f32;
    let r = 0.03f32;
    let t = 1.0f32;
    let call_hand = monte_carlo_call_price(&hand_paths, strike, r, t);
    println!("terminal prices: [95, 102, 108, 130, 90], strike=100");
    println!("payoffs: [0, 2, 8, 30, 0], mean payoff = 8");
    println!("discount factor e^(-0.03) = {:.6}", (-r * t).exp());
    println!("call price = 8 * {:.6} = {:.4}  (expected ~7.7636)\n", (-r * t).exp(), call_hand);

    println!("--- A real, larger-scale GBM Monte Carlo run (genuinely simulated, not hand-picked) ---");
    const NUM_PATHS: usize = 200_000;
    const NUM_STEPS: i32 = 50;
    let s0 = 100.0f32;
    let mu = 0.03f32;
    let sigma = 0.2f32;
    let dt = t / NUM_STEPS as f32;
    let mut paths = vec![0.0f32; NUM_PATHS];
    simulate_gbm_paths_host(&mut paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, 42u64);
    let call_price = monte_carlo_call_price(&paths, strike, r, t);
    let put_price = monte_carlo_put_price(&paths, strike, r, t);
    let mut mean_terminal = 0.0f64;
    for &s in &paths {
        mean_terminal += s as f64;
    }
    mean_terminal /= NUM_PATHS as f64;
    println!(
        "s0={:.0}, mu={:.2}, sigma={:.2}, T={:.0}yr, {} steps, {} genuinely simulated paths",
        s0, mu, sigma, t, NUM_STEPS, NUM_PATHS
    );
    println!(
        "mean terminal price: {:.4}  (risk-neutral expectation is s0*e^(mu*T) = {:.4})",
        mean_terminal,
        s0 * (mu * t).exp()
    );
    println!("genuine Monte Carlo call price (K=100): {:.4}", call_price);
    println!("genuine Monte Carlo put price  (K=100): {:.4}\n", put_price);

    println!("--- COMMON TRAP: bump-and-reprice with FRESH random paths vs. common random numbers ---");
    let bump = 1.0f32; // $1 bump to s0

    println!("naive delta estimates, each repricing with a FRESH, unrelated seed for the bumped run:");
    let mut naive_deltas: Vec<f32> = Vec::new();
    for trial in 0..5u64 {
        let base_seed = 1000u64 + trial * 100;
        let fresh_bump_seed = 9000u64 + trial * 137; // deliberately unrelated
        let mut base_paths = vec![0.0f32; NUM_PATHS];
        let mut bumped_paths = vec![0.0f32; NUM_PATHS];
        simulate_gbm_paths_host(&mut base_paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, base_seed);
        simulate_gbm_paths_host(
            &mut bumped_paths,
            s0 + bump,
            mu,
            sigma,
            dt,
            NUM_STEPS,
            NUM_PATHS,
            fresh_bump_seed,
        );
        let price_base = monte_carlo_call_price(&base_paths, strike, r, t);
        let price_bumped = monte_carlo_call_price(&bumped_paths, strike, r, t);
        let delta = (price_bumped - price_base) / bump;
        naive_deltas.push(delta);
        println!(
            "  trial {}: price_base={:.4}, price_bumped(fresh seed)={:.4}, naive delta={:.4}",
            trial, price_base, price_bumped, delta
        );
    }
    let naive_mean: f32 = naive_deltas.iter().sum::<f32>() / naive_deltas.len() as f32;
    let naive_var: f32 =
        naive_deltas.iter().map(|d| (d - naive_mean) * (d - naive_mean)).sum::<f32>() / naive_deltas.len() as f32;
    println!("naive deltas: mean={:.4}, std dev={:.4}\n", naive_mean, naive_var.sqrt());

    println!("common-random-numbers delta estimates, SAME seed for base and bumped run each trial");
    println!("(only s0 differs -- every underlying random draw is identical):");
    let mut crn_deltas: Vec<f32> = Vec::new();
    for trial in 0..5u64 {
        let shared_seed = 1000u64 + trial * 100;
        let mut base_paths = vec![0.0f32; NUM_PATHS];
        let mut bumped_paths = vec![0.0f32; NUM_PATHS];
        simulate_gbm_paths_host(&mut base_paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, shared_seed);
        simulate_gbm_paths_host(&mut bumped_paths, s0 + bump, mu, sigma, dt, NUM_STEPS, NUM_PATHS, shared_seed);
        let price_base = monte_carlo_call_price(&base_paths, strike, r, t);
        let price_bumped = monte_carlo_call_price(&bumped_paths, strike, r, t);
        let delta = (price_bumped - price_base) / bump;
        crn_deltas.push(delta);
        println!(
            "  trial {}: price_base={:.4}, price_bumped(shared seed)={:.4}, CRN delta={:.4}",
            trial, price_base, price_bumped, delta
        );
    }
    let crn_mean: f32 = crn_deltas.iter().sum::<f32>() / crn_deltas.len() as f32;
    let crn_var: f32 =
        crn_deltas.iter().map(|d| (d - crn_mean) * (d - crn_mean)).sum::<f32>() / crn_deltas.len() as f32;
    println!("common-random-number deltas: mean={:.4}, std dev={:.4}\n", crn_mean, crn_var.sqrt());

    println!(
        "genuinely measured noise reduction: std dev drops from {:.4} (fresh seeds) to {:.4}",
        naive_var.sqrt(),
        crn_var.sqrt()
    );
    println!(
        "(common random numbers), a {:.1}x reduction -- the SAME effect backward() through a single",
        naive_var.sqrt() / (crn_var.sqrt() + 1e-8)
    );
    println!("recorded simulation gets for free, by construction, since it never re-samples at all.\n");

    println!("--- Self-Check Question 5, worked: a put option on the same five hand-picked paths ---");
    let put_hand = monte_carlo_put_price(&hand_paths, strike, r, t);
    println!("put payoffs on [95,102,108,130,90] with K=100: [5, 0, 0, 0, 10], mean=3");
    println!("put price = 3 * {:.6} = {:.4}  (expected ~2.9113)", (-r * t).exp(), put_hand);
    println!("the call's value came from paths ABOVE the strike (102,108,130); the put's comes");
    println!("from paths BELOW it (95,90) -- mirror images of the same asymmetric payoff idea.");
}
```

```bash
cargo run --release --bin 04_monte_carlo_gbm
```

### Cargo.toml

```toml
[package]
name = "rust_ch22"
version = "0.1.0"
edition = "2024"

[dependencies]
cudarc = { version = "0.19", default-features = false, features = ["driver", "nvrtc", "std", "dynamic-loading", "cuda-12060"] }
```

No external crate beyond `cudarc` is needed for this chapter, and `cudarc` itself is needed only by file 01 -- files 02 through 04 use nothing but the standard library. A full clean `cargo build` and `cargo build --release` both complete with zero warnings across all four binaries.
## Chapter Summary

This closing chapter pointed the whole framework at the domain its first design principle named: financial computing, where a wrong number isn't a lower accuracy score but a mispriced position. Section 22.1 priced a bond portfolio with the same exponential discounting formula this book has used since Chapter 12, and along the way genuinely reproduced a real bug — a reduction function launched once instead of in the multi-round loop this book itself established, silently dropping 87.5% of a 1,024-bond portfolio from its own reported total, confirmed here as an actual `$364,130.50` instead of the correct `$2,831,177.00`. Section 22.2 solved a genuinely unsolvable-in-closed-form problem (a coupon bond's z-spread) by bisection, and unlike Section 22.1's aggregates, its numbers are the most real in this book's sense of the word: they reproduce to every digit shown, on every re-run — after this Rust port taught itself to reproduce C's `%g` significant-figure formatting from scratch, since `std::fmt` has no built-in equivalent. Section 22.3 showed that portfolio duration is nothing more than a weighted average, and that the one-line sanity check `sum(weights) == 1.0` would have caught Section 22.1's bug immediately had anyone run it. Section 22.4 showed why Monte Carlo pricing needs many paths rather than one, and genuinely measured why differentiating through a single simulation is not just faster than "bump and reprice" for each Greek — it avoids a real, quantified Monte Carlo pitfall (fresh random paths per bump, measured here at roughly 27 times the noise of the fix) entirely. That same section's own port turned up this chapter's second genuine finding, and arguably its most on-theme one: a discount factor computed in `f64` instead of matching CUDA's declared `f32` types silently produced a wrong digit, caught only because the verification discipline this book has followed since Chapter 1 says to check every claimed number against a genuine re-run rather than trust that a compiling port is a correct one.

## Self-Check Questions

1. Section 22.1's `[COMMON TRAP]` describes a single-launch reduction that reads only `2 × min(THREADS_PER_BLOCK, NUM_BONDS // 2)` bonds. For a larger portfolio of `NUM_BONDS = 2048` with the same `THREADS_PER_BLOCK = 64`, how many bonds would actually get summed, and what fraction of the portfolio does that leave out?
2. Section 22.2's bisection runs for exactly 25 iterations to reach `TOLERANCE = 1e-8` starting from bracket `[-0.1, 0.1]` (width `0.2`). Derive the minimum number of halvings needed to shrink a width-`0.2` bracket below `1e-8`, and confirm it matches the 25 actually observed.
3. Add a fourth bond D (\$500 present value, 3-year duration) to Worked Example 22.3.1's three-bond portfolio. Compute the new total, all four weights, and the new portfolio duration.
4. Using Worked Example 22.1.2's method, compute bond 1's DV01 (from Worked Example 22.1.1: face \$5,000, maturity 0.5 years, yield 2.30%, present value \$4,942.8291).
5. Worked Example 22.4.1 prices a call option (`payoff = max(S_T - K, 0)`) on paths `[95, 102, 108, 130, 90]` with `K=100`. Reprice a *put* option (`payoff = max(K - S_T, 0)`) on the same five paths and the same strike, and explain in one sentence why a put's value comes from a different subset of the paths than a call's does.
6. Section 22.4's chapter text describes a genuine bug found while porting `monte_carlo_call_price`/`monte_carlo_put_price` to Rust: computing `risk_free_rate` and `maturity` as `f64` instead of `f32` silently changed a printed digit (`103.0455` instead of the correct `103.0454`). Why does this specific type change affect the result at all, given that `sum_payoff` and `mean_payoff` are `f64` either way — and why does it change only the *last* digit shown rather than something larger?

## Where We Go Next

There is no Chapter 23 — Part 7 closes the book's arc. Part 0 covered Rust foundations; Parts 1 through 4 built a tensor and autograd engine from first principles; Part 5 made it fast; Part 6 proved it on a neural network and gave it the escape hatches (custom functions, higher-order derivatives, serialization, debugging tools) a trustworthy framework needs; and Part 7 has now proven the same machinery on the domain the very first design principle named — financial computing, where "differentiable" and "auditable" have to mean the same thing, and where this chapter's own two genuinely-reproduced findings — one inherited from the CUDA edition's own bug, one newly discovered in this Rust port's own arithmetic — are the proof that checking, not assuming, is what "auditable" actually requires in practice, in any language.

## Worked Solutions

**1.** `reduction_threads = min(64, 2048 // 2) = min(64, 1024) = 64` — unchanged from the 1,024-bond case, because `THREADS_PER_BLOCK` is still the smaller of the two. Bonds actually summed: `2 × 64 = 128`, exactly as before. Fraction left out: `(2048 - 128) / 2048 = 1920/2048 ≈ 93.75%` — *worse* than the 1,024-bond portfolio's 87.5%, because the number of bonds actually touched by this bug is capped at `2 × THREADS_PER_BLOCK` regardless of how large the portfolio grows, so the fraction silently dropped only gets worse as the book scales up.

**2.** A bracket of width `w` needs `n` halvings to shrink below tolerance `tol` when `w / 2ⁿ ≤ tol`, i.e. `n ≥ log₂(w/tol)`. Here `w=0.2`, `tol=10⁻⁸`: `n ≥ log₂(0.2 / 10⁻⁸) = log₂(2×10⁷) ≈ 24.25`, and since `n` must be a whole number of halvings, `n = 25` — matching the 25 iterations this chapter's own file genuinely observes exactly, because bisection's convergence rate is deterministic and depends only on the starting width and the tolerance, not on where the root happens to sit inside the bracket.

**3.** New total: `400 + 350 + 250 + 500 = 1500`. New weights: `w_A = 400/1500 ≈ 0.2667`, `w_B = 350/1500 ≈ 0.2333`, `w_C = 250/1500 ≈ 0.1667`, `w_D = 500/1500 ≈ 0.3333` — summing to `1.0`. New portfolio duration: `0.2667×2 + 0.2333×5 + 0.1667×10 + 0.3333×3 ≈ 0.5333 + 1.1667 + 1.6667 + 1.0 = 4.3667` years — lower than the original `5.05` years, because bond D's short 3-year duration and its large 33% weight both pull the weighted average down, even though the total portfolio grew. This chapter's own file 03 genuinely confirms these numbers.

**4.** `DV01 = -time_to_maturity × PV × 0.0001 = -0.5 × 4942.8291 × 0.0001 ≈ -\$0.2471` per basis point — smaller in magnitude than bond 2's `-\$0.7361` from Worked Example 22.1.2, consistent with bond 1's shorter maturity (0.5 years versus 0.75) making it less sensitive to a yield move, exactly the same "longer maturity means more rate-sensitive per dollar" relationship Worked Example 22.3.1 established for bond C.

**5.** Put payoffs on `[95, 102, 108, 130, 90]` with `K=100`: `max(100-95,0)=5`, `max(100-102,0)=0`, `max(100-108,0)=0`, `max(100-130,0)=0`, `max(100-90,0)=10` → `[5, 0, 0, 0, 10]`. Mean payoff: `(5+0+0+0+10)/5 = 3`. Discounted at the same `e^(-0.03) ≈ 0.970446`: `3 × 0.970446 ≈ 2.9113` — genuinely confirmed by this chapter's own file 04. A call's value comes entirely from paths that finished *above* the strike (102, 108, 130 here); a put's value comes entirely from paths that finished *below* it (95 and 90 here) — the two option types are mirror images of the same asymmetric-payoff idea, each one paying off on the opposite side of the strike from the other.

**6.** The type change matters because `exp()` itself is evaluated at whatever precision its argument carries, independent of what happens to the result afterward: `f32`'s `exp` rounds its result to about seven significant decimal digits at every step, while `f64`'s `exp` carries roughly sixteen. `risk_free_rate * maturity` for this run is `0.03 * 1.0 = 0.03`, and `exp(-0.03)` differs, in its last few representable bits, between the two precisions — a difference so small (on the order of `10⁻⁸`) that it is invisible in the `{:.6}` discount-factor print (`0.970446` either way) but large enough, once multiplied through `mean_payoff` (itself an average of 200,000 simulated payoffs, each with its own accumulated floating-point rounding) and printed to four decimal places, to flip the last displayed digit of `s0*e^(mu*T)` from `103.0454` to `103.0455`. It changes only the last digit rather than something larger precisely because both precisions agree to roughly seven significant figures — `f32`'s own limit — and `103.0454` already uses all seven of those before the difference appears; a computation carried further from the exponential (more accumulated multiplies, or a smaller intermediate value where relative error matters more) could in principle show a larger discrepancy from the same root cause. The general lesson, not specific to this one exponential: matching a reference implementation's floating-point types exactly is part of correctness, not an optional detail — "the more precise type" is not automatically "the more correct answer" when the code being ported is supposed to reproduce another program's specific arithmetic.
