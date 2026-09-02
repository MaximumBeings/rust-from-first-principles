# Appendix F: A Risk Engine (VaR/XVA)

> "A risk number is not a fact about the world; it is a fact about a model of the world, computed two different ways so the gap between the two answers becomes visible instead of hidden inside a single confident figure."

## What you will understand

By the end of this appendix you will be able to:

- Compute Value-at-Risk two genuinely different ways — a simulated/historical tail-quantile extraction and a closed-form parametric estimate — from the same GBM-simulated data Chapter 22.4 established, and explain why they land close but not identical.
- Compute Conditional VaR (Expected Shortfall) and verify, rather than assume, that `CVaR ≥ VaR` always holds.
- Build a genuine exposure profile (`EE(t)`/`ENE(t)`) from checkpointed Monte Carlo paths of a forward contract, and cross-check it against the closed-form GBM mean.
- Compute CVA, DVA, and FVA from that exposure profile, and explain why DVA's dependence on the bank's own credit quality has been considered a controversial property of XVA accounting.
- State honestly what this appendix does *not* model — MVA, KVA, and wrong-way risk — and why each is named rather than approximated with invented inputs.

## What you need to know first

This appendix builds on Chapter 22.4's GBM path simulation (`box_muller`, the inline xorshift PRNG, the GBM update step) — Sections F.1 and F.2 reuse that exact machinery, byte-for-byte, rather than introducing a new one. Basic familiarity with what a forward contract and a discount factor are helps for Section F.2, but every formula used there is derived and checked inline.

Unlike the CUDA C++ edition's own Appendix G, this appendix does not open with the C++ Rule of Five. That edition's Sections G.1-G.4 spend real space on a double free, five hand-written special member functions, and the self-assignment/self-move guards those five need — real, well-worth-knowing C++ material, but material Rust's ownership model and borrow checker prevent by construction: the compiler refuses to compile a program where two owners believe they each own the same heap allocation, rather than requiring five hand-written functions to make copying safe *after* the fact. There is nothing left to demonstrate by porting a bug class this language doesn't have, so this appendix begins directly where the CUDA edition's own Sections G.5-G.7 do: a risk engine built on Chapter 22.4's Monte Carlo machinery, needing no CUDA toolchain, no GPU, and no C++-specific resource-ownership discussion at all.

## F.1 Value-at-Risk and Its Variants

### Intuition

Value-at-Risk answers one question: at a given confidence level, how bad could tomorrow's loss get? Different VaR methodologies answer it with different assumptions about the distribution of outcomes — this section computes it two genuinely different ways from the same underlying data and checks that they agree approximately, for a reason that is itself informative rather than a discrepancy to explain away.

### Background

This section reuses Chapter 22.4's `box_muller`/xorshift-PRNG/GBM-update machinery byte-for-byte, at a 1-day horizon (`dt = 1/252`) instead of the full-year horizon Chapter 22.4 used for option pricing. Three methodologies are computed:

**Simulated ("historical") VaR** sorts a large sample of simulated 1-day P&L outcomes ascending and reads off the loss at the 1st percentile for 99% confidence. This appendix has no real external historical price series to draw from, so — honestly — "historical VaR" here is applied to the same GBM-simulated sample "Monte Carlo VaR" would use; in real practice these are two different data sources (an actual historical return series vs. a model-simulated one), but the *extraction algorithm* — sort, take the tail quantile — is the genuine, general-purpose technique either data source would use.

**Parametric (variance-covariance) VaR** assumes P&L is normally distributed and uses a closed form: `VaR = S0 * sigma_1day * z_99`, where `z_99 = 2.326348` is the standard normal 99th-percentile critical value and `sigma_1day = sigma * sqrt(dt)`.

**Conditional VaR (CVaR / Expected Shortfall)** is the average of the P&L values *beyond* the VaR cutoff — not just the one point at the boundary, but the full tail. CVaR is always at least as large as VaR by construction: it averages a set of losses every one of which is at least as large as the VaR loss itself, since that's the criterion for a loss belonging to the tail at all.

### Formulas and Key Terms

```
VaR_α = -PnL_(⌊α·N⌋)                          (simulated/historical VaR, PnL sorted ascending)

VaR_α = S0 · σ_Δt · z_α,    σ_Δt = σ·√Δt       (parametric / variance-covariance VaR)

CVaR_α = -(1/k) · Σ_(i=1)^k PnL_(i),   k = ⌊α·N⌋   (Conditional VaR / Expected Shortfall)
```

- **P&L (profit and loss)** — the change in position value over the horizon; `PnL = S_T - S0` in this section, one value per simulated scenario.
- **Confidence level** — the tail probability VaR is measured against; `99%` confidence corresponds to `α = 0.01`, the loss expected to be exceeded only 1% of the time.
- **z_α** — the standard normal critical value at tail probability `α`; `z_0.01 = 2.326348` for 99% confidence, the value `z_99` in the code above.
- **Horizon (Δt)** — the holding period over which the loss is measured; `Δt = 1/252` (one trading day) throughout this section.
- **σ_Δt** — volatility rescaled to the horizon via the square-root-of-time rule: `σ_Δt = σ·√Δt`, `sigma_1day` in the code above.
- **Tail quantile / worst-case index** — the sorted-P&L position VaR is read from, `⌊α·N⌋` for `N` total scenarios — `var_index` in the code above.
- **Expected Shortfall (ES)** — another name for Conditional VaR: the average loss *conditional on* already being in the tail, rather than the single boundary value VaR reports.

### Worked Example F.1.1 — Both methodologies, cross-checked

`01_var_engine.rs` reuses Chapter 22.4's `box_muller` and `simulate_gbm_paths_host` unmodified:

```
=== F.1: Value-at-Risk and Its Variants ===

--- Simulated ("Monte Carlo"/"Historical") VaR, 1-day horizon, 99% confidence ---
holding: 1 unit of the underlying, S0=100.00, mu=sigma=0.03/0.20, 200000 simulated 1-day scenarios
worst-case index for 99% VaR: floor(0.01 * 200000) = 2000
P&L at that index: -2.8665  ->  simulated 99% 1-day VaR = 2.8665

--- Parametric (Variance-Covariance) VaR, same horizon ---
sigma_1day = sigma * sqrt(dt) = 0.012599, z_99 = 2.326348
parametric VaR = S0 * sigma_1day * z_99 = 2.9309

--- Cross-check: parametric vs. simulated, and why they should be CLOSE, not identical ---
relative difference: 0.0225 (2.25%)
(GBM log-returns are exactly normally distributed by construction, but VaR here is
measured on ARITHMETIC P&L (S_T - S0), which is log-normally, not normally,
distributed -- parametric VaR's normal-P&L assumption is therefore an approximation
even against this appendix's own fully-consistent GBM data, not a bug in either method.)

--- Conditional VaR (CVaR / Expected Shortfall): the average loss BEYOND the VaR cutoff ---
averaging the 2000 worst-case P&L values (indices 0..1999): CVaR_99 = 3.2938
CVaR >= VaR: true (3.2938 >= 2.8665)
(this must always hold: CVaR is the average of a set of losses EVERY one of which is
at least as large as the VaR cutoff itself, by definition of which losses qualify.)
```

Genuinely compiled with `cargo build` (zero warnings, both the `dev` and `release` profiles) and run, with `S0=100.0, mu=0.03, sigma=0.20`, `NUM_PATHS=200000` — every digit above matches the CUDA C++ edition's own genuinely-observed output exactly, because the underlying PRNG, seed, and GBM update are byte-for-byte the same machinery, just reached from a different language.

The two methodologies land close (2.87 vs. 2.93, a 2.25% relative difference) precisely *because* the underlying GBM data is internally consistent — and they don't land identically because arithmetic P&L (`S_T - S0`) is log-normally, not normally, distributed even when the *log-returns* driving it are exactly normal by construction. Parametric VaR's normality assumption is therefore always an approximation, even measured against data this appendix itself generated to be as favorable to it as possible. CVaR's invariant (`3.2938 ≥ 2.8665`) is checked, not assumed, directly from the same sorted tail the simulated VaR was read from.

**[COMMON TRAP]** It's tempting to read a close match between simulated and parametric VaR as validation that "the model is right." What it actually validates is narrower: that arithmetic P&L over a short, low-volatility, one-day horizon is *close enough* to normal for the parametric shortcut to be a reasonable approximation — a property of the horizon and volatility level, not a general guarantee that would survive a longer horizon or a fatter-tailed underlying process.

## F.2 XVA and Its Variants

### Intuition

VaR asks about the risk of holding a position. XVA asks a related but different question: given that the counterparty on the other side of a trade, or the bank itself, might default before the trade matures, what is that default risk worth, in the price of the trade itself? Answering it requires more than a single terminal payoff — it requires the trade's *exposure profile over time*, since what matters is how much would be lost if default happened at each point along the way, not only at maturity.

### Background

An exposure profile requires simulating a path's value at multiple checkpoints, not just its terminal value — this section extends Chapter 22.4's simulation to record the price at four quarterly checkpoints per path, continuing each path's own state from one checkpoint to the next rather than resimulating from `t=0` each time. For a long forward contract with value `V(t) = S(t) - K`:

- **Expected Positive Exposure**, `EE(t) = E[max(V(t), 0)]`, is what the bank stands to lose if the *counterparty* defaults at `t` — averaged over paths where the trade is currently in the bank's favor.
- **Expected Negative Exposure**, `ENE(t) = E[min(V(t), 0)]`, is the mirror: what the bank currently owes, relevant to the bank's *own* default risk.
- **CVA** (Credit Valuation Adjustment) is the expected loss from counterparty default: `CVA = (1-R_C) * Σᵢ EE(tᵢ) * [Q_C(tᵢ₋₁) - Q_C(tᵢ)] * DF(tᵢ)`, where `Q_C(t) = exp(-λ_C·t)` is the counterparty's survival probability under a flat hazard rate `λ_C`, `R_C` its recovery rate, and `DF(t) = exp(-r·t)` the risk-free discount factor.
- **DVA** (Debit Valuation Adjustment) is the mirror using the bank's *own* hazard rate and `ENE`: `DVA = (1-R_B) * Σᵢ |ENE(tᵢ)| * [Q_B(tᵢ₋₁) - Q_B(tᵢ)] * DF(tᵢ)` — an expected *gain* to the bank's own valuation, since the bank's own default would extinguish an obligation it owed.
- **FVA** (Funding Valuation Adjustment) is the cost of funding expected positive exposure at a spread over the risk-free rate: `FVA = Σᵢ s_funding * EE(tᵢ) * DF(tᵢ) * Δtᵢ`.

### Formulas and Key Terms

```
V(t) = S(t) - K                                             (forward contract value, long position)

EE(t)  = E[max(V(t), 0)]                                    (Expected Positive Exposure)
ENE(t) = E[min(V(t), 0)]                                    (Expected Negative Exposure)

Q(t)  = e^(-λ·t)                                             (survival probability, flat hazard rate λ)
DF(t) = e^(-r·t)                                             (risk-free discount factor)

CVA = (1-R_C) · Σ_i EE(t_i) · [Q_C(t_(i-1)) - Q_C(t_i)] · DF(t_i)
DVA = (1-R_B) · Σ_i |ENE(t_i)| · [Q_B(t_(i-1)) - Q_B(t_i)] · DF(t_i)
FVA = Σ_i s_funding · EE(t_i) · DF(t_i) · Δt_i
```

- **Forward contract** — an agreement to buy the underlying at a fixed strike `K` at maturity; a long position's value at any earlier time `t` is `V(t) = S(t) - K`.
- **Exposure** — how much would be lost right now if the counterparty defaulted this instant; `EE(t)` and `ENE(t)` are its two one-sided halves.
- **Hazard rate (λ)** — the instantaneous default intensity: the (constant, under this appendix's flat-hazard model) rate at which default risk accrues per unit time. `λ_C` for the counterparty, `λ_B` for the bank's own credit.
- **Survival probability (Q(t))** — the probability of *not* having defaulted by time `t`; `Q(t) = e^(-λ·t)` under a flat hazard rate, so `Q(t_(i-1)) - Q(t_i)` is the probability of defaulting specifically *within* interval `i`.
- **Recovery rate (R)** — the fraction of exposure recovered in the event of default; `1 - R` is the *loss given default*, the multiplier on `EE`/`ENE` in the CVA/DVA formulas above.
- **Discount factor (DF(t))** — converts a cash flow at time `t` into its present value: `DF(t) = e^(-r·t)`, using the risk-free rate `r`.
- **Funding spread (s_funding)** — the spread over the risk-free rate a bank pays to fund a position; the multiplier on `EE(t)` in the FVA formula, since positive exposure must be funded.
- **CVA / DVA / FVA** — see Background above for what each one means; the formulas here are the same three equations, restated together for reference alongside the terms they're built from.

### Worked Example F.2.1 — A genuine exposure profile, and CVA/DVA/FVA from it

`02_xva_engine.rs` reuses `box_muller` again and extends the GBM update into a checkpointed simulator that records four quarterly prices per path instead of only a terminal one:

```
=== F.2: XVA and Its Variants ===

Instrument: a long forward contract, S0=100.00, K=100.00 (at-the-money), T=1.00y, r=3.00%
V(t) = S(t) - K.  Exposure profile from 200000 simulated paths, quarterly checkpoints:

t        EE(t)        ENE(t)       EE(t)+ENE(t)  
0.25     4.3690       -3.6382      0.7308        
0.50     6.4255       -4.9706      1.4549        
0.75     8.1167       -5.9087      2.2081        
1.00     9.6584       -6.6322      3.0262        

--- Independent cross-check: EE(T)+ENE(T) must equal E[S(T)]-K, and E[S(T)] must
    match the theoretical GBM real-world mean S0*exp(mu*T) ---
simulated  E[S(T)] = 103.0262
theoretical S0*exp(mu*T) = 103.0454  (relative diff 0.0187%)
EE(T)+ENE(T) = 3.0262  vs.  E[S(T)]-K = 3.0262  (must match: max(v,0)+min(v,0)=v identically)

--- Credit/funding assumptions ---
counterparty: flat hazard=0.0200, recovery=0.40
own (bank):   flat hazard=0.0100, recovery=0.40
funding spread over r: 0.0050

--- CVA (Credit Valuation Adjustment): expected loss from counterparty default ---
CVA = (1-R_C) * sum_i EE(t_i) * [Q_C(t_i-1)-Q_C(t_i)] * DF(t_i) = 0.0830

--- DVA (Debit Valuation Adjustment): expected GAIN from OUR OWN default ---
DVA = (1-R_B) * sum_i |ENE(t_i)| * [Q_B(t_i-1)-Q_B(t_i)] * DF(t_i) = 0.0309

--- FVA (Funding Valuation Adjustment): cost of funding expected positive exposure ---
FVA = sum_i funding_spread * EE(t_i) * DF(t_i) * dt_i = 0.0350

--- Hand cross-check of the FIRST interval's CVA term (t: 0.00 -> 0.25) ---
Q_C(0)=1.000000, Q_C(0.25)=0.995012, DF(0.25)=0.992528, EE(0.25)=4.3690
term = (1-0.40) * 4.3690 * (1.000000-0.995012) * 0.992528 = 0.012976

--- Net XVA adjustment to the trade's value (bank's perspective) ---
Net = -CVA + DVA - FVA = -0.0830 + 0.0309 - 0.0350 = -0.0870

--- Scope note: MVA and KVA ---
Margin Valuation Adjustment (MVA) and Capital Valuation Adjustment (KVA) are real,
widely-quoted XVA variants alongside CVA/DVA/FVA above, requiring an initial-margin
schedule and a regulatory-capital profile respectively -- inputs this appendix does
not model, so, honestly: they are named and defined here, not computed.
```

Genuinely compiled (zero warnings, both profiles) and run, for an at-the-money (`K = S0 = 100`) one-year forward, `r = 3%`, quarterly checkpoints, 200,000 paths — every digit matches the CUDA C++ edition's own genuinely-observed output exactly, the same bit-exact-machinery consequence Worked Example F.1.1 already established.

Two independent checks confirm the exposure profile itself before any credit or funding assumption is applied to it. First, `EE(t)+ENE(t)` must equal `E[V(t)]` identically, since `max(v,0)+min(v,0)=v` for any `v` — the table's own arithmetic confirms this at every checkpoint. Second, at the final checkpoint, `E[S(T)]` from the simulation (`103.0262`) is compared against the closed-form theoretical GBM mean `S0·exp(μT) = 103.0454`, a relative difference of `0.0187%` — evidence the checkpointed simulator is behaving like the same GBM process Chapter 22.4 established, not a new, unrelated one. The first interval's CVA contribution is then recomputed by hand from the same `EE(0.25)`, survival probabilities, and discount factor already printed, landing on the identical `0.012976` one of the four terms the full sum embeds.

`DVA` comes out smaller than `CVA` here specifically because the bank's own hazard rate (`1%`) was set lower than the counterparty's (`2%`) — a better-credit bank has a smaller DVA, all else equal, since DVA scales with the bank's *own* probability of defaulting. The net adjustment, `-CVA + DVA - FVA = -0.087`, is a genuine, if small, negative adjustment to this particular trade's value under these particular assumptions.

**Scope note — MVA and KVA.** Margin Valuation Adjustment (the funding cost of posting initial margin under bilateral or cleared margin rules) and Capital Valuation Adjustment (the cost of holding regulatory capital against the trade) are real, widely-quoted XVA variants alongside CVA/DVA/FVA above. Both require inputs this appendix does not model — an initial-margin schedule for MVA, a regulatory capital profile and cost-of-equity rate for KVA — so, honestly: they are named and defined here, not computed, rather than approximated with inputs invented for the occasion.

**Scope note — wrong-way risk.** The CVA formula above treats `EE(t)` and the counterparty's default probability as independent — `EE(tᵢ)` is computed once from the exposure profile, then multiplied by a survival-probability *difference* that never depends on which path produced that exposure. Real counterparties frequently violate this: **wrong-way risk** is the case where a counterparty's own default probability *rises* precisely when the bank's exposure to them is rising too (the standard example: an oil producer as the counterparty on a contract whose value depends on oil prices, where a sustained oil-price move that increases the bank's exposure is also exactly the scenario that stresses the producer's own credit). Modeling this genuinely requires coupling the hazard rate to the simulated path itself — computing a path-dependent `λ_C(t, S)` rather than the single flat `λ_C` used above — which this appendix does not build. The CVA figure computed here is therefore the *independent-exposure* baseline the industry starts from, not a wrong-way-risk-adjusted figure; where wrong-way risk is material, the real CVA is larger than what a flat-hazard model like this one reports.

**[COMMON TRAP]** It's tempting to compute CVA using the *terminal* exposure alone (`EE(T)`) rather than the full profile. That would ignore every default scenario before maturity — a counterparty that defaults at `t=0.25` when `EE(0.25)=4.37` is a genuinely different (and, here, smaller) loss than one that defaults at `T` when `EE(T)=9.66`, which is exactly why CVA is a *sum over the exposure profile*, weighted by the probability of default arriving in each specific interval, rather than a single point evaluated once.

## F.3 Complete Runnable Code

### File: 01_var_engine.rs

```rust
// F.1: Value-at-Risk and Its Variants.
//
// Value-at-Risk answers one question: at a given confidence level, how bad
// could tomorrow's loss get? This file computes it two genuinely different
// ways from the same underlying data and checks that they agree
// approximately, for a reason that is itself informative rather than a
// discrepancy to explain away.
//
// `box_muller` and `simulate_gbm_paths_host` below are reused byte-for-byte
// from Chapter 22.4's `04_monte_carlo_gbm.rs` -- the same Box-Muller
// normal-sampling technique and the same inline xorshift PRNG, at a 1-day
// horizon (`dt = 1/252`) instead of Chapter 22.4's full-year option-pricing
// horizon. This appendix needs no CUDA at all: every number below is
// ordinary host-side Rust arithmetic, exactly as it already was in
// Appendix C's tiling simulation.
use std::f32::consts::PI;

fn box_muller(u1: f32, u2: f32) -> f32 {
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * PI * u2).cos()
}

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

fn main() {
    println!("=== F.1: Value-at-Risk and Its Variants ===\n");

    const NUM_PATHS: usize = 200_000;
    let s0 = 100.0f32;
    let mu = 0.03f32;
    let sigma = 0.20f32;
    let horizon_days = 1.0f32;
    let dt_1day = horizon_days / 252.0f32;

    println!("--- Simulated (\"Monte Carlo\"/\"Historical\") VaR, 1-day horizon, 99% confidence ---");
    let mut one_day_prices = vec![0.0f32; NUM_PATHS];
    simulate_gbm_paths_host(&mut one_day_prices, s0, mu, sigma, dt_1day, 1, NUM_PATHS, 42u64);

    let mut pnl: Vec<f32> = one_day_prices.iter().map(|&s| s - s0).collect();
    pnl.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let confidence = 0.99f64;
    let var_index = ((1.0 - confidence) * NUM_PATHS as f64) as usize;
    let simulated_var_99 = -pnl[var_index];

    println!("holding: 1 unit of the underlying, S0={s0:.2}, mu=sigma={mu:.2}/{sigma:.2}, {NUM_PATHS} simulated 1-day scenarios");
    println!("worst-case index for 99% VaR: floor(0.01 * {NUM_PATHS}) = {var_index}");
    println!("P&L at that index: {:.4}  ->  simulated 99% 1-day VaR = {:.4}\n", pnl[var_index], simulated_var_99);

    println!("--- Parametric (Variance-Covariance) VaR, same horizon ---");
    let sigma_1day = sigma * dt_1day.sqrt();
    let z_99 = 2.326348f32;
    let parametric_var_99 = s0 * sigma_1day * z_99;
    println!("sigma_1day = sigma * sqrt(dt) = {sigma_1day:.6}, z_99 = {z_99:.6}");
    println!("parametric VaR = S0 * sigma_1day * z_99 = {parametric_var_99:.4}\n");

    println!("--- Cross-check: parametric vs. simulated, and why they should be CLOSE, not identical ---");
    let relative_diff = (parametric_var_99 - simulated_var_99).abs() / simulated_var_99;
    println!("relative difference: {:.4} ({:.2}%)", relative_diff, relative_diff * 100.0);
    println!("(GBM log-returns are exactly normally distributed by construction, but VaR here is");
    println!("measured on ARITHMETIC P&L (S_T - S0), which is log-normally, not normally,");
    println!("distributed -- parametric VaR's normal-P&L assumption is therefore an approximation");
    println!("even against this appendix's own fully-consistent GBM data, not a bug in either method.)\n");

    println!("--- Conditional VaR (CVaR / Expected Shortfall): the average loss BEYOND the VaR cutoff ---");
    let mut sum_tail_losses = 0.0f64;
    for i in 0..var_index {
        sum_tail_losses += -pnl[i] as f64;
    }
    let cvar_99 = (sum_tail_losses / var_index as f64) as f32;
    println!("averaging the {var_index} worst-case P&L values (indices 0..{}): CVaR_99 = {cvar_99:.4}", var_index - 1);
    println!(
        "CVaR >= VaR: {} ({cvar_99:.4} >= {simulated_var_99:.4})",
        if cvar_99 >= simulated_var_99 { "true" } else { "false" }
    );
    println!("(this must always hold: CVaR is the average of a set of losses EVERY one of which is");
    println!("at least as large as the VaR cutoff itself, by definition of which losses qualify.)");
}
```

### File: 02_xva_engine.rs

```rust
// F.2: XVA and Its Variants.
//
// VaR asks about the risk of holding a position. XVA asks a related but
// different question: given that the counterparty on the other side of a
// trade, or the bank itself, might default before the trade matures, what
// is that default risk worth, in the price of the trade itself? Answering
// it requires the trade's exposure profile over time, not just its
// terminal payoff -- this file extends Chapter 22.4's GBM machinery to
// record a path's value at four quarterly checkpoints, continuing each
// path's own state from one checkpoint to the next rather than
// resimulating from t=0 each time.
//
// `box_muller` is reused byte-for-byte from Chapter 22.4 / this appendix's
// own Section F.1.
use std::f32::consts::PI;

fn box_muller(u1: f32, u2: f32) -> f32 {
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * PI * u2).cos()
}

fn simulate_gbm_paths_checkpointed(
    checkpoint_prices: &mut [Vec<f32>],
    checkpoint_times: &[f32],
    s0: f32,
    mu: f32,
    sigma: f32,
    steps_per_checkpoint: i32,
    num_paths: usize,
    seed: u64,
) {
    let num_checkpoints = checkpoint_times.len();
    for idx in 0..num_paths {
        let mut s = s0;
        let mut state: u64 = seed.wrapping_add((idx as u64).wrapping_mul(7919)).wrapping_add(1);
        let mut t_prev = 0.0f32;
        for c in 0..num_checkpoints {
            let t_cur = checkpoint_times[c];
            let dt = (t_cur - t_prev) / steps_per_checkpoint as f32;
            for _step in 0..steps_per_checkpoint {
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
            checkpoint_prices[c][idx] = s;
            t_prev = t_cur;
        }
    }
}

fn main() {
    println!("=== F.2: XVA and Its Variants ===\n");

    const NUM_PATHS: usize = 200_000;
    const STEPS_PER_CHECKPOINT: i32 = 13;
    let s0 = 100.0f32;
    let mu = 0.03f32;
    let sigma = 0.20f32;
    let r = 0.03f32;
    let k = 100.0f32;

    let checkpoint_times = vec![0.25f32, 0.5, 0.75, 1.0];
    let nc = checkpoint_times.len();

    let mut checkpoint_prices: Vec<Vec<f32>> = (0..nc).map(|_| vec![0.0f32; NUM_PATHS]).collect();
    simulate_gbm_paths_checkpointed(&mut checkpoint_prices, &checkpoint_times, s0, mu, sigma, STEPS_PER_CHECKPOINT, NUM_PATHS, 42u64);

    println!(
        "Instrument: a long forward contract, S0={s0:.2}, K={k:.2} (at-the-money), T={:.2}y, r={:.2}%",
        checkpoint_times[nc - 1],
        r * 100.0
    );
    println!("V(t) = S(t) - K.  Exposure profile from {NUM_PATHS} simulated paths, quarterly checkpoints:\n");

    let mut ee = vec![0.0f32; nc];
    let mut ene = vec![0.0f32; nc];
    println!("{:<8} {:<12} {:<12} {:<14}", "t", "EE(t)", "ENE(t)", "EE(t)+ENE(t)");
    for c in 0..nc {
        let mut sum_pos = 0.0f64;
        let mut sum_neg = 0.0f64;
        for i in 0..NUM_PATHS {
            let v = checkpoint_prices[c][i] - k;
            if v > 0.0 {
                sum_pos += v as f64;
            } else {
                sum_neg += v as f64;
            }
        }
        ee[c] = (sum_pos / NUM_PATHS as f64) as f32;
        ene[c] = (sum_neg / NUM_PATHS as f64) as f32;
        println!("{:<8.2} {:<12.4} {:<12.4} {:<14.4}", checkpoint_times[c], ee[c], ene[c], ee[c] + ene[c]);
    }

    println!("\n--- Independent cross-check: EE(T)+ENE(T) must equal E[S(T)]-K, and E[S(T)] must");
    println!("    match the theoretical GBM real-world mean S0*exp(mu*T) ---");
    let mut sum_terminal = 0.0f64;
    for i in 0..NUM_PATHS {
        sum_terminal += checkpoint_prices[nc - 1][i] as f64;
    }
    let mean_terminal_simulated = (sum_terminal / NUM_PATHS as f64) as f32;
    let mean_terminal_theoretical = s0 * (mu * checkpoint_times[nc - 1]).exp();
    println!("simulated  E[S(T)] = {mean_terminal_simulated:.4}");
    println!(
        "theoretical S0*exp(mu*T) = {mean_terminal_theoretical:.4}  (relative diff {:.4}%)",
        100.0 * (mean_terminal_simulated - mean_terminal_theoretical).abs() / mean_terminal_theoretical
    );
    println!(
        "EE(T)+ENE(T) = {:.4}  vs.  E[S(T)]-K = {:.4}  (must match: max(v,0)+min(v,0)=v identically)\n",
        ee[nc - 1] + ene[nc - 1],
        mean_terminal_simulated - k
    );

    let hazard_counterparty = 0.02f32;
    let recovery_counterparty = 0.40f32;
    let hazard_own = 0.01f32;
    let recovery_own = 0.40f32;
    let funding_spread = 0.005f32;

    let survival = |hazard: f32, t: f32| (-hazard * t).exp();
    let discount = |t: f32| (-r * t).exp();

    println!("--- Credit/funding assumptions ---");
    println!("counterparty: flat hazard={hazard_counterparty:.4}, recovery={recovery_counterparty:.2}");
    println!("own (bank):   flat hazard={hazard_own:.4}, recovery={recovery_own:.2}");
    println!("funding spread over r: {funding_spread:.4}\n");

    let mut cva = 0.0f64;
    let mut dva = 0.0f64;
    let mut fva = 0.0f64;
    let mut t_prev = 0.0f32;
    for c in 0..nc {
        let t_cur = checkpoint_times[c];
        let q_c_prev = survival(hazard_counterparty, t_prev);
        let q_c_cur = survival(hazard_counterparty, t_cur);
        let q_b_prev = survival(hazard_own, t_prev);
        let q_b_cur = survival(hazard_own, t_cur);
        let df = discount(t_cur);
        let dt = t_cur - t_prev;

        cva += (1.0 - recovery_counterparty) as f64 * ee[c] as f64 * (q_c_prev - q_c_cur) as f64 * df as f64;
        dva += (1.0 - recovery_own) as f64 * (-ene[c]) as f64 * (q_b_prev - q_b_cur) as f64 * df as f64;
        fva += funding_spread as f64 * ee[c] as f64 * df as f64 * dt as f64;

        t_prev = t_cur;
    }

    println!("--- CVA (Credit Valuation Adjustment): expected loss from counterparty default ---");
    println!("CVA = (1-R_C) * sum_i EE(t_i) * [Q_C(t_i-1)-Q_C(t_i)] * DF(t_i) = {cva:.4}\n");

    println!("--- DVA (Debit Valuation Adjustment): expected GAIN from OUR OWN default ---");
    println!("DVA = (1-R_B) * sum_i |ENE(t_i)| * [Q_B(t_i-1)-Q_B(t_i)] * DF(t_i) = {dva:.4}\n");

    println!("--- FVA (Funding Valuation Adjustment): cost of funding expected positive exposure ---");
    println!("FVA = sum_i funding_spread * EE(t_i) * DF(t_i) * dt_i = {fva:.4}\n");

    println!("--- Hand cross-check of the FIRST interval's CVA term (t: 0.00 -> 0.25) ---");
    {
        let q_c_prev = survival(hazard_counterparty, 0.0);
        let q_c_cur = survival(hazard_counterparty, 0.25);
        let df = discount(0.25);
        let term1 = (1.0 - recovery_counterparty) as f64 * ee[0] as f64 * (q_c_prev - q_c_cur) as f64 * df as f64;
        println!("Q_C(0)={q_c_prev:.6}, Q_C(0.25)={q_c_cur:.6}, DF(0.25)={df:.6}, EE(0.25)={:.4}", ee[0]);
        println!("term = (1-0.40) * {:.4} * ({q_c_prev:.6}-{q_c_cur:.6}) * {df:.6} = {term1:.6}", ee[0]);
    }

    println!("\n--- Net XVA adjustment to the trade's value (bank's perspective) ---");
    let net_xva = -cva + dva - fva;
    println!("Net = -CVA + DVA - FVA = -{cva:.4} + {dva:.4} - {fva:.4} = {net_xva:.4}\n");

    println!("--- Scope note: MVA and KVA ---");
    println!("Margin Valuation Adjustment (MVA) and Capital Valuation Adjustment (KVA) are real,");
    println!("widely-quoted XVA variants alongside CVA/DVA/FVA above, requiring an initial-margin");
    println!("schedule and a regulatory-capital profile respectively -- inputs this appendix does");
    println!("not model, so, honestly: they are named and defined here, not computed.");
}
```

## Appendix Summary

This appendix computed Value-at-Risk two genuinely different ways — a simulated/historical quantile extraction and a closed-form parametric estimate — from the same GBM-simulated data Chapter 22.4 established, landing close but not identical for a specific, checked reason: arithmetic P&L on a log-normal terminal price is only approximately normal, even though GBM's log-returns are exactly normal by construction. Conditional VaR's `CVaR ≥ VaR` invariant was verified rather than assumed, directly from the same sorted tail. Section F.2 extended that same machinery to record an exposure profile across time rather than only at maturity, computing CVA, DVA, and FVA from genuine `EE(t)`/`ENE(t)` values cross-checked against the GBM process's own closed-form mean, with honest scope notes that MVA and KVA are named but not computed, since this appendix has no initial-margin schedule or capital profile to model them from, and that the CVA figure itself assumes exposure and counterparty default are independent — real wrong-way risk, where the two are correlated, would make the true CVA larger than what this flat-hazard model reports. Every number in both sections traces back to Chapter 22.4's `box_muller`/xorshift/GBM-update machinery, reused byte-for-byte rather than reimplemented — a property worth remembering if this book's own simulation parameters are ever revisited, since a change to that chapter's PRNG or GBM update would propagate its effect identically into every VaR and XVA number here.

## Self-Check Questions

1. Section F.1 reports simulated VaR (2.8665) and parametric VaR (2.9309) as "close but not identical," attributing the gap to arithmetic P&L being log-normal rather than normal. If the same VaR calculation were instead applied to LOG-returns (`ln(S_T/S0)`) rather than arithmetic P&L (`S_T - S0`), would you expect the simulated-vs-parametric gap to shrink, grow, or stay about the same? Justify your answer using what GBM assumes is normally distributed by construction.
2. Section F.2 computes `DVA < CVA` here specifically because the bank's own hazard rate (1%) was set lower than the counterparty's (2%). Using the DVA formula's own structure (`DVA = (1-R_B) * Σᵢ |ENE(tᵢ)| * [Q_B(tᵢ₋₁)-Q_B(tᵢ)] * DF(tᵢ)`), explain what would happen to DVA, holding everything else fixed, if the bank's OWN credit quality got WORSE (a higher hazard rate) — and why a bank's DVA increasing when its own credit gets worse has historically been considered a controversial property of DVA accounting.
3. Section F.2's exposure profile shows `EE(t)` growing from 4.3690 at `t=0.25` to 9.6584 at `t=1.00`, roughly proportional to `sqrt(t)` rather than growing linearly with `t`. Using what you know about GBM's volatility term (`sigma * sqrt(dt)` in the update step), explain why an at-the-money forward's expected positive exposure would be expected to grow with the SQUARE ROOT of time rather than linearly.

### Worked Solutions

1. The gap would shrink, likely substantially. GBM's defining assumption is that LOG-returns, `ln(S_T/S0) = (μ - σ²/2)T + σ√T·Z` for standard normal `Z`, are exactly normally distributed by construction — that's the entire mechanism the `box_muller`-driven update step implements. Arithmetic P&L, `S_T - S0`, is a nonlinear (exponential) transformation of that normal log-return, which is only approximately normal itself, and that approximation is precisely what Section F.1 identifies as the source of the 2.25% gap. Applying parametric VaR's normal-distribution assumption directly to log-returns instead removes that transformation entirely — parametric VaR would then be assuming normality for the exact quantity GBM actually makes normal, so the simulated and parametric figures should coincide far more closely, with the small remaining gap attributable only to finite-sample Monte Carlo noise from using 200,000 paths rather than the true underlying distribution.
2. Holding `R_B`, `ENE(t)`, and `DF(t)` fixed, a higher `hazard_own` makes `Q_B(t)` fall faster with time, which makes each `[Q_B(tᵢ₋₁)-Q_B(tᵢ)]` term — the incremental probability of the bank itself defaulting in that specific interval — LARGER, which makes DVA larger. This is exactly the property that has made DVA controversial in practice: it means a bank's own reported earnings can go UP as its own credit quality gets WORSE, since a higher own-default probability increases the "expected gain from extinguishing our own obligations" DVA represents — an accounting outcome many practitioners and regulators have criticized as counterintuitive (a firm should not look more profitable purely because the market believes it is more likely to fail), which is part of why post-financial-crisis accounting and regulatory treatments of DVA have been repeatedly revisited.
3. GBM's volatility contribution to each step accumulates through the `sigma * sqrt(dt)` term, and for INDEPENDENT increments, variance is additive across time while standard deviation — the `sqrt` of variance — is not: the standard deviation of the price after time `t` scales with `sqrt(t)`, not `t` itself (this is the same square-root-of-time scaling behind the `sigma_1day = sigma * sqrt(dt)` conversion Section F.1 uses to go from an annual to a 1-day volatility). Since an at-the-money forward's exposure is driven almost entirely by how far `S(t)` has plausibly wandered from `K = S0` — which is governed by that same standard deviation, not the mean drift alone at these parameter magnitudes — `EE(t)`, being roughly proportional to the spread of the distribution at time `t`, inherits that same `sqrt(t)` growth rather than growing linearly, which is exactly the shape the table's own numbers (4.37, 6.43, 8.12, 9.66 at `t=0.25, 0.5, 0.75, 1.0`) trace out.
