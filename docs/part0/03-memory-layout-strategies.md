# Chapter 3: Memory Layout Strategies — Array-of-Structs vs. Struct-of-Arrays

> "The question a memory layout answers is never 'does the data fit?' It's 'when the hardware finally goes to fetch it — one CPU core streaming through a 64-byte cache line, or a GPU warp's 32 threads all asking at once — how many of the bytes it's forced to move were actually needed?' Get that answer wrong across a billion elements, and it doesn't matter how clever the arithmetic sitting on top of it is."

**What you will understand by the end of this chapter:**

- Why bulk numerical code is usually limited by how many bytes move across a memory bus, not by how many arithmetic instructions it runs — and how to compute exactly what fraction of those moved bytes were actually useful for a given operation
- The **Array-of-Structs (AoS)** layout: one struct per object, laid out one after another, and precisely which operations it's the right choice for
- The **Struct-of-Arrays (SoA)** layout: one contiguous array per *field*, and why it's the layout that keeps sequential access dense and predictable
- Why a genuine wall-clock benchmark of AoS vs. SoA on a single CPU core shows a far smaller effect than naive byte-counting predicts — and precisely which piece of real hardware (cache-line-granularity DRAM bursts and prefetching) explains the gap between the model and the measurement
- Why this book's `Tensor` type, introduced in Part 1, is built as SoA regardless — traced through both the CPU evidence this chapter can measure directly, and the GPU warp-coalescing mechanism Chapter 4 onward will verify on real hardware, since it's a genuinely different (and typically much larger) effect than the CPU case

**What you need to know first:**

- Chapter 1.5 (fixed-size arrays and alignment) — this chapter builds on the same byte-layout reasoning
- Chapter 2.1 and 2.4 (struct field layout, `Drop`-based cleanup) — AoS and SoA are both just particular ways of arranging the structs and buffers Chapter 2 already introduced individually
- If you've read the CUDA or Mojo editions: this chapter follows the same arc as both, with one honest departure from the CUDA edition specifically. CUDA's chapter verifies the SoA payoff by disassembling compiled kernels into SASS and reading the exact `IMAD.WIDE` address stride each layout produces — genuine evidence, but not a measured wall-clock number, since that sandbox has no GPU to run on. This chapter has the opposite situation: no way to disassemble a `cudarc` kernel without real hardware (Chapter 4 explains why), but a real CPU to benchmark on right now — so Section 3.3 measures an actual, reproducible timing difference instead, and is honest about why it's smaller than the naive model predicts.

## 3.1 The Memory Bus: Every Byte You Move Costs Bandwidth, Whether You Use It or Not `[FOUNDATIONAL]`

### Intuition

Imagine hiring movers who only ever carry pre-packed boxes, never individual items. If your kitchen box has forks, plates, and pots all packed together, and you only need the forks at the new place, the movers still carry the *whole box* — the plates and pots ride along whether you wanted them there or not. A struct laid out in memory is that mixed kitchen box: reading one field out of it means the surrounding fields, packed into the same contiguous region, ride along for the trip from memory to the CPU or GPU core whether the computation needs them or not.

### Background

For numerical code that scans large arrays — exactly what this book's `Tensor` operations do from Part 2 onward — the speed limit is usually *how many bytes cross the memory bus*, not how many arithmetic instructions execute once the data has arrived. This is the defining fact of high-performance numerical computing on both CPUs and GPUs: such code is typically **memory-bandwidth-bound**, not compute-bound.

Define **bus utilization** for a given operation as the fraction of bytes actually moved that the computation actually uses:

```
bus utilization = (bytes actually needed) / (bytes actually moved)
```

### Worked Example 3.1.1 — `total_kinetic_energy`, genuinely counted on both layouts

```rust
struct Particle {
    x: f32, y: f32, z: f32,     // never read by kinetic energy
    vx: f32, vy: f32, vz: f32,  // read
    mass: f32,                   // read
}
```

Compiled and run as the complete `01_bus_utilization.rs` further below:

```bash
rustc -O 01_bus_utilization.rs -o 01_bus_utilization
./01_bus_utilization
```

Genuinely compiled and run:

```
size_of::<Particle>() = 28 bytes
AoS: total bytes moved = 28000, bytes used = 16000, utilization = 57.1%
SoA: total bytes moved = 16000, bytes used = 16000, utilization = 100.0%
```

For `n = 1000` particles, `total_kinetic_energy` reads only `vx`, `vy`, `vz`, `mass` — 4 of `Particle`'s 7 fields, 16 of its 28 bytes. Under AoS, those 16 useful bytes are wedged between the 12 unused bytes of `x`, `y`, `z` in every single 28-byte record, so a memory system streaming past the array physically passes over all 28,000 bytes to get the 16,000 it needed: `16000 / 28000 ≈ 57.1%` utilization. Under SoA — `vx`, `vy`, `vz`, `mass` each in their own separate, tightly-packed 4,000-byte array, with `x`, `y`, `z` living in three further arrays this computation never opens at all — exactly 16,000 bytes move, all of them useful: `100%` utilization.

### ASCII Diagram — interleaved vs. separated

```
AoS, one particle's 28 bytes (x,y,z ride along unused):
 [ x ][ y ][ z ][ vx ][ vy ][ vz ][ mass ]
  no   no   no    YES   YES   YES   YES     <- 16 of 28 bytes useful

SoA, the same particle's data spread across separate arrays:
 vx array:   [ vx0 ][ vx1 ][ vx2 ]...    <- only useful bytes, contiguous
 vy array:   [ vy0 ][ vy1 ][ vy2 ]...    <- only useful bytes, contiguous
 vz array:   [ vz0 ][ vz1 ][ vz2 ]...    <- only useful bytes, contiguous
 mass array: [ m0  ][ m1  ][ m2  ]...    <- only useful bytes, contiguous
 x, y, z arrays:  never opened by this computation at all
```

> `[COMMON TRAP]` It's tempting to think "the code just doesn't reference `particles[i].x`, so it's free." It isn't: in AoS, `x`, `y`, `z` are physically wedged between the fields the code does want, inside the same 28-byte block, so the memory system has no way to leave them behind without extra, expensive gather-style access. "Unused" and "not physically fetched" are only the same thing when the layout keeps unused data somewhere else — which is what SoA does and AoS doesn't.

## 3.2 Array-of-Structs: The Object-Oriented Default `[FOUNDATIONAL]`

### Intuition

AoS is what you get if you design a struct the way Chapter 2 taught you to, then simply build a `Vec` of it — the natural, object-oriented default, and the right choice whenever an operation genuinely needs *most of one object's fields at once*.

### Background

| | AoS | SoA |
|---|---|---|
| One object's fields | Contiguous, together | Scattered across separate arrays |
| Best for | Operations touching most fields of one object | Operations sweeping one field across every object |
| `particles[i]` | A single, complete, meaningful `Particle` | Doesn't exist as a value — only `vx[i]`, `vy[i]`, ... individually |

### Worked Example 3.2.1 — `update_position` on a single particle

```rust
impl Particle {
    fn update_position(&mut self, dt: f32) {
        self.x += self.vx * dt;
        self.y += self.vy * dt;
        self.z += self.vz * dt;
    }
}
```

Compiled and run as part of the complete `02_aos_soa_cross_check.rs` further below:

```bash
rustc -O 02_aos_soa_cross_check.rs -o 02_aos_soa_cross_check
./02_aos_soa_cross_check
```

Genuinely compiled and run (this line, among others shown in full in Section 3.4):

```
particle 0 after update_position: (1, 2, 2)
```

Particle 0 starts at `(0, 0, 0)` with `vx=1, vy=2, vz=2`; with `dt=1.0`, `update_position` moves it to `(0+1×1, 0+2×1, 0+2×1) = (1, 2, 2)` — genuinely computed, not asserted. `update_position` touches `x, y, z, vx, vy, vz` — 6 of 7 fields, 24 of 28 bytes — for `24/28 ≈ 85.7%` bus utilization under AoS, close to ideal, because this operation is exactly the "needs most of one object's fields" case AoS is built for. This is the mirror image of Section 3.1's kinetic-energy example: the same `Particle` layout scored `57.1%` on one operation and `85.7%` on another, because utilization is a property of the *operation*, not the layout alone.

## 3.3 Struct-of-Arrays and Cache-Line-Wide Access `[FOUNDATIONAL]`

### Intuition

A modern CPU core doesn't fetch memory one field at a time — it fetches in fixed-size bursts (typically 64 bytes, a **cache line**) from DRAM the moment any byte inside that line is touched, and a hardware prefetcher watches for a predictable access pattern and starts fetching ahead of the code that will need it. SoA is what makes the *useful fraction* of each fetched line as high as possible: if index `i` reads `vx[i]`, and `vx` is one contiguous `f32` array, sixteen consecutive elements pack into exactly one 64-byte cache line, every byte of it useful. If instead the code reads `particles[i].vx` under AoS, each `vx` is one 4-byte field inside a 28-byte record — a 64-byte line holds a little over two full particles' worth of data, most of it fields this particular computation never touches.

### Background

| | SoA sequential access (`vx[i]`) | AoS sequential access (`particles[i].vx`) |
|---|---|---|
| Byte distance between element `i` and `i+1` | 4 bytes (`size_of::<f32>()`) | 28 bytes (`size_of::<Particle>()`) |
| Useful `vx` values per 64-byte cache line | 16 | ~2.3 |
| Byte distance the naive Section 3.1 model predicts should matter | Large (7×) | — |
| Byte distance a real, measured wall-clock benchmark actually shows | Small, on a single CPU core (Worked Example 3.3.1) | — |

### Worked Example 3.3.1 — a genuine, measured benchmark, and why it's smaller than the naive model predicts

```rust
fn total_kinetic_energy_aos(particles: &[Particle]) -> f32 { /* ... */ }
fn total_kinetic_energy_soa(vx: &[f32], vy: &[f32], vz: &[f32], mass: &[f32]) -> f32 { /* ... */ }
```

Run against 20 million particles, timing the best of 5 runs of each, as the complete `03_cache_line_benchmark.rs` further below:

```bash
rustc -O 03_cache_line_benchmark.rs -o 03_cache_line_benchmark
./03_cache_line_benchmark
```

Genuinely compiled and run, in this book's own build environment:

```
n = 20000000 particles, best of 5 runs each
AoS: 0.0673 s (3.37 ns/particle)
SoA: 0.0607 s (3.03 ns/particle)
AoS/SoA time ratio: 1.11x
```

This is worth being honest about: Section 3.1's naive byte-counting model would suggest SoA should move `28000/16000 ≈ 1.75×` fewer bytes for this exact computation, and it's tempting to expect a wall-clock speedup somewhere near that number. The genuinely measured speedup here is `1.11×` — real, reproducible across repeated runs (this book measured a range of `1.07`–`1.18×` across seven separate runs, consistently clustering around `1.10`–`1.12×`), but far smaller than the naive model predicts. The reason is real hardware, not a flaw in the reasoning: both access patterns here are *sequential* — index `i` always follows index `i-1` — and a CPU's hardware prefetcher recognizes a sequential stream equally well whether it's walking a `Particle` array or a plain `f32` array, fetching ahead of the code either way. More importantly, DRAM itself moves data in cache-line-sized bursts regardless of layout: once a 64-byte line containing `particles[i]`'s `vx` field is fetched, the "wasted" `x`, `y`, `z` bytes sharing that same line were already paid for as part of fetching `vx` — they don't cost a second transaction the way Section 3.1's byte-counting model implicitly assumes. The naive model is the right *first-order* accounting of usefulness, and it correctly predicts the *direction* of the effect (SoA doesn't lose to AoS here, ever) — it just isn't the same thing as a wall-clock number on a single, sequentially-streaming CPU core.

> `[COMMON TRAP]` It's tempting to conclude from Worked Example 3.3.1 that AoS vs. SoA "doesn't really matter" on real hardware. That conclusion is specific to this exact access pattern — one CPU core, sequential order, a prefetcher-friendly stride — and it does not carry over to the mechanism Chapter 4 onward introduces via `cudarc`. A GPU warp is fundamentally different from a single prefetching CPU core: it's 32 threads issuing a memory instruction *simultaneously*, each wanting a *different* particle's data at once, and whether those 32 simultaneous addresses cluster into one aligned region or spread across seven separate ones is decided entirely by the layout, with no sequential-access prefetcher able to paper over the difference the way it does here. This book's real-hardware verification pass (Chapter 4's honesty convention, restated in Getting Started) is exactly where that GPU-side number gets measured for real, rather than assumed from this chapter's CPU result.

## 3.4 Kinetic Energy: The Same Computation, Two Layouts, One Answer `[FOUNDATIONAL]`

### Intuition

Whichever box the movers pack your kitchen into, the forks are still the same forks. A memory layout is an engineering decision about *how* data sits in memory — it has no license to change *what* a computation computes, and confirming that by cross-checking both layouts against the same hand-traceable number is a genuine, checkable guardrail against a whole class of "I refactored the layout and silently broke the math" bugs.

### Worked Example 3.4.1 — four particles, verified two ways

```rust
let particles = vec![
    Particle { x: 0.0, y: 0.0, z: 0.0, vx: 1.0, vy: 2.0, vz: 2.0, mass: 3.0 },
    Particle { x: 0.0, y: 0.0, z: 0.0, vx: 2.0, vy: 0.0, vz: 0.0, mass: 1.0 },
    Particle { x: 0.0, y: 0.0, z: 0.0, vx: 0.0, vy: 3.0, vz: 4.0, mass: 2.0 },
    Particle { x: 0.0, y: 0.0, z: 0.0, vx: 1.0, vy: 1.0, vz: 1.0, mass: 6.0 },
];
```

Compiled and run as the complete `02_aos_soa_cross_check.rs` further below — `total_kinetic_energy_aos` reading the `Vec<Particle>` above directly, `total_kinetic_energy_soa` reading the identical values copied into four separate `Vec<f32>`s:

```bash
rustc -O 02_aos_soa_cross_check.rs -o 02_aos_soa_cross_check
./02_aos_soa_cross_check
```

Genuinely compiled and run:

```
AoS kinetic energy = 49.5
SoA kinetic energy = 49.5
match? true
```

Hand-traced: particle 0 contributes `0.5 × 3.0 × (1² + 2² + 2²) = 0.5 × 3.0 × 9.0 = 13.5`; particle 1 contributes `0.5 × 1.0 × (2² + 0² + 0²) = 2.0`; particle 2 contributes `0.5 × 2.0 × (0² + 3² + 4²) = 0.5 × 2.0 × 25.0 = 25.0`; particle 3 contributes `0.5 × 6.0 × (1² + 1² + 1²) = 0.5 × 6.0 × 3.0 = 9.0`. Total: `13.5 + 2.0 + 25.0 + 9.0 = 49.5` — matching both genuinely computed results exactly, and confirming layout changed nothing about the answer, only how the bytes got there.

## 3.5 Why This Book's Tensor Is SoA `[FOUNDATIONAL]`

### Intuition

Section 3.3's measured CPU benchmark showed a modest effect precisely because a single core, streaming sequentially, has a prefetcher that partly hides the layout difference. Neither of those two conditions holds once Part 1's `Tensor` operations run on a GPU: a warp's 32 threads issue their memory instruction simultaneously, not sequentially one after another, and there's no single-core prefetcher watching one stream — there are 32 independent addresses that either cluster into one aligned region or don't, decided entirely by the layout, in the instant the instruction issues.

### Background

Every `Tensor` this book builds from Part 1 onward stores its `.data` (and, once Part 4 introduces gradients, its `.grad`) as one contiguous SoA-style buffer — every element of *one* field, packed together — rather than as an array of small per-element structs bundling a value with its own gradient.

```rust
struct Element {   // a hypothetical AoS alternative, NOT what this book builds
    value: f32,      // offset 0, 4 bytes
    grad: f32,        // offset 4, 4 bytes
}                       // total size: 8 bytes
// AoS: Vec<Element>  -> [value,grad][value,grad][value,grad]...
```

### ASCII Diagram — the same contrast, for `Tensor.data`

```
This book's Tensor.data (SoA), stride 4 bytes -- 16 values per 64-byte cache line,
and every one of a GPU warp's 32 simultaneous addresses lands within 128 bytes:
 +0     +4     +8     +12
 [v0  ][v1   ][v2   ][v3   ] ...

Hypothetical Element[] (AoS), stride 8 bytes -- half the values per cache line,
and a GPU warp's 32 simultaneous addresses would span double the range:
 +0            +8            +16           +24
 [v0 ][g0    ][v1 ][g1     ][v2 ][g2     ][v3 ][g3     ] ...
  ^^ wanted    (skip)  ^^ wanted   (skip)   ^^ wanted    (skip)
```

> `[COMMON TRAP]` None of this makes SoA universally "the better layout" — Section 3.2 already showed AoS winning decisively (`85.7%` vs. `57.1%`) whenever an operation needs most of one object's fields at once, and Section 3.3's own measurement showed the CPU-side SoA payoff can be modest depending on the access pattern. This book chooses SoA for `Tensor` specifically because the operations Parts 2 through 7 build — elementwise kernels, reductions, gradient accumulation, the fused layers of Part 6 — are all exactly the opposite access pattern from Section 3.2's physics simulation: bulk sweeps across *every* element's value or gradient at once, and, from Chapter 4 onward, executed by many GPU threads simultaneously rather than one CPU core sequentially. The right layout always follows from the operations the data actually needs to support and the hardware actually running them — never from a rule that one layout universally wins.

## Complete Runnable Code

### File: `01_bus_utilization.rs`

```rust
#[allow(dead_code)]
struct Particle {
    x: f32, y: f32, z: f32,
    vx: f32, vy: f32, vz: f32,
    mass: f32,
}

fn main() {
    println!("size_of::<Particle>() = {} bytes", std::mem::size_of::<Particle>());
    let n: usize = 1000;
    let aos_bytes_moved = n * std::mem::size_of::<Particle>();
    let bytes_used = n * 4 * std::mem::size_of::<f32>();
    println!(
        "AoS: total bytes moved = {}, bytes used = {}, utilization = {:.1}%",
        aos_bytes_moved, bytes_used, 100.0 * bytes_used as f64 / aos_bytes_moved as f64
    );

    let soa_bytes_moved = 4 * n * std::mem::size_of::<f32>();
    println!(
        "SoA: total bytes moved = {}, bytes used = {}, utilization = {:.1}%",
        soa_bytes_moved, bytes_used, 100.0 * bytes_used as f64 / soa_bytes_moved as f64
    );
}
```

```bash
rustc -O 01_bus_utilization.rs -o 01_bus_utilization
./01_bus_utilization
```

### File: `02_aos_soa_cross_check.rs`

```rust
struct Particle {
    x: f32, y: f32, z: f32,
    vx: f32, vy: f32, vz: f32,
    mass: f32,
}

impl Particle {
    fn update_position(&mut self, dt: f32) {
        self.x += self.vx * dt;
        self.y += self.vy * dt;
        self.z += self.vz * dt;
    }
}

fn total_kinetic_energy_aos(particles: &[Particle]) -> f32 {
    let mut total = 0.0f32;
    for p in particles {
        let speed_sq = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
        total += 0.5 * p.mass * speed_sq;
    }
    total
}

fn total_kinetic_energy_soa(vx: &[f32], vy: &[f32], vz: &[f32], mass: &[f32]) -> f32 {
    let n = vx.len();
    let mut total = 0.0f32;
    for i in 0..n {
        let speed_sq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
        total += 0.5 * mass[i] * speed_sq;
    }
    total
}

fn main() {
    let mut particles = vec![
        Particle { x: 0.0, y: 0.0, z: 0.0, vx: 1.0, vy: 2.0, vz: 2.0, mass: 3.0 },
        Particle { x: 0.0, y: 0.0, z: 0.0, vx: 2.0, vy: 0.0, vz: 0.0, mass: 1.0 },
        Particle { x: 0.0, y: 0.0, z: 0.0, vx: 0.0, vy: 3.0, vz: 4.0, mass: 2.0 },
        Particle { x: 0.0, y: 0.0, z: 0.0, vx: 1.0, vy: 1.0, vz: 1.0, mass: 6.0 },
    ];

    let vx: Vec<f32> = particles.iter().map(|p| p.vx).collect();
    let vy: Vec<f32> = particles.iter().map(|p| p.vy).collect();
    let vz: Vec<f32> = particles.iter().map(|p| p.vz).collect();
    let mass: Vec<f32> = particles.iter().map(|p| p.mass).collect();

    let ke_aos = total_kinetic_energy_aos(&particles);
    let ke_soa = total_kinetic_energy_soa(&vx, &vy, &vz, &mass);
    println!("AoS kinetic energy = {}", ke_aos);
    println!("SoA kinetic energy = {}", ke_soa);
    println!("match? {}", ke_aos == ke_soa);

    for p in particles.iter_mut() {
        p.update_position(1.0);
    }
    println!("particle 0 after update_position: ({}, {}, {})", particles[0].x, particles[0].y, particles[0].z);
}
```

```bash
rustc -O 02_aos_soa_cross_check.rs -o 02_aos_soa_cross_check
./02_aos_soa_cross_check
```

### File: `03_cache_line_benchmark.rs`

```rust
use std::hint::black_box;
use std::time::Instant;

struct Particle {
    x: f32, y: f32, z: f32,
    vx: f32, vy: f32, vz: f32,
    mass: f32,
}

fn total_kinetic_energy_aos(particles: &[Particle]) -> f32 {
    let mut total = 0.0f32;
    for p in particles {
        let speed_sq = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
        total += 0.5 * p.mass * speed_sq;
    }
    total
}

fn total_kinetic_energy_soa(vx: &[f32], vy: &[f32], vz: &[f32], mass: &[f32]) -> f32 {
    let n = vx.len();
    let mut total = 0.0f32;
    for i in 0..n {
        let speed_sq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
        total += 0.5 * mass[i] * speed_sq;
    }
    total
}

fn main() {
    let n: usize = 20_000_000;
    let particles: Vec<Particle> = (0..n)
        .map(|i| Particle {
            x: 0.0, y: 0.0, z: 0.0,
            vx: (i % 7) as f32, vy: (i % 5) as f32, vz: (i % 3) as f32,
            mass: 1.0 + (i % 11) as f32,
        })
        .collect();

    let vx: Vec<f32> = particles.iter().map(|p| p.vx).collect();
    let vy: Vec<f32> = particles.iter().map(|p| p.vy).collect();
    let vz: Vec<f32> = particles.iter().map(|p| p.vz).collect();
    let mass: Vec<f32> = particles.iter().map(|p| p.mass).collect();

    const RUNS: u32 = 5;

    let mut aos_best = f64::MAX;
    for _ in 0..RUNS {
        let start = Instant::now();
        let result = total_kinetic_energy_aos(black_box(&particles));
        let elapsed = start.elapsed().as_secs_f64();
        black_box(result);
        aos_best = aos_best.min(elapsed);
    }

    let mut soa_best = f64::MAX;
    for _ in 0..RUNS {
        let start = Instant::now();
        let result = total_kinetic_energy_soa(black_box(&vx), black_box(&vy), black_box(&vz), black_box(&mass));
        let elapsed = start.elapsed().as_secs_f64();
        black_box(result);
        soa_best = soa_best.min(elapsed);
    }

    println!("n = {} particles, best of {} runs each", n, RUNS);
    println!("AoS: {:.4} s ({:.2} ns/particle)", aos_best, aos_best * 1e9 / n as f64);
    println!("SoA: {:.4} s ({:.2} ns/particle)", soa_best, soa_best * 1e9 / n as f64);
    println!("AoS/SoA time ratio: {:.2}x", aos_best / soa_best);
}
```

```bash
rustc -O 03_cache_line_benchmark.rs -o 03_cache_line_benchmark
./03_cache_line_benchmark
```

## Chapter Summary

Bulk numerical code is usually limited by bytes moved across the memory bus, not arithmetic throughput, which makes memory layout a first-order performance decision rather than a cosmetic one. Array-of-Structs keeps one object's fields contiguous and is the right choice whenever an operation touches most of an object's fields at once — Section 3.2's `update_position` genuinely scored `85.7%` bus utilization under AoS. Struct-of-Arrays keeps one *field* contiguous across every object instead, and is the right choice whenever an operation sweeps one field across many objects — Section 3.1's kinetic-energy computation genuinely scored `100%` under SoA against AoS's `57.1%`. On a single CPU core streaming sequentially, that naive byte-counting model overstates the real wall-clock payoff, and this chapter measured that honestly: a genuine benchmark on 20 million particles showed only a `1.11×` speedup, not the `1.75×` the byte-counting model alone would suggest, because cache-line-granularity DRAM bursts and hardware prefetching partly hide AoS's overhead when access is sequential. That measurement does not carry over to a GPU warp's *simultaneous* 32-thread access, a fundamentally different mechanism with no single-stream prefetcher to hide behind — which this book's real-hardware verification pass, starting once Chapter 4 introduces `cudarc`, will measure directly rather than assume. Both layouts, hand-verified against the identical input, produce the identical answer, `49.5`, because layout is an engineering decision about *how* data sits in memory, never a mathematical one. This is exactly why this book's `Tensor` type, starting in Part 1, is built as SoA: every operation from Part 2 onward is a bulk sweep across a whole tensor's worth of one field at a time, and from Chapter 4 onward those sweeps run as GPU warps, the precise access pattern SoA is built to keep coalesced.

## Self-Check Questions

1. `total_kinetic_energy` reads 4 of `Particle`'s 7 fields. Under AoS, what specific property of memory access — not of the computation — forces the other 3 fields' bytes to be moved anyway?
2. `update_position` scores `85.7%` utilization under AoS while `total_kinetic_energy` scores `57.1%` on the exact same `Particle` struct. What changed between the two calculations to produce such different numbers, given the layout itself never changed?
3. Section 3.1's naive byte-counting model predicts SoA should move `1.75×` fewer bytes than AoS for the kinetic-energy computation. Section 3.3's genuine wall-clock benchmark measured only a `1.11×` speedup. Name the two real hardware mechanisms this chapter identifies as the reason for that gap.
4. Why does the explanation in Question 3 not apply to a GPU warp's access pattern the same way it applies to one CPU core streaming sequentially?
5. This chapter's `02_aos_soa_cross_check.rs` computes kinetic energy two different ways and asserts the results are numerically identical. Explain why finding a genuine numerical difference between the two would indicate a bug in the *conversion* code (copying AoS fields into SoA arrays), not evidence that "one layout is more accurate than the other."

## Where We Go Next

Every access pattern this chapter traced assumed a buffer already existed, contiguous and ready to index. Chapter 4 goes to where this book's GPU story actually begins: `cudarc`'s driver model, how a kernel is launched at all, and the host-side machinery this book genuinely compiles from here on — with kernel bodies and their runtime numbers clearly tagged unverified until this chapter's own CPU-vs-GPU distinction gets its real-hardware answer.

## Worked Solutions

**1.** AoS packs all seven fields of one `Particle` into a single contiguous 28-byte block, so the 3 unused fields (`x`, `y`, `z`) are physically interleaved between the 4 used ones inside that same block. A memory system reading a contiguous range cannot skip the middle of a range it's already streaming through without a separate, more expensive gather-style access — so the unused bytes are fetched as an unavoidable side effect of fetching the used ones sitting right next to them in the same struct instance.

**2.** The layout is identical in both cases — what changed is which fields the *operation* reads. `update_position` reads `x, y, z, vx, vy, vz` (6 of 7 fields, 24 of 28 bytes, `24/28 ≈ 85.7%`), while `total_kinetic_energy` reads only `vx, vy, vz, mass` (4 of 7 fields, 16 of 28 bytes, `57.1%`). Utilization is a property of how much of one struct's layout a specific operation happens to touch, not a fixed property of the layout by itself — the same `Particle` struct can score well or poorly depending entirely on what you ask of it.

**3.** Two mechanisms: cache-line-granularity DRAM bursts (memory is fetched in fixed 64-byte chunks regardless of layout, so bytes that happen to share a line with genuinely needed bytes are effectively "free," softening AoS's penalty), and hardware prefetching (a CPU's prefetcher recognizes a sequential access pattern and starts fetching ahead of the code equally well whether it's walking an AoS array or an SoA array, since both patterns here are purely sequential).

**4.** A GPU warp issues its memory instruction from 32 threads *simultaneously*, not sequentially one after another the way a single CPU core's instruction stream does — there is no single stream for a prefetcher to recognize and get ahead of. Whether those 32 simultaneous, independent addresses cluster into one aligned region or spread across seven separate ones is decided entirely by the layout at the instant the instruction issues, with no equivalent to a CPU's "the next request is probably close to this one" prediction papering over the gap.

**5.** The kinetic-energy *formula* — `0.5 × mass × (vx² + vy² + vz²)` — is applied identically in both functions; the only difference between them is which memory layout each one reads the same four numbers from. If the AoS and SoA versions disagreed, the formula itself (present, unchanged, in both) cannot be the cause — the only place a real discrepancy could originate is the step that copies values out of the `Vec<Particle>` into the four separate `Vec<f32>`s, which is ordinary data movement, not a second implementation of the physics. A layout choice has no mathematical content of its own; it can only ever be implemented correctly or incorrectly, never "more" or "less" correct than another layout holding the same values.
