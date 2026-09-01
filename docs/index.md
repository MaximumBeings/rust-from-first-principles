# Rust From First Principles — Under Development

**Building a GPU-Native Tensor and Autodiff Framework in Rust, One Layer at a Time**

!!! info "This book is just getting started"
    This is the fourth sibling project to [*Mojo From First Principles*](https://maximumbeings.github.io/mojo-from-first-principles/), [*Triton From First Principles*](https://maximumbeings.github.io/triton-from-first-principles/), and [*CUDA From First Principles*](https://maximumbeings.github.io/cuda-from-first-principles/), built the same way: one chapter at a time, in page order, each one carrying real code and honestly labeled output. This book's verification story has one more layer than its siblings': every CPU-side listing is genuinely compiled *and run* with `rustc`/`cargo` in this book's build environment — not just compiled, as the CUDA edition's kernels are. GPU chapters, built on the `cudarc` crate, are different: that same environment has no NVIDIA GPU, no `nvcc`, and no path to a Rust nightly toolchain, so host-side `cudarc` code (buffer management, launch configuration, driver calls) is genuinely compiled there, but kernel bodies and any runtime numbers from them carry an explicit **UNVERIFIED — pending real-GPU test** tag until independently confirmed on real hardware and folded back in. Every chapter says plainly which of its claims rest on which kind of evidence.

> "A `Tensor` in the CUDA and Mojo editions of this book has to answer what shape, what dtype, what device. A `Tensor` in this one has to answer a fourth question neither of those languages asks at compile time: who owns it, and who is therefore responsible for freeing it. This book is about building that answer once, correctly, and then letting the compiler enforce it for the rest of the book."

This is a practitioner's guide to Rust that follows the same arc as its Mojo and CUDA C++ siblings rather than its Triton one: instead of consuming an existing tensor library, it builds one — a small, real `Tensor` type with its own shape and stride bookkeeping, its own device abstraction, and its own memory management — and then builds a genuine reverse-mode automatic differentiation engine on top of it, node by node, the same way the Mojo and CUDA books do. Where it necessarily departs from both is where Rust itself is different: ownership and borrowing as a compiler-enforced second typing axis from the very first chapter, no garbage collector and no manual `free`/`cudaFree` calls anywhere in the book, and, from Part 0 Chapter 4 onward, the `cudarc` crate as the bridge to NVIDIA's driver API in place of `nvcc`'s built-in host/device split.

The design principles carried through the whole book:

- **Build the tensor before you build anything on top of it.** Shape, stride, dtype, and device ownership are modeled explicitly as a `Tensor` type in Part 1, the same way Mojo's `Tensor[dtype]` and CUDA's `Tensor` class are — not assumed from a framework, the way the Triton book assumes them from PyTorch.
- **Your own autodiff engine, not someone else's.** Parts 3 and 4 build a real computational graph and a real backward traversal in Rust — no `candle`, `tch-rs`, or `burn` anywhere in this book's core chapters.
- **Ownership as a first-class citizen, not an afterthought.** Every buffer this book allocates — CPU or GPU — is an ordinary owned Rust value, freed automatically through `Drop` when it goes out of scope. There is no equivalent chapter in either the CUDA or Mojo edition, because neither language tracks ownership at compile time the way Rust does.
- **Honest about verification on two different axes.** CPU-side code is compiled *and run* for real, cross-checked by a second independently-implemented method wherever it makes a numeric claim. GPU-side host code is genuinely compiled; kernel bodies and their output are clearly tagged unverified until tested on real hardware — this book never presents a number it didn't actually measure, or blurs the line between what compiled and what ran.
- **Financial computing ready.** The closing chapter validates these building blocks against the same Black-Scholes, rolling-volatility, and Monte Carlo option-pricing problems the Mojo, Triton, and CUDA books end on.

<div class="grid cards" markdown>

- :material-book-open-page-variant:{ .lg .middle } **22 chapters**

    Part 0 through Part 7, from stack-level types to a Monte Carlo option pricer.

- :material-shield-check:{ .lg .middle } **Ownership, compiler-enforced**

    Moves, borrows, and `Drop` — ownership as a real second typing axis, verified against genuine `rustc` errors.

- :material-chip:{ .lg .middle } **`cudarc`-driven GPU kernels**

    Host-side driver code, genuinely compiled; kernel bodies and their output honestly tagged until tested on real hardware.

- :material-finance:{ .lg .middle } **Real financial models**

    Black-Scholes, rolling volatility, and Monte Carlo pricing, cross-checked by a second independent method.

</div>

## How the book is organized

| Part | Focus |
|---|---|
| **Part 0 — Rust Foundations** | Variables, ownership, and types; struct design patterns; memory layout strategies; a first look at `cudarc`'s driver model; SIMD via `std::arch` and the `wide` crate |
| **Part 1 — Core Tensor Infrastructure** | The `Tensor` type itself, memory layout design, creation functions, specialized (half-precision) tensor types, the device abstraction layer, the memory management system |
| **Part 2 — Basic Tensor Operations** | Element-wise operations, matrix operations, reduction operations |
| **Part 3 — Computational Graph Foundation** | A real graph-node architecture, built from scratch |
| **Part 4 — Automatic Differentiation Engine** | Backward function implementation, the gradient computation engine (topological traversal, fan-in accumulation) |
| **Part 5 — GPU Acceleration and Performance** | GPU kernel implementation via `cudarc`, performance optimization techniques (CPU cache/SIMD tuning, GPU occupancy concepts) |
| **Part 6 — Neural Network Building Blocks** | Neural network layers, advanced fused features (Flash-Attention-style fusion, MoE gating) |
| **Part 7 — Financial Computing Applications** | Black-Scholes, rolling volatility, and Monte Carlo option pricing |

Start with [Getting Started](getting-started.md) to stand up a Rust toolchain, or jump straight into [Part 0: Rust Foundations](part0/01-variables-ownership-types.md) if `rustc` and `cargo` are already installed.

!!! note "How this book relates to its siblings"
    Mojo and CUDA C++ both build their own tensor type and their own autodiff engine because neither has anything else to lean on for this kind of from-scratch teaching arc. Triton deliberately does neither — it owns only the kernel and leans on PyTorch for the tensor, the device, and the autograd graph. Rust follows Mojo and CUDA's arc, not Triton's: it has no built-in tensor abstraction and no autograd engine of its own, so this book builds both. What's genuinely new here, with no equivalent in any of the three siblings, is ownership as a compile-time-enforced second typing axis from Chapter 1 onward, and a two-tier verification story that neither the CUDA book (compiled but never run) nor the Mojo or Triton books need: CPU code genuinely compiled and run in this book's own environment, GPU kernels genuinely compiled where possible but explicitly deferred to a real-hardware pass for anything that requires actually executing on a GPU.
