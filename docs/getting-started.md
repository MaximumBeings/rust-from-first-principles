# Getting Started

This book targets stable Rust throughout — every CPU-side listing compiles and runs on the current stable toolchain, no nightly required. GPU chapters (built on the `cudarc` crate) compile their host-side code on stable too; the kernels themselves need a real NVIDIA GPU and CUDA toolkit to execute, which this book flags explicitly wherever it applies (see [How this book verifies its claims](#how-this-book-verifies-its-claims) below).

## Environment

Install Rust via `rustup`, the standard toolchain manager:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Confirm both `rustc` and `cargo` are on your `PATH`:

```bash
rustc --version
cargo --version
```

This book was written and verified against:

```text
rustc 1.95.0 (59807616e 2026-04-14)
cargo 1.95.0 (f2d3ce0bd 2026-03-21)
```

Any reasonably recent stable toolchain (1.85+) should work identically for every CPU-side chapter — nothing in this book depends on unstable/nightly-only features.

## Verify the toolchain

```bash
cargo new hello_rust
cd hello_rust
```

```rust
// src/main.rs
fn main() {
    println!("Hello from Rust From First Principles!");
    let a: i64 = 7;
    let b: i64 = 35;
    println!("{a} + {b} = {}", a + b);
}
```

```bash
cargo run
```

```text
Hello from Rust From First Principles!
7 + 35 = 42
```

If you see that output, you're ready for [Part 0: Rust Foundations](part0/01-variables-ownership-types.md).

## GPU toolchain (needed for Part 0 Chapter 4 onward, and the GPU appendices)

This book's GPU chapters are written against `cudarc`, a Rust crate providing safe(r) bindings to the CUDA Driver API. Its host-side code (buffer allocation, stream/context management, kernel launch configuration) compiles on stable Rust with no CUDA toolkit installed at all, because `cudarc` resolves `libcuda.so` by dynamic loading at *runtime*, not link time:

```toml
# Cargo.toml
[dependencies]
cudarc = { version = "0.19", default-features = false, features = ["driver", "std", "dynamic-loading", "cuda-12060"] }
```

Swap the `cuda-12060` feature for whichever CUDA version your toolkit reports (`cudarc` supports 11.4 through 13.3 at minimum-supported-version granularity — check the crate's current feature list on [crates.io](https://crates.io/crates/cudarc) if your version isn't listed above).

Actually *running* a kernel needs a real NVIDIA GPU, its driver, and the CUDA toolkit (for `nvcc`, if you want to compile kernels from CUDA C, or a NVPTX-capable Rust nightly if you want to compile kernels from Rust source — see the note below). On Ubuntu/Debian:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

Confirm the driver and toolkit are both visible:

```bash
nvidia-smi      # real driver + GPU, needed to run anything
nvcc --version  # compiler, needed if you compile kernels from CUDA C
```

!!! note "A note on where this book's GPU code was written"
    This book is authored and its CPU-side listings are verified in a sandboxed environment with no NVIDIA GPU and no network access to `static.rust-lang.org` or NVIDIA's package repositories — which rules out both a Rust nightly toolchain (needed for compiling Rust source directly to PTX) and `nvcc`/NVRTC in that environment. So every GPU chapter's host-side `cudarc` code is genuinely compiled there, but kernel bodies and any runtime numbers from them are marked **UNVERIFIED — pending a real-GPU test pass** until independently run on real hardware and confirmed. If you're reading this after that pass, those tags should be gone and replaced with real captured output; if you still see one, that chapter hasn't been verified on hardware yet.

## How this book verifies its claims

Every CPU-side "Worked Example" in this book embeds real, unedited output from `rustc`/`cargo` run against the exact listing shown — nothing is hand-typed or guessed. Where a chapter makes a numeric or behavioral claim, it's cross-checked by a second, independently-implemented method (a naive reference algorithm against an optimized one, or an analytic result against a Monte Carlo estimate) before it appears in the text. GPU chapters follow the same rule for their host-side code; kernel-level claims are explicitly tagged as unverified until run on real hardware, as described above.
