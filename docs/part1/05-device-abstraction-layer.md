# Chapter 10: Device Abstraction Layer — Discovery and Memory Management, Minimally

> "Every `cudarc` call this book has made so far has simply assumed a device might be there and asked anyway, catching the honest failure afterward. A real program shouldn't gamble that way on every single allocation — it should ask once, remember the answer, and let that answer decide the strategy. This chapter builds the small struct that asks, and discovers along the way that cudarc's own asking mechanism works nothing like CUDA's C API does — with real, measurable, and at one point genuinely surprising consequences."

**What you will understand by the end of this chapter:**

- `DeviceManager`: a struct that checks for CUDA's presence exactly once and caches the answer — but built on cudarc's actual two-layer discovery shape, since `cudarc::driver::CudaContext::device_count()` *panics* rather than returning an error the moment `libcuda.so` can't be found, unlike CUDA's own `cudaGetDeviceCount`
- Why bypassing discovery and calling a device operation directly doesn't just fail with less context here — in cudarc specifically, it can take down the entire process, a genuinely stronger consequence than the CUDA edition's equivalent mistake
- `DeviceAwareAllocator`: routing between a `cudarc` device context and a plain `Vec<u8>` fallback based on the manager's cached answer, plus a second, independent fallback for when discovery itself turns out to have been wrong
- A genuine, measured gap between what a device allocation path would guarantee about memory alignment and what its host fallback silently does not
- A real benchmark result that comes out the *opposite* direction from the CUDA edition's own chapter — and a genuine, verified mechanism (an `OnceLock` that never caches a panicking initializer) explaining exactly why

**What you need to know first:**

- Chapter 4 (the honest `cudarc` panics this book has shown since its first device-touching call, and the `catch_unwind`-based `catch_it` helper this chapter reuses) — this chapter is the first one to build something that *uses* that knowledge to avoid the failure altogether, rather than just reporting it afterward
- Chapter 7.3 (alignment and padding) — Section 10.2's alignment gap is a direct, concrete instance of that chapter's general warning
- Chapter 9's `size_of`/behavioral differences from the CUDA edition — this chapter finds another one, in a benchmark result rather than a struct layout
- If you've read the CUDA or Mojo editions: this chapter follows their Chapter 10 section-for-section (device discovery, then a device-aware allocator), but the underlying mechanism diverges early and stays diverged — cudarc's dynamically-loaded FFI layer behaves nothing like CUDA's own C Runtime API once a library fails to load, and this chapter's structure exists specifically to work around that difference rather than translate past it.

## 10.1 Device Discovery: Ask Once, Trust the Answer `[FOUNDATIONAL]`

### Intuition

CUDA's own `cudaGetDeviceCount` is a Runtime API call that always returns a `cudaError_t` — even when no driver, no library, and no device exist at all, it fails *cleanly*, as this book's very first chapters showed. cudarc's `CudaContext::device_count()` does something different: internally, it tries to `dlopen` `libcuda.so` through a lazily-initialized, process-wide cache, and if that load fails, the function `panic!`s outright rather than returning `Err`. A `DeviceManager` built for cudarc has to discover this fact in two layers, not one — a genuinely non-panicking presence check first, and only then, if that succeeds, the real device count.

### Background

| | CUDA's own `cudaGetDeviceCount` | cudarc's `CudaContext::device_count()` |
|---|---|---|
| Behavior when the driver library can't be found | Returns `cudaErrorNoDevice` (a checked error code) | **Panics** — `cudarc::panic_no_lib_found`, deep inside the FFI thunk |
| Safe to call speculatively? | Yes — the whole point of the C Runtime API's design | No — a caller must check something else *first* |
| The "something else" this chapter uses | N/A | `cudarc::driver::sys::is_culib_present()` — attempts the same `dlopen`, but returns `bool` instead of panicking |

### Worked Example 10.1.1 — discovery in two layers, because cudarc's own doesn't return one

```rust
// A tiny, honest device manager -- but one built for cudarc's actual shape,
// not a direct translation of CUDA's cudaGetDeviceCount. cudarc's device_count()
// PANICS (not Result::Err) the moment libcuda.so can't be dlopen'd, so discovery
// has to happen in two layers: a genuinely non-panicking presence check first
// (is_culib_present()), and only THEN, if that succeeds, the real device count.
struct DeviceManager {
    lib_present: bool,
    device_count: i32,
    discovery_succeeded: bool,
}

impl DeviceManager {
    fn new() -> Self {
        let lib_present = unsafe { cudarc::driver::sys::is_culib_present() };
        if !lib_present {
            return DeviceManager { lib_present: false, device_count: 0, discovery_succeeded: false };
        }
        match catch_it(|| cudarc::driver::CudaContext::device_count()) {
            Ok(Ok(n)) => DeviceManager { lib_present: true, device_count: n, discovery_succeeded: true },
            Ok(Err(_)) | Err(_) => DeviceManager { lib_present: true, device_count: 0, discovery_succeeded: false },
        }
    }

    fn has_device(&self, id: i32) -> bool {
        self.discovery_succeeded && id < self.device_count
    }
}
```

Compiled and run as part of the complete `01_device_discovery.rs` further below:

```bash
cargo run --release --bin 01_device_discovery
```

Genuinely compiled and run:

```
DeviceManager::new():
  is_culib_present() = false
  discovery_succeeded=false, device_count=0

mgr.has_device(0) = false
```

`is_culib_present()` genuinely attempts the identical library search `CudaContext::device_count()` would (the same candidate list of `libcuda.so`/`libnvcuda.so` names), but returns a plain `bool` rather than unwinding — `Library::new(choice).is_ok()`, checked directly, with no panic anywhere in its own implementation. `DeviceManager::new()` never even attempts `device_count()` once this first check fails, so the panic that call would otherwise raise never has a chance to happen — `discovery_succeeded` ends up `false` for exactly the same underlying reason CUDA's own edition reports `cudaErrorNoDevice`, just reached through a structurally different check.

### Worked Example 10.1.2 — bypassing the manager, and what that costs in cudarc specifically

Compiled and run as part of the complete `01_device_discovery.rs` further below:

```bash
cargo run --release --bin 01_device_discovery
```

Genuinely compiled and run:

```
--- bypassing the layered manager: calling device_count() directly ---
code that skips is_culib_present() and calls CudaContext::device_count() directly:
  device_count() -> PANICKED (caught here only for this demo -- see below):
  Unable to dynamically load the "cuda" shared library - searched for library names: [...]
  same underlying fact the manager already knew -- but reached through a
  panic rather than a checked return value, and only caught here because
  this specific call is wrapped; see bypass_crash_demo.rs for what happens
  when nothing catches it at all.
```

Calling `CudaContext::device_count()` directly, without ever checking `is_culib_present()` first, produces the exact same underlying fact the manager already knew — but reaches it through a `panic!` rather than a `Result`, and this call is only survivable at all because it happens to be wrapped in `catch_it` for this specific demonstration. CUDA's own equivalent mistake (Chapter 10.1.2 in that edition) merely surfaces the failure later and with less context, still through a checked return value the caller could have inspected. cudarc's version is a strictly stronger consequence: skip the check, and *nothing* protects the rest of the program unless a caller specifically remembers to wrap the call in `catch_unwind`, which is not a mistake anyone would expect to need to guard against from a function whose signature returns an ordinary `Result`.

```rust
// Deliberately UNGUARDED: no is_culib_present() check, no catch_unwind.
// Kept as its own binary, purely to demonstrate what genuinely happens when a
// caller bypasses DeviceManager entirely. This program is EXPECTED to crash
// the whole process -- that crash is the point.
fn main() {
    println!("about to call CudaContext::device_count() with no guard at all...");
    let n = cudarc::driver::CudaContext::device_count();
    println!("device_count() = {:?} (if you see this, a device was genuinely found)", n);
}
```

Genuinely compiled and run as its own binary, `bypass_crash_demo.rs`:

```
about to call CudaContext::device_count() with no guard at all...

thread 'main' (1835) panicked at .../cudarc-0.19.9/src/lib.rs:200:5:
Unable to dynamically load the "cuda" shared library - searched for library names: [...]
stack backtrace:
   0: __rustc::rust_begin_unwind
   ...
   9: cudarc::driver::safe::core::CudaContext::device_count
  10: bypass_crash_demo::main
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
```

The process exits with status `101` — Rust's standard uncaught-panic exit code — having printed exactly one line of its own intended output before the unguarded call ended everything. Unlike Chapter 6's and Chapter 7's deliberately-failing files, this one compiles perfectly cleanly; the mistake it demonstrates is a runtime one, not a compile-time one, which is exactly why `DeviceManager`'s two-layer discipline matters here in a way it never needed to for a use-after-move or a dangling lifetime — the compiler has no way to catch this category of mistake for you.

## 10.2 A Device-Aware Allocator `[FOUNDATIONAL]`

### Intuition

`DeviceManager` answers one question; `DeviceAwareAllocator` acts on it — routing an allocation request to a `cudarc` device context when a device is known to exist, and to a plain `Vec<u8>` otherwise, without ever attempting the device path when discovery has already ruled it out. This is the first allocator in this book that can genuinely *succeed* under this environment's no-GPU constraints, rather than honestly reporting a panic (or, for the CUDA edition, `cudaErrorNoDevice`) the way every device-touching call since Chapter 4 has.

### Background

| | Device path | Host fallback path |
|---|---|---|
| Taken when | `DeviceManager` reports a device *and* the allocation itself succeeds | No device reported, or the device path panics anyway despite being reported |
| Allocator call | `cudarc::driver::CudaContext::new(0)`, then (on real hardware) a stream allocation | `vec![0u8; bytes]` |
| Alignment guarantee | Whatever the CUDA driver documents for device allocations (unverified here — no device to query) | Whatever Rust's global allocator's default alignment for `u8` happens to be — not requested, not guaranteed |
| Genuinely reachable in this no-GPU environment? | No (would need a real device) | Yes — both trigger conditions below are real |

### Worked Example 10.2.1 — the routing condition, both reachable cases

```rust
struct DeviceAwareAllocator {
    device_available: bool,
}

impl DeviceAwareAllocator {
    fn allocate(&self, bytes: usize) -> (Vec<u8>, &'static str) {
        if self.device_available {
            match catch_it(|| cudarc::driver::CudaContext::new(0)) {
                Ok(Ok(_ctx)) => {
                    // Real hardware would continue: ctx.default_stream().alloc_zeros(...).
                    // UNVERIFIED -- pending real-GPU test; unreachable in this sandbox.
                    (vec![0u8; bytes], "cudarc CudaContext (device) [UNVERIFIED -- pending real-GPU test]")
                }
                _ => (vec![0u8; bytes], "Vec<u8> (host fallback -- CudaContext::new panicked despite device_available)"),
            }
        } else {
            (vec![0u8; bytes], "Vec<u8> (host fallback -- no device reported by discovery)")
        }
    }
}
```

Compiled and run as part of the complete `02_device_aware_allocator.rs` further below:

```bash
cargo run --release --bin 02_device_aware_allocator
```

Genuinely compiled and run:

```
Case 1 (device_available=false, the real, measured discovery result):
  strategy used: Vec<u8> (host fallback -- no device reported by discovery)
  allocation succeeded: true

Case 2 (device_available=true, FORCED, to test the failure-recovery path):
  strategy used: Vec<u8> (host fallback -- CudaContext::new panicked despite device_available)
  allocation succeeded: true
```

Case 1 is this environment's real, honestly-discovered situation: `DeviceManager` reports no device, `device_available` is genuinely `false`, and `allocate()` never even attempts `CudaContext::new` — it goes straight to `vec![0u8; bytes]`, which genuinely succeeds. Case 2 deliberately forces `device_available = true` (simulating discovery having been correct at startup but the device becoming unavailable, or simply wrong) specifically to exercise the *second* fallback: `allocate()` still tries the device path first, genuinely catches a panic via `catch_it`, and falls back to `Vec<u8>` anyway — succeeding through a different path than Case 1, for a different reason. Both are real, both are genuinely exercised, and both produce a working allocation in an environment where every device-touching call this book has made has honestly failed or panicked.

### Worked Example 10.2.2 — the alignment guarantee that never arrives

Compiled and run as part of the complete `02_device_aware_allocator.rs` further below:

```bash
cargo run --release --bin 02_device_aware_allocator
```

Genuinely compiled and run:

```
checking whether the host-fallback path honors a 256-byte alignment request
it was never actually told about:
  5 of 5 Vec<u8> allocations were NOT 256-byte aligned
  std::alloc::alloc with Layout::from_size_align(_, 256), by contrast: aligned to 256 bytes? true
```

`allocate()`'s signature never takes an alignment parameter at all — `vec![0u8; bytes]` allocates through Rust's global allocator using `u8`'s own natural alignment (`1` byte), and all 5 genuinely tested calls confirm it: none land on a 256-byte boundary. `std::alloc::alloc` called with an explicit `Layout::from_size_align(bytes, 256)`, by contrast, genuinely honors the request — Rust's allocator API takes the alignment as a first-class, explicit part of what's being asked for, rather than inferring it from a type the caller may not have chosen with alignment in mind at all. This is the same structural gap Chapter 7.3's `Vec<f32>` discussion already found from the other direction: a `Vec<T>`'s allocation is only ever as aligned as `T` requires, never more, unless something *else* — a `#[repr(align(N))]` wrapper type, or a raw `Layout` request like this one — asks for more explicitly.

> `[COMMON TRAP]` A function whose parameter list doesn't mention alignment isn't necessarily a function that doesn't need to think about it — `DeviceAwareAllocator::allocate()`'s device path would, on real hardware, satisfy whatever alignment the CUDA driver documents for its own allocations, purely as a side effect of that API's own guarantee, while its `Vec<u8>` fallback satisfies nothing beyond `u8`'s trivial 1-byte requirement, for the identical reason: neither path was ever asked to guarantee anything about alignment beyond what its own underlying allocator happens to provide. The fix mirrors this worked example's own contrast — replace `vec![0u8; bytes]` with an explicit `Layout`-based allocation carrying the alignment the caller actually needs, so the guarantee survives the routing decision instead of depending on which branch happened to run.

### Worked Example 10.2.3 — a benchmark that comes out backwards from the CUDA edition, for a real reason

Compiled and run as part of the complete `02_device_aware_allocator.rs` further below:

```bash
cargo run --release --bin 02_device_aware_allocator
```

Genuinely compiled and run:

```
100 catch_unwind(CudaContext::new) calls (all 100 panicked): 47.968 ms total, 0.47968 ms/call
100 Vec<u8> allocations (all succeeded): 0.001 ms total, 0.00001 ms/call
```

(Rerunning the same binary produces slightly different absolute figures each time, ranging from roughly `45` to `50` ms total across every rerun performed while writing this chapter — but the qualitative result, and its order of magnitude, held every time.)

The CUDA edition's own version of this benchmark found the failing device call *faster* than the succeeding host allocation, because CUDA's runtime caches "no device is present" cheaply after the first failed call. This chapter's numbers say the opposite, by roughly three orders of magnitude — and the reason is a real, specific mechanism worth tracing all the way down, not just a difference to note and move past. `CudaContext::new`'s FFI thunk loads `libcuda.so` through a function-local `static LIB: OnceLock<Library>`, filled in by `LIB.get_or_init(|| { ...; panic_no_lib_found(...) })`. `OnceLock::get_or_init` only caches a value when its closure *returns* one — when the closure panics instead, the cell is left exactly as uninitialized as it started, by design (this is documented `OnceLock` behavior, not an oversight), so the library search runs again from scratch on every single subsequent call. A direct, isolated timing check confirms this: five successive, individually-timed calls to `catch_it(|| CudaContext::device_count())` each take essentially the same ~0.5 ms, with no speedup on later calls — genuine proof that nothing is being cached across the panics at all.

```
call 0: 0.5828 ms
call 1: 0.5076 ms
call 2: 0.5349 ms
call 3: 0.5041 ms
call 4: 0.5683 ms
```

Read honestly, this chapter's benchmark isn't measuring "device allocation versus host allocation" any more than the CUDA edition's was — it's measuring "the cost of a library search that fails and is never cached" against "the cost of a real host allocation." But where the CUDA edition's honest reading was a caution about benchmark *interpretation*, this chapter's honest reading is a caution about *architecture*: any code that calls a panic-guarded cudarc function inside a loop or a hot path, expecting the underlying `OnceLock` to behave like a typical successful-initialization cache, will silently pay the full, uncached search cost on every single iteration. `DeviceManager`'s entire value proposition — ask once, cache the answer — has to be implemented explicitly, in the caller's own struct field, precisely because cudarc's internal caching does not survive the panic path the way CUDA's own runtime-level caching survives an ordinary error return.

> `[COMMON TRAP]` It's tempting to assume any lazily-initialized, cached resource in Rust behaves the same way regardless of whether initialization succeeds or fails — `OnceLock`, `LazyLock`, and similar types all share this same property: a panicking initializer leaves the cell empty, ready to try again, rather than caching the failure the way a `Result`-returning cache (a `HashMap<Key, Result<Value, Error>>`, say) explicitly would. This is exactly the right behavior for a resource that might genuinely become available later (a library installed after the process starts, a device plugged in mid-run), but it means any code relying on "the first call pays the cost, later calls are free" needs to check specifically whether the thing being cached is a success value or an outcome that includes failure — the two are not interchangeable, and this chapter's own benchmark is the direct, measured consequence of assuming they were.

## 10.3 Complete Runnable Code

### File: `01_device_discovery.rs`

```rust
use std::panic;

fn catch_it<T>(f: impl FnOnce() -> T + panic::UnwindSafe) -> Result<T, String> {
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

// A tiny, honest device manager -- but one built for cudarc's actual shape,
// not a direct translation of CUDA's cudaGetDeviceCount. cudarc's device_count()
// PANICS (not Result::Err) the moment libcuda.so can't be dlopen'd, so discovery
// has to happen in two layers: a genuinely non-panicking presence check first
// (is_culib_present()), and only THEN, if that succeeds, the real device count.
struct DeviceManager {
    lib_present: bool,
    device_count: i32,
    discovery_succeeded: bool,
}

impl DeviceManager {
    fn new() -> Self {
        let lib_present = unsafe { cudarc::driver::sys::is_culib_present() };
        if !lib_present {
            return DeviceManager { lib_present: false, device_count: 0, discovery_succeeded: false };
        }
        // The library is present, so this call is genuinely safe to attempt --
        // still wrapped defensively, since a present-but-broken driver could
        // still fail in ways worth catching rather than propagating.
        match catch_it(|| cudarc::driver::CudaContext::device_count()) {
            Ok(Ok(n)) => DeviceManager { lib_present: true, device_count: n, discovery_succeeded: true },
            Ok(Err(_)) | Err(_) => DeviceManager { lib_present: true, device_count: 0, discovery_succeeded: false },
        }
    }

    fn has_device(&self, id: i32) -> bool {
        self.discovery_succeeded && id < self.device_count
    }
}

fn main() {
    println!("=== Section 10.1: Device Discovery ===\n");
    let mgr = DeviceManager::new();
    println!("DeviceManager::new():");
    println!("  is_culib_present() = {}", mgr.lib_present);
    println!("  discovery_succeeded={}, device_count={}", mgr.discovery_succeeded, mgr.device_count);

    println!("\nmgr.has_device(0) = {}", mgr.has_device(0));

    println!("\n--- bypassing the layered manager: calling device_count() directly ---");
    println!("code that skips is_culib_present() and calls CudaContext::device_count() directly:");
    match catch_it(|| cudarc::driver::CudaContext::device_count()) {
        Ok(Ok(n)) => println!("  device_count() -> Ok({}) (did not panic)", n),
        Ok(Err(e)) => println!("  device_count() -> Err({:?})", e),
        Err(msg) => {
            println!("  device_count() -> PANICKED (caught here only for this demo -- see below):");
            println!("  {}", msg.lines().next().unwrap_or(""));
        }
    }
    println!("  same underlying fact the manager already knew -- but reached through a");
    println!("  panic rather than a checked return value, and only caught here because");
    println!("  this specific call is wrapped; see bypass_crash_demo.rs for what happens");
    println!("  when nothing catches it at all.");
}
```

```bash
cargo run --release --bin 01_device_discovery
```

Produces exactly the output shown in Worked Examples 10.1.1 and 10.1.2 above.

### File: `bypass_crash_demo.rs`

Kept separate from the two numbered files above — not because it fails to compile (it compiles perfectly cleanly, and `cargo build --release` for this chapter's whole package succeeds with it included), but because it is a deliberately unguarded program whose entire purpose is to crash when run, which makes it unsuitable to invoke as part of any normal build-and-run sequence.

```rust
// Deliberately UNGUARDED: no is_culib_present() check, no catch_unwind.
// Kept as its own binary, outside the "Complete Runnable Code" set, purely to
// demonstrate what genuinely happens when a caller bypasses DeviceManager
// entirely and calls a cudarc device operation directly. This program is
// EXPECTED to crash the whole process -- that crash is the point.
fn main() {
    println!("about to call CudaContext::device_count() with no guard at all...");
    let n = cudarc::driver::CudaContext::device_count();
    println!("device_count() = {:?} (if you see this, a device was genuinely found)", n);
}
```

```bash
cargo run --release --bin bypass_crash_demo
echo "exit code: $?"
```

Produces exactly the crash and exit code `101` shown in Worked Example 10.1.2 above.

### File: `02_device_aware_allocator.rs`

```rust
use std::alloc::{alloc, dealloc, Layout};
use std::panic;
use std::time::Instant;

fn catch_it<T>(f: impl FnOnce() -> T + panic::UnwindSafe) -> Result<T, String> {
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

// Routes to a cudarc device context when the manager reports a device,
// falling back to a plain host Vec<u8> otherwise -- and ALSO falls back if
// the device path itself panics despite device_available, since discovery is
// a snapshot that can go stale (or, here, be flat-out wrong) between the
// check and the actual allocation attempt.
struct DeviceAwareAllocator {
    device_available: bool,
}

impl DeviceAwareAllocator {
    fn allocate(&self, bytes: usize) -> (Vec<u8>, &'static str) {
        if self.device_available {
            match catch_it(|| cudarc::driver::CudaContext::new(0)) {
                Ok(Ok(_ctx)) => {
                    // Real hardware would continue: ctx.default_stream().alloc_zeros(...).
                    // UNVERIFIED -- pending real-GPU test; unreachable in this sandbox.
                    (vec![0u8; bytes], "cudarc CudaContext (device) [UNVERIFIED -- pending real-GPU test]")
                }
                _ => (vec![0u8; bytes], "Vec<u8> (host fallback -- CudaContext::new panicked despite device_available)"),
            }
        } else {
            (vec![0u8; bytes], "Vec<u8> (host fallback -- no device reported by discovery)")
        }
    }
}

fn main() {
    println!("=== Section 10.2: Memory Management and Allocation ===\n");

    println!("--- routing condition, both reachable combinations in this environment ---");
    let lib_present = unsafe { cudarc::driver::sys::is_culib_present() };
    let honest_allocator = DeviceAwareAllocator { device_available: lib_present };
    let (p1, strategy1) = honest_allocator.allocate(1024);
    println!("Case 1 (device_available={}, the real, measured discovery result):", lib_present);
    println!("  strategy used: {}", strategy1);
    println!("  allocation succeeded: {}", !p1.is_empty() || 1024 == 0);

    let forced_allocator = DeviceAwareAllocator { device_available: true }; // simulate discovery having (wrongly) reported a device
    let (p2, strategy2) = forced_allocator.allocate(1024);
    println!("\nCase 2 (device_available=true, FORCED, to test the failure-recovery path):");
    println!("  strategy used: {}", strategy2);
    println!("  allocation succeeded: {}", !p2.is_empty() || 1024 == 0);

    println!("\n--- the alignment parameter that never arrives ---");
    println!("checking whether the host-fallback path honors a 256-byte alignment request");
    println!("it was never actually told about:");
    let mut misaligned_count = 0;
    for _ in 0..5 {
        let v: Vec<u8> = vec![0u8; 1024];
        let addr = v.as_ptr() as usize;
        if addr % 256 != 0 {
            misaligned_count += 1;
        }
    }
    println!("  {} of 5 Vec<u8> allocations were NOT 256-byte aligned", misaligned_count);

    let layout = Layout::from_size_align(1024, 256).unwrap();
    let aligned_ptr = unsafe { alloc(layout) };
    let addr2 = aligned_ptr as usize;
    println!("  std::alloc::alloc with Layout::from_size_align(_, 256), by contrast: aligned to 256 bytes? {}", addr2 % 256 == 0);
    unsafe { dealloc(aligned_ptr, layout) };
    println!("  a device-aware allocator that silently falls back to plain Vec<u8> loses");
    println!("  whatever alignment guarantee the device path would have honored -- exactly");
    println!("  the kind of silently-dropped parameter Chapter 7.3 warned about generally.");

    println!("\n--- reading a benchmark's ledger honestly ---");
    let trials = 100;
    let t0 = Instant::now();
    let mut real_failures = 0;
    for _ in 0..trials {
        if catch_it(|| cudarc::driver::CudaContext::new(0)).is_err() {
            real_failures += 1;
        }
    }
    let cudarc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    for _ in 0..trials {
        let v: Vec<u8> = vec![0u8; 1024 * 4];
        drop(v);
    }
    let vec_ms = t2.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  {} catch_unwind(CudaContext::new) calls (all {} panicked): {:.3} ms total, {:.5} ms/call",
        trials, real_failures, cudarc_ms, cudarc_ms / trials as f64
    );
    println!(
        "  {} Vec<u8> allocations (all succeeded): {:.3} ms total, {:.5} ms/call",
        trials, vec_ms, vec_ms / trials as f64
    );
    println!("\n  honest reading: this is NOT \"device allocation vs host allocation\" performance --");
    println!("  see the chapter text for what these two numbers actually measure, and why");
    println!("  the direction of this specific comparison is itself a real, Rust-specific");
    println!("  finding rather than a repeat of the CUDA edition's own conclusion.");
}
```

```bash
cargo run --release --bin 02_device_aware_allocator
```

Produces exactly the output shown in Worked Examples 10.2.1 through 10.2.3 above.

`Cargo.toml` for all three binaries:

```toml
[package]
name = "rust_ch10"
version = "0.1.0"
edition = "2024"

[dependencies]
cudarc = { version = "0.19", default-features = false, features = ["driver", "std", "dynamic-loading", "cuda-12060"] }
```

Every number here was independently verified earlier in this chapter, including the isolated five-call timing check that pinned down exactly why Worked Example 10.2.3's benchmark runs the direction it does. All three files genuinely compile and run in this sandbox against the real `cudarc` crate; none of their output is projected or assumed, including the parts of this chapter that turned out to disagree with the CUDA edition's own numbers.

## Chapter Summary

`DeviceManager` still turns "ask and see what happens" into "ask once, cache the answer," but cudarc's own shape forces that discovery into two layers rather than CUDA's one: a non-panicking `is_culib_present()` check first, and only then, if that succeeds, the real device count — because `CudaContext::device_count()` panics outright the moment the underlying library can't be found, rather than returning the checked error CUDA's own Runtime API would. Bypassing the manager here isn't merely worse in the same way the CUDA edition described; it's categorically worse, genuinely capable of crashing the entire process with exit code `101`, since nothing about a `Result`-returning function's signature warns a caller that it might not actually return at all. `DeviceAwareAllocator` routes between a `cudarc` device path and a `Vec<u8>` host fallback, and — exactly as in the CUDA edition — that fallback silently drops the alignment guarantee the device path would have provided. And this chapter's own benchmark of "device path versus host allocation" timing produced a genuinely different, opposite-direction result from the CUDA edition's, for a real and fully traced reason: `OnceLock`'s documented behavior of never caching a panicking initializer means every guarded attempt at the device path pays the full, uncached library-search cost again, roughly three orders of magnitude slower than the host fallback it's supposedly a defensive wrapper around — a genuine, measured architectural consequence of building "ask once" discipline on top of a resource that doesn't cache its own failures the way CUDA's runtime does.

## Self-Check Questions

1. Why can't `DeviceManager::new()` simply call `CudaContext::device_count()` directly and interpret a `Result::Err` the way the CUDA edition's `DeviceManager` interprets a non-`cudaSuccess` return code?
2. `is_culib_present()` and `CudaContext::device_count()` both ultimately search for the same `libcuda.so`/`libnvcuda.so` candidate names. What's the one concrete difference in how each function is implemented that lets one of them stay panic-free?
3. `bypass_crash_demo.rs` compiles without any errors or warnings, yet is described as "deliberately broken." In what sense is it broken, if the compiler accepts it without complaint?
4. `DeviceAwareAllocator::allocate()`'s device-path branch, on real hardware, would satisfy some CUDA-driver-documented alignment guarantee "as a side effect," not because the allocator explicitly requested it. Why does this make the guarantee fragile even before the host-fallback branch is considered at all?
5. Worked Example 10.2.3 found the device path roughly a thousand times slower than the host fallback, the opposite direction from the CUDA edition's own benchmark. If cudarc's `CudaContext::new` call *succeeded* on real hardware (a device genuinely present), would this same "no caching across calls" mechanism still apply on the next `CudaContext::new` call, or does it only affect the panicking path? Reason from `OnceLock`'s actual caching rule, not from this chapter's specific (failing) numbers.

## Where We Go Next

`DeviceAwareAllocator` still returns a plain `Vec<u8>` or an unused device context and expects the caller to manage its own lifetime from there — exactly the manual-lifetime problem Chapter 6's ownership rules solved for a single `Tensor`, not yet extended to something shared across multiple views or reused across repeated allocations. Chapter 11 builds the memory management layer this book's actual `Tensor` needs on top of everything Part 1 has built so far: shared ownership across multiple `Tensor` views of the same buffer, and a pooled allocator that reuses freed memory instead of paying this chapter's own discovery and allocation costs on every single request.

## Worked Solutions

**1.** Because `CudaContext::device_count()` doesn't reliably *reach* the point of returning a `Result` at all when the library is missing — the missing-library case is a `panic!`, not an `Err`, so a caller who only pattern-matches on `Ok`/`Err` and never wraps the call in `catch_unwind` would simply crash instead of ever seeing the `Err` branch execute. The CUDA edition's `cudaGetDeviceCount` always returns a value (success or a specific error code) no matter what's missing on the system; cudarc's `device_count()` only returns a value when the dynamic library load itself has already succeeded, making a same-day error-code check insufficient on its own.

**2.** `is_culib_present()`'s own implementation calls `Library::new(choice).is_ok()` directly inside an ordinary loop, checking each candidate name with a `Result` it inspects itself and never propagates as a panic. `CudaContext::device_count()` (via the FFI thunk it calls into) instead calls `culib()`, whose `OnceLock::get_or_init` closure calls `panic_no_lib_found(...)` if every candidate fails — a deliberate design choice in that specific code path, not an unavoidable consequence of trying the same library names.

**3.** It's "broken" in the sense that its author's evident intent — printing a `device_count()` result — is never fulfilled the way a normal successful run would fulfill it; the moment the unguarded call panics, the rest of `main()` never executes, and the process terminates with a nonzero exit code rather than completing normally. This is a runtime defect, not a type error or a borrow-checker violation, and Rust's compiler has no mechanism to catch it — nothing in `CudaContext::device_count()`'s type signature (`Result<i32, DriverError>`) discloses that calling it can also *not* return a `Result` at all.

**4.** Because nothing about the allocator's own type or method signature records that the guarantee exists or depends on the device path being taken — a caller who happens to rely on it (say, to safely reinterpret the returned buffer as a wider SIMD type) has no compile-time or even type-level signal that switching to the fallback branch (discovery reporting no device, or a stale `device_available` flag) would silently invalidate an assumption their code depends on. A guarantee that exists only "as a side effect of which branch happened to run" is invisible exactly where it matters most: at the call site that assumes it holds.

**5.** The mechanism is specific to a panicking initializer, not to `CudaContext::new` calls in general — `OnceLock::get_or_init` caches a value only when its closure *returns* one, and on real hardware, a successful `CudaContext::new(0)` call means the underlying `culib()` `OnceLock` closure actually completed and returned the loaded library handle, which *does* get cached from that point forward. Every *subsequent* call, on that same successful hardware, would reuse the cached library handle rather than repeating the dlopen search — this chapter's slow, repeated-search behavior is a direct consequence of the closure never completing (because it panics), and would not appear at all once the underlying resource is genuinely, successfully initialized even once.
