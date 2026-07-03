**Title:** `vulkan: drain the queue before vkResetCommandPool in ggml_vk_command_pool_cleanup`

---

Patch in `21-vulkan-async-command-pool-reset.patch` (~11 LOC, one function).
Applied locally in our vendored ggml; **crash-fix awaiting confirmation on a
native RADV/NVIDIA box** (we only have Apple/MoltenVK, which cannot reproduce the
fault — see below). File against **ggml-org/llama.cpp** (vulkan; per the
repo-routing table).

## Symptom

A user (our issue #215) hits a hard segfault running a small ASR model
(Qwen3-Omni audio encoder + Qwen3-1.7B) on the Vulkan backend, on **both** an
NVIDIA GPU (`libnvidia-eglcore`) and an AMD Radeon (`libvulkan_radeon`), with
q8_0 and q4_k weights. The fault is inside the vendor driver, one frame under:

```
#2 ggml_vk_command_pool_cleanup(...)          <- vkResetCommandPool
#3 ggml_vk_graph_cleanup(ggml_backend_vk_context*)
#4 ggml_backend_sched_compute_splits(...)
#5 ggml_backend_sched_graph_compute()
#6 <model>_run_encoder()
```

Crucially, the reporter found it needs **"multiple slices"** to trigger — a
single short clip runs clean; the same clip concatenated 3× (~33 s) crashes
reliably.

## Root cause

`ggml_vk_command_pool_cleanup()` calls `vkResetCommandPool`, whose spec
precondition is that every command buffer allocated from the pool is **not in
the pending state** (not still being executed by the GPU). The function's own
comment already says *"Requires command buffers to be done"* — it relies on the
caller having synchronized.

In **async** mode (`device->support_async`, the default on non-Intel GPUs)
`ggml_backend_vk_graph_compute` returns **without** synchronizing (the
`ggml_vk_synchronize` at the end is gated on `!support_async`); the wait is
deferred. That leaves a window in which a command pool can be reset while buffers
are still pending, from either cleanup path:

- `ggml_vk_graph_cleanup` (called at sched split boundaries — the reporter's
  stack), and
- `ggml_vk_queue_command_pools_cleanup`, which resets the pool once
  `buffers_in_use() >= cleanup_frequency` (10) and is reached from the
  synchronous transfer helpers `ggml_vk_buffer_write_2d` / `_read`, i.e. every
  `ggml_backend_tensor_set/get`.

The model that triggers this runs one small graph per ~1 s of audio (per
100-mel-frame chunk: `tensor_set(mel)` → conv `graph_compute` → `tensor_get`), so
after ~10 chunks a `tensor_set`/split reset fires while earlier chunks' command
buffers are still executing → `vkResetCommandPool` faults. That is exactly the
reporter's "≥10 buffers / multiple slices" threshold, and why a single short clip
(< 10 buffers) is fine. Ordinary LLM decode (one large graph per token, each
followed by a real synchronize) rarely accumulates 10 unsynced buffers, which is
why this has stayed latent.

## Why we can't reproduce it locally

Not reproducible on MoltenVK: Metal reference-counts and auto-manages
command-buffer lifetime, so "reset a pool while its buffers are pending" is simply
not a fault mode there. Every flash/manual × single/3× combination runs clean on
our M1. The crash is specific to native Vulkan drivers (RADV/NVIDIA) that enforce
the pending-state precondition. This is the same hardware-access gap as draft #19.

## Fix

Enforce the documented precondition at the single choke point — drain the pool's
own queue before resetting it:

```cpp
static void ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    if (p.q) {
        std::lock_guard<std::mutex> guard(queue_mutex);
        p.q->queue.waitIdle();
    }

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    ...
}
```

`vk_command_pool` already carries `q` (its owning `vk_queue*`), so we wait exactly
the right queue, and `queue_mutex` (the existing global that serializes submits to
the shared `vk::Queue`) satisfies `vkQueueWaitIdle`'s external-synchronization
requirement. It is a **no-op on the common already-synchronized path** (the queue
is idle, `waitIdle` returns immediately) and only blocks when a reset would
otherwise be undefined behaviour. No caller of `command_pool_cleanup` holds
`queue_mutex`, so there is no recursive-lock deadlock; `waitIdle` only awaits
already-submitted work, so no submit/wait cycle either.

A tighter alternative for reviewers to weigh: instead of an unconditional
`waitIdle`, have `ggml_backend_vk_graph_compute` fully synchronize before
returning even in async mode, or track a per-pool "has-unsynced-submits" flag and
`waitIdle` only when set. The choke-point `waitIdle` is the smallest correct
change and is self-documenting against the existing "Requires command buffers to
be done" comment.

## Confirmation requested from the reporter (native RADV/NVIDIA)

1. Build our branch (guard applied) and run the 3× clip on Vulkan — expect no
   crash. In our tree the model also force-disables Vulkan async as a belt-and-
   suspenders default; set `CRISPASR_MOSS_TRANSCRIBE_VULKAN_ASYNC=1` so the async
   path is actually exercised and the guard is what prevents the crash.
2. `test-backend-ops` won't cover this (it's a multi-graph lifecycle bug, not an
   op); the model repro is the test. If it holds on both NVIDIA and AMD, that is
   the signal to file.

## Requirements / AI usage

- Root cause, the `waitIdle` guard, and this write-up were developed with AI
  assistance (Claude) reading the ggml-vulkan source; a human reviewed the
  synchronization/deadlock reasoning and will author the PR prose and commit
  message directly per ggml-org/llama.cpp's AI-content policy.
- No `test-backend-ops` op is touched. Correctness + no-perf-regression verified
  on MoltenVK (async on/off produce byte-identical transcripts; the model's
  per-chunk encoder loop is a heavy exerciser of both cleanup paths). Crash-gone
  verification is pending native RADV/NVIDIA hardware.

## Status

Draft — patch applied locally, awaiting native-hardware confirmation from the #215
reporter before filing at ggml-org/llama.cpp.
