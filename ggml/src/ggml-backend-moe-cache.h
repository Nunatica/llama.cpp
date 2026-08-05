#pragma once

#include <stdint.h>
#include <stddef.h>

// MoE expert cache — dynamic VRAM cache for MoE expert weights on CPU-resident
// MUL_MAT_ID. Integration point is the CPU mul_mat_id kernel itself: thread 0
// dispatches cached expert rows to the GPU while the remaining threads compute
// the uncached rows, then results are collected into dst before the node ends.
//
// This table is the bridge between ggml-cpu (which cannot link CUDA) and the
// CUDA backend (which registers the implementations at backend-reg time).
// begin/plan/dispatch/collect for one node are called from a single thread
// (ith == 0); the implementation may assume no concurrent node processing.

#ifdef __cplusplus
extern "C" {
#endif


struct ggml_tensor;

struct ggml_moe_cache_api {
    // Decide whether the cache engages for this MUL_MAT_ID node.
    // Returns the device id to use (>= 0) or -1 to stay on the pure-CPU path.
    // Performs lazy per-device initialization on first use and selects the
    // internal slot pool matching (expert_size, wtype).
    //   tensor_name: src0->name (stable cache key source, e.g. "blk.7.ffn_up_exps.weight")
    //   host_base:   src0->data (source for async inserts)
    //   expert_size: src0->nb[2] (bytes per expert)
    //   n_in/n_out:  src0->ne[0] / src0->ne[1]
    //   wtype:       src0->type
    //   n_expert:    src0->ne[2]
    //   n_tokens:    ids->ne[1] (cache engages only when == 1)
    int (*begin)(const char * tensor_name, const void * host_base, size_t expert_size,
                 int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens);

    // For each of the n_ids expert ids: slot_idx[k] = cache slot index (hit) or
    // -1 (miss; CPU computes the row, an async insert may be enqueued).
    // Returns the number of hits.
    int (*plan)(int dev, const int32_t * ids, int n_ids, int32_t * slot_idx);

    // One batched GPU launch computing all n_hits rows:
    //   out_row[i] (n_out floats) = W[slot_idx_compact[i]] . act_rows[i]
    // act_rows are host fp32 pointers (they may all be the same row for
    // gate/up-style nodes; distinct rows for down-style nodes).
    // Asynchronous; results are pulled into dst by collect().
    void (*dispatch)(int dev, int wtype, int64_t n_in, int64_t n_out, int n_hits,
                     const int32_t * slot_idx_compact, const float * const * act_rows);

    // Synchronize the device's compute stream and copy the n_hits result rows
    // (in dispatch order) into dst_rows[0..n_hits-1] (n_out floats each).
    void (*collect)(int dev, int n_hits, float * const * dst_rows, int64_t n_out);

    // Periodic stats logging (rate-limited internally).
    void (*stats)(void);

    // ---- GPU-resident dst handoff (down-projection round-trip elimination) ----
    // The scheduler offers the GPU-side copy tensor of a CPU MUL_MAT_ID dst
    // BEFORE the CPU split runs. If the cache then computes that node, it
    // scatters its GPU rows directly into gpu_copy_data (async) instead of
    // pulling them to the host.
    void (*redirect_offer)(const void * host_dst_data, size_t nb1, int64_t n_rows,
                           void * gpu_copy_data, void * consumer_backend);
    // Called by the scheduler at the consumer split's input-copy site, after
    // the CPU split completed. Returns 1 if the cache fully populated the GPU
    // copy (it uploads the CPU-computed miss rows here and installs a
    // stream-order dependency on the consumer backend) — the scheduler must
    // then SKIP its own copy. Returns 0 for the normal copy path.
    int (*redirect_finalize)(const void * host_dst_data, void * consumer_backend);

    // ---- fused gate+up+GLU path ----
    // Called by the CPU GLU kernel (swiglu split variant) from EVERY thread:
    // returns the bitmask of dst rows the cache computed on the GPU (fused
    // silu(gate)*up) — those rows must be SKIPPED by the CPU loop. Thread 0
    // (ith == 0) additionally synchronizes the fused chain and scatters the
    // GPU rows into dst before returning. Returns 0 when the cache has no
    // pending fused work for this (src0, src1) pair.
    unsigned long long (*glu_hits)(const void * src0_data, const void * src1_data,
                                   void * dst_data, size_t dst_nb1, int ith);

    // Host weight buffer teardown notification (model unload): drops queued
    // insert jobs sourced from the range and resets per-block tensor-base
    // learning. Must be called before the memory is unmapped/freed.
    void (*invalidate)(const void * base, size_t size);

    // ---- complete-layer HIGH cache -----------------------------------------
    // Returns device ordinal + 1 when a MUL_MAT_ID node belongs to a layer that
    // is being promoted or is HIGH resident. Zero keeps normal assignment.
    int (*high_backend)(const struct ggml_tensor * node);

    // Allocate persistent layer-owned backend buffers without changing graph
    // tensor pointers. backend and buft are opaque here so this bridge does not
    // depend on ggml-backend.h.
    int (*high_preflight)(struct ggml_tensor * node, void * backend, void * buft);

    // Non-zero only after every required gate/up/down buffer for this layer was
    // prepared. The scheduler must not force a partially prepared layer.
    int (*high_ready)(const struct ggml_tensor * node);

    // Bind scheduler-owned copy-tensor metadata to the persistent HIGH buffer.
    // Returns non-zero when normal graph allocation and input copying should be
    // skipped for this tensor.
    int (*high_bind_copy)(const struct ggml_tensor * host,
                          struct ggml_tensor * gpu_copy,
                          void * backend,
                          void * buft);
    // Called for a host weight split input and its normal GPU copy:
    //   0 = use normal/selective copy, 1 = issue a complete copy, 2 = skip copy.
    int (*high_prepare)(const struct ggml_tensor * host,
                        const struct ggml_tensor * gpu_copy);
    // Records a complete copy after the scheduler enqueues it. Once every
    // gate/up/down tensor is committed, the layer becomes HIGH resident.
    void (*high_commit)(const struct ggml_tensor * host,
                        const struct ggml_tensor * gpu_copy);
    // Changes whenever promotion/demotion requires the scheduler graph to be
    // split and allocated again.
    uint64_t (*high_generation)(void);

    // Node wall-time sample for the bail-out judge. code is begin()'s return
    // value: -3 = pure-CPU baseline sample, >= 0 = cache-engaged sample.
    void (*node_time)(int code, int64_t wall_us);
};

// Zero-initialized in ggml-backend.cpp; populated by the CUDA backend in
// ggml_backend_cuda_reg() when the cache is enabled (GGML_CUDA_MOE_CACHE=1).
extern struct ggml_moe_cache_api ggml_moe_cache;

#ifdef __cplusplus
}
#endif
