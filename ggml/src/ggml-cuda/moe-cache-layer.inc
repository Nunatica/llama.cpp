// Two-tier MoE cache prototype.
//
// LOW  : the existing expert-slot cache in this translation unit.
// HIGH : complete, canonical layer weights in dedicated persistent GPU buffers.
//
// Scheduler split-input copies are intentionally temporary and may alias across
// layers. HIGH therefore owns backend buffers whose lifetime is independent of
// the graph allocator. Fresh scheduler copy-tensor metadata is bound to those
// buffers whenever a graph is rebuilt.

#define MOE_HIGH_MAX_BLK   1024
#define MOE_HIGH_MAX_ROLE  4

enum moe_high_state : uint8_t {
    MOE_HIGH_LOW = 0,
    MOE_HIGH_PROMOTING,
    MOE_HIGH_PROTECTED,
    MOE_HIGH_EVICTABLE,
};

struct moe_high_layer {
    moe_high_state state = MOE_HIGH_LOW;
    int dev = 0;

    const void * host[MOE_HIGH_MAX_ROLE] = {};
    size_t bytes[MOE_HIGH_MAX_ROLE] = {};
    uint8_t known_mask = 0;
    uint8_t required_mask = 0;
    uint8_t prepared_mask = 0; // persistent buffers allocated
    uint8_t content_mask = 0;  // canonical host bytes copied

    // Persistent canonical GPU weight buffers. Tensor metadata remains owned by
    // the scheduler context and is recreated safely on each graph rebuild.
    ggml_backend_buffer_t weight_buffer[MOE_HIGH_MAX_ROLE] = {};
    size_t alloc_bytes[MOE_HIGH_MAX_ROLE] = {};
    bool preflight_sync_done = false;

    uint8_t sample_seen = 0;
    uint8_t sample_hit = 0;
    std::deque<uint8_t> all_hit_window;
    int all_hit_count = 0;

    uint64_t protected_until = 0;
    uint64_t cooldown_until = 0;
    uint64_t promoted_serial = 0;
    bool empty_fill_promotion = false;
};

struct moe_high_global {
    std::mutex mu;
    bool enabled = true;
    int low_pct = 50;
    int high_pct = 50;
    int window = 128;
    int promote_bp = 9950; // 99.50%, in basis points
    int protect_tokens = 256;
    int cooldown_tokens = 128;

    size_t budget[MOE_CACHE_MAX_DEV] = {};
    size_t used[MOE_CACHE_MAX_DEV] = {};
    bool low_budget_claimed[MOE_CACHE_MAX_DEV] = {};
    moe_high_layer layer[MOE_HIGH_MAX_BLK];

    int last_start_blk = -1;
    int promoting_blk = -1; // serialize prototype promotions
    uint64_t token_epoch = 0;
    uint64_t generation = 1;
    uint64_t promotion_serial = 0;
};

static moe_high_global & g_high = *new moe_high_global();

static int moe_high_role(const char * name) {
    if (!name) return -1;
    if (strstr(name, "_gate_up_exps")) return 3;
    if (strstr(name, "_gate_exps"))    return 0;
    if (strstr(name, "_up_exps"))      return 1;
    if (strstr(name, "_down_exps"))    return 2;
    return -1;
}

static int moe_high_block(const char * name) {
    const char * p = name ? strstr(name, "blk.") : nullptr;
    return p ? atoi(p + 4) : -1;
}

static uint8_t moe_high_required_mask(const moe_high_layer & l) {
    // Separate gate/up/down or a combined gate_up plus down.
    if ((l.known_mask & 0x07) == 0x07) return 0x07;
    if ((l.known_mask & 0x0c) == 0x0c) return 0x0c;
    return 0;
}

static size_t moe_high_layer_bytes(const moe_high_layer & l) {
    size_t result = 0;
    const uint8_t mask = l.required_mask ? l.required_mask : moe_high_required_mask(l);
    for (int role = 0; role < MOE_HIGH_MAX_ROLE; ++role) {
        if (mask & (1u << role)) result += l.bytes[role];
    }
    return result;
}

static size_t moe_high_layer_charge(const moe_high_layer & l) {
    size_t result = 0;
    const uint8_t mask = l.required_mask ? l.required_mask : moe_high_required_mask(l);
    for (int role = 0; role < MOE_HIGH_MAX_ROLE; ++role) {
        if (!(mask & (1u << role))) continue;
        result += l.alloc_bytes[role] ? l.alloc_bytes[role] : l.bytes[role];
    }
    return result;
}

static int moe_high_env_int(const char * name, int value, int lo, int hi) {
    if (const char * e = getenv(name)) value = atoi(e);
    return std::max(lo, std::min(value, hi));
}

static void moe_high_configure(void) {
    int requested_high_pct = 50;
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_HIGH_PCT")) requested_high_pct = atoi(e);
    g_high.high_pct = std::max(0, std::min(requested_high_pct, 80));
    g_high.low_pct = 100 - g_high.high_pct;
    g_high.enabled = g_high.high_pct > 0;
    g_high.window = moe_high_env_int("GGML_CUDA_MOE_CACHE_PROMOTE_WINDOW", 128, 8, 4096);
    g_high.promote_bp = moe_high_env_int("GGML_CUDA_MOE_CACHE_PROMOTE_BP", 9950, 1, 10000);
    g_high.protect_tokens = moe_high_env_int("GGML_CUDA_MOE_CACHE_PROTECT_TOKENS", 256, 1, 1 << 20);
    g_high.cooldown_tokens = moe_high_env_int("GGML_CUDA_MOE_CACHE_COOLDOWN_TOKENS", 128, 0, 1 << 20);

    if (requested_high_pct != g_high.high_pct) {
        MOE_CACHE_LOG("[moe-cache] GGML_CUDA_MOE_CACHE_HIGH_PCT=%d clamped to %d\n",
                      requested_high_pct, g_high.high_pct);
    }
    if (!g_high.enabled) {
        MOE_CACHE_LOG("[moe-cache] tiered cache disabled: high=0%%, legacy LOW cache uses 100%% of cache budget\n");
        return;
    }

    MOE_CACHE_LOG("[moe-cache] tiers: low=%d%% high=%d%% promote=%d/%d bp window=%d protect=%d cooldown=%d\n",
                  g_high.low_pct, g_high.high_pct, g_high.promote_bp, 10000, g_high.window,
                  g_high.protect_tokens, g_high.cooldown_tokens);
}

static void moe_high_bump_generation_locked(void) {
    ++g_high.generation;
}

static size_t moe_high_set_device_budget(int dev, size_t total_budget) {
    if (dev < 0 || dev >= MOE_CACHE_MAX_DEV) return 0;
    std::lock_guard<std::mutex> lk(g_high.mu);
    if (g_high.budget[dev] == 0 && g_high.enabled) {
        g_high.budget[dev] = total_budget * (size_t) g_high.high_pct / 100;
    }

    // LOW pool construction may be retried when a late tensor shape is seen.
    // Grant the per-device LOW budget only once. With HIGH=0 this returns the
    // full budget, preserving the legacy expert-cache behavior.
    if (g_high.low_budget_claimed[dev]) return 0;
    g_high.low_budget_claimed[dev] = true;
    return total_budget * (size_t) g_high.low_pct / 100;
}

static void moe_high_note_begin(const char * name, const void * host_base,
                                size_t expert_size, int64_t n_expert, int64_t n_tokens) {
    if (!g_high.enabled || n_tokens != 1) return;
    const int blk = moe_high_block(name);
    const int role = moe_high_role(name);
    if (blk < 0 || blk >= MOE_HIGH_MAX_BLK || role < 0) return;

    std::lock_guard<std::mutex> lk(g_high.mu);
    moe_high_layer & l = g_high.layer[blk];
    l.dev = g.n_dev > 0 ? blk % g.n_dev : 0;
    l.host[role] = host_base;
    l.bytes[role] = expert_size * (size_t) n_expert;
    l.known_mask |= (uint8_t) (1u << role);
    l.required_mask = moe_high_required_mask(l);

    const bool starts_layer_sample = role == 0 || role == 3;
    if (starts_layer_sample) {
        if (g_high.last_start_blk >= 0 && blk <= g_high.last_start_blk) ++g_high.token_epoch;
        g_high.last_start_blk = blk;
        l.sample_seen = 0;
        l.sample_hit = 0;
    }
}

static bool moe_high_layout_compatible(const moe_high_layer & incoming, const moe_high_layer & victim) {
    if (!incoming.required_mask || incoming.required_mask != victim.required_mask) return false;
    for (int role = 0; role < MOE_HIGH_MAX_ROLE; ++role) {
        const uint8_t bit = (uint8_t) (1u << role);
        if (!(incoming.required_mask & bit)) continue;
        if (incoming.bytes[role] != victim.bytes[role] || !victim.weight_buffer[role]) return false;
    }
    return true;
}

static int moe_high_select_victim_locked(int dev, const moe_high_layer & incoming) {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int blk = 0; blk < MOE_HIGH_MAX_BLK; ++blk) {
        moe_high_layer & l = g_high.layer[blk];
        if (l.dev != dev) continue;
        if (l.state == MOE_HIGH_PROTECTED && g_high.token_epoch >= l.protected_until) {
            l.state = MOE_HIGH_EVICTABLE;
        }
        if (l.state != MOE_HIGH_EVICTABLE || !moe_high_layout_compatible(incoming, l)) continue;
        if (l.promoted_serial < oldest) {
            oldest = l.promoted_serial;
            victim = blk;
        }
    }
    return victim;
}

static void moe_high_transfer_victim_locked(int victim_blk, int incoming_blk) {
    moe_high_layer & victim = g_high.layer[victim_blk];
    moe_high_layer & incoming = g_high.layer[incoming_blk];
    const size_t bytes = moe_high_layer_charge(victim);

    for (int role = 0; role < MOE_HIGH_MAX_ROLE; ++role) {
        incoming.weight_buffer[role] = victim.weight_buffer[role];
        incoming.alloc_bytes[role] = victim.alloc_bytes[role];
        victim.weight_buffer[role] = nullptr;
        victim.alloc_bytes[role] = 0;
    }

    if (victim.dev >= 0 && victim.dev < MOE_CACHE_MAX_DEV) {
        g_high.used[victim.dev] = g_high.used[victim.dev] > bytes ? g_high.used[victim.dev] - bytes : 0;
    }
    victim.state = MOE_HIGH_LOW;
    victim.prepared_mask = 0;
    victim.content_mask = 0;
    victim.preflight_sync_done = false;
    victim.all_hit_window.clear();
    victim.all_hit_count = 0;
    victim.sample_seen = 0;
    victim.sample_hit = 0;
    victim.cooldown_until = g_high.token_epoch + (uint64_t) g_high.cooldown_tokens;
    victim.empty_fill_promotion = false;

    MOE_CACHE_LOG("[moe-cache] high-cache evict: blk=%d bytes=%zu MiB cooldown-until=%llu\n",
                  victim_blk, bytes >> 20, (unsigned long long) victim.cooldown_until);
}

static void moe_high_try_promote_locked(int blk) {
    moe_high_layer & l = g_high.layer[blk];
    if (!g_high.enabled || l.state != MOE_HIGH_LOW || !l.required_mask) return;
    if (g_high.promoting_blk >= 0) return;
    if (g_high.token_epoch < l.cooldown_until) return;

    const size_t bytes = moe_high_layer_bytes(l);
    const int dev = l.dev;
    if (bytes == 0 || dev < 0 || dev >= MOE_CACHE_MAX_DEV || bytes > g_high.budget[dev]) return;

    const int samples = (int) l.all_hit_window.size();
    const int ratio_bp = samples > 0 ? l.all_hit_count * 10000 / samples : 0;
    const bool latest_full_hit = samples > 0 && l.all_hit_window.back() != 0;
    const bool has_empty_slot = g_high.used[dev] + bytes <= g_high.budget[dev];
    const bool empty_fill = has_empty_slot && latest_full_hit;
    const bool ratio_qualified = samples >= g_high.window && ratio_bp >= g_high.promote_bp;
    if (!empty_fill && !ratio_qualified) return;

    int victim = -1;
    if (!has_empty_slot) {
        // The persistent-buffer prototype only recycles a layout-compatible
        // complete layer. Refuse rather than freeing a partial victim set.
        victim = moe_high_select_victim_locked(dev, l);
        if (victim < 0) return;
        moe_high_transfer_victim_locked(victim, blk);
    }

    l.state = MOE_HIGH_PROMOTING;
    l.prepared_mask = 0;
    l.content_mask = 0;
    l.preflight_sync_done = false;
    l.empty_fill_promotion = empty_fill;
    // A recycled victim carries its actual backend allocation sizes; a fresh
    // promotion reserves canonical tensor bytes and is adjusted in preflight.
    g_high.used[dev] += moe_high_layer_charge(l);
    g_high.promoting_blk = blk;
    moe_high_bump_generation_locked();

    MOE_CACHE_LOG("[moe-cache] high-cache promote: blk=%d dev=%d all-hit=%d.%02d%% samples=%d bytes=%zu MiB victim=%d mode=%s\n",
                  blk, dev, ratio_bp / 100, ratio_bp % 100, samples, bytes >> 20, victim,
                  empty_fill ? "empty-fill" : "window");
}

static void moe_high_note_plan(int blk, int role, bool all_rows_hit) {
    if (!g_high.enabled || blk < 0 || blk >= MOE_HIGH_MAX_BLK || role < 0 || role >= MOE_HIGH_MAX_ROLE) return;
    std::lock_guard<std::mutex> lk(g_high.mu);
    moe_high_layer & l = g_high.layer[blk];
    if (l.state != MOE_HIGH_LOW) return;

    const uint8_t bit = (uint8_t) (1u << role);
    l.sample_seen |= bit;
    if (all_rows_hit) l.sample_hit |= bit;

    if (role != 2 || !l.required_mask || (l.sample_seen & l.required_mask) != l.required_mask) return;
    const uint8_t hit = (l.sample_hit & l.required_mask) == l.required_mask ? 1 : 0;
    l.all_hit_window.push_back(hit);
    l.all_hit_count += hit;
    while ((int) l.all_hit_window.size() > g_high.window) {
        l.all_hit_count -= l.all_hit_window.front();
        l.all_hit_window.pop_front();
    }
    moe_high_try_promote_locked(blk);
}

static bool moe_high_layer_active(int blk) {
    if (blk < 0 || blk >= MOE_HIGH_MAX_BLK) return false;
    std::lock_guard<std::mutex> lk(g_high.mu);
    return g_high.layer[blk].state != MOE_HIGH_LOW;
}

// Remove the promoted layer's completed or queued LOW entries. In-flight
// worker entries observe HIGH state on completion and discard themselves.
static void moe_high_purge_low(int blk) {
    std::lock_guard<std::mutex> lk(g.mu);

    for (auto it = g.queue.begin(); it != g.queue.end();) {
        if (it->blk != blk) {
            ++it;
            continue;
        }
        moe_cache_pool & p = g.dev[it->dev].pools[it->pool];
        if (it->slot_idx >= 0 && it->slot_idx < p.n_used) {
            moe_cache_slot & s = p.slots[it->slot_idx];
            if (s.queued && s.key == it->key) {
                p.map.erase(s.key);
                moe_cache_slot_clear(p, it->slot_idx);
            }
        }
        it = g.queue.erase(it);
    }

    for (int dev = 0; dev < g.n_dev; ++dev) {
        for (int pi = 0; pi < g.dev[dev].n_pools; ++pi) {
            moe_cache_pool & p = g.dev[dev].pools[pi];
            for (int si = 0; si < p.n_used; ++si) {
                moe_cache_slot & s = p.slots[si];
                if (s.blk != blk || s.queued || (!s.valid && !s.key)) continue;
                p.map.erase(s.key);
                moe_cache_slot_clear(p, si);
            }
        }
    }
}

static int moe_high_backend(const struct ggml_tensor * node) {
    if (!g_high.enabled || !node || node->op != GGML_OP_MUL_MAT_ID || !node->src[0]) return 0;
    const int blk = moe_high_block(node->src[0]->name);
    if (blk < 0 || blk >= MOE_HIGH_MAX_BLK) return 0;
    std::lock_guard<std::mutex> lk(g_high.mu);
    moe_high_layer & l = g_high.layer[blk];
    if (l.state == MOE_HIGH_PROTECTED && g_high.token_epoch >= l.protected_until) {
        l.state = MOE_HIGH_EVICTABLE;
    }
    return l.state == MOE_HIGH_LOW ? 0 : l.dev + 1;
}

static void moe_high_free_buffers_locked(moe_high_layer & l) {
    for (int role = 0; role < MOE_HIGH_MAX_ROLE; ++role) {
        if (l.weight_buffer[role]) {
            ggml_backend_buffer_free(l.weight_buffer[role]);
            l.weight_buffer[role] = nullptr;
        }
        l.alloc_bytes[role] = 0;
    }
    l.prepared_mask = 0;
    l.content_mask = 0;
}

static void moe_high_abort_promotion_locked(int blk, const char * reason) {
    moe_high_layer & l = g_high.layer[blk];
    const size_t charge = moe_high_layer_charge(l);
    moe_high_free_buffers_locked(l);
    if (l.dev >= 0 && l.dev < MOE_CACHE_MAX_DEV) {
        g_high.used[l.dev] = g_high.used[l.dev] > charge ? g_high.used[l.dev] - charge : 0;
    }
    l.state = MOE_HIGH_LOW;
    l.preflight_sync_done = false;
    l.empty_fill_promotion = false;
    l.cooldown_until = g_high.token_epoch + (uint64_t) g_high.cooldown_tokens;
    if (g_high.promoting_blk == blk) g_high.promoting_blk = -1;
    MOE_CACHE_LOG("[moe-cache] high-cache promotion aborted: blk=%d reason=%s cooldown-until=%llu\n",
                  blk, reason, (unsigned long long) l.cooldown_until);
}

// Allocate persistent buffers before assigning any node in the layer
// to the GPU. This does not mutate graph tensor pointers.
static int moe_high_preflight(struct ggml_tensor * node, void * backend_ptr, void * buft_ptr) {
    if (!node || node->op != GGML_OP_MUL_MAT_ID || !node->src[0] || !backend_ptr || !buft_ptr) return 0;

    const struct ggml_tensor * host = node->src[0];
    const int blk = moe_high_block(host->name);
    const int role = moe_high_role(host->name);
    if (blk < 0 || blk >= MOE_HIGH_MAX_BLK || role < 0) return 0;

    ggml_backend_t backend = (ggml_backend_t) backend_ptr;
    ggml_backend_buffer_type_t buft = (ggml_backend_buffer_type_t) buft_ptr;

    std::lock_guard<std::mutex> lk(g_high.mu);
    moe_high_layer & l = g_high.layer[blk];
    if (l.state == MOE_HIGH_LOW) return 0;

    const uint8_t bit = (uint8_t) (1u << role);
    if (!(l.required_mask & bit)) return 0;
    if (host->data != l.host[role] || host->view_src != nullptr) {
        if (l.state == MOE_HIGH_PROMOTING) {
            moe_high_abort_promotion_locked(blk, "host tensor mismatch/view");
        }
        return 0;
    }
    if ((l.prepared_mask & bit) && l.weight_buffer[role]) return 1;

    if (!l.preflight_sync_done) {
        // A recycled victim buffer may still be referenced by the previous
        // graph. The prototype is single-context and graph generation changes
        // force a rebuild before this preflight executes.
        ggml_backend_synchronize(backend);
        l.preflight_sync_done = true;
    }

    struct ggml_tensor layout = *host;
    layout.buffer = nullptr;
    layout.data = nullptr;
    layout.view_src = nullptr;
    layout.view_offs = 0;
    layout.extra = nullptr;

    const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, &layout);
    const size_t old_charge = l.alloc_bytes[role] ? l.alloc_bytes[role] : l.bytes[role];
    ggml_backend_buffer_t buffer = l.weight_buffer[role];

    if (buffer && (ggml_backend_buffer_get_type(buffer) != buft ||
                   ggml_backend_buffer_get_size(buffer) < alloc_size)) {
        moe_high_abort_promotion_locked(blk, "recycled buffer layout mismatch");
        return 0;
    }

    if (!buffer) {
        if (alloc_size > old_charge) {
            const size_t delta = alloc_size - old_charge;
            if (g_high.used[l.dev] + delta > g_high.budget[l.dev]) {
                moe_high_abort_promotion_locked(blk, "actual allocation exceeds HIGH budget");
                return 0;
            }
            g_high.used[l.dev] += delta;
        } else {
            g_high.used[l.dev] -= std::min(g_high.used[l.dev], old_charge - alloc_size);
        }

        // Record the adjusted charge before allocation so the abort path can
        // unwind it exactly if the backend allocation fails.
        l.alloc_bytes[role] = alloc_size;
        buffer = ggml_backend_buft_alloc_buffer(buft, alloc_size);
        if (!buffer) {
            moe_high_abort_promotion_locked(blk, "GPU buffer allocation failed");
            return 0;
        }
        l.weight_buffer[role] = buffer;
    } else {
        l.alloc_bytes[role] = ggml_backend_buffer_get_size(buffer);
    }

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    l.prepared_mask |= bit;
    return 1;
}

static int moe_high_ready(const struct ggml_tensor * node) {
    if (!g_high.enabled || !node || node->op != GGML_OP_MUL_MAT_ID || !node->src[0]) return 0;
    const int blk = moe_high_block(node->src[0]->name);
    const int role = moe_high_role(node->src[0]->name);
    if (blk < 0 || blk >= MOE_HIGH_MAX_BLK || role < 0) return 0;

    std::lock_guard<std::mutex> lk(g_high.mu);
    moe_high_layer & l = g_high.layer[blk];
    if (l.state == MOE_HIGH_PROTECTED && g_high.token_epoch >= l.protected_until) {
        l.state = MOE_HIGH_EVICTABLE;
    }
    const uint8_t bit = (uint8_t) (1u << role);
    return l.state != MOE_HIGH_LOW &&
           (l.prepared_mask & l.required_mask) == l.required_mask &&
           (l.required_mask & bit) && l.weight_buffer[role] != nullptr;
}

// Bind scheduler-owned copy tensor metadata to a persistent HIGH buffer. The
// metadata is safe to discard on scheduler reset; only the backend buffer is
// retained. Returns non-zero when normal graph allocation/copy must be skipped.
static int moe_high_bind_copy(const struct ggml_tensor * host,
                              struct ggml_tensor * gpu_copy,
                              void * backend_ptr,
                              void * buft_ptr) {
    if (!host || !gpu_copy || !backend_ptr || !buft_ptr) return 0;
    const int blk = moe_high_block(host->name);
    const int role = moe_high_role(host->name);
    if (blk < 0 || blk >= MOE_HIGH_MAX_BLK || role < 0) return 0;

    bool completed = false;
    bool completed_empty_fill = false;
    size_t resident_bytes = 0;
    uint64_t protected_until = 0;

    {
        std::lock_guard<std::mutex> lk(g_high.mu);
        moe_high_layer & l = g_high.layer[blk];
        const uint8_t bit = (uint8_t) (1u << role);
        if (l.state == MOE_HIGH_LOW || l.host[role] != host->data ||
            !(l.prepared_mask & bit) || !l.weight_buffer[role]) {
            return 0;
        }

        ggml_backend_buffer_t buffer = l.weight_buffer[role];
        if (ggml_backend_buffer_get_type(buffer) != (ggml_backend_buffer_type_t) buft_ptr) return 0;

        gpu_copy->buffer = buffer;
        gpu_copy->data = ggml_backend_buffer_get_base(buffer);
        gpu_copy->view_src = nullptr;
        gpu_copy->view_offs = 0;
        gpu_copy->extra = nullptr;
        if (ggml_backend_buffer_init_tensor(buffer, gpu_copy) != GGML_STATUS_SUCCESS) {
            gpu_copy->buffer = nullptr;
            gpu_copy->data = nullptr;
            gpu_copy->extra = nullptr;
            return 0;
        }

        if (!(l.content_mask & bit)) {
            ggml_backend_tensor_set(gpu_copy, host->data, 0, ggml_nbytes(host));
            l.content_mask |= bit;
        }

        if (l.state == MOE_HIGH_PROMOTING &&
            (l.content_mask & l.required_mask) == l.required_mask) {
            l.state = MOE_HIGH_PROTECTED;
            const int protect_tokens = l.empty_fill_promotion
                ? std::max(1, g_high.protect_tokens / 2)
                : g_high.protect_tokens;
            l.protected_until = g_high.token_epoch + (uint64_t) protect_tokens;
            l.promoted_serial = ++g_high.promotion_serial;
            if (g_high.promoting_blk == blk) g_high.promoting_blk = -1;
            completed = true;
            completed_empty_fill = l.empty_fill_promotion;
            resident_bytes = moe_high_layer_charge(l);
            protected_until = l.protected_until;
        }
    }

    if (completed) {
        moe_high_purge_low(blk);
        MOE_CACHE_LOG("[moe-cache] high-cache resident: blk=%d bytes=%zu MiB protected-until=%llu mode=%s\n",
                      blk, resident_bytes >> 20, (unsigned long long) protected_until,
                      completed_empty_fill ? "empty-fill" : "window");
    }
    return 1;
}

// Existing compute-copy hook: a bound persistent tensor already contains the
// canonical full weight, so no selective or full scheduler copy is required.
static int moe_high_prepare(const struct ggml_tensor * host, const struct ggml_tensor * gpu_copy) {
    if (!g_high.enabled || !host || !gpu_copy || !gpu_copy->buffer) return 0;
    const int blk = moe_high_block(host->name);
    const int role = moe_high_role(host->name);
    if (blk < 0 || blk >= MOE_HIGH_MAX_BLK || role < 0) return 0;

    std::lock_guard<std::mutex> lk(g_high.mu);
    const moe_high_layer & l = g_high.layer[blk];
    const uint8_t bit = (uint8_t) (1u << role);
    if (l.state == MOE_HIGH_LOW || l.host[role] != host->data) return 0;
    return l.weight_buffer[role] == gpu_copy->buffer && (l.content_mask & bit) ? 2 : 0;
}

static uint64_t moe_high_generation(void) {
    std::lock_guard<std::mutex> lk(g_high.mu);
    return g_high.generation;
}

static void moe_high_invalidate(const void * base, size_t size) {
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    std::lock_guard<std::mutex> lk(g_high.mu);
    bool changed = false;
    for (int blk = 0; blk < MOE_HIGH_MAX_BLK; ++blk) {
        moe_high_layer & l = g_high.layer[blk];
        bool overlap = false;
        for (int role = 0; role < MOE_HIGH_MAX_ROLE; ++role) {
            const char * p = (const char *) l.host[role];
            if (p && p < hi && p + l.bytes[role] > lo) overlap = true;
        }
        if (!overlap) continue;
        const size_t bytes = moe_high_layer_charge(l);
        if (l.state != MOE_HIGH_LOW && l.dev >= 0 && l.dev < MOE_CACHE_MAX_DEV) {
            g_high.used[l.dev] = g_high.used[l.dev] > bytes ? g_high.used[l.dev] - bytes : 0;
            changed = true;
        }
        if (g_high.promoting_blk == blk) g_high.promoting_blk = -1;
        moe_high_free_buffers_locked(l);
        l = moe_high_layer{};
    }
    if (changed) moe_high_bump_generation_locked();
}
