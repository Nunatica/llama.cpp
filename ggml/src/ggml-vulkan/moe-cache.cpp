#include "moe-cache.hpp"

#include "ggml-vulkan.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml.h"

#include "../ggml-backend-moe-cache.h"
#include "../ggml-impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int    VK_MOE_MAX_HITS           = 64;
constexpr size_t VK_MOE_META_BYTES         = 2u * 1024u * 1024u;
constexpr size_t VK_MOE_DEFAULT_BUDGET_MB  = 1024;
constexpr size_t VK_MOE_DEFAULT_RESERVE_MB = 256;
constexpr int    VK_MOE_DEFAULT_POOL_DIV   = 2;
constexpr int    VK_MOE_DEFAULT_MAX_SLOTS  = 1024;
constexpr int    VK_MOE_DEFAULT_INSERTS    = 4;
constexpr int    VK_MOE_DEFAULT_ADMIT      = 2;
constexpr int    VK_MOE_DEFAULT_QUEUE      = 256;
constexpr int    VK_MOE_DEFAULT_STATS      = 10;
constexpr size_t VK_MOE_DEFAULT_MIN_KB     = 128;

static size_t env_size(const char * name, size_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        GGML_LOG_WARN("[vk-moe-cache] ignoring invalid %s=%s\n", name, value);
        return fallback;
    }
    if (parsed > std::numeric_limits<size_t>::max()) {
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(parsed);
}

static int env_int(const char * name, int fallback, int lo, int hi) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < lo || parsed > hi) {
        GGML_LOG_WARN("[vk-moe-cache] ignoring invalid %s=%s\n", name, value);
        return fallback;
    }
    return static_cast<int>(parsed);
}

static size_t scaled_bytes(size_t value, size_t scale) {
    if (value > std::numeric_limits<size_t>::max() / scale) {
        return std::numeric_limits<size_t>::max();
    }
    return value * scale;
}

static size_t saturated_mul(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return std::numeric_limits<size_t>::max();
    }
    return a * b;
}

static bool env_enabled(const char * name, bool fallback) {
    const char * value = std::getenv(name);
    return value == nullptr ? fallback : std::atoi(value) > 0;
}

static uint64_t fnv1a(const char * text) {
    uint64_t h = 1469598103934665603ull;
    if (text != nullptr) {
        while (*text != '\0') {
            h ^= static_cast<unsigned char>(*text++);
            h *= 1099511628211ull;
        }
    }
    return h;
}

static uint64_t pointer_hash(const void * pointer) {
    uint64_t x = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static uint64_t expert_key(uint64_t tensor_hash, int32_t expert) {
    uint64_t x = tensor_hash ^ (static_cast<uint64_t>(static_cast<uint32_t>(expert)) + 0x9e3779b97f4a7c15ull);
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static int parse_block(const char * name) {
    if (name == nullptr) {
        return 0;
    }
    const char * p = std::strstr(name, "blk.");
    p = p != nullptr ? p + 4 : name;
    while (*p != '\0' && (*p < '0' || *p > '9')) {
        ++p;
    }
    int block = 0;
    while (*p >= '0' && *p <= '9') {
        block = block * 10 + (*p++ - '0');
    }
    return block;
}

static bool overlaps(const void * a_ptr, size_t a_size, const void * b_ptr, size_t b_size) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(a_ptr);
    const uintptr_t b = reinterpret_cast<uintptr_t>(b_ptr);
    const uintptr_t a_end = a_size > std::numeric_limits<uintptr_t>::max() - a
        ? std::numeric_limits<uintptr_t>::max() : a + a_size;
    const uintptr_t b_end = b_size > std::numeric_limits<uintptr_t>::max() - b
        ? std::numeric_limits<uintptr_t>::max() : b + b_size;
    return a < b_end && b < a_end;
}

struct graph_state {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * activations = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * output = nullptr;
    ggml_gallocr_t allocator = nullptr;
    int n_hits = 0;
    std::vector<float> host_activations;
    std::vector<int32_t> host_ids;
    std::vector<float> host_output;

    ~graph_state() {
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct cache_slot {
    uint64_t key = 0;
    uint64_t generation = 0;
    uint64_t last_use = 0;
    const void * source = nullptr;
    int32_t expert = -1;
    bool valid = false;
    bool queued = false;
};

struct cache_pool {
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t n_in = 0;
    int64_t n_out = 0;
    size_t expert_size = 0;
    int capacity = 0;

    ggml_context * weights_ctx = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;

    std::vector<cache_slot> slots;
    std::unordered_map<uint64_t, int> key_to_slot;
    std::unordered_map<uint64_t, uint16_t> frequency;
    std::unordered_map<int, std::unique_ptr<graph_state>> graphs;
    uint64_t clock = 0;
    bool disabled = false;

    ~cache_pool() {
        graphs.clear();
        if (weights_buffer != nullptr) {
            ggml_backend_buffer_free(weights_buffer);
        }
        if (weights_ctx != nullptr) {
            ggml_free(weights_ctx);
        }
    }
};

struct device_state {
    int id = -1;
    ggml_backend_t backend = nullptr;
    size_t budget = 0;
    size_t used = 0;
    std::mutex gpu_mutex;
    std::vector<std::unique_ptr<cache_pool>> pools;

    graph_state * active_graph = nullptr;
    cache_pool * active_pool = nullptr;
    int active_hits = 0;
    int64_t active_out = 0;
    bool compute_locked = false;
    std::atomic<bool> dispatch_pending{false};
};

struct node_state {
    int device = -1;
    cache_pool * pool = nullptr;
    const void * host_base = nullptr;
    size_t expert_size = 0;
    int64_t n_expert = 0;
    int64_t n_in = 0;
    int64_t n_out = 0;
    enum ggml_type type = GGML_TYPE_COUNT;
    uint64_t tensor_hash = 0;
    bool valid = false;
};

struct insert_job {
    int device = -1;
    cache_pool * pool = nullptr;
    int slot = -1;
    uint64_t key = 0;
    uint64_t generation = 0;
    const void * source = nullptr;
    size_t size = 0;
};

struct cache_state {
    bool enabled = false;
    int n_devices = 0;
    size_t budget_mb = VK_MOE_DEFAULT_BUDGET_MB;
    size_t reserve_mb = VK_MOE_DEFAULT_RESERVE_MB;
    size_t min_expert_bytes = VK_MOE_DEFAULT_MIN_KB * 1024u;
    int pool_divisor = VK_MOE_DEFAULT_POOL_DIV;
    int max_slots = VK_MOE_DEFAULT_MAX_SLOTS;
    int inserts_per_plan = VK_MOE_DEFAULT_INSERTS;
    int admission_count = VK_MOE_DEFAULT_ADMIT;
    int max_queue = VK_MOE_DEFAULT_QUEUE;
    int stats_every = VK_MOE_DEFAULT_STATS;
    int device_override = -1;

    std::mutex mutex;
    std::condition_variable condition;
    std::array<std::unique_ptr<device_state>, GGML_VK_MAX_DEVICES> devices;
    std::deque<insert_job> queue;
    bool worker_started = false;
    const void * active_source = nullptr;
    size_t active_source_size = 0;
    node_state current;

    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t inserts = 0;
    uint64_t evictions = 0;
    uint64_t queue_drops = 0;
    uint64_t bytes_uploaded = 0;
    int64_t last_stats_us = 0;
};

static cache_state & state() {
    // The upload worker is detached, so the process-lifetime allocation avoids
    // static-destruction ordering hazards in dynamically loaded backends.
    static cache_state * value = new cache_state();
    return *value;
}

static void vk_cache_stats_locked(cache_state & g, int64_t now);

static graph_state * create_graph(device_state & dev, cache_pool & pool, int n_hits) {
    if (n_hits <= 0 || n_hits > VK_MOE_MAX_HITS) {
        return nullptr;
    }

    auto graph = std::make_unique<graph_state>();
    graph->n_hits = n_hits;

    ggml_init_params params = {};
    params.mem_size = VK_MOE_META_BYTES;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    graph->ctx = ggml_init(params);
    if (graph->ctx == nullptr) {
        return nullptr;
    }

    graph->activations = ggml_new_tensor_3d(graph->ctx, GGML_TYPE_F32, pool.n_in, n_hits, 1);
    graph->ids = ggml_new_tensor_2d(graph->ctx, GGML_TYPE_I32, n_hits, 1);
    graph->output = ggml_mul_mat_id(graph->ctx, pool.weights, graph->activations, graph->ids);
    if (graph->activations == nullptr || graph->ids == nullptr || graph->output == nullptr) {
        return nullptr;
    }

    ggml_set_name(graph->activations, "vk_moe_cache_activations");
    ggml_set_name(graph->ids, "vk_moe_cache_ids");
    ggml_set_name(graph->output, "vk_moe_cache_output");
    ggml_set_input(graph->activations);
    ggml_set_input(graph->ids);
    ggml_set_output(graph->output);

    if (!ggml_backend_supports_op(dev.backend, graph->output)) {
        return nullptr;
    }

    graph->graph = ggml_new_graph_custom(graph->ctx, 16, false);
    if (graph->graph == nullptr) {
        return nullptr;
    }
    ggml_build_forward_expand(graph->graph, graph->output);

    graph->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dev.backend));
    if (graph->allocator == nullptr || !ggml_gallocr_alloc_graph(graph->allocator, graph->graph)) {
        return nullptr;
    }

    graph->host_activations.resize(static_cast<size_t>(n_hits) * static_cast<size_t>(pool.n_in));
    graph->host_ids.resize(n_hits);
    graph->host_output.resize(static_cast<size_t>(n_hits) * static_cast<size_t>(pool.n_out));

    graph_state * result = graph.get();
    pool.graphs.emplace(n_hits, std::move(graph));
    return result;
}

static graph_state * get_graph(device_state & dev, cache_pool & pool, int n_hits) {
    const auto found = pool.graphs.find(n_hits);
    if (found != pool.graphs.end()) {
        return found->second.get();
    }
    return create_graph(dev, pool, n_hits);
}

static device_state * get_device_locked(cache_state & g, int id) {
    if (id < 0 || id >= g.n_devices || id >= GGML_VK_MAX_DEVICES) {
        return nullptr;
    }
    if (g.devices[id] != nullptr) {
        return g.devices[id].get();
    }

    auto dev = std::make_unique<device_state>();
    dev->id = id;
    dev->backend = ggml_backend_vk_init(static_cast<size_t>(id));
    if (dev->backend == nullptr) {
        GGML_LOG_WARN("[vk-moe-cache] failed to initialize Vulkan device %d\n", id);
        return nullptr;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_vk_get_device_memory(id, &free_bytes, &total_bytes);

    const size_t requested = scaled_bytes(g.budget_mb, 1024u * 1024u);
    const size_t reserve = scaled_bytes(g.reserve_mb, 1024u * 1024u);
    const size_t available = free_bytes > reserve ? free_bytes - reserve : 0;
    dev->budget = free_bytes == 0 ? requested : std::min(requested, available);
    if (dev->budget < g.min_expert_bytes * 2) {
        GGML_LOG_WARN("[vk-moe-cache] device %d has insufficient free VRAM for cache (%zu MiB usable)\n",
                id, dev->budget >> 20);
        ggml_backend_free(dev->backend);
        return nullptr;
    }

    GGML_LOG_INFO("[vk-moe-cache] device=%d budget=%zu MiB free=%zu MiB total=%zu MiB\n",
            id, dev->budget >> 20, free_bytes >> 20, total_bytes >> 20);

    device_state * result = dev.get();
    g.devices[id] = std::move(dev);
    return result;
}

static cache_pool * find_pool(device_state & dev, enum ggml_type type, int64_t n_in, int64_t n_out, size_t expert_size) {
    for (const auto & item : dev.pools) {
        cache_pool * pool = item.get();
        if (!pool->disabled && pool->type == type && pool->n_in == n_in && pool->n_out == n_out &&
                pool->expert_size == expert_size) {
            return pool;
        }
    }
    return nullptr;
}

static cache_pool * create_pool_locked(cache_state & g, device_state & dev, enum ggml_type type,
        int64_t n_in, int64_t n_out, size_t expert_size) {
    const size_t remaining = dev.budget > dev.used ? dev.budget - dev.used : 0;
    const size_t target = std::min(remaining, dev.budget / static_cast<size_t>(g.pool_divisor));
    const size_t raw_capacity = expert_size == 0 ? 0 : target / expert_size;
    const int capacity = static_cast<int>(std::min<size_t>(raw_capacity, static_cast<size_t>(g.max_slots)));
    if (capacity < 2) {
        GGML_LOG_WARN("[vk-moe-cache] no room for pool type=%s shape=%lldx%lld expert=%zu KiB\n",
                ggml_type_name(type), static_cast<long long>(n_in), static_cast<long long>(n_out), expert_size >> 10);
        return nullptr;
    }

    auto pool = std::make_unique<cache_pool>();
    pool->type = type;
    pool->n_in = n_in;
    pool->n_out = n_out;
    pool->expert_size = expert_size;
    pool->capacity = capacity;
    pool->slots.resize(capacity);

    ggml_init_params params = {};
    params.mem_size = VK_MOE_META_BYTES;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    pool->weights_ctx = ggml_init(params);
    if (pool->weights_ctx == nullptr) {
        return nullptr;
    }

    pool->weights = ggml_new_tensor_3d(pool->weights_ctx, type, n_in, n_out, capacity);
    if (pool->weights == nullptr || ggml_nbytes(pool->weights) != expert_size * static_cast<size_t>(capacity)) {
        GGML_LOG_WARN("[vk-moe-cache] tensor layout mismatch for type=%s shape=%lldx%lld\n",
                ggml_type_name(type), static_cast<long long>(n_in), static_cast<long long>(n_out));
        return nullptr;
    }
    ggml_set_name(pool->weights, "vk_moe_cache_weights");

    std::lock_guard<std::mutex> gpu_lock(dev.gpu_mutex);
    pool->weights_buffer = ggml_backend_alloc_ctx_tensors(pool->weights_ctx, dev.backend);
    if (pool->weights_buffer == nullptr) {
        GGML_LOG_WARN("[vk-moe-cache] Vulkan allocation failed for %zu MiB pool\n",
                (expert_size * static_cast<size_t>(capacity)) >> 20);
        return nullptr;
    }
    ggml_backend_buffer_set_usage(pool->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    cache_pool * result = pool.get();
    if (get_graph(dev, *result, 1) == nullptr) {
        GGML_LOG_WARN("[vk-moe-cache] Vulkan does not support cached MUL_MAT_ID for type=%s shape=%lldx%lld\n",
                ggml_type_name(type), static_cast<long long>(n_in), static_cast<long long>(n_out));
        return nullptr;
    }

    const size_t allocated = ggml_backend_buffer_get_size(pool->weights_buffer);
    dev.used += allocated;
    GGML_LOG_INFO("[vk-moe-cache] pool device=%d type=%s shape=%lldx%lld slots=%d size=%zu MiB\n",
            dev.id, ggml_type_name(type), static_cast<long long>(n_in), static_cast<long long>(n_out),
            capacity, allocated >> 20);
    dev.pools.push_back(std::move(pool));
    return result;
}

static int choose_victim(cache_pool & pool, const std::vector<uint8_t> & protected_slots) {
    int free_slot = -1;
    int victim = -1;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < pool.capacity; ++i) {
        const cache_slot & slot = pool.slots[i];
        if (protected_slots[i] || slot.queued) {
            continue;
        }
        if (!slot.valid && slot.key == 0) {
            free_slot = i;
            break;
        }
        if (slot.valid && slot.last_use < oldest) {
            oldest = slot.last_use;
            victim = i;
        }
    }
    return free_slot >= 0 ? free_slot : victim;
}

static void upload_worker() {
    cache_state & g = state();
    for (;;) {
        insert_job job;
        {
            std::unique_lock<std::mutex> lock(g.mutex);
            auto ready = [&g](const insert_job & candidate) {
                return candidate.device >= 0 && candidate.device < g.n_devices &&
                    g.devices[candidate.device] != nullptr &&
                    !g.devices[candidate.device]->dispatch_pending.load(std::memory_order_acquire);
            };
            g.condition.wait(lock, [&g, &ready] {
                return std::any_of(g.queue.begin(), g.queue.end(), ready);
            });
            const auto next = std::find_if(g.queue.begin(), g.queue.end(), ready);
            job = *next;
            g.queue.erase(next);

            if (job.device < 0 || job.device >= g.n_devices || g.devices[job.device] == nullptr ||
                    job.pool == nullptr || job.slot < 0 || job.slot >= job.pool->capacity) {
                continue;
            }
            const cache_slot & slot = job.pool->slots[job.slot];
            if (!slot.queued || slot.generation != job.generation || slot.key != job.key) {
                continue;
            }
            g.active_source = job.source;
            g.active_source_size = job.size;
        }

        device_state * dev = g.devices[job.device].get();
        {
            std::lock_guard<std::mutex> gpu_lock(dev->gpu_mutex);
            ggml_backend_tensor_set(job.pool->weights, job.source,
                    static_cast<size_t>(job.slot) * job.size, job.size);
        }

        {
            std::lock_guard<std::mutex> lock(g.mutex);
            cache_slot & slot = job.pool->slots[job.slot];
            if (slot.queued && slot.generation == job.generation && slot.key == job.key) {
                slot.valid = true;
                slot.queued = false;
                ++g.inserts;
                g.bytes_uploaded += job.size;
            }
            g.active_source = nullptr;
            g.active_source_size = 0;
        }
        g.condition.notify_all();
    }
}

static void start_worker_locked(cache_state & g) {
    if (!g.worker_started) {
        g.worker_started = true;
        std::thread(upload_worker).detach();
    }
}

static int vk_cache_begin(const char * tensor_name, const void * host_base, size_t expert_size,
        int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens) {
    cache_state & g = state();
    if (!g.enabled || n_tokens != 1 || host_base == nullptr || tensor_name == nullptr ||
            expert_size < g.min_expert_bytes || n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
            wtype < 0 || wtype >= GGML_TYPE_COUNT) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(g.mutex);
    int device = g.device_override;
    if (device < 0) {
        device = parse_block(tensor_name) % g.n_devices;
    }
    device_state * dev = get_device_locked(g, device);
    if (dev == nullptr) {
        return -1;
    }

    const enum ggml_type type = static_cast<enum ggml_type>(wtype);
    cache_pool * pool = find_pool(*dev, type, n_in, n_out, expert_size);
    if (pool == nullptr) {
        pool = create_pool_locked(g, *dev, type, n_in, n_out, expert_size);
    }
    if (pool == nullptr || pool->disabled) {
        return -1;
    }

    g.current.device = device;
    g.current.pool = pool;
    g.current.host_base = host_base;
    g.current.expert_size = expert_size;
    g.current.n_expert = n_expert;
    g.current.n_in = n_in;
    g.current.n_out = n_out;
    g.current.type = type;
    // Tensor names repeat across independently loaded models. Mix the current
    // host allocation into the key so two live contexts cannot alias slots.
    g.current.tensor_hash = fnv1a(tensor_name) ^ pointer_hash(host_base);
    g.current.valid = true;
    return device;
}

static int vk_cache_plan(int device, const int32_t * ids, int n_ids, int32_t * slot_idx) {
    cache_state & g = state();
    if (ids == nullptr || slot_idx == nullptr || n_ids <= 0 || n_ids > VK_MOE_MAX_HITS) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(g.mutex);
    std::fill(slot_idx, slot_idx + n_ids, -1);
    if (!g.current.valid || g.current.device != device || g.current.pool == nullptr) {
        return 0;
    }

    cache_pool & pool = *g.current.pool;
    std::vector<uint8_t> protected_slots(static_cast<size_t>(pool.capacity), 0);
    int n_hits = 0;
    int admitted = 0;

    for (int i = 0; i < n_ids; ++i) {
        const int32_t expert = ids[i];
        if (expert < 0 || expert >= g.current.n_expert) {
            continue;
        }
        ++g.lookups;
        const uint64_t key = expert_key(g.current.tensor_hash, expert);
        const auto found = pool.key_to_slot.find(key);
        if (found != pool.key_to_slot.end()) {
            cache_slot & slot = pool.slots[found->second];
            if (slot.valid && slot.key == key) {
                slot.last_use = ++pool.clock;
                slot_idx[i] = found->second;
                protected_slots[found->second] = 1;
                ++n_hits;
                ++g.hits;
                continue;
            }
            if (slot.queued && slot.key == key) {
                continue;
            }
        }

        uint16_t & frequency = pool.frequency[key];
        if (frequency != std::numeric_limits<uint16_t>::max()) {
            ++frequency;
        }
        if (frequency < g.admission_count || admitted >= g.inserts_per_plan ||
                static_cast<int>(g.queue.size()) >= g.max_queue) {
            if (static_cast<int>(g.queue.size()) >= g.max_queue) {
                ++g.queue_drops;
            }
            continue;
        }

        const int victim = choose_victim(pool, protected_slots);
        if (victim < 0) {
            continue;
        }
        cache_slot & slot = pool.slots[victim];
        if (slot.key != 0) {
            pool.key_to_slot.erase(slot.key);
            if (slot.valid) {
                ++g.evictions;
            }
        }

        slot.key = key;
        ++slot.generation;
        slot.last_use = ++pool.clock;
        slot.source = static_cast<const char *>(g.current.host_base) +
                static_cast<size_t>(expert) * g.current.expert_size;
        slot.expert = expert;
        slot.valid = false;
        slot.queued = true;
        pool.key_to_slot[key] = victim;

        insert_job job;
        job.device = device;
        job.pool = &pool;
        job.slot = victim;
        job.key = key;
        job.generation = slot.generation;
        job.source = slot.source;
        job.size = g.current.expert_size;
        g.queue.push_back(job);
        ++admitted;
    }

    if (pool.frequency.size() > (1u << 20)) {
        pool.frequency.clear();
    }
    device_state * dev = g.devices[device].get();
    if (dev != nullptr) {
        // Give the imminent hit dispatch priority over background H2D inserts.
        // Otherwise a synchronous Vulkan staging upload can win the mutex and
        // delay the very compute path the cache is intended to accelerate.
        dev->dispatch_pending.store(n_hits > 0, std::memory_order_release);
    }
    if (admitted > 0) {
        start_worker_locked(g);
        g.condition.notify_one();
    }

    // ggml-cpu does not invoke the optional stats callback. CUDA/HIP calls its
    // logger directly from collect(); do the equivalent here from plan() so
    // all cache activity (including all-miss warm-up periods) can emit stats.
    vk_cache_stats_locked(g, ggml_time_us());
    return n_hits;
}

[[noreturn]] static void dispatch_abort(device_state & dev, const char * reason) {
    if (dev.compute_locked) {
        dev.compute_locked = false;
        dev.gpu_mutex.unlock();
    }
    GGML_ABORT("Vulkan MoE expert cache: %s", reason);
}

static void vk_cache_dispatch(int device, int wtype, int64_t n_in, int64_t n_out, int n_hits,
        const int32_t * slot_idx_compact, const float * const * act_rows) {
    cache_state & g = state();
    if (device < 0 || device >= g.n_devices || n_hits <= 0 || n_hits > VK_MOE_MAX_HITS ||
            slot_idx_compact == nullptr || act_rows == nullptr) {
        GGML_ABORT("Vulkan MoE expert cache: invalid dispatch arguments");
    }

    device_state * dev = nullptr;
    cache_pool * pool = nullptr;
    {
        std::lock_guard<std::mutex> lock(g.mutex);
        if (!g.current.valid || g.current.device != device || g.current.pool == nullptr ||
                g.current.type != static_cast<enum ggml_type>(wtype) ||
                g.current.n_in != n_in || g.current.n_out != n_out) {
            GGML_ABORT("Vulkan MoE expert cache: dispatch does not match begin()");
        }
        dev = g.devices[device].get();
        pool = g.current.pool;
    }

    dev->gpu_mutex.lock();
    dev->compute_locked = true;
    graph_state * graph = get_graph(*dev, *pool, n_hits);
    if (graph == nullptr) {
        dispatch_abort(*dev, "failed to build cached MUL_MAT_ID graph");
    }

    for (int i = 0; i < n_hits; ++i) {
        if (slot_idx_compact[i] < 0 || slot_idx_compact[i] >= pool->capacity || act_rows[i] == nullptr) {
            dispatch_abort(*dev, "invalid cache slot or activation pointer");
        }
        graph->host_ids[i] = slot_idx_compact[i];
        std::memcpy(graph->host_activations.data() + static_cast<size_t>(i) * static_cast<size_t>(n_in),
                act_rows[i], static_cast<size_t>(n_in) * sizeof(float));
    }

    ggml_backend_tensor_set(graph->activations, graph->host_activations.data(), 0,
            graph->host_activations.size() * sizeof(float));
    ggml_backend_tensor_set(graph->ids, graph->host_ids.data(), 0,
            graph->host_ids.size() * sizeof(int32_t));

    const enum ggml_status status = ggml_backend_graph_compute_async(dev->backend, graph->graph);
    if (status != GGML_STATUS_SUCCESS) {
        dispatch_abort(*dev, ggml_status_to_string(status));
    }

    dev->active_graph = graph;
    dev->active_pool = pool;
    dev->active_hits = n_hits;
    dev->active_out = n_out;
}

static void vk_cache_collect(int device, int n_hits, float * const * dst_rows, int64_t n_out) {
    cache_state & g = state();
    if (device < 0 || device >= g.n_devices || g.devices[device] == nullptr) {
        GGML_ABORT("Vulkan MoE expert cache: invalid collect device");
    }
    device_state & dev = *g.devices[device];
    if (!dev.compute_locked || dev.active_graph == nullptr || dev.active_hits != n_hits ||
            dev.active_out != n_out || dst_rows == nullptr) {
        dispatch_abort(dev, "collect does not match dispatch");
    }

    graph_state & graph = *dev.active_graph;
    ggml_backend_synchronize(dev.backend);
    ggml_backend_tensor_get(graph.output, graph.host_output.data(), 0,
            graph.host_output.size() * sizeof(float));
    for (int i = 0; i < n_hits; ++i) {
        if (dst_rows[i] == nullptr) {
            dispatch_abort(dev, "null destination row");
        }
        std::memcpy(dst_rows[i], graph.host_output.data() + static_cast<size_t>(i) * static_cast<size_t>(n_out),
                static_cast<size_t>(n_out) * sizeof(float));
    }

    dev.active_graph = nullptr;
    dev.active_pool = nullptr;
    dev.active_hits = 0;
    dev.active_out = 0;
    dev.compute_locked = false;
    dev.dispatch_pending.store(false, std::memory_order_release);
    dev.gpu_mutex.unlock();
    g.condition.notify_all();
}

static void vk_cache_stats_locked(cache_state & g, int64_t now) {
    if (!g.enabled || g.stats_every <= 0) {
        return;
    }
    if (g.last_stats_us != 0 && now - g.last_stats_us < static_cast<int64_t>(g.stats_every) * 1000000ll) {
        return;
    }
    g.last_stats_us = now;
    const double hit_rate = g.lookups == 0 ? 0.0 : 100.0 * static_cast<double>(g.hits) / static_cast<double>(g.lookups);
    GGML_LOG_INFO("[vk-moe-cache] lookups=%llu hits=%llu (%.1f%%) inserts=%llu evictions=%llu "
                  "uploaded=%llu MiB queued=%zu drops=%llu\n",
            static_cast<unsigned long long>(g.lookups), static_cast<unsigned long long>(g.hits), hit_rate,
            static_cast<unsigned long long>(g.inserts), static_cast<unsigned long long>(g.evictions),
            static_cast<unsigned long long>(g.bytes_uploaded >> 20), g.queue.size(),
            static_cast<unsigned long long>(g.queue_drops));
}

static void vk_cache_stats() {
    cache_state & g = state();
    const int64_t now = ggml_time_us();
    std::lock_guard<std::mutex> lock(g.mutex);
    vk_cache_stats_locked(g, now);
}

static void vk_cache_invalidate(const void * base, size_t size) {
    if (base == nullptr || size == 0) {
        return;
    }
    cache_state & g = state();
    std::unique_lock<std::mutex> lock(g.mutex);

    g.queue.erase(std::remove_if(g.queue.begin(), g.queue.end(), [base, size](const insert_job & job) {
        return overlaps(job.source, job.size, base, size);
    }), g.queue.end());

    while (g.active_source != nullptr && overlaps(g.active_source, g.active_source_size, base, size)) {
        g.condition.wait(lock);
    }

    for (auto & dev_ptr : g.devices) {
        if (dev_ptr == nullptr) {
            continue;
        }
        for (auto & pool_ptr : dev_ptr->pools) {
            cache_pool & pool = *pool_ptr;
            for (cache_slot & slot : pool.slots) {
                if (slot.source != nullptr && overlaps(slot.source, pool.expert_size, base, size)) {
                    pool.key_to_slot.erase(slot.key);
                    slot.key = 0;
                    ++slot.generation;
                    slot.last_use = 0;
                    slot.source = nullptr;
                    slot.expert = -1;
                    slot.valid = false;
                    slot.queued = false;
                }
            }
            pool.frequency.clear();
        }
    }
    if (g.current.host_base != nullptr && overlaps(g.current.host_base,
            saturated_mul(g.current.expert_size, static_cast<size_t>(g.current.n_expert)), base, size)) {
        g.current = node_state{};
    }
}

static void vk_cache_node_time(int, int64_t) {
    // The CUDA implementation has a sampled bail-out judge. The portable
    // Vulkan core keeps policy deterministic; users can disable it explicitly.
}

} // namespace

void ggml_vk_moe_cache_register(void) {
    cache_state & g = state();
    static std::once_flag once;
    std::call_once(once, [&g] {
        if (!env_enabled("GGML_VK_MOE_CACHE", false)) {
            return;
        }
        if (ggml_moe_cache.begin != nullptr) {
            GGML_LOG_WARN("[vk-moe-cache] another MoE cache backend is already registered; Vulkan cache not installed\n");
            return;
        }

        g.n_devices = std::min(ggml_backend_vk_get_device_count(), GGML_VK_MAX_DEVICES);
        if (g.n_devices <= 0) {
            return;
        }
        g.budget_mb = env_size("GGML_VK_MOE_CACHE_BUDGET_MB", VK_MOE_DEFAULT_BUDGET_MB);
        g.reserve_mb = env_size("GGML_VK_MOE_CACHE_RESERVE_MB", VK_MOE_DEFAULT_RESERVE_MB);
        g.min_expert_bytes = scaled_bytes(
                env_size("GGML_VK_MOE_CACHE_MIN_EXPERT_KB", VK_MOE_DEFAULT_MIN_KB), 1024u);
        g.pool_divisor = env_int("GGML_VK_MOE_CACHE_POOL_DIV", VK_MOE_DEFAULT_POOL_DIV, 1, 16);
        g.max_slots = env_int("GGML_VK_MOE_CACHE_MAX_SLOTS", VK_MOE_DEFAULT_MAX_SLOTS, 2, 65536);
        g.inserts_per_plan = env_int("GGML_VK_MOE_CACHE_INSERTS", VK_MOE_DEFAULT_INSERTS, 0, VK_MOE_MAX_HITS);
        g.admission_count = env_int("GGML_VK_MOE_CACHE_ADMIT", VK_MOE_DEFAULT_ADMIT, 1, 255);
        g.max_queue = env_int("GGML_VK_MOE_CACHE_QUEUE", VK_MOE_DEFAULT_QUEUE, 1, 65536);
        g.stats_every = env_int("GGML_VK_MOE_CACHE_STATS", VK_MOE_DEFAULT_STATS, 0, 3600);
        g.device_override = env_int("GGML_VK_MOE_CACHE_DEVICE", -1, -1, g.n_devices - 1);
        g.enabled = true;

        ggml_moe_cache.begin = vk_cache_begin;
        ggml_moe_cache.plan = vk_cache_plan;
        ggml_moe_cache.dispatch = vk_cache_dispatch;
        ggml_moe_cache.collect = vk_cache_collect;
        ggml_moe_cache.stats = vk_cache_stats;
        ggml_moe_cache.redirect_offer = nullptr;
        ggml_moe_cache.redirect_finalize = nullptr;
        ggml_moe_cache.glu_hits = nullptr;
        ggml_moe_cache.invalidate = vk_cache_invalidate;
        ggml_moe_cache.node_time = vk_cache_node_time;

        GGML_LOG_INFO("[vk-moe-cache] enabled: devices=%d budget=%zu MiB/device inserts=%d admit=%d stats=%ds\n",
                g.n_devices, g.budget_mb, g.inserts_per_plan, g.admission_count, g.stats_every);
    });
}
