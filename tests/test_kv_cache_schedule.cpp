// Host-only qualification of per-layer mixed-precision KV storage: schedule parsing, the resolved
// per-layer plane schema, bytes per token, pool capacity totals and per-layer page copy sizes.
// Hand-computed expectations come from the closed D256 profiles in
// docs/maintainer/paged-kv-cache.md, not from the implementation.

#include "core/host_kv_arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using ninfer::KvCacheSchedule;
using ninfer::KvCacheStorage;

// Qwen3.8-27B text stack: 16 of 64 layers are full-attention, head_dim 256, 4 KV heads.
constexpr std::uint32_t kAttentionLayers = 16;
constexpr std::int32_t kHeadDim          = 256;
constexpr std::int32_t kKvHeads          = 4;

// Physical bytes per token and KV head, D=256 (code plane + scale plane, K then V).
constexpr std::size_t kInt8TokenHead    = (256 + 4 * 2) + (256 + 4 * 2);  // 528
constexpr std::size_t kRk4V4E8TokenHead = (128 + 4 * 2) + (128 + 4 * 2);  // 272
constexpr std::size_t kRk8V4TokenHead   = (256 + 4 * 2) + (128 + 4 * 2);  // 400
constexpr std::size_t kRk2V4E8TokenHead = (64 + 4 * 2) + (128 + 4 * 2);   // 208

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expect_size(std::size_t actual, std::size_t expected, const std::string& label) {
    expect(actual == expected, label + ": expected " + std::to_string(expected) + ", got " +
                                   std::to_string(actual));
}

template <typename Fn> void expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    std::cerr << "FAIL: " << label << " was accepted\n";
    ++failures;
}

void check_parsing() {
    const KvCacheSchedule uniform = ninfer::parse_kv_cache_schedule("int8");
    expect(uniform.uniform(), "`int8` is a uniform schedule");
    expect(uniform.storage_for(0) == KvCacheStorage::Int8Group64 &&
               uniform.storage_for(15) == KvCacheStorage::Int8Group64,
           "`int8` resolves to int8 on every layer");
    expect(uniform == KvCacheSchedule(KvCacheStorage::Int8Group64),
           "`int8` equals the implicit uniform schedule");
    expect(ninfer::kv_cache_schedule_spec(uniform) == "int8", "uniform spec round-trips");

    const KvCacheSchedule tiered = ninfer::parse_kv_cache_schedule("int8:4,rk4v4-e8");
    expect(!tiered.uniform(), "`int8:4,rk4v4-e8` is a two-tier schedule");
    expect(tiered.head_layers == 4, "two-tier boundary is 4 attention layers");
    expect(tiered.storage_for(0) == KvCacheStorage::Int8Group64 &&
               tiered.storage_for(3) == KvCacheStorage::Int8Group64 &&
               tiered.storage_for(4) == KvCacheStorage::RK4V4E8 &&
               tiered.storage_for(15) == KvCacheStorage::RK4V4E8,
           "two-tier boundary is exclusive at N");
    expect(tiered.trailing_storage() == KvCacheStorage::RK4V4E8,
           "trailing pools follow the tail kind");
    expect(ninfer::kv_cache_schedule_spec(tiered) == "int8:4,rk4v4-e8",
           "two-tier spec round-trips");

    const KvCacheSchedule degenerate = ninfer::parse_kv_cache_schedule("fp8:4,fp8");
    expect(degenerate.uniform() && ninfer::kv_cache_schedule_spec(degenerate) == "fp8",
           "a two-tier spec naming one kind twice normalizes to uniform");

    expect_throws([] { (void)ninfer::parse_kv_cache_schedule("int8:0,rk4v4-e8"); },
                  "zero-layer boundary");
    expect_throws([] { (void)ninfer::parse_kv_cache_schedule("int8:x,rk4v4-e8"); },
                  "non-numeric boundary");
    expect_throws([] { (void)ninfer::parse_kv_cache_schedule("int8:,rk4v4-e8"); },
                  "empty boundary");
    expect_throws([] { (void)ninfer::parse_kv_cache_schedule("int8:4"); },
                  "schedule without a tail kind");
    expect_throws([] { (void)ninfer::parse_kv_cache_schedule("int8:4,bogus"); }, "bad tail kind");
    expect_throws([] { (void)ninfer::parse_kv_cache_schedule("bogus:4,int8"); }, "bad head kind");
    expect_throws([] { (void)ninfer::parse_kv_cache_schedule("bogus"); }, "bad uniform kind");
    expect_throws([] { (void)ninfer::parse_kv_cache_schedule(""); }, "empty kv-dtype");

    expect(ninfer::parse_kv_cache_schedule("int8:4,rk4v4-e8").identity_tag() !=
               ninfer::parse_kv_cache_schedule("int8:8,rk4v4-e8").identity_tag(),
           "schedules differing only in boundary get distinct prefix identities");
    expect(ninfer::parse_kv_cache_schedule("int8").identity_tag() !=
               ninfer::parse_kv_cache_schedule("int8:4,rk4v4-e8").identity_tag(),
           "a uniform and a two-tier schedule get distinct prefix identities");
}

// The sm_89-only rotated family is rejected on 120a by the same predicate the startup gate uses,
// in either tier of a schedule.
void check_sm89_gate() {
    expect(KvCacheSchedule(KvCacheStorage::RK4V4E8).requires_sm89(), "uniform rk4v4-e8 is sm_89");
    expect(KvCacheSchedule(KvCacheStorage::RotatedInt8KeyInt4ValueGroup64).requires_sm89(),
           "uniform rk8v4 is sm_89");
    expect(ninfer::parse_kv_cache_schedule("int8:4,rk4v4-e8").requires_sm89(),
           "a schedule with an sm_89 tail is sm_89");
    expect(ninfer::parse_kv_cache_schedule("rk8v4:8,int8").requires_sm89(),
           "a schedule with an sm_89 head is sm_89");
    expect(!ninfer::parse_kv_cache_schedule("int8:4,fp8").requires_sm89(),
           "a schedule of portable kinds is not sm_89-only");
    expect(!KvCacheSchedule(KvCacheStorage::Fp8KeyNvfp4Value).requires_sm89(),
           "k8v4 is not sm_89-only");
}

struct Expectation {
    const char* spec;
    std::size_t head_token_head;
    std::size_t tail_token_head;
    std::uint32_t head_layers;
};

void check_layout_and_capacity(const Expectation& expected) {
    const KvCacheSchedule schedule = ninfer::parse_kv_cache_schedule(expected.spec);
    const std::vector<ninfer::PagedKVStorageLayout> layers =
        ninfer::paged_kv_schedule_layouts(schedule, kAttentionLayers, kHeadDim);
    const std::string label(expected.spec);

    expect_size(layers.size(), kAttentionLayers, label + " layer count");
    for (std::uint32_t layer = 0; layer < kAttentionLayers; ++layer) {
        const bool head = layer < expected.head_layers;
        expect(layers[layer].storage == schedule.storage_for(layer),
               label + " layer " + std::to_string(layer) + " kind");
        expect_size(layers[layer].physical_bytes_per_token_head(),
                    head ? expected.head_token_head : expected.tail_token_head,
                    label + " layer " + std::to_string(layer) + " bytes per token/head");
        expect_size(ninfer::paged_kv_layer_bytes_per_token(layers[layer], kKvHeads),
                    (head ? expected.head_token_head : expected.tail_token_head) * kKvHeads,
                    label + " layer " + std::to_string(layer) + " bytes per token");
    }

    const std::size_t bytes_per_token =
        (expected.head_layers * expected.head_token_head +
         (kAttentionLayers - expected.head_layers) * expected.tail_token_head) *
        kKvHeads;
    expect_size(ninfer::paged_kv_bytes_per_token(layers, kKvHeads), bytes_per_token,
                label + " pool bytes per token");

    // Physical pool: one page-group ID selects a differently sized slice in each layer, so the
    // pool pays exactly the summed per-layer bytes.
    const ninfer::KVPageGeometry geometry = ninfer::paged_kv_page_geometry(layers, kKvHeads);
    std::size_t planes                    = 0;
    for (const ninfer::PagedKVStorageLayout& layer : layers) { planes += layer.planes_per_layer(); }
    expect_size(geometry.planes.size(), planes, label + " plane inventory");

    constexpr std::uint32_t kPages = 512;
    ninfer::LayoutBuilder builder;
    const ninfer::DeviceKVPagePoolLayout pool = ninfer::plan_device_kv_page_pool(
        builder, {.page_group_count = kPages, .geometry = geometry});
    expect_size(pool.payload_bytes(),
                bytes_per_token * static_cast<std::size_t>(ninfer::kPagedKVPageSize) * kPages,
                label + " pool payload bytes");

    // The capacity curve stride is one page group of every layer's planes.
    ninfer::LayoutBuilder next_builder;
    const ninfer::DeviceKVPagePoolLayout next = ninfer::plan_device_kv_page_pool(
        next_builder, {.page_group_count = kPages + 1, .geometry = geometry});
    expect_size(next.payload_bytes() - pool.payload_bytes(),
                bytes_per_token * static_cast<std::size_t>(ninfer::kPagedKVPageSize),
                label + " capacity stride per page group");

    // Page copies are per plane, so each layer's page payload is its own size on both replicas.
    const ninfer::HostKVPageLayout host = ninfer::plan_host_kv_page_layout(geometry);
    expect_size(host.planes.size(), planes, label + " host plane inventory");
    for (std::uint32_t layer = 0; layer < kAttentionLayers; ++layer) {
        const std::size_t base = ninfer::paged_kv_plane_base(layers, layer);
        std::size_t layer_page = 0;
        for (std::size_t index = 0; index < layers[layer].planes_per_layer(); ++index) {
            const ninfer::KVPlaneGeometry& plane = geometry.planes[base + index];
            const std::size_t plane_page_bytes =
                static_cast<std::size_t>(plane.leading_extent) * ninfer::kPagedKVPageSize *
                static_cast<std::size_t>(plane.head_extent) * ninfer::dtype_size(plane.dtype);
            expect_size(host.planes[base + index].page_payload_bytes, plane_page_bytes,
                        label + " layer " + std::to_string(layer) + " plane " +
                            std::to_string(index) + " page payload");
            // The device slab stride over page IDs is the same per-page payload the D2H copy
            // moves, so a small layer never reads a large layer's page size.
            const ninfer::TensorRegion& region = pool.planes[base + index].storage;
            expect_size(static_cast<std::size_t>(region.shape[0]) * region.shape[1] *
                            region.shape[2] * ninfer::dtype_size(region.dtype),
                        plane_page_bytes,
                        label + " layer " + std::to_string(layer) + " plane " +
                            std::to_string(index) + " device page stride");
            layer_page += plane_page_bytes;
        }
        expect_size(layer_page,
                    (layer < expected.head_layers ? expected.head_token_head
                                                  : expected.tail_token_head) *
                        kKvHeads * ninfer::kPagedKVPageSize,
                    label + " layer " + std::to_string(layer) + " page bytes");
    }
    expect_size(host.page_stride, bytes_per_token * ninfer::kPagedKVPageSize,
                label + " host page stride");
}

void check_uniform_is_unchanged() {
    // A uniform schedule must plan exactly what a single storage kind planned before.
    for (const KvCacheStorage storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value,
          KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, KvCacheStorage::RK4V4E8,
          KvCacheStorage::RK2V4E8}) {
        const std::vector<ninfer::PagedKVStorageLayout> layers =
            ninfer::paged_kv_schedule_layouts(storage, kAttentionLayers, kHeadDim);
        const ninfer::PagedKVStorageLayout single =
            ninfer::paged_kv_storage_layout(storage, kHeadDim);
        for (const ninfer::PagedKVStorageLayout& layer : layers) {
            expect(layer == single, std::string("uniform ") +
                                        ninfer::kv_cache_storage_name(storage) +
                                        " layer matches the single-kind layout");
        }
        expect_size(ninfer::paged_kv_bytes_per_token(layers, kKvHeads),
                    single.physical_bytes_per_token_head() * kKvHeads * kAttentionLayers,
                    std::string("uniform ") + ninfer::kv_cache_storage_name(storage) +
                        " bytes per token");
    }
}

void check_mixed_plane_counts() {
    // bf16 layers have no scale planes, so plane ordinals are not a fixed stride.
    const std::vector<ninfer::PagedKVStorageLayout> layers = ninfer::paged_kv_schedule_layouts(
        ninfer::parse_kv_cache_schedule("bf16:2,int8"), 5, kHeadDim);
    expect_size(layers[0].planes_per_layer(), 2, "bf16 layer plane count");
    expect_size(layers[2].planes_per_layer(), 4, "int8 layer plane count");
    expect_size(ninfer::paged_kv_plane_base(layers, 0), 0, "layer 0 plane base");
    expect_size(ninfer::paged_kv_plane_base(layers, 2), 4, "layer 2 plane base");
    expect_size(ninfer::paged_kv_plane_base(layers, 4), 12, "layer 4 plane base");
    expect_throws([&] { (void)ninfer::paged_kv_plane_base(layers, 5); },
                  "plane base past the last layer");
}

void check_boundary_validation() {
    expect_throws(
        [] {
            (void)ninfer::paged_kv_schedule_layouts(
                ninfer::parse_kv_cache_schedule("int8:16,rk4v4-e8"), kAttentionLayers, kHeadDim);
        },
        "boundary at the attention-layer count");
    expect_throws(
        [] {
            (void)ninfer::paged_kv_schedule_layouts(
                ninfer::parse_kv_cache_schedule("int8:20,rk4v4-e8"), kAttentionLayers, kHeadDim);
        },
        "boundary past the attention-layer count");
    expect_throws(
        [] {
            (void)ninfer::paged_kv_schedule_layouts(
                ninfer::parse_kv_cache_schedule("int8:4,rk4v4-e8"), kAttentionLayers, 128);
        },
        "schedule at an unsupported head dimension");
}

} // namespace

int main() {
    check_parsing();
    check_sm89_gate();
    check_uniform_is_unchanged();
    check_mixed_plane_counts();
    check_boundary_validation();

    // 4 int8 layers at 2112 B/token plus 12 rk4v4-e8 layers at 1088 B/token = 21504 B/token,
    // against 33792 for uniform int8 and 17408 for uniform rk4v4-e8.
    check_layout_and_capacity({"int8:4,rk4v4-e8", kInt8TokenHead, kRk4V4E8TokenHead, 4});
    // 8 rk8v4 layers at 1600 B/token plus 8 rk2v4-e8 layers at 832 B/token = 19456 B/token,
    // against 25600 for uniform rk8v4 and 13312 for uniform rk2v4-e8.
    check_layout_and_capacity({"rk8v4:8,rk2v4-e8", kRk8V4TokenHead, kRk2V4E8TokenHead, 8});
    check_layout_and_capacity({"int8", kInt8TokenHead, kInt8TokenHead, 0});

    // The two-tier totals sit between the uniform totals they interpolate.
    const auto pool_bytes = [](const char* spec) {
        return ninfer::paged_kv_bytes_per_token(
            ninfer::paged_kv_schedule_layouts(ninfer::parse_kv_cache_schedule(spec),
                                              kAttentionLayers, kHeadDim),
            kKvHeads);
    };
    expect_size(pool_bytes("int8"), 33792, "uniform int8 bytes per token");
    expect_size(pool_bytes("rk4v4-e8"), 17408, "uniform rk4v4-e8 bytes per token");
    expect_size(pool_bytes("int8:4,rk4v4-e8"), 21504, "int8:4,rk4v4-e8 bytes per token");
    expect_size(pool_bytes("rk8v4"), 25600, "uniform rk8v4 bytes per token");
    expect_size(pool_bytes("rk2v4-e8"), 13312, "uniform rk2v4-e8 bytes per token");
    expect_size(pool_bytes("rk8v4:8,rk2v4-e8"), 19456, "rk8v4:8,rk2v4-e8 bytes per token");

    if (failures != 0) {
        std::cerr << failures << " KV schedule check(s) failed\n";
        return 1;
    }
    std::cout << "KV cache schedule checks passed\n";
    return 0;
}
