// CPU-only regression: cleanup must not access a destroyed Core context.
#include <infinicore/ops/common/cache.hpp>
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace {
infiniDevice_t selected_type = INFINI_DEVICE_CPU;
int selected_index = 0;
int selections = 0;
int releases = 0;
bool fail_query = false;
}

extern "C" infiniStatus_t infinirtGetDevice(infiniDevice_t *type, int *index) {
    if (fail_query) return INFINI_STATUS_INTERNAL_ERROR;
    *type = selected_type;
    *index = selected_index;
    return INFINI_STATUS_SUCCESS;
}
extern "C" infiniStatus_t infinirtSetDevice(infiniDevice_t type, int index) {
    selected_type = type;
    selected_index = index;
    ++selections;
    return INFINI_STATUS_SUCCESS;
}

namespace infinicore::context {
Device getDevice() { throw std::runtime_error("Core context already destroyed"); }
void setDevice(Device) { throw std::runtime_error("Core context already destroyed"); }
}

struct Descriptor {
    infiniDevice_t type;
    int index;
    Descriptor(infiniDevice_t t, int i) : type(t), index(i) {}
    ~Descriptor() {
        assert(selected_type == type && selected_index == index);
        ++releases;
    }
};
using Cache = infinicore::op::common::OpCache<int, std::shared_ptr<Descriptor>>;

int main() {
    {
        Cache cache;
        cache.getCache(infinicore::Device::Type::CPU, 0).put(0, std::make_shared<Descriptor>(INFINI_DEVICE_CPU, 0));
        cache.getCache(infinicore::Device::Type::HYGON, 0).put(1, std::make_shared<Descriptor>(INFINI_DEVICE_HYGON, 0));
        cache.getCache(infinicore::Device::Type::HYGON, 2).put(2, std::make_shared<Descriptor>(INFINI_DEVICE_HYGON, 2));
        cache.clear();
        assert(releases == 3 && selections == 4); // Empty device 1 was skipped.
        assert(selected_type == INFINI_DEVICE_CPU && selected_index == 0);
        cache.clear();
        assert(releases == 3 && selections == 4);
    }
    {
        Cache cache;
        cache.getCache(infinicore::Device::Type::HYGON, 1).put(0, std::make_shared<Descriptor>(INFINI_DEVICE_HYGON, 1));
        // No explicit clear: normal destruction must release on the owner device.
    }
    assert(releases == 4 && selections == 6);
    assert(selected_type == INFINI_DEVICE_CPU && selected_index == 0);
    {
        Cache cache;
        cache.getCache(infinicore::Device::Type::CPU, 0).put(0, std::make_shared<Descriptor>(INFINI_DEVICE_CPU, 0));
        fail_query = true;
        bool rejected = false;
        try { cache.clear(); } catch (const std::runtime_error &) { rejected = true; }
        assert(rejected && releases == 4); // Failed selection cannot report cleanup.
        fail_query = false;
        cache.clear();
    }
    assert(releases == 5 && selections == 6);
    std::cout << "OP_CACHE_LIFETIME_PASS releases=5 device_restore=PASS empty_idempotent=PASS runtime_error_propagated=PASS\n";
}
