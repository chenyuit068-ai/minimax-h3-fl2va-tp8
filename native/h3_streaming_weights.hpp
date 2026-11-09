#pragma once
// Host storage lifetime only. Numerical work remains on existing Core operators.
#include <infinicore/context/context.hpp>
#include <infinicore/nn/parameter.hpp>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace infinilm::h3_streaming {
class LinearLease {
    std::vector<std::shared_ptr<infinicore::Memory>> memories_;
    std::vector<std::unique_lock<std::mutex>> locks_;
    bool finished_ = false;
public:
    explicit LinearLease(const std::unordered_map<std::string,infinicore::nn::Parameter> &params) {
        for (const auto &[name,param] : params) {
            auto memory=param->storage();
            if (memory->is_streaming()) memories_.push_back(memory);
        }
        std::sort(memories_.begin(),memories_.end(),[](const auto &a,const auto &b){return std::less<const void *>{}(a.get(),b.get());});
        memories_.erase(std::unique(memories_.begin(),memories_.end()),memories_.end());
        for (const auto &memory : memories_) locks_.emplace_back(memory->residency_mutex());
        if (!memories_.empty() && infinicore::context::isGraphRecording())
            throw std::runtime_error("Streaming weights do not support graph capture");
        for (const auto &memory : memories_) memory->materialize();
    }
    void finish() {
        if (finished_) return;
        if (!memories_.empty()) infinicore::context::syncStream();
        for (const auto &memory : memories_) memory->evict();
        finished_=true;
    }
    // On a failed kernel, retain storage rather than free memory still in use.
    ~LinearLease() = default;
};
} // namespace infinilm::h3_streaming
