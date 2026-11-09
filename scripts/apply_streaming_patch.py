"""Experimental opt-in host storage patch; apply only to separate source copies."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def patch(core,lm):
    package=Path(__file__).resolve().parents[1]
    target=package/'streaming';target.mkdir(exist_ok=True)
    manifest=target/'patch-manifest.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else []
    records=[];diffs=[]
    def change(repo,name,edit):
        root=core if repo=='InfiniCore' else lm
        path=root/name;backup=path.with_suffix(path.suffix+'.h3-streaming-reference')
        if not backup.exists():shutil.copy2(path,backup)
        original=backup.read_text();updated=edit(original);current=path.read_text()
        known={r['modified_sha256'] for r in previous if r['repo']==repo and r['file']==name and r['reference_sha256']==digest(original)}
        if current not in (original,updated) and digest(current) not in known:
            raise ValueError(f'Unrelated source edits: {path}')
        path.write_text(updated)
        records.append(dict(repo=repo,file=name,reference_sha256=digest(original),modified_sha256=digest(updated)))
        diffs.extend(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile=f'a/{repo}/{name}',tofile=f'b/{repo}/{name}'))
    def replace_once(text,old,new):
        if text.count(old)!=1:raise ValueError(f'Unexpected reference source: {old[:80]}')
        return text.replace(old,new,1)

    memory_methods='''
    void mark_placeholder();
    bool is_placeholder() const;
    void enable_streaming();
    bool is_streaming() const;
    void add_stream_source(size_t target_offset, std::shared_ptr<Memory> source, size_t source_offset, size_t count);
    void materialize();
    void evict();
    std::mutex &residency_mutex();
'''
    memory_fields='''
    struct StreamSpan { size_t target_offset, source_offset, count; std::shared_ptr<Memory> source; };
    bool placeholder_ = false, streaming_ = false;
    std::shared_ptr<Memory> resident_;
    std::vector<StreamSpan> stream_sources_;
    std::mutex residency_mutex_;
'''
    def memory_header(text):
        text=replace_once(text,'#include <functional>','#include <functional>\n#include <memory>\n#include <mutex>\n#include <vector>')
        text=replace_once(text,'    bool is_pinned() const;','    bool is_pinned() const;\n'+memory_methods)
        return replace_once(text,'    bool is_pinned_;','    bool is_pinned_;\n'+memory_fields)
    change('InfiniCore','include/infinicore/memory.hpp',memory_header)
    implementation='''
void Memory::mark_placeholder() {
    if (data_ || deleter_) throw std::runtime_error("Placeholder storage must not own a raw allocation");
    placeholder_=true;
}
bool Memory::is_placeholder() const { return placeholder_; }
void Memory::enable_streaming() {
    if (!placeholder_ || resident_) throw std::runtime_error("Enable streaming before weight loading");
    streaming_=true;
}
bool Memory::is_streaming() const { return streaming_; }
std::mutex &Memory::residency_mutex() { return residency_mutex_; }
void Memory::add_stream_source(size_t target, std::shared_ptr<Memory> source, size_t offset, size_t count) {
    std::unique_lock lock(residency_mutex_,std::try_to_lock);
    if (!lock.owns_lock() || resident_) throw std::runtime_error("Cannot reload active streaming weights");
    if (!streaming_ || source.get()==this || source->device().getType()!=Device::Type::CPU ||
        target>size_ || count>size_-target || offset>source->size() || count>source->size()-offset)
        throw std::runtime_error("Invalid streaming weight span");
    for (auto &span : stream_sources_) {
        if (target==span.target_offset && count==span.count) { span={target,offset,count,std::move(source)}; return; }
        if (target<span.target_offset+span.count && span.target_offset<target+count)
            throw std::runtime_error("Overlapping streaming weight spans");
    }
    stream_sources_.push_back({target,offset,count,std::move(source)});
}
void Memory::materialize() {
    if (!placeholder_ || resident_) return;
    if (streaming_) {
        std::sort(stream_sources_.begin(),stream_sources_.end(),[](const auto &a,const auto &b){return a.target_offset<b.target_offset;});
        size_t covered=0;
        for (const auto &span : stream_sources_) {
            if (span.target_offset!=covered) throw std::runtime_error("Missing streaming weight bytes");
            covered+=span.count;
        }
        if (covered!=size_) throw std::runtime_error("Incomplete streaming weight coverage");
    }
    context::setDevice(device_);
    auto allocation=context::allocateMemory(size_);
    try {
        for (const auto &span : stream_sources_)
            context::memcpyH2D(allocation->data()+span.target_offset,span.source->data()+span.source_offset,span.count);
        if (streaming_) context::syncStream();
    } catch (...) {
        try { context::syncStream(); } catch (...) {}
        throw;
    }
    resident_=std::move(allocation);
}
void Memory::evict() {
    if (!streaming_) throw std::runtime_error("Only streaming storage can be evicted");
    resident_.reset();
}
'''
    def memory_source(text):
        text=replace_once(text,'#include "infinicore/memory.hpp"','#include "infinicore/memory.hpp"\n#include "infinicore/context/context.hpp"\n#include <algorithm>\n#include <stdexcept>')
        text=replace_once(text,'    return data_;','    if (resident_) return resident_->data();\n    if (placeholder_) throw std::runtime_error("Deferred weight accessed before materialization");\n    return data_;')
        return replace_once(text,'} // namespace infinicore',implementation+'\n} // namespace infinicore')
    change('InfiniCore','src/infinicore/memory.cc',memory_source)
    change('InfiniCore','include/infinicore/tensor.hpp',lambda s:replace_once(s,'    Device device() const;','    Device device() const;\n\n    const std::shared_ptr<Memory> &storage() const { return data_.memory; }'))
    def parameter_source(text):
        text=replace_once(text,'#include <cstring>','#include <cstring>\n#include <cstdlib>')
        helper='''namespace {
Tensor parameter_storage(const Shape &shape,const DataType &dtype,const Device &device) {
    const char *enabled=std::getenv("INFINICORE_DEFER_PARAMETER_STORAGE");
    if (enabled && std::strcmp(enabled,"1")==0 && device.getType()!=Device::Type::CPU) {
        auto result=Tensor::from_blob(nullptr,shape,dtype,device);
        result->storage()->mark_placeholder();
        return result;
    }
    return Tensor::empty(shape,dtype,device,false);
}
} // namespace
'''
        text=replace_once(text,'namespace infinicore::nn {','namespace infinicore::nn {\n'+helper)
        text=replace_once(text,'Tensor::empty(get_partition_shape_(shape, tp_dim, tp_size, num_shards), dtype, device, false)','parameter_storage(get_partition_shape_(shape, tp_dim, tp_size, num_shards), dtype, device)')
        # Replicated linears use split_dim=-1: never index it when tp_size=1.
        needle='expected_shape[tp_dim_] *= tp_size_;'
        assert text.count(needle)==2
        return text.replace(needle,'''if (tp_size_>1) {
        if (tp_dim_>=expected_shape.size()) throw std::runtime_error("Invalid parallel partition dimension");
        expected_shape[tp_dim_] *= tp_size_;
    }''')
    change('InfiniCore','src/infinicore/nn/parameter.cc',parameter_source)
    def copy_source(text):
        needle='    if (this->device() == src->device()) {'
        replacement='''    if (data_.memory->is_streaming()) {
        if (src->device().getType()!=Device::Type::CPU || !is_contiguous() || src->dtype()!=dtype())
            throw std::runtime_error("Streaming weight loads require contiguous CPU source and matching dtype");
        if (!src->is_contiguous()) src=src->contiguous();
        data_.memory->add_stream_source(data_.offset,src->data_.memory,src->data_.offset,nbytes());
        return;
    }
    if (data_.memory->is_placeholder()) data_.memory->materialize();
'''+needle
        return replace_once(text,needle,replacement)
    change('InfiniCore','src/infinicore/tensor/copy.cc',copy_source)
    def linear_source(text):
        text=replace_once(text,'#include "base_linear.hpp"','#include "base_linear.hpp"\n#include "h3_streaming_weights.hpp"')
        text=replace_once(text,'        this->register_parameter(desc.name, param);','''        if (param->storage()->is_placeholder()) {
            if (desc.tp_size!=1 || !std::dynamic_pointer_cast<infinilm::quantization::NoneQuantization>(quantization_))
                throw std::runtime_error("Streaming supports single-device unquantized linear weights only");
            param->storage()->enable_streaming();
        }
        this->register_parameter(desc.name, param);''')
        text=replace_once(text,'infinicore::Tensor BaseLinear::compute_linear(infinicore::Tensor &input) const {','infinicore::Tensor BaseLinear::compute_linear(infinicore::Tensor &input) const {\n    infinilm::h3_streaming::LinearLease lease(parameters_);')
        text=replace_once(text,'    return quantization_->forward(params, input, has_bias_, alpha_);','    auto result=quantization_->forward(params, input, has_bias_, alpha_);\n    lease.finish();\n    return result;')
        text=replace_once(text,'    infinicclComm_t communicator) const {','    infinicclComm_t communicator) const {\n    infinilm::h3_streaming::LinearLease lease(parameters_);')
        text=replace_once(text,'    return quantization_->forward_allreduce(\n        params, input, has_bias_, communicator, alpha_);','    auto result=quantization_->forward_allreduce(\n        params, input, has_bias_, communicator, alpha_);\n    lease.finish();\n    return result;')
        # Streaming keeps the original untransposed layout; skip eager packing.
        text=replace_once(text,'void BaseLinear::process_weights_after_loading() {','''void BaseLinear::process_weights_after_loading() {
    for (const auto &[name,param] : parameters_) {
        if (param->storage()->is_streaming()) return;
    }''')
        return text
    change('InfiniLM','csrc/layers/linear/base_linear.cpp',linear_source)
    def streaming_bindings(text):
        return replace_once(text,'    infinilm::h3_nine::bind(m);','''    infinilm::h3_nine::bind(m);
    m.def("_h3_streaming_storage_available", [] { return true; });
    m.def("_h3_trim_memory", [] {
        pybind11::gil_scoped_release release;
        infinicore::context::syncDevice();
        infinicore::context::trimMemory();
    });''')
    change('InfiniLM','csrc/pybind11/bindings.cc',streaming_bindings)
    shutil.copy2(package/'native/h3_streaming_weights.hpp',lm/'csrc/layers/linear/h3_streaming_weights.hpp')
    manifest.write_text(json.dumps(records,indent=2))
    (target/'InfiniCore-InfiniLM-streaming.patch').write_text(''.join(diffs))
    print('EXPERIMENTAL_STREAMING_PATCH_APPLIED; FULL_MODEL_NOT_VALIDATED')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--core',type=Path,required=True);parser.add_argument('--lm',type=Path,required=True)
    args=parser.parse_args();patch(args.core.resolve(),args.lm.resolve())
