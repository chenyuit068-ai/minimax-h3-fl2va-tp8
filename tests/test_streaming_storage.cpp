// Synthetic native storage/linear checks. Does not load the competition model.
#include "h3_streaming_weights.hpp"
#include <infinicore/ops/linear.hpp>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

using namespace infinicore;
using infinicore::nn::Parameter;
using infinilm::h3_streaming::LinearLease;

void require(bool value,const char *message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F operation,const char *message) {
    try { operation(); } catch (const std::runtime_error &) { return; }
    throw std::runtime_error(message);
}
Tensor host(const Shape &shape,const std::vector<float> &values) {
    auto result=Tensor::empty(shape,DataType::F32,Device(Device::Type::CPU,0));
    require(result->nbytes()==values.size()*sizeof(float),"Invalid test data");
    std::memcpy(result->data(),values.data(),result->nbytes());
    return result;
}
std::vector<float> read(Tensor value) {
    auto cpu=value->to(Device(Device::Type::CPU,0));
    context::syncDevice();
    std::vector<float> result(cpu->numel());
    std::memcpy(result.data(),cpu->data(),cpu->nbytes());
    return result;
}
int main(int argc,char **argv) {
    Device gpu(argc>1 && std::string(argv[1])=="hygon" ? Device::Type::HYGON : Device::Type::NVIDIA,0);
    context::setDevice(gpu);
    auto input_cpu=host({2,3},{1,2,3,-1,1,2});
    auto weights=host({4,3},{1,0,1,0,1,1,2,1,0,-1,2,1});
    auto bias_cpu=host({4},{1,2,3,4});
    auto input=input_cpu->to(gpu),resident_weight=weights->to(gpu),resident_bias=bias_cpu->to(gpu);
    auto expected=read(op::linear(input,resident_weight,resident_bias));
    setenv("INFINICORE_DEFER_PARAMETER_STORAGE","1",1);
    Parameter weight({4,3},DataType::F32,gpu,std::numeric_limits<size_t>::max());
    Parameter bias({4},DataType::F32,gpu);
    weight->storage()->enable_streaming();bias->storage()->enable_streaming();
    rejects([&]{weight->data();},"Unmaterialized weight was exposed");
    // Two independently loaded aliases cover the same fused backing allocation.
    Parameter first(weight->narrow({{0,0,2}})),second(weight->narrow({{0,2,2}}));
    first.load(weights->narrow({{0,0,2}}));second.load(weights->narrow({{0,2,2}}));
    bias.load(bias_cpu);
    weights.reset();bias_cpu.reset(); // Core storage must retain owned host bytes.
    std::unordered_map<std::string,Parameter> params{{"weight",weight},{"bias",bias}};
    for (int replay=0;replay<3;replay++) {
        LinearLease lease(params);
        auto actual=op::linear(input,weight,bias);
        lease.finish();
        require(read(actual)==expected,"Streaming linear differs from resident Core");
        rejects([&]{weight->data();},"Weight remained resident after finish");
    }
    auto replacement=host({2,3},{3,0,1,0,4,1});
    first.load(replacement);
    {
        LinearLease lease(params);auto actual=op::linear(input,weight,bias);lease.finish();
        require(read(actual)!=expected,"Reloaded weights reused old values");
    }
    Parameter incomplete({4,3},DataType::F32,gpu);
    incomplete->storage()->enable_streaming();
    Parameter fragment(incomplete->narrow({{0,0,2}}));fragment.load(replacement);
    rejects([&]{LinearLease lease({{"incomplete",incomplete}});},"Incomplete weight coverage was accepted");
    auto overlapping=incomplete->narrow({{0,1,2}});
    rejects([&]{overlapping->copy_from(replacement);},"Overlapping spans were accepted");
    unsetenv("INFINICORE_DEFER_PARAMETER_STORAGE");
    Parameter normal({4,3},DataType::F32,gpu);
    require(!normal->storage()->is_placeholder(),"Default allocation changed");
    std::cout << "{\"status\":\"PASS\",\"checks\":9,\"synthetic\":true,\"full_fl2va_tested\":false}" << std::endl;
}
