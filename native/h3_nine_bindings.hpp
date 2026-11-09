#pragma once
#include "h3_nine_bridge.hpp"
#include "h3_dlpack.h" // Unmodified public DLPack ABI; see THIRD_PARTY.md.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <infinicore/ops/add.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/mha.hpp>
#include <cmath>
#include <vector>

namespace infinilm::h3_nine {
struct DLPackOwner {
    DLManagedTensor managed{};
    Tensor tensor;
    std::vector<int64_t> shape, strides;
    explicit DLPackOwner(const Tensor &t) : tensor(t), shape(t->shape().begin(), t->shape().end()),
                                          strides(t->strides().begin(), t->strides().end()) {
        auto &dl = managed.dl_tensor;
        dl.data = const_cast<std::byte *>(t->data());
        auto type = t->device().getType();
        if (type != infinicore::Device::Type::NVIDIA && type != infinicore::Device::Type::HYGON)
            throw std::invalid_argument("DLPack bridge requires GPU tensors");
        dl.device = {type == infinicore::Device::Type::HYGON ? kDLROCM : kDLCUDA,
                     static_cast<int>(t->device().getIndex())};
        dl.ndim = static_cast<int>(shape.size());
        if (t->dtype() == infinicore::DataType::BF16) dl.dtype = {kDLBfloat, 16, 1};
        else if (t->dtype() == infinicore::DataType::F16) dl.dtype = {kDLFloat, 16, 1};
        else if (t->dtype() == infinicore::DataType::F32) dl.dtype = {kDLFloat, 32, 1};
        else throw std::invalid_argument("Unsupported H3 bridge dtype");
        dl.shape = shape.data(); dl.strides = strides.data(); dl.byte_offset = 0;
        managed.manager_ctx = this;
        managed.deleter = [](DLManagedTensor *m) { delete static_cast<DLPackOwner *>(m->manager_ctx); };
    }
};

inline void bind(pybind11::module_ &m) {
    namespace py = pybind11;
    m.def("_h3_to_dlpack", [](const Tensor &t) {
        auto *owner = new DLPackOwner(t);
        return py::capsule(&owner->managed, "dltensor", [](PyObject *capsule) {
            if (PyCapsule_IsValid(capsule, "dltensor")) {
                auto *managed = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
                managed->deleter(managed);
            }
        });
    });
    m.def("_h3_set_nine_callback", [](py::object fn, bool pointwise, bool attention) {
        if (fn.is_none()) { set_callback({}); return; }
        if (!PyCallable_Check(fn.ptr())) throw std::invalid_argument("Callback must be callable");
        set_callback([fn](const char *op, const Tensor &x, const Tensor &a,
                          const Tensor &b, const Tensor &out, uintptr_t stream) {
            py::gil_scoped_acquire acquire;
            // Immutable host metadata forms the cache key. Current pointers
            // are supplied anew, never retained as a generated-result cache.
            py::tuple signature(3 + x->ndim() * 5);
            size_t index = 0;
            signature[index++] = static_cast<int>(x->device().getType());
            signature[index++] = x->device().getIndex();
            signature[index++] = static_cast<int>(x->dtype());
            for (auto size : x->shape()) signature[index++] = size;
            py::tuple pointers(4);
            Tensor tensors[] = {x, a, b, out};
            for (size_t i = 0; i < 4; ++i) {
                pointers[i] = py::int_(reinterpret_cast<uintptr_t>(tensors[i]->data()));
                for (auto stride : tensors[i]->strides()) signature[index++] = stride;
            }
            fn(op, x, a, b, out, stream, signature, pointers);
        }, pointwise, attention);
    }, py::arg("fn"), py::arg("pointwise") = true, py::arg("attention") = true);
    m.def("_h3_nine_stats", [] { py::dict result; result["adaln"] = adaln_calls.load();
                                result["residual"] = residual_calls.load(); result["attention"] = attention_calls.load(); return result; });
    // Exercise the same dispatch used by the patched model, including fallback.
    m.def("_h3_test_dispatch", [](std::string op, const Tensor &x, const Tensor &a, const Tensor &b) {
        py::gil_scoped_release release;
        Tensor out;
        if (op != "adaln" && op != "residual" && op != "attention") throw std::invalid_argument("Unknown operation");
        if (try_compute(op.c_str(), x, a, b, out)) return out;
        if (op == "attention") throw std::runtime_error("Install a validated attention callback before testing attention");
        if (op == "adaln") return infinicore::op::add(infinicore::op::add(x, infinicore::op::mul(x, a)), b);
        return infinicore::op::add(x, infinicore::op::mul(a, b));
    });
}
} // namespace infinilm::h3_nine
