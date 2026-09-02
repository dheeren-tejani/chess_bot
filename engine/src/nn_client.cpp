#include "nn_client.h"
#include "policy_map.h"
#include <onnxruntime_cxx_api.h>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <sstream>

namespace chess {

struct NNEvaluator::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "chess_engine"};
    std::optional<Ort::Session> session;
    int batch = 0;

    const char* in_names[2] = {"planes", "scalars"};
    const char* out_names[3] = {"policy", "wdl", "material"};

    Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
};

NNEvaluator::NNEvaluator() = default;
NNEvaluator::~NNEvaluator() = default;

bool NNEvaluator::load(const std::string& onnx_path, int batch_size, bool prefer_gpu,
                       std::string* err) {
    try {
        impl_ = std::make_unique<Impl>();
        impl_->batch = batch_size;
        batch_size_ = batch_size;

        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(prefer_gpu ? 1 : 2);
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        gpu_ = false;
        if (prefer_gpu) {
            try {
                OrtCUDAProviderOptions opts{};
                opts.device_id = 0;
                opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                so.AppendExecutionProvider_CUDA(opts);
                gpu_ = true;
            } catch (const Ort::Exception& e) {
                if (err) *err = std::string("CUDA EP unavailable: ") + e.what();
                gpu_ = false;
            }
        }

        impl_->session.emplace(impl_->env, onnx_path.c_str(), so);

        auto in_count = impl_->session->GetInputCount();
        auto out_count = impl_->session->GetOutputCount();
        if (in_count != 2 || out_count != 3)
            throw std::runtime_error("unexpected model io signature");

        {
            auto planes_meta = impl_->session->GetInputTypeInfo(0);
            auto dims = planes_meta.GetTensorTypeAndShapeInfo().GetShape();
            // Expected shape from BitPackedInput wrapper: [Batch, 113, 8]
            if (dims.size() != 3 || dims[1] != NUM_PLANES || dims[2] != 8) {
                std::ostringstream os;
                os << "model planes mismatch: expected shape [?, " << NUM_PLANES << ", 8]";
                throw std::runtime_error(os.str());
            }
            auto scalars_meta = impl_->session->GetInputTypeInfo(1);
            auto sdims = scalars_meta.GetTensorTypeAndShapeInfo().GetShape();
            if (sdims.size() != 2 || sdims[1] != SCALAR_COUNT) {
                throw std::runtime_error("model scalars mismatch");
            }
        }

        // Warmup forward pass
        {
            std::vector<uint64_t> dummy_planes(
                static_cast<size_t>(batch_size_) * NUM_PLANES, 0ULL);
            std::vector<float> dummy_scalars(
                static_cast<size_t>(batch_size_) * SCALAR_COUNT, 0.f);
            std::vector<float> p, w, m;
            evaluate(dummy_planes.data(), dummy_scalars.data(), batch_size_, p, w, m);
        }

        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        impl_.reset();
        return false;
    }
}

void NNEvaluator::evaluate(const uint64_t* planes_words, const float* scalars, int n,
                           std::vector<float>& policy_out, std::vector<float>& wdl_out,
                           std::vector<float>& material_out) {
    if (n <= 0) return;
    n = std::min(n, batch_size_);
    Impl& I = *impl_;

    // Direct uint8 view over planes_words (8 bytes per uint64) -> [n, 113, 8]
    std::array<int64_t, 3> plane_shape{n, NUM_PLANES, 8};
    std::array<int64_t, 2> scalar_shape{n, SCALAR_COUNT};

    Ort::Value plane_tensor = Ort::Value::CreateTensor<uint8_t>(
        I.mem, const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(planes_words)),
        static_cast<size_t>(n) * NUM_PLANES * 8, plane_shape.data(), 3);
        
    Ort::Value scalar_tensor = Ort::Value::CreateTensor<float>(
        I.mem, const_cast<float*>(scalars), static_cast<size_t>(n) * SCALAR_COUNT,
        scalar_shape.data(), 2);

    const char* ins[] = {I.in_names[0], I.in_names[1]};
    const char* outs[] = {I.out_names[0], I.out_names[1], I.out_names[2]};
    Ort::Value inputs[] = {std::move(plane_tensor), std::move(scalar_tensor)};
    auto outputs = I.session->Run(Ort::RunOptions{nullptr}, ins, inputs, 2, outs, 3);

    policy_out.resize(static_cast<size_t>(n) * POLICY_ACTIONS);
    wdl_out.resize(static_cast<size_t>(n) * 3);
    material_out.resize(n);

    std::memcpy(policy_out.data(), outputs[0].GetTensorMutableData<float>(),
                sizeof(float) * n * POLICY_ACTIONS);
    std::memcpy(wdl_out.data(), outputs[1].GetTensorMutableData<float>(),
                sizeof(float) * n * 3);
    std::memcpy(material_out.data(), outputs[2].GetTensorMutableData<float>(),
                sizeof(float) * n);
}

}  // namespace chess