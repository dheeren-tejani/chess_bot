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

    // cached io names
    const char* in_names[2] = {"planes", "scalars"};
    const char* out_names[3] = {"policy", "wdl", "material"};

    // reusable host buffers
    std::vector<float> plane_floats;      // [B*112*64]
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
        so.SetIntraOpNumThreads(2);
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        gpu_ = false;
        if (prefer_gpu) {
            try {
                OrtCUDAProviderOptions opts{};
                opts.device_id = 0;
                opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
                so.AppendExecutionProvider_CUDA(opts);
                gpu_ = true;
            } catch (const Ort::Exception& e) {
                if (err) *err = std::string("CUDA EP unavailable: ") + e.what();
                gpu_ = false;   // fall through to CPU
            }
        }

        impl_->session.emplace(impl_->env, onnx_path.c_str(), so);

        // sanity-check io signature AND input shapes against current encoder
        Ort::AllocatorWithDefaultOptions alloc;
        auto in_count = impl_->session->GetInputCount();
        auto out_count = impl_->session->GetOutputCount();
        if (in_count != 2 || out_count != 3)
            throw std::runtime_error("unexpected model io signature");

        {
            auto planes_meta = impl_->session->GetInputTypeInfo(0);
            auto planes_t = planes_meta.GetTensorTypeAndShapeInfo();
            auto dims = planes_t.GetShape();
            if (dims.size() != 4 || dims[1] != NUM_PLANES) {
                std::ostringstream os;
                os << "model 'planes' input expects "
                   << (dims.size() > 1 && dims[1] > 0 ? dims[1] : -1)
                   << " plane channels but this engine build encodes " << NUM_PLANES
                   << " - the onnx file is stale; delete it and re-export/bootstrap";
                throw std::runtime_error(os.str());
            }
            auto scalars_meta = impl_->session->GetInputTypeInfo(1);
            auto scalars_t = scalars_meta.GetTensorTypeAndShapeInfo();
            auto sdims = scalars_t.GetShape();
            if (sdims.size() != 2 || sdims[1] != SCALAR_COUNT) {
                std::ostringstream os;
                os << "model 'scalars' input expects "
                   << (sdims.size() > 1 && sdims[1] > 0 ? sdims[1] : -1)
                   << " features but this engine build encodes " << SCALAR_COUNT
                   << " - stale onnx file";
                throw std::runtime_error(os.str());
            }
        }

        impl_->plane_floats.assign(static_cast<size_t>(batch_size_) * NUM_PLANES * 64, 0.f);
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        impl_.reset();
        return false;
    }
}

void NNEvaluator::evaluate(const uint64_t* planes_words, const float* scalars,
                           std::vector<float>& policy_out, std::vector<float>& wdl_out,
                           std::vector<float>& material_out) {
    Impl& I = *impl_;
    const int B = batch_size_;

    // unpack bit-planes to floats: [B][112][64]
    for (int b = 0; b < B; ++b) {
        float* dst = I.plane_floats.data() + static_cast<size_t>(b) * NUM_PLANES * 64;
        const uint64_t* src = planes_words + static_cast<size_t>(b) * NUM_PLANES;
        for (int p = 0; p < NUM_PLANES; ++p) {
            uint64_t w = src[p];
            float* d = dst + static_cast<size_t>(p) * 64;
            for (int s = 0; s < 64; ++s)
                d[s] = static_cast<float>((w >> s) & 1ULL);
        }
    }

    std::array<int64_t, 4> plane_shape{B, NUM_PLANES, 8, 8};
    std::array<int64_t, 2> scalar_shape{B, SCALAR_COUNT};

    Ort::Value plane_tensor = Ort::Value::CreateTensor<float>(
        I.mem, I.plane_floats.data(), I.plane_floats.size(), plane_shape.data(), 4);
    Ort::Value scalar_tensor = Ort::Value::CreateTensor<float>(
        I.mem, const_cast<float*>(scalars), static_cast<size_t>(B) * SCALAR_COUNT,
        scalar_shape.data(), 2);

    const char* ins[] = {I.in_names[0], I.in_names[1]};
    const char* outs[] = {I.out_names[0], I.out_names[1], I.out_names[2]};
    Ort::Value inputs[] = {std::move(plane_tensor), std::move(scalar_tensor)};
    auto outputs = I.session->Run(Ort::RunOptions{nullptr}, ins, inputs, 2, outs, 3);

    policy_out.resize(static_cast<size_t>(B) * POLICY_ACTIONS);
    wdl_out.resize(static_cast<size_t>(B) * 3);
    material_out.resize(B);
    std::memcpy(policy_out.data(), outputs[0].GetTensorMutableData<float>(),
                sizeof(float) * B * POLICY_ACTIONS);
    std::memcpy(wdl_out.data(), outputs[1].GetTensorMutableData<float>(),
                sizeof(float) * B * 3);
    std::memcpy(material_out.data(), outputs[2].GetTensorMutableData<float>(),
                sizeof(float) * B);
}

}  // namespace chess
