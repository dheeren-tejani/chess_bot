#pragma once
#include <string>
#include <vector>
#include <array>
#include <memory>
#include "encoder.h"
#include "policy_map.h"

namespace chess {

// Batched ONNX Runtime inference client.
// Contract: model inputs  "planes" uint8 [B,112,8,8], "scalars" float [B,4]
//           model outputs "policy" float [B,4672], "wdl" float [B,3], "material" float [B,1]
// Single-threaded use (one evaluator owned by one feeder thread).
class NNEvaluator {
public:
    NNEvaluator();
    ~NNEvaluator();
    NNEvaluator(const NNEvaluator&) = delete;
    NNEvaluator& operator=(const NNEvaluator&) = delete;

    bool load(const std::string& onnx_path, int batch_size, bool prefer_gpu,
              std::string* err = nullptr);

    int batch_size() const { return batch_size_; }
    bool is_gpu() const { return gpu_; }

    // Evaluate exactly `batch_size_` positions (caller pads).
    // planes_words: batch_size_ * NUM_PLANES packed uint64 words.
    // scalars:      batch_size_ * SCALAR_COUNT floats.
    // Outputs resized to [batch]*4672 / [*3] / [*1].
    void evaluate(const uint64_t* planes_words, const float* scalars,
                  std::vector<float>& policy_out, std::vector<float>& wdl_out,
                  std::vector<float>& material_out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int batch_size_ = 0;
    bool gpu_ = false;
};

}  // namespace chess
