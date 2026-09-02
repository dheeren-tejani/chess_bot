#pragma once
#include <string>
#include <vector>
#include <array>
#include <memory>
#include "encoder.h"
#include "policy_map.h"

namespace chess {

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

    // Evaluate up to `n` positions (1 <= n <= batch_size_).
    void evaluate(const uint64_t* planes_words, const float* scalars, int n,
                  std::vector<float>& policy_out, std::vector<float>& wdl_out,
                  std::vector<float>& material_out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int batch_size_ = 0;
    bool gpu_ = false;
};

}  // namespace chess