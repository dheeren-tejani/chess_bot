#pragma once
#include <string>

namespace chess {
// Blocking UCI loop on stdin/stdout.
//   `engine uci --model M [--visits N] [--workers W] [--batch B] [--cpu]`
//
// `num_workers` and `batch_size` control the root-parallel search. They
// should normally be equal (K workers → up to K tasks per round → batcher
// fills a batch of K). If batch_size < num_workers, some workers idle; if
// batch_size > num_workers, the batcher waits for tasks that can't arrive.
int run_uci(const std::string& model_path, int default_visits, bool prefer_gpu,
            int num_workers = 32, int batch_size = 32);
}  // namespace chess