#pragma once                                                                                                                                               
#include <string>                                                                                                                                          
                                                                                                                                                           
namespace chess {                                                                                                                                          
// Blocking UCI loop on stdin/stdout. `engine uci --model M [--visits N] [--cpu]`                                                                          
int run_uci(const std::string& model_path, int default_visits, bool prefer_gpu);                                                                           
}  // namespace chess 