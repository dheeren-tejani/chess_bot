cmake_minimum_required(VERSION 3.16)
project(chess_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime")

add_executable(engine
    src/main.cpp
    src/position.cpp
    src/policy_map.cpp
    src/encoder.cpp
    src/nn_client.cpp
    src/mcts.cpp
    src/selfplay.cpp
)

target_compile_options(engine PRIVATE -O3 -march=native -Wall -Wextra -funroll-loops)

target_include_directories(engine PRIVATE "${ORT_ROOT}/include")
target_link_libraries(engine PRIVATE
    Threads::Threads
    ZLIB::ZLIB
    "${ORT_ROOT}/lib/libonnxruntime.so"
)

set_target_properties(engine PROPERTIES
    BUILD_RPATH "\$ORIGIN:${ORT_ROOT}/lib"
    INSTALL_RPATH "\$ORIGIN:${ORT_ROOT}/lib"
)
