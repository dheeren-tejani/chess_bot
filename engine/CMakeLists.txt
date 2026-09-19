cmake_minimum_required(VERSION 3.16)
project(chess_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Portable build option: when ON, target x86-64-v2 (Nehalem, 2008) instead
# of `native`. Required for containerized deploys where the build host and
# runtime host may have different CPU features. `native` on a modern build
# host emits AVX-512/BMI2 instructions that SIGILL on older runtime hosts.
option(PORTABLE_BUILD "Build for x86-64-v2 baseline instead of -march=native" OFF)

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
    src/uci.cpp
)

if(PORTABLE_BUILD)
    set(ARCH_FLAGS -march=x86-64-v2 -mtune=generic)
    message(STATUS "Portable build: ${ARCH_FLAGS}")
else()
    set(ARCH_FLAGS -march=native)
    message(STATUS "Native build: ${ARCH_FLAGS}")
endif()

target_compile_options(engine PRIVATE
    -O3
    ${ARCH_FLAGS}
    -Wall -Wextra -funroll-loops
)

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