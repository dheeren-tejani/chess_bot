#pragma once
#include <cstdint>
#include <array>
#include "types.h"

namespace chess {

// Deterministic Zobrist keys (fixed seed -> identical across processes/runs).
class Zobrist {
public:
    std::array<std::array<uint64_t, 64>, 12> psq;
    std::array<uint64_t, 16> castle;
    std::array<uint64_t, 8> epfile;
    uint64_t side;

    static const Zobrist& instance() {
        static Zobrist z;
        return z;
    }

private:
    Zobrist() {
        uint64_t s = 0x9E3779B97F4A7C15ULL;  // golden ratio constant
        for (auto& row : psq)
            for (auto& k : row) k = next(s);
        for (auto& k : castle) k = next(s);
        for (auto& k : epfile) k = next(s);
        side = next(s);
    }
    static uint64_t next(uint64_t& s) {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

}  // namespace chess
