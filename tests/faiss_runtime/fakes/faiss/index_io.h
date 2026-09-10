#pragma once
#include "faiss/IndexFlat.h"
#include <cstdio>
#include <memory>
namespace faiss {
inline void write_index(const Index* index, const char* path) {
    FILE* file = std::fopen(path, "wb");
    if (!file) throw std::runtime_error("open failed");
    std::fwrite(&index->d, sizeof(index->d), 1, file);
    std::fwrite(&index->ntotal, sizeof(index->ntotal), 1, file);
    std::fwrite(index->vectors.data(), sizeof(float), index->vectors.size(), file);
    std::fclose(file);
}
inline Index* read_index(FILE* file) {
    int dimension = 0;
    idx_t count = 0;
    if (std::fread(&dimension, sizeof(dimension), 1, file) != 1 ||
        std::fread(&count, sizeof(count), 1, file) != 1 || dimension <= 0 || count <= 0 || count > 100000)
        throw std::runtime_error("corrupt fixture");
    auto result = std::make_unique<IndexFlatL2>(dimension);
    result->ntotal = count;
    result->vectors.resize(static_cast<size_t>(count) * dimension);
    if (std::fread(result->vectors.data(), sizeof(float), result->vectors.size(), file) != result->vectors.size())
        throw std::runtime_error("truncated fixture");
    return result.release();
}
inline Index* read_index(const char* path) {
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"), &std::fclose);
    if (!file) throw std::runtime_error("missing fixture");
    return read_index(file.get());
}
}
