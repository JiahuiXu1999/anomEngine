#pragma once
// Test double only: exercises production routing/lifetime without a Faiss SDK.
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>
namespace faiss {
using idx_t = int64_t;
enum MetricType { METRIC_INNER_PRODUCT = 0, METRIC_L2 = 1 };
struct Index {
    int d = 0;
    idx_t ntotal = 0;
    MetricType metric_type = METRIC_L2;
    bool is_trained = true;
    std::vector<float> vectors;
    explicit Index(int dimension = 0) : d(dimension) {}
    virtual ~Index() = default;
    virtual void add(idx_t count, const float* data) {
        vectors.insert(vectors.end(), data, data + count * d);
        ntotal += count;
    }
    virtual void search(idx_t count, const float* queries, idx_t k, float* distances, idx_t* labels) const {
        if (k <= 0 || k > ntotal) throw std::runtime_error("invalid k");
        for (idx_t q = 0; q < count; ++q) {
            std::vector<std::pair<float, idx_t>> sorted;
            for (idx_t i = 0; i < ntotal; ++i) {
                float distance = 0;
                for (int j = 0; j < d; ++j) {
                    const float delta = queries[q * d + j] - vectors[i * d + j];
                    distance += delta * delta;
                }
                sorted.emplace_back(distance, i);
            }
            std::sort(sorted.begin(), sorted.end());
            for (idx_t i = 0; i < k; ++i) {
                distances[q * k + i] = sorted[static_cast<size_t>(i)].first;
                labels[q * k + i] = sorted[static_cast<size_t>(i)].second;
            }
        }
    }
    virtual void reconstruct(idx_t key, float* output) const {
        if (key < 0 || key >= ntotal) throw std::runtime_error("invalid key");
        std::copy_n(vectors.data() + key * d, d, output);
    }
};
}
