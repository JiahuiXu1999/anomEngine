// Included in cuda_compute.cu's private namespace, after the shared CUDA
// helpers. PCA preserves the CPU float subtraction followed by double
// accumulation.
__global__ void dfkdeProject(const float* patches, const float* mean, const float* basis, float* projected,
                             size_t count, int dimension, int rank, int* invalid) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        const float* patch = patches + (i / rank) * dimension;
        const int component = int(i % rank);
        double value = 0;
        for (int d = 0; d < dimension; ++d) {
            const float difference = __fsub_rn(patch[d], mean[d]);
            value =
                __dadd_rn(value, __dmul_rn(double(difference), double(basis[size_t(d) * rank + component])));
        }
        const float result = float(value);
        if (!isfinite(result)) atomicExch(invalid, 1);
        // A failed projection is reported after all queued work completes.
        projected[i] = isfinite(result) ? result : 0.0F;
    }
}

__device__ double dfkdeWarpMinimum(double value) {
    for (int shift = 16; shift; shift /= 2)
        value = fmin(value, __shfl_down_sync(0xffffffff, value, shift));
    return value;
}
__device__ double dfkdeWarpSum(double value) {
    for (int shift = 16; shift; shift /= 2)
        value += __shfl_down_sync(0xffffffff, value, shift);
    return value;
}

// A block processes one query/gallery partition. Its only persisted output is
// a minimum squared distance and the exponential sum relative to that minimum.
// Smaller workspaces use fewer partitions and stream more gallery rows in each
// thread. No query-by-gallery distance matrix is allocated.
__global__ void dfkdeDensityPartitions(const float* projected, const float* gallery, double* minima,
                                       double* weights, size_t count, int rank, int galleryCount,
                                       int partitions, double inverseBandwidth) {
    __shared__ double warpValues[8];
    __shared__ double minimum;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (size_t i = blockIdx.x; i < count; i += gridDim.x) {
        const float* patch = projected + (i / partitions) * rank;
        const size_t partition = i % partitions;
        double localMinimum = CUDART_INF, localWeight = 0;
        for (size_t g = partition * blockDim.x + threadIdx.x; g < size_t(galleryCount);
             g += size_t(partitions) * blockDim.x) {
            const float* row = gallery + g * rank;
            double distance = 0;
            for (int k = 0; k < rank; ++k) {
                const double difference = double(patch[k]) - double(row[k]);
                distance = __dadd_rn(distance, __dmul_rn(difference, difference));
            }
            if (localWeight == 0) {
                localMinimum = distance;
                localWeight = 1;
            } else if (distance < localMinimum) {
                localWeight = localWeight * exp(-(localMinimum - distance) * inverseBandwidth) + 1;
                localMinimum = distance;
            } else {
                localWeight += exp(-(distance - localMinimum) * inverseBandwidth);
            }
        }
        const double warpMinimum = dfkdeWarpMinimum(localMinimum);
        if (lane == 0) warpValues[warp] = warpMinimum;
        __syncthreads();
        if (warp == 0) {
            double value = lane < 8 ? warpValues[lane] : CUDART_INF;
            value = dfkdeWarpMinimum(value);
            if (lane == 0) minimum = value;
        }
        __syncthreads();
        double weight = localWeight > 0 ? localWeight * exp(-(localMinimum - minimum) * inverseBandwidth) : 0;
        weight = dfkdeWarpSum(weight);
        if (lane == 0) warpValues[warp] = weight;
        __syncthreads();
        if (warp == 0) {
            weight = lane < 8 ? warpValues[lane] : 0;
            weight = dfkdeWarpSum(weight);
            if (lane == 0) {
                minima[i] = minimum;
                weights[i] = weight;
            }
        }
        __syncthreads();
    }
}

__global__ void dfkdeDensityScores(const double* minima, const double* weights, float* output, int rows,
                                   int partitions, double inverseBandwidth, int* invalid) {
    __shared__ double warpValues[8];
    __shared__ double minimum;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (size_t row = blockIdx.x; row < size_t(rows); row += gridDim.x) {
        const size_t base = row * partitions;
        double localMinimum = CUDART_INF;
        for (int p = threadIdx.x; p < partitions; p += blockDim.x)
            localMinimum = fmin(localMinimum, minima[base + p]);
        localMinimum = dfkdeWarpMinimum(localMinimum);
        if (lane == 0) warpValues[warp] = localMinimum;
        __syncthreads();
        if (warp == 0) {
            double value = lane < 8 ? warpValues[lane] : CUDART_INF;
            value = dfkdeWarpMinimum(value);
            if (lane == 0) minimum = value;
        }
        __syncthreads();
        double sum = 0;
        for (int p = threadIdx.x; p < partitions; p += blockDim.x)
            sum += weights[base + p] * exp(-(minima[base + p] - minimum) * inverseBandwidth);
        sum = dfkdeWarpSum(sum);
        if (lane == 0) warpValues[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = lane < 8 ? warpValues[lane] : 0;
            sum = dfkdeWarpSum(sum);
            if (lane == 0) {
                // The existing DFKDE score has no log(gallery_count) correction.
                const float score = float(minimum * inverseBandwidth - log(sum));
                if (!isfinite(score)) atomicExch(invalid, 1);
                output[row] = score;
            }
        }
        __syncthreads();
    }
}

void runDfkde(Context& c, const anom_compute_dfkde& a) {
    const size_t patchCount = elements(a.rows, a.dimension);
    const size_t basisCount = elements(a.dimension, a.rank);
    const size_t galleryCount = elements(a.gallery_count, a.rank);
    require(a.rank <= a.dimension, "DFKDE rank exceeds feature dimension");
    require(std::isfinite(a.sigma) && a.sigma > 0, "Invalid DFKDE bandwidth");
    const float inverseBandwidth = 1.0F / (2.0F * a.sigma * a.sigma);
    require(std::isfinite(inverseBandwidth), "DFKDE inverse bandwidth is non-finite");
    const size_t projectionRowBytes = size_t(a.rank) * sizeof(float);
    require(a.workspace_bytes <= SIZE_MAX &&
                a.workspace_bytes >= projectionRowBytes + 2 * sizeof(double) + sizeof(int),
            "DFKDE workspace is too small");
    const size_t budget = size_t(a.workspace_bytes) - sizeof(int);
    const int partitions = int(std::min((size_t(a.gallery_count) + 255) / 256,
                                        (budget - projectionRowBytes) / (2 * sizeof(double))));
    const size_t rowBytes = projectionRowBytes + size_t(partitions) * 2 * sizeof(double);
    const int chunkRows = int(std::min(size_t(a.rows), budget / rowBytes));
    const size_t partialCount = size_t(chunkRows) * partitions;
    Argument patches(c, a.patches, patchCount * sizeof(float));
    Argument mean(c, a.mean, size_t(a.dimension) * sizeof(float));
    Argument basis(c, a.basis, basisCount * sizeof(float));
    Argument gallery(c, a.gallery, galleryCount * sizeof(float));
    Argument output(c, a.output, size_t(a.rows) * sizeof(float), true);
    Buffer<float> projected(size_t(chunkRows) * a.rank);
    Buffer<double> minima(partialCount), weights(partialCount);
    Buffer<int> invalid(1);
    check(cudaMemsetAsync(invalid.p, 0, sizeof(int), c.stream));
    finiteFloats<<<blocks(patchCount), 256, 0, c.stream>>>(patches.as<float>(), patchCount, invalid.p);
    finiteFloats<<<blocks(a.dimension), 256, 0, c.stream>>>(mean.as<float>(), a.dimension, invalid.p);
    finiteFloats<<<blocks(basisCount), 256, 0, c.stream>>>(basis.as<float>(), basisCount, invalid.p);
    finiteFloats<<<blocks(galleryCount), 256, 0, c.stream>>>(gallery.as<float>(), galleryCount, invalid.p);
    int bad = 0;
    check(cudaMemcpyAsync(&bad, invalid.p, sizeof(int), cudaMemcpyDeviceToHost, c.stream));
    check(cudaStreamSynchronize(c.stream));
    require(!bad, "DFKDE features or statistics contain non-finite values");
    for (int begin = 0; begin < a.rows;) {
        const int rows = std::min(chunkRows, a.rows - begin);
        const size_t projections = size_t(rows) * a.rank;
        const size_t partials = size_t(rows) * partitions;
        dfkdeProject<<<blocks(projections), 256, 0, c.stream>>>(
            patches.as<float>() + size_t(begin) * a.dimension, mean.as<float>(), basis.as<float>(),
            projected.p, projections, a.dimension, a.rank, invalid.p);
        dfkdeDensityPartitions<<<int(std::min<size_t>(partials, 65535)), 256, 0, c.stream>>>(
            projected.p, gallery.as<float>(), minima.p, weights.p, partials, a.rank, a.gallery_count,
            partitions, double(inverseBandwidth));
        dfkdeDensityScores<<<std::min(rows, 65535), 256, 0, c.stream>>>(
            minima.p, weights.p, output.as<float>() + begin, rows, partitions, double(inverseBandwidth),
            invalid.p);
        begin += rows;
    }
    check(cudaGetLastError());
    check(cudaMemcpyAsync(&bad, invalid.p, sizeof(int), cudaMemcpyDeviceToHost, c.stream));
    check(cudaStreamSynchronize(c.stream));
    require(!bad, "DFKDE projection or density score is non-finite");
    output.finish(size_t(a.rows) * sizeof(float));
}
