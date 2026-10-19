// Included in cuda_compute.cu's private namespace, after the shared CUDA helpers.
// PaDiM keeps one mean and precision matrix per spatial location.
struct PadimChannel {
    const float* data;
    size_t batchStride;
    int height, width;
};

__global__ void padimSelect(const PadimChannel* channels, float* output, size_t count, int dimension,
                            int height, int width) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        const PadimChannel channel = channels[i % dimension];
        const size_t location = i / dimension;
        const int x = int(location % width), y = int(location / width % height);
        const size_t batch = location / (size_t(height) * width);
        const float* plane = channel.data + batch * channel.batchStride;
        output[i] = channel.height == height && channel.width == width
                        ? plane[size_t(y) * width + x]
                        : interpolate(plane, channel.height, channel.width, y, x, height, width);
    }
}

__global__ void padimDifference(const float* patches, const float* mean, float* difference, size_t count,
                                size_t meanCount) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x)
        difference[i] = __fsub_rn(patches[i], mean[i % meanCount]);
}

// Four warps cooperate on each location; columns are contiguous within a warp.
// Only the final reduction changes CPU's accumulation order. Every product and
// partial sum remains double precision, while differences remain float32.
__global__ void padimQuadratic(const float* difference, const float* precision, float* output, size_t count,
                               int locations, int dimension, int* invalid) {
    __shared__ double partial[4];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (size_t i = blockIdx.x; i < count; i += gridDim.x) {
        const float* diff = difference + i * dimension;
        const float* matrix = precision + (i % locations) * size_t(dimension) * dimension;
        double subtotal = 0;
        for (int row = warp; row < dimension; row += 4) {
            double projected = 0;
            for (int column = lane; column < dimension; column += 32)
                projected += double(matrix[size_t(row) * dimension + column]) * diff[column];
            for (int shift = 16; shift; shift /= 2)
                projected += __shfl_down_sync(0xffffffff, projected, shift);
            if (lane == 0) subtotal += double(diff[row]) * projected;
        }
        if (lane == 0) partial[warp] = subtotal;
        __syncthreads();
        if (threadIdx.x == 0) {
            const double value = (partial[0] + partial[1]) + (partial[2] + partial[3]);
            // Match the CPU adapter's float conversion before square root.
            const float distance = sqrtf(float(fmax(value, 0.0)));
            if (!isfinite(value) || value < -1.0e-4 || !isfinite(distance)) atomicExch(invalid, 1);
            output[i] = distance;
        }
        __syncthreads();
    }
}

__global__ void padimFiniteDoubles(const double* values, size_t count, int* invalid) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x)
        if (!isfinite(values[i])) atomicExch(invalid, 1);
}

__global__ void padimAccumulateSums(const float* patches, double* sums, size_t count, int n, int locations,
                                    int dimension, int begin) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        double value = sums[i];
        const size_t source = (size_t(begin) + i / dimension) * dimension + i % dimension;
        for (int batch = 0; batch < n; ++batch)
            value += double(patches[size_t(batch) * locations * dimension + source]);
        sums[i] = value;
    }
}

__global__ void padimAccumulateMoments(const float* patches, double* moments, size_t count, int n,
                                       int locations, int dimension, int begin) {
    const size_t matrixSize = size_t(dimension) * dimension;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        const int row = int(i / dimension % dimension), column = int(i % dimension);
        const size_t source = (size_t(begin) + i / matrixSize) * dimension;
        double value = moments[i];
        for (int batch = 0; batch < n; ++batch) {
            const float* patch = patches + size_t(batch) * locations * dimension + source;
            // Keep the CPU's multiply then add behavior for stable covariance.
            value = __dadd_rn(value, __dmul_rn(double(patch[row]), double(patch[column])));
        }
        moments[i] = value;
    }
}

__global__ void padimCovariance(const double* sums, const double* moments, double* covariance,
                                double* inverse, float* mean, size_t count, int dimension,
                                double inverseSamples, double regularization, int* invalid) {
    const size_t matrixSize = size_t(dimension) * dimension;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        const int row = int(i / dimension % dimension), column = int(i % dimension);
        const size_t base = i / matrixSize * dimension;
        const double rowMean = __dmul_rn(sums[base + row], inverseSamples);
        const double columnMean = __dmul_rn(sums[base + column], inverseSamples);
        double value = __dsub_rn(__dmul_rn(moments[i], inverseSamples), __dmul_rn(rowMean, columnMean));
        if (row == column) value = __dadd_rn(value, regularization);
        covariance[i] = value;
        inverse[i] = row == column ? 1.0 : 0.0;
        if (column == 0) mean[base + row] = float(rowMean);
        if (!isfinite(value) || !isfinite(float(rowMean))) atomicExch(invalid, 1);
    }
}

__global__ void padimMatrixPointers(double* matrices, double** pointers, int count, size_t matrixSize) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(count);
         i += size_t(blockDim.x) * gridDim.x)
        pointers[i] = matrices + i * matrixSize;
}

__global__ void padimPrecision(const double* inverse, float* precision, size_t count, int* invalid) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        const float value = float(inverse[i]);
        if (!isfinite(value)) atomicExch(invalid, 1);
        precision[i] = value;
    }
}

void padimCheckInvalid(Context& c, int* invalid, const char* message) {
    int value = 0;
    check(cudaMemcpyAsync(&value, invalid, sizeof(value), cudaMemcpyDeviceToHost, c.stream));
    check(cudaStreamSynchronize(c.stream));
    require(!value, message);
}

void runSelectFeatures(Context& c, const anom_compute_select_features& a) {
    require(a.layers && a.count > 0 && a.count <= 1024 && a.channels, "Invalid PaDiM layers/channels");
    const size_t total = elements(a.layers[0].n, a.height, a.width, a.dimension);
    require(a.layers[0].h == a.height && a.layers[0].w == a.width,
            "PaDiM first layer does not match statistics grid");
    int totalChannels = 0;
    std::vector<std::unique_ptr<Argument>> inputs;
    inputs.reserve(a.count);
    for (size_t j = 0; j < a.count; ++j) {
        const auto& layer = a.layers[j];
        const size_t count = elements(layer.n, layer.c, layer.h, layer.w);
        require(layer.n == a.layers[0].n, "PaDiM feature batch mismatch");
        require(layer.c <= INT_MAX - totalChannels, "PaDiM channel count overflow");
        totalChannels += layer.c;
        inputs.emplace_back(std::make_unique<Argument>(c, layer.buffer, count * sizeof(float)));
    }
    require(a.dimension <= totalChannels, "PaDiM selected channel count exceeds available channels");
    std::vector<int32_t> sorted(a.channels, a.channels + a.dimension);
    std::sort(sorted.begin(), sorted.end());
    require(sorted.front() >= 0 && sorted.back() < totalChannels &&
                std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
            "PaDiM channels must be unique and in range");
    std::vector<PadimChannel> channels;
    channels.reserve(a.dimension);
    for (int i = 0; i < a.dimension; ++i) {
        int channel = a.channels[i];
        size_t layerIndex = 0;
        while (channel >= a.layers[layerIndex].c) channel -= a.layers[layerIndex++].c;
        const auto& layer = a.layers[layerIndex];
        channels.push_back({inputs[layerIndex]->as<float>() + size_t(channel) * layer.h * layer.w,
                            size_t(layer.c) * layer.h * layer.w, layer.h, layer.w});
    }
    Buffer<PadimChannel> descriptors(channels.size());
    Argument out(c, a.output, total * sizeof(float), true);
    check(cudaMemcpyAsync(descriptors.p, channels.data(), channels.size() * sizeof(PadimChannel),
                          cudaMemcpyHostToDevice, c.stream));
    padimSelect<<<blocks(total), 256, 0, c.stream>>>(descriptors.p, out.as<float>(), total, a.dimension,
                                                     a.height, a.width);
    // Fitting may mutate persistent statistics next; reject all bad features now.
    validateFinite(c, out.as<float>(), total);
    out.finish(total * sizeof(float));
}

void runPadimDistance(Context& c, const anom_compute_padim_distance& a) {
    const size_t patchCount = elements(a.n, a.locations, a.dimension);
    const size_t meanCount = elements(a.locations, a.dimension);
    const size_t matrixCount = elements(a.locations, a.dimension, a.dimension);
    const size_t outputCount = elements(a.n, a.locations);
    Argument patches(c, a.patches, patchCount * sizeof(float));
    Argument mean(c, a.mean, meanCount * sizeof(float));
    Argument precision(c, a.precision, matrixCount * sizeof(float));
    Argument output(c, a.output, outputCount * sizeof(float), true);
    Buffer<float> difference(patchCount);
    Buffer<int> invalid(1);
    check(cudaMemsetAsync(invalid.p, 0, sizeof(int), c.stream));
    padimDifference<<<blocks(patchCount), 256, 0, c.stream>>>(patches.as<float>(), mean.as<float>(),
                                                              difference.p, patchCount, meanCount);
    padimQuadratic<<<int(std::min<size_t>(outputCount, 65535)), 128, 0, c.stream>>>(
        difference.p, precision.as<float>(), output.as<float>(), outputCount, a.locations, a.dimension,
        invalid.p);
    padimCheckInvalid(c, invalid.p, "PaDiM Mahalanobis quadratic form is invalid");
    output.finish(outputCount * sizeof(float));
}

void runPadimAccumulate(Context& c, const anom_compute_padim_accumulate& a) {
    const size_t patchCount = elements(a.n, a.locations, a.dimension);
    const size_t sumCount = elements(a.location_count, a.dimension);
    const size_t matrixCount = elements(a.location_count, a.dimension, a.dimension);
    require(a.location_begin >= 0 && a.location_begin <= a.locations &&
                a.location_count <= a.locations - a.location_begin,
            "PaDiM location chunk is out of range");
    Argument patches(c, a.patches, patchCount * sizeof(float));
    Argument sums(c, a.sums, sumCount * sizeof(double));
    Argument moments(c, a.moments, matrixCount * sizeof(double));
    validateFinite(c, patches.as<float>(), patchCount);
    Buffer<int> invalid(1);
    check(cudaMemsetAsync(invalid.p, 0, sizeof(int), c.stream));
    padimFiniteDoubles<<<blocks(sumCount), 256, 0, c.stream>>>(sums.as<double>(), sumCount, invalid.p);
    padimFiniteDoubles<<<blocks(matrixCount), 256, 0, c.stream>>>(moments.as<double>(), matrixCount,
                                                                  invalid.p);
    padimCheckInvalid(c, invalid.p, "PaDiM statistics contain non-finite values");
    padimAccumulateSums<<<blocks(sumCount), 256, 0, c.stream>>>(
        patches.as<float>(), sums.as<double>(), sumCount, a.n, a.locations, a.dimension, a.location_begin);
    padimAccumulateMoments<<<blocks(matrixCount), 256, 0, c.stream>>>(
        patches.as<float>(), moments.as<double>(), matrixCount, a.n, a.locations, a.dimension,
        a.location_begin);
    sums.finish(sumCount * sizeof(double));
    moments.finish(matrixCount * sizeof(double));
}

void runPadimFinalize(Context& c, const anom_compute_padim_finalize& a) {
    const size_t sumCount = elements(a.locations, a.dimension);
    const size_t matrixSize = elements(a.dimension, a.dimension);
    const size_t matrixCount = elements(a.locations, a.dimension, a.dimension);
    require(a.samples > 0 && std::isfinite(a.regularization) && a.regularization >= 0,
            "Invalid PaDiM population or regularization");
    Argument sums(c, a.sums, sumCount * sizeof(double));
    Argument moments(c, a.moments, matrixCount * sizeof(double));
    Argument mean(c, a.mean, sumCount * sizeof(float), true);
    Argument precision(c, a.precision, matrixCount * sizeof(float), true);
    // The caller bounds location chunks, so quadratic temporary memory is bounded.
    Buffer<double> covariance(matrixCount), inverse(matrixCount);
    Buffer<double*> covariancePointers(a.locations), inversePointers(a.locations);
    Buffer<int> info(a.locations), invalid(1);
    check(cudaMemsetAsync(invalid.p, 0, sizeof(int), c.stream));
    padimCovariance<<<blocks(matrixCount), 256, 0, c.stream>>>(
        sums.as<double>(), moments.as<double>(), covariance.p, inverse.p, mean.as<float>(), matrixCount,
        a.dimension, 1.0 / double(a.samples), a.regularization, invalid.p);
    padimCheckInvalid(c, invalid.p, "PaDiM covariance or mean is not finite");
    padimMatrixPointers<<<blocks(a.locations), 256, 0, c.stream>>>(covariance.p, covariancePointers.p,
                                                                   a.locations, matrixSize);
    padimMatrixPointers<<<blocks(a.locations), 256, 0, c.stream>>>(inverse.p, inversePointers.p, a.locations,
                                                                   matrixSize);
    solver(cusolverDnDpotrfBatched(c.eigen, CUBLAS_FILL_MODE_LOWER, a.dimension, covariancePointers.p,
                                   a.dimension, info.p, a.locations));
    std::vector<int> statuses(a.locations);
    check(cudaMemcpyAsync(statuses.data(), info.p, statuses.size() * sizeof(int), cudaMemcpyDeviceToHost,
                          c.stream));
    check(cudaStreamSynchronize(c.stream));
    for (int status : statuses) require(status == 0, "PaDiM covariance matrix is not positive definite");
    const double one = 1.0;
    blas(cublasDtrsmBatched(c.handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                            CUBLAS_DIAG_NON_UNIT, a.dimension, a.dimension, &one, covariancePointers.p,
                            a.dimension, inversePointers.p, a.dimension, a.locations));
    blas(cublasDtrsmBatched(c.handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                            CUBLAS_DIAG_NON_UNIT, a.dimension, a.dimension, &one, covariancePointers.p,
                            a.dimension, inversePointers.p, a.dimension, a.locations));
    padimPrecision<<<blocks(matrixCount), 256, 0, c.stream>>>(inverse.p, precision.as<float>(), matrixCount,
                                                              invalid.p);
    padimCheckInvalid(c, invalid.p, "PaDiM precision matrix cannot be represented as float32");
    mean.finish(sumCount * sizeof(float));
    precision.finish(matrixCount * sizeof(float));
}

void runPadimProbe(Context&, const uint32_t& version) {
    require(version == 1, "Unsupported PaDiM compute version");
}
