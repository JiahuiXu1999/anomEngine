// Included by cuda_compute.cu after its context, buffer and kernel helpers.
// One thread per spatial location keeps NCHW channel loads coalesced while
// preserving the CPU's sequential double-precision channel accumulation.
__global__ void efficientAdChannelL2(const float* input, const float* reference, float* output,
                                    size_t count, size_t plane, int channels, int* invalid) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        const size_t base = (i / plane) * size_t(channels) * plane + i % plane;
        double sum = 0.0;
        for (int channel = 0; channel < channels; ++channel) {
            const size_t index = base + size_t(channel) * plane;
            const double value = reference ? double(input[index]) - double(reference[index])
                                           : double(input[index]);
            sum = __dadd_rn(sum, __dmul_rn(value, value));
        }
        const float value = static_cast<float>(sqrt(sum));
        if (!isfinite(value)) atomicExch(invalid, 1);
        output[i] = value;
    }
}

struct EfficientAdExtrema {
    float hardMin, hardMax, softMin, softMax;
};

__device__ EfficientAdExtrema efficientAdCombine(EfficientAdExtrema a, EfficientAdExtrema b) {
    return {fminf(a.hardMin, b.hardMin), fmaxf(a.hardMax, b.hardMax),
            fminf(a.softMin, b.softMin), fmaxf(a.softMax, b.softMax)};
}

__global__ void efficientAdMapExtrema(const float* hard, const float* soft,
                                     EfficientAdExtrema* partial, size_t plane, size_t groups,
                                     size_t tasks, int* invalid) {
    __shared__ EfficientAdExtrema scratch[256];
    for (size_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const size_t image = task / groups, group = task % groups;
        EfficientAdExtrema value{CUDART_INF_F, -CUDART_INF_F, CUDART_INF_F, -CUDART_INF_F};
        for (size_t pixel = group * blockDim.x + threadIdx.x; pixel < plane;
             pixel += groups * blockDim.x) {
            const size_t index = image * plane + pixel;
            const float h = hard[index], s = soft[index];
            if (!isfinite(h) || !isfinite(s)) atomicExch(invalid, 1);
            value = efficientAdCombine(value, {h, h, s, s});
        }
        scratch[threadIdx.x] = value;
        __syncthreads();
        for (int stride = 128; stride > 0; stride /= 2) {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] = efficientAdCombine(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x == 0) partial[task] = scratch[0];
        __syncthreads();
    }
}

__global__ void efficientAdReduceExtrema(const EfficientAdExtrema* partial,
                                        EfficientAdExtrema* extrema, size_t groups, int images) {
    __shared__ EfficientAdExtrema scratch[256];
    for (size_t image = blockIdx.x; image < size_t(images); image += gridDim.x) {
        EfficientAdExtrema value{CUDART_INF_F, -CUDART_INF_F, CUDART_INF_F, -CUDART_INF_F};
        for (size_t group = threadIdx.x; group < groups; group += blockDim.x)
            value = efficientAdCombine(value, partial[image * groups + group]);
        scratch[threadIdx.x] = value;
        __syncthreads();
        for (int stride = 128; stride > 0; stride /= 2) {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] = efficientAdCombine(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x == 0) extrema[image] = scratch[0];
        __syncthreads();
    }
}

__device__ float efficientAdNormalize(float value, float minimum, float maximum) {
    if (!(maximum > minimum)) return 0.0F;
    // Match feature::normalizeMinMax. A float scale/shift expression suffers
    // catastrophic cancellation on maps with a large offset and small range.
    const double range = double(maximum) - double(minimum);
    return static_cast<float>((double(value) - double(minimum)) / range);
}

__global__ void efficientAdBlend(const float* hard, const float* soft, float* output,
                                const EfficientAdExtrema* extrema, size_t count, size_t plane,
                                float hardAlpha, float softAlpha, int* invalid) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        const auto bounds = extrema[i / plane];
        const float h = efficientAdNormalize(hard[i], bounds.hardMin, bounds.hardMax);
        const float s = efficientAdNormalize(soft[i], bounds.softMin, bounds.softMax);
        const float result = static_cast<float>(double(h) * hardAlpha + double(s) * softAlpha);
        if (!isfinite(result)) atomicExch(invalid, 1);
        output[i] = result;
    }
}

void efficientAdCheckFinite(Context& c, const Buffer<int>& invalid) {
    int bad = 0;
    check(cudaMemcpyAsync(&bad, invalid.p, sizeof(int), cudaMemcpyDeviceToHost, c.stream));
    check(cudaStreamSynchronize(c.stream));
    require(!bad, "EfficientAD input or result is non-finite");
}

void runChannelL2(Context& c, const anom_compute_channel_l2& a) {
    const auto& layer = a.input;
    const size_t count = elements(layer.n, layer.h, layer.w);
    const size_t inputCount = elements(layer.n, layer.c, layer.h, layer.w);
    require(a.difference == 0 || a.difference == 1, "Invalid EfficientAD difference mode");
    Argument input(c, layer.buffer, inputCount * sizeof(float));
    std::unique_ptr<Argument> reference;
    if (a.difference) {
        require(a.reference.n == layer.n && a.reference.c == layer.c &&
                    a.reference.h == layer.h && a.reference.w == layer.w,
                "EfficientAD channel difference requires equal shapes");
        reference = std::make_unique<Argument>(c, a.reference.buffer, inputCount * sizeof(float));
    }
    Argument output(c, a.output, count * sizeof(float), true);
    Buffer<int> invalid(1);
    check(cudaMemsetAsync(invalid.p, 0, sizeof(int), c.stream));
    efficientAdChannelL2<<<blocks(count), 256, 0, c.stream>>>(
        input.as<float>(), reference ? reference->as<float>() : nullptr, output.as<float>(), count,
        elements(layer.h, layer.w), layer.c, invalid.p);
    efficientAdCheckFinite(c, invalid);
    output.finish(count * sizeof(float));
}

void runNormalizedBlend(Context& c, const anom_compute_normalized_blend& a) {
    const size_t count = elements(a.n, a.height, a.width), plane = elements(a.height, a.width);
    const float totalWeight = a.hard_weight + a.soft_weight;
    require(std::isfinite(a.hard_weight) && a.hard_weight >= 0 && std::isfinite(a.soft_weight) &&
                a.soft_weight >= 0 && std::isfinite(totalWeight) && totalWeight > 0,
            "EfficientAD weights must have a positive finite sum");
    Argument hard(c, a.hard, count * sizeof(float)), soft(c, a.soft, count * sizeof(float)),
        output(c, a.output, count * sizeof(float), true);
    const size_t groups = std::min<size_t>((plane + 255) / 256, 1024);
    require(size_t(a.n) <= SIZE_MAX / groups / sizeof(EfficientAdExtrema), "EfficientAD workspace overflow");
    const size_t tasks = size_t(a.n) * groups;
    Buffer<EfficientAdExtrema> partial(tasks), extrema(a.n);
    Buffer<int> invalid(1);
    check(cudaMemsetAsync(invalid.p, 0, sizeof(int), c.stream));
    efficientAdMapExtrema<<<static_cast<int>(std::min<size_t>(tasks, 65535)), 256, 0, c.stream>>>(
        hard.as<float>(), soft.as<float>(), partial.p, plane, groups, tasks, invalid.p);
    efficientAdReduceExtrema<<<std::min(a.n, 65535), 256, 0, c.stream>>>(partial.p, extrema.p, groups, a.n);
    efficientAdBlend<<<blocks(count), 256, 0, c.stream>>>(hard.as<float>(), soft.as<float>(),
        output.as<float>(), extrema.p, count, plane, a.hard_weight / totalWeight,
        a.soft_weight / totalWeight, invalid.p);
    efficientAdCheckFinite(c, invalid);
    output.finish(count * sizeof(float));
}
