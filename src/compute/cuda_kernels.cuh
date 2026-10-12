// Device-only kernels. Included in the private namespace of cuda_compute.cu.
// All dimensions and buffer sizes are validated by the host dispatch layer.
__global__ void pool(const float* x, float* y, size_t total, int h, int w, int oh, int ow, int k, int stride,
                     int pad) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += size_t(blockDim.x) * gridDim.x) {
        int yy = int(i / ow % oh) * stride - pad, xx = int(i % ow) * stride - pad;
        size_t plane = i / (size_t(oh) * ow) * h * w;
        float sum = 0;
        for (int dy = 0; dy < k; ++dy)
            for (int dx = 0; dx < k; ++dx)
                if (yy + dy >= 0 && yy + dy < h && xx + dx >= 0 && xx + dx < w)
                    sum += x[plane + (yy + dy) * w + xx + dx];
        y[i] = sum / float(k * k);
    }
}
__device__ float interpolate(const float* x, int h, int w, int y, int z, int oh, int ow) {
    float fy = float((y + 0.5) * double(h) / oh - 0.5), fx = float((z + 0.5) * double(w) / ow - 0.5);
    int y0 = int(floorf(fy)), x0 = int(floorf(fx));
    fy -= y0;
    fx -= x0;
    int y1 = max(0, min(h - 1, y0 + 1)), x1 = max(0, min(w - 1, x0 + 1));
    y0 = max(0, min(h - 1, y0));
    x0 = max(0, min(w - 1, x0));
    float a = x[y0 * w + x0] * (1 - fx) + x[y0 * w + x1] * fx,
          b = x[y1 * w + x0] * (1 - fx) + x[y1 * w + x1] * fx;
    return a * (1 - fy) + b * fy;
}
__global__ void pack(const float* x, float* y, size_t total, int c, int h, int w, int oh, int ow,
                     int dimension, int offset, int mean, int layers) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += size_t(blockDim.x) * gridDim.x) {
        int channel = int(i % c);
        size_t location = i / c;
        int xx = int(location % ow), yy = int(location / ow % oh);
        size_t n = location / (size_t(oh) * ow);
        float v = interpolate(x + (n * c + channel) * h * w, h, w, yy, xx, oh, ow);
        size_t dst = location * dimension + offset + channel;
        if (mean)
            y[dst] += v;
        else
            y[dst] = v;
    }
}
__global__ void divide(float* x, size_t n, float d) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x)
        x[i] /= d;
}
struct Best {
    double value;
    int index;
};
__device__ Best better(Best a, Best b) {
    return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}
__global__ void distances(const float* x, double* minimum, Best* partial, const int* center, int n, int d) {
    __shared__ Best s[256];
    Best best{-1, INT_MAX};
    int selected = *center;
    for (int i = int(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += int(blockDim.x * gridDim.x)) {
        double sum = 0;
        for (int j = 0; j < d; ++j) {
            double diff = double(x[size_t(i) * d + j]) - x[size_t(selected) * d + j];
            sum += diff * diff;
        }
        double v = minimum[i];
        if (i == selected)
            v = -1;
        else if (v >= 0)
            v = fmin(v, sum);
        minimum[i] = v;
        best = better(best, {v, i});
    }
    s[threadIdx.x] = best;
    __syncthreads();
    for (int k = 128; k; k /= 2) {
        if (threadIdx.x < k) s[threadIdx.x] = better(s[threadIdx.x], s[threadIdx.x + k]);
        __syncthreads();
    }
    if (!threadIdx.x) partial[blockIdx.x] = s[0];
}
__global__ void nextCenter(const Best* p, int n, int* center, int64_t* selected, int iteration) {
    __shared__ Best s[256];
    Best b{-1, INT_MAX};
    for (int i = threadIdx.x; i < n; i += 256) b = better(b, p[i]);
    s[threadIdx.x] = b;
    __syncthreads();
    for (int k = 128; k; k /= 2) {
        if (threadIdx.x < k) s[threadIdx.x] = better(s[threadIdx.x], s[threadIdx.x + k]);
        __syncthreads();
    }
    if (!threadIdx.x) {
        *center = s[0].index;
        selected[iteration] = s[0].index;
    }
}
__global__ void initDistances(double* p, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) p[i] = CUDART_INF;
}
__global__ void centerData(const float* x, double* y, const double* mean, size_t total, int d) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += size_t(blockDim.x) * gridDim.x)
        y[i] = double(x[i]) - mean[i % d];
}
__global__ void scaleData(const double* x, float* y, const double* scale, size_t n, int r) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x)
        y[i] = float(x[i] * scale[i % r]);
}
__global__ void residualData(const double* x, const double* reconstruction, float* y, int n, int d) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        double s = 0;
        for (int j = 0; j < d; ++j) {
            double v = x[size_t(i) * d + j] - reconstruction[size_t(i) * d + j];
            s += v * v;
        }
        y[i] = float(s);
    }
}
__global__ void means(const float* x, double* mean, int n, int d) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < d; j += blockDim.x * gridDim.x) {
        double sum = 0;
        for (int i = 0; i < n; ++i) sum += x[size_t(i) * d + j];
        mean[j] = sum / n;
    }
}
__global__ void preprocess(const unsigned char* x, float* y, size_t stride, int h, int w, int channels,
                           anom_compute_preprocess a, int image) {
    size_t total = size_t(a.crop_h) * a.crop_w * 3;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += size_t(blockDim.x) * gridDim.x) {
        int c = int(i % 3), xx = int(i / 3 % a.crop_w) + (a.resize_w - a.crop_w) / 2,
            yy = int(i / 3 / a.crop_w) + (a.resize_h - a.crop_h) / 2;
        int sc = channels == 1 ? 0 : (a.rgb ? 2 - c : c);
        int value;
        if (a.interpolation == 0) {
            int sy = min(h - 1, int(double(yy) * h / a.resize_h)),
                sx = min(w - 1, int(double(xx) * w / a.resize_w));
            value = x[sy * stride + sx * channels + sc];
        } else {
            float fy = float((yy + 0.5) * double(h) / a.resize_h - 0.5),
                  fx = float((xx + 0.5) * double(w) / a.resize_w - 0.5);
            int y0 = int(floorf(fy)), x0 = int(floorf(fx));
            fy -= y0;
            fx -= x0;
            if (x0 < 0) {
                x0 = 0;
                fx = 0;
            }
            if (x0 >= w - 1) {
                x0 = w - 1;
                fx = 0;
            }
            int x1 = min(w - 1, x0 + 1), y1 = max(0, min(h - 1, y0 + 1));
            y0 = max(0, min(h - 1, y0));
            int ax = __float2int_rn((1 - fx) * 2048), bx = __float2int_rn(fx * 2048),
                ay = __float2int_rn((1 - fy) * 2048), by = __float2int_rn(fy * 2048);
            int top = x[y0 * stride + x0 * channels + sc] * ax + x[y0 * stride + x1 * channels + sc] * bx;
            int bottom = x[y1 * stride + x0 * channels + sc] * ax + x[y1 * stride + x1 * channels + sc] * bx;
            // OpenCV's 8-bit INTER_LINEAR separates fixed-point rounding
            // after the horizontal pass (resize.cpp, VResizeLinear<uchar>).
            const int verticalTop = (ay * (top >> 4)) >> 16;
            const int verticalBottom = (by * (bottom >> 4)) >> 16;
            value = (verticalTop + verticalBottom + 2) >> 2;
        }
        size_t dst = size_t(image) * total + (a.nchw ? size_t(c) * a.crop_h * a.crop_w + i / 3 : i);
        y[dst] = (float(value) * (1.0f / 255.0f) - a.mean[c]) / a.stddev[c];
    }
}
__global__ void resizeMaps(const float* x, float* y, size_t total, int h, int w, int oh, int ow) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += size_t(blockDim.x) * gridDim.x)
        y[i] = interpolate(x + i / (size_t(oh) * ow) * h * w, h, w, int(i / ow % oh), int(i % ow), oh, ow);
}
__device__ int reflect(int x, int n) {
    if (n == 1) return 0;
    int period = 2 * n - 2;
    x = ((x % period) + period) % period;
    return x < n ? x : period - x;
}
__global__ void gaussian(const float* x, float* y, const float* weights, size_t total, int h, int w, int k,
                         bool vertical) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += size_t(blockDim.x) * gridDim.x) {
        int yy = int(i / w % h), xx = int(i % w);
        size_t base = i / (size_t(h) * w) * h * w;
        float sum = 0;
        for (int j = 0; j < k; ++j) {
            int sy = vertical ? reflect(yy + j - k / 2, h) : yy,
                sx = vertical ? xx : reflect(xx + j - k / 2, w);
            sum += x[base + sy * w + sx] * weights[j];
        }
        y[i] = sum;
    }
}
__global__ void scores(const float* d, float* p, float* maxima, int64_t* argmax, int* invalid, int locations,
                       int k, bool root) {
    __shared__ Best s[256];
    Best best{-1, INT_MAX};
    for (int i = threadIdx.x; i < locations; i += 256) {
        size_t q = (size_t(blockIdx.x) * locations + i) * k;
        for (int j = 0; j < k; ++j)
            if (!isfinite(d[q + j]) || d[q + j] < 0) atomicExch(invalid, 1);
        float v = root ? sqrtf(d[q]) : d[q];
        p[size_t(blockIdx.x) * locations + i] = v;
        best = better(best, {v, i});
    }
    s[threadIdx.x] = best;
    __syncthreads();
    for (int t = 128; t; t /= 2) {
        if (threadIdx.x < t) s[threadIdx.x] = better(s[threadIdx.x], s[threadIdx.x + t]);
        __syncthreads();
    }
    if (!threadIdx.x) {
        maxima[blockIdx.x] = float(s[0].value);
        argmax[blockIdx.x] = s[0].index;
    }
}
// Reject non-finite vectors before a greedy reduction can produce an invalid index.
__global__ void finiteFloats(const float* x, size_t n, int* invalid) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x)
        if (!isfinite(x[i])) atomicExch(invalid, 1);
}
