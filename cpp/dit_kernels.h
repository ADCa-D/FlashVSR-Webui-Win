// dit_kernels.h -- fused elementwise kernels for the DiT block (NVRTC JIT).
//
// The ATen version spends ~80-100 launches per layer, and on WDDM each one is
// ~80 us even when the op itself is trivial. These four kernels collapse the
// whole elementwise chain:
//
//   layernorm(1536) + modulate            -> ln_modulate            (6 ops -> 1)
//   layernorm(1536, affine)               -> ln_affine              (7 ops -> 1)
//   rmsnorm(1536) * w + 3D rope           -> rms_rope   x2          (32 ops -> 2)
//   residual + gate*delta                 -> gate_res                (2 ops -> 1)
//   residual + delta                      -> add2                   (1 op  -> 1)
#include "cuda_jit.h"

#include <torch/torch.h>

namespace dk {

static const char* kSrc = R"(
__device__ __forceinline__ float b2f(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f2b(float f) {
    unsigned int u = __float_as_uint(f);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
    return (unsigned short)u;
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[32];
    for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
        for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}

// out = (layernorm(x, no affine) * (1 + scale) + shift)
extern "C" __global__ void ln_modulate(const unsigned short* __restrict__ x,
                                       const float* __restrict__ shift,
                                       const float* __restrict__ scale,
                                       unsigned short* __restrict__ out,
                                       int D, float eps)
{
    const int row = blockIdx.x;
    const unsigned short* xr = x + (size_t)row * D;
    unsigned short* orow = out + (size_t)row * D;
    float s1 = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s1 += b2f(xr[i]);
    const float mean = block_sum(s1) / (float)D;
    // variance in a second pass: x_patch rows are near-constant at ~1e-4, where
    // sum(x^2)/D - mean^2 loses every significant digit in float32
    float s2 = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float d = b2f(xr[i]) - mean;
        s2 += d * d;
    }
    const float inv = rsqrtf(block_sum(s2) / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float y = (b2f(xr[i]) - mean) * inv;
        orow[i] = f2b(y * (1.f + scale[i]) + shift[i]);
    }
}

// out = layernorm(x, affine) -- used by norm3 before cross-attention
extern "C" __global__ void ln_affine(const unsigned short* __restrict__ x,
                                     const unsigned short* __restrict__ w,
                                     const unsigned short* __restrict__ b,
                                     unsigned short* __restrict__ out,
                                     int D, float eps)
{
    const int row = blockIdx.x;
    const unsigned short* xr = x + (size_t)row * D;
    unsigned short* orow = out + (size_t)row * D;
    float s1 = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s1 += b2f(xr[i]);
    const float mean = block_sum(s1) / (float)D;
    float s2 = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float d = b2f(xr[i]) - mean;
        s2 += d * d;
    }
    const float inv = rsqrtf(block_sum(s2) / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float y = (b2f(xr[i]) - mean) * inv;
        orow[i] = f2b(y * b2f(w[i]) + b2f(b[i]));
    }
}

// out = rope(rmsnorm(q, weight))  -- rms over the full row, rope per head pair.
// cosT/sinT: [S, HD/2] float32, one row per token (3D grid frequencies).
extern "C" __global__ void rms_rope(const unsigned short* __restrict__ q,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ cosT,
                                    const float* __restrict__ sinT,
                                    unsigned short* __restrict__ out,
                                    int D, int HD, float eps)
{
    const int row = blockIdx.x;
    const unsigned short* qr = q + (size_t)row * D;
    unsigned short* orow = out + (size_t)row * D;
    float s2 = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = b2f(qr[i]);
        s2 += v * v;
    }
    s2 = block_sum(s2);
    const float inv = rsqrtf(s2 / (float)D + eps);

    const int half = HD >> 1;
    const float* c = cosT + (size_t)row * half;
    const float* s = sinT + (size_t)row * half;
    const int pairs = D >> 1;                     // 768 for D=1536
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int h = p / half;
        const int k = p - h * half;
        const int i0 = h * HD + 2 * k;
        const int i1 = i0 + 1;
        const float a = b2f(qr[i0]) * weight[i0] * inv;
        const float b = b2f(qr[i1]) * weight[i1] * inv;
        const float co = c[k], si = s[k];
        orow[i0] = f2b(a * co - b * si);
        orow[i1] = f2b(a * si + b * co);
    }
}

// out = rmsnorm(q, weight)  -- cross-attention q, no rope
extern "C" __global__ void rms_only(const unsigned short* __restrict__ q,
                                    const float* __restrict__ weight,
                                    unsigned short* __restrict__ out,
                                    int D, float eps)
{
    const int row = blockIdx.x;
    const unsigned short* qr = q + (size_t)row * D;
    unsigned short* orow = out + (size_t)row * D;
    float s2 = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = b2f(qr[i]);
        s2 += v * v;
    }
    s2 = block_sum(s2);
    const float inv = rsqrtf(s2 / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        orow[i] = f2b(b2f(qr[i]) * weight[i] * inv);
}

// x += gate * delta   (gate broadcast over the last dim)
extern "C" __global__ void gate_res(unsigned short* __restrict__ x,
                                    const float* __restrict__ gate,
                                    const unsigned short* __restrict__ delta,
                                    int D, long long n)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const int d = (int)(i % D);
        x[i] = f2b(b2f(x[i]) + gate[d] * b2f(delta[i]));
    }
}

// x += y
extern "C" __global__ void add2(unsigned short* __restrict__ x,
                                const unsigned short* __restrict__ y, long long n)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        x[i] = f2b(b2f(x[i]) + b2f(y[i]));
}
)";

struct Lib {
    cj::Module mod;
    void* ln_modulate = nullptr;
    void* ln_affine = nullptr;
    void* rms_rope = nullptr;
    void* rms_only = nullptr;
    void* gate_res = nullptr;
    void* add2 = nullptr;
    bool ok = false;
};

inline Lib& lib() {
    static Lib L = [] {
        Lib l;
        if (!cj::compile(kSrc, "ln_modulate", &l.mod, /*verbose=*/false)) return l;
        auto get = [&](const char* n) -> void* {
            void* f = nullptr;
            cj::driver().module_get_fn((cj::CUfunction*)&f, l.mod.mod, n);
            if (!f) std::printf("[dk] missing kernel %s\n", n);
            return f;
        };
        l.ln_modulate = get("ln_modulate");
        l.ln_affine = get("ln_affine");
        l.rms_rope = get("rms_rope");
        l.rms_only = get("rms_only");
        l.gate_res = get("gate_res");
        l.add2 = get("add2");
        l.ok = l.ln_modulate && l.ln_affine && l.rms_rope && l.rms_only && l.gate_res && l.add2;
        return l;
    }();
    return L;
}

// ---- launch helpers -------------------------------------------------------
inline bool call(void* fn, unsigned gx, unsigned bx, void** args, const char* tag) {
    cj::Module m;
    m.mod = lib().mod.mod;
    m.fn = fn;
    return cj::launch(m, gx, 1, 1, bx, 1, 1, 0, args, tag);
}

// x: bf16 [1,S,D]; shift/scale: float [D]; returns bf16 [1,S,D]
inline torch::Tensor ln_modulate(const torch::Tensor& x, const torch::Tensor& shift,
                                 const torch::Tensor& scale, double eps) {
    const auto xc = x.contiguous();
    auto out = torch::empty(x.sizes(), x.options());
    const int S = (int)xc.size(1), D = (int)xc.size(2);
    void* xp = xc.data_ptr();
    void* shp = shift.data_ptr();
    void* scp = scale.data_ptr();
    void* op = out.data_ptr();
    int iD = D;
    float feps = (float)eps;
    void* args[] = {&xp, &shp, &scp, &op, &iD, &feps};
    call(lib().ln_modulate, (unsigned)S, 256, args, "ln_modulate");
    return out;
}

inline torch::Tensor ln_affine(const torch::Tensor& x, const torch::Tensor& w,
                               const torch::Tensor& b, double eps) {
    const auto xc = x.contiguous();
    auto out = torch::empty(x.sizes(), x.options());
    const int S = (int)xc.size(1), D = (int)xc.size(2);
    void* xp = xc.data_ptr();
    void* wp = w.data_ptr();
    void* bp = b.data_ptr();
    void* op = out.data_ptr();
    int iD = D;
    float feps = (float)eps;
    void* args[] = {&xp, &wp, &bp, &op, &iD, &feps};
    call(lib().ln_affine, (unsigned)S, 256, args, "ln_affine");
    return out;
}

// q: bf16 [1,S,D]; weight: float [D]; cos/sin: float [S, HD/2]
inline torch::Tensor rms_rope(const torch::Tensor& q, const torch::Tensor& weight,
                              const torch::Tensor& cosT, const torch::Tensor& sinT,
                              int64_t head_dim, double eps) {
    const auto xc = q.contiguous();
    auto out = torch::empty(q.sizes(), q.options());
    const int S = (int)xc.size(1), D = (int)xc.size(2);
    void* qp = xc.data_ptr();
    void* wp = weight.data_ptr();
    void* cp = cosT.data_ptr();
    void* sp = sinT.data_ptr();
    void* op = out.data_ptr();
    int iD = D, iHD = (int)head_dim;
    float feps = (float)eps;
    void* args[] = {&qp, &wp, &cp, &sp, &op, &iD, &iHD, &feps};
    call(lib().rms_rope, (unsigned)S, 256, args, "rms_rope");
    return out;
}

inline torch::Tensor rms_only(const torch::Tensor& q, const torch::Tensor& weight, double eps) {
    const auto xc = q.contiguous();
    auto out = torch::empty(q.sizes(), q.options());
    const int S = (int)xc.size(1), D = (int)xc.size(2);
    void* qp = xc.data_ptr();
    void* wp = weight.data_ptr();
    void* op = out.data_ptr();
    int iD = D;
    float feps = (float)eps;
    void* args[] = {&qp, &wp, &op, &iD, &feps};
    call(lib().rms_only, (unsigned)S, 256, args, "rms_only");
    return out;
}

// x += gate * delta   (in place on x)
inline void gate_res_(torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& delta) {
    TORCH_CHECK(x.is_contiguous(), "gate_res_ needs contiguous x");
    const int D = (int)x.size(-1);
    long long n = x.numel();
    void* xp = x.data_ptr();
    void* gp = gate.data_ptr();
    void* dp = delta.data_ptr();
    int iD = D;
    void* args[] = {&xp, &gp, &dp, &iD, &n};
    call(lib().gate_res, 256, 256, args, "gate_res");
}

inline void add2_(torch::Tensor& x, const torch::Tensor& y) {
    TORCH_CHECK(x.is_contiguous(), "add2_ needs contiguous x");
    long long n = x.numel();
    void* xp = x.data_ptr();
    void* yp = y.data_ptr();
    void* args[] = {&xp, &yp, &n};
    call(lib().add2, 256, 256, args, "add2");
}

}  // namespace dk
