// attn_jit.h -- fused block-sparse self-attention, compiled by NVRTC at startup
// and launched through the driver API.
//
// Replaces the ATen version that did ~280 launches per layer (matmul/max/exp/
// sum/scatter per 128x64 block, ~100 us each on WDDM -> 840 ms for 30 layers).
// Here it is ONE launch per (q-block, head, batch): grid = (Lq, H, B).
//
// Layout assumptions (the ones the DiT actually uses):
//   q/k/v : bf16 [B, S, H*D] token-major, head_dim D = 128
//   mask  : fp32 [B, H, Lq, Lk] block mask, 128 query tokens x 64 kv tokens
#include "cuda_jit.h"

#include <torch/torch.h>

namespace attn {

static const char* kSrc = R"(
// head_dim is a compile-time constant on purpose: with a runtime D the arrays
// qr[]/s[] land in local memory and the whole kernel collapses to ~10x slower.
#define HD 128
#define BM 128
#define BN 64

__device__ __forceinline__ float b2f(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f2b(float f) {
    unsigned int u = __float_as_uint(f);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
    return (unsigned short)u;
}

extern "C" __global__ void sparse_attn(
    const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    const float* __restrict__ mask,
    unsigned short* __restrict__ out,
    int Lq, int Lk, int S, int Skv, int H, int qstride, int kvstride, float scale)
{
    // dynamic shared: acc[BM][HD+1] floats, then one bf16 tile [BN][HD]
    extern __shared__ char smem[];
    float* accs = reinterpret_cast<float*>(smem);
    unsigned short* tile = reinterpret_cast<unsigned short*>(smem + BM * (HD + 1) * 4);

    const int i = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int r = threadIdx.x;
    const int row = i * BM + r;
    float* accrow = accs + r * (HD + 1);

    float qr[HD];
    {
        const unsigned short* qp = q + ((size_t)b * S + row) * qstride + h * HD;
#pragma unroll
        for (int d = 0; d < HD; ++d) {
            qr[d] = b2f(qp[d]);
            accrow[d] = 0.f;
        }
    }

    float m = -1e30f, l = 0.f;
    const size_t mask_base = ((size_t)b * H + h) * Lq * Lk + (size_t)i * Lk;

    for (int j = 0; j < Lk; ++j) {
        if (!(mask[mask_base + j] > 0.5f)) continue;

        __syncthreads();
        for (int idx = threadIdx.x; idx < BN * HD; idx += BM * 1) {
            const int t = idx / HD, d = idx - t * HD;
            tile[idx] = k[((size_t)b * Skv + j * BN + t) * kvstride + h * HD + d];
        }
        __syncthreads();

        float s[BN];
        float bmax = -1e30f;
#pragma unroll 1
        for (int t = 0; t < BN; ++t) {
            const unsigned short* kt = tile + t * HD;
            float dot = 0.f;
#pragma unroll
            for (int d = 0; d < HD; ++d) dot = fmaf(qr[d], b2f(kt[d]), dot);
            dot *= scale;
            s[t] = dot;
            bmax = fmaxf(bmax, dot);
        }

        const float new_m = fmaxf(m, bmax);
        const float alpha = (m > -1e29f) ? __expf(m - new_m) : 0.f;
        float lsum = 0.f;
#pragma unroll
        for (int t = 0; t < BN; ++t) {
            s[t] = __expf(s[t] - new_m);
            lsum += s[t];
        }
        l = l * alpha + lsum;
        m = new_m;
#pragma unroll
        for (int d = 0; d < HD; ++d) accrow[d] *= alpha;

        __syncthreads();
        for (int idx = threadIdx.x; idx < BN * HD; idx += BM * 1) {
            const int t = idx / HD, d = idx - t * HD;
            tile[idx] = v[((size_t)b * Skv + j * BN + t) * kvstride + h * HD + d];
        }
        __syncthreads();

#pragma unroll
        for (int d = 0; d < HD; ++d) {
            float a = accrow[d];
#pragma unroll 1
            for (int t = 0; t < BN; ++t) a = fmaf(s[t], b2f(tile[t * HD + d]), a);
            accrow[d] = a;
        }
    }

    const float inv = (l > 1e-30f) ? (1.f / l) : 0.f;
    unsigned short* op = out + ((size_t)b * S + row) * qstride + h * HD;
#pragma unroll
    for (int d = 0; d < HD; ++d) op[d] = f2b(accrow[d] * inv);
}
)";

// Compile once, reuse. Returns false (with a message) if NVRTC fails.
inline std::vector<cj::Module>& modules() {
    static std::vector<cj::Module> v;
    return v;
}

inline bool ready() {
    static int state = -1;
    if (state >= 0) return state == 1;
    cj::Module m;
    if (!cj::compile(kSrc, "sparse_attn", &m, /*verbose=*/false)) {
        state = 0;
        return false;
    }
    // dynamic shared: acc[128][D+1] + tile[64][D] with D=128 -> ~80.5 KB
    const int attr = 8;   // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    using PFN = int (*)(void*, int, int);
    auto p = (PFN)GetProcAddress(GetModuleHandleA("nvcuda.dll"), "cuFuncSetAttribute");
    if (p) p(m.fn, attr, 96 * 1024);
    modules().push_back(m);
    state = 1;
    return true;
}

// q/k/v: bf16 CUDA [B,S,H*D] contiguous; mask: fp32 [B,H,Lq,Lk]; returns bf16 [B,S,H*D].
inline torch::Tensor forward(const torch::Tensor& q, const torch::Tensor& k,
                             const torch::Tensor& v, const torch::Tensor& mask, int64_t heads,
                             int64_t head_dim) {
    const auto B = q.size(0), S = q.size(1);
    const auto Skv = k.size(1);
    const auto Lq = S / 128, Lk = Skv / 64;
    auto out = torch::empty_like(q);
    if (!ready()) {
        std::printf("[attn] JIT unavailable, falling back\n");
        return out;
    }

    const int64_t qstride = heads * head_dim;
    int iLq = (int)Lq, iLk = (int)Lk, iS = (int)S, iSkv = (int)Skv, iH = (int)heads,
        iqs = (int)qstride, iks = (int)qstride;
    float scale = 1.0f / std::sqrt((float)head_dim);

    void* qp = q.data_ptr();
    void* kp = k.data_ptr();
    void* vp = v.data_ptr();
    void* mp = mask.data_ptr();
    void* op = out.data_ptr();
    void* args[] = {&qp, &kp, &vp, &mp, &op, &iLq, &iLk, &iS, &iSkv, &iH, &iqs, &iks, &scale};

    const size_t shmem = (size_t)(128 * (head_dim + 1) * 4 + 64 * head_dim * 2);
    auto& m = modules().back();
    if (!cj::launch(m, (unsigned)Lq, (unsigned)heads, (unsigned)B, 128, 1, 1, (unsigned)shmem,
                    args, "sparse_attn"))
        std::printf("[attn] launch failed\n");
    return out;
}

}  // namespace attn
