// nvrtc_spike.cpp -- can we JIT CUDA C in-process on this box (no Toolkit)?
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "cuda_jit.h"

#include <cstdio>

static const char* kSrc = R"(
extern "C" __global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + 1.0f;
}
)";

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (!LoadLibraryA("torch_cuda.dll")) {
        std::printf("LoadLibrary(torch_cuda.dll) failed\n");
        return 1;
    }
    if (!torch::cuda::is_available()) {
        std::printf("CUDA not available\n");
        return 1;
    }

    // touch the device so torch creates the context our JIT will attach to
    auto t = torch::arange(0, 1024, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    cudaDeviceSynchronize();

    int maj = 0, min = 0;
    if (!cj::init() || !cj::arch(&maj, &min)) {
        std::printf("cuda_jit init failed\n");
        return 1;
    }
    std::printf("device sm_%d%d\n", maj, min);

    cj::Module m;
    if (!cj::compile(kSrc, "add_one", &m)) return 1;
    std::printf("compiled + loaded\n");

    float* p = t.data_ptr<float>();
    int n = (int)t.numel();
    void* args[] = {&p, &n};
    if (!cj::launch(m, (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1, 0, args, "add_one")) return 1;
    cj::driver().sync();

    auto got = t.slice(0, 0, 4).cpu();
    auto want = torch::arange(1, 5, torch::TensorOptions().dtype(torch::kFloat));
    const double diff = (got - want).abs().max().item<double>();
    std::printf("result[0..3] = %.1f %.1f %.1f %.1f   maxdiff %.1e  %s\n", got[0].item<double>(),
                got[1].item<double>(), got[2].item<double>(), got[3].item<double>(), diff,
                diff == 0 ? "OK" : "FAIL");
    return 0;
}
