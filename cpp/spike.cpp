// FlashVSR C++ spike (ASCII only): can libtorch compile & run CUDA on this box?
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <iostream>

int main() {
    // The CUDA backend lives in torch_cuda.dll, and nothing in torch_cpu.lib
    // references it, so the linker drops it and no CUDA dispatch key gets
    // registered ("cuda=no, count=0"). Load it explicitly before probing.
    if (!LoadLibraryA("torch_cuda.dll")) {
        std::printf("LoadLibrary(torch_cuda.dll) failed: %lu\n", GetLastError());
        return 1;
    }

    std::cout << "torch " << TORCH_VERSION
              << "  cuda=" << (torch::cuda::is_available() ? "yes" : "no")
              << "  count=" << (int)torch::cuda::device_count() << std::endl;

    // Probe harder: actually allocate, and print the real error if it fails.
    try {
        auto probe = torch::zeros({8}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        std::cout << "probe alloc ok, sum=" << probe.sum().item<float>() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "probe failed: " << e.what() << std::endl;
        return 1;
    }

    auto opt = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    const int64_t M = 2304, K = 8960, N = 1536;
    auto a = torch::randn({M, K}, opt);
    auto b = torch::randn({N, K}, opt);

    for (int i = 0; i < 3; ++i) {
        auto c = torch::matmul(a, b.t());
        (void)c;
    }
    cudaDeviceSynchronize();

    const int n = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    torch::Tensor c;
    for (int i = 0; i < n; ++i) c = torch::matmul(a, b.t());
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n;
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    auto* alloc = c10::cuda::CUDACachingAllocator::get();
    size_t reserved = 0;
    if (alloc) reserved = alloc->getDeviceStats(0).reserved_bytes[0].current;
    std::printf("bf16 matmul %lldx%lldx%lld: %.3f ms  %.1f TFLOPS\n",
                (long long)M, (long long)K, (long long)N, ms, tflops);
    std::printf("vram: free %.0f MiB / total %.0f MiB ; torch reserved %.0f MiB\n",
                free_b / 1048576.0, total_b / 1048576.0, reserved / 1048576.0);
    return 0;
}
