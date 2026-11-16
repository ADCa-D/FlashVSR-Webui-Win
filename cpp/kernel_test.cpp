// kernel_test.cpp -- unit tests for the fused kernels, without loading the model.
#include <torch/torch.h>

#include "dit_kernels.h"

#include <cmath>
#include <cstdio>

static double md(const torch::Tensor& a, const torch::Tensor& b) {
    return (a.to(torch::kFloat).cpu() - b.to(torch::kFloat).cpu()).abs().max().item<double>();
}
static double mn(const torch::Tensor& a, const torch::Tensor& b) {
    return (a.to(torch::kFloat).cpu() - b.to(torch::kFloat).cpu()).abs().mean().item<double>();
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (!LoadLibraryA("torch_cuda.dll")) return 1;
    if (!torch::cuda::is_available()) return 1;
    const double eps = 1e-6;
    const int64_t S = 32, D = 1536, HD = 128, heads = 12;

    auto opt = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    torch::manual_seed(7);

    // ---------------- ln_modulate ----------------
    {
        auto x = torch::randn({1, S, D}, opt);
        auto shift = torch::randn({D}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)) * 0.1f;
        auto scale = torch::randn({D}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)) * 0.1f;
        auto got = dk::ln_modulate(x, shift, scale, eps);

        auto f = x.to(torch::kFloat);
        auto m = f.mean(-1, true);
        auto v = f.var(-1, false, true);
        auto y = ((f - m) * torch::rsqrt(v + eps)).to(torch::kBFloat16);
        auto ref = y * (1.0 + scale.to(torch::kBFloat16)) + shift.to(torch::kBFloat16);
        std::printf("ln_modulate : max %.3e  mean %.3e  %s\n", md(got, ref), mn(got, ref),
                    md(got, ref) < 5e-2 ? "OK" : "FAIL");
    }

    // ---------------- ln_modulate on DiT-like data (tiny values, big mean) ----
    {
        auto opt2 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        for (double base : {1e-4, 1e-2, 1.0}) {
            auto x = (torch::randn({1, S, D}, opt2) * 1e-2f + base).to(torch::kBFloat16);
            auto shift = torch::randn({D}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
            auto scale = torch::randn({D}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
            auto got = dk::ln_modulate(x, shift, scale, eps);
            auto f = x.to(torch::kFloat);
            auto m = f.mean(-1, true);
            auto v = f.var(-1, false, true);
            auto y = ((f - m) * torch::rsqrt(v + eps)).to(torch::kBFloat16);
            auto ref = y * (1.0 + scale.to(torch::kBFloat16)) + shift.to(torch::kBFloat16);
            std::printf("ln_mod@%g  : max %.3e  mean %.3e  %s\n", base, md(got, ref), mn(got, ref),
                        md(got, ref) < 5e-2 ? "OK" : "FAIL");
        }
    }

    // ---------------- ln_modulate at production row count (grid = 384) -------
    {
        const int64_t S2 = 384;
        auto x = (torch::randn({1, S2, D}, opt) * 1e-2f + 1e-4f).to(torch::kBFloat16);
        auto shift = torch::randn({D}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)) * 0.1f;
        auto scale = torch::randn({D}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)) * 0.1f;
        auto got = dk::ln_modulate(x, shift, scale, eps);
        auto f = x.to(torch::kFloat);
        auto m = f.mean(-1, true);
        auto v = f.var(-1, false, true);
        auto y = ((f - m) * torch::rsqrt(v + eps)).to(torch::kBFloat16);
        auto ref = y * (1.0 + scale.to(torch::kBFloat16)) + shift.to(torch::kBFloat16);
        std::printf("ln_mod S=384: max %.3e  mean %.3e  %s\n", md(got, ref), mn(got, ref),
                    md(got, ref) < 5e-2 ? "OK" : "FAIL");
    }

    // ---------------- ln_affine ----------------
    {
        auto x = torch::randn({1, S, D}, opt);
        auto w = (torch::randn({D}, opt) * 0.1f);
        auto b = (torch::randn({D}, opt) * 0.1f);
        auto got = dk::ln_affine(x, w, b, eps);

        auto f = x.to(torch::kFloat);
        auto m = f.mean(-1, true);
        auto v = f.var(-1, false, true);
        auto y = ((f - m) * torch::rsqrt(v + eps)).to(torch::kBFloat16);
        auto ref = y * w + b;
        std::printf("ln_affine   : max %.3e  mean %.3e  %s\n", md(got, ref), mn(got, ref),
                    md(got, ref) < 5e-2 ? "OK" : "FAIL");
    }

    // ---------------- rms_only ----------------
    {
        auto q = torch::randn({1, S, D}, opt);
        auto w = torch::randn({D}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)) * 0.1f;
        auto got = dk::rms_only(q, w, eps);

        auto f = q.to(torch::kFloat);
        auto rms = f * torch::rsqrt(f.pow(2).mean(-1, true) + eps);
        auto ref = rms.to(torch::kBFloat16) * w.to(torch::kBFloat16);
        std::printf("rms_only    : max %.3e  mean %.3e  %s\n", md(got, ref), mn(got, ref),
                    md(got, ref) < 5e-2 ? "OK" : "FAIL");
    }

    // ---------------- gate_res ----------------
    {
        auto x = torch::randn({1, S, D}, opt);
        auto d = torch::randn({1, S, D}, opt);
        auto g = torch::randn({D}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        auto ref = x + g.to(torch::kBFloat16) * d;
        dk::gate_res_(x, g, d);
        std::printf("gate_res    : max %.3e  mean %.3e  %s\n", md(x, ref), mn(x, ref),
                    md(x, ref) < 5e-2 ? "OK" : "FAIL");
    }

    // ---------------- rms_rope (vs double-precision reference) ----------------
    {
        auto q = torch::randn({1, S, D}, opt);
        auto w = torch::randn({D}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)) * 0.1f;

        // build cos/sin for the same shape: S tokens, one row each
        std::vector<double> cv((size_t)S * HD / 2), sv((size_t)S * HD / 2);
        for (int64_t s = 0; s < S; ++s)
            for (int64_t j = 0; j < HD / 2; ++j) {
                const double a = (double)s * (0.001 * (double)(j + 1));
                cv[(size_t)(s * (HD / 2) + j)] = std::cos(a);
                sv[(size_t)(s * (HD / 2) + j)] = std::sin(a);
            }
        auto opt64 = torch::TensorOptions().dtype(torch::kFloat64);
        auto cosF = torch::from_blob(cv.data(), {S, HD / 2}, opt64).clone().to(torch::kFloat).cuda();
        auto sinF = torch::from_blob(sv.data(), {S, HD / 2}, opt64).clone().to(torch::kFloat).cuda();
        auto got = dk::rms_rope(q, w, cosF, sinF, HD, eps);

        // reference in float64
        auto f = q.to(torch::kFloat);
        auto rms = (f * torch::rsqrt(f.pow(2).mean(-1, true) + eps)).to(torch::kBFloat16);
        auto scaled = rms * w.to(torch::kBFloat16);
        auto xr = scaled.view({1, S, heads, HD / 2, 2}).to(torch::kFloat64);
        auto re = xr.select(-1, 0), im = xr.select(-1, 1);
        auto c = cosF.to(torch::kFloat64).unsqueeze(1), s_ = sinF.to(torch::kFloat64).unsqueeze(1);
        auto o0 = re * c - im * s_;
        auto o1 = re * s_ + im * c;
        auto ref = torch::stack({o0, o1}, -1).reshape({1, S, D}).to(torch::kBFloat16);
        std::printf("rms_rope    : max %.3e  mean %.3e  %s\n", md(got, ref), mn(got, ref),
                    md(got, ref) < 5e-2 ? "OK" : "FAIL");
    }
    return 0;
}
