// dit.cpp -- FlashVSR DiT forward in C++ (libtorch/ATen), stage 1 of the port.
//
// Goal of this stage: reproduce the Python forward exactly (same kernels, same
// op order) so that any later hand-written CUDA kernel has a trustworthy
// reference. Compares against work/dit_ref/ref.safetensors, dumped by
// bench/dump_dit_ref.py from the real production path (model_fn_wan_video).
//
// Deliberately NOT here yet: LQ_proj_in latents, kv-cache streaming across
// windows, block-sparse mask *generation* (the mask is read from the dump so
// attention math and mask selection can be debugged separately).
#include <torch/torch.h>

#include "safetensors.h"

#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr int64_t kDim = 1536;
constexpr int64_t kHeads = 12;
constexpr int64_t kFfn = 8960;
constexpr int64_t kLayers = 30;
constexpr int64_t kFreqDim = 256;
constexpr double kEps = 1e-6;
constexpr int64_t kPatchF = 1, kPatchH = 2, kPatchW = 2;
constexpr int64_t kInDim = 16, kOutDim = 16;

using Tensor = torch::Tensor;

Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) {
    return torch::linear(x, w, b);
}

// RMSNorm: norm in float32, cast back, then scale (matches the Python module)
Tensor rms_norm(const Tensor& x, const Tensor& w, double eps) {
    auto f = x.to(torch::kFloat);
    auto n = f * torch::rsqrt(f.pow(2).mean(-1, true) + eps);
    return n.to(x.dtype()) * w;
}

// nn.LayerNorm(eps, elementwise_affine=False)
Tensor layer_norm_plain(const Tensor& x, double eps) {
    auto f = x.to(torch::kFloat);
    auto m = f.mean(-1, true);
    auto v = f.var(-1, /*unbiased=*/false, /*keepdim=*/true);
    return ((f - m) * torch::rsqrt(v + eps)).to(x.dtype());
}

// nn.LayerNorm(eps, elementwise_affine=True)
Tensor layer_norm_affine(const Tensor& x, const Tensor& w, const Tensor& b, double eps) {
    auto f = x.to(torch::kFloat);
    auto m = f.mean(-1, true);
    auto v = f.var(-1, false, true);
    auto y = ((f - m) * torch::rsqrt(v + eps)).to(x.dtype());
    return y * w + b;
}

Tensor modulate(const Tensor& x, const Tensor& shift, const Tensor& scale) {
    return x * (1.0 + scale) + shift;
}

// ---- RoPE: 3D complex frequencies, computed in float64 like the reference ----
struct RopeTable {
    Tensor cos_t;   // [end, head_dim/2] float64
    Tensor sin_t;
};

// RoPE table for the actual 3D grid: token s = ((f)*H + h)*W + w, and the 64
// complex freqs are split by axis (f: 0..21, h: 22..42, w: 43..63). Each axis
// advances with ITS OWN index, not with the flat token index -- getting this
// wrong leaves s=0 exact and everything else rotated by the wrong angle.
RopeTable build_rope(int64_t head_dim, int64_t F, int64_t H, int64_t W,
                     c10::DeviceType dev, double theta = 10000.0) {
    const int64_t dim_f = head_dim - 2 * (head_dim / 3);
    const int64_t dim_h = head_dim / 3;
    const int64_t dim_w = head_dim / 3;
    auto axis = [&](int64_t dim) {
        std::vector<double> v;
        for (int64_t i = 0; i < dim / 2; ++i) {
            double e = (double)(2 * i) / (double)dim;
            v.push_back(1.0 / std::pow(theta, e));
        }
        return v;
    };
    const auto ff = axis(dim_f), hh = axis(dim_h), ww = axis(dim_w);

    const int64_t half = head_dim / 2;
    const int64_t L = F * H * W;
    std::vector<double> cosv((size_t)(L * half)), sinv((size_t)(L * half));
    for (int64_t fi = 0; fi < F; ++fi) {
        for (int64_t hi = 0; hi < H; ++hi) {
            for (int64_t wi = 0; wi < W; ++wi) {
                const int64_t s = (fi * H + hi) * W + wi;
                int64_t j = 0;
                auto put = [&](double pos, const std::vector<double>& f) {
                    for (double fr : f) {
                        double a = pos * fr;
                        cosv[(size_t)(s * half + j)] = std::cos(a);
                        sinv[(size_t)(s * half + j)] = std::sin(a);
                        ++j;
                    }
                };
                put((double)fi, ff);
                put((double)hi, hh);
                put((double)wi, ww);
            }
        }
    }
    auto opt = torch::TensorOptions().dtype(torch::kFloat64);   // build on host, then move
    return {torch::from_blob(cosv.data(), {L, half}, opt).clone().to(dev),
            torch::from_blob(sinv.data(), {L, half}, opt).clone().to(dev)};
}

Tensor rope_apply(const Tensor& x, const RopeTable& rope, int64_t heads) {
    const auto B = x.size(0), L = x.size(1), D = x.size(2), hd = D / heads;
    auto xr = x.view({B, L, heads, hd / 2, 2}).to(torch::kFloat64);
    auto re = xr.select(-1, 0);
    auto im = xr.select(-1, 1);
    auto c = rope.cos_t.narrow(0, 0, L).unsqueeze(1);   // [L,1,hd/2]
    auto s = rope.sin_t.narrow(0, 0, L).unsqueeze(1);
    auto o0 = re * c - im * s;
    auto o1 = re * s + im * c;
    return torch::stack({o0, o1}, -1).reshape({B, L, D}).to(x.dtype());
}

// ---- WindowPartition3D (2,8,8) --------------------------------------------
constexpr int64_t kWinF = 2, kWinH = 8, kWinW = 8;

Tensor partition3d(const Tensor& x, int64_t f, int64_t h, int64_t w) {
    const auto B = x.size(0), D = x.size(2);   // input is [B, f*h*w, D]
    auto v = x.view({B, f, h, w, D});
    v = v.view({B, f / kWinF, kWinF, h / kWinH, kWinH, w / kWinW, kWinW, D});
    v = v.permute({0, 1, 3, 5, 2, 4, 6, 7}).contiguous();
    return v.view({-1, kWinF * kWinH * kWinW, D});
}

Tensor reverse3d(const Tensor& win, int64_t f, int64_t h, int64_t w) {
    const int64_t nf = f / kWinF, nh = h / kWinH, nw = w / kWinW;
    const int64_t B = win.size(0) / (nf * nh * nw);
    auto x = win.view({B, nf, nh, nw, kWinF, kWinH, kWinW, -1});
    x = x.permute({0, 1, 4, 2, 5, 3, 6, 7}).contiguous();
    return x.view({B, f * h * w, -1});
}

// ---- attention -------------------------------------------------------------
// mask: optional bool [B,H,S,Sq] (true = attend). Dense softmax over the allowed
// columns; this is the mathematically equivalent form of sage's masked kernel.
Tensor attention(const Tensor& q_in, const Tensor& k_in, const Tensor& v_in, int64_t heads,
                 const Tensor* mask) {
    // compute in float32: sage accumulates qk in fp32 and softmaxes in fp32
    auto q = q_in.to(torch::kFloat), k = k_in.to(torch::kFloat), v = v_in.to(torch::kFloat);
    const auto B = q.size(0), S = q.size(1), D = q.size(2), hd = D / heads;
    auto qh = q.view({B, S, heads, hd}).transpose(1, 2);
    auto kh = k.view({B, k.size(1), heads, hd}).transpose(1, 2);
    auto vh = v.view({B, v.size(1), heads, hd}).transpose(1, 2);
    auto sc = torch::matmul(qh, kh.transpose(2, 3)) * (1.0 / std::sqrt((double)hd));
    if (mask != nullptr) {
        const auto filled = mask->logical_not();
        const auto empty_rows = filled.to(torch::kInt).sum(-1).eq(mask->size(3)).sum().item<int64_t>();
        if (empty_rows > 0)
            std::printf("   [warn] %lld fully-masked rows (block mask has no allowed kv block)\n",
                        (long long)empty_rows);
        sc = sc.masked_fill(filled, -std::numeric_limits<double>::infinity());
    }
    auto p = torch::softmax(sc, -1);
    p = torch::nan_to_num(p, 0.0);
    return torch::matmul(p, vh).transpose(1, 2).reshape({B, S, D}).to(q_in.dtype());
}

// expand a block mask [B,H,S/128,S/64] to per-token [B,H,S,S] (true = attend)
Tensor expand_mask(const Tensor& blk) {
    const auto B = blk.size(0), H = blk.size(1), Lq = blk.size(2), Lk = blk.size(3);
    constexpr int64_t BM = 128, BN = 64;
    auto m = blk.gt(0.5).view({B, H, Lq, 1, Lk, 1}).expand({B, H, Lq, BM, Lk, BN});
    return m.reshape({B, H, Lq * BM, Lk * BN}).contiguous();
}

double maxdiff(const Tensor& a, const Tensor& b) {
    if (a.sizes() != b.sizes()) {
        std::printf("   shape mismatch:");
        for (auto s : a.sizes()) std::printf(" %lld", (long long)s);
        std::printf("  vs");
        for (auto s : b.sizes()) std::printf(" %lld", (long long)s);
        std::printf("\n");
        return -1;
    }
    return (a.to(torch::kFloat).cpu() - b.to(torch::kFloat).cpu()).abs().max().item<double>();
}

double meandiff(const Tensor& a, const Tensor& b) {
    return (a.to(torch::kFloat).cpu() - b.to(torch::kFloat).cpu()).abs().mean().item<double>();
}

void report(const char* tag, const Tensor& got, const Tensor& ref) {
    const double mx = maxdiff(got, ref);
    const double mn = meandiff(got, ref);
    std::printf("  %-9s maxdiff %.3e   meandiff %.3e   %s\n", tag, mx, mn,
                mx < 1e-2 ? "OK" : "**FAIL**");
}

}  // namespace

static int run_main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (!LoadLibraryA("torch_cuda.dll")) {
        std::printf("LoadLibrary(torch_cuda.dll) failed: %lu\n", GetLastError());
        return 1;
    }
    if (!torch::cuda::is_available()) {
        std::printf("CUDA not available\n");
        return 1;
    }
    auto t0 = std::chrono::high_resolution_clock::now();

    const char* ckpt =
        "D:\\ComfyUI_windows_portable\\ComfyUI\\models\\FlashVSR-v1.1"
        "\\diffusion_pytorch_model_streaming_dmd.safetensors";
    const char* refp = "work\\dit_ref\\ref.safetensors";

    // ref tensors were dumped as f32; converting back to bf16 is lossless
    auto ref = st::load_all(refp, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    if (ref.empty()) {
        std::printf("failed to load %s (run bench/dump_dit_ref.py first)\n", refp);
        return 1;
    }
    auto W = st::load_all(ckpt, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    if (W.empty()) {
        std::printf("failed to load %s\n", ckpt);
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("loaded %zu ref tensors + %zu weights in %.2f s\n", ref.size(), W.size(),
                std::chrono::duration<double>(t1 - t0).count());

    auto w = [&](const std::string& k) -> Tensor& {
        auto it = W.find(k);
        if (it == W.end()) {
            std::printf("missing weight: %s\n", k.c_str());
            std::exit(2);
        }
        return it->second;
    };

    // ---- inputs -----------------------------------------------------------
    auto x = ref.at("x").to(torch::kBFloat16);                    // [1,16,6,16,16]
    const int64_t f = x.size(2), hh = x.size(3), ww = x.size(4);
    const int64_t L = f * hh * ww;
    auto ts = ref.at("timestep").to(torch::kBFloat16);             // [1]
    // NOTE: the dumped context is the *raw* text-encoder output (4096-d); the
    // pipeline runs it through text_embedding inside reinit_cross_kv().
    auto ctx_raw = ref.at("context").to(torch::kBFloat16);        // [1,512,4096]
    auto ctx = linear(ctx_raw, w("text_embedding.0.weight"), w("text_embedding.0.bias"));
    ctx = torch::gelu(ctx, "tanh");
    ctx = linear(ctx, w("text_embedding.2.weight"), w("text_embedding.2.bias"));   // [1,512,dim]

    // ---- time embedding ---------------------------------------------------
    // sinusoidal_embedding_1d in float64, cast back to the position dtype
    Tensor sinus;
    {
        auto pos = ts.to(torch::kFloat64);
        auto opt64 = torch::TensorOptions().dtype(torch::kFloat64).device(ts.device());
        auto idx = torch::arange(0, kFreqDim / 2, opt64);
        // Python: freqs = 10000 ** (-arange(dim//2) / (dim//2))
        auto freqs = torch::pow(10000.0, -idx / (double)(kFreqDim / 2));
        auto ang = pos.unsqueeze(1) * freqs.unsqueeze(0);          // [1, freq/2]
        sinus = torch::cat({torch::cos(ang), torch::sin(ang)}, 1).to(ts.dtype());
    }
    auto t = linear(sinus, w("time_embedding.0.weight"), w("time_embedding.0.bias"));
    t = torch::silu(t);
    t = linear(t, w("time_embedding.2.weight"), w("time_embedding.2.bias"));   // [1,dim]
    auto t_mod = torch::silu(t);
    t_mod = linear(t_mod, w("time_projection.1.weight"), w("time_projection.1.bias"));
    t_mod = t_mod.unflatten(1, {6, kDim});                                     // [1,6,dim]

    report("t", t, ref.at("t"));
    report("t_mod", t_mod, ref.at("t_mod"));

    // ---- patch embedding --------------------------------------------------
    auto xp = torch::conv3d(x, w("patch_embedding.weight"), w("patch_embedding.bias"),
                            {kPatchF, kPatchH, kPatchW}, {0, 0, 0});           // [1,dim,f',h',w']
    const int64_t pf = xp.size(2), ph = xp.size(3), pw = xp.size(4);
    auto seq = xp.permute({0, 2, 3, 4, 1}).reshape({1, pf * ph * pw, kDim});
    std::printf("grid (f,h,w) = (%lld,%lld,%lld)  S = %lld\n", (long long)pf, (long long)ph,
                (long long)pw, (long long)seq.size(1));
    report("x_patch", seq, ref.at("x_patch"));

    // ---- cross-attn kv cache (from the text context) -----------------------
    auto rope = build_rope(kDim / kHeads, pf, ph, pw, x.device().type());
    {
        auto cc = rope.cos_t.cpu(), ss = rope.sin_t.cpu();
        auto ca = cc.accessor<double, 2>();
        auto sa = ss.accessor<double, 2>();
        std::printf("rope cos[1][0..3] %.8f %.8f %.8f %.8f\n", ca[1][0], ca[1][1], ca[1][2], ca[1][3]);
        std::printf("rope sin[1][0..3] %.8f %.8f %.8f %.8f\n", sa[1][0], sa[1][1], sa[1][2], sa[1][3]);
        std::printf("rope cos[0][0]    %.8f   half=%lld\n", ca[0][0], (long long)cc.size(1));
    }
    auto blk_mask = expand_mask(ref.at("mask"));   // [B,H,S,S]

    auto run = [&](Tensor cur, const char* tag) {
        for (int64_t i = 0; i < kLayers; ++i) {
            const std::string p = "blocks." + std::to_string(i) + ".";
            auto mods = (w(p + "modulation") + t_mod).chunk(6, 1);
            const Tensor& shift_msa = mods[0];
            const Tensor& scale_msa = mods[1];
            const Tensor& gate_msa = mods[2];
            const Tensor& shift_mlp = mods[3];
            const Tensor& scale_mlp = mods[4];
            const Tensor& gate_mlp = mods[5];

            // ---- self attention ----
            auto inp = modulate(layer_norm_plain(cur, kEps), shift_msa, scale_msa);
            auto q = rms_norm(linear(inp, w(p + "self_attn.q.weight"), w(p + "self_attn.q.bias")),
                               w(p + "self_attn.norm_q.weight"), kEps);
            auto k = rms_norm(linear(inp, w(p + "self_attn.k.weight"), w(p + "self_attn.k.bias")),
                               w(p + "self_attn.norm_k.weight"), kEps);
            auto v = linear(inp, w(p + "self_attn.v.weight"), w(p + "self_attn.v.bias"));
            if (i == 0 && ref.count("b0_inp")) {
                report("b0_inp", inp, ref.at("b0_inp"));
                report("b0_q_pre", q, ref.at("b0_q_pre"));
                report("b0_k_pre", k, ref.at("b0_k_pre"));
                report("b0_v_pre", v, ref.at("b0_v_pre"));
            }
            q = rope_apply(q, rope, kHeads);
            k = rope_apply(k, rope, kHeads);
            if (i == 0 && ref.count("b0_q_rope")) {
                report("b0_q_rope", q, ref.at("b0_q_rope"));
                report("b0_k_rope", k, ref.at("b0_k_rope"));
            }
            // window partition -> (b block_n) block_s d -> b (block_n block_s) d
            q = partition3d(q, pf, ph, pw);
            k = partition3d(k, pf, ph, pw);
            v = partition3d(v, pf, ph, pw);
            if (i == 0 && ref.count("q_w")) {
                report("q_w", q, ref.at("q_w"));
                report("k_w", k, ref.at("k_w"));
            }
            const int64_t block_n = q.size(0);
            q = q.view({1, block_n * q.size(1), kDim});
            k = k.view({1, block_n * k.size(1), kDim});
            v = v.view({1, block_n * v.size(1), kDim});
            auto attn = attention(q, k, v, kHeads, &blk_mask);
            // b (block_n block_s) d -> (b block_n) block_s d -> reverse partition
            attn = attn.view({block_n, attn.size(1) / block_n, kDim});
            attn = reverse3d(attn, pf, ph, pw);
            attn = linear(attn, w(p + "self_attn.o.weight"), w(p + "self_attn.o.bias"));
            cur = cur + gate_msa * attn;

            // ---- cross attention ----
            auto n3 = layer_norm_affine(cur, w(p + "norm3.weight"), w(p + "norm3.bias"), kEps);
            auto cq = rms_norm(linear(n3, w(p + "cross_attn.q.weight"), w(p + "cross_attn.q.bias")),
                                w(p + "cross_attn.norm_q.weight"), kEps);
            auto ck = rms_norm(linear(ctx, w(p + "cross_attn.k.weight"), w(p + "cross_attn.k.bias")),
                                w(p + "cross_attn.norm_k.weight"), kEps);
            auto cv = linear(ctx, w(p + "cross_attn.v.weight"), w(p + "cross_attn.v.bias"));
            auto ca = attention(cq, ck, cv, kHeads, nullptr);
            ca = linear(ca, w(p + "cross_attn.o.weight"), w(p + "cross_attn.o.bias"));
            cur = cur + ca;

            // ---- ffn ----
            auto inp2 = modulate(layer_norm_plain(cur, kEps), shift_mlp, scale_mlp);
            auto ff = linear(inp2, w(p + "ffn.0.weight"), w(p + "ffn.0.bias"));
            ff = torch::gelu(ff, "tanh");
            ff = linear(ff, w(p + "ffn.2.weight"), w(p + "ffn.2.bias"));
            cur = cur + gate_mlp * ff;

            if (i == 0 && ref.count("block00")) report("block00", cur, ref.at("block00"));
            if (i == 15 && ref.count("block15")) report("block15", cur, ref.at("block15"));
            if (i == 29 && ref.count("block29")) report("block29", cur, ref.at("block29"));
        }
        return cur;
    };

    auto out = run(seq, "final");

    // ---- head + unpatchify -------------------------------------------------
    auto hmods = (w("head.modulation") + t).chunk(2, 1);
    auto hn = modulate(layer_norm_plain(out, kEps), hmods[0], hmods[1]);
    auto hout = linear(hn, w("head.head.weight"), w("head.head.bias"));   // [1,S,out*px*py*pz]
    const int64_t px = kPatchF, py = kPatchH, pz = kPatchW;
    auto img = hout.view({1, pf, ph, pw, px, py, pz, kOutDim});
    img = img.permute({0, 7, 1, 4, 2, 5, 3, 6}).contiguous();             // b c f x h y w z
    img = img.view({1, kOutDim, pf * px, ph * py, pw * pz});
    report("y", img, ref.at("y"));

    auto t2 = std::chrono::high_resolution_clock::now();
    std::printf("forward: %.2f s\n", std::chrono::duration<double>(t2 - t1).count());
    std::printf("ref y mean %+.5f   got y mean %+.5f\n", ref.at("y").to(torch::kFloat).mean().item<double>(),
                img.to(torch::kFloat).mean().item<double>());
    return 0;
}

int main() {
    try {
        return run_main();
    } catch (const std::exception& e) {
        std::printf("EXC: %s\n", e.what());
        return 1;
    }
}
