// tcdecoder.cpp -- FlashVSR 的 TCDecoder（TAEHV decoder-only）在 C++ 里重实现。
//
// 结构（由 bench/dump_tcdec_spec2.py 从真实流水线反射出来，27 层）：
//   0 clamp(tanh(x/3)*3)
//   1 conv3x3 784->512 (+bias)     2 relu   3 conv3x3 512->512 (identity init)  4 relu
//   5,6,7 memblock(512)
//   8 up2   9 tgrow(stride=1)   10 conv3x3 512->256 (no bias)
//   11,12,13 memblock(256)
//   14 up2  15 tgrow(stride=2)  16 conv3x3 256->128 (no bias)
//   17,18,19 memblock(128)
//   20 up2  21 tgrow(stride=2)  22 conv3x3 128->128 (no bias)
//   23 relu 24 conv3x3 128->128   25 relu  26 conv3x3 128->3 (+bias)
//
// 时间维语义：TGrow(stride) 把 1 个时间步变成 stride 个（时间上采样）；
// MemBlock 携带上一时间步的输入作为 past（跨 window 也携带，见 decode_video 的 mem）。
// 第一帧的 cond 走 PixelShuffle3d(4,8,8)：21 帧先补到 24（前面重复首帧），
// 再重排成 768 通道 × 6 帧 × 64×64，和 16 通道的 latent 拼成 784 通道。
#include <torch/torch.h>

#include "safetensors.h"

#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cstdio>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace {

using Tensor = torch::Tensor;
constexpr int64_t kLatentC = 16;

Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b, int64_t pad) {
    return torch::conv2d(x, w, b, {1, 1}, {pad, pad});
}

// replay of apply_model_with_memblocks(parallel=False) for a fixed layer plan
struct Decoder {
    std::map<std::string, Tensor> sd;
    std::vector<Tensor> mem;        // per layer, [N,C,H,W] of the last timestep
    bool first_call = true;

    Tensor w(const std::string& k) {
        auto it = sd.find(k);
        if (it == sd.end()) {
            std::printf("missing weight: %s\n", k.c_str());
            std::exit(2);
        }
        return it->second;
    }

    // ---- individual blocks ------------------------------------------------
    Tensor clamp_(const Tensor& x) const { return torch::tanh(x / 3.0) * 3.0; }

    Tensor memblock(const Tensor& x, const Tensor& past, int64_t C, const std::string& p) {
        auto both = torch::cat({x, past}, 1);
        auto y = conv2d(both, w(p + "conv.0.weight"), w(p + "conv.0.bias"), 1);
        y = torch::relu(y);
        y = conv2d(y, w(p + "conv.2.weight"), w(p + "conv.2.bias"), 1);
        y = torch::relu(y);
        y = conv2d(y, w(p + "conv.4.weight"), w(p + "conv.4.bias"), 1);
        // skip is nn.Identity() here (n_in == n_out)
        return torch::relu(y + x);
    }

    Tensor tgrow(const Tensor& x, int64_t stride, const std::string& key) {
        return conv2d(x, w(key), Tensor(), 0);   // 1x1, no bias
    }

    // ---- the 27-layer plan ------------------------------------------------
    // returns the timesteps produced at the end of the stack (in order)
    std::vector<Tensor> run(const Tensor& input_nt, int64_t N) {
        // queue items: tensor + layer index (mirrors the python work queue)
        struct Item {
            Tensor x;
            int64_t layer;
        };
        std::deque<Item> q;
        for (int64_t t = 0; t < input_nt.size(0); ++t)
            q.push_back({input_nt.narrow(0, t, 1), 0});   // keep the batch dim: [1,C,H,W]

        std::vector<Tensor> out;
        const int64_t nlayers = 27;
        while (!q.empty()) {
            Item it = q.front();
            q.pop_front();
            if (it.layer == nlayers) {
                out.push_back(it.x);
                continue;
            }
            const int64_t i = it.layer;
            Tensor xt = it.x;
            switch (i) {
                case 0: xt = clamp_(xt); break;
                case 1: xt = conv2d(xt, w("decoder.1.weight"), w("decoder.1.bias"), 1); break;
                case 2: xt = torch::relu(xt); break;
                case 3: xt = conv2d(xt, w("decoder.3.weight"), Tensor(), 1); break;
                case 4: xt = torch::relu(xt); break;
                case 5: case 6: case 7: {
                    const std::string p = "decoder." + std::to_string(i) + ".";
                    if (mem[i].defined()) xt = memblock(xt, mem[i], 512, p);
                    else {
                        auto zero = xt * 0;
                        xt = memblock(xt, zero, 512, p);
                    }
                    mem[i] = it.x.clone();   // python stores the *input*
                    break;
                }
                case 8: xt = torch::upsample_nearest2d(xt, {xt.size(2) * 2, xt.size(3) * 2}); break;
                case 9: xt = tgrow(xt, 1, "decoder.9.conv.weight"); break;
                case 10: xt = conv2d(xt, w("decoder.10.weight"), Tensor(), 1); break;
                case 11: case 12: case 13: {
                    const std::string p = "decoder." + std::to_string(i) + ".";
                    if (mem[i].defined()) xt = memblock(xt, mem[i], 256, p);
                    else xt = memblock(xt, xt * 0, 256, p);
                    mem[i] = it.x.clone();
                    break;
                }
                case 14: xt = torch::upsample_nearest2d(xt, {xt.size(2) * 2, xt.size(3) * 2}); break;
                case 15: xt = tgrow(xt, 2, "decoder.15.conv.weight"); break;
                case 16: xt = conv2d(xt, w("decoder.16.weight"), Tensor(), 1); break;
                case 17: case 18: case 19: {
                    const std::string p = "decoder." + std::to_string(i) + ".";
                    if (mem[i].defined()) xt = memblock(xt, mem[i], 128, p);
                    else xt = memblock(xt, xt * 0, 128, p);
                    mem[i] = it.x.clone();
                    break;
                }
                case 20: xt = torch::upsample_nearest2d(xt, {xt.size(2) * 2, xt.size(3) * 2}); break;
                case 21: xt = tgrow(xt, 2, "decoder.21.conv.weight"); break;
                case 22: xt = conv2d(xt, w("decoder.22.weight"), Tensor(), 1); break;
                case 23: xt = torch::relu(xt); break;
                case 24: xt = conv2d(xt, w("decoder.24.weight"), Tensor(), 1); break;
                case 25: xt = torch::relu(xt); break;
                case 26: xt = conv2d(xt, w("decoder.26.weight"), w("decoder.26.bias"), 1); break;
                default: std::printf("unexpected layer %lld\n", (long long)i); std::exit(3);
            }

            if (i == 15 || i == 21) {          // TGrow with stride 2: 1 timestep -> 2
                const int64_t stride = 2;
                const int64_t C = xt.size(1) / stride;
                for (int64_t s = stride - 1; s >= 0; --s) {
                    auto piece = xt.narrow(1, s * C, C).contiguous();
                    q.push_front({piece, i + 1});
                }
                continue;
            }
            q.push_front({xt, i + 1});
        }
        (void)N;
        return out;
    }
};

// PixelShuffle3d(4,8,8) on cond [N,3,T,H,W] -> [N, T/4, 768, H/8, W/8]
Tensor pixel_shuffle_cond(const Tensor& cond) {
    const int64_t N = cond.size(0), C = cond.size(1), T = cond.size(2);
    const int64_t H = cond.size(3), W = cond.size(4);
    const int64_t ff = 4, hh = 8, ww = 8;
    Tensor x = cond;
    if (T % ff != 0) {
        // python prepends copies of frame 0
        auto first = x.narrow(2, 0, 1);
        auto pad = first.repeat({1, 1, ff - T % ff, 1, 1});
        x = torch::cat({pad, x}, 2);
    }
    const int64_t Tp = x.size(2);
    auto v = x.view({N, C, Tp / ff, ff, H / hh, hh, W / ww, ww});
    v = v.permute({0, 1, 3, 5, 7, 2, 4, 6});      // b c ff hh ww f h w
    v = v.contiguous().view({N, C * ff * hh * ww, Tp / ff, H / hh, W / ww});
    return v.transpose(1, 2);                       // -> [N, f, C', h, w]
}

double maxdiff(const Tensor& a, const Tensor& b) {
    if (a.sizes() != b.sizes()) {
        std::printf("   shape mismatch:");
        for (auto s : a.sizes()) std::printf(" %lld", (long long)s);
        std::printf(" vs");
        for (auto s : b.sizes()) std::printf(" %lld", (long long)s);
        std::printf("\n");
        return -1;
    }
    return (a.to(torch::kFloat).cpu() - b.to(torch::kFloat).cpu()).abs().max().item<double>();
}

}  // namespace

static int run_main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (!LoadLibraryA("torch_cuda.dll")) return 1;
    if (!torch::cuda::is_available()) {
        std::printf("CUDA not available\n");
        return 1;
    }

    const char* ckpt =
        "D:\\ComfyUI_windows_portable\\ComfyUI\\models\\FlashVSR-v1.1\\TCDecoder.ckpt";
    const char* refp = "work\\tcdec\\ref.safetensors";

    auto opt = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto ref = st::load_all(refp, opt);
    if (ref.empty()) {
        std::printf("missing %s (run bench/dump_tcdec_spec2.py + make_tcdec_ref.py)\n", refp);
        return 1;
    }

    // ---- weights: converted to safetensors by bench/make_tcdec_ref.py -----
    auto t0 = std::chrono::high_resolution_clock::now();
    Decoder dec;
    {
        auto all = st::load_all("work\\tcdec\\weights.safetensors", opt);
        if (all.empty()) {
            std::printf("missing work/tcdec/weights.safetensors (run bench/make_tcdec_ref.py)\n");
            return 1;
        }
        for (auto& kv : all) {
            if (kv.first.rfind("decoder.", 0) == 0) dec.sd[kv.first] = kv.second;
        }
    }
    dec.mem.assign(27 + 1, Tensor());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("loaded %zu weights in %.2f s\n", dec.sd.size(),
                std::chrono::duration<double>(t1 - t0).count());

    // ---- inputs ----------------------------------------------------------
    auto x = ref.at("x").to(torch::kBFloat16);       // [1,6,16,64,64]
    auto cond = ref.at("cond").to(torch::kBFloat16);  // [1,3,21,512,512]
    const int64_t N = x.size(0), T = x.size(1);
    std::printf("x %s  cond %s\n", [&] {
        std::string s;
        for (auto v : x.sizes()) s += std::to_string(v) + " ";
        return s;
    }().c_str(), [&] {
        std::string s;
        for (auto v : cond.sizes()) s += std::to_string(v) + " ";
        return s;
    }().c_str());

    // ---- preprocess: pixel-shuffle(cond) ++ latent along channels --------
    auto cps = pixel_shuffle_cond(cond);             // [N,T,768,64,64]
    std::printf("pixel_shuffle(cond) -> %s\n", [&] {
        std::string s;
        for (auto v : cps.sizes()) s += std::to_string(v) + " ";
        return s;
    }().c_str());
    auto joined = torch::cat({cps, x}, 2);            // [N,T,784,64,64]
    auto nt = joined.contiguous().view({N * T, joined.size(2), joined.size(3), joined.size(4)});
    std::printf("stack input %lld x %lld x %lld x %lld\n", (long long)nt.size(0),
                (long long)nt.size(1), (long long)nt.size(2), (long long)nt.size(3));

    // ---- run -------------------------------------------------------------
    auto t2 = std::chrono::high_resolution_clock::now();
    auto outs = dec.run(nt, N);
    auto t3 = std::chrono::high_resolution_clock::now();
    std::printf("forward %.2f s, %zu timesteps out\n",
                std::chrono::duration<double>(t3 - t2).count(), outs.size());

    // stack -> [N,T,3,H,W], then trim the first 3 frames on the first call
    auto y = torch::stack(outs, 1);
    if (dec.first_call) {
        const int64_t trim = 3;
        y = y.narrow(1, trim, y.size(1) - trim);
        dec.first_call = false;
    }
    std::printf("y %s\n", [&] {
        std::string s;
        for (auto v : y.sizes()) s += std::to_string(v) + " ";
        return s;
    }().c_str());

    const auto& want = ref.at("y");
    std::printf("ref y mean %+.5f   got y mean %+.5f\n",
                want.to(torch::kFloat).mean().item<double>(),
                y.to(torch::kFloat).mean().item<double>());
    std::printf("maxdiff vs python = %.4e\n", maxdiff(y, want));
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
