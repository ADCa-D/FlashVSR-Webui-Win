# 引擎：tile 预处理 / 画布混合 / 权重与激活量化

> 原样摘录，未做任何改写。行号为该文件内真实行号。

## 来源：`engine.py`（共 3435 行）

### `_feather_mask_nhwc`  （第 477–490 行，原样摘录）

```python
def _feather_mask_nhwc(size, overlap, dtype, device, smooth=True):
    """羽化权重按 (h, w, band, dtype, device) 缓存：每 tile 现建+转置+dtype 转换是纯开销。"""
    key = (int(size[0]), int(size[1]), int(overlap), str(dtype), str(device), bool(smooth))
    m = _FEATHER_CACHE.get(key)
    if m is None:
        m = _feather_mask(size, overlap, smooth=smooth)
        m = m.permute(0, 2, 3, 1).to(dtype=dtype, device=device).contiguous()
        if len(_FEATHER_CACHE) > 24:
            # 逐条淘汰最老的，不要 clear()：tile 尺寸/窗口帧数来回变时全清会把热点
            # （同一尺寸的 mask 每个 tile 都要用）一起丢掉，紧接着重建 → 周期性抖动。
            for _old in list(_FEATHER_CACHE)[:8]:
                _FEATHER_CACHE.pop(_old, None)
        _FEATHER_CACHE[key] = m
    return m
```

### `_prepare_input_gpu`  （第 561–592 行，原样摘录）

```python
def _prepare_input_gpu(mod, image_tensor, device, scale=4, dtype=torch.bfloat16):
    """prepare_input_tensor 的 GPU 版：输入可以是 uint8 NHWC 或 float NHWC（[0,1]）。

    与节点版逐位一致的部分：F = largest_8n1_leq(N0+4)、bicubic(align_corners=False)、
    居中裁剪到 128 的倍数、先 to(dtype) 再 *2-1、输出 (1,3,F,tH,tW)。
    不同处：不做 D2H、不在 CPU 上 stack、不在末尾 clean_vram（由 canvas 每 N 个 tile 清一次）。
    """
    N0, h0, w0, _ = image_tensor.shape
    sW, sH, tW, tH = mod.compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    F = mod.largest_8n1_leq(N0 + 4)
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {N0 + 4}.")

    if image_tensor.dtype == torch.uint8:
        # 必须**在 CPU 上**做 /255：GPU 的 float32 除法与 CPU 有 1 ULP 差异
        # （实测一半元素差 5.96e-8），经 fp8 量化放大后整段输出 PSNR 只有 ~37 dB。
        # 只对"当前 tile 的切片"做，512 KB 级别，代价可忽略；整窗仍保持 uint8。
        # _stage_uint8_to_gpu 用 pinned 双缓冲，让 H2D 真异步（见其 docstring）。
        src = _stage_uint8_to_gpu(image_tensor, device)
    else:
        src = image_tensor.to(device=device, non_blocking=True)
        if src.dtype != torch.float32:
            src = src.float()

    idx = torch.arange(F, device=src.device).clamp_(max=max(N0 - 1, 0))
    x = src.index_select(0, idx).permute(0, 3, 1, 2)          # (F,3,h0,w0)
    x = torch.nn.functional.interpolate(x, size=(sH, sW), mode="bicubic", align_corners=False)
    left, top = max(0, (sW - tW) // 2), max(0, (sH - tH) // 2)
    x = x[:, :, top:top + tH, left:left + tW]
    vid = (x.to(dtype) * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    setattr(mod, _PI_CROP_KEY, (top, left))
    return vid, tH, tW, F
```

### `flashvsr_gpu_canvas`  （第 623–746 行，原样摘录）

```python
def flashvsr_gpu_canvas(mod, pipe, frames, scale, color_fix, tile, tile_overlap,
                        sparse, kv, local, seed=0, seam_fix="smooth"):
    """和节点的 tiled_dit 分支等价，但**画布与混合都在 GPU 上**。

    省掉的成本：每 tile 一次 `.to("cpu")`（8 帧 256² ≈ 3 MB PCIe）+ CPU 上的
    float16 乘加（15 tile ≈ 0.5–0.8 s，且与 GPU 串行）。
    """
    device = pipe.device
    dtype = pipe.torch_dtype
    n0 = frames.shape[0]
    add_n = mod.next_8n5(n0) - n0
    if add_n > 0:
        frames = torch.cat([frames, frames[-1:, :, :, :].repeat(add_n, 1, 1, 1)], dim=0)

    N, H, W, C = frames.shape
    # 小视频的两个坑（都只在"窗口比 tile 小"的轴上出现）：
    #   1) prepare_input_tensor 的目标边长是 max(128, floor(s/128)*128)，s < 128 时它
    #      不补边、直接输出更小的 tile，却仍把 128 当 height/width 传给 pipe → 出图全黑；
    #   2) 只有一个 tile 时，居中裁掉的边距没有邻居 tile 来补 → 右/下留黑边。
    # 处理：把这条轴用边缘复制补到 128/scale 的整数倍，拼完再裁回原尺寸。
    # 窗口 >= tile 的轴上每块都正好是 tile 大小、边距为零，走原来的逻辑（不改 544p 等常规路径）。
    if isinstance(tile, (tuple, list)):
        _th_t, _tw_t = int(tile[0]), int(tile[1])
    else:
        _th_t = _tw_t = int(tile)
    unit = 128 // math.gcd(128, max(1, int(scale)))
    Hp = ((H + unit - 1) // unit) * unit if H < _th_t else H
    Wp = ((W + unit - 1) // unit) * unit if W < _tw_t else W
    if Hp != H or Wp != W:
        import torch.nn.functional as _F
        _x = frames.permute(0, 3, 1, 2)
        _x = _F.pad(_x, (0, Wp - W, 0, Hp - H), mode="replicate")
        frames = _x.permute(0, 2, 3, 1)
    # 累加精度：默认 fp32。实测同帧 fp16 vs fp32 只有 mean|d|=0.18/255（9% 像素），
    # 耗时差 <1%，但 fp16 在重叠区最多叠 4 块时会丢尾数，hi_precision 下更敏感。
    # FLASHVSR_CANVAS_FP32=0 可退回 fp16（显存占用减半，4K 输出省约 50 MiB/帧）。
    _acc = torch.float32 if os.environ.get("FLASHVSR_CANVAS_FP32", "1") != "0" else torch.float16
    canvas = torch.zeros((N, Hp * scale, Wp * scale, C), dtype=_acc, device=device)
    wsum = torch.zeros((N, Hp * scale, Wp * scale, 1), dtype=_acc, device=device)
    # run_win 可能传矩形 tile：(tile_h, tile_w)。以前这里直接 int(tile) → TypeError。
    if isinstance(tile, (tuple, list)):
        th_t, tw_t = int(tile[0]), int(tile[1])
    else:
        th_t = tw_t = int(tile)
    # 节点的 prepare_input_tensor 会把 tile 居中裁剪到 128 的倍数（nodes.py:53-71）。
    # tile*scale 不是 128 的倍数时，每块输出比自己的覆盖范围小 → 画布必然有空洞。
    sc = int(scale)
    if (th_t * sc) % 128 != 0 or (tw_t * sc) % 128 != 0:
        step = 128 // math.gcd(128, max(1, sc))
        if th_t == tw_t:
            print(f"[canvas] 警告：tile={tile}×scale={scale}={th_t * sc} 不是 128 的倍数，"
                  f"接缝处会有 %128 的错位/空洞（建议 tile 取 {step} 的倍数）", flush=True)
        else:
            print(f"[canvas] 警告：tile={th_t}×{tw_t}×scale={scale} 时 "
                  f"{th_t * sc}×{tw_t * sc} 不是 128 的倍数，"
                  f"接缝处会有 %128 的错位/空洞（建议 tile_h/tile_w 取 {step} 的倍数）", flush=True)
    coords = mod.calculate_tile_coords(Hp, Wp, tile, tile_overlap)
    if not coords:
        # calculate_tile_coords 在 H/W <= overlap 时返回空表（num_rows=0）→ 从这里出去就是整帧全黑
        coords = [(0, 0, Wp, Hp)]
    pipe_cls = getattr(mod, "FlashVSRTinyLongPipeline", None)
    _TILE_HK[0] = 0

    for x1, y1, x2, y2 in mod.cqdm(coords, desc="Processing Tiles"):
        input_tile = frames[:, y1:y2, x1:x2, :]
        LQ_tile, th, tw, F = mod.prepare_input_tensor(input_tile, device, scale=scale, dtype=dtype)
        if pipe_cls is not None and not isinstance(pipe, pipe_cls):
            LQ_tile = LQ_tile.to(device)
        # 静止块复用的"位置键"：必须在 pipe(...) 之前设，解码发生在其内部
        _TILE_POS[0] = (int(x1), int(y1), int(x2), int(y2))
        out_gpu = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
            tiled=True, LQ_video=LQ_tile, num_frames=F, height=th, width=tw,
            is_full_block=False, if_buffer=True,
            topk_ratio=sparse * 768 * 1280 / (th * tw), kv_ratio=kv, local_range=local,
            color_fix=color_fix, unload_dit=False, force_offload=False)
        tile_gpu = mod.tensor2video(out_gpu)                    # (F,H,W,C)
        # 实测：tiny-long 的 pipe(...) 输出会落到 CPU（流式解码内部就搬了），
        # 所以这里先对齐设备——若是 CPU 就 H2D（3 MB/tile ≈ 0.25 ms，可忽略），
        # 换取混合在显存里做（省掉 CPU 上的 float16 乘加）。
        if tile_gpu.device.type == "cpu":
            tile_gpu = tile_gpu.to(device, non_blocking=True)
        if tile_gpu.shape[0] != canvas.shape[0]:
            # pipe 内部还能再补帧（8n+1 对齐），多出来的尾帧不属于本窗
            tile_gpu = tile_gpu[:canvas.shape[0]]
        th_o, tw_o = int(tile_gpu.shape[1]), int(tile_gpu.shape[2])
        # seam_fix 在 GPU 画布上也要生效（以前只有 CPU 回退路径才看得到这个参数）：
        # off=硬边（不加羽化过渡），smooth=平滑羽化，align=羽化 + 逐 tile 亮度偏置对齐。
        mask_hw = _feather_mask_nhwc((th_o, tw_o), int(tile_overlap) * scale,
                                     tile_gpu.dtype, tile_gpu.device,
                                     smooth=(seam_fix != "off"))
        # prepare_input_tensor 会把 tile 居中裁剪到 128 的倍数，裁掉的偏移必须补回来，
        # 否则 tile 尺寸不是 128/scale 的倍数时每块都会整体偏移 + 右/下留黑边。
        ct, cl = getattr(mod, _PI_CROP_KEY, (0, 0))
        ox1, oy1 = x1 * scale + int(cl), y1 * scale + int(ct)
        ox2, oy2 = ox1 + tw_o, oy1 + th_o
        if seam_fix == "align":
            # 只在"重叠区且已有内容、且不在本 tile 的实心区"的像素上估每通道 DC 偏差，
            # 把整块 tile 平移到已有画布上，接缝两侧就没有亮暗台阶。
            _wp = wsum[:, oy1:oy2, ox1:ox2, :]
            _have = (_wp > 1e-3) & (mask_hw < 0.999)
            if int(_have.sum()) > 512:
                _cur = canvas[:, oy1:oy2, ox1:ox2, :] / _wp.clamp_min(1e-6)
                _sel = _have.expand(-1, -1, -1, canvas.shape[3])
                _d = (_cur.to(tile_gpu.dtype) - tile_gpu)[_sel].reshape(
                    -1, canvas.shape[3]).mean(dim=0)
                tile_gpu = tile_gpu + _d
        canvas[:, oy1:oy2, ox1:ox2, :] += (tile_gpu * mask_hw).to(_acc)
        wsum[:, oy1:oy2, ox1:ox2, :] += mask_hw.to(_acc)
        del LQ_tile, out_gpu, tile_gpu, mask_hw
        _TILE_POS[0] = None
        _tile_housekeeping()

    _debug_check_tensor("canvas", canvas)
    _debug_check_tensor("wsum", wsum)
    if float(wsum.min()) <= 1e-6:
        # 以前这里是静默的：没覆盖到的像素除完就是黑边，只有肉眼能发现
        print("[canvas] 警告：有像素未被任何 tile 覆盖（会是黑边），检查 tile/overlap/裁剪偏移",
              flush=True)
    out = canvas / wsum.clamp_min_(1e-6)
    out = out[:n0]
    if out.shape[1] != H * scale or out.shape[2] != W * scale:
        out = out[:, :H * scale, :W * scale, :]      # 去掉补边（复制的边缘像素）
    return out
```

### `install_stream_loader`  （第 835–931 行，原样摘录）

```python
def install_stream_loader(quant="none"):
    """把节点的 safetensors 加载改成"逐个张量读 → 立刻转 dtype → 立刻上卡"。

    原实现（utils.load_state_dict_from_safetensors）会攒一个完整的 CPU state_dict，
    实测 commit 峰值 **12.6 GB**（f32 视图 5.4 + bf16 副本 2.8 + 分配器缓存 + 显存），
    在 16 GB 内存的机器上加载时浏览器都得关掉。
    流式之后 CPU 侧同时只持有一个张量，峰值降到"一个最大的张量"量级。

    只影响加载路径，不改模型结构；失败就退回原实现。
    """
    # 即使已经 patch 过也要重算：切 quant（none/w8）后旧集合会让权重被错误跳过
    set_stream_skip(quant)
    try:
        utils = sys.modules.get("flashvsr_node.src.models.utils")
        if utils is None or getattr(utils, "_stream_patched", False):
            return False
        orig = utils.load_state_dict_from_safetensors

        # 缓存里已有的层：不读盘、不转 bf16，直接在显存里占位（真张量，不是 meta，
        # 这样节点的 model.to(dtype, device) 不会崩）。随后 quantize_dit_mxfp8 会用
        # 缓存里的 fp8 覆盖 forward 并把占位丢掉，所以这些空张量只活几秒。
        # 集合按文件判断：非本 DiT checkpoint 一律不跳过（否则别的模型也吃空张量）。
        dev_ph = "cuda" if torch.cuda.is_available() else "cpu"

        def patched(file_path, torch_dtype=None):
            """不用 safe_open（它 mmap 整个文件，5.4 GB 文件页会常驻工作集）。

            』 这里自己解析头 + read()，一次只读一个张量：
              - 工作集峰值 ≈ 最大的那个张量（~50 MB），而不是整个 checkpoint
              - 读出来 → 转 dtype → 立刻上卡 → 立刻释放
            等价于 cpp-port 分支里 load_dit.cpp 的做法（那边 host 峰值 542 MiB）。
            """
            import json
            import struct

            import numpy as np
            import torch as _t

            dev = "cuda" if _t.cuda.is_available() else "cpu"
            _tgt = _STREAM_SKIP_CFG["ckpt"]
            _is_dit = _tgt is not None and \
                os.path.normcase(os.path.abspath(file_path)) == _tgt
            skip = _STREAM_SKIP_CFG["keys"] if _is_dit else set()
            skipped_n = 0
            np_dtype = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16,
                        "I64": np.int64, "I32": np.int32, "I16": np.int16,
                        "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
            out = {}
            with open(file_path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
                base = 8 + hlen
                for k, info in hdr.items():
                    if k == "__metadata__" or not isinstance(info, dict):
                        continue
                    _base = k[:-7] if k.endswith(".weight") else k
                    if _base in skip:
                        out[k] = torch.empty(info["shape"],
                                             dtype=torch_dtype or torch.bfloat16,
                                             device=dev_ph)
                        skipped_n += 1
                        continue
                    beg, end = info["data_offsets"]
                    f.seek(base + beg)
                    raw = f.read(end - beg)
                    nd = np_dtype.get(info["dtype"])
                    if nd is None:
                        raise RuntimeError(f"不支持的 dtype {info['dtype']} ({k})")
                    arr = np.frombuffer(raw, dtype=nd)
                    if info["dtype"] == "BF16":
                        # bf16 在 numpy 里没有原生类型：按 uint16 读，再用 view 转 torch
                        ten = arr.copy().view(_t.bfloat16).reshape(info["shape"])
                    else:
                        ten = _t.from_numpy(arr.copy()).reshape(info["shape"])
                    if torch_dtype is not None:
                        ten = ten.to(torch_dtype)
                    out[k] = ten.to(dev)
                    del ten, arr, raw
            if skipped_n:
                print(f"[stream-load] 跳过 {skipped_n} 个已在缓存里的权重（不读盘、不转 bf16）",
                      flush=True)
            return out

        def guarded(file_path, torch_dtype=None):
            try:
                return patched(file_path, torch_dtype)
            except Exception as e:  # noqa
                print(f"[stream-load] 流式加载失败，回退原实现: {e}", flush=True)
                return orig(file_path, torch_dtype)

        utils.load_state_dict_from_safetensors = guarded
        utils._stream_patched = True
        print("[stream-load] 已启用：逐张量读→上卡，不再攒 CPU state_dict", flush=True)
        return True
    except Exception as e:  # noqa
        print(f"[stream-load] 启用失败（不影响运行）: {e}", flush=True)
        return False
```

### `quantize_dit_mxfp8`  （第 1281–1352 行，原样摘录）

```python
def quantize_dit_mxfp8(pipe, log=None):
    """MXFP8（e4m3 + 每 32 元素 e8m0）。Ada/Hopper/Blackwell 都有原生 FP8 张量核。

    实测：单段 77.0→65.4 s（−15%），峰值 6.19→4.93 GiB，PSNR 46.49 dB。
    """
    import comfy_kitchen as ck

    def _call(x, wq, wb, bias):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        xq, xb = ck.quantize_mxfp8(x2)
        y = ck.scaled_mm_mxfp8(xq, wq, xb, wb, bias=bias, out_dtype=x.dtype)
        return y.reshape(*shp[:-1], y.shape[-1])

    # 权重是**确定的产物**：算一次就落盘，之后直接复用。
    # 除了省一次量化，更重要的是把量化期间的 commit 峰值削掉——那一段会把机器逼进
    # 页文件交换（实测 commit 峰值 12.5 GB），页文件写入才是真伤 SSD 的东西。
    use_cache = os.environ.get("FLASHVSR_QCACHE", "1") != "0"
    cache_path = _qcache_path(pipe, "mxfp8") if use_cache else None
    cached = _qcache_load(cache_path, log) if cache_path else None
    # 流式加载已经把缓存里那些层换成**未初始化**的 torch.empty（省读盘）。此时若缓存
    # 读不出来（文件截断 / 显存不够 / 版本不符），下面会把随机垃圾量化再落盘，之后每次
    # 运行都从这份垃圾起跳 → 永久花屏。宁可报错并删掉坏缓存，让它下次从原模型重量化。
    _skip_keys = set(_STREAM_SKIP_CFG.get("keys") or ())
    if _skip_keys:
        _have = set((cached or {}).keys())
        _miss = _skip_keys - _have
        if _miss:
            if cache_path and os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    print(f"[FlashVSR] 量化缓存不完整，已删除 {os.path.basename(cache_path)}"
                          f"（下次运行将从原模型重新量化）", flush=True)
                except OSError as e:
                    print(f"[FlashVSR] 删除损坏的量化缓存失败：{e}", flush=True)
            raise RuntimeError(
                f"MXFP8 量化缓存不完整（缺 {len(_miss)} 层），而权重是按「跳过读盘」"
                f"加载的，继续跑会把未初始化显存量化后写回缓存。已清理缓存，请重新运行。")
    fresh = {}

    n = skipped = n_hit = 0
    for name, m in _iter_linears(pipe.dit):
        if not _should_quant(name):
            skipped += 1
            continue
        hit = cached.get(name) if cached else None
        if hit is not None:
            wq, wb = hit
            n_hit += 1
        else:
            w = _take_weight(m)
            if w is None or w.shape[1] % 32 != 0:
                skipped += 1
                continue
            try:
                wq, wb = ck.quantize_mxfp8(w)
                fresh[name] = (wq, wb)
            except Exception as e:  # noqa
                if log:
                    log(f"mxfp8 量化失败 {name}: {e}")
                skipped += 1
                continue
        bias = None if m.bias is None else m.bias.detach()
        m.forward = (lambda x, _q=wq, _s=wb, _b=bias: _call(x, _q, _s, _b))
        _drop_weight(m)
        n += 1
    if fresh and cache_path:
        _qcache_save(cache_path, fresh, log)
    torch.cuda.empty_cache()
    if log:
        log(f"MXFP8: {n} 个 Linear（缓存命中 {n_hit}，本次新量化 {len(fresh)}），跳过 {skipped}")
    return n
```

### `_qcache_load`  （第 1236–1258 行，原样摘录）

```python
def _qcache_load(path, log=None):
    """读缓存 → {layer: (wq, wb)}；格式不对就当没有。"""
    if not path or not os.path.exists(path):
        return None
    try:
        from safetensors.torch import load_file
        d = load_file(path, device="cuda")
        out = {}
        for k in d:
            if not k.endswith("|q"):
                continue
            name = k[:-2]
            b = d.get(name + "|b")
            if b is None:
                return None
            out[name] = (d[k].view(torch.float8_e4m3fn), b.view(torch.float8_e8m0fnu))
        if log:
            log(f"量化缓存命中：{len(out)} 层（{os.path.getsize(path) / 2**30:.2f} GB）")
        return out or None
    except Exception as e:  # noqa
        if log:
            log(f"量化缓存读取失败（忽略，重新量化）: {e}")
        return None
```

### `_qcache_save`  （第 1261–1278 行，原样摘录）

```python
def _qcache_save(path, items, log=None):
    """原子写：先写 .tmp 再 rename，中断不会留半截文件。"""
    if not path or not items:
        return
    try:
        from safetensors.torch import save_file
        blob = {}
        for name, (wq, wb) in items.items():
            blob[name + "|q"] = wq.contiguous().view(torch.uint8).cpu()
            blob[name + "|b"] = wb.contiguous().view(torch.uint8).cpu()
        tmp = path + ".tmp"
        save_file(blob, tmp)
        os.replace(tmp, path)
        if log:
            log(f"量化缓存已写盘：{len(items)} 层（{os.path.getsize(path) / 2**30:.2f} GB）")
    except Exception as e:  # noqa
        if log:
            log(f"量化缓存落盘失败（不影响本次运行）: {e}")
```

## 来源：`node:nodes.py`（共 553 行）

### `prepare_input_tensor`  （第 75–100 行，原样摘录）

```python
def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)
    
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
    
    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH).to('cpu').to(dtype) * 2.0 - 1.0
        frames.append(tensor_chw)
        del frame_slice

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    return vid_final, tH, tW, F
```

### `create_feather_mask`  （第 126–137 行，原样摘录）

```python
def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask
```
