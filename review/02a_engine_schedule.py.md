# 引擎：外层调度（分段 / 分窗 / tile / 复用）

> 原样摘录，未做任何改写。行号为该文件内真实行号。

## 来源：`engine.py`（共 3435 行）

### `_run_window`  （第 2846–2922 行，原样摘录）

```python
    def _run_window(self, frames_u8, scale, tile, kv, on_tile=None, sparse=2.0, local=11,
                    tile_overlap=24, color_fix=True, seam_fix="smooth"):
        """tile 可以是 int（正方形）或 (th, tw)（矩形）。

        force_offload=False：每 tile 结束把 2.84 GB DiT 卸到 CPU 再搬回要多花 ~2.9 s/tile，因此不做。
        """
        mod = self.mod
        orig = mod.cqdm

        class Hook:
            def __init__(self, iterable=None, total=None, desc=""):
                self.total = total if total is not None else (len(iterable) if iterable is not None else 1)
                self.it = iter(iterable) if iterable is not None else None
                self.n = 0

            def __iter__(self):
                return self

            def __next__(self):
                v = next(self.it)
                self.n += 1
                if on_tile:
                    on_tile(self.n, self.total)
                return v

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def __len__(self):
                return self.total

        mod.cqdm = Hook
        try:
            _f = torch.from_numpy(np.ascontiguousarray(frames_u8))
            n_f, h_f, w_f = int(_f.shape[0]), int(_f.shape[1]), int(_f.shape[2])
            use_gpu_canvas = (os.environ.get("FLASHVSR_GPU_CANVAS", "1") != "0")
            if use_gpu_canvas and not _gpu_canvas_fits(n_f, h_f, w_f, int(scale)):
                use_gpu_canvas = False
            want_gpu_prep = use_gpu_canvas and os.environ.get("FLASHVSR_GPU_PREPARE", "1") != "0"
            if want_gpu_prep:
                patched = _install_gpu_prepare(mod)
            else:
                # 关掉时必须还原：否则上一次窗口留下的 GPU 版 prepare 会一直生效
                _uninstall_gpu_prepare(mod)
                patched = False
            if patched and _f.dtype == torch.uint8:
                # 整窗保持 uint8：每个 tile 只把用到的切片搬上卡再转 float。
                # 以前是整窗先 float32（1080p/8 帧 ≈ 199 MB + 全量 CPU 遍历），而下游只用切片。
                frames = _f
            else:
                # uint8 → [0,1] 需要缩放；float32 已是 [0,1]（不能原地除，否则会把 numpy 缓冲改坏）
                frames = _f.float().div_(255.0) if _f.dtype == torch.uint8 else _f.float()
            if use_gpu_canvas:
                out = flashvsr_gpu_canvas(
                    mod, self.pipe, frames, int(scale), bool(color_fix), tile,
                    int(tile_overlap), float(sparse), float(kv), int(local), 0,
                    seam_fix=str(seam_fix or "smooth"))
            elif seam_fix == "align":
                _install_smooth_feather(mod)
                out = flashvsr_aligned(
                    mod, self.pipe, frames, int(scale), bool(color_fix), tile,
                    int(tile_overlap), float(sparse), float(kv), int(local), 0)
            else:
                if seam_fix != "off":
                    _install_smooth_feather(mod)
                else:
                    _uninstall_smooth_feather(mod)
                out = mod.flashvsr(
                    self.pipe, frames, int(scale), bool(color_fix), True, True,
                    tile, int(tile_overlap), False, float(sparse), float(kv), int(local), 0, False)
            _debug_check_tensor("window_out", out)
            return _quant8(out)
        finally:
            mod.cqdm = orig
```

### `upscale_video`  （第 2924–3434 行，原样摘录）

```python
    def upscale_video(self, src, dst, params, on_stage):
        pre_scale = float(params.get("pre_scale") or 1.0)
        chunk = max(0, int(params.get("chunk") or 0))
        overlap = max(0, min(int(params.get("overlap") or 8), max(0, chunk // 4)))

        # 分段时只解当前窗口需要的帧（原来会把整段视频一次解完，30 帧/段也照样吃满内存）
        info = probe(src)
        total = int(info.get("frames") or 0) or int(round(info.get("duration", 0) * info.get("fps", 25)))
        # 试跑：只处理前 N 帧（0/缺省 = 全片）。分段、续跑、指标都跟着这个上限走。
        max_frames = max(0, int(params.get("max_frames") or 0))
        if max_frames and 0 < max_frames < total:
            total = max_frames
        # 浮点解码：YUV→RGB 全程 float32（libplacebo），不再经过 rgb24 的 8bit 量化。
        # 不做预缩放时也有效（省掉 swscale 的四舍五入与抖动），代价是每帧内存 4×。
        fp32 = bool(params.get("hi_precision"))
        if fp32:
            tw = max(2, int(round(info["w"] * pre_scale / 2)) * 2)
            th = max(2, int(round(info["h"] * pre_scale / 2)) * 2)
            win_f = min(total, chunk) if chunk else total
            gb = win_f * tw * th * 3 * 4 / 2 ** 30
            if gb > 2.0:
                on_stage("decode", 0, total,
                         f"浮点解码需要 {gb:.1f} GB 内存，超过 2 GB 上限 → 回退 8bit")
                fp32 = False
            else:
                on_stage("decode", 0, total, f"浮点解码（gbrpf32le，窗口约 {gb:.2f} GB）")
        frames = None
        if chunk <= 0 or total <= chunk or total <= 0:
            if max_frames:
                # 试跑路径：只解前 total 帧，不把整段视频读进内存
                frames = decode_frames_range(
                    src, 0, total, pre_scale=pre_scale, fps=info["fps"], fp32=fp32,
                    scaler=params.get("prescale_scaler"),
                    deband=bool(params.get("prescale_deband")),
                    deband_grain=params.get("deband_grain"),
                    dither=str(params.get("prescale_dither") or ""),
                    on_progress=lambda n: on_stage("decode", n, total, f"解码中 {n}/{total} 帧"))
                if abs(pre_scale - 1.0) > 1e-6:
                    info["w"] = max(2, int(round(info["w"] * pre_scale / 2)) * 2)
                    info["h"] = max(2, int(round(info["h"] * pre_scale / 2)) * 2)
            else:
                frames, info = decode_frames(
                    src, on_progress=lambda n: on_stage("decode", n, 0, f"解码中 {n} 帧"),
                    pre_scale=pre_scale, fp32=fp32, scaler=params.get("prescale_scaler"),
                    deband=bool(params.get("prescale_deband")),
                    deband_grain=params.get("deband_grain"),
                    dither=str(params.get("prescale_dither") or ""))
            total = len(frames)
            on_stage("decode", total, total, f"共 {total} 帧 {info['w']}x{info['h']} @{info['fps']:.2f}fps")
        else:
            if abs(pre_scale - 1.0) > 1e-6:
                info["w"] = max(2, int(round(info["w"] * pre_scale / 2)) * 2)
                info["h"] = max(2, int(round(info["h"] * pre_scale / 2)) * 2)
            on_stage("decode", 0, total, f"分段解码：共 {total} 帧 {info['w']}x{info['h']} @{info['fps']:.2f}fps")

        # 预热尺寸必须和 run_win 实际用的 tile 一致：矩形 tile 下以前按方形算 → 白预热一次
        _bt = params.get("tile") or 0
        _bh = _bt[0] if isinstance(_bt, (tuple, list)) else _bt
        _bw = _bt[1] if isinstance(_bt, (tuple, list)) else _bt
        _pth = max(1, int(params.get("tile_h") or _bh))
        _ptw = max(1, int(params.get("tile_w") or _bw))
        _psc = max(1, int(params["scale"]))
        _prewarm = [(max(16, _pth * _psc // 8), max(16, _ptw * _psc // 8))]
        self.ensure_loaded(params.get("mode") or "tiny-long", params.get("attention") or "sparse",
                           on_stage, quant=params.get("quant") or "none",
                           prewarm=_prewarm)
        drop_on = os.environ.get("FLASHVSR_FRAME_DROP", "0") == "1"
        reuse_mode = str(params.get("reuse") or "off")
        self._reuse_tile = reuse_mode in ("tile", "window")
        self._reuse_win = reuse_mode == "window"
        # 复用缓存是"上一支视频"的：pos 里每项都持有 cond/out 两份 GPU 张量（最多
        # 192 项 ≈ 几百 MB），不清就是跨任务显存泄漏，首窗还可能撞上一支的同坐标。
        _ru0 = getattr(self, "_reuse_stats", None)
        if isinstance(_ru0, dict):
            _ru0["pos"].clear()
            _ru0["prev"] = None
            _ru0["out"] = None
            _ru0["run"] = 0
            _ru0["skip"] = 0
            _ru0["verdict"] = None
            if reuse_mode in ("tile", "window") and not os.environ.get("FLASHVSR_REUSE_KEY"):
                # reuse=tile 的静止块判定必须「同坐标跨窗比」；默认 seq 比的是扫描序
                # 相邻的不同 tile，永远不可能命中。
                _ru0["key"] = "pos"
        _wst0 = getattr(self, "_win", None)
        if isinstance(_wst0, dict):
            _wst0["prev"] = None
            _wst0["flag"] = False
        if reuse_mode != "off":
            on_stage("infer", 0, 0,
                     "复用已开启：" + ("整窗跳过 DiT（≈50.6 dB）" if reuse_mode == "window"
                                     else "静止 tile 跳过解码（≈54.9 dB）"))
        audio = src if info["has_audio"] else None
        quality = int(params.get("quality") or 15)
        bitdepth = int(params.get("bitdepth") or 8)
        # 能力探测用**输出分辨率**：老核显能编 H.264 不代表能编 4K
        caps = get_caps(w=int(info["w"]) * int(params["scale"]),
                        h=int(info["h"]) * int(params["scale"]))
        backend = resolve_encoder(params.get("encoder"), caps)
        if bitdepth == 10 and backend not in ("x265", "qsv"):
            # x265 与 hevc_qsv 能出 main10；其余（含 AMF/NVENC）只有 8bit
            on_stage("encode", 0, 0,
                     f"{backend} 不支持 10 bit，本次按 8 bit 输出（要 10 bit 请用 x265 或 qsv）")
            bitdepth = 8
        budget_mode = params.get("budget") or "auto"
        # 开发模式：用户自传编码参数（GPU 编码器的 -rc/-cq/-tune 之类没有统一入口）
        extra_args = []
        raw_extra = str(params.get("extra_args") or "").strip()
        if raw_extra:
            import shlex
            if len(raw_extra) > 300 or "\n" in raw_extra:
                on_stage("encode", 0, 0, "附加参数过长或含换行，已忽略")
            else:
                try:
                    extra_args = shlex.split(raw_extra, posix=False)
                    extra_args = [a.strip('"') for a in extra_args]
                    on_stage("encode", 0, 0, "已附加自定义编码参数：" + " ".join(extra_args))
                except ValueError as e:
                    on_stage("encode", 0, 0, f"附加参数解析失败，已忽略：{e}")

        # 编码器状态：后台编码线程会改、主线程会读，不能用 nonlocal 变量（赋值/读取跨线程
        # 无同步，主线程可能读到旧值，on_stage 与 concat 的 backend 参数就会和实际产物不符）。
        enc_lock = threading.Lock()
        enc_state = {"backend": backend, "bitdepth": bitdepth}

        def _cur_enc():
            with enc_lock:
                return enc_state["backend"], enc_state["bitdepth"]

        def encode_win(frames_in, path, audio_src=None):
            """统一编码入口：沿 FALLBACK_CHAIN 降级（qsv→qsv_h264→x265→cpu），并记住能用的那个。

            这样"哪代核显支持什么"不需要我们预先知道——本机当场试，失败就换下一个，
            且只在第一次失败时付出探测成本（之后 enc_state 已经改成可用的那个）。
            """
            primary, primary_bd = _cur_enc()
            chain = (primary,) + fallback_chain(primary)
            seen, last = set(), "未知错误"
            for be in chain:
                if be in seen:
                    continue
                seen.add(be)
                bd = primary_bd
                if bd == 10 and be not in ("x265", "qsv"):
                    bd = 8                 # 只有 x265 / hevc_qsv 能出 10bit
                try:
                    # svtav1 若不是用户的选择（而是兜底），用 p8：4K 实测 2.7 s/50 帧、
                    # VMAF 96.22，p4 要 7.7 s 才 96.90 —— 兜底不值得为 0.7 dB 多等 3 倍
                    svt_p = "8" if (be == "svtav1" and be != primary) else None
                    encode_frames(frames_in, info["fps"], path, audio_src,
                                  quality=quality, backend=be, bitdepth=bd,
                                  svt_preset=svt_p, tune=params.get("tune"),
                                  extra=extra_args)
                    if be != primary:
                        on_stage("encode", 0, 0,
                                 f"编码器 {primary} 不可用，已改用 {be}（后续段沿用）")
                        with enc_lock:
                            enc_state["backend"], enc_state["bitdepth"] = be, bd
                    return
                except RuntimeError as e:
                    last = (str(e).strip().splitlines() or ["未知错误"])[-1][:160]
                    on_stage("encode", 0, 0, f"{be} 编码失败：{last}")
            raise RuntimeError(f"所有编码器都失败，最后一次：{last}")

        def budget_for(win_u8, kv):
            if budget_mode != "auto" or len(win_u8) < 2:
                return 2.0, 11, kv
            sub = win_u8[:, ::8, ::8, 1]
            if np.issubdtype(np.asarray(sub).dtype, np.floating):
                # hi_precision：[0,1] float32；astype(int16) 会全变 0 → 永远判成"静止"，
                # 每窗都被压到最低预算（sparse 1.5 / local 9 / kv 2.0）。
                small = np.asarray(sub, dtype=np.float32) * 255.0
            else:
                small = np.asarray(sub, dtype=np.int16)
            motion = float(np.abs(np.diff(small.astype(np.float32), axis=0)).mean())
            if motion <= 3.0:
                return 1.5, 9, min(kv, 2.0)
            if motion <= 9.0:
                return 2.0, 11, kv
            return 2.0, 11, max(kv, 2.5)

        def run_win(win, tile, kv, cb=None):
            sp, lr, kk = budget_for(win, kv)
            base_h = tile[0] if isinstance(tile, (tuple, list)) else tile
            base_w = tile[1] if isinstance(tile, (tuple, list)) else tile
            th_t = int(params.get("tile_h") or base_h)
            tw_t = int(params.get("tile_w") or base_w)
            tshape = (th_t, tw_t) if th_t != tw_t else th_t
            res = self._run_window(win, params["scale"], tshape, kk, cb, sparse=sp, local=lr,
                                   tile_overlap=int(params.get("tile_overlap") or 24),
                                   color_fix=params.get("color_fix", True),
                                   seam_fix=str(params.get("seam_fix") or "smooth"))
            return res, sp, lr, kk

        if not chunk or total <= chunk:
            p_drop = _drop_stride(frames) if drop_on else 1
            fin = frames[::p_drop] if p_drop > 1 else frames
            if p_drop > 1:
                on_stage("decode", 0, total,
                         f"抽帧复用：检测到 {p_drop} 拍，{total} → {len(fin)} 帧"
                         f"（省 {(1-len(fin)/max(1,total))*100:.0f}% 计算）")
            self._noise_stride = p_drop
            out_t, sp, lr, kk = run_win(
                fin, params["tile"], params["kv"],
                lambda c, t: on_stage("infer", c, t, f"tile {c}/{t}"))
            self._noise_stride = 1
            frames_out = _expand(out_t[:len(fin)], p_drop, total)
            on_stage("encode", 0, 1, f"编码中（sparse={sp} kv={kk} local={lr}）")
            _tmp_out = _tmp_name(dst)
            encode_win(frames_out, _tmp_out, audio)
            try:
                os.replace(_tmp_out, dst)      # 编码成功才落正式名
            except OSError as e:
                raise RuntimeError(f"编码完成但无法改名为 {dst}：{e}") from e
            on_stage("encode", 1, 1, f"编码完成（{_cur_enc()[0]}）")
            return dict(frames=total, fps=info["fps"], in_wh=[info["w"], info["h"]],
                        out_wh=[int(frames_out.shape[2]), int(frames_out.shape[1])], segments=0)

        src_sig = dict(size=os.path.getsize(src), hash=quick_hash(src))
        pkey = _param_key(params)
        root = seg_root_dir(params)
        # 顺手清掉过期分段缓存（默认 7 天 / 最近 8 个任务，可用环境变量调）
        pruned = prune_segments(root)
        if pruned:
            on_stage("seg", 0, 0, f"已清理 {pruned} 个过期分段缓存（省盘）")
        seg_dir = os.path.join(root, f"{src_sig['hash']}_{pkey}")
        if params.get("restart"):
            shutil.rmtree(seg_dir, ignore_errors=True)
        try:
            os.makedirs(seg_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"无法创建分段缓存目录 {seg_dir}（磁盘满或没有写权限？）：{e}") from e

        man = load_manifest(seg_dir)
        expect = dict(src=src_sig, params_scale=params.get("scale"), params_tile=params.get("tile"),
                      params_kv=params.get("kv"), params_chunk=params.get("chunk"),
                      params_overlap=params.get("overlap"), params_mode=params.get("mode"),
                      params_attention=params.get("attention"), params_quant=params.get("quant"),
                      params_pre_scale=params.get("pre_scale"), total=total,
                      params_tile_overlap=params.get("tile_overlap"),
                      params_ctrl=dict(color_fix=bool(params.get("color_fix", True)),
                                       seam_fix=str(params.get("seam_fix") or "smooth"),
                                       hi_precision=bool(params.get("hi_precision")),
                                       max_frames=int(params.get("max_frames") or 0)))
        same = bool(man) and all(man.get(k) == v for k, v in expect.items())
        if not same:
            shutil.rmtree(seg_dir, ignore_errors=True)
            os.makedirs(seg_dir, exist_ok=True)
            man = dict(expect)
            man["done"] = {}
            save_manifest(seg_dir, man)
        done = man.get("done", {})

        # 上次被强杀/断电可能留下没改名成功的半成品段（*.tmp.mp4）。它们从没进过
        # manifest，留着只占盘，直接清掉。
        try:
            for _name in os.listdir(seg_dir):
                if _name.endswith(".tmp.mp4"):
                    try:
                        os.remove(os.path.join(seg_dir, _name))
                    except OSError:
                        pass
        except OSError:
            pass

        plans = _seg_plans(total, chunk, overlap)
        n_seg = len(plans)
        step = max(1, chunk - overlap)
        # 磁盘预检（只提示不阻断）：分段缓存 + 成片 ≈ 0.4 bpp（CRF 15 实测 ~0.2 bpp，
        # 再留一倍余量）。不够时至少让人先知道，而不是中途 ENOSPC 崩在 ffmpeg 里。
        _disk_note = ""
        try:
            _need = total * (int(info["w"]) * int(params["scale"])) \
                * (int(info["h"]) * int(params["scale"])) * 0.4 / 8
            _free = shutil.disk_usage(seg_dir).free
            if _free < _need:
                _disk_note = (f"；磁盘可能不够：预计约 {_need / 2**30:.1f} GB，"
                              f"剩余 {_free / 2**30:.1f} GB")
        except Exception:  # noqa  预检失败不影响主流程
            pass
        on_stage("seg", 0, n_seg,
                 f"共 {n_seg} 段：{total} 帧，{chunk} 帧/段"
                 f"（含 {overlap} 帧段间重叠 → 每段实际推进 {step} 帧）{_disk_note}")
        man["plans"] = plans
        save_manifest(seg_dir, man)

        resumed = 0
        for k in list(done.keys()):
            try:
                i = int(k)
            except ValueError:
                done.pop(k, None)
                continue
            if not (0 <= i < n_seg):
                done.pop(k, None)
                continue
            keep = plans[i][2] - plans[i][0]
            seg = os.path.join(seg_dir, f"seg_{i:04d}.mp4")
            if not os.path.exists(seg) or probe_frames(seg) != keep:
                done.pop(k, None)
            else:
                resumed += 1
        man["done"] = done
        save_manifest(seg_dir, man)
        if resumed:
            on_stage("seg", resumed, n_seg, f"检测到上次进度：{resumed}/{n_seg} 段已完成，从这里续跑")

        out_wh = None
        seg_paths = []
        async_encode = os.environ.get("FLASHVSR_ASYNC_ENCODE", "1") != "0"
        enc = _SegEncoder(encode_win) if async_encode else None

        # 连续流式解码：一条常驻 ffmpeg 顺着窗口往下读，替代每段 Popen 一次
        # （Windows 起进程 + vulkan/libplacebo 初始化 0.3–0.6 s/段，比软解本身贵）。
        # 整段已在内存（frames is not None）和浮点解码（fp32）路径不走这里。
        stream = None
        dec_pool = None
        pending = None          # (段号, future)：下一段的解码已经在线程里跑（双缓冲）

        def _next_open(after):
            """after（含）之后第一个还没完成的段号。"""
            for j in range(after, n_seg):
                if str(j) not in done:
                    return j
            return None

        def _decode_win(ws, we, prog=None):
            """逐段解码一个窗口（常驻流读短了时的兜底，内含 CFR 兜底）。"""
            return decode_frames_range(src, ws, we - ws, pre_scale=pre_scale,
                                       fps=info["fps"], fp32=fp32,
                                       scaler=params.get("prescale_scaler"),
                                       deband=bool(params.get("prescale_deband")),
                                       deband_grain=params.get("deband_grain"),
                                       dither=str(params.get("prescale_dither") or ""),
                                       on_progress=prog)

        if frames is None and not fp32 and os.environ.get("FLASHVSR_STREAM_DEC", "1") != "0":
            _first = _next_open(0)
            if _first is not None:
                try:
                    stream = _WindowStream(src, info["fps"], plans[_first][1],
                                           pre_scale=pre_scale,
                                           scaler=params.get("prescale_scaler"),
                                           deband=bool(params.get("prescale_deband")),
                                           deband_grain=params.get("deband_grain"),
                                           dither=str(params.get("prescale_dither") or ""))
                    dec_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="dec")
                except Exception as _e:  # noqa
                    print(f"[stream-dec] 常驻解码不可用，退回逐段解码: {_e}", flush=True)
                    stream = None
                    dec_pool = None

        def _finish_seg(idx, seg_path, meta):
            """写盘之后的收尾：校验帧数 → 原子改名 → 记 manifest → 抽预览帧。"""
            keep_n = int(meta.get("keep") or 0)
            got_n = probe_frames(seg_path)
            if got_n != keep_n:
                raise RuntimeError(f"第 {idx + 1} 段校验失败：期望 {keep_n} 帧，实际 {got_n} 帧")
            # 校验通过才把 *.tmp.mp4 改成正式名：进程被杀时只会留下 .tmp，
            # 续跑逻辑永远看不到"半个段"，也就不会拼出坏成片。
            final = seg_path
            if ".tmp" in os.path.basename(seg_path):
                final = os.path.join(os.path.dirname(seg_path),
                                     os.path.basename(seg_path).replace(".tmp", "", 1))
                os.replace(seg_path, final)
            done[str(idx)] = dict(frames=keep_n, bytes=os.path.getsize(final),
                                  at=int(time.time()))
            man["done"] = done
            save_manifest(seg_dir, man)
            pv = None
            pdir = meta.get("pdir")
            if pdir and keep_n > 0:
                try:
                    os.makedirs(pdir, exist_ok=True)
                    name = f"seg_{idx:04d}{preview_ext()}"
                    if extract_preview(final, random.randrange(keep_n), info["fps"],
                                       os.path.join(pdir, name)):
                        pv = name
                except Exception as e:  # noqa
                    print(f"[preview] 第 {idx + 1} 段抽帧失败: {e}", flush=True)
            extra = {"preview": pv, "pv_cur": idx + 1, "pv_total": n_seg} if pv else {}
            on_stage("seg", idx + 1, n_seg,
                     f"段 {idx + 1}/{n_seg} 完成（sparse={meta.get('sp')} "
                     f"kv={meta.get('kv')} local={meta.get('local')}）", **extra)

        def _drain():
            """收掉后台线程已编码完的段；错误在这里抛回主线程。"""
            if enc is None:
                return
            for idx, spath, meta, err in enc.poll():
                if err is not None:
                    raise RuntimeError(f"第 {idx + 1} 段编码失败：{err}")
                _finish_seg(idx, spath, meta)

        try:
            for i, (start, win_start, win_end) in enumerate(plans):
                _drain()
                seg_final = os.path.join(seg_dir, f"seg_{i:04d}.mp4")
                seg_paths.append(seg_final)     # 拼接用正式名（改名由 _finish_seg 完成）
                seg_path = _tmp_name(seg_final)
                keep_from = start - win_start
                keep = win_end - start
                if str(i) in done:
                    on_stage("seg", i + 1, n_seg, f"段 {i + 1}/{n_seg} 已完成，跳过")
                    continue
                if frames is not None:
                    win = frames[win_start:win_end]
                elif stream is not None:
                    on_stage("decode", win_start, total, f"解码 {win_start}–{win_end} 帧（常驻解码）")
                    if pending is not None and pending[0] == i:
                        win = pending[1].result()
                        pending = None
                    else:
                        # 预取错位（上一段被跳过 / 上次异常）→ 丢掉旧结果，同步读本窗
                        if pending is not None:
                            try:
                                pending[1].result()
                            except Exception:  # noqa
                                pass
                            pending = None
                        win = stream.read(win_start, win_end - win_start)
                    if len(win) != win_end - win_start:
                        # 常驻流水读短了（源已到尾，或 VFR 源按帧号取不满这一段）：
                        # 这一窗退回逐段解码，那里有 CFR 兜底与明确的 VFR 报错。
                        win = _decode_win(win_start, win_end, lambda n, ws=win_start: on_stage(
                            "decode", ws + n, total, f"解码 {ws + n}/{total} 帧"))
                    # 本窗已到手，立刻让后台线程去读下一窗：GPU 跑推理时解码在重叠
                    if pending is None:
                        _j = _next_open(i + 1)
                        if _j is not None:
                            _ws, _we = plans[_j][1], plans[_j][2]
                            pending = (_j, dec_pool.submit(stream.read, _ws, _we - _ws))
                else:
                    on_stage("decode", win_start, total, f"解码 {win_start}–{win_end} 帧")
                    win = _decode_win(win_start, win_end, lambda n, ws=win_start: on_stage(
                        "decode", ws + n, total, f"解码 {ws + n}/{total} 帧"))
                if frames is None and len(win) != win_end - win_start:
                    # 帧数与探测不符（典型：VFR 源，passthrough 下源的真实帧数不够）。
                    # 以前只修正本段的 keep、不动后续段起点，结果是成片中间**静默缺帧**。
                    # 宁可失败：
                    got_n = len(win)
                    raise RuntimeError(
                        f"解码帧数与探测不符：第 {i + 1}/{n_seg} 段期望 "
                        f"{win_end - win_start} 帧，实际 {got_n} 帧。\n"
                        f"这是 VFR（可变帧率）源的典型表现。先转成恒定帧率再跑：\n"
                        f'  ffmpeg -i "输入.mp4" -fps_mode cfr -r 30 -c:v libx264 -crf 12 "输入_cfr.mp4"')
                on_stage("seg", i, n_seg, f"段 {i + 1}/{n_seg}：{win_start}–{win_end} 帧")

                n_in = len(win)
                p_drop = _drop_stride(win) if drop_on else 1
                win_in = win[::p_drop] if p_drop > 1 else win
                if p_drop > 1:
                    on_stage("seg", i, n_seg,
                             f"段 {i + 1}/{n_seg} 抽帧：{p_drop} 拍，"
                             f"{n_in} → {len(win_in)} 帧（省 {(1-len(win_in)/n_in)*100:.0f}%）")

                def tile_cb(c, t, i=i):
                    on_stage("seg", i, n_seg, f"段 {i + 1}/{n_seg} tile {c}/{t}")

                self._noise_stride = p_drop
                res, sp, lr, kk = run_win(win_in, params["tile"], params["kv"], tile_cb)
                self._noise_stride = 1
                res = _expand(res[:len(win_in)], p_drop, n_in)
                out_frames = res[keep_from:keep_from + keep]
                out_wh = [int(out_frames.shape[2]), int(out_frames.shape[1])]
                on_stage("encode", 0, 1, f"段 {i + 1}/{n_seg} 写盘（{_cur_enc()[0]}）")
                meta = dict(keep=keep, pdir=params.get("preview_dir"), sp=sp, kv=kk, local=lr)
                if enc is None:
                    encode_win(out_frames, seg_path, None)
                    _finish_seg(i, seg_path, meta)
                else:
                    # 异步写盘：GPU 立刻去跑下一窗，编码在后台线程里与推理重叠
                    enc.submit(i, out_frames, seg_path, meta)
                del res, out_frames, win

            if enc is not None:
                enc.close()      # 等最后一两段写完，再拼接
                _drain()
        finally:
            # 异常路径也必须收掉后台编码线程（否则线程还持着整段输出帧在写盘）
            if enc is not None:
                enc.close()
            if dec_pool is not None:
                dec_pool.shutdown(wait=True)     # 先等预取线程退出，再关解码管道
            if stream is not None:
                stream.close()

        # 全段续跑（resumed == n_seg）时主循环一次都没解，out_wh 还是 None：
        # 直接从第一段探一下尺寸，别把 None 丢给前端。
        if out_wh is None and seg_paths and os.path.exists(seg_paths[0]):
            _sp = probe(seg_paths[0])
            if _sp.get("w") and _sp.get("h"):
                out_wh = [int(_sp["w"]), int(_sp["h"])]

        be_now, bd_now = _cur_enc()      # 后台线程可能已降级过，不能再用旧值
        on_stage("encode", 0, 1, f"拼接 {n_seg} 段 + 混入音轨")
        concat_segments(seg_paths, dst, audio_src=audio, backend=be_now, quality=quality,
                        bitdepth=bd_now, tune=params.get("tune"), extra=extra_args)
        got = probe_frames(dst)
        if got and got != total:
            on_stage("encode", 1, 1,
                     f"拼接完成，但输出 {got} 帧 ≠ 期望 {total} 帧（音画可能差 {(total - got) / info['fps']:.2f}s）")
        else:
            on_stage("encode", 1, 1, f"拼接完成（{be_now}，{got} 帧）")
        if not params.get("keep_seg"):
            shutil.rmtree(seg_dir, ignore_errors=True)
        return dict(frames=total, fps=info["fps"], in_wh=[info["w"], info["h"]],
                    out_wh=out_wh, segments=n_seg, resumed=resumed,
                    frames_out=got if got and got > 0 else None)
```

### `_patch_tile_coords`  （第 1070–1103 行，原样摘录）

```python
def _patch_tile_coords(mod):
    """节点的 calculate_tile_coords 只支持正方形 tile；这里换成支持 (th, tw)。

    只改坐标生成：flashvsr() 后面所有逻辑（裁剪、羽化、画布累加）都由坐标推导，
    所以替换它是安全的。羽化宽度仍由 tile_overlap 决定。
    """
    orig = getattr(mod, "calculate_tile_coords", None)
    if orig is None or getattr(orig, "_rect", False):
        return orig

    def coords(height, width, tile_size, overlap):
        if isinstance(tile_size, (tuple, list)):
            th, tw = int(tile_size[0]), int(tile_size[1])
        else:
            th = tw = int(tile_size)
        stride_h = max(1, th - overlap)
        stride_w = max(1, tw - overlap)
        rows = math.ceil((height - overlap) / stride_h)
        cols = math.ceil((width - overlap) / stride_w)
        out = []
        for r in range(rows):
            for c in range(cols):
                y1, x1 = r * stride_h, c * stride_w
                y2, x2 = min(y1 + th, height), min(x1 + tw, width)
                if y2 - y1 < th:
                    y1 = max(0, y2 - th)
                if x2 - x1 < tw:
                    x1 = max(0, x2 - tw)
                out.append((x1, y1, x2, y2))
        return out

    coords._rect = True
    mod.calculate_tile_coords = coords
    return coords
```

### `_tile_housekeeping`  （第 493–500 行，原样摘录）

```python
def _tile_housekeeping(every=10):
    """每 tile 都 gc+empty_cache 实测慢 7%（HANDOFF §4）；这里只每 N 个 tile 清一次缓存。"""
    _TILE_HK[0] += 1
    if every and _TILE_HK[0] % every == 0:
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa
            pass
```

### `_gpu_canvas_fits`  （第 457–464 行，原样摘录）

```python
def _gpu_canvas_fits(n_frames, h, w, scale, limit_gb=0.8):
    """画布（fp16 NHWC）+ 权重图要多少显存；太大就回退 CPU 画布。"""
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:  # noqa
        return False
    need = n_frames * (h * scale) * (w * scale) * 3 * 2 * 1.5   # canvas + wsum + 余量
    return need < min(limit_gb * 2 ** 30, free * 0.5)
```
