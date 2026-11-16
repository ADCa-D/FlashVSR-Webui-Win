# -*- coding: utf-8 -*-
"""FlashVSR 命令行入口（与面板共用 engine.resolve_params，参数语义一致）。

示例：
  python inference.py -s in.mp4 -o out.mp4 --preset quality
  python inference.py -s in.mp4 -o out.mp4 --preset fast --prescale half --quant mxfp8
  python inference.py -s in.mp4 -o out.mp4 --tile 128 --kv 3 --chunk 0 --encoder nvenc --crf 18
  python inference.py -s D:\\素材 -o D:\\成品 --batch          # 批量目录（跳过已存在）
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts")


def build_params(args, src):
    fields = dict(preset=args.preset)
    if args.preset == "custom":
        fields.update(scale=args.scale, tile=args.tile, kv=args.kv)
    fields.update(chunk=args.chunk, max_frames=args.frames,
                  quant=args.quant, encoder=args.encoder,
                  bitdepth=args.bitdepth, hi_precision=1 if args.hi_precision else 0,
                  color_fix=0 if args.no_color_fix else 1, seam_fix=args.seam_fix,
                  tune=args.tune, extra_args=args.extra, prescale_scaler=args.scaler,
                  quality=args.crf, prescale=args.prescale,
                  tile_overlap=args.overlap, mode=args.mode,
                  attention=args.attention, budget=args.budget)
    p = E.resolve_params(fields, src)
    p.pop("auto_note", None)
    p.update(restart=False, seg_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "work", "segments"))
    return p


def out_name(path, args):
    """试跑（--frames）产物加后缀，避免被当成全片成品。"""
    if not getattr(args, "frames", 0):
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_preview{args.frames}f{ext}"


def run_one(eng, src, dst, args):
    params = build_params(args, src)
    note = params.pop("auto_note", None)
    print(f"\n=== {os.path.basename(src)} ===")
    if note:
        print(f"  {note}")
    print(f"  scale={params['scale']} tile={params['tile']} kv={params['kv']} "
          f"pre={params['pre_scale']} chunk={params['chunk']} quant={params['quant']}")
    t0 = time.time()
    info = eng.upscale_video(src, dst, params, lambda *a: None)
    print(f"  完成：{info['frames']} 帧 {info['out_wh'][0]}×{info['out_wh'][1]} "
          f"用时 {time.time() - t0:.1f}s → {dst}")
    return info


def main():
    ap = argparse.ArgumentParser(description="FlashVSR 命令行超分")
    ap.add_argument("-s", "--src", required=True, help="输入视频或目录（配 --batch）")
    ap.add_argument("-o", "--dst", required=True, help="输出文件或目录（配 --batch）")
    ap.add_argument("--batch", action="store_true", help="把 src 当目录，批量处理并跳过已存在输出")
    ap.add_argument("--preset", default="quality", choices=list(E.PRESETS) + ["custom"])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--kv", type=float, default=3.0)
    ap.add_argument("--overlap", type=int, default=24, help="tile 重叠（羽化带，越大接缝越稳）")
    ap.add_argument("--chunk", type=int, default=48, help="帧/段，0=不分段")
    ap.add_argument("--frames", type=int, default=0,
                    help="只处理前 N 帧（试跑用，0=全片）")
    ap.add_argument("--quant", default="mxfp8", choices=list(E.QUANT_CHOICES))
    ap.add_argument("--prescale", default="auto", choices=["auto", "half", "off"],
                    help="源预缩放：auto=目标 2× 时先 /2")
    ap.add_argument("--encoder", default="auto", choices=list(E.ENCODER_CHOICES))
    ap.add_argument("--crf", type=int, default=15)
    ap.add_argument("--bitdepth", type=int, default=8, choices=[8, 10],
                    help="10 位输出（仅 x265 支持；保留模型浮点精度）")
    ap.add_argument("--mode", default="tiny-long", choices=["tiny", "tiny-long"])
    ap.add_argument("--attention", default="sparse", choices=["sparse", "dense"])
    ap.add_argument("--budget", default="auto", choices=["auto", "fixed"])
    ap.add_argument("--tune", default="", choices=list(E.TUNE_ALL),
                    help="CPU 编码器调优：ssim/psnr（x264/x265/svtav1）、vq（svtav1）、"
                         "film/animation/grain/stillimage/fastdecode/zerolatency（x264 系）")
    ap.add_argument("--extra", default="", help="附加编码参数（开发模式，整串透传）")
    ap.add_argument("--scaler", default="", choices=[""] + list(E.SCALERS),
                    help="预缩放所用的 libplacebo 缩放器（默认 ewa_lanczos4sharpest）")
    ap.add_argument("--seam-fix", default="smooth", choices=["smooth", "align", "off"],
                    help="tile 拼接方式：smooth=平滑羽化（默认）/ align=加 DC 对齐（实验）/ off=线性")
    ap.add_argument("--no-color-fix", action="store_true",
                    help="关掉逐 tile 颜色对齐（实测会明显掉色，仅排障用）")
    ap.add_argument("--hi-precision", action="store_true",
                    help="浮点解码（gbrpf32le）：YUV→RGB 全程 float32，不经 8bit 量化；内存 4×")
    args = ap.parse_args()

    eng = E.Engine()
    if not args.batch:
        run_one(eng, args.src, out_name(args.dst, args), args)
        return 0

    os.makedirs(args.dst, exist_ok=True)
    files = [f for f in sorted(glob.glob(os.path.join(args.src, "*")))
             if f.lower().endswith(VIDEO_EXT)]
    if not files:
        print(f"[错误] {args.src} 下没有视频文件")
        return 1
    done, skip = 0, 0
    for i, f in enumerate(files, 1):
        out = out_name(os.path.join(args.dst,
                                    os.path.splitext(os.path.basename(f))[0] + "_x.mp4"), args)
        if os.path.exists(out) and os.path.getsize(out) > 1024:
            print(f"[{i}/{len(files)}] 跳过（已存在）{os.path.basename(out)}")
            skip += 1
            continue
        try:
            run_one(eng, f, out, args)
            done += 1
        except Exception as e:  # noqa
            print(f"[{i}/{len(files)}] 失败 {os.path.basename(f)}: {type(e).__name__}: {e}")
    print(f"\n批量结束：完成 {done}，跳过 {skip}，共 {len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
