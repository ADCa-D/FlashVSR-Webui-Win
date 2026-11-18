# -*- coding: utf-8 -*-
"""把审查需要的源码原样抽取出来（按函数/类边界，带行号），分成 4 份。

不复制粘贴、不摘要：用 ast 定位节点，按原文切片，保证与仓库一致。
输出 -> D:\\flashvsr-web\\review\\NN_topic.py.md
"""
import ast
import os

NODE = r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast"
ENGINE = r"D:\flashvsr-web\engine.py"
CK = r"D:\ComfyUI_windows_portable\python_embeded\Lib\site-packages\comfy_kitchen\backends\cuda\__init__.py"
OUT = r"D:\flashvsr-web\review"

# (输出文件, 标题, [(绝对路径, [顶层定义名...])])
PARTS = [
    ("01_dit_hotpath.py.md", "DiT 热路径（每步 30 block 的实际算子）", [
        (NODE + r"\src\models\wan_video_dit.py", [
            "RMSNorm", "AttentionModule", "WindowPartition3D", "flash_attention",
            "rope_apply", "_local_mask_additive", "_mask_sig", "_select_topk_mask",
            "generate_draft_block_mask", "generate_draft_block_mask_sage",
            "build_step_freqs", "SelfAttention", "DiTBlock",
        ]),
    ]),
    ("02a_engine_schedule.py.md", "引擎：外层调度（分段 / 分窗 / tile / 复用）", [
        (ENGINE, ["_run_window", "upscale_video", "_patch_tile_coords",
                  "_tile_housekeeping", "_gpu_canvas_fits"]),
    ]),
    ("02b_engine_dataflow.py.md", "引擎：tile 预处理 / 画布混合 / 权重与激活量化", [
        (ENGINE, ["_feather_mask_nhwc", "_prepare_input_gpu", "flashvsr_gpu_canvas",
                  "install_stream_loader", "quantize_dit_mxfp8",
                  "_qcache_load", "_qcache_save"]),
        (NODE + r"\nodes.py", ["prepare_input_tensor", "create_feather_mask"]),
    ]),
    ("03_vae_tcdecoder.py.md", "VAE / TCDecoder（解码段的数据搬运）", [
        (NODE + r"\src\models\wan_video_vae.py", [
            "CausalConv3d", "RMS_norm", "Upsample", "Resample",
            "VideoVAE_", "WanVideoVAE",
        ]),
        (NODE + r"\src\models\TCDecoder.py", ["MemBlock", "TAEHV"]),
    ]),
    ("04_pipeline_vram_quant.py.md", "管线循环 / VRAM 管理 / 第三方量化接口", [
        (NODE + r"\src\pipelines\flashvsr_tiny_long.py", ["model_fn_wan_video", "__call__"]),
        (NODE + r"\src\pipelines\base.py", ["load_models_to_device"]),
        (NODE + r"\src\vram_management\layers.py", [
            "cast_to", "AutoWrappedModule", "AutoWrappedLinear", "enable_vram_management",
        ]),
        (CK, ["quantize_mxfp8", "_gemm_vector_arg", "_wrap_for_dlpack", "_aligned16"]),
    ]),
]


def collect(path, names):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found[node.name] = node
    # 类方法 / 被 if 包住的定义：ast.walk 兜底（不覆盖已找到的顶层定义）
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.setdefault(node.name, node)
    out = []
    for name in names:
        node = found.get(name)
        if node is None:
            # 也可能在类里，或者名字带前缀（如 scaled_mm_*）
            cand = [k for k in found if k.startswith(name)]
            if not cand:
                out.append((name, None, None))
                continue
            node = found[cand[0]]
            name = cand[0]
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        end = node.end_lineno
        out.append((name, start, end, "\n".join(lines[start - 1:end])))
    return out, len(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    index = []
    for fname, title, entries in PARTS:
        chunks = []
        total = 0
        for path, names in entries:
            rel = path.replace(NODE + "\\", "node:").replace(ENGINE, "engine.py")
            segs, nlines = collect(path, names)
            chunks.append(f"## 来源：`{rel}`（共 {nlines} 行）\n")
            for seg in segs:
                if seg[1] is None:
                    chunks.append(f"### {seg[0]} —— **未找到**\n")
                    continue
                name, start, end, body = seg
                total += end - start + 1
                chunks.append(f"### `{name}`  （第 {start}–{end} 行，原样摘录）\n")
                chunks.append("```python\n" + body + "\n```\n")
        with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n> 原样摘录，未做任何改写。行号为该文件内真实行号。\n\n")
            f.write("\n".join(chunks))
        size = os.path.getsize(os.path.join(OUT, fname)) / 1024
        index.append((fname, title, total, size))
        print("%-28s %5d 行代码  %6.1f KB" % (fname, total, size))
    return index


if __name__ == "__main__":
    main()
