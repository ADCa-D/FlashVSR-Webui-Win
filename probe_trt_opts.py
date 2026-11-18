# -*- coding: utf-8 -*-
"""用 FlashVSR trt_decoder.py 里那套完全相同的 TRT EP 选项跑一个小模型，
确认：选项合法、engine cache 生效、provider 真的是 TensorRT。"""
import os
import time

import torch

os.makedirs("_trtopt", exist_ok=True)
ONNX = "_trtopt/probe.onnx"
CACHE = os.path.abspath("_trtopt/engines")
os.makedirs(CACHE, exist_ok=True)

if not os.path.exists(ONNX):
    m = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, padding=1),
                            torch.nn.LeakyReLU(0.1),
                            torch.nn.Conv2d(64, 64, 3, padding=1)).cuda().half().eval()
    x = torch.zeros(1, 64, 64, 64, device="cuda", dtype=torch.float16)
    torch.onnx.export(m, x, ONNX, input_names=["x"], output_names=["y"], opset_version=17,
                      dynamo=False)

try:
    import tensorrt_libs
    os.environ["PATH"] = os.path.dirname(tensorrt_libs.__file__) + os.pathsep + os.environ["PATH"]
except Exception:
    pass

import onnxruntime as ort

so = ort.SessionOptions()
so.log_severity_level = 3
opts = {
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": CACHE,
    "trt_max_workspace_size": int(os.environ.get("FLASHVSR_TRT_WS", "3072")) << 20,
    "trt_builder_optimization_level": 3,
}
t0 = time.perf_counter()
sess = ort.InferenceSession(ONNX, so, providers=[("TensorrtExecutionProvider", opts)])
print("build %.1fs" % (time.perf_counter() - t0))
print("providers:", sess.get_providers())
print("engines:", os.listdir(CACHE))

import numpy as np
x = np.zeros((1, 64, 64, 64), np.float16)
t0 = time.perf_counter()
for _ in range(5):
    sess.run(None, {"x": x})
print("5 次推理 %.1f ms/次" % ((time.perf_counter() - t0) / 5 * 1000))
