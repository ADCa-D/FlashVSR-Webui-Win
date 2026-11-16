# -*- coding: utf-8 -*-
"""看 ck.quantize_mxfp8 的输出长什么样（缓存要按这个格式落盘）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import comfy_kitchen as ck

w = torch.randn(512, 1536, device="cuda", dtype=torch.bfloat16)
wq, wb = ck.quantize_mxfp8(w)
print("权重 fp8 :", wq.dtype, tuple(wq.shape), "字节/元素", wq.element_size())
print("块指数   :", wb.dtype, tuple(wb.shape), "字节/元素", wb.element_size())
print("原始     :", w.dtype, tuple(w.shape))
print("压缩比   :", (w.numel() * 2) / (wq.numel() * wq.element_size()
                                      + wb.numel() * wb.element_size()))
print("safetensors 版本:", __import__("safetensors").__version__)
print("支持 F8_E4M3 :", "F8_E4M3" in str(__import__("safetensors").torch))
