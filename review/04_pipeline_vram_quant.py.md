# 管线循环 / VRAM 管理 / 第三方量化接口

> 原样摘录，未做任何改写。行号为该文件内真实行号。

## 来源：`node:src\pipelines\flashvsr_tiny_long.py`（共 609 行）

### `model_fn_wan_video`  （第 535–609 行，原样摘录）

```python
def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    tea_cache: Optional[TeaCache] = None,
    use_unified_sequence_parallel: bool = False,
    LQ_latents: Optional[torch.Tensor] = None,
    is_full_block: bool = False,
    is_stream: bool = False,
    pre_cache_k: Optional[list[torch.Tensor]] = None,
    pre_cache_v: Optional[list[torch.Tensor]] = None,
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    cur_process_idx: int = 0,
    t_mod : torch.Tensor = None,
    t : torch.Tensor = None,
    local_range: int = 9,
    **kwargs,
):
    # patchify
    x, (f, h, w) = dit.patchify(x)

    win = (2, 8, 8)
    seqlen = f // win[0]
    local_num = seqlen
    window_size = win[0] * h * w // 128
    square_num = window_size * window_size
    topk = int(square_num * topk_ratio) - 1
    kv_len = int(kv_ratio)

    # RoPE 位置（分段）
    freqs = build_step_freqs(dit, f, h, w, cur_process_idx, x.device)

    # TeaCache（默认不启用）
    tea_cache_update = tea_cache.check(dit, x, t_mod) if tea_cache is not None else False

    # 统一序列并行（此处默认关闭）
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                             get_sequence_parallel_world_size,
                                             get_sp_group)
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    # Block 堆叠
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        for block_id, block in enumerate(dit.blocks):
            if LQ_latents is not None and block_id < len(LQ_latents):
                x = x + LQ_latents[block_id]
            x, last_pre_cache_k, last_pre_cache_v = block(
                x, context, t_mod, freqs, f, h, w,
                local_num, topk,
                block_id=block_id,
                kv_len=kv_len,
                is_full_block=is_full_block,
                is_stream=is_stream,
                pre_cache_k=pre_cache_k[block_id] if pre_cache_k is not None else None,
                pre_cache_v=pre_cache_v[block_id] if pre_cache_v is not None else None,
                local_range = local_range,
            )
            if pre_cache_k is not None: pre_cache_k[block_id] = last_pre_cache_k
            if pre_cache_v is not None: pre_cache_v[block_id] = last_pre_cache_v

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import get_sp_group
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    return x, pre_cache_k, pre_cache_v
```

### `__call__`  （第 302–478 行，原样摘录）

```python
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt="",
        denoising_strength=1.0,
        seed=None,
        rand_device="gpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(60, 104),
        tile_stride=(30, 52),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="Wan2.1-T2V-1.3B",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        LQ_video=None,
        is_full_block=False,
        if_buffer=False,
        topk_ratio=2.0,
        kv_ratio=3.0,
        local_range = 9,
        color_fix = True,
        unload_dit = False,
        force_offload = False,
    ):
        # 只接受 cfg=1.0（与原代码一致）
        assert cfg_scale == 1.0, "cfg_scale must be 1.0"
        
        # 要求：必须先 init_cross_kv()
        if self.prompt_emb_posi is None or 'context' not in self.prompt_emb_posi:
            raise RuntimeError(
                "Cross-Attn KV 未初始化。请在调用 __call__ 前先执行：\n"
                "    pipe.init_cross_kv()\n"
                "或传入自定义 context：\n"
                "    pipe.init_cross_kv(context_tensor=your_context_tensor)"
            )

        # 尺寸修正
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        # Tiler 参数
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # 初始化噪声
        if if_buffer:
            noise = self.generate_noise((1, 16, (num_frames - 1) // 4, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        else:
            noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        # noise = noise.to(dtype=self.torch_dtype, device=self.device)
        latents = noise

        process_total_num = (num_frames - 1) // 8 - 2
        is_stream = True
        
        if self.prompt_emb_posi['stats'] == "offload":
            self.init_cross_kv(context_tensor=self.prompt_emb_posi['context'])
        self.load_models_to_device(["dit"])
        self.dit.LQ_proj_in.to(self.device)
        self.TCDecoder.to(self.device)

        # 清理可能存在的 LQ_proj_in cache
        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()

        frames_total = []
        LQ_pre_idx = 0
        LQ_cur_idx = 0
        self.TCDecoder.clean_mem()

        with torch.no_grad():
            for cur_process_idx in progress_bar_cmd(range(process_total_num)):
                if cur_process_idx == 0:
                    pre_cache_k = [None] * len(self.dit.blocks)
                    pre_cache_v = [None] * len(self.dit.blocks)
                    LQ_latents = None
                    inner_loop_num = 7
                    for inner_idx in range(inner_loop_num):
                        cur = self.denoising_model().LQ_proj_in.stream_forward(
                            LQ_video[:, :, max(0, inner_idx*4-3):(inner_idx+1)*4-3, :, :].to(self.device)
                        ) if LQ_video is not None else None
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    LQ_cur_idx = (inner_loop_num-1)*4-3
                    cur_latents = latents[:, :, :6, :, :]
                else:
                    LQ_latents = None
                    inner_loop_num = 2
                    for inner_idx in range(inner_loop_num):
                        cur = self.denoising_model().LQ_proj_in.stream_forward(
                            LQ_video[:, :, cur_process_idx*8+17+inner_idx*4:cur_process_idx*8+21+inner_idx*4, :, :].to(self.device)
                        ) if LQ_video is not None else None
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    LQ_cur_idx = cur_process_idx*8+21+(inner_loop_num-2)*4
                    cur_latents = latents[:, :, 4+cur_process_idx*2:6+cur_process_idx*2, :, :]
                        
                # 推理（无 motion_controller / vace）
                noise_pred_posi, pre_cache_k, pre_cache_v = model_fn_wan_video(
                    self.dit,
                    x=cur_latents,
                    timestep=self.timestep,
                    context=None,
                    tea_cache=None,
                    use_unified_sequence_parallel=False,
                    LQ_latents=LQ_latents,
                    is_full_block=is_full_block,
                    is_stream=is_stream,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    topk_ratio=topk_ratio,
                    kv_ratio=kv_ratio,
                    cur_process_idx=cur_process_idx,
                    t_mod=self.t_mod,
                    t=self.t,
                    local_range = local_range,
                )

                # 更新 latent
                cur_latents = cur_latents - noise_pred_posi
                
                # Decode
                cur_LQ_frame = LQ_video[:,:,LQ_pre_idx:LQ_cur_idx,:,:].to(self.device)
                cur_frames = self.TCDecoder.decode_video(
                    cur_latents.transpose(1, 2),
                    parallel=False,
                    show_progress_bar=False,
                    cond=cur_LQ_frame).transpose(1, 2).mul_(2).sub_(1)

                # 颜色校正（wavelet）
                try:
                    if color_fix:
                        cur_frames = self.ColorCorrector(
                            cur_frames.to(device=self.device),
                            cur_LQ_frame,
                            clip_range=(-1, 1),
                            chunk_size=None,
                            method='adain'
                        )
                except:
                    pass
                
                frames_total.append(cur_frames.to('cpu'))
                LQ_pre_idx = LQ_cur_idx
                
                if unload_dit:
                    del noise_pred_posi, cur_frames, cur_latents, cur_LQ_frame
                    clean_vram()
                    
            if hasattr(self.dit, "LQ_proj_in"):
                self.dit.LQ_proj_in.clear_cache()
                
            self.TCDecoder.clean_mem()
            if force_offload:
                self.offload_model()
            
            frames = torch.cat(frames_total, dim=2)
            
        return frames[0]
```

## 来源：`node:src\pipelines\base.py`（共 130 行）

### `load_models_to_device`  （第 94–123 行，原样摘录）

```python
    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        # fresh the cuda cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
```

## 来源：`node:src\vram_management\layers.py`（共 95 行）

### `cast_to`  （第 5–8 行，原样摘录）

```python
def cast_to(weight, dtype, device):
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight)
    return r
```

### `AutoWrappedModule`  （第 11–38 行，原样摘录）

```python
class AutoWrappedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device):
        super().__init__()
        self.module = module.to(dtype=offload_dtype, device=offload_device)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            module = self.module
        else:
            module = copy.deepcopy(self.module).to(dtype=self.computation_dtype, device=self.computation_device)
        return module(*args, **kwargs)
```

### `AutoWrappedLinear`  （第 41–71 行，原样摘录）

```python
class AutoWrappedLinear(torch.nn.Linear):
    def __init__(self, module: torch.nn.Linear, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device):
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None, dtype=offload_dtype, device=offload_device)
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, x, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            weight, bias = self.weight, self.bias
        else:
            weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
            bias = None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
        return torch.nn.functional.linear(x, weight, bias)
```

### `enable_vram_management`  （第 92–94 行，原样摘录）

```python
def enable_vram_management(model: torch.nn.Module, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None):
    enable_vram_management_recursively(model, module_map, module_config, max_num_param, overflow_module_config, total_num_param=0)
    model.vram_management_enabled = True
```

## 来源：`D:\ComfyUI_windows_portable\python_embeded\Lib\site-packages\comfy_kitchen\backends\cuda\__init__.py`（共 4392 行）

### `quantize_mxfp8`  （第 1729–1762 行，原样摘录）

```python
def quantize_mxfp8(
    x: torch.Tensor,
    pad_32x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"

    orig_rows, orig_cols = x.shape
    if pad_32x:
        num_rows = roundup(orig_rows, 32)
        num_cols = roundup(orig_cols, 32)
    else:
        num_rows, num_cols = orig_rows, orig_cols
        assert num_rows % 32 == 0, f"num_rows must be divisible by 32, got {num_rows}"
        assert num_cols % 32 == 0, f"num_cols must be divisible by 32, got {num_cols}"

    qx = torch.empty((num_rows, num_cols), device=x.device, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format)

    scale_rows = roundup(num_rows, 128)
    scale_cols = roundup(num_cols // 32, 4)
    sx_uint8 = torch.zeros((scale_rows, scale_cols), device=x.device, dtype=torch.uint8)

    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    _C.quantize_mxfp8(
        _wrap_for_dlpack(x),
        _wrap_for_dlpack(qx),
        _wrap_for_dlpack(sx_uint8),
        pad_32x,
        stream_ptr,
    )

    # View uint8 scales as float8_e8m0fnu before returning
    sx = sx_uint8.view(torch.float8_e8m0fnu)

    return qx, sx
```

### `_gemm_vector_arg`  （第 502–508 行，原样摘录）

```python
def _gemm_vector_arg(v: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A per-channel vector as the CUTLASS epilogues read it: on device, in dtype,
    contiguous and 16-byte aligned (the row-broadcast visitors load 16-byte vectors)."""
    if v.device == device and v.dtype == dtype and v.is_contiguous() and _aligned16(v):
        return v
    v = v.to(device=device, dtype=dtype).contiguous()
    return v if _aligned16(v) else v.clone()
```

### `_wrap_for_dlpack`  （第 526–541 行，原样摘录）

```python
def _wrap_for_dlpack(tensor: torch.Tensor):
    """Export tensor via DLPack without cross-stream sync.

    Works around PyTorch issue where __dlpack__(stream=None) syncs with
    the default stream, breaking CUDA graph capture on non-default streams.
    See: https://github.com/pytorch/pytorch/pull/163242

    Detaches first so nn.Parameter (requires_grad=True) inputs like bias
    export without PyTorch's gradient-tracking refusal.

    Returns a PyCapsule containing the DLTensor that nanobind can import.
    """
    # stream=-1 tells PyTorch to skip synchronization (DLPack spec)
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.__dlpack__(stream=-1)
```

### `_aligned16`  （第 498–499 行，原样摘录）

```python
def _aligned16(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0
```
