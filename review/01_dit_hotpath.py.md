# DiT 热路径（每步 30 block 的实际算子）

> 原样摘录，未做任何改写。行号为该文件内真实行号。

## 来源：`node:src\models\wan_video_dit.py`（共 977 行）

### `RMSNorm`  （第 444–455 行，原样摘录）

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight
```

### `AttentionModule`  （第 458–465 行，原样摘录）

```python
class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attention_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attention_mask=attention_mask)
        return x
```

### `WindowPartition3D`  （第 100–119 行，原样摘录）

```python
class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B,F,H,W,C)."""
    @staticmethod
    def partition(x: torch.Tensor, win: Tuple[int, int, int]):
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        assert F % wf == 0 and H % wh == 0 and W % ww == 0, "Dims must divide by window size."
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(windows: torch.Tensor, win: Tuple[int, int, int], orig: Tuple[int, int, int]):
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)
```

### `flash_attention`  （第 285–363 行，原样摘录）

```python
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attention_mask=None, return_KV=False):
    if attention_mask is not None:
        seqlen = q.shape[1]
        seqlen_kv = k.shape[1]
        if USE_BLOCK_ATTN and BLOCK_ATTN_AVAILABLE:
            q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
            v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)
        else:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        streaming_info = None
        base_blockmask = attention_mask
        p_dropout = 0.0
        if USE_BLOCK_ATTN and BLOCK_ATTN_AVAILABLE:
            # 这三个常量张量只有 block_sparse_attn 需要；以前在分支外无条件创建，
            # 稀疏（sage）路径每层白做 3 次 CPU->GPU 拷贝（30 层 = 90 次/步），也会打断 CUDA Graph。
            cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
            cu_seqlens_k = torch.tensor([0, seqlen_kv], device=k.device, dtype=torch.int32)
            head_mask_type = torch.tensor([1] * num_heads, device=q.device, dtype=torch.int32)
            max_seqlen_q_ = seqlen
            max_seqlen_k_ = seqlen_kv
            x = block_sparse_attn_func(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                head_mask_type,
                streaming_info,
                base_blockmask,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                deterministic=False,
                softmax_scale=None,
                is_causal=False,
                exact_streaming=False,
                return_attn_probs=False,
            ).unsqueeze(0)
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        else:
            x = sparse_sageattn(
                q, k, v,
                mask_id=base_blockmask.to(torch.int8),
                is_causal=False,
                tensor_layout="HND"
            )
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x
```

### `rope_apply`  （第 433–438 行，原样摘录）

```python
def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)
```

### `_local_mask_additive`  （第 122–155 行，原样摘录）

```python
@torch.no_grad()
def _local_mask_additive(local_attn_mask, sig, repeat_len, repeat_num, repeat_head,
                         kv_double, scores_shape, dtype=torch.float32, device=None):
    """把 [bh, bw] 的局部窗 mask 展开成 scores 可直接广播相加的加性 mask。

    展开过程与入参形状一一对应，30 个 block 每步都要重复算一遍且内容完全相同 → 按参数缓存。
      · sig          : 由 SelfAttention 传入 (bh, bw, local_range, include_self)，**不含 .item()**，
                       否则首次展开要 30 次 GPU->CPU 同步，且会在 CUDA Graph 捕获期直接报错
      · kv_double    : sage 路径把 K 拆成前后两半、kv 长度翻倍，需要 repeat_interleave(2)；
                       非 sage（generate_draft_block_mask）不能做，否则末维是 scores 的 2 倍
      · 返回 [1, L, S]，head 维交给 broadcast：省掉 repeat_head 倍的拷贝与带宽
    """
    dev = device if device is not None else local_attn_mask.device
    key = (sig, repeat_len, repeat_num, repeat_head, bool(kv_double),
           tuple(scores_shape), dtype, str(dev))
    hit = _LOCAL_MASK_CACHE.get(key)
    if hit is not None:
        return hit
    bh, bw = local_attn_mask.shape
    m = local_attn_mask.to(dev).unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    # 直接 reshape 才等于原 einops 'x a y b -> (x a) (y b)'：
    # 布局是 [repeat_len, bh, repeat_num, bw]，最内两维 (rn, bw) 连续，
    # 所以按 (rl*bh, rn*bw) 压扁后 行=(x,a)、列=(y,b) 正是原顺序。
    # 不能先 permute(0,2,1,3)——那会变成 行=(x,y,a)，repeat_num>1 且 bh>1 时整行错位
    # （全 True 的掩码测不出来，小 tile 实测密度就是 1.0，所以之前没暴露）。
    m = m.reshape(repeat_len * bh, repeat_num * bw)
    if kv_double:
        m = m.repeat_interleave(2, dim=1)
    add = torch.zeros((1, m.shape[0], m.shape[1]), dtype=dtype, device=dev)
    add.masked_fill_(~m.unsqueeze(0), -float('inf'))
    if len(_LOCAL_MASK_CACHE) > 32:
        _LOCAL_MASK_CACHE.clear()
    _LOCAL_MASK_CACHE[key] = add
    return add
```

### `_mask_sig`  （第 158–169 行，原样摘录）

```python
def _mask_sig(local_attn_mask, sig):
    """sig 由调用方给（生产路径）；只给 shape 的旧式调用才退回内容指纹（会同步一次）。"""
    if sig is not None:
        return sig
    ent = _MASK_FP.get(id(local_attn_mask))
    if ent is not None and ent[0] is local_attn_mask:
        return ent[1]
    fp = (tuple(local_attn_mask.shape), int(local_attn_mask.sum().item()))
    if len(_MASK_FP) > 64:
        _MASK_FP.clear()
    _MASK_FP[id(local_attn_mask)] = (local_attn_mask, fp)   # 持有引用，防止 id 被复用
    return fp
```

### `_select_topk_mask`  （第 176–191 行，原样摘录）

```python
def _select_topk_mask(flat, apply_topk, loop_num, s1, s2):
    """每行从 flat 里选出要保留的块。

    FLASHVSR_MASK_TOPK=1 时用 topk 的 indices + scatter_（严格 k 个，行内一定非空）；
    默认保持 "> 第 k+1 大值" 的旧行为（逐位不变，便于 A/B）。
    """
    if apply_topk <= 0:
        return torch.zeros_like(flat, dtype=torch.bool).reshape(loop_num, s1, s2)
    if _MASK_TOPK_INDICES:
        idx = torch.topk(flat, k=apply_topk, dim=1, largest=True, sorted=False).indices
        m = torch.zeros_like(flat, dtype=torch.bool)
        m.scatter_(1, idx, True)
        return m.reshape(loop_num, s1, s2)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    return (flat > thresholds).reshape(loop_num, s1, s2)
```

### `generate_draft_block_mask`  （第 197–231 行，原样摘录）

```python
@torch.no_grad()
def generate_draft_block_mask(batch_size, nheads, seqlen,
                              q_w, k_w, topk=10, local_attn_mask=None, sig=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    avgpool_q = torch.mean(q_w, dim=1) 
    avgpool_k = torch.mean(k_w, dim=1)
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]
    # 非 sage：K 没有拆两半，mask 不能 repeat_interleave(2)
    add_mask = _local_mask_additive(local_attn_mask, _mask_sig(local_attn_mask, sig),
                                    repeat_len, repeat_num, repeat_head, False,
                                    scores.shape, torch.float32, scores.device)
    scores = scores + add_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    n = flat.shape[1]
    apply_topk = min(flat.shape[1]-1, topk)
    mask_new = _select_topk_mask(flat, apply_topk, loop_num, s1, s2)
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)  # keep shape note
    # 修正：上行变量名统一
    # mask_new = rearrange(attn_map, 'h (it s1) s2 -> h (it s1) s2', it=seqlen) * 0 + mask_new
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask
```

### `generate_draft_block_mask_sage`  （第 234–279 行，原样摘录）

```python
@torch.no_grad()
def generate_draft_block_mask_sage(batch_size, nheads, seqlen,
                                      q_w, k_w, topk=10, local_attn_mask=None, sig=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    
    avgpool_q = torch.mean(q_w, dim=1) 
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    
    k_w_split = k_w.view(k_w.shape[0], 2, 64, k_w.shape[2])
    avgpool_k_split = torch.mean(k_w_split, dim=2)
    avgpool_k_refined = rearrange(avgpool_k_split, 's two d -> (s two) d', two=2) # shape: (s*2, C)
    avgpool_k_refined = rearrange(avgpool_k_refined, 's (h d) -> s h d', h=nheads) # shape: (s*2, h, d)
    k_heads_doubled = avgpool_k_refined.permute(1, 0, 2) # shape: (h, s*2, d)
    
    k_heads_1, k_heads_2 = torch.chunk(k_heads_doubled, 2, dim=1)
    scores_1 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_1) / math.sqrt(D)
    scores_2 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_2) / math.sqrt(D)
    scores = torch.cat([scores_1, scores_2], dim=-1)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = (scores.shape[2] // 2) // local_attn_mask.shape[1]

    # 之前每层都要重建一遍这张大 mask（repeat + rearrange + repeat_interleave + float32 提升），
    # 30 层内容相同 → 按参数缓存；sage 路径 K 拆成两半，kv 维要翻倍
    add_mask = _local_mask_additive(local_attn_mask, _mask_sig(local_attn_mask, sig),
                                    repeat_len, repeat_num, repeat_head, True,
                                    scores.shape, torch.float32, scores.device)
    assert add_mask.shape[-2:] == scores.shape[-2:] and add_mask.shape[0] in (1, scores.shape[0]), \
        f"Mask shape {tuple(add_mask.shape)} 无法广播到 scores {tuple(scores.shape)}"
    scores = scores + add_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1]-1, topk)

    mask_new = _select_topk_mask(flat, apply_topk, loop_num, s1, s2)
        
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask
```

### `build_step_freqs`  （第 400–425 行，原样摘录）

```python
def build_step_freqs(dit, f, h, w, cur_process_idx=0, device=None):
    dev = device if device is not None else next(dit.parameters()).device
    axis_f, axis_h, axis_w = dit.freqs
    off = 0 if cur_process_idx == 0 else 4 + cur_process_idx * 2
    if off + f > axis_f.shape[0]:
        # 预计算表默认只有 1024 个时间位置（precompute_freqs_cis_3d 的 end 默认值）。
        # 超过时原代码会切出短于 f 的片段，然后在 .view(f,1,1,-1) 报一个看不懂的形状错误；
        # 这里按需把 f 轴扩到够用（RoPE 只是 i*theta，外推是合法的）。
        need = off + f + 64
        axis_f = precompute_freqs_cis(axis_f.shape[-1] * 2, need)
        dit.freqs = (axis_f, axis_h, axis_w)
    # key 在扩表之后算，且只带「每轴列数」（表长 end 不影响取值，带进 key 只会永远 miss）
    key = (f, h, w, cur_process_idx, int(axis_f.shape[-1]), int(axis_h.shape[-1]),
           int(axis_w.shape[-1]), str(dev))
    hit = _STEP_FREQS.get(key)
    if hit is not None:
        return hit
    fr = torch.cat([
        axis_f[off:off + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        axis_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        axis_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(dev)
    if len(_STEP_FREQS) > 8:
        _STEP_FREQS.clear()
    _STEP_FREQS[key] = fr
    return fr
```

### `SelfAttention`  （第 468–555 行，原样摘录）

```python
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
        self.local_attn_mask = None

    def forward(self, x, freqs, f=None, h=None, w=None, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range = 9):
        B, L, D = x.shape
        if is_stream and pre_cache_k is not None and pre_cache_v is not None:
            assert f==2, "f must be 2"
        if is_stream and (pre_cache_k is None or pre_cache_v is None):
            assert f==6, " start f must be 6"
        assert L == f * h * w, "Sequence length mismatch with provided (f,h,w)."

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        win = (2, 8, 8)
        q = q.view(B, f, h, w, D)
        k = k.view(B, f, h, w, D)
        v = v.view(B, f, h, w, D)

        q_w = WindowPartition3D.partition(q, win)
        k_w = WindowPartition3D.partition(k, win)
        v_w = WindowPartition3D.partition(v, win)

        seqlen = f//win[0]
        one_len = k_w.shape[0] // B // seqlen
        if pre_cache_k is not None and pre_cache_v is not None:
            k_w = torch.cat([pre_cache_k, k_w], dim=0)
            v_w = torch.cat([pre_cache_v, v_w], dim=0)

        block_n = q_w.shape[0] // B
        block_s = q_w.shape[1]
        block_n_kv = k_w.shape[0] // B

        reorder_q = q_w.reshape(B, block_n * block_s, D)
        reorder_k = k_w.reshape(B, block_n_kv * block_s, D)
        reorder_v = v_w.reshape(B, block_n_kv * block_s, D)

        window_size = win[0]*h*w//128

        if self.local_attn_mask is None or self.local_attn_mask_h!=h//8 or self.local_attn_mask_w!=w//8 or self.local_range!=local_range:
            self.local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(h//8, w//8, local_range, local_range, include_self=True, device=k_w.device)
            self.local_attn_mask_h = h//8
            self.local_attn_mask_w = w//8
            self.local_range = local_range
        # 掩码的全部输入参数：用它当缓存键，避免额外 .item() 同步（也是 CUDA Graph 捕获的前提）
        mask_sig = (h // 8, w // 8, int(local_range), True)
        if USE_BLOCK_ATTN and BLOCK_ATTN_AVAILABLE:
            attention_mask = generate_draft_block_mask(B, self.num_heads, seqlen, q_w, k_w, topk=topk, local_attn_mask=self.local_attn_mask, sig=mask_sig)
        else:
            attention_mask = generate_draft_block_mask_sage(B, self.num_heads, seqlen, q_w, k_w, topk=topk, local_attn_mask=self.local_attn_mask, sig=mask_sig)

        x = self.attn(reorder_q, reorder_k, reorder_v, attention_mask)

        cur_block_n, cur_block_s, _ = k_w.shape
        cache_num = cur_block_n // one_len
        if kv_len is not None and cache_num > kv_len:
            cache_k = k_w[one_len:, :, :]
            cache_v = v_w[one_len:, :, :]
        else:
            cache_k = k_w
            cache_v = v_w

        x = x.reshape(x.shape[0] * block_n, block_s, D)
        x = WindowPartition3D.reverse(x, win, (f, h, w))
        x = x.view(B, f*h*w, D)

        if is_stream:
            return self.o(x), cache_k, cache_v
        return self.o(x)
```

### `DiTBlock`  （第 613–654 行，原样摘录）

```python
class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, f, h, w, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range = 9):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_ret = self.self_attn(
            input_x, freqs, f, h, w, local_num, topk, train_img, block_id,
            kv_len=kv_len, is_full_block=is_full_block, is_stream=is_stream,
            pre_cache_k=pre_cache_k, pre_cache_v=pre_cache_v, local_range = local_range)
        # SelfAttention 非流式只返回单张量：无条件 3 元组解包会 ValueError；
        # 更阴的是 B==3 时把 batch 维拆成 3 个变量，静默算错
        if is_stream:
            self_attn_output, self_attn_cache_k, self_attn_cache_v = attn_ret
        else:
            self_attn_output = attn_ret

        x = self.gate(x, gate_msa, self_attn_output)
        x = x + self.cross_attn(self.norm3(x), context, is_stream=is_stream)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        if is_stream:
            return x, self_attn_cache_k, self_attn_cache_v
        return x
```
