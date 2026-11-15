# -*- coding: utf-8 -*-
"""
FlashVSR 独立面板（aiohttp，无 ComfyUI 依赖）

启动：双击「启动 FlashVSR 面板.bat」，或
      D:\\ComfyUI_windows_portable\\python_embeded\\python.exe app.py
然后浏览器打开 http://127.0.0.1:8288
"""
import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web

# 便携版 python 带 ._pth，脚本目录不会自动进 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine as E  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(APP_DIR, "work")
OUT = os.path.join(APP_DIR, "out")
WORKER = os.path.join(APP_DIR, "worker.py")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(WORK, "segments"), exist_ok=True)

JOBS = {}
EXECUTOR = ThreadPoolExecutor(max_workers=1)
ENGINE = E.Engine()  # 仅用于 /api/status 与历史状态，不影响子进程推理


def new_job(name):
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = dict(
        id=jid, name=name, stage="queued", cur=0, total=0, msg="排队中",
        started=time.time(), done=False, error=None, out=None, info=None, params=None,
    )
    return jid


def safe_name(s):
    s = re.sub(r"[^\w.\-\u4e00-\u9fa5]+", "_", s or "video")
    return s[:120] or "video"


# ---------------------------------------------------------------- 参数解析
def _resolve_params(fields):
    """把表单字段解析成 worker 认识的一组参数（含 tile 对齐 / 2× 降采样换算）。"""
    preset = fields.get("preset", "quality")
    if preset in E.PRESETS:
        p = dict(E.PRESETS[preset])
        scale, tile, kv = p["scale"], p["tile"], p["kv"]
        # 量化后显存省 ~1.26 GB，可以把 tile 提一档（实测 t192/kv2 = 49.5 s vs t160 = 65.4 s）
        if preset in ("quality", "balance"):
            tile = 192 if (fields.get("quant") or "none") != "none" else p["tile"]
    else:
        scale = int(fields.get("scale", 4))
        tile = int(fields.get("tile", 128))
        kv = float(fields.get("kv", 3.0))

    # 节点内部会把每个 tile 的输出裁到 128 的倍数（compute_scaled_and_target_dims），
    # 若 tile*scale 不是 128 的倍数，裁掉的部分会让拼贴几何错位 → 接缝。这里提前对齐。
    step = 128 // math.gcd(128, max(1, int(scale)))
    if tile % step:
        tile = max(step, (tile // step) * step)

    # 输入预缩放：控制「源先 /2 再 4×」这条路径，自定义档也能用
    #   auto → 目标是 2× 时自动 /2（模型走 4×）
    #   half → 强制 源/2 + 4×模型（不管上面填的 scale 是多少，输出恒为 2×）
    #   off  → 不预缩放（scale=2 就是原生 2×，由模型自己放大）
    pre_scale = 1.0
    twox = fields.get("twox") or "downscale"      # 兼容旧字段
    prescale = (fields.get("prescale") or "").strip()
    if prescale not in ("auto", "half", "off"):
        prescale = "auto"
    if prescale == "auto":
        prescale = "half" if (int(scale) == 2 and twox == "downscale") else "off"
    if prescale == "half":
        scale, pre_scale = 4, 0.5
        tile = max(64, tile // 2)          # 输入减半，tile 相应缩小才是同一块画面
        step = 128 // math.gcd(128, int(scale))
        if tile % step:
            tile = max(step, (tile // step) * step)

    chunk = int(fields.get("chunk", 0) or 0)
    mode = fields.get("mode") or "tiny-long"
    if mode not in ("tiny", "tiny-long"):
        mode = "tiny-long"
    attention = fields.get("attention") or "sparse"
    if attention not in ("sparse", "dense"):
        attention = "sparse"
    encoder = fields.get("encoder") or "auto"
    if encoder not in E.ENCODER_CHOICES:
        encoder = "auto"
    quant = fields.get("quant") or "none"
    if quant not in E.QUANT_CHOICES:
        quant = "none"
    budget = fields.get("budget") or "auto"
    if budget not in ("auto", "fixed"):
        budget = "auto"
    try:
        quality = int(float(fields.get("quality") or 15))
    except ValueError:
        quality = 15
    quality = max(10, min(28, quality))

    return dict(scale=scale, tile=tile, kv=kv, chunk=chunk, mode=mode,
                attention=attention, encoder=encoder, quality=quality,
                quant=quant, budget=budget, pre_scale=pre_scale)


# ---------------------------------------------------------------- 待确认任务 / 分段缓存
PEND = os.path.join(WORK, "pending")


def _gc_pending(max_age=24 * 3600):
    """清掉超过 24h 没确认的临时上传（崩溃/关页面留下的）。"""
    if not os.path.isdir(PEND):
        return
    now = time.time()
    for fn in os.listdir(PEND):
        p = os.path.join(PEND, fn)
        try:
            if now - os.path.getmtime(p) > max_age:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
        except OSError:
            pass


def _save_pending(pid, spec):
    os.makedirs(PEND, exist_ok=True)
    with open(os.path.join(PEND, f"{pid}.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)


def _load_pending(pid):
    if not re.fullmatch(r"[0-9a-f]{12}", pid or ""):
        return None
    try:
        with open(os.path.join(PEND, f"{pid}.json"), "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception:
        return None
    if not os.path.isfile(spec.get("src", "")):
        return None
    return spec


def _del_pending(pid):
    try:
        os.remove(os.path.join(PEND, f"{pid}.json"))
    except OSError:
        pass


def _seg_key_dir(src, P, root):
    """与 engine 完全一致的键：源指纹 + 参数指纹。"""
    probe = dict(scale=P["scale"], tile=P["tile"], kv=P["kv"], chunk=P["chunk"], overlap=8,
                 mode=P["mode"], attention=P["attention"], quant=P["quant"],
                 pre_scale=P["pre_scale"])
    return os.path.join(root, f"{E.quick_hash(src)}_{E._param_key(probe)}")


def _resume_info(src, P, root):
    """返回 (已完成分段数, 总分段数, 分段目录)。"""
    d = _seg_key_dir(src, P, root)
    man = E.load_manifest(d) or {}
    # 注意：manifest["total"] 是帧数，分段数要数 plans
    total = len(man.get("plans") or [])
    done = 0
    for k in (man.get("done") or {}):
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if os.path.exists(os.path.join(d, f"seg_{i:04d}.mp4")):
            done += 1
    return done, max(total, done), d


def _dir_size(root):
    n = files = 0
    for base, _dirs, fs in os.walk(root):
        for f in fs:
            try:
                n += os.path.getsize(os.path.join(base, f))
                files += 1
            except OSError:
                pass
    return n, files


async def index(request):
    # 禁缓存：改完 UI 后刷新就能看到，不会拿到旧页面
    return web.FileResponse(os.path.join(APP_DIR, "ui.html"),
                            headers={"Cache-Control": "no-store, must-revalidate"})


async def presets(request):
    return web.json_response({k: v for k, v in E.PRESETS.items()})


async def capabilities(request):
    caps = await asyncio.get_running_loop().run_in_executor(EXECUTOR, E.get_caps)
    return web.json_response({"encoders": caps, "labels": E.ENCODER_LABEL,
                              "choices": list(E.ENCODER_CHOICES)})


async def run(request):
    tmp_path = None
    if request.content_type.startswith("multipart/"):
        reader = await request.multipart()
        fields, fname = {}, None
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "file":
                fname = safe_name(part.filename)
                tmp_path = os.path.join(
                    WORK, f"up_{uuid.uuid4().hex}{os.path.splitext(fname)[1] or '.mp4'}"
                )
                got = 0
                with open(tmp_path, "wb") as f:
                    while True:
                        b = await part.read_chunk(1024 * 1024)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                if got < 1024:
                    raise web.HTTPBadRequest(text="上传不完整（收到 0 字节）")
            else:
                fields[part.name] = await part.text()
        if fname is None:
            raise web.HTTPBadRequest(text="没有收到文件")
    else:
        data = await request.json()
        fields, fname = data, safe_name(data.get("name", "video.mp4"))

    # 续跑确认：前端第二次提交只带 {pending_id, resume_action}，文件已在上一次上传时落盘
    pending_id = (fields.get("pending_id") or "").strip()
    resume_action = (fields.get("resume_action") or "").strip()
    if pending_id:
        spec = _load_pending(pending_id)
        if not spec:
            raise web.HTTPBadRequest(text="待确认的任务已过期（临时文件已清理），请重新选择文件")
        fields = spec["fields"]
        fname = spec["name"]
        tmp_path = spec["src"]
        _del_pending(pending_id)

    P = _resolve_params(fields)
    scale, tile, kv = P["scale"], P["tile"], P["kv"]
    chunk, mode, attention = P["chunk"], P["mode"], P["attention"]
    encoder, quality, quant = P["encoder"], P["quality"], P["quant"]
    budget, pre_scale = P["budget"], P["pre_scale"]

    if resume_action == "restart":
        restart = True
    elif resume_action == "resume":
        restart = False                      # 用户明确选了「继续」，优先于表单里的续跑勾选
    else:
        restart = bool(int(fields.get("restart", 0) or 0))

    # ---- 有历史分段先问：继续上次 / 从头开始 / 忘记缓存 ----
    seg_root = os.path.join(WORK, "segments")
    if tmp_path and chunk > 0:
        done, seg_total, key_dir = _resume_info(tmp_path, P, seg_root)
        if done and not resume_action:
            pid = uuid.uuid4().hex[:12]
            os.makedirs(PEND, exist_ok=True)
            ext = os.path.splitext(fname)[1] or ".mp4"
            dst = os.path.join(PEND, f"{pid}{ext}")
            os.replace(tmp_path, dst)
            _save_pending(pid, dict(src=dst, name=fname, fields=fields))
            return web.json_response(dict(
                need_confirm=True, pending_id=pid, upload=fname,
                done=done, total=seg_total,
                params=dict(scale=scale, tile=tile, kv=kv, quant=quant, pre_scale=pre_scale)))
        if restart or fields.get("forget"):
            shutil.rmtree(key_dir, ignore_errors=True)

    jid = new_job(fname)
    # 只保留最近 50 个已完成任务，避免内存字典无限增长
    if len(JOBS) > 80:
        done_ids = [k for k, v in JOBS.items() if v.get("done")]
        for k in done_ids[: len(JOBS) - 50]:
            JOBS.pop(k, None)
    JOBS[jid]["params"] = dict(scale=scale, tile=tile, kv=kv, chunk=chunk,
                               mode=mode, attention=attention,
                               encoder=encoder, quality=quality,
                               quant=quant, budget=budget, pre_scale=pre_scale)
    waiting = sum(1 for k, j in JOBS.items() if k != jid and not j.get("done"))
    if waiting:
        JOBS[jid]["msg"] = f"排队中（前面还有 {waiting} 个任务）"

    base, ext = os.path.splitext(fname)
    src = os.path.join(WORK, f"{jid}_{fname}")
    if tmp_path:
        os.replace(tmp_path, src)
    out = os.path.join(OUT, f"{base}_{jid}_{scale}x.mp4")

    loop = asyncio.get_running_loop()
    cfg_path = os.path.join(WORK, f"{jid}.json")
    base_params = dict(scale=scale, tile=tile, kv=kv, chunk=chunk, overlap=8,
                       mode=mode, attention=attention,
                       encoder=encoder, quality=quality,
                       quant=quant, budget=budget, pre_scale=pre_scale,
                       restart=restart,
                       seg_dir=os.path.join(WORK, "segments"))

    def run_once(params):
        """跑一次 worker，返回 (exit_code, tail_lines)。"""
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"src": src, "dst": out, "params": params}, f, ensure_ascii=False)
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        tail = []
        proc = subprocess.Popen(
            [sys.executable, "-u", WORKER, cfg_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", cwd=APP_DIR, env=env,
        )
        j = JOBS[jid]
        j["_proc"] = proc
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("@@STAGE "):
                d = json.loads(line[8:])
                j["stage"] = d["stage"]
                j["cur"], j["total"], j["msg"] = d["cur"], d["total"], d["msg"]
            elif line.startswith("@@DONE "):
                j["info"] = json.loads(line[7:])
            elif line.startswith("@@ERR "):
                j["error"] = json.loads(line[6:]).get("error")
            elif line.strip():
                tail.append(line[-400:])
                del tail[:-16]
        return proc.wait(), tail

    def work():
        """子进程跑推理；显存不够就自动降档重试（tile/kv 下调，稠密→稀疏）。"""
        j = JOBS[jid]
        last_tail, code = [], None
        if j.get("cancelled"):
            j.update(done=True, stage="error", msg="已取消", error="排队中被取消，未开始执行")
            return
        for attempt in range(4):
            if j.get("cancelled"):
                break
            params = dict(base_params)
            if attempt:
                params["attention"] = "sparse"
                if attempt == 1:
                    params["tile"] = min(int(params["tile"]), 128)
                    params["kv"] = min(float(params["kv"]), 2.0)
                elif attempt == 2:
                    params["tile"], params["kv"] = 96, 2.0
                else:
                    params["tile"], params["kv"] = 64, 1.5
            j["params"] = dict(scale=params["scale"], tile=params["tile"], kv=params["kv"],
                               chunk=params["chunk"], mode=params["mode"], attention=params["attention"],
                               pre_scale=params.get("pre_scale", 1.0))
            j["attempt"] = attempt
            j["error"] = None          # 每次重试清掉上一次的失败信息，否则成功也会被判失败
            j["info"] = None
            if attempt:
                j["msg"] = (f"显存不足，自动降档重试（第 {attempt} 次）："
                            f"tile={params['tile']}, kv={params['kv']}, "
                            f"注意力={'稀疏' if params['attention'] == 'sparse' else '稠密'}")
            try:
                code, last_tail = run_once(params)
            except Exception as e:  # noqa
                code, last_tail = -1, [f"{type(e).__name__}: {e}"]
            j["elapsed"] = round(time.time() - j["started"], 1)
            if j.get("cancelled") or os.path.exists(out):
                break
            if attempt == 3:
                break
        j["done"] = True
        if j.get("cancelled"):
            j["stage"] = "error"
            j["msg"] = "已取消"
            j["error"] = "任务已取消（已完成的分段会保留，重跑同文件同参数会续跑）"
        elif j.get("error"):
            j["stage"] = "error"
            j["msg"] = "失败"
        elif code == 0 and os.path.exists(out):
            j["out"] = out
            j["stage"] = "done"
            j["msg"] = "完成"
        else:
            j["stage"] = "error"
            j["msg"] = "失败"
            hint = {
                3221225477: "CUDA 访问冲突（0xC0000005）",
                3221225725: "栈溢出（0xC00000FD）",
                3221226505: "栈缓冲区溢出（0xC0000409）",
            }.get(code, f"退出码 {code}")
            j["error"] = (f"推理进程崩溃：{hint}。已自动降档重试 3 次仍失败 —— "
                          f"把 tile 降到 96/64、kv 降到 1.0，或改用「快速」档（2×）再试。"
                          + ("\n\n最后输出：\n" + "\n".join(last_tail[-6:]) if last_tail else ""))

    loop.run_in_executor(EXECUTOR, work)
    return web.json_response({"job": jid})


async def progress(request):
    jid = request.match_info["job"]
    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)
    last = None
    while True:
        j = JOBS.get(jid)
        if not j:
            try:
                await resp.write(b'data: {"stage":"error","msg":"job not found"}\n\n')
            except Exception:
                pass
            break
        payload = {k: j.get(k) for k in
                   ("stage", "cur", "total", "msg", "done", "error", "elapsed", "info", "params")}
        payload["out_name"] = os.path.basename(j["out"]) if j.get("out") else None
        s = json.dumps(payload, ensure_ascii=False)
        if s != last:
            try:
                await resp.write(("data: " + s + "\n\n").encode("utf-8"))
            except Exception:
                return resp  # 浏览器关掉了连接，别把 traceback 打到日志
            last = s
        if j.get("done"):
            break
        await asyncio.sleep(0.4)
    try:
        await resp.write_eof()
    except Exception:
        pass
    return resp


async def result(request):
    jid = request.match_info["job"]
    j = JOBS.get(jid)
    if not j or not j.get("out") or not os.path.exists(j["out"]):
        raise web.HTTPNotFound()
    return await send_file(request, j["out"])


async def send_file(request, path):
    size = os.path.getsize(path)
    rng = request.headers.get("Range")
    if not rng:
        return web.FileResponse(path)
    m = re.match(r"bytes=(\d+)-(\d*)", rng)
    start = int(m.group(1)) if m else 0
    end = int(m.group(2)) if (m and m.group(2)) else size - 1
    end = min(end, size - 1)
    if start > end:
        raise web.HTTPRequestRangeNotSatisfiable()
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start + 1)
    return web.Response(
        body=data, status=206, content_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(data)),
        },
    )


async def download(request):
    jid = request.match_info["job"]
    j = JOBS.get(jid)
    if not j or not j.get("out"):
        raise web.HTTPNotFound()
    return web.FileResponse(j["out"], headers={
        "Content-Disposition": f'attachment; filename="{os.path.basename(j["out"])}"'
    })


async def history(request):
    items = []
    for fn in os.listdir(OUT):
        p = os.path.join(OUT, fn)
        if not os.path.isfile(p):
            continue
        st = os.stat(p)
        items.append(dict(name=fn, size=st.st_size, mtime=int(st.st_mtime)))
    items.sort(key=lambda x: -x["mtime"])
    return web.json_response(items[:30])


async def old_file(request):
    fn = os.path.basename(request.match_info["name"])
    p = os.path.join(OUT, fn)
    if not os.path.isfile(p):
        raise web.HTTPNotFound()
    return await send_file(request, p)


async def delete_out(request):
    fn = os.path.basename(request.match_info["name"])
    p = os.path.join(OUT, fn)
    if not os.path.isfile(p):
        raise web.HTTPNotFound()
    try:
        os.remove(p)
    except OSError as e:
        raise web.HTTPInternalServerError(text=f"删除失败：{e}")
    return web.json_response({"ok": True})


async def clear_history(request):
    if any(not j.get("done") for j in JOBS.values()):
        raise web.HTTPConflict(text="还有任务在跑，等它结束再清")
    n = 0
    for fn in os.listdir(OUT):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p):
            try:
                os.remove(p)
                n += 1
            except OSError:
                pass
    # 顺带清上传缓存 / 残留分段 / 任务 json
    for fn in os.listdir(WORK):
        p = os.path.join(WORK, fn)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        except OSError:
            pass
    return web.json_response({"ok": True, "deleted": n})


async def segments(request):
    """查看分段缓存：总量 + 每个任务目录的进度。"""
    root = os.path.join(WORK, "segments")
    size, files = _dir_size(root)
    dirs = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if not os.path.isdir(p):
                continue
            man = E.load_manifest(p) or {}
            done = len(man.get("done") or {})
            total = int(man.get("total") or len(man.get("plans") or []) or 0)
            b, _ = _dir_size(p)
            dirs.append(dict(key=name, done=done, total=total, mb=round(b / 1048576, 1)))
    return web.json_response(dict(bytes=size, mb=round(size / 1048576, 1),
                                  files=files, dirs=dirs))


async def clear_segments(request):
    """忘记历史缓存：删掉所有分段（下一次同文件同参数会从头算）。"""
    if any(not j.get("done") for j in JOBS.values()):
        raise web.HTTPConflict(text="还有任务在跑，等它结束再清（否则正在写的那一段会损坏）")
    root = os.path.join(WORK, "segments")
    size, files = _dir_size(root)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    return web.json_response(dict(ok=True, freed_mb=round(size / 1048576, 1), files=files))


async def unload(request):
    busy = any(not j.get("done") for j in JOBS.values())
    if busy:
        return web.json_response({"ok": False, "note": "任务进行中，显存被 worker 占用，无法释放（可先取消任务）"})
    return web.json_response({"ok": True, "note": "模型随任务子进程退出已释放；无需手动操作"})


async def cancel(request):
    jid = request.match_info["job"]
    j = JOBS.get(jid)
    if not j:
        raise web.HTTPNotFound()
    if j.get("done"):
        return web.json_response({"ok": True, "note": "已经结束"})
    proc = j.get("_proc")
    j["cancelled"] = True
    killed = False
    if proc is not None:
        try:
            # 用 taskkill 杀进程树（worker 可能还带着 ffmpeg 子进程）
            # taskkill 在中文 Windows 上输出 GBK，指定 errors=replace 否则读线程会抛
            # UnicodeDecodeError（管道不再被消费，长输出时还可能卡死）
            r = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, text=True, errors="replace", timeout=20)
            killed = r.returncode == 0
        except Exception as e:  # noqa
            print(f"[cancel] taskkill 失败: {e}", flush=True)
        if not killed:
            try:
                proc.kill()
                killed = True
            except Exception:
                pass
    # 兜底：按命令行里含 cfg 文件名的方式再找一次孤儿 worker
    return web.json_response({"ok": True, "killed": killed})


async def status(request):
    loaded = ENGINE.pipe is not None
    busy = any(not j.get("done") for j in JOBS.values())
    return web.json_response({"loaded": loaded, "busy": busy})


def main():
    app = web.Application(client_max_size=8 * 1024 ** 3)
    _gc_pending()
    app.add_routes([
        web.get("/", index),
        web.get("/api/presets", presets),
        web.get("/api/capabilities", capabilities),
        web.post("/api/run", run),
        web.get("/api/progress/{job}", progress),
        web.get("/api/result/{job}", result),
        web.get("/api/download/{job}", download),
        web.get("/api/history", history),
        web.get("/api/out/{name}", old_file),
        web.delete("/api/out/{name}", delete_out),
        web.post("/api/history/clear", clear_history),
        web.get("/api/segments", segments),
        web.post("/api/segments/clear", clear_segments),
        web.post("/api/unload", unload),
        web.post("/api/job/cancel/{job}", cancel),
        web.get("/api/status", status),
    ])
    print("FlashVSR 面板：http://127.0.0.1:8288", flush=True)
    web.run_app(app, host="127.0.0.1", port=8288, print=None)


if __name__ == "__main__":
    main()
