# -*- coding: utf-8 -*-
"""
FlashVSR 独立面板（aiohttp，无 ComfyUI 依赖）

启动：双击「启动 FlashVSR 面板.bat」，或
      D:\\ComfyUI_windows_portable\\python_embeded\\python.exe app.py
然后浏览器打开 http://127.0.0.1:8288
"""
import asyncio
import json
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

    jid = new_job(fname)
    # 只保留最近 50 个已完成任务，避免内存字典无限增长
    if len(JOBS) > 80:
        done_ids = [k for k, v in JOBS.items() if v.get("done")]
        for k in done_ids[: len(JOBS) - 50]:
            JOBS.pop(k, None)
    JOBS[jid]["params"] = dict(scale=scale, tile=tile, kv=kv, chunk=chunk,
                               mode=mode, attention=attention,
                               encoder=encoder, quality=quality,
                               quant=quant, budget=budget)
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
                       quant=quant, budget=budget,
                       restart=bool(int(fields.get("restart", 0) or 0)),
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
                               chunk=params["chunk"], mode=params["mode"], attention=params["attention"])
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
            r = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, text=True, timeout=20)
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
        web.post("/api/unload", unload),
        web.post("/api/job/cancel/{job}", cancel),
        web.get("/api/status", status),
    ])
    print("FlashVSR 面板：http://127.0.0.1:8288", flush=True)
    web.run_app(app, host="127.0.0.1", port=8288, print=None)


if __name__ == "__main__":
    main()
