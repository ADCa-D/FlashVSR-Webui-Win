# -*- coding: utf-8 -*-
"""
FlashVSR 独立面板（aiohttp，无 ComfyUI 依赖）

启动：双击「启动 FlashVSR 面板.bat」，或
      D:\\ComfyUI_windows_portable\\python_embeded\\python.exe app.py
然后浏览器打开 http://127.0.0.1:8288
"""
import asyncio
import base64
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import ssl
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

# ---------------------------------------------------------------- 鉴权
# 规则：本机回环免登录；其他 IP 必须（a）在放行名单里，或（b）持有有效会话 cookie。
# 登录成功后把该 IP 记入放行名单，下次同 IP 直接进。
TLS_CERT = os.environ.get(
    "FLASHVSR_TLS_CERT", r"C:\Users\MECHREVO\Desktop\server\certs\server.crt")
TLS_KEY = os.environ.get(
    "FLASHVSR_TLS_KEY", r"C:\Users\MECHREVO\Desktop\server\certs\server.key")
AUTH_USERS = os.path.join(WORK, "auth_users.json")
AUTH_IPS = os.path.join(WORK, "auth_ips.json")
AUTH_SECRET = os.path.join(WORK, "auth_secret.txt")
PWD_HINT = os.path.join(WORK, "首次登录密码.txt")
COOKIE = "fs_sess"
COOKIE_AGE = 30 * 86400


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


PBKDF2_ITER = 200_000


def _hash_pw(salt, pwd, it=PBKDF2_ITER):
    return hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), salt.encode("utf-8"), it).hex()


def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _secret():
    if os.path.isfile(AUTH_SECRET):
        try:
            with open(AUTH_SECRET, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            pass
    s = uuid.uuid4().hex + uuid.uuid4().hex
    with open(AUTH_SECRET, "w", encoding="utf-8") as f:
        f.write(s)
    return s


SECRET = _secret()


def _init_users():
    """首次启动：建 admin 账号，密码随机并打印（也可用环境变量指定）。"""
    users = _load_json(AUTH_USERS, {})
    env_user = os.environ.get("FLASHVSR_USER")
    env_pass = os.environ.get("FLASHVSR_PASS")
    if env_user and env_pass:
        salt = uuid.uuid4().hex[:16]
        users[env_user] = dict(salt=salt, iter=PBKDF2_ITER, hash=_hash_pw(salt, env_pass))
        _save_json(AUTH_USERS, users)
        return users
    if not users:
        pwd = uuid.uuid4().hex[:10]
        salt = uuid.uuid4().hex[:16]
        users["admin"] = dict(salt=salt, iter=PBKDF2_ITER, hash=_hash_pw(salt, pwd))
        _save_json(AUTH_USERS, users)
        # 首次密码同时写一份到文件：控制台容易被关掉/日志被覆盖，丢了就只能重置
        try:
            with open(PWD_HINT, "w", encoding="utf-8") as f:
                f.write(f"用户名: admin\n密码: {pwd}\n"
                        f"（登录成功一次后本文件自动删除；要改密码就删 work/auth_users.json 重启）\n")
        except OSError:
            pass
        print("=" * 56, flush=True)
        print(f"  已创建登录账号   用户名: admin   密码: {pwd}", flush=True)
        print(f"  （记录在 {AUTH_USERS}，可用 FLASHVSR_USER/FLASHVSR_PASS 覆盖）", flush=True)
        print("=" * 56, flush=True)
    return users


USERS = _init_users()


def _sign(payload):
    sig = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return base64.urlsafe_b64encode(f"{payload}|{sig}".encode()).decode()


def _unsign(token):
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        payload, sig = raw.rsplit("|", 1)
    except Exception:
        return None
    want = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig, want):
        return None
    user, exp = payload.split("|", 1)
    if time.time() > float(exp):
        return None
    return user


def _is_local(ip):
    return ip in ("127.0.0.1", "::1", "::ffff:127.0.0.1")


def _ip_allowed(ip):
    return ip in _load_json(AUTH_IPS, {})


def _allow_ip(ip, user):
    data = _load_json(AUTH_IPS, {})
    data[ip] = dict(user=user, at=int(time.time()))
    _save_json(AUTH_IPS, data)


def _cookie_user(request):
    tok = request.cookies.get(COOKIE)
    return _unsign(tok) if tok else None


@web.middleware
async def auth_middleware(request, handler):
    path = request.path
    if path in ("/login", "/favicon.ico"):
        return await handler(request)
    ip = request.remote or ""
    if _is_local(ip) or _ip_allowed(ip) or _cookie_user(request):
        return await handler(request)
    if path.startswith("/api/"):
        return web.json_response({"error": "未登录"}, status=401)
    raise web.HTTPFound("/login")


LOGIN_HTML = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>登录 · FlashVSR</title>
<style>
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
 background:#0f1115;color:#e8ecf3;font:14px/1.6 "Segoe UI","Microsoft YaHei",sans-serif}
form{background:#171a21;border:1px solid #262b36;border-radius:14px;padding:26px 28px;width:320px}
h1{font-size:16px;margin:0 0 18px}
label{display:block;color:#8b93a5;font-size:12.5px;margin:12px 0 4px}
input{width:100%;box-sizing:border-box;background:#0f131b;border:1px solid #262b36;color:#e8ecf3;
 border-radius:8px;padding:8px 10px;font-size:14px}
button{margin-top:18px;width:100%;background:#5b8cff;color:#fff;border:0;border-radius:10px;
 padding:10px;font-size:14px;font-weight:600;cursor:pointer}
.err{color:#ff5f5f;font-size:12.5px;margin-top:12px;min-height:18px}
.note{color:#5c6579;font-size:11.5px;margin-top:14px}
</style></head><body>
<form method="post" action="/login">
  <h1>FlashVSR 面板登录</h1>
  <label>用户名</label><input name="username" autocomplete="username" autofocus>
  <label>密码</label><input name="password" type="password" autocomplete="current-password">
  <button type="submit">登录</button>
  <div class="err">%ERR%</div>
  <div class="note">登录成功后本机 IP 会被记入放行名单，下次免密。</div>
</form></body></html>"""


async def login_page(request):
    return web.Response(text=LOGIN_HTML.replace("%ERR%", ""), content_type="text/html")


async def login_submit(request):
    data = await request.post()
    user = (data.get("username") or "").strip()
    pwd = data.get("password") or ""
    rec = USERS.get(user)
    ok = False
    if rec:
        if rec.get("iter"):
            ok = hmac.compare_digest(rec["hash"], _hash_pw(rec["salt"], pwd, rec["iter"]))
        else:                                   # 旧格式（sha256），验证后自动升级
            ok = hmac.compare_digest(rec["hash"], _sha(rec["salt"] + pwd))
            if ok:
                rec.update(iter=PBKDF2_ITER, hash=_hash_pw(rec["salt"], pwd))
                _save_json(AUTH_USERS, USERS)
    if not ok:
        await asyncio.sleep(0.6)          # 拖慢暴力尝试
        html = LOGIN_HTML.replace("%ERR%", "用户名或密码不对")
        return web.Response(text=html, content_type="text/html", status=401)
    ip = request.remote or ""
    _allow_ip(ip, user)                    # 记住这个 IP
    try:                                   # 已成功登录一次，删掉明文提示文件
        if os.path.isfile(PWD_HINT):
            os.remove(PWD_HINT)
    except OSError:
        pass
    resp = web.HTTPFound("/")
    resp.set_cookie(COOKIE, _sign(f"{user}|{int(time.time()) + COOKIE_AGE}"),
                    max_age=COOKIE_AGE, httponly=True, samesite="Lax",
                    secure=request.scheme == "https")
    print(f"[auth] 登录成功 user={user} ip={ip}", flush=True)
    return resp


async def logout(request):
    resp = web.HTTPFound("/login")
    resp.del_cookie(COOKIE)
    return resp


async def auth_ips(request):
    if not _is_local(request.remote or ""):
        raise web.HTTPForbidden()
    return web.json_response(_load_json(AUTH_IPS, {}))


async def auth_ips_clear(request):
    if not _is_local(request.remote or ""):
        raise web.HTTPForbidden()
    _save_json(AUTH_IPS, {})
    return web.json_response({"ok": True})



def _pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                             capture_output=True, text=True, errors="replace", timeout=10)
        return str(int(pid)) in (out.stdout or "")
    except Exception:
        return False


def _adopt_orphans():
    """面板重启后，把还没跑完的 worker 认领回来（进度不可见，但知道它在跑）。"""
    try:
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Where-Object { $_.CommandLine -match 'worker\\.py' } | "
              "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, errors="replace", timeout=30)
        for line in (out.stdout or "").splitlines():
            line = line.strip()
            m = re.match(r"^(\d+)\s+.*?work[\\/]([0-9a-f]{12})\.json", line)
            if not m:
                continue
            pid, jid = int(m.group(1)), m.group(2)
            if jid in JOBS or not _pid_alive(pid):
                continue
            JOBS[jid] = dict(
                id=jid, name="(重启前启动的任务)", stage="seg", cur=0, total=0,
                msg=f"面板重启前启动，进度不可见（worker PID {pid}）；跑完会出现在历史里",
                started=time.time(), done=False, error=None, out=None, info=None,
                params=None, orphan=True, pid=pid,
            )
            print(f"[adopt] 认领 worker PID {pid} → job {jid}", flush=True)
    except Exception as e:  # noqa
        print(f"[adopt] 跳过：{type(e).__name__}: {e}", flush=True)


def _prune_orphans():
    for jid, j in list(JOBS.items()):
        if j.get("orphan") and not j.get("done") and not _pid_alive(j.get("pid", 0)):
            j["done"] = True
            j["stage"] = "done"
            j["msg"] = "重启前的任务已结束（成品在历史里）"


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
def _resolve_params(fields, src_path=None):
    return E.resolve_params(fields, src_path)


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


def _gc_preview(max_age=7 * 86400):
    """清掉一周前的分段预览图（work/preview/<jid>/）。"""
    root = os.path.join(WORK, "preview")
    if not os.path.isdir(root):
        return
    now = time.time()
    for d in os.listdir(root):
        p = os.path.join(root, d)
        try:
            if os.path.isdir(p) and now - os.path.getmtime(p) > max_age:
                shutil.rmtree(p, ignore_errors=True)
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
                            headers={"Cache-Control": "no-store, must-revalidate",
                                     "Content-Type": "text/html; charset=utf-8"})


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

    P = _resolve_params(fields, tmp_path)
    scale, tile, kv = P["scale"], P["tile"], P["kv"]
    chunk, mode, attention = P["chunk"], P["mode"], P["attention"]
    encoder, quality, quant = P["encoder"], P["quality"], P["quant"]
    budget, pre_scale = P["budget"], P["pre_scale"]
    tile_overlap = P["tile_overlap"]
    preview = P["preview"]
    auto_note = P.get("auto_note")

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
                               quant=quant, budget=budget, pre_scale=pre_scale,
                               tile_overlap=tile_overlap)
    waiting = sum(1 for k, j in JOBS.items() if k != jid and not j.get("done"))
    if waiting:
        JOBS[jid]["msg"] = f"排队中（前面还有 {waiting} 个任务）"
    if auto_note:
        JOBS[jid]["auto_note"] = auto_note
        JOBS[jid]["msg"] = auto_note

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
                       tile_overlap=tile_overlap, bitdepth=P["bitdepth"],
                       hi_precision=P["hi_precision"],
                       preview_dir=(os.path.join(WORK, "preview", jid) if preview else None),
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
                if d.get("preview"):
                    j["preview"] = dict(file=f"{jid}/{os.path.basename(str(d['preview']))}",
                                        cur=d.get("pv_cur"), total=d.get("pv_total"))
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
                   ("stage", "cur", "total", "msg", "done", "error", "elapsed", "info",
                    "params", "preview")}
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


async def preview_frame(request):
    """分段预览帧（每段随机抽一帧）。"""
    jid = request.match_info["job"]
    name = os.path.basename(request.match_info["name"])
    if not re.fullmatch(r"[0-9a-f]{12}", jid or ""):
        raise web.HTTPNotFound()
    p = os.path.join(WORK, "preview", jid, name)
    if not os.path.isfile(p):
        raise web.HTTPNotFound()
    return web.FileResponse(p, headers={"Cache-Control": "no-store"})


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
            total = len(man.get("plans") or []) or int(man.get("total") or 0)
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


async def current_job(request):
    """正在跑/最近的任务，用于刷新页面后自动接上进度。"""
    _prune_orphans()
    for jid, j in JOBS.items():
        if not j.get("done"):
            return web.json_response({"job": jid, "name": j.get("name"),
                                      "stage": j.get("stage"),
                                      "orphan": bool(j.get("orphan"))})
    return web.json_response({"job": None})


async def status(request):
    loaded = ENGINE.pipe is not None
    busy = any(not j.get("done") for j in JOBS.values())
    return web.json_response({"loaded": loaded, "busy": busy})


def main():
    app = web.Application(client_max_size=8 * 1024 ** 3, middlewares=[auth_middleware])
    _gc_pending()
    _gc_preview()
    _adopt_orphans()
    app.add_routes([
        web.get("/login", login_page),
        web.post("/login", login_submit),
        web.get("/logout", logout),
        web.get("/api/auth/ips", auth_ips),
        web.delete("/api/auth/ips", auth_ips_clear),
        web.get("/", index),
        web.get("/api/presets", presets),
        web.get("/api/capabilities", capabilities),
        web.post("/api/run", run),
        web.get("/api/progress/{job}", progress),
        web.get("/api/preview/{job}/{name}", preview_frame),
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
        web.get("/api/current", current_job),
    ])
    asyncio.run(serve(app))


async def serve(app):
    """同时监听：127.0.0.1 明文（本机免登录）+ EasyTier IP 上的 HTTPS（要登录）。

    必须让面板自己终结 TLS：字节级代理会把所有客户端都显示成 127.0.0.1，
    按 IP 放行就形同虚设。
    """
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    sites = [("本机", web.TCPSite(runner, "127.0.0.1", 8288))]

    tls_host = os.environ.get("FLASHVSR_TLS_HOST", "66.66.66.67")
    cert, key = TLS_CERT, TLS_KEY
    if os.path.isfile(cert) and os.path.isfile(key):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        sites.append(("网络", web.TCPSite(runner, tls_host, 8288, ssl_context=ctx)))
        # 手机等装不了自签 CA 的客户端：再开一个明文端口（仍要登录，仅限 EasyTier 内网）
        plain_port = int(os.environ.get("FLASHVSR_PLAIN_PORT", "8289"))
        sites.append(("网络(明文)", web.TCPSite(runner, tls_host, plain_port)))
    else:
        print(f"[提示] 没找到证书（{cert}），只开本机明文端口", flush=True)

    for label, s in sites:
        try:
            await s.start()
            if label == "本机":
                print("FlashVSR 面板（本机，免登录）：http://127.0.0.1:8288", flush=True)
            elif label == "网络":
                print(f"FlashVSR 面板（网络，需登录）：https://{tls_host}:8288", flush=True)
            else:
                print(f"FlashVSR 面板（手机/无证书客户端，明文需登录）："
                      f"http://{tls_host}:{plain_port}", flush=True)
        except OSError as e:
            if label == "本机":
                # 本机端口都绑不上（通常是另一个面板实例还在跑）→ 直接退出，别静默空转
                print(f"[错误] 本机端口 8288 绑定失败：{e}", flush=True)
                raise
            # 典型场景：EasyTier 还没连上，66.66.66.67 不存在 → 不能因此拖死本机端口
            print(f"[警告] {label}端口启动失败（{e}）；其它入口继续工作", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    main()
