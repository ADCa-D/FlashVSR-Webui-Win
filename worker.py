# -*- coding: utf-8 -*-
"""推理工作进程：由 app.py 以子进程方式拉起，跑完即退（显存随之释放）。
崩溃（含 CUDA 原生 abort）不会带走面板本体，父进程会收到非 0 退出码。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine as E  # noqa: E402

# 内存回收守卫：本进程存活期间让 D:\tools\mem-reclaim 跳过全局回收
# （EmptyWorkingSets 是全局的，会把推理进程的工作集踢进页面文件）
PAUSE_FLAG = r"D:\tools\mem-reclaim\pause.flag"


def set_pause(on: bool):
    try:
        if on:
            with open(PAUSE_FLAG, "w", encoding="ascii") as f:
                f.write(str(os.getpid()))
        elif os.path.exists(PAUSE_FLAG):
            os.remove(PAUSE_FLAG)
    except Exception:
        pass


def emit(tag, payload):
    print("@@" + tag + " " + json.dumps(payload, ensure_ascii=False), flush=True)


def stage(st, cur, total, msg, **extra):
    payload = {"stage": st, "cur": cur, "total": total, "msg": msg}
    payload.update(extra)
    emit("STAGE", payload)


def main():
    cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    set_pause(True)
    try:
        info = E.Engine().upscale_video(cfg["src"], cfg["dst"], cfg["params"], stage)
        emit("DONE", info)
        return 0
    except Exception as e:  # noqa
        emit("ERR", {"error": f"{type(e).__name__}: {str(e)[:600]}"})
        return 3
    finally:
        set_pause(False)


if __name__ == "__main__":
    sys.exit(main())
