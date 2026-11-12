"""Run diag.py for every model, packing jobs onto free GPUs within the disk budget.

Each job downloads its weights, evaluates, then deletes the weights. A job starts only when enough
GPUs are idle and the sum of the download sizes of running jobs stays under the disk budget.
Usage: python src/run_all.py [--disk-gb 95] [--gpus 0,1,2,3] [models ...]   (logs in $B/logs/<model>.log)
"""
import argparse, os, subprocess, sys, time
from models import MODELS, BY_NAME

B = os.environ["B"]
ap = argparse.ArgumentParser()
ap.add_argument("--disk-gb", type=float, default=95)
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("models", nargs="*")
a = ap.parse_args()
todo = [BY_NAME[m] for m in a.models] if a.models else list(MODELS)
todo = [m for m in todo if not os.path.exists(f"{B}/results/models/{m['name']}/meta.json")]
free, running, failed, tries = a.gpus.split(","), {}, [], {}
os.makedirs(f"{B}/logs", exist_ok=True)
while todo or running:
    for name, (p, gpus, gb) in list(running.items()):
        if p.poll() is not None:
            free += gpus; del running[name]
            print(time.strftime("%H:%M"), "finished" if p.returncode == 0 else f"FAILED ({p.returncode})", name, flush=True)
            if p.returncode:
                tries[name] = tries.get(name, 0) + 1
                if tries[name] <= 2:  # transient errors (DNS, Hub timeouts): retry twice
                    todo.append(BY_NAME[name])
                else:
                    failed.append(name)
    used = sum(gb for _, _, gb in running.values())
    for m in list(todo):
        if len(free) >= m["gpus"] and used + m["size_gb"] <= a.disk_gb:
            gpus, free = free[:m["gpus"]], free[m["gpus"]:]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
            log = open(f"{B}/logs/{m['name']}.log", "w")
            p = subprocess.Popen([sys.executable, f"{B}/src/diag.py", "--model", m["name"]], env=env, stdout=log, stderr=subprocess.STDOUT)
            running[m["name"]] = (p, gpus, m["size_gb"]); used += m["size_gb"]; todo.remove(m)
            print(time.strftime("%H:%M"), "started", m["name"], "on GPUs", ",".join(gpus), flush=True)
    time.sleep(20)
print("ALL_DONE", "failed:", failed or "none", flush=True)
