"""Measures batch-1 latency for a finished evaluation and adds it to its results JSON (same code as pub_eval.py --latency).

Use this when the evaluation itself ran on a busy machine: run it afterwards, with nothing else running.
    python latency_only.py --dataset cifar10 --seed 0 --data data
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "04_Practical_Implementation"))
import lcd_core as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="cifar10")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arch", default="resnet20")
ap.add_argument("--data", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
torch.set_num_threads(a.threads)
DEV = a.device

tag = f"{a.dataset}_{a.arch}_s{a.seed}"
res_path = HERE / "results" / f"{tag}.json"
R = json.loads(res_path.read_text(encoding="utf-8"))
cfg = {k: v for k, v in R["frozen_config"].items() if k != "conf_thr"}
model = L.make_model(a.arch)
model.load_state_dict(torch.load(HERE / "checkpoints" / f"{a.dataset}_{a.arch}_clean_s{a.seed}.pt", map_location=DEV))
model.to(DEV).eval()
_, _, (xte, _) = L.load_dataset(a.dataset, a.data)


def ms(fn, xs):
    for i in range(5):
        fn(xs[i:i + 1])
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(len(xs)):
        fn(xs[i:i + 1])
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(xs) * 1000


xl = xte[:200]
lat = {"none": ms(lambda z: L.baseline_predict(model, z, "none", device=DEV), xl),
       "LCD": ms(lambda z: L.lcd_predict(model, z, cfg, device=DEV), xl),
       "always_stage2": ms(lambda z: L.baseline_predict(model, z, "always_stage2", cfg["n_views"], device=DEV, bits=cfg["bits"]), xl)}
lat.update(LCD_x=lat["LCD"] / lat["none"], always_stage2_x=lat["always_stage2"] / lat["none"])
R["latency_ms"] = lat
res_path.write_text(json.dumps(R, indent=2), encoding="utf-8")
print(tag, {k: round(v, 2) for k, v in lat.items()})
