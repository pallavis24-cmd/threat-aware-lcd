"""Backdoor track for the paper: BadNets at 1%, 5%, 10% poison (CIFAR-10) plus a clean model as negative control.

For every model: clean accuracy; trigger attack-success rate (ASR) without defence, with squeezing, with always-on Stage 2
and with Threat-Aware LCD (frozen configuration of the clean model, temperature refitted); gate routing of triggered
inputs; and two offline audits -- activation clustering [Chen et al. 2019] and Neural Cleanse [Wang et al. 2019].

    python pub_backdoor.py --data <cifar root> [--device cuda] [--arch resnet20]
Writes results/backdoor_<arch>.json. Needs results/cifar10_<arch>_s0.json (frozen LCD configuration).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "04_Practical_Implementation"))
import lcd_core as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--arch", default="resnet20")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--nc_steps", type=int, default=150)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads)
DEV = a.device
N_TEST = 200 if a.smoke else 2000
NC_STEPS = 3 if a.smoke else a.nc_steps
log_f = open(HERE / "logs" / f"backdoor_{a.arch}{'_smoke' if a.smoke else ''}.txt", "w", encoding="utf-8")


def log(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()


def neural_cleanse(model, x_clean, steps, n_classes=10, lam=1e-2, lr=0.1, bs=128, seed=0):
    """Wang et al. (S&P 2019): for each class, find the smallest mask+pattern that sends clean images to that class.
    A backdoor target needs an abnormally small mask; anomaly index = (median - L1) / (1.4826 * MAD)."""
    g = torch.Generator().manual_seed(seed)
    norms, success = [], []
    for t in range(n_classes):
        m_raw = torch.full((1, 1, 32, 32), -3.0, device=DEV, requires_grad=True)
        p_raw = torch.zeros((1, 3, 32, 32), device=DEV, requires_grad=True)
        opt = torch.optim.Adam([m_raw, p_raw], lr=lr)
        tgt = torch.full((bs,), t, dtype=torch.long, device=DEV)
        for _ in range(steps):
            idx = torch.randint(0, len(x_clean), (bs,), generator=g)
            xb = x_clean[idx].to(DEV)
            m, p = torch.sigmoid(m_raw), torch.sigmoid(p_raw)
            loss = F.cross_entropy(model((1 - m) * xb + m * p), tgt) + lam * m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            m, p = torch.sigmoid(m_raw), torch.sigmoid(p_raw)
            xb = x_clean[:500].to(DEV)
            success.append(float((model((1 - m) * xb + m * p).argmax(1) == t).float().mean()))
            norms.append(float(m.sum()))
    norms = np.array(norms)
    med = np.median(norms); mad = np.median(np.abs(norms - med)) * 1.4826 + 1e-9
    anomaly = (med - norms) / mad                       # large positive = abnormally small trigger
    flagged = int(np.argmax(anomaly))
    return {"mask_l1": norms.round(2).tolist(), "reversed_trigger_success": success, "anomaly_index": anomaly.round(3).tolist(),
            "flagged_class": flagged, "max_anomaly": float(anomaly.max()), "detected": bool(anomaly.max() > 2.0)}


t0 = time.time()
(xtr, ytr), (xva, yva), (xte, yte) = L.load_dataset("cifar10", a.data)
cfg0 = json.loads((HERE / "results" / f"cifar10_{a.arch}_s0.json").read_text(encoding="utf-8"))["frozen_config"]
BITS = cfg0["bits"]
xt, yt = xte[:N_TEST], yte[:N_TEST]
nt = yt != 0
xtrig = L.add_trigger(xt[nt])
R = {"arch": a.arch, "n_test": N_TEST, "nc_steps": NC_STEPS, "bits": BITS, "models": {}}
models = [("clean_control", f"cifar10_{a.arch}_clean_s0.pt", 0.0)] + \
         [(f"badnets_{p}pct", f"cifar10_{a.arch}_badnets_p{p}.pt", p / 100) for p in (1, 5, 10)]
for name, fname, rate in models:
    ck = HERE / "checkpoints" / fname
    if not ck.exists():
        log(f"skip {name}: {fname} not trained yet"); continue
    m = L.make_model(a.arch); m.load_state_dict(torch.load(ck, map_location=DEV)); m.to(DEV).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    Tm = L.fit_temperature(L.predict_logits(m, xva, device=DEV), yva)
    cfg = {**{k: cfg0[k] for k in ("tau_D", "rho_U", "u2_max", "d2_max", "n_views", "bits", "alpha")}, "T": Tm}
    o = L.lcd_predict(m, xtrig, cfg, device=DEV, seed=9)
    cov = o["pred"] != -1
    r = {"poison_rate": rate,
         "clean_acc": L.accuracy(m, xt, yt, device=DEV),
         "ASR_none": float((L.predict_logits(m, xtrig, device=DEV).argmax(1) == 0).float().mean()),
         "ASR_squeeze": float((L.baseline_predict(m, xtrig, "squeeze_only", device=DEV, bits=BITS) == 0).float().mean()),
         "ASR_always_stage2": float((L.baseline_predict(m, xtrig, "always_stage2", cfg["n_views"], device=DEV, bits=BITS) == 0).float().mean()),
         "ASR_LCD": float(((o["pred"] == 0) & cov).float().mean()),
         "LCD_routed_triggered": float((o["route"] > 0).float().mean())}
    if rate:
        xp, yp, flag = L.poisoned_dataset(xtr, ytr, rate=rate, target=0, seed=3)
    else:                                               # control: audit the clean training set
        xp, yp, flag = xtr, ytr, torch.zeros(len(xtr), dtype=torch.bool)
    ac = L.activation_clustering(m, xp, yp, device=DEV)
    fl, pf = ac["flagged"], flag.numpy()
    zs = [c["robust_z"] for c in ac["per_class"]]
    r["activation_clustering"] = {"suspect_class": ac["suspect_class"], "max_robust_z": float(max(zs)),
                                  "precision": float((fl & pf).sum() / max(fl.sum(), 1)) if rate else None,
                                  "recall": float((fl & pf).sum() / max(pf.sum(), 1)) if rate else None,
                                  "flagged": int(fl.sum())}
    r["neural_cleanse"] = neural_cleanse(m, xva[:2000], NC_STEPS)
    R["models"][name] = r
    log(f"{name}: clean {r['clean_acc']:.3f} | ASR none {r['ASR_none']:.3f} sq {r['ASR_squeeze']:.3f} S2 {r['ASR_always_stage2']:.3f} "
        f"LCD {r['ASR_LCD']:.3f} (routed {r['LCD_routed_triggered']:.3f}) | AC class {ac['suspect_class']} z {max(zs):.2f} | "
        f"NC class {r['neural_cleanse']['flagged_class']} anomaly {r['neural_cleanse']['max_anomaly']:.2f} "
        f"({'detected' if r['neural_cleanse']['detected'] else 'not detected'}) [{(time.time() - t0) / 60:.0f} min]")
    (HERE / "results" / f"backdoor_{a.arch}{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
log(f"done in {(time.time() - t0) / 60:.0f} min")
