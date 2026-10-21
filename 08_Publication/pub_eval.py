"""Publication-grade evaluation of Threat-Aware LCD for ONE (dataset, model seed).

    python pub_eval.py --dataset cifar10 --seed 0 --data <root>          # CPU pilot (ResNet-20)
    python pub_eval.py --dataset cifar10 --seed 0 --arch resnet18 --device cuda --data <root> --full   # Colab

Protocol (validation only for every choice; the test split is used once):
  calibration  T, then squeeze bit depth b in {3,4,5,6}, tau_D, rho_U, alpha, Stage-2 limits (block-D objective)
  attacks      FGSM eps 4/8/16, BIM 8, C&W-L2, AutoAttack (APGD-CE + APGD-DLR) transferred from the undefended model,
               AutoAttack against a BPDA+EOT surrogate of the whole pipeline (adaptive), and BPDA+EOT PGD with a
               gate-evasion term (adaptive)
  systems      none, squeeze only, randomisation only, always Stage 2, Threat-Aware LCD, and an MTP-1-style
               confidence gate (routes on 1 - max p at the same 10% clean budget)
  statistics   per-condition bootstrap 95% CIs over test images; seeds are aggregated by aggregate.py
Writes results/<dataset>_<arch>_s<seed>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
ap.add_argument("--full", action="store_true", help="larger test sets and default AutoAttack iterations (GPU)")
ap.add_argument("--latency", action="store_true", help="measure batch-1 latency (only on an otherwise idle machine)")
ap.add_argument("--smoke", action="store_true", help="tiny sizes: checks every code path in a few minutes")
a = ap.parse_args()
torch.set_num_threads(a.threads)
DEV = a.device
N_TEST, N_CW, N_AA, N_ADA = (5000, 1000, 1000, 1000) if a.full else (2000, 300, 500, 500)
AA_ITERS, CW_STEPS = (100, 200) if a.full else (40, 100)
N_VAL, N_VIEWS, BUDGET = 1000, 10, 0.10
if a.smoke:
    N_TEST, N_CW, N_AA, N_ADA, AA_ITERS, CW_STEPS, N_VAL = 60, 20, 20, 20, 3, 5, 100

OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
tag = f"{a.dataset}_{a.arch}_s{a.seed}" + ("_smoke" if a.smoke else "")
log_f = open(HERE / "logs" / f"eval_{tag}.txt", "w", encoding="utf-8")


def log(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()


def boot_ci(v, n_boot=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def rate(mask):
    m = np.asarray(mask, float)
    return {"mean": float(m.mean()), "ci95": boot_ci(m)}


t_start = time.time()
torch.manual_seed(2073 + a.seed)
np.random.seed(2073 + a.seed)
(xtr, ytr), (xva, yva), (xte, yte) = L.load_dataset(a.dataset, a.data)
ck = HERE / "checkpoints" / f"{a.dataset}_{a.arch}_clean_s{a.seed}.pt"
model = L.make_model(a.arch); model.load_state_dict(torch.load(ck, map_location=DEV)); model.to(DEV).eval()
for p_ in model.parameters():
    p_.requires_grad_(False)
R = {"dataset": a.dataset, "arch": a.arch, "seed": a.seed, "full": a.full,
     "sizes": {"test": N_TEST, "cw": N_CW, "autoattack": N_AA, "adaptive": N_ADA, "val_attack": N_VAL}}
R["clean_val_acc"] = L.accuracy(model, xva, yva, device=DEV)
log(f"[{tag}] clean val acc {R['clean_val_acc']:.4f}")

# ================================================================ calibration (validation only)
vl = L.predict_logits(model, xva, device=DEV)
T = L.fit_temperature(vl, yva)
R["calibration"] = {"T": T, "ece_before": L.ece(torch.softmax(vl, 1), yva), "ece_after": L.ece(torch.softmax(vl / T, 1), yva)}
xv, yv = xva[:N_VAL], yva[:N_VAL]
val_adv = [L.attack_in_batches(L.fgsm, model, xv, yv, eps=8 / 255),
           L.attack_in_batches(L.bim, model, xv, yv, eps=8 / 255, steps=20, step_size=1 / 255)]
study = {}
for b in (3, 4, 5, 6):
    sc = L.probe(model, xv, T, bits=b, device=DEV)
    sa = [L.probe(model, z, T, bits=b, device=DEV) for z in val_adv]
    Dc, Uc = sc["D"].numpy(), sc["U"].numpy()
    Da = np.concatenate([s["D"].numpy() for s in sa]); Ua = np.concatenate([s["U"].numpy() for s in sa])
    thr = L.select_thresholds(Dc, Uc, Da, Ua, max_clean_route=BUDGET)
    study[b] = {"AUROC_D": L.auroc(Dc, Da), "AUROC_U": L.auroc(Uc, Ua), "tpr": thr["tpr"], "fpr": thr["fpr"], "thr": thr,
                "Dc": Dc, "Uc": Uc, "Da": Da, "Ua": Ua}
BITS = max(study, key=lambda b: study[b]["tpr"])
s_ = study[BITS]
alpha, auc_ts = L.select_alpha(s_["Dc"], s_["Uc"], s_["Da"], s_["Ua"])
st = L.stage2(model, xva[:200 if a.smoke else 2000], N_VIEWS, T, bits=BITS, seed=11, device=DEV)
cfg = {"T": T, "bits": BITS, "tau_D": s_["thr"]["tau_D"], "rho_U": s_["thr"]["rho_U"], "alpha": alpha, "n_views": N_VIEWS,
       "u2_max": float(np.quantile(st["U2"].numpy(), 0.95)), "d2_max": float(np.quantile(st["D2"].numpy(), 0.95))}
# MTP-1-style confidence gate at the same clean budget: route if 1 - max p0 >= its 90th clean percentile
conf_clean = 1 - L.probe(model, xv, T, bits=BITS, device=DEV)["p0"].max(1).values.numpy()
conf_att = np.concatenate([1 - L.probe(model, z, T, bits=BITS, device=DEV)["p0"].max(1).values.numpy() for z in val_adv])
conf_thr = float(np.quantile(conf_clean, 1 - BUDGET))
R["validation"] = {"bit_study": {b: {k: v for k, v in study[b].items() if k in ("AUROC_D", "AUROC_U", "tpr", "fpr")} for b in study},
                   "selected_bits": BITS, "AUROC_TS": auc_ts, "alpha": alpha,
                   "AUROC_maxprob": L.auroc(conf_clean, conf_att), "conf_gate_tpr": float(np.mean(conf_att >= conf_thr))}
R["frozen_config"] = {**cfg, "conf_thr": conf_thr}
log(f"[{tag}] bits={BITS} tau_D={cfg['tau_D']:.4f} rho_U={cfg['rho_U']:.4f} alpha={alpha:.2f} | val AUROC D "
    f"{s_['AUROC_D']:.3f} maxprob {R['validation']['AUROC_maxprob']:.3f} | val TPR LCD {s_['tpr']:.3f} conf {R['validation']['conf_gate_tpr']:.3f}")


# ================================================================ defended predictors
def conf_gate_predict(x, seed=7, bs=250):
    """Same Stage 2 and abstention as LCD, but the gate is the MTP-1 confidence score."""
    preds, routes = [], []
    for s in range(0, len(x), bs):
        xb = x[s:s + bs]
        sig = L.probe(model, xb, T, bits=BITS, device=DEV)
        pred = sig["y_fast"].clone(); r = torch.zeros(len(xb), dtype=torch.long)
        hot = (1 - sig["p0"].max(1).values) >= conf_thr
        if hot.any():
            st2 = L.stage2(model, xb[hot], N_VIEWS, T, bits=BITS, seed=seed + s, device=DEV)
            ok = (st2["U2"] < cfg["u2_max"]) & (st2["D2"] < cfg["d2_max"])
            pred[hot] = torch.where(ok, st2["y2"], torch.full_like(st2["y2"], -1)); r[hot] = torch.where(ok, 1, 2)
        preds.append(pred); routes.append(r)
    return torch.cat(preds), torch.cat(routes)


def evaluate(x, y):
    out = {}
    for sname in ("none", "squeeze_only", "randomization_only", "always_stage2"):
        pr = L.baseline_predict(model, x, sname, N_VIEWS, device=DEV, seed=5, bits=BITS)
        out[sname] = {"acc": rate((pr == y).numpy())}
    o = L.lcd_predict(model, x, cfg, device=DEV, seed=7)
    cov = o["pred"] != -1
    out["LCD"] = {"acc": rate((o["pred"] == y).numpy()), "coverage": rate(cov.numpy()),
                  "wrong_accepted": rate(((o["pred"] != y) & cov).numpy()),
                  "routed": rate((o["route"] > 0).numpy()), "abstain": rate((o["route"] == 2).numpy()),
                  "passes": o["forward_passes_per_image"]}
    pc, rc = conf_gate_predict(x)
    covc = pc != -1
    out["conf_gate"] = {"acc": rate((pc == y).numpy()), "coverage": rate(covc.numpy()),
                        "wrong_accepted": rate(((pc != y) & covc).numpy()), "routed": rate((rc > 0).numpy())}
    return out


# ================================================================ attacks on the test split
xt, yt = xte[:N_TEST], yte[:N_TEST]
conds = {"clean": xt}
for e in (4, 8, 16):
    conds[f"FGSM_{e}"] = L.attack_in_batches(L.fgsm, model, xt, yt, eps=e / 255, device=DEV)
conds["BIM_8"] = L.attack_in_batches(L.bim, model, xt, yt, eps=8 / 255, steps=20, step_size=1 / 255, device=DEV)
log(f"[{tag}] FGSM/BIM done ({(time.time() - t_start) / 60:.0f} min)")
conds["CW_L2"] = L.attack_in_batches(L.cw_l2, model, xt[:N_CW], yt[:N_CW], steps=CW_STEPS, search=3, device=DEV)
log(f"[{tag}] C&W done ({(time.time() - t_start) / 60:.0f} min)")

from autoattack import AutoAttack  # noqa: E402
import autoattack.autopgd_base as _apb  # noqa: E402


def _check_zero_gradients(grad, logger=None):
    """Same diagnostic as autoattack.checks.check_zero_gradients, but with reshape(): gradients that flow through the
    BPDA squeeze / resize views are not contiguous, and the original .view() call raises on them."""
    z = grad.reshape(grad.shape[0], -1).abs().sum(-1)
    if (z == 0).any():
        msg = f"there are {int((z == 0).sum())} points with zero gradient"
        print("warning:", msg) if logger is None else logger.log(msg)


_apb.check_zero_gradients = _check_zero_gradients


def run_aa(net, x, y, eot_note):
    adv = AutoAttack(net, norm="Linf", eps=8 / 255, version="custom", attacks_to_run=["apgd-ce", "apgd-dlr"], device=DEV,
                     verbose=False)
    adv.apgd.n_iter = AA_ITERS
    adv.apgd.n_restarts = 1
    adv.apgd_targeted.n_iter = AA_ITERS
    out = []
    for s in range(0, len(x), 250):
        out.append(adv.run_standard_evaluation(x[s:s + 250].to(DEV), y[s:s + 250].to(DEV), bs=250).cpu())
    return torch.cat(out)


conds["AutoAttack_transfer"] = run_aa(model, xt[:N_AA], yt[:N_AA], "none")
log(f"[{tag}] AutoAttack (transfer) done ({(time.time() - t_start) / 60:.0f} min)")


class PipelineSurrogate(nn.Module):
    """Differentiable surrogate of the whole defence for an adaptive AutoAttack: BPDA through the squeeze and EOT over
    Stage-2 views, mixing the fast-path and Stage-2 distributions so both routes are attacked."""

    def __init__(self, eot=4):
        super().__init__(); self.eot = eot

    def forward(self, x):
        if x.requires_grad:                             # gradients through resize/median come back non-contiguous;
            x.register_hook(lambda g: g.contiguous())   # AutoAttack's gradient check needs a contiguous tensor
        xs = L.squeeze_bpda(x, BITS)
        views = torch.cat([L.random_resize_pad(xs) for _ in range(self.eot)])
        p_bar = F.softmax(model(views) / T, -1).view(self.eot, len(x), -1).mean(0)
        p0 = F.softmax(model(x) / T, -1)
        return torch.log(0.5 * p0 + 0.5 * p_bar + 1e-12)


conds["AutoAttack_adaptive"] = run_aa(PipelineSurrogate().to(DEV).eval(), xt[:N_AA], yt[:N_AA], "bpda+eot")
log(f"[{tag}] AutoAttack (adaptive) done ({(time.time() - t_start) / 60:.0f} min)")
conds["BPDA_EOT_PGD"] = L.attack_in_batches(L.adaptive_bpda_eot, model, xt[:N_ADA], yt[:N_ADA], eps=8 / 255, steps=20, eot=4,
                                            beta=1.0, bits=BITS, device=DEV)
log(f"[{tag}] BPDA+EOT PGD done ({(time.time() - t_start) / 60:.0f} min)")

# ================================================================ evaluation
R["test"] = {}
for cname, xc in conds.items():
    yc = yt[: len(xc)]
    R["test"][cname] = evaluate(xc, yc)
    t = R["test"][cname]
    log(f"   {cname:20s} none {t['none']['acc']['mean']:.3f} | sq {t['squeeze_only']['acc']['mean']:.3f} | rnd "
        f"{t['randomization_only']['acc']['mean']:.3f} | S2 {t['always_stage2']['acc']['mean']:.3f} | LCD "
        f"{t['LCD']['acc']['mean']:.3f} (routed {t['LCD']['routed']['mean']:.3f}) | conf-gate {t['conf_gate']['acc']['mean']:.3f}")

# gate quality on test: AUROC of TS and of max-prob, routing of successful attacks
cs = L.probe(model, xt, T, bits=BITS, device=DEV)
ts_clean = alpha * cs["D"].numpy() + (1 - alpha) * cs["U"].numpy()
mp_clean = 1 - cs["p0"].max(1).values.numpy()
R["gate"] = {"clean_route_rate": float(((cs["D"] >= cfg["tau_D"]) | (cs["U"] >= cfg["rho_U"])).float().mean())}
for cname in [c for c in conds if c != "clean"]:
    s = L.probe(model, conds[cname], T, bits=BITS, device=DEV); yc = yt[: len(conds[cname])]
    fooled = (s["y_fast"] != yc).numpy()
    routed = ((s["D"] >= cfg["tau_D"]) | (s["U"] >= cfg["rho_U"])).numpy()
    R["gate"][cname] = {"AUROC_TS": L.auroc(ts_clean, alpha * s["D"].numpy() + (1 - alpha) * s["U"].numpy()),
                        "AUROC_maxprob": L.auroc(mp_clean, 1 - s["p0"].max(1).values.numpy()),
                        "routed_successful": float(routed[fooled].mean()) if fooled.any() else None,
                        "base_attack_success": float(fooled.mean())}

# ablations on clean / FGSM-8 / BIM-8
variants = {"full": cfg, "D_only": {**cfg, "rho_U": 1.01}, "U_only": {**cfg, "tau_D": 1.01},
            "no_abstention": {**cfg, "u2_max": 9, "d2_max": 9}, "N1": {**cfg, "n_views": 1}, "N5": {**cfg, "n_views": 5}}
R["ablations"] = {}
for v, vc in variants.items():
    R["ablations"][v] = {}
    for c in ("clean", "FGSM_8", "BIM_8"):
        o = L.lcd_predict(model, conds[c], vc, device=DEV, seed=7)
        R["ablations"][v][c] = {"acc": float((o["pred"] == yt).float().mean()), "coverage": float((o["pred"] != -1).float().mean()),
                                "passes": o["forward_passes_per_image"]}

if a.latency:
    def ms(fn, xs):
        for i in range(5): fn(xs[i:i + 1])
        if DEV == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(len(xs)): fn(xs[i:i + 1])
        if DEV == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - t0) / len(xs) * 1000
    xl = xt[:200]
    lat = {"none": ms(lambda z: L.baseline_predict(model, z, "none", device=DEV), xl),
           "LCD": ms(lambda z: L.lcd_predict(model, z, cfg, device=DEV), xl),
           "always_stage2": ms(lambda z: L.baseline_predict(model, z, "always_stage2", N_VIEWS, device=DEV, bits=BITS), xl)}
    lat.update(LCD_x=lat["LCD"] / lat["none"], always_stage2_x=lat["always_stage2"] / lat["none"])
    R["latency_ms"] = lat

R["minutes"] = (time.time() - t_start) / 60
(OUT / f"{tag}.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
log(f"[{tag}] done in {R['minutes']:.0f} min")
