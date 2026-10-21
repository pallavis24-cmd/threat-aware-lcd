"""Aggregates results/*.json into the paper's tables, figures and number macros.

    python aggregate.py [--arch resnet20]

Writes paper/numbers.tex (\\res{key} macros), paper/tables/*.tex, paper/figures/*.pdf and results/summary_<arch>.json.
Multi-seed cells are mean ± std over seeds; single-run cells show the bootstrap 95% CI of the run.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="resnet20")
a = ap.parse_args()
RES, PAPER = HERE / "results", HERE / "paper"
(PAPER / "tables").mkdir(parents=True, exist_ok=True); (PAPER / "figures").mkdir(parents=True, exist_ok=True)

SYS = [("none", "None"), ("squeeze_only", "Squeeze"), ("randomization_only", "Random"), ("always_stage2", "Always S2"),
       ("conf_gate", "Conf.\\ gate"), ("LCD", "LCD")]
CONDS = [("clean", "Clean"), ("FGSM_4", "FGSM 4/255"), ("FGSM_8", "FGSM 8/255"), ("FGSM_16", "FGSM 16/255"),
         ("BIM_8", "BIM 8/255"), ("CW_L2", "C\\&W-$L_2$"), ("AutoAttack_transfer", "AutoAttack (transfer)"),
         ("AutoAttack_adaptive", "AutoAttack (adaptive)"), ("BPDA_EOT_PGD", "BPDA+EOT PGD (adaptive)")]
ATT = [c for c, _ in CONDS if c != "clean"]


def load(ds):
    files = sorted(f for f in glob.glob(str(RES / f"{ds}_{a.arch}_s*.json")) if "smoke" not in f)
    return [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]


runs = {"cifar10": load("cifar10"), "svhn": load("svhn")}
bd_path = RES / f"backdoor_{a.arch}.json"
bd = json.loads(bd_path.read_text(encoding="utf-8")) if bd_path.exists() else None
macros = {}


def stat(vals):
    v = np.array([x for x in vals if x is not None], float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)) if len(v) else (np.nan, 0.0, 0)


def cell(rs, getter, pct=True, ci_getter=None):
    m, s, n = stat([getter(r) for r in rs])
    f = 100 if pct else 1
    if np.isnan(m):
        return "--", m
    if n > 1:
        return (f"{m * f:.1f}$\\pm${s * f:.1f}" if pct else f"{m:.3f}$\\pm${s:.3f}"), m
    if ci_getter is not None:
        lo, hi = ci_getter(rs[0])
        return (f"{m * f:.1f} [{lo * f:.0f},{hi * f:.0f}]" if pct else f"{m:.3f}"), m
    return (f"{m * f:.1f}" if pct else f"{m:.3f}"), m


def acc_cell(rs, cond, sysk):
    return cell(rs, lambda r: r["test"][cond][sysk]["acc"]["mean"], ci_getter=lambda r: r["test"][cond][sysk]["acc"]["ci95"])


def write_table(name, header, rows, colspec, caption, label, star=False, size="\\footnotesize"):
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{3pt}", "\\begin{adjustbox}{max width=\\linewidth}", f"\\begin{{tabular}}{{{colspec}}}",
             "\\toprule", " & ".join(header) + " \\\\", "\\midrule"] + [" & ".join(r) + " \\\\" for r in rows] + \
            ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", f"\\end{{{env}}}"]
    (PAPER / "tables" / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


# ======================================================================== main accuracy tables
for ds, rs in runs.items():
    if not rs:
        continue
    rows = []
    for c, cname in CONDS:
        row = [cname]
        best = max(stat([r["test"][c][k]["acc"]["mean"] for r in rs])[0] for k, _ in SYS if k != "none")
        for k, _ in SYS:
            txt, m = acc_cell(rs, c, k)
            macros[f"{ds}-{c}-{k}"] = f"{m * 100:.1f}"
            row.append(f"\\textbf{{{txt}}}" if k != "none" and abs(m - best) < 1e-9 else txt)
        cov, _ = cell(rs, lambda r: r["test"][c]["LCD"]["coverage"]["mean"])
        wr, wm = cell(rs, lambda r: r["test"][c]["LCD"]["wrong_accepted"]["mean"])
        rt, rm = cell(rs, lambda r: r["test"][c]["LCD"]["routed"]["mean"])
        macros[f"{ds}-{c}-LCD-cov"] = f"{stat([r['test'][c]['LCD']['coverage']['mean'] for r in rs])[0] * 100:.1f}"
        macros[f"{ds}-{c}-LCD-wrong"] = f"{wm * 100:.1f}"
        macros[f"{ds}-{c}-LCD-routed"] = f"{rm * 100:.1f}"
        row += [cov, wr]
        rows.append(row)
    nseeds = len(rs)
    what = f"mean$\\pm$std over {nseeds} seeds" if nseeds > 1 else "single run, bootstrap 95\\% CI in brackets"
    dsname = "CIFAR-10" if ds == "cifar10" else "SVHN"
    write_table(f"main_{ds}", ["Condition"] + [n for _, n in SYS] + ["LCD cov.", "LCD wrong-acc."], rows, "l" + "c" * (len(SYS) + 2),
                f"{dsname} held-out accuracy (\\%, abstentions counted as wrong; {what}). Best defended system in bold. "
                "\\emph{LCD cov.}\\ = share of inputs answered; \\emph{wrong-acc.}\\ = share answered incorrectly.",
                f"tab:main_{ds}", star=True)

# ======================================================================== gate quality
rs = runs["cifar10"]
if rs:
    rows = []
    for c in ATT:
        cname = dict(CONDS)[c]
        ts, tm = cell(rs, lambda r: r["gate"][c]["AUROC_TS"], pct=False)
        mp, mm = cell(rs, lambda r: r["gate"][c]["AUROC_maxprob"], pct=False)
        ro, rm = cell(rs, lambda r: r["gate"][c]["routed_successful"])
        bs_, bm = cell(rs, lambda r: r["gate"][c]["base_attack_success"])
        macros[f"auc-ts-{c}"] = f"{tm:.2f}"; macros[f"auc-mp-{c}"] = f"{mm:.2f}"; macros[f"routed-{c}"] = f"{rm * 100:.1f}"
        rows.append([cname, ts, mp, ro, bs_])
    write_table("gate", ["Attack", "AUROC $D$/TS", "AUROC $1-\\max p$", "Routed (\\%)", "Base ASR (\\%)"], rows, "lcccc",
                "Gate quality on CIFAR-10 test images. AUROC compares clean and attacked inputs; values below 0.5 mean the "
                "score ranks attacked inputs as \\emph{less} suspicious than clean ones. \\emph{Routed} is the share of successful "
                "attacks sent to Stage~2.", "tab:gate", star=True)
    vm = stat([r["validation"]["AUROC_maxprob"] for r in rs]); vd = stat([r["validation"]["bit_study"][str(r["validation"]["selected_bits"])]["AUROC_D"] for r in rs])
    macros["val-auc-mp"] = f"{vm[0]:.2f}"; macros["val-auc-d"] = f"{vd[0]:.2f}"
    macros["val-auc-mp-std"] = f"{vm[1]:.2f}"; macros["val-auc-d-std"] = f"{vd[1]:.2f}"
    macros["conf-tpr"] = f"{stat([r['validation']['conf_gate_tpr'] for r in rs])[0] * 100:.1f}"
    macros["lcd-tpr"] = f"{stat([r['validation']['bit_study'][str(r['validation']['selected_bits'])]['tpr'] for r in rs])[0] * 100:.1f}"
    macros["clean-acc"] = f"{stat([r['test']['clean']['none']['acc']['mean'] for r in rs])[0] * 100:.1f}"
    macros["nseeds"] = str(len(rs))
    macros["bits-selected"] = ", ".join(str(r["validation"]["selected_bits"]) for r in rs)
    macros["clean-route"] = f"{stat([r['gate']['clean_route_rate'] for r in rs])[0] * 100:.1f}"

    # ---------------------------------------------------------------- bit-depth study
    rows = []
    for b in ("3", "4", "5", "6"):
        d_ = cell(rs, lambda r: r["validation"]["bit_study"][b]["AUROC_D"], pct=False)[0]
        t_ = cell(rs, lambda r: r["validation"]["bit_study"][b]["tpr"])[0]
        sel = sum(int(r["validation"]["selected_bits"]) == int(b) for r in rs)
        rows.append([f"{b}-bit", d_, t_, f"{sel}/{len(rs)}"])
    write_table("bits", ["Bit depth $b$", "AUROC of $D$", "Attacks routed (\\%)", "Selected"], rows, "lccc",
                "Validation-only choice of the squeeze bit depth at a 10\\% clean-routing budget (CIFAR-10).", "tab:bits")

    # ---------------------------------------------------------------- ablations and cost
    rows = []
    names = {"full": "Full LCD", "D_only": "$D$ only", "U_only": "$U$ only", "no_abstention": "No abstention", "N1": "$N=1$", "N5": "$N=5$"}
    for v, vn in names.items():
        row = [vn]
        for c in ("clean", "FGSM_8", "BIM_8"):
            row.append(cell(rs, lambda r: r["ablations"][v][c]["acc"])[0])
        row.append(cell(rs, lambda r: r["ablations"][v]["clean"]["passes"], pct=False)[0].replace(".000", ""))
        rows.append(row)
        macros[f"abl-{v}-fgsm"] = f"{stat([r['ablations'][v]['FGSM_8']['acc'] for r in rs])[0] * 100:.1f}"
    write_table("ablation", ["Variant", "Clean", "FGSM 8/255", "BIM 8/255", "Passes (clean)"], rows, "lcccc",
                "Ablations on CIFAR-10 (accuracy \\%, abstentions counted as wrong).", "tab:ablation")
    lat = [r["latency_ms"] for r in rs if "latency_ms" in r]
    if lat:
        macros["lat-lcd"] = f"{lat[0]['LCD_x']:.2f}"; macros["lat-s2"] = f"{lat[0]['always_stage2_x']:.2f}"
    macros["passes-clean"] = f"{stat([r['test']['clean']['LCD']['passes'] for r in rs])[0]:.2f}"
    macros["passes-fgsm"] = f"{stat([r['test']['FGSM_8']['LCD']['passes'] for r in rs])[0]:.2f}"

# ======================================================================== backdoor table
if bd:
    rows = []
    for name, r in bd["models"].items():
        label = "Clean (control)" if name == "clean_control" else f"BadNets {int(round(r['poison_rate'] * 100))}\\%"
        acl = r["activation_clustering"]; nc = r["neural_cleanse"]
        ac_txt = (f"{acl['suspect_class']} ({acl['precision'] * 100:.0f}/{acl['recall'] * 100:.0f})" if acl["precision"] is not None
                  else f"{acl['suspect_class']} (z={acl['max_robust_z']:.1f})")
        nc_txt = f"{nc['flagged_class']} ({nc['max_anomaly']:.1f}){'*' if nc['detected'] else ''}"
        rows.append([label, f"{r['clean_acc'] * 100:.1f}", f"{r['ASR_none'] * 100:.1f}", f"{r['ASR_always_stage2'] * 100:.1f}",
                     f"{r['ASR_LCD'] * 100:.1f}", f"{r['LCD_routed_triggered'] * 100:.1f}", ac_txt, nc_txt])
        key = name.replace("_", "")
        macros[f"bd-{key}-asr"] = f"{r['ASR_none'] * 100:.1f}"; macros[f"bd-{key}-lcd"] = f"{r['ASR_LCD'] * 100:.1f}"
        macros[f"bd-{key}-routed"] = f"{r['LCD_routed_triggered'] * 100:.1f}"; macros[f"bd-{key}-s2"] = f"{r['ASR_always_stage2'] * 100:.1f}"
        if acl["precision"] is not None:
            macros[f"bd-{key}-acp"] = f"{acl['precision'] * 100:.1f}"; macros[f"bd-{key}-acr"] = f"{acl['recall'] * 100:.1f}"
        macros[f"bd-{key}-nc"] = f"{nc['max_anomaly']:.1f}"; macros[f"bd-{key}-ncclass"] = str(nc["flagged_class"])
        macros[f"bd-{key}-acclass"] = str(acl["suspect_class"])
    write_table("backdoor", ["Model", "Clean acc.", "ASR none", "ASR always S2", "ASR LCD", "Routed", "AC class (P/R)", "NC class (idx)"],
                rows, "lccccccc",
                "Backdoor track (CIFAR-10, target class 0). ASR = triggered non-target images classified as the target (\\%). "
                "AC = activation clustering: flagged class (precision/recall of flagged training samples, \\%). NC = Neural Cleanse: "
                "flagged class (anomaly index; * = index $>2$, i.e.\\ detected).", "tab:backdoor", star=True)

# ======================================================================== figures
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)


rs = runs["cifar10"]
if rs:
    short = {"clean": "Clean", "FGSM_4": "FGSM\n4", "FGSM_8": "FGSM\n8", "FGSM_16": "FGSM\n16", "BIM_8": "BIM\n8", "CW_L2": "C&W",
             "AutoAttack_transfer": "AA\ntrans.", "AutoAttack_adaptive": "AA\nadapt.", "BPDA_EOT_PGD": "BPDA\n+EOT"}
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    xk = np.arange(len(CONDS)); w = 0.8 / len(SYS)
    for i, (k, nm) in enumerate(SYS):
        ms = [stat([r["test"][c][k]["acc"]["mean"] for r in rs]) for c, _ in CONDS]
        ax.bar(xk + (i - (len(SYS) - 1) / 2) * w, [m[0] for m in ms], w * 0.9, yerr=[m[1] for m in ms], color=SLOTS[i],
               label=nm.replace("\\ ", " ").replace("\\", ""), error_kw=dict(lw=0.6, capsize=1.2, ecolor=INK))
    ax.set_xticks(xk, [short[c] for c, _ in CONDS], fontsize=7.5, color=INK)
    ax.set_ylabel("Accuracy", color=INK, fontsize=8); ax.set_ylim(0, 1)
    style(ax); ax.legend(frameon=False, ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "accuracy.pdf"); fig.savefig(PAPER / "figures" / "accuracy.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    xk = np.arange(len(ATT))
    for j, (key, nm, col) in enumerate([("AUROC_TS", "$D$ / $TS$ (ours)", SLOTS[0]), ("AUROC_maxprob", "$1-\\max p$ (confidence)", SLOTS[1])]):
        ms = [stat([r["gate"][c][key] for r in rs]) for c in ATT]
        ax.bar(xk + (j - 0.5) * 0.38, [m[0] for m in ms], 0.36, yerr=[m[1] for m in ms], color=col, label=nm,
               error_kw=dict(lw=0.6, capsize=1.2, ecolor=INK))
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8); ax.text(len(ATT) - 0.5, 0.51, "chance", fontsize=6.5, color=MUTED, ha="right")
    ax.set_xticks(xk, [short[c].replace("\n", " ") for c in ATT], fontsize=6, color=INK, rotation=40, ha="right"); ax.set_ylim(0, 1.05); ax.set_ylabel("AUROC", fontsize=8, color=INK)
    style(ax); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "auroc.pdf"); fig.savefig(PAPER / "figures" / "auroc.png", dpi=200); plt.close(fig)

# ======================================================================== macros + summary
lines = ["% generated by aggregate.py -- do not edit by hand",
         "\\makeatletter", "\\newcommand{\\res}[1]{\\@ifundefined{res@#1}{\\textbf{??}}{\\@nameuse{res@#1}}}"]
lines += [f"\\@namedef{{res@{k}}}{{{v}}}" for k, v in sorted(macros.items())]
lines += ["\\makeatother"]
(PAPER / "numbers.tex").write_text("\n".join(lines), encoding="utf-8")
(RES / f"summary_{a.arch}.json").write_text(json.dumps(macros, indent=2), encoding="utf-8")
print(f"aggregated: cifar10 seeds={len(runs['cifar10'])}, svhn runs={len(runs['svhn'])}, backdoor={'yes' if bd else 'no'}; "
      f"{len(macros)} macros")
