"""Trains one model for the publication experiments and saves it atomically.

    python train_job.py --dataset cifar10 --seed 1                 # clean model, training seed 1
    python train_job.py --dataset cifar10 --poison 0.10            # BadNets model (10% poison, target class 0)
    python train_job.py --dataset svhn --seed 0 --epochs 6         # SVHN (no horizontal flips: digits)

The data split is fixed (seed 2073) for every model, so seeds vary initialisation, batch order and augmentation only.
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
ap.add_argument("--poison", type=float, default=0.0)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--arch", default="resnet20")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--data", required=True)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
torch.set_num_threads(a.threads)

CK = HERE / "checkpoints"
LOGS = HERE / "logs"
CK.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
tag = f"{a.dataset}_{a.arch}_" + (f"badnets_p{int(round(a.poison * 100))}" if a.poison else f"clean_s{a.seed}")
out = CK / f"{tag}.pt"
if out.exists():
    print("exists:", out.name); sys.exit(0)
log_f = open(LOGS / f"train_{tag}.txt", "w", encoding="utf-8")
log = lambda s: (log_f.write(s + "\n"), log_f.flush(), print(s, flush=True))

(xtr, ytr), (xva, yva), _ = L.load_dataset(a.dataset, a.data)
if a.poison:
    xtr, ytr, _ = L.poisoned_dataset(xtr, ytr, rate=a.poison, target=0, seed=3)     # same recipe as the pilot
torch.manual_seed(a.seed)
model = L.make_model(a.arch).to(a.device)
t0 = time.time()
log(f"training {tag}: {len(xtr)} images, {a.epochs} epochs, flip={a.dataset != 'svhn'}")
L.train_model(model, xtr, ytr, a.epochs, device=a.device, log=log, seed=a.seed if not a.poison else 2073, flip=(a.dataset != "svhn"))
val_acc = L.accuracy(model, xva, yva, device=a.device)
tmp = out.with_suffix(".tmp")
torch.save(model.state_dict(), tmp)
tmp.replace(out)
meta = {"tag": tag, "dataset": a.dataset, "seed": a.seed, "poison": a.poison, "epochs": a.epochs,
        "val_acc": val_acc, "train_minutes": (time.time() - t0) / 60}
(CK / f"{tag}.json").write_text(json.dumps(meta, indent=2))
log(f"done {tag}: val acc {val_acc:.4f} in {meta['train_minutes']:.1f} min")
