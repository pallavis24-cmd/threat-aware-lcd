"""Builds Threat_Aware_LCD_Publication_Colab.ipynb: the full GPU run (ResNet-18, 3 seeds, CIFAR-10 + SVHN, BadNets 1/5/10%).

The notebook embeds the *current* local copies of lcd_core.py and the publication scripts with %%writefile, so the Colab run
uses exactly the code that produced the CPU pilot. Re-run this builder after editing any script.
    python build_colab_notebook.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "04_Practical_Implementation" / "lcd_core.py"
SCRIPTS = ["train_job.py", "pub_eval.py", "pub_backdoor.py", "aggregate.py"]

nb = nbf.v4.new_notebook()
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Threat-Aware LCD: full publication run (Colab GPU)

This notebook reproduces every table and figure of the paper with the **ResNet-18** configuration:

| Step | What runs | Approx. time on a T4 |
|---|---|---|
| 1-3 | setup, code, data | 5 min |
| 4 | train 3 CIFAR-10 seeds, 3 SVHN seeds, BadNets 1/5/10% | 2-3 h |
| 5 | `pub_eval.py --full` for every clean model | 3-5 h |
| 6 | backdoor track (AC + Neural Cleanse) | 30-45 min |
| 7 | aggregate tables, figures and `numbers.tex` | 1 min |

Everything is written to Google Drive, and every step **skips work that is already finished**, so if Colab disconnects
just run all cells again. Choose *Runtime → Change runtime type → GPU* first.
""")

md("## 1. Environment")
code("""
!nvidia-smi -L
!pip -q install git+https://github.com/fra31/auto-attack
import torch; print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
""")

md("## 2. Drive folder and code\nThe folder layout mirrors the project: `04_Practical_Implementation/lcd_core.py` and `08_Publication/*.py`.")
code("""
from google.colab import drive
drive.mount('/content/drive')
import os
ROOT = '/content/drive/MyDrive/ThreatAwareLCD_publication'
for d in ['04_Practical_Implementation', '08_Publication/checkpoints', '08_Publication/results', '08_Publication/logs',
          '08_Publication/paper/tables', '08_Publication/paper/figures', 'data']:
    os.makedirs(f'{ROOT}/{d}', exist_ok=True)
%cd {ROOT}/08_Publication
""")

code(f"%%writefile {{ROOT}}/04_Practical_Implementation/lcd_core.py\n{CORE.read_text(encoding='utf-8')}")
for s in SCRIPTS:
    code(f"%%writefile {{ROOT}}/08_Publication/{s}\n{(HERE / s).read_text(encoding='utf-8')}")

md("## 3. Data\nCIFAR-10 and SVHN are downloaded once with torchvision and cached as uint8 arrays (`data/*.npz`), "
   "the same format the CPU pilot used.")
code("""
import numpy as np, torchvision, os
DATA = f'{ROOT}/data'
if not os.path.exists(f'{DATA}/cifar10.npz'):
    tr = torchvision.datasets.CIFAR10('/content/raw', train=True, download=True)
    te = torchvision.datasets.CIFAR10('/content/raw', train=False, download=True)
    np.savez(f'{DATA}/cifar10.npz', x_train=tr.data, y_train=np.array(tr.targets, dtype=np.int64),
             x_test=te.data, y_test=np.array(te.targets, dtype=np.int64))
if not os.path.exists(f'{DATA}/svhn.npz'):
    tr = torchvision.datasets.SVHN('/content/raw', split='train', download=True)
    te = torchvision.datasets.SVHN('/content/raw', split='test', download=True)
    np.savez(f'{DATA}/svhn.npz', x_train=tr.data.transpose(0, 2, 3, 1), y_train=tr.labels.astype(np.int64),
             x_test=te.data.transpose(0, 2, 3, 1), y_test=te.labels.astype(np.int64))
for f in ('cifar10', 'svhn'):
    d = np.load(f'{DATA}/{f}.npz'); print(f, {k: v.shape for k, v in d.items()})
""")

md("## 4. Training\nSeeds vary initialisation, batch order and augmentation; the train/validation split is fixed. "
   "Finished checkpoints are skipped automatically.")
code("""
ARCH = 'resnet18'
jobs = [f'--dataset cifar10 --seed {s} --epochs 30' for s in (0, 1, 2)] + \\
       [f'--dataset svhn --seed {s} --epochs 15' for s in (0, 1, 2)] + \\
       [f'--dataset cifar10 --poison {p} --epochs 30' for p in (0.01, 0.05, 0.10)]
for j in jobs:
    !python -u train_job.py {j} --arch {ARCH} --device cuda --data {DATA} 2>&1 | grep -v -i warn | tail -3
""")

md("## 5. Evaluation (`--full`: 5,000 test images, 1,000 per expensive attack, full AutoAttack iterations)\n"
   "Latency is measured once (seed 0). Each run writes `results/<dataset>_resnet18_s<seed>.json`.")
code("""
import os
for ds, seeds in (('cifar10', (0, 1, 2)), ('svhn', (0, 1, 2))):
    for s in seeds:
        if os.path.exists(f'results/{ds}_{ARCH}_s{s}.json'):
            print('done:', ds, s); continue
        lat = '--latency' if (ds, s) == ('cifar10', 0) else ''
        !python -u pub_eval.py --dataset {ds} --seed {s} --arch {ARCH} --device cuda --full {lat} --data {DATA} 2>&1 | grep -v -i warn | tail -12
""")

md("## 6. Backdoor track")
code("""
if not os.path.exists(f'results/backdoor_{ARCH}.json'):
    !python -u pub_backdoor.py --arch {ARCH} --device cuda --data {DATA} 2>&1 | grep -v -i warn | tail -6
""")

md("## 7. Tables, figures and number macros\nCopy `paper/tables`, `paper/figures` and `paper/numbers.tex` into the paper folder "
   "(or compile there). `results/summary_resnet18.json` lists every reported number.")
code("""
!python aggregate.py --arch {ARCH}
!cd {ROOT}/08_Publication && zip -qr /content/lcd_publication_results.zip results paper logs
from google.colab import files; files.download('/content/lcd_publication_results.zip')
""")

out = HERE / "Threat_Aware_LCD_Publication_Colab.ipynb"
nb.metadata.update({"accelerator": "GPU", "colab": {"provenance": []},
                    "kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, out)
print("wrote", out.name, len(nb.cells), "cells")
