# Threat-Aware LCD: evaluation code

Code and per-run results for the paper *"Do Cheap Pre-processing Defences Survive Adaptive Attacks? A Calibrated,
Cost-Aware Evaluation of Threat-Aware Compound Defence"* (Pallavi Sarangi, Nitin Niranjan Awathare, IIT Jodhpur).

Threat-Aware LCD compares a frozen classifier's predictions on an image and on its feature-squeezed copy, routes inputs
with high divergence or entropy to a randomised second stage, abstains when that stage stays uncertain, and audits
backdoors offline. The evaluation covers CIFAR-10 (3 seeds) and SVHN, FGSM/BIM/C&W, AutoAttack (transferred and adaptive),
a gate-aware BPDA+EOT attack, BadNets at 1/5/10 % poisoning, activation clustering and Neural Cleanse.

## Layout

```
04_Practical_Implementation/lcd_core.py   the defence, attacks, calibration and audits (one module)
08_Publication/train_job.py               train one model (clean seed, or BadNets at a poison rate)
08_Publication/pub_eval.py                calibrate on validation data, freeze, evaluate every attack x system
08_Publication/latency_only.py            batch-1 latency on an idle machine
08_Publication/pub_backdoor.py            backdoor study (ASR, gate routing, activation clustering, Neural Cleanse)
08_Publication/aggregate.py               results/*.json -> paper tables, figures and number macros
08_Publication/Threat_Aware_LCD_Publication_Colab.ipynb   full ResNet-18 run on a GPU
08_Publication/results/                   the per-run JSON results reported in the paper
```

## Requirements

Python 3.10+, PyTorch, torchvision, NumPy, scikit-learn, matplotlib, and
[AutoAttack](https://github.com/fra31/auto-attack) (`pip install git+https://github.com/fra31/auto-attack`).

## Reproduce

Data are cached as `data/cifar10.npz` and `data/svhn.npz` (uint8 arrays `x_train, y_train, x_test, y_test`); the Colab
notebook shows how to create them with torchvision.

```bash
cd 08_Publication
python train_job.py --dataset cifar10 --seed 0 --data data            # also seeds 1, 2 and --poison 0.01/0.05/0.10
python train_job.py --dataset svhn --seed 0 --epochs 6 --data data
python pub_eval.py --dataset cifar10 --seed 0 --data data             # and seeds 1, 2; and --dataset svhn
python latency_only.py --dataset cifar10 --seed 0 --data data
python pub_backdoor.py --data data
python aggregate.py                                                   # writes paper/tables, paper/figures, paper/numbers.tex
```

Every script has `--smoke` (tiny sizes) to check all code paths in a few minutes, and `--device cuda` for a GPU.
Running `python aggregate.py` on the included `results/` regenerates every number in the paper.

## Protocol

Thresholds, temperature, squeeze bit depth and the gate weight are chosen on validation data only and frozen before the test
split is used. C&W and all adaptive attacks are never used for calibration. Abstentions count as errors.

## License

MIT (see `LICENSE`).
