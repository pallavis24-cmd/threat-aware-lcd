"""
Threat-Aware LCD (ARMOR-inspired) - core implementation.

Pallavi Sarangi (M25DE2073), M.Tech Data Engineering, IIT Jodhpur
Supervisor: Dr. Nitin Niranjan Awathare

Everything the architecture needs lives here so that the local pilot (run_pilot.py) and the
Colab notebook use exactly the same code:

    models        ResNet-18 (MTP-1 setting) and ResNet-20 (fast CPU pilot), with input normalisation inside
    squeeze S     3-bit colour depth + 2x2 median filter                 (Xu et al., NDSS 2018)
    signals       p0, ps, D = JS/ln2, U = max normalised entropy, TS      (architecture block 3)
    router        fast path / Stage 2 / abstain                           (architecture block 4)
    Stage 2       N random resize+pad views of S(x), averaged             (Xie et al., ICLR 2018)
    calibration   temperature scaling + validation-only threshold search  (Guo et al., ICML 2017)
    attacks       FGSM, BIM, C&W-L2, adaptive BPDA+EOT, BadNets
    audit         activation clustering for the poisoned model             (Chen et al., 2019)

All images are tensors in pixel space [0, 1]; the model normalises internally, so attacks and
defences never have to think about CIFAR mean/std.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
K = 10


# =============================================================================================
# Models
# =============================================================================================
class Normalize(nn.Module):
    def __init__(self, mean=CIFAR_MEAN, std=CIFAR_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class _BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(out)) + self.short(x))


class ResNet20(nn.Module):
    """CIFAR ResNet-20 (He et al., 2016, Sec. 4.2): 3 stages x 3 blocks, 16/32/64 channels."""

    def __init__(self, num_classes=K):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        layers, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(3):
                layers.append(_BasicBlock(cin, cout, stride if i == 0 else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

    def features(self, x):
        return F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))


def resnet18_cifar(num_classes=K):
    """torchvision ResNet-18 with the usual CIFAR stem (3x3 conv, no max-pool)."""
    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def make_model(arch="resnet18"):
    net = resnet18_cifar() if arch == "resnet18" else ResNet20()
    return nn.Sequential(Normalize(), net)


def penultimate(model, x):
    """Penultimate-layer activations (used by the activation-clustering backdoor audit)."""
    net = model[1]
    x = model[0](x)
    if isinstance(net, ResNet20):
        return net.features(x)
    feats = nn.Sequential(*list(net.children())[:-1])
    return feats(x).flatten(1)


# =============================================================================================
# Data
# =============================================================================================
def load_cifar10(root, val_size=5000, seed=2073):
    """Returns float tensors in [0,1]: (x_train, y_train), (x_val, y_val), (x_test, y_test).
    Validation is carved out of the official training split; the official test split is held out."""
    import os
    to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255.0)
    cache = os.path.join(root, "cifar10.npz")      # optional local cache (uint8 HWC arrays, official order)
    if os.path.exists(cache):
        d = np.load(cache)
        x, y = to_t(d["x_train"]), torch.from_numpy(d["y_train"])
        x_test, y_test = to_t(d["x_test"]), torch.from_numpy(d["y_test"])
    else:
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        x, y = to_t(tr.data), torch.tensor(tr.targets)
        x_test, y_test = to_t(te.data), torch.tensor(te.targets)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    vi, ti = perm[:val_size], perm[val_size:]
    return (x[ti], y[ti]), (x[vi], y[vi]), (x_test, y_test)


def load_dataset(name, root, val_size=5000, seed=2073):
    """Generic loader: 'cifar10' (see load_cifar10) or 'svhn' (official train/test split; root must hold svhn.npz with
    uint8 HWC arrays x_train/y_train/x_test/y_test). Validation is carved out of the training split."""
    import os
    if name == "cifar10":
        return load_cifar10(root, val_size, seed)
    to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255.0)
    d = np.load(os.path.join(root, f"{name}.npz"))
    x, y = to_t(d["x_train"]), torch.from_numpy(d["y_train"])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    vi, ti = perm[:val_size], perm[val_size:]
    return (x[ti], y[ti]), (x[vi], y[vi]), (to_t(d["x_test"]), torch.from_numpy(d["y_test"]))


def augment(x, flip=True):
    """Standard augmentation on a batch: random crop (pad 4) + horizontal flip (flip=False for digits, e.g. SVHN)."""
    n = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (n,))
    j = torch.randint(0, 9, (n,))
    ar = torch.arange(32)
    rows = (i[:, None] + ar)[:, None, :, None].expand(n, 3, 32, 40)
    out = xp.gather(2, rows)                                            # crop rows -> B,3,32,40
    cols = (j[:, None] + ar)[:, None, None, :].expand(n, 3, 32, 32)
    out = out.gather(3, cols)                                           # crop cols -> B,3,32,32
    if flip:
        f = torch.rand(n) < 0.5
        out[f] = out[f].flip(3)
    return out


def batches(x, y, bs, shuffle=False, seed=0):
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(len(x))
    for s in range(0, len(x), bs):
        b = idx[s:s + bs]
        yield x[b], y[b]


# =============================================================================================
# Training (clean, BadNets-poisoned, optional fast adversarial training)
# =============================================================================================
def train_model(model, x, y, epochs, bs=128, lr=0.1, wd=5e-4, device="cpu", poison_fn=None,
                adv_eps=None, log=print, seed=2073, flip=True):
    """SGD + Nesterov momentum + one-cycle LR (a short schedule that converges in few epochs).
    poison_fn(xb, yb) -> (xb, yb) injects BadNets triggers per batch (separate poisoned model).
    adv_eps enables FGSM-RS fast adversarial training (Wong et al., ICLR 2020) as a baseline."""
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    steps = epochs * math.ceil(len(x) / bs)
    if steps >= 20:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    else:                                   # too few steps for a schedule (quick smoke tests only)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    for ep in range(epochs):
        t0, correct, total, loss_sum = time.time(), 0, 0, 0.0
        for xb, yb in batches(x, y, bs, shuffle=True, seed=seed + ep):
            xb = augment(xb, flip)
            if poison_fn is not None:
                xb, yb = poison_fn(xb, yb)
            xb, yb = xb.to(device), yb.to(device)
            if adv_eps:
                delta = (torch.rand_like(xb) * 2 - 1) * adv_eps
                delta.requires_grad_(True)
                l_adv = F.cross_entropy(model((xb + delta).clamp(0, 1)), yb)
                g, = torch.autograd.grad(l_adv, delta)
                delta = (delta + 1.25 * adv_eps * g.sign()).clamp(-adv_eps, adv_eps).detach()
                xb = (xb + delta).clamp(0, 1)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        log(f"  epoch {ep + 1}/{epochs}  loss {loss_sum / total:.3f}  train-acc {correct / total:.3f}  ({time.time() - t0:.0f}s)")
    return model.eval()


@torch.no_grad()
def predict_logits(model, x, bs=500, device="cpu"):
    model.eval()
    return torch.cat([model(xb.to(device)).cpu() for xb, _ in batches(x, torch.zeros(len(x)), bs)])


def accuracy(model, x, y, device="cpu"):
    return (predict_logits(model, x, device=device).argmax(1) == y).float().mean().item()


# =============================================================================================
# Block 1 - Squeeze transform S(x): 3-bit colour depth + 2x2 median filter
# =============================================================================================
def bit_depth(x, bits=3):
    levels = 2 ** bits - 1
    return torch.round(x * levels) / levels


def median_filter(x, k=2):
    """k x k median filter (for k=2 the lower median, as scipy/Xu et al. effectively use)."""
    pad_l, pad_r = (k - 1) // 2, k // 2
    xp = F.pad(x, (pad_l, pad_r, pad_l, pad_r), mode="reflect")
    patches = xp.unfold(2, k, 1).unfold(3, k, 1)                 # B,C,H,W,k,k
    return patches.reshape(*patches.shape[:4], -1).median(dim=-1).values


def squeeze(x, bits=3, k=2):
    """S(x). Order is fixed and versioned: bit-depth reduction first, then median filter."""
    return median_filter(bit_depth(x, bits), k)


def squeeze_bpda(x, bits=3, k=2):
    """S(x) in the forward pass, identity gradient in the backward pass (BPDA, Athalye et al. 2018)."""
    return x + (squeeze(x, bits, k) - x).detach()


# =============================================================================================
# Block 2-3 - probes and threat signals
# =============================================================================================
def entropy_norm(p, eps=1e-12):
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(-1) / math.log(p.shape[-1])


def js_norm(p, q, eps=1e-12):
    """Jensen-Shannon divergence divided by ln 2, so it lies in [0, 1] (Lin, 1991)."""
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.log() - b.log())).sum(-1)
    return ((0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2)).clamp(0, 1)


@torch.no_grad()
def probe(model, x, T=1.0, alpha=0.5, bits=3, k=2, device="cpu"):
    """Stage-1 probes for a batch. Returns p0, ps, D, U, TS and the fast-path prediction.
    x and S(x) are sent through the network as ONE batch of 2B images (one call, two views)."""
    x = x.to(device)
    z = model(torch.cat([x, squeeze(x, bits, k)]))
    z0, zs = z.chunk(2)
    p0, ps = F.softmax(z0 / T, -1), F.softmax(zs / T, -1)
    D = js_norm(p0, ps)
    U = torch.maximum(entropy_norm(p0), entropy_norm(ps))
    return {"p0": p0.cpu(), "ps": ps.cpu(), "D": D.cpu(), "U": U.cpu(),
            "TS": (alpha * D + (1 - alpha) * U).cpu(), "y_fast": p0.argmax(1).cpu()}


# =============================================================================================
# Stage 2 - randomised resize + pad views of S(x)
# =============================================================================================
def random_resize_pad(x, min_size=26, out=32, generator=None):
    """Xie et al.-style view: resize to r x r (r ~ U{min_size..out}), then pad at a random offset."""
    n = x.shape[0]
    r = int(torch.randint(min_size, out + 1, (1,), generator=generator))
    xr = F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False)
    gap = out - r
    top = int(torch.randint(0, gap + 1, (1,), generator=generator))
    left = int(torch.randint(0, gap + 1, (1,), generator=generator))
    return F.pad(xr, (left, gap - left, top, gap - top), value=0.0)


@torch.no_grad()
def stage2(model, x, n_views=10, T=1.0, bits=3, k=2, seed=0, device="cpu"):
    """p_bar = mean_r softmax(z(T_r(S(x)))/T); U2 = H(p_bar)/ln K; D2 = mean_r JS(p_r, p_bar)/ln 2."""
    g = torch.Generator().manual_seed(seed)
    xs = squeeze(x.to(device), bits, k)
    views = torch.cat([random_resize_pad(xs, generator=g) for _ in range(n_views)])
    P = F.softmax(model(views) / T, -1).view(n_views, x.shape[0], -1)
    p_bar = P.mean(0)
    D2 = torch.stack([js_norm(P[r], p_bar) for r in range(n_views)]).mean(0)
    return {"p_bar": p_bar.cpu(), "y2": p_bar.argmax(1).cpu(), "U2": entropy_norm(p_bar).cpu(), "D2": D2.cpu()}


# =============================================================================================
# Block 4 + full LCD inference
# =============================================================================================
def lcd_predict(model, x, cfg, device="cpu", bs=250, seed=0):
    """Full Threat-Aware LCD. Returns predictions (-1 = abstain) and per-sample diagnostics.
    cfg keys: T, tau_D, rho_U, u2_max, d2_max, n_views, alpha, bits (squeeze bit depth, default 3)."""
    bits = cfg.get("bits", 3)
    preds, route, Ds, Us, n_forward = [], [], [], [], 0
    for s in range(0, len(x), bs):
        xb = x[s:s + bs]
        sig = probe(model, xb, cfg["T"], cfg.get("alpha", 0.5), bits=bits, device=device)
        n_forward += 2 * len(xb)
        pred = sig["y_fast"].clone()
        r = torch.zeros(len(xb), dtype=torch.long)                         # 0 fast, 1 stage-2, 2 abstain
        hot = (sig["D"] >= cfg["tau_D"]) | (sig["U"] >= cfg["rho_U"])
        if hot.any():
            st = stage2(model, xb[hot], cfg["n_views"], cfg["T"], bits=bits, seed=seed + s, device=device)
            n_forward += cfg["n_views"] * int(hot.sum())
            ok = (st["U2"] < cfg["u2_max"]) & (st["D2"] < cfg["d2_max"])
            y2 = torch.where(ok, st["y2"], torch.full_like(st["y2"], -1))
            pred[hot] = y2
            r[hot] = torch.where(ok, 1, 2)
        preds.append(pred); route.append(r); Ds.append(sig["D"]); Us.append(sig["U"])
    return {"pred": torch.cat(preds), "route": torch.cat(route), "D": torch.cat(Ds), "U": torch.cat(Us),
            "forward_passes_per_image": n_forward / len(x)}


def baseline_predict(model, x, method, n_views=10, device="cpu", bs=250, seed=0, bits=3):
    """Single-defence baselines under matched conditions."""
    out = []
    for s in range(0, len(x), bs):
        xb = x[s:s + bs].to(device)
        with torch.no_grad():
            if method == "none":
                out.append(model(xb).argmax(1).cpu())
            elif method == "squeeze_only":                          # Xu et al. squeezing as a preprocessor
                out.append(model(squeeze(xb, bits)).argmax(1).cpu())
            elif method == "randomization_only":                    # Xie et al. on the raw input
                g = torch.Generator().manual_seed(seed + s)
                v = torch.cat([random_resize_pad(xb, generator=g) for _ in range(n_views)])
                out.append(F.softmax(model(v), -1).view(n_views, len(xb), -1).mean(0).argmax(1).cpu())
            elif method == "always_stage2":                          # squeeze + randomisation on every input
                out.append(stage2(model, xb, n_views, bits=bits, seed=seed + s, device=device)["y2"])
    return torch.cat(out)


# =============================================================================================
# Offline calibration (blocks C-D): temperature, alpha, (tau_D, rho_U), Stage-2 acceptance
# =============================================================================================
def fit_temperature(logits, labels):
    """Minimise validation NLL over a single scalar T > 0 (Guo et al., 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    bins = torch.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * ((pred[m] == labels[m]).float().mean() - conf[m].mean()).abs()
    return float(e)


def auroc(scores_clean, scores_attack):
    """Probability that a random attacked sample scores higher than a random clean one (Mann-Whitney)."""
    s = np.concatenate([scores_clean, scores_attack])
    lab = np.concatenate([np.zeros(len(scores_clean)), np.ones(len(scores_attack))])
    ranks = s.argsort().argsort() + 1.0
    n1, n0 = lab.sum(), len(lab) - lab.sum()
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def select_thresholds(D_clean, U_clean, D_att, U_att, max_clean_route=0.10, grid=101):
    """Choose (tau_D, rho_U) on VALIDATION data: maximise the fraction of attacked inputs routed to
    Stage 2 subject to routing at most `max_clean_route` of clean inputs (a declared operating point)."""
    best = None
    qs = np.linspace(0.0, 1.0, grid)
    tD = np.unique(np.quantile(D_clean, qs))
    tU = np.unique(np.quantile(U_clean, qs))
    for td in tD:
        cd = D_clean >= td
        ad = D_att >= td
        for tu in tU:
            fpr = float(np.mean(cd | (U_clean >= tu)))
            if fpr > max_clean_route:
                continue
            tpr = float(np.mean(ad | (U_att >= tu)))
            if best is None or tpr > best["tpr"] or (tpr == best["tpr"] and fpr < best["fpr"]):
                best = {"tau_D": float(td), "rho_U": float(tu), "tpr": tpr, "fpr": fpr}
    return best


def select_alpha(D_clean, U_clean, D_att, U_att):
    """alpha for the scalar summary TS = alpha*D + (1-alpha)*U, chosen by validation AUROC."""
    best = (0.5, -1)
    for a in np.linspace(0, 1, 21):
        s = auroc(a * D_clean + (1 - a) * U_clean, a * D_att + (1 - a) * U_att)
        if s > best[1]:
            best = (float(a), s)
    return best


# =============================================================================================
# Attacks (all in pixel space [0,1])
# =============================================================================================
def fgsm(model, x, y, eps):
    x = x.clone().requires_grad_(True)
    g, = torch.autograd.grad(F.cross_entropy(model(x), y), x)
    return (x + eps * g.sign()).clamp(0, 1).detach()


def bim(model, x, y, eps, steps=20, step_size=None):
    """Basic Iterative Method / I-FGSM (Kurakin et al., 2017): no random start, L_inf projection."""
    step_size = step_size or eps / 8
    xa = x.clone()
    for _ in range(steps):
        xa.requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)
        xa = (xa + step_size * g.sign()).detach()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1)
    return xa


def cw_l2(model, x, y, steps=100, c_init=1.0, search=4, kappa=0.0, lr=0.01):
    """Untargeted Carlini-Wagner L2 (2017) with tanh change of variables and binary search on c.
    Returns the successful adversarial example with the smallest L2 norm (original x if none)."""
    n, dev = x.shape[0], x.device
    lo, hi = torch.zeros(n, device=dev), torch.full((n,), 1e10, device=dev)
    c = torch.full((n,), c_init, device=dev)
    best_l2 = torch.full((n,), 1e10, device=dev)
    best_adv = x.clone()
    onehot = F.one_hot(y, K).bool()
    w0 = torch.atanh((x * 2 - 1).clamp(-0.999999, 0.999999))
    for _ in range(search):
        w = w0.clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        for _ in range(steps):
            xa = (torch.tanh(w) + 1) / 2
            z = model(xa)
            real = z[onehot]
            other = z.masked_fill(onehot, -1e9).max(1).values
            f = (real - other + kappa).clamp_min(0)
            l2 = ((xa - x) ** 2).flatten(1).sum(1)
            loss = (l2 + c * f).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                win = (z.argmax(1) != y) & (l2 < best_l2)
                best_l2[win] = l2[win].detach()
                best_adv[win] = xa[win].detach()
                succeeded |= (z.argmax(1) != y)
        hi = torch.where(succeeded, torch.minimum(hi, c), hi)
        lo = torch.where(~succeeded, torch.maximum(lo, c), lo)
        c = torch.where(hi < 1e9, (lo + hi) / 2, c * 10)
    return best_adv.detach()


def adaptive_bpda_eot(model, x, y, eps=8 / 255, steps=20, eot=4, beta=1.0, step_size=None, seed=0, bits=3):
    """White-box adaptive attack against the COMPLETE LCD pipeline (Athalye et al. 2018; Tramer et al. 2020):
       - BPDA through the non-differentiable squeeze S,
       - EOT over Stage-2 random resize/pad views,
       - a gate-evasion term that keeps D (JS between x and S(x) predictions) small, so the
         input is pushed towards the fast path while fooling both the fast path and Stage 2."""
    step_size = step_size or eps / 8
    g = torch.Generator().manual_seed(seed)
    xa = x.clone()
    for _ in range(steps):
        xa.requires_grad_(True)
        z0 = model(xa)
        xs = squeeze_bpda(xa, bits)
        zs = model(xs)
        views = torch.cat([random_resize_pad(xs, generator=g) for _ in range(eot)])
        p_bar = F.softmax(model(views), -1).view(eot, len(x), -1).mean(0)
        # all three terms are per-image sums, so each image's own gradient balances them 1 : 1 : beta
        loss = (F.cross_entropy(z0, y, reduction="sum") + F.nll_loss(p_bar.clamp_min(1e-12).log(), y, reduction="sum")
                - beta * js_norm(F.softmax(z0, -1), F.softmax(zs, -1)).sum())
        grad, = torch.autograd.grad(loss, xa)
        xa = (xa + step_size * grad.sign()).detach()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1)
    return xa


def attack_in_batches(fn, model, x, y, bs=250, device=None, **kw):
    """Runs an attack batch by batch on the model's device and returns CPU tensors."""
    model.eval()
    device = device or next(model.parameters()).device
    return torch.cat([fn(model, x[s:s + bs].to(device), y[s:s + bs].to(device), **kw).cpu()
                      for s in range(0, len(x), bs)])


# =============================================================================================
# BadNets (Gu et al., 2017) and the activation-clustering audit (Chen et al., 2019)
# =============================================================================================
def add_trigger(x, size=3):
    """3x3 white square in the bottom-right corner (MTP-1 trigger)."""
    x = x.clone()
    x[:, :, -size:, -size:] = 1.0
    return x


def make_poison_fn(rate=0.05, target=0, size=3, seed=0):
    g = torch.Generator().manual_seed(seed)

    def poison(xb, yb):
        m = torch.rand(len(xb), generator=g) < rate
        if m.any():
            xb = xb.clone(); yb = yb.clone()
            xb[m] = add_trigger(xb[m], size)
            yb[m] = target
        return xb, yb
    return poison


def poisoned_dataset(x, y, rate=0.05, target=0, size=3, seed=0):
    """Fixed poisoned training set (the indices are known, so the audit can be scored)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[: int(rate * len(x))]
    xp, yp = x.clone(), y.clone()
    xp[idx] = add_trigger(xp[idx], size)
    yp[idx] = target
    flag = torch.zeros(len(x), dtype=torch.bool)
    flag[idx] = True
    return xp, yp, flag


@torch.no_grad()
def activation_clustering(model, x, y, n_components=10, device="cpu"):
    """Per class: penultimate activations -> PCA(10) -> 2-means. A class whose smaller cluster is
    unusually large and well separated is suspicious; samples in that cluster are flagged."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    acts = torch.cat([penultimate(model, xb.to(device)).cpu() for xb, _ in batches(x, y, 500)]).numpy()
    report, flagged = [], np.zeros(len(x), bool)
    for c in range(K):
        idx = np.where(y.numpy() == c)[0]
        a = PCA(n_components=n_components, random_state=0).fit_transform(acts[idx])
        km = KMeans(2, n_init=10, random_state=0).fit(a)
        small = np.argmin(np.bincount(km.labels_))
        frac = float(np.mean(km.labels_ == small))
        sil = float(silhouette_score(a[:3000], km.labels_[:3000]))
        report.append({"class": c, "small_cluster_fraction": frac, "silhouette": sil})
    sil_all = np.array([r["silhouette"] for r in report])
    z = (sil_all - np.median(sil_all)) / (np.median(np.abs(sil_all - np.median(sil_all))) * 1.4826 + 1e-9)
    suspect = int(np.argmax(z))
    idx = np.where(y.numpy() == suspect)[0]
    a = PCA(n_components=n_components, random_state=0).fit_transform(acts[idx])
    km = KMeans(2, n_init=10, random_state=0).fit(a)
    small = np.argmin(np.bincount(km.labels_))
    flagged[idx[km.labels_ == small]] = True
    for r, zz in zip(report, z):
        r["robust_z"] = float(zz)
    return {"suspect_class": suspect, "per_class": report, "flagged": flagged}
