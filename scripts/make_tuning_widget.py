"""Data for the landing page's 1-D widget: one neuron, scalar stimulus x ~ N(0, 1), response r = f(x) + noise.

Three tuning curves, all non-negative firing rates (Fig. 1 of the paper's companion work):
  monotonic        f(x) = (A / 2) (1 + tanh(x / w))        rate 0 .. A
  low-gain         f(x) = (A / 4) (1 + tanh(x / w))        rate 0 .. A/2
  single-peaked    f(x) = (A / 2) (1 - tanh(|x| / w))      rate 0 .. A/2, peak at x = 0
The monotonic and single-peaked curves have the same slope magnitude |f'(x)| everywhere.
For each: Fisher information J(x) = f'(x)^2 / sigma^2; blurred-stimulus Fisher J_t(x_t) of
p_t(r | x_t) = int N(r; f(x), sigma^2) p(x | x_t) dx, computed on grids; multi-scale Fisher
G(x) = int_0^inf E_{x_t ~ N(x, t)}[J_t(x_t)] dt; distances from a reference stimulus; and the mutual information,
computed directly and as 1/2 <G>_{p(x)} (the identity of the paper), as a check.
Output: static/data/tuning.json
"""
import json
import os

import numpy as np

A, W, SIGMA = 1.0, 0.6, 0.08
X = np.linspace(-4, 4, 321)                         # stimulus grid (also x_t grid core)
XT = np.linspace(-9, 9, 721)                         # x_t grid
R = np.linspace(-0.6, 1.6, 700)                      # response grid
TS = np.logspace(-4, 2.3, 64)                         # blur scales t
prior = np.exp(-X ** 2 / 2); prior /= prior.sum()
dx, dr = X[1] - X[0], R[1] - R[0]

sech2 = lambda u: 1 / np.cosh(u) ** 2
SLOPES = {                                           # exact f'(x)
    "monotonic": lambda x: 0.5 * A / W * sech2(x / W),
    "low-gain": lambda x: 0.25 * A / W * sech2(x / W),
    "single-peaked": lambda x: -0.5 * A / W * np.where(x >= 0, 1.0, -1.0) * sech2(np.abs(x) / W),  # |slope| at the cusp: its limit
}
CURVES = {
    "monotonic": lambda x: 0.5 * A * (1 + np.tanh(x / W)),
    "low-gain": lambda x: 0.25 * A * (1 + np.tanh(x / W)),
    "single-peaked": lambda x: 0.5 * A * (1 - np.tanh(np.abs(x) / W)),
}


def mi_direct(f):
    lik = np.exp(-(R[:, None] - f[None, :]) ** 2 / (2 * SIGMA ** 2)) / np.sqrt(2 * np.pi * SIGMA ** 2)
    pr = lik @ prior
    h_r = -np.sum(pr * np.log(pr + 1e-300)) * dr
    return h_r - 0.5 * np.log(2 * np.pi * np.e * SIGMA ** 2)


def J_t_curve(f, t):
    """J_t on the XT grid."""
    lw = -(XT[:, None] - X[None, :]) ** 2 / (2 * t) + np.log(prior)[None, :]              # [xt, x], log weights
    post = np.exp(lw - lw.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    m = post @ X
    lik = np.exp(-(R[:, None] - f[None, :]) ** 2 / (2 * SIGMA ** 2))                    # [r, x] (unnormalised)
    lik /= np.sqrt(2 * np.pi * SIGMA ** 2)
    p = post @ lik.T                                                                       # [xt, r]
    dp = (post * (X[None, :] - m[:, None]) / t) @ lik.T
    return np.sum(dp ** 2 / (p + 1e-300), 1) * dr


def smooth(Jt, t):
    """E_{x_t ~ N(x, t)}[J_t(x_t)] on the X grid."""
    lk = -(X[:, None] - XT[None, :]) ** 2 / (2 * t)
    k = np.exp(lk - lk.max(1, keepdims=True)); k /= k.sum(1, keepdims=True)
    return k @ Jt


out = {"x": X.round(4).tolist(), "prior": (prior / dx).round(5).tolist(), "t": TS.tolist(), "curves": {}}
for name, fn in CURVES.items():
    f = fn(X)
    fp = SLOPES[name](X)
    J = fp ** 2 / SIGMA ** 2
    contrib = []                                      # t * E[J_t] (per unit log t), on X
    for t in TS:
        contrib.append(t * smooth(J_t_curve(f, t), t))
    contrib = np.array(contrib)                       # [nt, nx]
    lt = np.log(TS)
    G = np.trapz(contrib, lt, axis=0)                 # integral over log t
    # small-t tail: below TS[0], J_t ~ J (smoothed by ~0): add J * TS[0]
    G = G + TS[0] * J
    mi = mi_direct(f)
    mi_G = 0.5 * np.sum(prior * G)
    x0 = np.argmin(np.abs(X + 4.0))                   # reference stimulus x0 = -4 (left end of the axis)
    dFR = np.abs(np.cumsum(np.sqrt(J)) * dx - (np.cumsum(np.sqrt(J)) * dx)[x0])
    dG = np.abs(np.cumsum(np.sqrt(G)) * dx - (np.cumsum(np.sqrt(G)) * dx)[x0])
    out["curves"][name] = dict(f=f.round(5).tolist(), J=J.round(4).tolist(), G=G.round(4).tolist(),
                               contrib=contrib.round(5).tolist(), dFR=dFR.round(4).tolist(), dG=dG.round(4).tolist(),
                               mi_bits=mi / np.log(2), mi_from_G_bits=mi_G / np.log(2))
    print(f"{name:14s} MI direct {mi / np.log(2):.3f} bits, 1/2<G> {mi_G / np.log(2):.3f} bits")
out.update(sigma=SIGMA, x0=-4.0)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static", "data", "tuning.json")
os.makedirs(os.path.dirname(p), exist_ok=True)
json.dump(out, open(p, "w"), separators=(",", ":"))
print("wrote", os.path.relpath(p), os.path.getsize(p) // 1024, "kB")
