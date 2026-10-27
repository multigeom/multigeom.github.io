"""The frog of the eigenvector demo (first image of static/data/teaser.json), computed with the 200-epoch diffusion models.

Estimates G(x) and J(x) of the frog of paper Fig. 3D (first TinyImageNet training image of class n01644900) for the
Inception model of V1 and V4, as in notebook 3 of multigeom-tutorials, and rewrites the image and the leading 4
eigenvectors of G(x) and J(x) in static/data/teaser.json (the "cycle" list is kept). Needs multigeom, a GPU, and the
checkpoints and TinyImageNet in the multigeom-tutorials folder:

    python make_teaser_frog.py --tutorials <multigeom-tutorials> --out <site root>
"""
import argparse
import base64
import copy
import json
import os

import numpy as np
import torch
from PIL import Image

from make_eigvec_assets import K, q8

IMG = "data/tiny-imagenet-200/train/n01644900/images/n01644900_0.JPEG"
ENC = "inception"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutorials", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mc", type=int, default=200)          # the paper uses 200
    a = ap.parse_args()
    os.chdir(a.tutorials)

    from multigeom import compute_fisher, compute_g
    from multigeom.config import load_config
    from multigeom.datasets import CANONICAL_TRANSFORM
    from multigeom.diffusion.model import build_diffusion_model, build_scheduler
    from multigeom.encoders import build_encoder
    from multigeom.utils import get_device, make_generator

    device = get_device(None)
    cfg = copy.deepcopy(load_config("estimator"))
    cfg.monte_carlo.n_samples = a.n_mc
    sigma = float(cfg.monte_carlo.observation_noise_sigma)
    diff_cfg = load_config("diffusion")
    x = CANONICAL_TRANSFORM(Image.open(IMG)).unsqueeze(0).to(device)

    T = {"k": K}
    T["image"] = base64.b64encode(((x[0, 0].cpu().numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8).tobytes()).decode()
    for area in ("V1", "V4"):
        encoder = build_encoder(load_config(ENC)).to(device)
        encoder.load_state_dict(torch.load(f"runs/encoders/{ENC}/{area}/best_model.pt", map_location=device,
                                           weights_only=False)["model_state_dict"])
        encoder.eval()
        ckpt = torch.load(f"runs/diffusion/{ENC}/{area}/final_model.pt", map_location=device, weights_only=False)
        assert ckpt["epoch"] == 200, ckpt["epoch"]
        unet = build_diffusion_model(diff_cfg).to(device)
        unet.load_state_dict(ckpt["model_state_dict"])
        unet.eval()
        alphas_cumprod = build_scheduler(diff_cfg).alphas_cumprod.to(device)
        g = compute_g(image=x, encoder=encoder, unet=unet, alphas_cumprod=alphas_cumprod, cfg=cfg, device=device,
                      generator=make_generator(int(cfg.seed), device))
        j = compute_fisher(image=x, encoder=encoder, sigma=sigma, top_k=int(cfg.fisher_baseline.top_k), device=device)
        for m, r in (("G", g), ("J", j)):
            V = r.eigenvectors.cpu().numpy().astype(np.float64)[:K]
            V = V * np.sign(V.sum(1, keepdims=True) + 1e-12)          # display sign: positive mean
            T[f"{m}_{area}"] = [dict(zip(("b64", "scale"), q8(v))) for v in V]
        print(area, "done")

    path = f"{a.out}/static/data/teaser.json"
    old = json.load(open(path))
    if "cycle" in old:
        T["cycle"] = old["cycle"]
    json.dump(T, open(path, "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
