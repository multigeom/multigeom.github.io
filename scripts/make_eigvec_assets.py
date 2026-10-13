"""Real-data assets of the landing page, from the stage-4 outputs of the paper's pipeline (repo
sazio/MultiscaleInformationGeometry, paper-recipe diffusion models: 200 epochs, classifier-free guidance).

  static/data/teaser.json      the example image (the frog of the paper) and the leading 4 eigenvectors of G(x) and
                               J(x) for the Inception model of V1 and V4, int8-quantised, for the in-browser slider
  static/img/explorer/*.png    for 12 held-out test images: the image, and the leading 4 eigenvectors of G and J for
                               the Inception and ResNet-50 models of V1 and V4 (one sprite per image)
  static/data/explorer.json    index of the sprites

    python make_eigvec_assets.py --frog <runs_frog>/per_image --val <runs>/per_image --out <site root>
"""
import argparse
import base64
import json
import os

import numpy as np
import torch
from PIL import Image

K = 4
CMAP_N = 256


def rdbu():
    import matplotlib
    return (matplotlib.colormaps["RdBu_r"](np.linspace(0, 1, CMAP_N))[:, :3] * 255).astype(np.uint8)


def q8(v):
    s = float(np.abs(v).max()) or 1.0
    return base64.b64encode(np.round(v / s * 127).astype(np.int8).tobytes()).decode(), s


def tile(v, lut):
    v = v.reshape(64, 64)
    m = np.percentile(np.abs(v), 99.5) or 1.0
    i = np.clip((v / m + 1) / 2 * (CMAP_N - 1), 0, CMAP_N - 1).astype(int)
    return lut[i]


def gray(x):
    return np.repeat(((np.clip(x.reshape(64, 64), -1, 1) + 1) * 127.5).astype(np.uint8)[:, :, None], 3, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frog", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", type=int, nargs="+", default=[0, 3, 9, 16, 27, 41, 58, 72, 95, 118, 150, 187])
    a = ap.parse_args()
    lut = rdbu()
    # ---- teaser ---------------------------------------------------------------------------------------------------
    T = {"k": K}
    for area in ("V1", "V4"):
        d = torch.load(f"{a.frog}/inception/{area}/img0000.pt", map_location="cpu", weights_only=False)
        if "image" not in T:
            T["image"] = base64.b64encode(((d["image"][0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8).tobytes()).decode()
        for m in ("G", "J"):
            V = d[f"{m}_eigenvectors"].numpy().astype(np.float64)[:K]
            V = V * np.sign(V.sum(1, keepdims=True) + 1e-12)          # display sign: positive mean
            T[f"{m}_{area}"] = [dict(zip(("b64", "scale"), q8(v))) for v in V]
    os.makedirs(f"{a.out}/static/data", exist_ok=True)
    json.dump(T, open(f"{a.out}/static/data/teaser.json", "w"), separators=(",", ":"))
    # ---- explorer sprites: rows (G V1 Inc, G V1 Res, G V4 Inc, G V4 Res, J ...), cols image + K eigvecs -----------
    os.makedirs(f"{a.out}/static/img/explorer", exist_ok=True)
    rows = [(m, ar, e) for m in ("G", "J") for ar in ("V1", "V4") for e in ("inception", "resnet50")]
    pad = 4
    index = {"rows": [dict(metric=m, area=ar, encoder=e) for m, ar, e in rows], "k": K, "tile": 64, "pad": pad,
             "images": []}
    for idx in a.images:
        S = np.full((len(rows) * (64 + pad) - pad, (K + 1) * (64 + pad) - pad, 3), 255, np.uint8)
        thumb = None
        for r, (m, ar, e) in enumerate(rows):
            d = torch.load(f"{a.val}/{e}/{ar}/img{idx:04d}.pt", map_location="cpu", weights_only=False)
            img = gray(d["image"][0, 0].numpy())
            thumb = img if thumb is None else thumb
            y = r * (64 + pad)
            S[y:y + 64, 0:64] = img
            for k in range(K):
                x = (k + 1) * (64 + pad)
                S[y:y + 64, x:x + 64] = tile(d[f"{m}_eigenvectors"][k].numpy().astype(np.float64), lut)
        Image.fromarray(S).save(f"{a.out}/static/img/explorer/img{idx:04d}.png", optimize=True)
        Image.fromarray(thumb).save(f"{a.out}/static/img/explorer/thumb{idx:04d}.png", optimize=True)
        index["images"].append(idx)
    json.dump(index, open(f"{a.out}/static/data/explorer.json", "w"), separators=(",", ":"))
    print("ok", len(index["images"]), "images")


if __name__ == "__main__":
    main()
