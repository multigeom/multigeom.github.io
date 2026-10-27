"""Extra images for the eigenvector demo at the top of the landing page.

For each image index, writes static/data/teaser_img<idx>.json (same format as teaser.json: the image and the leading
4 eigenvectors of G(x) and J(x) for the Inception model of V1 and V4, int8-quantised), and lists the indices under
"cycle" in static/data/teaser.json. The frog of teaser.json stays the first image.

    python make_teaser_cycle.py --val <runs>/per_image --out <site root>
"""
import argparse
import base64
import json

import numpy as np
import torch

from make_eigvec_assets import K, q8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", type=int, nargs="+", default=[0, 3, 9, 16, 27, 41, 58, 72, 95, 118, 150, 187])
    a = ap.parse_args()
    for idx in a.images:
        T = {"k": K}
        for area in ("V1", "V4"):
            d = torch.load(f"{a.val}/inception/{area}/img{idx:04d}.pt", map_location="cpu", weights_only=False)
            if "image" not in T:
                T["image"] = base64.b64encode(
                    ((d["image"][0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8).tobytes()).decode()
            for m in ("G", "J"):
                V = d[f"{m}_eigenvectors"].numpy().astype(np.float64)[:K]
                V = V * np.sign(V.sum(1, keepdims=True) + 1e-12)          # display sign: positive mean
                T[f"{m}_{area}"] = [dict(zip(("b64", "scale"), q8(v))) for v in V]
        json.dump(T, open(f"{a.out}/static/data/teaser_img{idx:04d}.json", "w"), separators=(",", ":"))
    path = f"{a.out}/static/data/teaser.json"
    main_T = json.load(open(path))
    main_T["cycle"] = a.images
    json.dump(main_T, open(path, "w"), separators=(",", ":"))
    print("ok", len(a.images), "images")


if __name__ == "__main__":
    main()
