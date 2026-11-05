"""Experimento D — robustez: blur e oclusão em 4 níveis (original, baixa, média, alta).

O modelo e o threshold ficam CONGELADOS (results/threshold.json); só as imagens de teste são degradadas.
Uso: python robustness.py
"""
from model import get_device, load_checkpoint, load_config
from evaluate import embed_pairs, full_metrics, get_pairs_and_root, save_scores
from PIL import ImageDraw, ImageFilter
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


LEVELS = ["original", "baixa", "media", "alta"]


class Blur:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        return img if self.sigma <= 0 else img.filter(ImageFilter.GaussianBlur(self.sigma))


class Occlude:
    """Quadrado preto central cujo lado é `frac` do lado da imagem."""

    def __init__(self, frac):
        self.frac = frac

    def __call__(self, img):
        if self.frac <= 0:
            return img
        img = img.copy()
        w, h = img.size
        s = int(round(self.frac * min(w, h)))
        x0, y0 = (w - s) // 2, (h - s) // 2
        ImageDraw.Draw(img).rectangle([x0, y0, x0 + s, y0 + s], fill=(0, 0, 0))
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", choices=["cfp_fp", "lfw"], default="cfp_fp")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg)
    ckpt = args.checkpoint or str(
        Path(cfg["paths"]["checkpoints_dir"]) / f"best_seed{cfg['project']['seed']}.pt")
    model, _ = load_checkpoint(cfg, ckpt, device)
    _, pairs, root = get_pairs_and_root(cfg, args.dataset)
    thr = json.load(open(cfg["threshold"]["file"]))["threshold"]

    out = Path(cfg["paths"]["results_dir"]) / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    rcfg = cfg["robustness"]
    rows = []
    for pert in rcfg["perturbations"]:
        params = rcfg["levels"]["blur_sigma" if pert ==
                                "blur" else "occlusion_fraction"]
        for name, v in zip(LEVELS, params):
            perturb = Blur(v) if pert == "blur" else Occlude(v)
            scores, labels, _ = embed_pairs(
                model, cfg, device, root, pairs, perturb)
            m = full_metrics(labels, scores, thr)
            save_scores(out / f"scores_{pert}_{name}.csv", pairs, scores)
            rows.append({"perturbacao": pert, "nivel": name, "parametro": v,
                         **{k: m[k] for k in ("eer", "auc", "fmr", "fnmr", "tar_at_far_1pct")}})
            print(f"{pert:9s} {name:9s} param={v:<5} EER={m['eer']:.4f} AUC={m['auc']:.4f} "
                  f"FMR={m['fmr']:.4f} FNMR={m['fnmr']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    json.dump({"dataset": args.dataset, "frozen_threshold": thr, "results": rows},
              open(out / "metrics.json", "w"), indent=2)
    plt.figure(figsize=(6, 4))
    for pert, g in df.groupby("perturbacao", sort=False):
        plt.plot(g.nivel, g.eer, marker="o", label=pert)
    plt.ylabel("EER")
    plt.xlabel("Nível de degradação")
    plt.title("Robustez: EER x degradação")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "robustness_eer.png", dpi=150)
    plt.close()
    print(f"Resultados em {out}")


if __name__ == "__main__":
    main()
