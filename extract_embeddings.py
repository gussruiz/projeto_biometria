"""Extrai e salva os embeddings (L2-normalizados) de um conjunto.

Uso: python extract_embeddings.py --split test          (train | validation | test | lfw)
Saída: results/embeddings/<split>.npz  (chaves 'keys' e 'embeddings')
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model import extract_embeddings, get_device, load_checkpoint, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--split", choices=["train", "validation", "test", "lfw"], default="test")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg)
    proto = Path(cfg["paths"]["protocols_dir"])
    ckpt = args.checkpoint or str(
        Path(cfg["paths"]["checkpoints_dir"]) / f"best_seed{cfg['project']['seed']}.pt")
    model, _ = load_checkpoint(cfg, ckpt, device)

    if args.split == "train":
        keys = pd.read_csv(proto / "train.csv", dtype=str).key.tolist()
        root = cfg["datasets"]["primary"]["processed_path"]
    else:
        name, root_key = {"validation": ("validation.csv", "primary"), "test": ("test_pairs.csv", "primary"),
                          "lfw": ("lfw_test_pairs.csv", "secondary")}[args.split]
        pairs = pd.read_csv(proto / name, dtype=str)
        keys = sorted(set(pairs.amostra_1) | set(pairs.amostra_2))
        root = cfg["datasets"][root_key]["processed_path"]

    emb = extract_embeddings(model, root, keys, cfg, device)
    out = Path(cfg["paths"]["results_dir"]) / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{args.split}.npz",
                        keys=np.array(keys), embeddings=emb)
    print(
        f"{len(keys)} embeddings ({emb.shape[1]}-d) salvos em {out / (args.split + '.npz')}")


if __name__ == "__main__":
    main()
