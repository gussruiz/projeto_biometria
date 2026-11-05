"""Treino / fine-tuning da ResNet50 com ArcFace nas identidades de treino do CFP-FP.

Uso: python train.py --seed 42
A validação (pares de validação.csv) é usada para escolher a melhor época. O teste não é tocado.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import compute_eer, embed_pairs
from model import (ArcMarginProduct, ImageDataset, build_model, build_transform, get_device,
                   load_config, set_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    set_seed(seed)
    device = get_device(cfg)
    tcfg = cfg["training"]
    proto = Path(cfg["paths"]["protocols_dir"])
    root = cfg["datasets"]["primary"]["processed_path"]

    train_df = pd.read_csv(proto / "train.csv", dtype=str)
    val_pairs = pd.read_csv(proto / "validation.csv", dtype=str)
    classes = sorted(train_df.identity.unique())
    cmap = {c: i for i, c in enumerate(classes)}
    labels = [cmap[i] for i in train_df.identity]
    print(
        f"Treino: {len(train_df)} imagens, {len(classes)} identidades | dispositivo: {device}")

    g = torch.Generator()
    g.manual_seed(seed)
    ds = ImageDataset(root, train_df.key.tolist(), labels,
                      build_transform(cfg, train=True))
    dl = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                    num_workers=tcfg["num_workers"], generator=g)

    model = build_model(cfg).to(device)
    arc = ArcMarginProduct(cfg["model"]["embedding_dim"], len(classes),
                           tcfg["arcface_scale"], tcfg["arcface_margin"]).to(device)
    lr = tcfg["learning_rate"]
    opt = torch.optim.Adam(
        [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": lr},
         {"params": list(model.head.parameters()) + list(arc.parameters()),
          "lr": lr * tcfg["head_lr_multiplier"]}],
        weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=tcfg["epochs"])
    crit = nn.CrossEntropyLoss()

    ck_dir = Path(cfg["paths"]["checkpoints_dir"])
    ck_dir.mkdir(parents=True, exist_ok=True)
    res = Path(cfg["paths"]["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    ck_path = ck_dir / f"best_seed{seed}.pt"

    best, bad, log = 1.0, 0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        arc.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in tqdm(dl, desc=f"época {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = crit(arc(model(x), y), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        s, l, _ = embed_pairs(model, cfg, device, root, val_pairs)
        val_eer, _ = compute_eer(l, s)
        log.append({"epoch": epoch, "train_loss": total / n,
                   "val_eer": val_eer, "sec": time.time() - t0})
        print(
            f"época {epoch:02d} | loss {total / n:.4f} | EER validação {val_eer:.4f}")
        if val_eer < best:
            best, bad = val_eer, 0
            torch.save({"model": model.state_dict(), "seed": seed,
                       "epoch": epoch, "val_eer": val_eer}, ck_path)
        else:
            bad += 1
            if bad >= tcfg["early_stopping_patience"]:
                print("Early stopping.")
                break
        pd.DataFrame(log).to_csv(
            res / f"train_log_seed{seed}.csv", index=False)
    print(f"Melhor EER de validação: {best:.4f} -> {ck_path}")


if __name__ == "__main__":
    main()
