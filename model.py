"""Modelo (ResNet50 + embedding + ArcFace), Dataset e extração de embeddings."""
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ----------------------------------------------------------------- utilidades
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg):
    want = cfg["project"].get("device", "cuda")
    return torch.device("cuda" if want == "cuda" and torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------- modelo
class FaceEmbeddingNet(nn.Module):
    """ResNet50 -> pooling (2048) -> Dropout -> Linear(embedding_dim) -> BatchNorm1d."""

    def __init__(self, embedding_dim=512, dropout=0.4, pretrained=True,
                 weights="IMAGENET1K_V1", freeze_layers=()):
        super().__init__()
        w = models.ResNet50_Weights[weights] if pretrained else None
        backbone = models.resnet50(weights=w)
        feat = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.freeze_layers = list(freeze_layers)
        for name in self.freeze_layers:
            for p in getattr(self.backbone, name).parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for name in self.freeze_layers:  # mantém BatchNorm das camadas congeladas em eval
            getattr(self.backbone, name).eval()
        return self

    def forward(self, x):
        # embedding sem normalizar; normalizar na extração
        return self.head(self.backbone(x))


class ArcMarginProduct(nn.Module):
    """Camada ArcFace (Deng et al., 2019): logits com margem angular aditiva."""

    def __init__(self, in_features, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, labels):
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, cosine.size(1)).float()
        return self.s * (one_hot * phi + (1.0 - one_hot) * cosine)


def build_model(cfg, pretrained=None):
    m = cfg["model"]
    use_pre = m["pretrained"] if pretrained is None else pretrained
    return FaceEmbeddingNet(m["embedding_dim"], m["dropout"], use_pre,
                            m["weights"], m["freeze_layers"])


def load_checkpoint(cfg, path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ck["model"])
    meta = {k: v for k, v in ck.items() if k != "model"}
    return model.to(device).eval(), meta


def model_info(model, ckpt_path, device, image_size, runs=50):
    """Nº de parâmetros, tamanho do arquivo, tempo médio de inferência (1 imagem) e hardware."""
    params = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(ckpt_path) / 1e6
    x = torch.randn(1, 3, image_size, image_size, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / runs * 1000
    hw = torch.cuda.get_device_name(0) if device.type == "cuda" else (
        platform.processor() or "CPU")
    return {"num_params": int(params), "model_size_mb": round(size_mb, 2),
            "inference_ms_per_image": round(ms, 3), "hardware": hw}


# ---------------------------------------------------------------------- dados
def build_transform(cfg, train=False):
    p = cfg["preprocessing"]
    size = p["image_size"]
    ops = [transforms.Resize((size, size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.RandomRotation(10)]
    ops += [transforms.ToTensor(),
            transforms.Normalize(p["normalize_mean"], p["normalize_std"])]
    return transforms.Compose(ops)


def processed_file(root, key):
    """Chave 'id/frontal/01.jpg' -> <root>/id/frontal/01.png (imagem já pré-processada)."""
    return Path(root) / Path(key).with_suffix(".png")


class ImageDataset(Dataset):
    def __init__(self, root, keys, labels=None, transform=None, perturb=None):
        self.root, self.keys, self.labels = root, list(keys), labels
        self.transform, self.perturb = transform, perturb

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        img = Image.open(processed_file(
            self.root, self.keys[i])).convert("RGB")
        if self.perturb is not None:
            img = self.perturb(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, (i if self.labels is None else self.labels[i])


@torch.no_grad()
def extract_embeddings(model, root, keys, cfg, device, perturb=None, batch_size=64):
    """Retorna matriz (len(keys), dim) de embeddings L2-normalizados."""
    ds = ImageDataset(root, keys, None, build_transform(
        cfg, train=False), perturb)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=cfg["training"].get("num_workers", 0))
    model.eval()
    out = np.zeros(
        (len(keys), cfg["model"]["embedding_dim"]), dtype=np.float32)
    for x, idx in dl:
        e = F.normalize(model(x.to(device)), dim=1).cpu().numpy()
        out[idx.numpy()] = e
    return out
