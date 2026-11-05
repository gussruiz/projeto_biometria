"""Pré-processamento (detecção -> alinhamento -> recorte) e geração dos protocolos.

Uso:
  python preprocess.py                 # protocolos + imagens alinhadas (CFP-FP e LFW)
  python preprocess.py --step protocols
  python preprocess.py --step align --dataset cfp_fp
  python preprocess.py --no-align      # ablação: só recorte central (saída em *_noalign)
"""
import argparse
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import load_config

# Template de 5 pontos do ArcFace para imagens 112x112
# (olho esq., olho dir., nariz, canto esq. da boca, canto dir. da boca)
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


# ----------------------------------------------------- detecção e alinhamento
def get_detector(cfg):
    p = cfg["preprocessing"]
    mp = Path(p["detector_model"])
    if not mp.exists():
        mp.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando detector YuNet para {mp} ...")
        urllib.request.urlretrieve(p["detector_url"], str(mp))
    return cv2.FaceDetectorYN.create(str(mp), "", (320, 320), p["detector_score_threshold"])


def center_crop_resize(img, size):
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA)


def align_by_landmarks(img, landmarks, size):
    """Alinhamento por transformação de similaridade (5 landmarks -> template ArcFace)."""
    dst = TEMPLATE * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)


def align_face(img, detector, size):
    """Detecta o maior rosto, alinha e recorta. Retorna None se a detecção falhar."""
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    f = faces[np.argmax(faces[:, 2] * faces[:, 3])]
    # olho dir., olho esq., nariz, boca dir., boca esq. (YuNet)
    lmk = f[4:14].reshape(5, 2)
    return align_by_landmarks(img, lmk, size)


def process_keys(keys, src_root, out_root, size, detector):
    """Processa as imagens e grava <out_root>/<chave>.png + preprocess_log.csv.

    detector=None -> apenas recorte central (ablação). Falha do detector -> recorte
    central como fallback, registrado como detected=False (conta como FTA).
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in tqdm(keys, desc=f"pré-processando -> {out_root.name}"):
        img = cv2.imread(str(Path(src_root) / key))
        if img is None:
            raise FileNotFoundError(f"Não consegui ler {Path(src_root) / key}")
        out = align_face(img, detector, size) if detector is not None else None
        detected = out is not None
        if out is None:
            out = center_crop_resize(img, size)
        dst = out_root / Path(key).with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out)
        rows.append((key, detected if detector is not None else True))
    log = pd.DataFrame(rows, columns=["key", "detected"])
    log.to_csv(out_root / "preprocess_log.csv", index=False)
    if detector is not None:
        print(
            f"  detecção: {log.detected.mean() * 100:.1f}% ({(~log.detected).sum()} falhas -> FTA)")


# ------------------------------------------------------------------ protocolos
def _read_list(path):
    """Pair_list_*.txt -> {indice: 'id/frontal|profile/arquivo.jpg'}."""
    d = {}
    for n, line in enumerate(open(path, encoding="utf-8"), start=1):
        m = re.search(r"(\d+)/(frontal|profile)/([^/\s]+\.jpg)", line, re.I)
        if not m:
            continue
        tok = line.split()
        idx = int(tok[0]) if tok and tok[0].isdigit() else n
        d[idx] = f"{m.group(1)}/{m.group(2).lower()}/{m.group(3)}"
    return d


def _ident(key):
    return key.split("/")[0]


def build_cfp_protocols(cfg):
    ds = cfg["datasets"]["primary"]
    raw = Path(ds["raw_path"])
    pr = cfg["protocol"]
    proto = Path(cfg["paths"]["protocols_dir"])
    proto.mkdir(parents=True, exist_ok=True)

    F_ = _read_list(raw / "Protocol" / "Pair_list_F.txt")
    P_ = _read_list(raw / "Protocol" / "Pair_list_P.txt")

    def fold_dir(k):
        for name in (f"{k:02d}", str(k)):
            d = raw / "Protocol" / "Split" / "FP" / name
            if d.exists():
                return d
        raise FileNotFoundError(
            f"Fold {k} não encontrado em {raw / 'Protocol' / 'Split' / 'FP'}")

    pairs = {}
    for k in range(1, 11):
        rows = []
        for tipo, fname in (("genuine", "same.txt"), ("impostor", "diff.txt")):
            for line in open(fold_dir(k) / fname, encoding="utf-8"):
                n = re.findall(r"\d+", line)
                if len(n) >= 2:
                    rows.append((F_[int(n[0])], P_[int(n[1])], tipo, k))
        pairs[k] = rows

    bad = sum(1 for k in pairs for a, b, t,
              _ in pairs[k] if t == "genuine" and _ident(a) != _ident(b))
    if bad:
        raise ValueError(f"{bad} pares 'same' com identidades diferentes: confira a ordem "
                         "frontal/perfil e as listas Pair_list_F/P.")

    ids = {k: {_ident(x)
               for a, b, _, _ in pairs[k] for x in (a, b)} for k in pairs}
    tr, va, te = (set().union(*[ids[k] for k in pr[s]])
                  for s in ("train_folds", "val_folds", "test_folds"))
    if tr & va or tr & te or va & te:
        raise ValueError(
            "Vazamento de identidade entre treino/validação/teste!")

    base = raw / "Data" / "Images"
    all_keys = sorted(p.relative_to(base).as_posix()
                      for p in base.glob("*/*/*.jpg"))
    pd.DataFrame([(k, _ident(k)) for k in all_keys if _ident(k) in tr],
                 columns=["key", "identity"]).to_csv(proto / "train.csv", index=False)

    cols = ["amostra_1", "amostra_2", "tipo", "fold"]
    val = pd.DataFrame([r for k in pr["val_folds"]
                       for r in pairs[k]], columns=cols)
    test = pd.DataFrame([r for k in pr["test_folds"]
                        for r in pairs[k]], columns=cols)
    val.to_csv(proto / "validation.csv", index=False)
    test.to_csv(proto / "test_pairs.csv", index=False)
    both = pd.concat([val.assign(split="validation"),
                     test.assign(split="test")])
    both[both.tipo == "genuine"].to_csv(
        proto / "genuine_pairs.csv", index=False)
    both[both.tipo == "impostor"].to_csv(
        proto / "impostor_pairs.csv", index=False)

    ntr = pd.read_csv(proto / "train.csv").identity.nunique()
    print(
        f"CFP-FP: treino {ntr} identidades | validação {len(val)} pares | teste {len(test)} pares")
    print(
        f"        identidades disjuntas (treino/val/teste): {len(tr)}/{len(va)}/{len(te)}")


def build_lfw_protocol(cfg):
    ds = cfg["datasets"]["secondary"]
    d, raw = Path(ds["pairs_dir"]), Path(ds["raw_path"])
    proto = Path(cfg["paths"]["protocols_dir"])
    proto.mkdir(parents=True, exist_ok=True)
    match = pd.read_csv(d / "matchpairsDevTest.csv")
    mism = pd.read_csv(d / "mismatchpairsDevTest.csv")

    def key(name, num):
        return f"{name}/{name}_{int(num):04d}.jpg"

    try:
        rows = [(key(r[0], r[1]), key(r[0], r[2]), "genuine", "devtest")
                for r in match.itertuples(index=False)]
        rows += [(key(r[0], r[1]), key(r[2], r[3]), "impostor", "devtest")
                 for r in mism.itertuples(index=False)]
    except (ValueError, IndexError) as e:
        raise ValueError("Colunas inesperadas nos CSVs do LFW. Esperado: match = nome,img1,img2 | "
                         f"mismatch = nome1,img1,nome2,img2. Detalhe: {e}")
    df = pd.DataFrame(rows, columns=["amostra_1", "amostra_2", "tipo", "fold"])
    missing = [k for k in set(df.amostra_1) | set(
        df.amostra_2) if not (raw / k).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} imagens do LFW não encontradas em {raw}. Ex.: {missing[:3]}")
    df.to_csv(proto / "lfw_test_pairs.csv", index=False)
    print(
        f"LFW (DevTest): {(df.tipo == 'genuine').sum()} genuínos + {(df.tipo == 'impostor').sum()} impostores")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--step", choices=["protocols", "align", "all"], default="all")
    ap.add_argument(
        "--dataset", choices=["cfp_fp", "lfw", "both"], default="both")
    ap.add_argument("--no-align", action="store_true",
                    help="ablação: sem detecção/alinhamento")
    args = ap.parse_args()
    cfg = load_config(args.config)
    size = cfg["preprocessing"]["image_size"]
    suffix = "_noalign" if args.no_align else ""
    do_cfp = args.dataset in ("cfp_fp", "both")
    do_lfw = args.dataset in ("lfw", "both")

    if args.step in ("protocols", "all"):
        if do_cfp:
            build_cfp_protocols(cfg)
        if do_lfw:
            build_lfw_protocol(cfg)

    if args.step in ("align", "all"):
        det = None if args.no_align else get_detector(cfg)
        if do_cfp:
            ds = cfg["datasets"]["primary"]
            base = Path(ds["raw_path"]) / "Data" / "Images"
            keys = sorted(p.relative_to(base).as_posix()
                          for p in base.glob("*/*/*.jpg"))
            process_keys(keys, base, ds["processed_path"] + suffix, size, det)
        if do_lfw:
            ds = cfg["datasets"]["secondary"]
            pairs = pd.read_csv(
                Path(cfg["paths"]["protocols_dir"]) / "lfw_test_pairs.csv", dtype=str)
            keys = sorted(set(pairs.amostra_1) | set(pairs.amostra_2))
            process_keys(keys, ds["raw_path"],
                         ds["processed_path"] + suffix, size, det)


if __name__ == "__main__":
    main()
