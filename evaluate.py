"""Avaliação: scores (cosseno), threshold na VALIDAÇÃO, métricas, ROC, distribuições e erros.

Uso:
  python evaluate.py --dataset cfp_fp          # define o threshold na validação, congela e avalia no teste
  python evaluate.py --dataset lfw             # cross-dataset: modelo e threshold congelados, sem fine-tuning
  python evaluate.py --dataset cfp_fp --processed-suffix _noalign --frozen --tag cfp_fp_noalign   # ablação
"""
from model import (extract_embeddings, get_device, load_checkpoint, load_config,
                   model_info, processed_file)
from sklearn.metrics import roc_auc_score, roc_curve
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


# -------------------------------------------------------------------- métricas
def score_pairs(emb, keys, pairs):
    """Similaridade de cosseno (embeddings já normalizados) para cada par."""
    idx = {k: i for i, k in enumerate(keys)}
    a = np.array([idx[k] for k in pairs.amostra_1])
    b = np.array([idx[k] for k in pairs.amostra_2])
    scores = np.sum(emb[a] * emb[b], axis=1)
    labels = (pairs.tipo.values == "genuine").astype(int)
    return scores, labels


def compute_eer(labels, scores):
    """Retorna (EER, threshold no ponto do EER). Decisão: score >= threshold -> genuíno."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def metrics_at(labels, scores, thr):
    gen, imp = scores[labels == 1], scores[labels == 0]
    fmr = float(np.mean(imp >= thr))     # falsas aceitações
    fnmr = float(np.mean(gen < thr))     # falsas rejeições
    acc = float((np.sum(gen >= thr) + np.sum(imp < thr)) / len(scores))
    return fmr, fnmr, acc


def full_metrics(labels, scores, thr):
    fpr, tpr, _ = roc_curve(labels, scores)
    eer, _ = compute_eer(labels, scores)
    fmr, fnmr, acc = metrics_at(labels, scores, thr)
    return {"eer": eer, "auc": float(roc_auc_score(labels, scores)),
            "fmr": fmr, "fnmr": fnmr,
            "tar_at_far_1pct": float(np.interp(0.01, fpr, tpr)),
            "accuracy_at_threshold": acc, "threshold": float(thr),
            "n_genuine": int((labels == 1).sum()), "n_impostor": int((labels == 0).sum())}


def fta_rate(processed_root, keys):
    """Fração de imagens em que o detector falhou (usou recorte central)."""
    log = Path(processed_root) / "preprocess_log.csv"
    if not log.exists():
        return None
    df = pd.read_csv(log).set_index("key").reindex(keys)
    return float((~df.detected.fillna(True).astype(bool)).mean())


# ------------------------------------------------------------- dados / pares
def get_pairs_and_root(cfg, dataset, suffix=""):
    proto = Path(cfg["paths"]["protocols_dir"])
    if dataset == "cfp_fp":
        root = cfg["datasets"]["primary"]["processed_path"] + suffix
        return (pd.read_csv(proto / "validation.csv", dtype=str),
                pd.read_csv(proto / "test_pairs.csv", dtype=str), root)
    root = cfg["datasets"]["secondary"]["processed_path"] + suffix
    return None, pd.read_csv(proto / "lfw_test_pairs.csv", dtype=str), root


def embed_pairs(model, cfg, device, root, pairs, perturb=None):
    keys = sorted(set(pairs.amostra_1) | set(pairs.amostra_2))
    emb = extract_embeddings(model, root, keys, cfg, device, perturb)
    scores, labels = score_pairs(emb, keys, pairs)
    return scores, labels, keys


# ------------------------------------------------------------------- saídas
def save_scores(path, pairs, scores):
    pd.DataFrame({"amostra_1": pairs.amostra_1.values, "amostra_2": pairs.amostra_2.values,
                  "tipo": pairs.tipo.values, "score": np.round(scores, 6)}).to_csv(path, index=False)


def plot_roc(labels, scores, path):
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(labels, scores):.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FMR (FAR)")
    plt.ylabel("TAR (1 - FNMR)")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_distributions(labels, scores, thr, path):
    plt.figure(figsize=(6, 4))
    plt.hist(scores[labels == 1], bins=40, alpha=0.6,
             density=True, label="Genuínos")
    plt.hist(scores[labels == 0], bins=40, alpha=0.6,
             density=True, label="Impostores")
    plt.axvline(thr, color="k", ls="--", label=f"Threshold = {thr:.3f}")
    plt.xlabel("Similaridade de cosseno")
    plt.ylabel("Densidade")
    plt.title("Distribuição dos scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_error_analysis(out_dir, pairs, scores, labels, thr, root, n=3):
    """3 falsos matches (impostores com maior score) e 3 falsos non-matches (genuínos com menor score)."""
    df = pairs.reset_index(drop=True).assign(score=scores)
    fm = df[df.tipo == "impostor"].sort_values(
        "score", ascending=False).head(n).assign(erro="falso_match")
    fnm = df[df.tipo == "genuine"].sort_values(
        "score").head(n).assign(erro="falso_non_match")
    cases = pd.concat([fm, fnm])
    cases["threshold"] = thr
    cases[["erro", "amostra_1", "amostra_2", "score", "threshold"]].to_csv(
        Path(out_dir) / "errors.csv", index=False)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4))
    for ax, (_, r) in zip(np.array(axes).ravel(), cases.iterrows()):
        a = Image.open(processed_file(root, r.amostra_1)).convert("RGB")
        b = Image.open(processed_file(root, r.amostra_2)
                       ).convert("RGB").resize(a.size)
        ax.imshow(np.hstack([np.asarray(a), np.asarray(b)]))
        ax.set_title(
            f"{r.erro}\nscore={r.score:.3f} | thr={thr:.3f}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "errors.png", dpi=150)
    plt.close()


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", choices=["cfp_fp", "lfw"], default="cfp_fp")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--processed-suffix", default="")
    ap.add_argument("--frozen", action="store_true",
                    help="usa o threshold já congelado em results/threshold.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg)
    res = Path(cfg["paths"]["results_dir"])
    primary = args.dataset == "cfp_fp"
    default_tag = "cfp_fp_test" if primary else "lfw_cross"
    tag = args.tag or default_tag
    out = res / tag
    out.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint or str(
        Path(cfg["paths"]["checkpoints_dir"]) / f"best_seed{cfg['project']['seed']}.pt")
    model, meta = load_checkpoint(cfg, ckpt, device)
    val_pairs, test_pairs, root = get_pairs_and_root(
        cfg, args.dataset, args.processed_suffix)
    thr_file = Path(cfg["threshold"]["file"])

    if args.frozen or not primary:                      # threshold congelado
        if not thr_file.exists():
            sys.exit(
                f"{thr_file} não existe. Rode primeiro: python evaluate.py --dataset cfp_fp")
        thr = json.load(open(thr_file))["threshold"]
        thr_source = "congelado (validação CFP-FP)"
    else:                                               # threshold definido SÓ na validação
        s, l, _ = embed_pairs(model, cfg, device, root, val_pairs)
        val_eer, thr = compute_eer(l, s)
        thr_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"threshold": thr, "criterion": "eer", "selected_on": "validation",
                   "val_eer": val_eer, "checkpoint": ckpt, "seed": meta.get("seed")},
                  open(thr_file, "w"), indent=2)
        thr_source = "validação (EER)"
        print(
            f"Threshold definido na validação: {thr:.4f} (EER val = {val_eer:.4f}) -> congelado")

    scores, labels, keys = embed_pairs(model, cfg, device, root, test_pairs)
    m = full_metrics(labels, scores, thr)
    result = {"tag": tag, "modality": cfg["project"]["modality"],
              "dataset": cfg["datasets"]["primary" if primary else "secondary"]["name"],
              "cnn": cfg["model"]["architecture"], "pretrain": cfg["model"]["pretrain_dataset"],
              "embedding_dim": cfg["model"]["embedding_dim"], "matching": "cosine",
              "threshold_source": thr_source, "seed": meta.get("seed"), "metrics": m,
              "fta": fta_rate(root, keys),
              "efficiency": model_info(model, ckpt, device, cfg["preprocessing"]["image_size"],
                                       cfg["evaluation"]["inference_runs"])}
    json.dump(result, open(out / "metrics.json", "w"), indent=2)
    save_scores(out / "scores.csv", test_pairs, scores)
    plot_roc(labels, scores, out / "roc.png")
    plot_distributions(labels, scores, thr, out / "distributions.png")
    save_error_analysis(out, test_pairs, scores, labels, thr, root)

    # arquivos "oficiais" na raiz de results/ (formato do enunciado)
    if tag == default_tag:
        for f in ("scores.csv", "metrics.json", "roc.png", "distributions.png"):
            shutil.copy(out / f, res / f)

    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
          for k, v in m.items()}, indent=2))
    print(f"FTA: {result['fta']} | resultados em {out}")


if __name__ == "__main__":
    main()
