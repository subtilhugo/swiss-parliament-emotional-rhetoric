#!/usr/bin/env python3
"""Compute Gennaro & Ash (2022) emotionality scores for each speech.

Method: encode speeches and word-dictionary poles with a multilingual
sentence transformer, then apply equation (1) from the paper:

    Yi = (sim(speech, Affect_pole) + b) / (sim(speech, Cognition_pole) + b)

where b = 1.0 (smoothing constant from the paper).

Reference:
  Gennaro & Ash (2022). "Emotion and Reason in Political Language."
  The Economic Journal, 132(643), 1037-1059.

Dictionaries: Ash & Gennaro EN + FR stemmed word lists, stored as JSON in
data/frozen/ (629 affect words, 169 cognition words per language).

Reads:  data/raw/speeches.csv
Writes: data/processed/ga_scores.csv  (ID, emotionality, emotionality_z,
        sim_emotion, sim_cognition)

Embeddings are cached in data/processed/embeddings.npy to speed up reruns.

Runtime: ~10 min on CPU (MPS/CUDA if available), ~2 min on GPU.
No API key needed — uses a local sentence-transformer model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "raw" / "speeches.csv"
FROZEN = ROOT / "data" / "frozen"
OUT_DIR = ROOT / "data" / "processed"
OUT_CSV = OUT_DIR / "ga_scores.csv"
EMB_FILE = OUT_DIR / "_embeddings.npy"
EMB_IDS = OUT_DIR / "_embeddings_ids.csv"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SMOOTHING = 1.0
MAX_CHARS = 1_000  # truncate per paper's convention


def load_words(name: str) -> list[str]:
    return json.loads((FROZEN / f"{name}.json").read_text())


def build_poles(model):
    en_affect = load_words("dictionary_affect")
    en_cognition = load_words("dictionary_cognition")
    fr_affect = load_words("fr_dictionary_affect")
    fr_cognition = load_words("fr_dictionary_cognition")

    affect_words = list(set(en_affect + fr_affect))
    cognition_words = list(set(en_cognition + fr_cognition))
    print(f"Affect pole: {len(affect_words)} words  |  Cognition pole: {len(cognition_words)} words")

    A = model.encode(affect_words, normalize_embeddings=True, show_progress_bar=False).mean(axis=0)
    C = model.encode(cognition_words, normalize_embeddings=True, show_progress_bar=False).mean(axis=0)
    A = A / np.linalg.norm(A)
    C = C / np.linalg.norm(C)
    print(f"cos(Affect, Cognition) = {float(np.dot(A, C)):.3f}")
    return A, C


def encode_speeches(model, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    if EMB_FILE.exists() and EMB_IDS.exists():
        cached_ids = pd.read_csv(EMB_IDS)["ID"].astype(str).tolist()
        id_set = set(cached_ids)
        present = df[df["ID"].astype(str).isin(id_set)].copy()
        if len(present) == len(df):
            print(f"Loading cached embeddings ({EMB_FILE.name})…")
            all_embs = np.load(EMB_FILE)
            id_to_idx = {sid: i for i, sid in enumerate(cached_ids)}
            idx = [id_to_idx[str(sid)] for sid in df["ID"]]
            return all_embs[idx], df

    print(f"Encoding {len(df):,} speeches (first {MAX_CHARS} chars each)…")
    texts = df["text_clean"].fillna("").astype(str).str[:MAX_CHARS].tolist()
    embs = model.encode(texts, batch_size=512, normalize_embeddings=True, show_progress_bar=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_FILE, embs.astype(np.float32))
    pd.DataFrame({"ID": df["ID"].astype(str).tolist()}).to_csv(EMB_IDS, index=False)
    print(f"Cached embeddings → {EMB_FILE.name}")
    return embs, df


def main() -> None:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise SystemExit(
            "Missing dependencies. Run: pip install sentence-transformers scikit-learn"
        )

    device = "cpu"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    except AttributeError:
        pass
    print(f"Device: {device}  |  Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    A, C = build_poles(model)

    df = pd.read_csv(IN_CSV, usecols=["ID", "text_clean", "n_words"], low_memory=False)
    df = df[df["n_words"] >= 20].dropna(subset=["text_clean"]).copy()
    df["ID"] = df["ID"].astype(str)
    print(f"Speeches to score: {len(df):,}")

    embs, df = encode_speeches(model, df)

    sim_A = cosine_similarity(embs, A.reshape(1, -1)).flatten()
    sim_C = cosine_similarity(embs, C.reshape(1, -1)).flatten()

    df["emotionality"] = (sim_A + SMOOTHING) / (sim_C + SMOOTHING)
    df["sim_emotion"] = sim_A
    df["sim_cognition"] = sim_C

    mu, sd = df["emotionality"].mean(), df["emotionality"].std()
    df["emotionality_z"] = (df["emotionality"] - mu) / sd

    out = df[["ID", "emotionality", "emotionality_z", "sim_emotion", "sim_cognition"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out):,} rows → {OUT_CSV.relative_to(ROOT)}")
    print(f"  mean={mu:.4f}  std={sd:.4f}  "
          f"min={df['emotionality'].min():.4f}  max={df['emotionality'].max():.4f}")


if __name__ == "__main__":
    main()
