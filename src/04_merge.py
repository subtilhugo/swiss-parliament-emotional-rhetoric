#!/usr/bin/env python3
"""Assemble the final analysis dataset from raw speeches + LLM labels.

Reads:
  data/raw/speeches.csv                   -- base speech corpus (00_fetch.py)
  data/processed/rhetoric_labels.csv      -- reason/emotion labels (01_label_rhetoric.py)
  data/processed/emotion_labels.csv       -- 8-emotion scores (02_label_emotions.py)
  data/processed/topic_labels.csv         -- CAP topic codes (03_label_topics.py)

Writes:
  data/processed/final_dataset.csv        -- input for build.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "speeches.csv"
PROC = ROOT / "data" / "processed"
OUT = PROC / "final_dataset.csv"

LABEL_FILES = {
    "rhetoric":  (PROC / "rhetoric_labels.csv",
                  ["ID", "reason_emotion_label", "reason_emotion_confidence"]),
    "emotions":  (PROC / "emotion_labels.csv",
                  ["ID", "emotional", "dominant_emotion", "confidence",
                   "emo_anger", "emo_fear", "emo_joy", "emo_sadness",
                   "emo_enthusiasm", "emo_disgust", "emo_hope", "emo_pride"]),
    "topics":    (PROC / "topic_labels.csv",
                  ["ID", "CAP_Topic", "CAP_Code", "CAP_Confidence"]),
}


def main() -> None:
    df = pd.read_csv(RAW, low_memory=False)
    df["ID"] = df["ID"].astype(str)
    print(f"Base speeches: {len(df):,}")

    for name, (path, cols) in LABEL_FILES.items():
        if not path.exists():
            print(f"  Warning: {path.name} not found — skipping {name} labels.")
            continue
        labels = pd.read_csv(path, usecols=[c for c in cols if c in
                             pd.read_csv(path, nrows=0).columns], low_memory=False)
        labels["ID"] = labels["ID"].astype(str)
        before = len(df)
        df = df.merge(labels, on="ID", how="left")
        print(f"  Merged {name}: {labels['ID'].nunique():,} labels "
              f"({df.shape[1] - before + 1} new columns)")

    # Derived column expected by build.py
    if "reason_emotion_label" in df.columns:
        df["reason_emotion_score"] = (
            df["reason_emotion_label"]
            .map({"emotion": 1.0, "neutral": 0.0, "reason": -1.0})
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df):,} rows, {df.shape[1]} columns → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
