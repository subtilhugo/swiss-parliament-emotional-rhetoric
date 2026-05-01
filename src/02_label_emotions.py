#!/usr/bin/env python3
"""Score each speech on 8 emotion dimensions and derive the dominant emotion.

Reads:  data/raw/speeches.csv
Writes: data/processed/emotion_labels.csv
        (ID, emotional, dominant_emotion, confidence, emo_anger, emo_fear, …)

Requires: OPENAI_API_KEY in environment (or .env at project root).
Model:    gpt-4o-mini  (override with OPENAI_MODEL)

Runtime: ~8 h for ~110 k speeches at 8 workers; supports resume.

Usage:
    python src/02_label_emotions.py              # full run
    python src/02_label_emotions.py --smoke 10   # quick test
    python src/02_label_emotions.py --resume
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "raw" / "speeches.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_CSV = OUT_DIR / "emotion_labels.csv"
CHECKPOINT = OUT_DIR / "_emotion_checkpoint.json"

EMOTIONS = ["anger", "fear", "joy", "sadness", "enthusiasm", "disgust", "hope", "pride"]
MIN_WORDS = 50
MAX_CHARS = 8_000
WORKERS = 8


def get_client() -> OpenAI:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file.")
    return OpenAI(api_key=key, timeout=120, max_retries=5)


def build_prompt(text: str, lang: str) -> str:
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n[TRUNCATED]"
    scores_block = "\n".join(f'    "{e}": <float 0.0-1.0>' for e in EMOTIONS)
    emotion_list = "|".join(EMOTIONS)
    return f"""\
You are an expert in political discourse analysis.
Rate the RHETORICAL TONE of this Swiss parliamentary speech.

Language: {lang.upper()} (DE/FR/IT/RM). Analyse in the original language.
Judge HOW it is said, not WHAT topic it covers.
A factual speech about an emotional topic scores low. A passionate speech about a mundane topic scores high.

EMOTION DEFINITIONS:
- anger      : indignation, outrage, hostile accusations, sarcasm.
- fear       : alarm, threat-framing, catastrophising, urgency.
- disgust    : revulsion, contempt, moral condemnation.
- sadness    : grief, regret, lamentation, compassion.
- joy        : celebration, optimism, gratitude.
- enthusiasm : energetic advocacy, passionate calls to action.
- hope       : forward-looking optimism, aspirational framing.
- pride      : collective achievement, honour, affirming identity.

INSTRUCTIONS:
1. Score each emotion 0.0 (absent) → 1.0 (very strong).
2. Set "emotional" to false if ALL scores < 0.15; true otherwise.
3. Set "dominant_emotion" to the highest-scoring emotion, or "none" if emotional is false.
4. Set "confidence" to your overall classification confidence (0.0-1.0).

Return ONLY valid JSON (no markdown):
{{
  "emotional": true|false,
  "emotions": {{
{scores_block}
  }},
  "dominant_emotion": "{emotion_list}|none",
  "confidence": <float 0.0-1.0>
}}

---BEGIN SPEECH---
{text}
---END SPEECH---"""


def classify_one(client: OpenAI, model: str, sid: str, text: str, lang: str) -> dict:
    prompt = build_prompt(text, lang)
    r = client.chat.completions.create(
        model=model, temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.choices[0].message.content or ""
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    obj = json.loads(raw)

    emotional = bool(obj.get("emotional", False))
    scores = {e: max(0.0, min(1.0, float(obj.get("emotions", {}).get(e, 0.0))))
              for e in EMOTIONS}
    dominant = str(obj.get("dominant_emotion", "none")).strip().lower()
    if dominant not in EMOTIONS + ["none"]:
        dominant = "none"
    if not emotional:
        dominant = "none"
        scores = {e: 0.0 for e in EMOTIONS}
    elif emotional:
        dominant = max(scores, key=scores.get)

    conf = obj.get("confidence")
    try:
        conf = round(max(0.0, min(1.0, float(conf))), 3)
    except (TypeError, ValueError):
        conf = None

    return {"ID": sid, "emotional": emotional, "dominant_emotion": dominant,
            "confidence": conf, **{f"emo_{e}": round(v, 3) for e, v in scores.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, metavar="N")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    df = pd.read_csv(IN_CSV, usecols=["ID", "text_clean", "LanguageOfText", "n_words"],
                     low_memory=False)
    df = df[df["n_words"] >= MIN_WORDS].dropna(subset=["text_clean"]).copy()
    df["ID"] = df["ID"].astype(str)

    if args.smoke:
        df = df.sample(n=min(args.smoke, len(df)), random_state=42)
        print(f"Smoke mode: {len(df)} speeches.")

    done: dict[str, dict] = {}
    if args.resume and CHECKPOINT.exists():
        done = json.loads(CHECKPOINT.read_text())
        print(f"Resumed: {len(done):,} already labelled.")

    todo = df[~df["ID"].isin(done)].reset_index(drop=True)
    print(f"To classify: {len(todo):,} | model: {model}")

    def _one(row):
        return classify_one(client, model, str(row["ID"]),
                            str(row["text_clean"]), str(row.get("LanguageOfText", "")))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_one, row): row["ID"] for _, row in todo.iterrows()}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                result = f.result()
                done[result["ID"]] = result
            except Exception as e:
                print(f"  error id={futs[f]}: {e}")
            if i % 500 == 0 or i == len(todo):
                CHECKPOINT.write_text(json.dumps(done))
                print(f"  {i}/{len(todo)}", end="\r", flush=True)

    print()
    CHECKPOINT.write_text(json.dumps(done))

    out = pd.DataFrame(list(done.values()))
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out):,} rows → {OUT_CSV.relative_to(ROOT)}")
    print(out["dominant_emotion"].value_counts().to_string())


if __name__ == "__main__":
    main()
