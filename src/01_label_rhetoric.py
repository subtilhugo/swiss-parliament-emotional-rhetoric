#!/usr/bin/env python3
"""Classify each speech as emotion | reason | neutral (rhetorical mode).

Reads:  data/raw/speeches.csv
Writes: data/processed/rhetoric_labels.csv  (ID, reason_emotion_label, reason_emotion_confidence)

Requires: OPENAI_API_KEY in environment (or a .env file at the project root).
Model:    gpt-4o-mini  (override with OPENAI_MODEL)

Runtime: ~4 h for ~110 k speeches at 8 workers; supports resume.

Usage:
    python src/01_label_rhetoric.py              # full run
    python src/01_label_rhetoric.py --smoke 10   # quick test on 10 speeches
    python src/01_label_rhetoric.py --resume     # continue interrupted run
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
OUT_CSV = OUT_DIR / "rhetoric_labels.csv"
CHECKPOINT = OUT_DIR / "_rhetoric_checkpoint.json"

MIN_WORDS = 50
MAX_CHARS = 8_000
WORKERS = 8
ALLOWED = {"emotion", "reason", "neutral"}


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
    return f"""\
You are an expert in political discourse analysis.
Classify the RHETORICAL MODE of this Swiss parliamentary speech:

- emotion  : affective persuasion dominates (outrage, fear framing, enthusiasm, moral charge).
- reason   : argumentation dominates (evidence, causal logic, procedural/legal reasoning).
- neutral  : neither side clearly dominates; mostly descriptive or formal.

Language hint: {lang.upper()} (DE/FR/IT/RM). Analyse in the original language.
Judge HOW it is said, not WHAT topic it covers.

Return ONLY valid JSON:
{{"label": "emotion|reason|neutral", "confidence": <float 0.0-1.0>}}

---BEGIN SPEECH---
{text}
---END SPEECH---"""


def classify_one(client: OpenAI, model: str, sid: str, text: str, lang: str):
    prompt = build_prompt(text, lang)
    r = client.chat.completions.create(
        model=model, temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.choices[0].message.content or ""
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    obj = json.loads(raw)
    label = str(obj.get("label", "")).strip().lower()
    if label not in ALLOWED:
        raise ValueError(f"unexpected label: {label!r}")
    conf = float(obj.get("confidence", 0.5))
    return sid, label, round(max(0.0, min(1.0, conf)), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, metavar="N",
                        help="Test on N speeches only and exit.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the last checkpoint.")
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

    done: dict[str, tuple[str, float]] = {}
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
                sid, label, conf = f.result()
                done[sid] = (label, conf)
            except Exception as e:
                print(f"  error id={futs[f]}: {e}")
            if i % 500 == 0 or i == len(todo):
                CHECKPOINT.write_text(json.dumps(done))
                print(f"  {i}/{len(todo)}", end="\r", flush=True)

    print()
    CHECKPOINT.write_text(json.dumps(done))

    out = pd.DataFrame([
        {"ID": sid, "reason_emotion_label": v[0], "reason_emotion_confidence": v[1]}
        for sid, v in done.items()
    ])
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out):,} labels → {OUT_CSV.relative_to(ROOT)}")
    print(out["reason_emotion_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
