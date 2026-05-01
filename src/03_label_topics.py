#!/usr/bin/env python3
"""Assign a Comparative Agendas Project (CAP) topic code to each speech.

Uses the 21-topic CAP codebook as defined in Comparative Agendas Project
(https://www.comparativeagendas.net/). The prompt mirrors the multilingual
classifier approach of the ParlaCAP project (Burst et al., 2023).

Reads:  data/raw/speeches.csv
Writes: data/processed/topic_labels.csv  (ID, CAP_Topic, CAP_Code, CAP_Confidence)

Requires: OPENAI_API_KEY in environment (or .env at project root).
Model:    gpt-4o-mini  (override with OPENAI_MODEL)

Runtime: ~6 h for ~110 k speeches at 8 workers; supports resume.

Usage:
    python src/03_label_topics.py              # full run
    python src/03_label_topics.py --smoke 10   # quick test
    python src/03_label_topics.py --resume
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
OUT_CSV = OUT_DIR / "topic_labels.csv"
CHECKPOINT = OUT_DIR / "_topic_checkpoint.json"

MIN_WORDS = 20
MAX_CHARS = 6_000
WORKERS = 8

# CAP major topic codes (Comparative Agendas Project codebook v3)
CAP_TOPICS = {
    1:  "Macroeconomics",
    2:  "Civil Rights",
    3:  "Health",
    4:  "Agriculture",
    5:  "Labor",
    6:  "Education",
    7:  "Environment",
    8:  "Energy",
    9:  "Immigration",
    10: "Transportation",
    12: "Law and Crime",
    13: "Social Welfare",
    14: "Housing",
    15: "Domestic Commerce",
    16: "Defense",
    17: "Technology",
    18: "Foreign Trade",
    19: "International Affairs",
    20: "Government Operations",
    21: "Public Lands",
    23: "Culture",
    98: "Other",
    99: "Mix",
}


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
    topic_list = "\n".join(f"  {code}: {label}" for code, label in CAP_TOPICS.items())
    return f"""\
You are an expert in policy analysis and legislative text classification.
Assign the single most appropriate Comparative Agendas Project (CAP) major topic code
to this Swiss parliamentary speech. Choose from the list below.

Language: {lang.upper()} (DE/FR/IT/RM). Read the speech in its original language.
Focus on the POLICY DOMAIN being addressed, not the rhetorical style.
Use 99 (Mix) only when multiple equal-weight topics are present.
Use 98 (Other) only when no listed topic applies.

CAP MAJOR TOPICS:
{topic_list}

Return ONLY valid JSON:
{{"CAP_Code": <integer>, "CAP_Topic": "<label>", "confidence": <float 0.0-1.0>}}

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
    code = int(obj["CAP_Code"])
    if code not in CAP_TOPICS:
        code = 98
    label = CAP_TOPICS[code]
    conf = round(max(0.0, min(1.0, float(obj.get("confidence", 0.5)))), 3)
    return {"ID": sid, "CAP_Topic": label, "CAP_Code": code, "CAP_Confidence": conf}


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
    print(out["CAP_Topic"].value_counts().to_string())


if __name__ == "__main__":
    main()
