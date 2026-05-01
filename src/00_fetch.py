#!/usr/bin/env python3
"""Fetch Swiss Parliament speeches from the official OData API.

Uses swissparlpy (https://github.com/metaodi/swissparlpy), a thin wrapper
around ws.parlament.ch/odata.svc/. No authentication required.

Output: data/raw/speeches.csv (~600 MB, ~1 h on a standard connection).
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import swissparlpy as spp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "speeches.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 1999, 2025


def _strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<pd_text>|</pd_text>", "", text)
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"</p>|</?i[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[PAGE\s*\d+\]|\[VS\]", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_all(endpoint: str, **kwargs) -> pd.DataFrame:
    records = spp.get_data(endpoint, **kwargs)
    total = records.count
    print(f"  {endpoint}: {total:,} records", flush=True)
    rows = []
    for i, rec in enumerate(records):
        rows.append(dict(rec))
        if (i + 1) % 50_000 == 0:
            print(f"    fetched {i + 1:,} / {total:,}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    print("1/5  Transcripts …")
    df = _fetch_all("Transcript", Language="DE")

    print("2/5  Persons …")
    persons = _fetch_all("Person", Language="DE")
    df = df.merge(persons[["PersonNumber", "GenderAsString", "DateOfBirth"]],
                  on="PersonNumber", how="left")

    print("3/5  MemberCouncil (party per mandate) …")
    mc = _fetch_all("MemberCouncil", Language="DE")
    for col in ("DateJoining", "DateLeaving"):
        mc[col] = pd.to_datetime(mc[col], errors="coerce")
    df["_date"] = pd.to_datetime(df["MeetingDate"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.merge(mc[["PersonNumber", "PartyAbbreviation",
                       "DateJoining", "DateLeaving"]].dropna(subset=["DateJoining"]),
                  on="PersonNumber", how="left")
    in_mandate = (df["_date"] >= df["DateJoining"]) & (
        df["DateLeaving"].isna() | (df["_date"] <= df["DateLeaving"]))
    df.loc[~in_mandate, "PartyAbbreviation"] = pd.NA
    df = (df.sort_values("PartyAbbreviation", na_position="last")
            .drop_duplicates(subset=["ID"], keep="first"))

    print("4/5  SubjectBusiness (bill titles) …")
    sb = _fetch_all("SubjectBusiness", Language="DE")
    sb = sb.sort_values("SortOrder").drop_duplicates("IdSubject", keep="first")
    biz = _fetch_all("Business", Language="DE")
    biz = (biz[["ID", "BusinessShortNumber", "Title"]]
           .rename(columns={"ID": "BusinessID", "Title": "BusinessTitle"})
           .drop_duplicates("BusinessShortNumber", keep="first"))
    sb = sb.merge(biz, on="BusinessShortNumber", how="left")
    if "IdSubject" in df.columns:
        df = df.merge(sb[["IdSubject", "BusinessID", "BusinessTitle"]],
                      on="IdSubject", how="left")

    print("5/5  Clean & filter …")
    df["date"] = df["_date"].dt.date
    df["year"] = df["_date"].dt.year
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX) & df["PersonNumber"].gt(0)].copy()
    df["text_clean"] = df["Text"].fillna("").apply(_strip_html)
    df["n_words"] = df["text_clean"].str.split().str.len().fillna(0).astype(int)
    df = df[df["n_words"] >= 5].copy()

    # Normalise party codes: pre-2021 PDC/M-E → Mitte
    df["NormalizedPartyFinal"] = (
        df["PartyAbbreviation"]
        .replace({"PDC": "Mitte", "M-E": "Mitte", "PBD": "BDP"})
    )

    keep = [
        "ID", "date", "year", "PersonNumber", "SpeakerFullName",
        "NormalizedPartyFinal", "CouncilName", "CantonName", "LanguageOfText",
        "GenderAsString", "DateOfBirth", "BusinessID", "BusinessTitle",
        "text_clean", "n_words",
    ]
    df[keep].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nWrote {len(df):,} speeches → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
