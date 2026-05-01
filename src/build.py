#!/usr/bin/env python3
"""Build one chart-level CSV per figure in the deck.

Reads the frozen analysis dataset and the small frozen chart tables in
data/frozen/source_csv/, parses slides_letemps_parlacap.tex to know which
figures are used, and writes a compact CSV per figure to output/figure_data/
plus a manifest.csv mapping figures to their data files.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides" / "slides_letemps_parlacap.tex"
FROZEN = ROOT / "data" / "frozen"
SRC_CSV = FROZEN / "source_csv"
DATA_CSV = FROZEN / "final_dataset_capstone_with_parlacap.csv"
AHEF_CSV = FROZEN / "emotions_ahef_data.csv"
OUT = ROOT / "output" / "figure_data"

PARTIES = ["SP", "Grüne", "GLP", "Mitte", "FDP", "SVP"]
EMOTIONS = ["anger", "enthusiasm", "hope", "fear", "sadness", "disgust"]
CHAMBER_LABEL = {"Nationalrat": "National Council", "Ständerat": "Council of States"}
N_TOP_TOPICS = 12  # match `select_topics` in the figure-generation script
EXCLUDE_CAP = {"Mix", "Other"}
CAP_BROAD = {
    "International Affairs": "Foreign Policy & Security",
    "Foreign Trade": "Foreign Policy & Security",
    "Defense": "Defense & Military",
    "Macroeconomics": "Budget, Tax & Public Finance",
    "Government Operations": "Budget, Tax & Public Finance",
    "Domestic Commerce": "Economy & Infrastructure",
    "Transportation": "Economy & Infrastructure",
    "Technology": "Economy & Infrastructure",
    "Law and Crime": "Criminal Justice & Rights",
    "Civil Rights": "Criminal Justice & Rights",
    "Health": "Healthcare",
    "Energy": "Energy & Environment",
    "Environment": "Energy & Environment",
    "Agriculture": "Agriculture & Food",
    "Public Lands": "Agriculture & Food",
    "Immigration": "Migration & Asylum",
    "Labor": "Social Policy & Labor",
    "Social Welfare": "Social Policy & Labor",
    "Education": "Social Policy & Labor",
    "Housing": "Social Policy & Labor",
    "Culture": "Social Policy & Labor",
}
LEG_EDGES = pd.to_datetime(
    ["1998-01-01", "2003-10-19", "2007-10-21", "2011-10-23", "2015-10-18",
     "2019-10-20", "2023-10-22", "2032-01-01"]
)
LEG_LABELS = ["46 (99-03)", "47 (04-07)", "48 (08-11)", "49 (12-15)",
              "50 (16-19)", "51 (20-23)", "52 (24-25)"]


def load_dataset() -> pd.DataFrame:
    cols = [
        "ID", "date", "year", "NormalizedPartyFinal", "CouncilName",
        "CantonName", "LanguageOfText", "GenderAsString", "DateOfBirth",
        "PersonNumber", "SpeakerFullName", "n_words", "brackets", "BusinessTitle",
        "dominant_emotion", "reason_emotion_label", "emotional",
        "CAP_Topic", "text_clean",
    ]
    df = pd.read_csv(DATA_CSV, low_memory=False, encoding="utf-8-sig", usecols=cols)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["party"] = df["NormalizedPartyFinal"]
    df["text_len"] = df["text_clean"].fillna("").astype(str).str.strip().str.len()
    df.drop(columns=["text_clean"], inplace=True)
    label = df["reason_emotion_label"].astype(str).str.lower()
    df["emotional_rhetoric"] = label.eq("emotion")
    df["reason_emotion_score"] = label.map({"emotion": 1.0, "neutral": 0.0, "reason": -1.0})
    df["dominant_emotion"] = df["dominant_emotion"].astype(str).str.lower()
    df["topic_broad"] = df["CAP_Topic"].map(CAP_BROAD)
    df["legislature"] = pd.cut(
        pd.to_datetime(df["date"], errors="coerce"),
        bins=LEG_EDGES, labels=LEG_LABELS, right=False, ordered=True,
    ).astype(object)
    return df


def base(df: pd.DataFrame, chamber: str | None = None) -> pd.DataFrame:
    out = df[df["party"].isin(PARTIES)]
    if chamber is not None:
        out = out[out["CouncilName"].eq(chamber)]
    out = out[out["year"].between(1999, 2025)]
    out = out[out["text_len"] > 20]
    return out.copy()


_GA_PARTY_MAP = {"PSS": "SP", "PLR": "FDP", "UDC": "SVP", "PDC": "Mitte",
                 "M-E": "Mitte", "Le Centre": "Mitte", "VERT-E-S": "Grüne", "PBD": "BDP"}


def _ga_for_chamber(label: str) -> pd.DataFrame:
    ga = pd.read_csv(SRC_CSV / "gennaro_ash_scores.csv", low_memory=False)
    council = "Conseil national" if label == "Nationalrat" else "Conseil des Etats"
    ga = ga[ga["CouncilName"].eq(council)].copy()
    ga["party"] = ga["PartyAbbreviation"].replace(_GA_PARTY_MAP)
    return ga[ga["year"].between(1999, 2025)]


def _ga_with_glp(df: pd.DataFrame) -> pd.DataFrame:
    """Calibrate GLP rows for Nationalrat (the GA file has none) using
    the deck's reason_emotion_score on the same z-score scale, mirroring
    34_letemps_parlacap_rebuild.gennaro_ash_chamber_figures."""
    ga = _ga_for_chamber("Nationalrat")
    main = base(df, "Nationalrat").dropna(subset=["reason_emotion_score"])
    main_py = (main[main["party"].isin(PARTIES)]
               .groupby(["party", "year"])["reason_emotion_score"].mean()
               .reset_index())
    ga_py = (ga[ga["party"].isin(PARTIES)]
             .groupby(["party", "year"])["emotionality_z"].mean()
             .reset_index())
    cal = main_py.merge(ga_py, on=["party", "year"])
    if len(cal) >= 10:
        slope, intercept = np.polyfit(cal["reason_emotion_score"].to_numpy(),
                                       cal["emotionality_z"].to_numpy(), 1)
        glp = main_py[main_py["party"].eq("GLP") & main_py["year"].between(2008, 2025)].copy()
        if not glp.empty:
            glp["emotionality_z"] = slope * glp["reason_emotion_score"] + intercept
            ga = pd.concat([ga, glp[["year", "party", "emotionality_z"]]], ignore_index=True)
    return ga


def _w(df: pd.DataFrame, name: str) -> str:
    df.to_csv(OUT / f"{name}.csv", index=False)
    return f"{name}.csv"


# ---------- recipes ----------

def floor_time_share(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    tab = (d.groupby(["year", "party"], dropna=False)["n_words"].sum()
             .unstack(fill_value=0).reindex(columns=PARTIES, fill_value=0))
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    out = pct.reset_index().melt("year", var_name="party", value_name="share_words_pct")
    return _w(out, name), "Yearly word-share by party (% of words)."


def topic_distribution_broad(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    d = d[d["topic_broad"].notna()]
    tab = (d.groupby(["year", "topic_broad"]).size().unstack(fill_value=0))
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    out = pct.reset_index().melt("year", var_name="topic_broad", value_name="share_pct")
    return _w(out, name), "Yearly share of speeches per broad ParlaCAP topic (%)."


def _top_cap_topics(df: pd.DataFrame, n: int = N_TOP_TOPICS) -> list[str]:
    # Top-N CAP topics ranked from Nationalrat speeches, mirroring select_topics
    # in the figure-generation script. Same list used for both chambers.
    d = base(df, "Nationalrat")
    d = d[d["CAP_Topic"].notna() & ~d["CAP_Topic"].isin(EXCLUDE_CAP)]
    return d["CAP_Topic"].value_counts().head(n).index.tolist()


def topic_distribution_top(df: pd.DataFrame, chamber: str, name: str,
                           n: int = N_TOP_TOPICS) -> tuple[str, str]:
    topics = _top_cap_topics(df, n)
    d = base(df, chamber)
    d = d[d["CAP_Topic"].isin(topics)]
    tab = d.groupby(["year", "CAP_Topic"]).size().unstack(fill_value=0).reindex(columns=topics, fill_value=0)
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    out = pct.reset_index().melt("year", var_name="CAP_Topic", value_name="share_pct")
    return _w(out, name), f"Yearly share of speeches per top-{n} ParlaCAP topic (%)."


def topic_emphasis_party(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    topics = _top_cap_topics(df)
    d = base(df, chamber)
    d = d[d["CAP_Topic"].isin(topics)]
    tab = d.groupby(["party", "CAP_Topic"]).size().unstack(fill_value=0)
    pct = (tab.div(tab.sum(axis=1), axis=0) * 100).reindex(index=PARTIES, columns=topics, fill_value=0)
    out = pct.reset_index().melt("party", var_name="CAP_Topic", value_name="share_pct")
    return _w(out, name), "Topic emphasis by party across top-12 CAP topics (% of party speeches)."


def topic_shift(df: pd.DataFrame, chamber: str, name: str, kind: str) -> tuple[str, str]:
    topics = _top_cap_topics(df)
    d = base(df, chamber)
    d = d[d["CAP_Topic"].isin(topics)]
    early = d[d["year"].between(2000, 2005)]["CAP_Topic"].value_counts(normalize=True).reindex(topics, fill_value=0) * 100
    late = d[d["year"].between(2020, 2025)]["CAP_Topic"].value_counts(normalize=True).reindex(topics, fill_value=0) * 100
    out = pd.DataFrame({"early_pct": early, "late_pct": late})
    if kind == "delta":
        out["delta_pp"] = out["late_pct"] - out["early_pct"]
    else:
        out["ratio"] = np.where(out["early_pct"] > 0, out["late_pct"] / out["early_pct"], np.nan)
    return _w(out.reset_index().rename(columns={"index": "CAP_Topic"}), name), \
        "Top-12 CAP topics, share early (2000-05) vs late (2020-25); " + (
            "delta in pp." if kind == "delta" else "ratio late/early.")


def defD_overall(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    out = (d.groupby("year")["emotional_rhetoric"].mean() * 100).reset_index(name="share_emotion_pct")
    return _w(out, name), "Yearly share of emotional speeches (%)."


def defD_by_party(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    out = (d.groupby(["year", "party"])["emotional_rhetoric"].mean() * 100).reset_index(name="share_emotion_pct")
    return _w(out, name), "Share of emotional speeches by party-year (%)."


def defD_by_legislature(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    tab = (d.groupby(["legislature", "party"])["emotional_rhetoric"].mean() * 100).reset_index()
    tab = tab.rename(columns={"emotional_rhetoric": "share_emotion_pct"})
    return _w(tab, name), "Share of emotional speeches per legislature × party (%)."


def defD_topic_shift(df: pd.DataFrame, chamber: str, name: str,
                     diff_col: str = "shift_pp") -> tuple[str, str]:
    d = base(df, chamber)
    d = d[d["CAP_Topic"].notna() & ~d["CAP_Topic"].isin(["Mix", "Other"])]
    early = d[d["year"].between(2000, 2005)].groupby("CAP_Topic")["emotional_rhetoric"].mean() * 100
    late = d[d["year"].between(2020, 2025)].groupby("CAP_Topic")["emotional_rhetoric"].mean() * 100
    out = pd.DataFrame({"early_pct": early, "late_pct": late}).reset_index()
    out[diff_col] = out["late_pct"] - out["early_pct"]
    return _w(out, name), "Within-topic shift in emotional-rhetoric share (pp)."


def defD_topic_shift_party(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    d = d[d["CAP_Topic"].notna() & ~d["CAP_Topic"].isin(["Mix", "Other"])]
    early = (d[d["year"].between(2000, 2005)]
             .groupby(["CAP_Topic", "party"])["emotional_rhetoric"].mean() * 100)
    late = (d[d["year"].between(2020, 2025)]
            .groupby(["CAP_Topic", "party"])["emotional_rhetoric"].mean() * 100)
    out = pd.concat({"early_pct": early, "late_pct": late}, axis=1).reset_index()
    out["shift_pp"] = out["late_pct"] - out["early_pct"]
    return _w(out, name), "Within-topic emotional-rhetoric shift by party (pp)."


def emotion_score_time(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    ga = _ga_with_glp(df) if chamber == "Nationalrat" else _ga_for_chamber(chamber)
    ga = ga[ga["party"].isin(PARTIES)]
    out = ga.groupby("year")["emotionality_z"].mean().reset_index()
    return _w(out, name), "Yearly mean Gennaro-Ash emotionality z-score (top-6 parties)."


def emotion_score_party(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    ga = _ga_with_glp(df) if chamber == "Nationalrat" else _ga_for_chamber(chamber)
    ga = ga[ga["party"].isin(PARTIES)]
    out = ga.groupby(["year", "party"])["emotionality_z"].mean().reset_index()
    return _w(out, name), "Yearly mean Gennaro-Ash z-score by party (Mitte = ex-PDC/M-E)."


def emotion_index_100(df: pd.DataFrame, chamber: str, name: str, base_year: int = 2000) -> tuple[str, str]:
    d = base(df, chamber)
    s = d.groupby("year")["emotional_rhetoric"].mean() * 100
    if base_year in s.index and s.loc[base_year]:
        idx = s / s.loc[base_year] * 100
    else:
        idx = s
    out = pd.DataFrame({"year": s.index, "share_emotion_pct": s.values, "index_2000_100": idx.values})
    return _w(out, name), f"Emotional-rhetoric share, indexed to {base_year}=100."


def dominant_party(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    d = d[d["emotional_rhetoric"]]
    tab = d.groupby(["party", "dominant_emotion"]).size().unstack(fill_value=0)
    tab = tab.div(tab.sum(axis=1), axis=0) * 100
    out = tab.reindex(PARTIES).reset_index().melt(
        "party", var_name="dominant_emotion", value_name="share_pct")
    return _w(out, name), "Distribution of dominant emotion within emotional speeches, by party (%)."


def emotion_share_party_facet(df: pd.DataFrame, chamber: str, name: str, emotion: str) -> tuple[str, str]:
    d = base(df, chamber)
    d = d.copy()
    d["match"] = d["emotional_rhetoric"] & d["dominant_emotion"].eq(emotion)
    out = (d.groupby(["year", "party"])["match"].mean() * 100).reset_index()
    out.columns = ["year", "party", f"{emotion}_share_pct"]
    return _w(out, name), f"Share of speeches with dominant emotion = {emotion}, by year × party (%)."


def emotion_topic(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    d = d[d["CAP_Topic"].notna() & ~d["CAP_Topic"].isin(["Mix", "Other"])]
    out = (d.groupby("CAP_Topic")["emotional_rhetoric"].mean() * 100).reset_index()
    out.columns = ["CAP_Topic", "share_emotion_pct"]
    return _w(out.sort_values("share_emotion_pct", ascending=False), name), \
        "Mean share of emotional speeches per topic (%)."


def emotion_topic_party(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    d = d[d["CAP_Topic"].notna() & ~d["CAP_Topic"].isin(["Mix", "Other"])]
    out = (d.groupby(["CAP_Topic", "party"])["emotional_rhetoric"].mean() * 100).reset_index()
    out.columns = ["CAP_Topic", "party", "share_emotion_pct"]
    return _w(out, name), "Mean share of emotional speeches per topic × party (%)."


def party_ratio_topic(chamber: str, name: str) -> tuple[str, str]:
    ga = _ga_for_chamber(chamber)
    g = ga.groupby(["topic_merged", "party"])["emotionality_z"].mean().unstack()
    out = pd.DataFrame({"topic": g.index, "SP_z": g.get("SP"), "SVP_z": g.get("SVP")})
    out["svp_minus_sp"] = out["SVP_z"] - out["SP_z"]
    return _w(out.sort_values("svp_minus_sp"), name), \
        "Mean Gennaro-Ash z-score per topic for SVP and SP, with SVP-SP gap."


def gender_mp_vs_speeches(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    rows = []
    for party in PARTIES:
        sub = d[d["party"].eq(party)]
        if sub.empty:
            continue
        mps = sub.drop_duplicates("PersonNumber")
        rows.append({
            "party": party,
            "frac_mp_female_pct": (mps["GenderAsString"].eq("f").mean() * 100),
            "frac_speeches_female_pct": (sub["GenderAsString"].eq("f").mean() * 100),
        })
    return _w(pd.DataFrame(rows), name), \
        "Female share among MPs vs among speeches, by party (%)."


def gender_metrics_chamber(chamber: str, name: str) -> tuple[str, str]:
    g = pd.read_csv(SRC_CSV / "lt_gender_parliament_metrics_summary.csv")
    label = "nationalrat" if chamber == "Nationalrat" else "staenderat"
    out = g[g["council"].eq(label)].copy()
    return _w(out, name), "Pooled 1999-2025 gender metrics for the chamber (gap, ratios)."


def canton_share(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    out = (d.groupby("CantonName").size() / len(d) * 100).reset_index(name="share_pct")
    return _w(out.sort_values("share_pct", ascending=False), name), \
        "Speech share per canton (%)."


def canton_per_mp(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    speeches = d.groupby("CantonName").size()
    mps = d.groupby("CantonName")["PersonNumber"].nunique()
    out = pd.DataFrame({
        "canton": speeches.index,
        "n_speeches": speeches.values,
        "n_mps": mps.reindex(speeches.index).values,
    })
    out["speeches_per_mp"] = out["n_speeches"] / out["n_mps"]
    return _w(out.sort_values("speeches_per_mp", ascending=False), name), \
        "Speeches per MP by canton."


def canton_gap(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    speeches = d["CantonName"].value_counts()
    mps = d.drop_duplicates(["PersonNumber", "CantonName"])["CantonName"].value_counts()
    speech_share = speeches / speeches.sum() * 100
    mp_share = mps / mps.sum() * 100
    out = pd.concat([speech_share.rename("speech_share"),
                     mp_share.rename("mp_share")], axis=1).dropna()
    out["gap_pp"] = out["speech_share"] - out["mp_share"]
    out = out.sort_values("gap_pp").reset_index().rename(columns={"index": "canton"})
    return _w(out, name), "Speech share vs MP share gap (pp) by canton."


def svp_anger_time(df: pd.DataFrame, name: str) -> tuple[str, str]:
    d = base(df, "Nationalrat")
    d = d[d["party"].eq("SVP")].copy()
    d["anger"] = d["dominant_emotion"].eq("anger") & d["emotional_rhetoric"]
    out = (d.groupby("year")["anger"].mean() * 100).reset_index(name="anger_share_pct")
    return _w(out, name), "Share of SVP speeches dominated by anger, by year (%)."


def anger_prepost_2007(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    d = d.copy()
    d["period"] = np.where(d["year"] < 2008, "pre-2007", "post-2007")
    d["anger"] = d["dominant_emotion"].eq("anger") & d["emotional_rhetoric"]
    out = (d.groupby(["party", "period"])["anger"].mean() * 100).reset_index(name="anger_share_pct")
    return _w(out, name), "Share of speeches dominated by anger, pre vs post 2007, by party (%)."


def cohort_emotional(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    by = pd.to_datetime(d["DateOfBirth"], errors="coerce", dayfirst=True).dt.year
    d = d.assign(birth_decade=(by // 10 * 10).astype("Int64"))
    d = d[d["birth_decade"].notna() & d["birth_decade"].between(1900, 2000)]
    out = (d.groupby("birth_decade")["emotional_rhetoric"].mean() * 100).reset_index(name="share_emotion_pct")
    return _w(out, name), "Share of emotional speeches by speaker birth decade (%)."


def age_group_emotional(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber).copy()
    by = pd.to_datetime(d["DateOfBirth"], errors="coerce", dayfirst=True).dt.year
    d["age"] = d["year"].astype("float") - by
    d["age_group"] = pd.cut(d["age"], bins=[0, 30, 45, 60, 100],
                            labels=["<30", "30-44", "45-59", "60+"], right=False)
    d = d[d["age_group"].notna()]
    out = (d.groupby("age_group", observed=False)["emotional_rhetoric"].mean() * 100).reset_index(name="share_emotion_pct")
    return _w(out, name), "Share of emotional speeches by age group (%)."


def emotion_language(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = base(df, chamber)
    out = (d.groupby("LanguageOfText")["emotional_rhetoric"].mean() * 100).reset_index(name="share_emotion_pct")
    return _w(out, name), "Share of emotional speeches by language (%)."


_APPLAUSE_RX = re.compile(
    r"beifall|applaus|applaudissement|applauso", re.IGNORECASE)
_LAUGHTER_RX = re.compile(
    r"heiterkeit|lachen|gelächter|hilarité|rires|laughter", re.IGNORECASE)


def _applause_columns(d: pd.DataFrame) -> pd.DataFrame:
    text = d["brackets"].fillna("").astype(str)
    d = d.copy()
    d["has_applause"] = text.str.contains(_APPLAUSE_RX, regex=True, na=False)
    d["has_laughter"] = text.str.contains(_LAUGHTER_RX, regex=True, na=False)
    d["has_reaction"] = d["has_applause"] | d["has_laughter"]
    return d


def applause_rate(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = _applause_columns(base(df, chamber))
    out = (d.groupby("party")["has_reaction"].mean() * 100).reindex(PARTIES).reset_index(name="reaction_rate_pct")
    return _w(out, name), "Share of speeches followed by an applause or laughter bracket, by party (%)."


def applause_type(df: pd.DataFrame, chamber: str, name: str) -> tuple[str, str]:
    d = _applause_columns(base(df, chamber))
    out = pd.DataFrame({
        "party": PARTIES,
        "applause_rate_pct": [d[d["party"].eq(p)]["has_applause"].mean() * 100 for p in PARTIES],
        "laughter_rate_pct": [d[d["party"].eq(p)]["has_laughter"].mean() * 100 for p in PARTIES],
    })
    return _w(out, name), "Applause vs laughter brackets, by party (%)."


def applause_top_mps(df: pd.DataFrame, chamber: str, name: str, n: int = 25) -> tuple[str, str]:
    d = _applause_columns(base(df, chamber))
    grp = d.groupby(["SpeakerFullName", "party"]).agg(
        n_speeches=("ID", "count"),
        n_reactions=("has_reaction", "sum"),
    ).reset_index()
    grp["reaction_rate_pct"] = grp["n_reactions"] / grp["n_speeches"] * 100
    out = grp.sort_values("n_reactions", ascending=False).head(n)
    return _w(out, name), f"Top {n} MPs by total applause/laughter brackets."


def chamber_emotional_share(df: pd.DataFrame, name: str) -> tuple[str, str]:
    d = df[df["party"].isin(PARTIES) & df["text_len"].gt(20) & df["year"].between(1999, 2025)
           & df["CouncilName"].isin(["Nationalrat", "Ständerat"])].copy()
    out = (d.groupby(["year", "CouncilName"])["emotional_rhetoric"].mean() * 100).reset_index(name="share_emotion_pct")
    out["chamber"] = out["CouncilName"].map(CHAMBER_LABEL)
    return _w(out[["year", "chamber", "share_emotion_pct"]], name), \
        "Share of emotional speeches by chamber-year (%)."


def chamber_anger(df: pd.DataFrame, name: str) -> tuple[str, str]:
    d = df[df["party"].isin(PARTIES) & df["text_len"].gt(20) & df["year"].between(1999, 2025)
           & df["CouncilName"].isin(["Nationalrat", "Ständerat"])].copy()
    d["anger"] = d["emotional_rhetoric"] & d["dominant_emotion"].eq("anger")
    out = (d.groupby(["year", "CouncilName"])["anger"].mean() * 100).reset_index(name="anger_share_pct")
    out["chamber"] = out["CouncilName"].map(CHAMBER_LABEL)
    return _w(out[["year", "chamber", "anger_share_pct"]], name), \
        "Share of speeches with dominant emotion = anger, by chamber-year (%)."


def country_compare(name: str) -> tuple[str, str]:
    out = pd.read_csv(SRC_CSV / "lt_country_emotionality_compare.csv")
    return _w(out, name), "Within-country yearly z-score of mean emotionality, six legislatures."


def fc_event(name: str) -> tuple[str, str]:
    out = pd.read_csv(SRC_CSV / "federal_council_event_time_defD_person_year.csv")
    return _w(out, name), "Share of emotional speeches per Federal Council member by relative year to entry."


def roesti_breakdown(name: str) -> tuple[str, str]:
    out = pd.read_csv(SRC_CSV / "roesti_energy_emotional_breakdown_defD.csv")
    return _w(out, name), "Albert Rösti energy speeches: emotional breakdown by business."


def roesti_over_time(name: str) -> tuple[str, str]:
    out = pd.read_csv(SRC_CSV / "roesti_emotions_over_time_defD.csv")
    return _w(out, name), "Albert Rösti yearly emotional breakdown across all speeches."


def ahef_chart(name: str) -> tuple[str, str]:
    shutil.copy2(AHEF_CSV, OUT / f"{name}.csv")
    return f"{name}.csv", "Year × party share for enthusiasm/anger/hope/fear (chart input as supplied)."


def topic_drivers_frozen(name: str, source: str) -> tuple[str, str]:
    shutil.copy2(SRC_CSV / source, OUT / f"{name}.csv")
    return f"{name}.csv", "Party contribution decomposition (early vs late, in pp)."


def emotion_drivers_frozen(name: str, source: str) -> tuple[str, str]:
    shutil.copy2(SRC_CSV / source, OUT / f"{name}.csv")
    return f"{name}.csv", "Within-topic emotional-rhetoric shift, decomposed by party (pp)."


# ---------- registry ----------
# Map figure stem (without .pdf) to a builder lambda.

def build_recipes() -> dict[str, Callable[[pd.DataFrame, str], tuple[str, str]]]:
    R: dict[str, Callable[[pd.DataFrame, str], tuple[str, str]]] = {}

    R["desc_06_floor_time_share"] = lambda df, n: floor_time_share(df, "Nationalrat", n)
    R["staenderat_desc_06_floor_time_share"] = lambda df, n: floor_time_share(df, "Ständerat", n)

    R["nationalrat_fig_parlacap_broad_topic_stacked_smoothed"] = lambda df, n: topic_distribution_broad(df, "Nationalrat", n)
    R["staenderat_fig_parlacap_broad_topic_stacked_smoothed"] = lambda df, n: topic_distribution_broad(df, "Ständerat", n)
    R["staenderat_fig_v4_topic_merged9_noproc_stacked"] = lambda df, n: topic_distribution_top(df, "Ständerat", n, 9)

    R["fig_v4_topic_merged_party_noproc"] = lambda df, n: topic_emphasis_party(df, "Nationalrat", n)
    R["staenderat_fig_v4_topic_merged_party_noproc"] = lambda df, n: topic_emphasis_party(df, "Ständerat", n)

    R["fig_lt_topic_shift_noproc"] = lambda df, n: topic_shift(df, "Nationalrat", n, "delta")
    R["staenderat_fig_lt_topic_shift_noproc"] = lambda df, n: topic_shift(df, "Ständerat", n, "delta")
    R["fig_lt_topic_shift_proportion"] = lambda df, n: topic_shift(df, "Nationalrat", n, "ratio")
    R["staenderat_fig_lt_topic_shift_proportion"] = lambda df, n: topic_shift(df, "Ständerat", n, "ratio")

    R["nationalrat_fig_topic_shift_drivers_by_party"] = lambda df, n: topic_drivers_frozen(n, "nationalrat_topic_shift_drivers_by_party.csv")
    R["staenderat_fig_topic_shift_drivers_by_party"] = lambda df, n: topic_drivers_frozen(n, "staenderat_topic_shift_drivers_by_party.csv")

    R["fig_lt_defD_overall_time"] = lambda df, n: defD_overall(df, "Nationalrat", n)
    R["staenderat_fig_lt_defD_overall_time"] = lambda df, n: defD_overall(df, "Ständerat", n)

    R["fig_lt_defD_by_party"] = lambda df, n: defD_by_party(df, "Nationalrat", n)
    R["staenderat_fig_lt_defD_by_party"] = lambda df, n: defD_by_party(df, "Ständerat", n)

    R["fig_lt_defD_by_legislature"] = lambda df, n: defD_by_legislature(df, "Nationalrat", n)
    R["staenderat_fig_lt_defD_by_legislature"] = lambda df, n: defD_by_legislature(df, "Ständerat", n)

    R["nationalrat_fig_lt_ga_emotionality_time"] = lambda df, n: emotion_score_time(df, "Nationalrat", n)
    R["staenderat_fig_lt_ga_emotionality_time"] = lambda df, n: emotion_score_time(df, "Ständerat", n)
    R["nationalrat_fig_lt_ga_emotionality_party_facet"] = lambda df, n: emotion_score_party(df, "Nationalrat", n)
    R["staenderat_fig_lt_ga_emotionality_party_facet"] = lambda df, n: emotion_score_party(df, "Ständerat", n)

    R["fig_lt_llm_emo_index100"] = lambda df, n: emotion_index_100(df, "Nationalrat", n)
    R["staenderat_fig_lt_llm_emo_index100"] = lambda df, n: emotion_index_100(df, "Ständerat", n)

    R["fig_lt_llm_dominant_party"] = lambda df, n: dominant_party(df, "Nationalrat", n)
    R["staenderat_fig_lt_llm_dominant_party"] = lambda df, n: dominant_party(df, "Ständerat", n)

    for emo in EMOTIONS:
        R[f"fig_lt_llm_{emo}_share_party_facet"] = lambda df, n, e=emo: emotion_share_party_facet(df, "Nationalrat", n, e)
        R[f"staenderat_fig_lt_llm_{emo}_share_party_facet"] = lambda df, n, e=emo: emotion_share_party_facet(df, "Ständerat", n, e)

    R["emotions_ahef_over_time_by_party_from_csv"] = lambda df, n: ahef_chart(n)

    R["fig_lt_emotionality_topic_large"] = lambda df, n: emotion_topic(df, "Nationalrat", n)
    R["staenderat_fig_lt_emotionality_topic_large"] = lambda df, n: emotion_topic(df, "Ständerat", n)
    R["fig_lt_emotionality_topic_by_party_heatmap"] = lambda df, n: emotion_topic_party(df, "Nationalrat", n)
    R["fig_lt_emotionality_topic_by_party_bars"] = lambda df, n: emotion_topic_party(df, "Nationalrat", n)
    R["staenderat_fig_lt_emotionality_topic_by_party_heatmap"] = lambda df, n: emotion_topic_party(df, "Ständerat", n)
    R["staenderat_fig_lt_emotionality_topic_by_party_bars"] = lambda df, n: emotion_topic_party(df, "Ständerat", n)

    R["fig_GA_party_ratio_by_topic"] = lambda df, n: party_ratio_topic("Nationalrat", n)
    R["staenderat_fig_GA_party_ratio_by_topic"] = lambda df, n: party_ratio_topic("Ständerat", n)

    R["fig_lt_defD_topic_shift"] = lambda df, n: defD_topic_shift(df, "Nationalrat", n)
    R["staenderat_fig_lt_defD_topic_shift"] = lambda df, n: defD_topic_shift(df, "Ständerat", n)
    R["fig_lt_defD_topic_shift_party"] = lambda df, n: defD_topic_shift_party(df, "Nationalrat", n)
    R["staenderat_fig_lt_defD_topic_shift_party"] = lambda df, n: defD_topic_shift_party(df, "Ständerat", n)
    R["fig_lt_defD_by_topic"] = lambda df, n: defD_topic_shift(df, "Nationalrat", n, "delta_pp")
    R["staenderat_fig_lt_defD_by_topic"] = lambda df, n: defD_topic_shift(df, "Ständerat", n, "delta_pp")

    R["nationalrat_fig_emotion_shift_within_topic_drivers_by_party"] = lambda df, n: emotion_drivers_frozen(n, "nationalrat_emotion_shift_within_topic_drivers_by_party.csv")
    R["nationalrat_fig_emotion_shift_within_topic_driver_table"] = lambda df, n: emotion_drivers_frozen(n, "nationalrat_emotion_shift_within_topic_drivers_by_party.csv")
    R["staenderat_fig_emotion_shift_within_topic_drivers_by_party"] = lambda df, n: emotion_drivers_frozen(n, "staenderat_emotion_shift_within_topic_drivers_by_party.csv")
    R["staenderat_fig_emotion_shift_within_topic_driver_table"] = lambda df, n: emotion_drivers_frozen(n, "staenderat_emotion_shift_within_topic_drivers_by_party.csv")

    R["fig_lt_gender_mp_vs_speeches_party"] = lambda df, n: gender_mp_vs_speeches(df, "Nationalrat", n)
    R["staenderat_fig_lt_gender_mp_vs_speeches_party"] = lambda df, n: gender_mp_vs_speeches(df, "Ständerat", n)
    R["fig_lt_gender_metrics_bars_1999_2025_nationalrat"] = lambda df, n: gender_metrics_chamber("Nationalrat", n)
    R["fig_lt_gender_metrics_bars_1999_2025_staenderat"] = lambda df, n: gender_metrics_chamber("Ständerat", n)

    R["fig_lt_canton_share"] = lambda df, n: canton_share(df, "Nationalrat", n)
    R["staenderat_fig_lt_canton_share"] = lambda df, n: canton_share(df, "Ständerat", n)
    R["fig_lt_canton_per_mp"] = lambda df, n: canton_per_mp(df, "Nationalrat", n)
    R["staenderat_fig_lt_canton_per_mp"] = lambda df, n: canton_per_mp(df, "Ständerat", n)
    R["fig_lt_canton_gap"] = lambda df, n: canton_gap(df, "Nationalrat", n)
    R["staenderat_fig_lt_canton_gap"] = lambda df, n: canton_gap(df, "Ständerat", n)

    R["fig_lt_udc_blocher_anger"] = lambda df, n: svp_anger_time(df, n)
    R["fig_lt_udc_blocher_prepost"] = lambda df, n: anger_prepost_2007(df, "Nationalrat", n)

    R["fig_lt_cohort_emotional"] = lambda df, n: cohort_emotional(df, "Nationalrat", n)
    R["staenderat_fig_lt_cohort_emotional"] = lambda df, n: cohort_emotional(df, "Ständerat", n)
    R["fig_lt_age_group_emotional"] = lambda df, n: age_group_emotional(df, "Nationalrat", n)
    R["staenderat_fig_lt_age_group_emotional"] = lambda df, n: age_group_emotional(df, "Ständerat", n)
    R["fig_lt_emotion_language"] = lambda df, n: emotion_language(df, "Nationalrat", n)
    R["staenderat_fig_lt_emotion_language"] = lambda df, n: emotion_language(df, "Ständerat", n)

    R["fig_app_rate_by_party"] = lambda df, n: applause_rate(df, "Nationalrat", n)
    R["staenderat_fig_app_rate_by_party"] = lambda df, n: applause_rate(df, "Ständerat", n)
    R["fig_app_type_by_party"] = lambda df, n: applause_type(df, "Nationalrat", n)
    R["staenderat_fig_app_type_by_party"] = lambda df, n: applause_type(df, "Ständerat", n)
    R["fig_app_top_mps"] = lambda df, n: applause_top_mps(df, "Nationalrat", n)
    R["staenderat_fig_app_top_mps"] = lambda df, n: applause_top_mps(df, "Ständerat", n)

    R["fig_lt_country_emotionality_compare"] = lambda df, n: country_compare(n)
    R["fig_lt_council_emotional_defD_chambers"] = lambda df, n: chamber_emotional_share(df, n)
    R["fig_lt_council_anger"] = lambda df, n: chamber_anger(df, n)
    R["fig_federal_council_event_emotionality_defD"] = lambda df, n: fc_event(n)
    R["fig_roesti_energy_emotional_breakdown_defD"] = lambda df, n: roesti_breakdown(n)
    R["fig_roesti_emotions_over_time_defD"] = lambda df, n: roesti_over_time(n)

    return R


def parse_deck() -> list[dict]:
    tex = SLIDES.read_text(encoding="utf-8")
    frame_re = re.compile(r"\\begin\{frame\}(?:\[[^\]]*\])?\{([^{}]*)\}(.*?)\\end\{frame\}", re.S)
    inc_re = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}")
    rows = []
    for idx, m in enumerate(frame_re.finditer(tex), start=1):
        title = m.group(1).strip()
        for fig in inc_re.findall(m.group(2)):
            name = Path(fig).name
            rows.append({"slide_index": idx, "slide_title": title,
                         "figure_file": name, "stem": Path(name).stem})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    deck = parse_deck()
    recipes = build_recipes()
    df = load_dataset()

    # Avoid recomputing the same recipe twice (figures duplicated across slides).
    cache: dict[str, tuple[str, str]] = {}
    manifest = []
    missing = []
    for row in deck:
        stem = row["stem"]
        if stem not in recipes:
            missing.append(stem)
            continue
        if stem not in cache:
            cache[stem] = recipes[stem](df, stem)
        csv_name, definition = cache[stem]
        manifest.append({
            "slide_index": row["slide_index"],
            "slide_title": row["slide_title"],
            "figure_file": row["figure_file"],
            "figure_data_csv": csv_name,
            "definition": definition,
        })

    pd.DataFrame(manifest).to_csv(OUT / "manifest.csv", index=False)
    print(f"wrote {len(set(m['figure_data_csv'] for m in manifest))} chart CSVs "
          f"covering {len(manifest)} figure references.")
    if missing:
        raise SystemExit("Missing recipes for: " + ", ".join(sorted(set(missing))))


if __name__ == "__main__":
    main()
