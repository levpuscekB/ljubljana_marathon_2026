"""Data preparation and statistical utilities for 10/21/42 km Ljubljana results.

This module intentionally does not import Streamlit: it can be tested directly.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from race_config import RACES, ALL_SPLITS

def distances_for(race):
    config = RACES[str(race)]
    return [0.0] + [float(k) for k in config["checkpoints"]] + [config["km"]]

def checkpoints_for(race):
    return [(float(k), f"{k}km_net") for k in RACES[str(race)]["checkpoints"]]
MALE = "#2D77D0"
FEMALE = "#EB6EAA"
SELECTED = "#F3A528"
COMPARE = "#0CA899"
BASELINE = "#161D29"

COL_ALIASES = {
    "gender": "gender", "sex": "gender", "spol": "gender",
    "rank_gross": "rank_gross", "gross rank": "rank_gross", "rank": "rank_gross",
    "bib": "bib", "startna st": "bib", "st": "bib", "st.": "bib",
    "name": "name", "priimek in ime": "name", "runner": "name",
    "country": "country", "club": "club", "klub": "club",
    "5km_net": "5km_net", "5 km (net)": "5km_net", "5km": "5km_net",
    "10km_net": "10km_net", "10 km (net)": "10km_net", "10km": "10km_net",
    "15km_net": "15km_net", "15 km (net)": "15km_net", "15km": "15km_net",
    "20km_net": "20km_net", "20 km (net)": "20km_net", "20km": "20km_net",
    "30km_net": "30km_net", "30 km (net)": "30km_net", "30km": "30km_net",
    "distance_km": "distance_km", "distance (km)": "distance_km",
    "net_time": "net_time", "net time": "net_time", "netto": "net_time",
    "gross_time": "gross_time", "gross time": "gross_time", "rezultat": "gross_time",
}


def normalize_text(x):
    s = unicodedata.normalize("NFKD", str(x or ""))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def clock(seconds, signed=False, compact=False):
    if seconds is None or not np.isfinite(float(seconds)):
        return "—"
    seconds = float(seconds)
    sign = "+" if seconds >= 0 else "−"
    seconds = int(round(abs(seconds)))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    value = f"{h}:{m:02}:{s:02}" if (h or not compact) else f"{m}:{s:02}"
    return sign + value if signed else value


def parse_seconds(v):
    # IMPORTANT: pandas.NaT is an instance of datetime, but calling its .time()
    # raises "NaTType does not support time". Check nulls BEFORE datetime.
    # Missing split times are normal and must not invalidate the finisher.
    if v is None or v is pd.NaT or v is pd.NA:
        return np.nan
    try:
        if bool(pd.isna(v)):
            return np.nan
    except (TypeError, ValueError):
        pass
    if isinstance(v, (pd.Timedelta, timedelta)):
        return float(v.total_seconds())
    if isinstance(v, (time, datetime)):
        t = v.time() if isinstance(v, datetime) else v
        return float(t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6)
    if isinstance(v, (int, float, np.number)):
        # Excel stores times as a fraction of 24h. For CSVs, numeric durations
        # above 1 are interpreted as seconds, not as spreadsheet fractions.
        val = float(v)
        return 86400.0 * val if 0 < val < 1 else val
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "nat", "—", "-", "dnf", "dns"}:
        return np.nan
    s = s.replace(",", ".")
    if re.fullmatch(r"\d{1,3}:\d{1,2}(?::\d{1,2}(?:\.\d+)?)?", s):
        parts = list(map(float, s.split(":")))
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    try:
        return float(pd.to_timedelta(s).total_seconds())
    except (ValueError, TypeError):
        return np.nan


def read_results(source, name=None):
    """Read the scraper's CSV/XLSX as well as its older displayed-header XLSX."""
    if name is None:
        name = getattr(source, "name", str(source))
    if str(name).lower().endswith(".xlsx"):
        raw = pd.read_excel(source, dtype={"Bib": str, "bib": str})
    else:
        raw = pd.read_csv(source, encoding="utf-8-sig", dtype={"bib": str, "Bib": str})
    return prepare_results(raw)


def prepare_results(raw):
    ren = {}
    for c in raw.columns:
        key = normalize_text(str(c).strip()).replace("\u00a0", " ")
        ren[c] = COL_ALIASES.get(key, c)
    df = raw.rename(columns=ren).copy()
    required = {"gender", "bib", "name", "net_time", "gross_time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}. Expected the scraper CSV or XLSX.")
    for col in [f"{km}km_net" for km in ALL_SPLITS]:
        if col not in df:
            df[col] = np.nan
    df["gender"] = df["gender"].astype(str).str.upper().str.strip().replace({
        "MALE": "M", "MOŠKI": "M", "MOSKI": "M", "FEMALE": "F", "Ž": "F", "Z": "F", "ZENSKE": "F", "ŽENSKE": "F"})
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["bib"] = df["bib"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    for src in ["net_time", "gross_time"] + [f"{km}km_net" for km in ALL_SPLITS]:
        df[src + "_sec"] = df[src].map(parse_seconds)
    df["start_delay_sec"] = df["gross_time_sec"] - df["net_time_sec"]
    before = len(df)
    valid = (
        df.gender.isin(["M", "F"]) & df.name.ne("") & df.bib.ne("") &
        (df.net_time_sec > 0) & (df.gross_time_sec >= df.net_time_sec) &
        (df.start_delay_sec >= 0)
    )
    df = df.loc[valid].drop_duplicates(["gender", "bib"], keep="first").reset_index(drop=True)
    df["runner_id"] = df.gender + ":" + df.bib
    df["search"] = (df.name.map(normalize_text) + " " + df.bib.astype(str))
    df["color"] = np.where(df.gender == "M", MALE, FEMALE)
    df.attrs["rejected"] = before - len(df)
    return df


def percentile_summary(df, col):
    arr = df[col].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return None
    q = np.percentile(arr, [5, 25, 50, 75, 95])
    return {"n": len(arr), "mean": float(arr.mean()),
            "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4])}


def detect_checkpoint_basis(df, race="21"):
    """Estimate checkpoint clock basis; uncertain where checkpoints are far from finish.

    In the existing 21k file we can directly see many 20k checkpoints later
    than the runner's net finish, proving they are gun elapsed. That inference
    cannot be transferred as proof to 42k or 10k, where splits are earlier.
    The same timing provider suggests gross as a default; make it overridable.
    """
    checkpoint = f"{RACES[str(race)]['checkpoints'][-1]}km_net_sec"
    subset = df[[checkpoint, "net_time_sec"]].dropna()
    if str(race) == '21' and len(subset) >= 20:
        fraction = (subset[checkpoint] > subset['net_time_sec'] + 5).mean()
        return 'gross' if fraction > .10 else 'net'
    return 'gross'  # assumption, not verified from these finish times


def get_profile(rec, checkpoint_basis="net", race="21"):
    """Actual observed checkpoint times, no interpolated intermediate splits."""
    net = float(rec["net_time_sec"])
    delay = float(rec["start_delay_sec"])
    pts = [(0.0, 0.0)]
    last = 0.0
    for km, col in checkpoints_for(race):
        val = rec.get(col + "_sec", np.nan)
        if pd.isna(val):
            continue
        sec = float(val) - (delay if checkpoint_basis == "gross" else 0.0)
        if last < sec < net:
            pts.append((km, sec))
            last = sec
    pts.append((RACES[str(race)]['km'], net))
    return pts


def pack_profiles(df, checkpoint_basis="net", sample_limit=None, race="21"):
    """Combine all grey trajectories into a single separated GPU trace."""
    if sample_limit and len(df) > sample_limit:
        idx = np.linspace(0, len(df)-1, sample_limit, dtype=int)
        df = df.iloc[idx]
    x, y = [], []
    n = 0
    for _, rec in df.iterrows():
        prof = get_profile(rec, checkpoint_basis, race)
        if len(prof) < 3:
            continue  # do not invent pacing from only a finish result
        for km, sec in prof:
            x.append(km); y.append(sec / 60.0)
        x.append(None); y.append(None)
        n += 1
    return x, y, n


def mean_profile(df, checkpoint_basis="net", race="21"):
    distances = distances_for(race)
    observed, complete = [], []
    for _, rec in df.iterrows():
        prof = get_profile(rec, checkpoint_basis, race)
        if len(prof) >= 3:
            observed.append(dict(prof))
        if len(prof) == len(distances):
            complete.append(dict(prof))
    if complete:
        return distances, [np.mean([v[d] for v in complete]) / 60 for d in distances], len(complete), True
    if observed:
        mean = [float(np.mean([v[d] for v in observed if d in v])) / 60
                if any(d in v for v in observed) else np.nan for d in distances]
        return distances, mean, len(observed), False
    return [], [], 0, False


def pava(values, weights):
    """Weighted isotonic regression (pool adjacent violators), non-decreasing."""
    blocks = []
    for i, (v, w) in enumerate(zip(values, weights)):
        blocks.append([float(v), float(w), i, i])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            b = blocks.pop()
            a = blocks.pop()
            tw = a[1] + b[1]
            blocks.append([(a[0]*a[1] + b[0]*b[1]) / tw, tw, a[2], b[3]])
    out = np.empty(len(values), float)
    for level, _, first, last in blocks:
        out[first:last+1] = level
    return out


def fit_delay_model(group, min_n=60):
    """Estimate typical chip time conditional on start delay, within one gender.

    Empirical equal-count bins then an increasing isotonic baseline. This is an
    observational association, not an effect caused by the starting position.
    """
    group = group.dropna(subset=["start_delay_sec", "net_time_sec"]).copy()
    if len(group) < min_n:
        return None
    n_bins = int(np.clip(len(group)//110, 4, 28))
    group = group.sort_values(["start_delay_sec", "net_time_sec"]).reset_index(drop=True)
    group["bin"] = (np.arange(len(group)) * n_bins // len(group)).clip(0, n_bins - 1)
    tab = group.groupby("bin", sort=True).agg(
        delay=("start_delay_sec", "median"),
        median=("net_time_sec", "median"),
        q25=("net_time_sec", lambda s: s.quantile(.25)),
        q75=("net_time_sec", lambda s: s.quantile(.75)),
        n=("net_time_sec", "size")
    ).reset_index(drop=True)
    # Dense ties at the same delay may place several equal x values in bins.
    # Merge equal delays with count-weighted summaries. Avoid DataFrameGroupBy
    # apply(include_groups=False), which is not supported by pandas 2.1.
    counts = tab.groupby("delay", sort=True)["n"].sum()
    merged = {}
    for column in ["median", "q25", "q75"]:
        merged[column] = ((tab[column] * tab["n"]).groupby(tab["delay"]).sum() / counts)
    tab = pd.DataFrame({**merged, "n": counts}).reset_index()
    if len(tab) < 2:
        return None
    tab["expected"] = pava(tab["median"].to_numpy(), tab["n"].to_numpy())
    return tab


def predict_delay(model, delays):
    return np.interp(np.asarray(delays, dtype=float), model["delay"].to_numpy(),
                     model["expected"].to_numpy())


def apply_delay_models(df):
    out = df.copy()
    out["expected_sec"] = np.nan
    out["overperformance_sec"] = np.nan
    models = {}
    for gender in ["M", "F"]:
        sub = out[out.gender == gender]
        model = fit_delay_model(sub)
        models[gender] = model
        if model is None:
            continue
        exp = predict_delay(model, sub.start_delay_sec)
        out.loc[sub.index, "expected_sec"] = exp
        out.loc[sub.index, "overperformance_sec"] = exp - sub.net_time_sec.to_numpy(dtype=float)
    return out, models


def local_percentile(df, selected, k_floor=65):
    """Nearby-delay comparison; percent faster than people with similar delay."""
    pool = df[df.gender == selected["gender"]].copy()
    if len(pool) < 60:
        return None
    k = min(len(pool), max(k_floor, min(300, int(round(len(pool) * .04)))))
    pool["distance"] = (pool.start_delay_sec - float(selected["start_delay_sec"])).abs()
    nearby = pool.nsmallest(k, ["distance", "net_time_sec"])
    pct = float((nearby.net_time_sec > float(selected["net_time_sec"])).mean() * 100)
    return {"percentile": pct, "n": len(nearby),
            "delay_min": float(nearby.start_delay_sec.min()),
            "delay_max": float(nearby.start_delay_sec.max())}
