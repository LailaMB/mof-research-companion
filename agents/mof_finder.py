# mof_finder_app.py
import re
import math
import io
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st

#from llama_index.llms.google_genai import GoogleGenAI
#from llama_index.llms import PandasQueryEngine


st.set_page_config(page_title="MOF Finder", layout="wide")


# -------------------------
# Utility & caching
# -------------------------
@st.cache_data
def load_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not path or not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None


def unify_columns(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a unified dataframe with common columns used for search/ranking.
    Returns dataframe with columns:
      ['source','name','refcode','density_g_cm3','gsa_m2g','vsa_m2_cm3','vf','pv_cm3_g',
       'lcd_A','pld_A','ug_at_ps','uv_at_ps','ug_at_tps','uv_at_tps','metal_types','has_oms','thermal_stability_C', ...]
    """
    records = []
    for source, df in dfs.items():
        if df is None:
            continue
        cols = {c.lower(): c for c in df.columns}

        def get(col_options, default=None):
            for c in col_options:
                if c.lower() in cols:
                    return df[cols[c.lower()]]
            return pd.Series([default] * len(df)) if df is not None else None

        # Best-effort column mappings
        density = get(["Density", "density", "Density (g/cm3)", "density_g_cm3", "Density_g_cm3"], default=math.nan)
        # surface area: try several names (ASA, GSA, GSA?)
        gsa = get(["ASA (m2/g)", "ASA (m2/cm3)", "GSA", "GSA (m2/g)", "gsa", "ASA", "ASA (A2)"], default=math.nan)
        # volumetric surface area / VSA
        vsa = get(["VSA", "VSA (m2/cm3)", "vsa"], default=math.nan)
        # pore volume
        pv = get(["PV (cm3/g)", "PV", "pv", "PV (A3)"], default=math.nan)
        # VF
        vf = get(["VF", "vf"], default=math.nan)
        lcd = get(["LCD (Å)", "LCD", "lcd", "lcd_A"], default=math.nan)
        pld = get(["PLD (Å)", "PLD", "pld", "pld_A"], default=math.nan)
        # hydrogen uptakes: names differ; check UG/UV
        ug_ps = get(["UG at PS", "UG at PS", "UG at PS (wt%)", "ug_at_ps", "UG at PS (wt%)"], default=math.nan)
        uv_ps = get(["UV at PS", "UV at PS", "uv_at_ps"], default=math.nan)
        ug_tps = get(["UG at TPS", "UG at TPS", "UG at TPS (wt%)", "ug_at_tps"], default=math.nan)
        uv_tps = get(["UV at TPS", "UV at TPS", "uv_at_tps"], default=math.nan)

        # Name and refcode
        name = df[cols.get("name", "Name")] if "name" in cols else df.iloc[:, 0]
        ref = df[cols.get("CSD refc.", "CSD refc.")] if "csd refc." in cols or "CSD refc." in df.columns else df.columns[0] if df.shape[1] > 1 else None

        # other metadata
        metal = get(["Metal Types", "Metal", "metal", "metal_types"], default="").astype(str)
        oms = get(["Has OMS", "Has OMS?", "Has OMS", "Has_OMS", "has_oms"], default="").astype(str)
        thermal = get(["Thermal_stability (℃)", "Thermal_stability", "Thermal Stability", "thermal_stability_C", "thermal_stability (℃)"], default=math.nan)

        # Build records
        for i in range(len(df)):
            rec = {
                "source": source,
                "name": str(name.iloc[i]) if name is not None else "",
                "refcode": str(df.iloc[i][0]) if ref is None else (str(ref.iloc[i]) if isinstance(ref, pd.Series) else ""),
                "density_g_cm3": try_float_from_series(density, i),
                "gsa_m2_g": try_float_from_series(gsa, i),
                "vsa_m2_cm3": try_float_from_series(vsa, i),
                "vf": try_float_from_series(vf, i),
                "pv_cm3_g": try_float_from_series(pv, i),
                "lcd_A": try_float_from_series(lcd, i),
                "pld_A": try_float_from_series(pld, i),
                "ug_at_ps": try_float_from_series(ug_ps, i),
                "uv_at_ps": try_float_from_series(uv_ps, i),
                "ug_at_tps": try_float_from_series(ug_tps, i),
                "uv_at_tps": try_float_from_series(uv_tps, i),
                "metal_types": metal.iloc[i] if isinstance(metal, pd.Series) else str(metal),
                "has_oms": str(oms.iloc[i]) if isinstance(oms, pd.Series) else str(oms),
                "thermal_stability_C": try_float_from_series(thermal, i),
                # keep original row index for traceability
                "_orig_index": i
            }
            records.append(rec)
    unified = pd.DataFrame.from_records(records)
    return unified

def try_float_from_series(series, i):
    try:
        if isinstance(series, pd.Series):
            v = series.iloc[i]
            if pd.isna(v):
                return math.nan
            # remove commas and non-numeric chars
            if isinstance(v, str):
                v2 = re.sub(r"[^\d\.\-eE]", "", v)
                return float(v2) if v2 not in ("", "-", None) else math.nan
            return float(v)
    except Exception:
        return math.nan

# -------------------------
# Simple NL parser for objective
# -------------------------
def parse_user_request(text: str) -> Dict[str, Any]:
    """
    Extract objective and simple filters from text.
    returns dict with keys:
      - objective: "gravimetric" | "volumetric" | "surface_area" | "pore_volume" | "density_low" | "density_high" | None
      - metals: list of metals requested
      - require_oms: True/False/None
      - min_density / max_density (floats or None)
      - min_gsa / max_gsa (floats or None)
      - keywords: list of other tokens to match in name
    """
    t = text.lower()
    out = {
        "objective": None,
        "metals": [],
        "require_oms": None,
        "min_density": None,
        "max_density": None,
        "min_gsa": None,
        "max_gsa": None,
        "keywords": []
    }

    # objective keywords
    if any(k in t for k in ["gravimetric", "weight percent", "wt%", "wt %", "gravimetric uptake", "gravimetric capacity"]):
        out["objective"] = "gravimetric"
    if any(k in t for k in ["volumetric", "volumetric uptake", "volume uptake", "v/v"]):
        out["objective"] = "volumetric"
    if any(k in t for k in ["surface area", "high surface area", "asa", "gsa"]):
        out["objective"] = "surface_area"
    if any(k in t for k in ["pore volume", "pv", "pore-volume"]):
        out["objective"] = "pore_volume"

    # metal detection (simple)
    metals = ["cu", "zn", "mg", "al", "fe", "co", "ni", "mn", "li", "ca", "zro", "zr", "cr"]
    for m in metals:
        # word boundaries to avoid partial matches
        if re.search(rf"\b{re.escape(m)}\b", t):
            out["metals"].append(m.upper())

    # OMS requirement
    if "oms" in t or "open metal" in t or "open metal site" in t:
        out["require_oms"] = True
    if "no oms" in t or "without oms" in t:
        out["require_oms"] = False

    # numeric filters: density, gsa, pv
    # patterns like "density > 0.4" or "density < 0.5 g/cm3" or "density between 0.2 and 0.6"
    for field in ["density", "gsa", "pv"]:
        # between
        m_between = re.search(rf"{field}.*?between\s*([\d\.]+)\s*(?:and|to)\s*([\d\.]+)", t)
        if m_between:
            v1 = float(m_between.group(1))
            v2 = float(m_between.group(2))
            lo, hi = min(v1, v2), max(v1, v2)
            if field == "density":
                out["min_density"] = lo
                out["max_density"] = hi
            if field == "gsa":
                out["min_gsa"] = lo
                out["max_gsa"] = hi
            if field == "pv":
                out["min_pv"] = lo
                out["max_pv"] = hi

        # greater than / less than
        m_gt = re.search(rf"{field}.*?(?:>=|>|greater than|above)\s*([\d\.]+)", t)
        m_lt = re.search(rf"{field}.*?(?:<=|<|less than|below)\s*([\d\.]+)", t)
        if m_gt:
            val = float(m_gt.group(1))
            if field == "density":
                out["min_density"] = val
            if field == "gsa":
                out["min_gsa"] = val
            if field == "pv":
                out["min_pv"] = val
        if m_lt:
            val = float(m_lt.group(1))
            if field == "density":
                out["max_density"] = val
            if field == "gsa":
                out["max_gsa"] = val
            if field == "pv":
                out["max_pv"] = val

    # fallback: if user mentions "best hydrogen uptake" prefer gravimetric
    if "hydrogen uptake" in t or "hydrogen storage" in t or "store hydrogen" in t:
        if out["objective"] is None:
            out["objective"] = "gravimetric"

    # free keywords (names, e.g., "MOF-5")
    tokens = re.findall(r"[A-Za-z0-9\-\_]{2,}", text)
    out["keywords"] = tokens

    return out

# -------------------------
# Filtering & ranking
# -------------------------
def filter_and_rank(df: pd.DataFrame, parsed: Dict[str, Any]) -> pd.DataFrame:
    # copy
    d = df.copy()

    # Apply metal filter if requested
    if parsed["metals"]:
        # metal_types may be multi token; do simple substring matching
        metals_upper = [m.upper() for m in parsed["metals"]]
        d = d[d["metal_types"].fillna("").str.upper().apply(lambda s: any(m in s for m in metals_upper))]

    # OMS filter
    if parsed["require_oms"] is True:
        d = d[d["has_oms"].fillna("").str.lower().isin(["yes", "true", "y", "1"])]
    elif parsed["require_oms"] is False:
        d = d[~d["has_oms"].fillna("").str.lower().isin(["yes", "true", "y", "1"])]

    # numeric filters
    if parsed.get("min_density") is not None:
        d = d[d["density_g_cm3"].fillna(-1) >= parsed["min_density"]]
    if parsed.get("max_density") is not None:
        d = d[d["density_g_cm3"].fillna(1e9) <= parsed["max_density"]]
    if parsed.get("min_gsa") is not None:
        d = d[d["gsa_m2_g"].fillna(-1) >= parsed["min_gsa"]]
    if parsed.get("max_gsa") is not None:
        d = d[d["gsa_m2_g"].fillna(1e9) <= parsed["max_gsa"]]

    # Keyword matching boosts scores for names/refcodes that contain requested tokens
    d["_keyword_score"] = 0.0
    keywords = [k.lower() for k in parsed.get("keywords", []) if len(k) > 2]
    if keywords:
        def kscore(row):
            s = (str(row["name"]) + " " + str(row.get("refcode",""))).lower()
            score = sum(2.0 if k in s else 0.0 for k in keywords)
            return score
        d["_keyword_score"] = d.apply(kscore, axis=1)

    # Ranking logic: depends on objective
    obj = parsed.get("objective")
    # primary ranking metric name and fallback
    if obj == "gravimetric":
        # prefer UG at PS or UG at TPS; fallback to gsa (higher surface area -> often higher gravimetric uptake)
        d["_rank_metric"] = d["ug_at_ps"].fillna(d["ug_at_tps"]).fillna(d["gsa_m2_g"] * 0.001)
    elif obj == "volumetric":
        # prefer UV at PS or UV at TPS; fallback to vsa or density * ug estimate
        d["_rank_metric"] = d["uv_at_ps"].fillna(d["uv_at_tps"]).fillna(d["vsa_m2_cm3"]).fillna(d["density_g_cm3"] * d["ug_at_ps"].fillna(0))
    elif obj == "surface_area":
        d["_rank_metric"] = d["gsa_m2_g"].fillna(d["vsa_m2_cm3"])
    elif obj == "pore_volume":
        d["_rank_metric"] = d["pv_cm3_g"].fillna(d["vf"])
    else:
        # default ranking: gravimetric-like if H2 mentioned else GSA
        d["_rank_metric"] = d["ug_at_ps"].fillna(d["gsa_m2_g"] * 0.001)

    # Combine ranking metric and keyword score
    # If metric missing, treat as very small
    d["_rank_metric"] = pd.to_numeric(d["_rank_metric"], errors="coerce").fillna(-1e9)
    d["_score"] = d["_rank_metric"] + d["_keyword_score"]

    # Sort desc by score
    d_sorted = d.sort_values("_score", ascending=False).reset_index(drop=True)
    # keep only relevant display columns
    display_cols = ["source", "name", "refcode", "density_g_cm3", "gsa_m2_g", "vsa_m2_cm3", "pv_cm3_g",
                    "lcd_A", "pld_A", "ug_at_ps", "uv_at_ps", "ug_at_tps", "uv_at_tps", "metal_types", "has_oms", "thermal_stability_C", "_score"]
    present_cols = [c for c in display_cols if c in d_sorted.columns]
    return d_sorted[present_cols]

# -------------------------
# Pagination helper
# -------------------------
def display_table_with_pagination(df: pd.DataFrame, page_size: int = 10, key_prefix: str = "results"):
    if df is None or df.empty:
        st.info("No results to display.")
        return
    total = len(df)
    pages = math.ceil(total / page_size)
    if f"{key_prefix}_page" not in st.session_state:
        st.session_state[f"{key_prefix}_page"] = 0
    page = st.session_state[f"{key_prefix}_page"]

    left_col, mid_col, right_col = st.columns([1, 6, 1])
    with mid_col:
        st.write(f"Showing results {(page*page_size)+1}–{min((page+1)*page_size, total)} of {total}")
        start = page * page_size
        end = start + page_size
        sub = df.iloc[start:end].copy()
        # show an interactive table
        st.dataframe(sub, use_container_width=True)
        # CSV download
        csv_buf = io.StringIO()
        sub.to_csv(csv_buf, index=False)
        st.download_button("Download page CSV", csv_buf.getvalue(), file_name=f"mof_results_page{page+1}.csv", mime="text/csv")

    # pagination controls
    colp1, colp2, colp3, colp4 = st.columns([1,1,1,1])
    if colp1.button("⏮ First"):
        st.session_state[f"{key_prefix}_page"] = 0
    if colp2.button("◀ Prev"):
        st.session_state[f"{key_prefix}_page"] = max(0, st.session_state[f"{key_prefix}_page"] - 1)
    if colp3.button("Next ▶"):
        st.session_state[f"{key_prefix}_page"] = min(pages-1, st.session_state[f"{key_prefix}_page"] + 1)
    if colp4.button("Last ⏭"):
        st.session_state[f"{key_prefix}_page"] = pages-1
