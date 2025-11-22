# --- MOF Analyzer (Streamlit-friendly) ---
import io
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_xrd_filebuffer(buffer: io.BytesIO) -> dict:
    """Expect CSV with columns: 2theta,intensity"""
    try:
        df = pd.read_csv(buffer)
    except Exception as e:
        return {"error": f"Failed to read CSV: {e}"}
    # try find appropriate columns
    cols = [c.lower() for c in df.columns]
    # find intensity and angle columns
    angle_col = None
    inten_col = None
    for c in df.columns:
        cl = c.lower()
        if "2theta" in cl or "2 theta" in cl or "theta" == cl or "angle" in cl:
            angle_col = c
        if "intensity" in cl or "counts" in cl:
            inten_col = c
    if angle_col is None:
        # try first two columns
        if df.shape[1] >= 2:
            angle_col = df.columns[0]
            inten_col = df.columns[1]
        else:
            return {"error": "Could not find angle/intensity columns in XRD CSV."}

    if inten_col is None:
        inten_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]

    x = pd.to_numeric(df[angle_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[inten_col], errors="coerce").to_numpy()
    if len(x) < 10:
        return {"error": "XRD CSV appears too short."}

    # peak detection
    try:
        threshold = np.nanmax(y) * 0.05 if np.nanmax(y) > 0 else 0.0
        peaks_idx, props = find_peaks(y, height=threshold, distance=max(1, len(x)//100))
        peaks_2theta = x[peaks_idx].round(3).tolist()
    except Exception as e:
        return {"error": f"Peak detection failed: {e}"}

    result = {
        "instrument": "XRD",
        "n_points": int(len(x)),
        "peaks_2theta": peaks_2theta,
        "n_peaks": int(len(peaks_2theta)),
        "notes": "Detected peaks using scipy.find_peaks (demo).",
        "columns": [angle_col, inten_col],
        "raw_preview": df.head(5).to_dict(orient="records")
    }
    return {"xrd": result, "_df": df}  # include df for plotting downstream

def analyze_bet_filebuffer(buffer: io.BytesIO) -> dict:
    """Expect CSV with columns: p_over_p0,volume (or similar)"""
    try:
        df = pd.read_csv(buffer)
    except Exception as e:
        return {"error": f"Failed to read CSV: {e}"}
    # find columns
    pcol = None
    vcol = None
    for c in df.columns:
        cl = c.lower()
        if "p_over" in cl or "p/p0" in cl or "p_over_p0" in cl or "relative" in cl or "p0" in cl:
            pcol = c
        if "volume" in cl or "loading" in cl or "amount" in cl:
            vcol = c
    if pcol is None or vcol is None:
        # fallback to first two columns
        if df.shape[1] >= 2:
            pcol = df.columns[0]
            vcol = df.columns[1]
        else:
            return {"error": "Could not find isotherm columns in BET CSV."}

    df = df.sort_values(pcol)
    # simple fallback BET-like estimate (demo)
    try:
        mask = (df[pcol] > 0.05) & (df[pcol] < 0.30)
        if mask.sum() >= 3:
            x = df.loc[mask, pcol].astype(float).to_numpy()
            y = df.loc[mask, vcol].astype(float).to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            demo_area = abs(slope) * 1000.0
            method = "fallback_polyfit"
        else:
            demo_area = None
            method = "insufficient_points"
    except Exception as e:
        demo_area = None
        method = f"error:{e}"

    result = {
        "instrument": "BET",
        "n_points": int(len(df)),
        "surface_area_m2_g": float(demo_area) if demo_area is not None else None,
        "pore_volume_cm3_g": float(df[vcol].max()) if vcol in df else None,
        "isotherm_type": "I/II (demo)",
        "method": method,
        "columns": [pcol, vcol],
        "raw_preview": df.head(5).to_dict(orient="records")
    }
    return {"bet": result, "_df": df}

def analyze_xps_filebuffer(buffer: io.BytesIO) -> dict:
    """Expect CSV with columns: be,intensity (binding energy)")
    Performs baseline median subtraction + peak detection; returns detected ranges as demo.
    """
    try:
        df = pd.read_csv(buffer)
    except Exception as e:
        return {"error": f"Failed to read CSV: {e}"}
    # guess columns
    becol = None
    icol = None
    for c in df.columns:
        cl = c.lower()
        if "be" in cl or "binding" in cl or "energy" in cl:
            becol = c
        if "intensity" in cl or "counts" in cl:
            icol = c
    if becol is None or icol is None:
        if df.shape[1] >= 2:
            becol = df.columns[0]
            icol = df.columns[1]
        else:
            return {"error": "Could not find BE/intensity columns in XPS CSV."}
    x = pd.to_numeric(df[becol], errors="coerce").to_numpy()
    y = pd.to_numeric(df[icol], errors="coerce").to_numpy()
    if len(x) < 10:
        return {"error": "XPS CSV appears too short."}
    # baseline and peak detect
    baseline = pd.Series(y).rolling(window=max(3, len(y)//80), min_periods=1, center=True).median().to_numpy()
    signal = y - baseline
    try:
        peaks_idx, props = find_peaks(signal, height=np.nanmax(signal)*0.05 if np.nanmax(signal)>0 else 0, distance=max(1, len(x)//100))
        be_peaks = x[peaks_idx].round(2).tolist()
    except Exception as e:
        return {"error": f"Peak detection failed: {e}"}

    # naive element hints (demo)
    detected = []
    for bev in be_peaks:
        if 280 <= bev <= 290:
            detected.append("C1s")
        elif 528 <= bev <= 535:
            detected.append("O1s")
        elif 330 <= bev <= 360:
            detected.append("Pd3d_or_metal")
    detected = list(dict.fromkeys(detected))

    result = {
        "instrument": "XPS",
        "n_points": int(len(x)),
        "peaks_be": be_peaks,
        "elements_detected": detected,
        "notes": "Baseline-subtracted signal + peak detection (demo).",
        "columns": [becol, icol],
        "raw_preview": df.head(5).to_dict(orient="records")
    }
    return {"xps": result, "_df": df}

# ---------- Simple synthesizer (rule-based merger) ----------
def synthesize_results(results: dict) -> dict:
    """
    results: dict may contain keys "xrd","bet","xps" (each is dict)
    Returns structured JSON with summary & mechanism_hint.
    """
    summary = []
    mechanism_hint = "insufficient data"
    try:
        bet = results.get("bet")
        xrd = results.get("xrd")
        xps = results.get("xps")

        if bet:
            sa = bet.get("surface_area_m2_g")
            pv = bet.get("pore_volume_cm3_g")
            if sa:
                summary.append(f"BET area ~ {sa:.0f} m²/g (demo estimate).")
            if pv:
                summary.append(f"Pore volume ~ {pv:.3f} cm³/g.")
        if xrd:
            n_peaks = xrd.get("n_peaks", 0)
            summary.append(f"XRD: {n_peaks} peaks detected (structural features visible).")
        if xps:
            elems = xps.get("elements_detected", [])
            if elems:
                summary.append("XPS indicates presence of " + ", ".join(elems) + ".")
        # simple mechanism rules
        if bet and bet.get("surface_area_m2_g") and bet.get("surface_area_m2_g") > 3000:
            mechanism_hint = "physisorption-dominated; high surface area suggests high gravimetric capacity at low T."
        elif xps and ("Pd3d_or_metal" in xps.get("elements_detected", [])):
            mechanism_hint = "possible spillover or metal-mediated adsorption (check XPS Pd signature)."
        elif bet and bet.get("surface_area_m2_g") and bet.get("surface_area_m2_g") < 500:
            mechanism_hint = "low surface area — adsorption likely limited; consider functionalization or OMS."

        final = {
            "summary": " ".join(summary) if summary else "No significant signatures found.",
            "mechanism_hint": mechanism_hint,
            "components": {k: v for k, v in results.items()}
        }
    except Exception as e:
        final = {"summary": "Synthesis failed", "error": str(e)}
    return final