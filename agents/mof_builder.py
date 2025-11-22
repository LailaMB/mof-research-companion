
import streamlit as st
import pandas as pd

import numpy as np
from itertools import product

#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline


# -------------------------
# Simple rule-based predictor
# -------------------------
def rule_based_predict(metal, linker):
    """
    Return a dict of simple heuristic predictions for density, gsa, pv.
    These numbers are illustrative and should be tuned by you.
    """

    topology = "pcu"
    target_pore = 20

    # Base priors (very rough)
    metal_density = {
        "ZN": 0.45, "CU": 0.50, "MG": 0.35, "AL": 0.60, "LI": 0.30
    }
    linker_area = {
        "BDC": 800, "BPDC": 1200, "BTB": 1500, "TPDC": 1400
    }
    topo_factor = {
        "pcu": 1.0, "fcu": 1.05, "dia": 0.9, "rht": 1.15, "qom": 1.1
    }
    # defaults
    d0 = metal_density.get(metal.upper(), 0.45)
    gsa0 = linker_area.get(linker.upper(), 800)
    topo_mult = topo_factor.get(topology, 1.0)

    # heuristics: larger pore -> lower density, higher pv, slightly higher surface area for certain linkers
    density = max(0.15, d0 * (1.0 - (target_pore - 5) * 0.02))
    pv = max(0.1, 0.2 + (target_pore - 5) * 0.03) * topo_mult
    gsa = max(100, gsa0 * topo_mult * (1.0 + (target_pore - 5) * 0.01))

    return {"density_g_cm3": round(density, 3), "gsa_m2_g": int(round(gsa)), "pv_cm3_g": round(pv, 3)}

