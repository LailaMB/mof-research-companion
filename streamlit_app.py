import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st
import os
import pandas as pd
import math
from typing import Any
import concurrent.futures
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json

import traceback


from agents.mof_finder import load_csv_if_exists, unify_columns, parse_user_request, filter_and_rank, display_table_with_pagination
from agents.mof_analyzer import analyze_xrd_filebuffer, analyze_bet_filebuffer, analyze_xps_filebuffer, synthesize_results
from agents.mof_builder import rule_based_predict

# -------------------------
# Dataset paths
# -------------------------
CORE_CSV = "data/core_mofs.csv"        # dataset 1 (CoRE)
HYMARC_CSV = "data/tps_clean.csv"  # dataset 2 (hymarc)
# -------------------------


# --- Page config ---
st.set_page_config(page_title="MOF Research Companion",
                   layout="wide",
                   initial_sidebar_state="collapsed")


# ----------------- CONFIG  -----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "models/gemini-2.5-flash-lite"
PAGE_SIZE = 5
# ---------------------------------------------


# --- Centering help: create wide center column ---
left, main, right = st.columns([1, 6, 1])

with main:
    # Header area
    col_logo, col_text = st.columns([1, 7])
    with col_logo:
        st.image("image/logo.png" if Path("image/logo.png").exists() else "https://via.placeholder.com/88.png?text=🔬", width=150)

    with col_text:
        st.title("MOF Research Companion")
        st.markdown("I am your companion in building MOFs for H₂ storage")

    st.write("")  # spacing

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔎 MOF Finder",
        "🧩 MOF Builder",
        "⚙️ MOF Modifier",
        "🔬 MOF Analyzer"
    ])

    with tab1:

        # -------------------------
        # Main Streamlit UI
        # -------------------------
        st.header("🔎 MOF Finder for H₂ Storage")
        st.markdown("Enter your objective and constraints and the finder will search datasets for candidate MOFs.")

        # dataset selection
        use_core = True
        use_hymarc = True

        # load requested datasets
        dfs_raw = {}
        if use_core:
            dfs_raw["CoRE"] = load_csv_if_exists(CORE_CSV)
        if use_hymarc:
            dfs_raw["HYMARC_real"] = load_csv_if_exists(HYMARC_CSV)


        if not dfs_raw:
            st.warning("No datasets selected or no dataset files found. Please enable a dataset in the sidebar and ensure file paths exist.")
            st.stop()

        # unify
        with st.spinner("Normalizing datasets..."):
            unified = unify_columns(dfs_raw)

        st.success(f"Loaded {len(unified)} MOF entries from 2 datasets.")

        # user query
        user_query = st.text_area("Describe what you need (example: 'Find MOFs to maximize gravimetric hydrogen uptake with density < 0.5 g/cm3 and open metal sites, prefer Cu')", height=120)
        run_search = st.button("Search MOFs", width="stretch", type="primary")

        # show parsed extraction for transparency
        #if user_query.strip():
        #    parsed = parse_user_request(user_query)
        #    st.markdown("**Parsed request (debug)**")
        #    st.json(parsed)

        if run_search:
            parsed = parse_user_request(user_query)
            candidates = filter_and_rank(unified, parsed)
            # show top N filter
            top_n = st.number_input("Top N results to show per page", min_value=5, max_value=200, value=10, step=5)
            display_table_with_pagination(candidates, page_size=top_n)
            st.session_state["last_results"] = candidates

        # quick show last results if exist
        #if "last_results" in st.session_state and st.session_state["last_results"] is not None:
        #    st.markdown("---")
        #    st.subheader("Last results (cached)")
        #    display_table_with_pagination(st.session_state["last_results"], page_size=st.number_input("Page size (use same value as above)", min_value=5, max_value=200, value=10, key="ps2"))

        # End of app

    with tab2:
        
        st.header("🧩 MOF Builder")
        st.write("Build your MOF from SBUs and linkers and get a quick property estimate.")

        c1, c2 = st.columns(2)
        with c1:
            metal = st.multiselect("Select metal node", ["Zn", "Cu", "Mg", "Al", "Li"], 
                                   default=["Zn"])
            topology = "pcu"

        with c2:
            linker = st.multiselect("Linker", ["BDC", "BPDC", "BTB", "TPDC"],
                                    default=["BDC"])
            target_pore = 20
        # choose prediction mode
        mode = "rule-based"
        if st.button("Predict properties", key="predict_props", width="stretch", type="primary"):
            result = rule_based_predict(metal[0], linker[0])

            st.success(f"Predicted properties for {metal}-{linker} {result}.")

    with tab3:
        # --- MOF Modification Module (Yaghi-inspired) ---
        st.markdown("## ⚙️ Modification Module — Omar Yaghi pathways")
        st.markdown(
            "Choose a pathway below."
        )

        def _heuristic_surface_area(linker, avoid_interpenetration, topology, target_pore_A):
            base = {"BDC": 800, "BPDC": 1200, "BTB": 1500, "TPDC": 1400}.get(linker.upper(), 800)
            topo_mult = {"pcu": 1.0, "rht": 1.15, "fcu": 1.05}.get(topology, 1.0)
            interpen_mult = 1.15 if avoid_interpenetration else 1.0
            pore_mult = 1.0 + max(0, (target_pore_A - 8) * 0.02)
            sa = base * topo_mult * interpen_mult * pore_mult
            pv = max(0.1, 0.18 + (target_pore_A - 5) * 0.03) * topo_mult
            # rough H2 wt% estimate at 77K (very naive)
            h2_wt_77k = min(10.0, (sa / 1000.0) * 1.6 + pv * 2.0)
            return round(sa), round(pv, 3), round(h2_wt_77k, 2)

        def _heuristic_qst(functional_group, metal, enable_oms):
            base_qst = {"none": 6.0, "-oh": 7.5, "-nh2": 8.5, "-f": 6.5}.get(functional_group.lower(), 6.0)
            metal_boost = {"LI": 3.0, "MG": 2.5, "NI": 1.2, "CO": 1.2}.get(metal.upper().replace("+",""), 0.5)
            oms_boost = 2.0 if enable_oms else 0.0
            qst = base_qst + metal_boost + oms_boost
            # percent change at 298 K: very rough mapping
            uptake_change_pct = min(200, (qst - 6.0) * 10)
            return round(qst, 2), round(uptake_change_pct, 1)

        def _heuristic_functionalization(groups, apply_on):
            # sum simple contributions
            mapping = {"-f": 1.0, "-oh": 1.2, "-nh2": 1.5, "-no2": 1.8}
            total = sum(mapping.get(g.lower(), 0.5) for g in groups)
            binding_improvement = min(6.0, total * 1.2)
            updated_formula = "Linker–" + ",".join(groups) if apply_on == "Linker" else "Node–" + ",".join(groups)
            return updated_formula, round(binding_improvement, 2)

        def _heuristic_spillover(catalyst, support, two_step):
            base = {"Pd": 0.5, "Pt": 0.6, "Ni": 0.3}.get(catalyst, 0.1)
            support_mult = {"Carbon": 1.0, "Graphene": 1.15, "CNTs": 1.2}.get(support, 1.0)
            two_step_mult = 1.25 if two_step else 1.0
            uptake_298 = round(min(5.0, base * support_mult * two_step_mult * 5.0), 2)
            recommended_loading = round(min(5.0, base * 3.0), 2)  # wt% placeholder
            return uptake_298, recommended_loading

        def _heuristic_pore_opt(target_nm, linker_length, preserve_topo):
            target_A = target_nm * 10.0
            # predicted pore window (Å) approximated
            predicted_window = round(target_A * (1.0 if not preserve_topo else 0.95), 2)
            # surface area vs pore size (naive)
            sa = int(max(200, 1000 * (target_nm / 1.2)))
            deliverable = round(max(0.5, (sa / 1000.0) * 1.2 * (1 - abs(target_nm-1.05)/2.5)), 2)
            stability_warning = None
            if target_nm > 1.6:
                stability_warning = "Warning: pore size may exceed typical stability limits for some MOFs."
            return predicted_window, sa, deliverable, stability_warning

        # Layout: we'll show 5 expanders, each with left inputs and right outputs
        st.write("")  # spacer

        # 1. Increasing Surface Area
        with st.expander("1️⃣ Increasing Surface Area", expanded=False):
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Goal:** Maximize total adsorption capacity (wt%) of H₂.")
                st.write("- Use longer linkers (e.g., BTB instead of BDC).")
                st.write("- Reduce interpenetration between frameworks.")
                st.write("- Select open topologies (e.g., rht, pcu).")
                st.write("**Inputs**")
                sa_linker = st.selectbox("Select linker type", options=["BDC", "BTB", "BPDC", "TPDC"], index=0, key="sa_linker")
                sa_avoid_inter = st.radio("Avoid interpenetration?", options=["Yes","No"], index=0, key="sa_inter")
                sa_topology = st.selectbox("Topology preference", options=["pcu", "rht", "fcu"], index=0, key="sa_topo")
                sa_target_pore = st.slider("Target pore size (Å)", min_value=3, max_value=30, value=int(target_pore), key="sa_pore")

            with right:
                st.markdown("**Predicted outputs**")
                sa_pred, pv_pred, h2w_pred = _heuristic_surface_area(sa_linker, sa_avoid_inter=="Yes", sa_topology, sa_target_pore)
                st.metric("Predicted surface area (m²/g)", f"{sa_pred:,}")
                st.metric("Predicted pore volume (cm³/g)", f"{pv_pred}")
                st.metric("Estimated H₂ wt% at 77 K", f"{h2w_pred}%")
                #st.image("image/3d_placeholder.png" if Path("image/3d_placeholder.png").exists() else None,
                #       caption="3D model placeholder (before/after)", use_container_width=True)

        # 2. Increase Isosteric Heat of Adsorption (Qst)
        with st.expander("2️⃣ Increase Isosteric Heat of Adsorption (Qst)", expanded=False):
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Goal:** Strengthen H₂–framework interactions (target Qst ≈ 10–15 kJ/mol).")
                st.write("- Add polar functional groups (–OH, –NH₂).")
                st.write("- Use metals with higher charge density (Li⁺, Mg²⁺).")
                st.write("- Expose open metal sites (OMS).")
                st.write("**Inputs**")
                qst_func = st.selectbox("Add functional group", options=["none", "-OH", "-NH2", "-F"], index=0, key="qst_func")
                qst_metal = st.selectbox("Select metal ion", options=["Li+", "Mg2+", "Ni2+", "Co2+"], index=1, key="qst_metal")
                qst_oms = st.checkbox("Enable open metal sites (OMS)?", value=False, key="qst_oms")
            with right:
                st.markdown("**Predicted outputs**")
                qst_val, uptake_pct = _heuristic_qst(qst_func, qst_metal, qst_oms)
                st.metric("Predicted Qst (kJ/mol)", f"{qst_val}")
                st.metric("Change in H₂ uptake at 298 K", f"{uptake_pct}%")
                st.markdown("**Recommended metal–linker combo**")
                st.write(f"- Example: {qst_metal.split('+')[0]}–BDC{qst_func}")
                st.write("**Heat map placeholder:**")
                st.write(" (heatmap of adsorption energy would be displayed here) ")
                #st.image("image/heatmap_placeholder.png" if Path("image/heatmap_placeholder.png").exists() else None, use_column_width=True)

        # 3. Functionalization
        with st.expander("3️⃣ Functionalization", expanded=False):
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Goal:** Introduce chemical groups to improve H₂ binding strength.")
                st.write("- Add electron-withdrawing / donating groups: –F, –OH, –NH₂, –NO₂.")
                st.write("- Enhance dipole interactions and local fields.")
                st.write("**Inputs**")
                func_groups = st.multiselect("Choose functional groups", options=["-F", "-OH", "-NH2", "-NO2"], default=["-NH2"], key="func_groups")
                apply_on = st.radio("Apply on linker or node?", options=["Linker", "Metal Node"], index=0, key="apply_on")
            with right:
                st.markdown("**Predicted outputs**")
                if func_groups:
                    updated_formula, binding_imp = _heuristic_functionalization(func_groups, apply_on)
                    st.write("**Updated linker/node formula:**")
                    st.code(updated_formula)
                    st.metric("Predicted improvement in H₂ binding (kJ/mol)", f"+{binding_imp}")
                    st.write("**Simulated IR bands:**")
                    st.write("- New bands expected at ~3200–3600 cm⁻¹ (OH/NH) and ~1100–1350 cm⁻¹ (C–F/C–N).")
                    st.write("**Reference examples:**")
                    st.markdown("- MOF-74–NH₂, UiO-66–NH₂ (examples)")
                else:
                    st.info("Select functional groups.")

        # 4. Spillover Mechanism
        with st.expander("4️⃣ Spillover Mechanism", expanded=False):
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Goal:** Use catalytic routes to enhance storage at ambient temperature.")
                st.write("- Add catalysts (Pd, Pt) to dissociate H₂ and enable spillover.")
                st.write("**Inputs**")
                spill_cat = st.selectbox("Select catalyst", ["Pd", "Pt", "Ni"], index=0, key="spill_cat")
                spill_support = st.selectbox("Select support material", ["Carbon", "Graphene", "CNTs"], index=0, key="spill_support")
                spill_two = st.checkbox("Two-step spillover model?", value=False, key="spill_two")
            with right:
                st.markdown("**Predicted outputs**")
                uptake_298, rec_loading = _heuristic_spillover(spill_cat, spill_support, spill_two)
                st.metric("Predicted H₂ uptake at 298 K (wt%)", f"{uptake_298}")
                st.metric("Recommended catalyst loading (wt%)", f"{rec_loading}")
                st.write("**Visual diagram:**")
                #st.write("(spillover pathway animation or simple SVG would be here)")

        # 5. Pore Size Optimization
        with st.expander("5️⃣ Pore Size Optimization", expanded=False):
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Goal:** Adjust pore diameter to maximize storage efficiency (ideal 0.9–1.2 nm).")
                st.write("**Inputs**")
                pore_target_nm = st.slider("Target pore size (nm)", min_value=0.5, max_value=2.0, value=round(target_pore/10, 2), step=0.05, key="pore_target")
                pore_linker_length = st.selectbox("Select linker length", ["BDC", "BPDC", "BTB", "TPDC"], index=0, key="pore_linker")
                preserve_topo = st.checkbox("Preserve original topology", value=True, key="preserve_topo")
            with right:
                st.markdown("**Predicted outputs**")
                pred_window_A, sa_est, deliverable, warning = _heuristic_pore_opt(pore_target_nm, pore_linker_length, preserve_topo)
                st.metric("Predicted pore window (Å)", f"{pred_window_A}")
                st.metric("Surface area (approx m²/g)", f"{sa_est}")
                st.metric("Estimated deliverable H₂ capacity (wt%)", f"{deliverable}")
                if warning:
                    st.warning(warning)

            #st.caption("This module is an interface scaffold. Replace placeholder heuristics with trained predictors, physics-based calculators, or visualizers (CIF → 3D viewer) as you build them.")      

    with tab4:

        st.header("🔬 MOF Analyzer")
        st.write("Upload experimental files (BET, XRD, XPS/FTIR). The analyzer will detect type by filename/columns.")

        # Advanced option: use ADK/LLM agents if available (guarded)
        use_adk = False
        try:
            import google.adk  # type: ignore
            use_adk = st.checkbox("Use Google ADK agents for synthesis (requires ADK config)", value=False)
        except Exception:
            st.info("Google ADK not found: using local deterministic analyzers + rule-based synthesizer.")

        uploaded_files = st.file_uploader("Upload files (CSV)", accept_multiple_files=True, help="Upload BET/XRD/XPS CSV files", type=["csv"])
        if not uploaded_files:
            st.markdown("**Instrument suggestions**")
            st.markdown("- BET → Surface area & porosity (isotherm CSV: p_over_p0, volume)")
            st.markdown("- XRD → Structural confirmation (2theta,intensity)")
            st.markdown("- XPS → Elemental hints & peak positions (be,intensity)")
        else:
            # map filenames to analyzer function
            def guess_instrument(filename: str, df_preview: pd.DataFrame = None) -> str:
                fn = filename.lower()
                if "xrd" in fn or "2theta" in fn or (df_preview is not None and any("2theta" in c.lower() for c in df_preview.columns)):
                    return "XRD"
                if "bet" in fn or "isotherm" in fn or "p_over" in fn or (df_preview is not None and any("p_over" in c.lower() or "p/p0" in c.lower() for c in df_preview.columns)):
                    return "BET"
                if "xps" in fn or "be" in fn or (df_preview is not None and any("be" in c.lower() for c in df_preview.columns)):
                    return "XPS"
                # fallback: try to open and inspect columns
                return "unknown"

            st.info(f"Received {len(uploaded_files)} file(s). Running analyzers...")

            results = {}
            plots_data = {}

            progress_bar = st.progress(0)
            total = len(uploaded_files)
            completed = 0

            # run analyzers in threadpool
            with ThreadPoolExecutor(max_workers=3) as ex:
                future_map = {}
                for f in uploaded_files:
                    # read few bytes to decide
                    raw = f.read()
                    buf = io.BytesIO(raw)  # keep buffer to pass to analyzer
                    # attempt a quick preview to guide guess
                    try:
                        preview_df = pd.read_csv(io.BytesIO(raw), nrows=5)
                    except Exception:
                        preview_df = None
                    instr = guess_instrument(f.name, preview_df)
                    if instr == "XRD":
                        future = ex.submit(analyze_xrd_filebuffer, io.BytesIO(raw))
                    elif instr == "BET":
                        future = ex.submit(analyze_bet_filebuffer, io.BytesIO(raw))
                    elif instr == "XPS":
                        future = ex.submit(analyze_xps_filebuffer, io.BytesIO(raw))
                    else:
                        # try all analyzers and pick the one that returns non-error first (best-effort)
                        def try_all(buff):
                            for fntry in (analyze_xrd_filebuffer, analyze_bet_filebuffer, analyze_xps_filebuffer):
                                try:
                                    r = fntry(io.BytesIO(buff.getvalue()))
                                    # accept result if it has instrument key
                                    if isinstance(r, dict) and any(k in r for k in ("xrd","bet","xps")):
                                        return r
                                except Exception:
                                    continue
                            return {"error": "Unknown file type or parsing failed."}
                        future = ex.submit(try_all, io.BytesIO(raw))
                    future_map[future] = f.name

                for fut in as_completed(future_map):
                    fname = future_map[fut]
                    try:
                        res = fut.result(timeout=30)
                    except Exception as e:
                        res = {"error": f"Analyzer crashed: {e}\n{traceback.format_exc()}"}
                    # detect which analyzer returned
                    if "xrd" in res:
                        results["xrd"] = res["xrd"]
                        plots_data["xrd_df"] = res.get("_df")
                    elif "bet" in res:
                        results["bet"] = res["bet"]
                        plots_data["bet_df"] = res.get("_df")
                    elif "xps" in res:
                        results["xps"] = res["xps"]
                        plots_data["xps_df"] = res.get("_df")
                    else:
                        # unknown or error
                        results.setdefault("errors", []).append({fname: res})
                    completed += 1
                    progress_bar.progress(int(completed / total * 100))

            # show results in columns
            col_left, col_right = st.columns([2, 3])
            with col_left:
                st.subheader("Instrument outputs (JSON)")
                st.json(results)
                st.download_button("Download report (JSON)", json.dumps(results, indent=2), file_name="analysis_results.json", mime="application/json")

            with col_right:
                st.subheader("Visuals & quick metrics")
                # BET plot
                if "bet_df" in plots_data and plots_data["bet_df"] is not None:
                    dfbet = plots_data["bet_df"]
                    # detect columns
                    pcol, vcol = dfbet.columns[0], dfbet.columns[1]
                    fig, ax = plt.subplots()
                    ax.plot(dfbet[pcol], dfbet[vcol], marker="o", linestyle="-")
                    ax.set_xlabel(pcol); ax.set_ylabel(vcol); ax.set_title("BET isotherm (demo)")
                    st.pyplot(fig)
                    if "bet" in results and results["bet"].get("surface_area_m2_g") is not None:
                        st.metric("Estimated surface area (m²/g)", f"{results['bet']['surface_area_m2_g']:.1f}")
                        st.metric("Estimated pore volume (cm³/g)", f"{results['bet']['pore_volume_cm3_g']:.3f}")

                # XRD plot with peak markers
                if "xrd_df" in plots_data and plots_data["xrd_df"] is not None:
                    dfx = plots_data["xrd_df"]
                    ang = dfx.iloc[:, 0]; inten = dfx.iloc[:, 1]
                    fig2, ax2 = plt.subplots()
                    ax2.plot(ang, inten, lw=0.8)
                    if "xrd" in results:
                        peaks = results["xrd"]["peaks_2theta"]
                        for pk in peaks:
                            ax2.axvline(pk, color="red", alpha=0.6, linestyle="--")
                    ax2.set_xlabel(dfx.columns[0]); ax2.set_ylabel(dfx.columns[1]); ax2.set_title("XRD pattern (demo)")
                    st.pyplot(fig2)
                    if "xrd" in results:
                        st.write(f"Detected peaks: {results['xrd']['peaks_2theta']}")

                # XPS plot
                if "xps_df" in plots_data and plots_data["xps_df"] is not None:
                    dfx = plots_data["xps_df"]
                    be = dfx.iloc[:, 0]; inten = dfx.iloc[:, 1]
                    fig3, ax3 = plt.subplots()
                    ax3.plot(be, inten, lw=0.8)
                    if "xps" in results:
                        for pk in results["xps"]["peaks_be"]:
                            ax3.axvline(pk, color="green", alpha=0.6, linestyle="--")
                    ax3.set_xlabel(dfx.columns[0]); ax3.set_ylabel(dfx.columns[1]); ax3.set_title("XPS (demo)")
                    st.pyplot(fig3)
                    if "xps" in results:
                        st.write(f"XPS element hints: {results['xps'].get('elements_detected', [])}")

            # Synthesize (rule-based) and show final report
            st.markdown("---")
            st.subheader("Synthesis / Final Report (rule-based)")
            final_report = synthesize_results(results)
            st.json(final_report)
            st.download_button("Download final report (JSON)", json.dumps(final_report, indent=2), file_name="final_characterization_report.json", mime="application/json")

            # Optional: If user checked use_adk and adk available, we could pass results to ADK merger agent.
            if use_adk:
                st.warning("ADK integration requested. ADK path is not implemented in this demo block; if you want, I can add guarded code to call your ParallelAgent and merger (requires ADK SDK + credentials).")


# end of centered main column




