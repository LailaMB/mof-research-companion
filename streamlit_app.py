import streamlit as st
import os
import pandas as pd
import math
from typing import Any
import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np


# ----------------- CONFIG  -----------------
CSV_PATH = "data/input.csv"
#os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
GEMINI_API_KEY = "AIzaSyAIBUKDUPGEBLDBYhzJD1DOa75etX1u044"
GEMINI_MODEL = "models/gemini-2.5-flash-lite"
PAGE_SIZE = 5
# ---------------------------------------------


# ---- Add Logo + Centered Title ----
st.markdown(
    """
    <div style="text-align: center;">
        <img src="image/logo.png" alt="Logo" width="150">
        <h1 style="font-size: 3rem; margin-top: 10px;">MOF Research Companion</h1>
        <p> I am your companion in building MOFs for H2 storage </p>

    </div>
    """,
    unsafe_allow_html=True
)


# Page configuration
st.set_page_config(
    page_title="MOF Research Companion",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-processing {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    #if 'reranking_results' not in st.session_state:
    #    st.session_state.reranking_results = {}
    #if 'current_query' not in st.session_state:
    #    st.session_state.current_query = ""
    #if 'current_documents' not in st.session_state:
    #    st.session_state.current_documents = []


# ---- Load CSV ----
@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def mof_finder_tab():
    """MOF finder interface"""

    try:
        df = load_df(CSV_PATH)
    except Exception as e:
        st.error(f"Failed to load CSV at {CSV_PATH}: {e}")
        st.stop()


    # ---- Initialize PandasQueryEngine with Gemini ----
    pandas_query_engine = None
    #try:
        #import llama_index
        #from llama_index.experimental.query_engine import PandasQueryEngine
        #from llama_index.llms.gemini import Gemini

        #llm = Gemini(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)
        #pandas_query_engine = PandasQueryEngine(df=df, llm=llm, verbose=False)
    #except Exception as e:
    #    st.error(f"Failed to initialize PandasQueryEngine or Gemini: {e}")
    #    st.stop()



    st.markdown("## 🎯 MOF Finder")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        #st.markdown("### Query Input")
        #query = st.text_input(
        #    "Enter your search query:",
        #    #value=st.session_state.current_query,
        #    placeholder="e.g., machine learning applications in healthcare"
        #)


        # ---- UI: query input ----
        query = st.text_input("Enter your search query:", value="", placeholder="e.g. Give me the MOF with the highest surface area")
        run = st.button("Run Query")

        # session state for results & pagination
        if "results_df" not in st.session_state:
            st.session_state["results_df"] = None
        if "page" not in st.session_state:
            st.session_state["page"] = 0
        if "raw_response" not in st.session_state:
            st.session_state["raw_response"] = None

        # ---- helper to convert LLM response into pandas DataFrame if possible ----
        def convert_response_to_dataframe(resp: Any) -> pd.DataFrame | None:
            """
            Try to convert the PandasQueryEngine response to a DataFrame.
            Handles: pandas Series / DataFrame / list-of-dicts / list-of-values.
            Otherwise returns None.
            """
            # If it's already a pandas object
            try:
                import pandas as _pd
                if isinstance(resp, _pd.DataFrame):
                    return resp.reset_index(drop=True)
                if isinstance(resp, _pd.Series):
                    return resp.to_frame().reset_index()
            except Exception:
                pass

            # If resp has metadata with pandas instruction and executed output
            try:
                # resp could be a llama-index response object; attempt to access .response or .metadata
                # Convert to string first then attempt to eval
                s = str(resp)
                # try eval safely for simple Python literal (list/tuple/dict)
                try:
                    obj = eval(s, {"__builtins__": {}}, {})
                except Exception:
                    obj = None
                if isinstance(obj, list):
                    # list of dicts?
                    if len(obj) > 0 and isinstance(obj[0], dict):
                        return pd.DataFrame(obj)
                    else:
                        # list of values -> single column
                        return pd.DataFrame({"value": obj})
                if isinstance(obj, dict):
                    return pd.DataFrame([obj])
            except Exception:
                pass

            # Last fallback: no parsable dataframe
            return None

        # ---- run query handler ----
        def run_query_and_store(q: str):
            st.session_state["raw_response"] = None
            st.session_state["results_df"] = None
            st.session_state["page"] = 0
            try:
                resp = pandas_query_engine.query(q)
                st.session_state["raw_response"] = resp  # keep raw for debug
                # try to convert to dataframe
                df_resp = convert_response_to_dataframe(resp)
                if df_resp is not None and len(df_resp) > 0:
                    st.session_state["results_df"] = df_resp
                else:
                    # if no dataframe, try to detect MOF-like strings within response and map to rows
                    text = str(resp)
                    # simple heuristic: search for MOF names existing in original df (case-insensitive)
                    candidates = []
                    lowered = text.lower()
                    for col in df.columns:
                        # consider only string columns
                        if pd.api.types.is_string_dtype(df[col]):
                            for idx, val in df[col].dropna().astype(str).iteritems():
                                v = val.strip()
                                if v.lower() in lowered or v.lower().replace(" ", "") in lowered:
                                    candidates.append(idx)
                    if candidates:
                        st.session_state["results_df"] = df.loc[list(dict.fromkeys(candidates))].reset_index(drop=True)
                    else:
                        # no matches -> return textual response only
                        st.session_state["results_df"] = None
            except Exception as e:
                st.error(f"PandasQueryEngine query failed: {e}")
                st.session_state["raw_response"] = None
                st.session_state["results_df"] = None

        # ---- run on submit ----
        if run and query.strip():
            run_query_and_store(query.strip())

        # ---- display results (pagination) ----
        results_df = st.session_state.get("results_df", None)
        raw_resp = st.session_state.get("raw_response", None)

        if results_df is None:
            st.write("No tabular results to show from the query.")
            if raw_resp is not None:
                st.subheader("Raw response")
                st.text(str(raw_resp))
        else:
            total = len(results_df)
            pages = math.ceil(total / PAGE_SIZE)
            page = st.session_state["page"]
            st.markdown(f"**Showing results page {page+1} / {pages}  —  total rows: {total}**")
            start = page * PAGE_SIZE
            end = min(start + PAGE_SIZE, total)
            page_df = results_df.iloc[start:end].copy()

            # show main information (pick first few columns to display)
            # Use up to 5 columns or user-friendly names if present
            display_cols = list(page_df.columns[:5])
            st.table(page_df[display_cols].reset_index(drop=True))

            # expanders to show details per row
            for i, row in page_df.iterrows():
                with st.expander(f"Details: row {i} — {row[display_cols[0]] if display_cols else i}"):
                    st.write(row.to_dict())

            # pagination controls
            cols = st.columns([1,1,6])
            if cols[0].button("Previous") and page > 0:
                st.session_state["page"] -= 1
            if cols[1].button("Next") and page < pages - 1:
                st.session_state["page"] += 1

      

    
    with col2:
        st.markdown("### Settings")
        top_k = st.slider("Results to return:", 1, 15, 8)
        techniques = st.multiselect(
            "Select techniques:",
            ["Cross-Encoder", "LLM-based", "Cohere API", "Hybrid", "Learning to Rank"],
            default=["Cross-Encoder", "Cohere API"]
        )

        st.markdown("### Quick Start with Presets")
        preset_category = st.selectbox("Choose a preset:", ["Custom", "Technology", "Science", "Business", "Healthcare"])
        
        if preset_category != "Custom":
            if st.button(f"Load {preset_category} Preset"):
                pass
                #preset_data = SAMPLE_QUERIES[preset_category]
                #st.session_state.current_query = preset_data["query"]
                #st.session_state.current_documents = preset_data["documents"]
                #st.rerun()
    
    
def mof_characterization_tab():
    """MOF characterization interface"""
    col1, col2 = st.columns([2, 1])
    with col1:

        # --------- Placeholders for your real analyzer functions ----------
        # If you already have analyze_xrd/analyze_bet/analyze_xps, import them instead.
        def analyze_xrd(path_or_bytes):
            # accept path or bytes-like (uploaded file)
            if hasattr(path_or_bytes, "read"):
                df = pd.read_csv(path_or_bytes)
            else:
                df = pd.read_csv(path_or_bytes)
            two_theta = df['2theta'].to_numpy()
            intensity = df['intensity'].to_numpy()
            # find peaks naive
            from scipy.signal import find_peaks
            peaks_idx, _ = find_peaks(intensity, height=np.max(intensity)*0.05, distance=5)
            peaks = two_theta[peaks_idx].round(3).tolist()
            return {"peaks": peaks, "n_peaks": len(peaks), "raw": {"two_theta": two_theta.tolist()[:10], "intensity": intensity.tolist()[:10]}}

        def analyze_bet(path_or_bytes):
            if hasattr(path_or_bytes, "read"):
                df = pd.read_csv(path_or_bytes)
            else:
                df = pd.read_csv(path_or_bytes)
            # simple demo
            mask = (df['p_over_p0']>0.05)&(df['p_over_p0']<0.3)
            if mask.sum()>2:
                slope, intercept = np.polyfit(df.loc[mask,'p_over_p0'].to_numpy(), df.loc[mask,'volume'].to_numpy(), 1)
                area = abs(slope)*1000
            else:
                area = None
            return {"surface_area_m2_g": area, "pore_vol_cm3_g": float(df['volume'].max()), "isotherm_sample": df.head(3).to_dict(orient='list')}

        def analyze_xps(path_or_bytes):
            if hasattr(path_or_bytes, "read"):
                df = pd.read_csv(path_or_bytes)
            else:
                df = pd.read_csv(path_or_bytes)
            be = df['be'].to_numpy()
            intensity = df['intensity'].to_numpy()
            from scipy.signal import find_peaks
            peaks_idx, _ = find_peaks(intensity - pd.Series(intensity).rolling(50,min_periods=1,center=True).median().to_numpy(),
                                    height=np.max(intensity)*0.05 if np.max(intensity)>0 else 0, distance=10)
            peaks_be = be[peaks_idx].round(2).tolist()
            return {"peaks_be": peaks_be, "elements_guess": ["C1s" if 280<p<290 else "O1s" if 525<p<535 else "metal" for p in peaks_be]}

# ---------------- Streamlit UI ----------------
#st.set_page_config(layout="wide", page_title="MOF Characterization Chat")

# Sidebar: file upload and settings
    with col2:
        st.header("Upload data")
        xrd_file = st.file_uploader("XRD CSV (2theta,intensity)", type=["csv"])
        bet_file = st.file_uploader("BET CSV (p_over_p0,volume)", type=["csv"])
        xps_file = st.file_uploader("XPS CSV (be,intensity)", type=["csv"])
        st.markdown("---")
        st.write("Or use example data")
        use_example = st.checkbox("Use example synthetic data", value=False)
        st.markdown("---")
        run_mode = st.selectbox("Run mode", ["Local (fast demo)", "ADK Agents (requires ADK)"])
        st.markdown("Progress / logs:")
        log_area = st.empty()

        # central layout: left chat, right tabs for agent outputs
        #col1, col2 = st.columns([1, 2])

    # Chat area (col1)
    with col1:
        # Agent tabs (col2)
        tabs = st.tabs(["Summary", "XRD", "BET", "XPS", "Function calls / Logs"])

        # Helper to run analyzers (local parallel)
        def run_local_analysis(xrd_in, bet_in, xps_in):
            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futures = {
                    ex.submit(analyze_xrd, xrd_in): "xrd",
                    ex.submit(analyze_bet, bet_in): "bet",
                    ex.submit(analyze_xps, xps_in): "xps"
                }
                for fut in concurrent.futures.as_completed(futures):
                    name = futures[fut]
                    try:
                        results[name] = fut.result()
                    except Exception as e:
                        results[name] = {"error": str(e)}
            return results

        # Provide example data if requested
        if use_example:
            # Create in-memory CSVs (simple)
            import io
            # XRD example
            two_theta = np.linspace(5,50,1000)
            intensity = np.exp(-0.5*((two_theta-14.8)/0.12)**2)*1200 + np.random.normal(0,2,size=two_theta.shape)
            xrd_buf = io.StringIO()
            pd.DataFrame({"2theta": two_theta, "intensity": intensity}).to_csv(xrd_buf, index=False)
            xrd_buf.seek(0)
            xrd_in = io.StringIO(xrd_buf.getvalue())
            # BET example
            p = np.linspace(0.01,0.99,200)
            V = 1.2*(p/(0.05+p))
            bet_buf = io.StringIO()
            pd.DataFrame({"p_over_p0": p, "volume": V}).to_csv(bet_buf, index=False)
            bet_buf.seek(0)
            bet_in = io.StringIO(bet_buf.getvalue())
            # XPS
            be = np.linspace(0,1200,2400)
            inten = np.exp(-0.5*((be-284.8)/0.6)**2)*120 + np.random.normal(0,1,size=be.shape)
            xps_buf = io.StringIO()
            pd.DataFrame({"be": be, "intensity": inten}).to_csv(xps_buf, index=False)
            xps_buf.seek(0)
            xps_in = io.StringIO(xps_buf.getvalue())
        else:
            xrd_in = xrd_file
            bet_in = bet_file
            xps_in = xps_file

        # Run analysis button
        if st.button("Run Analysis"):
            if not (xrd_in and bet_in and xps_in):
                st.warning("Please upload all three files or select example data.")
            else:
                log_area.info("Starting analysis...")
                if run_mode == "Local (fast demo)":
                    results = run_local_analysis(xrd_in, bet_in, xps_in)
                else:
                    # ADK calling placeholder: you should call runner.run_async(...).result() or run_live to stream
                    # For the hackathon keep local. If you want, I can add ADK runner code here.
                    results = run_local_analysis(xrd_in, bet_in, xps_in)

                # Save to session for display
                st.session_state.latest_results = results
                # Append to chat
                st.session_state.chat_history.append({"role":"system", "text": "Analysis complete. See per-agent tabs for details."})
                st.experimental_rerun()

        # Render tabs (if results exist)
        results = st.session_state.get("latest_results", None)

        with tabs[0]:
            st.header("Summary")
            if results:
                mech_hint = "physisorption" if (results.get("bet",{}).get("surface_area_m2_g") or 0) > 800 else "unknown"
                st.markdown(f"**Mechanism hint:** {mech_hint}")
                st.json({k: results[k] for k in results})
            else:
                st.info("No results yet. Upload files and press Run Analysis.")

        with tabs[1]:
            st.header("XRD")
            if results and "xrd" in results:
                st.json(results["xrd"])
                # show plot
                try:
                    # read data again for plotting
                    if hasattr(xrd_in, "read"):
                        xrd_in.seek(0)
                        df = pd.read_csv(xrd_in)
                    else:
                        df = pd.read_csv(xrd_in)
                    fig, ax = plt.subplots()
                    ax.plot(df['2theta'], df['intensity'])
                    ax.set_xlabel("2θ")
                    ax.set_ylabel("Intensity")
                    st.pyplot(fig)
                except Exception as e:
                    st.write("Could not plot XRD:", e)
            else:
                st.info("No XRD results")

        with tabs[2]:
            st.header("BET")
            if results and "bet" in results:
                st.json(results["bet"])
                # optional isotherm plot
                try:
                    if hasattr(bet_in, "read"):
                        bet_in.seek(0)
                        df = pd.read_csv(bet_in)
                    else:
                        df = pd.read_csv(bet_in)
                    fig, ax = plt.subplots()
                    ax.plot(df['p_over_p0'], df['volume'], marker='o')
                    ax.set_xlabel("P/P0")
                    ax.set_ylabel("Loading")
                    st.pyplot(fig)
                except Exception as e:
                    st.write("Could not plot BET:", e)
            else:
                st.info("No BET results")

        with tabs[3]:
            st.header("XPS")
            if results and "xps" in results:
                st.json(results["xps"])
                try:
                    if hasattr(xps_in, "read"):
                        xps_in.seek(0)
                        df = pd.read_csv(xps_in)
                    else:
                        df = pd.read_csv(xps_in)
                    fig, ax = plt.subplots()
                    ax.plot(df['be'], df['intensity'])
                    ax.set_xlabel("Binding energy (eV)")
                    ax.set_ylabel("Intensity")
                    st.pyplot(fig)
                except Exception as e:
                    st.write("Could not plot XPS:", e)
            else:
                st.info("No XPS results")

        with tabs[4]:       
            st.subheader("Chat")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            # display chat messages
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['text']}")
                else:
                    st.markdown(f"**System:** {msg['text']}")
            # input
            user_input = st.text_input("Enter a question or command (or press Run Analysis):", key="user_input")
            if st.button("Send"):
                st.session_state.chat_history.append({"role":"user","text": user_input})
                st.experimental_rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 MOF Finder", 
    "🧪 MOF Builder", 
    "🔬 MOF Modification",
    "📚 MOF Characterization"
])

with tab1:
    mof_finder_tab();

with tab2:
    pass;

with tab3:
    pass;

with tab4:
    mof_characterization_tab();

