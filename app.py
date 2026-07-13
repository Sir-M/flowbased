import os
import streamlit as st
import polars as pl
import pandas as pd
import plotly.graph_objects as go
from jao import JaoPublicationToolPandasClient

# --- STEP 0: INITIALIZE APP LAYOUT & FULL SCREEN VIEWPORT CONFIGS ---
st.set_page_config(
    layout="wide", 
    page_title="CoreSGMv8 Grid Map",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }
        [data-testid="stVerticalBlock"] {
            gap: 0rem !important;
        }
        .stPlotlyChart {
            height: 100vh !important;
            width: 100vw !important;
        }
    </style>
""", unsafe_allow_html=True)

def resolve_file_path(default_path):
    if os.path.exists(default_path):
        return default_path
    stripped = default_path.lstrip("./").lstrip("../")
    if os.path.exists(stripped):
        return stripped
    relative = os.path.join("..", stripped)
    if os.path.exists(relative):
        return relative
    return default_path

GEO_CSV_PATH = resolve_file_path("../datasets/CoreSGMv8/substation_locations_exact.csv")
LINES_PATH = resolve_file_path("../datasets/CoreSGMv8/v8_Lines.parquet")
TIELINES_PATH = resolve_file_path("../datasets/CoreSGMv8/v8_Tielines.parquet")

def clean_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip().lower()

# --- STEP 1: LOAD GEOLOCATION DATABASE ---
@st.cache_data
def load_geolocation_map(path):
    if not os.path.exists(path):
        st.error(f"Critical asset missing: {path}")
        return {}, pl.DataFrame()
        
    df = pl.read_csv(path).drop_nulls("lat_lon").filter(pl.col("lat_lon") != "")
    
    df = df.with_columns(pl.col("lat_lon").str.replace_all(" ", ""))
    df = df.with_columns([
        pl.col("lat_lon").str.split(",").list.get(0).cast(pl.Float64).alias("lat"),
        pl.col("lat_lon").str.split(",").list.get(1).cast(pl.Float64).alias("lon")
    ])
    
    loc_map = {}
    for row in df.to_dicts():
        keys = list(row.keys())
        raw_name = str(row[keys[0]]).strip()
        tso_raw = str(row[keys[1]]).strip().lower() if len(keys) > 1 else ""
        
        coords = (row["lon"], row["lat"])
        
        loc_map[raw_name] = coords
        loc_map[raw_name.lower()] = coords
        loc_map[raw_name.lower().replace(" ", "")] = coords
        
        norm_check = raw_name.lower().replace(" ", "").replace(".", "").replace("-", "")
        if "stpeter" in norm_check:
            if "apg" in tso_raw or "at" in tso_raw:
                loc_map["st.peter:at"] = coords
            elif "tennet" in tso_raw or "de" in tso_raw:
                loc_map["st.peter:de"] = coords
                
    return loc_map, df

loc_map, geo_df = load_geolocation_map(GEO_CSV_PATH)

# --- STEP 2: LOAD NETWORK CONNECTIVITY DATA ---
@st.cache_data
def load_network_data(lines_path, tielines_path):
    ldf = pl.read_parquet(lines_path) if os.path.exists(lines_path) else pl.DataFrame()
    tdf = pl.read_parquet(tielines_path) if os.path.exists(tielines_path) else pl.DataFrame()
    return ldf, tdf

lines_df, tielines_df = load_network_data(LINES_PATH, TIELINES_PATH)

# --- TERMINAL DIAGNOSTIC: PRINT ENTIRE RAW PARQUET ROWS ---
print("\n" + "="*80)
print("TERMINAL LOG: SCANNING FOR ST. PETER ROWS IN PARQUET DATABASE")
print("="*80)

for current_df, label_name in [(lines_df, "v8_Lines.parquet"), (tielines_df, "v8_Tielines.parquet")]:
    if not current_df.is_empty():
        # Identify correct substation column fields dynamically
        sub1_col = next((c for c in current_df.columns if "substation_1" in c.lower()), None)
        sub2_col = next((c for c in current_df.columns if "substation_2" in c.lower()), None)
        
        if sub1_col and sub2_col:
            matching_rows = current_df.filter(
                pl.col(sub1_col).cast(pl.Utf8).str.to_lowercase().str.contains("peter") |
                pl.col(sub2_col).cast(pl.Utf8).str.to_lowercase().str.contains("peter")
            )
            
            print(f"\n[Asset: {label_name}] -> Found {len(matching_rows)} matching records:")
            if len(matching_rows) > 0:
                # Force Polars terminal configuration to display all columns and wide text fields cleanly
                with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=60, tbl_width_chars=100):
                    print(matching_rows)
            else:
                print(" -> No rows matched query string threshold.")
        else:
            print(f"\n[Asset: {label_name}] -> Missing 'substation_1' or 'substation_2' schema columns.")
    else:
        print(f"\n[Asset: {label_name}] -> Target dataframe asset is empty.")
print("="*80 + "\n")


# --- STEP 3: SIDEBAR CONTROLS ---
st.sidebar.header("Data Controls")
target_date = st.sidebar.date_input("Select Deployment Date", pd.Timestamp("2026-07-13").date())
target_hour = st.sidebar.slider("Target Hour Interval (UTC)", 0, 23, 12)

mtu_timestamp = pd.Timestamp(
    year=target_date.year, month=target_date.month, day=target_date.day,
    hour=target_hour, minute=0, tz="Europe/Amsterdam"
)

if "jao_data" not in st.session_state:
    st.session_state.jao_data = pd.DataFrame()

fetch_requested = st.sidebar.button("Fetch JAO Data", type="primary")

# --- STEP 4: FETCH LIVE DATA VIA JAO-PY ---
@st.cache_data(ttl=3600)
def fetch_jao_congestion_domain(mtu):
    try:
        client = JaoPublicationToolPandasClient()
        return client.query_final_domain(mtu=mtu, presolved=True)
    except Exception as e:
        st.sidebar.error(f"API Error: {e}")
        return pd.DataFrame()

if fetch_requested:
    with st.sidebar.spinner("Querying JAO API..."):
        st.session_state.jao_data = fetch_jao_congestion_domain(mtu_timestamp)

jao_df = st.session_state.jao_data

congested_eics = set()
if not jao_df.empty:
    possible_columns = ["cneEic", "cneeic", "cne_eic", "eic"]
    active_col = next((col for col in possible_columns if col in jao_df.columns), None)
    if active_col:
        raw_values = jao_df[active_col].dropna().values
        congested_eics = {str(x).strip().lower() for x in raw_values}
        st.sidebar.success(f"Active CNE constraints loaded: {len(congested_eics)}")
    else:
        st.sidebar.info("JAO records loaded, but no matching EIC column was found.")
elif not fetch_requested:
    st.sidebar.info("Select parameters and click 'Fetch JAO Data'.")

all_plotted_eics = set()

# --- STEP 5: JOINT NETWORK ELEMENT COORDINATE RESOLVER ---
def resolve_line_coordinates(sub1, sub2, row_tso, is_tieline=False):
    if sub1 is None or sub2 is None:
        return None, None
        
    s1_raw, s2_raw = str(sub1).strip(), str(sub2).strip()
    s1_norm = s1_raw.lower().replace(" ", "").replace(".", "").replace("-", "")
    s2_norm = s2_raw.lower().replace(" ", "").replace(".", "").replace("-", "")
    
    def single_lookup(s_raw, s_norm, force_side=None):
        if "stpeter" in s_norm:
            if force_side == "at":
                return loc_map.get("st.peter:at")
            if force_side == "de":
                return loc_map.get("st.peter:de")
                
            tso_clean = str(row_tso).lower()
            if any(x in tso_clean for x in ["apg", "at", "austria"]):
                return loc_map.get("st.peter:at")
            if any(x in tso_clean for x in ["tennet", "de", "germany", "ttg"]):
                return loc_map.get("st.peter:de")
                
            return loc_map.get("st.peter:de") if " " in s_raw else loc_map.get("st.peter:at")
            
        if s_raw in loc_map: return loc_map[s_raw]
        if s_raw.lower() in loc_map: return loc_map[s_raw.lower()]
        if s_raw.lower().replace(" ", "") in loc_map: return loc_map[s_raw.lower().replace(" ", "")]
        return None

    if "stpeter" in s1_norm and "stpeter" in s2_norm:
        return loc_map.get("st.peter:de"), loc_map.get("st.peter:at")
        
    if is_tieline:
        if "stpeter" in s1_norm:
            side = "de" if " " in s1_raw else "at"
            return single_lookup(s1_raw, s1_norm, force_side=side), single_lookup(s2_raw, s2_norm)
        if "stpeter" in s2_norm:
            side = "de" if " " in s2_raw else "at"
            return single_lookup(s1_raw, s1_norm), single_lookup(s2_raw, s2_norm, force_side=side)

    return single_lookup(s1_raw, s1_norm), single_lookup(s2_raw, s2_norm)

def build_split_line_traces(df: pl.DataFrame, label_prefix: str, congestion_set: set, tracking_set: set, is_tieline: bool):
    lons_norm, lats_norm, hovers_norm = [], [], []
    lons_cong, lats_cong, hovers_cong = [], [], []
    
    if df.is_empty():
        return lons_norm, lats_norm, hovers_norm, lons_cong, lats_cong, hovers_cong

    for row in df.to_dicts():
        row_tso = str(row.get("tso", "")).strip().lower()
        
        coords_1, coords_2 = resolve_line_coordinates(
            row.get("substation_1"), row.get("substation_2"), row_tso, is_tieline=is_tieline
        )
        
        if coords_1 and coords_2:
            lon1, lat1 = coords_1
            lon2, lat2 = coords_2
            
            ne_name = row.get("ne_name") or row.get("c_ne_name") or row.get("cnename") or "Unknown"
            eic_code = row.get("eic_code") or row.get("c_ne_eic") or row.get("cneeic") or "Unknown"
            clean_eic = clean_str(eic_code)
            
            tracking_set.add(clean_eic)
            is_congested = clean_eic in congestion_set
            
            hover_text = (
                f"<b>{label_prefix}: {ne_name}</b><br>"
                f"EIC: {eic_code}<br>"
                f"TSO: {row.get('tso', 'Unknown')}<br>"
                f"Path: {row.get('substation_1')} ➔ {row.get('substation_2')}<br>"
                f"Status: {'CONGESTED' if is_congested else 'Normal'}"
            )
            
            if is_congested:
                lons_cong.extend([lon1, lon2, None])
                lats_cong.extend([lat1, lat2, None])
                hovers_cong.extend([hover_text, hover_text, None])
            else:
                lons_norm.extend([lon1, lon2, None])
                lats_norm.extend([lat1, lat2, None])
                hovers_norm.extend([hover_text, hover_text, None])
                
    return lons_norm, lats_norm, hovers_norm, lons_cong, lats_cong, hovers_cong

# --- STEP 6: GENERATE CANVAS MAP ELEMENTS ---
fig = go.Figure()

ln_no, lt_no, h_no, ln_co, lt_co, h_co = build_split_line_traces(lines_df, "Internal Line", congested_eics, all_plotted_eics, is_tieline=False)
if ln_no:
    fig.add_trace(go.Scattermapbox(
        lon=ln_no, lat=lt_no, mode="lines", line=dict(width=1.5, color="#34495e"),
        hoverinfo="text", text=h_no, name="Internal Lines (Normal)"
    ))
if ln_co:
    fig.add_trace(go.Scattermapbox(
        lon=ln_co, lat=lt_co, mode="lines", line=dict(width=3.5, color="#f39c12"),
        hoverinfo="text", text=h_co, name="Internal Lines (Congested)"
    ))

t_ln_no, t_lt_no, t_h_no, t_ln_co, t_lt_co, t_h_co = build_split_line_traces(tielines_df, "Cross-Border Tieline", congested_eics, all_plotted_eics, is_tieline=True)
if t_ln_no:
    fig.add_trace(go.Scattermapbox(
        lon=t_ln_no, lat=t_lt_no, mode="lines", line=dict(width=2.5, color="#2980b9"),
        hoverinfo="text", text=t_h_no, name="Cross-Border Tielines (Normal)"
    ))
if t_ln_co:
    fig.add_trace(go.Scattermapbox(
        lon=t_ln_co, lat=t_lt_co, mode="lines", line=dict(width=4.5, color="#e74c3c"),
        hoverinfo="text", text=t_h_co, name="Cross-Border Tielines (Congested)"
    ))

if not geo_df.is_empty():
    node_lons, node_lats, node_hovers = [], [], []
    for row in geo_df.to_dicts():
        node_lons.append(row["lon"])
        node_lats.append(row["lat"])
        keys = list(row.keys())
        node_name = row.get(keys[0]) or "Unknown"
        tso = row.get(keys[1]) or "Unknown"
        node_hovers.append(f"<b>Node: {node_name}</b><br>TSO: {tso}")

    fig.add_trace(go.Scattermapbox(
        lon=node_lons, lat=node_lats, mode="markers",
        marker=dict(size=7, color="#2c3e50", opacity=0.9),
        hoverinfo="text", text=node_hovers, name="Substations / Nodes"
    ))

fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=51.1657, lon=10.4515),
        zoom=4.5
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=1000,
    showlegend=True,
    legend=dict(
        yanchor="top", y=0.98,
        xanchor="left", x=0.01,
        bgcolor="rgba(255, 255, 255, 0.9)"
    )
)

st.plotly_chart(fig, use_container_width=True)

# --- STEP 7: PRINT METRICS ---
if not jao_df.empty:
    st.sidebar.markdown("---")
    matched_cne_count = len(congested_eics.intersection(all_plotted_eics))
    total_fetched_count = len(congested_eics)
    st.sidebar.metric(
        label="Plotted / Fetched CNEs",
        value=f"{matched_cne_count} / {total_fetched_count}"
    )