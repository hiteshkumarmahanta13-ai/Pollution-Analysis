# air_dashboard_multi_page.py
# Patched: Mapbox token handling + default CSVs from /mnt/data if present
import os
import math
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Air Pollution Dashboard (2-pages)", layout="wide")

# ---------------------------
# Mapbox token setup (robust)
# ---------------------------
def get_mapbox_token():
    # 1) Try Streamlit secrets (Streamlit Cloud)
    try:
        token = st.secrets.get("MAPBOX_API_KEY")  # returns None if not set
        if token:
            return token
    except Exception:
        pass
    # 2) Try environment variables (common names)
    token = os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_TOKEN") or os.getenv("MAPBOX_ACCESS_TOKEN")
    return token

MAPBOX_TOKEN = get_mapbox_token()
if MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN
    st.sidebar.success("Mapbox token loaded — using Mapbox basemaps.")
else:
    st.sidebar.warning("No Mapbox token found. App will use a token-free basemap (OpenStreetMap). Add MAPBOX_API_KEY in app secrets to enable Mapbox styles.")

# ---------------------------
# Model save/load helpers
# ---------------------------
MODEL_DIR = Path.cwd() / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "aqi_model.joblib"

def save_model_artifact(obj, model_path=MODEL_PATH):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, str(model_path))
    return str(model_path)

def try_load_model(candidate_paths=None):
    if candidate_paths is None:
        candidate_paths = [MODEL_PATH, Path.cwd() / "aqi_model.joblib"]
    for p in candidate_paths:
        p = Path(p)
        if p.exists():
            try:
                return joblib.load(str(p))
            except Exception as e:
                st.warning(f"Found model at {p} but loading failed: {e}")
    return None

# ---------------------------
# Load helpers (auto or upload)
# ---------------------------
@st.cache_data
def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_data
def find_default_files():
    """
    Prefer CSVs provided in /mnt/data (uploaded to session), otherwise look in repo working dir or data/ folder.
    """
    found = {}
    # 1) container/session upload location
    m_b = Path("/mnt/data/bbsr_cleaned.csv")
    m_d = Path("/mnt/data/delhi_cleaned.csv")
    if m_b.exists(): found["bbsr"] = str(m_b)
    if m_d.exists(): found["delhi"] = str(m_d)
    # 2) working directory + data folder fallback
    base_dirs = [Path.cwd(), Path.cwd() / "data"]
    for base in base_dirs:
        if not base.exists():
            continue
        b = base / "bbsr_cleaned.csv"
        d = base / "delhi_cleaned.csv"
        if b.exists() and "bbsr" not in found: found["bbsr"] = str(b)
        if d.exists() and "delhi" not in found: found["delhi"] = str(d)
    return found

preloaded = find_default_files()

# ---------------------------
# Sidebar: global controls, including shared Map City
# ---------------------------
st.sidebar.title("Controls")
page = st.sidebar.radio("Select page", ["Overview (Visuals)", "ML & Recommendations"])

st.sidebar.header("Data (upload or use defaults)")
use_preloaded = False
if "bbsr" in preloaded or "delhi" in preloaded:
    st.sidebar.success("Default CSVs detected in working folder or /mnt/data.")
    use_preloaded = st.sidebar.checkbox("Use default CSVs", value=True)

uploaded_bbsr = st.sidebar.file_uploader("Upload Bhubaneswar CSV", type=["csv"], key="u_bbsr")
uploaded_delhi = st.sidebar.file_uploader("Upload Delhi CSV", type=["csv"], key="u_delhi")

# Shared map city control (applies to maps on both pages)
map_city = st.sidebar.selectbox("Map city (applies to maps on both pages)", ["Bhubaneswar", "Delhi"])

def get_df(uploaded, keyname):
    """
    Returns dataframe either from upload, or from preloaded default file if selected.
    """
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except Exception as e:
            st.sidebar.error(f"Could not read uploaded {keyname} CSV: {e}")
            return None
    elif use_preloaded and keyname in preloaded:
        return load_csv_safe(preloaded[keyname])
    return None

bbsr_df = get_df(uploaded_bbsr, "bbsr")
delhi_df = get_df(uploaded_delhi, "delhi")

if bbsr_df is None and delhi_df is None:
    st.sidebar.warning("No datasets loaded yet. Upload CSVs or enable defaults in the sidebar.")

# ---------------------------
# Common parsing and helpers
# ---------------------------
def parse_datesafe(df):
    if df is None: return None
    df = df.copy()
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if date_cols:
        try:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
            if "year" not in df.columns: df["year"] = df[date_cols[0]].dt.year
            if "month" not in df.columns: df["month"] = df[date_cols[0]].dt.month
            if "day" not in df.columns: df["day"] = df[date_cols[0]].dt.day
        except Exception:
            pass
    return df

bbsr_df = parse_datesafe(bbsr_df)
delhi_df = parse_datesafe(delhi_df)

def find_pm25_cols(df):
    if df is None: return []
    return [c for c in df.columns if ("pm" in c.lower() and ("2.5" in c.lower() or "pm25" in c.lower()))]

def ensure_latlon(df, city_name):
    df = df.copy()
    default_coords = (20.2961, 85.8245) if "bhubaneswar" in city_name.lower() else (28.7041, 77.1025)
    lat_center, lon_center = default_coords
    if "lat" not in df.columns or "lon" not in df.columns:
        np.random.seed(42)
        df["lat"] = lat_center + np.random.uniform(-0.02, 0.02, len(df))
        df["lon"] = lon_center + np.random.uniform(-0.02, 0.02, len(df))
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce").fillna(lat_center)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce").fillna(lon_center)
    return df

# ---------------------------
# Robust heatmap builder (uses Mapbox if token present; otherwise OpenStreetMap)
# ---------------------------
def build_heatmap_deck(df_in, intensity_col="intensity", prefer_mapbox_style="mapbox://styles/mapbox/light-v9"):
    """
    Returns a pydeck.Deck object.
    If MAPBOX_TOKEN is available, uses Mapbox style; otherwise falls back to token-free OpenStreetMap.
    """
    df_local = df_in.copy()
    if intensity_col not in df_local.columns:
        df_local[intensity_col] = 0

    if MAPBOX_TOKEN:
        map_style = prefer_mapbox_style
    else:
        map_style = "open-street-map"

    layer = pdk.Layer(
        "HeatmapLayer",
        data=df_local,
        get_position='[lon, lat]',
        get_weight=intensity_col,
        radiusPixels=60,
        intensity=1,
        threshold=0.2,
        aggregation='SUM',
        colorRange=[
            [0, 255, 0, 0],
            [255, 255, 0, 80],
            [255, 140, 0, 150],
            [255, 0, 0, 200],
            [180, 0, 0, 255]
        ]
    )

    lat_mean = float(df_local["lat"].mean()) if "lat" in df_local.columns else (20.2961 if "bhubaneswar" in map_city.lower() else 28.7041)
    lon_mean = float(df_local["lon"].mean()) if "lon" in df_local.columns else (85.8245 if "bhubaneswar" in map_city.lower() else 77.1025)

    view = pdk.ViewState(latitude=lat_mean, longitude=lon_mean, zoom=10, pitch=40)
    deck = pdk.Deck(map_style=map_style, initial_view_state=view, layers=[layer],
                    tooltip={"text": "Lat: {lat}\nLon: {lon}\nIntensity: {intensity}"})
    return deck

def render_legend(min_val, max_val, width=340, title="Intensity"):
    """
    Render an HTML gradient legend with numeric ticks between min_val and max_val.
    """
    if min_val is None or max_val is None:
        st.markdown("No data for legend.")
        return
    if math.isclose(min_val, max_val):
        min_val, max_val = min_val - 0.1, max_val + 0.1
    tick_vals = [min_val + (max_val - min_val) * f for f in [0, 0.25, 0.5, 0.75, 1.0]]
    html = f"""
    <div style="display:flex; flex-direction:column; width:{width}px; font-family: sans-serif;">
      <div style="font-weight:600; margin-bottom:6px;">{title}</div>
      <div style="height:16px; background: linear-gradient(to right, rgba(0,255,0,0), #ffff00, #ff8c00, #ff0000, #b40000); border-radius:4px;"></div>
      <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:12px; color:#111;">
        <span>{tick_vals[0]:.1f}</span>
        <span>{tick_vals[1]:.1f}</span>
        <span>{tick_vals[2]:.1f}</span>
        <span>{tick_vals[3]:.1f}</span>
        <span>{tick_vals[4]:.1f}</span>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------
# Page: Overview (Visuals)
# ---------------------------
if page == "Overview (Visuals)":
    st.title("Overview — Visual Analysis")
    if bbsr_df is None or delhi_df is None:
        st.warning("Both Bhubaneswar and Delhi data are recommended for full comparisons. Upload or enable defaults.")

    st.sidebar.header("Overview Filters")
    view_mode = st.sidebar.selectbox("View mode", ["Single city analysis", "Compare cities"])

    if view_mode == "Single city analysis":
        # Single-city analysis controls (city for analysis is independent from map_city)
        analysis_city = st.sidebar.selectbox("Analysis city", ["Bhubaneswar", "Delhi"])

        # Safely pick analysis dataframe
        if analysis_city == "Bhubaneswar":
            if bbsr_df is None:
                st.error("Bhubaneswar data not available. Upload it in the sidebar or enable defaults.")
                st.stop()
            df = bbsr_df.copy()
        else:
            if delhi_df is None:
                st.error("Delhi data not available. Upload it in the sidebar or enable defaults.")
                st.stop()
            df = delhi_df.copy()

        # pollutants auto-detect
        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        exclude = set(date_cols + ["year","month","day","city","AQI","aqi"])
        pollutants = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        pm25 = find_pm25_cols(df)
        for p in pm25:
            if p not in pollutants: pollutants.append(p)
        selected_pollutants = st.multiselect("Select pollutants to visualize", pollutants, default=pollutants[:3])

        # KPIs and composition
        st.subheader(f"{analysis_city} — Key Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Dataset sample")
            st.dataframe(df.head())
        with col2:
            if selected_pollutants:
                nums = df[selected_pollutants].apply(pd.to_numeric, errors="coerce")
                st.metric("Max", f"{nums.max().max():.2f}")
                st.metric("Avg", f"{nums.mean().mean():.2f}")
        with col3:
            if selected_pollutants:
                avg = df[selected_pollutants].apply(pd.to_numeric, errors="coerce").mean()
                fig = px.pie(values=avg.values, names=avg.index, title="Pollutant Composition")
                st.plotly_chart(fig, use_container_width=True)

        # Trends
        st.subheader("Trends")
        if "month" in df.columns and selected_pollutants:
            month_map = {i: m for i, m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}
            df["month_norm"] = df["month"].map(month_map).fillna(df["month"])
            monthwise = df[selected_pollutants].apply(pd.to_numeric, errors="coerce").groupby(df["month_norm"]).mean()
            if not monthwise.empty:
                fig = px.line(monthwise, x=monthwise.index, y=selected_pollutants, markers=True, title="Monthly Trends")
                st.plotly_chart(fig, use_container_width=True)
        elif selected_pollutants:
            date_col_local = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if date_col_local:
                fig = px.line(df, x=date_col_local[0], y=selected_pollutants, title="Time Series")
                st.plotly_chart(fig, use_container_width=True)

        # Distribution & relationships
        st.subheader("Distribution & Relationships")
        if selected_pollutants:
            col1, col2 = st.columns(2)
            with col1:
                st.write("Bar chart (mean by pollutant)")
                means = df[selected_pollutants].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=False)
                fig = px.bar(x=means.index, y=means.values, labels={'x':'Pollutant','y':'Mean'}, title="Mean pollutant levels")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.write("Boxplot")
                fig, ax = plt.subplots(figsize=(6,4))
                sns.boxplot(data=df[selected_pollutants].apply(pd.to_numeric, errors="coerce"), ax=ax)
                st.pyplot(fig)

            if len(selected_pollutants) >= 2:
                st.write("Scatter (choose two pollutants)")
                p1 = st.selectbox("X pollutant", selected_pollutants, index=0)
                p2 = st.selectbox("Y pollutant", selected_pollutants, index=1)
                # safe scatter + trendline fallback if statsmodels missing
                x = pd.to_numeric(df[p1], errors="coerce")
                y = pd.to_numeric(df[p2], errors="coerce")
                valid = x.notna() & y.notna()
                if valid.sum() < 2:
                    st.info("Not enough valid points for scatter/trendline.")
                else:
                    import plotly.graph_objects as go
                    scatter = go.Scatter(x=x[valid], y=y[valid], mode="markers", name="Data",
                                         marker=dict(opacity=0.6))
                    coeffs = np.polyfit(x[valid].to_numpy(), y[valid].to_numpy(), deg=1)
                    poly = np.poly1d(coeffs)
                    xs = np.linspace(x[valid].min(), x[valid].max(), 100)
                    ys = poly(xs)
                    trend = go.Scatter(x=xs, y=ys, mode="lines", name="Linear trend", line=dict(width=2))
                    layout = go.Layout(title=f"{p1} vs {p2}", xaxis_title=p1, yaxis_title=p2)
                    fig = go.Figure(data=[scatter, trend], layout=layout)
                    st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # -----------------------
        # Heatmap + Month Slider + Play (map uses map_city selection)
        # -----------------------
        st.subheader("🌡️ Pollution Heat Map (monthly)")

        # Use the shared map_city selection for the map data source
        if map_city == "Bhubaneswar":
            if bbsr_df is None:
                st.error("Map city Bhubaneswar data not available. Upload it or enable defaults.")
                st.stop()
            df_map_source = bbsr_df.copy()
        else:
            if delhi_df is None:
                st.error("Map city Delhi data not available. Upload it or enable defaults.")
                st.stop()
            df_map_source = delhi_df.copy()

        df_map = ensure_latlon(df_map_source, map_city)
        pollutants_present = [p for p in selected_pollutants if p in df_map.columns]
        if pollutants_present:
            df_map[pollutants_present] = df_map[pollutants_present].apply(pd.to_numeric, errors="coerce").fillna(0)
            df_map["intensity"] = df_map[pollutants_present].mean(axis=1)
            intensity_source = f"Average of {', '.join(pollutants_present)}"
        else:
            pm25c = find_pm25_cols(df_map)
            if pm25c:
                df_map["intensity"] = pd.to_numeric(df_map[pm25c[0]], errors="coerce").fillna(0)
                intensity_source = pm25c[0]
            else:
                df_map["intensity"] = 0
                intensity_source = "N/A"

        if "month" in df_map.columns:
            month_label_map = {0: "All", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            selected_month = st.slider("Select month (0 = All)", min_value=0, max_value=12, value=0, step=1)
            deck_placeholder = st.empty()
            legend_placeholder = st.empty()

            def render_month_map(month_val):
                if month_val == 0:
                    df_plot = df_map.copy()
                else:
                    df_plot = df_map[df_map["month"] == month_val]
                    if df_plot.empty:
                        deck_placeholder.info(f"No data for {month_label_map.get(month_val, month_val)} in {map_city}")
                        legend_placeholder.empty()
                        return
                deck_placeholder.pydeck_chart(build_heatmap_deck(df_plot))
                vmin = float(df_plot["intensity"].min())
                vmax = float(df_plot["intensity"].max())
                with legend_placeholder:
                    render_legend(vmin, vmax, title=f"Intensity ({map_city} - {month_label_map.get(month_val)})")

            render_month_map(selected_month)

            if st.button("Play animation (Jan → Dec)"):
                for m in range(1, 13):
                    render_month_map(m)
                    st.markdown(f"**Showing month:** {month_label_map[m]} (Map city: {map_city})")
                    time.sleep(0.6)
                render_month_map(selected_month)
        else:
            st.info("No 'month' column detected for map city; showing full heatmap.")
            st.pydeck_chart(build_heatmap_deck(df_map))
            vmin = float(df_map["intensity"].min()) if "intensity" in df_map.columns else 0.0
            vmax = float(df_map["intensity"].max()) if "intensity" in df_map.columns else 0.0
            render_legend(vmin, vmax, title=f"Intensity ({map_city} - All months)")

        st.markdown(f"**Heatmap intensity source (map city = {map_city}):** {intensity_source}")

    else:
        # Compare cities view
        st.sidebar.header("Comparison settings")
        if bbsr_df is None or delhi_df is None:
            st.error("Both datasets required for comparison. Upload or use defaults.")
            st.stop()
        b_poll = [c for c in bbsr_df.select_dtypes(include=[np.number]).columns]
        d_poll = [c for c in delhi_df.select_dtypes(include=[np.number]).columns]
        common = sorted(list(set(b_poll).intersection(d_poll)))
        if not common:
            st.info("No common numeric pollutant columns to compare.")
        else:
            selected = st.multiselect("Select pollutants to compare (common columns)", common, default=common[:3])
            if selected:
                b_means = bbsr_df[selected].apply(pd.to_numeric, errors="coerce").mean()
                d_means = delhi_df[selected].apply(pd.to_numeric, errors="coerce").mean()
                comp_df = pd.DataFrame({"Bhubaneswar": b_means, "Delhi": d_means}).reset_index().rename(columns={'index':'Pollutant'})
                fig = px.bar(comp_df, x="Pollutant", y=["Bhubaneswar","Delhi"], barmode="group", title="City comparison (mean values)")
                st.plotly_chart(fig, use_container_width=True)
                comp_df["PctDiff(Delhi_vs_Bbsr)"] = ((comp_df["Delhi"] - comp_df["Bhubaneswar"]) / (comp_df["Bhubaneswar"].replace(0, np.nan))).fillna(0)*100
                st.dataframe(comp_df.style.format({"Bhubaneswar":"{:.2f}","Delhi":"{:.2f}","PctDiff(Delhi_vs_Bbsr)":"{:.1f}%"}))

                # Side-by-side heatmaps with own legends
                st.subheader("🌡️ Side-by-side Monthly Heat Maps (Comparison)")
                b_df_map = ensure_latlon(bbsr_df, "Bhubaneswar")
                d_df_map = ensure_latlon(delhi_df, "Delhi")
                if selected:
                    b_df_map[selected] = b_df_map[selected].apply(pd.to_numeric, errors="coerce").fillna(0)
                    d_df_map[selected] = d_df_map[selected].apply(pd.to_numeric, errors="coerce").fillna(0)
                    b_df_map["intensity"] = b_df_map[selected].mean(axis=1)
                    d_df_map["intensity"] = d_df_map[selected].mean(axis=1)
                else:
                    b_df_map["intensity"] = 0
                    d_df_map["intensity"] = 0

                has_month_b = "month" in b_df_map.columns
                has_month_d = "month" in d_df_map.columns
                if has_month_b and has_month_d:
                    month_label_map = {0: "All", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                    selected_month = st.slider("Select month (0 = All)", min_value=0, max_value=12, value=0, step=1, key="comp_month")
                    deck_col1 = st.empty()
                    legend_col1 = st.empty()
                    deck_col2 = st.empty()
                    legend_col2 = st.empty()

                    def render_both(month_val):
                        if month_val == 0:
                            b_plot = b_df_map.copy()
                            d_plot = d_df_map.copy()
                        else:
                            b_plot = b_df_map[b_df_map["month"] == month_val]
                            d_plot = d_df_map[d_df_map["month"] == month_val]
                        # Bhubaneswar
                        if b_plot.empty:
                            deck_col1.info(f"No Bhubaneswar data for {month_label_map.get(month_val, month_val)}")
                            legend_col1.empty()
                        else:
                            deck_col1.pydeck_chart(build_heatmap_deck(b_plot, map_style="mapbox://styles/mapbox/light-v9"))
                            vmin_b = float(b_plot["intensity"].min())
                            vmax_b = float(b_plot["intensity"].max())
                            with legend_col1:
                                render_legend(vmin_b, vmax_b, title=f"Intensity (Bhubaneswar - {month_label_map.get(month_val)})")
                        # Delhi
                        if d_plot.empty:
                            deck_col2.info(f"No Delhi data for {month_label_map.get(month_val, month_val)}")
                            legend_col2.empty()
                        else:
                            deck_col2.pydeck_chart(build_heatmap_deck(d_plot, map_style="mapbox://styles/mapbox/light-v9"))
                            vmin_d = float(d_plot["intensity"].min())
                            vmax_d = float(d_plot["intensity"].max())
                            with legend_col2:
                                render_legend(vmin_d, vmax_d, title=f"Intensity (Delhi - {month_label_map.get(month_val)})")

                    render_both(selected_month)

                    if st.button("Play animation (Jan → Dec)", key="comp_play"):
                        for m in range(1,13):
                            render_both(m)
                            st.markdown(f"**Showing month:** {month_label_map[m]}")
                            time.sleep(0.6)
                        render_both(selected_month)
                else:
                    st.info("Month column missing in one or both datasets; showing full heatmaps")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Bhubaneswar Heatmap**")
                        st.pydeck_chart(build_heatmap_deck(b_df_map, map_style="mapbox://styles/mapbox/light-v9"))
                        render_legend(float(b_df_map["intensity"].min()), float(b_df_map["intensity"].max()), title="Intensity (Bhubaneswar - All months)")
                    with col2:
                        st.markdown("**Delhi Heatmap**")
                        st.pydeck_chart(build_heatmap_deck(d_df_map, map_style="mapbox://styles/mapbox/light-v9"))
                        render_legend(float(d_df_map["intensity"].min()), float(d_df_map["intensity"].max()), title="Intensity (Delhi - All months)")

# ---------------------------
# Page: ML & Recommendations (city-specific map uses shared map_city)
# ---------------------------
else:
    st.title("ML & Tree Recommendations")
    if bbsr_df is None and delhi_df is None:
        st.error("At least one dataset required. Upload data in sidebar.")
        st.stop()

    # Combine datasets for ML & maps
    frames = []
    if bbsr_df is not None: frames.append(bbsr_df.assign(city="Bhubaneswar"))
    if delhi_df is not None: frames.append(delhi_df.assign(city="Delhi"))
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    st.subheader("Data preview (combined)")
    st.dataframe(combined.head())

    # safely detect date_col for combined
    date_cols_combined = [c for c in combined.columns if "date" in c.lower() or "time" in c.lower()]
    date_col = date_cols_combined[0] if date_cols_combined else None

    # select target & features
    possible_targets = [c for c in combined.columns if c.lower() == "aqi"]
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in combined data to train model.")
    target_col = st.selectbox("Select target column (AQI recommended)", options=(possible_targets + numeric_cols) if (possible_targets + numeric_cols) else numeric_cols)
    feature_cols = st.multiselect("Feature columns (numeric recommended)", options=[c for c in numeric_cols if c != target_col], default=[c for c in numeric_cols if c != target_col][:6])

    # Train model
    if st.button("Train AQI model"):
        if not feature_cols:
            st.error("Choose at least one feature.")
        else:
            X = combined[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            y = combined[target_col].apply(pd.to_numeric, errors="coerce").fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            with st.spinner("Training..."):
                model.fit(X_train, y_train)
            preds = model.predict(X_test)
            st.success("Training finished.")
            st.metric("R²", f"{r2_score(y_test, preds):.3f}")
            st.metric("RMSE", f"{math.sqrt(mean_squared_error(y_test, preds)):.3f}")
            fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            st.subheader("Feature importances")
            st.bar_chart(fi.set_index("feature")["importance"])
            save_model_artifact({"model": model, "features": feature_cols, "target": target_col})
            st.success("Model saved to ./models/aqi_model.joblib")

    # Prediction block (batch)
    st.subheader("Batch prediction (upload CSV with feature columns)")
    pred_file = st.file_uploader("Upload CSV to predict", type=["csv"], key="pred_batch")
    if pred_file is not None:
        pred_df = pd.read_csv(pred_file)
        saved = try_load_model()
        if saved is None:
            st.error("No saved model found. Train and save a model first.")
        else:
            model = saved["model"]
            features = saved["features"]
            missing = [c for c in features if c not in pred_df.columns]
            if missing:
                st.warning(f"Uploaded file missing features: {missing}. Missing features will be filled with 0.")
            X_new = pd.DataFrame({c: pd.to_numeric(pred_df.get(c, 0), errors="coerce").fillna(0) for c in features})
            preds = model.predict(X_new)
            pred_df["Predicted_" + saved.get("target", "target")] = preds
            st.dataframe(pred_df.head())
            st.download_button("Download predictions CSV", pred_df.to_csv(index=False).encode(), "predictions.csv")

    # Tree recommendation logic
    st.markdown("---")
    st.subheader("Tree Recommendations & Zones")

    # Use same map_city selector for the ML page map as well (shared control)
    city_for_map = map_city  # using shared control

    env_type = st.selectbox("Environment Type", ["Urban / Roadside","Industrial","Residential","Rural / Agricultural"])

    pollutant_tree_map = {
        "PM2.5": ["Neem","Peepal","Banyan"],
        "PM10": ["Ashoka","Gulmohar","Cassia Fistula"],
        "SO2": ["Arjuna","Amaltas","Bael"],
        "NO2": ["Neem","Peepal","Mango"],
        "CO": ["Banyan","Jamun"],
        "O3": ["Neem","Peepal","Mango"]
    }
    env_tree_map = {
        "Urban / Roadside":["Neem","Ashoka","Cassia","Gulmohar"],
        "Industrial":["Arjuna","Amaltas","Banyan"],
        "Residential":["Mango","Jamun","Bael"],
        "Rural / Agricultural":["Peepal","Neem","Tamarind"]
    }

    def recommend_trees(pollutants, env):
        trees = set()
        for p in pollutants:
            for key, lst in pollutant_tree_map.items():
                if key.lower() in p.lower():
                    trees.update(lst)
        trees.update(env_tree_map.get(env, []))
        return sorted(trees)

    # Determine selected pollutants to base recommendations on (from combined or user pick)
    all_pollutants = [c for c in combined.columns if c not in [date_col, "year","month","day","city","AQI","aqi"] and pd.api.types.is_numeric_dtype(combined[c])]
    all_pollutants = [c for c in all_pollutants if c is not None]
    selected_pollutants_for_rec = st.multiselect("Select pollutants that concern you (for recommendations)", all_pollutants, default=[c for c in all_pollutants if "pm" in c.lower()][:2])

    if selected_pollutants_for_rec:
        rec_trees = recommend_trees(selected_pollutants_for_rec, env_type)
        st.success(f"Recommended trees: {', '.join(rec_trees)}")
    else:
        rec_trees = []
        st.info("Select pollutants to get tree recommendations.")

    # Interactive tree zone map (city-specific using shared map_city)
    st.subheader("Tree Zone Map (interactive markers & benefits)")

    # pick the dataframe for the selected map city safely
    if city_for_map == "Bhubaneswar":
        if bbsr_df is None:
            st.error("Bhubaneswar data not available for map. Upload it or enable defaults.")
            st.stop()
        df_city = bbsr_df.copy()
    else:
        if delhi_df is None:
            st.error("Delhi data not available for map. Upload it or enable defaults.")
            st.stop()
        df_city = delhi_df.copy()

    df_city = ensure_latlon(df_city, city_for_map)

    # compute intensity using only pollutants available in this city
    if selected_pollutants_for_rec:
        pollutants_in_city = [p for p in selected_pollutants_for_rec if p in df_city.columns]
        if pollutants_in_city:
            df_city[pollutants_in_city] = df_city[pollutants_in_city].apply(pd.to_numeric, errors="coerce").fillna(0)
            df_city["intensity"] = df_city[pollutants_in_city].mean(axis=1)
            intensity_source = f"Average of {', '.join(pollutants_in_city)} (city-specific)"
        else:
            pm25_cols_local = [c for c in df_city.columns if "pm" in c.lower() and ("2.5" in c.lower() or "pm25" in c.lower())]
            if pm25_cols_local:
                df_city["intensity"] = pd.to_numeric(df_city[pm25_cols_local[0]], errors="coerce").fillna(0)
                intensity_source = pm25_cols_local[0]
            else:
                df_city["intensity"] = 0
                intensity_source = "N/A"
    else:
        pm25_cols_local = [c for c in df_city.columns if "pm" in c.lower() and ("2.5" in c.lower() or "pm25" in c.lower())]
        if pm25_cols_local:
            df_city["intensity"] = pd.to_numeric(df_city[pm25_cols_local[0]], errors="coerce").fillna(0)
            intensity_source = pm25_cols_local[0]
        else:
            df_city["intensity"] = 0
            intensity_source = "N/A"

    if "intensity" in df_city.columns and df_city["intensity"].sum() > 0:
        hotspots = df_city.nlargest(10, "intensity")[["lat", "lon", "intensity"]].reset_index(drop=True)
    else:
        hotspots = pd.DataFrame(columns=["lat", "lon", "intensity"])

    tree_benefits = {
        "Neem":"Absorbs PM2.5, NO2, SO2, and CO2.",
        "Peepal":"Releases oxygen at night; filters PM2.5.",
        "Banyan":"Dust absorber; traps particulates.",
        "Ashoka":"Good roadside dust absorber.",
        "Gulmohar":"Traps dust; cooling canopy.",
        "Cassia Fistula":"Absorbs SO2; ornamental.",
        "Arjuna":"Tolerates gaseous pollutants.",
        "Amaltas":"Cleans SO2-rich air.",
        "Bael":"Absorbs toxic gases.",
        "Mango":"Filters NO2, CO.",
        "Jamun":"CO absorption; urban tolerant.",
        "Tamarind":"Dust control, soil benefits."
    }

    markers = []
    if len(hotspots) > 0 and rec_trees:
        for i, row in hotspots.iterrows():
            tree = rec_trees[i % len(rec_trees)]
            markers.append({
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "tree": tree,
                "benefit": tree_benefits.get(tree, "Improves air"),
                "intensity": round(float(row["intensity"]), 2)
            })

        marker_layer = pdk.Layer(
            "ScatterplotLayer",
            data=markers,
            get_position='[lon, lat]',
            get_color='[0, 150, 50, 200]',
            get_radius=120,
            pickable=True
        )
        text_layer = pdk.Layer(
            "TextLayer",
            data=markers,
            get_position='[lon, lat]',
            get_text='tree',
            get_size=14,
            get_color='[255,255,255]',
            get_alignment_baseline="'bottom'"
        )
        view = pdk.ViewState(latitude=float(df_city["lat"].mean()), longitude=float(df_city["lon"].mean()), zoom=10)
        tooltip = {"text":"🌳 Tree: {tree}\n💚 Benefit: {benefit}\n🔥 Intensity: {intensity}"}
        deck = pdk.Deck(map_style="mapbox://styles/mapbox/outdoors-v12" if MAPBOX_TOKEN else "open-street-map", initial_view_state=view, layers=[marker_layer, text_layer], tooltip=tooltip)
        st.pydeck_chart(deck)
        # legend for hotspots (intensity min/max)
        if "intensity" in df_city.columns and not df_city["intensity"].isna().all():
            render_legend(float(df_city["intensity"].min()), float(df_city["intensity"].max()), title=f"Intensity (hotspots - {city_for_map})")
        st.markdown(f"Markers show suggested tree species & benefits for top pollution hotspots in **{city_for_map}** (intensity source: {intensity_source}).")
    else:
        st.info(f"No hotspots or recommended trees available for {city_for_map}. Try selecting different pollutants or check dataset completeness.")

# ---------------------------
# End
# ---------------------------
st.markdown("---")
st.caption("Two-page dashboard with robust Mapbox handling and defaults from /mnt/data when provided. Add MAPBOX_API_KEY to Streamlit app secrets to enable Mapbox basemaps; otherwise the app uses OpenStreetMap.")
