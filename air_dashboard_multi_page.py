# air_dashboard_multi_page.py
# Updated: persist ML training outputs in st.session_state; remove main-page uploads (sidebar-only).
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

# Attempt to import folium + streamlit_folium (preferred for MapTiler)
try:
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

st.set_page_config(page_title="Air Pollution Dashboard (MapTiler maps)", layout="wide")

# ===== MapTiler token =====
def get_maptiler_key():
    try:
        token = st.secrets.get("MAPTILER_KEY")
        if token:
            return token
    except Exception:
        pass
    return os.getenv("MAPTILER_KEY") or os.getenv("MAPTILER_TOKEN")

MAPTILER_KEY = get_maptiler_key()

if MAPTILER_KEY and FOLIUM_AVAILABLE:
    st.sidebar.success("MapTiler key loaded and Folium available (MapTiler backend ready).")
elif MAPTILER_KEY and not FOLIUM_AVAILABLE:
    st.sidebar.warning("MapTiler key found but folium/streamlit-folium not installed. Install requirements to enable MapTiler.")
else:
    st.sidebar.info("MapTiler key not found. Folium will fallback to OpenStreetMap tiles if used.")

USE_FOLIUM = FOLIUM_AVAILABLE  # prefer folium maps when available

# ===== Model helpers =====
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

# ===== Data loading helpers =====
@st.cache_data
def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_data
def find_default_files():
    found = {}
    m_b = Path("/mnt/data/bbsr_cleaned.csv")
    m_d = Path("/mnt/data/delhi_cleaned.csv")
    if m_b.exists(): found["bbsr"] = str(m_b)
    if m_d.exists(): found["delhi"] = str(m_d)
    base_dirs = [Path.cwd(), Path.cwd() / "data"]
    for base in base_dirs:
        if not base.exists(): continue
        b = base / "bbsr_cleaned.csv"
        d = base / "delhi_cleaned.csv"
        if b.exists() and "bbsr" not in found: found["bbsr"] = str(b)
        if d.exists() and "delhi" not in found: found["delhi"] = str(d)
    return found

preloaded = find_default_files()

# Sidebar controls & file uploads (sidebar-only uploads)
st.sidebar.title("Controls")
page = st.sidebar.radio("Select page", ["Overview (Visuals)", "ML & Recommendations"])

st.sidebar.header("Data (upload or use defaults)")
use_preloaded = False
if "bbsr" in preloaded or "delhi" in preloaded:
    st.sidebar.success("Default CSVs detected in working folder or /mnt/data.")
    use_preloaded = st.sidebar.checkbox("Use default CSVs", value=True)

# Sidebar uploaders (the only upload UI)
uploaded_bbsr = st.sidebar.file_uploader("Upload Bhubaneswar CSV", type=["csv"], key="u_bbsr")
uploaded_delhi = st.sidebar.file_uploader("Upload Delhi CSV", type=["csv"], key="u_delhi")

# Shared map city control
map_city = st.sidebar.selectbox("Map city (applies to maps on both pages)", ["Bhubaneswar", "Delhi"])

def get_df(uploaded, keyname):
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

# ===== Utilities =====
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

# ===== Map builders =====
def build_folium_heatmap(df_in, intensity_col="intensity", maptiler_key=None):
    if not FOLIUM_AVAILABLE:
        raise RuntimeError("folium or streamlit_folium is not installed.")
    df_local = df_in.copy()
    if intensity_col not in df_local.columns:
        df_local[intensity_col] = 0
    lat_mean = float(df_local["lat"].mean()) if "lat" in df_local.columns else (20.2961 if "bhubaneswar" in map_city.lower() else 28.7041)
    lon_mean = float(df_local["lon"].mean()) if "lon" in df_local.columns else (85.8245 if "bhubaneswar" in map_city.lower() else 77.1025)
    if maptiler_key:
        tiles_url = f"https://api.maptiler.com/maps/streets/256/{{z}}/{{x}}/{{y}}@2x.png?key={maptiler_key}"
        m = folium.Map(location=[lat_mean, lon_mean], zoom_start=11, tiles=None)
        folium.TileLayer(tiles=tiles_url, attr='MapTiler', name='MapTiler Streets', overlay=False, control=True).add_to(m)
    else:
        m = folium.Map(location=[lat_mean, lon_mean], zoom_start=11, tiles="OpenStreetMap")
    heat_data = df_local[['lat','lon', intensity_col]].dropna().values.tolist()
    if len(heat_data) > 0:
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
    if "intensity" in df_local.columns and not df_local["intensity"].isna().all():
        hotspots = df_local.nlargest(10, "intensity")[["lat", "lon", "intensity"]].reset_index(drop=True)
        for _, row in hotspots.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color='crimson',
                fill=True,
                fill_opacity=0.9,
                popup=folium.Popup(f"Intensity: {row['intensity']:.2f}", parse_html=True)
            ).add_to(m)
    return m

def build_pydeck_heatmap(df_in, intensity_col="intensity"):
    df_local = df_in.copy()
    if intensity_col not in df_local.columns:
        df_local[intensity_col] = 0
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
            [0,255,0,0],
            [255,255,0,80],
            [255,140,0,150],
            [255,0,0,200],
            [180,0,0,255]
        ]
    )
    lat_mean = float(df_local["lat"].mean()) if "lat" in df_local.columns else (20.2961 if "bhubaneswar" in map_city.lower() else 28.7041)
    lon_mean = float(df_local["lon"].mean()) if "lon" in df_local.columns else (85.8245 if "bhubaneswar" in map_city.lower() else 77.1025)
    view = pdk.ViewState(latitude=lat_mean, longitude=lon_mean, zoom=10, pitch=40)
    deck = pdk.Deck(map_style="open-street-map", initial_view_state=view, layers=[layer],
                    tooltip={"text":"Lat: {lat}\nLon: {lon}\nIntensity: {intensity}"})
    return deck

def render_legend(min_val, max_val, width=340, title="Intensity"):
    if min_val is None or max_val is None:
        st.markdown("No data for legend.")
        return
    if math.isclose(min_val, max_val):
        min_val, max_val = min_val - 0.1, max_val + 0.1
    tick_vals = [min_val + (max_val - min_val) * f for f in [0,0.25,0.5,0.75,1.0]]
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

# Initialize persistent storage for training if not present
if "trained" not in st.session_state:
    st.session_state.trained = False
if "model_artifact" not in st.session_state:
    st.session_state.model_artifact = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "feature_importances" not in st.session_state:
    st.session_state.feature_importances = None

# ===== Page: Overview =====
if page == "Overview (Visuals)":
    st.title("Overview — Visual Analysis")
    if bbsr_df is None or delhi_df is None:
        st.warning("Both Bhubaneswar and Delhi data are recommended for full comparisons. Upload or enable defaults.")

    st.sidebar.header("Overview Filters")
    view_mode = st.sidebar.selectbox("View mode", ["Single city analysis", "Compare cities"])

    if view_mode == "Single city analysis":
        analysis_city = st.sidebar.selectbox("Analysis city", ["Bhubaneswar", "Delhi"])

        if analysis_city == "Bhubaneswar":
            if bbsr_df is None:
                st.error("Bhubaneswar data not available. Upload or enable defaults.")
                st.stop()
            df = bbsr_df.copy()
        else:
            if delhi_df is None:
                st.error("Delhi data not available. Upload or enable defaults.")
                st.stop()
            df = delhi_df.copy()

        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        exclude = set(date_cols + ["year","month","day","city","AQI","aqi"])
        pollutants = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        pm25 = find_pm25_cols(df)
        for p in pm25:
            if p not in pollutants: pollutants.append(p)
        selected_pollutants = st.multiselect("Select pollutants to visualize", pollutants, default=pollutants[:3])

        # KPI & composition
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
            month_map = {i:m for i,m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}
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
            c1, c2 = st.columns(2)
            with c1:
                st.write("Bar chart (mean by pollutant)")
                means = df[selected_pollutants].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=False)
                fig = px.bar(x=means.index, y=means.values, labels={'x':'Pollutant','y':'Mean'}, title="Mean pollutant levels")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.boxplot(data=df[selected_pollutants].apply(pd.to_numeric, errors="coerce"), ax=ax)
                st.pyplot(fig)

            if len(selected_pollutants) >= 2:
                st.write("Scatter (choose two pollutants)")
                p1 = st.selectbox("X pollutant", selected_pollutants, index=0)
                p2 = st.selectbox("Y pollutant", selected_pollutants, index=1)
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

        # Map (heatmap + month slider)
        st.subheader("🌡️ Pollution Heat Map (monthly)")

        if map_city == "Bhubaneswar":
            if bbsr_df is None:
                st.error("Map city Bhubaneswar data not available.")
                st.stop()
            df_map_source = bbsr_df.copy()
        else:
            if delhi_df is None:
                st.error("Map city Delhi data not available.")
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

        if USE_FOLIUM:
            month_label_map = {0: "All", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            selected_month = st.slider("Select month (0 = All)", min_value=0, max_value=12, value=0, step=1)
            if selected_month == 0:
                df_plot = df_map.copy()
            else:
                df_plot = df_map[df_map["month"] == selected_month]
                if df_plot.empty:
                    st.info(f"No data for {month_label_map[selected_month]} in {map_city}")
            m = build_folium_heatmap(df_plot, intensity_col="intensity", maptiler_key=MAPTILER_KEY if MAPTILER_KEY else None)
            st_data = st_folium(m, width=900, height=500)
            if not df_plot["intensity"].isna().all():
                render_legend(float(df_plot["intensity"].min()), float(df_plot["intensity"].max()), title=f"Intensity ({map_city} - {month_label_map[selected_month]})")
        else:
            if "month" in df_map.columns:
                selected_month = st.slider("Select month (0 = All)", min_value=0, max_value=12, value=0, step=1)
                if selected_month == 0:
                    df_plot = df_map.copy()
                else:
                    df_plot = df_map[df_map["month"] == selected_month]
                st.pydeck_chart(build_pydeck_heatmap(df_plot))
                render_legend(float(df_plot["intensity"].min()), float(df_plot["intensity"].max()), title=f"Intensity ({map_city})")
            else:
                st.pydeck_chart(build_pydeck_heatmap(df_map))
                render_legend(float(df_map["intensity"].min()), float(df_map["intensity"].max()), title=f"Intensity ({map_city})")

        st.markdown(f"**Heatmap intensity source (map city = {map_city}):** {intensity_source}")

    else:
        # Comparison view
        st.sidebar.header("Comparison settings")
        if bbsr_df is None or delhi_df is None:
            st.error("Both datasets required for comparison.")
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

                # Prepare heatmaps
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

                month_label_map = {0:"All",1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
                selected_month = st.slider("Select month (0 = All)", min_value=0, max_value=12, value=0, step=1, key="comp_month")

                if selected_month != 0:
                    if "month" in b_df_map.columns:
                        b_plot = b_df_map[b_df_map["month"] == selected_month]
                    else:
                        b_plot = b_df_map.copy()
                    if "month" in d_df_map.columns:
                        d_plot = d_df_map[d_df_map["month"] == selected_month]
                    else:
                        d_plot = d_df_map.copy()
                else:
                    b_plot = b_df_map.copy()
                    d_plot = d_df_map.copy()

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Bhubaneswar Heatmap**")
                    if USE_FOLIUM:
                        m1 = build_folium_heatmap(b_plot, intensity_col="intensity", maptiler_key=MAPTILER_KEY if MAPTILER_KEY else None)
                        st_folium(m1, width=500, height=400)
                    else:
                        st.pydeck_chart(build_pydeck_heatmap(b_plot))
                    if not b_plot["intensity"].isna().all():
                        render_legend(float(b_plot["intensity"].min()), float(b_plot["intensity"].max()), title=f"Intensity (Bhubaneswar - {month_label_map[selected_month]})")
                with col2:
                    st.markdown("**Delhi Heatmap**")
                    if USE_FOLIUM:
                        m2 = build_folium_heatmap(d_plot, intensity_col="intensity", maptiler_key=MAPTILER_KEY if MAPTILER_KEY else None)
                        st_folium(m2, width=500, height=400)
                    else:
                        st.pydeck_chart(build_pydeck_heatmap(d_plot))
                    if not d_plot["intensity"].isna().all():
                        render_legend(float(d_plot["intensity"].min()), float(d_plot["intensity"].max()), title=f"Intensity (Delhi - {month_label_map[selected_month]})")

# ===== Page: ML & Recommendations =====
else:
    st.title("ML & Tree Recommendations")
    if bbsr_df is None and delhi_df is None:
        st.error("At least one dataset required. Upload data in sidebar.")
        st.stop()

    frames = []
    if bbsr_df is not None: frames.append(bbsr_df.assign(city="Bhubaneswar"))
    if delhi_df is not None: frames.append(delhi_df.assign(city="Delhi"))
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    st.subheader("Data preview (combined)")
    st.dataframe(combined.head())

    date_cols_combined = [c for c in combined.columns if "date" in c.lower() or "time" in c.lower()]
    date_col = date_cols_combined[0] if date_cols_combined else None

    # ML model UI
    possible_targets = [c for c in combined.columns if c.lower() == "aqi"]
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in combined data to train model.")
    target_col = st.selectbox("Select target column (AQI recommended)", options=(possible_targets + numeric_cols) if (possible_targets + numeric_cols) else numeric_cols)
    feature_cols = st.multiselect("Feature columns (numeric recommended)", options=[c for c in numeric_cols if c != target_col], default=[c for c in numeric_cols if c != target_col][:6])

    # Train button: model training persisted to session_state
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
            r2 = r2_score(y_test, preds)
            rmse = math.sqrt(mean_squared_error(y_test, preds))
            fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            # persist into session_state
            st.session_state.trained = True
            st.session_state.model_artifact = {"model": model, "features": feature_cols, "target": target_col}
            st.session_state.metrics = {"r2": r2, "rmse": rmse}
            st.session_state.feature_importances = fi
            # save to disk
            save_model_artifact(st.session_state.model_artifact)
            st.success("Training finished and model saved.")

    # Display persisted training results (if any)
    if st.session_state.trained:
        st.markdown("### Training results (persisted)")
        m = st.session_state.metrics
        st.metric("R²", f"{m.get('r2', 0):.3f}")
        st.metric("RMSE", f"{m.get('rmse', 0):.3f}")
        if st.session_state.feature_importances is not None:
            fi_df = st.session_state.feature_importances
            st.subheader("Feature importances")
            st.bar_chart(fi_df.set_index("feature")["importance"])
    else:
        st.info("No trained model in this session yet. Train a model to see persisted metrics.")

    # Batch prediction (kept — upload for prediction is optional)
    st.subheader("Batch prediction (upload CSV with feature columns)")
    pred_file = st.file_uploader("Upload CSV for predictions (features must match trained model)", type=["csv"], key="pred_batch")
    if pred_file is not None:
        pred_df = pd.read_csv(pred_file)
        saved = st.session_state.model_artifact or try_load_model()
        if saved is None:
            st.error("No saved model found in session or disk. Train and save a model first.")
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

    # Tree recommendation logic (restored)
    st.markdown("---")
    st.subheader("Tree Recommendations & Zones")
    city_for_map = map_city  # shared control
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

    all_pollutants = [c for c in combined.columns if c not in [date_col, "year","month","day","city","AQI","aqi"] and pd.api.types.is_numeric_dtype(combined[c])]
    all_pollutants = [c for c in all_pollutants if c is not None]
    selected_pollutants_for_rec = st.multiselect("Select pollutants that concern you (for recommendations)", all_pollutants, default=[c for c in all_pollutants if "pm" in c.lower()][:2])

    if selected_pollutants_for_rec:
        rec_trees = recommend_trees(selected_pollutants_for_rec, env_type)
        st.success(f"Recommended trees: {', '.join(rec_trees)}")
    else:
        rec_trees = []
        st.info("Select pollutants to get tree recommendations.")

    # Tree zone map (MapTiler via Folium)
    st.subheader("Tree Zone Map (interactive markers & benefits)")
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
        hotspots = pd.DataFrame(columns=["lat","lon","intensity"])

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

        if USE_FOLIUM:
            st.markdown("Top hotspots (list):")
            st.dataframe(hotspots.style.format({"intensity":"{:.2f}"}))
            m_city = build_folium_heatmap(df_city, intensity_col="intensity", maptiler_key=MAPTILER_KEY if MAPTILER_KEY else None)
            for mk in markers:
                folium.Marker(
                    location=[mk["lat"], mk["lon"]],
                    popup=folium.Popup(f"Tree: {mk['tree']}<br>Benefit: {mk['benefit']}<br>Intensity: {mk['intensity']}", max_width=300),
                    icon=folium.Icon(color="green", icon="tree", prefix='fa')
                ).add_to(m_city)
            st_folium(m_city, width=900, height=500)
        else:
            marker_layer = pdk.Layer(
                "ScatterplotLayer",
                data=markers,
                get_position='[lon, lat]',
                get_color='[0,150,50,200]',
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
            tooltip = {"text":"Tree: {tree}\nBenefit: {benefit}\nIntensity: {intensity}"}
            deck = pdk.Deck(map_style="open-street-map", initial_view_state=view, layers=[marker_layer, text_layer], tooltip=tooltip)
            st.pydeck_chart(deck)

        if "intensity" in df_city.columns and not df_city["intensity"].isna().all():
            render_legend(float(df_city["intensity"].min()), float(df_city["intensity"].max()), title=f"Intensity (hotspots - {city_for_map})")
        st.markdown(f"Markers show suggested tree species & benefits for top pollution hotspots in **{city_for_map}** (intensity source: {intensity_source}).")
    else:
        st.info(f"No hotspots or recommended trees available for {city_for_map}. Try selecting different pollutants or check dataset completeness.")

# End
st.markdown("---")
st.caption("Maps use MapTiler via Folium when configured (set MAPTILER_KEY). Folium falls back to OpenStreetMap if no key; pydeck is fallback if folium is not installed.")
