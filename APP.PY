import io
import json
import socket
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from sklearn.ensemble import IsolationForest

# --- OPTIONAL INTEL OPENVINO CHECK (Casablanca Workshop Resource) ---
try:
    import openvino as ov
    import openvino.opset13 as ops
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Team Concord | NASA Earth System Trend Detective",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. EARTH SCIENCE VARIABLES CATALOG (MERRA-2 / GLDAS / MODIS / GPM) ---
VARIABLES = {
    "T2M": {
        "name": "Surface Air Temperature (2m)",
        "unit": "°C",
        "mission": "NASA MERRA-2 / AIRS",
        "definition": "Average air temperature at 2 meters above Earth's surface, essential for tracking regional warming trends."
    },
    "PRECTOTCORR": {
        "name": "Bias-Corrected Precipitation",
        "unit": "mm/day",
        "mission": "NASA GPM / MERRA-2",
        "definition": "Total daily rain and snow fall rate, used to detect shifting monsoons, droughts, or extreme rainfall."
    },
    "GWETTOP": {
        "name": "Surface Soil Wetness (Top 5cm)",
        "unit": "fraction (0-1)",
        "mission": "NASA GLDAS / SMAP",
        "definition": "Relative saturation of the top 5cm of soil, a direct indicator of agricultural drought and evaporation stress."
    },
    "QV2M": {
        "name": "Specific Humidity (2m)",
        "unit": "g/kg",
        "mission": "NASA MERRA-2 / MODIS",
        "definition": "Concentration of water vapor in the air, driving atmospheric moisture transport and extreme heat index."
    },
    "ALLSKY_SFC_SW_DWN": {
        "name": "Surface Solar Irradiance (All-Sky)",
        "unit": "W/m²",
        "mission": "NASA CERES / MERRA-2",
        "definition": "Solar radiation reaching Earth's surface, reflecting cloud cover trends and atmospheric aerosols."
    }
}

# --- 2. AI DETECTIVE QUESTIONNAIRE KNOWLEDGE BASE ---
DETECTIVE_SOLUTIONS = {
    ("Semi-Arid Plain / Agricultural Land (e.g., Morocco)", "Drought, Soil Drying & Crop Stress"): {
        "lat": 33.26, "lon": -7.58, "var1": "GWETTOP", "var2": "T2M",
        "region_label": "Berrechid / Chaouia Plain, Morocco",
        "hypothesis": (
            "In semi-arid agricultural plains like Berrechid and Chaouia, rising Surface Air Temperature (T2M) "
            "increases potential evapotranspiration, driving a downward trend in Surface Soil Wetness (GWETTOP). "
            "Notice how the two variables trend in opposite directions from the same warming process."
        )
    },
    ("Tropical Rainforest (e.g., Amazon Basin)", "Deforestation & Water Cycle Shift"): {
        "lat": -3.46, "lon": -62.21, "var1": "PRECTOTCORR", "var2": "T2M",
        "region_label": "Central Amazon Basin, Brazil",
        "hypothesis": (
            "In tropical rainforests, canopy stress and regional warming (T2M) disrupt evapotranspiration "
            "recycling, causing localized precipitation (PRECTOTCORR) and soil moisture patterns to shift."
        )
    },
    ("Coastal Megacity (e.g., Grand Casablanca)", "Urban Heat & Atmospheric Moisture"): {
        "lat": 33.57, "lon": -7.59, "var1": "T2M", "var2": "QV2M",
        "region_label": "Grand Casablanca Coast, Morocco",
        "hypothesis": (
            "In coastal urban atmospheres, ocean warming increases atmospheric water vapor (QV2M) while urban "
            "heat retention raises air temperatures (T2M), creating a compounding upward trend."
        )
    },
    ("Polar / Cryosphere (e.g., High Arctic)", "Accelerated Warming & Ice-Albedo Feedback"): {
        "lat": 78.22, "lon": 15.63, "var1": "T2M", "var2": "PRECTOTCORR",
        "region_label": "Svalbard / High Arctic",
        "hypothesis": (
            "Loss of reflective sea ice exposes darker ocean water, causing Arctic air temperatures (T2M) "
            "to rise rapidly while warmer air holds and precipitates more moisture (PRECTOTCORR)."
        )
    }
}


# --- 3. DATA INGESTION PIPELINE (LIVE NASA API + GIOVANNI CSV LOADER) ---
@st.cache_data(ttl=3600)
def fetch_nasa_merra2_live(lat, lon, var1, var2, start_year, end_year):
    """Fetches real monthly satellite/model time series from NASA POWER (MERRA-2 / GLDAS)."""
    params = f"{var1},{var2}" if var1 != var2 else var1
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters={params}&community=SB&longitude={lon}&latitude={lat}"
        f"&start={start_year}&end={end_year}&format=JSON"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    raw = resp.json()["properties"]["parameter"]

    records = []
    for ym, val1 in raw[var1].items():
        if ym.endswith("13"):  # Skip annual summary key
            continue
        val2 = raw[var2].get(ym, -999)
        if val1 != -999 and val2 != -999:
            year = int(ym[:4])
            month = int(ym[4:6])
            records.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "year": year,
                "month": month,
                var1: float(val1),
                var2: float(val2)
            })
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def parse_giovanni_csv(uploaded_file):
    """Parses downloaded NASA Giovanni or AppEEARS CSV sheets (skips metadata header lines)."""
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = content.splitlines()

    start_idx = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if low.startswith("time,") or low.startswith("date,") or low.startswith("year,") or low.startswith("timestamp,"):
            start_idx = i
            break

    csv_body = "\n".join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(csv_body))
    df.columns = [c.strip() for c in df.columns]

    time_col = df.columns[0]
    val_col1 = df.columns[1]
    val_col2 = df.columns[2] if len(df.columns) > 2 else df.columns[1]

    df["date"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df[val_col1] = pd.to_numeric(df[val_col1], errors="coerce")
    df[val_col2] = pd.to_numeric(df[val_col2], errors="coerce")
    df = df.dropna(subset=[val_col1, val_col2]).reset_index(drop=True)
    return df, val_col1, val_col2


def generate_sample_giovanni_csv():
    """Generates a ready-to-download NASA Giovanni format CSV for testing."""
    years = np.repeat(np.arange(1990, 2026), 12)
    months = np.tile(np.arange(1, 13), 36)
    t_idx = np.arange(len(years))
    # Simulated warming + soil drying with seasonal cycle
    temp = 18.5 + 0.0038 * t_idx + 6.5 * np.sin(2 * np.pi * months / 12.0) + np.random.normal(0, 0.6, len(years))
    soil = 0.42 - 0.00025 * t_idx - 0.12 * np.sin(2 * np.pi * months / 12.0) + np.random.normal(0, 0.02, len(years))
    out_df = pd.DataFrame({
        "time": [f"{y}-{m:02d}-01" for y, m in zip(years, months)],
        "MERRA2_T2M_degC": np.round(temp, 3),
        "GLDAS_SoilWetness": np.round(np.clip(soil, 0.05, 0.95), 3)
    })
    header = (
        "NASA Giovanni Area-Averaged Time Series Export\n"
        "Region: Morocco Chaouia Plain (33.26N, 7.58W)\n"
    )
    return header + out_df.to_csv(index=False)


# --- 4. STATISTICAL SIGNIFICANCE & AI ENGINE (MANN-KENDALL + T-TEST + OPENVINO) ---
def deseasonalize_series(df, col):
    """Removes monthly seasonal cycles to reveal underlying long-term trends."""
    monthly_means = df.groupby("month")[col].transform("mean")
    return df[col] - monthly_means


def mann_kendall_test(y):
    """Performs non-parametric Mann-Kendall monotonic trend test + Sen's slope."""
    n = len(y)
    s = 0
    slopes = []
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = y[j] - y[k]
            s += np.sign(diff)
            slopes.append(diff / (j - k))
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    sens_slope = float(np.median(slopes)) if slopes else 0.0
    return z, p_value, sens_slope


def forecast_with_openvino(years, values, horizon=10):
    """
    Trains an autoregressive sliding-window model combined with linear trend 
    and accelerates inference using Intel OpenVINO Runtime (if installed).
    """
    window = min(5, max(2, len(values) // 4))
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
    weights, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    W = weights[:-1].reshape(window, 1).astype(np.float32)
    b = np.array([weights[-1]], dtype=np.float32)

    compiled_model = None
    engine_name = "NumPy / Scikit-Learn Autoregressive Engine"

    if OPENVINO_AVAILABLE:
        try:
            param = ops.parameter([1, window], np.float32, name="input_window")
            const_w = ops.constant(W, np.float32)
            const_b = ops.constant(b, np.float32)
            matmul = ops.matmul(param, const_w, transpose_a=False, transpose_b=False)
            add_op = ops.add(matmul, const_b)
            ov_model = ov.Model([add_op], [param], "concord_openvino_forecaster")
            core = ov.Core()
            compiled_model = core.compile_model(ov_model, "CPU")
            infer_req = compiled_model.create_infer_request()
            engine_name = "Intel OpenVINO CPU IR Graph"
        except Exception:
            compiled_model = None

    history = list(values[-window:].astype(np.float32))
    preds = []
    for _ in range(horizon):
        inp = np.array([history[-window:]], dtype=np.float32)
        if compiled_model is not None:
            res = infer_req.infer({0: inp})
            next_val = float(list(res.values())[0].flatten()[0])
        else:
            next_val = float(np.dot(inp, W).flatten()[0] + b[0])
        preds.append(next_val)
        history.append(next_val)

    future_years = np.arange(int(years[-1]) + 1, int(years[-1]) + horizon + 1)
    return future_years, np.array(preds), engine_name


def analyze_timeseries(df, col):
    """Full statistical & AI analysis for an Earth science variable."""
    df["anom_" + col] = deseasonalize_series(df, col)
    annual = df.groupby("year").agg(
        raw_mean=(col, "mean"),
        anom_mean=("anom_" + col, "mean")
    ).reset_index()

    x = annual["year"].values
    y = annual["raw_mean"].values

    # 1. Linear Regression + Student's t-test
    lin = stats.linregress(x, y)
    slope_yr = lin.slope
    slope_dec = slope_yr * 10.0
    ttest_p = lin.pvalue

    # 2. 95% Confidence Interval
    t_crit = stats.t.ppf(0.975, max(1, len(x) - 2))
    ci_95_dec = t_crit * lin.stderr * 10.0

    # 3. Mann-Kendall Non-Parametric Trend Test
    mk_z, mk_p, sens_slope = mann_kendall_test(y)

    # 4. Isolation Forest Anomaly Detection (flags extreme climate years)
    iso = IsolationForest(contamination=0.12, random_state=42)
    annual["is_anomaly"] = iso.fit_predict(annual[["raw_mean"]].values) == -1
    anomalies_df = annual[annual["is_anomaly"]]

    # 5. OpenVINO 10-Year AI Forecast
    future_years, forecast_y, ai_engine = forecast_with_openvino(x, y, horizon=10)

    significant = (mk_p < 0.05) and (ttest_p < 0.05)
    direction = "INCREASING" if slope_yr > 0 else "DECREASING"

    return {
        "annual": annual,
        "anomalies": anomalies_df,
        "slope_yr": slope_yr,
        "slope_dec": slope_dec,
        "ci_95_dec": ci_95_dec,
        "ttest_p": ttest_p,
        "mk_z": mk_z,
        "mk_p": mk_p,
        "sens_slope_dec": sens_slope * 10.0,
        "significant": significant,
        "direction": direction,
        "future_years": future_years,
        "forecast_y": forecast_y,
        "ai_engine": ai_engine,
        "r_squared": lin.rvalue ** 2
    }


# --- 5. OPTIONAL LLM AGENT CALL (GEMINI API) + BUILT-IN SCIENTIFIC AGENT ---
def generate_ai_scientific_brief(var1, var2, res1, res2, corr, region_label, hypothesis, gemini_api_key=""):
    """Generates an educational Earth system interpretation of the statistical numbers."""
    paradox_flag = (res1["slope_yr"] * res2["slope_yr"]) < 0
    default_brief = (
        f"### 🧠 AI Detective Synthesis — {region_label}\n"
        f"- **Primary Finding (`{var1}`):** Moving in an **{res1['direction']}** direction at "
        f"**{res1['slope_dec']:+.3f} per decade** (±{res1['ci_95_dec']:.3f} at 95% CI). "
        f"Mann-Kendall $Z = {res1['mk_z']:.2f}$ ($p = {res1['mk_p']:.4f}$) and Student's t-test ($p = {res1['ttest_p']:.4f}$) "
        f"confirm this trend is **{'STATISTICALLY SIGNIFICANT' if res1['significant'] else 'NOT statistically significant (within natural variability)'}**.\n"
        f"- **Secondary Variable (`{var2}`):** Moving in an **{res2['direction']}** direction at "
        f"**{res2['slope_dec']:+.3f} per decade** (Mann-Kendall $p = {res2['mk_p']:.4f}$).\n"
        f"- **Multi-Variable Interaction ($r = {corr:+.2f}$):** "
        f"{'⚠️ **Opposite Trend / Regional Paradox Detected!** Even though a single environmental process drives the region, one variable rises while the other falls.' if paradox_flag else 'Both variables are trending in the same direction over this period.'}\n"
        f"- **Physical Earth System Mechanism:** {hypothesis}"
    )

    if not gemini_api_key.strip():
        return default_brief

    # Optional Live Gemini LLM Enhancement if user provides an API Key
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_api_key.strip()}"
        prompt = (
            f"You are a NASA Earth Science Detective helping college students. Summarize these statistical findings in 4 concise bullet points:\n"
            f"Region: {region_label}\n"
            f"Variable 1 ({var1}): {res1['direction']}, slope={res1['slope_dec']:+.3f}/decade, Mann-Kendall p={res1['mk_p']:.4f}, t-test p={res1['ttest_p']:.4f}.\n"
            f"Variable 2 ({var2}): {res2['direction']}, slope={res2['slope_dec']:+.3f}/decade, Mann-Kendall p={res2['mk_p']:.4f}.\n"
            f"Pearson correlation r={corr:+.2f}. Explain why these variables interact this way physically."
        )
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=8)
        if r.status_code == 200:
            llm_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return f"### ✨ Live Gemini AI Detective Report ({region_label})\n\n" + llm_text
    except Exception:
        pass

    return default_brief


# --- 6. ESP32 UDP & FIREBASE CLOUD DISPATCHER ---
def dispatch_to_hardware(esp_ip, esp_port, payload, firebase_url=""):
    """Sends real-time telemetry to the ESP32 over Wi-Fi UDP + optional Firebase Realtime DB."""
    logs = []
    # Format expected by ESP32 C++ code: VAR|DIR|SLOPE_DEC|P_VAL|LED_CODE
    udp_str = f"{payload['var']}|{payload['dir']}|{payload['slope']:.3f}|{payload['p']:.4f}|{payload['led']}"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        sock.sendto(udp_str.encode("utf-8"), (esp_ip.strip(), int(esp_port)))
        sock.close()
        logs.append(f"✅ UDP Packet Sent to `{esp_ip}:{esp_port}` -> `{udp_str}`")
    except Exception as e:
        logs.append(f"⚠️ Local UDP Notice (normal if hosted on cloud): {e}")

    if firebase_url.strip():
        try:
            endpoint = firebase_url.rstrip("/") + "/concord_station.json"
            requests.put(endpoint, json=payload, timeout=4)
            logs.append("✅ Synced to Firebase Cloud DB")
        except Exception as e:
            logs.append(f"⚠️ Firebase Sync Error: {e}")

    return " | ".join(logs)


# =====================================================================
# 7. MAIN STREAMLIT DASHBOARD UI
# =====================================================================
st.title("🛰️ Team Concord — Earth System Trend Detective")
st.caption(
    "**2026 NASA Space Apps Challenge (Casablanca)** | *Multi-Variable Satellite Time-Series, "
    "Mann-Kendall Significance Engine, Intel OpenVINO Forecaster & ESP32 Telemetry Station*"
)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🗂️ 1. Data Source Mode")
    data_mode = st.radio(
        "Choose NASA Data Pipeline:",
        ["🛰️ Live NASA MERRA-2 / GLDAS API", "📄 NASA Giovanni / AppEEARS CSV Sheet"]
    )

    st.divider()
    st.header("🕵️‍♂️ 2. AI Detective Questionnaire")
    st.caption("Answer two quick questions so the AI Agent selects the best variables & coordinates:")
    q_biome = st.selectbox(
        "Where would you like to investigate?",
        list(dict.fromkeys([k[0] for k in DETECTIVE_SOLUTIONS.keys()]))
    )
    valid_topics = [k[1] for k in DETECTIVE_SOLUTIONS.keys() if k[0] == q_biome]
    q_topic = st.selectbox("What Earth process are you investigating?", valid_topics)

    agent_rec = DETECTIVE_SOLUTIONS[(q_biome, q_topic)]

    if data_mode == "🛰️ Live NASA MERRA-2 / GLDAS API":
        st.subheader("📍 Coordinates & Variables")
        lat = st.number_input("Latitude (-90 to 90)", value=float(agent_rec["lat"]), format="%.2f")
        lon = st.number_input("Longitude (-180 to 180)", value=float(agent_rec["lon"]), format="%.2f")
        var_keys = list(VARIABLES.keys())
        var1 = st.selectbox(
            "Primary Variable",
            var_keys,
            index=var_keys.index(agent_rec["var1"]),
            format_func=lambda k: f"{k} — {VARIABLES[k]['name']}"
        )
        var2 = st.selectbox(
            "Comparison Variable",
            var_keys,
            index=var_keys.index(agent_rec["var2"]),
            format_func=lambda k: f"{k} — {VARIABLES[k]['name']}"
        )
        year_range = st.slider("Satellite Record Period", 1981, 2025, (1990, 2025))
    else:
        st.subheader("📥 Upload Giovanni / AppEEARS CSV")
        sample_csv_bytes = generate_sample_giovanni_csv().encode("utf-8")
        st.download_button(
            "⬇️ Download Sample NASA Giovanni CSV",
            data=sample_csv_bytes,
            file_name="giovanni_morocco_merra2_gldas.csv",
            mime="text/csv"
        )
        uploaded_csv = st.file_uploader("Upload `.csv` time-series file", type=["csv"])

    st.divider()
    st.header("🔌 3. ESP32 Hardware & Cloud")
    esp_ip = st.text_input("ESP32 Local IP (UDP)", value="192.168.1.105")
    esp_port = st.number_input("ESP32 UDP Port", value=4210, step=1)
    firebase_url = st.text_input("Firebase Realtime DB URL (Optional)", value="")
    gemini_key = st.text_input("Gemini API Key (Optional Live LLM)", type="password", value="")

    if OPENVINO_AVAILABLE:
        st.success("⚡ Intel OpenVINO Runtime: ACTIVE")
    else:
        st.info("🧠 AI Engine: Autoregressive + IsolationForest")

# --- EDUCATIONAL VARIABLE DEFINITIONS BANNER ---
if data_mode == "🛰️ Live NASA MERRA-2 / GLDAS API":
    c_info1, c_info2 = st.columns(2)
    with c_info1:
        st.info(
            f"**📘 Primary Variable: `{var1}` — {VARIABLES[var1]['name']}**\n\n"
            f"**Mission:** {VARIABLES[var1]['mission']} | **Unit:** `{VARIABLES[var1]['unit']}`\n\n"
            f"{VARIABLES[var1]['definition']}"
        )
    with c_info2:
        st.info(
            f"**📙 Comparison Variable: `{var2}` — {VARIABLES[var2]['name']}**\n\n"
            f"**Mission:** {VARIABLES[var2]['mission']} | **Unit:** `{VARIABLES[var2]['unit']}`\n\n"
            f"{VARIABLES[var2]['definition']}"
        )

# --- RUN ANALYSIS BUTTON ---
if st.button("🚀 Run Earth System Trend Detective & Dispatch to ESP32", type="primary", use_container_width=True):
    with st.spinner("Acquiring NASA time-series, removing seasonal cycles, and running Mann-Kendall & OpenVINO models..."):
        try:
            # 1. Load Data from either Live NASA API or Uploaded Giovanni CSV
            if data_mode == "🛰️ Live NASA MERRA-2 / GLDAS API":
                df = fetch_nasa_merra2_live(lat, lon, var1, var2, year_range[0], year_range[1])
                unit1 = VARIABLES[var1]["unit"]
                unit2 = VARIABLES[var2]["unit"]
                region_title = f"{agent_rec['region_label']} ({lat:.2f}°, {lon:.2f}°)"
            else:
                if uploaded_csv is not None:
                    df, var1, var2 = parse_giovanni_csv(uploaded_csv)
                    region_title = f"Uploaded NASA Giovanni CSV ({uploaded_csv.name})"
                else:
                    df, var1, var2 = parse_giovanni_csv(io.BytesIO(generate_sample_giovanni_csv().encode("utf-8")))
                    region_title = "Sample NASA Giovanni Dataset (Morocco Chaouia Plain)"
                lat, lon = agent_rec["lat"], agent_rec["lon"]
                unit1, unit2 = "units", "units"

            # 2. Run Statistical & AI Analysis on Both Variables
            res1 = analyze_timeseries(df, var1)
            res2 = analyze_timeseries(df, var2)

            corr, _ = stats.pearsonr(res1["annual"]["raw_mean"], res2["annual"]["raw_mean"])
            opposite_directions = (res1["slope_yr"] * res2["slope_yr"]) < 0

            # 3. Compute ESP32 5-LED Code
            # LED 2 (Green) = No Sig Trend, LED 3 (Yellow) = Mod Sig (p < 0.05), LED 4 (Red) = Crit Sig (p < 0.01)
            if res1["mk_p"] < 0.01:
                led_code = 4
            elif res1["mk_p"] < 0.05:
                led_code = 3
            else:
                led_code = 2

            # Add +10 offset if opposite trends (Paradox LED 5)
            if opposite_directions and (res1["significant"] or res2["significant"]):
                led_code += 10

            # 4. Top Summary KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                f"{var1} Rate of Change",
                f"{res1['slope_dec']:+.3f} {unit1}/dec",
                f"±{res1['ci_95_dec']:.3f} (95% CI)"
            )
            k2.metric(
                f"{var1} Mann-Kendall Test",
                f"p = {res1['mk_p']:.4f}",
                "SIGNIFICANT TREND" if res1["significant"] else "Not Significant",
                delta_color="normal" if res1["significant"] else "off"
            )
            k3.metric(
                f"{var2} Rate of Change",
                f"{res2['slope_dec']:+.3f} {unit2}/dec",
                f"p = {res2['mk_p']:.4f} ({'SIG' if res2['significant'] else 'NS'})"
            )
            k4.metric(
                "Cross-Variable Correlation",
                f"r = {corr:+.2f}",
                "Opposite Directions (Paradox)" if opposite_directions else "Same Direction"
            )

            # 5. Primary Interactive Chart (Observed + Linear Trend + Anomalies + OpenVINO 10-Yr Forecast)
            st.subheader(f"📈 {var1} Time-Series, Linear Significance Fit & 10-Year AI Forecast — {region_title}")
            ann1 = res1["annual"]
            anom1 = res1["anomalies"]

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=ann1["year"], y=ann1["raw_mean"],
                mode="lines+markers", name=f"Annual Mean {var1}",
                line=dict(color="#38bdf8", width=2)
            ))
            trend_line = res1["slope_yr"] * ann1["year"] + (ann1["raw_mean"].mean() - res1["slope_yr"] * ann1["year"].mean())
            fig1.add_trace(go.Scatter(
                x=ann1["year"], y=trend_line,
                mode="lines",
                name=f"Linear Trend ({res1['slope_dec']:+.3f}/dec, t-test p={res1['ttest_p']:.4f})",
                line=dict(color="#ef4444", width=3, dash="dash")
            ))
            fig1.add_trace(go.Scatter(
                x=anom1["year"], y=anom1["raw_mean"],
                mode="markers", name="Isolation Forest Extreme Anomaly Years",
                marker=dict(color="#f43f5e", size=11, symbol="star")
            ))
            fig1.add_trace(go.Scatter(
                x=res1["future_years"], y=res1["forecast_y"],
                mode="lines+markers", name=f"10-Yr Forecast ({res1['ai_engine']})",
                line=dict(color="#f59e0b", width=3, dash="dot")
            ))
            fig1.update_layout(height=420, template="plotly_dark", xaxis_title="Year", yaxis_title=unit1)
            st.plotly_chart(fig1, use_container_width=True)

            # 6. Multi-Variable Z-Score Comparison & Geographic Map
            col_chart, col_map = st.columns([3, 2])
            with col_chart:
                st.subheader("🔄 Deseasonalized Multi-Variable Comparison (Z-Score Anomalies)")
                z1 = stats.zscore(ann1["anom_mean"])
                z2 = stats.zscore(res2["annual"]["anom_mean"])
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=ann1["year"], y=z1, mode="lines", name=f"{var1} Anomaly (σ)", line=dict(color="#38bdf8", width=2.5)))
                fig2.add_trace(go.Scatter(x=res2["annual"]["year"], y=z2, mode="lines", name=f"{var2} Anomaly (σ)", line=dict(color="#a855f7", width=2.5)))
                fig2.update_layout(height=340, template="plotly_dark", xaxis_title="Year", yaxis_title="Standardized Anomaly (σ)")
                st.plotly_chart(fig2, use_container_width=True)

            with col_map:
                st.subheader("🌍 Investigated Coordinates")
                st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=4)

            # 7. AI Detective Scientific Synthesis Report
            report_md = generate_ai_scientific_brief(
                var1, var2, res1, res2, corr, region_title, agent_rec["hypothesis"], gemini_key
            )
            st.info(report_md)

            # 8. Dispatch Telemetry to Physical ESP32 Station + Visual LED Preview
            hw_payload = {
                "var": var1[:10],
                "dir": res1["direction"][:3],
                "slope": float(res1["slope_dec"]),
                "p": float(res1["mk_p"]),
                "led": int(led_code)
            }
            dispatch_msg = dispatch_to_hardware(esp_ip, esp_port, hw_payload, firebase_url)

            st.subheader("📟 ESP32 Physical Station Status (OLED + 5 LEDs)")
            base_led = led_code - 10 if led_code >= 10 else led_code
            paradox_on = led_code >= 10
            l1, l2, l3, l4, l5 = st.columns(5)
            l1.markdown("🔵 **LED 1 (Blue):** `ON` (Packet Rx)")
            l2.markdown(f"🟢 **LED 2 (Green):** `{'ON (Stable)' if base_led == 2 else 'OFF'}`")
            l3.markdown(f"🟡 **LED 3 (Yellow):** `{'ON (p < 0.05)' if base_led == 3 else 'OFF'}`")
            l4.markdown(f"🔴 **LED 4 (Red):** `{'ON (p < 0.01)' if base_led == 4 else 'OFF'}`")
            l5.markdown(f"🟣 **LED 5 (Paradox):** `{'ON (Opposite!)' if paradox_on else 'OFF'}`")
            st.caption(dispatch_msg)

        except Exception as e:
            st.error(f"Error during NASA data acquisition or statistical processing: {e}")