import io
import re
import json
import socket
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# --- OPTIONAL INTEL OPENVINO RUNTIME CHECK ---
try:
    import openvino as ov
    import openvino.opset13 as ops
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# ==============================================================================
# PAGE & SESSION STATE INITIALIZATION
# ==============================================================================
st.set_page_config(
    page_title="Team Concord | Earth System Trend Detective",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = []

if "active_case" not in st.session_state:
    st.session_state["active_case"] = None

# ==============================================================================
# EARTH SCIENCE VARIABLES CATALOG (MERRA-2 / GLDAS / GPM / CERES)
# ==============================================================================
VARIABLES = {
    "T2M": {
        "name": "Surface Air Temperature (2m)",
        "unit": "°C",
        "mission": "NASA MERRA-2 / AIRS",
        "category": "Thermal",
        "desc": "Air temperature at 2 meters above Earth's surface, primary indicator of regional warming."
    },
    "PRECTOTCORR": {
        "name": "Bias-Corrected Precipitation",
        "unit": "mm/day",
        "mission": "NASA GPM / MERRA-2",
        "category": "Hydrological",
        "desc": "Total daily rain and snow fall rate, tracking droughts, monsoons, and extreme rainfall."
    },
    "GWETTOP": {
        "name": "Surface Soil Wetness (Top 5cm)",
        "unit": "fraction (0-1)",
        "mission": "NASA GLDAS / SMAP",
        "category": "Hydrological",
        "desc": "Relative water saturation in the top 5cm of soil, measuring agricultural drought and evaporation."
    },
    "QV2M": {
        "name": "Specific Humidity (2m)",
        "unit": "g/kg",
        "mission": "NASA MERRA-2 / MODIS",
        "category": "Hydrological",
        "desc": "Mass of water vapor per kilogram of air, driving atmospheric moisture transport and heat stress."
    },
    "ALLSKY_SFC_SW_DWN": {
        "name": "Surface Solar Irradiance",
        "unit": "W/m²",
        "mission": "NASA CERES / MERRA-2",
        "category": "Radiative",
        "desc": "Shortwave solar radiation reaching the surface, influenced by cloud cover and aerosols."
    },
    "WS2M": {
        "name": "Wind Speed at 2 Meters",
        "unit": "m/s",
        "mission": "NASA MERRA-2",
        "category": "Atmospheric Dynamics",
        "desc": "Near-surface wind speed driving surface evaporation, dust transport, and coastal upwelling."
    }
}

# ==============================================================================
# STEP 1: AI AGENT ROUTER (USER PROMPT -> MULTI-VARIABLE JSON PAYLOAD)
# ==============================================================================
def parse_user_prompt_with_ai(user_prompt, gemini_api_key=""):
    """
    Uses Gemini 2.5 Flash (or a deterministic fallback parser) to convert a natural 
    language investigation request into coordinates, time range, 1 Target Variable, 
    and 3 physical Driver Variables.
    """
    valid_vars = list(VARIABLES.keys())

    if gemini_api_key.strip():
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={gemini_api_key.strip()}"
            )
            system_instruction = (
                "You are a NASA Earth Science AI Router. Given the user's investigation prompt, "
                "return ONLY a valid JSON object (no markdown backticks) with these exact keys:\n"
                "{\n"
                '  "region_name": "String name of region",\n'
                '  "lat": float (-90 to 90),\n'
                '  "lon": float (-180 to 180),\n'
                '  "start_year": int (between 1981 and 2015),\n'
                '  "end_year": int (between 2020 and 2025),\n'
                f'  "target_var": one of {valid_vars},\n'
                f'  "driver_vars": list of exactly 3 distinct variables from {valid_vars} (excluding target_var),\n'
                '  "hypothesis": "1-sentence physical explanation of how the drivers affect the target variable in this region."\n'
                "}"
            )
            payload = {
                "contents": [{"parts": [{"text": f"{system_instruction}\n\nUser Prompt: {user_prompt}"}]}]
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                raw_text = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
                parsed = json.loads(raw_text)
                if parsed.get("target_var") in valid_vars:
                    drivers = [v for v in parsed.get("driver_vars", []) if v in valid_vars and v != parsed["target_var"]]
                    for fallback_v in valid_vars:
                        if len(drivers) < 3 and fallback_v != parsed["target_var"] and fallback_v not in drivers:
                            drivers.append(fallback_v)
                    parsed["driver_vars"] = drivers[:3]
                    parsed["router_source"] = "Gemini 2.5 Flash Live Agent"
                    return parsed
        except Exception:
            pass

    # Deterministic Intelligent Fallback Router (works offline or without API key)
    p_low = user_prompt.lower()
    if "amazon" in p_low or "rainforest" in p_low or "brazil" in p_low:
        base = {
            "region_name": "Amazon Rainforest Basin, Brazil",
            "lat": -3.46, "lon": -62.21, "start_year": 1990, "end_year": 2025,
            "target_var": "PRECTOTCORR", "driver_vars": ["T2M", "QV2M", "ALLSKY_SFC_SW_DWN"],
            "hypothesis": "Canopy warming (T2M) and solar radiation shifts alter atmospheric humidity (QV2M) and rainfall recycling (PRECTOTCORR)."
        }
    elif "arctic" in p_low or "polar" in p_low or "ice" in p_low or "greenland" in p_low:
        base = {
            "region_name": "Svalbard / High Arctic",
            "lat": 78.22, "lon": 15.63, "start_year": 1990, "end_year": 2025,
            "target_var": "T2M", "driver_vars": ["ALLSKY_SFC_SW_DWN", "QV2M", "PRECTOTCORR"],
            "hypothesis": "Ice-albedo feedback amplifies surface warming (T2M), increasing specific humidity (QV2M) and polar precipitation."
        }
    elif "tangier" in p_low or "north morocco" in p_low or "rif" in p_low:
        base = {
            "region_name": "Tangier / Northern Morocco",
            "lat": 35.76, "lon": -5.83, "start_year": 1990, "end_year": 2025,
            "target_var": "PRECTOTCORR", "driver_vars": ["T2M", "QV2M", "WS2M"],
            "hypothesis": "Mediterranean moisture transport (QV2M, WS2M) interacts with rising air temperatures (T2M) to shift coastal rainfall regimes."
        }
    elif "sahara" in p_low or "errachidia" in p_low or "desert" in p_low or "ouarzazate" in p_low:
        base = {
            "region_name": "Pre-Sahara / Ouarzazate Basin, Morocco",
            "lat": 30.92, "lon": -6.90, "start_year": 1990, "end_year": 2025,
            "target_var": "GWETTOP", "driver_vars": ["T2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"],
            "hypothesis": "Intense solar irradiance and rising temperatures accelerate arid topsoil desiccation while sporadic rainfall creates flash anomalies."
        }
    else:
        base = {
            "region_name": "Berrechid / Grand Casablanca Plain, Morocco",
            "lat": 33.26, "lon": -7.58, "start_year": 1990, "end_year": 2025,
            "target_var": "GWETTOP", "driver_vars": ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
            "hypothesis": "Rising air temperature (T2M) and solar radiation (ALLSKY_SFC_SW_DWN) accelerate evapotranspiration, depleting topsoil moisture (GWETTOP) faster than rainfall replenishes it."
        }

    base["router_source"] = "Built-in Domain Knowledge Router"
    return base


# ==============================================================================
# STEP 2 & 3.1: NASA API FETCHER + DATA TABLE STRUCTURING
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_nasa_multivariable_data(lat, lon, start_year, end_year, var_list):
    """
    Queries NASA POWER API (MERRA-2 & GLDAS satellite/model archive) for all 4 variables 
    simultaneously and returns a clean monthly Pandas DataFrame.
    """
    unique_vars = list(dict.fromkeys(var_list))
    params_str = ",".join(unique_vars)
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters={params_str}&community=SB&longitude={lon}&latitude={lat}"
        f"&start={start_year}&end={end_year}&format=JSON"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    param_data = resp.json()["properties"]["parameter"]

    time_keys = sorted(list(param_data[unique_vars[0]].keys()))
    records = []

    for ym in time_keys:
        if ym.endswith("13"):  # Skip NASA's annual average summary key
            continue
        row = {
            "date": pd.Timestamp(year=int(ym[:4]), month=int(ym[4:6]), day=1),
            "year": int(ym[:4]),
            "month": int(ym[4:6])
        }
        valid_row = True
        for v in unique_vars:
            val = float(param_data[v].get(ym, -999.0))
            if val <= -990.0:
                valid_row = False
                break
            row[v] = val
        if valid_row:
            records.append(row)

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


# ==============================================================================
# STEP 3.2: DESEASONALIZATION (REMOVING THE SEASONAL ZIG-ZAG CYCLE)
# ==============================================================================
def deseasonalize_dataframe(df, var_list):
    """
    Removes the regular monthly seasonal cycle (summer/winter wave) from each variable 
    so true underlying anomalies and multi-decade trends are exposed.
    """
    df_clean = df.copy()
    for v in var_list:
        monthly_climatology = df_clean.groupby("month")[v].transform("mean")
        df_clean[f"{v}_anom"] = df_clean[v] - monthly_climatology
    return df_clean


# ==============================================================================
# STEP 3.3: 4-LAYER MULTI-VARIABLE INTELLIGENCE ENGINE
# ==============================================================================
def mann_kendall_math(y):
    """Calculates Mann-Kendall S statistic, Z-score, p-value, and Sen's median slope."""
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
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    sens_slope = float(np.median(slopes)) if slopes else 0.0
    return z, p_val, sens_slope


def run_multivariable_intelligence_engine(df_monthly, target_var, driver_vars):
    """
    Executes the 4 analytical layers after Step 3.2 deseasonalization:
      Layer 1: Per-variable Mann-Kendall, Student's t-test, and 95% Confidence Intervals.
      Layer 2: Driver-Response Cross-Correlation, Time-Lag (0-3 mos), & Paradox Detection.
      Layer 3: CUSUM Tipping-Point / Regime Shift Year & Acceleration Ratio.
      Layer 4: Multivariate AI Forecast (Random Forest + OpenVINO IR Graph) & What-If Model.
    """
    all_vars = [target_var] + driver_vars
    df_deseason = deseasonalize_dataframe(df_monthly, all_vars)

    # Annual aggregation of raw and deseasonalized anomalies
    agg_dict = {v: "mean" for v in all_vars}
    for v in all_vars:
        agg_dict[f"{v}_anom"] = "mean"
    annual_df = df_deseason.groupby("year").agg(agg_dict).reset_index()
    years = annual_df["year"].values

    # --- LAYER 1: BASELINE SIGNIFICANCE TESTS FOR ALL 4 VARIABLES ---
    var_stats = {}
    for v in all_vars:
        y_anom = annual_df[f"{v}_anom"].values
        y_raw = annual_df[v].values

        lin = stats.linregress(years, y_anom)
        t_crit = stats.t.ppf(0.975, max(1, len(years) - 2))
        ci_95_dec = t_crit * lin.stderr * 10.0
        mk_z, mk_p, sens_slope = mann_kendall_math(y_anom)

        # Unsupervised Isolation Forest for extreme year detection
        iso = IsolationForest(contamination=0.12, random_state=42)
        anom_flags = iso.fit_predict(y_anom.reshape(-1, 1)) == -1

        var_stats[v] = {
            "mean_raw": float(np.mean(y_raw)),
            "std_raw": float(np.std(y_raw)) if np.std(y_raw) > 0 else 1.0,
            "slope_yr": float(lin.slope),
            "slope_dec": float(lin.slope * 10.0),
            "sens_slope_dec": float(sens_slope * 10.0),
            "ci_95_dec": float(ci_95_dec),
            "ttest_p": float(lin.pvalue),
            "mk_z": float(mk_z),
            "mk_p": float(mk_p),
            "r_squared": float(lin.rvalue ** 2),
            "significant": bool((mk_p < 0.05) and (lin.pvalue < 0.05)),
            "direction": "INCREASING" if lin.slope > 0 else "DECREASING",
            "extreme_years": years[anom_flags].tolist(),
            "extreme_vals": y_raw[anom_flags].tolist()
        }

    # --- LAYER 2: CROSS-CORRELATION MATRIX, TIME-LAG & PARADOX DISCOVERY ---
    anom_cols = [f"{v}_anom" for v in all_vars]
    corr_matrix = df_deseason[anom_cols].corr()
    corr_matrix.index = all_vars
    corr_matrix.columns = all_vars

    driver_couplings = {}
    strongest_driver = driver_vars[0]
    max_corr_mag = -1.0

    for d_var in driver_vars:
        r_annual, p_annual = stats.pearsonr(annual_df[f"{target_var}_anom"], annual_df[f"{d_var}_anom"])

        # Test monthly time lags (0, 1, 2, 3 months) to see if driver precedes target
        best_lag = 0
        best_lag_r = float(df_deseason[f"{target_var}_anom"].corr(df_deseason[f"{d_var}_anom"]))
        for lag in [1, 2, 3]:
            r_lag = float(df_deseason[f"{target_var}_anom"].corr(df_deseason[f"{d_var}_anom"].shift(lag)))
            if abs(r_lag) > abs(best_lag_r):
                best_lag_r = r_lag
                best_lag = lag

        opposite_slopes = (var_stats[target_var]["slope_dec"] * var_stats[d_var]["slope_dec"]) < 0
        paradox_coupled = bool(opposite_slopes and (abs(r_annual) >= 0.35))

        driver_couplings[d_var] = {
            "r_annual": float(r_annual),
            "p_annual": float(p_annual),
            "best_lag_months": int(best_lag),
            "lag_r": float(best_lag_r),
            "opposite_slopes": bool(opposite_slopes),
            "paradox_coupled": paradox_coupled
        }

        if abs(r_annual) > max_corr_mag:
            max_corr_mag = abs(r_annual)
            strongest_driver = d_var

    # --- LAYER 3: TIPPING-POINT / REGIME SHIFT DETECTION (CUSUM) ---
    target_anom = annual_df[f"{target_var}_anom"].values
    cusum = np.cumsum(target_anom - np.mean(target_anom))
    margin = max(4, len(years) // 5)
    valid_cusum = np.abs(cusum[margin:-margin])
    tip_idx = int(np.argmax(valid_cusum)) + margin
    tipping_year = int(years[tip_idx])

    slope_pre = float(stats.linregress(years[:tip_idx], target_anom[:tip_idx]).slope * 10.0)
    slope_post = float(stats.linregress(years[tip_idx:], target_anom[tip_idx:]).slope * 10.0)
    accel_ratio = abs(slope_post) / max(abs(slope_pre), 1e-6)
    regime_shifted = bool(accel_ratio >= 1.4 and var_stats[target_var]["ttest_p"] < 0.10)

    # --- LAYER 4: MULTIVARIATE AI FORECASTING (RANDOM FOREST + OPENVINO) ---
    # Feature matrix uses all 4 variables at year t to predict Target at year t+1
    X_raw = annual_df[all_vars].values
    X_train = X_raw[:-1]
    y_train = annual_df[target_var].values[1:]

    rf_model = RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)

    # Also fit a linear multivariate weight matrix for OpenVINO IR Graph acceleration
    X_aug = np.hstack([X_train, np.ones((X_train.shape[0], 1), dtype=np.float32)]).astype(np.float32)
    weights, _, _, _ = np.linalg.lstsq(X_aug, y_train.astype(np.float32), rcond=None)
    W_mv = weights[:-1].reshape(len(all_vars), 1).astype(np.float32)
    b_mv = np.array([weights[-1]], dtype=np.float32)

    ov_infer_req = None
    ai_engine_label = "Multivariate Random Forest + Linear Coupling"
    if OPENVINO_AVAILABLE:
        try:
            param = ops.parameter([1, len(all_vars)], np.float32, name="multivar_input")
            matmul = ops.matmul(param, ops.constant(W_mv, np.float32), transpose_a=False, transpose_b=False)
            add_op = ops.add(matmul, ops.constant(b_mv, np.float32))
            ov_model = ov.Model([add_op], [param], "concord_multivar_forecaster")
            compiled = ov.Core().compile_model(ov_model, "CPU")
            ov_infer_req = compiled.create_infer_request()
            ai_engine_label = "Intel OpenVINO IR Graph + Random Forest Ensemble"
        except Exception:
            ov_infer_req = None

    future_years = np.arange(int(years[-1]) + 1, int(years[-1]) + 11)
    curr_vec = X_raw[-1].copy().astype(np.float32)
    forecast_vals = []

    for _ in range(10):
        rf_pred = float(rf_model.predict([curr_vec])[0])
        if ov_infer_req is not None:
            ov_res = ov_infer_req.infer({0: curr_vec.reshape(1, -1)})
            lin_pred = float(list(ov_res.values())[0].flatten()[0])
        else:
            lin_pred = float(np.dot(curr_vec, W_mv).flatten()[0] + b_mv[0])

        # Ensemble blend of non-linear Random Forest + OpenVINO linear coupling + trend slope
        next_target = 0.50 * rf_pred + 0.30 * lin_pred + 0.20 * (curr_vec[0] + var_stats[target_var]["slope_yr"])
        forecast_vals.append(next_target)

        # Advance state vector for next year
        curr_vec[0] = next_target
        for idx, d_var in enumerate(driver_vars, start=1):
            curr_vec[idx] += var_stats[d_var]["slope_yr"]

    # Calculate feature importances from Random Forest
    importances = dict(zip(all_vars, rf_model.feature_importances_))

    # --- STEP 3.5: 5-LED BITMASK GENERATION (BLUE, ORANGE, RED, WHITE, GREEN) ---
    hydro_vars = {"PRECTOTCORR", "GWETTOP", "QV2M"}
    led_blue   = int(any(var_stats[v]["significant"] for v in all_vars if v in hydro_vars))
    led_orange = int(regime_shifted)
    led_red    = int(var_stats[target_var]["mk_p"] < 0.01)
    led_white  = int(any(driver_couplings[d]["paradox_coupled"] for d in driver_vars))
    led_green  = int((not var_stats[target_var]["significant"]) and (not regime_shifted))
    led_bitmask = f"{led_blue}{led_orange}{led_red}{led_white}{led_green}"

    return {
        "df_monthly": df_deseason,
        "annual_df": annual_df,
        "target_var": target_var,
        "driver_vars": driver_vars,
        "all_vars": all_vars,
        "var_stats": var_stats,
        "corr_matrix": corr_matrix,
        "driver_couplings": driver_couplings,
        "strongest_driver": strongest_driver,
        "tipping_year": tipping_year,
        "slope_pre": slope_pre,
        "slope_post": slope_post,
        "accel_ratio": accel_ratio,
        "regime_shifted": regime_shifted,
        "future_years": future_years,
        "forecast_vals": np.array(forecast_vals),
        "rf_model": rf_model,
        "importances": importances,
        "ai_engine_label": ai_engine_label,
        "led_bitmask": led_bitmask
    }


# ==============================================================================
# STEP 3.5: ESP32 UDP + FIREBASE HARDWARE DISPATCHER
# ==============================================================================
def send_multivariable_udp_to_esp32(esp_ip, esp_port, case_data, firebase_url=""):
    """
    Sends the compact multi-variable intelligence string over Wi-Fi UDP to the ESP32:
    Format: TARGET|SLOPE_DEC|STRONGEST_DRIVER|CORR|TIP_YEAR|PRED_2035|BITMASK_5LEDS
    Example: GWETTOP|-0.045|T2M|-0.74|2016|0.210|11110
    """
    res = case_data["intel"]
    t_var = res["target_var"]
    s_drv = res["strongest_driver"]
    slope_dec = res["var_stats"][t_var]["slope_dec"]
    corr_val = res["driver_couplings"][s_drv]["r_annual"]
    tip_yr = res["tipping_year"]
    pred_end = float(res["forecast_vals"][-1])
    bitmask = res["led_bitmask"]

    udp_payload = f"{t_var}|{slope_dec:+.3f}|{s_drv}|{corr_val:+.2f}|{tip_yr}|{pred_end:.3f}|{bitmask}"
    status_notes = []

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        sock.sendto(udp_payload.encode("utf-8"), (esp_ip.strip(), int(esp_port)))
        sock.close()
        status_notes.append(f"✅ UDP Sent to `{esp_ip}:{esp_port}` ➔ `{udp_payload}`")
    except Exception as e:
        status_notes.append(f"⚠️ Local UDP Warning: {e}")

    if firebase_url.strip():
        try:
            fb_target = firebase_url.rstrip("/") + "/concord_mission_control.json"
            requests.put(fb_target, json={
                "region": case_data["meta"]["region_name"],
                "udp_string": udp_payload,
                "target_var": t_var,
                "slope_decade": slope_dec,
                "strongest_driver": s_drv,
                "driver_corr": corr_val,
                "tipping_year": tip_yr,
                "led_bitmask": bitmask
            }, timeout=4)
            status_notes.append("✅ Synced to Firebase Cloud")
        except Exception as e:
            status_notes.append(f"⚠️ Firebase Error: {e}")

    return " | ".join(status_notes), udp_payload


# ==============================================================================
# SIDEBAR: CONFIGURATION, MANUAL OVERRIDE & ESP32 LINK
# ==============================================================================
with st.sidebar:
    st.header("🛰️ Concord Control Tower")
    # Check if key exists in secrets, otherwise leave blank
    default_gemini = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""

    gemini_key = st.text_input("🔑 Gemini 2.5 Flash API Key", value=default_gemini, type="password")
    st.divider()
    st.subheader("🎛️ Manual Variable Override")
    enable_override = st.checkbox("Override AI Variable Selection", value=False)
    var_keys = list(VARIABLES.keys())
    man_lat = st.number_input("Latitude", value=33.26, format="%.2f", disabled=not enable_override)
    man_lon = st.number_input("Longitude", value=-7.58, format="%.2f", disabled=not enable_override)
    man_target = st.selectbox("Target Variable", var_keys, index=2, disabled=not enable_override)
    man_drivers = st.multiselect(
        "Driver Variables (Pick 3)",
        [k for k in var_keys if k != man_target],
        default=["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"][:3],
        disabled=not enable_override
    )
    man_years = st.slider("Year Range", 1981, 2025, (1990, 2025), disabled=not enable_override)

    st.divider()
    st.subheader("🔌 ESP32 Hardware Telemetry")
    esp_ip = st.text_input("ESP32 Local IP Address", value="192.168.1.105")
    esp_port = st.number_input("ESP32 UDP Port", value=4210, step=1)
    firebase_url = st.text_input("Firebase DB URL (Optional)", value="")

    if st.button("🗑️ Clear Top Watchlist Ribbon", use_container_width=True):
        st.session_state["watchlist"] = []
        st.rerun()


# ==============================================================================
# DASHBOARD HEADER & TOP RIBBON: PREVIOUS INVESTIGATIONS WATCHLIST MODULES
# ==============================================================================
st.title("🛰️ Team Concord — Multi-Variable Earth System Trend Detective")
st.caption(
    "**2026 NASA Space Apps Challenge (Casablanca)** | *AI Prompt Router • Seasonal Cycle Filter • "
    "Cross-Variable Coupling • Tipping-Point Detection • Multivariate AI Forecast & 5-LED ESP32 Station*"
)

st.subheader("📌 Top Watchlist Ribbon — Investigated Regions & Significant Alerts")
if len(st.session_state["watchlist"]) == 0:
    st.info("No investigations saved yet. Enter a prompt in **Module 1** below to populate your live Mission Control Watchlist!")
else:
    cols_watch = st.columns(min(4, len(st.session_state["watchlist"])))
    for idx, item in enumerate(st.session_state["watchlist"][-4:]):
        w_meta = item["meta"]
        w_intel = item["intel"]
        t_v = w_intel["target_var"]
        t_stat = w_intel["var_stats"][t_v]
        sig_badge = "🔴 SIGNIFICANT" if t_stat["significant"] else "🟢 STABLE/NOISE"
        shift_badge = f"🟠 Shift: {w_intel['tipping_year']}" if w_intel["regime_shifted"] else f"Tip: {w_intel['tipping_year']}"
        with cols_watch[idx]:
            with st.container(border=True):
                st.markdown(f"**#{idx + 1} {w_meta['region_name']}**")
                st.markdown(
                    f"**Target:** `{t_v}` ({t_stat['slope_dec']:+.3f} {VARIABLES[t_v]['unit']}/dec)\n\n"
                    f"{sig_badge} (`p={t_stat['mk_p']:.4f}`) | {shift_badge}\n\n"
                    f"**Strongest Driver:** `{w_intel['strongest_driver']}` ($r={w_intel['driver_couplings'][w_intel['strongest_driver']]['r_annual']:+.2f}$) | **LEDs:** `{w_intel['led_bitmask']}`"
                )
                if st.button(f"Load Case #{idx + 1}", key=f"load_watch_{idx}", use_container_width=True):
                    st.session_state["active_case"] = item
                    st.rerun()

    # Cross-Region Paradox Comparison Expander when 2+ investigations exist
    if len(st.session_state["watchlist"]) >= 2:
        with st.expander("🌍 Compare Two Watchlist Regions Side-by-Side (Regional Paradox Detector)"):
            c_sel1, c_sel2 = st.columns(2)
            names = [f"#{i+1}: {w['meta']['region_name']} ({w['intel']['target_var']})" for i, w in enumerate(st.session_state["watchlist"])]
            idx_a = c_sel1.selectbox("Region A", range(len(names)), format_func=lambda i: names[i], index=0)
            idx_b = c_sel2.selectbox("Region B", range(len(names)), format_func=lambda i: names[i], index=min(1, len(names) - 1))

            case_a = st.session_state["watchlist"][idx_a]
            case_b = st.session_state["watchlist"][idx_b]
            va = case_a["intel"]["target_var"]
            vb = case_b["intel"]["target_var"]
            sa = case_a["intel"]["var_stats"][va]["slope_dec"]
            sb = case_b["intel"]["var_stats"][vb]["slope_dec"]

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=case_a["intel"]["annual_df"]["year"],
                y=stats.zscore(case_a["intel"]["annual_df"][f"{va}_anom"]),
                mode="lines+markers", name=f"{case_a['meta']['region_name']} ({va}: {sa:+.3f}/dec)"
            ))
            fig_cmp.add_trace(go.Scatter(
                x=case_b["intel"]["annual_df"]["year"],
                y=stats.zscore(case_b["intel"]["annual_df"][f"{vb}_anom"]),
                mode="lines+markers", name=f"{case_b['meta']['region_name']} ({vb}: {sb:+.3f}/dec)"
            ))
            fig_cmp.update_layout(height=300, template="plotly_dark", yaxis_title="Deseasonalized Z-Anomaly (σ)")
            st.plotly_chart(fig_cmp, use_container_width=True)
            if sa * sb < 0:
                st.warning("⚡ **Regional Divergence Confirmed:** These two regions are trending in opposite directions over the same period!")

st.divider()

# ==============================================================================
# MODULE 1: AI PROMPT ROUTER & 4-VARIABLE CLUSTER LAUNCHER
# ==============================================================================
st.subheader("🧭 Module 1 — AI Prompt Router & Multi-Variable Cluster Selection")
col_prompt, col_quick = st.columns([3, 2])

with col_quick:
    preset_choice = st.selectbox(
        "Or pick a quick investigation prompt:",
        [
            "Analyze agricultural drought, soil drying, and heat stress in Berrechid / Casablanca, Morocco from 1990 to 2025",
            "Investigate extreme rainfall shifts, humidity, and Mediterranean wind patterns in Tangier, Northern Morocco",
            "Analyze Amazon Rainforest deforestation water cycle shifts and temperature anomalies in Brazil",
            "Examine Arctic amplification, solar radiation, and polar warming in Svalbard"
        ]
    )

with col_prompt:
    user_query = st.text_area(
        "Ask the AI Detective to investigate any region & environmental process on Earth:",
        value=preset_choice,
        height=85
    )

if st.button("🚀 Execute Step 1 ➔ Step 3.5 Pipeline & Update ESP32 Station", type="primary", use_container_width=True):
    with st.spinner("Step 1: Routing via AI Agent ➔ Step 2: Fetching 4 NASA Variables ➔ Step 3.2: Filtering Seasonal Zig-Zag ➔ Step 3.3: Running 4-Layer Intelligence..."):
        try:
            meta = parse_user_prompt_with_ai(user_query, gemini_key)
            if enable_override and len(man_drivers) >= 1:
                meta["lat"] = float(man_lat)
                meta["lon"] = float(man_lon)
                meta["target_var"] = man_target
                meta["driver_vars"] = man_drivers[:3]
                meta["start_year"] = int(man_years[0])
                meta["end_year"] = int(man_years[1])
                meta["region_name"] = f"Custom Coordinates ({man_lat:.2f}°, {man_lon:.2f}°)"
                meta["router_source"] = "Manual Sidebar Override"

            all_requested_vars = [meta["target_var"]] + meta["driver_vars"]
            df_raw = fetch_nasa_multivariable_data(
                meta["lat"], meta["lon"], meta["start_year"], meta["end_year"], all_requested_vars
            )
            intel_res = run_multivariable_intelligence_engine(
                df_raw, meta["target_var"], meta["driver_vars"]
            )

            case_bundle = {"meta": meta, "intel": intel_res}
            udp_status, udp_str = send_multivariable_udp_to_esp32(esp_ip, esp_port, case_bundle, firebase_url)
            case_bundle["udp_status"] = udp_status
            case_bundle["udp_str"] = udp_str

            st.session_state["active_case"] = case_bundle
            st.session_state["watchlist"].append(case_bundle)
            st.rerun()
        except Exception as err:
            st.error(f"Pipeline Execution Error: {err}")


# ==============================================================================
# RENDER MODULES 2 TO 5 WHEN AN ACTIVE INVESTIGATION EXISTS
# ==============================================================================
if st.session_state["active_case"] is not None:
    active = st.session_state["active_case"]
    meta = active["meta"]
    intel = active["intel"]
    t_var = intel["target_var"]
    d_vars = intel["driver_vars"]
    all_vars = intel["all_vars"]
    v_stats = intel["var_stats"]
    ann_df = intel["annual_df"]

    # Show JSON payload generated by Step 1 + 4-Variable Significance Cards
    with st.expander(f"🔍 View Step 1 AI Router JSON Output ({meta['router_source']})", expanded=False):
        st.json(meta)

    st.markdown(f"#### 🧪 Active 4-Variable Cluster for **{meta['region_name']}** ({meta['lat']:.2f}°, {meta['lon']:.2f}°)")
    st.caption(f"**Physical Hypothesis:** {meta['hypothesis']}")

    card_cols = st.columns(len(all_vars))
    for i, v in enumerate(all_vars):
        st_v = v_stats[v]
        role_tag = "🎯 TARGET" if v == t_var else f"⚙️ DRIVER #{i}"
        sig_label = "SIGNIFICANT (p < 0.05)" if st_v["significant"] else "Not Significant (Noise)"
        with card_cols[i]:
            with st.container(border=True):
                st.markdown(f"**{role_tag}: `{v}`**")
                st.caption(f"{VARIABLES[v]['name']} ({VARIABLES[v]['mission']})")
                st.metric(
                    label="Deseasonalized Trend / Decade",
                    value=f"{st_v['slope_dec']:+.3f} {VARIABLES[v]['unit']}",
                    delta=f"±{st_v['ci_95_dec']:.3f} (95% CI)"
                )
                st.markdown(
                    f"**Mann-Kendall $p$:** `{st_v['mk_p']:.4f}`  \n"
                    f"**t-test $p$:** `{st_v['ttest_p']:.4f}`  \n"
                    f"**Verdict:** {'🔴 ' if st_v['significant'] else '🟢 '}**{sig_label}**"
                )

    st.divider()

    # ==========================================================================
    # MODULE 2 & MODULE 3: MULTI-VARIABLE OVERLAY, FORECAST & COUPLING MATRIX
    # ==========================================================================
    col_mod3, col_mod2 = st.columns([3, 2])

    with col_mod3:
        st.subheader("📈 Module 3 — Deseasonalized Trend, Tipping Point & Multivariate AI Forecast")
        view_mode = st.radio(
            "Select Diagram Mode:",
            ["Target Variable + Tipping Point + 10-Yr Multivariate AI Forecast", "4-Variable Standardized Z-Score Overlay (After Zig-Zag Removal)"],
            horizontal=True
        )

        if view_mode.startswith("Target"):
            fig_main = go.Figure()
            y_obs = ann_df[t_var].values
            x_yr = ann_df["year"].values

            # 1. Observed Annual Series
            fig_main.add_trace(go.Scatter(
                x=x_yr, y=y_obs, mode="lines+markers",
                name=f"Observed {t_var} ({VARIABLES[t_var]['unit']})",
                line=dict(color="#38bdf8", width=2.5)
            ))

            # 2. Piecewise Pre- vs Post-Tipping Point Trendlines
            tip_yr = intel["tipping_year"]
            tip_idx = int(np.where(x_yr == tip_yr)[0][0])
            lin_pre = stats.linregress(x_yr[:tip_idx+1], y_obs[:tip_idx+1])
            lin_post = stats.linregress(x_yr[tip_idx:], y_obs[tip_idx:])

            fig_main.add_trace(go.Scatter(
                x=x_yr[:tip_idx+1],
                y=lin_pre.slope * x_yr[:tip_idx+1] + lin_pre.intercept,
                mode="lines", name=f"Pre-{tip_yr} Regime ({intel['slope_pre']:+.3f}/dec)",
                line=dict(color="#22c55e", width=3, dash="dash")
            ))
            fig_main.add_trace(go.Scatter(
                x=x_yr[tip_idx:],
                y=lin_post.slope * x_yr[tip_idx:] + lin_post.intercept,
                mode="lines", name=f"Post-{tip_yr} Regime ({intel['slope_post']:+.3f}/dec)",
                line=dict(color="#ef4444", width=3.5, dash="dash")
            ))

            # 3. Extreme Anomaly Stars
            fig_main.add_trace(go.Scatter(
                x=v_stats[t_var]["extreme_years"],
                y=v_stats[t_var]["extreme_vals"],
                mode="markers", name="Isolation Forest Extreme Anomalies",
                marker=dict(color="#f43f5e", size=12, symbol="star")
            ))

            # 4. 10-Year Multivariate AI Forecast
            fig_main.add_trace(go.Scatter(
                x=intel["future_years"], y=intel["forecast_vals"],
                mode="lines+markers", name=f"10-Yr Multivariate AI Forecast ({intel['ai_engine_label']})",
                line=dict(color="#f59e0b", width=3, dash="dot")
            ))

            # Vertical Tipping-Point Line
            fig_main.add_vline(
                x=tip_yr, line_width=2, line_dash="dot", line_color="#fb923c",
                annotation_text=f"Tipping Point: {tip_yr}", annotation_position="top left"
            )
            fig_main.update_layout(height=430, template="plotly_dark", xaxis_title="Year", yaxis_title=VARIABLES[t_var]["unit"])
            st.plotly_chart(fig_main, use_container_width=True)

        else:
            fig_z = go.Figure()
            palette = ["#38bdf8", "#f43f5e", "#a855f7", "#eab308"]
            for idx, v in enumerate(all_vars):
                z_series = stats.zscore(ann_df[f"{v}_anom"].values)
                fig_z.add_trace(go.Scatter(
                    x=ann_df["year"], y=z_series, mode="lines",
                    name=f"{v} ({'Target' if v == t_var else 'Driver'})",
                    line=dict(color=palette[idx % len(palette)], width=3 if v == t_var else 1.8)
                ))
            fig_z.add_vline(x=intel["tipping_year"], line_width=2, line_dash="dot", line_color="#fb923c")
            fig_z.update_layout(
                height=430, template="plotly_dark",
                xaxis_title="Year", yaxis_title="Deseasonalized Z-Score Anomaly (σ)"
            )
            st.plotly_chart(fig_z, use_container_width=True)

    with col_mod2:
        st.subheader("🔗 Module 2 — Driver Coupling, Time-Lag & Regime Shift")

        # Cross-Correlation Heatmap
        fig_heat = px.imshow(
            np.round(intel["corr_matrix"].values, 2),
            x=all_vars, y=all_vars,
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=True, aspect="auto"
        )
        fig_heat.update_layout(height=230, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Tipping Point & Time-Lag Intelligence Table
        st.markdown(
            f"**⚡ CUSUM Tipping-Point Year:** `{intel['tipping_year']}` "
            f"(Slope changed from `{intel['slope_pre']:+.3f}` to `{intel['slope_post']:+.3f}`/decade — "
            f"**{intel['accel_ratio']:.1f}x acceleration**)"
        )

        lag_rows = []
        for d_v in d_vars:
            dc = intel["driver_couplings"][d_v]
            lag_rows.append({
                "Driver": d_v,
                "Annual Corr (r)": f"{dc['r_annual']:+.2f}",
                "Optimal Lag": f"{dc['best_lag_months']} mo",
                "Lagged r": f"{dc['lag_r']:+.2f}",
                "Coupling Type": "⚪ INVERSE PARADOX" if dc["paradox_coupled"] else "Co-Trending"
            })
        st.dataframe(pd.DataFrame(lag_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ==========================================================================
    # MODULE 4 & MODULE 5: "WHAT-IF" AI SIMULATOR + ESP32 5-LED DIGITAL TWIN
    # ==========================================================================
    col_mod4, col_mod5 = st.columns([3, 2])

    with col_mod4:
        st.subheader("🎛️ Module 4 — Interactive Multivariate 'What-If' Scenario Simulator")
        st.caption(
            f"Adjust the physical Driver Variables below to simulate how `{t_var}` ({VARIABLES[t_var]['name']}) "
            "responds according to the trained Multivariate Random Forest model:"
        )

        sim_cols = st.columns(len(d_vars))
        baseline_vec = ann_df[all_vars].values[-1].copy()
        simulated_vec = baseline_vec.copy()

        for i, d_v in enumerate(d_vars):
            d_mean = v_stats[d_v]["mean_raw"]
            d_std = v_stats[d_v]["std_raw"]
            with sim_cols[i]:
                delta_pct = st.slider(
                    f"Δ {d_v} ({VARIABLES[d_v]['unit']})",
                    min_value=-3.0, max_value=3.0, value=0.0, step=0.25,
                    help=f"Shift {d_v} by standard deviations (1σ = {d_std:.3f} {VARIABLES[d_v]['unit']})"
                )
                simulated_vec[i + 1] = baseline_vec[i + 1] + (delta_pct * d_std)

        base_pred = float(intel["rf_model"].predict([baseline_vec])[0])
        # Combine Random Forest response with linear cross-correlation sensitivity for smooth physical feedback
        rf_sim_pred = float(intel["rf_model"].predict([simulated_vec])[0])
        linear_delta = 0.0
        for i, d_v in enumerate(d_vars):
            std_shift = (simulated_vec[i + 1] - baseline_vec[i + 1]) / v_stats[d_v]["std_raw"]
            r_val = intel["driver_couplings"][d_v]["r_annual"]
            linear_delta += std_shift * r_val * v_stats[t_var]["std_raw"] * 0.45

        final_sim_pred = 0.5 * rf_sim_pred + 0.5 * (base_pred + linear_delta)
        sim_diff = final_sim_pred - base_pred

        m_sim1, m_sim2, m_sim3 = st.columns(3)
        m_sim1.metric(f"Current Baseline {t_var}", f"{baseline_vec[0]:.3f} {VARIABLES[t_var]['unit']}")
        m_sim2.metric(
            f"Simulated {t_var} Response",
            f"{final_sim_pred:.3f} {VARIABLES[t_var]['unit']}",
            f"{sim_diff:+.3f} {VARIABLES[t_var]['unit']}"
        )
        top_imp_var = max(intel["importances"], key=intel["importances"].get)
        m_sim3.metric("Top AI Feature Importance", f"{top_imp_var}", f"{intel['importances'][top_imp_var]*100:.1f}% weight")

    with col_mod5:
        st.subheader("📟 Module 5 — ESP32 5-LED Hardware Station & Map")
        st.map(pd.DataFrame({"lat": [meta["lat"]], "lon": [meta["lon"]]}), zoom=4)

        bitmask = intel["led_bitmask"]
        b_on, o_on, r_on, w_on, g_on = [c == "1" for c in bitmask]

        st.markdown("##### 💡 Live 5-LED Bitmask State (`" + bitmask + "`)")
        l1, l2, l3, l4, l5 = st.columns(5)
        l1.markdown(f"🔵 **BLUE**\n\n`{'ON' if b_on else 'OFF'}`\n\n*Hydro Sig*")
        l2.markdown(f"🟠 **ORANGE**\n\n`{'ON' if o_on else 'OFF'}`\n\n*Tip Shift*")
        l3.markdown(f"🔴 **RED**\n\n`{'ON' if r_on else 'OFF'}`\n\n*p < 0.01*")
        l4.markdown(f"⚪ **WHITE**\n\n`{'ON' if w_on else 'OFF'}`\n\n*Paradox*")
        l5.markdown(f"🟢 **5th LED**\n\n`{'ON' if g_on else 'OFF'}`\n\n*Stable*")

        st.caption(active.get("udp_status", ""))