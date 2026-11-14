import io
import re
import json
import socket
from html import escape
from concord_theme import (inject_theme, nav_bar, section_header, stat_tile, status_chip, arc_gauge_svg, sparkline_svg, led_tile, footer, hero_card, icon, safe_prose, html_table, info_note, with_info, variable_color, variable_icon, status_icon, render_map, render_hero, label_chart, chart_key)
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from concord_utils import (
    validate_metadata, normalize_monthly_data, mann_kendall_details,
    safe_linregress, safe_correlation, standardize, estimate_additional_years,
    geocode_city, prompt_years, clamp_physical,
)

# --- OPTIONAL INTEL OPENVINO RUNTIME CHECK ---
try:
    import openvino as ov
    import openvino.opset13 as ops
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# ==============================================================================
# PAGE CONFIGURATION & SESSION STATE INITIALIZATION
# ==============================================================================
st.set_page_config(
    page_title="CONCORD | Earth System Trend Detective",
    layout="wide",
    initial_sidebar_state="auto"
)

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = []

if "active_case" not in st.session_state:
    st.session_state["active_case"] = None

# ==============================================================================
# CUSTOM CSS: PITCH BLACK BASE + PURE WHITE ACCENT + NYT SERIF + BORDERLESS
# ==============================================================================
inject_theme()

# ==============================================================================
# EARTH SCIENCE VARIABLES CATALOG (SCIENTIFIC + ACCESSIBLE STUDENT DEFINITIONS)
# ==============================================================================
VARIABLES = {
    "T2M": {
        "name": "Surface Air Temperature (2m)",
        "unit": "deg C",
        "mission": "NASA POWER / MERRA-2",
        "category": "Thermal Regime",
        "sci_desc": "Dry-bulb air temperature at 2 meters above the surface displacement height.",
        "plain_desc": "How hot or cold the air feels near the ground where people, forests, and crops live."
    },
    "PRECTOTCORR": {
        "name": "Bias-Corrected Precipitation",
        "unit": "mm/day",
        "mission": "NASA POWER / MERRA-2",
        "category": "Hydrological Flux",
        "sci_desc": "Satellite- and gauge-corrected total liquid and solid precipitation rate.",
        "plain_desc": "The average daily depth of rain and snow falling on a region."
    },
    "GWETTOP": {
        "name": "Surface Soil Wetness (0-5cm)",
        "unit": "fraction",
        "mission": "NASA POWER / MERRA-2",
        "category": "Land Hydrology",
        "sci_desc": "Dimensionless relative surface soil wetness in the model's top soil layer (0 to 1); not volumetric water content.",
        "plain_desc": "How dry or soaked the top layer of ground soil is, from 0 (bone dry) to 1 (completely saturated)."
    },
    "QV2M": {
        "name": "Specific Humidity (2m)",
        "unit": "g/kg",
        "mission": "NASA POWER / MERRA-2",
        "category": "Atmospheric Moisture",
        "sci_desc": "Mass mixing ratio of water vapor per kilogram of moist air at 2 meters.",
        "plain_desc": "The exact weight of invisible water vapor held inside every kilogram of air."
    },
    "ALLSKY_SFC_SW_DWN": {
        "name": "Surface Downward Shortwave Irradiance",
        "unit": "W/m2",
        "mission": "NASA POWER / GEWEX SRB / CERES",
        "category": "Radiative Forcing",
        "sci_desc": "Total direct and diffuse solar irradiance reaching a horizontal surface under all-sky conditions.",
        "plain_desc": "The intensity of sunlight energy hitting the ground after passing through clouds, smog, and dust."
    },
    "WS2M": {
        "name": "Near-Surface Wind Speed (2m)",
        "unit": "m/s",
        "mission": "NASA MERRA-2",
        "category": "Atmospheric Dynamics",
        "sci_desc": "Horizontal wind velocity vector magnitude at 2 meters above the surface.",
        "plain_desc": "How fast the wind blows near the surface, driving coastal flushing, dust transport, and evaporation."
    }
}

# ==============================================================================
# BIOME & REGIONAL ARCHETYPE CATALOG (COVERS ALL NASA CHALLENGE ARCHETYPES)
# ==============================================================================
BIOME_QUESTIONNAIRE_CATALOG = {
    "Desert / Arid Basin": {
        "questions": {
            "Are rising desert temperatures and intense sunlight drying out oasis topsoil?": {
                "region_name": "Ouarzazate & Draa Arid Basin, Pre-Saharan Morocco",
                "lat": 30.92, "lon": -6.90,
                "target_var": "GWETTOP", "driver_vars": ["T2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"],
                "biome_type": "Desert / Hyper-Arid Basin",
                "hypothesis": "High downward shortwave irradiance and rising air temperatures accelerate potential evapotranspiration, desiccating arid topsoil.",
                "literature_context": "Research on North African arid margins demonstrates that low cloud cover and high surface albedo create radiative-thermal feedbacks where warming accelerates oasis soil moisture loss even when sporadic rainfall occurs."
            },
            "Are desert wind speeds and atmospheric moisture shifting across the Sahara margin?": {
                "region_name": "Errachidia / Tafilalet Desert Margin, Morocco",
                "lat": 31.93, "lon": -4.42,
                "target_var": "WS2M", "driver_vars": ["T2M", "ALLSKY_SFC_SW_DWN", "QV2M"],
                "biome_type": "Desert / Hyper-Arid Basin",
                "hypothesis": "Surface thermal gradients alter near-surface wind speeds and dust-mobilizing boundary layer dynamics.",
                "literature_context": "Climatological studies of Saharan boundary layers link steepening land-ocean thermal gradients to shifts in surface wind stress and Harmattan dust transport."
            }
        }
    },
    "Tropical Rain Forest": {
        "questions": {
            "Is canopy warming disrupting the Amazon's self-generated rainfall cycle?": {
                "region_name": "Central Amazon Basin (Manaus Sector), Brazil",
                "lat": -3.46, "lon": -62.21,
                "target_var": "PRECTOTCORR", "driver_vars": ["T2M", "QV2M", "ALLSKY_SFC_SW_DWN"],
                "biome_type": "Tropical Rain Forest",
                "hypothesis": "Canopy warming and solar radiation shifts alter transpired specific humidity and regional convective rainfall recycling.",
                "literature_context": "Satellite hydroclimatology papers show the Amazon recycles up to 50% of its own rainfall via canopy transpiration; rising temperatures and land-cover stress lengthen dry seasons and perturb precipitation."
            },
            "Is tropical topsoil drying out under rising temperatures in the Congo Basin?": {
                "region_name": "Congo Tropical Rain Forest Basin, Central Africa",
                "lat": 0.52, "lon": 22.40,
                "target_var": "GWETTOP", "driver_vars": ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
                "biome_type": "Tropical Rain Forest",
                "hypothesis": "Increasing surface air temperature and solar radiation anomalies reduce root-zone and surface soil wetness across equatorial forests.",
                "literature_context": "Observational studies of equatorial African forests indicate a multi-decade drying trend driven by warming Indian and Atlantic Ocean sea surface temperatures."
            }
        }
    },
    "Tidal Estuary / Coastal Zone": {
        "questions": {
            "How are freshwater rainfall runoff, wind flushing, and warming changing in a Moroccan tidal estuary?": {
                "region_name": "Sebou River Tidal Estuary (Kenitra / Mehdia Coast), Morocco",
                "lat": 34.26, "lon": -6.65,
                "target_var": "PRECTOTCORR", "driver_vars": ["T2M", "WS2M", "GWETTOP"],
                "biome_type": "Tidal Estuary / Coastal Mixing Zone",
                "hypothesis": "Declining upstream precipitation and shifting coastal winds alter estuarine freshwater discharge, salinity intrusion, and thermal stratification.",
                "literature_context": "Hydrological studies of Atlantic Moroccan estuaries show that reduced watershed precipitation combined with coastal thermal warming intensifies marine salinity intrusion upstream."
            },
            "Is surface warming accelerating across the Chesapeake Bay tidal estuary?": {
                "region_name": "Chesapeake Bay Tidal Estuary, United States",
                "lat": 37.80, "lon": -76.10,
                "target_var": "T2M", "driver_vars": ["ALLSKY_SFC_SW_DWN", "QV2M", "WS2M"],
                "biome_type": "Tidal Estuary / Coastal Mixing Zone",
                "hypothesis": "Atmospheric warming and weakened wind mixing elevate estuarine surface temperatures, reducing dissolved oxygen capacity.",
                "literature_context": "Estuarine time-series research (including SeaWiFS/MODIS coastal analyses) demonstrates that shallow tidal estuaries warm faster than the open ocean, triggering nutrient and chlorophyll regime shifts."
            }
        }
    },
    "Atmosphere Above a Large City": {
        "questions": {
            "Is the Urban Heat Island intensifying air temperatures and moisture above Grand Casablanca?": {
                "region_name": "Grand Casablanca Megacity Urban Core, Morocco",
                "lat": 33.57, "lon": -7.59,
                "target_var": "T2M", "driver_vars": ["ALLSKY_SFC_SW_DWN", "QV2M", "WS2M"],
                "biome_type": "Atmosphere Above a Large City",
                "hypothesis": "Urban thermal mass, radiative trapping, and altered coastal wind ventilation amplify near-surface air temperature trends.",
                "literature_context": "Urban climatology research shows dense metropolitan corridors absorb shortwave solar irradiance in concrete/asphalt and release longwave heat overnight, steepening decadal warming slopes."
            },
            "How are solar irradiance (urban haze/aerosols) and heat changing above Cairo Megacity?": {
                "region_name": "Greater Cairo Metropolitan Atmosphere, Egypt",
                "lat": 30.04, "lon": 31.23,
                "target_var": "ALLSKY_SFC_SW_DWN", "driver_vars": ["T2M", "WS2M", "QV2M"],
                "biome_type": "Atmosphere Above a Large City",
                "hypothesis": "Boundary-layer wind stagnation, urban aerosol loading, and humidity modulate surface downward solar irradiance and heat retention.",
                "literature_context": "Atmospheric remote sensing over megacities tracks how urban boundary layers trap particulates and heat when ventilation winds weaken."
            }
        }
    },
    "Agricultural Plain / Breadbasket": {
        "questions": {
            "Is agricultural topsoil moisture in Berrechid & Chaouia crossing a permanent drying threshold?": {
                "region_name": "Berrechid & Chaouia Agricultural Plain, Morocco",
                "lat": 33.26, "lon": -7.58,
                "target_var": "GWETTOP", "driver_vars": ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
                "biome_type": "Semi-Arid Agricultural Plain",
                "hypothesis": "Positive thermal anomalies and solar radiative forcing accelerate evapotranspiration, depleting topsoil moisture faster than rainfall recharge.",
                "literature_context": "Agro-hydrological monitoring across Moroccan cereal plains highlights a structural shift toward multi-year meteorological and soil moisture droughts since the mid-2010s."
            }
        }
    },
    "Polar / Cryosphere": {
        "questions": {
            "How fast is Arctic Amplification warming Svalbard compared to solar and moisture shifts?": {
                "region_name": "Svalbard Archipelago, High Arctic",
                "lat": 78.22, "lon": 15.63,
                "target_var": "T2M", "driver_vars": ["ALLSKY_SFC_SW_DWN", "QV2M", "PRECTOTCORR"],
                "biome_type": "Polar / Cryosphere",
                "hypothesis": "Sea-ice retreat lowers surface albedo, amplifying surface air warming and increasing polar atmospheric water vapor.",
                "literature_context": "Polar reanalysis datasets (MERRA-2) confirm the High Arctic is warming three to four times faster than the global average due to ice-albedo and water-vapor feedbacks."
            }
        }
    }
}

# ==============================================================================
# STEP 1: AI AGENT ROUTER (USER PROMPT -> MULTI-VARIABLE JSON PAYLOAD)
# ==============================================================================
def parse_user_prompt_with_ai(user_prompt, gemini_api_key="", user_location=""):
    """
    Routes a user's natural language query via Gemini 2.5 Flash (if a valid key is supplied)
    or falls back to the built-in domain knowledge engine covering all NASA biome archetypes.
    """
    valid_vars = list(VARIABLES.keys())
    start_year, end_year = prompt_years(user_prompt)
    geo = geocode_city(user_location) if user_location.strip() else None

    if gemini_api_key.strip():
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={gemini_api_key.strip()}"
            )
            system_instruction = (
                "You are a NASA Earth System Science Router. Given the user's investigation prompt, "
                "return ONLY a valid JSON object (no markdown fences) with these exact keys:\n"
                "{\n"
                '  "region_name": "String name of region",\n'
                '  "biome_type": "Desert / Rain Forest / Tidal Estuary / Large City Atmosphere / Agricultural / Polar",\n'
                '  "lat": float (-90 to 90),\n'
                '  "lon": float (-180 to 180),\n'
                '  "start_year": int (between 1981 and 2018),\n'
                '  "end_year": int (between 2020 and 2025),\n'
                f'  "target_var": one of {valid_vars},\n'
                f'  "driver_vars": list of exactly 3 distinct variables from {valid_vars} (excluding target_var),\n'
                '  "hypothesis": "1-sentence physical mechanism linking the drivers to the target variable.",\n'
                '  "literature_context": "1-sentence scientific background drawing on regional characteristics or published climate science."\n'
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
                    drivers = list(dict.fromkeys(v for v in parsed.get("driver_vars", []) if v in valid_vars and v != parsed["target_var"]))
                    for fallback_v in valid_vars:
                        if len(drivers) < 3 and fallback_v != parsed["target_var"] and fallback_v not in drivers:
                            drivers.append(fallback_v)
                    parsed["driver_vars"] = drivers[:3]
                    parsed.setdefault("biome_type", "Regional Earth System")
                    parsed.setdefault("literature_context", "Selected variables capture coupled thermal, radiative, and hydrological feedbacks.")
                    parsed["router_source"] = "LIVE GEMINI 2.5 FLASH API"
                    if geo:
                        parsed.update(geo)
                    if re.search(r"\b(?:19|20)\d{2}\b", user_prompt):
                        parsed["start_year"], parsed["end_year"] = start_year, end_year
                    if not 1981 <= int(parsed["start_year"]) <= int(parsed["end_year"]) <= 2025:
                        raise ValueError("Unsupported date range from router.")
                    return validate_metadata(parsed, VARIABLES)
        except Exception:
            pass

    p_low = user_prompt.lower()
    if "estuary" in p_low or "tidal" in p_low or "sebou" in p_low or "kenitra" in p_low or "chesapeake" in p_low:
        base = dict(BIOME_QUESTIONNAIRE_CATALOG["Tidal Estuary / Coastal Zone"]["questions"][
            "How are freshwater rainfall runoff, wind flushing, and warming changing in a Moroccan tidal estuary?"
        ])
    elif "city" in p_low or "urban" in p_low or "megacity" in p_low or "cairo" in p_low or "heat island" in p_low:
        base = dict(BIOME_QUESTIONNAIRE_CATALOG["Atmosphere Above a Large City"]["questions"][
            "Is the Urban Heat Island intensifying air temperatures and moisture above Grand Casablanca?"
        ])
    elif "amazon" in p_low or "rainforest" in p_low or "rain forest" in p_low or "brazil" in p_low:
        base = dict(BIOME_QUESTIONNAIRE_CATALOG["Tropical Rain Forest"]["questions"][
            "Is canopy warming disrupting the Amazon's self-generated rainfall cycle?"
        ])
    elif "arctic" in p_low or "polar" in p_low or "ice" in p_low or "svalbard" in p_low:
        base = dict(BIOME_QUESTIONNAIRE_CATALOG["Polar / Cryosphere"]["questions"][
            "How fast is Arctic Amplification warming Svalbard compared to solar and moisture shifts?"
        ])
    elif "sahara" in p_low or "desert" in p_low or "ouarzazate" in p_low or "arid" in p_low:
        base = dict(BIOME_QUESTIONNAIRE_CATALOG["Desert / Arid Basin"]["questions"][
            "Are rising desert temperatures and intense sunlight drying out oasis topsoil?"
        ])
    elif "tangier" in p_low or "north morocco" in p_low or "rif" in p_low:
        base = {
            "region_name": "Tangier Peninsula, Northern Morocco",
            "biome_type": "Mediterranean Coastal Strait",
            "lat": 35.76, "lon": -5.83,
            "target_var": "PRECTOTCORR", "driver_vars": ["T2M", "QV2M", "WS2M"],
            "hypothesis": "Atlantic-Mediterranean moisture advection and thermal shifts alter coastal precipitation regimes.",
            "literature_context": "Mediterranean basin studies document shifting storm tracks and winter precipitation variability across the Strait of Gibraltar."
        }
    else:
        base = dict(BIOME_QUESTIONNAIRE_CATALOG["Agricultural Plain / Breadbasket"]["questions"][
            "Is agricultural topsoil moisture in Berrechid & Chaouia crossing a permanent drying threshold?"
        ])

    # Match every named catalog region before applying generic biome defaults.
    region_aliases = {
        "chesapeake": "Chesapeake", "cairo": "Cairo", "congo": "Congo",
        "berrechid": "Berrechid", "chaouia": "Berrechid",
        "errachidia": "Errachidia", "casablanca": "Grand Casablanca",
        "ouarzazate": "Ouarzazate", "draa": "Ouarzazate",
        "sebou": "Sebou", "kenitra": "Sebou", "amazon": "Amazon", "svalbard": "Svalbard",
    }
    matched_region = False
    for alias, name in region_aliases.items():
        if alias in p_low:
            base = next(dict(p) for b in BIOME_QUESTIONNAIRE_CATALOG.values()
                        for p in b["questions"].values() if name in p["region_name"])
            matched_region = True
            break
    if geo is None and not matched_region and not any(k in p_low for k in ("tangier", "north morocco", "rif")):
        match = re.search(r"\b(?:in|near|at)\s+(.+?)(?:\s+(?:from|between|since|during)\b|$)", user_prompt, re.I)
        generic_biome = any(k in p_low for k in ("desert", "rainforest", "rain forest", "arctic", "polar", "tidal", "estuary", "agricultur"))
        if match or not generic_biome:
            city = match.group(1).strip(" .") if match else re.sub(r"\b(?:19|20)\d{2}\b.*", "", user_prompt).strip()
            geo = geocode_city(city)
    if geo:
        base.update(geo)
        base["biome_type"] = "User-selected regional grid point"
    original_target = base["target_var"]
    for pattern, target in [
        (r"\b(?:soil|drought|drying)\b", "GWETTOP"),
        (r"\b(?:temperature|warming|heat)\b", "T2M"),
        (r"\b(?:rain|rainfall|precipitation|flood)\b", "PRECTOTCORR"),
        (r"\b(?:wind|winds)\b", "WS2M"),
        (r"\b(?:humidity|vapor)\b", "QV2M"),
        (r"\b(?:solar|irradiance|radiation|sunlight)\b", "ALLSKY_SFC_SW_DWN"),
    ]:
        if re.search(pattern, p_low):
            base["target_var"] = target
            break
    base["driver_vars"] = [v for v in dict.fromkeys(base["driver_vars"] + valid_vars) if v != base["target_var"]][:3]
    if geo or original_target != base["target_var"]:
        base["hypothesis"] = "Test whether thermal, radiative, and hydrological anomalies covary at the selected grid point."
        base["literature_context"] = "An exploratory regional hypothesis; correlation alone cannot establish a physical cause."
    base["start_year"] = start_year
    base["end_year"] = end_year
    base["router_source"] = "BUILT-IN DOMAIN KNOWLEDGE ENGINE"
    if geo:
        base["router_source"] += " + OPEN-METEO GEOCODING"
    return validate_metadata(base, VARIABLES)


# ==============================================================================
# STEP 2 & 3.1: NASA API FETCHER + DATA TABLE STRUCTURING
# ==============================================================================
@st.cache_data(ttl=3600, max_entries=32)
def fetch_nasa_multivariable_data(lat, lon, start_year, end_year, var_list):
    """
    Fetches monthly historical records from NASA POWER (meteorology and solar archives)
    for all requested variables and structures a clean Pandas DataFrame.
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
        if ym.endswith("13"):
            continue
        row = {
            "date": pd.Timestamp(year=int(ym[:4]), month=int(ym[4:6]), day=1),
            "year": int(ym[:4]),
            "month": int(ym[4:6])
        }
        valid_row = True
        for v in unique_vars:
            val = float(param_data[v].get(ym, -999.0))
            if not np.isfinite(val) or val <= -990.0:
                valid_row = False
                break
            row[v] = val
        if valid_row:
            records.append(row)

    if not records:
        raise ValueError("NASA returned no valid overlapping monthly data for these variables and coordinates.")
    result = normalize_monthly_data(pd.DataFrame(records), unique_vars)
    result.attrs["source_units"] = {v: resp.json().get("parameters", {}).get(v, {}).get("units", "") for v in unique_vars}
    return result


# ==============================================================================
# STEP 3.2: DESEASONALIZATION (REMOVING THE SEASONAL ZIG-ZAG CYCLE)
# ==============================================================================
def deseasonalize_dataframe(df, var_list, reference_df=None):
    """
    Subtracts the multi-decade monthly climatological mean from each observation
    to isolate non-seasonal anomalies and expose hidden trends.
    """
    df_clean = df.copy()
    for v in var_list:
        reference = df_clean if reference_df is None else reference_df
        climatology = reference.groupby("month")[v].mean()
        monthly_climatology = df_clean["month"].map(climatology)
        if monthly_climatology.isna().any():
            raise ValueError("Climatology needs valid observations for every calendar month.")
        df_clean[f"{v}_clim"] = monthly_climatology
        df_clean[f"{v}_anom"] = df_clean[v] - monthly_climatology
    return df_clean


# ==============================================================================
# STEP 3.3: 4-LAYER MULTI-VARIABLE INTELLIGENCE ENGINE + RECORD SUFFICIENCY MATH
# ==============================================================================
def mann_kendall_math(y, years=None):
    """Computes non-parametric Mann-Kendall S, Z-statistic, two-tailed p-value, and Sen's slope."""
    _, z, p_val, sens_slope = mann_kendall_details(y, years)
    return z, p_val, sens_slope


def build_plain_english_variable_summary(var_code, st_dict, n_years):
    """
    Generates an accessible explanation of the statistical findings for students and the public,
    including how many additional years of observation are needed if a trend is not yet significant.
    """
    v_name = escape(VARIABLES.get(var_code, {}).get("name", var_code))
    unit = escape(VARIABLES.get(var_code, {}).get("unit", "units"))
    direction_word = "rising" if st_dict["slope_dec"] > 0 else "falling" if st_dict["slope_dec"] < 0 else "flat"
    abs_rate = abs(st_dict["slope_dec"])

    if st_dict["significant"]:
        return (
            f"<strong>Plain-Language Takeaway:</strong> {v_name} is {direction_word} by {abs_rate:.3f} {unit} every 10 years. "
            f"Because both the Mann-Kendall p-value ({st_dict['mk_p']:.4f}) and Student's t-test p-value ({st_dict['ttest_p']:.4f}) "
            f"are below 0.05, both tests support a trend over these {n_years} complete years. "
            "A p-value measures compatibility with a no-trend model; it is not the probability that the trend is false."
        )

    extra_yrs = st_dict["additional_years_needed"]
    estimate = (
        "No finite observation-length estimate is available for a flat or unresolved slope."
        if extra_yrs is None else
        "OLS already meets 0.05, but Mann-Kendall does not; the tests disagree."
        if extra_yrs == 0 else
        f"The OLS planning approximation suggests about {extra_yrs} additional complete years if slope and variability persist."
    )
    status = "Emerging" if st_dict["emerging"] else "Unconfirmed"
    return (f"<strong>{status} trend:</strong> {v_name} is {direction_word}, with a fitted change of "
            f"{st_dict['slope_dec']:+.3f} {unit}/decade. OLS p={st_dict['ttest_p']:.4f}; "
            f"Mann-Kendall p={st_dict['mk_p']:.4f}. {estimate} "
            "This is not a guarantee of future significance or evidence of no change.")


def observation_window_label(stat):
    extra = stat["additional_years_needed"]
    if stat["significant"]:
        return "Both tests meet p < 0.05"
    if extra is None:
        return "OLS window estimate unavailable (flat slope)"
    if extra == 0:
        return "OLS significant; Mann-Kendall unconfirmed"
    return f"OLS estimate: +{extra} complete years (conditional)"


def run_multivariable_intelligence_engine(df_monthly, target_var, driver_vars, reference_df=None):
    """
    Executes the 4 analytical layers after Step 3.2 deseasonalization:
      Layer 1: Parametric OLS t-test, 95% CI Fan, Mann-Kendall, Sen's slope & Record-Length Sufficiency.
      Layer 2: Driver-Response Cross-Correlation, Time-Lag (0-3 mos), & Paradox Detection.
      Layer 3: CUSUM Tipping-Point / Regime Shift Year & Acceleration Ratio.
      Layer 4: Multivariate AI Forecast (Random Forest + OpenVINO IR Graph) & Feature Importance.
    """
    all_vars = [target_var] + driver_vars
    if len(driver_vars) != 3 or len(set(all_vars)) != 4:
        raise ValueError("The analysis requires one target and three distinct drivers.")
    df_monthly = normalize_monthly_data(df_monthly, all_vars)
    counts = df_monthly.groupby("year")["month"].nunique()
    complete_years = counts[counts == 12].index
    excluded_years = counts[counts != 12].index.tolist()
    if len(complete_years) < 6:
        raise ValueError("At least six complete years (12 valid months each) are required for the four-layer analysis.")
    df_deseason = deseasonalize_dataframe(df_monthly, all_vars, reference_df)

    agg_dict = {v: "mean" for v in all_vars}
    for v in all_vars:
        agg_dict[f"{v}_anom"] = "mean"
    annual_df = df_deseason.groupby("year").agg(agg_dict).reset_index()
    annual_df = annual_df[annual_df.year.isin(complete_years)].reset_index(drop=True)
    years = annual_df["year"].values
    n_years = len(years)

    var_stats = {}
    for v in all_vars:
        y_anom = annual_df[f"{v}_anom"].values
        y_raw = annual_df[v].values

        # Annual OLS Regression on Deseasonalized Series & Raw Series
        lin = safe_linregress(years, y_anom)
        lin_raw = safe_linregress(years, y_raw)
        dof = max(1, n_years - 2)
        t_crit = float(stats.t.ppf(0.975, dof))
        ci_95_dec = float(t_crit * lin.stderr * 10.0)
        mk_s, mk_z, mk_p, sens_slope = mann_kendall_details(y_anom, years)

        # Monthly Deseasonalized OLS Regression (for the Seasonal Cycle Decomposition chart)
        m_x = (df_deseason["year"] * 12 + df_deseason["month"]).to_numpy(dtype=float)
        m_x -= m_x[0]
        m_lin = safe_linregress(m_x, df_deseason[f"{v}_anom"].values)

        # 95% Confidence Interval Fan Curve across the historical years
        y_hat_raw = lin_raw.intercept + lin_raw.slope * years
        residuals = y_raw - y_hat_raw
        s_err = np.sqrt(np.sum(residuals ** 2) / dof)
        x_mean = np.mean(years)
        sxx = max(1e-9, np.sum((years - x_mean) ** 2))
        ci_band = t_crit * s_err * np.sqrt((1.0 / n_years) + ((years - x_mean) ** 2) / sxx)

        # Observation Window Sufficiency Estimator (Years needed for p < 0.05)
        t_stat_abs = abs(lin.slope) / max(lin.stderr, 1e-9)
        sig_flag = bool((mk_p < 0.05) and (lin.pvalue < 0.05))
        emerging_flag = bool((not sig_flag) and ((mk_p < 0.20) or (lin.pvalue < 0.20)))

        add_years_needed = estimate_additional_years(n_years, lin.slope, lin.stderr, lin.pvalue)

        iso = IsolationForest(contamination=0.12, random_state=42)
        anom_flags = iso.fit_predict(y_anom.reshape(-1, 1)) == -1

        st_entry = {
            "mean_raw": float(np.mean(y_raw)),
            "std_raw": float(np.std(y_raw)),
            "slope_yr": float(lin.slope),
            "slope_dec": float(lin.slope * 10.0),
            "intercept_raw": float(lin_raw.intercept),
            "y_hat_raw": y_hat_raw,
            "ci_upper_raw": y_hat_raw + ci_band,
            "ci_lower_raw": y_hat_raw - ci_band,
            "monthly_trend_line": m_lin.intercept + m_lin.slope * m_x,
            "sens_slope_dec": float(sens_slope * 10.0),
            "ci_95_dec": ci_95_dec,
            "ttest_p": float(lin.pvalue),
            "mk_z": float(mk_z),
            "mk_s": float(mk_s),
            "mk_p": float(mk_p),
            "r_squared": float(lin.rvalue ** 2),
            "significant": sig_flag,
            "emerging": emerging_flag,
            "additional_years_needed": add_years_needed,
            "direction": "INCREASING" if lin.slope > 0 else "DECREASING" if lin.slope < 0 else "FLAT",
            "extreme_years": years[anom_flags].tolist(),
            "extreme_vals": y_raw[anom_flags].tolist()
        }
        st_entry["plain_summary"] = build_plain_english_variable_summary(v, st_entry, n_years)
        var_stats[v] = st_entry

    # --- LAYER 2: CROSS-CORRELATION MATRIX, TIME-LAG & PARADOX DISCOVERY ---
    anom_cols = [f"{v}_anom" for v in all_vars]
    corr_matrix = df_deseason[anom_cols].corr()
    corr_matrix.index = all_vars
    corr_matrix.columns = all_vars

    driver_couplings = {}
    calendar_months = df_deseason.set_index("date").asfreq("MS")
    strongest_driver = driver_vars[0]
    max_corr_mag = -1.0

    for d_var in driver_vars:
        r_annual, p_annual = safe_correlation(annual_df[f"{target_var}_anom"], annual_df[f"{d_var}_anom"])

        best_lag = 0
        best_lag_r, _ = safe_correlation(calendar_months[f"{target_var}_anom"], calendar_months[f"{d_var}_anom"])
        for lag in [1, 2, 3]:
            r_lag, _ = safe_correlation(calendar_months[f"{target_var}_anom"], calendar_months[f"{d_var}_anom"].shift(lag))
            if np.isfinite(r_lag) and (not np.isfinite(best_lag_r) or abs(r_lag) > abs(best_lag_r)):
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
    margin = max(3, len(years) // 5)
    if len(years) > 2 * margin:
        valid_cusum = np.abs(cusum[margin:-margin])
        tip_idx = int(np.argmax(valid_cusum)) + margin
    else:
        tip_idx = len(years) // 2
    tipping_year = int(years[tip_idx])

    slope_pre = float(safe_linregress(years[:tip_idx], target_anom[:tip_idx]).slope * 10.0)
    slope_post = float(safe_linregress(years[tip_idx:], target_anom[tip_idx:]).slope * 10.0)
    accel_ratio = abs(slope_post) / max(abs(slope_pre), 1e-6)
    regime_shifted = bool(accel_ratio >= 1.4 and var_stats[target_var]["ttest_p"] < 0.10)

    # --- LAYER 4: MULTIVARIATE AI FORECASTING (RANDOM FOREST + OPENVINO) ---
    X_raw = annual_df[all_vars].values
    consecutive = np.diff(years) == 1
    X_train = X_raw[:-1][consecutive]
    y_train = annual_df[target_var].values[1:][consecutive]
    if len(y_train) < 4:
        raise ValueError("Forecasting requires at least four pairs of consecutive complete years.")

    rf_model = RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)

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

        next_target = 0.50 * rf_pred + 0.30 * lin_pred + 0.20 * (curr_vec[0] + var_stats[target_var]["slope_yr"])
        next_target = clamp_physical(target_var, next_target)
        forecast_vals.append(next_target)

        curr_vec[0] = next_target
        for idx, d_var in enumerate(driver_vars, start=1):
            curr_vec[idx] = clamp_physical(d_var, curr_vec[idx] + var_stats[d_var]["slope_yr"])

    importances = dict(zip(all_vars, rf_model.feature_importances_))

    hydro_vars = {"PRECTOTCORR", "GWETTOP", "QV2M"}
    led_blue   = int(any(var_stats[v]["significant"] for v in all_vars if v in hydro_vars))
    led_orange = int(regime_shifted)
    led_red    = int(var_stats[target_var]["mk_p"] < 0.01)
    led_white  = int(any(driver_couplings[d]["paradox_coupled"] for d in driver_vars))
    led_green  = int(not any((led_blue, led_orange, led_red, led_white)) and not var_stats[target_var]["significant"])
    led_bitmask = f"{led_blue}{led_orange}{led_red}{led_white}{led_green}"

    return {
        "df_monthly": df_deseason,
        "annual_df": annual_df,
        "n_years": n_years,
        "excluded_years": excluded_years,
        "dropped_rows": df_monthly.attrs.get("dropped_rows", 0),
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
    Sends the compact multi-variable telemetry string over Wi-Fi UDP to the ESP32:
    Format: TARGET|SLOPE_DEC|STRONGEST_DRIVER|CORR|TIP_YEAR|PRED_END|BITMASK_5LEDS
    PRED_END is ten years after the last complete observation year (2035 for 2025).
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
        if not esp_ip.strip():
            status_notes.append("UDP disabled: no destination configured.")
        elif not np.isfinite(corr_val):
            status_notes.append("UDP skipped: driver correlation is undefined for constant series.")
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(1.0)
                sock.sendto(udp_payload.encode("utf-8"), (esp_ip.strip(), int(esp_port)))
            status_notes.append(f"UDP SENT TO {esp_ip}:{esp_port}; device receipt is not acknowledged [{udp_payload}]")
    except Exception as e:
        status_notes.append(f"LOCAL UDP NOTICE: {e}")

    if firebase_url.strip():
        try:
            fb_target = firebase_url.rstrip("/") + "/concord_mission_control.json"
            response = requests.put(fb_target, json={
                "region": case_data["meta"]["region_name"],
                "udp_string": udp_payload,
                "target_var": t_var,
                "slope_decade": slope_dec,
                "strongest_driver": s_drv,
                "driver_corr": corr_val if np.isfinite(corr_val) else None,
                "forecast_end_year": int(res["future_years"][-1]),
                "forecast_end_value": pred_end,
                "tipping_year": tip_yr,
                "led_bitmask": bitmask
            }, timeout=4)
            response.raise_for_status()
            status_notes.append("FIREBASE CLOUD SYNCHRONIZED")
        except Exception as e:
            status_notes.append(f"FIREBASE SYNC ERROR: {e}")

    return " | ".join(status_notes), udp_payload


def execute_and_store_investigation(meta, esp_ip, esp_port, firebase_url, custom_df=None):
    """Runs the complete mathematical pipeline and archives the case into session state."""
    meta = validate_metadata(meta, VARIABLES)
    all_requested_vars = [meta["target_var"]] + meta["driver_vars"]
    reference_df = None
    if custom_df is None:
        reference_df = fetch_nasa_multivariable_data(
            meta["lat"], meta["lon"], min(1990, meta["start_year"]), 2025, all_requested_vars)
        df_raw = reference_df[reference_df.year.between(meta["start_year"], meta["end_year"])].copy()
        reference_df = reference_df[reference_df.year.between(1990, 2025)]
        meta["climatology"] = "Available monthly observations in the fixed 1990-2025 reference period"
    else:
        df_raw = normalize_monthly_data(custom_df, all_requested_vars)
        meta["climatology"] = "Uploaded record's calendar-month means"

    intel_res = run_multivariable_intelligence_engine(
        df_raw, meta["target_var"], meta["driver_vars"], reference_df=reference_df
    )
    case_bundle = {"meta": meta, "intel": intel_res,
                   "variables": {v: dict(VARIABLES[v]) for v in all_requested_vars}}
    udp_status, udp_str = send_multivariable_udp_to_esp32(esp_ip, esp_port, case_bundle, firebase_url)
    case_bundle["udp_status"] = udp_status
    case_bundle["udp_str"] = udp_str

    st.session_state["active_case"] = case_bundle
    st.session_state["watchlist"].append(case_bundle)


def execute_demo(meta, esp_ip, esp_port, firebase_url):
    try:
        with st.spinner("Loading the demonstration and computing its results..."):
            execute_and_store_investigation(meta, esp_ip, esp_port, firebase_url)
        st.rerun()
    except Exception as err:
        st.error(f"Demonstration unavailable: {err}")
NFC_REGION_MAP = {
    "BERRECHID":  ("Agricultural Plain / Breadbasket", "Is agricultural topsoil moisture in Berrechid & Chaouia crossing a permanent drying threshold?"),
    "CASABLANCA": ("Atmosphere Above a Large City", "Is the Urban Heat Island intensifying air temperatures and moisture above Grand Casablanca?"),
    "SEBOU":      ("Tidal Estuary / Coastal Zone", "How are freshwater rainfall runoff, wind flushing, and warming changing in a Moroccan tidal estuary?"),
    "OUARZAZATE": ("Desert / Arid Basin", "Are rising desert temperatures and intense sunlight drying out oasis topsoil?"),
    "AMAZON":     ("Tropical Rain Forest", "Is canopy warming disrupting the Amazon's self-generated rainfall cycle?"),
    "SVALBARD":   ("Polar / Cryosphere", "How fast is Arctic Amplification warming Svalbard compared to solar and moisture shifts?")
}

def sync_nfc_badge_from_esp32(esp_ip, esp_port, firebase_url, force_load=False):
    if not esp_ip.strip():
        return None
    try:
        resp = requests.get(f"http://{esp_ip.strip()}/status", timeout=1.2)
        if resp.status_code == 200:
            hw = resp.json()
            scan_id = int(hw.get("scan_count", 0))
            last_seen = st.session_state.get("last_nfc_scan_id", 0)
            if (scan_id > 0 and scan_id != last_seen) or force_load:
                st.session_state["last_nfc_scan_id"] = scan_id
                reg_key = hw.get("region_key", "BERRECHID")
                win_mode = hw.get("window_mode", "FULL_35Y")
                biome_k, quest_k = NFC_REGION_MAP.get(reg_key, NFC_REGION_MAP["BERRECHID"])
                meta = dict(BIOME_QUESTIONNAIRE_CATALOG[biome_k]["questions"][quest_k])
                meta["start_year"] = 2013 if win_mode == "SHORT_12Y" else 1990
                meta["end_year"] = 2025
                meta["router_source"] = f"ESP32 NFC BADGE ({hw.get('nfc_uid', 'TAG')} -> {reg_key})"
                execute_and_store_investigation(meta, esp_ip, esp_port, firebase_url)
                return hw
    except Exception:
        return None
    return None

# ==============================================================================
# HELPER: DARK MONOCHROME EDITORIAL PLOTLY LAYOUT (BLACK BASE, WHITE ACCENT)
# ==============================================================================
def apply_dark_monochrome_theme(fig, height=430, xaxis_title="Year", yaxis_title=""):
    """Apply the variable chart palette while preserving data and straight segments."""
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI Variable, Segoe UI, system-ui, Inter, sans-serif", color="#F5F7FF", size=12),
        colorway=["#FF8F5A", "#F5B79B", "#F0C66C", "#79B9EC"],
        xaxis=dict(title=None, showgrid=False, zeroline=False,
                   tickfont=dict(size=11, color="#AFBCD2"), automargin=True),
        yaxis=dict(title=None, showgrid=True, gridcolor="rgba(255,255,255,.06)",
                   zeroline=True, zerolinecolor="rgba(255,255,255,.15)",
                   tickfont=dict(size=11, color="#AFBCD2"), automargin=True),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#252D4B", bordercolor="#404963", font=dict(family="Segoe UI Variable, Segoe UI, system-ui, Inter, sans-serif", color="#F5F7FF", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
                    bgcolor="rgba(255,255,255,.04)", font=dict(family="Segoe UI Variable, Segoe UI, system-ui, Inter, sans-serif", size=10, color="#D0DAEC")),
        margin=dict(l=12, r=76, t=64, b=24), showlegend=False
    )
    fig.update_traces(line_shape="linear", selector=dict(type="scatter"))
    label_chart(fig, yaxis_title)
    return fig


# ==============================================================================
# SIDEBAR: CONFIGURATION, EXTERNAL CSV INGESTION & HARDWARE TELEMETRY
# ==============================================================================
with st.sidebar:
    st.html(with_info('<div class="wordmark">' + icon("settings") + 'Control Tower</div>', 'What this section does: Sets system parameters, external dataset ingestion, and ESP32 hardware routing.'))

    try:
        default_gemini = st.secrets["GEMINI_API_KEY"]
    except Exception:
        default_gemini = ""

    gemini_key = st.text_input("Gemini 2.5 Flash API Key", value=default_gemini, type="password")
    if gemini_key.strip():
        st.caption("ROUTING ENGINE: Live Gemini 2.5 Flash API key loaded.")
    else:
        st.caption("ROUTING ENGINE: Built-in Domain Knowledge Engine active (No external key required).")

    st.markdown("---")
    st.html('<div class="wordmark" style="font-size:.875rem">' + icon("settings") + 'Manual variable override</div>')
    enable_override = st.checkbox("Override Direct Prompt with Manual Settings", value=False)
    var_keys = list(VARIABLES.keys())
    man_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=33.26, format="%.2f", disabled=not enable_override)
    man_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-7.58, format="%.2f", disabled=not enable_override)
    man_target = st.selectbox("Target Variable", var_keys, index=2, disabled=not enable_override)
    man_drivers = st.multiselect(
        "Driver Variables (Select 3)",
        [k for k in var_keys if k != man_target],
        default=[k for k in ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "QV2M"] if k != man_target][:3],
        disabled=not enable_override,
        max_selections=3, key=f"manual_drivers_{man_target}"
    )
    man_years = st.slider("Observation Window", 1981, 2025, (1990, 2025), disabled=not enable_override)

    st.markdown("---")
    st.html('<div class="wordmark" style="font-size:.875rem">' + icon("link") + 'Hardware telemetry</div>')
    esp_ip = st.text_input("ESP32 Local IP Address", value="192.168.1.105")
    esp_port = st.number_input("ESP32 UDP Port", min_value=1, max_value=65535, value=4210, step=1)
    firebase_url = st.text_input("Firebase Realtime DB URL (Optional)", value="")
    if st.session_state["active_case"] is None:
        sync_nfc_badge_from_esp32(esp_ip, esp_port, firebase_url)
    if st.button("Sync NFC badge from ESP32", width="content"):
        if sync_nfc_badge_from_esp32(esp_ip, esp_port, firebase_url, force_load=True):
            st.rerun()

    st.markdown("---")
    st.html('<div class="wordmark" style="font-size:.875rem">' + icon("upload") + 'External CSV ingestion</div>')
    st.caption("Upload a time-series CSV from NASA Giovanni, AppEEARS, NOAA CDO, or CUAHSI Hydrological archives.")
    uploaded_csv = st.file_uploader("Select Time-Series CSV", type=["csv"])
    if uploaded_csv is not None:
        try:
            raw_ext = pd.read_csv(uploaded_csv)
            raw_ext.columns = [str(c).strip() for c in raw_ext.columns]
            num_cols = [c for c in raw_ext if c.lower() not in ("date", "year", "month") and not c.endswith(("_anom", "_clim"))]
            st.caption("Use monthly rows with Date or Year + Month, and at least four measurement columns. NASA codes must use the catalog units. Other codes use unspecified units.")
            if len(num_cols) < 4:
                st.error("Provide a target and three measurement columns.")
            else:
                t_ext = st.selectbox("CSV target column", num_cols)
                d_ext = st.multiselect("CSV driver columns", [c for c in num_cols if c != t_ext],
                                      default=[c for c in num_cols if c != t_ext][:3], max_selections=3,
                                      key=f"csv_drivers_{t_ext}")
            if len(num_cols) >= 4 and st.button("Analyze CSV", width="content"):
                if len(d_ext) != 3:
                    raise ValueError("Select exactly three CSV driver columns.")
                if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", c) for c in [t_ext] + d_ext):
                    raise ValueError("Measurement codes must start with a letter and contain only letters, digits or underscores (up to 40 characters).")
                raw_ext = normalize_monthly_data(raw_ext, [t_ext] + d_ext)
                for c_name in [t_ext] + d_ext:
                    if c_name not in VARIABLES:
                        VARIABLES[c_name] = {
                            "name": c_name, "unit": "units", "mission": "External CSV Archive (Giovanni/CUAHSI)",
                            "category": "External Series", "sci_desc": "User-supplied time-series variable.",
                            "plain_desc": "Custom environmental measurement imported from an external CSV file."
                        }
                ext_meta = {
                    "region_name": f"External Dataset ({uploaded_csv.name})",
                    "biome_type": "External Hydrological / Satellite Record",
                    "lat": float(man_lat), "lon": float(man_lon),
                    "start_year": int(raw_ext["year"].min()), "end_year": int(raw_ext["year"].max()),
                    "target_var": t_ext, "driver_vars": d_ext,
                    "hypothesis": "Empirical trend and cross-variable coupling test on imported CSV series.",
                    "literature_context": "Direct statistical evaluation of external hydrological or satellite station records.",
                    "router_source": "EXTERNAL CSV INGESTION ENGINE"
                }
                execute_and_store_investigation(ext_meta, esp_ip, esp_port, firebase_url, custom_df=raw_ext)
                st.rerun()
        except Exception as e:
            st.caption(f"CSV Parse Notice: {e}")

    if st.button("Clear watchlist", width="content"):
        st.session_state["watchlist"] = []
        st.rerun()


# ==============================================================================
# MASTHEAD & INTERACTIVE DATA DETECTIVE DEMONSTRATION WALKTHROUGH
# ==============================================================================
st.html(nav_bar('What this section does: Introduces a satellite time-series observatory guiding students, researchers, and citizens to uncover hidden environmental trends, filter seasonal cycles, test statistical significance, and trigger physical ESP32 hardware alerts.'))
hero_slot = st.container()

with st.expander("How it works", expanded=False):
    st.html(info_note('What this section does: Walks you through the 5 steps Earth scientists use to investigate whether an environmental change is a real long-term trend or short-term weather variability, with two instant demonstration cases.'))
    tutorial_tiles = []
    steps_info = [
        ("STEP 01", "Formulate & Acquire", "Choose a specific biome (Desert, Rain Forest, Tidal Estuary, or City Atmosphere) and acquire multi-decade NASA satellite records."),
        ("STEP 02", "Remove Seasonal Cycles", "Raw monthly data zig-zags wildly between summer and winter. Subtracting the monthly average exposes the hidden anomaly signal."),
        ("STEP 03", "Measure Linear Rate", "Fit an Ordinary Least Squares (OLS) linear regression line and Sen's median slope to measure the exact change per decade."),
        ("STEP 04", "Test Significance", "Run the Mann-Kendall and Student's t-test (p < 0.05) and plot the 95% Confidence Interval fan to assess evidence against a no-trend model."),
        ("STEP 05", "Compare & Forecast", "Discover delayed driver couplings, test whether a longer time window is needed, and transmit the alert bitmask to the ESP32 unit.")
    ]
    for idx, (s_num, s_title, s_desc) in enumerate(steps_info):
        tutorial_tiles.append('<div class="tutorial-step"><div class="tutorial-step-num">' + escape(s_num) + '</div><div class="tutorial-step-title">' + escape(s_title) + '</div><div class="plain-summary">' + escape(s_desc) + '</div></div>')
    st.html('<div class="tutorial-grid">' + ''.join(tutorial_tiles) + '</div>')

    st.write("")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.caption("Casablanca · 1990–2025 · 36 annual means")
        if st.button("Load Demo 1", width="content"):
            demo_meta = dict(BIOME_QUESTIONNAIRE_CATALOG["Atmosphere Above a Large City"]["questions"][
                "Is the Urban Heat Island intensifying air temperatures and moisture above Grand Casablanca?"
            ])
            demo_meta["start_year"] = 1990
            demo_meta["end_year"] = 2025
            demo_meta["router_source"] = "GUIDED DEMONSTRATION CASE #1 (35-YEAR WINDOW)"
            execute_demo(demo_meta, esp_ip, esp_port, firebase_url)
    with d_col2:
        st.caption("Same Casablanca variables · 2015–2025 · 11 annual means")
        if st.button("Load Demo 2", width="content"):
            demo_meta2 = dict(BIOME_QUESTIONNAIRE_CATALOG["Atmosphere Above a Large City"]["questions"][
                "Is the Urban Heat Island intensifying air temperatures and moisture above Grand Casablanca?"
            ])
            demo_meta2["start_year"] = 2015
            demo_meta2["end_year"] = 2025
            demo_meta2["router_source"] = "GUIDED DEMONSTRATION CASE #2 (SHORT 10-YEAR WINDOW)"
            execute_demo(demo_meta2, esp_ip, esp_port, firebase_url)

# ==============================================================================
# TOP WATCHLIST RIBBON
# ==============================================================================
st.html(with_info('<div class="section-heading"><h2>Saved investigations</h2></div>', 'What this section does: Stores every region you investigate during your session so you can track significant alerts at a glance and compare two regions side-by-side to see if they trend in opposite directions.'))

if len(st.session_state["watchlist"]) == 0:
    st.html('<div class="empty-state">' + icon("pin") + '<div><h2>No saved investigations</h2><p>No regional investigations have been archived in this session yet. Use the Guided Detective Questionnaire or AI Prompt Router in Section I below, or click one of the Demonstration buttons above.</p></div></div>')
else:
    cols_watch = st.container(horizontal=True, key="watchlist-row", gap="small")
    for slot, item in enumerate(st.session_state["watchlist"][-4:]):
        idx = max(0, len(st.session_state["watchlist"]) - 4) + slot
        w_meta = item["meta"]
        w_intel = item["intel"]
        t_v = w_intel["target_var"]
        t_stat = w_intel["var_stats"][t_v]
        if t_stat["significant"]:
            sig_badge = "SIGNIFICANT (p < 0.05)"
        elif t_stat["emerging"]:
            sig_badge = "EMERGING: " + observation_window_label(t_stat)
        else:
            sig_badge = "UNCONFIRMED: " + observation_window_label(t_stat)
        shift_badge = f"REGIME SHIFT: {w_intel['tipping_year']}" if w_intel["regime_shifted"] else f"INFLECTION: {w_intel['tipping_year']}"
        with cols_watch, st.container(key=f"watchlist-chip-{idx}", width=340):
            selected_class = " selected" if item is st.session_state["active_case"] else ""
            st.html(
                '<div class="location-chip' + selected_class + '"><div class="location-name">' + '<span class="watch-dot ' + ('significant' if t_stat['significant'] else 'emerging' if t_stat['emerging'] else 'unconfirmed') + '"></span>' + '<span title="' + escape(w_meta['region_name']) + '">' + escape(w_meta['region_name']) + '</span>' + '</div>'
                + f'<div class="location-value">{escape(t_v)} · {t_stat["slope_dec"]:+.3f} {escape(item.get("variables", VARIABLES)[t_v]["unit"])}/dec</div>'
                + '<details><summary>Case ' + str(idx + 1) + ' · ' + str(w_meta['start_year']) + '–' + str(w_meta['end_year']) + ' · Details</summary>'
                + '<div class="var-stats-grid">Status: ' + escape(sig_badge)
                + f'<br>Mann-Kendall p: {t_stat["mk_p"]:.4f}<br>' + escape(shift_badge)
                + '<br>Primary Driver: ' + escape(w_intel['strongest_driver'])
                + f' (r = {w_intel["driver_couplings"][w_intel["strongest_driver"]]["r_annual"]:+.2f})'
                + '<br>ESP32 Bitmask: [' + escape(w_intel['led_bitmask']) + ']</div><div class="plain-summary">'
                + safe_prose(t_stat['plain_summary']) + '</div></details></div>'
            )
            if st.button("Load", key=f"load_watch_{idx}", width="content"):
                st.session_state["active_case"] = item
                st.rerun()

    if len(st.session_state["watchlist"]) >= 2:
        with st.expander("Compare saved regions"):
            c_sel1, c_sel2 = st.columns(2)
            names = [f"Case 0{i+1}: {w['meta']['region_name']} ({w['intel']['target_var']})" for i, w in enumerate(st.session_state["watchlist"])]
            idx_a = c_sel1.selectbox("Select Primary Region", range(len(names)), format_func=lambda i: names[i], index=0)
            idx_b = c_sel2.selectbox("Select Secondary Region", range(len(names)), format_func=lambda i: names[i], index=min(1, len(names) - 1))

            case_a = st.session_state["watchlist"][idx_a]
            case_b = st.session_state["watchlist"][idx_b]
            va = case_a["intel"]["target_var"]
            vb = case_b["intel"]["target_var"]
            sa = case_a["intel"]["var_stats"][va]["slope_dec"]
            sb = case_b["intel"]["var_stats"][vb]["slope_dec"]

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=case_a["intel"]["annual_df"]["year"],
                y=standardize(case_a["intel"]["annual_df"][f"{va}_anom"]),
                mode="lines+markers",
                name=f"{case_a['meta']['region_name']} ({va}: {sa:+.3f}/dec)",
                line=dict(color=variable_color(va), width=2.5)
            ))
            fig_cmp.add_trace(go.Scatter(
                x=case_b["intel"]["annual_df"]["year"],
                y=standardize(case_b["intel"]["annual_df"][f"{vb}_anom"]),
                mode="lines+markers",
                name=f"{case_b['meta']['region_name']} ({vb}: {sb:+.3f}/dec)",
                line=dict(color=variable_color(vb), width=2, dash="dash")
            ))
            fig_cmp = apply_dark_monochrome_theme(fig_cmp, height=300, yaxis_title="Standardized Anomaly (sigma)")
            st.plotly_chart(fig_cmp, width="stretch", theme=None, config={"displayModeBar": False})
            st.html(chart_key(fig_cmp))
            st.markdown('<div class="plain-summary">Each record is standardized against its own mean and variability. '
                        'Check the variable and date labels before comparing; this overlay is descriptive and does not test a difference between regional slopes.</div>', unsafe_allow_html=True)
            if sa * sb < 0:
                st.markdown(
                    '<div class="plain-summary">'
                    '<strong>Opposite fitted slopes:</strong> The selected records have slopes with opposite signs. '
                    'Their variables and time windows may differ; this does not establish a common cause or a statistically significant regional contrast.'
                    '</div>',
                    unsafe_allow_html=True
                )

st.markdown("---")

# ==============================================================================
# SECTION I: GUIDED DETECTIVE QUESTIONNAIRE & AI NATURAL LANGUAGE ROUTER
# ==============================================================================
st.html(with_info(section_header("I", "Start an investigation", "inquiry"), 'What this section does: Lets you either answer three guided questions about an environment (Desert, Rain Forest, Tidal Estuary, or Large City Atmosphere) to automatically identify the right NASA variables, or type a custom research question in plain English.'))

tab_guided, tab_ai_prompt = st.tabs([
    "Guided questions",
    "Ask a question"
])

with tab_guided:
    q_col1, q_col2, q_col3 = st.columns([2, 3, 2])
    with q_col1:
        selected_biome = st.selectbox(
            "Region or biome",
            list(BIOME_QUESTIONNAIRE_CATALOG.keys())
        )
    with q_col2:
        available_questions = list(BIOME_QUESTIONNAIRE_CATALOG[selected_biome]["questions"].keys())
        selected_question = st.selectbox(
            "Research question",
            available_questions
        )
    with q_col3:
        window_choice = st.selectbox(
            "Record length",
            [
                "Full Record (1990-2025, 36 Annual Means)",
                "Medium Record (2005-2025, 21 Annual Means)",
                "Short Record (2013-2025, 13 Annual Means)"
            ]
        )

    matched_preset = BIOME_QUESTIONNAIRE_CATALOG[selected_biome]["questions"][selected_question]
    st.markdown(
        f'<div class="plain-summary" style="margin-bottom: 1rem;">'
        f'<strong>Auto-Matched NASA Variable Cluster:</strong> Target Variable: <code>{escape(str(matched_preset["target_var"]))}</code> '
        f'({escape(str(VARIABLES[matched_preset["target_var"]]["name"]))}) &nbsp;|&nbsp; '
        f'Driver Variables: <code>{escape(str(", ".join(matched_preset["driver_vars"])))}</code><br>'
        f'<strong>Regional & Scientific Literature Context:</strong> {escape(str(matched_preset["literature_context"]))}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.caption("Results are sent to ESP32 when a device address is configured.")
    if st.button("Run guided investigation", width="content"):
        with st.spinner("Acquiring NASA POWER time series, removing seasonal cycles, and computing statistical significance..."):
            try:
                meta = dict(matched_preset)
                if "2005" in window_choice:
                    meta["start_year"], meta["end_year"] = 2005, 2025
                elif "2013" in window_choice:
                    meta["start_year"], meta["end_year"] = 2013, 2025
                else:
                    meta["start_year"], meta["end_year"] = 1990, 2025
                meta["router_source"] = "GUIDED 3-STEP DETECTIVE QUESTIONNAIRE"
                execute_and_store_investigation(meta, esp_ip, esp_port, firebase_url)
                st.rerun()
            except Exception as err:
                st.error(f"Pipeline Execution Failure: {err}")

with tab_ai_prompt:
    user_location = st.text_input("City, country (optional)", key="user_location",
                                  help="An explicit city takes priority over locations mentioned in the prompt. For ambiguous places use the manual coordinates.")
    col_prompt, col_quick = st.columns([3, 2])
    with col_quick:
        preset_choice = st.selectbox(
            "Example question",
            [
                "Analyze agricultural drought, soil drying, and heat stress in Berrechid and Casablanca, Morocco from 1990 to 2025",
                "Investigate estuarine rainfall runoff, wind flushing, and coastal warming in the Sebou River Tidal Estuary, Morocco",
                "Examine Urban Heat Island warming, humidity, and solar radiation in the atmosphere above Grand Casablanca",
                "Analyze Amazon Rainforest hydrological shifts, canopy warming, and solar radiation in Brazil",
                "Analyze arid soil desiccation and solar radiative forcing in Ouarzazate and Pre-Saharan Morocco"
            ]
        )
    with col_prompt:
        user_query = st.text_area(
            "Your question",
            value=preset_choice,
            height=90
        )

    st.caption("Results are sent to ESP32 when a device address is configured.")
    if st.button("Run prompt investigation", width="content"):
        with st.spinner("Routing query, retrieving NASA POWER archives, removing seasonal cycles, and computing statistical significance..."):
            try:
                if enable_override:
                    meta = {"hypothesis": "Explore trends and associations at manually selected coordinates.",
                            "biome_type": "Manual regional grid point", "literature_context": "User-defined exploratory investigation."}
                    meta["lat"] = float(man_lat)
                    meta["lon"] = float(man_lon)
                    meta["target_var"] = man_target
                    meta["driver_vars"] = man_drivers[:3]
                    meta["start_year"] = int(man_years[0])
                    meta["end_year"] = int(man_years[1])
                    meta["region_name"] = f"Custom Coordinates ({man_lat:.2f}, {man_lon:.2f})"
                    meta["router_source"] = "MANUAL SIDEBAR OVERRIDE"
                else:
                    meta = parse_user_prompt_with_ai(user_query, gemini_key, user_location)
                execute_and_store_investigation(meta, esp_ip, esp_port, firebase_url)
                st.rerun()
            except Exception as err:
                st.error(f"Pipeline Execution Failure: {err}")


# ==============================================================================
# RENDER SECTIONS II TO VI WHEN AN ACTIVE INVESTIGATION EXISTS
# ==============================================================================
if st.session_state["active_case"] is None:
    st.html('<div class="empty-state" style="margin-top:24px">' + icon("pin") + '<div><h2>No results yet</h2><p>Run an investigation to reveal its trend, annual record, significance, driver relationships and hardware signals. No measurements are shown until a case is loaded.</p></div></div>')

if st.session_state["active_case"] is not None:
    active = st.session_state["active_case"]
    VARIABLES.update(active.get("variables", {}))
    meta = active["meta"]
    intel = active["intel"]
    t_var = intel["target_var"]
    d_vars = intel["driver_vars"]
    all_vars = intel["all_vars"]
    v_stats = intel["var_stats"]
    ann_df = intel["annual_df"]
    df_mon = intel["df_monthly"]
    with hero_slot:
        render_hero(meta, intel, VARIABLES)
        with st.expander("Notes · record and assumptions"):
            st.caption(f"Analysis uses {intel['n_years']} complete years, {int(ann_df.year.min())}-{int(ann_df.year.max())}. "
                       f"Climatology: {meta.get('climatology', 'record monthly means')}. "
                       f"Incomplete years excluded from annual statistics: {intel.get('excluded_years', [])}. "
                       f"Invalid monthly rows removed: {intel.get('dropped_rows', 0)}.")
            st.caption("OLS and Mann-Kendall results assume independent annual observations. Serial correlation and multiple testing are not corrected. "
                       "CUSUM splits and model forecasts are exploratory. POWER point data represent a grid cell, not a city-wide or basin-wide average.")

    st.markdown("---")
    st.html(with_info(section_header("II", "Significance", "significance"), "What this section does: Displays the decadal rate of change, 95% confidence intervals, Mann-Kendall non-parametric test, Student's t-test, and an Observation Window Sufficiency estimate explaining how many additional years of data are needed if a trend is still emerging."))

    with st.expander("Notes · investigation context"):
        st.markdown(
            f'<div class="telemetry-banner">'
            f'<strong>Routing Engine:</strong> {escape(meta["router_source"])} &nbsp;&nbsp;|&nbsp;&nbsp; '
            f'<strong>Biome Archetype:</strong> {escape(meta.get("biome_type", "Regional System"))} &nbsp;&nbsp;|&nbsp;&nbsp; '
            f'<strong>Coordinates:</strong> Latitude {meta["lat"]:+.2f}, Longitude {meta["lon"]:+.2f} ({escape(str(intel["n_years"]))} Years / {escape(str(len(df_mon)))} Monthly Observations)<br>'
            f'<strong>Physical Hypothesis:</strong> {escape(meta["hypothesis"])}<br>'
            f'<strong>Regional Background (not a verified citation):</strong> {escape(meta.get("literature_context", ""))}'
            f'</div>',
            unsafe_allow_html=True
        )

    card_html = []
    for i, v in enumerate(all_vars):
        st_v = v_stats[v]
        role_tag = "TARGET VARIABLE" if v == t_var else f"DRIVER VARIABLE 0{i}"
        if st_v["significant"]:
            verdict_class = "var-verdict-sig"
            verdict_text = "STATISTICALLY SIGNIFICANT (p < 0.05)"
            window_line = "Observation Window : Sufficient (Confirmed at 95% CI)"
        elif st_v["emerging"]:
            verdict_class = "var-verdict-emerging"
            verdict_text = "EMERGING TREND (BELOW 95% THRESHOLD)"
            window_line = observation_window_label(st_v)
        else:
            verdict_class = "var-verdict-noise"
            verdict_text = "UNCONFIRMED TREND (HIGH VARIABILITY)"
            window_line = observation_window_label(st_v)

        card_html.append(
                f"""
                <div class="var-card">
                    <div>
                        <div class="var-role">{escape(str(role_tag))} — {escape(str(v))}</div>
                        <div class="var-title">{variable_icon(v)}{escape(str(VARIABLES[v]['name']))}</div>
                        <div class="tile-sparkline">{sparkline_svg(ann_df[v], "Annual observed " + v, variable_color(v))}</div>
                        <div class="var-metric-box">
                            <div class="var-metric-label">Deseasonalized Linear Rate / Decade</div>
                            <div class="var-metric-val">{st_v['slope_dec']:+.3f} {escape(str(VARIABLES[v]['unit']))}</div>
                            <div class="var-metric-ci">+/- {st_v['ci_95_dec']:.3f} (95% Confidence Interval)</div>
                        </div>
                        <div class="{escape(verdict_class)}">{status_icon("significant" if st_v["significant"] else "emerging" if st_v["emerging"] else "unconfirmed")}{escape(str(verdict_text))}</div>
                        <details class="variable-notes"><summary>Statistics and notes</summary>
                        <div class="var-meta">{escape(str(VARIABLES[v]['mission']))} | {escape(str(VARIABLES[v]['category']))}</div>
                        <div class="var-plain-what">Simple definition: {escape(str(VARIABLES[v]['plain_desc']))}</div>
                        <details><summary>Scientific definition</summary>{escape(str(VARIABLES[v]['sci_desc']))}</details>
                        <div class="var-stats-grid">
                            Sen's Median Slope : {st_v['sens_slope_dec']:+.3f} /dec<br>
                            Mann-Kendall Test  : S = {st_v['mk_s']:+.0f}, Z = {st_v['mk_z']:+.2f} (p = {st_v['mk_p']:.4f})<br>
                            Student's t-test   : p = {st_v['ttest_p']:.4f} (R2 = {st_v['r_squared']:.2f})<br>
                            {escape(str(window_line))}
                        </div>
                    <div class="plain-summary" style="margin-top: 0.4rem;">
                        {safe_prose(st_v['plain_summary'])}
                    </div></details></div>
                </div>
                """
        )
    st.html('<div class="statistics-grid">' + "".join(card_html) + '</div>')

    st.markdown("---")

    # ==========================================================================
    # SECTION III & SECTION IV: TEMPORAL TRAJECTORY & DRIVER COUPLING MATRIX
    # ==========================================================================
    col_mod3, col_mod2 = st.container(), st.container()

    with col_mod3:
        st.html(with_info(section_header("III", "Trends over time", "time-series"), 'What this section does: Lets you inspect (1) the primary linear trendline and shaded 95% confidence interval fan, (2) how removing the monthly summer-winter seasonal cycle exposes hidden trends, (3) tipping-point regime shifts and AI forecasts, or (4) all four variables overlaid on a standardized scale.'))

        view_mode = st.radio(
            "Chart view",
            [
                "1. Primary Linear Trend & 95% Confidence Interval Fan",
                "2. Seasonal Cycle Removal (Raw Monthly Zig-Zag vs. Deseasonalized Signal)",
                "3. Piecewise Regime Inflection (CUSUM) & 10-Yr Multivariate Forecast",
                "4. Standardized 4-Variable Z-Score Overlay (Dimensionless Sigma Scale)"
            ],
            horizontal=True,
            format_func=lambda value: {"1.": "Trend & CI", "2.": "Seasonal removal", "3.": "Regime & forecast", "4.": "Z-score overlay"}[value[:2]]
        )

        chart_area, chart_stats = st.columns([2, 1], gap="large")
        with chart_stats:
            st.html('<div class="chart-facts">' + stat_tile("OLS slope / decade", f"{v_stats[t_var]['slope_dec']:+.3f} {VARIABLES[t_var]['unit']}")
                    + stat_tile("95% confidence interval", f"+/- {v_stats[t_var]['ci_95_dec']:.3f}")
                    + stat_tile("Mann-Kendall p", f"{v_stats[t_var]['mk_p']:.4f}")
                    + stat_tile("Student's t-test p", f"{v_stats[t_var]['ttest_p']:.4f}")
                    + stat_tile("Annual means", str(intel['n_years'])) + '</div>')
        with chart_area:
            if view_mode.startswith("1."):
                fig_lin = go.Figure()
                x_yr = ann_df["year"].values
                y_obs = ann_df[t_var].values
                st_t = v_stats[t_var]

                # Shaded 95% Confidence Interval Envelope around the Linear Regression Line
                fig_lin.add_trace(go.Scatter(
                    x=np.concatenate([x_yr, x_yr[::-1]]),
                    y=np.concatenate([st_t["ci_upper_raw"], st_t["ci_lower_raw"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(255,143,90,.14)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="95% Confidence Interval Envelope"
                ))

                # Observed Annual Time Series
                fig_lin.add_trace(go.Scatter(
                    x=x_yr, y=y_obs, mode="lines+markers",
                    name=f"Observed {t_var} ({VARIABLES[t_var]['unit']})",
                    line=dict(color=variable_color(t_var), width=1.8),
                    marker=dict(color=variable_color(t_var), size=5)
                ))

                # Full-Period Linear Regression Line
                fig_lin.add_trace(go.Scatter(
                    x=x_yr, y=st_t["y_hat_raw"], mode="lines",
                    name=f"OLS Linear Trend ({st_t['slope_dec']:+.3f} {VARIABLES[t_var]['unit']}/dec, p={st_t['ttest_p']:.4f})",
                    line=dict(color=variable_color(t_var), width=3.2)
                ))

                if "fillgradient" in go.Scatter()._valid_props:
                    fig_lin.data[0].fillgradient = dict(type="vertical", colorscale=[[0, "rgba(255,143,90,.02)"], [1, "rgba(255,143,90,.22)"]])
                fig_lin = apply_dark_monochrome_theme(
                    fig_lin, height=430, yaxis_title=f"{VARIABLES[t_var]['name']} ({VARIABLES[t_var]['unit']})"
                )
                st.plotly_chart(fig_lin, width="stretch", theme=None, config={"displayModeBar": False})
                st.html(chart_key(fig_lin))

                st.markdown(
                    f'<div class="plain-summary">'
                    f'<strong>How to read this chart:</strong> The solid colored line is the full-period linear trendline '
                    f'({st_t["slope_dec"]:+.3f} {escape(str(VARIABLES[t_var]["unit"]))} per decade), and the shaded band around it is the '
                    f'<strong>95% Confidence Interval</strong>. A narrow band with a steep slope indicates a highly reliable trend, '
                    f'whereas a wide fan indicates larger year-to-year uncertainty.'
                    f'</div>',
                    unsafe_allow_html=True
                )

            elif view_mode.startswith("2."):
                fig_seas = go.Figure()
                fig_seas.add_trace(go.Scatter(
                    x=df_mon["date"], y=df_mon[t_var], mode="lines",
                    name=f"Raw Monthly Series (With Seasonal Cycle)",
                    line=dict(color="#7D90AC", width=1.2)
                ))
                fig_seas.add_trace(go.Scatter(
                    x=df_mon["date"], y=df_mon[f"{t_var}_anom"] + v_stats[t_var]["mean_raw"], mode="lines",
                    name=f"Deseasonalized Series (Seasonal Cycle Removed)",
                    line=dict(color=variable_color(t_var), width=1.8)
                ))
                fig_seas.add_trace(go.Scatter(
                    x=df_mon["date"],
                    y=v_stats[t_var]["monthly_trend_line"] + v_stats[t_var]["mean_raw"],
                    mode="lines",
                    name=f"Exposed Linear Trend ({v_stats[t_var]['slope_dec']:+.3f}/dec)",
                    line=dict(color=variable_color(t_var), width=3.2)
                ))
                fig_seas = apply_dark_monochrome_theme(
                    fig_seas, height=430, xaxis_title="Monthly Timeline",
                    yaxis_title=f"{VARIABLES[t_var]['name']} ({VARIABLES[t_var]['unit']})"
                )
                st.plotly_chart(fig_seas, width="stretch", theme=None, config={"displayModeBar": False})
                st.html(chart_key(fig_seas))

                st.markdown(
                    '<div class="plain-summary">'
                    '<strong>Why removing the seasonal cycle matters:</strong> The muted blue wave shows the raw monthly satellite measurements, '
                    'which swing up and down every summer and winter. That regular seasonal cycle hides gradual climate shifts. '
                    'By subtracting the historical average for each month (January through December), the anomaly curve and solid trendline '
                    'expose the true underlying anomaly trajectory across all monthly data points.'
                    '</div>',
                    unsafe_allow_html=True
                )

            elif view_mode.startswith("3."):
                fig_main = go.Figure()
                y_obs = ann_df[t_var].values
                x_yr = ann_df["year"].values

                fig_main.add_trace(go.Scatter(
                    x=x_yr, y=y_obs, mode="lines+markers",
                    name=f"Observed {t_var} ({VARIABLES[t_var]['unit']})",
                    line=dict(color=variable_color(t_var), width=2.2),
                    marker=dict(color=variable_color(t_var), size=5)
                ))

                tip_yr = intel["tipping_year"]
                tip_idx = int(np.where(x_yr == tip_yr)[0][0])
                lin_pre = safe_linregress(x_yr[:tip_idx], y_obs[:tip_idx])
                lin_post = safe_linregress(x_yr[tip_idx:], y_obs[tip_idx:])

                fig_main.add_trace(go.Scatter(
                    x=x_yr[:tip_idx],
                    y=lin_pre.slope * x_yr[:tip_idx] + lin_pre.intercept,
                    mode="lines", name=f"Pre-{tip_yr} Regime ({intel['slope_pre']:+.3f}/dec)",
                    line=dict(color=variable_color(t_var), width=2.2, dash="dash")
                ))
                fig_main.add_trace(go.Scatter(
                    x=x_yr[tip_idx:],
                    y=lin_post.slope * x_yr[tip_idx:] + lin_post.intercept,
                    mode="lines", name=f"Post-{tip_yr} Regime ({intel['slope_post']:+.3f}/dec)",
                    line=dict(color=variable_color(t_var), width=3.2, dash="dash")
                ))

                fig_main.add_trace(go.Scatter(
                    x=v_stats[t_var]["extreme_years"],
                    y=v_stats[t_var]["extreme_vals"],
                    mode="markers", name="Isolation Forest Extreme Anomalies",
                    marker=dict(color="#151B45", size=11, symbol="diamond", line=dict(width=2, color=variable_color(t_var)))
                ))

                fig_main.add_trace(go.Scatter(
                    x=intel["future_years"], y=intel["forecast_vals"],
                    mode="lines+markers", name=f"10-Yr Forecast ({intel['ai_engine_label']})",
                    line=dict(color=variable_color(t_var), width=2.5, dash="dot"),
                    marker=dict(color=variable_color(t_var), size=6, symbol="square")
                ))

                fig_main.add_vline(
                    x=tip_yr, line_width=1.5, line_dash="dot", line_color=variable_color(t_var),
                    annotation_text=f"CUSUM Inflection: {tip_yr}",
                    annotation_position="bottom left",
                    annotation_font=dict(color="#281509", family="Segoe UI Variable, Segoe UI, system-ui, Inter, sans-serif", size=11),
                    annotation_bgcolor=variable_color(t_var), annotation_borderpad=6
                )
                fig_main = apply_dark_monochrome_theme(
                    fig_main, height=430, yaxis_title=f"{VARIABLES[t_var]['name']} ({VARIABLES[t_var]['unit']})"
                )
                st.plotly_chart(fig_main, width="stretch", theme=None, config={"displayModeBar": False})
                st.html(chart_key(fig_main))

                st.markdown(
                    f'<div class="plain-summary">'
                    f'<strong>How to read this non-linear regime chart:</strong> Environmental trends rarely move at a constant speed. '
                    f'The vertical line at <strong>{escape(str(tip_yr))}</strong> marks an exploratory CUSUM split where the fitted decadal slope changes from '
                    f'{intel["slope_pre"]:+.3f} to {intel["slope_post"]:+.3f} {escape(str(VARIABLES[t_var]["unit"]))}/decade, followed by the 10-year '
                    f'multivariate scenario forecast. This split is not proof of a physical tipping point; the forecast has not been independently validated.'
                    f'</div>',
                    unsafe_allow_html=True
                )

            else:
                fig_z = go.Figure()
                styles = [
                    dict(color=variable_color(t_var), width=3.0, dash="solid"),
                    dict(color=variable_color(t_var), width=2.0, dash="dash"),
                    dict(color="#F0C66C", width=2.0, dash="dot"),
                    dict(color=variable_color(t_var), width=1.8, dash="dashdot")
                ]
                for idx, v in enumerate(all_vars):
                    z_series = standardize(ann_df[f"{v}_anom"].values)
                    fig_z.add_trace(go.Scatter(
                        x=ann_df["year"], y=z_series, mode="lines",
                        name=f"{v} ({'Target' if v == t_var else 'Driver'})",
                        line=dict(styles[idx % len(styles)], color=variable_color(v))
                    ))
                fig_z.add_vline(x=intel["tipping_year"], line_width=1.5, line_dash="dot", line_color=variable_color(t_var))
                fig_z = apply_dark_monochrome_theme(
                    fig_z, height=430, yaxis_title="Deseasonalized Z-Score Anomaly (sigma)"
                )
                st.plotly_chart(fig_z, width="stretch", theme=None, config={"displayModeBar": False})
                st.html(chart_key(fig_z))

                st.markdown(
                    '<div class="plain-summary">'
                    '<strong>How to read this chart:</strong> Because temperature, sunlight, and water are measured in different units, '
                    'this chart converts all four variables onto a single standardized scale (standard deviations from normal, where 0 is average). '
                    'When one line rises above 0 while another drops below 0, their anomalies have opposite signs at that time.'
                    '</div>',
                    unsafe_allow_html=True
                )

            # Processed CSV Download Button for Students & Researchers
            csv_buffer = df_mon.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download monthly CSV",
                data=csv_buffer,
                file_name=f"concord_deseasonalized_{t_var}_{meta['start_year']}_{meta['end_year']}.csv",
                mime="text/csv",
                width="content"
            )

    with col_mod2:
        st.html(with_info(section_header("IV", "Driver relationships", "coupling"), 'What this section does: Measures how strongly the three driver variables vary with the target variable, checks for delayed 1-to-3 month reactions, and detects opposite-direction paradoxes.'))

        fig_heat = px.imshow(
            np.round(intel["corr_matrix"].values, 2),
            x=all_vars, y=all_vars,
            color_continuous_scale=[[0.0, "#3E7197"], [0.5, "#252D4B"], [1.0, "#9B6146"]],
            zmin=-1, zmax=1,
            text_auto=True, aspect="auto"
        )
        fig_heat = apply_dark_monochrome_theme(fig_heat, height=360, xaxis_title="", yaxis_title="")
        fig_heat.update_layout(hovermode="closest", coloraxis_showscale=False, margin=dict(l=16, r=16, t=48, b=32))
        heat_labels = [escape(v).replace("ALLSKY_SFC_SW_DWN", "ALLSKY_<br>SFC_SW_<br>DWN") for v in all_vars]
        fig_heat.update_xaxes(side="top", tickangle=0, tickvals=all_vars, ticktext=heat_labels, tickfont_size=10)
        fig_heat.update_yaxes(tickvals=all_vars, ticktext=heat_labels, tickfont_size=10)
        fig_heat.update_traces(texttemplate="%{z:.2f}", textfont=dict(color="#F5F7FF", size=12))
        st.plotly_chart(fig_heat, width="stretch", theme=None, config={"displayModeBar": False})

        lag_rows = []
        for d_v in d_vars:
            dc = intel["driver_couplings"][d_v]
            lag_rows.append({
                "Driver": d_v,
                "Correlation (r)": f"{dc['r_annual']:+.2f}",
                "Response Lag": f"{dc['best_lag_months']} mo",
                "Lagged r": f"{dc['lag_r']:+.2f}",
                "Interaction": "INVERSE PARADOX" if dc["paradox_coupled"] else "UNDEFINED" if not np.isfinite(dc["r_annual"]) else "EXPLORATORY ASSOCIATION"
            })
        st.html(html_table(lag_rows))

        s_drv = intel["strongest_driver"]
        s_corr = intel["driver_couplings"][s_drv]["r_annual"]
        rel_type = "an inverse (opposite-direction)" if s_corr < 0 else "a direct (same-direction)"
        st.markdown(
            f'<div class="plain-summary">'
            f'<strong>Plain-Language Coupling Summary:</strong> The largest available annual correlation with <code>{escape(str(t_var))}</code> is '
            f'<strong>{escape(str(VARIABLES[s_drv]["name"]))} (<code>{escape(str(s_drv))}</code>)</strong> with a correlation of <strong>r = {s_corr:+.2f}</strong>, '
            f'indicating {escape(str(rel_type))} association when defined. Correlation and the best tested lag do not establish causation; nan means a correlation is undefined. The fitted slope ratio is '
            f'<strong>{intel["accel_ratio"]:.1f}x</strong> after the inflection year <strong>{escape(str(intel["tipping_year"]))}</strong>.'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ==========================================================================
    # SECTION V & SECTION VI: PERTURBATION SIMULATOR & ESP32 HARDWARE TWIN
    # ==========================================================================
    col_mod4, col_mod5 = st.container(), st.container()

    with col_mod4:
        st.html(with_info(section_header("V", "What if?", "simulator"), 'What this section does: Lets you manually increase or decrease the three physical driver variables to test how the target variable will react in the future according to the trained machine learning model.'))

        with st.container(key="simulator-controls"):
            sim_cols = st.columns(len(d_vars))
            baseline_vec = ann_df[all_vars].values[-1].copy()
            simulated_vec = baseline_vec.copy()

            for i, d_v in enumerate(d_vars):
                d_std = v_stats[d_v]["std_raw"]
                with sim_cols[i]:
                    delta_sigma = st.slider(
                        f"Perturb {d_v} (sigma)",
                        min_value=-3.0, max_value=3.0, value=0.0, step=0.25,
                        help=f"1.0 sigma = {d_std:.3f} {VARIABLES[d_v]['unit']}",
                        disabled=d_std == 0, key=f"sim_{len(st.session_state['watchlist'])}_{meta['region_name']}_{meta['start_year']}_{meta['end_year']}_{t_var}_{d_v}"
                    )
                    simulated_vec[i + 1] = clamp_physical(d_v, baseline_vec[i + 1] + (delta_sigma * d_std))

        base_pred = float(intel["rf_model"].predict([baseline_vec])[0])
        rf_sim_pred = float(intel["rf_model"].predict([simulated_vec])[0])
        linear_delta = 0.0
        for i, d_v in enumerate(d_vars):
            std_shift = (simulated_vec[i + 1] - baseline_vec[i + 1]) / v_stats[d_v]["std_raw"] if v_stats[d_v]["std_raw"] > 0 else 0.0
            r_val = intel["driver_couplings"][d_v]["r_annual"]
            if np.isfinite(r_val):
                linear_delta += std_shift * r_val * v_stats[t_var]["std_raw"] * 0.45

        final_sim_pred = clamp_physical(t_var, 0.5 * rf_sim_pred + 0.5 * (base_pred + linear_delta))
        sim_diff = final_sim_pred - base_pred

        m_sim1, m_sim2, m_sim3 = st.columns(3)
        m_sim1.metric(f"Unperturbed Next-Year Prediction ({t_var})", f"{base_pred:.3f} {VARIABLES[t_var]['unit']}")
        m_sim2.metric(
            f"Simulated {t_var} Output",
            f"{final_sim_pred:.3f} {VARIABLES[t_var]['unit']}",
            f"{sim_diff:+.3f} {VARIABLES[t_var]['unit']}", delta_color="off"
        )
        top_imp_var = max(intel["importances"], key=intel["importances"].get)
        with m_sim3:
            st.html('<div class="stat-tile gauge-tile"><div><div class="tile-label">Dominant ML Predictor</div><div class="stat-value">' + escape(top_imp_var) + '</div><div class="stat-note">' + f'{intel["importances"][top_imp_var]*100:.1f}% Weight' + '</div></div>' + arc_gauge_svg(intel["importances"][top_imp_var]*100) + '</div>')

        st.markdown(
            f'<div class="plain-summary">'
            f'<strong>Plain-Language Simulation Takeaway:</strong> Under your selected slider adjustments, '
            f'{escape(str(VARIABLES[t_var]["name"]))} shifts by <strong>{sim_diff:+.3f} {escape(str(VARIABLES[t_var]["unit"]))}</strong> from its unperturbed next-year prediction. '
            f'Across the historical record, <code>{escape(str(top_imp_var))}</code> accounts for {intel["importances"][top_imp_var]*100:.1f}% '
            f'of the Random Forest impurity-based feature importance. These exploratory sensitivities are not causal effects or measured prediction accuracy.'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_mod5:
        st.html(with_info(section_header("VI", "Map and device", "telemetry"), 'What this section does: Displays the exact satellite sampling coordinates and mirrors the live 5-LED bitmask transmitted over Wi-Fi UDP to the physical ESP32 hardware unit.'))

        render_map(meta, height=330, context="telemetry")
        st.markdown('<div class="plain-summary">The orange point marks the requested sampling coordinates. '
                    'NASA POWER returns a model grid cell, not direct measurements throughout the named region. '
                    'The LED table shows computed conditions; UDP provides no device acknowledgement.</div>', unsafe_allow_html=True)

        bitmask = intel["led_bitmask"]
        b_on, o_on, r_on, w_on, g_on = [c == "1" for c in bitmask]

        led_table = pd.DataFrame([
            {"Hardware Channel": "LED 1 (Blue)",   "State": "ACTIVE [1]" if b_on else "OFF [0]", "Scientific Trigger": "Significant Hydrological Trend (p < 0.05)"},
            {"Hardware Channel": "LED 2 (Orange)", "State": "ACTIVE [1]" if o_on else "OFF [0]", "Scientific Trigger": f"CUSUM Regime Acceleration ({intel['tipping_year']})"},
            {"Hardware Channel": "LED 3 (Red)",    "State": "ACTIVE [1]" if r_on else "OFF [0]", "Scientific Trigger": "High-Confidence Anomaly (Mann-Kendall p < 0.01)"},
            {"Hardware Channel": "LED 4 (White)",  "State": "ACTIVE [1]" if w_on else "OFF [0]", "Scientific Trigger": "Inverse Driver-Target Paradox Coupling"},
            {"Hardware Channel": "LED 5 (Green)",  "State": "ACTIVE [1]" if g_on else "OFF [0]", "Scientific Trigger": "System Within Natural Baseline Variability"}
        ])
        st.html('<div class="led-grid">' + ''.join(
            led_tile(row["Hardware Channel"], row["State"] == "ACTIVE [1]", row["Scientific Trigger"], color)
            for row, color in zip(led_table.to_dict("records"), ["blue", "orange", "red", "white", "green"])
        ) + '</div>')
        st.html('<div class="device-link">' + icon("link") + '<span>Device link · ' + escape(active.get("udp_status", "")) + '</span></div>')

st.html(footer())
