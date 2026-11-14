"""Validated inputs and numerical helpers shared by the Concord observatory."""
import re

import numpy as np
import pandas as pd
import requests
from scipy import stats


def validate_metadata(meta, variables):
    result = dict(meta)
    target = result.get("target_var")
    drivers = result.get("driver_vars", [])
    if target not in variables or not isinstance(drivers, list):
        raise ValueError("Select a supported target and three driver variables.")
    if len(drivers) != 3 or len(set([target] + drivers)) != 4 or any(v not in variables for v in drivers):
        raise ValueError("Select exactly three distinct drivers, excluding the target.")
    for key, limit in (("lat", 90), ("lon", 180)):
        result[key] = float(result[key])
        if not np.isfinite(result[key]) or abs(result[key]) > limit:
            raise ValueError(f"{key} must be between {-limit} and {limit}.")
    for key in ("start_year", "end_year"):
        value = float(result[key])
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError("Observation years must be whole numbers.")
        result[key] = int(value)
    if not 1 <= result["start_year"] <= result["end_year"] <= 9999:
        raise ValueError("The start year must precede the end year.")
    for key in ("region_name", "hypothesis", "router_source"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"Missing investigation field: {key}.")
    return result


def normalize_monthly_data(frame, variables):
    """Require real dates and one row per month; never invent timestamps."""
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: c.lower() for c in df if c.lower() in {"date", "year", "month"}})
    if df.columns.duplicated().any():
        raise ValueError("CSV column names must be unique.")
    missing = set(variables) - set(df.columns)
    if missing:
        raise ValueError("Missing variable columns: " + ", ".join(sorted(missing)))
    if "date" in df:
        dates = pd.to_datetime(df["date"], errors="coerce")
    elif {"year", "month"}.issubset(df):
        year = pd.to_numeric(df["year"], errors="coerce")
        month = pd.to_numeric(df["month"], errors="coerce")
        if ((year % 1 != 0) | (month % 1 != 0)).any():
            raise ValueError("Year and month must be whole numbers.")
        dates = pd.to_datetime(dict(year=year, month=month, day=1), errors="coerce")
    else:
        raise ValueError("Provide a date column (YYYY-MM-DD), or year and month columns. Dates are never inferred.")
    if dates.isna().any():
        raise ValueError("The dataset contains invalid or missing dates.")
    df["date"] = dates.dt.to_period("M").dt.to_timestamp()
    if df["date"].duplicated().any():
        raise ValueError("Provide one observation per calendar month; aggregate daily data before uploading.")
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    for var in variables:
        values = pd.to_numeric(df[var], errors="coerce")
        df[var] = values.where(np.isfinite(values) & (values > -990))
    result = df.dropna(subset=variables).sort_values("date").reset_index(drop=True)
    if result.empty:
        raise ValueError("No valid monthly observations remain for all four variables.")
    result.attrs["dropped_rows"] = frame.attrs.get("dropped_rows", 0) + len(df) - len(result)
    return result


def mann_kendall_details(y, years=None):
    """Tie-corrected two-sided MK test and Sen slope per actual calendar year."""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if years is None else np.asarray(years, dtype=float)
    if len(y) < 2 or len(x) != len(y) or not np.isfinite(y).all() or not np.all(np.diff(x) > 0):
        raise ValueError("Mann-Kendall needs finite values at increasing observation times.")
    i, j = np.triu_indices(len(y), 1)
    differences = y[j] - y[i]
    s = float(np.sign(differences).sum())
    _, ties = np.unique(y, return_counts=True)
    n = len(y)
    variance = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18
    z = (s - np.sign(s)) / np.sqrt(variance) if variance > 0 else 0.0
    return s, float(z), float(2 * stats.norm.sf(abs(z))), float(np.median(differences / (x[j] - x[i])))


def safe_linregress(x, y):
    if np.ptp(y) == 0:
        from types import SimpleNamespace
        return SimpleNamespace(slope=0.0, intercept=float(y[0]), stderr=0.0, pvalue=1.0, rvalue=0.0)
    return stats.linregress(x, y)


def safe_correlation(x, y):
    pairs = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)}).dropna()
    if len(pairs) < 3 or pairs.x.nunique() < 2 or pairs.y.nunique() < 2:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(pairs.x, pairs.y)
    return float(r), float(p)


def standardize(values):
    values = np.asarray(values, dtype=float)
    std = np.std(values)
    return (values - np.mean(values)) / std if std > 0 else np.zeros_like(values)


def estimate_additional_years(n, slope, stderr, p_value):
    """OLS planning approximation, not a promise of future significance."""
    if p_value < 0.05:
        return 0
    if slope == 0 or stderr <= 0 or not np.isfinite(stderr):
        return None
    t_value = abs(slope / stderr)
    required = np.ceil(n * (stats.t.ppf(0.975, n - 2) / t_value) ** (2 / 3))
    return max(0, int(required) - n) if np.isfinite(required) else None


def geocode_city(city):
    """Resolve an explicit city field, rejecting ambiguous/unmatched locations."""
    city = city.strip()
    if not city:
        return None
    parts = [p.strip() for p in city.split(",") if p.strip()]
    response = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                            params={"name": parts[0], "count": 10, "language": "en", "format": "json"}, timeout=10)
    response.raise_for_status()
    results = response.json().get("results", [])
    if len(parts) > 1:
        results = [r for r in results if all(
            any(p.casefold() == str(r.get(k, "")).casefold() for k in ("country", "country_code", "admin1", "admin2"))
            for p in parts[1:])]
    if not results:
        raise ValueError("City not found. Use City, Country or the manual coordinate override.")
    if len(results) > 1:
        raise ValueError("Several places match this city. Add its country and administrative region, or use coordinates.")
    top = results[0]
    name = ", ".join(dict.fromkeys(str(top[k]) for k in ("name", "admin1", "country") if top.get(k)))
    return {"region_name": name, "lat": float(top["latitude"]), "lon": float(top["longitude"])}


def prompt_years(prompt):
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", prompt)]
    start, end = (years[0], years[-1]) if len(years) >= 2 else (years[0] if years else 1990, 2025)
    if not 1981 <= start <= end <= 2025:
        raise ValueError("Use a year range within the supported 1981-2025 archive.")
    return start, end


def clamp_physical(var, value):
    if not np.isfinite(value):
        raise ValueError("The exploratory forecast became unstable for this dataset.")
    if var == "GWETTOP":
        return float(np.clip(value, 0, 1))
    if var in {"PRECTOTCORR", "QV2M", "ALLSKY_SFC_SW_DWN", "WS2M"}:
        return float(max(0, value))
    return float(value)
