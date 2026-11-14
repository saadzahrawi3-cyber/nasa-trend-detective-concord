"""Read-only integration check; does not load secrets or send hardware packets."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_concord import load_engine

app = load_engine()
variables = ["T2M", "ALLSKY_SFC_SW_DWN", "QV2M", "WS2M"]
reference = app["fetch_nasa_multivariable_data"](33.57, -7.59, 1990, 2025, variables)
print("NASA rows:", len(reference), "source units:", reference.attrs.get("source_units"))
for start in (1990, 2015):
    result = app["run_multivariable_intelligence_engine"](
        reference[reference.year >= start], "T2M", variables[1:], reference_df=reference)
    print(json.dumps({"requested_start": start, "complete_years": result["n_years"],
                      "excluded_years": result["excluded_years"],
                      "slope_decade": result["var_stats"]["T2M"]["slope_dec"],
                      "ols_p": result["var_stats"]["T2M"]["ttest_p"],
                      "forecast_end_year": int(result["future_years"][-1])}))
print("Geocoder:", app["geocode_city"]("Casablanca, Morocco"))
print("Playwright installed:", importlib.util.find_spec("playwright") is not None)
