"""Offline numerical regressions and real Streamlit rerun tests."""
import ast
import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests
from scipy import stats
from streamlit.testing.v1 import AppTest

from concord_utils import (normalize_monthly_data, mann_kendall_details,
                           estimate_additional_years, validate_metadata, geocode_city)

ROOT = Path(__file__).resolve().parents[1]


def load_engine():
    # Keep APP.PY a drop-in Streamlit script while testing its pure functions
    # without executing the UI, reading credentials, or sending telemetry.
    tree = ast.parse((ROOT / "APP.PY").read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"VARIABLES", "BIOME_QUESTIONNAIRE_CATALOG"} for t in node.targets):
            nodes.append(node)
    namespace = {"OPENVINO_AVAILABLE": False}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "APP.PY", "exec"), namespace)
    return namespace


def fixture(years=12, constant=False):
    date = pd.date_range("2000-01-01", periods=years * 12, freq="MS")
    rng = np.random.default_rng(42)
    t = np.arange(len(date)) / 12
    season = np.sin(2 * np.pi * date.month / 12)
    df = pd.DataFrame({"date": date})
    for v, baseline, slope, seasonal in [("T2M", 20, .1, 4), ("PRECTOTCORR", 3, -.02, 1),
                                         ("GWETTOP", .5, -.002, .05), ("QV2M", 8, .02, 2)]:
        df[v] = baseline if constant else baseline + slope * t + seasonal * season + rng.normal(0, abs(slope) + .01, len(t))
    return df


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = load_engine()

    def engine(self, df):
        return self.app["run_multivariable_intelligence_engine"](df, "T2M", ["PRECTOTCORR", "GWETTOP", "QV2M"])

    def test_ties_and_calendar_sen_slope(self):
        y = np.array([1, 1, 2, 2, 3, 4.])
        years = np.array([2000, 2001, 2004, 2005, 2006, 2009])
        s, z, p, slope = mann_kendall_details(y, years)
        self.assertEqual(s, 13)
        self.assertAlmostEqual(slope, stats.theilslopes(y, years).slope)
        self.assertAlmostEqual(z, 12 / np.sqrt((6 * 5 * 17 - 2 * 2 * 1 * 9) / 18))
        self.assertTrue(0 < p < 1)

    def test_no_invented_dates_and_duplicate_months(self):
        with self.assertRaisesRegex(ValueError, "date column"):
            normalize_monthly_data(fixture().drop(columns="date"), ["T2M"])
        with self.assertRaisesRegex(ValueError, "one observation"):
            normalize_monthly_data(pd.concat([fixture(), fixture().iloc[:1]]), ["T2M"])

    def test_case_insensitive_year_month_and_flags(self):
        df = fixture()
        df["YEAR"], df["MONTH"] = df.date.dt.year, df.date.dt.month
        df = df.drop(columns="date")
        df.loc[0, "T2M"] = -999
        df.loc[1, "T2M"] = np.inf
        result = normalize_monthly_data(df, ["T2M"])
        self.assertEqual(len(result), len(df) - 2)
        self.assertEqual(result.attrs["dropped_rows"], 2)

    def test_full_engine_and_constant_series(self):
        for constant in (False, True):
            result = self.engine(fixture(constant=constant))
            self.assertTrue(np.isfinite(result["forecast_vals"]).all())
            self.assertEqual(len(result["forecast_vals"]), 10)
            self.assertEqual(len(result["led_bitmask"]), 5)
            self.assertEqual(result["n_years"], 12)
            if constant:
                self.assertEqual(result["var_stats"]["T2M"]["ttest_p"], 1)
                self.assertIsNone(result["var_stats"]["T2M"]["additional_years_needed"])
                self.assertEqual(result["led_bitmask"], "00001")

    def test_missing_month_excludes_year_but_keeps_calendar_gaps(self):
        result = self.engine(fixture().drop(index=14))
        self.assertEqual(result["n_years"], 11)
        self.assertEqual(result["excluded_years"], [2001])
        self.assertEqual(len(result["df_monthly"]), 143)

    def test_short_record_rejected_before_regression(self):
        with self.assertRaisesRegex(ValueError, "six complete years"):
            self.engine(fixture(years=3))

    def test_no_capped_or_fabricated_year_estimate(self):
        self.assertIsNone(estimate_additional_years(10, 0, .2, 1))
        self.assertGreater(estimate_additional_years(10, .001, 1, .99), 85)
        self.assertEqual(estimate_additional_years(10, 1, .1, .001), 0)

    def test_router_preserves_dates_and_all_catalog_regions(self):
        for query, region in [("Cairo 1995 to 2020", "Cairo"), ("Congo 2005 to 2020", "Congo"),
                              ("Chesapeake 1995 to 2020", "Chesapeake"), ("Errachidia 1995 to 2020", "Errachidia")]:
            result = self.app["parse_user_prompt_with_ai"](query)
            self.assertIn(region, result["region_name"])
            self.assertEqual(result["end_year"], 2020)

    def test_unresolved_city_never_silently_relocated(self):
        response = Mock()
        response.json.return_value = {"results": []}
        with patch("requests.get", return_value=response), self.assertRaisesRegex(ValueError, "City not found"):
            self.app["parse_user_prompt_with_ai"]("temperature in Unknownplace")

    def test_offline_biome_and_known_city_variable_selection(self):
        with patch("requests.get", side_effect=AssertionError("Unexpected network request")):
            self.assertEqual(self.app["parse_user_prompt_with_ai"]("desert drought")["target_var"], "GWETTOP")
            self.assertEqual(self.app["parse_user_prompt_with_ai"]("rainfall in Casablanca")["target_var"], "PRECTOTCORR")
            self.assertEqual(self.app["parse_user_prompt_with_ai"]("wind in Casablanca")["target_var"], "WS2M")

    def test_fixed_climatology_for_short_and_long_records(self):
        variables = ["T2M", "PRECTOTCORR", "GWETTOP", "QV2M"]
        df = normalize_monthly_data(fixture(), variables)
        full = self.app["deseasonalize_dataframe"](df, variables, df)
        short = self.app["deseasonalize_dataframe"](df[df.year >= 2005], variables, df)
        np.testing.assert_allclose(full.loc[full.year >= 2005, "T2M_anom"], short.T2M_anom)

    def test_invalid_router_metadata_rejected(self):
        meta = {"target_var": "T2M", "driver_vars": ["T2M", "QV2M", "WS2M"], "lat": 0, "lon": 0,
                "start_year": 1990, "end_year": 2025, "region_name": "Test", "hypothesis": "Test", "router_source": "TEST"}
        with self.assertRaisesRegex(ValueError, "three distinct"):
            validate_metadata(meta, self.app["VARIABLES"])
        meta["driver_vars"] = ["GWETTOP", "QV2M", "WS2M"]
        meta["lat"] = float("nan")
        with self.assertRaisesRegex(ValueError, "lat must"):
            validate_metadata(meta, self.app["VARIABLES"])

    def test_geocoder_honors_country(self):
        response = Mock()
        response.json.return_value = {"results": [
            {"name": "Paris", "country": "United States", "latitude": 1, "longitude": 1},
            {"name": "Paris", "country": "France", "latitude": 48.85, "longitude": 2.35}]}
        with patch("requests.get", return_value=response):
            self.assertEqual(geocode_city("Paris, France")["lat"], 48.85)

    def test_nasa_empty_data_is_actionable(self):
        response = Mock()
        response.json.return_value = {"properties": {"parameter": {"T2M": {"200001": -999, "200013": 20}}}}
        with patch("requests.get", return_value=response), self.assertRaisesRegex(ValueError, "no valid"):
            self.app["fetch_nasa_multivariable_data"](0, 0, 2000, 2001, ["T2M"])

    def test_firebase_http_error_not_success(self):
        result = self.engine(fixture())
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("403 forbidden")
        with patch("requests.put", return_value=response):
            status, packet = self.app["send_multivariable_udp_to_esp32"]("", 4210,
                {"meta": {"region_name": "Test"}, "intel": result}, "https://example.invalid")
        self.assertIn("FIREBASE SYNC ERROR", status)
        self.assertNotIn("SYNCHRONIZED", status)
        self.assertEqual(len(packet.split("|")), 7)


class StreamlitTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "APP.PY"), default_timeout=40)

    def test_startup_and_every_manual_target(self):
        at = self.app().run()
        self.assertFalse(at.exception)
        at.checkbox[0].check().run()
        for target in ("T2M", "PRECTOTCORR", "QV2M", "WS2M", "ALLSKY_SFC_SW_DWN", "GWETTOP"):
            next(w for w in at.selectbox if w.label == "Target Variable").select(target).run()
            self.assertFalse(at.exception, str(at.exception))

    def test_active_case_all_four_views_simulator_and_watchlist(self):
        ns = load_engine()
        result = ns["run_multivariable_intelligence_engine"](fixture(), "T2M", ["PRECTOTCORR", "GWETTOP", "QV2M"])
        meta = {"region_name": "Test region", "lat": 33.5, "lon": -7.5, "start_year": 2000, "end_year": 2011,
                "hypothesis": "Test", "router_source": "TEST", "target_var": "T2M", "driver_vars": ["PRECTOTCORR", "GWETTOP", "QV2M"]}
        case = {"meta": meta, "intel": result, "variables": ns["VARIABLES"]}
        at = self.app()
        at.session_state["active_case"] = case
        at.session_state["watchlist"] = [case, case]
        at.run()
        self.assertFalse(at.exception, str(at.exception))
        for option in at.radio[0].options:
            at.radio[0].set_value(option).run()
            self.assertFalse(at.exception, str(at.exception))
        next(s for s in at.slider if s.label.startswith("Perturb")).set_value(1.0).run()
        self.assertFalse(at.exception, str(at.exception))

    def test_demo_network_failure_stays_in_ui(self):
        with patch("requests.get", side_effect=requests.Timeout("offline")):
            at = self.app().run()
            next(b for b in at.button if b.label.startswith("Load Demo 1")).click().run()
            self.assertFalse(at.exception, str(at.exception))
            self.assertTrue(any("Demonstration unavailable" in e.value for e in at.error))

    def test_external_csv_survives_rerun_and_case_switch(self):
        data = fixture().rename(columns={"T2M": "CUSTOM_TEMP", "PRECTOTCORR": "CUSTOM_RAIN",
                                        "GWETTOP": "CUSTOM_SOIL", "QV2M": "CUSTOM_HUMIDITY"})
        def uploaded(*args, **kwargs):
            stream = io.BytesIO(data.to_csv(index=False).encode())
            stream.name = "custom.csv"
            return stream
        with patch("streamlit.file_uploader", side_effect=uploaded):
            at = self.app().run()
            next(w for w in at.text_input if w.label == "ESP32 Local IP Address").set_value("").run()
            next(b for b in at.button if b.label == "Analyze Uploaded CSV Dataset").click().run()
            self.assertFalse(at.exception, str(at.exception))
            self.assertIsNotNone(at.session_state["active_case"])
            self.assertEqual(at.session_state["active_case"]["intel"]["target_var"], "CUSTOM_TEMP")
            at.run()
            self.assertFalse(at.exception, str(at.exception))
            next(b for b in at.button if b.label.startswith("Load Case")).click().run()
            self.assertFalse(at.exception, str(at.exception))

    def test_constant_case_simulation_does_not_show_nan(self):
        ns = load_engine()
        result = ns["run_multivariable_intelligence_engine"](fixture(constant=True), "T2M", ["PRECTOTCORR", "GWETTOP", "QV2M"])
        meta = {"region_name": "Flat record", "lat": 0., "lon": 0., "start_year": 2000, "end_year": 2011,
                "hypothesis": "Constant record", "router_source": "TEST"}
        at = self.app()
        at.session_state["active_case"] = {"meta": meta, "intel": result, "variables": ns["VARIABLES"]}
        at.run()
        self.assertFalse(at.exception, str(at.exception))
        self.assertTrue(all("nan" not in str(m.value).lower() for m in at.metric))


if __name__ == "__main__":
    unittest.main()
