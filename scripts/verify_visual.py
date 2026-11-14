"""Compare the redesign against its local baseline using identical NASA data."""
import ast
import json
import sys
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts" / "ui-baseline"


def load(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"VARIABLES", "BIOME_QUESTIONNAIRE_CATALOG"} for t in node.targets):
            nodes.append(node)
    ns = {"OPENVINO_AVAILABLE": False}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def check_equal(left, right, path="intel"):
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            if key != "rf_model":
                check_equal(left[key], right[key], path + "." + key)
    elif isinstance(left, (float, np.floating)) and np.isnan(left):
        assert np.isnan(right), path
    else:
        assert left == right, path


def verify():
    original = load(BASE / "APP.PY")
    current = load(ROOT / "APP.PY")
    assert (BASE / "concord_utils.py").read_bytes() == (ROOT / "concord_utils.py").read_bytes()
    old_tree = ast.parse((BASE / "APP.PY").read_text(encoding="utf-8"))
    new_tree = ast.parse((ROOT / "APP.PY").read_text(encoding="utf-8"))
    old_funcs = {n.name: ast.dump(n) for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    new_funcs = {n.name: ast.dump(n) for n in new_tree.body if isinstance(n, ast.FunctionDef)}
    for name in old_funcs:
        if name != "apply_dark_monochrome_theme":
            assert old_funcs[name] == new_funcs[name], "Protected function changed: " + name
    def widget_keys(tree):
        return sorted(ast.dump(kw.value) for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute)
                      and node.func.attr in {'slider', 'button', 'text_input', 'selectbox', 'multiselect', 'radio', 'text_area'}
                      for kw in node.keywords if kw.arg == 'key')
    assert widget_keys(old_tree) == widget_keys(new_tree), 'Existing widget keys changed'
    def radio_options(tree):
        return [ast.dump(node.args[1]) for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'radio']
    assert radio_options(old_tree) == radio_options(new_tree), 'Underlying radio values changed'
    data_path = BASE / "nasa.csv"
    variables = ["T2M", "ALLSKY_SFC_SW_DWN", "QV2M", "WS2M"]
    if not data_path.exists():
        reference = original["fetch_nasa_multivariable_data"](33.57, -7.59, 1990, 2025, variables)
        reference.to_csv(data_path, index=False)
    reference = pd.read_csv(data_path, parse_dates=["date"])
    reports = []
    for start in (1990, 2015):
        frame = reference[reference.year >= start].copy()
        before = original["run_multivariable_intelligence_engine"](frame, "T2M", variables[1:], reference)
        after = current["run_multivariable_intelligence_engine"](frame, "T2M", variables[1:], reference)
        check_equal(before, after)
        reports.append({"start": start, "numeric_outputs": "exactly identical", "n_years": after["n_years"],
                        "slope": after["var_stats"]["T2M"]["slope_dec"], "ols_p": after["var_stats"]["T2M"]["ttest_p"]})
    # Execute the actual two demo buttons with the saved live response, without telemetry.
    response = Mock()
    response.json.return_value = {"properties": {"parameter": {
        v: {f"{r.year:04d}{r.month:02d}": getattr(r, v) for r in reference.itertuples()} for v in variables}}}
    with patch("requests.get", return_value=response):
        at = AppTest.from_file(str(ROOT / "APP.PY"), default_timeout=60).run()
        assert not at.exception, at.exception
        next(w for w in at.text_input if w.label == "ESP32 Local IP Address").set_value("").run()
        for prefix in ("Load Demo 1", "Load Demo 2"):
            next(b for b in at.button if b.label.startswith(prefix)).click().run()
            assert not at.exception, at.exception
            assert at.session_state["active_case"] is not None
        next(b for b in at.button if b.label == "Clear watchlist").click().run()
        assert not at.exception and not at.session_state["watchlist"]
    raw = (ROOT / "APP.PY").read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n"), "APP.PY must retain CRLF"
    (BASE / "verification.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print("Protected functions unchanged; concord_utils.py byte-identical; both demos and clear passed; CRLF preserved.")


if __name__ == "__main__":
    verify()
