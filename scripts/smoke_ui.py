"""Run the actual dashboard in AppTest after a presentation change."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from streamlit.testing.v1 import AppTest
from test_concord import fixture, load_engine


def smoke():
    app = AppTest.from_file(str(ROOT / 'APP.PY'), default_timeout=60).run()
    assert not app.exception, app.exception
    ns = load_engine()
    intel = ns['run_multivariable_intelligence_engine'](fixture(), 'T2M', ['PRECTOTCORR', 'GWETTOP', 'QV2M'])
    meta = dict(region_name='Casablanca preview', lat=33.57, lon=-7.59, start_year=2000, end_year=2011,
                hypothesis='Presentation check', router_source='LOCAL TEST', target_var='T2M',
                driver_vars=['PRECTOTCORR', 'GWETTOP', 'QV2M'])
    case = dict(meta=meta, intel=intel, variables=ns['VARIABLES'])
    app.session_state['active_case'] = case
    app.session_state['watchlist'] = [case, case]
    app.run()
    assert not app.exception, app.exception
    print('PASS: actual app first load, active case, and comparator render without exceptions.')


if __name__ == '__main__':
    smoke()
