# CONCORD visual redesign

The presentation layer uses midnight blue glass surfaces, aurora mint controls, violet model series, and original stroke icons. The scientific pipeline, NASA acquisition, model training, telemetry functions, widget keys, and session state remain unchanged. `APP.PY` retains CRLF line endings; `concord_utils.py` is byte-identical to the saved baseline.

## Files

- `concord_theme.py`: CSS, theme injection, navigation, section headings, statistics, status chips, linear sparklines, schematic mini-map, 270-degree gauge, HTML tables, LED tiles, and footer.
- `APP.PY`: presentation integration across phases A–E, Plotly styling, and escaped dynamic HTML.
- `.streamlit/config.toml`: native widget theme and fonts.
- `tests/test_theme.py` and `tests/test_concord.py`: presentation integrity, contrast, and simulator interaction checks.
- `scripts/verify_visual.py`: exact numerical comparison with the locally saved pre-redesign source and NASA response in `artifacts/ui-baseline/`.
- `scripts/capture_ui.py`: local headless Edge screenshots using Python's standard library. Its URL must point to a running app with an active case; the local preview harness seeds the saved NASA case without network requests or device dispatch.

No pip dependencies were added. SVGs are embedded as self-contained image data URLs because this Streamlit version's HTML sanitizer removes inline SVG elements. No application JavaScript was added.

## Tokens

| Token | Value |
| --- | --- |
| `--bg-0`, `--bg-1` | `#060B1F`, `#0A1731` |
| `--surface`, `--surface-2`, `--surface-hover` | white at 5.5%, 9%, 12% |
| `--hairline` | white at 9% |
| `--text` | `#F5F7FF` |
| `--text-muted`, `--text-faint` | `rgba(235,240,255,.66)`, `rgba(235,240,255,.42)` |
| `--accent`, `--accent-ink` | `#5EEAD4`, `#04221F` |
| `--accent-2`, `--warn`, `--alert` | `#A78BFA`, `#FFB454`, `#FF7A7A` |
| `--series-4` | `#7DD3FC` |
| `--radius-card`, `--radius-inner`, `--radius-pill` | `24px`, `16px`, `999px` |
| `--shadow` | `0 10px 30px rgba(0,0,0,.25)` |
| `--space` | `8px` |

Page glows use teal `rgba(45,212,191,.16)` and violet `rgba(139,92,246,.14)`. Manrope supplies headings and numbers, Inter supplies prose and controls, and JetBrains Mono supplies statistics. Glass cards locally increase muted-text opacity to 75% for contrast on nested surfaces; the base token stays at 66%. Streamlit's additional caption opacity is disabled. Faint text is not used for required information.

## Verification

- All 24 unit and Streamlit interaction tests pass: `python -m unittest discover -s tests -v`.
- First load, both actual demo buttons, watchlist load/clear, CSV input, comparator, all four chart modes, and simulator updates are covered. Demo acquisition is replayed from a saved NASA response for repeatability.
- `python scripts/verify_visual.py` confirms exact equality of numerical engine outputs for both demos, unchanged protected function syntax trees, unchanged utility bytes, and CRLF preservation.
- The 1990–2025 temperature demo retains 36 years, slope `0.1942438867438867`, and OLS p-value `0.01626866286086482`. The 2015–2025 demo retains 11 years, slope `1.1903030303030304`, and OLS p-value `0.07478547997561118`.
- Headless Edge screenshots at 1440, 1024, and 390 pixels are in `artifacts/ui-screenshots/`. Checks report no page overflow, clipped cards, Streamlit exceptions, or broken SVG images. Significance grids render four, two, and one columns respectively. The lag table scrolls within its own container on narrow screens.
- Contrast tests cover muted text on nested glass surfaces; focus rings, 44-pixel control targets, and reduced-motion CSS are included. This is not a complete assistive-technology audit.

## Deliberate scope choices

Weather-specific day/month selectors, daylight markers, news content, and a decorative Chart/List toggle were omitted. CONCORD keeps its existing scientific controls and four functioning chart modes. All existing explanations, limitations, numerical statistics, and hardware trigger descriptions remain accessible. The hero's mini-map is explicitly schematic; Section VI retains the geographic Plotly map.

Validation did not send packets to physical hardware or exercise external Firebase writes. Those functions were preserved without changes.
