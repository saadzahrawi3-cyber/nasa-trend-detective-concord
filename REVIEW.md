# Concord merge and bug review

`APP.PY` is the merged application. `APP2.py` remains unchanged as the teammate's reference version. Existing unrelated working-tree edits were preserved.

## Merge decisions

The first application already covered more of the challenge: the biome questionnaire, variable definitions, record-length estimator, four chart modes, walkthrough, CSV import/export, and monochrome editorial styling. The second application contributed useful city lookup and explicit date parsing, but shared the same statistical engine and several of its defects. Its colored interface and mandatory home-city screen were not carried over.

The merge adapts city lookup and date parsing, adds reusable validation in `concord_utils.py`, and retains all six dashboard sections, the watchlist, the model simulator, and the seven-field ESP32 protocol.

## Fixed problems

| Area | Original problem | Result |
| --- | --- | --- |
| CSV execution | Hardware variables were referenced before assignment. | Hardware controls initialize before CSV execution. |
| CSV dates | Missing timestamps were fabricated starting in 1990. | Requires real dates or year/month columns. |
| CSV variables | Two measurements could enter a four-variable engine. | Requires exactly one target and three distinct drivers; columns are explicitly selectable. |
| CSV reruns | Custom variable definitions disappeared after `st.rerun()`. | Each saved case carries its own variable catalog. |
| Manual target | Default driver list could include the selected target and crash Streamlit. | Defaults exclude the target, with separate widget state and three-driver validation. |
| Router | Named regions such as Cairo and Chesapeake could map to a different preset. Dates were ignored. | Matches catalog regions, preserves requested supported years, validates AI metadata, and honors explicit variables. |
| City lookup | A failed geocoder could silently return Casablanca; a home location could override an explicit place. | Explicit city field takes documented precedence; unknown or ambiguous cities require a clearer location or coordinates. |
| NASA ingestion | Empty responses failed during sorting; nonfinite values survived. | Actionable empty-data errors and finite-value validation. |
| Annual statistics | Partial years were treated as comparable full annual means. | Requires six complete years; excludes partial years from annual statistics and reports them. Monthly observations remain available for plotting. |
| Deseasonalization | Short and long demos used different reference climatologies. | NASA investigations use available 1990–2025 reference data; CSVs use their uploaded record. |
| Mann–Kendall | Ties were not corrected and S was not displayed. | Corrected variance, S/Z/p display, finite handling for constant records. |
| Sen slope | Missing years were counted as one-year intervals. | Uses actual calendar-year distances. |
| Lag scan | A row shift crossed missing months as though they were consecutive. | Reindexes to calendar months before testing lags. |
| Forecast training | Missing annual records created false next-year training pairs. | Trains only on consecutive complete years; requires four such pairs. |
| Constant records | Undefined regressions, correlations, z-scores, and simulator divisions. | Flat trend results, undefined correlations, zero standardized values, and disabled zero-variance perturbations. |
| Sufficiency estimate | Invented minimum extra years and an arbitrary 85-year cap. | Uses the stated OLS approximation; flat slopes return unavailable; disagreements with Mann–Kendall are explicit. |
| Explanation | A p-value was called the probability of random weather; correlations were called causes. | Corrected plain-language interpretation and exploratory labels. |
| CUSUM plot | The split year was included in the pre-fit on the plot but excluded in the engine. | Matching partitions for plotted and reported slopes. |
| Simulator | Reported change was measured against a prediction but described as change from the observation. | Baseline and change use the same unperturbed next-year prediction. Physical NASA bounds are applied. |
| Telemetry | Firebase HTTP errors were reported as success; UDP was called confirmed delivery. | Checks HTTP status, closes sockets on failure, and distinguishes sending from device acknowledgement. |
| LED baseline | Green could illuminate alongside another alert. | Green requires absence of every configured alert and target significance. |
| Demo comparison | Different regions and variables confounded the window-length demonstration. | Same Casablanca case for both periods; significance is computed rather than promised. |
| Layout | Narrow nested columns squeezed long chart labels; card heights were unreliable. | Full-width analysis sections and a responsive equal-height statistics grid, with four/two/one columns. |
| Theme | The documented theme lock did not exist; map and metric deltas could use colors. | Added tracked monochrome config, neutral deltas, grayscale map, and modern Streamlit width settings. |
| Watchlist | The last four saved cases restarted numbering at one. | Archive labels retain their actual case positions. |

1990–2025 spans 35 years between endpoints but includes 36 calendar annual means. Likewise 2015–2025 contains 11 annual means. The interface distinguishes those counts.

## Validation

Run the offline regression suite:

```powershell
python -m unittest discover -s tests -v
```

Tests exercise tied and constant series, incomplete records, invalid metadata, CSV date handling and reruns, failed NASA/Firebase requests, all manual targets, all four chart modes, the simulator, watchlist restoration, and demo failure handling. Tests extract functions from the monolithic entry point without executing its UI; Streamlit interaction tests execute the actual app with `AppTest`.

Run the optional read-only external-service check:

```powershell
python scripts/verify_live.py
```

Live results on 2026-11-14 at Casablanca (33.57, -7.59):

| Record | Valid monthly observations | Complete years | T2M OLS slope / decade | OLS p |
| --- | ---: | ---: | ---: | ---: |
| 1990–2025 | 432 | 36 | +0.19424 °C | 0.01627 |
| 2015–2025 | 132 | 11 | +1.19030 °C | 0.07479 |

The live response confirmed SB units of C, W m-2, g/kg, and m/s for the demo variables. Open-Meteo successfully resolved Casablanca, Morocco. No Gemini request, live Firebase write, or ESP32 packet was sent during verification. Browser CSS rendering and responsive breakpoints still need visual inspection; AppTest verifies Python-side rendering and widget behavior only.

## Scientific limits to address before claiming research-grade inference

- OLS and Mann–Kendall assume independent observations here. Serial-correlation corrections and multiple-comparison control are not implemented.
- The extra-years estimate is conditional on unchanged slope and variability and addresses OLS only. It cannot promise future significance or resolve Mann–Kendall disagreement.
- CUSUM selects an exploratory split, not an independently verified ecological tipping point. Acceleration ratios are unstable when the initial slope is near zero.
- The forecast ensemble and perturbation response are exploratory, with no time-blocked validation or prediction intervals. Feature importance is not causal attribution.
- NASA POWER's point endpoint returns gridded model/radiation products. A city coordinate does not establish an urban heat island, a basin-wide trend, tidal salinity, canopy temperature, or an area average. Those claims need additional measurements or spatial comparisons.
- Background paragraphs in the inherited catalog are hypotheses, not verified literature citations. They are labeled accordingly; attach real sources before presenting them as literature evidence.
- External CSV import supports normalized monthly tables, not every provider's raw export format. Daily exports, metadata preambles, unfamiliar missing-value flags, and unit conversion need preprocessing. NASA variable codes must use the catalog units.
- The archive window intentionally remains 1981–2025. Availability differs by variable; incomplete data are reported rather than invented. The fixed 1990–2025 climatology uses available monthly data and is not necessarily a complete reference for every location.
- The seven-field hardware packet retains its existing shape. Its forecast value refers to ten years after the last complete observation, which is only 2035 when the final observation is 2025. The firmware must share that convention.

The NASA source labels were aligned with the products used by the POWER point endpoint, rather than implying direct SMAP/MODIS/AIRS measurements. See [NASA POWER monthly API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/monthly/). City search uses the [Open-Meteo geocoding API](https://open-meteo.com/en/docs/geocoding-api).
