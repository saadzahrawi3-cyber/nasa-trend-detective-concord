# The Concord Earth System Detective

The merged Team Concord application runs from `APP.PY`. `APP2.py` is retained as the original teammate version. See [REVIEW.md](REVIEW.md) for the comparison, fixes, validation results, and scientific limitations.

## Run

Use Python 3.12 or later with the declared dependencies:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run APP.PY
```

The black-and-white theme is configured in `.streamlit/config.toml`. Gemini is optional: put `GEMINI_API_KEY` in the local `.streamlit/secrets.toml` or enter it in the sidebar. The deterministic catalog router works without a key; NASA acquisition and arbitrary-city geocoding require internet access. Existing credentials should remain local.

Choose a guided biome/question/record window or enter a research prompt. The optional City, Country field overrides the prompt's location. Manual coordinates and variable settings apply to the direct-prompt action. Blank the ESP32 address to disable UDP; Firebase stays disabled unless a URL is supplied.

CSV uploads require one row per month, a `date` column or `year` and `month`, and four measurement columns. Select the target and three drivers. At least six complete years and four pairs of consecutive complete years are required for the complete analysis. Codes must contain letters, digits, or underscores, beginning with a letter. Use the displayed NASA units for recognized NASA codes; custom codes have unspecified units.

## Checks

```powershell
python -m unittest discover -s tests -v
python scripts/verify_live.py
```

The second command makes read-only requests to NASA POWER and Open-Meteo. Neither command needs an API key or a physical ESP32. OpenVINO remains optional and falls back to NumPy if unavailable.
