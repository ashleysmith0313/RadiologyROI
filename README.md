
# Radiology Locums ROI (Editable, Logic-First)

This Streamlit app shows the ROI of locums radiology reads using *logic only* — all numbers are editable in the UI.
Use it for client-facing demos without exposing real contract rates.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app_editable.py
```

## Deploy to Streamlit Cloud
1. Push this folder to a public GitHub repo.
2. On https://share.streamlit.io, click **New app**.
3. Choose your repo + branch, and set the main file to `app_editable.py`.
4. Click **Deploy**.

## Using the app
- Adjust ED visits, ED imaging rates per 100, non-ED multipliers.
- Set payer mix, bad-debt %, and the % of studies you would have completed without locums.
- Enter total locums spend, FTE-days, and reads per FTE-day.
- Edit the **Rate Table** inline (or upload a CSV) to set Medicare anchors, payer multipliers, contrast mix, and pro-share.
- Download a **JSON config** of everything to reuse; upload it later to prefill all fields.

## Files
- `app_editable.py` – the Streamlit app
- `sample_config.json` – example config you can load to prefill the app
- `radiology_roi_rates_template.csv` – starting point for a rates CSV (optional if you edit rates inline)
- `requirements.txt` – Python packages
- `.gitignore` – repo hygiene

