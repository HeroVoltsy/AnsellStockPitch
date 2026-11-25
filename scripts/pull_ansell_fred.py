"""
Ansell APAC stock pitch — FRED macro data downloader
----------------------------------------------------
Downloads selected FRED series, preserves native frequencies, and exports:
  1) fred_raw_<SERIES_ID>.csv
  2) fred_monthly_panel.csv
  3) fred_quarterly_panel.csv
  4) fred_series_manifest.csv
  5) fred_download_log.csv
  6) Ansell_FRED_Macro_Data.xlsx

Setup:
  pip install pandas requests openpyxl
  export FRED_API_KEY="YOUR_32_CHARACTER_KEY"   # macOS/Linux
  set FRED_API_KEY=YOUR_32_CHARACTER_KEY        # Windows CMD
  $env:FRED_API_KEY="YOUR_32_CHARACTER_KEY"    # PowerShell

Run:
  python pull_ansell_fred.py

Notes:
- FRED API requires a free API key.
- Daily series are averaged to monthly frequency; monthly series remain monthly.
- Quarterly series are kept quarterly and are NOT forward-filled into monthly data.
- Missing/unavailable series are logged. Verify each series' definition and units
  on FRED before using it in a causal model.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    raise EnvironmentError(
        "FRED_API_KEY not found. Check your .env file."
    )


API_BASE = "https://api.stlouisfed.org/fred"
START_DATE = "2015-01-01"
END_DATE = None  # None = latest available observation
OUTDIR = Path("ansell_fred_data")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Grouped by economic transmission channel.
# Some FX candidates may not be available/current in FRED; failures are logged.
SERIES: dict[str, dict[str, str]] = {
    # Demand / industrial activity
    "INDPRO": {"group": "demand_us", "description": "US Industrial Production Index"},
    "IPFINAL": {"group": "demand_us", "description": "US Industrial Production: Final Products"},
    "PAYEMS": {"group": "demand_us", "description": "US Total Nonfarm Payrolls"},
    "CES6562000001": {"group": "healthcare_us", "description": "US Health Care and Social Assistance Employment"},
    "UNRATE": {"group": "demand_us", "description": "US Unemployment Rate"},
    "GDPC1": {"group": "demand_quarterly", "description": "US Real GDP"},

    # Price / input-cost proxies (not Ansell-specific procurement prices)
    "CPIAUCSL": {"group": "prices", "description": "US CPI: All Urban Consumers"},
    "PPIACO": {"group": "prices", "description": "US PPI: All Commodities"},
    "DCOILWTICO": {"group": "energy", "description": "WTI Crude Oil Spot Price"},
    "DCOILBRENTEU": {"group": "energy", "description": "Brent Crude Oil Spot Price"},
    "DHHNGSP": {"group": "energy", "description": "Henry Hub Natural Gas Spot Price"},

    # FX: quote conventions differ by series; inspect FRED metadata before interpretation.
    "DEXUSEU": {"group": "fx", "description": "US Dollars to Euro (USD per EUR)"},
    "DEXUSAL": {"group": "fx", "description": "US Dollars to Australian Dollar (USD per AUD)"},
    "DEXMAUS": {"group": "fx_candidate", "description": "Malaysia Ringgit per US Dollar candidate"},
    "DEXTHUS": {"group": "fx_candidate", "description": "Thailand Baht per US Dollar candidate"},
    "DEXINUS": {"group": "fx_candidate", "description": "India Rupees per US Dollar candidate"},
    "DEXCHUS": {"group": "fx_candidate", "description": "China Yuan per US Dollar candidate"},
    "DEXSLUS": {"group": "fx_candidate", "description": "Sri Lanka Rupees per US Dollar candidate"},

    # Financing conditions / discount-rate context
    "FEDFUNDS": {"group": "rates", "description": "Federal Funds Effective Rate"},
    "DGS2": {"group": "rates", "description": "2-Year US Treasury Constant Maturity Rate"},
    "DGS10": {"group": "rates", "description": "10-Year US Treasury Constant Maturity Rate"},
    "BAA10Y": {"group": "credit", "description": "Moody's Seasoned Baa Corporate Bond Yield Relative to 10-Year Treasury"},
}

def fred_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY environment variable is missing. "
            "Register for a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    p = dict(params)
    p.update({"api_key": api_key, "file_type": "json"})
    response = requests.get(f"{API_BASE}/{endpoint}", params=p, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if "error_code" in payload:
        raise RuntimeError(f"FRED API error {payload.get('error_code')}: {payload.get('error_message')}")
    return payload

def get_series_metadata(series_id: str) -> dict[str, Any]:
    payload = fred_get("series", {"series_id": series_id})
    return payload["seriess"][0]

def get_observations(series_id: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "observation_start": START_DATE,
        "sort_order": "asc",
        "limit": 100000,
    }
    if END_DATE:
        params["observation_end"] = END_DATE

    payload = fred_get("series/observations", params)
    obs = pd.DataFrame(payload.get("observations", []))
    if obs.empty:
        return pd.DataFrame(columns=["date", series_id])

    obs = obs[["date", "value"]].copy()
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    obs[series_id] = pd.to_numeric(obs["value"].replace(".", pd.NA), errors="coerce")
    obs = obs.drop(columns="value").dropna(subset=["date"])
    return obs.sort_values("date").drop_duplicates("date")

def main() -> None:
    manifest_rows = []
    log_rows = []
    downloaded: dict[str, pd.DataFrame] = {}
    metadata_by_id: dict[str, dict[str, Any]] = {}

    for i, (series_id, info) in enumerate(SERIES.items(), start=1):
        print(f"[{i}/{len(SERIES)}] Downloading {series_id} ...")
        try:
            meta = get_series_metadata(series_id)
            data = get_observations(series_id)
            if data.empty or data[series_id].notna().sum() == 0:
                raise ValueError("No numeric observations returned.")

            downloaded[series_id] = data
            metadata_by_id[series_id] = meta
            data.to_csv(OUTDIR / f"fred_raw_{series_id}.csv", index=False)

            manifest_rows.append({
                "series_id": series_id,
                "group": info["group"],
                "description_model": info["description"],
                "fred_title": meta.get("title"),
                "frequency": meta.get("frequency"),
                "units": meta.get("units"),
                "seasonal_adjustment": meta.get("seasonal_adjustment"),
                "observation_start": meta.get("observation_start"),
                "observation_end": meta.get("observation_end"),
                "last_updated": meta.get("last_updated"),
                "n_observations": int(data[series_id].notna().sum()),
                "status": "downloaded",
            })
            log_rows.append({"series_id": series_id, "status": "success", "message": ""})
            print(f"    OK — {len(data):,} rows; {meta.get('frequency')}; {meta.get('units')}")
        except Exception as exc:
            manifest_rows.append({
                "series_id": series_id,
                "group": info["group"],
                "description_model": info["description"],
                "fred_title": None,
                "frequency": None,
                "units": None,
                "seasonal_adjustment": None,
                "observation_start": None,
                "observation_end": None,
                "last_updated": None,
                "n_observations": 0,
                "status": "failed",
            })
            log_rows.append({"series_id": series_id, "status": "failed", "message": str(exc)})
            print(f"    FAILED — {exc}")
        time.sleep(0.15)

    manifest = pd.DataFrame(manifest_rows)
    log = pd.DataFrame(log_rows)
    manifest.to_csv(OUTDIR / "fred_series_manifest.csv", index=False)
    log.to_csv(OUTDIR / "fred_download_log.csv", index=False)

    # Build separate panels by native frequency.
    monthly_parts = []
    quarterly_parts = []

    for series_id, data in downloaded.items():
        meta = metadata_by_id[series_id]
        freq = str(meta.get("frequency", "")).lower()
        s = data.set_index("date")[series_id].sort_index()

        if "daily" in freq or "weekly" in freq:
            s = s.resample("MS").mean()
            monthly_parts.append(s.rename(series_id))
        elif "monthly" in freq:
            s.index = s.index.to_period("M").to_timestamp()
            monthly_parts.append(s.rename(series_id))
        elif "quarterly" in freq:
            s.index = s.index.to_period("Q").to_timestamp(how="end").normalize()
            quarterly_parts.append(s.rename(series_id))
        elif "annual" in freq:
            # Keep annual observations out of the monthly/quarterly panels.
            annual = s.rename(series_id).to_frame()
            annual.to_csv(OUTDIR / f"fred_annual_{series_id}.csv")

    monthly = pd.concat(monthly_parts, axis=1).sort_index() if monthly_parts else pd.DataFrame()
    quarterly = pd.concat(quarterly_parts, axis=1).sort_index() if quarterly_parts else pd.DataFrame()

    if not monthly.empty:
        monthly.index.name = "date"
        monthly.to_csv(OUTDIR / "fred_monthly_panel.csv")
        # Derived changes are separate columns; raw levels remain intact.
        changes = pd.DataFrame(index=monthly.index)
        for col in monthly.columns:
            x = monthly[col]
            changes[f"{col}_mom_pct"] = x.pct_change() * 100
            changes[f"{col}_yoy_pct"] = x.pct_change(12) * 100
            changes[f"{col}_yoy_log_pct"] = (100 * (x / x.shift(12)).apply(
                lambda v: __import__("math").log(v) if pd.notna(v) and v > 0 else float("nan")
            ))
        changes.index.name = "date"
        changes.to_csv(OUTDIR / "fred_monthly_changes.csv")

    if not quarterly.empty:
        quarterly.index.name = "date"
        quarterly.to_csv(OUTDIR / "fred_quarterly_panel.csv")

    # Excel delivery: native frequency panels + manifest + log + all raw series.
    xlsx_path = OUTDIR / "Ansell_FRED_Macro_Data.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if not monthly.empty:
            monthly.to_excel(writer, sheet_name="MONTHLY_LEVELS")
        if not quarterly.empty:
            quarterly.to_excel(writer, sheet_name="QUARTERLY_LEVELS")
        if not monthly.empty:
            changes.to_excel(writer, sheet_name="MONTHLY_CHANGES")
        manifest.to_excel(writer, sheet_name="SERIES_MANIFEST", index=False)
        log.to_excel(writer, sheet_name="DOWNLOAD_LOG", index=False)
        for series_id, data in downloaded.items():
            sheet = f"RAW_{series_id}"[:31]
            data.to_excel(writer, sheet_name=sheet, index=False)

    print("\nCompleted.")
    print(f"Output folder: {OUTDIR.resolve()}")
    print(f"Workbook: {xlsx_path.resolve()}")
    print(f"Downloaded: {len(downloaded)} / {len(SERIES)} series")
    print("Review SERIES_MANIFEST and DOWNLOAD_LOG before modelling.")

if __name__ == "__main__":
    main()
