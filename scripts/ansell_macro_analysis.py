
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "Projects" / "APACStockPitch"
INPUT = ROOT / "ansell_fred_data" / "Ansell_FRED_Macro_Data.xlsx"
OUTPUT = ROOT / "ansell_model" / "Ansell_Macro_Regime_Analysis.xlsx"

SERIES = [
    "INDPRO", "IPFINAL", "PAYEMS", "CES6562000001", "UNRATE",
    "CPIAUCSL", "PPIACO", "DCOILWTICO", "DCOILBRENT EU",
    "DHHNGSP", "DEXUSEU", "DEXUSAL", "DEXMAUS", "DEXTHUS",
    "DEXINUS", "DEXCHUS", "DEXSLUS", "FEDFUNDS", "DGS2",
    "DGS10", "BAA10Y"
]

def load_monthly(path):
    df = pd.read_excel(path, sheet_name="MONTHLY_LEVELS")
    df.columns = [str(c).strip() for c in df.columns]

    date_col = next(
        (c for c in df.columns if c.lower() in {"date", "observation_date"}),
        df.columns[0]
    )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")

    return df.loc[:, ~df.columns.duplicated()]

def summarize(df):
    available = [c for c in SERIES if c in df.columns]
    selected = df[available].copy()

    changes = selected.pct_change(fill_method=None)
    changes = changes.replace([np.inf, -np.inf], np.nan)

    # Interest rates and unemployment are more interpretable
    # as percentage-point changes than percentage changes.
    for col in ["UNRATE", "FEDFUNDS", "DGS2", "DGS10", "BAA10Y"]:
        if col in selected.columns:
            changes[col] = selected[col].diff()

    rows = []

    for col in available:
        x = selected[col].dropna()
        dx = changes[col].dropna()

        if x.empty:
            continue

        rows.append({
            "Series": col,
            "Observations": len(x),
            "Start": x.index.min(),
            "End": x.index.max(),
            "Level_Mean": x.mean(),
            "Level_Median": x.median(),
            "Level_Min": x.min(),
            "Level_Max": x.max(),
            "Change_Mean": dx.mean(),
            "Change_SD": dx.std(),
            "Change_P05": dx.quantile(0.05),
            "Change_P25": dx.quantile(0.25),
            "Change_P50": dx.quantile(0.50),
            "Change_P75": dx.quantile(0.75),
            "Change_P95": dx.quantile(0.95),
        })

    return selected, changes, pd.DataFrame(rows)

def regime_summary(df):
    # Regimes are descriptive, not estimated causal states.
    output = []

    for col in ["INDPRO", "IPFINAL", "PPIACO",
                "CPIAUCSL", "DCOILWTICO", "DCOILBRENT EU"]:
        if col not in df.columns:
            continue

        x = df[col].dropna()
        if len(x) < 24:
            continue

        change = x.pct_change(fill_method=None).dropna()

        output.append({
            "Series": col,
            "Negative_Months": int((change < 0).sum()),
            "Positive_Months": int((change > 0).sum()),
            "Negative_Month_Share": (change < 0).mean(),
            "Positive_Month_Share": (change > 0).mean(),
            "Worst_Month": change.min(),
            "Best_Month": change.max(),
            "P05_Monthly_Change": change.quantile(0.05),
            "P95_Monthly_Change": change.quantile(0.95),
            "Worst_12M_Change": x.pct_change(12).min(),
            "Best_12M_Change": x.pct_change(12).max(),
        })

    return pd.DataFrame(output)

def rolling_volatility(changes, window=12):
    return changes.rolling(window).std() * np.sqrt(12)

def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    monthly = load_monthly(INPUT)
    levels, changes, stats = summarize(monthly)
    regimes = regime_summary(levels)
    volatility = rolling_volatility(changes)

    missingness = (
        monthly.isna().mean()
        .rename("Missing_Share")
        .reset_index()
        .rename(columns={"index": "Series"})
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        levels.to_excel(writer, sheet_name="LEVELS")
        changes.to_excel(writer, sheet_name="MONTHLY_CHANGES")
        stats.to_excel(writer, sheet_name="SERIES_STATISTICS", index=False)
        regimes.to_excel(writer, sheet_name="REGIME_RANGES", index=False)
        volatility.to_excel(writer, sheet_name="ROLLING_VOLATILITY")
        missingness.to_excel(writer, sheet_name="MISSINGNESS", index=False)

    print("\nINPUT:", INPUT)
    print("MONTHLY DATA:", monthly.shape)
    print("\nAVAILABLE SERIES:")
    print(list(monthly.columns))
    print("\nSERIES STATISTICS:")
    print(stats.to_string(index=False))
    print("\nREGIME RANGES:")
    print(regimes.to_string(index=False))
    print("\nOUTPUT:", OUTPUT)

if __name__ == "__main__":
    main()
