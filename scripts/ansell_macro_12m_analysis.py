
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "Projects/APACStockPitch"
INPUT = ROOT / "ansell_fred_data" / "Ansell_FRED_Macro_Data.xlsx"
OUTPUT = ROOT / "ansell_model" / "Ansell_Macro_12M_Analysis.xlsx"

df = pd.read_excel(INPUT, sheet_name="MONTHLY_LEVELS")

# Identify and normalize the date column.
date_col = next(
    c for c in df.columns
    if "date" in str(c).lower() or "observation" in str(c).lower()
)
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).set_index(date_col)

# Exclude November 2026: it may be an incomplete monthly observation.
df = df.loc[df.index < "2026-11-01"]

# Convert numeric columns.
df = df.apply(pd.to_numeric, errors="coerce")

# Percentage changes for index, price, commodity and FX series.
pct_series = [
    "INDPRO", "IPFINAL", "PAYEMS", "CES6562000001",
    "CPIAUCSL", "PPIACO", "DCOILWTICO", "DCOILBRENTEU",
    "DHHNGSP", "DEXUSEU", "DEXUSAL", "DEXMAUS",
    "DEXTHUS", "DEXINUS", "DEXCHUS", "DEXSLUS"
]

# 12-month percentage changes.
pct_data = {}
for col in pct_series:
    if col in df.columns:
        pct_data[col] = df[col].pct_change(12)

# 12-month absolute changes for rates and unemployment.
level_series = ["UNRATE", "FEDFUNDS", "DGS2", "DGS10", "BAA10Y"]
for col in level_series:
    if col in df.columns:
        pct_data[col + "_PP_CHANGE"] = df[col].diff(12)

changes = pd.DataFrame(pct_data, index=df.index)

# Distribution summary.
rows = []
for col in changes.columns:
    s = changes[col].dropna()
    if s.empty:
        continue

    rows.append({
        "Series": col,
        "Observations": len(s),
        "Mean": s.mean(),
        "Median": s.median(),
        "P05": s.quantile(0.05),
        "P25": s.quantile(0.25),
        "P75": s.quantile(0.75),
        "P95": s.quantile(0.95),
        "Minimum": s.min(),
        "Maximum": s.max(),
        "Latest": s.iloc[-1],
        "Latest_Date": s.index[-1],
    })

summary = pd.DataFrame(rows)

# Historical stress periods: worst and best 12-month observations.
stress_rows = []
for col in changes.columns:
    s = changes[col].dropna()
    if s.empty:
        continue

    stress_rows.append({
        "Series": col,
        "Worst_12M_Value": s.min(),
        "Worst_12M_Date": s.idxmin(),
        "Best_12M_Value": s.max(),
        "Best_12M_Date": s.idxmax(),
    })

stress = pd.DataFrame(stress_rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="12M_DISTRIBUTIONS", index=False)
    stress.to_excel(writer, sheet_name="STRESS_PERIODS", index=False)
    changes.to_excel(writer, sheet_name="12M_CHANGES")

print("\n12-MONTH DISTRIBUTIONS")
print(summary.to_string(index=False))

print("\nSTRESS PERIODS")
print(stress.to_string(index=False))

print(f"\nOUTPUT: {OUTPUT}")
