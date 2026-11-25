
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/home/voltsy/Projects/APACStockPitch")

FRED_FILE = (
    ROOT / "ansell_fred_data" /
    "Ansell_FRED_Macro_Data.xlsx"
)

ANSell_FILE = (
    ROOT / "Ansell_Causal_Valuation_Historical_Base_v0_1.xlsx"
)

OUTPUT_DIR = ROOT / "ansell_model"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "Ansell_Causal_Panel.xlsx"

# Set these to the actual financial workbook sheet names.
FINANCIAL_SHEET = "RAW_HISTORICAL"

# Ansell's financial reporting year ends June.
# FY2026 = July 2025 to June 2026.
FISCAL_YEAR_END_MONTH = 6


# ============================================================
# 1. LOAD FRED DATA
# ============================================================

def load_fred():

    monthly = pd.read_excel(
        FRED_FILE,
        sheet_name="MONTHLY_LEVELS"
    )

    changes = pd.read_excel(
        FRED_FILE,
        sheet_name="MONTHLY_CHANGES"
    )

    quarterly = pd.read_excel(
        FRED_FILE,
        sheet_name="QUARTERLY_LEVELS"
    )

    for frame in [monthly, changes, quarterly]:
        frame["date"] = pd.to_datetime(frame["date"])

    monthly = monthly.sort_values("date")
    changes = changes.sort_values("date")
    quarterly = quarterly.sort_values("date")

    return monthly, changes, quarterly


# ============================================================
# 2. LOAD ANSELL FINANCIAL DATA
# ============================================================

def load_financials():

    df = pd.read_excel(
        ANSell_FILE,
        sheet_name=FINANCIAL_SHEET
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    print("\nFINANCIAL DATA COLUMNS")
    print(df.columns.tolist())

    return df


# ============================================================
# 3. BUILD MONTHLY MACRO FEATURES
# ============================================================

def build_monthly_features(monthly, changes):

    df = monthly.merge(
        changes,
        on="date",
        how="left",
        validate="one_to_one"
    )

    # Demand
    df["industrial_demand_yoy"] = (
        df["INDPRO_yoy_pct"]
    )

    df["final_products_demand_yoy"] = (
        df["IPFINAL_yoy_pct"]
    )

    df["healthcare_employment_yoy"] = (
        df["CES6562000001_yoy_pct"]
    )

    df["employment_yoy"] = (
        df["PAYEMS_yoy_pct"]
    )

    # Input cost proxies
    df["oil_wti_yoy"] = (
        df["DCOILWTICO_yoy_pct"]
    )

    df["oil_brent_yoy"] = (
        df["DCOILBRENTEU_yoy_pct"]
    )

    df["natural_gas_yoy"] = (
        df["DHHNGSP_yoy_pct"]
    )

    df["ppi_yoy"] = (
        df["PPIACO_yoy_pct"]
    )

    df["cpi_yoy"] = (
        df["CPIAUCSL_yoy_pct"]
    )

    # FX changes.
    # Convert all FX series into a common interpretation:
    # positive = local currency appreciates against USD.
    #
    # DEXUSEU / DEXUSAL:
    # USD per unit of foreign currency -> already direct.
    #
    # DEXMAUS etc.:
    # local currency per USD -> invert direction.

    direct_fx = [
        "DEXUSEU",
        "DEXUSAL",
    ]

    inverse_fx = [
        "DEXMAUS",
        "DEXTHUS",
        "DEXINUS",
        "DEXCHUS",
        "DEXSLUS",
    ]

    for col in direct_fx:
        df[f"{col}_appreciation_yoy"] = (
            df[col].pct_change(12) * 100
        )

    for col in inverse_fx:
        df[f"{col}_appreciation_yoy"] = (
            (df[col].shift(12) / df[col] - 1) * 100
        )

    # Interest rates and spreads are level variables,
    # not percentage-growth variables.

    df["yield_curve_10y_2y"] = (
        df["DGS10"] - df["DGS2"]
    )

    df["credit_spread"] = (
        df["BAA10Y"] - df["DGS10"]
    )

    df["fed_funds_rate"] = df["FEDFUNDS"]

    # Volatility proxies: rolling standard deviation
    # of monthly returns.

    for col in [
        "DCOILWTICO",
        "DCOILBRENTEU",
        "DEXMAUS",
        "DEXTHUS",
        "DEXINUS",
        "DEXCHUS",
        "DEXSLUS",
        "DEXUSEU",
        "DEXUSAL",
    ]:

        returns = df[col].pct_change()

        df[f"{col}_vol_12m"] = (
            returns.rolling(12).std() * np.sqrt(12)
        )

    return df


# ============================================================
# 4. FISCAL PERIOD AGGREGATION
# ============================================================

def add_fiscal_period(df):

    df = df.copy()

    df["fiscal_year"] = (
        df["date"].dt.year
        + (
            df["date"].dt.month >
            FISCAL_YEAR_END_MONTH
        ).astype(int)
    )

    df["fiscal_half"] = np.where(
        df["date"].dt.month.isin([7, 8, 9, 10, 11, 12]),
        "H1",
        "H2"
    )

    df["period"] = (
        df["fiscal_year"].astype(str)
        + "_"
        + df["fiscal_half"]
    )

    return df


def aggregate_fiscal(df):

    df = add_fiscal_period(df)

    # Explicitly exclude raw date and identifiers.
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # Use means for rates, indices and monthly averages.
    # We retain separate volatility measures.
    grouped = (
        df.groupby(
            ["fiscal_year", "fiscal_half", "period"],
            as_index=False
        )[numeric_cols]
        .mean()
    )

    return grouped


# ============================================================
# 5. QUARTERLY GDP AGGREGATION
# ============================================================

def prepare_quarterly_gdp(quarterly):

    q = quarterly.copy()

    q["fiscal_year"] = (
        q["date"].dt.year
        + (
            q["date"].dt.month >
            FISCAL_YEAR_END_MONTH
        ).astype(int)
    )

    q["fiscal_half"] = np.where(
        q["date"].dt.month.isin([7, 8, 9, 10, 11, 12]),
        "H1",
        "H2"
    )

    q["period"] = (
        q["fiscal_year"].astype(str)
        + "_"
        + q["fiscal_half"]
    )

    # GDP is quarterly. Average the two quarterly
    # observations belonging to each fiscal half.
    q = (
        q.groupby("period", as_index=False)
        .agg(
            GDPC1=("GDPC1", "mean"),
            GDP_observations=("GDPC1", "count")
        )
    )

    return q


# ============================================================
# 6. LOAD AND NORMALIZE FINANCIAL DATA
# ============================================================

def normalize_financials(financials):

    df = financials.copy()

    # Historical workbook contains annual observations.
    # Identify year column.
    year_candidates = [
        "fiscal_year",
        "fy",
        "year",
    ]

    year_col = next(
        (
            col for col in year_candidates
            if col in df.columns
        ),
        None
    )

    if year_col is None:
        raise ValueError(
            "Could not identify fiscal year column. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["fiscal_year"] = pd.to_numeric(
        df[year_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=["fiscal_year"]
    )

    df["fiscal_year"] = (
        df["fiscal_year"].astype(int)
    )

    return df


# ============================================================
# 7. MERGE MACRO DATA WITH ANSELL FINANCIALS
# ============================================================

def merge_annual_macro(financials, macro):

    # Annual macro aggregation.
    annual = (
        macro.groupby("fiscal_year", as_index=False)
        .mean(numeric_only=True)
    )

    merged = financials.merge(
        annual,
        on="fiscal_year",
        how="left",
        validate="one_to_one"
    )

    return merged


# ============================================================
# 8. FINANCIAL FEATURE ENGINEERING
# ============================================================

def financial_features(df):

    df = df.sort_values(
        "fiscal_year"
    ).copy()

    # Find common historical fields.
    revenue_cols = [
        c for c in df.columns
        if "revenue" in c or "sales" in c
    ]

    ebit_cols = [
        c for c in df.columns
        if "ebit" in c
    ]

    # Derive growth for identified financial columns.
    for col in revenue_cols + ebit_cols:

        if pd.api.types.is_numeric_dtype(df[col]):

            df[f"{col}_growth_pct"] = (
                df[col].pct_change() * 100
            )

    # Identify likely segment revenue/EBIT fields.
    # Avoid automatically selecting ambiguous columns
    # for causal models.

    return df


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("Loading FRED data...")

    monthly, changes, quarterly = load_fred()

    print("Building monthly features...")

    macro_monthly = build_monthly_features(
        monthly,
        changes
    )

    print("Aggregating fiscal periods...")

    macro_half = aggregate_fiscal(
        macro_monthly
    )

    macro_gdp = prepare_quarterly_gdp(
        quarterly
    )

    macro_half = macro_half.merge(
        macro_gdp,
        on="period",
        how="left",
        validate="one_to_one"
    )

    print("Loading Ansell financials...")

    financials = load_financials()

    financials = normalize_financials(
        financials
    )

    print("Merging annual macro data...")

    macro_annual = (
        macro_half.groupby(
            "fiscal_year",
            as_index=False
        )
        .mean(numeric_only=True)
    )

    annual_panel = merge_annual_macro(
        financials,
        macro_annual
    )

    annual_panel = financial_features(
        annual_panel
    )

    # Export
    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        macro_monthly.to_excel(
            writer,
            sheet_name="MACRO_MONTHLY",
            index=False
        )

        macro_half.to_excel(
            writer,
            sheet_name="MACRO_HALF_YEAR",
            index=False
        )

        financials.to_excel(
            writer,
            sheet_name="ANSell_FINANCIALS",
            index=False
        )

        annual_panel.to_excel(
            writer,
            sheet_name="DATA",
            index=False
        )

    print("\n" + "=" * 70)
    print("CAUSAL PANEL BUILD COMPLETE")
    print("=" * 70)

    print("Monthly macro:", macro_monthly.shape)
    print("Half-year macro:", macro_half.shape)
    print("Annual panel:", annual_panel.shape)

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nDATA COLUMNS:")
    print(annual_panel.columns.tolist())


if __name__ == "__main__":
    main()
