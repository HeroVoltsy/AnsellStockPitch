
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. CONFIGURATION
# ============================================================

INPUT_FILE = (
    "/home/voltsy/Projects/APACStockPitch/"
    "ansell_fred_data/Ansell_FRED_Macro_Data.xlsx"
)
INPUT_SHEET = "DATA"

OUTPUT_DIR = Path("ansell_scm_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_COL = "date"

# Each specification:
# treatment = external driver X
# mediator  = operating mechanism M
# outcome   = financial outcome Y
# controls  = confounder proxies C
#
# Replace these names with the actual column names in your
# merged Ansell + FRED dataset.
#
# IMPORTANT:
# A proxy is not necessarily the actual business mechanism.
# Document the interpretation of each proxy.

SPECS = [
    {
        "name": "input_cost_to_gross_margin",
        "treatment": "input_cost_index",
        "mediator": "pricing_proxy",
        "outcome": "gross_margin",
        "controls": [
            "industrial_production",
            "fx_index",
        ],
        "description": (
            "Input cost pressure -> pricing response proxy -> "
            "gross margin"
        ),
    },
    {
        "name": "demand_to_segment_revenue",
        "treatment": "industrial_production",
        "mediator": "volume_mix_proxy",
        "outcome": "industrial_revenue_growth",
        "controls": [
            "input_cost_index",
            "fx_index",
        ],
        "description": (
            "Industrial demand -> volume/mix proxy -> "
            "Industrial revenue growth"
        ),
    },
    {
        "name": "fx_to_reported_revenue",
        "treatment": "fx_index",
        "mediator": "translation_proxy",
        "outcome": "reported_revenue_growth",
        "controls": [
            "industrial_production",
            "input_cost_index",
        ],
        "description": (
            "FX movements -> translation proxy -> "
            "reported revenue growth"
        ),
    },
]


# Model settings
HAC_MAXLAGS = 2
BOOTSTRAP_ITERATIONS = 2000
BLOCK_LENGTH = 4
RANDOM_SEED = 42

# Standardized effects make coefficients comparable across units.
STANDARDIZE = True

# Minimum observations after removing missing values.
MIN_OBSERVATIONS = 20


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    path = Path(INPUT_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path.resolve()}"
        )

    df = pd.read_excel(
        path,
        sheet_name=INPUT_SHEET
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def validate_spec(df, spec):

    required = [
        TIME_COL,
        spec["treatment"],
        spec["mediator"],
        spec["outcome"],
        *spec["controls"],
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{spec['name']}: Missing columns: {missing}"
        )


# ============================================================
# 2. PREPROCESSING
# ============================================================

def prepare_data(df, spec):

    validate_spec(df, spec)

    cols = [
        TIME_COL,
        spec["treatment"],
        spec["mediator"],
        spec["outcome"],
        *spec["controls"],
    ]

    data = df[cols].copy()

    data[TIME_COL] = pd.to_datetime(
        data[TIME_COL],
        errors="coerce"
    )

    data = (
        data
        .dropna(subset=[TIME_COL])
        .sort_values(TIME_COL)
        .drop_duplicates(subset=[TIME_COL])
        .reset_index(drop=True)
    )

    numeric_cols = [
        spec["treatment"],
        spec["mediator"],
        spec["outcome"],
        *spec["controls"],
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = (
        data
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=numeric_cols)
        .reset_index(drop=True)
    )

    if len(data) < MIN_OBSERVATIONS:
        raise ValueError(
            f"{spec['name']}: Only {len(data)} usable "
            f"observations. Minimum is {MIN_OBSERVATIONS}."
        )

    if STANDARDIZE:

        scaler = StandardScaler()

        data[numeric_cols] = scaler.fit_transform(
            data[numeric_cols]
        )

    return data


# ============================================================
# 3. STRUCTURAL EQUATION ESTIMATION
# ============================================================

def fit_equations(data, spec):

    x = spec["treatment"]
    m = spec["mediator"]
    y = spec["outcome"]
    controls = spec["controls"]

    # Mediator equation:
    # M = alpha + aX + controls + error

    Xm = sm.add_constant(
        data[[x] + controls],
        has_constant="add"
    )

    mediator_model = sm.OLS(
        data[m],
        Xm
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_MAXLAGS}
    )

    # Outcome equation:
    # Y = beta + c'X + bM + controls + error

    Xy = sm.add_constant(
        data[[x, m] + controls],
        has_constant="add"
    )

    outcome_model = sm.OLS(
        data[y],
        Xy
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_MAXLAGS}
    )

    # Total effect equation:
    # Y = tau + cX + controls + error

    Xt = sm.add_constant(
        data[[x] + controls],
        has_constant="add"
    )

    total_model = sm.OLS(
        data[y],
        Xt
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_MAXLAGS}
    )

    a = mediator_model.params[x]
    b = outcome_model.params[m]
    direct = outcome_model.params[x]

    indirect = a * b
    total = direct + indirect

    return {
        "a": a,
        "b": b,
        "direct": direct,
        "indirect": indirect,
        "total": total,
        "regression_total": total_model.params[x],
        "mediator_model": mediator_model,
        "outcome_model": outcome_model,
        "total_model": total_model,
    }


# ============================================================
# 4. MOVING-BLOCK BOOTSTRAP
# ============================================================

def moving_block_indices(n, block_length, rng):

    if n <= block_length:
        raise ValueError(
            "Sample must be longer than block length."
        )

    blocks_needed = int(
        np.ceil(n / block_length)
    )

    max_start = n - block_length

    starts = rng.integers(
        0,
        max_start + 1,
        size=blocks_needed
    )

    indices = []

    for start in starts:

        block = np.arange(
            start,
            start + block_length
        )

        indices.extend(block.tolist())

    return np.asarray(indices[:n])


def bootstrap_effects(data, spec):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    n = len(data)

    for _ in range(BOOTSTRAP_ITERATIONS):

        idx = moving_block_indices(
            n,
            BLOCK_LENGTH,
            rng
        )

        sample = (
            data.iloc[idx]
            .reset_index(drop=True)
        )

        try:

            estimates = fit_equations(
                sample,
                spec
            )

            results.append([
                estimates["direct"],
                estimates["indirect"],
                estimates["total"],
            ])

        except Exception:
            continue

    boot = np.asarray(results)

    if len(boot) < BOOTSTRAP_ITERATIONS * 0.8:
        warnings.warn(
            f"{spec['name']}: More than 20% of bootstrap "
            "iterations failed."
        )

    if len(boot) == 0:
        raise RuntimeError(
            f"{spec['name']}: Bootstrap failed."
        )

    columns = [
        "direct",
        "indirect",
        "total",
    ]

    output = {}

    for i, name in enumerate(columns):

        values = boot[:, i]

        output[name] = {
            "bootstrap_mean": np.mean(values),
            "ci_lower": np.percentile(values, 2.5),
            "ci_upper": np.percentile(values, 97.5),
            "bootstrap_sd": np.std(values, ddof=1),
        }

    return output, boot


# ============================================================
# 5. DIAGNOSTICS
# ============================================================

def calculate_diagnostics(estimates, data, spec):

    mediator_model = estimates["mediator_model"]
    outcome_model = estimates["outcome_model"]

    x = spec["treatment"]
    m = spec["mediator"]
    y = spec["outcome"]

    residuals = pd.DataFrame({
        "mediator_residual": mediator_model.resid,
        "outcome_residual": outcome_model.resid,
    })

    # Correlation between mediator and outcome residuals.
    # High correlation may indicate unobserved mediator-outcome
    # confounding or misspecification.
    residual_corr = residuals.corr().iloc[0, 1]

    # Variance inflation factors.
    from statsmodels.stats.outliers_influence import (
        variance_inflation_factor
    )

    design = sm.add_constant(
        data[[x, m] + spec["controls"]],
        has_constant="add"
    )

    vif = {}

    for i, col in enumerate(design.columns):

        if col == "const":
            continue

        vif[col] = variance_inflation_factor(
            design.values,
            i
        )

    return {
        "n": len(data),
        "mediator_r2": mediator_model.rsquared,
        "outcome_r2": outcome_model.rsquared,
        "mediator_f_stat": mediator_model.fvalue,
        "outcome_f_stat": outcome_model.fvalue,
        "residual_correlation": residual_corr,
        "vif": vif,
    }


# ============================================================
# 6. SENSITIVITY ANALYSIS
# ============================================================

def mediator_sensitivity(estimates):

    a = estimates["a"]
    b = estimates["b"]
    direct = estimates["direct"]

    return {
        "a": a,
        "b": b,
        "direct": direct,
        "indirect": a * b,
        "total": direct + a * b,
        "indirect_share": (
            (a * b) / (direct + a * b)
            if abs(direct + a * b) > 1e-10
            else np.nan
        ),
    }


# ============================================================
# 7. EXECUTE ONE SPECIFICATION
# ============================================================

def run_specification(df, spec):

    print("\n" + "=" * 75)
    print(spec["name"].upper())
    print("=" * 75)

    print(spec["description"])

    data = prepare_data(
        df,
        spec
    )

    estimates = fit_equations(
        data,
        spec
    )

    bootstrap, boot_samples = bootstrap_effects(
        data,
        spec
    )

    diagnostics = calculate_diagnostics(
        estimates,
        data,
        spec
    )

    sensitivity = mediator_sensitivity(
        estimates
    )

    print("\nEFFECT DECOMPOSITION")

    print(
        f"Direct effect:   "
        f"{estimates['direct']:.5f}"
    )

    print(
        f"Indirect effect: "
        f"{estimates['indirect']:.5f}"
    )

    print(
        f"Total effect:    "
        f"{estimates['total']:.5f}"
    )

    print(
        f"Regression total:"
        f" {estimates['regression_total']:.5f}"
    )

    print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS")

    for effect in [
        "direct",
        "indirect",
        "total",
    ]:

        result = bootstrap[effect]

        print(
            f"{effect}: "
            f"[{result['ci_lower']:.5f}, "
            f"{result['ci_upper']:.5f}]"
        )

    print("\nDIAGNOSTICS")

    for key, value in diagnostics.items():

        if key != "vif":
            print(f"{key}: {value}")

    print("\nVIF")

    for key, value in diagnostics["vif"].items():
        print(f"{key}: {value:.3f}")

    return {
        "spec": spec,
        "data": data,
        "estimates": estimates,
        "bootstrap": bootstrap,
        "bootstrap_samples": boot_samples,
        "diagnostics": diagnostics,
        "sensitivity": sensitivity,
    }


# ============================================================
# 8. EXPORT RESULTS FOR EXCEL
# ============================================================

def export_results(results):

    effect_rows = []
    diagnostic_rows = []
    coefficient_rows = []

    for result in results:

        name = result["spec"]["name"]
        estimates = result["estimates"]
        bootstrap = result["bootstrap"]
        diagnostics = result["diagnostics"]

        for effect in [
            "direct",
            "indirect",
            "total",
        ]:

            effect_rows.append({
                "specification": name,
                "effect": effect,
                "estimate": estimates[effect],
                "bootstrap_mean": bootstrap[effect][
                    "bootstrap_mean"
                ],
                "bootstrap_sd": bootstrap[effect][
                    "bootstrap_sd"
                ],
                "ci_lower_95": bootstrap[effect][
                    "ci_lower"
                ],
                "ci_upper_95": bootstrap[effect][
                    "ci_upper"
                ],
                "observations": diagnostics["n"],
                "standardized": STANDARDIZE,
                "interpretation": "SURROGATE MODEL ESTIMATE",
            })

        for equation_name in [
            "mediator_model",
            "outcome_model",
            "total_model",
        ]:

            model = estimates[equation_name]

            for variable in model.params.index:

                coefficient_rows.append({
                    "specification": name,
                    "equation": equation_name,
                    "variable": variable,
                    "coefficient": model.params[variable],
                    "std_error": model.bse[variable],
                    "p_value": model.pvalues[variable],
                    "ci_lower": model.conf_int().loc[
                        variable, 0
                    ],
                    "ci_upper": model.conf_int().loc[
                        variable, 1
                    ],
                })

        diagnostic_rows.append({
            "specification": name,
            "observations": diagnostics["n"],
            "mediator_r2": diagnostics["mediator_r2"],
            "outcome_r2": diagnostics["outcome_r2"],
            "residual_correlation": diagnostics[
                "residual_correlation"
            ],
            "indirect_share": result["sensitivity"][
                "indirect_share"
            ],
        })

    effects_df = pd.DataFrame(effect_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    coefficients_df = pd.DataFrame(coefficient_rows)

    with pd.ExcelWriter(
        OUTPUT_DIR / "Ansell_SCM_Results.xlsx",
        engine="openpyxl"
    ) as writer:

        effects_df.to_excel(
            writer,
            sheet_name="EFFECTS",
            index=False
        )

        coefficients_df.to_excel(
            writer,
            sheet_name="COEFFICIENTS",
            index=False
        )

        diagnostics_df.to_excel(
            writer,
            sheet_name="DIAGNOSTICS",
            index=False
        )

        # Store bootstrap draws for reproducibility.
        for result in results:

            name = result["spec"]["name"][:25]

            boot_df = pd.DataFrame(
                result["bootstrap_samples"],
                columns=[
                    "direct",
                    "indirect",
                    "total",
                ]
            )

            boot_df.to_excel(
                writer,
                sheet_name=f"BOOT_{name}",
                index=False
            )

    effects_df.to_csv(
        OUTPUT_DIR / "effects.csv",
        index=False
    )

    coefficients_df.to_csv(
        OUTPUT_DIR / "coefficients.csv",
        index=False
    )

    diagnostics_df.to_csv(
        OUTPUT_DIR / "diagnostics.csv",
        index=False
    )

    print("\nEXPORT COMPLETE")
    print(
        OUTPUT_DIR / "Ansell_SCM_Results.xlsx"
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    df = load_data()

    results = []

    for spec in SPECS:

        try:

            result = run_specification(
                df,
                spec
            )

            results.append(result)

        except Exception as exc:

            print(
                f"\nFAILED: {spec['name']}"
            )

            print(str(exc))

    if not results:
        raise RuntimeError(
            "No specifications completed successfully."
        )

    export_results(results)


if __name__ == "__main__":
    main()
