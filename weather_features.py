from __future__ import annotations
import numpy as np
import pandas as pd

def monthly_precip_features(
    daily_df: pd.DataFrame,
    county_col: str = "county_fips",
    date_col: str = "date",
    precip_col: str = "precip_mm",
    wet_threshold_mm: float = 1.0,
) -> pd.DataFrame:
    df = daily_df[[county_col, date_col, precip_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    # guard: clamp negatives to 0 before aggregations
    df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    df["year"]  = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["wet"]   = df[precip_col] >= wet_threshold_mm

    gb = df.groupby([county_col, "year", "month"], as_index=False)
    out = gb.agg(
        total_mm=(precip_col, "sum"),
        rx1day_mm=(precip_col, "max"),
        wet_days=("wet", "sum"),
    )
    p95 = (
        df.loc[df["wet"]]
          .groupby([county_col, "year", "month"], as_index=False)[precip_col]
          .apply(lambda x: np.nanpercentile(x, 95) if x.size else np.nan)
          .rename(columns={precip_col: "p95_mm"})
    )
    out = out.merge(p95, on=[county_col, "year", "month"], how="left")
    out["sdii_mm_per_wetday"] = out["total_mm"] / out["wet_days"].replace({0: np.nan})
    out["sdii_mm_per_wetday"] = out["sdii_mm_per_wetday"].fillna(0.0)
    out = out.sort_values([county_col, "year", "month"])
    out["total_mm_lag1"] = out.groupby(county_col)["total_mm"].shift(1)
    return out
