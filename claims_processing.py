from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Iterable, Dict

STATE_FIPS: Dict[str, str] = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11",
    "FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21",
    "LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30",
    "NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39",
    "OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49",
    "VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","PR":"72","VI":"78","GU":"66",
    "AS":"60","MP":"69"
}

def _paid_components(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["amountPaidOnBuildingClaim",
            "amountPaidOnContentsClaim",
            "amountPaidOnIncreasedCostOfComplianceClaim"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    df["paid_loss_nominal"] = df[cols].sum(axis=1)
    return df

def _build_county_fips(state_abbrev: pd.Series, county_code: pd.Series) -> pd.Series:
    """
    OpenFEMA v2 sometimes gives countyCode as 3-digit (county only) and sometimes
    as already-formed 5-digit FIPS; handle both robustly.
    """
    s2 = state_abbrev.map(STATE_FIPS).fillna("")
    cc = county_code.astype(str).str.strip()

    is5 = cc.str.fullmatch(r"\d{5}")
    out = pd.Series(index=cc.index, dtype="object")
    out.loc[is5]  = cc.loc[is5]
    out.loc[~is5] = (s2.loc[~is5] + cc.loc[~is5].str.zfill(3)).str.zfill(5)

    out = out.where(out.str.fullmatch(r"\d{5}"))
    return out

def clean_nfip_claims_openfema(nfip_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NFIP claims from the OpenFEMA FimaNfipClaims dataset.

    - Parses dateOfLoss -> year/month
    - Constructs county_fips from (state, countyCode)
    - Computes paid_loss_nominal = building + contents + ICC
    """
    df = nfip_raw.copy()

    # Date handling
    df["dateOfLoss"] = pd.to_datetime(df["dateOfLoss"], errors="coerce")
    df = df.dropna(subset=["dateOfLoss"])
    df["year"] = df["dateOfLoss"].dt.year
    df["month"] = df["dateOfLoss"].dt.month

    df = _paid_components(df)

    # Build county_fips
    df["county_fips"] = _build_county_fips(df.get("state"), df.get("countyCode"))

    county = df["county_fips"].astype(str)
    mask = county.str.fullmatch(r"\d{5}")
    mask = mask.fillna(False)
    df = df[mask].copy()

    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    return df

def clean_nfip_claims(claims_df: pd.DataFrame,
                      county_col: str = "county_fips",
                      date_col: str = "dateOfLoss",
                      paid_col: str = "paid_loss_nominal") -> pd.DataFrame:
    df = claims_df.copy()
    df[county_col] = df[county_col].astype(str).str.zfill(5)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if paid_col in df.columns:
        df[paid_col] = pd.to_numeric(df[paid_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["year"]  = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    return df

def attach_cpi_deflator(df: pd.DataFrame, cpi_df: pd.DataFrame,
                        year_col: str = "year", base_year: int = 2024,
                        out_col: str = "deflator") -> pd.DataFrame:
    """
    cpi_df columns: ['year','cpi_u'] (CPI-U).
    Produces a deflator such that real_loss = nominal_loss * deflator,
    where deflator = CPI(base_year) / CPI(year).

    Handles both the case where df has a 'year' column (year_col='year')
    and where the year column has a different name.
    """
    cpi = cpi_df[["year","cpi_u"]].dropna().copy()

    # Get CPI for base year
    base = float(cpi.loc[cpi["year"] == base_year, "cpi_u"].iloc[0])

    # deflator
    cpi[out_col] = base / cpi["cpi_u"]

    if year_col == "year":
        out = df.merge(cpi[["year", out_col]], on="year", how="left")
    else:
        out = df.merge(cpi[["year", out_col]], left_on=year_col, right_on="year", how="left")
        out = out.drop(columns=["year"]).rename(columns={year_col: "year"})

    return out


def claim_level_severity(claims_df: pd.DataFrame,
                         county_col: str = "county_fips",
                         date_col: str = "dateOfLoss",
                         paid_col: str = "paid_loss_nominal",
                         cpi_df: pd.DataFrame | None = None,
                         base_year: int = 2024) -> pd.DataFrame:
    """
    Returns claim-level rows with inflation-adjusted loss and log_loss:
    [county_fips, date_of_loss, year, month, loss_adj, log_loss]
    """
    df = claims_df[[county_col, date_col, paid_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"]  = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["loss_adj"] = pd.to_numeric(df[paid_col], errors="coerce").fillna(0.0)
    if cpi_df is not None:
        tmp = df[["year"]].drop_duplicates()
        tmp = attach_cpi_deflator(tmp, cpi_df, year_col="year", base_year=base_year)
        df = df.merge(tmp[["year","deflator"]], on="year", how="left")
        df["loss_adj"] = df["loss_adj"] * df["deflator"].fillna(1.0)
        df.drop(columns=["deflator"], inplace=True)
    df = df[df["loss_adj"] > 0].copy()
    df["log_loss"] = np.log(df["loss_adj"])
    df = df.rename(columns={date_col: "date_of_loss", county_col: "county_fips"})
    return df

def monthly_claim_counts(claims_df: pd.DataFrame,
                         county_col: str = "county_fips",
                         date_col: str = "dateOfLoss") -> pd.DataFrame:
    df = claims_df[[county_col, date_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"]  = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    out = (df.groupby([county_col, "year", "month"], as_index=False)
              .size().rename(columns={"size": "count_claims"}))
    out[county_col] = out[county_col].astype(str).str.zfill(5)
    return out

def merge_panel(monthly_weather: pd.DataFrame,
                monthly_entropy: pd.DataFrame,
                monthly_claims: pd.DataFrame,
                exposure_df: pd.DataFrame | None = None,
                county_col: str = "county_fips") -> pd.DataFrame:
    keys = [county_col, "year", "month"]
    panel = (monthly_weather.merge(monthly_entropy, on=keys, how="left")
                           .merge(monthly_claims, on=keys, how="left"))
    panel["count_claims"] = panel["count_claims"].fillna(0).astype(int)
    if exposure_df is not None:
        ex = exposure_df.copy()
        ex[county_col] = ex[county_col].astype(str).str.zfill(5)
        panel = panel.merge(ex[keys + ["exposure"]], on=keys, how="left")
    else:
        panel["exposure"] = np.nan
    # safe log offset
    panel["log_exposure"] = np.log(panel["exposure"].clip(lower=1)).where(panel["exposure"].notna())
    return panel
