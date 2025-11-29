from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd

def shannon_entropy(intensities: np.ndarray, bin_edges: Iterable[float], eps: float = 1e-12) -> float:
    """
    Standard Shannon entropy for binned intensities, normalized by log(B).
    Caller is responsible for passing a bin grid whose first bin explicitly
    captures zero (e.g., [-inf, 0, 1, 5, ... , +inf]).
    """
    x = np.asarray(intensities, dtype=float)
    x = x[~np.isnan(x)]
    counts, _ = np.histogram(x, bins=np.asarray(list(bin_edges), dtype=float))
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    p = np.clip(p, eps, 1.0)  # avoid log(0)
    H = -(p * np.log(p)).sum()
    H /= np.log(len(counts)) if len(counts) > 1 else 1.0
    return float(H)

def compute_shannon_monthly(daily_df: pd.DataFrame,
                            bin_edges: Iterable[float],
                            county_col: str = "county_fips",
                            date_col:   str = "date",
                            precip_col: str = "precip_mm",
                            out_col:    str = "H_shannon") -> pd.DataFrame:
    df = daily_df[[county_col, date_col, precip_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"]  = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    rows = []
    for (c, y, m), sub in df.groupby([county_col, "year", "month"]):
        H = shannon_entropy(sub[precip_col].values, bin_edges=bin_edges)
        rows.append((c, y, m, H))
    return pd.DataFrame(rows, columns=[county_col, "year", "month", out_col])

def permutation_entropy(series: np.ndarray,
                        m: int = 4, tau: int = 1,
                        wet_only: bool = False,
                        jitter_scale: float = 1e-6,
                        seed: int = 42) -> float:
    """
    Permutation entropy with deterministic, small Gaussian jitter to break ties.
    - Normalized by log(m!)
    - If wet_only=True, zeros are dropped before computing PE.
    """
    x = np.asarray(series, dtype=float)
    if wet_only:
        x = x[x > 0]
    n = x.size
    if n < m + (m - 1) * tau:
        return np.nan
    # small deterministic jitter proportional to series scale
    rng = np.random.default_rng(seed)
    sig = np.nanstd(x)
    if sig == 0 or np.isnan(sig):
        sig = 1.0
    x = x + rng.normal(0, jitter_scale * sig, size=n)
    pats = []
    for i in range(n - (m - 1) * tau):
        w = x[i:i + m * tau:tau]
        pats.append(tuple(np.argsort(w)))
    if not pats:
        return np.nan
    from collections import Counter
    counts = np.array(list(Counter(pats).values()), dtype=float)
    p = counts / counts.sum()
    H = -(p * np.log(p)).sum()
    H /= np.log(np.math.factorial(m))
    return float(H)

def compute_permutation_monthly(daily_df: pd.DataFrame,
                                county_col: str = "county_fips",
                                date_col:   str = "date",
                                precip_col: str = "precip_mm",
                                m: int = 4, tau: int = 1,
                                wet_only: bool = False,
                                out_col: str = "H_perm") -> pd.DataFrame:
    df = daily_df[[county_col, date_col, precip_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"]  = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    rows = []
    for (c, y, mm), sub in df.groupby([county_col, "year", "month"]):
        H = permutation_entropy(sub.sort_values(date_col)[precip_col].values,
                                m=m, tau=tau, wet_only=wet_only)
        rows.append((c, y, mm, H))
    return pd.DataFrame(rows, columns=[county_col, "year", "month", out_col])
