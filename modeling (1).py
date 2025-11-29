from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns and (("year" not in df.columns) or ("month" not in df.columns)):
        d = pd.to_datetime(df["date"])
        df["year"] = d.dt.year
        df["month"] = d.dt.month
    return df

def standardize_after_split(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mu, sd = train[cols].mean(), train[cols].std(ddof=0).replace(0, 1)
    train = train.copy(); test = test.copy()
    for c in cols:
        train[c + "_std"] = (train[c] - mu[c]) / sd[c]
        test [c + "_std"] = (test [c] - mu[c]) / sd[c]
    return train, test, mu, sd

def fit_zinb(df: pd.DataFrame,
             y: str = "count_claims",
             x_vars: List[str] = None,
             infl_vars: List[str] = None,
             offset_col: Optional[str] = "log_exposure",
             county_fe: bool = False,
             time_fe: bool = True,
             state_fe: bool = True,
             maxiter: int = 200) -> Dict:
    """
    Fit ZINB (NB2), exposure offset (log_exposure), and optional FE via dummies.
    Returns dict with model, result, design matrices, and formulas.
    """
    df = add_time_parts(df.copy())
    x_vars = x_vars or ["total_mm_std","rx1day_mm_std","wet_days_std","sdii_mm_per_wetday_std","H_shannon_std","H_perm_std"]
    infl_vars = infl_vars or ["rx1day_mm_std","H_shannon_std","H_perm_std"]

    # FE via pandas.get_dummies (memory-safe on sub-samples)
    rhs_fe_parts = []
    if time_fe:
        rhs_fe_parts += ["C(year)", "C(month)"]
    if state_fe:
        rhs_fe_parts += ["C(state)"]
    if county_fe:
        rhs_fe_parts += ["C(county_fips)"]
    fe_rhs = " + ".join(rhs_fe_parts)

    count_rhs = " + ".join(x_vars + ([fe_rhs] if fe_rhs else []))
    infl_rhs  = " + ".join(infl_vars + ([fe_rhs] if fe_rhs else []))

    ymat, X = patsy.dmatrices(f"{y} ~ {count_rhs}", df, return_type="dataframe")
    _,   Z  = patsy.dmatrices(f"{y} ~ {infl_rhs}",  df, return_type="dataframe")

    # offset
    offset = None
    if offset_col and offset_col in df.columns:
        off = df[offset_col]
        if off.notna().all():
            offset = off.values

    model = ZeroInflatedNegativeBinomialP(
        endog=ymat.iloc[:, 0],
        exog=X,
        exog_infl=Z,
        inflation="logit",
        p=2,
        offset=offset,
    )
    res = model.fit(method="bfgs", maxiter=maxiter, disp=False)
    return {"model": model, "result": res, "X": X, "Z": Z, "offset": offset,
            "formula_count": count_rhs, "formula_infl": infl_rhs}

def zinb_predict_means(fit: Dict, df_new: pd.DataFrame) -> np.ndarray:
    df_new = add_time_parts(df_new.copy())
    Xn = patsy.dmatrix(fit["formula_count"], df_new, return_type="dataframe")
    Zn = patsy.dmatrix(fit["formula_infl"],  df_new, return_type="dataframe")
    offset = df_new.get("log_exposure", None)
    return fit["result"].predict(exog=Xn, exog_infl=Zn, which="mean", offset=offset)

def fit_quantiles(df_claims: pd.DataFrame, y: str = "log_loss",
                  x_vars: List[str] = None, taus: List[float] = [0.5,0.75,0.9,0.95],
                  county_fe: bool = False, time_fe: bool = True, state_fe: bool = True,
                  cluster_col: Optional[str] = "county_fips",
                  n_boot: int = 200, random_state: int = 42):
    """
    Quantile regressions with FE and (optional) simple cluster bootstrap of SEs.
    Returns dict: tau -> fitted result (with bse possibly replaced by bootstrap SD).
    """
    import numpy as np
    rng = np.random.default_rng(random_state)
    x_vars = x_vars or ["total_mm_std","rx1day_mm_std","wet_days_std","sdii_mm_per_wetday_std","H_shannon_std","H_perm_std","count_claims_std"]
    fe_parts = []
    if time_fe:  fe_parts += ["C(year)", "C(month)"]
    if state_fe: fe_parts += ["C(state)"]
    if county_fe: fe_parts += ["C(county_fips)"]
    rhs = " + ".join(x_vars + fe_parts)
    formula = f"{y} ~ {rhs}"

    models = {}
    for q in taus:
        res = smf.quantreg(formula, df_claims).fit(q=q)
        if cluster_col and cluster_col in df_claims.columns and n_boot > 0:
            clusters = df_claims[cluster_col].unique()
            boot = []
            for _ in range(n_boot):
                samp = rng.choice(clusters, size=len(clusters), replace=True)
                boot_df = pd.concat([df_claims[df_claims[cluster_col]==c] for c in samp], ignore_index=True)
                boot_res = smf.quantreg(formula, boot_df).fit(q=q)
                boot.append(boot_res.params.values)
            boot = np.vstack(boot)
            res.bse = pd.Series(boot.std(axis=0, ddof=1), index=res.params.index)
        models[q] = res
    return models
