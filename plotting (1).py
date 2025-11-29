from __future__ import annotations
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from io_utils import save_fig

def hist_series(s: pd.Series, title: str, filename: str, bins: int = 40):
    plt.figure()
    s.plot(kind="hist", bins=bins)
    plt.title(title); plt.xlabel(s.name)
    save_fig(None, filename); plt.close()

def map_counties(county_gdf: gpd.GeoDataFrame, value_col: str, title: str, filename: str):
    plt.figure(figsize=(10,6))
    county_gdf.plot(column=value_col, legend=True, linewidth=0.1, edgecolor="none")
    plt.title(title); plt.axis("off")
    save_fig(None, filename); plt.close()

def coefplot_from_result(result, title: str, filename: str, keep=None):
    import pandas as pd, matplotlib.pyplot as plt
    params = result.params; conf = result.conf_int()
    df = pd.DataFrame({"param": params.index, "coef": params.values, "lo": conf[0].values, "hi": conf[1].values})
    if keep is not None: df = df[df["param"].isin(keep)]
    df = df.sort_values("coef")
    plt.figure(figsize=(6, max(4, 0.3*len(df))))
    plt.hlines(range(len(df)), df["lo"], df["hi"])
    plt.plot(df["coef"], range(len(df)), "o"); plt.axvline(0, color="k", lw=1)
    plt.yticks(range(len(df)), df["param"]); plt.title(title)
    save_fig(None, filename); plt.close()

def mi_barplot(mi_dict: dict, title: str, filename: str):
    plt.figure(figsize=(6,4))
    plt.bar(list(mi_dict.keys()), list(mi_dict.values()))
    plt.ylabel("Mutual Information (units)"); plt.title(title)
    save_fig(None, filename); plt.close()
