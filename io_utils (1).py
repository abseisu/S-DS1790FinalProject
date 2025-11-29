from __future__ import annotations
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import matplotlib.pyplot as plt
import json

ROOT = Path(__file__).resolve().parents[1]

DIR_RAW = ROOT / "data_raw"
DIR_INT = ROOT / "data_intermediate"
DIR_FIN = ROOT / "data_final"
DIR_FIG = ROOT / "figures"
DIR_TAB = ROOT / "tables"

for d in (DIR_RAW, DIR_INT, DIR_FIN, DIR_FIG, DIR_TAB):
    d.mkdir(parents=True, exist_ok=True)

def read_df(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

def write_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def save_json(obj, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_fig(fig: Optional[plt.Figure], filename: str, dpi: int = 200) -> Path:
    out = DIR_FIG / filename
    (fig or plt).savefig(out, dpi=dpi, bbox_inches="tight")
    return out
