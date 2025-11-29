from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple, List
import tarfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import geopandas as gpd

def load_us_stations(stations_file: Path) -> pd.DataFrame:
    df = pd.read_fwf(
        stations_file,
        names=["ID","LAT","LON","ELEV","STATE","NAME"],
        widths=[11,9,10,7,3,31],
        dtype={"ID":str},
    )
    df["ID"] = df["ID"].str.strip()
    df = df[df["ID"].str.startswith("US")].copy()
    return df[["ID","LAT","LON","STATE","NAME"]]

def build_station_to_county(stations_df: pd.DataFrame, counties_shp: Path) -> pd.DataFrame:
    counties = gpd.read_file(counties_shp).to_crs("EPSG:4326")
    if "GEOID" in counties.columns:
        counties = counties.rename(columns={"GEOID":"county_fips"})
    counties["county_fips"] = counties["county_fips"].astype(str).str.zfill(5)
    gst = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df["LON"], stations_df["LAT"]),
        crs="EPSG:4326",
    )
    j = gst.sjoin(counties[["county_fips","geometry"]], how="inner", predicate="within")
    return j[["ID","county_fips"]].drop_duplicates()

def _parse_prcp_lines(lines: List[str], station_id: str,
                      years: Optional[Tuple[int,int]] = None) -> pd.DataFrame:
    recs = []
    for line in lines:
        if line[17:21] != "PRCP":
            continue
        year = int(line[11:15]); month = int(line[15:17])
        if years is not None:
            y0, y1 = years
            if not (y0 <= year <= y1):
                continue
        for d in range(31):
            pos = 21 + d*8
            try:
                val = int(line[pos:pos+5])
            except ValueError:
                continue
            if val == -9999:
                continue
            day = d + 1
            try:
                dt = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue
            prcp_mm = val / 10.0
            recs.append((station_id, dt, prcp_mm))
    if not recs:
        return pd.DataFrame(columns=["ID","date","precip_mm"])
    return pd.DataFrame(recs, columns=["ID","date","precip_mm"])

def stream_tar_to_county_day(
    tar_gz: Path,
    station_to_county: pd.DataFrame,
    out_dir: Path,
    years: Optional[Tuple[int,int]] = None,
    stations_subset: Optional[Iterable[str]] = None,
    chunk_files: int = 400,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "county_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_ids = set(station_to_county["ID"].unique())
    if stations_subset is not None:
        valid_ids &= set(stations_subset)

    chunk_accum: List[pd.DataFrame] = []
    chunk_idx = 0

    with tarfile.open(tar_gz, "r|gz") as tar:
        try:
            for m in tqdm(tar, desc="Parsing GHCN-D (.dly)"):
                if not m.name.endswith(".dly"):
                    continue
                station_id = Path(m.name).stem
                fobj = tar.extractfile(m)
                if fobj is None:
                    continue

                if station_id not in valid_ids:
                    _ = fobj.read()
                    continue
                lines = fobj.read().decode("ascii", errors="ignore").splitlines()
                df = _parse_prcp_lines(lines, station_id, years=years)
                if df.empty:
                    continue
                df = df.merge(station_to_county, on="ID", how="left").dropna(subset=["county_fips"])
                if df.empty:
                    continue
                df_grp = df.groupby(["county_fips","date"], as_index=False)["precip_mm"].mean()
                chunk_accum.append(df_grp)
                if len(chunk_accum) >= chunk_files:
                    chunk = pd.concat(chunk_accum, ignore_index=True)
                    chunk = chunk.groupby(["county_fips","date"], as_index=False)["precip_mm"].mean()
                    chunk.to_parquet(tmp_dir / f"chunk_{chunk_idx:05d}.parquet", index=False)
                    chunk_accum = []
                    chunk_idx += 1
        except EOFError:
            print("Unexpected end of gzip; using data parsed so far.")

    if chunk_accum:
        chunk = pd.concat(chunk_accum, ignore_index=True)
        chunk = chunk.groupby(["county_fips","date"], as_index=False)["precip_mm"].mean()
        chunk.to_parquet(tmp_dir / f"chunk_{chunk_idx:05d}.parquet", index=False)

    parts = list(tmp_dir.glob("chunk_*.parquet"))
    final = out_dir / "county_day.parquet"
    if not parts:
        pd.DataFrame(columns=["county_fips","date","precip_mm"]).to_parquet(final, index=False)
        return final
    all_df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    all_df = all_df.groupby(["county_fips","date"], as_index=False)["precip_mm"].mean()
    all_df["county_fips"] = all_df["county_fips"].astype(str).str.zfill(5)
    all_df.to_parquet(final, index=False)
    return final
