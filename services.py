# dashboard/services.py
from pymongo import MongoClient
import pandas as pd

def _db():
    client = MongoClient("mongodb://localhost:27017/")
    return client["snrt_db"]

def _to_df(cursor):
    return pd.DataFrame(list(cursor))

def _convert_station_id(df, col="station_id"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=[col], inplace=True)
        df[col] = df[col].astype(int)
    return df

def get_kpi(site: str | None = None,
            family: str | None = None,
            user: str | None = None):
    """Read MongoDB, apply filters, compute KPIs, return dicts for charts."""
    db = _db()

    station_df = _to_df(db.material_station.find({}, {"_id": 1, "ConfigUser": 1}))
    family_df  = _to_df(db.material_family.find({}, {"station_id": 1, "family_id": 1}))
    site_df    = _to_df(db.material_site.find({}, {"station_id": 1, "site_id": 1}))
    eqpt_df    = _to_df(db.material_eqpt.find({}, {"station_id": 1}))

    if "_id" in station_df.columns:
        station_df.rename(columns={"_id": "station_id"}, inplace=True)
    station_df = _convert_station_id(station_df, "station_id")
    family_df  = _convert_station_id(family_df, "station_id")
    site_df    = _convert_station_id(site_df, "station_id")
    eqpt_df    = _convert_station_id(eqpt_df, "station_id")

    # Merge
    sf = station_df.merge(family_df[["station_id","family_id"]], on="station_id", how="left")
    ss = station_df.merge(site_df[["station_id","site_id"]],     on="station_id", how="left")

    # Filters
    if site:
        try:
            site_val = int(float(site))
            ss = ss[ss["site_id"] == site_val]
            sf = sf[sf["station_id"].isin(ss["station_id"])]
        except ValueError:
            ss = ss[ss["site_id"].astype(str) == site]
            sf = sf[sf["station_id"].isin(ss["station_id"])]

    if family:
        try:
            fam_val = int(float(family))
            sf = sf[sf["family_id"] == fam_val]
            ss = ss[ss["station_id"].isin(sf["station_id"])]
        except ValueError:
            sf = sf[sf["family_id"].astype(str) == family]
            ss = ss[ss["station_id"].isin(sf["station_id"])]

    if user:
        station_df = station_df[station_df["ConfigUser"].fillna("").astype(str).str.contains(user, case=False, na=False)]
        sf = sf[sf["station_id"].isin(station_df["station_id"])]
        ss = ss[ss["station_id"].isin(station_df["station_id"])]
        eqpt_df = eqpt_df[eqpt_df["station_id"].isin(station_df["station_id"])]

    # Build series
    stations_per_family = sf["family_id"].dropna().astype(int).value_counts()
    stations_per_site   = ss["site_id"].dropna().astype(int).value_counts()
    stations_per_user   = station_df["ConfigUser"].fillna("—").astype(str).value_counts()
    eqpt_per_station    = eqpt_df["station_id"].value_counts()

    def to_payload(series):
        s = series.sort_values(ascending=False)
        return {"labels": [str(idx) for idx in s.index.tolist()],
                "values": s.values.tolist()}

    return {
        "family": to_payload(stations_per_family),
        "site":   to_payload(stations_per_site),
        "user":   to_payload(stations_per_user),
        "eqpt":   to_payload(eqpt_per_station),
    }
