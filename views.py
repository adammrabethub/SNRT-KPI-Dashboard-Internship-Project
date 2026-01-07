# dashboard/views.py
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
import pandas as pd
import io
import re
import unicodedata

# NEW: imports for the model
import numpy as np  # NEW
import joblib       # NEW
from pathlib import Path  # NEW
import os           # NEW
from datetime import datetime  # NEW

from django.conf import settings
MONGO_URI = getattr(settings, "MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "snrt_db"

# -------- Mongo helpers (robust) --------
_mongo_client = None
def _get_db():
    """Return a cached DB handle with short timeout."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    return _mongo_client[DB_NAME]

def _safe_find(col_name, projection=None):
    """
    Safe find: returns list of docs; on any error (collection missing / mongo down) returns [].
    """
    try:
        db = _get_db()
        if projection is None:
            cursor = db[col_name].find({})
        else:
            cursor = db[col_name].find({}, projection)
        return list(cursor)
    except (PyMongoError, ServerSelectionTimeoutError, Exception):
        return []

# -----------------
# Utilities / clean
# -----------------
def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def clean_login(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"", "nan", "none", "null", "-", "_", "—"}:
        return None
    s = _strip_accents(s)
    s = re.sub(r"[^a-z0-9]+", ".", s)
    s = re.sub(r"\.+", ".", s).strip(".")
    return s or None

def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _safe_vc(series: pd.Series) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=int)
    return (
        series.astype(str)
        .replace({"": None, "nan": None, "None": None})
        .dropna()
        .value_counts()
        .sort_values(ascending=False)
    )

# ★ Helpers to ensure labels are human-friendly
_NUMLIKE_RE = re.compile(r"^\s*\d+\s*$")
_TAG_NUM_RE = re.compile(r"^(des|fam|sub|loc)\s*\d+\s*$", re.IGNORECASE)

def _niceify_label(primary: pd.Series, fallback: pd.Series = None, default_text="Sans libellé") -> pd.Series:
    """
    Make a label series readable:
      - if it's just a number or 'des 12' / 'fam 5' → replace by fallback
      - if still empty → default_text
    """
    if primary is None or len(primary) == 0:
        primary = pd.Series(dtype=object)
    s = primary.astype(str)

    def _is_bad(x: str) -> bool:
        if not x or x.strip().lower() in {"nan", "none", "null", "inconnu"}:
            return True
        if _NUMLIKE_RE.match(x) or _TAG_NUM_RE.match(x):
            return True
        return False

    bad = s.apply(_is_bad)
    if fallback is not None and len(fallback):
        fb = fallback.astype(str)
        s = s.where(~bad, fb)
        bad = s.apply(_is_bad)  # re-check after fallback

    s = s.where(~bad, default_text)
    return s

# -------------------------
# NEW — Model loader (cached)
# -------------------------
_model_cache = {"pipe": None, "path": None}

def _load_model():
    """
    Load the scikit-learn pipeline saved as 'logit_site_has_station.pkl'.
    Path is settings.ML_MODEL_PATH. Cached after first load.
    """
    model_path = Path(getattr(settings, "ML_MODEL_PATH", Path(settings.BASE_DIR) / "logit_site_has_station.pkl"))
    if not model_path.exists():
        return None
    if _model_cache["pipe"] is not None and _model_cache["path"] == str(model_path):
        return _model_cache["pipe"]
    pipe = joblib.load(model_path)
    _model_cache["pipe"] = pipe
    _model_cache["path"] = str(model_path)
    return pipe

def _model_meta():
    """Small metadata blob used by UI (threshold line, features, version-ish)."""
    path = _model_cache["path"] or getattr(settings, "ML_MODEL_PATH", None)
    mtime = None
    if path and os.path.exists(path):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
        except Exception:
            mtime = None
    pipe = _model_cache["pipe"]
    feats = getattr(pipe, "feature_names_in_", [])
    # UI threshold for "risk badge / line" — configurable:
    thresh = getattr(settings, "ML_NO_STATION_RISK_THRESHOLD", 0.60)
    return {
        "model_path": path,
        "last_modified": mtime,
        "feature_count": int(len(feats)),
        "features": list(map(str, feats)) if isinstance(feats, (list, np.ndarray)) else [],
        "risk_threshold_no_station": float(thresh),
    }

# -------------------------
# NEW — Feature table loader
# -------------------------
def _load_site_features_df():
    """
    Read the engineered site features from Mongo: 'features_sites_v1'.
    Returns a pandas DataFrame; empty frame if the collection doesn't exist.
    """
    docs = _safe_find("features_sites_v1")
    df = pd.DataFrame(docs)
    if df.empty:
        return df

    # Normalize typical field names used in training (no-ops if already there)
    rename_map = {}
    for k in ["site_id", "Site", "Province", "Region", "SNRT_RS",
              "Latitude", "Longitude", "Altitude", "altitude_bin",
              "amctl_cnt", "climctl_cnt", "fmctl_cnt", "gectl_cnt",
              "tntctl_cnt", "upsctl_cnt", "counterctl_cnt",
              "brigades_cnt", "brigades_present", "ctl_total",
              "dist_to_nearest_station_km", "has_station"]:
        if k in df.columns:
            rename_map[k] = k
    df = df.rename(columns=rename_map)

    # Ensure numeric types for numeric columns (silently coerce)
    num_cols = [
        "Latitude", "Longitude", "Altitude", "ctl_total",
        "brigades_cnt", "dist_to_nearest_station_km"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Categorical columns used by the pipeline
    if "altitude_bin" in df.columns:
        df["altitude_bin"] = df["altitude_bin"].astype(str).str.lower().fillna("low")
    if "Region" in df.columns:
        df["Region"] = df["Region"].astype(str).fillna("unknown")

    return df

# -------------------------
# NEW — Reason builder (storytelling)
# -------------------------
def _mk_reason(row_like: dict) -> str:
    """Produce a compact explanation string for a site's risk."""
    dist = pd.to_numeric(pd.Series([row_like.get("dist_to_nearest_station_km")]), errors="coerce").iloc[0]
    brig = pd.to_numeric(pd.Series([row_like.get("brigades_cnt")]), errors="coerce").iloc[0]
    ctl  = pd.to_numeric(pd.Series([row_like.get("ctl_total")]), errors="coerce").iloc[0]
    altb = str(row_like.get("altitude_bin", "") or "").lower()

    bits = []
    if pd.notna(dist):
        if dist >= 100:    bits.append("éloigné de toute station (≥100 km)")
        elif dist >= 50:   bits.append("distance importante (≥50 km)")
        elif dist >= 25:   bits.append("relativement éloigné (≥25 km)")
    if pd.notna(brig):
        if brig <= 0:      bits.append("aucune brigade recensée")
        elif brig < 2:     bits.append("peu de brigades (<2)")
    if pd.notna(ctl):
        if ctl <= 0:       bits.append("aucun équipement déclaré")
        elif ctl < 5:      bits.append("peu d’équipements (<5)")
    if altb in {"high","haute","élevé","eleve"}:
        bits.append("altitude élevée (accès/logistique difficile)")

    if not bits:
        return "signaux limités — vérifier terrain"
    if len(bits) == 1:
        return bits[0]
    return bits[0] + "; " + ", ".join(bits[1:])

# -------------------------
# NEW — Reusable feature builder
# -------------------------
def _build_features(df: pd.DataFrame, pipe):
    """
    Build an X that matches model expectations (uses engineered fallbacks when needed).
    """
    # helpers producing Series
    def _num_series(df_, col, default=0.0):
        if col in df_.columns:
            return pd.to_numeric(df_[col], errors="coerce")
        return pd.Series(default, index=df_.index, dtype=float)

    def _str_series(df_, col, default=""):
        if col in df_.columns:
            return df_[col].astype(str)
        return pd.Series([default] * len(df_), index=df_.index, dtype=object)

    lat = _num_series(df, "Latitude").fillna(0.0)
    lon = _num_series(df, "Longitude").fillna(0.0)
    alt = _num_series(df, "Altitude").fillna(0.0)
    ctl = _num_series(df, "ctl_total").fillna(0.0)
    brig = _num_series(df, "brigades_cnt").fillna(0.0)
    dist = _num_series(df, "dist_to_nearest_station_km").fillna(0.0)
    alt_bin = _str_series(df, "altitude_bin", "low").str.lower().fillna("low")

    def derive_alt_onehot(level: str) -> pd.Series:
        if "altitude_bin" in df.columns:
            return (alt_bin == level).astype(float)
        if level == "low":  return (alt < 400).astype(float)
        if level == "mid":  return ((alt >= 400) & (alt < 1200)).astype(float)
        if level == "high": return (alt >= 1200).astype(float)
        return pd.Series(0.0, index=df.index)

    def derive_brigades_present() -> pd.Series:
        if "brigades_present" in df.columns:
            return pd.to_numeric(df["brigades_present"], errors="coerce").fillna(0.0).clip(0, 1)
        return (brig > 0).astype(float)

    def derive_brigade_ratio() -> pd.Series:
        denom = ctl.where(ctl != 0, 1.0)
        return (brig / denom).fillna(0.0).clip(0.0, 1.0)

    expected = getattr(pipe, "feature_names_in_", None)
    if expected is not None:
        X = pd.DataFrame(index=df.index)
        for col in expected:
            if col in df.columns:
                s = df[col]
                try:
                    X[col] = pd.to_numeric(s, errors="coerce").fillna(0.0)
                except Exception:
                    X[col] = s.astype(str).fillna("")
                continue
            # engineered fallbacks
            if col in ("alt_low", "alt_mid", "alt_high"):
                X[col] = derive_alt_onehot(col.split("_", 1)[1])
            elif col == "brigades_present":
                X[col] = derive_brigades_present()
            elif col in ("brigade_presence_ratio", "brigade_presence_ratio_reg"):
                X[col] = derive_brigade_ratio()
            elif col == "Latitude":  X[col] = lat
            elif col == "Longitude": X[col] = lon
            elif col == "Altitude":  X[col] = alt
            elif col == "ctl_total": X[col] = ctl
            elif col == "brigades_cnt": X[col] = brig
            elif col == "dist_to_nearest_station_km": X[col] = dist
            else:
                X[col] = pd.Series(0.0, index=df.index)
        X = X[expected]
        return X

    # No feature_names_in_ → best-effort matrix
    return pd.DataFrame({
        "Latitude": lat,
        "Longitude": lon,
        "Altitude": alt,
        "ctl_total": ctl,
        "brigades_cnt": brig,
        "dist_to_nearest_station_km": dist,
        "alt_low":  derive_alt_onehot("low"),
        "alt_mid":  derive_alt_onehot("mid"),
        "alt_high": derive_alt_onehot("high"),
        "brigades_present": derive_brigades_present(),
        "brigade_presence_ratio_reg": derive_brigade_ratio(),
    }, index=df.index)

# ----- NEW: robust predictor that prefers raw columns for a Pipeline -----
def _predict_has_proba(pipe, df_raw: pd.DataFrame) -> np.ndarray:
    """
    Return P(has_station=1) for each row.
    If the pipeline exposes its original expected raw columns (feature_names_in_),
    try predict_proba on those raw columns directly (best for sklearn Pipelines).
    Otherwise, fall back to the engineered _build_features().
    """
    raw_cols = getattr(pipe, "feature_names_in_", None)
    if raw_cols is not None:
        raw_cols = list(map(str, raw_cols))
        if all(c in df_raw.columns for c in raw_cols):
            try:
                return pipe.predict_proba(df_raw[raw_cols])[:, 1]
            except Exception:
                pass  # fall back below if the pipeline still complains

    # Fallback to engineered features
    X = _build_features(df_raw, pipe)
    return pipe.predict_proba(X)[:, 1]

# -------------------------
# Raw loader (base 4 tables)
# -------------------------
def _load_raw_frames():
    """
    Returns: station_df, family_df, site_df, eqpt_df (station<->equipment legacy)
    All frames are safe/empty on DB issues.
    """
    station_df = pd.DataFrame(_safe_find("material_station", {"_id": 1, "ConfigUser": 1}))
    family_df  = pd.DataFrame(_safe_find("material_family", {"station_id": 1, "family_id": 1}))
    site_df    = pd.DataFrame(_safe_find("material_site", {"station_id": 1, "site_id": 1}))
    eqpt_df    = pd.DataFrame(_safe_find("material_eqpt", {"station_id": 1, "equipment_id": 1}))

    if "_id" in station_df.columns:
        station_df = station_df.rename(columns={"_id": "station_id"})

    for df in (station_df, family_df, site_df, eqpt_df):
        if "station_id" in df.columns:
            df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")

    station_df["ConfigUser_clean"] = station_df.get("ConfigUser", pd.Series(dtype=object)).apply(clean_login)

    return station_df, family_df, site_df, eqpt_df

# ----------------------------------------------
# Prepare merged frames + equipment master/labels
# ----------------------------------------------
def _load_frames():
    station_df, family_df, site_df, eqpt_df = _load_raw_frames()

    # merges for station KPIs
    station_family = (
        pd.merge(
            station_df, family_df[["station_id", "family_id"]] if "family_id" in family_df.columns else family_df,
            on="station_id", how="left"
        ) if not station_df.empty else pd.DataFrame(columns=["station_id", "family_id"])
    )

    station_site = (
        pd.merge(
            station_df, site_df[["station_id", "site_id"]] if "site_id" in site_df.columns else site_df,
            on="station_id", how="left"
        ) if not station_df.empty else pd.DataFrame(columns=["station_id", "site_id"])
    )

    # Equipment master (pull human-friendly fields too)
    equipment_df = pd.DataFrame(_safe_find(
        "material_equipment",
        {
            "equipment_id": 1, "station_id": 1,
            "family_id": 1, "subfamily_id": 1, "designation_id": 1, "location_id": 1,
            "family": 1, "subfamily": 1, "designation": 1, "location": 1,
            "model": 1, "Model": 1, "Situation": 1, "Serial": 1, "Barcode": 1,
            "price": 1, "prix": 1, "devise": 1,
        }
    ))

    if not equipment_df.empty:
        for col in ["equipment_id", "station_id", "family_id", "subfamily_id", "designation_id", "location_id"]:
            if col in equipment_df.columns:
                equipment_df[col] = pd.to_numeric(equipment_df[col], errors="coerce")

    # Lookups (legacy)
    fam_lkp = pd.DataFrame(_safe_find("material_equipment_family"))
    sub_lkp = pd.DataFrame(_safe_find("material_equipment_subfamily"))
    des_lkp = pd.DataFrame(_safe_find("material_equipment_designation"))
    loc_lkp = pd.DataFrame(_safe_find("material_equipment_location"))

    fam_name_col = _pick_col(fam_lkp, ["family_name", "name", "label", "FamilyName", "libelle", "value"])
    sub_name_col = _pick_col(sub_lkp, ["subfamily_name", "name", "label", "SubFamilyName", "libelle", "value"])
    des_name_col = _pick_col(des_lkp, ["designation_name", "name", "label", "DesignationName", "libelle", "value"])
    loc_name_col = _pick_col(loc_lkp, ["location_name", "name", "label", "LocationName", "libelle", "value"])

    fam_id_col = _pick_col(fam_lkp, ["family_id", "FamilyID", "_id", "id", "code"]) or "family_id"
    sub_id_col = _pick_col(sub_lkp, ["subfamily_id", "SubfamilyID", "_id", "id", "code"]) or "subfamily_id"
    des_id_col = _pick_col(des_lkp, ["designation_id", "DesignationID", "_id", "id", "code"]) or "designation_id"
    loc_id_col = _pick_col(loc_lkp, ["location_id", "LocationID", "_id", "id", "code"]) or "location_id"

    fam_map = (pd.Series(fam_lkp[fam_name_col].astype(str).values,
                         index=pd.to_numeric(fam_lkp[fam_id_col], errors="coerce").astype("Int64")).to_dict()
               if (not fam_lkp.empty and fam_name_col in fam_lkp.columns) else {})
    sub_map = (pd.Series(sub_lkp[sub_name_col].astype(str).values,
                         index=pd.to_numeric(sub_lkp[sub_id_col], errors="coerce").astype("Int64")).to_dict()
               if (not sub_lkp.empty and sub_name_col in sub_lkp.columns) else {})
    des_map = (pd.Series(des_lkp[des_name_col].astype(str).values,
                         index=pd.to_numeric(des_lkp[des_id_col], errors="coerce").astype("Int64")).to_dict()
               if (not des_lkp.empty and des_name_col in des_lkp.columns) else {})
    loc_map = (pd.Series(loc_lkp[loc_name_col].astype(str).values,
                         index=pd.to_numeric(loc_lkp[loc_id_col], errors="coerce").astype("Int64")).to_dict()
               if (not loc_lkp.empty and loc_name_col in loc_lkp.columns) else {})

    # ---------- NEW: Alternative lookups (if present) ----------
    achat_fam_lkp = pd.DataFrame(_safe_find("achat_family_alt"))
    if not achat_fam_lkp.empty:
        alt_fam_id_col = _pick_col(achat_fam_lkp, ["family_id", "FamilyID", "_id", "id", "code"]) or "family_id"
        alt_fam_name_col = _pick_col(achat_fam_lkp, ["family_name", "FamilyName", "name", "label", "libelle", "value"])
        if alt_fam_name_col in (achat_fam_lkp.columns if not achat_fam_lkp.empty else []):
            alt_fam_map = pd.Series(
                achat_fam_lkp[alt_fam_name_col].astype(str).values,
                index=pd.to_numeric(achat_fam_lkp[alt_fam_id_col], errors="coerce").astype("Int64")
            ).to_dict()
            fam_map.update({k: v for k, v in alt_fam_map.items() if k is not pd.NA})

    station_des_lkp = pd.DataFrame(_safe_find("station_designation"))
    if not station_des_lkp.empty:
        alt_des_id_col = _pick_col(station_des_lkp, ["designation_id", "DesignationID", "_id", "id", "code"]) or "designation_id"
        alt_des_name_col = _pick_col(station_des_lkp, ["designation_name", "DesignationName", "name", "label", "libelle", "value"])
        if alt_des_name_col in (station_des_lkp.columns if not station_des_lkp.empty else []):
            alt_des_map = pd.Series(
                station_des_lkp[alt_des_name_col].astype(str).values,
                index=pd.to_numeric(station_des_lkp[alt_des_id_col], errors="coerce").astype("Int64")
            ).to_dict()
            des_map.update({k: v for k, v in alt_des_map.items() if k is not pd.NA})
    # ---------- END NEW ----------

    def _label_from(df, id_col, raw_txt_candidates, mapping, prefix):
        """
        Priority:
        1) mapping[id] if id present
        2) first non-empty text among raw_txt_candidates
        3) prefix + id (as string) if id exists
        4) 'Inconnu'
        """
        if df is None or df.empty:
            return pd.Series(dtype=object)

        raw_col = _pick_col(df, raw_txt_candidates)
        if raw_col:
            base = df[raw_col].astype(str).str.strip()
            base = base.replace({"": None, "nan": None, "None": None})
        else:
            base = pd.Series([None]*len(df), index=df.index, dtype=object)

        if id_col in df.columns:
            ids = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
            mapped = ids.map(mapping)
            out = mapped.where(mapped.notna(), base)
            with_id = ids.astype(str)
            out = out.where(out.notna(), prefix + with_id)
            out = out.replace({prefix + "<NA>": None})
            return out.fillna("Inconnu")

        return base.fillna("Inconnu")

    if not equipment_df.empty:
        equipment_df["family_label"]      = _label_from(
            equipment_df, "family_id",
            ["family", "Family", "Famille"], fam_map, "fam "
        )
        equipment_df["subfamily_label"]   = _label_from(
            equipment_df, "subfamily_id",
            ["subfamily", "Subfamily", "SousFamille", "Sous-famille"], sub_map, "sub "
        )
        equipment_df["designation_label"] = _label_from(
            equipment_df, "designation_id",
            ["designation", "Designation", "model", "Model", "Modele"], des_map, "des "
        )
        equipment_df["location_label"]    = _label_from(
            equipment_df, "location_id",
            ["location", "Location", "lieu", "Lieu", "Situation"], loc_map, "loc "
        )

    return station_df, station_family, station_site, eqpt_df, equipment_df

# ---------------------
# Filters
# ---------------------
def _apply_filters(station_df, station_family, station_site, eqpt_df,
                   site=None, family=None, user=None, station=None):
    if site and not station_site.empty:
        try:
            site_val = int(float(site))
            station_site = station_site[station_site["site_id"] == site_val]
        except ValueError:
            station_site = station_site[station_site["site_id"].astype(str) == str(site)]
        keep_ids = set(station_site["station_id"].dropna().tolist())
        station_family = station_family[station_family["station_id"].isin(keep_ids)]
        station_df     = station_df[station_df["station_id"].isin(keep_ids)]
        eqpt_df        = eqpt_df[eqpt_df.get("station_id", pd.Series(dtype=float)).isin(keep_ids)] if not eqpt_df.empty else eqpt_df

    if family and not station_family.empty:
        try:
            fam_val = int(float(family))
            station_family = station_family[station_family["family_id"] == fam_val]
        except ValueError:
            station_family = station_family[station_family["family_id"].astype(str) == str(family)]
        keep_ids = set(station_family["station_id"].dropna().tolist())
        station_site = station_site[station_site["station_id"].isin(keep_ids)]
        station_df   = station_df[station_df["station_id"].isin(keep_ids)]
        eqpt_df      = eqpt_df[eqpt_df.get("station_id", pd.Series(dtype=float)).isin(keep_ids)] if not eqpt_df.empty else eqpt_df

    if user and not station_df.empty:
        needle = str(user).strip().lower()
        mask = station_df["ConfigUser_clean"].fillna("").str.contains(needle, case=False, na=False)
        station_df = station_df[mask]
        keep_ids = set(station_df["station_id"].dropna().tolist())
        station_family = station_family[station_family["station_id"].isin(keep_ids)]
        station_site   = station_site[station_site["station_id"].isin(keep_ids)]
        eqpt_df        = eqpt_df[eqpt_df.get("station_id", pd.Series(dtype=float)).isin(keep_ids)] if not eqpt_df.empty else eqpt_df

    if station:
        try:
            st_val = int(float(station))
            keep_ids = {st_val}
        except ValueError:
            keep_ids = set(
                station_df.loc[station_df["station_id"].astype(str) == str(station), "station_id"].tolist()
            )
        if keep_ids:
            station_df     = station_df[station_df["station_id"].isin(keep_ids)]
            station_family = station_family[station_family["station_id"].isin(keep_ids)]
            station_site   = station_site[station_site["station_id"].isin(keep_ids)]
            eqpt_df        = eqpt_df[eqpt_df.get("station_id", pd.Series(dtype=float)).isin(keep_ids)] if not eqpt_df.empty else eqpt_df

    return station_df, station_family, station_site, eqpt_df

# ---------------
# KPI computation
# ---------------
def get_kpis(site=None, family=None, user=None, station=None):
    station_df, station_family, station_site, eqpt_df, equipment_df = _load_frames()

    station_df, station_family, station_site, eqpt_df = _apply_filters(
        station_df, station_family, station_site, eqpt_df, site=site, family=family, user=user, station=station
    )

    stations_per_family = (
        pd.to_numeric(station_family.get("family_id", pd.Series(dtype=float)), errors="coerce")
        .dropna().astype(int).value_counts().sort_values(ascending=False)
    )
    stations_per_site = (
        pd.to_numeric(station_site.get("site_id", pd.Series(dtype=float)), errors="coerce")
        .dropna().astype(int).value_counts().sort_values(ascending=False)
    )
    stations_per_user = (
        station_df.get("ConfigUser_clean", pd.Series(dtype=object)).dropna().value_counts().sort_values(ascending=False)
    )

    # ---- Equipment counts by station (JOIN mapping ↔ master for names) ----
    equip_view = pd.DataFrame()
    equipment_count_per_station = pd.Series(dtype=int)

    if not eqpt_df.empty and not equipment_df.empty:
        map_key = _pick_col(eqpt_df, ["equipment_id", "EquipmentID", "equipmentId", "equip_id", "eqpt_id", "id", "code"])
        master_key = _pick_col(equipment_df, ["equipment_id", "_id", "EquipmentID", "equipmentId", "id", "code"]) or "equipment_id"
        if map_key and master_key:
            m = eqpt_df.copy()
            m["station_id"] = pd.to_numeric(m.get("station_id", pd.Series(dtype=float)), errors="coerce")
            m["__ekey__"] = m[map_key].astype(str).str.strip()
            e = equipment_df.copy()
            e["__ekey__"] = e[master_key].astype(str).str.strip()
            joined = m.merge(e, on="__ekey__", how="left", suffixes=("_map", ""))
            keep_ids = set(pd.to_numeric(station_df.get("station_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
            if keep_ids:
                joined = joined[pd.to_numeric(joined.get("station_id"), errors="coerce").isin(keep_ids)]
            if not joined.empty:
                equip_view = joined
                equipment_count_per_station = (
                    pd.to_numeric(joined["station_id"], errors="coerce").dropna().astype(int)
                    .value_counts().sort_values(ascending=False)
                )

    # Fallback A: master already has station_id
    if equipment_count_per_station.empty and not equipment_df.empty:
        st_col = _pick_col(equipment_df, ["station_id", "StationID", "station", "stationCode"])
        if st_col:
            station_vals = pd.to_numeric(equipment_df[st_col], errors="coerce").dropna().astype(int)
            equipment_count_per_station = station_vals.value_counts().sort_values(ascending=False)
            keep_ids = set(station_df.get("station_id", pd.Series(dtype=float)).dropna().astype(int).tolist())
            equip_view = equipment_df[pd.to_numeric(equipment_df[st_col], errors="coerce").isin(keep_ids)]

    # Fallback B: legacy counts
    if equipment_count_per_station.empty:
        equipment_count_per_station = (
            pd.to_numeric(eqpt_df.get("station_id", pd.Series(dtype=float)), errors="coerce")
            .dropna().astype(int).value_counts().sort_values(ascending=False)
        )

    # ★ Human-friendly labels for global name summaries
    if equip_view is None or equip_view.empty:
        eq_by_family      = pd.Series(dtype=int)
        eq_by_subfamily   = pd.Series(dtype=int)
        eq_by_designation = pd.Series(dtype=int)
        eq_by_location    = pd.Series(dtype=int)
    else:
        equip_view = equip_view.copy()
        equip_view["designation_display"] = _niceify_label(
            equip_view.get("designation_label", pd.Series(dtype=object)),
            fallback=equip_view.get(_pick_col(equip_view, ["model","Model"]), pd.Series(dtype=object)),
            default_text="Sans désignation"
        )
        equip_view["family_display"] = _niceify_label(
            equip_view.get("family_label", pd.Series(dtype=object)),
            fallback=equip_view.get(_pick_col(equip_view, ["family","Family"]), pd.Series(dtype=object)),
            default_text="Sans famille"
        )
        equip_view["subfamily_display"] = _niceify_label(
            equip_view.get("subfamily_label", pd.Series(dtype=object)),
            fallback=equip_view.get(_pick_col(equip_view, ["subfamily","Subfamily"]), pd.Series(dtype=object)),
            default_text="Sans sous-famille"
        )
        equip_view["location_display"] = _niceify_label(
            equip_view.get("location_label", pd.Series(dtype=object)),
            fallback=equip_view.get(_pick_col(equip_view, ["location","Location","Situation"]), pd.Series(dtype=object)),
            default_text="Sans lieu"
        )

        eq_by_family      = _safe_vc(equip_view["family_display"])
        eq_by_subfamily   = _safe_vc(equip_view["subfamily_display"])
        eq_by_designation = _safe_vc(equip_view["designation_display"])
        eq_by_location    = _safe_vc(equip_view["location_display"])

    return {
        "stations_per_family": stations_per_family,
        "stations_per_site": stations_per_site,
        "stations_per_user": stations_per_user,
        "equipment_count_per_station": equipment_count_per_station,
        "eq_by_family": eq_by_family,
        "eq_by_subfamily": eq_by_subfamily,
        "eq_by_designation": eq_by_designation,
        "eq_by_location": eq_by_location,
        "station_df": station_df,
    }

# ------------------------------
# JSON endpoint for charts (AJAX)
# ------------------------------
@login_required
def kpi_data(request):
    site    = request.GET.get("site")
    family  = request.GET.get("family")
    user    = request.GET.get("user")
    station = request.GET.get("station")
    limit   = request.GET.get("limit", "15")

    kpis = get_kpis(site=site, family=family, user=user, station=station)

    def series_to_payload(s):
        if s is None:
            return {"labels": [], "values": []}
        if limit != "all":
            try:
                n = int(limit)
            except ValueError:
                n = 15
            s = s.head(n)
        labels = [str(k) if not pd.isna(k) else "" for k in s.index.tolist()]
        values = [int(v) for v in s.values.tolist()]
        return {"labels": labels, "values": values}

    data = {
        "family": series_to_payload(kpis["stations_per_family"]),
        "site":   series_to_payload(kpis["stations_per_site"]),
        "user":   series_to_payload(kpis["stations_per_user"]),
        "eqpt":   series_to_payload(kpis["equipment_count_per_station"]),
        "eq_by_family":      series_to_payload(kpis["eq_by_family"]),
        "eq_by_subfamily":   series_to_payload(kpis["eq_by_subfamily"]),
        "eq_by_designation": series_to_payload(kpis["eq_by_designation"]),
        "eq_by_location":    series_to_payload(kpis["eq_by_location"]),
    }
    return JsonResponse(data)

# --------------------------------------------
# UPDATED — Site predictions (no-station centers) + CSV
# --------------------------------------------
@login_required
def site_predictions(request):
    """
    Returns ranked sites by probability of being WITHOUT a station (proba_no_station).
    Query params:
      - top={int|all}
      - region=<exact string>
      - format=csv   → CSV download
      - debug=1|true|yes  → include diagnostics in JSON
    """
    top_q = request.GET.get("top", "50")
    region_filter = request.GET.get("region")
    fmt = (request.GET.get("format") or "").lower().strip()

    # Parse "top"
    if str(top_q).lower() == "all":
        top_n = None
    else:
        try:
            top_n = int(top_q)
        except Exception:
            top_n = 50

    # 1) Load model
    pipe = _load_model()
    if pipe is None:
        if fmt == "csv":
            resp = HttpResponse("error\nModel file missing\n", content_type="text/csv; charset=utf-8")
            resp["Content-Disposition"] = 'attachment; filename="site_predictions_error.csv"'
            return resp
        return JsonResponse({"error": "Model not found. Train/save 'logit_site_has_station.pkl'."}, status=500)

    # 2) Load features
    df = _load_site_features_df()
    if df.empty:
        if fmt == "csv":
            resp = HttpResponse("site_id,Site,Region,Province,proba_no_station,reason\n",
                                content_type="text/csv; charset=utf-8")
            resp["Content-Disposition"] = 'attachment; filename="site_predictions_no_station.csv"'
            return resp
        return JsonResponse({"items": [], "count": 0, "returned": 0, "meta": _model_meta()})

    # Optional region filter
    if region_filter and "Region" in df.columns:
        df = df[df["Region"].astype(str) == str(region_filter)]

    # 3) Build features and predict (prefer raw columns expected by the Pipeline)
    try:
        proba_has = _predict_has_proba(pipe, df)  # P(has_station=1)
    except Exception as e:
        if fmt == "csv":
            resp = HttpResponse(f"error\nModel failed to predict: {e}\n",
                                content_type="text/csv; charset=utf-8")
            resp["Content-Disposition"] = 'attachment; filename="site_predictions_error.csv"'
            return resp
        return JsonResponse({"error": f"Model failed to predict: {e}"}, status=500)

    # 4) Rank by probability of NO station + reasons
    out = df.copy()
    out["proba_no_station"] = 1.0 - proba_has
    out["reason"] = out.apply(_mk_reason, axis=1)

    cols_show = [
        "site_id","Site","Region","Province",
        "proba_no_station","reason","SNRT_RS",
        "Latitude","Longitude","Altitude","altitude_bin",
        "ctl_total","brigades_cnt","dist_to_nearest_station_km","has_station",
    ]

    ranked = out.sort_values("proba_no_station", ascending=False)
    if top_n is not None:
        ranked = ranked.head(top_n)

    # CSV export — keep rich columns and ensure proba/ reason are included
    if fmt == "csv":
        csv_df = ranked.copy()
        for c in cols_show:
            if c not in csv_df.columns:
                csv_df[c] = ""
        # Put the “interesting” columns first, then the rest (deduped)
        rest = [c for c in csv_df.columns if c not in cols_show]
        ordered = cols_show + rest
        buf = io.StringIO()
        csv_df[ordered].to_csv(buf, index=False)
        resp = HttpResponse(buf.getvalue().encode("utf-8"),
                            content_type="text/csv; charset=utf-8")
        resp["Content-Disposition"] = 'attachment; filename="site_predictions_no_station.csv"'
        return resp

    # JSON for the UI (same columns as CSV’s front block; UI can ask for more later)
    keep_cols = [c for c in cols_show if c in ranked.columns]
    items = ranked[keep_cols].fillna("").to_dict(orient="records")

    # Optional debug payload
    debug_switch = (request.GET.get("debug") or "").lower() in {"1", "true", "yes"}
    payload = {
        "items": items,
        "count": int(len(out)),
        "returned": int(len(ranked)),
        "meta": _model_meta()
    }
    if debug_switch:
        try:
            raw_cols = list(getattr(pipe, "feature_names_in_", []) or [])
            if raw_cols and all(c in df.columns for c in raw_cols):
                M = df[raw_cols]
            else:
                M = _build_features(df, pipe)
            nz_ratios = {}
            for c in list(M.columns)[:40]:
                s = pd.to_numeric(M[c], errors="coerce")
                nz_ratios[str(c)] = float((s.fillna(0) != 0).mean())
            uniq = len(pd.Series(out["proba_no_station"]).round(6).unique())
            payload["debug"] = {
                "unique_probabilities": int(uniq),
                "nonzero_ratio_head": nz_ratios
            }
        except Exception:
            payload["debug"] = {"error": "debug computation failed"}

    return JsonResponse(payload)

# --------------------------------------------
# NEW — What-if simulation (interactive re-prediction)
# --------------------------------------------
@login_required
def site_predict_simulate(request):
    """
    Recompute probability for one site with optional overrides (for interactive what-if).
    Params (GET):
      - site_id (required)
      - brigades_cnt, dist_to_nearest_station_km, ctl_total, altitude_bin (optional overrides)
    """
    site_id = request.GET.get("site_id")
    if not site_id:
        return JsonResponse({"error": "site_id required"}, status=400)

    pipe = _load_model()
    if pipe is None:
        return JsonResponse({"error": "Model not found"}, status=500)

    df = _load_site_features_df()
    if df.empty or "site_id" not in df.columns:
        return JsonResponse({"error": "features unavailable"}, status=500)

    # Locate row by id (numeric or str)
    try:
        sid = int(float(site_id))
        row_df = df[df["site_id"] == sid]
    except Exception:
        row_df = df[df["site_id"].astype(str) == str(site_id)]

    if row_df.empty:
        return JsonResponse({"error": "site not found"}, status=404)

    base_row = row_df.iloc[0].to_dict()

    # Apply overrides into a single-row frame copy
    new_df = row_df.copy()
    for param, col in [
        ("brigades_cnt", "brigades_cnt"),
        ("dist_to_nearest_station_km", "dist_to_nearest_station_km"),
        ("ctl_total", "ctl_total"),
        ("altitude_bin", "altitude_bin"),
    ]:
        v = request.GET.get(param)
        if v is not None and col in new_df.columns:
            # numeric where appropriate
            if col in {"brigades_cnt", "dist_to_nearest_station_km", "ctl_total"}:
                try:
                    new_df.at[new_df.index[0], col] = float(v)
                except Exception:
                    new_df.at[new_df.index[0], col] = v  # fallback
            else:
                new_df.at[new_df.index[0], col] = str(v)

    # Build features and predict base/new (prefer raw columns for Pipeline)
    try:
        p_has_base = float(_predict_has_proba(pipe, row_df)[0])
    except Exception:
        p_has_base = None

    p_has_new = float(_predict_has_proba(pipe, new_df)[0])

    payload = {
        "site_id": base_row.get("site_id"),
        "site_name": base_row.get("Site") or base_row.get("site_name"),
        "meta": _model_meta(),
        "base": {
            "proba_no_station": None if p_has_base is None else (1.0 - p_has_base),
            "reason": _mk_reason(base_row)
        },
        "new": {
            "proba_no_station": 1.0 - p_has_new,
            "reason": _mk_reason(new_df.iloc[0].to_dict())
        }
    }
    return JsonResponse(payload)

# --------------------------------------------
# NEW — Model meta (for UI to draw threshold line / show features)
# --------------------------------------------
@login_required
def site_predict_meta(request):
    pipe = _load_model()
    if pipe is None:
        return JsonResponse({"error": "Model not found"}, status=500)
    return JsonResponse(_model_meta())

# --------------------------------------------
# Station equipment breakdown (modal)
# --------------------------------------------
@login_required
def station_equipment_detail(request):
    st = request.GET.get("station")
    if not st:
        return JsonResponse({"error": "missing station"}, status=400)
    try:
        st_id = int(float(st))
    except ValueError:
        return JsonResponse({"error": "invalid station id"}, status=400)

    _, _, _, eqpt_df, equipment_df = _load_frames()

    if equipment_df is None or equipment_df.empty:
        return JsonResponse({
            "station": st_id, "total": 0,
            "by_designation": {"labels": [], "values": []},
            "by_family": {"labels": [], "values": []},
            "by_subfamily": {"labels": [], "values": []},
            "by_location": {"labels": [], "values": []},
            "insights": {}, "top_lists": {}, "sample_rows": []
        })

    # Prefer JOIN mapping ↔ master for this station
    view = pd.DataFrame()
    if eqpt_df is not None and not eqpt_df.empty:
        map_key = _pick_col(eqpt_df, ["equipment_id", "EquipmentID", "equipmentId", "equip_id", "eqpt_id", "id", "code"])
        master_key = _pick_col(equipment_df, ["equipment_id", "_id", "EquipmentID", "equipmentId", "id", "code"]) or "equipment_id"
        if map_key and master_key:
            map_df = eqpt_df.copy()
            map_df["station_id"] = pd.to_numeric(map_df.get("station_id", pd.Series(dtype=float)), errors="coerce")
            map_df = map_df[map_df["station_id"] == st_id]
            if not map_df.empty:
                map_df["__ekey__"] = map_df[map_key].astype(str).str.strip()
                eq2 = equipment_df.copy()
                eq2["__ekey__"] = eq2[master_key].astype(str).str.strip()
                view = map_df.merge(eq2, on="__ekey__", how="left", suffixes=("_map", ""))

    # Fallback: station_id directly in master
    if view.empty:
        st_col = _pick_col(equipment_df, ["station_id", "StationID", "station", "stationCode"])
        if st_col:
            mask = pd.to_numeric(equipment_df[st_col], errors="coerce") == st_id
            view = equipment_df[mask]

    total = int(len(view))

    view = view.copy()
    view["designation_display"] = _niceify_label(
        view.get("designation_label", pd.Series(dtype=object)),
        fallback=view.get(_pick_col(view, ["model","Model"]), pd.Series(dtype=object)),
        default_text="Sans désignation"
    )
    view["family_display"] = _niceify_label(
        view.get("family_label", pd.Series(dtype=object)),
        fallback=view.get(_pick_col(view, ["family","Family"]), pd.Series(dtype=object)),
        default_text="Sans famille"
    )
    view["subfamily_display"] = _niceify_label(
        view.get("subfamily_label", pd.Series(dtype=object)),
        fallback=view.get(_pick_col(view, ["subfamily","Subfamily"]), pd.Series(dtype=object)),
        default_text="Sans sous-famille"
    )
    view["location_display"] = _niceify_label(
        view.get("location_label", pd.Series(dtype=object)),
        fallback=view.get(_pick_col(view, ["location","Location","Situation"]), pd.Series(dtype=object)),
        default_text="Sans lieu"
    )

    def pack(series: pd.Series, top=50):
        s = _safe_vc(series).head(top)
        return {"labels": [str(i) for i in s.index.tolist()],
                "values": [int(v) for v in s.values.tolist()]}

    def top_list(series: pd.Series, top=6):
        s = _safe_vc(series).head(top)
        return [{"label": str(k), "count": int(v)} for k, v in zip(s.index.tolist(), s.values.tolist())]

    def dominant(series: pd.Series):
        s = _safe_vc(series)
        if s.empty or total == 0:
            return {"label": "—", "count": 0, "pct": 0.0}
        lbl = str(s.index[0])
        cnt = int(s.iloc[0])
        pct = round(100.0 * cnt / max(total, 1), 1)
        return {"label": lbl, "count": cnt, "pct": pct}

    model_col     = _pick_col(view, ["model", "Model"])
    situation_col = _pick_col(view, ["Situation"])

    by_designation = pack(view["designation_display"])
    by_family      = pack(view["family_display"])
    by_subfamily   = pack(view["subfamily_display"])
    by_location    = pack(view["location_display"])

    insights = {
        "total": total,
        "top_model":      dominant(view.get(model_col, pd.Series(dtype=object))) if model_col else {"label": "—", "count": 0, "pct": 0.0},
        "top_designation":dominant(view["designation_display"]),
        "top_family":     dominant(view["family_display"]),
        "top_subfamily":  dominant(view["subfamily_display"]),
        "top_location":   dominant(view["location_display"]),
        "top_situation":  dominant(view.get(situation_col, pd.Series(dtype=object))) if situation_col else {"label": "—", "count": 0, "pct": 0.0},
    }

    top_lists = {
        "designation": top_list(view["designation_display"]),
        "family":      top_list(view["family_display"]),
        "subfamily":   top_list(view["subfamily_display"]),
        "location":    top_list(view["location_display"]),
        "models":      top_list(view.get(model_col, pd.Series(dtype=object))) if model_col else [],
        "situations":  top_list(view.get(situation_col, pd.Series(dtype=object))) if situation_col else [],
    }

    cols = [
        _pick_col(view, ["equipment_id", "_id", "EquipmentID", "equipmentId", "id", "code"]),
        model_col, _pick_col(view, ["Serial","serial"]), _pick_col(view, ["Barcode","barcode"]),
        "designation_display", "family_display", "subfamily_display", "location_display",
    ]
    cols = [c for c in cols if c]
    sample_rows = []
    if cols:
        sample_rows = view[cols].head(20).fillna("").to_dict(orient="records")

    payload = {
        "station": st_id,
        "total": total,
        "by_designation": by_designation,
        "by_family":      by_family,
        "by_subfamily":   by_subfamily,
        "by_location":    by_location,
        "insights": insights,
        "top_lists": top_lists,
        "sample_rows": sample_rows,
    }
    return JsonResponse(payload)

# -------------
# HTML dashboard
# -------------
@login_required
def dashboard_view(request):
    site    = request.GET.get("site")
    family  = request.GET.get("family")
    user    = request.GET.get("user")
    station = request.GET.get("station")

    kpis = get_kpis(site=site, family=family, user=user, station=station)

    tables = {
        "Stations par famille": kpis["stations_per_family"]
            .rename_axis("ID Famille").to_frame("Nombre de stations")
            .to_html(classes="table table-striped table-sm", border=0),
        "Stations par site": kpis["stations_per_site"]
            .rename_axis("ID Site").to_frame("Nombre de stations")
            .to_html(classes="table table-striped table-sm", border=0),
        "Stations par utilisateur": kpis["stations_per_user"]
            .rename_axis("Utilisateur").to_frame("Nombre de stations")
            .to_html(classes="table table-striped table-sm", border=0),
        "Équipements par station": kpis["equipment_count_per_station"]
            .rename_axis("ID Station").to_frame("Nombre d’équipements")
            .to_html(classes="table table-striped table-sm", border=0),
        "Équipements par famille (nom)": kpis["eq_by_family"]
            .rename_axis("Famille").to_frame("Nombre d’équipements")
            .to_html(classes="table table-striped table-sm", border=0),
        "Équipements par sous-famille (nom)": kpis["eq_by_subfamily"]
            .rename_axis("Sous-famille").to_frame("Nombre d’équipements")
            .to_html(classes="table table-striped table-sm", border=0),
        "Équipements par désignation (nom)": kpis["eq_by_designation"]
            .rename_axis("Désignation").to_frame("Nombre d’équipements")
            .to_html(classes="table table-striped table-sm", border=0),
        "Équipements par lieu (nom)": kpis["eq_by_location"]
            .rename_axis("Lieu").to_frame("Nombre d’équipements")
            .to_html(classes="table table-striped table-sm", border=0),
    }

    station_df = kpis["station_df"]
    user_logins = (
        station_df.get("ConfigUser_clean", pd.Series(dtype=object)).dropna().value_counts().index.tolist()
        if not station_df.empty else []
    )

    station_ids = [str(x) for x in kpis["equipment_count_per_station"].index.tolist()]
    def _numkey(s):
        try:
            return int(s)
        except Exception:
            return float("inf")
    station_ids = sorted(station_ids, key=_numkey)

    return render(request, "dashboard/index.html", {
        "tables": tables,
        "user_logins": user_logins,
        "station_ids": station_ids,
    })

# ----------------
# Excel download
# ----------------
@login_required
def download_excel(request):
    site    = request.GET.get("site")
    family  = request.GET.get("family")
    user    = request.GET.get("user")
    station = request.GET.get("station")

    kpis = get_kpis(site=site, family=family, user=user, station=station)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        kpis["stations_per_family"].rename_axis("ID Famille").to_frame("Nombre de stations").to_excel(writer, sheet_name="Stations par famille")
        kpis["stations_per_site"].rename_axis("ID Site").to_frame("Nombre de stations").to_excel(writer, sheet_name="Stations par site")
        kpis["stations_per_user"].rename_axis("Utilisateur").to_frame("Nombre de stations").to_excel(writer, sheet_name="Stations par utilisateur")
        kpis["equipment_count_per_station"].rename_axis("ID Station").to_frame("Nombre d’équipements").to_excel(writer, sheet_name="Éqpts par station")
        kpis["eq_by_family"].rename_axis("Famille").to_frame("Nombre d’équipements").to_excel(writer, sheet_name="Éqpts par famille (nom)")
        kpis["eq_by_subfamily"].rename_axis("Sous-famille").to_frame("Nombre d’équipements").to_excel(writer, sheet_name="Éqpts par sous-famille")
        kpis["eq_by_designation"].rename_axis("Désignation").to_frame("Nombre d’équipements").to_excel(writer, sheet_name="Éqpts par désignation")
        kpis["eq_by_location"].rename_axis("Lieu").to_frame("Nombre d’équipements").to_excel(writer, sheet_name="Éqpts par lieu")

    output.seek(0)
    resp = HttpResponse(output.read(), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    resp["Content-Disposition"] = 'attachment; filename="kpi_results.xlsx"'
    return resp

# ----------------------
# Data health (JSON)
# ----------------------
@login_required
def data_health(request):
    """
    Always returns a JSON structure (zeros if DB is down) so the UI panel never breaks.
    """
    try:
        station_df, family_df, site_df, eqpt_df = _load_raw_frames()

        station_id_null = int(station_df.get("station_id", pd.Series(dtype=float)).isna().sum())
        station_id_dup  = int(station_df.get("station_id", pd.Series(dtype=float)).duplicated().sum())

        orig  = station_df.get("ConfigUser", pd.Series(dtype=object))
        clean = station_df.get("ConfigUser_clean", pd.Series(dtype=object))
        became_null = int(((orig.notna()) & (clean.isna())).sum())
        changed     = int((orig.fillna("").astype(str).str.lower() != clean.fillna("").astype(str)).sum())
        valid_mask  = clean.dropna().astype(str).str.match(r"^[a-z0-9]+(\.[a-z0-9]+)*$")
        invalid_cnt = int((~valid_mask).sum()) if len(valid_mask) else 0
        invalid_examples = clean.dropna().astype(str)[~clean.dropna().astype(str).str.match(r"^[a-z0-9]+(\.[a-z0-9]+)*$")].head(10).tolist()

        site_id_num = pd.to_numeric(site_df.get("site_id", pd.Series(dtype=float)), errors="coerce")
        site_non_numeric = int(site_id_num.isna().sum())

        base_ids = set(station_df.get("station_id", pd.Series(dtype=float)).dropna().tolist())
        site_ids = set(site_df.get("station_id", pd.Series(dtype=float)).dropna().tolist())
        site_orphans = sorted(list(site_ids - base_ids))[:20]
        stations_missing_site = sorted(list(base_ids - site_ids))[:20]

        if not site_df.empty and "station_id" in site_df.columns:
            sites_per_station = site_df.groupby("station_id")["site_id"].nunique(dropna=True)
            stations_with_multi_sites = int((sites_per_station > 1).sum())
            sample_multi_site = sites_per_station[sites_per_station > 1].head(10).index.astype("Int64").dropna().tolist()
        else:
            stations_with_multi_sites = 0
            sample_multi_site = []

        fam_id_num = pd.to_numeric(family_df.get("family_id", pd.Series(dtype=float)), errors="coerce")
        family_non_numeric = int(fam_id_num.isna().sum())

        fam_ids = set(family_df.get("station_id", pd.Series(dtype=float)).dropna().tolist())
        fam_orphans = sorted(list(fam_ids - base_ids))[:20]
        stations_missing_family = sorted(list(base_ids - fam_ids))[:20]

        if not family_df.empty and "station_id" in family_df.columns:
            fams_per_station = family_df.groupby("station_id")["family_id"].nunique(dropna=True)
            stations_with_multi_families = int((fams_per_station > 1).sum())
            sample_multi_family = fams_per_station[fams_per_station > 1].head(10).index.astype("Int64").dropna().tolist()
        else:
            stations_with_multi_families = 0
            sample_multi_family = []

        eqpt_ids = set(eqpt_df.get("station_id", pd.Series(dtype=float)).dropna().tolist())
        eqpt_orphans = sorted(list(eqpt_ids - base_ids))[:20]
        stations_missing_eqpt = sorted(list(base_ids - eqpt_ids))[:20]
        eqpt_station_num = pd.to_numeric(eqpt_df.get("station_id", pd.Series(dtype=float)), errors="coerce")
        eqpt_non_numeric_station = int(eqpt_station_num.isna().sum())

        equipment_df = pd.DataFrame(_safe_find(
            "material_equipment",
            {"equipment_id": 1, "station_id": 1, "family_id": 1, "subfamily_id": 1, "designation_id": 1, "location_id": 1}
        ))

        fam_lkp = pd.DataFrame(_safe_find("material_equipment_family"))
        sub_lkp = pd.DataFrame(_safe_find("material_equipment_subfamily"))
        des_lkp = pd.DataFrame(_safe_find("material_equipment_designation"))
        loc_lkp = pd.DataFrame(_safe_find("material_equipment_location"))

        for col in ["equipment_id", "station_id", "family_id", "subfamily_id", "designation_id", "location_id"]:
            if col in equipment_df.columns:
                equipment_df[col] = pd.to_numeric(equipment_df[col], errors="coerce")

        def _orphans(left_df, left_key, right_df, right_key):
            if left_df is None or right_df is None or left_key not in left_df.columns or right_key not in right_df.columns:
                return 0
            merged = left_df[[left_key]].dropna().drop_duplicates().merge(
                right_df[[right_key]].dropna().drop_duplicates(),
                how="left", left_on=left_key, right_on=right_key
            )
            return int(merged[right_key].isna().sum())

        def _pct_non_null(df, col):
            if df is None or df.empty or col not in df.columns: return 0.0
            total = len(df)
            return float(df[col].notna().mean() * 100.0) if total else 0.0

        eq_lookup = {
            "rows_equipment": int(len(equipment_df)),
            "missing_station_id_rows": int(equipment_df.get("station_id", pd.Series(dtype=float)).isna().sum()) if "station_id" in equipment_df.columns else int(len(equipment_df)),
            "family_orphans": _orphans(equipment_df, "family_id",      fam_lkp, _pick_col(fam_lkp, ["family_id", "FamilyID", "id", "code"]) or "family_id"),
            "subfamily_orphans": _orphans(equipment_df, "subfamily_id", sub_lkp, _pick_col(sub_lkp, ["subfamily_id", "SubfamilyID", "id", "code"]) or "subfamily_id"),
            "designation_orphans": _orphans(equipment_df, "designation_id", des_lkp, _pick_col(des_lkp, ["designation_id", "DesignationID", "id", "code"]) or "designation_id"),
            "location_orphans": _orphans(equipment_df, "location_id",  loc_lkp, _pick_col(loc_lkp, ["location_id", "LocationID", "id", "code"]) or "location_id"),
            "pct_with_family_id": _pct_non_null(equipment_df, "family_id"),
            "pct_with_subfamily_id": _pct_non_null(equipment_df, "subfamily_id"),
            "pct_with_designation_id": _pct_non_null(equipment_df, "designation_id"),
            "pct_with_location_id": _pct_non_null(equipment_df, "location_id"),
        }

        report = {
            "rows": {
                "stations": int(len(station_df)),
                "families_rows": int(len(family_df)),
                "sites_rows": int(len(site_df)),
                "eqpt_rows": int(len(eqpt_df)),
                "equipment_master_rows": int(len(equipment_df))
            },
            "station": {
                "station_id_null": station_id_null,
                "station_id_duplicates": station_id_dup,
                "configuser_changed_after_clean": changed,
                "configuser_became_null_after_clean": became_null,
                "configuser_clean_invalid_pattern_count": invalid_cnt,
                "configuser_clean_invalid_examples": invalid_examples,
            },
            "site_mapping": {
                "non_numeric_site_id_rows": site_non_numeric,
                "stations_with_multiple_sites": stations_with_multi_sites,
                "sample_stations_with_multiple_sites": sample_multi_site,
                "orphan_rows_station_ids_not_in_base": site_orphans,
                "stations_missing_site_mapping": stations_missing_site,
            },
            "family_mapping": {
                "non_numeric_family_id_rows": family_non_numeric,
                "stations_with_multiple_families": stations_with_multi_families,
                "sample_stations_with_multiple_families": sample_multi_family,
                "orphan_rows_station_ids_not_in_base": fam_orphans,
                "stations_missing_family_mapping": stations_missing_family,
            },
            "equipment_mapping": {
                "non_numeric_station_id_rows_in_eqpt": eqpt_non_numeric_station,
                "orphan_rows_station_ids_not_in_base": eqpt_orphans,
                "stations_missing_equipment_rows": stations_missing_eqpt,
            },
            "equipment_lookup": eq_lookup,
        }
        return JsonResponse(report)

    except Exception:
        # Absolute fallback so the panel still renders
        return JsonResponse({
            "rows": {
                "stations": 0, "families_rows": 0, "sites_rows": 0, "eqpt_rows": 0, "equipment_master_rows": 0
            },
            "station": {
                "station_id_null": 0, "station_id_duplicates": 0,
                "configuser_changed_after_clean": 0, "configuser_became_null_after_clean": 0,
                "configuser_clean_invalid_pattern_count": 0, "configuser_clean_invalid_examples": []
            },
            "site_mapping": {
                "non_numeric_site_id_rows": 0, "stations_with_multiple_sites": 0,
                "sample_stations_with_multiple_sites": [], "orphan_rows_station_ids_not_in_base": [],
                "stations_missing_site_mapping": []
            },
            "family_mapping": {
                "non_numeric_family_id_rows": 0, "stations_with_multiple_families": 0,
                "sample_stations_with_multiple_families": [], "orphan_rows_station_ids_not_in_base": [],
                "stations_missing_family_mapping": []
            },
            "equipment_mapping": {
                "non_numeric_station_id_rows_in_eqpt": 0,
                "orphan_rows_station_ids_not_in_base": [],
                "stations_missing_equipment_rows": []
            },
            "equipment_lookup": {
                "rows_equipment": 0, "missing_station_id_rows": 0,
                "family_orphans": 0, "subfamily_orphans": 0, "designation_orphans": 0, "location_orphans": 0,
                "pct_with_family_id": 0.0, "pct_with_subfamily_id": 0.0,
                "pct_with_designation_id": 0.0, "pct_with_location_id": 0.0
            }
        })
