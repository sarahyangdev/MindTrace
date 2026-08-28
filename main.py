"""
MindTrace — FastAPI Backend
Serves ML predictions, SHAP explainability, facility finder, and Groq AI chat.
Designed for Render deployment with static frontend.
"""

import os
import json
import math
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Global State ──────────────────────────────────────────────
model: XGBRegressor = None
explainer = None
df: pd.DataFrame = None
features: list = None
feature_spec: dict = None
facilities: list = []
_tracts_cache = None

PRETTY = {
    "poverty_rate": "Poverty Rate",
    "housing_burden_pct": "Housing Burden",
    "education_pct_less_hs": "Education Gap",
    "unemployment_rate": "Unemployment",
    "linguistic_isolation_pct": "Language Isolation",
    "nb10_poverty_rate": "Neighbourhood Poverty",
    "nb10_education_pct_less_hs": "Neighbourhood Education",
    "poverty_vs_neighbours": "Poverty vs Neighbours",
    "housing_burden_pct_vs_county": "Housing vs County Avg",
    "pop_density_per_sqmi": "Population Density",
    "pct_adult": "Adult Share",
    "pm25_annual": "PM2.5 Exposure",
    "nb10_pm25_annual": "Neighbourhood PM2.5",
    "pm25_peak_ratio": "Smoke Peak Ratio",
    "ozone_mean_ppb_2yr": "Ozone Level",
    "ces4_diesel_pm": "Diesel Exhaust",
    "dist_to_mental_health": "Distance to Care",
    "gravity_access_mh": "Care Accessibility",
    "n_primary_care_within_25mi": "Nearby Clinics",
    "poverty_x_pm25": "Poverty × PM2.5",
}

CONTEXT_ONLY = {"pct_adult", "pop_density_per_sqmi"}


# ─── Startup / Shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, df, features, feature_spec, facilities, _tracts_cache

    feature_spec = json.load(open("mindtrace_model_v2_features.json"))
    features = feature_spec["features"]

    model = XGBRegressor()
    model.load_model("mindtrace_model_v2.json")

    df = pd.read_csv(
        "mindtrace_master_dataset_ca_tracts_v2.csv",
        dtype={"tract_fips": str, "county_fips": str, "state_fips": str},
    )

    explainer = shap.TreeExplainer(model)

    if os.path.exists("facilities.json"):
        with open("facilities.json") as f:
            facilities = json.load(f)
    else:
        facilities = _generate_sample_facilities(df)

    _tracts_cache = _build_tracts_cache(df)

    print(
        f"MindTrace ready: {len(df)} tracts, "
        f"{len(features)} features, {len(facilities)} facilities"
    )
    yield


def _build_tracts_cache(frame):
    data = []
    for _, r in frame.iterrows():
        data.append({
            "f": r.tract_fips,
            "la": round(float(r.tract_lat), 4),
            "lo": round(float(r.tract_lon), 4),
            "d": round(float(r.mhlth_crude_prev), 1),
            "p": round(float(r.mhlth_state_percentile), 0),
            "c": r.county_name,
            "n": int(r.total_population),
        })
    return data


def _generate_sample_facilities(frame):
    candidates = frame[frame.n_mental_health_within_5mi > 0].copy()
    candidates = candidates.sort_values(
        "n_mental_health_within_5mi", ascending=False
    )
    sampled = candidates.groupby("county_name").head(3).head(200)

    types = [
        "Outpatient Mental Health", "Crisis Intervention",
        "Community Counseling", "Behavioral Health Center",
        "Psychiatric Services",
    ]
    prefixes = [
        "Pacific", "Valley", "Metro", "Sunrise", "Gateway",
        "Coastal", "Central", "Sierra", "Bay Area", "Inland",
        "Golden", "Mountain View", "Harbor", "Westside", "Downtown",
    ]
    streets = ["Main", "Oak", "Elm", "Pine", "Cedar", "Maple", "Willow", "Birch"]

    np.random.seed(42)
    facs = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        facs.append({
            "name": f"{prefixes[i % len(prefixes)]} {types[i % len(types)]}",
            "lat": round(float(row.tract_lat) + np.random.uniform(-0.015, 0.015), 4),
            "lon": round(float(row.tract_lon) + np.random.uniform(-0.015, 0.015), 4),
            "address": f"{100 + (i * 37) % 9000} {streets[i % len(streets)]} St, {row.county_name}, CA",
            "phone": f"({310 + (i * 17) % 680}) {100 + (i * 31) % 900}-{1000 + (i * 41) % 9000}",
            "services": types[i % len(types)],
            "is_open": i % 3 != 0,
        })
    return facs


# ─── App ───────────────────────────────────────────────────────
app = FastAPI(title="MindTrace API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Utility ───────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest_tract(lat, lon):
    dlat = np.radians(df.tract_lat.values - lat)
    dlon = np.radians(df.tract_lon.values - lon)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat))
        * np.cos(np.radians(df.tract_lat.values))
        * np.sin(dlon / 2) ** 2
    )
    dists = 3958.8 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    idx = int(dists.argmin())
    return df.iloc[idx]


# ─── Request Models ────────────────────────────────────────────
class LocationQuery(BaseModel):
    lat: float
    lon: float
    radius_miles: float = 15.0


class PredictQuery(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    tract_fips: Optional[str] = None


class ExplainQuery(BaseModel):
    tract_fips: str
    open_facilities_count: int = 0


# ─── Endpoints ─────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "tracts": len(df),
        "features": len(features),
        "facilities": len(facilities),
        "model_r2": feature_spec.get("r2_county_holdout"),
        "model_mae": feature_spec.get("mae_county_holdout"),
    }


@app.get("/api/tracts-overview")
async def tracts_overview():
    return JSONResponse(content=_tracts_cache)


@app.post("/api/predict-community")
async def predict_community(query: PredictQuery):
    if query.tract_fips:
        match = df[df.tract_fips == query.tract_fips]
        if match.empty:
            raise HTTPException(404, "Census tract not found")
        row = match.iloc[0]
    elif query.lat is not None and query.lon is not None:
        row = find_nearest_tract(query.lat, query.lon)
    else:
        raise HTTPException(400, "Provide tract_fips or lat/lon")

    X_df = pd.DataFrame([row[features].values], columns=features)
    pred = float(model.predict(X_df)[0])
    pctl = float((df.mhlth_crude_prev < pred).mean() * 100)

    sv = explainer.shap_values(X_df)[0]

    shap_drivers = []
    for name, value in sorted(zip(features, sv), key=lambda t: -abs(t[1])):
        if name not in CONTEXT_ONLY:
            shap_drivers.append({
                "feature": name,
                "label": PRETTY.get(name, name),
                "shap_value": round(float(value), 3),
                "raw_value": round(float(row[name]), 2) if pd.notna(row[name]) else None,
            })

    return {
        "tract_fips": row.tract_fips,
        "county_name": row.county_name,
        "lat": round(float(row.tract_lat), 4),
        "lon": round(float(row.tract_lon), 4),
        "predicted_distress": round(pred, 1),
        "actual_distress": round(float(row.mhlth_crude_prev), 1),
        "state_percentile": round(pctl, 0),
        "population": int(row.total_population),
        "baseline": round(float(explainer.expected_value), 1),
        "shap_drivers": shap_drivers[:8],
        "dist_to_mental_health": round(float(row.dist_to_mental_health), 1),
    }


@app.post("/api/locate-facilities")
async def locate_facilities(query: LocationQuery):
    results = []
    for fac in facilities:
        dist = haversine(query.lat, query.lon, fac["lat"], fac["lon"])
        if dist <= query.radius_miles:
            results.append({**fac, "distance_miles": round(dist, 2)})
    results.sort(key=lambda x: x["distance_miles"])
    return {"facilities": results[:15], "total_found": len(results)}


@app.post("/api/groq-explain")
async def groq_explain(query: ExplainQuery):
    match = df[df.tract_fips == query.tract_fips]
    if match.empty:
        raise HTTPException(404, "Census tract not found")

    row = match.iloc[0]

    X_df = pd.DataFrame([row[features].values], columns=features)
    pred = float(model.predict(X_df)[0])
    pctl = float((df.mhlth_crude_prev < pred).mean() * 100)
    sv = explainer.shap_values(X_df)[0]

    top = sorted(
        ((PRETTY.get(n, n), v) for n, v in zip(features, sv) if n not in CONTEXT_ONLY),
        key=lambda t: -abs(t[1]),
    )[:3]
    drivers_str = " | ".join(f"{n} ({v:+.1f})" for n, v in top)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        stressors = [n for n, v in top if v > 0]
        buffers = [n for n, v in top if v < 0]
        return {
            "response": (
                f"This census tract in {row.county_name} County shows a predicted mental "
                f"distress prevalence of {pred:.1f}%, placing it at the {pctl:.0f}th "
                f"percentile statewide. Primary factors associated with elevated distress: "
                f"{', '.join(stressors) if stressors else 'none identified'}. "
                f"{'Protective factors: ' + ', '.join(buffers) + '. ' if buffers else ''}"
                f"The nearest mental health facility is "
                f"{row.dist_to_mental_health:.1f} miles away."
            ),
            "is_fallback": True,
        }

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        system_prompt = (
            "You are an empathetic community health navigator. Explain model-derived "
            "risk factors non-diagnostically using 'associated with' rather than causal "
            "claims. Translate SHAP drivers into accessible insights and practical "
            "environmental health recommendations. Keep responses under 200 words. "
            "Structure as: 1) Community Snapshot, 2) Key Factors, 3) Recommendations."
        )
        user_content = (
            f"Tract: {row.tract_fips} | County: {row.county_name} | "
            f"Predicted Distress: {pctl:.0f}th Percentile ({pred:.1f}%) | "
            f"Top Drivers: {drivers_str} | "
            f"Nearest MH facility: {row.dist_to_mental_health:.1f} mi | "
            f"Open facilities nearby: {query.open_facilities_count}"
        )

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=350,
        )
        return {
            "response": response.choices[0].delta.content,
            "is_fallback": False,
        }
    except Exception as e:
        return {"response": f"AI insights temporarily unavailable.", "is_fallback": True}


# ─── Serve Frontend ────────────────────────────────────────────
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
