import os
import json
import math
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq

# APP SETUP

app = FastAPI(
    title="MindTrace API",
    version="1.0.0",
    description="Backend API for the MindTrace mental health resource navigator."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# GLOBAL DATA CACHES

TRACT_CACHE = {}
FACILITIES_CACHE = []


# LOAD DATA WHEN SERVER STARTS

@app.on_event("startup")
def load_data():
    global TRACT_CACHE, FACILITIES_CACHE

    # Load ML / SHAP tract cache
    if os.path.exists("mindtrace_tracts_cache.json"):
        with open("mindtrace_tracts_cache.json", "r", encoding="utf-8") as file:
            TRACT_CACHE = json.load(file)

        print(f"Loaded {len(TRACT_CACHE)} census tracts.")
    else:
        print("WARNING: mindtrace_tracts_cache.json not found.")

    # Load mental health facility data
    if os.path.exists("samhsa_facilities.geojson"):
        with open("samhsa_facilities.geojson", "r", encoding="utf-8") as file:
            facility_data = json.load(file)

        FACILITIES_CACHE = facility_data.get("features", [])

        print(f"Loaded {len(FACILITIES_CACHE)} facilities.")
    else:
        print("WARNING: samhsa_facilities.geojson not found.")


# REQUEST MODELS

class LocationQuery(BaseModel):
    lat: float
    lon: float
    radius_miles: float = 15.0


class ExplainQuery(BaseModel):
    tract_fips: str
    open_facilities_count: int = 0
    question: Optional[str] = None


# UTILITY FUNCTIONS

def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate great-circle distance between two coordinates.

    Returns distance in miles.
    """

    earth_radius_miles = 3958.8

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_miles * c


def validate_coordinates(lat: float, lon: float):
    """
    Make sure latitude and longitude are valid.
    """

    if lat < -90 or lat > 90:
        raise HTTPException(
            status_code=400,
            detail="Latitude must be between -90 and 90."
        )

    if lon < -180 or lon > 180:
        raise HTTPException(
            status_code=400,
            detail="Longitude must be between -180 and 180."
        )


def find_nearest_tract(lat: float, lon: float):
    """
    MVP method:
    Find the census tract whose centroid is closest
    to the requested coordinates.

    NOTE:
    This is an approximation. A point-in-polygon lookup
    would be more geographically accurate.
    """

    if not TRACT_CACHE:
        raise HTTPException(
            status_code=503,
            detail="Census tract data is not loaded."
        )

    nearest_tract = None
    nearest_distance = float("inf")

    for fips, tract in TRACT_CACHE.items():

        tract_lat = tract.get("lat")
        tract_lon = tract.get("lon")

        if tract_lat is None or tract_lon is None:
            continue

        distance = haversine(
            lat,
            lon,
            float(tract_lat),
            float(tract_lon)
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_tract = tract

    if nearest_tract is None:
        raise HTTPException(
            status_code=404,
            detail="No census tract could be matched."
        )

    return nearest_tract, nearest_distance


def get_top_shap_factors(shap_values: dict, limit: int = 3):
    """
    Return the strongest positive and negative SHAP drivers.
    """

    if not shap_values:
        return [], []

    sorted_values = sorted(
        shap_values.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    stressors = []
    buffers = []

    for feature, value in sorted_values:

        item = {
            "feature": feature,
            "shap_value": round(float(value), 4)
        }

        if value > 0 and len(stressors) < limit:
            stressors.append(item)

        elif value < 0 and len(buffers) < limit:
            buffers.append(item)

    return stressors, buffers


# HEALTH CHECK

@app.get("/api/health")
def health_check():

    return {
        "status": "ok",
        "tracts_loaded": len(TRACT_CACHE),
        "facilities_loaded": len(FACILITIES_CACHE)
    }


# FACILITY LOCATOR

@app.post("/api/locate-facilities")
def locate_facilities(query: LocationQuery):

    validate_coordinates(query.lat, query.lon)

    if query.radius_miles <= 0 or query.radius_miles > 100:
        raise HTTPException(
            status_code=400,
            detail="Radius must be greater than 0 and at most 100 miles."
        )

    if not FACILITIES_CACHE:
        return {
            "facilities": [],
            "total_found": 0,
            "message": "Facility dataset is not currently loaded."
        }

    results = []

    for feature in FACILITIES_CACHE:

        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        coordinates = geometry.get("coordinates")

        if not coordinates or len(coordinates) < 2:
            continue

        facility_lon = coordinates[0]
        facility_lat = coordinates[1]

        try:
            facility_lon = float(facility_lon)
            facility_lat = float(facility_lat)
        except (TypeError, ValueError):
            continue

        distance = haversine(
            query.lat,
            query.lon,
            facility_lat,
            facility_lon
        )

        if distance <= query.radius_miles:

            results.append({
                "id": properties.get("id"),
                "name": properties.get(
                    "name",
                    "Mental Health Facility"
                ),
                "address": properties.get("address", ""),
                "phone": properties.get("phone", ""),
                "services": properties.get("services", []),
                "is_open": properties.get("is_open"),
                "distance_miles": round(distance, 2),
                "lat": facility_lat,
                "lon": facility_lon
            })

    results.sort(
        key=lambda facility: facility["distance_miles"]
    )

    nearest_results = results[:10]

    return {
        "facilities": nearest_results,
        "total_found": len(results),
        "radius_miles": query.radius_miles
    }


# COMMUNITY ML / SHAP LOOKUP

@app.post("/api/predict-community")
def predict_community(query: LocationQuery):

    validate_coordinates(query.lat, query.lon)

    tract, centroid_distance = find_nearest_tract(
        query.lat,
        query.lon
    )

    shap_values = tract.get("shap_values", {})

    stressors, buffers = get_top_shap_factors(
        shap_values
    )

    return {
        "tract_fips": tract.get("tract_fips"),
        "county_name": tract.get("county_name"),

        "predicted_mhlth": tract.get(
            "mhlth_crude_prev"
        ),

        "predicted_mhlth_percentile": tract.get(
            "mhlth_state_percentile"
        ),

        "features": tract.get(
            "features",
            {}
        ),

        "shap_values": shap_values,

        "top_stressors": stressors,
        "top_buffers": buffers,

        "tract_centroid": {
            "lat": tract.get("lat"),
            "lon": tract.get("lon")
        },

        "centroid_distance_miles": round(
            centroid_distance,
            3
        ),

        "lookup_method": "nearest_tract_centroid"
    }


# GROQ COMMUNITY EXPLAINER

@app.post("/api/groq-explain")
def groq_explain(query: ExplainQuery):

    tract_fips = str(query.tract_fips).zfill(11)

    tract = TRACT_CACHE.get(tract_fips)

    if tract is None:
        raise HTTPException(
            status_code=404,
            detail="Census tract not found."
        )

    api_key = os.environ.get("GROQ_API_KEY")

    shap_values = tract.get(
        "shap_values",
        {}
    )

    stressors, buffers = get_top_shap_factors(
        shap_values
    )

    # Fallback if Groq API key is not configured
    if not api_key:

        return {
            "response": (
                "Community-level environmental and socioeconomic "
                "conditions vary in this area. The model output should "
                "be interpreted as a population-level statistical "
                "estimate rather than an individual diagnosis."
            ),
            "is_fallback": True
        }

    client = Groq(
        api_key=api_key
    )

    system_prompt = """
You are the MindTrace community health information assistant.

Your role is to explain population-level model results in clear,
neutral, accessible language.

Important requirements:

- Do not diagnose individuals.
- Do not claim that a model feature causes a mental health condition.
- Describe relationships as associations or correlations.
- Clearly distinguish community-level statistics from individual health.
- Explain SHAP values as model contributions rather than causal effects.
- Avoid making medical treatment recommendations.
- Encourage appropriate professional or local support when relevant.
- Keep explanations concise and understandable.
"""

    user_prompt = f"""
Census tract: {tract_fips}

Community predicted mental distress percentile:
{tract.get("mhlth_state_percentile")}

Community model features:
{json.dumps(tract.get("features", {}), indent=2)}

Top positive SHAP contributors:
{json.dumps(stressors, indent=2)}

Top negative SHAP contributors:
{json.dumps(buffers, indent=2)}

Nearby available facilities:
{query.open_facilities_count}

User question:
{query.question or "Explain the community-level model results."}

Provide a short community-level explanation.
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.3,
            max_tokens=400
        )

        return {
            "response": response.choices[0].message.content,
            "is_fallback": False
        }

    except Exception as error:

        print("Groq error:", error)

        return {
            "response": (
                "The AI explanation service is temporarily "
                "unavailable. The community model results can "
                "still be viewed directly."
            ),
            "is_fallback": True
        }


# SERVE FRONTEND

if os.path.exists("static"):
    app.mount(
        "/",
        StaticFiles(
            directory="static",
            html=True
        ),
        name="static"
    )
