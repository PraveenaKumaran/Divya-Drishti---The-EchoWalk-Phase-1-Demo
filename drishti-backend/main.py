"""
Drishti backend (Groq vision + OpenRouteService walking directions).

Two endpoints:
  POST /analyze  -> Brain 2: describe the scene in Tamil (Groq vision)
  POST /route    -> Brain 4: foot-walking turn-by-turn steps (OpenRouteService)

.env needs TWO free keys (no credit card required for either):
  GROQ_API_KEY=...   (from https://console.groq.com)
  ORS_API_KEY=...    (from https://openrouteservice.org -> Dashboard)

Run:  uvicorn main:app --reload --port 8000
"""

import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
ors_key = os.getenv("ORS_API_KEY")

print("--- SERVER STARTING ---")
print("[OK] Groq key loaded." if groq_key else "[X] GROQ_API_KEY missing in .env")
print("[OK] ORS key loaded." if ors_key else "[X] ORS_API_KEY missing in .env")

client = Groq(api_key=groq_key)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replaces the decommissioned llama-3.2-90b-vision-preview. If this ever
# errors with "model_decommissioned", check https://console.groq.com/docs/models
# for the current vision model and swap the string below.
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ===========================================================================
# Brain 2 — scene description (vision)
# ===========================================================================
class ImageData(BaseModel):
    image_base64: str


@app.post("/analyze")
def analyze_image(data: ImageData):
    print("\n[SCAN] new deep-scan request")
    try:
        prompt = (
            "You are an AI assistant helping a visually impaired person in India. "
            "Look at this image and describe what is directly in front of them in "
            "ONE or TWO short sentences. Focus on safety and obstacles. "
            "You MUST reply ONLY in natural Tamil script."
        )
        image_url = f"data:image/jpeg;base64,{data.image_base64}"
        chat = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            model=VISION_MODEL,
        )
        tamil_text = chat.choices[0].message.content.strip()
        print("[OK] Groq replied:", tamil_text)
        return {"tamil_text": tamil_text}
    except Exception as e:
        print("[X] ERROR AT GROQ API:", str(e))
        return {"tamil_text": "மன்னிக்கவும், இப்போது பார்க்க முடியவில்லை."}


# ===========================================================================
# Brain 4 — walking turn-by-turn directions
# ===========================================================================
class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float


@app.post("/route")
def route(req: RouteRequest):
    """Return foot-walking steps from start -> end via OpenRouteService.

    Each step includes its maneuver `type` (an ORS code the frontend maps to a
    Tamil phrase), the distance of that step, and the lat/lon of the maneuver
    point (so the frontend can announce it when the user gets close)."""
    if not ors_key:
        raise HTTPException(status_code=500, detail="ORS_API_KEY not set in .env")

    # NOTE: ORS expects coordinates as [longitude, latitude] (lon first!).
    url = "https://api.openrouteservice.org/v2/directions/foot-walking/geojson"
    body = {"coordinates": [
        [req.start_lon, req.start_lat],
        [req.end_lon, req.end_lat],
    ]}
    headers = {"Authorization": ors_key, "Content-Type": "application/json"}

    try:
        r = requests.post(url, json=body, headers=headers, timeout=20)
        r.raise_for_status()
    except requests.HTTPError:
        # ORS returns a useful JSON error body; pass a trimmed version through.
        raise HTTPException(status_code=r.status_code,
                            detail=f"OpenRouteService error: {r.text[:200]}")
    except requests.RequestException as e:
        raise HTTPException(status_code=502,
                            detail=f"Could not reach OpenRouteService: {e}")

    data = r.json()
    try:
        feature = data["features"][0]
        coords = feature["geometry"]["coordinates"]          # [[lon,lat], ...]
        segments = feature["properties"]["segments"]
        total_m = feature["properties"]["summary"]["distance"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=502,
                            detail="No walking route found between those points.")

    steps = []
    for seg in segments:
        for s in seg.get("steps", []):
            wp = s.get("way_points", [0, 0])
            lon, lat = coords[wp[0]][0], coords[wp[0]][1]
            steps.append({
                "type": s.get("type"),               # ORS maneuver code 0..13
                "distance_m": round(s.get("distance", 0)),
                "name": s.get("name", ""),
                "lat": lat,
                "lon": lon,
            })

    return {"total_distance_m": round(total_m), "steps": steps}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vision_model": VISION_MODEL,
        "groq_key": bool(groq_key),
        "ors_key": bool(ors_key),
    }