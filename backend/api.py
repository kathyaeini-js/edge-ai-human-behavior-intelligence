from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Edge Behavior Intelligence API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

EVENTS_FILE = Path("data/outputs/events.jsonl")

@app.get("/health")
def health():
    return {"status": "ok", "events_file_exists": EVENTS_FILE.exists()}

@app.get("/events")
def recent_events(limit: int = Query(default=100, ge=1, le=1000)):
    if not EVENTS_FILE.exists():
        return {"items": [], "message": "No events yet. Run process_video.py first."}
    rows = []
    with EVENTS_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"items": rows[-limit:][::-1]}
