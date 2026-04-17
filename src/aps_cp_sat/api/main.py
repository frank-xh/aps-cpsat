from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aps_cp_sat.persistence.db import load_env
from aps_cp_sat.api.routers import comparison, runs, slots, violations


def create_app() -> FastAPI:
    load_env()
    app = FastAPI(title="APS Schedule Analysis API", version="0.1.0")

    allow_origins = os.getenv("ANALYSIS_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allow_origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    app.include_router(runs.router, prefix="/runs", tags=["runs"])
    app.include_router(slots.router, prefix="/runs", tags=["slots"])
    app.include_router(violations.router, prefix="/runs", tags=["violations"])
    app.include_router(comparison.router, prefix="/compare", tags=["comparison"])

    return app


app = create_app()

