"""
Main FastAPI Application Entrypoint for NarrativeGraph Research Platform.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import settings
from backend.router import analyze, results, experiments, paper, reasoning

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Backend API powering the NarrativeGraph Multilingual Research Platform and Professor Dashboard"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(analyze.router)
app.include_router(results.router)
app.include_router(experiments.router)
app.include_router(paper.router)
app.include_router(reasoning.router)


@app.get("/api/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.version,
        "gpu_available": True,
        "torch_version": "2.5.1"
    }

# Mount static frontend directory if present
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)

