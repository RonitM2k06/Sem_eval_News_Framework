"""
Unified NarrativeGraph Research Dashboard Launcher.
Starts the FastAPI Backend and serves the Interactive Research Dashboard.
"""

import sys
import os
sys.path.insert(0, ".")

import uvicorn
from backend.config import settings


def launch_dashboard():
    print("=================================================================")
    print("      NARRATIVEGRAPH: RESEARCH PLATFORM & PROFESSOR DASHBOARD    ")
    print("=================================================================")
    print(f"[Launcher] Starting FastAPI Server on http://127.0.0.1:8000")
    print(f"[Launcher] Interactive Dashboard available at http://127.0.0.1:8000")
    print(f"[Launcher] API Documentation available at http://127.0.0.1:8000/docs")
    print("=================================================================")

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    launch_dashboard()
