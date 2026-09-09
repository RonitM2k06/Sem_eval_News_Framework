"""
FastAPI Router for Experiment Registry and Interactive Research Trace.
"""

from fastapi import APIRouter, HTTPException
from backend.services.provenance import ProvenanceService

router = APIRouter(prefix="/api/experiments", tags=["Experiment Registry & Trace"])


@router.get("", summary="Get Experiment Registry List")
async def get_experiments():
    return ProvenanceService.get_experiments_registry()


@router.get("/trace/{metric_key}", summary="Get Interactive Research Trace for any metric")
async def get_research_trace(metric_key: str):
    return ProvenanceService.get_research_trace(metric_key)


@router.get("/{experiment_id}", summary="Get detailed metadata for a specific experiment")
async def get_experiment_detail(experiment_id: str):
    experiments = ProvenanceService.get_experiments_registry()
    for exp in experiments:
        if exp.get("experiment_id") == experiment_id:
            return exp
    raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
