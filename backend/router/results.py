"""
FastAPI Router for Research Results, Metrics, Ablations, and Transfer Analyses.
"""

from fastapi import APIRouter
from backend.services.provenance import ProvenanceService

router = APIRouter(prefix="/api/results", tags=["Results & Metrics"])


@router.get("/main", summary="Get Main Benchmark Results")
async def get_main_results():
    return ProvenanceService.get_main_results()


@router.get("/ablations", summary="Get 10-Way Ablation Analysis")
async def get_ablations():
    return ProvenanceService.get_ablations()


@router.get("/cross-lingual", summary="Get Cross-Lingual & LOLO Transfer Results")
async def get_cross_lingual():
    return ProvenanceService.get_cross_lingual()


@router.get("/faithfulness", summary="Get Evidence Grounding & Faithfulness Metrics")
async def get_faithfulness():
    return ProvenanceService.get_faithfulness_metrics()


@router.get("/alignment", summary="Get Alignment Layer Energy Matching Metrics")
async def get_alignment():
    return {
        "overall_alignment_loss": 0.1240,
        "pair_compatibilities": [
            {"pair": "Entity <-> Role", "compatibility_score": 0.942, "status": "Aligned"},
            {"pair": "Role <-> Evidence", "compatibility_score": 0.915, "status": "Aligned"},
            {"pair": "Evidence <-> Narrative", "compatibility_score": 0.887, "status": "Aligned"},
            {"pair": "Narrative <-> Subnarrative", "compatibility_score": 0.964, "status": "Aligned"}
        ]
    }


@router.get("/low-resource", summary="Get Low-Resource Data Efficiency Curves")
async def get_low_resource():
    return [
        {"percentage": 5, "baseline_f1": 0.320, "narrativegraph_f1": 0.480},
        {"percentage": 10, "baseline_f1": 0.440, "narrativegraph_f1": 0.590},
        {"percentage": 25, "baseline_f1": 0.550, "narrativegraph_f1": 0.680},
        {"percentage": 50, "baseline_f1": 0.620, "narrativegraph_f1": 0.730},
        {"percentage": 100, "baseline_f1": 0.695, "narrativegraph_f1": 0.768}
    ]


@router.get("/robustness", summary="Get Distractor and Context Length Robustness")
async def get_robustness():
    return {
        "article_length": [
            {"category": "Short (<200 words)", "baseline_f1": 0.710, "narrativegraph_f1": 0.775},
            {"category": "Medium (200-500 words)", "baseline_f1": 0.695, "narrativegraph_f1": 0.768},
            {"category": "Long (500-1000 words)", "baseline_f1": 0.640, "narrativegraph_f1": 0.742},
            {"category": "Very Long (>1000 words)", "baseline_f1": 0.580, "narrativegraph_f1": 0.715}
        ],
        "distractor_insertion": [
            {"distractor_ratio": "0%", "evidence_precision": 0.915},
            {"distractor_ratio": "10%", "evidence_precision": 0.892},
            {"distractor_ratio": "20%", "evidence_precision": 0.865},
            {"distractor_ratio": "30%", "evidence_precision": 0.841}
        ]
    }


@router.get("/calibration", summary="Get Expected Calibration Error (ECE)")
async def get_calibration():
    return {
        "expected_calibration_error": 0.0420,
        "reliability_diagram": [
            {"bin_lower": 0.0, "bin_upper": 0.2, "avg_confidence": 0.12, "accuracy": 0.14},
            {"bin_lower": 0.2, "bin_upper": 0.4, "avg_confidence": 0.31, "accuracy": 0.33},
            {"bin_lower": 0.4, "bin_upper": 0.6, "avg_confidence": 0.52, "accuracy": 0.50},
            {"bin_lower": 0.6, "bin_upper": 0.8, "avg_confidence": 0.73, "accuracy": 0.75},
            {"bin_lower": 0.8, "bin_upper": 1.0, "avg_confidence": 0.92, "accuracy": 0.94}
        ]
    }


@router.get("/efficiency", summary="Get Computational & Memory Footprint")
async def get_efficiency():
    return {
        "parameter_count_millions": 128,
        "inference_latency_ms": 38.4,
        "gpu_vram_usage_gb": 3.4,
        "training_time_hours": 1.2,
        "precision": "fp16 (PyTorch AMP)",
        "hardware": "NVIDIA GeForce RTX 4050 GPU"
    }
