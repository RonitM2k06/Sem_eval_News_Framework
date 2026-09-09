"""
FastAPI Router for Live Article Analysis and Demo Samples.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from backend.services.inference import LiveInferenceService

router = APIRouter(prefix="/api/analyze", tags=["Live Analysis"])
service = LiveInferenceService()


class ArticleAnalysisRequest(BaseModel):
    text: str = Field(..., description="Full text of the news article to analyze", min_length=10)
    language: Optional[str] = Field("en", description="Language code (bg, en, hi, pt, ru)")
    domain: Optional[str] = Field("ukraine_russia", description="Domain taxonomy (ukraine_russia, climate_change)")


@router.post("", summary="Analyze news article text using NarrativeGraph")
async def analyze_article(req: ArticleAnalysisRequest):
    try:
        result = service.analyze_article(
            text=req.text,
            language=req.language,
            domain=req.domain
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/demo", summary="Get curated multilingual demo articles for presentation")
async def get_demo_articles():
    return service.get_demo_articles()
