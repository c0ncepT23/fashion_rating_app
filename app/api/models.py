# app/api/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message")

class ClothingLabels(BaseModel):
    """Model for clothing item labels"""
    fit: str = Field(..., description="Fit style (e.g. 'structured_contemporary')")
    pants_type: str = Field(..., description="Type of pants/bottoms")
    top_type: str = Field(..., description="Type of top/upper body wear")
    color_palette: List[str] = Field(..., description="Dominant colors in the outfit")
    footwear_type: str = Field(..., description="Type of footwear")
    accessories: List[str] = Field(..., description="List of accessories detected")
    style: str = Field(..., description="Overall style classification")

class ScoreBreakdown(BaseModel):
    """Model for score breakdown by component"""
    fit: int = Field(..., ge=0, le=25, description="Fit score (0-25)")
    color: int = Field(..., ge=0, le=25, description="Color score (0-25)")
    footwear: int = Field(..., ge=0, le=20, description="Footwear score (0-20)")
    accessories: int = Field(..., ge=0, le=15, description="Accessories score (0-15)")
    style: int = Field(..., ge=0, le=15, description="Style score (0-15)")

class FeedbackComponents(BaseModel):
    """Model for textual feedback on each component"""
    fit: str = Field(..., description="Feedback on the fit")
    color: str = Field(..., description="Feedback on the color palette")
    footwear: str = Field(..., description="Feedback on the footwear")
    accessories: str = Field(..., description="Feedback on the accessories")
    style: str = Field(..., description="Feedback on the overall style")

class FashionRatingResponse(BaseModel):
    """Response model for fashion outfit rating"""
    image_path: str = Field(..., description="Path to the analyzed image")
    score: int = Field(..., ge=0, le=100, description="Overall score (0-100)")
    labels: ClothingLabels = Field(..., description="Detected clothing items and styles")
    score_breakdown: ScoreBreakdown = Field(..., description="Breakdown of scores by component")
    feedback: FeedbackComponents = Field(..., description="Detailed feedback on each aspect")
    
    class Config:
        schema_extra = {
            "example": {
                "image_path": "20250410-123045-a1b2c3d4.jpg",
                "score": 88,
                "labels": {
                    "fit": "structured_contemporary",
                    "pants_type": "neon_yellow_shorts",
                    "top_type": "gray_tee_with_black_blazer",
                    "color_palette": ["black", "gray", "neon_yellow"],
                    "footwear_type": "black_platform_sandals",
                    "accessories": ["oversized_sunglasses", "black_crossbody"],
                    "style": "edgy_chic"
                },
                "score_breakdown": {
                    "fit": 22,
                    "color": 23,
                    "footwear": 18,
                    "accessories": 15,
                    "style": 10
                },
                "feedback": {
                    "fit": "The structured black blazer creates sharp contrast with the casual shorts. The proportions balance perfectly - fitted blazer with mini shorts creates a leggy silhouette with strong shoulders.",
                    "color": "The bold neon yellow shorts create a striking focal point against the monochromatic black and gray. This strategic color blocking demonstrates confident styling and creates visual impact.",
                    "footwear": "The black platform sandals elongate the leg line and add sophistication. They coordinate with the top half while providing height and balance to the bold shorts.",
                    "accessories": "The oversized sunglasses add celebrity-inspired glamour. The minimal black crossbody is perfectly sized and maintains the clean aesthetic.",
                    "style": "Successfully combines formal and casual elements for a look that's both edgy and feminine. The outfit demonstrates confident understanding of proportion and strategic color use."
                }
            }
        }