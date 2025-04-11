# app/config.py
import os
from pathlib import Path
from pydantic import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings"""
    
    # Basic app info
    PROJECT_NAME: str = "Fashion Rating API"
    PROJECT_DESCRIPTION: str = "API for analyzing and rating fashion outfits from images"
    VERSION: str = "0.1.0"
    
    # API configuration
    API_PREFIX: str = "/api/v1"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["*"]  # In production, specify actual origins
    
    # File upload settings
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png"]
    
    # Model settings
    MODEL_DIR: Path = Path("app/models/weights")
    
    # Performance settings
    BATCH_SIZE: int = 1
    NUM_WORKERS: int = 4
    
    # Rating settings
    SCORE_COMPONENTS: List[str] = ["fit", "color", "footwear", "accessories", "style"]
    SCORE_WEIGHTS: dict = {
        "fit": 0.25,
        "color": 0.25,
        "footwear": 0.20,
        "accessories": 0.15,
        "style": 0.15
    }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)