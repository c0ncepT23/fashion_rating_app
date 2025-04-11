# app/api/endpoints.py
import os
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import aiofiles
from datetime import datetime

from app.config import settings
from app.api.models import (
    FashionRatingResponse,
    ErrorResponse
)
from app.core.pipeline import process_fashion_image

router = APIRouter()

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save uploaded file to the uploads directory
    
    Args:
        upload_file: The uploaded file
        
    Returns:
        str: Path to the saved file
    """
    # Create unique filename
    file_extension = upload_file.filename.split(".")[-1].lower()
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Create unique filename with timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    new_filename = f"{timestamp}-{unique_id}.{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, new_filename)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE/1024/1024}MB"
            )
        await out_file.write(content)
    
    return file_path

@router.post(
    "/rate-outfit", 
    response_model=FashionRatingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def rate_outfit(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Rate a fashion outfit from an uploaded image.
    
    - **file**: JPG or PNG image of the outfit to be rated
    
    Returns a detailed rating including scores, style classification, and feedback.
    """
    try:
        # Save uploaded file
        file_path = await save_upload_file(file)
        
        # Process the image and get rating
        rating_result = await process_fashion_image(file_path)
        
        # Add the file path to the response
        rating_result["image_path"] = os.path.basename(file_path)
        
        # Clean up file in background if needed
        # background_tasks.add_task(os.remove, file_path)
        
        return rating_result
        
    except Exception as e:
        # Log the error
        print(f"Error processing image: {str(e)}")
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}