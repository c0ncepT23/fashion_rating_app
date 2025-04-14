# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.api.endpoints import router as api_router
from app.api.mobile_routes import router as mobile_router
from app.config import settings
from fastapi.staticfiles import StaticFiles

def create_application() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    application = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Configure CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    application.include_router(api_router, prefix=settings.API_PREFIX)
    application.include_router(mobile_router, prefix=settings.API_PREFIX)
    
    return application

app = create_application()

# After creating your app:
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {"status": "online", "message": "Fashion Rating API is running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)