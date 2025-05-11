from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import load_libraries_from_disk
from app.api.exceptions.handlers import register_exception_handlers
from app.api.router import router
from app.config.project_settings import get_project_settings

config = get_project_settings()
# Create FastAPI app
app = FastAPI(
    title="Vector Search API",
    description="API for managing and searching vector embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "libraries", "description": "Operations with libraries"},
        {"name": "documents", "description": "Operations with documents"},
        {"name": "chunks", "description": "Operations with chunks"},
        {"name": "search", "description": "Operations with search"},
    ],
    openapi_url="/api/openapi.json",
)

register_exception_handlers(app)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register main router
app.include_router(router)

app.add_event_handler("startup", load_libraries_from_disk)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Vector Database API",
        "version": "1.0.0",
        "documentation": "/docs",
    }


if __name__ == "__main__":
    # reinstate all libraries from data folder if any
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
