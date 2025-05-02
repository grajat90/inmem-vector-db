from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import libraries, documents, chunks, search

# Create FastAPI app
app = FastAPI(
    title="Vector Search API",
    description="API for managing and searching libraries, documents, and chunks",
    version="1.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(libraries.router)
app.include_router(documents.router)
app.include_router(chunks.router)
app.include_router(search.router)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Vector Database API",
        "version": "1.0.0",
        "documentation": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 