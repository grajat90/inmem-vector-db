import os
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class ProjectSettings(BaseSettings):
    # Server settings
    port: int = Field(
        default=int(os.getenv("API_PORT", 8000)),
        description="Port to run the API server on",
    )
    host: str = Field(
        default=os.getenv("API_HOST", "0.0.0.0"),
        description="Host to run the API server on",
    )

    # Embedding settings
    cohere_api_key: str = Field(
        default=os.getenv("COHERE_API_KEY"),
        description="API key for the embedding service",
    )
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "embed-v4.0"),
        description="Model to use for embeddings",
    )

    # Data storage settings
    data_directory: str = Field(
        default=os.getenv("DATA_DIRECTORY", "./data"),
        description="Directory to store library data",
    )

    @field_validator("cohere_api_key")
    def validate_cohere_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Cohere API key is not set")
        return v

    @field_validator("embedding_model")
    def validate_embedding_model(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Embedding model is not set")
        return v

    @field_validator("data_directory")
    def validate_data_directory(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Data directory is not set")
        return v

    @field_validator("host")
    def validate_host(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Host is not set")
        return v


@lru_cache
def get_project_settings() -> ProjectSettings:
    return ProjectSettings()
