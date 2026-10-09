from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Individual service status."""

    status: str = Field(..., description="Service status", example="healthy")
    message: Optional[str] = Field(None, description="Status message", example="Connected successfully")


class CacheOperationalStatus(BaseModel):
    """Safe process-local visibility for the optional response cache."""

    enabled: bool
    status: Literal["disabled", "healthy", "unavailable"]
    backend: Literal["disabled", "redis"]
    ttl_seconds: int = Field(gt=0)
    schema_version: str
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    bypasses: int = Field(ge=0)
    writes: int = Field(ge=0)
    read_failures: int = Field(ge=0)
    write_failures: int = Field(ge=0)
    invalid_entries: int = Field(ge=0)
    hit_rate: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall health status", example="ok")
    version: str = Field(..., description="Application version", example="0.1.0")
    environment: str = Field(..., description="Deployment environment", example="development")
    service_name: str = Field(..., description="Service identifier", example="rag-api")
    services: Optional[Dict[str, ServiceStatus]] = Field(None, description="Individual service statuses")
    cache: CacheOperationalStatus

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "environment": "development",
                "service_name": "rag-api",
                "services": {
                    "database": {"status": "healthy", "message": "Connected successfully"},
                    "pdf_parser": {"status": "healthy", "message": "Docling parser ready"},
                },
            }
        }
