from fastapi import APIRouter
from sqlalchemy import text
from src.dependencies import DatabaseDep, RAGResponseCacheDep, SettingsDep
from src.schemas.health import CacheOperationalStatus, HealthResponse, ServiceStatus
from src.services.cache import RAG_CACHE_SCHEMA_VERSION

router = APIRouter()


@router.get("/ping", tags=["Health"])
async def ping():
    """Simple ping endpoint for basic connectivity tests."""
    return {"status": "ok", "message": "pong"}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health and status of the API service including database connectivity.",
    response_description="Service health information",
    tags=["Health"],
)
async def health_check(
    settings: SettingsDep,
    database: DatabaseDep,
    response_cache: RAGResponseCacheDep,
) -> HealthResponse:
    """
    Comprehensive health check endpoint for monitoring and load balancer probes.

    This endpoint provides information about the service health, version,
    environment, and checks connectivity to dependent services like database.

    Returns:
        HealthResponse: Contains service status, version, environment, and service checks

    Example:
        Response:
        ```
        {
            "status": "ok",
            "version": "0.1.0",
            "environment": "development",
            "service_name": "rag-api",
            "services": {
                "database": {"status": "healthy", "message": "Connected successfully"}
            }
        }
        ```
    """
    services = {}
    overall_status = "ok"

    # Test database connectivity
    try:
        with database.get_session() as session:
            # Simple query to test connection
            session.execute(text("SELECT 1"))
            services["database"] = ServiceStatus(status="healthy", message="Connected successfully")
    except Exception as e:
        services["database"] = ServiceStatus(status="unhealthy", message=f"Connection failed: {str(e)}")
        overall_status = "degraded"

    cache_status = await response_cache.health()
    cache_stats = response_cache.stats_snapshot()

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
        cache=CacheOperationalStatus(
            enabled=response_cache.enabled,
            status=cache_status.value,
            backend="redis" if response_cache.enabled else "disabled",
            ttl_seconds=response_cache.ttl_seconds,
            schema_version=RAG_CACHE_SCHEMA_VERSION,
            hits=cache_stats.hits,
            misses=cache_stats.misses,
            bypasses=cache_stats.bypasses,
            writes=cache_stats.writes,
            read_failures=cache_stats.read_failures,
            write_failures=cache_stats.write_failures,
            invalid_entries=cache_stats.invalid_entries,
            hit_rate=cache_stats.hit_rate,
        ),
    )
