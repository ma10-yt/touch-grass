from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.api import games, submissions
from app.db.client import connect_to_mongo, close_mongo_connection
from app.config import settings
import httpx
import os

logging.basicConfig()

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo_connection()

app = FastAPI(title="Touch Grass API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    issues = [
        {
            "location": error.get("loc"),
            "type": error.get("type"),
            "message": error.get("msg"),
        }
        for error in exc.errors()
    ]
    logger.warning(
        "Request validation failed method=%s path=%s issues=%s",
        request.method,
        request.url.path,
        issues,
    )
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(games.router, prefix="/api/games", tags=["games"])
app.include_router(submissions.router, prefix="/api/games", tags=["submissions"])

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Go outside and touch grass!",
        "demo_mode": settings.DEMO_MODE,
    }


@app.get("/api/health/ai")
async def ai_health_check():
    if settings.AI_PROVIDER == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            return {
                "provider": "openrouter",
                "reachable": False,
                "model": settings.OPENROUTER_MODEL,
                "model_available": False,
                "detail": "OpenRouter is selected but OPENROUTER_API_KEY is not configured.",
            }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                response = await client.get(
                    f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/models",
                    headers={
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"
                    },
                )
                response.raise_for_status()
                payload = response.json()
            models = payload.get("data", []) if isinstance(payload, dict) else []
            model_available = any(
                isinstance(model, dict)
                and model.get("id") == settings.OPENROUTER_MODEL
                for model in models
            )
            return {
                "provider": "openrouter",
                "reachable": True,
                "model": settings.OPENROUTER_MODEL,
                "model_available": model_available,
            }
        except (httpx.HTTPError, ValueError):
            return {
                "provider": "openrouter",
                "reachable": False,
                "model": settings.OPENROUTER_MODEL,
                "model_available": False,
                "detail": "Could not connect to the configured OpenRouter service.",
            }

    ollama_url = settings.OLLAMA_URL.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        available_names = {
            str(model.get("name") or model.get("model") or "")
            for model in models
            if isinstance(model, dict)
        }
        model_available = settings.GEMMA_MODEL in available_names
        result = {
            "ollama_reachable": True,
            "model": settings.GEMMA_MODEL,
            "model_available": model_available,
        }
        if not model_available:
            result["detail"] = (
                f"Gemma model '{settings.GEMMA_MODEL}' is not installed. "
                f"Run: ollama pull {settings.GEMMA_MODEL}"
            )
        return result
    except (httpx.HTTPError, ValueError):
        return {
            "ollama_reachable": False,
            "model": settings.GEMMA_MODEL,
            "model_available": False,
            "detail": "Could not connect to the configured local Ollama service.",
        }

# Serve frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
