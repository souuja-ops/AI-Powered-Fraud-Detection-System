"""
AI Fraud Detection API

FastAPI backend for real-time trade fraud detection.
Security-first implementation with strict input validation.

Features:
- Real-time trade analysis with multi-signal fraud detection
- Background trade generator for demo/testing
- Explainable risk scoring
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from core.config import settings
from api.routes import router
from schemas import ErrorResponse


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


# =============================================================================
# BACKGROUND TASK MANAGEMENT
# =============================================================================

# Global reference to the generator task
_generator_task: Optional[asyncio.Task] = None


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Startup:
    - Log configuration
    - Initialize ensemble ML model
    - Start background trade generator
    
    Shutdown:
    - Stop trade generator gracefully
    """
    global _generator_task
    
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"CORS allowed origins: {settings.ALLOWED_ORIGINS}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize Ensemble ML Model (Isolation Forest + XGBoost)
    from services.ensemble_model import initialize_ensemble_model, get_ensemble_model
    
    logger.info("=" * 60)
    logger.info("Initializing Ensemble Fraud Detection Model...")
    logger.info("  - Isolation Forest (unsupervised anomaly detection)")
    logger.info("  - XGBoost (supervised fraud classification)")
    logger.info("  - Heuristic Rules (domain knowledge)")
    logger.info("=" * 60)
    
    try:
        metrics = initialize_ensemble_model(n_samples=5000, fraud_ratio=0.1)
        logger.info("✅ Ensemble model trained successfully!")
        logger.info(f"   Precision: {metrics.precision:.3f}")
        logger.info(f"   Recall:    {metrics.recall:.3f}")
        logger.info(f"   F1-Score:  {metrics.f1_score:.3f}")
        logger.info(f"   ROC-AUC:   {metrics.roc_auc:.3f}")
    except Exception as e:
        logger.error(f"❌ Ensemble model initialization failed: {e}")
        logger.warning("Falling back to heuristics only")
    
    # Also initialize the legacy ML model for backward compatibility
    from services.anomaly_service import initialize_ml_model
    
    logger.info("Initializing legacy ML model...")
    if initialize_ml_model():
        logger.info("✅ Legacy ML model initialized")
    else:
        logger.warning("⚠️ Legacy ML model initialization failed")
    
    # Start the trade generator in the background
    from services.trade_generator import generate_trades_continuously, stop_generator
    
    logger.info("Starting background trade generator...")
    _generator_task = asyncio.create_task(generate_trades_continuously())
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Stop the trade generator gracefully
    if _generator_task:
        stop_generator()
        _generator_task.cancel()
        try:
            await _generator_task
        except asyncio.CancelledError:
            pass
        logger.info("Trade generator stopped")
    
    logger.info("Application shutdown complete")


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Real-time AI-powered fraud detection for financial trades",
        docs_url="/docs" if settings.DEBUG else None,  # Disable in production
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )
    
    # -------------------------------------------------------------------------
    # CORS MIDDLEWARE (Strict - No wildcards)
    # -------------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,  # Only http://localhost:3000
        allow_credentials=True,
        allow_methods=["GET", "POST"],  # Only methods we use
        allow_headers=["Content-Type", "Accept"],  # Minimal headers
        max_age=600,  # Cache preflight for 10 minutes
    )
    
    # -------------------------------------------------------------------------
    # EXCEPTION HANDLERS (Centralized, safe error messages)
    # -------------------------------------------------------------------------
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle Pydantic validation errors.
        Returns safe, user-friendly error messages without exposing internals.
        """
        # Extract first error for user-friendly message
        errors = exc.errors()
        if errors:
            first_error = errors[0]
            field = " -> ".join(str(loc) for loc in first_error.get("loc", []))
            msg = first_error.get("msg", "Invalid value")
            safe_message = f"Validation error in '{field}': {msg}"
        else:
            safe_message = "Invalid request data"
        
        logger.warning(f"Validation error: {safe_message}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="validation_error",
                message=safe_message
            ).model_dump()
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        request: Request,
        exc: ValidationError
    ) -> JSONResponse:
        """Handle Pydantic model validation errors."""
        logger.warning(f"Pydantic validation error: {exc.error_count()} errors")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="validation_error",
                message="Invalid request data format"
            ).model_dump()
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request,
        exc: ValueError
    ) -> JSONResponse:
        """Handle value errors with safe messages."""
        # Don't expose internal error details
        logger.error(f"Value error: {str(exc)}")
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="bad_request",
                message="Invalid request parameters"
            ).model_dump()
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """
        Global exception handler.
        NEVER expose stack traces or internal errors to clients.
        """
        # Log the full error internally
        logger.exception(f"Unhandled exception: {type(exc).__name__}")
        
        # Return safe generic message to client
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred. Please try again later."
            ).model_dump()
        )
    
    # -------------------------------------------------------------------------
    # ROUTES
    # -------------------------------------------------------------------------
    
    # Include API routes
    app.include_router(router, prefix="/api", tags=["Fraud Detection"])
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint - basic info."""
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running"
        }
    
    return app


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

app = create_app()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
        access_log=True,
    )
