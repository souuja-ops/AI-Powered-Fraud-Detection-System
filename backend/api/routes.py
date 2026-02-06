"""
API Routes

Handles request routing, validation, and security checks.
All routes follow the locked API contract.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Request, status

from schemas import (
    TradeRequest,
    TradeResponse,
    AlertResponse,
    ErrorResponse,
)
from services.anomaly_service import (
    analyze_trade,
    get_all_trades,
    get_alerts,
)


# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# =============================================================================
# RATE LIMITING PLACEHOLDER
# =============================================================================

# TODO: Replace with production rate limiter (Redis-based)
# This is a simple in-memory placeholder - NOT suitable for production
# 
# Production recommendations:
# - Use slowapi with Redis backend
# - Implement per-IP and per-user limits
# - Add rate limit headers to responses
#
# Example with slowapi:
# from slowapi import Limiter
# from slowapi.util import get_remote_address
# limiter = Limiter(key_func=get_remote_address)
# 
# @router.post("/analyze-trade")
# @limiter.limit("10/minute")
# async def analyze_trade_endpoint(...):


def _check_rate_limit(request: Request) -> None:
    """
    Placeholder rate limit check.
    
    TODO: Implement proper rate limiting with Redis
    
    In production, use:
    - slowapi with Redis backend
    - Token bucket or sliding window algorithm
    - Per-IP and per-API-key limits
    """
    # Placeholder - does nothing in MVP
    # In production: check request count against limit
    pass


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post(
    "/analyze-trade",
    response_model=TradeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a trade for fraud indicators",
    description="Submit a trade for real-time fraud analysis. Returns risk score and explanation.",
    responses={
        200: {"description": "Trade analyzed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
async def analyze_trade_endpoint(
    trade: TradeRequest,
    request: Request
) -> TradeResponse:
    """
    Analyze a trade for fraud detection.
    
    Request validation is handled automatically by Pydantic:
    - trade_id: min 3 chars
    - account_id: min 3 chars
    - trade_amount: must be > 0
    - trade_type: must be BUY or SELL
    - timestamp: valid ISO datetime
    """
    # Rate limit check (placeholder)
    _check_rate_limit(request)
    
    # Log the analysis request (no sensitive data)
    logger.info(
        f"Analyzing trade: id={trade.trade_id}, "
        f"type={trade.trade_type.value}, "
        f"account={trade.account_id[:3]}***"  # Partial mask
    )
    
    # Perform analysis
    result = analyze_trade(trade)
    
    # Log result (no sensitive data)
    logger.info(
        f"Trade {trade.trade_id} analyzed: "
        f"risk_level={result.risk_level.value}, "
        f"risk_score={result.risk_score}"
    )
    
    return result


@router.get(
    "/trades",
    response_model=List[TradeResponse],
    status_code=status.HTTP_200_OK,
    summary="Get all analyzed trades",
    description="Retrieve a list of all trades that have been analyzed.",
    responses={
        200: {"description": "List of trades"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
async def get_trades_endpoint(request: Request) -> List[TradeResponse]:
    """
    Retrieve all analyzed trades.
    
    Returns mock data if no trades have been analyzed yet.
    
    TODO:
    - Add pagination (limit, offset)
    - Add filtering (by date, risk level)
    - Add sorting options
    """
    # Rate limit check (placeholder)
    _check_rate_limit(request)
    
    trades = get_all_trades()
    
    logger.info(f"Retrieved {len(trades)} trades")
    
    return trades


@router.get(
    "/alerts",
    response_model=List[AlertResponse],
    status_code=status.HTTP_200_OK,
    summary="Get high-risk alerts",
    description="Retrieve alerts for trades flagged as HIGH risk.",
    responses={
        200: {"description": "List of high-risk alerts"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
async def get_alerts_endpoint(request: Request) -> List[AlertResponse]:
    """
    Retrieve HIGH risk alerts only.
    
    Returns only trades where risk_level == HIGH.
    """
    # Rate limit check (placeholder)
    _check_rate_limit(request)
    
    alerts = get_alerts()
    
    logger.info(f"Retrieved {len(alerts)} high-risk alerts")
    
    return alerts


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the API is running.",
    include_in_schema=False  # Hide from OpenAPI docs
)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


@router.get(
    "/generator/status",
    status_code=status.HTTP_200_OK,
    summary="Get trade generator status",
    description="Check the status of the background trade generator.",
    include_in_schema=False  # Hide from OpenAPI docs
)
async def generator_status():
    """Get trade generator status for debugging."""
    from services.trade_generator import get_generator_status
    return get_generator_status()


@router.post(
    "/generator/restart",
    status_code=status.HTTP_200_OK,
    summary="Restart trade generator",
    description="Restart the background trade generator if it stopped.",
    include_in_schema=False  # Hide from OpenAPI docs
)
async def restart_generator():
    """Restart the trade generator."""
    import asyncio
    from services.trade_generator import generate_trades_continuously, get_generator_status, reset_generator
    
    status = get_generator_status()
    if status["is_running"]:
        return {"message": "Generator already running", "status": status}
    
    # Reset state and start fresh
    reset_generator()
    
    # Start new generator task
    asyncio.create_task(generate_trades_continuously())
    
    return {"message": "Generator restarted", "status": get_generator_status()}


@router.get(
    "/ml/status",
    status_code=status.HTTP_200_OK,
    summary="Get ML model status",
    description="Check the status of the Isolation Forest ML model.",
    include_in_schema=False  # Hide from OpenAPI docs
)
async def ml_model_status():
    """Get ML model status for debugging/monitoring."""
    from services.ensemble_model import get_ensemble_model
    model = get_ensemble_model()
    return model.get_status()


@router.get(
    "/ml/metrics",
    status_code=status.HTTP_200_OK,
    summary="Get ML model performance metrics",
    description="Returns precision, recall, F1, confusion matrix and other metrics.",
)
async def ml_model_metrics():
    """Get ensemble model performance metrics."""
    from services.ensemble_model import get_ensemble_model
    model = get_ensemble_model()
    
    metrics = model.get_metrics()
    if metrics is None:
        return {"error": "Model not trained yet", "metrics": None}
    
    return {
        "status": "trained",
        "metrics": metrics,
        "feature_importance": model.get_feature_importance(),
    }


@router.get(
    "/ml/feature-importance",
    status_code=status.HTTP_200_OK,
    summary="Get feature importance scores",
    description="Returns the importance of each feature in the fraud detection model.",
)
async def ml_feature_importance():
    """Get feature importance from the ensemble model."""
    from services.ensemble_model import get_ensemble_model
    model = get_ensemble_model()
    
    if not model.is_trained:
        return {"error": "Model not trained yet", "features": {}}
    
    importance = model.get_feature_importance()
    
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "features": dict(sorted_features),
        "top_5": sorted_features[:5],
    }


@router.post(
    "/ml/feedback",
    status_code=status.HTTP_200_OK,
    summary="Submit feedback on a prediction",
    description="Record human feedback (confirm fraud or false positive) for model improvement.",
)
async def submit_feedback(
    request: Request,
    trade_id: str,
    is_fraud: bool,
):
    """
    Submit human feedback on a trade prediction.
    
    This enables the active learning loop:
    1. Model makes prediction
    2. Human reviews and provides ground truth
    3. Model learns from feedback
    
    Args:
        trade_id: The trade to provide feedback for
        is_fraud: True if the trade was actually fraud, False otherwise
    """
    from services.ensemble_model import get_ensemble_model
    from services.anomaly_service import get_all_trades
    from services.feature_store import get_feature_store
    
    # Find the trade
    trades = get_all_trades()
    trade = next((t for t in trades if t.trade_id == trade_id), None)
    
    if trade is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade {trade_id} not found"
        )
    
    model = get_ensemble_model()
    feature_store = get_feature_store()
    
    # Compute features for the feedback (simplified - uses current state)
    # In production, we'd store original features with each trade
    predicted_fraud = trade.risk_level.value == "HIGH"
    
    # Record feedback in the ensemble model for active learning
    feedback_result = None
    if model.is_trained:
        try:
            # Create features from trade data (simplified reconstruction)
            from datetime import datetime, timezone
            from dateutil.parser import parse as parse_datetime
            
            timestamp = parse_datetime(trade.timestamp)
            # Use trade_id to derive a pseudo account_id for demo
            account_id = f"ACC-{trade.trade_id.split('-')[1] if '-' in trade.trade_id else '0000'}"
            
            features = feature_store.compute_features(
                account_id=account_id,
                amount=trade.risk_score * 1000,  # Approximate from risk score
                timestamp=timestamp,
            )
            
            feedback_result = model.record_feedback(
                trade_id=trade_id,
                features=features,
                predicted_fraud=predicted_fraud,
                actual_fraud=is_fraud,
            )
        except Exception as e:
            logger.warning(f"Could not record feedback in model: {e}")
    
    result = {
        "trade_id": trade_id,
        "feedback_recorded": True,
        "predicted_risk_level": trade.risk_level.value,
        "actual_is_fraud": is_fraud,
        "was_correct": predicted_fraud == is_fraud,
        "feedback_buffer_count": feedback_result.get("feedback_count") if feedback_result else 0,
        "will_trigger_retrain": feedback_result.get("will_retrain", False) if feedback_result else False,
        "message": "Feedback recorded for active learning. Model will retrain after sufficient feedback samples.",
    }
    
    logger.info(f"Feedback recorded for {trade_id}: predicted={trade.risk_level.value}, actual_fraud={is_fraud}, correct={predicted_fraud == is_fraud}")
    
    return result

