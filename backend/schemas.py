"""
Pydantic Schemas

Defines all request/response models with strict validation.
These schemas EXACTLY match the frontend API contract.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from dateutil.parser import parse as parse_datetime


# =============================================================================
# ENUMS
# =============================================================================

class TradeType(str, Enum):
    """Allowed trade types."""
    BUY = "BUY"
    SELL = "SELL"
    TRANSFER = "TRANSFER"  # Wire/ACH transfers - higher risk for fraud


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class TradeRequest(BaseModel):
    """
    Request payload for trade analysis.
    
    Strict validation rules:
    - trade_id: minimum 3 characters
    - account_id: minimum 3 characters  
    - trade_amount: must be positive
    - trade_type: must be BUY or SELL
    - timestamp: valid ISO datetime string
    """
    
    trade_id: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Unique trade identifier",
        examples=["TRD-001"]
    )
    
    account_id: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Account identifier",
        examples=["ACC-7821"]
    )
    
    trade_amount: float = Field(
        ...,
        gt=0,
        le=1_000_000_000,  # 1 billion max for sanity
        description="Trade amount in USD (must be positive)",
        examples=[125000.00]
    )
    
    trade_type: TradeType = Field(
        ...,
        description="Type of trade operation",
        examples=[TradeType.BUY]
    )
    
    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp",
        examples=["2026-02-04T14:32:15.000Z"]
    )
    
    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate timestamp is a valid ISO datetime string."""
        try:
            # Parse to validate, return original string
            parse_datetime(v)
            return v
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid ISO datetime format: {v}")
    
    @field_validator("trade_id", "account_id")
    @classmethod
    def sanitize_ids(cls, v: str) -> str:
        """Sanitize ID fields - strip whitespace, basic validation."""
        v = v.strip()
        # Reject potentially dangerous characters
        if any(char in v for char in ['<', '>', '"', "'", ';', '&']):
            raise ValueError("ID contains invalid characters")
        return v


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class TradeResponse(BaseModel):
    """
    Response payload from trade analysis.
    
    Contract fields:
    - trade_id: string
    - account_id: string
    - trade_amount: float
    - trade_type: BUY | SELL | TRANSFER
    - risk_score: integer (0-100)
    - risk_level: LOW | MEDIUM | HIGH
    - anomaly_score: float
    - explanation: string
    - timestamp: ISO datetime string
    """
    
    trade_id: str = Field(
        ...,
        description="Unique trade identifier"
    )
    
    account_id: str = Field(
        ...,
        description="Account identifier"
    )
    
    trade_amount: float = Field(
        ...,
        gt=0,
        description="Trade amount in USD"
    )
    
    trade_type: TradeType = Field(
        ...,
        description="Type of trade operation"
    )
    
    risk_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Risk score from 0 to 100"
    )
    
    risk_level: RiskLevel = Field(
        ...,
        description="Categorized risk level"
    )
    
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Anomaly detection score (normalized 0-1)"
    )
    
    explanation: str = Field(
        ...,
        description="Human-readable explanation of risk assessment"
    )
    
    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "trade_id": "TRD-001",
                "account_id": "ACC-7821",
                "trade_amount": 125000.00,
                "trade_type": "BUY",
                "risk_score": 85,
                "risk_level": "HIGH",
                "anomaly_score": 0.85,
                "explanation": "Unusual trade volume detected",
                "timestamp": "2026-02-04T14:32:15.000Z"
            }
        }


class AlertResponse(BaseModel):
    """
    Alert response for HIGH risk trades only.
    
    Contract fields:
    - trade_id: string
    - risk_score: integer
    - explanation: string
    - timestamp: ISO datetime string
    """
    
    trade_id: str = Field(
        ...,
        description="Trade identifier that triggered the alert"
    )
    
    risk_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Risk score"
    )
    
    explanation: str = Field(
        ...,
        description="Explanation of why alert was triggered"
    )
    
    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "trade_id": "TRD-003",
                "risk_score": 92,
                "explanation": "Cross-border transfer to flagged jurisdiction",
                "timestamp": "2026-02-04T14:25:12.000Z"
            }
        }


# =============================================================================
# ERROR SCHEMAS
# =============================================================================

class ErrorResponse(BaseModel):
    """Standardized error response (safe for clients)."""
    
    error: str = Field(
        ...,
        description="Error type"
    )
    
    message: str = Field(
        ...,
        description="Safe, user-friendly error message"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid trade amount: must be greater than 0"
            }
        }
