"""
Application Configuration

Centralized configuration management with security defaults.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings with secure defaults."""
    
    # Application
    APP_NAME: str = "AI Fraud Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS - Strict whitelist (no wildcards)
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # Rate Limiting (placeholder values)
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    # Risk Score Thresholds
    HIGH_RISK_THRESHOLD: int = 70
    MEDIUM_RISK_THRESHOLD: int = 40
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton settings instance
settings = Settings()
