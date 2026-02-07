"""
Application Configuration

Centralized configuration management with security defaults.
"""

import os
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
    
    # CORS - Strict whitelist (no wildcards in production)
    # Set via environment variable: ALLOWED_ORIGINS=["https://your-app.vercel.app"]
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ai-powered-fraud-detection-system.vercel.app",
    ]
    
    # Frontend URL (for CORS in production)
    FRONTEND_URL: str = "https://ai-powered-fraud-detection-system.vercel.app"
    
    # Rate Limiting (placeholder values)
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    # Risk Score Thresholds (calibrated for realistic fraud detection)
    # With additive scoring, multiple mild signals can stack
    # Raised thresholds to reduce false positives
    HIGH_RISK_THRESHOLD: int = 80
    MEDIUM_RISK_THRESHOLD: int = 50
    
    def get_allowed_origins(self) -> List[str]:
        """Get all allowed origins including FRONTEND_URL if set."""
        origins = list(self.ALLOWED_ORIGINS)
        if self.FRONTEND_URL and self.FRONTEND_URL not in origins:
            origins.append(self.FRONTEND_URL)
        # Also allow Vercel preview deployments
        return origins
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton settings instance
settings = Settings()
