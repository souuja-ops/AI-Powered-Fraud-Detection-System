"""
Feature Store Service

Centralized feature engineering for fraud detection ML models.
Maintains account profiles, computes features, and provides consistent
feature vectors for both training and inference.

Features computed:
- Account-level: age, lifetime stats, risk history
- Transaction-level: amount, time, type
- Behavioral: velocity, deviation from norm, sequence patterns
- Derived: ratios, z-scores, rolling statistics

This module ensures feature consistency between training and prediction.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class AccountProfile(NamedTuple):
    """Comprehensive account profile for feature computation."""
    account_id: str
    created_at: datetime
    total_trades: int
    total_volume: float
    avg_trade_amount: float
    std_trade_amount: float
    max_trade_amount: float
    avg_trades_per_day: float
    preferred_hours: List[int]  # Most common trading hours
    risk_flags_count: int       # Historical flags
    last_trade_at: Optional[datetime]


class TransactionFeatures(NamedTuple):
    """Complete feature vector for a single transaction."""
    # Account features
    account_age_days: float
    account_total_trades: int
    account_avg_amount: float
    account_std_amount: float
    account_risk_history: int
    
    # Transaction features
    amount: float
    amount_log: float  # Log-transformed amount
    hour: int
    day_of_week: int
    is_weekend: int
    is_unusual_hour: int  # Outside 6am-10pm
    
    # Behavioral features
    amount_zscore: float          # How unusual is this amount for the account
    velocity_1min: int            # Trades in last 1 minute
    velocity_5min: int            # Trades in last 5 minutes
    velocity_10min: int           # Trades in last 10 minutes
    time_since_last_trade: float  # Seconds since last trade (0 if first)
    amount_vs_avg_ratio: float    # amount / account_avg
    amount_vs_max_ratio: float    # amount / account_max
    
    # Sequence features
    is_first_trade: int           # First trade for this account
    is_amount_spike: int          # Amount > 3x account average
    is_rapid_succession: int      # < 10 seconds since last trade
    hour_deviation: float         # Distance from preferred trading hours
    
    # Risk indicators (derived)
    risk_score_heuristic: float   # Pre-computed heuristic risk (0-1)


# =============================================================================
# FEATURE STORE (In-Memory)
# =============================================================================

class FeatureStore:
    """
    In-memory feature store for fraud detection.
    
    Maintains:
    - Account profiles with historical statistics
    - Recent transaction cache for velocity calculations
    - Feature computation logic
    
    In production, this would be backed by Redis/PostgreSQL.
    """
    
    def __init__(self):
        # Account profiles: account_id -> AccountProfile
        self._accounts: Dict[str, AccountProfile] = {}
        
        # Recent transactions per account: account_id -> [(timestamp, amount), ...]
        self._recent_transactions: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Transaction history for sequence analysis
        self._transaction_sequence: Dict[str, List[dict]] = defaultdict(list)
        
        # Global statistics for normalization
        self._global_stats = {
            "total_transactions": 0,
            "total_amount": 0.0,
            "amount_mean": 500.0,  # Initial estimates
            "amount_std": 1000.0,
        }
        
        # Sliding window size
        self._max_recent_transactions = 100
        
        logger.info("FeatureStore initialized")
    
    def get_or_create_account(self, account_id: str) -> AccountProfile:
        """Get existing account profile or create new one."""
        if account_id not in self._accounts:
            self._accounts[account_id] = AccountProfile(
                account_id=account_id,
                created_at=datetime.now(timezone.utc),
                total_trades=0,
                total_volume=0.0,
                avg_trade_amount=0.0,
                std_trade_amount=0.0,
                max_trade_amount=0.0,
                avg_trades_per_day=0.0,
                preferred_hours=[9, 10, 11, 14, 15, 16],  # Default business hours
                risk_flags_count=0,
                last_trade_at=None,
            )
        return self._accounts[account_id]
    
    def update_account(
        self,
        account_id: str,
        amount: float,
        timestamp: datetime,
        was_flagged: bool = False
    ) -> None:
        """Update account profile after a transaction."""
        profile = self.get_or_create_account(account_id)
        
        # Compute new statistics
        new_total_trades = profile.total_trades + 1
        new_total_volume = profile.total_volume + amount
        new_avg = new_total_volume / new_total_trades
        
        # Compute running std (Welford's algorithm approximation)
        if new_total_trades == 1:
            new_std = 0.0
        else:
            delta = amount - profile.avg_trade_amount
            new_std = np.sqrt(
                ((profile.std_trade_amount ** 2) * (profile.total_trades - 1) + delta * (amount - new_avg))
                / (new_total_trades - 1)
            ) if new_total_trades > 1 else 0.0
        
        new_max = max(profile.max_trade_amount, amount)
        
        # Compute trades per day
        account_age_days = max((timestamp - profile.created_at).days, 1)
        new_avg_per_day = new_total_trades / account_age_days
        
        # Update preferred hours
        hour = timestamp.hour
        new_preferred = list(profile.preferred_hours)
        if hour not in new_preferred and len(new_preferred) < 10:
            new_preferred.append(hour)
        
        # Update risk flags
        new_risk_flags = profile.risk_flags_count + (1 if was_flagged else 0)
        
        # Create updated profile
        self._accounts[account_id] = AccountProfile(
            account_id=account_id,
            created_at=profile.created_at,
            total_trades=new_total_trades,
            total_volume=new_total_volume,
            avg_trade_amount=new_avg,
            std_trade_amount=new_std,
            max_trade_amount=new_max,
            avg_trades_per_day=new_avg_per_day,
            preferred_hours=new_preferred,
            risk_flags_count=new_risk_flags,
            last_trade_at=timestamp,
        )
        
        # Update recent transactions cache
        self._recent_transactions[account_id].append((timestamp, amount))
        # Keep only recent
        if len(self._recent_transactions[account_id]) > self._max_recent_transactions:
            self._recent_transactions[account_id] = self._recent_transactions[account_id][-self._max_recent_transactions:]
        
        # Update global stats
        self._global_stats["total_transactions"] += 1
        self._global_stats["total_amount"] += amount
        self._global_stats["amount_mean"] = (
            self._global_stats["total_amount"] / self._global_stats["total_transactions"]
        )
    
    def compute_features(
        self,
        account_id: str,
        amount: float,
        timestamp: datetime,
        trade_type: str = "BUY"
    ) -> TransactionFeatures:
        """
        Compute complete feature vector for a transaction.
        
        This is the main entry point for feature engineering.
        Returns consistent features for both training and inference.
        """
        profile = self.get_or_create_account(account_id)
        recent = self._recent_transactions.get(account_id, [])
        
        # Account features
        account_age_days = max((timestamp - profile.created_at).total_seconds() / 86400, 0.01)
        
        # Transaction features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_unusual_hour = 1 if hour < 6 or hour > 22 else 0
        amount_log = np.log1p(amount)  # log(1 + amount) to handle 0
        
        # Behavioral features - Amount analysis
        if profile.total_trades > 0 and profile.std_trade_amount > 0:
            amount_zscore = (amount - profile.avg_trade_amount) / profile.std_trade_amount
        else:
            amount_zscore = 0.0
        
        # Velocity calculations
        now = timestamp
        velocity_1min = sum(1 for t, _ in recent if (now - t).total_seconds() <= 60)
        velocity_5min = sum(1 for t, _ in recent if (now - t).total_seconds() <= 300)
        velocity_10min = sum(1 for t, _ in recent if (now - t).total_seconds() <= 600)
        
        # Time since last trade
        if profile.last_trade_at:
            time_since_last = (timestamp - profile.last_trade_at).total_seconds()
        else:
            time_since_last = 0.0
        
        # Ratios
        amount_vs_avg = amount / profile.avg_trade_amount if profile.avg_trade_amount > 0 else 1.0
        amount_vs_max = amount / profile.max_trade_amount if profile.max_trade_amount > 0 else 1.0
        
        # Sequence features
        is_first_trade = 1 if profile.total_trades == 0 else 0
        is_amount_spike = 1 if amount_vs_avg > 3.0 else 0
        is_rapid_succession = 1 if 0 < time_since_last < 10 else 0
        
        # Hour deviation from preferred
        if profile.preferred_hours:
            hour_deviation = min(abs(hour - h) for h in profile.preferred_hours)
        else:
            hour_deviation = 0.0
        
        # Compute heuristic risk score (0-1)
        risk_score = self._compute_heuristic_risk(
            amount=amount,
            amount_zscore=amount_zscore,
            velocity_10min=velocity_10min,
            is_unusual_hour=is_unusual_hour,
            is_rapid_succession=is_rapid_succession,
            is_amount_spike=is_amount_spike,
            account_age_days=account_age_days,
        )
        
        return TransactionFeatures(
            account_age_days=account_age_days,
            account_total_trades=profile.total_trades,
            account_avg_amount=profile.avg_trade_amount,
            account_std_amount=profile.std_trade_amount,
            account_risk_history=profile.risk_flags_count,
            amount=amount,
            amount_log=amount_log,
            hour=hour,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            is_unusual_hour=is_unusual_hour,
            amount_zscore=amount_zscore,
            velocity_1min=velocity_1min,
            velocity_5min=velocity_5min,
            velocity_10min=velocity_10min,
            time_since_last_trade=time_since_last,
            amount_vs_avg_ratio=amount_vs_avg,
            amount_vs_max_ratio=amount_vs_max,
            is_first_trade=is_first_trade,
            is_amount_spike=is_amount_spike,
            is_rapid_succession=is_rapid_succession,
            hour_deviation=hour_deviation,
            risk_score_heuristic=risk_score,
        )
    
    def _compute_heuristic_risk(
        self,
        amount: float,
        amount_zscore: float,
        velocity_10min: int,
        is_unusual_hour: int,
        is_rapid_succession: int,
        is_amount_spike: int,
        account_age_days: float,
    ) -> float:
        """Compute heuristic risk score (0-1) based on rules."""
        risk = 0.0
        
        # Amount risk
        if amount > 100000:
            risk += 0.3
        elif amount > 50000:
            risk += 0.15
        elif amount > 10000:
            risk += 0.05
        
        # Z-score risk
        if abs(amount_zscore) > 3:
            risk += 0.2
        elif abs(amount_zscore) > 2:
            risk += 0.1
        
        # Velocity risk
        if velocity_10min > 5:
            risk += 0.25
        elif velocity_10min > 3:
            risk += 0.15
        
        # Time risk
        if is_unusual_hour:
            risk += 0.1
        
        # Sequence risk
        if is_rapid_succession:
            risk += 0.15
        if is_amount_spike:
            risk += 0.15
        
        # New account risk
        if account_age_days < 1:
            risk += 0.1
        
        return min(risk, 1.0)
    
    def features_to_array(self, features: TransactionFeatures) -> np.ndarray:
        """Convert TransactionFeatures to numpy array for ML models."""
        return np.array([
            features.account_age_days,
            features.account_total_trades,
            features.account_avg_amount,
            features.account_std_amount,
            features.account_risk_history,
            features.amount,
            features.amount_log,
            features.hour,
            features.day_of_week,
            features.is_weekend,
            features.is_unusual_hour,
            features.amount_zscore,
            features.velocity_1min,
            features.velocity_5min,
            features.velocity_10min,
            features.time_since_last_trade,
            features.amount_vs_avg_ratio,
            features.amount_vs_max_ratio,
            features.is_first_trade,
            features.is_amount_spike,
            features.is_rapid_succession,
            features.hour_deviation,
            features.risk_score_heuristic,
        ])
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names in order."""
        return [
            "account_age_days",
            "account_total_trades",
            "account_avg_amount",
            "account_std_amount",
            "account_risk_history",
            "amount",
            "amount_log",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_unusual_hour",
            "amount_zscore",
            "velocity_1min",
            "velocity_5min",
            "velocity_10min",
            "time_since_last_trade",
            "amount_vs_avg_ratio",
            "amount_vs_max_ratio",
            "is_first_trade",
            "is_amount_spike",
            "is_rapid_succession",
            "hour_deviation",
            "risk_score_heuristic",
        ]
    
    def get_stats(self) -> dict:
        """Get feature store statistics."""
        return {
            "total_accounts": len(self._accounts),
            "total_transactions": self._global_stats["total_transactions"],
            "global_amount_mean": round(self._global_stats["amount_mean"], 2),
        }
    
    def clear(self) -> None:
        """Clear all stored data."""
        self._accounts.clear()
        self._recent_transactions.clear()
        self._transaction_sequence.clear()
        self._global_stats = {
            "total_transactions": 0,
            "total_amount": 0.0,
            "amount_mean": 500.0,
            "amount_std": 1000.0,
        }
        logger.info("FeatureStore cleared")


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get the global feature store instance."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store


def reset_feature_store() -> None:
    """Reset the feature store (for testing)."""
    global _feature_store
    if _feature_store:
        _feature_store.clear()
    _feature_store = FeatureStore()
