"""
Anomaly Detection Service

Combines deterministic rule-based fraud detection with ML-based anomaly detection
using Isolation Forest for enhanced risk scoring.

Signals:
1. Amount Anomaly - Large transaction amounts increase risk (Heuristic)
2. Velocity Burst - Multiple trades in short time window (Heuristic)
3. Time-based Anomaly - Trades at unusual hours (Heuristic)
4. Behavior Drift - Deviation from account's historical average (Heuristic)
5. ML Anomaly Score - Isolation Forest unsupervised anomaly detection (ML)

All logic produces deterministic, explainable results.

ML INTEGRATION NOTES:
- Isolation Forest runs in-memory only (no persistence)
- Model is trained on startup with synthetic sample data
- Can be incrementally updated with new trades (in-memory)
- Training is fast (<2 seconds on small samples)
- ML score is combined as an additional signal to heuristics
"""

import logging
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, NamedTuple, Optional
from dateutil.parser import parse as parse_datetime

# =============================================================================
# ML IMPORTS (Isolation Forest)
# =============================================================================
# TODO: For production, consider:
# - Model versioning and persistence
# - A/B testing framework
# - Model performance monitoring
# - Feature store integration

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from schemas import TradeRequest, TradeResponse, AlertResponse, RiskLevel
from core.config import settings


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class SignalResult(NamedTuple):
    """Result from a single fraud signal evaluation."""
    score: int            # Direct contribution to risk score (0-100)
    triggered: bool       # Whether this signal was triggered
    explanation: str      # Human-readable explanation


class TradeRecord(NamedTuple):
    """Lightweight record for tracking trade history."""
    trade_id: str
    amount: float
    timestamp: datetime
    trade_type: str
    account_id: str       # Added for ML features


class MLFeatures(NamedTuple):
    """Feature vector for ML model."""
    amount: float
    hour: int
    is_unusual_hour: int
    trades_last_10min: int
    amount_vs_avg: float  # ratio to account average


# =============================================================================
# IN-MEMORY STORAGE (Stateless across restarts)
# =============================================================================

# Analyzed trades storage (for API responses)
_analyzed_trades: List[TradeResponse] = []

# Per-account trade history for behavior analysis
# Key: account_id, Value: list of recent TradeRecords
_account_history: Dict[str, List[TradeRecord]] = {}

# Configuration constants for fraud detection
_CONFIG = {
    # Amount thresholds (in USD) - RAISED for realistic trading
    # Most legitimate trades are under $25K, institutional can be $100K+
    "AMOUNT_LOW": 25_000,           # Below this = minimal risk
    "AMOUNT_MEDIUM": 75_000,        # Above this = moderate risk
    "AMOUNT_HIGH": 150_000,         # Above this = elevated risk
    "AMOUNT_VERY_HIGH": 500_000,    # Above this = high risk
    
    # Velocity detection
    "VELOCITY_WINDOW_MINUTES": 10,  # Time window for burst detection
    "VELOCITY_THRESHOLD": 4,        # Trades in window to trigger (raised from 3)
    
    # Time-based detection (unusual hours in UTC)
    "UNUSUAL_HOUR_START": 1,        # 1 AM UTC
    "UNUSUAL_HOUR_END": 5,          # 5 AM UTC
    
    # Behavior drift
    "MIN_HISTORY_FOR_DRIFT": 5,     # Minimum trades needed for drift detection (raised from 3)
    "DRIFT_MULTIPLIER": 4.0,        # Current amount / avg > this = anomaly (raised from 3.0)
    
    # History limits
    "MAX_HISTORY_PER_ACCOUNT": 100, # Cap history to prevent memory bloat
    
    # ML Configuration
    "ML_SIGNAL_MAX_SCORE": 20,      # Maximum points from ML signal (reduced from 25)
    "ML_CONTAMINATION": 0.05,       # Expected proportion of anomalies (reduced from 0.1)
    "ML_N_ESTIMATORS": 100,         # Number of trees in forest
    "ML_RANDOM_STATE": 42,          # Fixed seed for deterministic results
    "ML_MIN_SAMPLES_TO_TRAIN": 20,  # Minimum samples before training
    "ML_RETRAIN_INTERVAL": 50,      # Retrain every N new trades
}


# =============================================================================
# ML MODEL STATE
# =============================================================================

class MLModelState:
    """
    Manages the Isolation Forest model state.
    
    TODO for production:
    - Add model serialization/deserialization
    - Add feature importance tracking
    - Add model drift detection
    - Add shadow mode for new models
    """
    
    def __init__(self):
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False
        self.training_samples: int = 0
        self.trades_since_retrain: int = 0
        self.feature_buffer: List[np.ndarray] = []  # Buffer for incremental updates
        
    def reset(self):
        """Reset model state (for testing)."""
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_samples = 0
        self.trades_since_retrain = 0
        self.feature_buffer = []


# Global ML model state
_ml_state = MLModelState()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime object."""
    dt = parse_datetime(timestamp_str)
    # Ensure timezone-aware (default to UTC if naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _calculate_risk_level(risk_score: int) -> RiskLevel:
    """
    Determine risk level from score.
    
    Thresholds (from config):
    - HIGH: >= 70
    - MEDIUM: >= 40
    - LOW: < 40
    """
    if risk_score >= settings.HIGH_RISK_THRESHOLD:
        return RiskLevel.HIGH
    elif risk_score >= settings.MEDIUM_RISK_THRESHOLD:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _get_account_history(account_id: str) -> List[TradeRecord]:
    """Get trade history for an account (empty list if none)."""
    return _account_history.get(account_id, [])


def _add_to_history(account_id: str, record: TradeRecord) -> None:
    """Add a trade to account history with size limit."""
    if account_id not in _account_history:
        _account_history[account_id] = []
    
    _account_history[account_id].append(record)
    
    # Enforce history size limit (keep most recent)
    max_size = _CONFIG["MAX_HISTORY_PER_ACCOUNT"]
    if len(_account_history[account_id]) > max_size:
        _account_history[account_id] = _account_history[account_id][-max_size:]


# =============================================================================
# ML FEATURE EXTRACTION
# =============================================================================

def _extract_ml_features(trade: TradeRequest, history: List[TradeRecord]) -> np.ndarray:
    """
    Extract feature vector for ML model from trade.
    
    Features:
    1. amount - Trade amount (will be scaled)
    2. hour - Hour of day (0-23)
    3. is_unusual_hour - Binary flag for unusual hours
    4. trades_last_10min - Count of recent trades (velocity)
    5. amount_vs_avg - Ratio to account average (behavior drift)
    
    Returns:
        numpy array of shape (5,)
    """
    # Parse timestamp
    trade_time = _parse_timestamp(trade.timestamp)
    hour = trade_time.hour
    
    # Unusual hour flag
    is_unusual = 1 if _CONFIG["UNUSUAL_HOUR_START"] <= hour < _CONFIG["UNUSUAL_HOUR_END"] else 0
    
    # Count trades in last 10 minutes
    trades_in_window = 0
    if history:
        window_minutes = _CONFIG["VELOCITY_WINDOW_MINUTES"]
        for record in history:
            time_diff = abs((trade_time - record.timestamp).total_seconds() / 60.0)
            if time_diff <= window_minutes:
                trades_in_window += 1
    
    # Amount vs average ratio
    if history and len(history) >= _CONFIG["MIN_HISTORY_FOR_DRIFT"]:
        avg_amount = sum(r.amount for r in history) / len(history)
        amount_vs_avg = trade.trade_amount / max(avg_amount, 1.0)
    else:
        # Default ratio for new accounts
        amount_vs_avg = 1.0
    
    return np.array([
        trade.trade_amount,
        hour,
        is_unusual,
        trades_in_window,
        amount_vs_avg
    ], dtype=np.float64)


def _generate_synthetic_training_data(n_samples: int = 100) -> np.ndarray:
    """
    Generate synthetic training data for initial model training.
    
    Creates a mix of normal trades and anomalies to train the model
    before real trades are available.
    
    Distribution: 90% normal, 7% moderate anomalies, 3% severe
    (More realistic fraud ratios)
    
    TODO for production:
    - Use historical data from database
    - Implement proper train/test split
    - Add cross-validation
    
    Returns:
        numpy array of shape (n_samples, 5)
    """
    np.random.seed(_CONFIG["ML_RANDOM_STATE"])
    
    data = []
    
    # 90% normal trades (increased from 85%)
    n_normal = int(n_samples * 0.90)
    for _ in range(n_normal):
        data.append([
            np.random.uniform(100, 10000),       # amount: $100-$10K (typical range)
            np.random.randint(8, 18),            # hour: 8 AM - 6 PM
            0,                                    # not unusual hour
            np.random.randint(0, 3),             # 0-2 trades in window (normal)
            np.random.uniform(0.7, 1.3),         # close to average (tight range)
        ])
    
    # 7% moderate anomalies (reduced from 10%)
    n_moderate = int(n_samples * 0.07)
    for _ in range(n_moderate):
        data.append([
            np.random.uniform(75000, 200000),    # amount: $75K-$200K
            np.random.randint(0, 24),            # any hour
            np.random.choice([0, 1]),            # maybe unusual
            np.random.randint(3, 5),             # moderate velocity
            np.random.uniform(2.5, 5),           # 2.5-5x average
        ])
    
    # 3% severe anomalies (reduced from 5%)
    n_severe = n_samples - n_normal - n_moderate
    for _ in range(n_severe):
        data.append([
            np.random.uniform(300000, 750000),   # amount: $300K-$750K
            np.random.randint(1, 5),             # unusual hours
            1,                                    # unusual hour
            np.random.randint(5, 10),            # high velocity
            np.random.uniform(6, 12),            # 6-12x average
        ])
    
    return np.array(data, dtype=np.float64)


# =============================================================================
# ML MODEL MANAGEMENT
# =============================================================================

def train_ml_model(force: bool = False) -> bool:
    """
    Train or retrain the Isolation Forest model.
    
    Training uses either:
    1. Accumulated feature buffer from real trades
    2. Synthetic data if insufficient real trades
    
    Args:
        force: If True, force retrain even if recently trained
        
    Returns:
        True if training succeeded, False otherwise
        
    TODO for production:
    - Add model validation metrics
    - Implement A/B testing
    - Add model performance logging
    """
    global _ml_state
    
    start_time = time.time()
    
    try:
        # Determine training data
        min_samples = _CONFIG["ML_MIN_SAMPLES_TO_TRAIN"]
        
        if len(_ml_state.feature_buffer) >= min_samples:
            # Use real trade data
            training_data = np.array(_ml_state.feature_buffer)
            logger.info(f"🧠 Training ML model on {len(training_data)} real trade samples")
        else:
            # Use synthetic data for initial training
            training_data = _generate_synthetic_training_data(100)
            logger.info(f"🧠 Training ML model on synthetic data (100 samples)")
        
        # Initialize scaler
        _ml_state.scaler = StandardScaler()
        scaled_data = _ml_state.scaler.fit_transform(training_data)
        
        # Initialize and train Isolation Forest
        _ml_state.model = IsolationForest(
            n_estimators=_CONFIG["ML_N_ESTIMATORS"],
            contamination=_CONFIG["ML_CONTAMINATION"],
            random_state=_CONFIG["ML_RANDOM_STATE"],
            n_jobs=1,  # Single thread for deterministic results
        )
        
        _ml_state.model.fit(scaled_data)
        _ml_state.is_trained = True
        _ml_state.training_samples = len(training_data)
        _ml_state.trades_since_retrain = 0
        
        elapsed = time.time() - start_time
        logger.info(f"✅ ML model trained in {elapsed:.3f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ML model training failed: {e}")
        _ml_state.is_trained = False
        return False


def _maybe_retrain_model() -> None:
    """
    Check if model should be retrained based on new data.
    
    Retrains every ML_RETRAIN_INTERVAL new trades to incorporate
    recent patterns.
    """
    if _ml_state.trades_since_retrain >= _CONFIG["ML_RETRAIN_INTERVAL"]:
        if len(_ml_state.feature_buffer) >= _CONFIG["ML_MIN_SAMPLES_TO_TRAIN"]:
            logger.info("🔄 Triggering incremental model retrain...")
            train_ml_model(force=True)


def get_ml_model_status() -> Dict:
    """
    Get current ML model status (for monitoring/debugging).
    
    Returns:
        Dict with model status information
    """
    return {
        "is_trained": _ml_state.is_trained,
        "training_samples": _ml_state.training_samples,
        "feature_buffer_size": len(_ml_state.feature_buffer),
        "trades_since_retrain": _ml_state.trades_since_retrain,
        "config": {
            "n_estimators": _CONFIG["ML_N_ESTIMATORS"],
            "contamination": _CONFIG["ML_CONTAMINATION"],
            "ml_signal_max_score": _CONFIG["ML_SIGNAL_MAX_SCORE"],
        }
    }


# =============================================================================
# FRAUD SIGNAL DETECTORS (HEURISTIC)
# =============================================================================

def _signal_amount_anomaly(trade: TradeRequest) -> SignalResult:
    """
    Signal 1: Amount Anomaly Detection (Heuristic)
    
    Additive scoring (direct points added to risk score):
    - Trades below $25K: 0 points (minimal risk - normal trading)
    - Trades $25K-$75K: 5 points (low risk - larger but common)
    - Trades $75K-$150K: 15 points (moderate risk)
    - Trades $150K-$500K: 30 points (elevated risk)
    - Trades above $500K: 45 points (high risk)
    """
    amount = trade.trade_amount
    
    if amount < _CONFIG["AMOUNT_LOW"]:
        return SignalResult(
            score=0,
            triggered=False,
            explanation=f"Trade amount ${amount:,.2f} is within normal range"
        )
        
    elif amount < _CONFIG["AMOUNT_MEDIUM"]:
        return SignalResult(
            score=5,
            triggered=False,
            explanation=f"Trade amount ${amount:,.2f} is slightly elevated"
        )
        
    elif amount < _CONFIG["AMOUNT_HIGH"]:
        return SignalResult(
            score=15,
            triggered=True,
            explanation=f"Trade amount ${amount:,.2f} exceeds typical transaction size"
        )
        
    elif amount < _CONFIG["AMOUNT_VERY_HIGH"]:
        return SignalResult(
            score=30,
            triggered=True,
            explanation=f"Large trade amount ${amount:,.2f} requires attention"
        )
        
    else:
        return SignalResult(
            score=45,
            triggered=True,
            explanation=f"Very large trade amount ${amount:,.2f} significantly exceeds normal patterns"
        )


def _signal_velocity_burst(trade: TradeRequest, history: List[TradeRecord]) -> SignalResult:
    """
    Signal 2: Velocity Burst Detection (Heuristic)
    
    Additive scoring - detects rapid-fire trading patterns:
    - 0-3 trades in window: 0 points (normal activity)
    - 4 trades: 25 points (elevated)
    - 5 trades: 40 points (high)
    - 6+ trades: 55 points (very high - likely automated)
    """
    if not history:
        return SignalResult(
            score=0,
            triggered=False,
            explanation="First trade from this account"
        )
    
    # Parse current trade timestamp
    current_time = _parse_timestamp(trade.timestamp)
    window_minutes = _CONFIG["VELOCITY_WINDOW_MINUTES"]
    
    # Count trades within the time window
    trades_in_window = 0
    for record in history:
        time_diff = abs((current_time - record.timestamp).total_seconds() / 60.0)
        if time_diff <= window_minutes:
            trades_in_window += 1
    
    # Determine score based on velocity
    threshold = _CONFIG["VELOCITY_THRESHOLD"]
    
    if trades_in_window < threshold:
        return SignalResult(
            score=0,
            triggered=False,
            explanation=f"Normal trading velocity ({trades_in_window} trades in {window_minutes}min window)"
        )
        
    elif trades_in_window == threshold:
        return SignalResult(
            score=25,
            triggered=True,
            explanation=f"Elevated trading velocity: {trades_in_window} trades in {window_minutes}min window"
        )
        
    elif trades_in_window == threshold + 1:
        return SignalResult(
            score=40,
            triggered=True,
            explanation=f"High trading velocity: {trades_in_window} trades in {window_minutes}min window indicates possible layering"
        )
        
    else:
        return SignalResult(
            score=55,
            triggered=True,
            explanation=f"Extreme trading velocity: {trades_in_window} trades in {window_minutes}min window - potential automated fraud"
        )


def _signal_time_anomaly(trade: TradeRequest) -> SignalResult:
    """
    Signal 3: Time-based Anomaly Detection (Heuristic)
    
    Additive scoring (reduced weight - timezone considerations):
    - Normal hours: 0 points
    - Unusual hours (1-5 AM UTC): 10 points (reduced from 20)
    
    Note: This is a weak signal alone since traders operate globally.
    It becomes meaningful when combined with other risk factors.
    """
    current_time = _parse_timestamp(trade.timestamp)
    hour = current_time.hour
    
    unusual_start = _CONFIG["UNUSUAL_HOUR_START"]
    unusual_end = _CONFIG["UNUSUAL_HOUR_END"]
    
    if unusual_start <= hour < unusual_end:
        return SignalResult(
            score=10,
            triggered=True,
            explanation=f"Trade executed at unusual hour ({hour}:00 UTC) - outside typical trading window"
        )
    
    return SignalResult(
        score=0,
        triggered=False,
        explanation="Trade executed during normal business hours"
    )


def _signal_behavior_drift(trade: TradeRequest, history: List[TradeRecord]) -> SignalResult:
    """
    Signal 4: Behavior Drift Detection (Heuristic)
    
    Additive scoring - compares to account's historical average:
    - Within 2x average: 0 points
    - 2x-4x average: 10 points
    - 4x-6x average: 25 points
    - >6x average: 40 points
    
    New accounts (insufficient history) get 0 points - we don't penalize
    accounts just for being new. Other signals handle new account risk.
    """
    min_history = _CONFIG["MIN_HISTORY_FOR_DRIFT"]
    
    if len(history) < min_history:
        return SignalResult(
            score=0,
            triggered=False,
            explanation=f"Insufficient history for drift analysis ({len(history)} trades)"
        )
    
    # Calculate historical average amount
    total_amount = sum(record.amount for record in history)
    avg_amount = total_amount / len(history)
    
    # Avoid division by zero
    if avg_amount == 0:
        avg_amount = 1.0
    
    # Calculate drift ratio
    drift_ratio = trade.trade_amount / avg_amount
    drift_threshold = _CONFIG["DRIFT_MULTIPLIER"]
    
    if drift_ratio <= 2.0:
        return SignalResult(
            score=0,
            triggered=False,
            explanation=f"Trade amount consistent with account history (avg: ${avg_amount:,.2f})"
        )
        
    elif drift_ratio <= drift_threshold:
        return SignalResult(
            score=10,
            triggered=True,
            explanation=f"Trade amount {drift_ratio:.1f}x higher than account average (${avg_amount:,.2f})"
        )
        
    elif drift_ratio <= 6.0:
        return SignalResult(
            score=25,
            triggered=True,
            explanation=f"Significant deviation: trade is {drift_ratio:.1f}x the account average (${avg_amount:,.2f})"
        )
        
    else:
        return SignalResult(
            score=40,
            triggered=True,
            explanation=f"Extreme behavior drift: trade is {drift_ratio:.1f}x the account average (${avg_amount:,.2f}) - potential account takeover"
        )


def _signal_structuring(trade: TradeRequest, history: List[TradeRecord]) -> SignalResult:
    """
    Signal 5: Money Laundering Structuring Detection
    
    Detects transactions structured to avoid $10,000 reporting threshold.
    
    Pattern indicators:
    - Amount between $9,000-$9,999
    - Multiple similar transactions from same account
    - Transfer type transactions
    
    Scoring (only triggers on PATTERNS, not single transactions):
    - Single transaction $9,000-$9,999: 0 points (could be legitimate)
    - 2 structuring transactions: 20 points
    - 3 structuring transactions: 35 points
    - 4+ structuring transactions: 50 points (strong AML indicator)
    """
    STRUCTURING_MIN = 9000
    STRUCTURING_MAX = 9999
    
    amount = trade.trade_amount
    
    # Check if amount is in structuring range
    if not (STRUCTURING_MIN <= amount <= STRUCTURING_MAX):
        return SignalResult(
            score=0,
            triggered=False,
            explanation="Transaction amount not in structuring range"
        )
    
    # Count recent structuring-like transactions
    structuring_count = 1  # Current transaction
    if history:
        for record in history[-20:]:  # Check last 20 transactions
            if STRUCTURING_MIN <= record.amount <= STRUCTURING_MAX:
                structuring_count += 1
    
    # Single transaction in range - not suspicious by itself
    if structuring_count == 1:
        return SignalResult(
            score=0,
            triggered=False,
            explanation=f"Transaction amount ${amount:,.2f} noted (single occurrence)"
        )
    elif structuring_count == 2:
        return SignalResult(
            score=20,
            triggered=True,
            explanation=f"Two transactions ({structuring_count}) just under $10K threshold - monitoring for structuring"
        )
    elif structuring_count == 3:
        return SignalResult(
            score=35,
            triggered=True,
            explanation=f"Multiple transactions ({structuring_count}) just under $10K threshold - potential structuring/smurfing"
        )
    else:
        return SignalResult(
            score=50,
            triggered=True,
            explanation=f"Pattern of {structuring_count} transactions under $10K threshold - strong structuring indicator (AML alert)"
        )


def _signal_new_account_risk(trade: TradeRequest, history: List[TradeRecord]) -> SignalResult:
    """
    Signal 6: New Account Fraud Risk Detection
    
    Detects risky activity from newly created accounts.
    
    Pattern indicators:
    - Account has < 5 historical trades
    - Large transaction amount for first trades
    - Transfer type on new account
    
    Scoring (reduced - don't over-penalize legitimate new users):
    - New account + small trade (<$10K): 0 points (normal onboarding)
    - New account + medium trade ($10K-$50K): 10 points
    - New account + large trade ($50K-$100K): 20 points
    - New account + very large trade (>$100K): 35 points
    """
    is_new_account = len(history) < 5
    
    if not is_new_account:
        return SignalResult(
            score=0,
            triggered=False,
            explanation="Established account with trading history"
        )
    
    amount = trade.trade_amount
    
    # Small trade on new account - completely normal
    if amount < 10000:
        return SignalResult(
            score=0,
            triggered=False,
            explanation=f"New account ({len(history)} trades), normal transaction size"
        )
    
    # Medium trade on new account - slight caution
    if amount < 50000:
        return SignalResult(
            score=10,
            triggered=True,
            explanation=f"New account ({len(history)} trades) with elevated transaction ${amount:,.0f}"
        )
    
    # Large trade on new account - elevated risk
    if amount < 100000:
        return SignalResult(
            score=20,
            triggered=True,
            explanation=f"New account ({len(history)} trades) with large transaction ${amount:,.0f}"
        )
    
    # Very large trade on new account - high risk
    return SignalResult(
        score=35,
        triggered=True,
        explanation=f"New account ({len(history)} trades) with very large transaction ${amount:,.0f} - potential new account fraud"
    )


# =============================================================================
# ML SIGNAL DETECTOR
# =============================================================================

def _signal_ml_anomaly(trade: TradeRequest, history: List[TradeRecord]) -> SignalResult:
    """
    Signal 5: Isolation Forest ML Anomaly Detection
    
    Uses trained Isolation Forest model to detect anomalies based on
    multi-dimensional feature space.
    
    Scoring:
    - Anomaly score from model is converted to 0-25 points
    - Normal samples: 0-5 points
    - Suspicious: 5-15 points
    - Anomalous: 15-25 points
    
    The ML signal acts as a "second opinion" to heuristic signals,
    catching patterns that rule-based systems might miss.
    
    TODO for production:
    - Add confidence intervals
    - Implement ensemble with other ML models
    - Add feature importance explanations
    """
    global _ml_state
    
    # If model not trained, return neutral score
    if not _ml_state.is_trained or _ml_state.model is None:
        return SignalResult(
            score=0,
            triggered=False,
            explanation="ML model not yet trained"
        )
    
    try:
        # Extract features
        features = _extract_ml_features(trade, history)
        
        # Add to feature buffer for future training
        _ml_state.feature_buffer.append(features)
        # Keep buffer bounded
        if len(_ml_state.feature_buffer) > 1000:
            _ml_state.feature_buffer = _ml_state.feature_buffer[-500:]
        
        # Scale features
        features_scaled = _ml_state.scaler.transform(features.reshape(1, -1))
        
        # Get anomaly score
        # score_samples returns negative values, more negative = more anomalous
        # Typical range: -0.5 to 0.5, with negative being anomalous
        raw_score = _ml_state.model.score_samples(features_scaled)[0]
        
        # Convert to 0-1 scale (0 = normal, 1 = anomaly)
        # Map typical range [-0.5, 0.2] to [0, 1]
        normalized_score = 1.0 - (raw_score + 0.5) / 0.7
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        # Convert to point score (0-25)
        max_ml_score = _CONFIG["ML_SIGNAL_MAX_SCORE"]
        point_score = int(normalized_score * max_ml_score)
        
        # Increment counter for potential retrain
        _ml_state.trades_since_retrain += 1
        
        # Determine explanation based on score
        if normalized_score < 0.3:
            return SignalResult(
                score=point_score,
                triggered=False,
                explanation=f"ML model indicates normal pattern (confidence: {(1-normalized_score)*100:.0f}%)"
            )
        elif normalized_score < 0.6:
            return SignalResult(
                score=point_score,
                triggered=True,
                explanation=f"ML model detects unusual pattern (anomaly score: {normalized_score:.2f})"
            )
        else:
            return SignalResult(
                score=point_score,
                triggered=True,
                explanation=f"ML model flags significant anomaly (anomaly score: {normalized_score:.2f})"
            )
            
    except Exception as e:
        logger.error(f"ML signal error: {e}")
        return SignalResult(
            score=0,
            triggered=False,
            explanation="ML model unavailable"
        )


# =============================================================================
# RISK SCORE AGGREGATION
# =============================================================================

def _aggregate_signals(signals: List[SignalResult], ml_signal: SignalResult) -> Tuple[int, float, str]:
    """
    Aggregate heuristic and ML signals into a single risk score.
    
    Method: Simple additive scoring
    - Each heuristic signal contributes its score directly
    - ML signal adds 0-25 points as a "second opinion"
    - Final score is clamped to 0-100
    
    This approach makes scoring predictable and explainable:
    - Small trade alone: ~0-10 points → LOW
    - Large trade ($100K+): 40 points → MEDIUM  
    - Large trade + velocity burst: 40 + 50 = 90 → HIGH
    - ML can push borderline cases over thresholds
    
    Returns:
        Tuple of (risk_score, anomaly_score, combined_explanation)
    """
    # Sum all heuristic signal scores
    heuristic_score = sum(signal.score for signal in signals)
    
    # Add ML signal score
    total_score = heuristic_score + ml_signal.score
    
    # Clamp to valid range (0-100)
    risk_score = int(max(0, min(100, total_score)))
    
    # Normalize to 0-1 for anomaly score
    anomaly_score = round(risk_score / 100.0, 2)
    
    # Build explanation from triggered signals (most severe first)
    all_signals = signals + [ml_signal]
    triggered_signals = [s for s in all_signals if s.triggered]
    
    if not triggered_signals:
        explanation = "No significant anomalies detected. Trade patterns are consistent with account history."
    elif len(triggered_signals) == 1:
        explanation = triggered_signals[0].explanation
    else:
        # Sort by score (highest first) and combine explanations
        triggered_signals.sort(key=lambda s: s.score, reverse=True)
        primary = triggered_signals[0].explanation
        
        # Check if ML contributed
        ml_contributed = ml_signal.triggered
        secondary_count = len(triggered_signals) - 1
        
        if ml_contributed and ml_signal.score >= 10:
            explanation = f"{primary}. ML model corroborates with {secondary_count} additional signal(s)."
        else:
            explanation = f"{primary}. Additionally, {secondary_count} other risk signal(s) detected."
    
    return risk_score, anomaly_score, explanation


# =============================================================================
# PUBLIC SERVICE FUNCTIONS
# =============================================================================

def analyze_trade(trade: TradeRequest) -> TradeResponse:
    """
    Analyze a trade for fraud indicators using multi-signal detection + ML.
    
    Process:
    1. Retrieve account history
    2. Evaluate each heuristic fraud signal independently
    3. Evaluate ML anomaly signal
    4. Aggregate all signals into final risk score
    5. Update account history for future analysis
    6. Trigger model retrain if needed
    7. Return explainable risk assessment
    
    Args:
        trade: Validated trade request
        
    Returns:
        TradeResponse with risk assessment and explanation
    """
    # Step 1: Get account history for behavior-based signals
    history = _get_account_history(trade.account_id)
    
    # Step 2: Evaluate all heuristic fraud signals
    heuristic_signals = [
        _signal_amount_anomaly(trade),           # Signal 1: Amount-based risk
        _signal_velocity_burst(trade, history),  # Signal 2: Trading velocity (layering/spoofing)
        _signal_time_anomaly(trade),             # Signal 3: Unusual hours
        _signal_behavior_drift(trade, history),  # Signal 4: Behavior deviation (account takeover)
        _signal_structuring(trade, history),     # Signal 5: Money laundering structuring
        _signal_new_account_risk(trade, history), # Signal 6: New account fraud
    ]
    
    # Step 3: Evaluate ML signal
    ml_signal = _signal_ml_anomaly(trade, history)
    
    # Step 4: Aggregate signals into final score
    risk_score, anomaly_score, explanation = _aggregate_signals(heuristic_signals, ml_signal)
    risk_level = _calculate_risk_level(risk_score)
    
    # Step 5: Build response with full trade details
    response = TradeResponse(
        trade_id=trade.trade_id,
        account_id=trade.account_id,
        trade_amount=trade.trade_amount,
        trade_type=trade.trade_type,
        risk_score=risk_score,
        risk_level=risk_level,
        anomaly_score=anomaly_score,
        explanation=explanation,
        timestamp=trade.timestamp
    )
    
    # Step 6: Update history for future analysis
    trade_record = TradeRecord(
        trade_id=trade.trade_id,
        amount=trade.trade_amount,
        timestamp=_parse_timestamp(trade.timestamp),
        trade_type=trade.trade_type.value,
        account_id=trade.account_id
    )
    _add_to_history(trade.account_id, trade_record)
    
    # Step 7: Store for API retrieval
    _analyzed_trades.append(response)
    
    # Step 8: Check if model should be retrained
    _maybe_retrain_model()
    
    return response


def get_all_trades() -> List[TradeResponse]:
    """
    Retrieve all analyzed trades.
    
    Returns:
        List of all TradeResponse objects
        
    TODO:
    - Add pagination
    - Add filtering by date range
    - Add sorting options
    - Connect to database
    """
    # Return mock data if no trades analyzed yet
    if not _analyzed_trades:
        return _get_mock_trades()
    
    return _analyzed_trades.copy()


def get_alerts() -> List[AlertResponse]:
    """
    Retrieve alerts for suspicious trades (HIGH and MEDIUM risk).
    
    Returns:
        List of AlertResponse for trades with HIGH or MEDIUM risk level.
        HIGH risk alerts appear first, then MEDIUM, sorted by timestamp.
        
    TODO:
    - Add alert acknowledgment tracking
    - Add alert escalation logic
    - Connect to notification service
    """
    trades = get_all_trades()
    
    # Include both HIGH and MEDIUM risk as alerts
    suspicious_trades = [
        trade for trade in trades
        if trade.risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM)
    ]
    
    # Sort by risk level (HIGH first) then by timestamp (newest first)
    suspicious_trades.sort(
        key=lambda t: (0 if t.risk_level == RiskLevel.HIGH else 1, t.timestamp),
        reverse=True
    )
    
    alerts = [
        AlertResponse(
            trade_id=trade.trade_id,
            risk_score=trade.risk_score,
            explanation=trade.explanation,
            timestamp=trade.timestamp
        )
        for trade in suspicious_trades
    ]
    
    return alerts


def clear_trades() -> None:
    """
    Clear all stored trades and account history (for testing).
    
    This resets the service to initial state.
    """
    global _analyzed_trades, _account_history, _ml_state
    _analyzed_trades = []
    _account_history = {}
    _ml_state.reset()


def get_account_stats(account_id: str) -> Dict:
    """
    Get statistics for an account (for debugging/demo).
    
    Returns:
        Dict with account statistics
    """
    history = _get_account_history(account_id)
    
    if not history:
        return {
            "account_id": account_id,
            "trade_count": 0,
            "total_volume": 0,
            "average_amount": 0,
            "message": "No trade history for this account"
        }
    
    total = sum(r.amount for r in history)
    return {
        "account_id": account_id,
        "trade_count": len(history),
        "total_volume": total,
        "average_amount": total / len(history),
        "oldest_trade": history[0].timestamp.isoformat() if history else None,
        "newest_trade": history[-1].timestamp.isoformat() if history else None,
    }


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_ml_model() -> bool:
    """
    Initialize the ML model at startup.
    
    Called by main.py during application startup.
    Trains the model on synthetic data to be ready for first trade.
    
    Returns:
        True if initialization succeeded
    """
    logger.info("🚀 Initializing ML anomaly detection model...")
    return train_ml_model(force=True)


# =============================================================================
# MOCK DATA (for initial API testing)
# =============================================================================

def _get_mock_trades() -> List[TradeResponse]:
    """
    Return mock trades matching frontend contract.
    
    TODO: Remove when database is connected
    """
    from schemas import TradeType
    
    return [
        TradeResponse(
            trade_id="TRD-001",
            account_id="ACC-7821",
            trade_amount=125000.00,
            trade_type=TradeType.BUY,
            risk_score=85,
            risk_level=RiskLevel.HIGH,
            anomaly_score=0.85,
            explanation="Rapid succession of trades from new account. ML model corroborates with 2 additional signal(s).",
            timestamp="2026-02-04T14:32:15.000Z"
        ),
        TradeResponse(
            trade_id="TRD-002",
            account_id="ACC-4523",
            trade_amount=2500.00,
            trade_type=TradeType.SELL,
            risk_score=23,
            risk_level=RiskLevel.LOW,
            anomaly_score=0.23,
            explanation="No significant anomalies detected. Trade patterns are consistent with account history.",
            timestamp="2026-02-04T14:28:42.000Z"
        ),
        TradeResponse(
            trade_id="TRD-003",
            account_id="ACC-9012",
            trade_amount=450000.00,
            trade_type=TradeType.TRANSFER,
            risk_score=92,
            risk_level=RiskLevel.HIGH,
            anomaly_score=0.92,
            explanation="Unusual transfer volume exceeds 5x daily average. ML model corroborates with 1 additional signal(s).",
            timestamp="2026-02-04T14:25:10.000Z"
        ),
        TradeResponse(
            trade_id="TRD-004",
            account_id="ACC-3344",
            trade_amount=1200.00,
            trade_type=TradeType.BUY,
            risk_score=15,
            risk_level=RiskLevel.LOW,
            anomaly_score=0.15,
            explanation="No significant anomalies detected. Trade patterns are consistent with account history.",
            timestamp="2026-02-04T14:22:33.000Z"
        ),
        TradeResponse(
            trade_id="TRD-005",
            account_id="ACC-5566",
            trade_amount=35000.00,
            trade_type=TradeType.SELL,
            risk_score=45,
            risk_level=RiskLevel.MEDIUM,
            anomaly_score=0.45,
            explanation="Trade pattern deviates from account history",
            timestamp="2026-02-04T14:18:55.000Z"
        ),
        TradeResponse(
            trade_id="TRD-006",
            account_id="ACC-2211",
            trade_amount=89000.00,
            trade_type=TradeType.BUY,
            risk_score=78,
            risk_level=RiskLevel.HIGH,
            anomaly_score=0.78,
            explanation="Transaction pattern indicates potential layering activity. ML model corroborates with 2 additional signal(s).",
            timestamp="2026-02-04T14:15:20.000Z"
        ),
        TradeResponse(
            trade_id="TRD-007",
            account_id="ACC-8899",
            trade_amount=800.00,
            trade_type=TradeType.SELL,
            risk_score=12,
            risk_level=RiskLevel.LOW,
            anomaly_score=0.12,
            explanation="No significant anomalies detected. Trade patterns are consistent with account history.",
            timestamp="2026-02-04T14:12:08.000Z"
        ),
        TradeResponse(
            trade_id="TRD-008",
            account_id="ACC-1122",
            trade_amount=275000.00,
            trade_type=TradeType.TRANSFER,
            risk_score=88,
            risk_level=RiskLevel.HIGH,
            anomaly_score=0.88,
            explanation="Cross-border transfer to flagged jurisdiction. ML model corroborates with 1 additional signal(s).",
            timestamp="2026-02-04T14:08:45.000Z"
        ),
    ]
