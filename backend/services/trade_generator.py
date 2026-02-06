"""
Trade Generator Service

Simulates a live trading feed for demo purposes.
Generates both normal trades and anomalous trades to test the fraud detection system.

Features:
- Generates realistic normal trades
- Injects anomalies at configurable intervals:
  * Large amount anomalies
  * Velocity bursts (rapid trades from same account)
  * Unusual hour trades
  * Behavior drift (sudden change in trading pattern)
- Integrates with anomaly_service for real-time evaluation
- Keeps trades in memory for API endpoints

This module is designed for MVP demos and testing.
NOT for production use with real trading data.
"""

import asyncio
import logging
import random
import string
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Tuple
from collections import deque

from schemas import TradeRequest, TradeResponse, TradeType
from services.anomaly_service import analyze_trade, get_all_trades, get_alerts, clear_trades


# =============================================================================
# CONFIGURATION
# =============================================================================

class GeneratorConfig:
    """
    Trade generator configuration.
    Adjust these values to control trade generation behavior.
    """
    
    # Generation speed
    TRADE_INTERVAL_SECONDS: float = 1.0      # Time between trades
    
    # Memory limits
    MAX_TRADES_IN_MEMORY: int = 100          # Keep last N trades
    
    # Anomaly injection frequency - MUCH LESS FREQUENT
    # Real fraud is rare (0.1-1% of transactions)
    ANOMALY_EVERY_N_TRADES: int = 50         # Inject anomaly every N trades
    
    # ADDITIONAL: Random chance to skip anomaly (makes it more realistic)
    ANOMALY_SKIP_PROBABILITY: float = 0.3   # 30% chance to skip scheduled anomaly
    
    # Normal trade parameters
    NORMAL_AMOUNT_MIN: float = 50.0          # Minimum normal trade amount
    NORMAL_AMOUNT_MAX: float = 5000.0        # Maximum normal trade amount
    
    # Account behavior parameters
    ACCOUNT_TYPICAL_AMOUNTS: Dict[str, tuple] = {}  # Learned per account
    
    # Anomaly parameters - REAL FRAUD PATTERNS
    
    # 1. Layering/Spoofing - fake orders to manipulate price
    LAYERING_ORDER_COUNT: int = 10           # Number of fake orders
    LAYERING_CANCEL_RATIO: float = 0.9       # 90% get cancelled
    
    # 2. Account Takeover - sudden behavior change
    TAKEOVER_AMOUNT_MULTIPLIER: float = 10.0 # 10x typical amount
    
    # 3. Money Laundering Structuring - just under $10K threshold
    STRUCTURING_AMOUNTS: tuple = (9000, 9500, 9800, 9900, 9950)
    STRUCTURING_COUNT: int = 4               # Multiple structured transactions
    
    # 4. Velocity Attack / Rapid Trading
    BURST_TRADE_COUNT: int = 8               # Number of trades in a burst
    BURST_INTERVAL_SECONDS: float = 0.3      # Very rapid
    
    # 5. New Account Fraud - large amount on new account
    NEW_ACCOUNT_LARGE_AMOUNT: tuple = (50000, 200000)
    
    # Account pools
    NORMAL_ACCOUNT_POOL_SIZE: int = 50       # More accounts for realistic diversity
    
    # Unusual hours (UTC) - more realistic window
    UNUSUAL_HOUR_START: int = 2              # 2 AM UTC
    UNUSUAL_HOUR_END: int = 5                # 5 AM UTC


# Global config instance
config = GeneratorConfig()


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# GENERATOR STATE
# =============================================================================

class GeneratorState:
    """Maintains state for the trade generator with realistic account behavior."""
    
    def __init__(self):
        self.trade_counter: int = 0
        self.is_running: bool = False
        self.account_pool: List[str] = []
        self.anomaly_cycle: int = 0  # Tracks which anomaly type to inject next
        self.error_count: int = 0  # Track errors for monitoring
        
        # Track account behavior for realistic fraud patterns
        self.account_history: Dict[str, List[Tuple[float, datetime]]] = {}  # amount, time
        self.account_creation_time: Dict[str, datetime] = {}  # When account was "created"
        self.account_typical_amount: Dict[str, float] = {}  # Running average
        
        # New accounts created for fraud
        self.new_fraud_accounts: List[str] = []
        
        # Initialize account pool
        self._init_account_pool()
    
    def _init_account_pool(self) -> None:
        """Create a pool of realistic account IDs with varied creation dates."""
        now = datetime.now(timezone.utc)
        
        self.account_pool = []
        for i in range(config.NORMAL_ACCOUNT_POOL_SIZE):
            account_id = f"ACC-{random.randint(1000, 9999)}"
            self.account_pool.append(account_id)
            
            # Assign varied account ages (some old, some recent)
            age_days = random.choice([
                random.randint(365, 1000),   # Old accounts (1-3 years)
                random.randint(30, 365),     # Established accounts (1-12 months)
                random.randint(7, 30),       # Recent accounts (1-4 weeks)
            ])
            self.account_creation_time[account_id] = now - timedelta(days=age_days)
            
            # Set typical trading amounts for established accounts
            self.account_typical_amount[account_id] = random.uniform(100, 2000)
    
    def get_next_trade_id(self) -> str:
        """Generate sequential trade ID."""
        self.trade_counter += 1
        return f"TRD-{self.trade_counter:05d}"
    
    def get_random_account(self) -> str:
        """Get a random account from the pool."""
        return random.choice(self.account_pool)
    
    def get_established_account(self) -> str:
        """Get an account with trading history."""
        candidates = [acc for acc in self.account_pool if acc in self.account_history and len(self.account_history[acc]) >= 5]
        if candidates:
            return random.choice(candidates)
        return self.account_pool[0]
    
    def create_new_account(self) -> str:
        """Create a brand new account (for new account fraud patterns)."""
        account_id = f"ACC-NEW-{random.randint(9000, 9999)}"
        self.account_creation_time[account_id] = datetime.now(timezone.utc)
        self.new_fraud_accounts.append(account_id)
        return account_id
    
    def record_trade(self, account_id: str, amount: float, timestamp: datetime) -> None:
        """Record a trade for account behavior tracking."""
        if account_id not in self.account_history:
            self.account_history[account_id] = []
        self.account_history[account_id].append((amount, timestamp))
        
        # Keep only last 50 trades per account
        if len(self.account_history[account_id]) > 50:
            self.account_history[account_id] = self.account_history[account_id][-50:]
        
        # Update typical amount (running average)
        amounts = [t[0] for t in self.account_history[account_id]]
        self.account_typical_amount[account_id] = sum(amounts) / len(amounts)
    
    def get_account_age_hours(self, account_id: str) -> float:
        """Get account age in hours."""
        if account_id not in self.account_creation_time:
            return 1000  # Default to old account
        age = datetime.now(timezone.utc) - self.account_creation_time[account_id]
        return age.total_seconds() / 3600
    
    def should_inject_anomaly(self) -> bool:
        """Check if it's time to inject an anomaly (with randomness)."""
        if self.trade_counter % config.ANOMALY_EVERY_N_TRADES != 0:
            return False
        # Additional random skip for realism
        if random.random() < config.ANOMALY_SKIP_PROBABILITY:
            return False
        return True
    
    def get_next_anomaly_type(self) -> str:
        """
        Cycle through REAL fraud patterns.
        
        Types:
        0 - Layering/Spoofing (fake orders)
        1 - Account Takeover (sudden behavior change)
        2 - Money Laundering Structuring (just under $10K)
        3 - Velocity Attack (rapid trading)
        4 - New Account Fraud (large amount on new account)
        5 - Off-hours suspicious activity
        """
        patterns = [
            "layering_spoofing",
            "account_takeover", 
            "structuring",
            "velocity_attack",
            "new_account_fraud",
            "off_hours_fraud"
        ]
        anomaly_type = self.anomaly_cycle % len(patterns)
        self.anomaly_cycle += 1
        return patterns[anomaly_type]


# Global state instance
_state = GeneratorState()


# =============================================================================
# TRADE GENERATORS
# =============================================================================

def _generate_normal_trade() -> TradeRequest:
    """
    Generate a normal, low-risk trade with realistic account behavior.
    
    Characteristics:
    - Amount close to account's typical trading pattern
    - Random account from pool
    - Normal business hours
    - BUY or SELL randomly
    """
    trade_id = _state.get_next_trade_id()
    account_id = _state.get_random_account()
    
    # Use account's typical amount with some variance
    typical = _state.account_typical_amount.get(account_id, 500)
    variance = random.uniform(0.5, 1.5)  # 50% to 150% of typical
    amount = round(typical * variance, 2)
    amount = max(config.NORMAL_AMOUNT_MIN, min(amount, config.NORMAL_AMOUNT_MAX))
    
    # Random trade type
    trade_type = random.choice([TradeType.BUY, TradeType.SELL])
    
    # Current timestamp (normal hours)
    timestamp = datetime.now(timezone.utc)
    
    # Record this trade for behavior tracking
    _state.record_trade(account_id, amount, timestamp)
    
    return TradeRequest(
        trade_id=trade_id,
        account_id=account_id,
        trade_amount=amount,
        trade_type=trade_type,
        timestamp=timestamp.isoformat()
    )


# =============================================================================
# REAL FRAUD PATTERN GENERATORS
# =============================================================================

def _generate_layering_spoofing() -> List[TradeRequest]:
    """
    Generate Layering/Spoofing pattern - fake orders to manipulate price.
    
    Real pattern characteristics:
    - Many orders placed rapidly at different price levels
    - Most orders cancelled before execution (we simulate as rejected)
    - Used to create false impression of demand
    - Often from single account or related accounts
    """
    trades = []
    account_id = _state.get_established_account()
    base_time = datetime.now(timezone.utc)
    
    logger.info(f"🎭 Injecting LAYERING/SPOOFING pattern: {config.LAYERING_ORDER_COUNT} rapid orders")
    
    # Generate rapid succession of orders
    for i in range(config.LAYERING_ORDER_COUNT):
        trade_id = _state.get_next_trade_id()
        
        # Varying amounts to create fake order book depth
        amount = round(random.uniform(5000, 25000), 2)
        
        # Very rapid timestamps (milliseconds apart in real systems)
        trade_time = base_time + timedelta(milliseconds=i * 100)
        
        # Alternate buy/sell to manipulate both sides
        trade_type = TradeType.BUY if i % 2 == 0 else TradeType.SELL
        
        trade = TradeRequest(
            trade_id=trade_id,
            account_id=account_id,
            trade_amount=amount,
            trade_type=trade_type,
            timestamp=trade_time.isoformat()
        )
        trades.append(trade)
    
    return trades


def _generate_account_takeover() -> TradeRequest:
    """
    Generate Account Takeover pattern - sudden dramatic behavior change.
    
    Real pattern characteristics:
    - Established account with consistent history
    - Suddenly trades 10x+ their typical amount
    - Often different asset types than usual
    - May occur at unusual times
    """
    trade_id = _state.get_next_trade_id()
    account_id = _state.get_established_account()
    
    # Get typical amount and multiply significantly
    typical = _state.account_typical_amount.get(account_id, 500)
    amount = round(typical * config.TAKEOVER_AMOUNT_MULTIPLIER * random.uniform(1.0, 2.0), 2)
    
    # Often occurs at odd hours (attacker in different timezone)
    now = datetime.now(timezone.utc)
    unusual_hour = random.randint(2, 5)
    timestamp = now.replace(hour=unusual_hour, minute=random.randint(0, 59))
    
    logger.info(f"🚨 Injecting ACCOUNT TAKEOVER: {account_id} typical=${typical:.0f}, now=${amount:.0f} ({amount/typical:.0f}x)")
    
    return TradeRequest(
        trade_id=trade_id,
        account_id=account_id,
        trade_amount=amount,
        trade_type=random.choice([TradeType.BUY, TradeType.SELL, TradeType.TRANSFER]),
        timestamp=timestamp.isoformat()
    )


def _generate_structuring() -> List[TradeRequest]:
    """
    Generate Money Laundering Structuring pattern.
    
    Real pattern characteristics:
    - Multiple transactions just under $10,000 reporting threshold
    - Same account or related accounts
    - Spread across short time window
    - Often round-ish numbers
    """
    trades = []
    account_id = _state.get_random_account()
    base_time = datetime.now(timezone.utc)
    
    logger.info(f"� Injecting STRUCTURING pattern: {config.STRUCTURING_COUNT} transactions under $10K")
    
    for i in range(config.STRUCTURING_COUNT):
        trade_id = _state.get_next_trade_id()
        
        # Amounts just under $10K threshold
        amount = random.choice(config.STRUCTURING_AMOUNTS) + random.uniform(0, 50)
        
        # Spread across minutes/hours
        trade_time = base_time + timedelta(minutes=i * random.randint(5, 30))
        
        trade = TradeRequest(
            trade_id=trade_id,
            account_id=account_id,
            trade_amount=round(amount, 2),
            trade_type=TradeType.TRANSFER,  # Structuring often uses transfers
            timestamp=trade_time.isoformat()
        )
        trades.append(trade)
    
    return trades


async def _generate_velocity_attack() -> List[TradeRequest]:
    """
    Generate Velocity Attack pattern - rapid fire trading.
    
    Real pattern characteristics:
    - Extremely rapid succession of trades
    - Often algorithmic/bot behavior
    - Same direction (all buys or all sells)
    - Can indicate market manipulation or compromised account
    """
    trades = []
    account_id = f"ACC-VELOCITY-{random.randint(8000, 8999)}"
    base_time = datetime.now(timezone.utc)
    
    # Add this as a new account
    _state.account_creation_time[account_id] = base_time - timedelta(hours=random.randint(1, 24))
    
    logger.info(f"⚡ Injecting VELOCITY ATTACK: {config.BURST_TRADE_COUNT} rapid trades")
    
    # All same direction
    trade_type = random.choice([TradeType.BUY, TradeType.SELL])
    
    for i in range(config.BURST_TRADE_COUNT):
        trade_id = _state.get_next_trade_id()
        
        # Consistent amounts (algorithmic pattern)
        base_amount = random.uniform(8000, 15000)
        amount = round(base_amount + random.uniform(-100, 100), 2)
        
        # Very rapid timestamps
        trade_time = base_time + timedelta(seconds=i * 1)
        
        trade = TradeRequest(
            trade_id=trade_id,
            account_id=account_id,
            trade_amount=amount,
            trade_type=trade_type,
            timestamp=trade_time.isoformat()
        )
        trades.append(trade)
        
        # Small delay
        await asyncio.sleep(config.BURST_INTERVAL_SECONDS)
    
    return trades


def _generate_new_account_fraud() -> TradeRequest:
    """
    Generate New Account Fraud pattern.
    
    Real pattern characteristics:
    - Brand new account (hours old)
    - First trade is unusually large
    - Often uses stolen credentials/funds
    - No established behavior pattern
    """
    trade_id = _state.get_next_trade_id()
    account_id = _state.create_new_account()
    
    # Large amount for first trade
    amount_min, amount_max = config.NEW_ACCOUNT_LARGE_AMOUNT
    amount = round(random.uniform(amount_min, amount_max), 2)
    
    timestamp = datetime.now(timezone.utc)
    
    logger.info(f"� Injecting NEW ACCOUNT FRAUD: {account_id} first trade ${amount:,.0f}")
    
    return TradeRequest(
        trade_id=trade_id,
        account_id=account_id,
        trade_amount=amount,
        trade_type=random.choice([TradeType.BUY, TradeType.TRANSFER]),
        timestamp=timestamp.isoformat()
    )


def _generate_off_hours_fraud() -> TradeRequest:
    """
    Generate Off-Hours Suspicious Activity pattern.
    
    Real pattern characteristics:
    - Trading at unusual hours (2-5 AM)
    - Combined with other risk factors
    - May indicate international fraud or compromised account
    """
    trade_id = _state.get_next_trade_id()
    account_id = _state.get_random_account()
    
    # Moderate to large amount
    amount = round(random.uniform(10000, 50000), 2)
    
    # Off hours timestamp
    now = datetime.now(timezone.utc)
    unusual_hour = random.randint(config.UNUSUAL_HOUR_START, config.UNUSUAL_HOUR_END - 1)
    timestamp = now.replace(hour=unusual_hour, minute=random.randint(0, 59))
    
    logger.info(f"🌙 Injecting OFF-HOURS FRAUD: {unusual_hour}:00 UTC, ${amount:,.0f}")
    
    return TradeRequest(
        trade_id=trade_id,
        account_id=account_id,
        trade_amount=amount,
        trade_type=random.choice([TradeType.BUY, TradeType.SELL]),
        timestamp=timestamp.isoformat()
    )


# =============================================================================
# MAIN GENERATOR LOOP
# =============================================================================

async def _process_trade(trade: TradeRequest) -> TradeResponse:
    """
    Process a single trade through the anomaly detection system.
    
    Args:
        trade: The trade to analyze
        
    Returns:
        TradeResponse with risk assessment
    """
    # Analyze trade using the anomaly service
    result = analyze_trade(trade)
    
    # Log result
    level_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
    emoji = level_emoji.get(result.risk_level.value, "⚪")
    
    logger.info(
        f"{emoji} Trade {result.trade_id}: "
        f"${trade.trade_amount:,.2f} | "
        f"Score: {result.risk_score} | "
        f"Level: {result.risk_level.value}"
    )
    
    return result


async def generate_trades_continuously():
    """
    Main generator loop - runs continuously in background.
    
    Generates trades at configured intervals, injecting REAL fraud patterns
    periodically to test the fraud detection system.
    
    Fraud Patterns:
    - Layering/Spoofing
    - Account Takeover
    - Money Laundering Structuring
    - Velocity Attacks
    - New Account Fraud
    - Off-Hours Fraud
    """
    _state.is_running = True
    logger.info("🚀 Trade generator started")
    logger.info(f"   - Interval: {config.TRADE_INTERVAL_SECONDS}s")
    logger.info(f"   - Anomaly every: {config.ANOMALY_EVERY_N_TRADES} trades (with {config.ANOMALY_SKIP_PROBABILITY*100:.0f}% skip chance)")
    logger.info(f"   - Fraud patterns: layering, account_takeover, structuring, velocity, new_account, off_hours")
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    try:
        while _state.is_running:
            try:
                # Check if we should inject an anomaly
                if _state.should_inject_anomaly() and _state.trade_counter > 10:
                    anomaly_type = _state.get_next_anomaly_type()
                    
                    if anomaly_type == "layering_spoofing":
                        # Layering generates multiple rapid trades
                        trades = _generate_layering_spoofing()
                        for trade in trades:
                            await _process_trade(trade)
                            await asyncio.sleep(0.1)  # Very rapid
                        
                    elif anomaly_type == "account_takeover":
                        trade = _generate_account_takeover()
                        await _process_trade(trade)
                        
                    elif anomaly_type == "structuring":
                        # Structuring generates multiple trades under threshold
                        trades = _generate_structuring()
                        for trade in trades:
                            await _process_trade(trade)
                            await asyncio.sleep(0.5)
                        
                    elif anomaly_type == "velocity_attack":
                        # Velocity attack generates rapid burst
                        trades = await _generate_velocity_attack()
                        for trade in trades:
                            await _process_trade(trade)
                        
                    elif anomaly_type == "new_account_fraud":
                        trade = _generate_new_account_fraud()
                        await _process_trade(trade)
                        
                    elif anomaly_type == "off_hours_fraud":
                        trade = _generate_off_hours_fraud()
                        await _process_trade(trade)
                else:
                    # Generate normal trade
                    trade = _generate_normal_trade()
                    await _process_trade(trade)
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Wait before next trade
                await asyncio.sleep(config.TRADE_INTERVAL_SECONDS)
                
            except asyncio.CancelledError:
                raise  # Re-raise cancellation
            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"⚠️ Trade generator error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"❌ Too many consecutive errors, stopping generator")
                    _state.is_running = False
                    break
                
                # Wait a bit before retrying
                await asyncio.sleep(2)
            
    except asyncio.CancelledError:
        logger.info("🛑 Trade generator stopped")
        _state.is_running = False


def stop_generator():
    """Stop the trade generator."""
    _state.is_running = False
    logger.info("Trade generator stop requested")


def reset_generator():
    """Reset the generator state and clear all trades."""
    global _state
    _state = GeneratorState()
    clear_trades()
    logger.info("Trade generator reset")


def get_generator_status() -> Dict:
    """Get current generator status."""
    return {
        "is_running": _state.is_running,
        "trades_generated": _state.trade_counter,
        "anomaly_cycle": _state.anomaly_cycle,
        "error_count": _state.error_count,
        "config": {
            "interval_seconds": config.TRADE_INTERVAL_SECONDS,
            "anomaly_every_n_trades": config.ANOMALY_EVERY_N_TRADES,
            "max_trades_in_memory": config.MAX_TRADES_IN_MEMORY,
        }
    }


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def get_recent_trades(limit: int = 50) -> List[TradeResponse]:
    """
    Get recent trades for API endpoint.
    
    Args:
        limit: Maximum number of trades to return
        
    Returns:
        List of recent TradeResponse objects
    """
    all_trades = get_all_trades()
    # Return most recent trades (last N)
    return all_trades[-limit:] if len(all_trades) > limit else all_trades


def get_high_risk_alerts() -> List:
    """
    Get HIGH risk alerts for API endpoint.
    
    Returns:
        List of AlertResponse objects for HIGH risk trades only
    """
    return get_alerts()
