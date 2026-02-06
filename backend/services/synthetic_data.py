"""
Synthetic Fraud Data Generator

Generates labeled synthetic data for training fraud detection models.
Creates realistic fraud patterns based on known fraud typologies:

Fraud Types:
1. Large Amount Fraud - Unusually large transactions
2. Velocity Attack - Rapid-fire transactions (layering/spoofing)
3. Off-Hours Fraud - Transactions at unusual times
4. Account Takeover - Sudden behavior change
5. Structuring - Multiple transactions just under reporting thresholds
6. New Account Fraud - Fraud on newly created accounts

This module provides:
- Labeled training data (X, y) for supervised learning
- Balanced datasets with realistic class ratios
- Reproducible generation with seed control
"""

import logging
import random
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Optional
import numpy as np

from services.feature_store import FeatureStore, TransactionFeatures, get_feature_store

logger = logging.getLogger(__name__)


# =============================================================================
# FRAUD PATTERN DEFINITIONS
# =============================================================================

class FraudPatterns:
    """
    Defines synthetic fraud patterns for training data generation.
    Each pattern has specific characteristics that the model should learn.
    """
    
    # Pattern 1: Large Amount Fraud
    LARGE_AMOUNT = {
        "name": "large_amount",
        "description": "Unusually large transaction amounts",
        "amount_range": (100_000, 1_000_000),
        "probability": 0.15,  # 15% of fraud cases
    }
    
    # Pattern 2: Velocity Attack (Layering/Spoofing)
    VELOCITY_ATTACK = {
        "name": "velocity_attack",
        "description": "Rapid succession of trades indicating layering or spoofing",
        "trades_per_minute": (5, 15),
        "amount_range": (1_000, 50_000),
        "probability": 0.20,
    }
    
    # Pattern 3: Off-Hours Fraud
    OFF_HOURS = {
        "name": "off_hours",
        "description": "Trading at unusual hours (midnight to 5am)",
        "hours": list(range(0, 5)),
        "amount_range": (5_000, 100_000),
        "probability": 0.15,
    }
    
    # Pattern 4: Account Takeover
    ACCOUNT_TAKEOVER = {
        "name": "account_takeover",
        "description": "Sudden dramatic change in account behavior",
        "amount_multiplier": (5, 20),  # 5-20x normal amount
        "probability": 0.20,
    }
    
    # Pattern 5: Structuring (Smurfing)
    STRUCTURING = {
        "name": "structuring",
        "description": "Multiple transactions just under $10K reporting threshold",
        "amount_range": (9_000, 9_999),
        "num_transactions": (3, 8),
        "probability": 0.15,
    }
    
    # Pattern 6: New Account Fraud
    NEW_ACCOUNT = {
        "name": "new_account",
        "description": "Large transaction from newly created account",
        "account_age_hours": (0, 24),
        "amount_range": (10_000, 200_000),
        "probability": 0.15,
    }


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

class SyntheticDataGenerator:
    """
    Generates labeled synthetic data for fraud detection training.
    
    Creates both legitimate and fraudulent transactions with known labels.
    Ensures realistic patterns and class balance for effective training.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator with random seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Track generated accounts for realistic behavior
        self._account_profiles: Dict[str, dict] = {}
        
        logger.info(f"SyntheticDataGenerator initialized with seed={seed}")
    
    def generate_training_data(
        self,
        n_samples: int = 5000,
        fraud_ratio: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate labeled training data.
        
        Args:
            n_samples: Total number of samples to generate
            fraud_ratio: Proportion of fraud cases (default 10%)
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=legitimate, 1=fraud)
            fraud_types: List of fraud type names for each sample
        """
        n_fraud = int(n_samples * fraud_ratio)
        n_legitimate = n_samples - n_fraud
        
        logger.info(f"Generating {n_samples} samples: {n_legitimate} legitimate, {n_fraud} fraud")
        
        # Create temporary feature store for consistent feature computation
        feature_store = FeatureStore()
        
        X_list = []
        y_list = []
        fraud_types = []
        
        # Generate legitimate transactions
        for i in range(n_legitimate):
            features = self._generate_legitimate_transaction(feature_store, i)
            X_list.append(feature_store.features_to_array(features))
            y_list.append(0)
            fraud_types.append("legitimate")
        
        # Generate fraud transactions (distributed across patterns)
        fraud_patterns = [
            FraudPatterns.LARGE_AMOUNT,
            FraudPatterns.VELOCITY_ATTACK,
            FraudPatterns.OFF_HOURS,
            FraudPatterns.ACCOUNT_TAKEOVER,
            FraudPatterns.STRUCTURING,
            FraudPatterns.NEW_ACCOUNT,
        ]
        
        for i in range(n_fraud):
            # Select fraud pattern based on probabilities
            pattern = self._select_fraud_pattern(fraud_patterns)
            features = self._generate_fraud_transaction(feature_store, pattern, i)
            X_list.append(feature_store.features_to_array(features))
            y_list.append(1)
            fraud_types.append(pattern["name"])
        
        # Shuffle the data
        indices = list(range(len(X_list)))
        self.rng.shuffle(indices)
        
        X = np.array([X_list[i] for i in indices])
        y = np.array([y_list[i] for i in indices])
        fraud_types = [fraud_types[i] for i in indices]
        
        logger.info(f"Generated training data: X shape={X.shape}, fraud ratio={y.mean():.2%}")
        
        return X, y, fraud_types
    
    def _select_fraud_pattern(self, patterns: List[dict]) -> dict:
        """Select a fraud pattern based on probability weights."""
        probabilities = [p["probability"] for p in patterns]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        r = self.rng.random()
        cumulative = 0
        for pattern, prob in zip(patterns, probabilities):
            cumulative += prob
            if r <= cumulative:
                return pattern
        return patterns[-1]
    
    def _generate_legitimate_transaction(
        self,
        feature_store: FeatureStore,
        index: int
    ) -> TransactionFeatures:
        """Generate a legitimate (non-fraud) transaction."""
        # Use realistic account distribution
        account_id = f"ACC-{self.rng.randint(1000, 1100)}"
        
        # Normal business hours
        hour = self.rng.choice([9, 10, 11, 12, 13, 14, 15, 16, 17])
        
        # Normal amount distribution (log-normal)
        amount = self.np_rng.lognormal(mean=5.5, sigma=1.2)  # ~$250 median
        amount = min(max(amount, 10), 5000)  # Clip to reasonable range
        
        # Generate timestamp
        base_time = datetime.now(timezone.utc) - timedelta(days=self.rng.randint(0, 30))
        timestamp = base_time.replace(hour=hour, minute=self.rng.randint(0, 59))
        
        # Ensure account has some history for realistic features
        self._ensure_account_history(feature_store, account_id, timestamp, is_fraud=False)
        
        features = feature_store.compute_features(
            account_id=account_id,
            amount=round(amount, 2),
            timestamp=timestamp,
        )
        
        # Update feature store
        feature_store.update_account(account_id, amount, timestamp, was_flagged=False)
        
        return features
    
    def _generate_fraud_transaction(
        self,
        feature_store: FeatureStore,
        pattern: dict,
        index: int
    ) -> TransactionFeatures:
        """Generate a fraudulent transaction based on pattern."""
        pattern_name = pattern["name"]
        
        if pattern_name == "large_amount":
            return self._generate_large_amount_fraud(feature_store, pattern)
        elif pattern_name == "velocity_attack":
            return self._generate_velocity_fraud(feature_store, pattern)
        elif pattern_name == "off_hours":
            return self._generate_off_hours_fraud(feature_store, pattern)
        elif pattern_name == "account_takeover":
            return self._generate_account_takeover_fraud(feature_store, pattern)
        elif pattern_name == "structuring":
            return self._generate_structuring_fraud(feature_store, pattern)
        elif pattern_name == "new_account":
            return self._generate_new_account_fraud(feature_store, pattern)
        else:
            # Fallback to large amount
            return self._generate_large_amount_fraud(feature_store, pattern)
    
    def _generate_large_amount_fraud(
        self,
        feature_store: FeatureStore,
        pattern: dict
    ) -> TransactionFeatures:
        """Generate large amount fraud pattern."""
        account_id = f"ACC-FRAUD-{self.rng.randint(2000, 2500)}"
        amount_min, amount_max = pattern["amount_range"]
        amount = self.rng.uniform(amount_min, amount_max)
        
        timestamp = datetime.now(timezone.utc) - timedelta(hours=self.rng.randint(0, 72))
        
        self._ensure_account_history(feature_store, account_id, timestamp, is_fraud=True)
        
        features = feature_store.compute_features(
            account_id=account_id,
            amount=round(amount, 2),
            timestamp=timestamp,
        )
        
        feature_store.update_account(account_id, amount, timestamp, was_flagged=True)
        return features
    
    def _generate_velocity_fraud(
        self,
        feature_store: FeatureStore,
        pattern: dict
    ) -> TransactionFeatures:
        """Generate velocity attack fraud pattern."""
        account_id = f"ACC-VELOCITY-{self.rng.randint(3000, 3500)}"
        amount_min, amount_max = pattern["amount_range"]
        
        # Generate burst of transactions
        base_time = datetime.now(timezone.utc) - timedelta(hours=self.rng.randint(0, 48))
        
        # Add several rapid transactions to history
        for i in range(self.rng.randint(5, 10)):
            rapid_time = base_time + timedelta(seconds=i * self.rng.randint(2, 10))
            rapid_amount = self.rng.uniform(amount_min, amount_max)
            feature_store.update_account(account_id, rapid_amount, rapid_time, was_flagged=False)
        
        # The final transaction in the burst
        final_time = base_time + timedelta(seconds=self.rng.randint(30, 60))
        amount = self.rng.uniform(amount_min, amount_max)
        
        features = feature_store.compute_features(
            account_id=account_id,
            amount=round(amount, 2),
            timestamp=final_time,
        )
        
        feature_store.update_account(account_id, amount, final_time, was_flagged=True)
        return features
    
    def _generate_off_hours_fraud(
        self,
        feature_store: FeatureStore,
        pattern: dict
    ) -> TransactionFeatures:
        """Generate off-hours fraud pattern."""
        account_id = f"ACC-OFFHOURS-{self.rng.randint(4000, 4500)}"
        amount_min, amount_max = pattern["amount_range"]
        amount = self.rng.uniform(amount_min, amount_max)
        
        # Unusual hour
        hour = self.rng.choice(pattern["hours"])
        base_time = datetime.now(timezone.utc) - timedelta(days=self.rng.randint(0, 14))
        timestamp = base_time.replace(hour=hour, minute=self.rng.randint(0, 59))
        
        self._ensure_account_history(feature_store, account_id, timestamp, is_fraud=True)
        
        features = feature_store.compute_features(
            account_id=account_id,
            amount=round(amount, 2),
            timestamp=timestamp,
        )
        
        feature_store.update_account(account_id, amount, timestamp, was_flagged=True)
        return features
    
    def _generate_account_takeover_fraud(
        self,
        feature_store: FeatureStore,
        pattern: dict
    ) -> TransactionFeatures:
        """Generate account takeover fraud pattern."""
        # Use an existing "legitimate" account
        account_id = f"ACC-{self.rng.randint(1000, 1100)}"
        
        # First, build normal history
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        for i in range(20):  # 20 normal transactions
            normal_time = base_time + timedelta(days=i)
            normal_amount = self.np_rng.lognormal(mean=5.0, sigma=0.8)  # ~$150 median
            feature_store.update_account(account_id, normal_amount, normal_time, was_flagged=False)
        
        # Now the "takeover" - dramatic change
        mult_min, mult_max = pattern["amount_multiplier"]
        profile = feature_store.get_or_create_account(account_id)
        takeover_amount = profile.avg_trade_amount * self.rng.uniform(mult_min, mult_max)
        
        timestamp = datetime.now(timezone.utc) - timedelta(hours=self.rng.randint(0, 24))
        
        features = feature_store.compute_features(
            account_id=account_id,
            amount=round(takeover_amount, 2),
            timestamp=timestamp,
        )
        
        feature_store.update_account(account_id, takeover_amount, timestamp, was_flagged=True)
        return features
    
    def _generate_structuring_fraud(
        self,
        feature_store: FeatureStore,
        pattern: dict
    ) -> TransactionFeatures:
        """Generate structuring (smurfing) fraud pattern."""
        account_id = f"ACC-STRUCT-{self.rng.randint(5000, 5500)}"
        amount_min, amount_max = pattern["amount_range"]
        
        base_time = datetime.now(timezone.utc) - timedelta(hours=self.rng.randint(0, 48))
        
        # Multiple transactions just under $10K
        num_min, num_max = pattern["num_transactions"]
        for i in range(self.rng.randint(num_min, num_max) - 1):
            struct_time = base_time + timedelta(minutes=i * self.rng.randint(10, 60))
            struct_amount = self.rng.uniform(amount_min, amount_max)
            feature_store.update_account(account_id, struct_amount, struct_time, was_flagged=False)
        
        # Final structuring transaction
        final_time = base_time + timedelta(hours=self.rng.randint(2, 6))
        amount = self.rng.uniform(amount_min, amount_max)
        
        features = feature_store.compute_features(
            account_id=account_id,
            amount=round(amount, 2),
            timestamp=final_time,
        )
        
        feature_store.update_account(account_id, amount, final_time, was_flagged=True)
        return features
    
    def _generate_new_account_fraud(
        self,
        feature_store: FeatureStore,
        pattern: dict
    ) -> TransactionFeatures:
        """Generate new account fraud pattern."""
        # Brand new account
        account_id = f"ACC-NEW-{self.rng.randint(6000, 6500)}"
        amount_min, amount_max = pattern["amount_range"]
        amount = self.rng.uniform(amount_min, amount_max)
        
        # Very recent account creation
        hours_min, hours_max = pattern["account_age_hours"]
        account_age_hours = self.rng.uniform(hours_min, hours_max)
        
        # Create account profile with recent creation time
        timestamp = datetime.now(timezone.utc)
        
        # Don't add history - this is a new account
        features = feature_store.compute_features(
            account_id=account_id,
            amount=round(amount, 2),
            timestamp=timestamp,
        )
        
        feature_store.update_account(account_id, amount, timestamp, was_flagged=True)
        return features
    
    def _ensure_account_history(
        self,
        feature_store: FeatureStore,
        account_id: str,
        current_time: datetime,
        is_fraud: bool
    ) -> None:
        """Ensure account has some history for realistic features."""
        profile = feature_store.get_or_create_account(account_id)
        
        if profile.total_trades == 0 and not is_fraud:
            # Add some historical transactions for legitimate accounts
            for i in range(self.rng.randint(5, 20)):
                hist_time = current_time - timedelta(days=self.rng.randint(1, 60))
                hist_amount = self.np_rng.lognormal(mean=5.0, sigma=1.0)
                feature_store.update_account(account_id, hist_amount, hist_time, was_flagged=False)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_training_dataset(
    n_samples: int = 5000,
    fraud_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Generate a complete training dataset.
    
    Args:
        n_samples: Number of samples
        fraud_ratio: Fraction of fraud cases
        seed: Random seed
    
    Returns:
        X: Feature matrix
        y: Labels
        fraud_types: Type of each sample
        feature_names: Names of features
    """
    generator = SyntheticDataGenerator(seed=seed)
    X, y, fraud_types = generator.generate_training_data(n_samples, fraud_ratio)
    feature_names = get_feature_store().feature_names
    
    return X, y, fraud_types, feature_names
