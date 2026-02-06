"""
Ensemble Fraud Detection Model

Combines multiple ML models for robust fraud detection:
1. Isolation Forest - Unsupervised anomaly detection
2. XGBoost Classifier - Supervised fraud classification
3. Heuristic Rules - Domain knowledge-based scoring

The ensemble uses weighted voting to combine predictions,
providing both accuracy and interpretability.

Model Metrics:
- Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance
- ROC-AUC Score
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
from collections import deque

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

# XGBoost for supervised learning
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, using RandomForest fallback")

if not XGBOOST_AVAILABLE:
    from sklearn.ensemble import RandomForestClassifier

from services.feature_store import (
    FeatureStore,
    TransactionFeatures,
    get_feature_store,
)
from services.synthetic_data import generate_training_dataset

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class ModelMetrics(NamedTuple):
    """Comprehensive model performance metrics."""
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    accuracy: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class PredictionResult(NamedTuple):
    """Result from ensemble prediction."""
    is_fraud: bool
    fraud_probability: float
    risk_score: int  # 0-100
    risk_level: str  # LOW, MEDIUM, HIGH
    
    # Individual model contributions
    isolation_forest_score: float
    xgboost_score: float
    heuristic_score: float
    
    # Feature importance for this prediction
    top_features: List[Tuple[str, float]]
    
    # Explanation
    explanation: str


class FeedbackRecord(NamedTuple):
    """Record of human feedback on predictions."""
    trade_id: str
    timestamp: datetime
    predicted_fraud: bool
    actual_fraud: bool  # From human feedback
    features: np.ndarray


# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

class EnsembleFraudDetector:
    """
    Ensemble model combining multiple fraud detection approaches.
    
    Architecture:
    1. Isolation Forest (30% weight) - Catches novel anomalies
    2. XGBoost/RandomForest (50% weight) - Learns from labeled patterns
    3. Heuristic Rules (20% weight) - Domain knowledge baseline
    
    The model provides:
    - Fraud probability scores
    - Feature importance explanations
    - Performance metrics
    - Feedback loop for continuous learning
    """
    
    def __init__(
        self,
        isolation_weight: float = 0.30,
        supervised_weight: float = 0.50,
        heuristic_weight: float = 0.20,
    ):
        """
        Initialize ensemble model.
        
        Args:
            isolation_weight: Weight for Isolation Forest
            supervised_weight: Weight for XGBoost/RandomForest
            heuristic_weight: Weight for heuristic rules
        """
        self.weights = {
            "isolation_forest": isolation_weight,
            "supervised": supervised_weight,
            "heuristic": heuristic_weight,
        }
        
        # Models
        self.isolation_forest: Optional[IsolationForest] = None
        self.supervised_model = None  # XGBoost or RandomForest
        self.scaler: Optional[StandardScaler] = None
        
        # Feature names
        self.feature_names: List[str] = []
        
        # Training state
        self.is_trained = False
        self.training_timestamp: Optional[datetime] = None
        self.training_samples: int = 0
        
        # Performance metrics
        self.metrics: Optional[ModelMetrics] = None
        
        # Feedback buffer for incremental learning
        self._feedback_buffer: deque = deque(maxlen=1000)
        self._retrain_threshold: int = 100  # Retrain after N feedback samples
        
        # Feature importance cache
        self._feature_importance: Dict[str, float] = {}
        
        logger.info(f"EnsembleFraudDetector initialized with weights: {self.weights}")
    
    def train(
        self,
        n_samples: int = 5000,
        fraud_ratio: float = 0.1,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> ModelMetrics:
        """
        Train the ensemble model on synthetic data.
        
        Args:
            n_samples: Number of training samples
            fraud_ratio: Proportion of fraud cases
            test_size: Proportion for testing
            seed: Random seed
        
        Returns:
            ModelMetrics with performance on test set
        """
        logger.info(f"Training ensemble model with {n_samples} samples...")
        start_time = time.time()
        
        # Generate synthetic training data
        X, y, fraud_types, self.feature_names = generate_training_dataset(
            n_samples=n_samples,
            fraud_ratio=fraud_ratio,
            seed=seed,
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        # Fit scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Isolation Forest
        logger.info("Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=fraud_ratio,
            random_state=seed,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_train_scaled)
        
        # Train Supervised Model (XGBoost or RandomForest)
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost classifier...")
            self.supervised_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=seed,
                eval_metric='logloss',
            )
        else:
            logger.info("Training RandomForest classifier (XGBoost fallback)...")
            self.supervised_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=seed,
                n_jobs=-1,
            )
        
        self.supervised_model.fit(X_train_scaled, y_train)
        
        # Compute feature importance
        self._compute_feature_importance()
        
        # Evaluate on test set
        self.metrics = self._evaluate(X_test_scaled, y_test)
        
        # Update state
        self.is_trained = True
        self.training_timestamp = datetime.now(timezone.utc)
        self.training_samples = n_samples
        
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f}s")
        logger.info(f"Test metrics: F1={self.metrics.f1_score:.3f}, "
                   f"Precision={self.metrics.precision:.3f}, "
                   f"Recall={self.metrics.recall:.3f}")
        
        return self.metrics
    
    def predict(self, features: TransactionFeatures) -> PredictionResult:
        """
        Make fraud prediction using ensemble.
        
        Args:
            features: Transaction features from FeatureStore
        
        Returns:
            PredictionResult with scores and explanations
        """
        if not self.is_trained:
            # Fallback to heuristic-only if not trained
            return self._heuristic_only_prediction(features)
        
        # Convert features to array and scale
        feature_store = get_feature_store()
        X = feature_store.features_to_array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get individual model scores
        
        # 1. Isolation Forest (-1 to 1, where -1 is anomaly)
        if_score_raw = self.isolation_forest.decision_function(X_scaled)[0]
        # Convert to probability-like (0 to 1, where 1 is anomaly)
        if_score = 1 - (if_score_raw + 0.5)  # Normalize roughly to [0, 1]
        if_score = max(0, min(1, if_score))
        
        # 2. Supervised model probability
        if hasattr(self.supervised_model, 'predict_proba'):
            sup_score = self.supervised_model.predict_proba(X_scaled)[0, 1]
        else:
            sup_score = float(self.supervised_model.predict(X_scaled)[0])
        
        # 3. Heuristic score (from features)
        heur_score = features.risk_score_heuristic
        
        # Combine with weights
        combined_score = (
            self.weights["isolation_forest"] * if_score +
            self.weights["supervised"] * sup_score +
            self.weights["heuristic"] * heur_score
        )
        
        # Determine risk level
        risk_score = int(combined_score * 100)
        if risk_score >= 70:
            risk_level = "HIGH"
            is_fraud = True
        elif risk_score >= 40:
            risk_level = "MEDIUM"
            is_fraud = False
        else:
            risk_level = "LOW"
            is_fraud = False
        
        # Get top contributing features
        top_features = self._get_top_features(X[0])
        
        # Generate explanation
        explanation = self._generate_explanation(
            features, risk_score, risk_level, if_score, sup_score, heur_score, top_features
        )
        
        return PredictionResult(
            is_fraud=is_fraud,
            fraud_probability=combined_score,
            risk_score=risk_score,
            risk_level=risk_level,
            isolation_forest_score=if_score,
            xgboost_score=sup_score,
            heuristic_score=heur_score,
            top_features=top_features,
            explanation=explanation,
        )
    
    def record_feedback(
        self,
        trade_id: str,
        features: TransactionFeatures,
        predicted_fraud: bool,
        actual_fraud: bool,
    ) -> Dict:
        """
        Record human feedback for model improvement.
        
        Args:
            trade_id: Trade identifier
            features: Original features
            predicted_fraud: Model's prediction
            actual_fraud: Human's label (ground truth)
        
        Returns:
            Feedback status and potential retraining info
        """
        feature_store = get_feature_store()
        feature_array = feature_store.features_to_array(features)
        
        feedback = FeedbackRecord(
            trade_id=trade_id,
            timestamp=datetime.now(timezone.utc),
            predicted_fraud=predicted_fraud,
            actual_fraud=actual_fraud,
            features=feature_array,
        )
        
        self._feedback_buffer.append(feedback)
        
        result = {
            "recorded": True,
            "feedback_count": len(self._feedback_buffer),
            "is_correct": predicted_fraud == actual_fraud,
            "will_retrain": len(self._feedback_buffer) >= self._retrain_threshold,
        }
        
        # Check if we should retrain
        if result["will_retrain"]:
            logger.info(f"Feedback threshold reached ({self._retrain_threshold}), triggering retrain...")
            self._incremental_retrain()
            result["retrained"] = True
        
        logger.info(f"Feedback recorded for {trade_id}: predicted={predicted_fraud}, actual={actual_fraud}")
        
        return result
    
    def get_metrics(self) -> Optional[Dict]:
        """Get current model metrics as dictionary."""
        if self.metrics is None:
            return None
        
        return {
            "precision": round(self.metrics.precision, 4),
            "recall": round(self.metrics.recall, 4),
            "f1_score": round(self.metrics.f1_score, 4),
            "roc_auc": round(self.metrics.roc_auc, 4),
            "accuracy": round(self.metrics.accuracy, 4),
            "confusion_matrix": self.metrics.confusion_matrix,
            "true_positives": self.metrics.true_positives,
            "true_negatives": self.metrics.true_negatives,
            "false_positives": self.metrics.false_positives,
            "false_negatives": self.metrics.false_negatives,
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return dict(self._feature_importance)
    
    def get_status(self) -> Dict:
        """Get model status information."""
        return {
            "is_trained": self.is_trained,
            "model_type": "XGBoost" if XGBOOST_AVAILABLE else "RandomForest",
            "training_samples": self.training_samples,
            "training_timestamp": self.training_timestamp.isoformat() if self.training_timestamp else None,
            "feedback_buffer_size": len(self._feedback_buffer),
            "retrain_threshold": self._retrain_threshold,
            "weights": self.weights,
            "feature_count": len(self.feature_names),
        }
    
    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """Evaluate ensemble on test data."""
        # Get predictions
        y_pred = []
        y_prob = []
        
        for i in range(len(X_test)):
            # Isolation Forest
            if_score_raw = self.isolation_forest.decision_function(X_test[i:i+1])[0]
            if_score = 1 - (if_score_raw + 0.5)
            if_score = max(0, min(1, if_score))
            
            # Supervised
            if hasattr(self.supervised_model, 'predict_proba'):
                sup_score = self.supervised_model.predict_proba(X_test[i:i+1])[0, 1]
            else:
                sup_score = float(self.supervised_model.predict(X_test[i:i+1])[0])
            
            # Heuristic (last feature is heuristic score)
            heur_score = X_test[i, -1] if len(X_test[i]) > 0 else 0.5
            
            # Combine
            combined = (
                self.weights["isolation_forest"] * if_score +
                self.weights["supervised"] * sup_score +
                self.weights["heuristic"] * heur_score
            )
            
            y_prob.append(combined)
            y_pred.append(1 if combined >= 0.5 else 0)
        
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Compute metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            roc_auc = 0.5
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / len(y_test)
        
        return ModelMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm.tolist(),
            accuracy=accuracy,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
        )
    
    def _compute_feature_importance(self) -> None:
        """Compute and cache feature importance."""
        if self.supervised_model is None:
            return
        
        # Get importance from supervised model
        if hasattr(self.supervised_model, 'feature_importances_'):
            importances = self.supervised_model.feature_importances_
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)
        
        # Normalize to sum to 1
        importances = importances / importances.sum()
        
        # Store as dict
        self._feature_importance = {
            name: float(imp) 
            for name, imp in zip(self.feature_names, importances)
        }
    
    def _get_top_features(self, x: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top contributing features for a prediction."""
        if not self._feature_importance:
            return []
        
        # Weight feature values by importance
        contributions = []
        for i, (name, importance) in enumerate(zip(self.feature_names, self._feature_importance.values())):
            # Contribution = importance * normalized feature value
            value = x[i] if i < len(x) else 0
            contribution = importance * abs(value)
            contributions.append((name, contribution, value))
        
        # Sort by contribution
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k with original values
        return [(name, round(val, 4)) for name, _, val in contributions[:top_k]]
    
    def _generate_explanation(
        self,
        features: TransactionFeatures,
        risk_score: int,
        risk_level: str,
        if_score: float,
        sup_score: float,
        heur_score: float,
        top_features: List[Tuple[str, float]],
    ) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        # Main risk statement
        if risk_level == "HIGH":
            parts.append(f"High risk transaction detected (score: {risk_score}/100).")
        elif risk_level == "MEDIUM":
            parts.append(f"Moderate risk indicators present (score: {risk_score}/100).")
        else:
            parts.append(f"Transaction appears normal (score: {risk_score}/100).")
        
        # Model agreement
        models_flagging = sum([
            1 if if_score > 0.5 else 0,
            1 if sup_score > 0.5 else 0,
            1 if heur_score > 0.5 else 0,
        ])
        
        if models_flagging == 3:
            parts.append("All 3 models agree on risk assessment.")
        elif models_flagging >= 2:
            parts.append(f"{models_flagging}/3 models flagged this transaction.")
        
        # Specific factors
        factors = []
        if features.amount > 50000:
            factors.append(f"large amount (${features.amount:,.2f})")
        if features.velocity_10min > 3:
            factors.append(f"high velocity ({features.velocity_10min} trades/10min)")
        if features.is_unusual_hour:
            factors.append("unusual trading hours")
        if features.is_amount_spike:
            factors.append("amount spike vs account average")
        if features.is_rapid_succession:
            factors.append("rapid succession of trades")
        if features.account_age_days < 1:
            factors.append("new account")
        
        if factors:
            parts.append(f"Key factors: {', '.join(factors)}.")
        
        # Top features
        if top_features and risk_score >= 40:
            feature_str = ", ".join([f"{name}={val}" for name, val in top_features[:3]])
            parts.append(f"Top signals: {feature_str}.")
        
        return " ".join(parts)
    
    def _heuristic_only_prediction(self, features: TransactionFeatures) -> PredictionResult:
        """Fallback prediction using only heuristics."""
        risk_score = int(features.risk_score_heuristic * 100)
        
        if risk_score >= 70:
            risk_level = "HIGH"
            is_fraud = True
        elif risk_score >= 40:
            risk_level = "MEDIUM"
            is_fraud = False
        else:
            risk_level = "LOW"
            is_fraud = False
        
        return PredictionResult(
            is_fraud=is_fraud,
            fraud_probability=features.risk_score_heuristic,
            risk_score=risk_score,
            risk_level=risk_level,
            isolation_forest_score=0.0,
            xgboost_score=0.0,
            heuristic_score=features.risk_score_heuristic,
            top_features=[],
            explanation=f"Heuristic-only prediction (model not trained). Risk score: {risk_score}/100",
        )
    
    def _incremental_retrain(self) -> None:
        """Incrementally update model with feedback data."""
        if len(self._feedback_buffer) < 10:
            return
        
        # Extract feedback data
        X_feedback = np.array([f.features for f in self._feedback_buffer])
        y_feedback = np.array([1 if f.actual_fraud else 0 for f in self._feedback_buffer])
        
        logger.info(f"🔄 Incremental retrain with {len(self._feedback_buffer)} feedback samples...")
        
        try:
            # Scale the feedback features
            X_feedback_scaled = self.scaler.transform(X_feedback)
            
            # XGBoost doesn't support true incremental learning, but we can:
            # 1. Generate new synthetic data
            # 2. Combine with feedback data (weighted higher)
            # 3. Retrain from scratch
            
            # Generate synthetic base data (smaller batch)
            X_base, y_base, _, _ = generate_training_dataset(
                n_samples=2000, 
                fraud_ratio=0.1
            )
            X_base_scaled = self.scaler.transform(X_base)
            
            # Combine: feedback data is repeated to give it more weight
            X_combined = np.vstack([X_base_scaled, X_feedback_scaled, X_feedback_scaled])
            y_combined = np.hstack([y_base, y_feedback, y_feedback])
            
            # Shuffle
            indices = np.random.permutation(len(X_combined))
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]
            
            # Retrain supervised model
            self.supervised_model.fit(X_combined, y_combined)
            
            # Update Isolation Forest with combined data
            self.isolation_forest.fit(X_combined)
            
            # Update feature importance
            self._compute_feature_importance()
            
            # Clear feedback buffer
            self._feedback_buffer.clear()
            
            logger.info("✅ Incremental retrain complete - model updated with human feedback")
            
        except Exception as e:
            logger.error(f"❌ Incremental retrain failed: {e}")
            # Still clear buffer to prevent repeated failures
            self._feedback_buffer.clear()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_ensemble_model: Optional[EnsembleFraudDetector] = None


def get_ensemble_model() -> EnsembleFraudDetector:
    """Get the global ensemble model instance."""
    global _ensemble_model
    if _ensemble_model is None:
        _ensemble_model = EnsembleFraudDetector()
    return _ensemble_model


def initialize_ensemble_model(
    n_samples: int = 5000,
    fraud_ratio: float = 0.1,
) -> ModelMetrics:
    """Initialize and train the ensemble model."""
    global _ensemble_model
    _ensemble_model = EnsembleFraudDetector()
    return _ensemble_model.train(n_samples=n_samples, fraud_ratio=fraud_ratio)


def reset_ensemble_model() -> None:
    """Reset the ensemble model."""
    global _ensemble_model
    _ensemble_model = None
