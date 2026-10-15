"""
Random Forest Classifier for Net-track (Lee, Joo & Lee, WWW 2023).

Hyperparameters per Section 5.1:
- criterion: 'gini'
- max_depth: None
- max_features: 'sqrt'
- n_estimators: 55

NOTE: Decision Tree (DT), k-NN, MLP, and LSTM are intentionally excluded per
the paper's own conclusions (Section 5.1) establishing Random Forest as the
best-performing, most practical choice.
"""

import time
import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class NetTrackRandomForest:
    """Best-performing model from Net-track (Lee, Joo & Lee, WWW 2023).

    Hyperparameters per Section 5.1:
    - criterion: 'gini'
    - max_depth: None (pure leaves)
    - max_features: 'sqrt' (~9 features sampled per split)
    - n_estimators: 55
    """

    def __init__(
        self,
        n_estimators: int = 55,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.last_train_time_sec: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        start_t = time.perf_counter()
        self.model.fit(X, y)
        self.last_train_time_sec = time.perf_counter() - start_t
        return self.last_train_time_sec

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate_with_inference_time(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        start_t = time.perf_counter()
        preds = self.model.predict(X_test)
        total_inf_time_sec = time.perf_counter() - start_t

        num_samples = len(y_test)
        per_trace_inf_ms = (total_inf_time_sec / num_samples * 1000.0) if num_samples > 0 else 0.0

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "total_inference_time_sec": float(total_inf_time_sec),
            "per_trace_inference_ms": float(per_trace_inf_ms),
        }

    @property
    def feature_importances_(self) -> np.ndarray:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        raise ValueError("Model is not fitted yet.")
