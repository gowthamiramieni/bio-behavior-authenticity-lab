"""
Model training, improvement pipeline, cross-validation, multimodal fusion,
and persistence for BioAuth Lab.
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.base import BaseEstimator, ClassifierMixin

from core.metrics import compute_eer, evaluate_predictions
from core.db import get_session, Sample, ModelRecord, EvaluationResult

MODELS_DIR = Path(__file__).resolve().parent.parent / "models_store"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Anomaly Model Wrappers (Normalized Continuous Scores)
# ---------------------------------------------------------
class OneClassSVMWrapper(BaseEstimator, ClassifierMixin):
    """Wraps OneClassSVM to output normalized [0, 1] genuine scores."""
    _estimator_type = "classifier"

    def __init__(self, kernel="rbf", gamma="scale", nu=0.1):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.model = None
        self._min_score = -1.0
        self._max_score = 1.0

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        # Fit primarily on genuine class (y == 1) if y provided
        if y is not None:
            mask = (np.asarray(y) == 1)
            X_train = X[mask] if np.sum(mask) >= 3 else X
        else:
            X_train = X

        self.model = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)
        self.model.fit(X_train)
        raw_scores = self.model.decision_function(X_train)
        self._min_score = float(np.percentile(raw_scores, 2))
        self._max_score = float(np.percentile(raw_scores, 98))
        if self._max_score == self._min_score:
            self._max_score += 1e-4
        return self

    def predict_proba(self, X):
        raw = self.model.decision_function(X)
        # Min-max scale to [0, 1] with clipping
        norm = (raw - self._min_score) / (self._max_score - self._min_score)
        norm = np.clip(norm, 0.0, 1.0)
        # Return probability matrix [[P(0), P(1)]]
        return np.vstack([1.0 - norm, norm]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class IsolationForestWrapper(BaseEstimator, ClassifierMixin):
    """Wraps IsolationForest to output normalized [0, 1] genuine scores."""
    _estimator_type = "classifier"

    def __init__(self, n_estimators=100, contamination=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self._min_score = -0.5
        self._max_score = 0.5

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        if y is not None:
            mask = (np.asarray(y) == 1)
            X_train = X[mask] if np.sum(mask) >= 3 else X
        else:
            X_train = X

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model.fit(X_train)
        raw_scores = self.model.decision_function(X_train)
        self._min_score = float(np.percentile(raw_scores, 2))
        self._max_score = float(np.percentile(raw_scores, 98))
        if self._max_score == self._min_score:
            self._max_score += 1e-4
        return self

    def predict_proba(self, X):
        raw = self.model.decision_function(X)
        norm = (raw - self._min_score) / (self._max_score - self._min_score)
        norm = np.clip(norm, 0.0, 1.0)
        return np.vstack([1.0 - norm, norm]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


# ---------------------------------------------------------
# Candidate Model Factory
# ---------------------------------------------------------
def get_candidate_models() -> Dict[str, Any]:
    """Returns the 5 candidate models required for comparison."""
    return {
        "One-Class SVM": OneClassSVMWrapper(kernel="rbf", gamma="scale", nu=0.1),
        "Isolation Forest": IsolationForestWrapper(n_estimators=100, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    }


# ---------------------------------------------------------
# Data Loader from SQLite
# ---------------------------------------------------------
def load_dataset(user_id: Optional[int] = None, modality: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Loads samples from SQLite database for specified user and modality.
    Returns (X_df, y_array, feature_names).
    """
    session = get_session()
    query = session.query(Sample)

    if user_id is not None:
        query = query.filter(Sample.user_id == user_id)
    if modality is not None:
        query = query.filter(Sample.modality == modality)

    samples = query.all()
    session.close()

    if not samples:
        return pd.DataFrame(), np.array([]), []

    records = []
    labels = []
    for s in samples:
        feats = s.get_features()
        if feats:
            records.append(feats)
            labels.append(1 if s.label == "genuine" else 0)

    df = pd.DataFrame(records).fillna(0.0)
    y = np.array(labels)
    feature_names = list(df.columns)
    return df, y, feature_names


# ---------------------------------------------------------
# Cross-Validation Model Comparison
# ---------------------------------------------------------
def compare_models(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
    """
    Compares 5 models using Stratified K-Fold cross validation.
    Returns ranking table, detailed metrics per model, and picks the best model by EER.
    """
    models = get_candidate_models()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    for name, model in models.items():
        fold_eers = []
        fold_aucs = []
        all_y_true = []
        all_y_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            try:
                # Clone/re-initialize
                if name == "One-Class SVM":
                    clf = OneClassSVMWrapper(kernel="rbf", gamma="scale", nu=0.1)
                elif name == "Isolation Forest":
                    clf = IsolationForestWrapper(n_estimators=100, random_state=42)
                elif name == "Random Forest":
                    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
                elif name == "K-Nearest Neighbors":
                    clf = KNeighborsClassifier(n_neighbors=min(5, max(len(X_tr) - 1, 1)), weights="distance")
                else:
                    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

                clf.fit(X_tr, y_tr)
                scores = clf.predict_proba(X_val)[:, 1]

                fold_eer, _, _, _, _ = compute_eer(y_val, scores)
                fold_eers.append(fold_eer)
                all_y_true.extend(y_val)
                all_y_scores.extend(scores)
            except Exception as e:
                fold_eers.append(0.5)

        eval_summary = evaluate_predictions(np.array(all_y_true), np.array(all_y_scores))
        results[name] = {
            "mean_eer": round(float(np.mean(fold_eers)), 4),
            "std_eer": round(float(np.std(fold_eers)), 4),
            "summary_metrics": eval_summary,
            "y_true": all_y_true,
            "y_scores": [round(float(s), 4) for s in all_y_scores],
        }

    # Best model has the lowest mean EER (and highest AUC as tiebreaker)
    best_model_name = min(
        results.keys(),
        key=lambda k: (results[k]["mean_eer"], -results[k]["summary_metrics"]["auc"])
    )

    return {
        "results": results,
        "best_model": best_model_name,
        "best_eer": results[best_model_name]["mean_eer"]
    }


# ---------------------------------------------------------
# Improvement Pipeline
# ---------------------------------------------------------
class BiometricPipeline:
    """
    Full improvement pipeline:
    1. Outlier filtering on training genuine samples
    2. StandardScaler
    3. Feature Selection (Mutual Information SelectKBest)
    4. Stratified K-Fold GridSearchCV with class weighting
    """
    def __init__(self, base_model_name: str = "Random Forest"):
        self.base_model_name = base_model_name
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.model = None
        self.selected_indices_ = None
        self.best_params_ = {}

    def _remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Removes extreme noise/outliers from genuine samples using Isolation Forest."""
        genuine_mask = (y == 1)
        if np.sum(genuine_mask) <= 6:
            return X, y

        iso = IsolationForest(contamination=0.08, random_state=42)
        inlier_pred = iso.fit_predict(X[genuine_mask])
        keep_genuine = (inlier_pred == 1)

        # Retain all impostors and clean genuine samples
        clean_X = np.vstack([X[genuine_mask][keep_genuine], X[~genuine_mask]])
        clean_y = np.hstack([y[genuine_mask][keep_genuine], y[~genuine_mask]])
        return clean_X, clean_y

    def fit(self, X: np.ndarray, y: np.ndarray, tune_grid: bool = True):
        # 1. Outlier removal on training set
        X_clean, y_clean = self._remove_outliers(X, y)

        # 2. StandardScaler
        X_scaled = self.scaler.fit_transform(X_clean)

        # 3. Feature selection using Mutual Information
        k_features = min(max(4, X_scaled.shape[1] // 2), X_scaled.shape[1])
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
        X_selected = self.feature_selector.fit_transform(X_scaled, y_clean)
        self.selected_indices_ = self.feature_selector.get_support(indices=True)

        # 4. Model selection & Hyperparameter Tuning
        if self.base_model_name == "Random Forest":
            estimator = RandomForestClassifier(class_weight="balanced", random_state=42)
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [None, 6, 10],
                "min_samples_split": [2, 4],
            } if tune_grid else {}
        elif self.base_model_name == "Gradient Boosting":
            estimator = GradientBoostingClassifier(random_state=42)
            param_grid = {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            } if tune_grid else {}
        elif self.base_model_name == "K-Nearest Neighbors":
            estimator = KNeighborsClassifier(weights="distance")
            param_grid = {
                "n_neighbors": [3, 5, 7],
                "metric": ["euclidean", "manhattan"],
            } if tune_grid else {}
        elif self.base_model_name == "One-Class SVM":
            estimator = OneClassSVMWrapper()
            param_grid = {
                "gamma": ["scale", "auto", 0.05],
                "nu": [0.05, 0.1, 0.15],
            } if tune_grid else {}
        else:  # Isolation Forest
            estimator = IsolationForestWrapper(random_state=42)
            param_grid = {
                "n_estimators": [50, 100, 150],
                "contamination": [0.05, 0.1],
            } if tune_grid else {}

        if param_grid and tune_grid and len(np.unique(y_clean)) > 1:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid = GridSearchCV(estimator, param_grid, cv=cv, scoring="roc_auc", n_jobs=1)
            grid.fit(X_selected, y_clean)
            self.model = grid.best_estimator_
            self.best_params_ = grid.best_params_
        else:
            estimator.fit(X_selected, y_clean)
            self.model = estimator
            self.best_params_ = {}

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.feature_selector.transform(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_trans = self.transform(X)
        return self.model.predict_proba(X_trans)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def run_improvement_benchmark(user_id: int, modality: str) -> Dict[str, Any]:
    """
    Runs Before vs After Improvement comparison for a given user & modality.
    Saves before and after evaluation results to the SQLite results table.
    """
    X_df, y, feat_names = load_dataset(user_id=user_id, modality=modality)
    if len(X_df) < 10:
        return {}

    X = X_df.values
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 1. Before Improvement (Baseline: default unscaled Random Forest, no feature selection)
    before_y_true = []
    before_y_scores = []
    for tr_idx, val_idx in skf.split(X, y):
        base_clf = RandomForestClassifier(n_estimators=10, random_state=42)
        base_clf.fit(X[tr_idx], y[tr_idx])
        scores = base_clf.predict_proba(X[val_idx])[:, 1]
        before_y_true.extend(y[val_idx])
        before_y_scores.extend(scores)

    metrics_before = evaluate_predictions(np.array(before_y_true), np.array(before_y_scores))

    # 2. After Improvement (Tuned pipeline with Outlier removal + Scaling + SelectKBest + GridSearch)
    after_y_true = []
    after_y_scores = []
    for tr_idx, val_idx in skf.split(X, y):
        pipeline = BiometricPipeline(base_model_name="Random Forest")
        pipeline.fit(X[tr_idx], y[tr_idx], tune_grid=True)
        scores = pipeline.predict_proba(X[val_idx])[:, 1]
        after_y_true.extend(y[val_idx])
        after_y_scores.extend(scores)

    metrics_after = evaluate_predictions(np.array(after_y_true), np.array(after_y_scores))

    # Save to SQLite results table
    session = get_session()
    # Remove prior entries for this user and modality
    session.query(EvaluationResult).filter_by(user_id=user_id, modality=modality).delete()

    res_before = EvaluationResult(
        user_id=user_id,
        stage="before_improvement",
        modality=modality,
        model_name="Random Forest (Baseline)",
        metrics=json.dumps(metrics_before),
    )
    res_after = EvaluationResult(
        user_id=user_id,
        stage="after_improvement",
        modality=modality,
        model_name="Random Forest (Tuned Pipeline)",
        metrics=json.dumps(metrics_after),
    )
    session.add(res_before)
    session.add(res_after)
    session.commit()
    session.close()

    return {
        "modality": modality,
        "before": metrics_before,
        "after": metrics_after,
        "eer_improvement": round(metrics_before["eer"] - metrics_after["eer"], 4),
        "auc_improvement": round(metrics_after["auc"] - metrics_before["auc"], 4),
    }


# ---------------------------------------------------------
# Multimodal Score-Level Fusion
# ---------------------------------------------------------
class MultimodalFusionSystem:
    """
    Score-level fusion across Keystroke, Mouse, and Touch.
    Combines sub-scores via learned optimal convex weights:
      S_fusion = w_k * S_k + w_m * S_m + w_t * S_t
    """
    def __init__(self, weights: Optional[List[float]] = None):
        # Default equal weighting across 3 modalities
        self.weights = weights if weights else [0.34, 0.33, 0.33]
        self.modality_models = {}

    def fit_weights(self, scores_dict: Dict[str, np.ndarray], y_true: np.ndarray, step: float = 0.05):
        """
        Grid-searches optimal simplex weights (w_k, w_m, w_t) to minimize EER on validation scores.
        """
        s_k = scores_dict["keystroke"]
        s_m = scores_dict["mouse"]
        s_t = scores_dict["touch"]

        best_eer = 1.0
        best_w = [0.34, 0.33, 0.33]

        # Search over simplex
        w_range = np.arange(0.0, 1.01, step)
        for w1 in w_range:
            for w2 in w_range:
                if w1 + w2 > 1.0:
                    continue
                w3 = 1.0 - (w1 + w2)
                fused_scores = w1 * s_k + w2 * s_m + w3 * s_t
                eer, _, _, _, _ = compute_eer(y_true, fused_scores)
                if eer < best_eer:
                    best_eer = eer
                    best_w = [round(float(w1), 3), round(float(w2), 3), round(float(w3), 3)]

        self.weights = best_w
        return best_w, best_eer

    def fuse(self, score_keystroke: float, score_mouse: float, score_touch: float) -> Tuple[float, Dict[str, float]]:
        """
        Fuses three modality genuine scores into a single overall biometric score.
        Returns (fusion_score, modality_contributions).
        """
        wk, wm, wt = self.weights
        fusion_score = float(wk * score_keystroke + wm * score_mouse + wt * score_touch)
        fusion_score = min(max(fusion_score, 0.0), 1.0)

        # Modality contribution shares
        total = (wk * score_keystroke + wm * score_mouse + wt * score_touch) + 1e-6
        contributions = {
            "keystroke": round(float((wk * score_keystroke) / total * 100.0), 1),
            "mouse": round(float((wm * score_mouse) / total * 100.0), 1),
            "touch": round(float((wt * score_touch) / total * 100.0), 1),
        }
        return round(fusion_score, 4), contributions


# ---------------------------------------------------------
# Training & Persistence Pipeline
# ---------------------------------------------------------
def train_and_save_user_models(user_id: int) -> Dict[str, Any]:
    """
    Trains and persists the best models for each modality for a specific user,
    tunes multimodal fusion weights, and records everything in SQLite.
    """
    modalities = ["keystroke", "mouse", "touch"]
    trained_pipelines = {}
    validation_scores = {}
    y_labels = None

    session = get_session()

    for mod in modalities:
        X_df, y, feat_names = load_dataset(user_id=user_id, modality=mod)
        if len(X_df) < 8:
            continue

        if y_labels is None:
            y_labels = y

        X = X_df.values

        # 1. Compare models and pick the best
        comparison = compare_models(X, y)
        best_name = comparison["best_model"]

        # 2. Fit improved pipeline on best model
        pipeline = BiometricPipeline(base_model_name=best_name)
        pipeline.fit(X, y, tune_grid=True)
        trained_pipelines[mod] = pipeline

        # Save model artifact to disk
        filename = f"model_user_{user_id}_{mod}.joblib"
        file_path = MODELS_DIR / filename
        joblib.dump(pipeline, file_path)

        # In-sample validation scores for fusion tuning
        val_scores = pipeline.predict_proba(X)[:, 1]
        validation_scores[mod] = val_scores

        # Save/update ModelRecord in DB
        session.query(ModelRecord).filter_by(user_id=user_id, modality=mod).delete()
        model_rec = ModelRecord(
            user_id=user_id,
            modality=mod,
            model_type=best_name,
            filepath=str(file_path),
            params=json.dumps(pipeline.best_params_),
            metrics=json.dumps(comparison["results"][best_name]["summary_metrics"]),
        )
        session.add(model_rec)

    # 3. Train multimodal score-level fusion if all 3 modalities present
    fusion_metrics = {}
    if len(validation_scores) == 3 and y_labels is not None:
        fusion_sys = MultimodalFusionSystem()
        best_weights, best_eer = fusion_sys.fit_weights(validation_scores, y_labels)

        # Fused scores
        wk, wm, wt = best_weights
        fused_val = wk * validation_scores["keystroke"] + wm * validation_scores["mouse"] + wt * validation_scores["touch"]
        fusion_eval = evaluate_predictions(y_labels, fused_val)

        # Persist fusion system
        fusion_filename = f"model_user_{user_id}_fusion.joblib"
        fusion_path = MODELS_DIR / fusion_filename
        joblib.dump(fusion_sys, fusion_path)

        session.query(ModelRecord).filter_by(user_id=user_id, modality="fusion").delete()
        fusion_rec = ModelRecord(
            user_id=user_id,
            modality="fusion",
            model_type="MultimodalScoreFusion",
            filepath=str(fusion_path),
            params=json.dumps({"weights": best_weights}),
            metrics=json.dumps(fusion_eval),
        )
        session.add(fusion_rec)
        fusion_metrics = {
            "weights": best_weights,
            "metrics": fusion_eval,
        }

    session.commit()
    session.close()

    return {
        "user_id": user_id,
        "modalities": list(trained_pipelines.keys()),
        "fusion": fusion_metrics,
    }


def load_user_pipeline(user_id: int, modality: str) -> Optional[BiometricPipeline]:
    """Loads a serialized biometric pipeline from disk."""
    filename = f"model_user_{user_id}_{modality}.joblib"
    path = MODELS_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None


def load_user_fusion(user_id: int) -> Optional[MultimodalFusionSystem]:
    """Loads a serialized multimodal fusion system from disk."""
    filename = f"model_user_{user_id}_fusion.joblib"
    path = MODELS_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None
