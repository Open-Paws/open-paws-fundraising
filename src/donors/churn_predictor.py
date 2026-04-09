"""
Donor churn predictor — CatBoost patterns from DonorPulse.

Predicts which supporters are likely to stop donating within the next 180 days.
Uses CatBoost when installed; falls back to sklearn GradientBoostingClassifier.
Includes a rules-based fallback when no model has been trained.

Feature engineering mirrors DonorPulse's build_donor_training_table.py:
- Recency, frequency, monetary (RFM) signals
- Communication engagement rates
- Gift trend (recent 90d vs prior 90d)
- Categorical: preferred_campaign_topic, acquisition_source, region

No PII is present in any feature. All identifiers are pseudonymous.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .models import ChurnRisk, Donor

logger = logging.getLogger(__name__)

# Risk band thresholds — match DonorPulse model_config.json
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40

# Feature columns expected by the trained model
FEATURE_COLS = [
    "days_since_last_donation",
    "days_since_last_communication",
    "open_rate_last_90d",
    "click_rate_last_90d",
    "donation_count_12mo",
    "total_donations_12mo",
    "average_gift_size",
    "campaigns_engaged",
    "is_recurring_donor",
    "preferred_campaign_topic",  # categorical
]

CAT_COLS = ["preferred_campaign_topic"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]


def _risk_band(score: float) -> ChurnRisk:
    if score >= HIGH_RISK_THRESHOLD:
        return ChurnRisk.HIGH
    if score >= MEDIUM_RISK_THRESHOLD:
        return ChurnRisk.MEDIUM
    return ChurnRisk.LOW


def _rules_based_risk(donor: Donor) -> tuple[ChurnRisk, float]:
    """
    Simple rules-based fallback when no trained model is available.

    Rules derived from DonorPulse's predict_churn_risk.py risk_band logic:
    - days_since_last_donation >= 365 → HIGH (0.85)
    - days_since_last_donation >= 180 → MEDIUM (0.55)
    - no engagement in 90 days → bump one tier
    - else LOW (0.15)
    """
    days = donor.days_since_last_donation

    if days >= 365:
        return ChurnRisk.LAPSED, 0.90

    base_score: float
    if days >= 180:
        base_score = 0.75
    elif days >= 90:
        base_score = 0.50
    else:
        base_score = 0.20

    # Low engagement pushes score up
    if donor.open_rate_last_90d < 0.10 and donor.campaigns_engaged == 0:
        base_score = min(base_score + 0.15, 0.95)

    # Recurring donors are more stable
    if donor.is_recurring_donor:
        base_score = max(base_score - 0.10, 0.05)

    return _risk_band(base_score), round(base_score, 4)


def _recommendation(donor: Donor) -> str:
    """Generate a recommended action string from churn risk + donor signals."""
    risk = donor.churn_risk
    if risk in (ChurnRisk.HIGH, ChurnRisk.LAPSED):
        if donor.is_recurring_donor:
            return (
                "Personal outreach from fundraiser; review recurring payment health; "
                "send impact update."
            )
        if donor.open_rate_last_90d < 0.20:
            return (
                "Try a reactivation campaign with a different channel and stronger "
                "mission story."
            )
        return "Prioritize personal email/call and a targeted renewal ask within 7 days."
    if risk == ChurnRisk.MEDIUM:
        return (
            "Send tailored impact content and a soft re-engagement ask; "
            "monitor for 30 days."
        )
    return "Keep in regular stewardship flow; no urgent action needed."


class ChurnPredictor:
    """
    Donor churn predictor wrapping CatBoost (preferred) or sklearn fallback.

    Usage:
        predictor = ChurnPredictor()

        # Option A: load a pre-trained model
        predictor.load("models/")

        # Option B: train from donor event data
        auc = predictor.train(training_records)

        # Score donors
        scored_donors = predictor.predict(donors)
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._manifest: dict = {}
        self._trained = False

    def load(self, model_dir: str | Path) -> None:
        """Load a pre-trained model from a directory containing model_manifest.json."""
        model_dir = Path(model_dir)
        manifest_path = model_dir / "model_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"No model_manifest.json found in {model_dir}")

        with open(manifest_path, encoding="utf-8") as f:
            self._manifest = json.load(f)

        fmt = self._manifest.get("model_format", "")
        if fmt == "catboost_cbm":
            try:
                from catboost import CatBoostClassifier  # type: ignore[import]

                model = CatBoostClassifier()
                model.load_model(str(model_dir / self._manifest["model_path"]))
                self._model = model
            except ImportError:
                logger.warning("catboost not installed; falling back to rules-based scoring")
        elif fmt == "joblib_pipeline":
            import joblib  # type: ignore[import]

            self._model = joblib.load(model_dir / self._manifest["model_path"])

        self._trained = self._model is not None
        logger.info("Loaded model: %s (format=%s)", self._manifest.get("model_name"), fmt)

    def train(self, training_data: list[dict]) -> float:
        """
        Train a churn model from a list of donor feature dicts.

        Each dict must contain the FEATURE_COLS keys plus 'churned_in_next_180d' (0/1).

        Returns validation ROC-AUC. Falls back to sklearn if catboost not installed.
        Saves model to models/ directory.
        """
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        df = pd.DataFrame(training_data)
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feature_cols]
        y = df["churned_in_next_180d"]

        num_cols_present = [c for c in NUM_COLS if c in feature_cols]
        cat_cols_present = [c for c in CAT_COLS if c in feature_cols]

        split_idx = int(len(df) * 0.80)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Try CatBoost first
        try:
            from catboost import CatBoostClassifier  # type: ignore[import]

            X_train_cb = X_train.copy()
            X_val_cb = X_val.copy()
            for col in cat_cols_present:
                X_train_cb[col] = X_train_cb[col].astype(str)
                X_val_cb[col] = X_val_cb[col].astype(str)

            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                iterations=300,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=5,
                random_seed=42,
                auto_class_weights="Balanced",
                verbose=False,
            )
            model.fit(
                X_train_cb,
                y_train,
                cat_features=cat_cols_present,
                eval_set=(X_val_cb, y_val),
                use_best_model=True,
                verbose=False,
            )
            val_proba = model.predict_proba(X_val_cb)[:, 1]
            auc = float(roc_auc_score(y_val, val_proba))

            Path("models").mkdir(exist_ok=True)
            model.save_model("models/churn_model.cbm")
            manifest = {
                "model_name": "catboost",
                "model_format": "catboost_cbm",
                "model_path": "churn_model.cbm",
                "feature_columns": feature_cols,
                "categorical_columns": cat_cols_present,
                "decision_threshold": 0.5,
            }
            with open("models/model_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            self._model = model
            self._manifest = manifest
            self._trained = True
            logger.info("Trained CatBoost model — val AUC: %.4f", auc)
            return auc

        except ImportError:
            logger.info("catboost not installed; training sklearn GradientBoosting fallback")

        # Sklearn fallback
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]),
                    num_cols_present,
                ),
                (
                    "cat",
                    Pipeline([
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    cat_cols_present,
                ),
            ]
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
            )),
        ])
        pipeline.fit(X_train, y_train)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, val_proba))

        Path("models").mkdir(exist_ok=True)
        joblib.dump(pipeline, "models/churn_model.joblib")
        manifest = {
            "model_name": "sklearn_gradient_boosting",
            "model_format": "joblib_pipeline",
            "model_path": "churn_model.joblib",
            "feature_columns": feature_cols,
            "categorical_columns": cat_cols_present,
            "decision_threshold": 0.5,
        }
        with open("models/model_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self._model = pipeline
        self._manifest = manifest
        self._trained = True
        logger.info("Trained sklearn GradientBoosting fallback — val AUC: %.4f", auc)
        return auc

    def predict(self, donors: list[Donor]) -> list[Donor]:
        """
        Score a list of donors and populate churn_risk, churn_probability,
        and recommended_action on each.

        Falls back to rules-based scoring if no model is loaded.
        """
        if not self._trained:
            return self._predict_rules_based(donors)
        return self._predict_model(donors)

    def _predict_rules_based(self, donors: list[Donor]) -> list[Donor]:
        """Rules-based fallback: no model required."""
        for donor in donors:
            risk, prob = _rules_based_risk(donor)
            donor.churn_risk = risk
            donor.churn_probability = prob
            donor.recommended_action = _recommendation(donor)
        return donors

    def _predict_model(self, donors: list[Donor]) -> list[Donor]:
        """Model-based scoring using loaded CatBoost or sklearn pipeline."""
        rows = []
        for d in donors:
            rows.append({
                "days_since_last_donation": d.days_since_last_donation,
                "days_since_last_communication": d.days_since_last_donation,  # proxy
                "open_rate_last_90d": d.open_rate_last_90d,
                "click_rate_last_90d": d.click_rate_last_90d,
                "donation_count_12mo": d.donation_count_12mo,
                "total_donations_12mo": d.total_donations_12mo,
                "average_gift_size": d.average_gift_size,
                "campaigns_engaged": d.campaigns_engaged,
                "is_recurring_donor": int(d.is_recurring_donor),
                "preferred_campaign_topic": d.preferred_campaign_topic or "unknown",
            })

        feature_cols = self._manifest.get("feature_columns", FEATURE_COLS)
        cat_cols_present = self._manifest.get("categorical_columns", CAT_COLS)

        df = pd.DataFrame(rows)
        # Only use columns the model was trained on
        X = df[[c for c in feature_cols if c in df.columns]]

        model_name = self._manifest.get("model_name", "")
        if model_name == "catboost":
            X_score = X.copy()
            for col in cat_cols_present:
                if col in X_score.columns:
                    X_score[col] = X_score[col].astype(str)
            scores = self._model.predict_proba(X_score)[:, 1]
        else:
            scores = self._model.predict_proba(X)[:, 1]

        for donor, score in zip(donors, scores):
            score_f = float(score)
            donor.churn_probability = round(score_f, 6)
            donor.churn_risk = _risk_band(score_f)
            donor.recommended_action = _recommendation(donor)

        return donors
