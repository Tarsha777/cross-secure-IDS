"""Reusable ensemble inference helper for CROSS-SECURE."""

# Import standard library modules for string matching and path resolution.
import difflib
from pathlib import Path

# Import third-party libraries already present in the project.
import joblib
import pandas as pd


# Resolve model artifact paths relative to the project root.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

CICIDS_MODEL_PATH = MODEL_DIR / "cross_secure_model.pkl"
CICIDS_SCALER_PATH = MODEL_DIR / "scaler.pkl"
CICIDS_FEATURES_PATH = MODEL_DIR / "feature_names.pkl"
CICIDS_THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"

NSL_MODEL_PATH = MODEL_DIR / "nslkdd_model.pkl"
NSL_SCALER_PATH = MODEL_DIR / "nslkdd_scaler.pkl"
NSL_FEATURES_PATH = MODEL_DIR / "nslkdd_feature_names.pkl"


# Use the requested manual cross-domain feature mapping dictionary.
nsl_to_cicids = {
    "duration": "Flow Duration",
    "src_bytes": "Total Length of Fwd Packets",
    "dst_bytes": "Total Length of Bwd Packets",
    "wrong_fragment": "URG Flag Count",
    "urgent": "Fwd URG Flags",
    "hot": "PSH Flag Count",
    "count": "Total Fwd Packets",
    "srv_count": "Total Backward Packets",
    "serror_rate": "Fwd IAT Mean",
    "rerror_rate": "Bwd IAT Mean",
    "same_srv_rate": "Flow IAT Mean",
    "diff_srv_rate": "Flow IAT Std",
    "dst_host_count": "ACK Flag Count",
    "dst_host_srv_count": "SYN Flag Count",
    "num_failed_logins": "FIN Flag Count",
    "logged_in": "Init_Win_bytes_forward",
    "srv_serror_rate": "Bwd IAT Std",
    "rerror_rate": "RST Flag Count",
}


def normalize_name(name: str) -> str:
    """Normalize feature names so exact and approximate matches are easier to detect."""
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in str(name))
    tokens = [token for token in cleaned.split() if token]
    return " ".join(sorted(tokens))


def resolve_target_name(source_name: str, target_feature_names: list[str]) -> str | None:
    """Resolve a source feature into the target model's feature space."""
    target_set = set(target_feature_names)
    reverse_mapping = {value: key for key, value in nsl_to_cicids.items()}

    if source_name in target_set:
        return source_name

    mapped_name = nsl_to_cicids.get(source_name)
    if mapped_name in target_set:
        return mapped_name

    reverse_name = reverse_mapping.get(source_name)
    if reverse_name in target_set:
        return reverse_name

    normalized_target_lookup = {normalize_name(name): name for name in target_feature_names}
    normalized_source = normalize_name(source_name)

    if normalized_source in normalized_target_lookup:
        return normalized_target_lookup[normalized_source]

    close_matches = difflib.get_close_matches(
        normalized_source,
        list(normalized_target_lookup.keys()),
        n=1,
        cutoff=0.92,
    )
    if close_matches:
        return normalized_target_lookup[close_matches[0]]

    return None


def align_features(
    X_source: pd.DataFrame,
    source_feature_names: list[str],
    target_feature_names: list[str],
) -> pd.DataFrame:
    """Create a zero-filled dataframe in the target feature order and map compatible fields."""
    aligned = pd.DataFrame(0.0, index=X_source.index, columns=target_feature_names)

    for source_name in source_feature_names:
        if source_name not in X_source.columns:
            continue

        target_name = resolve_target_name(source_name, target_feature_names)
        if target_name is None:
            continue

        aligned[target_name] = pd.to_numeric(X_source[source_name], errors="coerce").fillna(0.0)

    return aligned


class CrossSecureEnsemble:
    """Run weighted ensemble inference across CICIDS2017 and NSL-KDD models."""

    # Keep the ensemble weights explicit and easy to inspect.
    CICIDS_WEIGHT = 0.7
    NSLKDD_WEIGHT = 0.3

    def __init__(self):
        """Load all model artifacts needed for ensemble inference."""
        try:
            self.cicids_model = joblib.load(CICIDS_MODEL_PATH)
            self.cicids_scaler = joblib.load(CICIDS_SCALER_PATH)
            self.cicids_feature_names = list(joblib.load(CICIDS_FEATURES_PATH))
            self.threshold = float(joblib.load(CICIDS_THRESHOLD_PATH))

            self.nslkdd_model = joblib.load(NSL_MODEL_PATH)
            self.nslkdd_scaler = joblib.load(NSL_SCALER_PATH)
            self.nslkdd_feature_names = list(joblib.load(NSL_FEATURES_PATH))

            print("Ensemble loaded: 2 models active")
        except Exception as exc:
            raise RuntimeError(f"Failed to load ensemble artifacts: {exc}") from exc

    def _feature_frame(self, feature_dict: dict) -> pd.DataFrame:
        """Convert an arbitrary feature dictionary into a single-row numeric dataframe."""
        source_frame = pd.DataFrame([feature_dict])
        return source_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def _predict_cicids(self, feature_dict: dict) -> tuple[float, int]:
        """Run the CICIDS2017 branch of the ensemble."""
        source_frame = self._feature_frame(feature_dict)
        aligned = align_features(
            X_source=source_frame,
            source_feature_names=source_frame.columns.tolist(),
            target_feature_names=self.cicids_feature_names,
        )
        scaled = pd.DataFrame(
            self.cicids_scaler.transform(aligned),
            columns=self.cicids_feature_names,
        )
        probability = float(self.cicids_model.predict_proba(scaled)[0][1])
        prediction = 1 if probability >= self.threshold else 0
        return probability, prediction

    def _predict_nslkdd(self, feature_dict: dict) -> tuple[float, int]:
        """Run the NSL-KDD branch of the ensemble."""
        source_frame = self._feature_frame(feature_dict)
        aligned = align_features(
            X_source=source_frame,
            source_feature_names=source_frame.columns.tolist(),
            target_feature_names=self.nslkdd_feature_names,
        )
        scaled = pd.DataFrame(
            self.nslkdd_scaler.transform(aligned),
            columns=self.nslkdd_feature_names,
        )
        probability = float(self.nslkdd_model.predict_proba(scaled)[0][1])
        prediction = 1 if probability >= 0.5 else 0
        return probability, prediction

    def predict(self, feature_dict: dict) -> dict:
        """Run weighted ensemble inference and fall back to CICIDS-only mode if needed."""
        try:
            cicids_proba, cicids_prediction = self._predict_cicids(feature_dict)
        except Exception as exc:
            raise RuntimeError(f"CICIDS ensemble branch failed: {exc}") from exc

        try:
            nslkdd_proba, nslkdd_prediction = self._predict_nslkdd(feature_dict)

            final_proba = (cicids_proba * self.CICIDS_WEIGHT) + (nslkdd_proba * self.NSLKDD_WEIGHT)
            prediction = 1 if final_proba >= self.threshold else 0
            confidence_attack = round(final_proba * 100, 2)
            confidence_normal = round((1 - final_proba) * 100, 2)

            return {
                "prediction": prediction,
                "label": "ATTACK" if prediction == 1 else "NORMAL",
                "confidence": confidence_attack if prediction == 1 else confidence_normal,
                "cicids_proba": round(cicids_proba, 4),
                "nslkdd_proba": round(nslkdd_proba, 4),
                "ensemble_proba": round(final_proba, 4),
                "models_agreed": bool(cicids_prediction == nslkdd_prediction),
            }
        except Exception as exc:
            fallback_prediction = 1 if cicids_proba >= self.threshold else 0
            fallback_attack = round(cicids_proba * 100, 2)
            fallback_normal = round((1 - cicids_proba) * 100, 2)

            return {
                "prediction": fallback_prediction,
                "label": "ATTACK" if fallback_prediction == 1 else "NORMAL",
                "confidence": fallback_attack if fallback_prediction == 1 else fallback_normal,
                "cicids_proba": round(cicids_proba, 4),
                "nslkdd_proba": 0.0,
                "ensemble_proba": round(cicids_proba, 4),
                "models_agreed": False,
                "fallback": True,
                "fallback_reason": str(exc),
            }

    def get_status(self) -> dict:
        """Return the ensemble readiness payload for the Flask API."""
        return {
            "models_loaded": 2,
            "cicids_weight": self.CICIDS_WEIGHT,
            "nslkdd_weight": self.NSLKDD_WEIGHT,
            "datasets": ["CICIDS2017", "NSL-KDD"],
            "status": "ready",
        }
