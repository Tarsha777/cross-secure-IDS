"""Train CROSS-SECURE intrusion detection models from CICIDS2017 CSV files."""

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Silence noisy library warnings to keep the terminal output clean.
warnings.filterwarnings("ignore")


# Define project-relative paths so the script runs from the project root in VS Code.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "CICIDS2017"
MODEL_DIR = BASE_DIR / "model"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MODEL_PATH = MODEL_DIR / "cross_secure_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"


# List identifier-style columns that should not be used as model features.
COLUMNS_TO_DROP = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Timestamp",
]


def print_section(title: str) -> None:
    """Print a visible section header for progress tracking."""
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def make_console_safe(text: str) -> str:
    """Convert text into a Windows-console-safe representation."""
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def print_value_counts(series: pd.Series, header: str) -> None:
    """Print value counts without crashing on Windows console encoding issues."""
    print(header)
    counts = series.astype(str).value_counts(dropna=False)
    for label, count in counts.items():
        safe_label = make_console_safe(label)
        print(f"{safe_label}: {count:,}")


def save_artifact(path: Path, artifact, label: str) -> float:
    """Save an artifact with joblib and print its file size in MB."""
    joblib.dump(artifact, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[SAVE] {label}: {path} ({size_mb:.2f} MB)")
    return size_mb


def load_all_csv_files(data_dir: Path) -> pd.DataFrame:
    """Load and merge all CSV files found inside the CICIDS2017 folder."""
    print_section("STEP 1 - LOAD DATA")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    print(f"[INFO] Found {len(csv_files)} CSV files in {data_dir}")

    dataframes = []
    loaded_files = 0

    for csv_file in csv_files:
        try:
            print(f"[LOAD] Reading file: {csv_file.name}")
            df = pd.read_csv(csv_file, low_memory=False)
            df.columns = df.columns.str.strip()
            dataframes.append(df)
            loaded_files += 1
            print(f"[LOAD] Loaded {len(df):,} rows from {csv_file.name}")
        except FileNotFoundError:
            print(f"[SKIP] File missing, skipping: {csv_file.name}")
        except Exception as exc:
            print(f"[SKIP] Could not load {csv_file.name}: {exc}")

    if not dataframes:
        raise ValueError("No CSV files could be loaded successfully.")

    print(f"[INFO] Successfully loaded {loaded_files}/{len(csv_files)} CSV files")
    print("[INFO] Merging all loaded files into one dataframe...")
    merged_df = pd.concat(dataframes, ignore_index=True)

    if "Label" not in merged_df.columns:
        raise KeyError("Required column 'Label' was not found after stripping column names.")

    print(f"[INFO] Total rows after merge: {len(merged_df):,}")
    print_value_counts(merged_df["Label"], "[INFO] Label distribution immediately after loading:")

    return merged_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean whitespace, duplicates, infinities, and missing values."""
    print_section("STEP 2 - CLEAN DATA")

    rows_before = len(df)

    df["Label"] = df["Label"].astype(str).str.strip()
    after_label_strip = len(df)

    df = df.drop_duplicates()
    after_dedup = len(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    rows_after = len(df)

    duplicate_rows_removed = after_label_strip - after_dedup
    nan_rows_removed = after_dedup - rows_after
    total_removed = rows_before - rows_after

    print(f"[CLEAN] Duplicate rows removed: {duplicate_rows_removed:,}")
    print(f"[CLEAN] Rows removed due to NaN/inf cleanup: {nan_rows_removed:,}")
    print(f"[CLEAN] Total rows removed: {total_removed:,}")
    print(f"[CLEAN] Rows remaining after cleaning: {rows_after:,}")

    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert BENIGN to 0 and all other labels to 1."""
    print_section("STEP 3 - ENCODE LABELS")

    y = np.where(df["Label"].eq("BENIGN"), 0, 1)
    y = pd.Series(y, name="target")

    class_counts = y.value_counts().sort_index()
    print("[LABEL] Encoded class distribution:")
    print(f"0 (BENIGN): {class_counts.get(0, 0):,}")
    print(f"1 (ATTACK): {class_counts.get(1, 0):,}")

    return df, y


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop unused columns and keep only numeric model features."""
    print_section("STEP 4 - FEATURE SELECTION")

    columns_present = [column for column in COLUMNS_TO_DROP if column in df.columns]
    if columns_present:
        print(f"[FEATURES] Dropping identifier columns: {columns_present}")
        df = df.drop(columns=columns_present)
    else:
        print("[FEATURES] No identifier columns from the drop list were present.")

    X = df.drop(columns=["Label"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    feature_names = X.columns.tolist()

    print(f"[FEATURES] Numeric feature count: {len(feature_names)}")
    print("[FEATURES] Feature names:")
    print(feature_names)

    return X, feature_names


def normalize_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler on all features, save it, and return scaled data."""
    print_section("STEP 5 - NORMALIZE")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X = X.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    save_artifact(SCALER_PATH, scaler, "Scaler")
    print(f"[NORMALIZE] Scaled feature matrix shape: {X_scaled.shape}")

    return X_scaled, scaler


def split_data(X: np.ndarray, y: pd.Series):
    """Create an 80/20 train-test split with stratification."""
    print_section("STEP 6 - SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print(f"[SPLIT] X_train shape: {X_train.shape}")
    print(f"[SPLIT] X_test shape: {X_test.shape}")
    print(f"[SPLIT] y_train size: {len(y_train):,}")
    print(f"[SPLIT] y_test size: {len(y_test):,}")

    return X_train, X_test, y_train, y_test


def build_models() -> dict[str, object]:
    """Create the three candidate models using the requested hyperparameters."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            eval_metric="logloss",
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),
    }


def train_and_compare_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, dict]:
    """Train all models, print metrics, and return their results."""
    print_section("STEP 7 - TRAIN 3 MODELS AND COMPARE")

    models = build_models()
    results = {}

    for model_name, model in models.items():
        print(f"\n[MODEL] Training {model_name}...")
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
        )

        results[model_name] = {
            "model": model,
            "accuracy": accuracy,
            "training_time": training_time,
            "report": report,
        }

        print(f"[MODEL] {model_name} training time: {training_time:.2f} seconds")
        print(f"[MODEL] {model_name} accuracy: {accuracy:.4f}")
        print(f"[MODEL] {model_name} classification report:")
        print(report)

    return results


def pick_best_model(results: dict[str, dict]) -> tuple[str, object, float]:
    """Select the highest-accuracy model."""
    print_section("STEP 8 - PICK BEST MODEL")

    best_model_name = max(results, key=lambda name: results[name]["accuracy"])
    best_model = results[best_model_name]["model"]
    best_accuracy = results[best_model_name]["accuracy"]

    print(f"[WINNER] Best model: {best_model_name}")
    print(f"[WINNER] Best accuracy: {best_accuracy:.4f}")

    return best_model_name, best_model, best_accuracy


def tune_threshold(best_model, X_test: np.ndarray, y_test: pd.Series) -> float:
    """Find a probability threshold that keeps attack recall at or above 0.95."""
    print_section("STEP 9 - THRESHOLD TUNING")

    probabilities = best_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probabilities)

    optimal_threshold = 0.5

    if len(thresholds) > 0:
        valid_indices = np.where(recalls[:-1] >= 0.95)[0]
        if len(valid_indices) > 0:
            best_index = valid_indices[np.argmax(precisions[:-1][valid_indices])]
            optimal_threshold = float(thresholds[best_index])

    print(f"[THRESHOLD] Optimal threshold: {optimal_threshold:.4f}")
    return optimal_threshold


def save_outputs(best_model_name: str, best_model, feature_names: list[str], threshold: float) -> float:
    """Save the best model and related artifacts, then print their sizes."""
    print_section("STEP 10 - SAVE FILES")

    model_size_mb = save_artifact(MODEL_PATH, best_model, f"{best_model_name} model")
    save_artifact(FEATURE_NAMES_PATH, feature_names, "Feature names")
    save_artifact(THRESHOLD_PATH, threshold, "Threshold")

    scaler_size_mb = SCALER_PATH.stat().st_size / (1024 * 1024)
    print(f"[SAVE] Existing scaler: {SCALER_PATH} ({scaler_size_mb:.2f} MB)")

    if model_size_mb < 10:
        print("[WARNING] cross_secure_model.pkl is under 10 MB.")

    return model_size_mb


def print_summary(
    total_samples: int,
    feature_count: int,
    best_model_name: str,
    best_accuracy: float,
    threshold: float,
    model_size_mb: float,
) -> None:
    """Print a clean final summary table."""
    print_section("STEP 11 - SUMMARY TABLE")

    summary_rows = [
        ("Total samples", f"{total_samples:,}"),
        ("Features used", str(feature_count)),
        ("Best model", best_model_name),
        ("Accuracy", f"{best_accuracy * 100:.2f}%"),
        ("Optimal threshold", f"{threshold:.4f}"),
        ("Model file size", f"{model_size_mb:.2f} MB"),
    ]

    key_width = max(len(key) for key, _ in summary_rows)
    value_width = max(len(value) for _, value in summary_rows)
    border = f"+-{'-' * key_width}-+-{'-' * value_width}-+"

    print(border)
    for key, value in summary_rows:
        print(f"| {key.ljust(key_width)} | {value.ljust(value_width)} |")
    print(border)


def main() -> None:
    """Run the full CROSS-SECURE model training pipeline."""
    print_section("CROSS-SECURE MODEL TRAINING")

    df = load_all_csv_files(DATA_DIR)
    df = clean_data(df)
    total_samples = len(df)

    df, y = encode_labels(df)
    X, feature_names = select_features(df)
    X_scaled, _ = normalize_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    results = train_and_compare_models(X_train, X_test, y_train, y_test)
    best_model_name, best_model, best_accuracy = pick_best_model(results)
    optimal_threshold = tune_threshold(best_model, X_test, y_test)
    model_size_mb = save_outputs(best_model_name, best_model, feature_names, optimal_threshold)
    print_summary(
        total_samples=total_samples,
        feature_count=len(feature_names),
        best_model_name=best_model_name,
        best_accuracy=best_accuracy,
        threshold=optimal_threshold,
        model_size_mb=model_size_mb,
    )


if __name__ == "__main__":
    main()
