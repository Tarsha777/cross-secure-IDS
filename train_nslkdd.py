"""Train the NSL-KDD model and run CROSS-SECURE cross-domain evaluation."""

# Import standard library modules for timing, file handling, and string matching.
import difflib
import time
import warnings
from pathlib import Path

# Import third-party libraries already used in the project.
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Silence noisy warnings so progress messages stay readable in the terminal.
warnings.filterwarnings("ignore")


# Resolve all project-relative paths from the repository root.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
NSL_DIR = DATA_DIR / "NSLKDD"
CICIDS_DIR = DATA_DIR / "CICIDS2017"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

NSL_MODEL_PATH = MODEL_DIR / "nslkdd_model.pkl"
NSL_SCALER_PATH = MODEL_DIR / "nslkdd_scaler.pkl"
NSL_FEATURES_PATH = MODEL_DIR / "nslkdd_feature_names.pkl"

CICIDS_MODEL_PATH = MODEL_DIR / "cross_secure_model.pkl"
CICIDS_SCALER_PATH = MODEL_DIR / "scaler.pkl"
CICIDS_FEATURES_PATH = MODEL_DIR / "feature_names.pkl"
CICIDS_THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"

REPORT_PATH = RESULTS_DIR / "cross_domain_report.txt"


# Define the exact NSL-KDD column order because the CSV files do not contain headers.
NSL_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]


# Track the categorical NSL-KDD columns that need LabelEncoder transformation.
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]


# Reuse the CICIDS training script's identifier drop list for sampled CICIDS evaluation.
CICIDS_COLUMNS_TO_DROP = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Timestamp",
]


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


def print_section(title: str) -> None:
    """Print a visible training section header."""
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")


def log(report_lines: list[str], message: str = "") -> None:
    """Print a message and mirror it into the saved report content."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))
    report_lines.append(message)


def save_artifact(path: Path, artifact, label: str, report_lines: list[str]) -> None:
    """Persist an artifact to disk and print a short confirmation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    log(report_lines, f"[SAVE] {label}: {path}")


def normalize_name(name: str) -> str:
    """Normalize a feature name to improve exact and fuzzy matching."""
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in str(name))
    tokens = [token for token in cleaned.split() if token]
    return " ".join(sorted(tokens))


def print_label_distribution(df: pd.DataFrame, title: str, report_lines: list[str]) -> None:
    """Print the label distribution for an NSL-KDD dataframe."""
    log(report_lines, title)
    for label, count in df["label"].astype(str).str.strip().value_counts(dropna=False).items():
        log(report_lines, f"  {label}: {count:,}")


def print_binary_counts(y: pd.Series, title: str, report_lines: list[str]) -> None:
    """Print NORMAL vs ATTACK counts for encoded labels."""
    normal_count = int((y == 0).sum())
    attack_count = int((y == 1).sum())
    log(report_lines, title)
    log(report_lines, f"  NORMAL (0): {normal_count:,}")
    log(report_lines, f"  ATTACK (1): {attack_count:,}")


def load_nslkdd_data(report_lines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official NSL-KDD train and test CSV files with manual headers."""
    print_section("STEP 1 - LOAD NSL-KDD")

    try:
        train_path = NSL_DIR / "KDDTrain+.csv"
        test_path = NSL_DIR / "KDDTest+.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"NSL-KDD training file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"NSL-KDD test file not found: {test_path}")

        train_df = pd.read_csv(train_path, header=None, names=NSL_COLUMNS)
        test_df = pd.read_csv(test_path, header=None, names=NSL_COLUMNS)

        train_df = train_df.drop(columns=["difficulty"])
        test_df = test_df.drop(columns=["difficulty"])

        log(report_lines, f"[LOAD] Train shape after dropping difficulty: {train_df.shape}")
        print_label_distribution(train_df, "[LOAD] Train label distribution:", report_lines)

        log(report_lines, f"[LOAD] Test shape after dropping difficulty: {test_df.shape}")
        print_label_distribution(test_df, "[LOAD] Test label distribution:", report_lines)

        return train_df, test_df
    except Exception as exc:
        raise RuntimeError(f"Failed to load NSL-KDD data: {exc}") from exc


def clean_and_encode_nslkdd(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report_lines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean NSL-KDD data, encode labels, and label-encode categorical columns."""
    print_section("STEP 2 - CLEAN AND ENCODE")

    try:
        combined = pd.concat(
            [
                train_df.copy().assign(_split="train"),
                test_df.copy().assign(_split="test"),
            ],
            ignore_index=True,
        )

        combined["label"] = combined["label"].astype(str).str.strip()
        combined["label"] = np.where(combined["label"].eq("normal"), 0, 1)

        for column in CATEGORICAL_COLUMNS:
            encoder = LabelEncoder()
            combined[column] = encoder.fit_transform(combined[column].astype(str).str.strip())
            log(report_lines, f"[ENCODE] LabelEncoder applied to: {column}")

        rows_before = len(combined)
        combined = combined.replace([np.inf, -np.inf], np.nan)
        combined = combined.dropna().reset_index(drop=True)
        rows_after = len(combined)

        log(report_lines, f"[CLEAN] Rows removed after inf/NaN cleanup: {rows_before - rows_after:,}")
        print_binary_counts(combined["label"], "[CLEAN] Combined NSL-KDD class counts:", report_lines)

        cleaned_train = combined.loc[combined["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
        cleaned_test = combined.loc[combined["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)
        cleaned_combined = combined.drop(columns=["_split"]).reset_index(drop=True)

        log(report_lines, f"[CLEAN] Cleaned train shape: {cleaned_train.shape}")
        log(report_lines, f"[CLEAN] Cleaned test shape: {cleaned_test.shape}")

        return cleaned_train, cleaned_test, cleaned_combined
    except Exception as exc:
        raise RuntimeError(f"Failed to clean and encode NSL-KDD data: {exc}") from exc


def prepare_nsl_features(
    combined_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report_lines: list[str],
) -> dict[str, object]:
    """Prepare NSL-KDD features, save artifacts, and create the 80/20 split."""
    print_section("STEP 3 - FEATURE SELECTION AND NORMALIZE")

    try:
        X_all = combined_df.drop(columns=["label"]).select_dtypes(include=[np.number]).copy()
        y_all = combined_df["label"].astype(int).copy()
        feature_names = X_all.columns.tolist()

        log(report_lines, f"[FEATURES] Numeric NSL-KDD feature count: {len(feature_names)}")
        save_artifact(NSL_FEATURES_PATH, feature_names, "NSL-KDD feature names", report_lines)

        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X_all).astype(np.float32)
        save_artifact(NSL_SCALER_PATH, scaler, "NSL-KDD scaler", report_lines)

        (
            X_train_raw,
            X_test_raw,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
        ) = train_test_split(
            X_all,
            X_all_scaled,
            y_all,
            test_size=0.2,
            stratify=y_all,
            random_state=42,
        )

        official_test_features = (
            test_df.drop(columns=["label"])
            .select_dtypes(include=[np.number])
            .reindex(columns=feature_names, fill_value=0.0)
            .copy()
        )
        official_test_labels = test_df["label"].astype(int).copy()

        log(report_lines, f"[SPLIT] X_train shape: {X_train_scaled.shape}")
        log(report_lines, f"[SPLIT] X_test shape: {X_test_scaled.shape}")
        print_binary_counts(y_test, "[SPLIT] Held-out 80/20 NSL-KDD test class counts:", report_lines)

        return {
            "X_train_raw": X_train_raw.reset_index(drop=True),
            "X_test_raw": X_test_raw.reset_index(drop=True),
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
            "feature_names": feature_names,
            "scaler": scaler,
            "official_test_features": official_test_features.reset_index(drop=True),
            "official_test_labels": official_test_labels.reset_index(drop=True),
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to prepare NSL-KDD features: {exc}") from exc


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, label: str) -> dict[str, object]:
    """Build a consistent metrics dictionary for console output and the report."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NORMAL", "ATTACK"],
        zero_division=0,
    )
    _, _, f1_scores, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "label": label,
        "accuracy": float(accuracy),
        "attack_f1": float(f1_scores[1]),
        "report": report,
    }


def print_metrics(metrics: dict[str, object], report_lines: list[str]) -> None:
    """Print an evaluation block to both the console and the saved report."""
    log(report_lines, f"[RESULT] {metrics['label']} accuracy: {metrics['accuracy'] * 100:.2f}%")
    log(report_lines, f"[RESULT] {metrics['label']} F1-Attack: {metrics['attack_f1']:.4f}")
    log(report_lines, f"[RESULT] {metrics['label']} classification report:")
    for line in str(metrics["report"]).splitlines():
        log(report_lines, line)


def train_nsl_model(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    report_lines: list[str],
) -> tuple[LGBMClassifier, dict[str, object], float]:
    """Train the dedicated NSL-KDD LightGBM model and evaluate it in-domain."""
    print_section("STEP 4 - TRAIN LIGHTGBM ON NSL-KDD")

    try:
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        start_time = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        training_time = time.perf_counter() - start_time

        y_pred = model.predict(X_test_scaled)
        metrics = compute_metrics(y_test, y_pred, "NSL-KDD in-domain")

        log(report_lines, f"[TRAIN] NSL-KDD LightGBM training time: {training_time:.2f} seconds")
        print_metrics(metrics, report_lines)
        save_artifact(NSL_MODEL_PATH, model, "NSL-KDD model", report_lines)

        return model, metrics, training_time
    except Exception as exc:
        raise RuntimeError(f"Failed to train the NSL-KDD model: {exc}") from exc


def resolve_target_name(source_name: str, target_feature_names: list[str]) -> str | None:
    """Resolve a source feature to the closest compatible target feature."""
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
    """Align source features into the target feature space using exact and fuzzy matches."""
    aligned = pd.DataFrame(0.0, index=X_source.index, columns=target_feature_names)

    for source_name in source_feature_names:
        if source_name not in X_source.columns:
            continue

        target_name = resolve_target_name(source_name, target_feature_names)
        if target_name is None:
            continue

        aligned[target_name] = pd.to_numeric(X_source[source_name], errors="coerce").fillna(0.0)

    return aligned


def load_cicids_artifacts(report_lines: list[str]) -> tuple[object, StandardScaler, list[str], float]:
    """Load the already-trained CICIDS2017 artifacts used for cross-domain testing."""
    try:
        cicids_model = joblib.load(CICIDS_MODEL_PATH)
        cicids_scaler = joblib.load(CICIDS_SCALER_PATH)
        cicids_feature_names = list(joblib.load(CICIDS_FEATURES_PATH))
        cicids_threshold = float(joblib.load(CICIDS_THRESHOLD_PATH))

        log(report_lines, f"[LOAD] CICIDS model loaded from: {CICIDS_MODEL_PATH}")
        log(report_lines, f"[LOAD] CICIDS scaler loaded from: {CICIDS_SCALER_PATH}")
        log(report_lines, f"[LOAD] CICIDS feature count: {len(cicids_feature_names)}")
        log(report_lines, f"[LOAD] CICIDS threshold: {cicids_threshold:.4f}")

        return cicids_model, cicids_scaler, cicids_feature_names, cicids_threshold
    except Exception as exc:
        raise RuntimeError(f"Failed to load CICIDS artifacts: {exc}") from exc


def load_cicids_sample(report_lines: list[str]) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load the first 50,000 rows from a CICIDS CSV that contains usable labels."""
    try:
        csv_files = sorted(CICIDS_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CICIDS2017 CSV files found in: {CICIDS_DIR}")

        selected_df = None
        selected_file = None

        for csv_file in csv_files:
            candidate = pd.read_csv(csv_file, nrows=50000, low_memory=False)
            candidate.columns = candidate.columns.str.strip()

            if "Label" not in candidate.columns:
                continue

            labels = candidate["Label"].astype(str).str.strip()
            if labels.nunique() > 1:
                selected_df = candidate
                selected_file = csv_file.name
                break

            if selected_df is None:
                selected_df = candidate
                selected_file = csv_file.name

        if selected_df is None:
            raise RuntimeError("Could not find a readable CICIDS2017 CSV with a Label column.")

        selected_df = selected_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        selected_df["Label"] = selected_df["Label"].astype(str).str.strip()

        y = pd.Series(np.where(selected_df["Label"].eq("BENIGN"), 0, 1), name="label")
        X = selected_df.drop(columns=["Label"], errors="ignore")

        columns_present = [column for column in CICIDS_COLUMNS_TO_DROP if column in X.columns]
        if columns_present:
            X = X.drop(columns=columns_present)

        X = X.select_dtypes(include=[np.number]).reset_index(drop=True)

        log(report_lines, f"[LOAD] CICIDS sample source file: {selected_file}")
        log(report_lines, f"[LOAD] CICIDS sample shape after cleanup: {X.shape}")
        print_binary_counts(y, "[LOAD] CICIDS sample class counts:", report_lines)

        return X, y.reset_index(drop=True), selected_file
    except Exception as exc:
        raise RuntimeError(f"Failed to load CICIDS sample data: {exc}") from exc


def evaluate_with_proba_threshold(
    model,
    X_scaled,
    y_true: pd.Series,
    label: str,
    threshold: float,
) -> dict[str, object]:
    """Evaluate a model using predict_proba and a configurable decision threshold."""
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return compute_metrics(y_true, predictions, label)


def run_cross_domain_evaluation(
    nsl_model,
    nsl_scaler: StandardScaler,
    nsl_feature_names: list[str],
    nsl_test_features: pd.DataFrame,
    nsl_test_labels: pd.Series,
    report_lines: list[str],
) -> dict[str, dict[str, object]]:
    """Run the requested cross-domain checks in both directions."""
    print_section("STEP 5 - CROSS DOMAIN EVALUATION")

    try:
        cicids_model, cicids_scaler, cicids_feature_names, cicids_threshold = load_cicids_artifacts(report_lines)

        log(report_lines, "[TEST A] CICIDS model on NSL-KDD test data")
        aligned_nsl_to_cicids = align_features(
            X_source=nsl_test_features,
            source_feature_names=nsl_feature_names,
            target_feature_names=cicids_feature_names,
        )
        scaled_nsl_for_cicids = cicids_scaler.transform(aligned_nsl_to_cicids)
        cicids_on_nsl = evaluate_with_proba_threshold(
            cicids_model,
            scaled_nsl_for_cicids,
            nsl_test_labels,
            "CICIDS→NSL (cross-domain)",
            cicids_threshold,
        )
        print_metrics(cicids_on_nsl, report_lines)

        log(report_lines, "[TEST B] NSL-KDD model on CICIDS2017 sample data")
        cicids_sample_X, cicids_sample_y, sample_file = load_cicids_sample(report_lines)
        aligned_cicids_to_nsl = align_features(
            X_source=cicids_sample_X,
            source_feature_names=cicids_sample_X.columns.tolist(),
            target_feature_names=nsl_feature_names,
        )
        scaled_cicids_for_nsl = nsl_scaler.transform(aligned_cicids_to_nsl)
        nsl_on_cicids = evaluate_with_proba_threshold(
            nsl_model,
            scaled_cicids_for_nsl,
            cicids_sample_y,
            "NSL→CICIDS (cross-domain)",
            0.5,
        )
        print_metrics(nsl_on_cicids, report_lines)

        log(report_lines, "[TEST C] CICIDS in-domain baseline (sample sanity check for F1 only)")
        aligned_cicids_sample = align_features(
            X_source=cicids_sample_X,
            source_feature_names=cicids_sample_X.columns.tolist(),
            target_feature_names=cicids_feature_names,
        )
        scaled_cicids_sample = cicids_scaler.transform(aligned_cicids_sample)
        cicids_sample_metrics = evaluate_with_proba_threshold(
            cicids_model,
            scaled_cicids_sample,
            cicids_sample_y,
            f"CICIDS sample check ({sample_file})",
            cicids_threshold,
        )
        print_metrics(cicids_sample_metrics, report_lines)

        return {
            "cicids_on_nsl": cicids_on_nsl,
            "nsl_on_cicids": nsl_on_cicids,
            "cicids_sample_metrics": cicids_sample_metrics,
        }
    except Exception as exc:
        raise RuntimeError(f"Failed during cross-domain evaluation: {exc}") from exc


def build_comparison_table(
    nsl_in_domain: dict[str, object],
    cicids_cross: dict[str, object],
    nsl_cross: dict[str, object],
    cicids_sample_metrics: dict[str, object],
) -> tuple[str, float]:
    """Create the requested summary table and compute the domain adaptation gap."""
    cicids_in_domain_accuracy = 99.86
    cicids_in_domain_f1 = cicids_sample_metrics["attack_f1"]
    nsl_in_domain_accuracy = nsl_in_domain["accuracy"] * 100
    nsl_in_domain_f1 = nsl_in_domain["attack_f1"]
    cicids_cross_accuracy = cicids_cross["accuracy"] * 100
    cicids_cross_f1 = cicids_cross["attack_f1"]
    nsl_cross_accuracy = nsl_cross["accuracy"] * 100
    nsl_cross_f1 = nsl_cross["attack_f1"]

    best_in_domain = max(cicids_in_domain_accuracy, nsl_in_domain_accuracy)
    best_cross_domain = max(cicids_cross_accuracy, nsl_cross_accuracy)
    domain_gap = best_in_domain - best_cross_domain

    table = "\n".join(
        [
            "╔══════════════════════════════════════════════════════════════╗",
            "║         CROSS-SECURE: CROSS-DOMAIN EVALUATION               ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ Test Scenario          │ Accuracy │ F1-Attack │ Type       ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ CICIDS → CICIDS        │  {cicids_in_domain_accuracy:05.2f}% │   {cicids_in_domain_f1:.2f}    │ ✅ In      ║",
            f"║ NSL-KDD → NSL-KDD      │  {nsl_in_domain_accuracy:05.2f}% │   {nsl_in_domain_f1:.2f}    │ ✅ In      ║",
            f"║ CICIDS → NSL-KDD       │  {cicids_cross_accuracy:05.2f}% │   {cicids_cross_f1:.2f}    │ 🔀 Cross   ║",
            f"║ NSL-KDD → CICIDS       │  {nsl_cross_accuracy:05.2f}% │   {nsl_cross_f1:.2f}    │ 🔀 Cross   ║",
            "╚══════════════════════════════════════════════════════════════╝",
            f"Domain Adaptation Gap: {domain_gap:.2f}%",
        ]
    )

    return table, domain_gap


def save_report(report_lines: list[str]) -> None:
    """Persist the assembled report to results/cross_domain_report.txt."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report saved to {REPORT_PATH.as_posix()}")


def main() -> None:
    """Run the full NSL-KDD training and cross-domain evaluation workflow."""
    report_lines: list[str] = []

    try:
        print_section("CROSS-SECURE NSL-KDD TRAINING")
        log(report_lines, "CROSS-SECURE NSL-KDD TRAINING")

        train_df, test_df = load_nslkdd_data(report_lines)
        cleaned_train, cleaned_test, cleaned_combined = clean_and_encode_nslkdd(train_df, test_df, report_lines)
        feature_bundle = prepare_nsl_features(cleaned_combined, cleaned_test, report_lines)

        nsl_model, nsl_in_domain_metrics, training_time = train_nsl_model(
            feature_bundle["X_train_scaled"],
            feature_bundle["X_test_scaled"],
            feature_bundle["y_train"],
            feature_bundle["y_test"],
            report_lines,
        )

        cross_domain_results = run_cross_domain_evaluation(
            nsl_model=nsl_model,
            nsl_scaler=feature_bundle["scaler"],
            nsl_feature_names=feature_bundle["feature_names"],
            nsl_test_features=feature_bundle["official_test_features"],
            nsl_test_labels=feature_bundle["official_test_labels"],
            report_lines=report_lines,
        )

        print_section("STEP 6 - PRINT COMPARISON TABLE")
        comparison_table, domain_gap = build_comparison_table(
            nsl_in_domain=nsl_in_domain_metrics,
            cicids_cross=cross_domain_results["cicids_on_nsl"],
            nsl_cross=cross_domain_results["nsl_on_cicids"],
            cicids_sample_metrics=cross_domain_results["cicids_sample_metrics"],
        )
        log(report_lines, comparison_table)
        log(report_lines, f"[SUMMARY] NSL-KDD training time: {training_time:.2f} seconds")
        log(report_lines, f"[SUMMARY] Domain adaptation gap: {domain_gap:.2f}%")

        print_section("STEP 7 - SAVE REPORT")
        save_report(report_lines)
    except Exception as exc:
        error_message = f"[FATAL] train_nslkdd.py failed: {exc}"
        print(error_message)
        report_lines.append(error_message)
        try:
            save_report(report_lines)
        except Exception as save_exc:
            print(f"[FATAL] Could not save failure report: {save_exc}")


if __name__ == "__main__":
    main()
