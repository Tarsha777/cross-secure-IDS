"""Evaluate CROSS-SECURE models on the UNSW-NB15 dataset."""

# Import standard library modules for file handling and warning control.
import warnings
from pathlib import Path

# Import third-party libraries already available in the project.
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


# Silence noisy warnings so the evaluation output stays readable.
warnings.filterwarnings("ignore")


# Resolve all paths relative to the project root.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UNSW_DIR = DATA_DIR / "UNSW-NB15"
NSL_DIR = DATA_DIR / "NSLKDD"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

CICIDS_MODEL_PATH = MODEL_DIR / "cross_secure_model.pkl"
CICIDS_SCALER_PATH = MODEL_DIR / "scaler.pkl"
CICIDS_FEATURES_PATH = MODEL_DIR / "feature_names.pkl"
CICIDS_THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"

NSL_MODEL_PATH = MODEL_DIR / "nslkdd_model.pkl"
NSL_SCALER_PATH = MODEL_DIR / "nslkdd_scaler.pkl"
NSL_FEATURES_PATH = MODEL_DIR / "nslkdd_feature_names.pkl"

REPORT_PATH = RESULTS_DIR / "unsw_nb15_report.txt"


# Define the NSL-KDD header again so we can build reference encoders for shared categories.
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


def print_section(title: str) -> None:
    """Print a visible console section header."""
    print(f"\n{'=' * 88}")
    print(title)
    print(f"{'=' * 88}")


def log(report_lines: list[str], message: str = "") -> None:
    """Print a message safely and also store it for the saved report."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))
    report_lines.append(message)


def safe_divide(numerator, denominator) -> np.ndarray:
    """Divide arrays safely and return zeros where the denominator is zero."""
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator_array,
        denominator_array,
        out=np.zeros_like(numerator_array, dtype=np.float64),
        where=denominator_array != 0,
    )


def clip_rate(values) -> np.ndarray:
    """Clip a numeric array into the range expected by KDD-style rate fields."""
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def compute_metrics(y_true: pd.Series, predictions: np.ndarray, label: str) -> dict[str, object]:
    """Return accuracy, attack F1, and a text classification report."""
    accuracy = accuracy_score(y_true, predictions)
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["NORMAL", "ATTACK"],
        zero_division=0,
    )
    _, _, f1_scores, _ = precision_recall_fscore_support(
        y_true,
        predictions,
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
    """Print one evaluation block to console and report output."""
    log(report_lines, f"[RESULT] {metrics['label']} accuracy: {metrics['accuracy'] * 100:.2f}%")
    log(report_lines, f"[RESULT] {metrics['label']} F1-Attack: {metrics['attack_f1']:.4f}")
    log(report_lines, f"[RESULT] {metrics['label']} classification report:")
    for line in str(metrics["report"]).splitlines():
        log(report_lines, line)


def load_unsw_data(report_lines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official UNSW-NB15 train and test CSV files."""
    print_section("STEP 1 - LOAD UNSW-NB15")

    try:
        train_path = UNSW_DIR / "UNSW_NB15_training-set.csv"
        test_path = UNSW_DIR / "UNSW_NB15_testing-set.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"UNSW-NB15 training file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"UNSW-NB15 testing file not found: {test_path}")

        train_df = pd.read_csv(train_path, low_memory=False)
        test_df = pd.read_csv(test_path, low_memory=False)

        train_df.columns = train_df.columns.str.strip()
        test_df.columns = test_df.columns.str.strip()

        log(report_lines, f"[LOAD] UNSW train shape: {train_df.shape}")
        log(report_lines, f"[LOAD] UNSW test shape: {test_df.shape}")

        log(report_lines, "[LOAD] UNSW test attack category distribution:")
        for attack_name, count in test_df["attack_cat"].astype(str).str.strip().value_counts().items():
            log(report_lines, f"  {attack_name}: {count:,}")

        return train_df, test_df
    except Exception as exc:
        raise RuntimeError(f"Failed to load UNSW-NB15 data: {exc}") from exc


def clean_unsw_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report_lines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean UNSW-NB15 rows and normalize key text columns."""
    print_section("STEP 2 - CLEAN UNSW-NB15")

    try:
        combined = pd.concat(
            [
                train_df.copy().assign(_split="train"),
                test_df.copy().assign(_split="test"),
            ],
            ignore_index=True,
        )

        for column in ["proto", "service", "state", "attack_cat"]:
            if column in combined.columns:
                combined[column] = combined[column].astype(str).str.strip()

        combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(0).astype(int)
        rows_before = len(combined)
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        rows_after = len(combined)

        log(report_lines, f"[CLEAN] Rows removed after NaN/inf cleanup: {rows_before - rows_after:,}")
        log(report_lines, "[CLEAN] Binary class counts across both UNSW files:")
        log(report_lines, f"  NORMAL (0): {int((combined['label'] == 0).sum()):,}")
        log(report_lines, f"  ATTACK (1): {int((combined['label'] == 1).sum()):,}")

        cleaned_train = combined.loc[combined["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
        cleaned_test = combined.loc[combined["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)

        log(report_lines, f"[CLEAN] Cleaned train shape: {cleaned_train.shape}")
        log(report_lines, f"[CLEAN] Cleaned test shape: {cleaned_test.shape}")

        return cleaned_train, cleaned_test
    except Exception as exc:
        raise RuntimeError(f"Failed to clean UNSW-NB15 data: {exc}") from exc


def build_nsl_reference_maps() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Load NSL-KDD categories so shared UNSW categories can be encoded consistently."""
    nsl_reference = pd.concat(
        [
            pd.read_csv(NSL_DIR / "KDDTrain+.csv", header=None, names=NSL_COLUMNS, usecols=["protocol_type", "service", "flag"]),
            pd.read_csv(NSL_DIR / "KDDTest+.csv", header=None, names=NSL_COLUMNS, usecols=["protocol_type", "service", "flag"]),
        ],
        ignore_index=True,
    )

    protocol_map = {value: index for index, value in enumerate(sorted(nsl_reference["protocol_type"].astype(str).str.strip().unique()))}
    service_map = {value: index for index, value in enumerate(sorted(nsl_reference["service"].astype(str).str.strip().unique()))}
    flag_map = {value: index for index, value in enumerate(sorted(nsl_reference["flag"].astype(str).str.strip().unique()))}

    return protocol_map, service_map, flag_map


def encode_nsl_category(series: pd.Series, mapping: dict[str, int], fallback_text: str | None = None) -> pd.Series:
    """Map text categories into NSL-compatible integer values with -1 for unknowns."""
    cleaned = series.astype(str).str.strip().str.lower()
    lookup = {str(key).strip().lower(): value for key, value in mapping.items()}

    if fallback_text is not None:
        fallback_value = lookup.get(fallback_text.lower(), -1)
    else:
        fallback_value = -1

    return cleaned.map(lookup).fillna(fallback_value).astype(float)


def build_unsw_cicids_view(unsw_df: pd.DataFrame, cicids_feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Create a CICIDS-style feature frame from UNSW-NB15 columns and simple derived fields."""
    numeric = unsw_df.copy()
    for column in numeric.columns:
        if column not in {"proto", "service", "state", "attack_cat"}:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    dur_seconds = numeric["dur"].astype(float)
    flow_duration = dur_seconds * 1_000_000.0
    total_packets = numeric["spkts"] + numeric["dpkts"]
    total_bytes = numeric["sbytes"] + numeric["dbytes"]

    smean = pd.to_numeric(numeric["smean"], errors="coerce").fillna(0.0)
    dmean = pd.to_numeric(numeric["dmean"], errors="coerce").fillna(0.0)
    packet_mean = pd.Series(safe_divide(total_bytes, total_packets), index=numeric.index)
    packet_std = pd.Series(np.std(np.vstack([smean.to_numpy(), dmean.to_numpy()]), axis=0), index=numeric.index)
    packet_max = pd.Series(np.maximum(smean.to_numpy(), dmean.to_numpy()), index=numeric.index)
    packet_min = pd.Series(np.where(numeric["dpkts"] > 0, np.minimum(smean.to_numpy(), dmean.to_numpy()), smean.to_numpy()), index=numeric.index)

    flow_iat_mean = pd.Series(safe_divide(flow_duration, np.maximum(total_packets - 1, 1)), index=numeric.index)
    flow_iat_std = ((numeric["sjit"] + numeric["djit"]) / 2.0) * 1000.0
    fwd_iat_total = np.maximum(numeric["spkts"] - 1, 0) * numeric["sinpkt"] * 1000.0
    bwd_iat_total = np.maximum(numeric["dpkts"] - 1, 0) * numeric["dinpkt"] * 1000.0

    header_unit = np.where(numeric["proto"].astype(str).str.lower().eq("tcp"), 20.0, np.where(numeric["proto"].astype(str).str.lower().eq("udp"), 8.0, 0.0))
    fin_count = numeric["state"].astype(str).str.upper().eq("FIN").astype(float)
    rst_count = numeric["state"].astype(str).str.upper().eq("RST").astype(float)
    syn_count = (numeric["synack"] > 0).astype(float)
    ack_count = ((numeric["ackdat"] > 0) | (numeric["tcprtt"] > 0)).astype(float)

    aligned = pd.DataFrame(0.0, index=unsw_df.index, columns=cicids_feature_names)

    explicit_values = {
        "Flow Duration": flow_duration,
        "Total Fwd Packets": numeric["spkts"],
        "Total Backward Packets": numeric["dpkts"],
        "Total Length of Fwd Packets": numeric["sbytes"],
        "Total Length of Bwd Packets": numeric["dbytes"],
        "Fwd Packet Length Max": smean,
        "Fwd Packet Length Min": smean,
        "Fwd Packet Length Mean": smean,
        "Fwd Packet Length Std": pd.Series(0.0, index=numeric.index),
        "Bwd Packet Length Max": dmean,
        "Bwd Packet Length Min": dmean,
        "Bwd Packet Length Mean": dmean,
        "Bwd Packet Length Std": pd.Series(0.0, index=numeric.index),
        "Flow Bytes/s": pd.Series(safe_divide(total_bytes, dur_seconds), index=numeric.index),
        "Flow Packets/s": pd.Series(safe_divide(total_packets, dur_seconds), index=numeric.index),
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": pd.Series(np.maximum(numeric["sinpkt"], numeric["dinpkt"]) * 1000.0, index=numeric.index),
        "Flow IAT Min": pd.Series(np.minimum(numeric["sinpkt"], numeric["dinpkt"]) * 1000.0, index=numeric.index),
        "Fwd IAT Total": fwd_iat_total,
        "Fwd IAT Mean": numeric["sinpkt"] * 1000.0,
        "Fwd IAT Std": numeric["sjit"] * 1000.0,
        "Fwd IAT Max": numeric["sinpkt"] * 1000.0,
        "Fwd IAT Min": numeric["sinpkt"] * 1000.0,
        "Bwd IAT Total": bwd_iat_total,
        "Bwd IAT Mean": numeric["dinpkt"] * 1000.0,
        "Bwd IAT Std": numeric["djit"] * 1000.0,
        "Bwd IAT Max": numeric["dinpkt"] * 1000.0,
        "Bwd IAT Min": numeric["dinpkt"] * 1000.0,
        "Fwd PSH Flags": pd.Series(0.0, index=numeric.index),
        "Bwd PSH Flags": pd.Series(0.0, index=numeric.index),
        "Fwd URG Flags": pd.Series(0.0, index=numeric.index),
        "Bwd URG Flags": pd.Series(0.0, index=numeric.index),
        "Fwd Header Length": pd.Series(header_unit * numeric["spkts"], index=numeric.index),
        "Bwd Header Length": pd.Series(header_unit * numeric["dpkts"], index=numeric.index),
        "Fwd Packets/s": pd.Series(safe_divide(numeric["spkts"], dur_seconds), index=numeric.index),
        "Bwd Packets/s": pd.Series(safe_divide(numeric["dpkts"], dur_seconds), index=numeric.index),
        "Min Packet Length": packet_min,
        "Max Packet Length": packet_max,
        "Packet Length Mean": packet_mean,
        "Packet Length Std": packet_std,
        "Packet Length Variance": packet_std ** 2,
        "FIN Flag Count": fin_count,
        "SYN Flag Count": syn_count,
        "RST Flag Count": rst_count,
        "PSH Flag Count": pd.Series(0.0, index=numeric.index),
        "ACK Flag Count": ack_count,
        "URG Flag Count": pd.Series(0.0, index=numeric.index),
        "CWE Flag Count": pd.Series(0.0, index=numeric.index),
        "ECE Flag Count": pd.Series(0.0, index=numeric.index),
        "Down/Up Ratio": pd.Series(safe_divide(numeric["dpkts"], np.maximum(numeric["spkts"], 1)), index=numeric.index),
        "Average Packet Size": packet_mean,
        "Avg Fwd Segment Size": smean,
        "Avg Bwd Segment Size": dmean,
        "Fwd Header Length.1": pd.Series(header_unit * numeric["spkts"], index=numeric.index),
        "Fwd Avg Bytes/Bulk": pd.Series(0.0, index=numeric.index),
        "Fwd Avg Packets/Bulk": pd.Series(0.0, index=numeric.index),
        "Fwd Avg Bulk Rate": pd.Series(0.0, index=numeric.index),
        "Bwd Avg Bytes/Bulk": pd.Series(0.0, index=numeric.index),
        "Bwd Avg Packets/Bulk": pd.Series(0.0, index=numeric.index),
        "Bwd Avg Bulk Rate": pd.Series(0.0, index=numeric.index),
        "Subflow Fwd Packets": numeric["spkts"],
        "Subflow Fwd Bytes": numeric["sbytes"],
        "Subflow Bwd Packets": numeric["dpkts"],
        "Subflow Bwd Bytes": numeric["dbytes"],
        "Init_Win_bytes_forward": numeric["swin"],
        "Init_Win_bytes_backward": numeric["dwin"],
        "act_data_pkt_fwd": numeric["spkts"],
        "min_seg_size_forward": pd.Series(np.where(header_unit > 0, header_unit, 0.0), index=numeric.index),
        "Active Mean": flow_duration,
        "Active Std": pd.Series(0.0, index=numeric.index),
        "Active Max": flow_duration,
        "Active Min": flow_duration,
        "Idle Mean": pd.Series(0.0, index=numeric.index),
        "Idle Std": pd.Series(0.0, index=numeric.index),
        "Idle Max": pd.Series(0.0, index=numeric.index),
        "Idle Min": pd.Series(0.0, index=numeric.index),
    }

    for column_name, values in explicit_values.items():
        if column_name in aligned.columns:
            aligned[column_name] = pd.to_numeric(values, errors="coerce").fillna(0.0)

    return aligned, sorted(explicit_values.keys())


def build_unsw_nsl_view(
    unsw_df: pd.DataFrame,
    nsl_feature_names: list[str],
    protocol_map: dict[str, int],
    service_map: dict[str, int],
    flag_map: dict[str, int],
) -> tuple[pd.DataFrame, list[str]]:
    """Create an NSL-KDD-style feature frame from UNSW-NB15 columns."""
    numeric = unsw_df.copy()
    for column in numeric.columns:
        if column not in {"proto", "service", "state", "attack_cat"}:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    same_srv_rate = clip_rate(safe_divide(numeric["ct_srv_dst"], np.maximum(numeric["ct_dst_ltm"], 1)))
    dst_same_src_port_rate = clip_rate(safe_divide(numeric["ct_dst_sport_ltm"], np.maximum(numeric["ct_dst_ltm"], 1)))
    srv_diff_host_rate = clip_rate(safe_divide(numeric["ct_dst_src_ltm"], np.maximum(numeric["ct_srv_src"], 1)))
    srv_diff_host_rate_alt = clip_rate(safe_divide(numeric["ct_dst_src_ltm"], np.maximum(numeric["ct_srv_dst"], 1)))
    serror_rate = clip_rate(safe_divide(numeric["synack"], np.maximum(numeric["tcprtt"], 1e-9)))
    rerror_rate = numeric["state"].astype(str).str.upper().eq("RST").astype(float)

    state_mapping = {
        "ACC": "SF",
        "CON": "SF",
        "FIN": "SF",
        "CLO": "S2",
        "REQ": "S0",
        "RST": "RSTR",
        "INT": "OTH",
    }
    mapped_flags = unsw_df["state"].astype(str).str.strip().str.upper().map(state_mapping).fillna("OTH")

    service_values = unsw_df["service"].astype(str).str.strip().replace({"-": "other"})
    protocol_values = unsw_df["proto"].astype(str).str.strip()

    aligned = pd.DataFrame(0.0, index=unsw_df.index, columns=nsl_feature_names)

    explicit_values = {
        "duration": numeric["dur"],
        "protocol_type": encode_nsl_category(protocol_values, protocol_map),
        "service": encode_nsl_category(service_values, service_map, fallback_text="other"),
        "flag": encode_nsl_category(mapped_flags, flag_map, fallback_text="OTH"),
        "src_bytes": numeric["sbytes"],
        "dst_bytes": numeric["dbytes"],
        "land": numeric["is_sm_ips_ports"],
        "wrong_fragment": pd.Series(0.0, index=numeric.index),
        "urgent": pd.Series(0.0, index=numeric.index),
        "hot": numeric["trans_depth"],
        "num_failed_logins": pd.Series(0.0, index=numeric.index),
        "logged_in": numeric["is_ftp_login"],
        "num_compromised": numeric["sloss"] + numeric["dloss"],
        "root_shell": pd.Series(0.0, index=numeric.index),
        "su_attempted": pd.Series(0.0, index=numeric.index),
        "num_root": pd.Series(0.0, index=numeric.index),
        "num_file_creations": pd.Series(0.0, index=numeric.index),
        "num_shells": pd.Series(0.0, index=numeric.index),
        "num_access_files": pd.Series(0.0, index=numeric.index),
        "num_outbound_cmds": pd.Series(0.0, index=numeric.index),
        "is_host_login": pd.Series(0.0, index=numeric.index),
        "is_guest_login": pd.Series(0.0, index=numeric.index),
        "count": numeric["ct_dst_ltm"],
        "srv_count": numeric["ct_srv_src"],
        "serror_rate": pd.Series(serror_rate, index=numeric.index),
        "srv_serror_rate": pd.Series(serror_rate, index=numeric.index),
        "rerror_rate": rerror_rate,
        "srv_rerror_rate": rerror_rate,
        "same_srv_rate": pd.Series(same_srv_rate, index=numeric.index),
        "diff_srv_rate": pd.Series(1.0 - same_srv_rate, index=numeric.index),
        "srv_diff_host_rate": pd.Series(srv_diff_host_rate, index=numeric.index),
        "dst_host_count": numeric["ct_dst_ltm"],
        "dst_host_srv_count": numeric["ct_srv_dst"],
        "dst_host_same_srv_rate": pd.Series(same_srv_rate, index=numeric.index),
        "dst_host_diff_srv_rate": pd.Series(1.0 - same_srv_rate, index=numeric.index),
        "dst_host_same_src_port_rate": pd.Series(dst_same_src_port_rate, index=numeric.index),
        "dst_host_srv_diff_host_rate": pd.Series(srv_diff_host_rate_alt, index=numeric.index),
        "dst_host_serror_rate": pd.Series(serror_rate, index=numeric.index),
        "dst_host_srv_serror_rate": pd.Series(serror_rate, index=numeric.index),
        "dst_host_rerror_rate": rerror_rate,
        "dst_host_srv_rerror_rate": rerror_rate,
    }

    for column_name, values in explicit_values.items():
        if column_name in aligned.columns:
            aligned[column_name] = pd.to_numeric(values, errors="coerce").fillna(0.0)

    return aligned, sorted(explicit_values.keys())


def evaluate_unsw_on_models(
    unsw_test_df: pd.DataFrame,
    report_lines: list[str],
) -> dict[str, dict[str, object]]:
    """Evaluate CICIDS, NSL-KDD, and the weighted ensemble on the UNSW-NB15 test set."""
    print_section("STEP 3 - EVALUATE UNSW-NB15 ON CURRENT MODELS")

    try:
        y_test = pd.to_numeric(unsw_test_df["label"], errors="coerce").fillna(0).astype(int)

        cicids_model = joblib.load(CICIDS_MODEL_PATH)
        cicids_scaler = joblib.load(CICIDS_SCALER_PATH)
        cicids_feature_names = list(joblib.load(CICIDS_FEATURES_PATH))
        cicids_threshold = float(joblib.load(CICIDS_THRESHOLD_PATH))

        nsl_model = joblib.load(NSL_MODEL_PATH)
        nsl_scaler = joblib.load(NSL_SCALER_PATH)
        nsl_feature_names = list(joblib.load(NSL_FEATURES_PATH))

        protocol_map, service_map, flag_map = build_nsl_reference_maps()

        cicids_view, cicids_populated = build_unsw_cicids_view(unsw_test_df, cicids_feature_names)
        nsl_view, nsl_populated = build_unsw_nsl_view(
            unsw_df=unsw_test_df,
            nsl_feature_names=nsl_feature_names,
            protocol_map=protocol_map,
            service_map=service_map,
            flag_map=flag_map,
        )

        log(report_lines, f"[MAP] CICIDS populated feature count from UNSW: {len(cicids_populated)}/{len(cicids_feature_names)}")
        log(report_lines, f"[MAP] NSL-KDD populated feature count from UNSW: {len(nsl_populated)}/{len(nsl_feature_names)}")

        cicids_scaled = pd.DataFrame(
            cicids_scaler.transform(cicids_view),
            columns=cicids_feature_names,
        )
        nsl_scaled = pd.DataFrame(
            nsl_scaler.transform(nsl_view),
            columns=nsl_feature_names,
        )

        cicids_proba = cicids_model.predict_proba(cicids_scaled)[:, 1]
        nsl_proba = nsl_model.predict_proba(nsl_scaled)[:, 1]
        ensemble_proba = (cicids_proba * 0.7) + (nsl_proba * 0.3)

        cicids_predictions = (cicids_proba >= cicids_threshold).astype(int)
        nsl_predictions = (nsl_proba >= 0.5).astype(int)
        ensemble_predictions = (ensemble_proba >= cicids_threshold).astype(int)

        cicids_metrics = compute_metrics(y_test, cicids_predictions, "CICIDS model on UNSW-NB15")
        nsl_metrics = compute_metrics(y_test, nsl_predictions, "NSL-KDD model on UNSW-NB15")
        ensemble_metrics = compute_metrics(y_test, ensemble_predictions, "CROSS-SECURE ensemble on UNSW-NB15")

        print_metrics(cicids_metrics, report_lines)
        print_metrics(nsl_metrics, report_lines)
        print_metrics(ensemble_metrics, report_lines)

        agreement_rate = float(np.mean(cicids_predictions == nsl_predictions) * 100.0)
        attack_rate = float(np.mean(ensemble_predictions == 1) * 100.0)
        log(report_lines, f"[ENSEMBLE] CICIDS/NSL agreement rate on UNSW test set: {agreement_rate:.2f}%")
        log(report_lines, f"[ENSEMBLE] Ensemble predicted ATTACK on {attack_rate:.2f}% of UNSW test rows")

        return {
            "cicids": cicids_metrics,
            "nsl": nsl_metrics,
            "ensemble": ensemble_metrics,
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to evaluate UNSW-NB15 on the trained models: {exc}") from exc


def build_summary_table(results: dict[str, dict[str, object]]) -> str:
    """Create a compact text table for the saved UNSW report."""
    rows = [
        ("CICIDS model", results["cicids"]["accuracy"] * 100, results["cicids"]["attack_f1"]),
        ("NSL-KDD model", results["nsl"]["accuracy"] * 100, results["nsl"]["attack_f1"]),
        ("Ensemble (0.7/0.3)", results["ensemble"]["accuracy"] * 100, results["ensemble"]["attack_f1"]),
    ]

    lines = [
        "+--------------------+----------+-----------+",
        "| Model              | Accuracy | F1-Attack |",
        "+--------------------+----------+-----------+",
    ]
    for name, accuracy, f1_score in rows:
        lines.append(f"| {name.ljust(18)} | {accuracy:7.2f}% | {f1_score:9.4f} |")
    lines.append("+--------------------+----------+-----------+")

    return "\n".join(lines)


def save_report(report_lines: list[str]) -> None:
    """Save the UNSW evaluation report inside the results folder."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report saved to {REPORT_PATH.as_posix()}")


def main() -> None:
    """Run the full UNSW-NB15 cross-dataset evaluation workflow."""
    report_lines: list[str] = []

    try:
        print_section("CROSS-SECURE UNSW-NB15 EVALUATION")
        log(report_lines, "CROSS-SECURE UNSW-NB15 EVALUATION")

        train_df, test_df = load_unsw_data(report_lines)
        _, cleaned_test = clean_unsw_data(train_df, test_df, report_lines)
        results = evaluate_unsw_on_models(cleaned_test, report_lines)

        print_section("STEP 4 - UNSW SUMMARY")
        summary_table = build_summary_table(results)
        log(report_lines, summary_table)

        print_section("STEP 5 - SAVE REPORT")
        save_report(report_lines)
    except Exception as exc:
        error_message = f"[FATAL] test_unsw_nb15.py failed: {exc}"
        print(error_message)
        report_lines.append(error_message)
        try:
            save_report(report_lines)
        except Exception as save_exc:
            print(f"[FATAL] Could not save failure report: {save_exc}")


if __name__ == "__main__":
    main()
