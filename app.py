"""
Folder Path: cross-secure/app.py
"""

# Import standard library modules used for runtime state and system details.
import platform
import socket
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from threading import Lock

# Import Flask components and CORS support for the dashboard frontend.
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    # Prefer Scapy for Windows-friendly interface discovery.
    from scapy.arch.windows import get_windows_if_list
except Exception:  # pragma: no cover - fallback for non-Windows or missing backend.
    get_windows_if_list = None

try:
    # Fallback interface listing for non-Windows environments.
    from scapy.all import get_if_list
except Exception:  # pragma: no cover - Scapy may be unavailable during partial installs.
    get_if_list = None


# Create the Flask application and enable cross-origin requests.
app = Flask(__name__)
CORS(app)


# Keep only the most recent live alerts in memory for dashboard polling.
alerts_lock = Lock()
live_alerts = deque(maxlen=100)
service_started_at = datetime.now(timezone.utc)


# Store rolling aggregate stats in memory for fast dashboard responses.
stats_lock = Lock()
stats = {
    "total": 0,
    "normal": 0,
    "attack": 0,
}


def utc_now() -> datetime:
    """Return the current UTC time with timezone metadata."""
    return datetime.now(timezone.utc)


def safe_float(value, default=0.0):
    """Convert incoming values to float while tolerating bad payloads."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value, default=0):
    """Convert incoming values to int while tolerating bad payloads."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def get_interfaces():
    """Return available network interfaces in a Windows-friendly format."""
    interfaces = []

    try:
        if platform.system().lower() == "windows" and get_windows_if_list:
            for item in get_windows_if_list():
                ips = item.get("ips") or []
                if not isinstance(ips, list):
                    ips = [ips]
                interfaces.append(
                    {
                        "name": item.get("name") or item.get("description") or "Unknown",
                        "description": item.get("description") or item.get("name") or "Unknown",
                        "ips": ips,
                        "guid": item.get("guid"),
                    }
                )
        elif get_if_list:
            for name in get_if_list():
                interfaces.append(
                    {
                        "name": name,
                        "description": name,
                        "ips": [],
                        "guid": None,
                    }
                )
    except Exception as exc:
        interfaces.append(
            {
                "name": "unavailable",
                "description": f"Interface lookup failed: {exc}",
                "ips": [],
                "guid": None,
            }
        )

    return interfaces


def update_stats(label: str):
    """Update aggregate counters for each received live alert."""
    normalized = "ATTACK" if str(label).upper() == "ATTACK" else "NORMAL"

    with stats_lock:
        stats["total"] += 1
        if normalized == "ATTACK":
            stats["attack"] += 1
        else:
            stats["normal"] += 1


def parse_timestamp(value) -> datetime:
    """Parse ISO timestamps safely and fall back to the current time."""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            pass

    return utc_now()


def attacks_last_minute() -> int:
    """Count attack alerts received within the last 60 seconds."""
    cutoff = utc_now() - timedelta(minutes=1)

    with alerts_lock:
        return sum(
            1
            for alert in live_alerts
            if str(alert.get("label", "")).upper() == "ATTACK"
            and parse_timestamp(alert.get("timestamp")) >= cutoff
        )


def compute_system_status() -> str:
    """Report MONITORING when alerts are arriving, otherwise IDLE."""
    with alerts_lock:
        if not live_alerts:
            return "IDLE"

        latest = parse_timestamp(live_alerts[-1].get("timestamp"))

    if utc_now() - latest <= timedelta(seconds=15):
        return "MONITORING"

    return "IDLE"


def stats_snapshot():
    """Return a consistent copy of current aggregate statistics."""
    with stats_lock:
        total = stats["total"]
        normal = stats["normal"]
        attack = stats["attack"]

    detection_rate = round((attack / total) * 100, 2) if total else 0.0

    return {
        "total": total,
        "normal": normal,
        "attack": attack,
        "detection_rate": detection_rate,
        "attacks_last_minute": attacks_last_minute(),
        "system_status": compute_system_status(),
    }


def sanitize_alert(payload: dict) -> dict:
    """Normalize an incoming live alert into a stable dashboard-friendly shape."""
    label = "ATTACK" if safe_int(payload.get("prediction"), 0) == 1 or str(payload.get("label", "")).upper() == "ATTACK" else "NORMAL"

    return {
        "src_ip": str(payload.get("src_ip", "unknown")),
        "dst_ip": str(payload.get("dst_ip", "unknown")),
        "src_port": safe_int(payload.get("src_port"), 0),
        "dst_port": safe_int(payload.get("dst_port"), 0),
        "protocol": str(payload.get("protocol", "UNKNOWN")).upper(),
        "prediction": 1 if label == "ATTACK" else 0,
        "label": label,
        "confidence": f"{safe_float(str(payload.get('confidence', '0')).replace('%', ''), 0.0):.2f}%",
        "timestamp": parse_timestamp(payload.get("timestamp")).isoformat(),
        "flow_duration": round(safe_float(payload.get("flow_duration"), 0.0), 3),
        "packets_count": safe_int(payload.get("packets_count"), 0),
    }


@app.get("/")
def index():
    """Return a service health message and basic system information."""
    return jsonify(
        {
            "message": "CROSS-SECURE live IDS API is running",
            "system_info": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "started_at": service_started_at.isoformat(),
                "uptime_seconds": int((utc_now() - service_started_at).total_seconds()),
                "alerts_buffer_size": live_alerts.maxlen,
                "interfaces_detected": len(get_interfaces()),
            },
            "stats": stats_snapshot(),
        }
    )


@app.post("/live-alert")
def receive_live_alert():
    """Receive completed flow predictions from capture.py and store them in memory."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON request body is required."}), 400

    required = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "prediction", "label", "confidence", "timestamp"]
    missing = [field for field in required if field not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        alert = sanitize_alert(payload)
        with alerts_lock:
            live_alerts.append(alert)
        update_stats(alert["label"])
        return jsonify({"message": "Alert received", "stats": stats_snapshot(), "alert": alert}), 201
    except Exception as exc:
        return jsonify({"error": f"Failed to process alert: {exc}"}), 500


@app.get("/alerts")
def get_alerts():
    """Return the most recent live alerts in reverse-chronological order."""
    with alerts_lock:
        ordered_alerts = list(reversed(live_alerts))

    return jsonify({"count": len(ordered_alerts), "alerts": ordered_alerts})


@app.get("/stats")
def get_stats():
    """Return aggregate detection statistics for the dashboard."""
    return jsonify(stats_snapshot())


@app.get("/interfaces")
def interfaces():
    """Expose available network interfaces so capture.py or the UI can inspect them."""
    interfaces_list = get_interfaces()
    counts = Counter("has_ip" if item.get("ips") else "no_ip" for item in interfaces_list)

    return jsonify({"count": len(interfaces_list), "summary": dict(counts), "interfaces": interfaces_list})


if __name__ == "__main__":
    # Run the API on all interfaces so the dashboard and local capture agent can reach it.
    app.run(host="0.0.0.0", port=5000, debug=True)
