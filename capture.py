"""
Folder Path: cross-secure/capture.py
"""

# Import standard library modules for threading, timing, and Windows-safe runtime checks.
import asyncio
import ctypes
import os
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Import third-party modules for packet capture, inference, and API delivery.
import joblib
import numpy as np
import pandas as pd
import pyshark
import requests

try:
    # Prefer Windows interface discovery via Scapy when available.
    from scapy.arch.windows import get_windows_if_list
except Exception:  # pragma: no cover - fallback for non-Windows or limited installs.
    get_windows_if_list = None

try:
    # General Scapy fallback for interface listing.
    from scapy.all import get_if_list
except Exception:  # pragma: no cover
    get_if_list = None


# Resolve important project paths relative to this file so the script works from the project root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cross_secure_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.pkl")


# Configure flow lifecycle thresholds and the Flask API target.
API_URL = "http://127.0.0.1:5000/live-alert"
FLOW_IDLE_TIMEOUT_SECONDS = 5.0
FLOW_MAX_PACKETS = 100
ACTIVE_IDLE_SPLIT_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 5


# Define the default CICIDS-style feature names this capture engine computes.
DEFAULT_FEATURE_ORDER = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def safe_float(value, default=0.0):
    """Convert arbitrary values to float without raising."""
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value, default=0):
    """Convert arbitrary values to int without raising."""
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def stat_mean(values):
    """Return the arithmetic mean for a numeric list."""
    return float(np.mean(values)) if values else 0.0


def stat_std(values):
    """Return the population standard deviation for a numeric list."""
    return float(np.std(values)) if values else 0.0


def stat_var(values):
    """Return the population variance for a numeric list."""
    return float(np.var(values)) if values else 0.0


def stat_max(values):
    """Return the maximum of a numeric list."""
    return float(max(values)) if values else 0.0


def stat_min(values):
    """Return the minimum of a numeric list."""
    return float(min(values)) if values else 0.0


def rate(count_or_bytes, duration_seconds):
    """Compute rates while avoiding division by zero."""
    if duration_seconds <= 0:
        return 0.0
    return float(count_or_bytes) / float(duration_seconds)


def compute_iats(timestamps):
    """Compute inter-arrival times from a list of packet timestamps."""
    if len(timestamps) < 2:
        return []
    return [max(0.0, timestamps[index] - timestamps[index - 1]) for index in range(1, len(timestamps))]


def compute_active_idle_periods(timestamps):
    """Split a flow timeline into active and idle periods using a gap threshold."""
    if not timestamps:
        return [], []

    active_periods = []
    idle_periods = []
    current_start = timestamps[0]
    previous = timestamps[0]

    for current in timestamps[1:]:
        gap = current - previous
        if gap > ACTIVE_IDLE_SPLIT_SECONDS:
            active_periods.append(max(0.0, previous - current_start))
            idle_periods.append(gap)
            current_start = current
        previous = current

    active_periods.append(max(0.0, previous - current_start))
    return active_periods, idle_periods


def is_admin():
    """Return True when the current Windows process has administrator privileges."""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def list_interfaces():
    """Discover available capture interfaces with Windows-first logic."""
    interfaces = []

    try:
        if get_windows_if_list:
            for item in get_windows_if_list():
                interfaces.append(
                    {
                        "name": item.get("name") or item.get("description") or "Unknown",
                        "description": item.get("description") or item.get("name") or "Unknown",
                        "ips": item.get("ips") or [],
                    }
                )
        elif get_if_list:
            for name in get_if_list():
                interfaces.append({"name": name, "description": name, "ips": []})
    except Exception as exc:
        print(f"[WARN] Failed to list interfaces: {exc}")

    return interfaces


def detect_active_interface():
    """Pick the most likely active interface on Windows for live packet capture."""
    interfaces = list_interfaces()
    if not interfaces:
        raise RuntimeError("No network interfaces were detected.")

    preferred_names = []
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = None

    ignored_tokens = [
        "loopback",
        "local area connection*",
        "isatap",
        "teredo",
        "pseudo-interface",
        "bluetooth",
        "virtual",
        "vmware",
        "hyper-v",
    ]

    filtered_interfaces = [
        item
        for item in interfaces
        if not any(token in str(item["name"]).lower() for token in ignored_tokens)
    ] or interfaces

    for item in filtered_interfaces:
        ips = item.get("ips") or []
        ip_values = ips if isinstance(ips, list) else [ips]
        if local_ip and local_ip in ip_values:
            preferred_names.append(item["name"])
        elif any(str(ip).count(".") == 3 and not str(ip).startswith("127.") for ip in ip_values):
            preferred_names.append(item["name"])

    if preferred_names:
        return preferred_names[0]

    for item in filtered_interfaces:
        name = str(item["name"]).lower()
        if "wi-fi" in name or "ethernet" in name or "wlan" in name:
            return item["name"]

    return filtered_interfaces[0]["name"]


def get_layer_field(packet, layer_name, field_name, default=None):
    """Safely retrieve a field from a packet layer."""
    try:
        layer = getattr(packet, layer_name, None)
        if layer is None:
            return default
        return getattr(layer, field_name, default)
    except Exception:
        return default


def parse_tcp_flags(packet):
    """Return common TCP flag booleans from a packet."""
    raw_flags = get_layer_field(packet, "tcp", "flags", "0x0000")
    text = str(raw_flags).strip().lower()
    if text.startswith("0x"):
        try:
            value = int(text, 16)
        except ValueError:
            value = 0
    else:
        try:
            value = int(text, 0)
        except ValueError:
            value = 0

    return {
        "fin": 1 if value & 0x01 else 0,
        "syn": 1 if value & 0x02 else 0,
        "rst": 1 if value & 0x04 else 0,
        "psh": 1 if value & 0x08 else 0,
        "ack": 1 if value & 0x10 else 0,
        "urg": 1 if value & 0x20 else 0,
    }


def packet_length(packet):
    """Extract the on-wire packet length."""
    for candidate in [getattr(packet, "length", None), get_layer_field(packet, "ip", "len"), get_layer_field(packet, "ipv6", "plen")]:
        if candidate is not None:
            return max(0.0, safe_float(candidate, 0.0))
    return 0.0


def header_length(packet, protocol):
    """Estimate IP and transport header length in bytes."""
    ip_header = 0.0
    transport = 0.0

    ip_hl = get_layer_field(packet, "ip", "hdr_len")
    if ip_hl is not None:
        ip_header = safe_float(ip_hl, 0.0)
    elif hasattr(packet, "ipv6"):
        ip_header = 40.0

    if protocol == "TCP":
        transport = safe_float(get_layer_field(packet, "tcp", "hdr_len"), 20.0)
    elif protocol == "UDP":
        transport = 8.0

    return ip_header + transport


def normalize_protocol(packet):
    """Return a consistent protocol label for a packet."""
    if hasattr(packet, "tcp"):
        return "TCP"
    if hasattr(packet, "udp"):
        return "UDP"
    if hasattr(packet, "icmp"):
        return "ICMP"
    protocol = get_layer_field(packet, "ip", "proto") or get_layer_field(packet, "ipv6", "nxt")
    return str(protocol or "OTHER").upper()


def packet_endpoints(packet):
    """Extract flow endpoint information from IPv4 or IPv6 packets."""
    src_ip = get_layer_field(packet, "ip", "src") or get_layer_field(packet, "ipv6", "src")
    dst_ip = get_layer_field(packet, "ip", "dst") or get_layer_field(packet, "ipv6", "dst")
    protocol = normalize_protocol(packet)

    if not src_ip or not dst_ip:
        return None

    if protocol == "TCP":
        src_port = safe_int(get_layer_field(packet, "tcp", "srcport"), 0)
        dst_port = safe_int(get_layer_field(packet, "tcp", "dstport"), 0)
    elif protocol == "UDP":
        src_port = safe_int(get_layer_field(packet, "udp", "srcport"), 0)
        dst_port = safe_int(get_layer_field(packet, "udp", "dstport"), 0)
    else:
        src_port = 0
        dst_port = 0

    return src_ip, dst_ip, src_port, dst_port, protocol


@dataclass
class PacketRecord:
    """Store the per-packet fields needed for flow feature extraction."""
    timestamp: float
    direction: str
    length: float
    header_length: float
    flags: dict
    window_size: int


@dataclass
class Flow:
    """Maintain packet state and derived statistics for a bidirectional flow."""
    key: tuple
    origin: tuple
    display_endpoints: tuple
    first_seen: float
    last_seen: float
    packets: list = field(default_factory=list)

    def add_packet(self, record: PacketRecord):
        """Append a packet record and update the flow timestamp."""
        self.packets.append(record)
        self.last_seen = record.timestamp

    @property
    def packet_count(self):
        """Return the number of packets currently tracked in the flow."""
        return len(self.packets)


class LiveTrafficCapture:
    """Capture live traffic, build flows, extract features, and run predictions."""

    def __init__(self):
        # Load model artifacts once on startup.
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.threshold = joblib.load(THRESHOLD_PATH)
        self.feature_names = self.load_feature_names()

        # Keep active flows in memory behind a lock because capture and cleanup run in separate threads.
        self.flows = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.capture = None
        self.interface = None

    def load_feature_names(self):
        """Load saved feature names and fall back to scaler metadata or defaults."""
        loaded = None

        try:
            loaded = joblib.load(FEATURES_PATH)
        except Exception:
            try:
                import pickle
                with open(FEATURES_PATH, "rb") as handle:
                    loaded = pickle.load(handle)
            except Exception as exc:
                print(f"[WARN] Could not load feature_names.pkl: {exc}")

        if isinstance(loaded, (list, tuple)) and loaded:
            print(f"[INFO] Loaded {len(loaded)} feature name(s) from feature_names.pkl")
            return list(loaded)

        scaler_features = getattr(self.scaler, "feature_names_in_", None)
        if scaler_features is not None and len(scaler_features):
            print(f"[INFO] Falling back to scaler feature metadata with {len(scaler_features)} column(s)")
            return list(scaler_features)

        print(f"[INFO] Falling back to default CICIDS-style feature order with {len(DEFAULT_FEATURE_ORDER)} columns")
        return DEFAULT_FEATURE_ORDER[:]

    def canonical_key(self, endpoints):
        """Build a bidirectional flow key so both directions map to the same flow."""
        src_ip, dst_ip, src_port, dst_port, protocol = endpoints
        left = (src_ip, src_port)
        right = (dst_ip, dst_port)
        if left <= right:
            origin = left
            key = (src_ip, dst_ip, src_port, dst_port, protocol)
        else:
            origin = right
            key = (dst_ip, src_ip, dst_port, src_port, protocol)
        return key, origin

    def packet_direction(self, flow, endpoints):
        """Determine whether the packet is forward or backward for this flow."""
        src_ip, _, src_port, _, _ = endpoints
        if (src_ip, src_port) == flow.origin:
            return "fwd"
        return "bwd"

    def build_feature_dict(self, flow):
        """Extract CICIDS-style numerical features from a completed flow."""
        packets = flow.packets
        timestamps = [packet.timestamp for packet in packets]
        all_lengths = [packet.length for packet in packets]
        fwd_packets = [packet for packet in packets if packet.direction == "fwd"]
        bwd_packets = [packet for packet in packets if packet.direction == "bwd"]

        fwd_lengths = [packet.length for packet in fwd_packets]
        bwd_lengths = [packet.length for packet in bwd_packets]
        fwd_headers = [packet.header_length for packet in fwd_packets]
        bwd_headers = [packet.header_length for packet in bwd_packets]

        duration_seconds = max(0.0, flow.last_seen - flow.first_seen)
        duration_micros = duration_seconds * 1_000_000.0
        total_bytes = sum(all_lengths)

        flow_iat = compute_iats(timestamps)
        fwd_iat = compute_iats([packet.timestamp for packet in fwd_packets])
        bwd_iat = compute_iats([packet.timestamp for packet in bwd_packets])
        active_periods, idle_periods = compute_active_idle_periods(timestamps)

        flags = defaultdict(int)
        for packet in packets:
            for flag_name, flag_value in packet.flags.items():
                flags[flag_name] += safe_int(flag_value, 0)

        return {
            "Flow Duration": duration_micros,
            "Total Fwd Packets": len(fwd_packets),
            "Total Backward Packets": len(bwd_packets),
            "Total Length of Fwd Packets": float(sum(fwd_lengths)),
            "Total Length of Bwd Packets": float(sum(bwd_lengths)),
            "Fwd Packet Length Max": stat_max(fwd_lengths),
            "Fwd Packet Length Min": stat_min(fwd_lengths),
            "Fwd Packet Length Mean": stat_mean(fwd_lengths),
            "Bwd Packet Length Max": stat_max(bwd_lengths),
            "Bwd Packet Length Min": stat_min(bwd_lengths),
            "Bwd Packet Length Mean": stat_mean(bwd_lengths),
            "Flow Bytes/s": rate(total_bytes, duration_seconds),
            "Flow Packets/s": rate(len(packets), duration_seconds),
            "Flow IAT Mean": stat_mean(flow_iat),
            "Flow IAT Std": stat_std(flow_iat),
            "Flow IAT Max": stat_max(flow_iat),
            "Flow IAT Min": stat_min(flow_iat),
            "Fwd IAT Mean": stat_mean(fwd_iat),
            "Fwd IAT Std": stat_std(fwd_iat),
            "Fwd IAT Max": stat_max(fwd_iat),
            "Fwd IAT Min": stat_min(fwd_iat),
            "Bwd IAT Mean": stat_mean(bwd_iat),
            "Bwd IAT Std": stat_std(bwd_iat),
            "Bwd IAT Max": stat_max(bwd_iat),
            "Bwd IAT Min": stat_min(bwd_iat),
            "Fwd PSH Flags": sum(packet.flags.get("psh", 0) for packet in fwd_packets),
            "Bwd PSH Flags": sum(packet.flags.get("psh", 0) for packet in bwd_packets),
            "Fwd URG Flags": sum(packet.flags.get("urg", 0) for packet in fwd_packets),
            "Bwd URG Flags": sum(packet.flags.get("urg", 0) for packet in bwd_packets),
            "Fwd Header Length": float(sum(fwd_headers)),
            "Bwd Header Length": float(sum(bwd_headers)),
            "Fwd Packets/s": rate(len(fwd_packets), duration_seconds),
            "Bwd Packets/s": rate(len(bwd_packets), duration_seconds),
            "Packet Length Min": stat_min(all_lengths),
            "Packet Length Max": stat_max(all_lengths),
            "Packet Length Mean": stat_mean(all_lengths),
            "Packet Length Std": stat_std(all_lengths),
            "Packet Length Variance": stat_var(all_lengths),
            "FIN Flag Count": flags["fin"],
            "SYN Flag Count": flags["syn"],
            "RST Flag Count": flags["rst"],
            "PSH Flag Count": flags["psh"],
            "ACK Flag Count": flags["ack"],
            "URG Flag Count": flags["urg"],
            "Average Packet Size": stat_mean(all_lengths),
            "Avg Fwd Segment Size": stat_mean(fwd_lengths),
            "Avg Bwd Segment Size": stat_mean(bwd_lengths),
            "Init_Win_bytes_forward": safe_int(fwd_packets[0].window_size if fwd_packets else 0, 0),
            "Init_Win_bytes_backward": safe_int(bwd_packets[0].window_size if bwd_packets else 0, 0),
            "Active Mean": stat_mean(active_periods),
            "Active Std": stat_std(active_periods),
            "Active Max": stat_max(active_periods),
            "Active Min": stat_min(active_periods),
            "Idle Mean": stat_mean(idle_periods),
            "Idle Std": stat_std(idle_periods),
            "Idle Max": stat_max(idle_periods),
            "Idle Min": stat_min(idle_periods),
            "duration": duration_seconds,
        }

    def align_features(self, feature_dict):
        """Align extracted features to the trained model's expected input columns."""
        values = {name: safe_float(feature_dict.get(name, 0.0), 0.0) for name in self.feature_names}
        return pd.DataFrame([values], columns=self.feature_names)

    def predict_flow(self, flow):
        """Run inference for a completed flow and prepare a JSON-safe alert payload."""
        features = self.build_feature_dict(flow)
        model_input = self.align_features(features)

        scaled = self.scaler.transform(model_input)
        proba = float(self.model.predict_proba(scaled)[0][1])
        prediction = 1 if proba >= self.threshold else 0
        confidence = round(proba * 100, 2)

        src_ip, dst_ip, src_port, dst_port, protocol = flow.display_endpoints
        return {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "prediction": prediction,
            "label": "ATTACK" if prediction == 1 else "NORMAL",
            "confidence": f"{confidence:.2f}%",
            "timestamp": utc_now_iso(),
            "flow_duration": round((flow.last_seen - flow.first_seen) * 1000.0, 3),
            "packets_count": flow.packet_count,
        }

    def send_alert(self, payload):
        """Send the completed flow prediction to the Flask API."""
        try:
            response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except Exception as exc:
            print(f"[WARN] Failed to POST /live-alert: {exc}")

    def complete_flow(self, key, reason):
        """Remove a flow, run inference, and notify the API."""
        with self.lock:
            flow = self.flows.pop(key, None)

        if flow is None or not flow.packets:
            return

        try:
            payload = self.predict_flow(flow)
            print(
                f"[FLOW] reason={reason} {payload['src_ip']}:{payload['src_port']} -> "
                f"{payload['dst_ip']}:{payload['dst_port']} {payload['protocol']} "
                f"packets={payload['packets_count']} duration_ms={payload['flow_duration']:.3f} "
                f"label={payload['label']} confidence={payload['confidence']}"
            )
            self.send_alert(payload)
        except Exception as exc:
            print(f"[ERROR] Failed to finalize flow {key}: {exc}")

    def should_close_from_flags(self, packet):
        """Return True when TCP FIN or RST is present."""
        if not hasattr(packet, "tcp"):
            return False
        flags = parse_tcp_flags(packet)
        return bool(flags["fin"] or flags["rst"])

    def process_packet(self, packet):
        """Convert an individual packet into a flow record."""
        try:
            endpoints = packet_endpoints(packet)
            if not endpoints:
                return

            timestamp = safe_float(getattr(packet, "sniff_timestamp", time.time()), time.time())
            key, origin = self.canonical_key(endpoints)

            with self.lock:
                flow = self.flows.get(key)
                if flow is None:
                    flow = Flow(
                        key=key,
                        origin=origin,
                        display_endpoints=endpoints,
                        first_seen=timestamp,
                        last_seen=timestamp,
                    )
                    self.flows[key] = flow

            direction = self.packet_direction(flow, endpoints)
            protocol = endpoints[4]
            record = PacketRecord(
                timestamp=timestamp,
                direction=direction,
                length=packet_length(packet),
                header_length=header_length(packet, protocol),
                flags=parse_tcp_flags(packet) if protocol == "TCP" else {"fin": 0, "syn": 0, "rst": 0, "psh": 0, "ack": 0, "urg": 0},
                window_size=safe_int(get_layer_field(packet, "tcp", "window_size_value"), 0),
            )

            with self.lock:
                current_flow = self.flows.get(key)
                if current_flow is None:
                    return
                current_flow.add_packet(record)
                packet_total = current_flow.packet_count

            if packet_total >= FLOW_MAX_PACKETS:
                self.complete_flow(key, "max_packets")
            elif self.should_close_from_flags(packet):
                self.complete_flow(key, "tcp_fin_rst")
        except Exception as exc:
            print(f"[WARN] Packet processing error: {exc}")

    def cleanup_idle_flows(self):
        """Continuously expire flows that have been idle for too long."""
        while not self.stop_event.is_set():
            try:
                expired = []
                now = time.time()
                with self.lock:
                    for key, flow in list(self.flows.items()):
                        if now - flow.last_seen >= FLOW_IDLE_TIMEOUT_SECONDS:
                            expired.append(key)
                for key in expired:
                    self.complete_flow(key, "idle_timeout")
            except Exception as exc:
                print(f"[WARN] Idle cleanup error: {exc}")
            self.stop_event.wait(1.0)

    def capture_packets(self):
        """Run the blocking pyshark sniff loop in a background thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.interface = detect_active_interface()
            print(f"[INFO] Using interface: {self.interface}")
            self.capture = pyshark.LiveCapture(interface=self.interface, eventloop=loop)
            self.capture.apply_on_packets(self.process_packet)
        except Exception as exc:
            print(f"[ERROR] Live capture failed on Windows: {exc}")
            print("[HINT] Verify Npcap/TShark installation and run this script as Administrator.")
        finally:
            try:
                loop = asyncio.get_event_loop()
                loop.close()
            except Exception:
                pass

    def start(self):
        """Start the idle cleanup worker and the live packet capture thread."""
        if os.name == "nt" and not is_admin():
            raise PermissionError("capture.py must be run as Administrator on Windows.")

        cleanup_thread = threading.Thread(target=self.cleanup_idle_flows, daemon=True)
        capture_thread = threading.Thread(target=self.capture_packets, daemon=True)

        cleanup_thread.start()
        capture_thread.start()

        print("[INFO] CROSS-SECURE live capture started. Press Ctrl+C to stop.")

        try:
            while capture_thread.is_alive():
                capture_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping live capture...")
            self.stop_event.set()
            try:
                if self.capture is not None:
                    self.capture.close()
            except Exception as exc:
                print(f"[WARN] Failed to close capture cleanly: {exc}")

            with self.lock:
                remaining_keys = list(self.flows.keys())
            for key in remaining_keys:
                self.complete_flow(key, "shutdown")


if __name__ == "__main__":
    # Start the Windows live capture engine only when the script is run directly.
    try:
        engine = LiveTrafficCapture()
        engine.start()
    except Exception as exc:
        print(f"[FATAL] {exc}")
