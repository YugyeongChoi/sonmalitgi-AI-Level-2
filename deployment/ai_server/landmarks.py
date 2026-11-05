"""Wire format for browser MediaPipe hand output (unmirrored coordinates)."""
import math

SCHEMA = "mediapipe-holistic-v1"
LEGACY_COUNTS = {"pose": 33, "face": 478, "left_hand": 21, "right_hand": 21}


def decode_landmarks(message, previous_timestamp=-1):
    schema = message.get("schema")
    if message.get("type") != "landmarks" or schema not in (SCHEMA, "mediapipe-hands-v1"):
        raise ValueError("Unknown landmark schema")
    timestamp = message.get("timestamp_ms")
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    if not finite(timestamp) or timestamp < 0 or timestamp <= previous_timestamp:
        raise ValueError("Landmark timestamps must increase")
    counts = LEGACY_COUNTS if schema == SCHEMA else {"left_hand": 21, "right_hand": 21}
    frame = {"schema": schema, "timestamp_ms": timestamp}
    for name, count in counts.items():
        points = message.get(name)
        if not isinstance(points, list) or len(points) not in (0, count):
            raise ValueError("Invalid landmark count")
        frame[name] = []
        for point in points:
            if not isinstance(point, dict) or not all(finite(point.get(k)) for k in ("x", "y", "z")):
                raise ValueError("Invalid landmark coordinates")
            clean = {k: point[k] for k in ("x", "y", "z")}
            if "visibility" in point:
                if not finite(point["visibility"]) or not 0 <= point["visibility"] <= 1:
                    raise ValueError("Invalid visibility")
                clean["visibility"] = point["visibility"]
            frame[name].append(clean)
    if "image_width" in message or "image_height" in message:
        for key in ("image_width", "image_height"):
            value = message.get(key)
            if type(value) is not int or not 1 <= value <= 8192:
                raise ValueError("Invalid camera dimensions")
            frame[key] = value
    return frame
