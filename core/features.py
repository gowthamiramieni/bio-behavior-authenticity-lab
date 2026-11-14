"""
Feature extraction algorithms for behavioural biometrics:
Keystroke Dynamics, Mouse Dynamics, and Touch/Swipe Gestures.
"""
from typing import List, Dict, Any
import numpy as np


# ---------------------------------------------------------
# 1. KEYSTROKE DYNAMICS
# ---------------------------------------------------------
def extract_keystroke_features(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extracts keystroke timing features from raw key events.
    Expected format: list of dicts with:
      - 'key': character or key name
      - 'event': 'keydown' or 'keyup'
      - 'timestamp': timestamp in milliseconds
    """
    if not events or len(events) < 4:
        return _empty_keystroke_features()

    # Sort events chronologically by timestamp
    sorted_events = sorted(events, key=lambda x: x["timestamp"])

    # Match keydown with corresponding keyup to calculate dwell times
    dwell_times = []
    keydown_dict = {}  # key -> list of down timestamps

    for ev in sorted_events:
        k = ev.get("key", "")
        t = float(ev.get("timestamp", 0))
        etype = ev.get("event")

        if etype == "keydown":
            if k not in keydown_dict:
                keydown_dict[k] = []
            keydown_dict[k].append(t)
        elif etype == "keyup":
            if k in keydown_dict and len(keydown_dict[k]) > 0:
                t_down = keydown_dict[k].pop(0)
                dwell = t - t_down
                if 0 <= dwell <= 3000:  # sanity check: 0 to 3 seconds
                    dwell_times.append(dwell)

    # Order of all keydown events for sequential flight times
    keydown_events = [ev for ev in sorted_events if ev.get("event") == "keydown"]
    keyup_events = [ev for ev in sorted_events if ev.get("event") == "keyup"]

    flight_up_down = []
    flight_down_down = []

    # Flight down-down: t_down[i] - t_down[i-1]
    for i in range(1, len(keydown_events)):
        dd = float(keydown_events[i]["timestamp"]) - float(keydown_events[i - 1]["timestamp"])
        if 0 < dd <= 5000:
            flight_down_down.append(dd)

    # Flight up-down: t_down[i] - t_up[i-1] (for sequential keypresses)
    # Match the preceding keyup before each keydown
    for i in range(1, len(keydown_events)):
        t_down_curr = float(keydown_events[i]["timestamp"])
        # Find latest keyup that occurred before t_down_curr
        prev_ups = [ev["timestamp"] for ev in keyup_events if ev["timestamp"] <= t_down_curr]
        if prev_ups:
            ud = t_down_curr - float(prev_ups[-1])
            if -500 <= ud <= 5000:
                flight_up_down.append(ud)

    # Total typing session duration and speed
    t_start = float(sorted_events[0]["timestamp"])
    t_end = float(sorted_events[-1]["timestamp"])
    total_duration_sec = max((t_end - t_start) / 1000.0, 0.05)
    char_count = len(keydown_events)
    typing_speed_cps = char_count / total_duration_sec  # characters per second
    typing_speed_wpm = (char_count / 5.0) / (total_duration_sec / 60.0)  # standard WPM

    dwell_arr = np.array(dwell_times) if dwell_times else np.array([100.0])
    fud_arr = np.array(flight_up_down) if flight_up_down else np.array([80.0])
    fdd_arr = np.array(flight_down_down) if flight_down_down else np.array([180.0])

    features = {
        "dwell_mean": float(np.mean(dwell_arr)),
        "dwell_std": float(np.std(dwell_arr)) if len(dwell_arr) > 1 else 0.0,
        "dwell_median": float(np.median(dwell_arr)),
        "flight_ud_mean": float(np.mean(fud_arr)),
        "flight_ud_std": float(np.std(fud_arr)) if len(fud_arr) > 1 else 0.0,
        "flight_ud_median": float(np.median(fud_arr)),
        "flight_dd_mean": float(np.mean(fdd_arr)),
        "flight_dd_std": float(np.std(fdd_arr)) if len(fdd_arr) > 1 else 0.0,
        "flight_dd_median": float(np.median(fdd_arr)),
        "typing_speed_cps": float(typing_speed_cps),
        "typing_speed_wpm": float(typing_speed_wpm),
        "total_duration_sec": float(total_duration_sec),
    }
    return features


def _empty_keystroke_features() -> Dict[str, float]:
    return {
        "dwell_mean": 0.0,
        "dwell_std": 0.0,
        "dwell_median": 0.0,
        "flight_ud_mean": 0.0,
        "flight_ud_std": 0.0,
        "flight_ud_median": 0.0,
        "flight_dd_mean": 0.0,
        "flight_dd_std": 0.0,
        "flight_dd_median": 0.0,
        "typing_speed_cps": 0.0,
        "typing_speed_wpm": 0.0,
        "total_duration_sec": 0.0,
    }


# ---------------------------------------------------------
# 2. MOUSE DYNAMICS
# ---------------------------------------------------------
def extract_mouse_features(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts mouse movement and click features.
    Expected format: dict with:
      - 'trajectory': list of {'x': float, 'y': float, 't': float}
      - 'clicks': list of {'x': float, 'y': float, 't_down': float, 't_up': float}
    """
    trajectory = data.get("trajectory", [])
    clicks = data.get("clicks", [])

    if len(trajectory) < 3:
        return _empty_mouse_features()

    coords = [(float(p["x"]), float(p["y"]), float(p["t"])) for p in trajectory]
    # Remove consecutive duplicate timestamps to prevent zero division
    clean_coords = [coords[0]]
    for p in coords[1:]:
        if p[2] > clean_coords[-1][2]:
            clean_coords.append(p)

    if len(clean_coords) < 3:
        return _empty_mouse_features()

    xs = np.array([p[0] for p in clean_coords])
    ys = np.array([p[1] for p in clean_coords])
    ts = np.array([p[2] for p in clean_coords])

    # Inter-point distances and time deltas
    dxs = np.diff(xs)
    dys = np.diff(ys)
    dts = np.diff(ts) / 1000.0  # seconds
    dts = np.where(dts <= 0, 1e-4, dts)

    distances = np.hypot(dxs, dys)
    total_path_length = float(np.sum(distances))

    # Euclidean distance between start and end point
    direct_distance = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    path_efficiency = float(direct_distance / total_path_length) if total_path_length > 1e-4 else 1.0
    path_efficiency = min(max(path_efficiency, 0.0), 1.0)

    # Speeds (px/s)
    speeds = distances / dts
    speed_mean = float(np.mean(speeds))
    speed_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0
    speed_max = float(np.max(speeds))

    # Accelerations (px/s^2)
    if len(speeds) > 1:
        dt_acc = (dts[:-1] + dts[1:]) / 2.0
        dt_acc = np.where(dt_acc <= 0, 1e-4, dt_acc)
        accels = np.diff(speeds) / dt_acc
        accel_mean = float(np.mean(np.abs(accels)))
        accel_std = float(np.std(accels))
        accel_max = float(np.max(np.abs(accels)))
    else:
        accels = np.array([0.0])
        accel_mean, accel_std, accel_max = 0.0, 0.0, 0.0

    # Jerk (px/s^3)
    if len(accels) > 1:
        dt_jerk = (dt_acc[:-1] + dt_acc[1:]) / 2.0
        dt_jerk = np.where(dt_jerk <= 0, 1e-4, dt_jerk)
        jerks = np.diff(accels) / dt_jerk
        jerk_mean = float(np.mean(np.abs(jerks)))
        jerk_std = float(np.std(jerks))
    else:
        jerk_mean, jerk_std = 0.0, 0.0

    # Angle changes (curvature / angular velocity)
    angles = np.arctan2(dys, dxs)
    if len(angles) > 1:
        angle_diffs = np.diff(angles)
        # Wrap angle to [-pi, pi]
        angle_diffs = np.abs((angle_diffs + np.pi) % (2 * np.pi) - np.pi)
        angle_change_mean = float(np.mean(angle_diffs))
        angle_change_sum = float(np.sum(angle_diffs))
    else:
        angle_change_mean, angle_change_sum = 0.0, 0.0

    # Click durations
    click_durations = []
    for c in clicks:
        t_down = float(c.get("t_down", 0))
        t_up = float(c.get("t_up", 0))
        duration = t_up - t_down
        if 0 < duration < 3000:
            click_durations.append(duration)

    click_arr = np.array(click_durations) if click_durations else np.array([100.0])
    click_dur_mean = float(np.mean(click_arr))
    click_dur_std = float(np.std(click_arr)) if len(click_arr) > 1 else 0.0

    features = {
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "speed_max": speed_max,
        "accel_mean": accel_mean,
        "accel_std": accel_std,
        "accel_max": accel_max,
        "jerk_mean": jerk_mean,
        "jerk_std": jerk_std,
        "path_efficiency": path_efficiency,
        "total_path_length": total_path_length,
        "angle_change_mean": angle_change_mean,
        "angle_change_sum": angle_change_sum,
        "click_duration_mean": click_dur_mean,
        "click_duration_std": click_dur_std,
    }
    return features


def _empty_mouse_features() -> Dict[str, float]:
    return {
        "speed_mean": 0.0,
        "speed_std": 0.0,
        "speed_max": 0.0,
        "accel_mean": 0.0,
        "accel_std": 0.0,
        "accel_max": 0.0,
        "jerk_mean": 0.0,
        "jerk_std": 0.0,
        "path_efficiency": 0.0,
        "total_path_length": 0.0,
        "angle_change_mean": 0.0,
        "angle_change_sum": 0.0,
        "click_duration_mean": 0.0,
        "click_duration_std": 0.0,
    }


# ---------------------------------------------------------
# 3. TOUCH / SWIPE GESTURES
# ---------------------------------------------------------
def extract_touch_features(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts swipe gesture features.
    Expected format: dict with:
      - 'stroke': list of {'x': float, 'y': float, 't': float, 'pressure': float}
    """
    stroke = data.get("stroke", [])
    if len(stroke) < 3:
        return _empty_touch_features()

    coords = [(float(p["x"]), float(p["y"]), float(p["t"]), float(p.get("pressure", 0.5))) for p in stroke]
    clean_coords = [coords[0]]
    for p in coords[1:]:
        if p[2] > clean_coords[-1][2]:
            clean_coords.append(p)

    if len(clean_coords) < 3:
        return _empty_touch_features()

    xs = np.array([p[0] for p in clean_coords])
    ys = np.array([p[1] for p in clean_coords])
    ts = np.array([p[2] for p in clean_coords])
    pressures = np.array([p[3] for p in clean_coords])

    # Inter-point distances and times
    dxs = np.diff(xs)
    dys = np.diff(ys)
    dts = np.diff(ts) / 1000.0  # seconds
    dts = np.where(dts <= 0, 1e-4, dts)

    distances = np.hypot(dxs, dys)
    total_length = float(np.sum(distances))

    # Total duration (ms)
    duration_ms = float(ts[-1] - ts[0])
    duration_sec = max(duration_ms / 1000.0, 1e-4)

    # Average velocity (px/s)
    velocity_mean = float(total_length / duration_sec)
    instant_velocities = distances / dts
    velocity_max = float(np.max(instant_velocities))
    velocity_std = float(np.std(instant_velocities)) if len(instant_velocities) > 1 else 0.0

    # Straightness: euclidean distance / total path length
    direct_distance = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    straightness = float(direct_distance / total_length) if total_length > 1e-4 else 1.0
    straightness = min(max(straightness, 0.0), 1.0)

    # Pressure stats
    pressure_mean = float(np.mean(pressures))
    pressure_std = float(np.std(pressures)) if len(pressures) > 1 else 0.0
    pressure_median = float(np.median(pressures))
    pressure_max = float(np.max(pressures))

    features = {
        "velocity_mean": velocity_mean,
        "velocity_max": velocity_max,
        "velocity_std": velocity_std,
        "pressure_mean": pressure_mean,
        "pressure_std": pressure_std,
        "pressure_median": pressure_median,
        "pressure_max": pressure_max,
        "straightness": straightness,
        "duration_ms": duration_ms,
        "stroke_length": total_length,
    }
    return features


def _empty_touch_features() -> Dict[str, float]:
    return {
        "velocity_mean": 0.0,
        "velocity_max": 0.0,
        "velocity_std": 0.0,
        "pressure_mean": 0.0,
        "pressure_std": 0.0,
        "pressure_median": 0.0,
        "pressure_max": 0.0,
        "straightness": 0.0,
        "duration_ms": 0.0,
        "stroke_length": 0.0,
    }


def extract_features_for_modality(modality: str, raw_data: Any) -> Dict[str, float]:
    """Unified entry point to extract features based on modality name."""
    if modality == "keystroke":
        # raw_data is either list of key events or dict with 'events'
        events = raw_data.get("events", raw_data) if isinstance(raw_data, dict) else raw_data
        return extract_keystroke_features(events)
    elif modality == "mouse":
        return extract_mouse_features(raw_data if isinstance(raw_data, dict) else {})
    elif modality == "touch":
        return extract_touch_features(raw_data if isinstance(raw_data, dict) else {})
    else:
        raise ValueError(f"Unknown modality: {modality}")
