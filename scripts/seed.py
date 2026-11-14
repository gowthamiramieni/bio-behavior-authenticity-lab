"""
Synthetic data generator for BioAuth Lab.
Generates realistic keystroke, mouse, and touch biometric raw data
for 20 distinct users (genuine and impostor samples), extracts features,
and populates the SQLite database.
"""
import sys
import json
import uuid
import random
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.db import init_db, get_session, User, Sample
from core.auth import hash_password
from core.features import extract_features_for_modality

TARGET_PHRASE = "bioauth secure access protocol"
TARGET_POINTS = [(120, 150), (450, 220), (280, 500)]


def generate_user_archetypes(n_users: int = 20):
    """Generate distinct behavioural archetypes for each user."""
    np.random.seed(42)
    random.seed(42)

    archetypes = {}
    for i in range(1, n_users + 1):
        username = f"user_{i:02d}"

        # Keystroke archetype
        keystroke_profile = {
            "mean_dwell": float(np.random.uniform(75.0, 155.0)),
            "std_dwell": float(np.random.uniform(10.0, 25.0)),
            "mean_flight": float(np.random.uniform(120.0, 240.0)),
            "std_flight": float(np.random.uniform(15.0, 40.0)),
            "digraph_boost": {
                ch: float(np.random.uniform(-20.0, 25.0))
                for ch in set(TARGET_PHRASE)
            }
        }

        # Mouse archetype
        mouse_profile = {
            "base_speed": float(np.random.uniform(320.0, 850.0)),   # px/s
            "speed_std": float(np.random.uniform(35.0, 110.0)),
            "curvature": float(np.random.uniform(0.05, 0.35)),     # deviation from straight line
            "jitter": float(np.random.uniform(0.5, 3.5)),           # micro-hand tremors
            "click_duration": float(np.random.uniform(65.0, 140.0)),
            "click_duration_std": float(np.random.uniform(8.0, 20.0)),
        }

        # Touch archetype
        touch_profile = {
            "mean_speed": float(np.random.uniform(450.0, 1200.0)),  # px/s
            "mean_pressure": float(np.random.uniform(0.38, 0.88)),
            "pressure_var": float(np.random.uniform(0.04, 0.12)),
            "stroke_len": float(np.random.uniform(160.0, 360.0)),   # px
            "wobble": float(np.random.uniform(0.02, 0.18)),
            "angle_deg": float(np.random.uniform(25.0, 65.0)),      # typical swipe angle
        }

        archetypes[username] = {
            "keystroke": keystroke_profile,
            "mouse": mouse_profile,
            "touch": touch_profile,
        }
    return archetypes


def simulate_keystroke_raw(profile: dict, is_impostor: bool = False, phrase: str = TARGET_PHRASE) -> list:
    """Simulate keydown and keyup events for the phrase."""
    # If impostor, distort the baseline timing
    mean_dwell = profile["mean_dwell"] * (random.uniform(0.65, 1.45) if is_impostor else 1.0)
    std_dwell = profile["std_dwell"] * (random.uniform(1.3, 2.2) if is_impostor else 1.0)
    mean_flight = profile["mean_flight"] * (random.uniform(0.60, 1.50) if is_impostor else 1.0)
    std_flight = profile["std_flight"] * (random.uniform(1.4, 2.5) if is_impostor else 1.0)

    events = []
    current_time = 100.0 + random.uniform(0, 50.0)

    for i, char in enumerate(phrase):
        # Calculate dwell for this key
        char_boost = profile["digraph_boost"].get(char, 0.0) if not is_impostor else 0.0
        dwell = max(35.0, np.random.normal(mean_dwell + char_boost, std_dwell))
        
        t_down = current_time
        t_up = t_down + dwell

        events.append({"key": char, "event": "keydown", "timestamp": round(t_down, 2)})
        events.append({"key": char, "event": "keyup", "timestamp": round(t_up, 2)})

        # Flight time to next key
        flight = max(40.0, np.random.normal(mean_flight, std_flight))
        # Sequential next keydown timestamp
        current_time = t_down + flight

    # Sort events by timestamp as real keyboard event stream
    events = sorted(events, key=lambda x: x["timestamp"])
    return events


def simulate_mouse_raw(profile: dict, is_impostor: bool = False) -> dict:
    """Simulate mouse trajectory and clicks across target points."""
    base_speed = profile["base_speed"] * (random.uniform(0.6, 1.5) if is_impostor else 1.0)
    curvature = profile["curvature"] * (random.uniform(1.5, 3.0) if is_impostor else 1.0)
    jitter = profile["jitter"] * (random.uniform(2.0, 4.0) if is_impostor else 1.0)
    click_mean = profile["click_duration"] * (random.uniform(0.7, 1.4) if is_impostor else 1.0)
    click_std = profile["click_duration_std"] * (random.uniform(1.2, 2.0) if is_impostor else 1.0)

    trajectory = []
    clicks = []
    current_t = 100.0

    # Start point
    cur_x, cur_y = 50.0 + random.uniform(-10, 10), 80.0 + random.uniform(-10, 10)
    trajectory.append({"x": round(cur_x, 1), "y": round(cur_y, 1), "t": round(current_t, 1)})

    for target_x, target_y in TARGET_POINTS:
        # Move from (cur_x, cur_y) to (target_x, target_y)
        dist = float(np.hypot(target_x - cur_x, target_y - cur_y))
        duration_sec = max(0.15, dist / base_speed)
        steps = max(12, int(duration_sec * 60))  # 60 Hz sample rate

        # Perpendicular normal vector for curvature
        nx = -(target_y - cur_y) / (dist + 1e-5)
        ny = (target_x - cur_x) / (dist + 1e-5)
        arc_offset = curvature * dist * (1 if random.random() > 0.5 else -1)

        dt_step = (duration_sec * 1000.0) / steps
        for step_i in range(1, steps + 1):
            s = step_i / steps
            # Smooth step bell curve (Hermite / easeInOut)
            alpha = 3 * s**2 - 2 * s**3
            # Baseline linear interpolation
            interp_x = cur_x + (target_x - cur_x) * alpha
            interp_y = cur_y + (target_y - cur_y) * alpha

            # Add curve displacement and random jitter
            curve_disp = np.sin(np.pi * s) * arc_offset
            j_x = np.random.normal(0, jitter)
            j_y = np.random.normal(0, jitter)

            pt_x = interp_x + nx * curve_disp + j_x
            pt_y = interp_y + ny * curve_disp + j_y
            current_t += dt_step + np.random.normal(0, 1.0)

            trajectory.append({
                "x": round(float(pt_x), 1),
                "y": round(float(pt_y), 1),
                "t": round(float(current_t), 1)
            })

        # Click event at the target
        click_dur = max(30.0, np.random.normal(click_mean, click_std))
        t_click_down = current_t + 10.0
        t_click_up = t_click_down + click_dur
        current_t = t_click_up + 25.0

        clicks.append({
            "x": round(target_x, 1),
            "y": round(target_y, 1),
            "t_down": round(t_click_down, 1),
            "t_up": round(t_click_up, 1)
        })

        cur_x, cur_y = target_x, target_y

    return {"trajectory": trajectory, "clicks": clicks}


def simulate_touch_raw(profile: dict, is_impostor: bool = False) -> dict:
    """Simulate a swipe gesture stroke."""
    speed = profile["mean_speed"] * (random.uniform(0.6, 1.5) if is_impostor else 1.0)
    length = profile["stroke_len"] * (random.uniform(0.7, 1.3) if is_impostor else 1.0)
    wobble = profile["wobble"] * (random.uniform(1.8, 3.5) if is_impostor else 1.0)
    mean_pres = profile["mean_pressure"] * (random.uniform(0.7, 1.3) if is_impostor else 1.0)
    mean_pres = min(max(mean_pres, 0.2), 0.95)
    pres_var = profile["pressure_var"]

    angle_rad = np.radians(profile["angle_deg"] + (random.uniform(-25, 25) if is_impostor else random.uniform(-4, 4)))
    total_dur_sec = max(0.12, length / speed)
    total_dur_ms = total_dur_sec * 1000.0

    steps = max(10, int(total_dur_sec * 60))
    start_x, start_y = 150.0 + random.uniform(-20, 20), 400.0 + random.uniform(-20, 20)
    end_x = start_x + length * np.cos(angle_rad)
    end_y = start_y - length * np.sin(angle_rad)  # swipe upwards

    stroke = []
    t_start = 50.0

    for step_i in range(steps + 1):
        s = step_i / steps
        # Natural hand acceleration-deceleration
        alpha = 3 * s**2 - 2 * s**3
        base_x = start_x + (end_x - start_x) * alpha
        base_y = start_y + (end_y - start_y) * alpha

        # Perpendicular wobble
        perp_dx = -(end_y - start_y) / (length + 1e-5)
        perp_dy = (end_x - start_x) / (length + 1e-5)
        w_offset = np.sin(2 * np.pi * s) * wobble * length + np.random.normal(0, 1.5)

        px = base_x + perp_dx * w_offset
        py = base_y + perp_dy * w_offset
        pt_t = t_start + s * total_dur_ms

        # Pressure forms an inverted parabolic envelope (light at start/end, heavy in middle)
        p_env = np.sin(np.pi * s)
        pressure = np.clip(mean_pres * p_env + np.random.normal(0, pres_var), 0.05, 1.0)

        stroke.append({
            "x": round(float(px), 1),
            "y": round(float(py), 1),
            "t": round(float(pt_t), 1),
            "pressure": round(float(pressure), 3)
        })

    return {"stroke": stroke}


def seed_database(num_users: int = 20, genuine_per_modality: int = 15, impostors_per_modality: int = 15):
    """Seed SQLite database with users and biometric samples."""
    print(f"[*] Initializing database schema...")
    init_db()
    session = get_session()

    # Clear existing data if any
    session.query(Sample).delete()
    session.query(User).delete()
    session.commit()

    print(f"[*] Generating {num_users} users and archetypes...")
    archetypes = generate_user_archetypes(num_users)
    users_dict = {}

    default_pw_hash = hash_password("pass1234")

    for uname in archetypes.keys():
        user = User(username=uname, password_hash=default_pw_hash)
        session.add(user)
        session.flush()
        users_dict[uname] = user

    user_names = list(archetypes.keys())
    print(f"[+] Created {len(users_dict)} users.")

    modalities = ["keystroke", "mouse", "touch"]
    total_samples = 0

    print("[*] Generating synthetic genuine and impostor biometric samples...")

    for i, uname in enumerate(user_names):
        user_obj = users_dict[uname]
        profile = archetypes[uname]

        for mod in modalities:
            # 1. Genuine samples
            for g_idx in range(genuine_per_modality):
                if mod == "keystroke":
                    raw = simulate_keystroke_raw(profile["keystroke"], is_impostor=False)
                elif mod == "mouse":
                    raw = simulate_mouse_raw(profile["mouse"], is_impostor=False)
                else:  # touch
                    raw = simulate_touch_raw(profile["touch"], is_impostor=False)

                feats = extract_features_for_modality(mod, raw)
                sample = Sample(
                    user_id=user_obj.id,
                    modality=mod,
                    raw_data=json.dumps(raw),
                    features=json.dumps(feats),
                    label="genuine",
                    session_id=str(uuid.uuid4())[:8],
                )
                session.add(sample)
                total_samples += 1

            # 2. Impostor samples
            # Pick other users to act as impostors trying to impersonate this user
            other_users = [u for u in user_names if u != uname]
            for imp_idx in range(impostors_per_modality):
                impostor_uname = random.choice(other_users)
                imp_profile = archetypes[impostor_uname]

                if mod == "keystroke":
                    raw = simulate_keystroke_raw(imp_profile["keystroke"], is_impostor=True)
                elif mod == "mouse":
                    raw = simulate_mouse_raw(imp_profile["mouse"], is_impostor=True)
                else:
                    raw = simulate_touch_raw(imp_profile["touch"], is_impostor=True)

                feats = extract_features_for_modality(mod, raw)
                sample = Sample(
                    user_id=user_obj.id,
                    modality=mod,
                    raw_data=json.dumps(raw),
                    features=json.dumps(feats),
                    label="impostor",
                    session_id=str(uuid.uuid4())[:8],
                )
                session.add(sample)
                total_samples += 1

        if (i + 1) % 5 == 0 or (i + 1) == num_users:
            session.commit()
            print(f"    Progress: {i + 1}/{num_users} users seeded ({total_samples} samples so far)...")

    session.commit()

    # Verification query
    user_count = session.query(User).count()
    sample_count = session.query(Sample).count()
    keystroke_count = session.query(Sample).filter_by(modality="keystroke").count()
    mouse_count = session.query(Sample).filter_by(modality="mouse").count()
    touch_count = session.query(Sample).filter_by(modality="touch").count()
    genuine_count = session.query(Sample).filter_by(label="genuine").count()
    impostor_count = session.query(Sample).filter_by(label="impostor").count()

    print("\n" + "=" * 50)
    print("DATABASE SEEDING SUMMARY")
    print("=" * 50)
    print(f"Total Users:      {user_count}")
    print(f"Total Samples:    {sample_count}")
    print(f"  - Keystroke:    {keystroke_count}")
    print(f"  - Mouse:        {mouse_count}")
    print(f"  - Touch:        {touch_count}")
    print(f"  - Genuine:      {genuine_count}")
    print(f"  - Impostor:     {impostor_count}")
    print(f"Database Path:    {PROJECT_ROOT / 'data' / 'bioauth.db'}")
    print("=" * 50)

    session.close()


if __name__ == "__main__":
    seed_database(num_users=20, genuine_per_modality=15, impostors_per_modality=15)
