"""
Streamlit Custom Components for Behavioural Biometric Data Capture.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent

# Declare the 3 custom biometric capture components
_keystroke_component = components.declare_component(
    "keystroke_capture",
    path=str(BASE_DIR / "keystroke")
)

_mouse_component = components.declare_component(
    "mouse_capture",
    path=str(BASE_DIR / "mouse")
)

_touch_component = components.declare_component(
    "touch_capture",
    path=str(BASE_DIR / "touch")
)


def capture_keystrokes(target_phrase: str = "bioauth secure access protocol", key: Optional[str] = "keystroke_comp") -> Optional[Dict[str, Any]]:
    """
    Renders browser keystroke dynamics recorder.
    Returns dictionary with raw events on submission, or None.
    """
    return _keystroke_component(targetPhrase=target_phrase, key=key, default=None)


def capture_mouse(key: Optional[str] = "mouse_comp") -> Optional[Dict[str, Any]]:
    """
    Renders browser mouse pointer trajectory and target click tracker.
    Returns dictionary with raw trajectory and clicks on completion, or None.
    """
    return _mouse_component(key=key, default=None)


def capture_touch(key: Optional[str] = "touch_comp") -> Optional[Dict[str, Any]]:
    """
    Renders browser touch/swipe pointer pad.
    Returns dictionary with raw swipe stroke points and pressures, or None.
    """
    return _touch_component(key=key, default=None)
