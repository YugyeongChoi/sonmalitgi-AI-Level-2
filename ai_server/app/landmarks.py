"""Hands-only browser conversion. Pose is never read or added to model input."""
from typing import Optional
import numpy as np

LH, RH = 0, 21

def _hand_px(lms, width, height, mirrored):
    h = np.asarray(lms, np.float32)
    if h.ndim != 2 or h.shape[0] != 21 or h.shape[1] < 2:
        return None
    xy = np.stack([h[:, 0] * width, h[:, 1] * height], axis=1)
    if mirrored:
        xy[:, 0] = width - xy[:, 0]
    return xy


def _person_side_from_label(label: Optional[str], mirrored: bool) -> Optional[str]:
    """MediaPipe handedness 라벨 → 사람 기준 'L'/'R'. 원본 영상이면 라벨이 반대."""
    if not label:
        return None
    if label.lower() not in ("left", "right"):
        return None
    side = "L" if label.lower() == "left" else "R"
    if not mirrored:
        side = "R" if side == "L" else "L"
    return side


def mediapipe_to_hands42(hands: list, width: int, height: int,
                         mirrored: bool = False) -> np.ndarray:
    """Only hands, x/y/conf. MediaPipe z is depth and is not confidence."""
    out = np.full((42, 3), np.nan, np.float32)
    out[:, 2] = 0
    scores = {"L": -1., "R": -1.}
    for hand in hands or []:
        lm = hand["landmarks"] if isinstance(hand, dict) else hand.landmarks
        label = hand.get("handedness") if isinstance(hand, dict) else hand.handedness
        score = hand.get("score") if isinstance(hand, dict) else hand.score
        score = 1.0 if score is None else score
        xy = _hand_px(lm, width, height, mirrored)
        side = _person_side_from_label(label, mirrored)
        if xy is None or side is None or not np.isfinite(xy).all() or score <= scores[side]:
            continue
        scores[side] = score
        base = 0 if side == "L" else 21
        out[base:base + 21, :2] = xy
        out[base:base + 21, 2] = 1.
    return out


def presence(arr: np.ndarray, thr: float = .3):
    return False, bool(arr[0, 2] >= thr), bool(arr[21, 2] >= thr)
