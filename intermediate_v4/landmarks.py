"""
MediaPipe 좌표 → 학습 데이터와 같은 54점(OpenPose 순서) 배열 (54, 3) = (x, y, conf).

학습 데이터(npy)의 54점: 왼손 21 + 오른손 21 + 상체 12 (BODY_25 idx 0,1,2,3,4,5,6,7,15,16,17,18)
  - 왼/오른 = 사람 기준
  - 좌표 = 원본 영상 픽셀 → 여기서도 정규화 좌표 × (width, height)로 픽셀로 바꿔 종횡비를 맞춤
    (데이터로더가 목 원점·어깨너비로 다시 정규화하므로 해상도 자체는 상관없음)

MediaPipe Pose 33점 → 상체 12점
  코 0 / 목 = (11+12)/2 / R 어깨·팔꿈치·손목 12·14·16 / L 11·13·15 / R눈 5 / L눈 2 / R귀 8 / L귀 7
MediaPipe Hand 21점은 OpenPose 손 21점과 순서가 같음 (손목, 엄지 1~4, 검지 5~8, 중지 9~12, 약지, 소지)

손 좌우 배정: MediaPipe Hands의 handedness 라벨은 "입력이 셀카(좌우반전)라고 가정"하고 붙여서
원본 영상에서는 반대로 나온다. 그래서 라벨보다 포즈의 손목 위치에 가까운 쪽으로 배정하고,
포즈가 없을 때만 라벨을 쓴다.
"""
from typing import List, Optional

import numpy as np

N_PTS = 54
LH, RH, BODY = 0, 21, 42
# 상체 12점의 MediaPipe 인덱스 (목은 None → 어깨 중점)
BODY_FROM_MP = [0, None, 12, 14, 16, 11, 13, 15, 5, 2, 8, 7]
MP_R_WRIST, MP_L_WRIST = 16, 15
# 좌우반전 입력일 때 서로 바꿔야 하는 포즈 인덱스 쌍 (L ↔ R)
MP_LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
               (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def _empty():
    a = np.full((N_PTS, 3), np.nan, np.float32)
    a[:, 2] = 0.0
    return a


def _pose_px(pose, width, height, mirrored):
    """(33, 4) [x, y, z, vis] 정규화 → (33, 3) [px, py, vis]. 좌우반전 입력이면 되돌림."""
    p = np.asarray(pose, np.float32)
    if p.ndim != 2 or p.shape[0] < 33 or p.shape[1] < 2:
        return None
    vis = p[:33, 3] if p.shape[1] >= 4 else np.ones(33, np.float32)
    out = np.stack([p[:33, 0] * width, p[:33, 1] * height, vis], axis=1)
    if mirrored:
        out[:, 0] = width - out[:, 0]
        for a, b in MP_LR_PAIRS:
            out[[a, b]] = out[[b, a]]
    return out


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
    side = "L" if label.lower().startswith("l") else "R"
    if not mirrored:
        side = "R" if side == "L" else "L"
    return side


def mediapipe_to_openpose54(pose: Optional[List[List[float]]], hands: list,
                            width: int, height: int, mirrored: bool = False) -> np.ndarray:
    """프레임 하나 변환. hands: [{"landmarks":[[x,y,z]×21], "handedness":"Left", "score":..}, ...]"""
    out = _empty()

    # ---- 상체
    pp = _pose_px(pose, width, height, mirrored) if pose else None
    if pp is not None:
        for k, j in enumerate(BODY_FROM_MP):
            if j is None:                                   # 목 = 어깨 중점
                pt = (pp[11] + pp[12]) / 2
                pt[2] = min(pp[11, 2], pp[12, 2])
            else:
                pt = pp[j]
            out[BODY + k] = pt

    # ---- 손
    parsed = []
    for h in hands or []:
        lm = h["landmarks"] if isinstance(h, dict) else h.landmarks
        label = h.get("handedness") if isinstance(h, dict) else h.handedness
        xy = _hand_px(lm, width, height, mirrored)
        if xy is not None:
            parsed.append((xy, _person_side_from_label(label, mirrored)))
    parsed = parsed[:2]

    sides = [s for _, s in parsed]
    if pp is not None and parsed:
        wr = {"L": pp[MP_L_WRIST, :2], "R": pp[MP_R_WRIST, :2]}
        d = [{s: float(np.linalg.norm(xy[0] - wr[s])) for s in "LR"} for xy, _ in parsed]
        if len(parsed) == 1:
            sides = [min(d[0], key=d[0].get)]
        else:                                               # 두 손: 거리 합이 작은 배정
            sides = ["L", "R"] if d[0]["L"] + d[1]["R"] <= d[0]["R"] + d[1]["L"] else ["R", "L"]
    used = set()
    for (xy, _), side in zip(parsed, sides):
        if side is None or side in used:
            continue
        used.add(side)
        base = LH if side == "L" else RH
        out[base:base + 21, :2] = xy
        out[base:base + 21, 2] = 1.0                        # MediaPipe 손은 점별 신뢰도 없음
    return out


def presence(arr54: np.ndarray, thr: float = 0.3):
    """실시간 상태용: (몸, 왼손, 오른손) 보임 여부."""
    c = arr54[:, 2]
    return (bool((c[BODY + 1:BODY + 8] >= thr).mean() >= 0.5),
            bool(c[LH] >= thr), bool(c[RH] >= thr))
