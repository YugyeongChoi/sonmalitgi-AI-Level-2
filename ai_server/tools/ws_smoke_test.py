#!/usr/bin/env python3
"""
서버 동작 확인: 학습 데이터 npy 클립을 "브라우저가 보내는 MediaPipe 형식"으로 바꿔 WebSocket으로 보낸다.

웹캠 없이 서버 전체 경로(좌표 변환 → 구간 → 전처리 → 모델)를 검증할 수 있다.
같은 클립을 train_word_lstm.py의 val 예측과 비교하면 서버와 학습 코드가 같은 결과를 내는지 확인된다.
(목 좌표만 어깨 중점으로 바뀌므로 확률은 약간 다를 수 있음)

사용 예 (서버 실행 중에):
  python tools/ws_smoke_test.py --npy ../data/word/clips/NIA_SL_WORD1157_REAL02_F.npy --target WORD1157
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.landmarks import BODY, BODY_FROM_MP, LH, RH  # noqa: E402

W, H = 1920, 1080   # 정규화 기준. 서버가 같은 값으로 되돌리므로 원래 해상도와 같을 필요 없음


def openpose54_to_mediapipe(fr):
    """(54, 3) → (pose 33×4, hands). MediaPipe가 원본 영상에서 낼 법한 형태로."""
    pose = np.zeros((33, 4), np.float32)
    for k, j in enumerate(BODY_FROM_MP):
        if j is None:
            continue
        x, y, c = fr[BODY + k]
        if np.isfinite(x) and c > 0:
            pose[j] = [x / W, y / H, 0.0, 1.0]
    hands = []
    for base, label in [(RH, "Left"), (LH, "Right")]:   # 원본 영상에서 MediaPipe 라벨은 사람 기준과 반대
        h = fr[base:base + 21]
        if np.isfinite(h[:, 0]).all() and (h[:, 2] > 0).all():
            hands.append({"landmarks": [[float(x) / W, float(y) / H, 0.0] for x, y in h[:, :2]],
                          "handedness": label, "score": 1.0})
    return pose.tolist(), hands


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--target", default=None, help="목표 WORD ID (정답 판정 확인용)")
    ap.add_argument("--url", default="ws://localhost:8000/ws/practice")
    args = ap.parse_args()

    from websockets.sync.client import connect

    arr = np.load(args.npy)
    with connect(args.url) as ws:
        print("<", ws.recv())
        ws.send(json.dumps({"type": "start", "lesson": "word", "target": args.target,
                            "width": W, "height": H, "mirrored": False}))
        print("<", ws.recv())
        for t, fr in enumerate(arr):
            pose, hands = openpose54_to_mediapipe(fr)
            ws.send(json.dumps({"type": "frame", "t": t * 33.3, "pose": pose, "hands": hands}))
        ws.send(json.dumps({"type": "end"}))
        while True:
            m = json.loads(ws.recv())
            if m["type"] in ("result", "error"):
                print("<", json.dumps(m, ensure_ascii=False, indent=1))
                break


if __name__ == "__main__":
    main()
