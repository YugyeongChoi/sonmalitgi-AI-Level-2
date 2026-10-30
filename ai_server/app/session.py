"""WebSocket 연결 하나 = 연습 세션 하나. 프레임을 모았다가 end에서 판정한다."""
import numpy as np

from .landmarks import mediapipe_to_openpose54, presence
from .protocol import FrameMsg, StartMsg


class SessionError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


class PracticeSession:
    def __init__(self, registry, segmenter, settings):
        self.registry, self.segmenter, self.s = registry, segmenter, settings
        self.start_msg = None
        self.frames = []          # (54, 3) 배열 목록
        self.times = []

    @property
    def recording(self):
        return self.start_msg is not None

    def start(self, msg: StartMsg):
        if msg.lesson not in self.registry:
            raise SessionError("unknown_lesson", f"사용할 수 없는 레슨: {msg.lesson} "
                                                 f"(가능: {list(self.registry)})")
        rec = self.registry[msg.lesson]
        if msg.target and msg.target not in {c["id"] for c in rec.classes}:
            raise SessionError("unknown_target", f"모델에 없는 목표 단어: {msg.target}")
        self.start_msg, self.frames, self.times = msg, [], []
        return {"type": "started", "lesson": msg.lesson, "target": msg.target}

    def add_frame(self, msg: FrameMsg):
        """프레임 추가. status_every마다 실시간 상태 dict 반환, 아니면 None."""
        if not self.recording:
            raise SessionError("not_started", "start 전에 frame을 보냈습니다")
        if len(self.frames) >= self.s.max_frames:
            raise SessionError("too_long", f"최대 {self.s.max_frames}프레임을 넘었습니다. end를 보내세요")
        st = self.start_msg
        arr = mediapipe_to_openpose54(msg.pose, msg.hands, st.width, st.height, st.mirrored)
        self.frames.append(arr)
        self.times.append(msg.t)
        if len(self.frames) % self.s.status_every == 0:
            body, lh, rh = presence(arr)
            return {"type": "status", "frames": len(self.frames), "body": body,
                    "left_hand": lh, "right_hand": rh}
        return None

    def cancel(self):
        self.start_msg, self.frames, self.times = None, [], []

    def finish(self):
        """end 처리: 구간 → 인식 → 결과 dict. CPU 연산이라 호출 측에서 스레드로 돌린다."""
        if not self.recording:
            raise SessionError("not_started", "start 없이 end를 보냈습니다")
        st, frames, times = self.start_msg, self.frames, self.times
        self.cancel()
        if len(frames) < self.s.min_frames:
            raise SessionError("too_short", f"프레임이 {len(frames)}개뿐입니다 (최소 {self.s.min_frames})")

        arr = np.stack(frames)
        s, e = self.segmenter(arr)
        rec = self.registry[st.lesson]
        prob = rec.predict(arr[s:e])

        order = np.argsort(-prob)
        top = [{"id": rec.classes[i]["id"], "label": rec.classes[i]["label"],
                "prob": round(float(prob[i]), 4)} for i in order[: self.s.top_k]]
        result = {"type": "result", "lesson": st.lesson, "target": st.target, "top": top,
                  "frames": len(frames), "used_frames": [int(s), int(e)]}
        if st.target:
            ti = next(i for i, c in enumerate(rec.classes) if c["id"] == st.target)
            result["target_prob"] = round(float(prob[ti]), 4)
            result["correct"] = bool(order[0] == ti and prob[ti] >= self.s.target_threshold)

        # 진단용: 실제 fps, 손이 보인 프레임 비율 (학습 데이터와 입력 차이 확인)
        ts = [t for t in times if t is not None]
        if len(ts) >= 2 and ts[-1] > ts[0]:
            result["fps"] = round((len(ts) - 1) / ((ts[-1] - ts[0]) / 1000), 1)
        seg = arr[s:e]
        result["hand_visible"] = {"left": round(float((seg[:, 0, 2] > 0).mean()), 2),
                                  "right": round(float((seg[:, 21, 2] > 0).mean()), 2)}
        if self.s.save_dir:
            self._save(arr, result)
        return result

    def _save(self, arr, result):
        """KSL_SAVE_DIR 설정 시 시도마다 54점 배열(npy) + 결과(json) 저장 → 시각화·오답 분석용."""
        import json
        import time
        from pathlib import Path
        d = Path(self.s.save_dir)
        d.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d_%H%M%S')}_{result.get('target') or 'none'}"
        np.save(d / f"{name}.npy", arr)
        (d / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        result["saved"] = name
