from collections import deque
import math
from .model import Prediction

class Evaluator:
    """Thresholds are starting configuration, not validated accuracy guarantees."""
    def __init__(self, target: str, threshold: float = .9, stable: int = 3, require_calibrated: bool = True):
        self.require_calibrated = require_calibrated
        self.target = target
        self.threshold = threshold
        self.history = deque(maxlen=stable)
        self.last = None

    def update(self, prediction: Prediction):
        p = prediction
        if not math.isfinite(p.confidence) or not 0 <= p.confidence <= 1:
            raise ValueError("Invalid model confidence")
        self.last = p
        if not p.capture_ok or (self.require_calibrated and not p.calibrated) or p.predicted_id is None or p.confidence < self.threshold:
            self.history.clear()
        else:
            self.history.append(p.predicted_id)
        stable = (len(self.history) == self.history.maxlen
                  and len(set(self.history)) == 1)
        result = "RETRY"
        if not p.capture_ok:
            result = "CAPTURE_ERROR"
        elif stable:
            result = "PASS" if p.predicted_id == self.target else "FAIL"
        feedback = {
            "PASS": "목표 수어가 인식됐어요.",
            "RETRY": "동작을 이어가거나 다시 표현해 주세요.",
            "FAIL": "다른 수어가 인식됐어요. 기준 영상을 확인해 주세요.",
            "CAPTURE_ERROR": "촬영 상태를 확인해 주세요.",
        }[result]
        consecutive = 0
        for predicted in reversed(self.history):
            if predicted != self.target:
                break
            consecutive += 1
        return dict(stable_count=consecutive, stable_required=self.history.maxlen,
                    stable_target=self.target,
                    predicted_id=p.predicted_id, confidence=p.confidence,
                    result=result, feedback=p.diagnostic or feedback)


class CumulativeEvaluator:
    """중급의 정답 인식 횟수를 시도 안에서 누적한다. 연속 인식은 요구하지 않는다."""
    def __init__(self, target, threshold=.5, required=2):
        self.target, self.threshold, self.required = target, threshold, required
        self.success_count = 0
        self.armed = True
        self.rest_since = None
        self.previous_frame_time = None

    def finalize(self, reason, elapsed_seconds):
        from datetime import datetime, timezone
        result=self.progress()
        passed=self.success_count>=self.required
        result.update(result="PASS" if passed else "FAIL", waiting_for_reset=False,
                      termination_reason="passed" if passed else reason,
                      evaluation_seconds=round(elapsed_seconds,3),
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      evaluation_policy="middle-two-performances-20s-v1")
        return result

    def prepare_next(self):
        if self.success_count <= 0 or self.success_count >= self.required or self.armed:
            return False
        self.armed=True
        self.rest_since=None
        self.previous_frame_time=None
        return True

    def observe_frame(self, frame):
        if self.armed or self.success_count >= self.required: return False
        timestamp=frame.get('timestamp_ms')
        if not isinstance(timestamp,(int,float)) or not math.isfinite(timestamp) or not middle_rest_pose(frame):
            self.rest_since=None
            self.previous_frame_time=None
            return False
        if self.previous_frame_time is None or not 0 < timestamp-self.previous_frame_time <= 300:
            self.rest_since=timestamp
        self.previous_frame_time=timestamp
        if timestamp-self.rest_since >= 600:
            self.armed=True
            self.rest_since=None
            self.previous_frame_time=None
            return True
        return False

    def progress(self, prediction=None):
        return dict(predicted_id=prediction.predicted_id if prediction else None,
                    confidence=prediction.confidence if prediction else 0,
                    result="PASS" if self.success_count >= self.required else "RETRY",
                    feedback="", success_count=self.success_count, success_required=self.required,
                    count_mode="cumulative", waiting_for_reset=not self.armed and self.success_count<self.required)

    def update(self, prediction):
        p = prediction
        if not math.isfinite(p.confidence) or not 0 <= p.confidence <= 1:
            raise ValueError("Invalid model confidence")
        if self.armed and p.capture_ok and p.predicted_id == self.target and p.confidence >= self.threshold:
            self.success_count = min(self.required, self.success_count + 1)
            self.armed = False
        return self.progress(p)


def middle_rest_pose(frame):
    """손 검출 실패가 아닌, 보이는 어깨·양손목의 내려간 자세만 종료로 인정."""
    try:
        pose=frame['pose'];w,h=frame['image_width'],frame['image_height']
        if len(pose)!=33 or any(pose[i].get('visibility',0)<.5 for i in (11,12)): return False
        shoulder=math.hypot((pose[11]['x']-pose[12]['x'])*w,(pose[11]['y']-pose[12]['y'])*h)
        if shoulder<max(w,h)*.1:return False
        neck=(pose[11]['y']+pose[12]['y'])*h/2
        boundary=neck+.55*shoulder
        for key,index in [('left_hand',15),('right_hand',16)]:
            hand=frame.get(key,[])
            if len(hand)==21:
                wrist_y=hand[0]['y']*h
            elif pose[index].get('visibility',0)>=.5:
                wrist_y=pose[index]['y']*h
            else:
                return False
            if not math.isfinite(wrist_y) or wrist_y<boundary:return False
        return True
    except (KeyError,TypeError,ValueError):return False
