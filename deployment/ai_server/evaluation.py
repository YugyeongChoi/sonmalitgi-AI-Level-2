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
    def __init__(self, target, threshold=.5, required=3):
        self.target, self.threshold, self.required = target, threshold, required
        self.success_count = 0

    def update(self, prediction):
        p = prediction
        if not math.isfinite(p.confidence) or not 0 <= p.confidence <= 1:
            raise ValueError("Invalid model confidence")
        if p.capture_ok and p.predicted_id == self.target and p.confidence >= self.threshold:
            self.success_count = min(self.required, self.success_count + 1)
        return dict(predicted_id=p.predicted_id, confidence=p.confidence,
                    result="PASS" if self.success_count >= self.required else "RETRY",
                    feedback="", success_count=self.success_count, success_required=self.required,
                    count_mode="cumulative")
