from collections import deque
import math
from .model import Prediction

class Evaluator:
    """Thresholds are starting configuration, not validated accuracy guarantees."""
    def __init__(self, target: str, threshold: float = .9, stable: int = 3):
        self.target = target
        self.threshold = threshold
        self.history = deque(maxlen=stable)
        self.last = None

    def update(self, prediction: Prediction):
        p = prediction
        if not math.isfinite(p.confidence) or not 0 <= p.confidence <= 1:
            raise ValueError("Invalid model confidence")
        self.last = p
        if not p.capture_ok or not p.calibrated or p.predicted_id is None or p.confidence < self.threshold:
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
            "PASS": "목표 수어가 연속으로 인식됐어요.",
            "RETRY": "동작을 이어가거나 다시 표현해 주세요.",
            "FAIL": "다른 수어가 연속으로 인식됐어요. 기준 영상을 확인해 주세요.",
            "CAPTURE_ERROR": "촬영 상태를 확인해 주세요.",
        }[result]
        return dict(predicted_id=p.predicted_id, confidence=p.confidence,
                    result=result, feedback=p.diagnostic or feedback)
