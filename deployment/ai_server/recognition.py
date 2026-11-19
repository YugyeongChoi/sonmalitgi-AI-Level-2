"""Target-free recognition policy. Scores are model scores, not calibrated accuracy."""
import math

class RecognitionState:
    def __init__(self, supported_ids, threshold=.8, stable=2):
        self.supported = set(supported_ids)
        self.threshold, self.stable = threshold, stable
        self.reset()

    def reset(self):
        self.candidate, self.count = None, 0

    def update(self, prediction):
        score = prediction.confidence
        valid = (prediction.capture_ok and prediction.predicted_id in self.supported
                 and math.isfinite(score) and self.threshold <= score <= 1)
        if not valid:
            self.reset()
            return dict(predicted_id=None, confidence=0, stable=False)
        self.count = self.count + 1 if self.candidate == prediction.predicted_id else 1
        self.candidate = prediction.predicted_id
        return dict(predicted_id=self.candidate if self.count >= self.stable else None,
                    confidence=score, stable=self.count >= self.stable)
