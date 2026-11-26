from .fist_gate import driving_fists_visible, driving_motion_visible
from .evaluation import Evaluator

SEQUENCES = {"SEN0009": ("운전", "천천히"), "SEN0011": ("운전", "빨리")}


class WordSequence:
    def __init__(self, target):
        self.target = target
        self.first, self.second = SEQUENCES[target]
        self.stage = 0
        self.judge = self._judge(self.first)

    @staticmethod
    def _judge(word):
        return Evaluator(word, threshold=.9, stable=3, require_calibrated=False)

    @property
    def window(self):
        return 12 if self.stage else 8

    def update(self, prediction, frames):
        if self.stage == 0 and not driving_fists_visible(frames):
            self.judge.history.clear()
            return dict(result="RETRY", predicted_id=prediction.predicted_id,
                        confidence=prediction.confidence, recognized_words=[],
                        stable_count=0, stable_required=self.judge.history.maxlen,
                        stable_target=self.first,
                        feedback="운전은 두 손을 주먹 모양으로 표현해 주세요. 손가락을 굽히고 양손이 잘 보이도록 해 주세요.")
        if self.stage == 0 and not driving_motion_visible(frames):
            self.judge.history.clear()
            return dict(result="RETRY", predicted_id=prediction.predicted_id,
                        confidence=prediction.confidence, recognized_words=[],
                        stable_count=0, stable_required=self.judge.history.maxlen,
                        stable_target=self.first,
                        feedback="양손 주먹을 유지하고, 운전 기준 영상처럼 핸들을 좌우로 돌리는 동작을 이어가세요.")
        result = self.judge.update(prediction)
        result["recognized_words"] = [self.first] if self.stage else []
        if result["result"] == "PASS":
            if self.stage:
                return dict(result, predicted_id=self.target,
                            recognized_words=[self.first, self.second])
            self.stage = 1
            self.judge = self._judge(self.second)
            # Keep the tail where a naturally connected second sign may have started.
            while len(frames) > 2:
                frames.popleft()
            return dict(result, result="RETRY", recognized_words=[self.first],
                        stable_count=0, stable_required=self.judge.history.maxlen,
                        stable_target=self.second,
                        feedback=f"운전을 인식했어요. 이어서 ‘{self.second}’를 표현해 주세요.")
        expected = self.second if self.stage else self.first
        if result["result"] != "CAPTURE_ERROR":
            result.update(result="RETRY", feedback=f"‘{expected}’를 표현해 주세요.")
        return result
