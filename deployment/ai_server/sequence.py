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
        return Evaluator(word, threshold=.9, stable=2, require_calibrated=False)

    def update(self, prediction, frames):
        result = self.judge.update(prediction)
        result["recognized_words"] = [self.first] if self.stage else []
        if result["result"] == "PASS":
            if self.stage:
                return dict(result, predicted_id=self.target,
                            recognized_words=[self.first, self.second])
            self.stage = 1
            self.judge = self._judge(self.second)
            frames.clear()
            return dict(result, result="RETRY", recognized_words=[self.first],
                        feedback=f"운전을 인식했어요. 이어서 ‘{self.second}’를 표현해 주세요.")
        expected = self.second if self.stage else self.first
        if result["result"] != "CAPTURE_ERROR":
            result.update(result="RETRY", feedback=f"‘{expected}’를 표현해 주세요.")
        return result
