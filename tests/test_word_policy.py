import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deployment'))
from ai_server.evaluation import Evaluator
from ai_server.model import Prediction


def test_word_practice_policy_without_claiming_calibration():
    judge = Evaluator('WORD1290', threshold=.5, stable=1, require_calibrated=False)
    assert judge.update(Prediction('WORD1290', .499, True, False))['result'] == 'RETRY'
    assert judge.update(Prediction('WORD1290', .5, True, False))['result'] == 'PASS'
    assert judge.update(Prediction('WORD1191', .8, True, False))['result'] == 'FAIL'
    assert judge.update(Prediction(None, 0., False, False))['result'] == 'CAPTURE_ERROR'


def test_other_models_still_require_calibration():
    judge = Evaluator('WORD1290')
    for _ in range(4):
        assert judge.update(Prediction('WORD1290', .99, True, False))['result'] == 'RETRY'
