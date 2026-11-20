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


def test_sentence_sequence_requires_three_consecutive_predictions():
    from collections import deque
    from ai_server.sequence import WordSequence
    sequence = WordSequence('SEN0009')
    frames = deque(range(12))
    driving = Prediction('운전', .99, True, False)
    for _ in range(2):
        assert sequence.update(driving, frames)['result'] == 'RETRY'
        assert sequence.stage == 0
    sequence.update(driving, frames)
    assert sequence.stage == 1
    slow = Prediction('천천히', .99, True, False)
    assert sequence.update(slow, frames)['result'] == 'RETRY'
    sequence.update(Prediction('빨리', .99, True, False), frames)
    for _ in range(2):
        assert sequence.update(slow, frames)['result'] == 'RETRY'
    assert sequence.update(slow, frames)['result'] == 'PASS'


def test_consecutive_progress_tracks_target_and_resets():
    judge = Evaluator('목표', threshold=.9, stable=3, require_calibrated=False)
    good = Prediction('목표', .95, True, False)
    assert judge.update(good)['stable_count'] == 1
    assert judge.update(good)['stable_count'] == 2
    assert judge.update(Prediction('다른말', .99, True, False))['stable_count'] == 0
    assert judge.update(good)['stable_count'] == 1
    assert judge.update(Prediction('목표', .8, True, False))['stable_count'] == 0
    for expected in (1, 2, 3):
        result = judge.update(good)
        assert result['stable_count'] == expected
        assert result['stable_required'] == 3
        assert result['stable_target'] == '목표'
    assert result['result'] == 'PASS'
