"""Target-free dictionary protocol tests with a deterministic model, no weights."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from ai_server.main import app
from ai_server.model import Prediction
from ai_server.recognition import RecognitionState


def test_uncertain_unknown_and_missing_hands_clear_candidate():
    state = RecognitionState(['운전'])
    good = Prediction('운전', .95, True)
    assert state.update(good)['predicted_id'] is None
    assert state.update(good)['predicted_id'] == '운전'
    for prediction in (Prediction('운전', .2, True), Prediction('빨리', .99, True),
                       Prediction('운전', .99, False), Prediction('운전', float('nan'), True)):
        assert state.update(prediction)['predicted_id'] is None
        assert state.update(good)['predicted_id'] is None
        assert state.update(good)['predicted_id'] == '운전'


def frame(timestamp, hands=True):
    return dict(type='landmarks', schema='mediapipe-hands-v1', timestamp_ms=timestamp,
                image_width=640, image_height=480,
                left_hand=[dict(x=.4, y=.5, z=0)] * 21 if hands else [], right_hand=[])


def test_continues_after_recognition_and_clears_when_hands_disappear():
    with TestClient(app) as client:
        app.state.unit5_words = SimpleNamespace(mode='test', supported_ids=['운전'],
            predict_landmarks=lambda frames: Prediction('운전', .95, True))
        with client.websocket_connect('/ws/v1/dictionary/recognize', headers={'origin': 'http://localhost:5173'}) as ws:
            ws.send_json(dict(type='start', scope='unit5'))
            assert ws.receive_json()['supported_ids'] == ['운전']
            predictions = []
            for index in range(12):
                ws.send_json(frame(index + 1))
                message = ws.receive_json()
                if message['type'] == 'prediction':
                    predictions.append(message)
                    message = ws.receive_json()
                assert message['type'] == 'ack'
            assert predictions[0]['predicted_id'] is None
            assert predictions[-1]['predicted_id'] == '운전'
            ws.send_json(frame(13, hands=False))
            assert ws.receive_json()['predicted_id'] is None
            assert ws.receive_json()['type'] == 'ack'
            ws.send_json(frame(14))
            assert ws.receive_json()['type'] == 'ack'
            ws.send_json(dict(type='stop'))


def test_invalid_scope_is_rejected():
    with TestClient(app) as client:
        with client.websocket_connect('/ws/v1/dictionary/recognize', headers={'origin': 'http://localhost:5173'}) as ws:
            ws.send_json(dict(type='start', scope='unknown'))
            assert ws.receive_json()['type'] == 'error'
