import sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'deployment'),str(ROOT)]
from intermediate_v4.adapter import IntermediateAdapter
from intermediate_v4.landmarks import mediapipe_to_openpose54
from intermediate_v4.word_dataset import make_features

def sample():
    pose=[dict(x=.5,y=.3,z=0,visibility=1) for _ in range(33)]
    pose[11]=dict(x=.65,y=.4,z=0,visibility=1);pose[12]=dict(x=.35,y=.4,z=0,visibility=1)
    pose[15]=dict(x=.65,y=.6,z=0,visibility=1);pose[16]=dict(x=.35,y=.6,z=0,visibility=1)
    hand=lambda x:[dict(x=x+j*.002,y=.6+j*.001,z=0) for j in range(21)]
    return dict(type='landmarks',schema='mediapipe-body-hands-v1',timestamp_ms=1,image_width=640,image_height=480,pose=pose,left_hand=hand(.65),right_hand=hand(.35))

def test_actual_v4_checkpoint_and_original_preprocessing():
    torch.set_num_threads(2)
    adapter=IntermediateAdapter(ROOT/'intermediate_v4/best.pt')
    assert len(adapter.supported_ids)==11
    assert adapter.model.inp[1].in_features==194
    f=sample();frames=[f]*16
    p=adapter.predict_landmarks(frames)
    hands=[dict(landmarks=[[v['x'],v['y'],v['z']] for v in f[k]],handedness=l) for k,l in [('left_hand','Left'),('right_hand','Right')]]
    arr=np.stack([mediapipe_to_openpose54([[v['x'],v['y'],v['z'],v['visibility']] for v in f['pose']],hands,640,480,False)]*16)
    start,end=adapter.segmenter(arr)
    with torch.inference_mode():expected=adapter.model(torch.from_numpy(make_features(arr[start:end],adapter.cfg))[None]).softmax(-1)[0]
    assert p.predicted_id==adapter.supported_ids[int(expected.argmax())]
    assert p.confidence==pytest.approx(float(expected.max()),abs=1e-7)
    assert not p.calibrated
    f['pose']=[]
    assert not adapter.predict_landmarks([f]*16).capture_ok
    f['schema']='mediapipe-hands-v1'
    with pytest.raises(ValueError):adapter.predict_landmarks([f]*16)

def test_body_wire_decoder_and_missing_body():
    from ai_server.landmarks import decode_landmarks
    f=sample()
    assert len(decode_landmarks(f)['pose'])==33
    f['pose']=[]
    assert decode_landmarks(f)['pose']==[]

def test_websocket_bridge_loads_middle_and_preserves_advanced(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv('WORD_MODEL_ADAPTER','intermediate_v4.adapter:create_model')
    monkeypatch.setenv('INTERMEDIATE_CHECKPOINT',str(ROOT/'intermediate_v4/best.pt'))
    monkeypatch.setenv('UNIT5_CHECKPOINT',str(ROOT/'unit5.pt'))
    monkeypatch.setenv('UNIT5_WORD_CHECKPOINT',str(ROOT/'unit5_words.pt'))
    from ai_server.main import app
    import uuid
    with TestClient(app) as client:
        health=client.get('/health').json()
        assert health['word_model_mode']=='intermediate-v4-body-hands-v1'
        assert len(health['unit5_supported_ids'])==5
        with client.websocket_connect('/ws/v1/word/evaluate',headers={'origin':'http://localhost:5173'}) as ws:
            ws.send_json(dict(type='start',input_mode='landmarks',target_id='WORD1290',attempt_id=str(uuid.uuid4())))
            assert ws.receive_json()['type']=='ready'
            for i in range(16):
                f=sample();f['timestamp_ms']=i+1;ws.send_json(f)
                r=ws.receive_json()
                if r['type']=='prediction':
                    assert 'confidence' in r
                    r=ws.receive_json()
                    if r['type']=='final':break
                assert r['type']=='ack',r
            else:
                ws.send_json(dict(type='stop'))
                assert ws.receive_json()['type']=='final'
