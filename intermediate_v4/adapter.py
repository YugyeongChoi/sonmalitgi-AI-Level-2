"""공식 v4_11words 가중치·54점 전처리를 웹 평가 프로토콜에 연결한다."""
import os
import numpy as np
from .landmarks import mediapipe_to_openpose54
from .word_model import load_checkpoint
from .word_dataset import Config, make_features
import torch
from .segmenter import make_segmenter

class IntermediateAdapter:
    mode = 'intermediate-v4-body-hands-v1'
    input_schema = 'mediapipe-body-hands-v1'
    def __init__(self, checkpoint):
        self.model, ckpt = load_checkpoint(checkpoint)
        self.supported_ids = ckpt["classes"]
        self.cfg = Config(**ckpt["cfg"])
        self.cfg.augment = False
        self.segmenter = make_segmenter('hands')
    def predict_landmarks(self, frames):
        from ai_server.model import Prediction
        arrays=[];body_seen=False;hands_seen=False
        for f in frames:
            if f.get('schema') != self.input_schema: raise ValueError('상체와 손 좌표가 필요합니다')
            pose=f.get('pose',[])
            if len(pose)==33 and all(pose[i].get('visibility',0)>=.3 for i in (11,12)):
                body_seen=True
            else:
                # 목·어깨 정규화가 필요한 모델에 몸 없는 가짜 좌표를 넣지 않는다.
                continue
            hands=[]
            for key,label in [('left_hand','Left'),('right_hand','Right')]:
                if f[key]:
                    hands_seen=True
                    hands.append(dict(landmarks=[[p['x'],p['y'],p['z']] for p in f[key]],handedness=label))
            arrays.append(mediapipe_to_openpose54(
                [[p['x'],p['y'],p['z'],p.get('visibility',0)] for p in pose],hands,
                f['image_width'],f['image_height'],False))
        if not body_seen or len(arrays)<8:
            return Prediction(None,0,False,False,'어깨와 양팔이 화면에 보이도록 카메라 위치를 조정해 주세요.')
        if not hands_seen:return Prediction(None,0,False,False,'손이 화면에 보이도록 해 주세요.')
        arr=np.stack(arrays);start,end=self.segmenter(arr)
        with torch.inference_mode():
            probs=self.model(torch.from_numpy(make_features(arr[start:end],self.cfg))[None]).softmax(-1)[0].numpy()
        i=int(np.argmax(probs))
        return Prediction(self.supported_ids[i],float(probs[i]),True,False)

def create_model():return IntermediateAdapter(os.environ['INTERMEDIATE_CHECKPOINT'])
