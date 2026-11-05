"""Bridge for sonmalitgi-web WORD_MODEL_ADAPTER; experimental, no PASS."""
import importlib.util
import math
import os
from pathlib import Path
import numpy as np
import torch
from hands_dataset import Config, make_features


class HandsWordAdapter:
    mode = "hands-word-experimental-v1"
    input_schema = "mediapipe-hands-v1"

    def __init__(self, checkpoint, aspect=None):
        if aspect is not None and (not math.isfinite(aspect) or aspect <= 0):
            raise ValueError("HANDS_VIDEO_ASPECT must be the actual video width/height")
        self.aspect = aspect
        path = Path(__file__).parent / "ai_server/ml/word_model.py"
        spec = importlib.util.spec_from_file_location("hands_word_network", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model, ckpt = module.load_checkpoint(checkpoint)
        self.supported_ids = ckpt["classes"]
        self.cfg = Config(**ckpt["cfg"])
        self.cfg.augment = False

    def predict_landmarks(self, frames):
        # Imported from the host website, not this fork's independent app package.
        from ai_server.model import Prediction
        arr = np.full((len(frames), 42, 3), np.nan, np.float32)
        arr[..., 2] = 0
        seen = False
        for i, frame in enumerate(frames):
            if frame.get("schema") != self.input_schema:
                raise ValueError("Hands-only landmarks required")
            width, height = frame.get("image_width"), frame.get("image_height")
            aspect = width / height if width and height else self.aspect
            if aspect is None:
                raise ValueError("Actual camera dimensions are required")
            for base, name in ((0, "left_hand"), (21, "right_hand")):
                points = frame[name]
                if not points:
                    continue
                if len(points) != 21:
                    raise ValueError("Expected 21 hand points")
                seen = True
                for j, p in enumerate(points):
                    arr[i, base+j] = [p["x"] * aspect, p["y"], 1.]
        if not seen:
            return Prediction(None, 0., False, False, "손이 카메라에 보이도록 위치를 조정해 주세요.")
        x = make_features(arr, self.cfg)
        with torch.inference_mode():
            prob = self.model(torch.from_numpy(x)[None]).softmax(-1)[0]
        idx = int(prob.argmax())
        return Prediction(self.supported_ids[idx], float(prob[idx]), True, False,
                          "손 전용 시험 모델의 예측입니다. 웹캠 성능 검증 전이므로 통과 판정은 하지 않습니다.")


def create_model():
    return HandsWordAdapter(os.environ["HANDS_WORD_CHECKPOINT"],
                            float(os.environ["HANDS_VIDEO_ASPECT"]) if os.getenv("HANDS_VIDEO_ASPECT") else None)
