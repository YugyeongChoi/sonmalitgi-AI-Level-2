"""
레슨 종류별 인식기. 모두 같은 인터페이스:
  classes: [{"id": "WORD1157", "label": "나"}, ...]
  predict(arr54: (T, 42, 3)) -> 확률 (n_classes,)

새 모델(팀원의 지문자·지숫자)을 붙일 때는 Recognizer를 상속한 클래스를 만들고 LOADERS에 등록한다.
입력이 54점이 아니라면 predict 안에서 arr54를 그 모델의 입력 형식으로 바꾸면 된다.
"""
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("ksl.recognizers")


class Recognizer:
    lesson = "base"
    classes: list = []

    def predict(self, arr54: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class WordRecognizer(Recognizer):
    """train_word_lstm.py가 저장한 best.pt (+ 같은 폴더의 classes.json으로 라벨)."""
    lesson = "word"

    def __init__(self, ckpt_path, device="cpu"):
        import torch
        from hands_dataset import Config, make_features
        from word_model import load_checkpoint

        self._torch, self._make_features, self.device = torch, make_features, device
        self.model, ckpt = load_checkpoint(ckpt_path, device)
        cfg = dict(ckpt.get("cfg", {}))
        if isinstance(cfg.get("seg_len"), list):
            cfg["seg_len"] = tuple(cfg["seg_len"])
        self.cfg = Config(**cfg)
        self.cfg.augment = False                       # 추론은 증강 없음

        labels = {}
        cj = Path(ckpt_path).with_name("classes.json")
        if cj.exists():
            labels = {e["word_id"]: e.get("label", "")
                      for e in json.loads(cj.read_text(encoding="utf-8"))}
        self.classes = [{"id": w, "label": labels.get(w, "")} for w in ckpt["classes"]]

    def predict(self, arr54):
        x = self._make_features(arr54, self.cfg)      # (32, 170), 학습과 같은 전처리
        with self._torch.no_grad():
            t = self._torch.from_numpy(x)[None].to(self.device)
            return self._torch.softmax(self.model(t), -1)[0].cpu().numpy()


LOADERS = {
    "word": WordRecognizer,
    # "fingerspell": FingerspellRecognizer,   ← 팀원 모델 연결 시 추가
}


def load_registry(models: dict, device="cpu") -> dict:
    """{lesson: path} → {lesson: Recognizer}. 파일이 없거나 실패한 레슨은 건너뛰고 로그."""
    reg = {}
    for lesson, path in models.items():
        if lesson not in LOADERS:
            log.warning("알 수 없는 레슨 종류 '%s' (LOADERS에 등록 필요)", lesson)
            continue
        if not Path(path).exists():
            log.warning("모델 파일 없음: %s=%s", lesson, path)
            continue
        try:
            reg[lesson] = LOADERS[lesson](path, device)
            log.info("모델 로드: %s (%d 클래스) ← %s", lesson, len(reg[lesson].classes), path)
        except Exception:
            log.exception("모델 로드 실패: %s=%s", lesson, path)
    return reg
