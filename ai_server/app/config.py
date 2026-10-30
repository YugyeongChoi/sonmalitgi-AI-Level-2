"""서버 설정. 환경 변수로 바꾼다 (Windows cmd: set KSL_MODELS=word=runs\\...\\best.pt)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _models_from_env():
    """KSL_MODELS="word=path/best.pt;fingerspell=path/fs.pt" → {lesson: path}."""
    raw = os.environ.get("KSL_MODELS", f"word={ROOT / 'models' / 'word' / 'best.pt'}")
    out = {}
    for item in filter(None, (s.strip() for s in raw.split(";"))):
        lesson, _, path = item.partition("=")
        out[lesson.strip()] = path.strip()
    return out


@dataclass
class Settings:
    models: dict = field(default_factory=_models_from_env)
    device: str = os.environ.get("KSL_DEVICE", "cpu")
    # 판정: 목표 단어가 1위이고 확률이 이 값 이상이면 정답
    target_threshold: float = float(os.environ.get("KSL_TARGET_THRESHOLD", "0.5"))
    top_k: int = int(os.environ.get("KSL_TOP_K", "3"))
    # 구간 처리: none = 클라이언트 start~end 그대로 / hands = 쉬는 손 구간을 앞뒤에서 잘라냄 (실험)
    segmenter: str = os.environ.get("KSL_SEGMENTER", "none")
    # 한 번의 시도에서 받을 프레임 수 제한 (30fps 기준 10초, 0.3초)
    max_frames: int = int(os.environ.get("KSL_MAX_FRAMES", "300"))
    min_frames: int = int(os.environ.get("KSL_MIN_FRAMES", "8"))
    # 녹화 중 실시간 상태(손 보임 여부)를 몇 프레임마다 보낼지
    status_every: int = int(os.environ.get("KSL_STATUS_EVERY", "5"))
    # 비우면 저장 안 함. 경로를 주면 시도마다 좌표(npy)와 결과(json) 저장
    save_dir: str = os.environ.get("KSL_SAVE_DIR", "")


settings = Settings()
