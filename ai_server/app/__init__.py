"""KSL AI 서버. ml/ 폴더의 학습 코드(word_dataset.py, word_model.py)를 최상위 모듈로 import 한다."""
import sys
from pathlib import Path

_ML = Path(__file__).resolve().parent.parent / "ml"
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))
