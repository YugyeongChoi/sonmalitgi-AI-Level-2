"""체크포인트의 Dropout 설정과 평가 모드 호환성을 검증한다."""
import importlib.util
from pathlib import Path
import pytest
import torch
from hands_dataset import N_FEAT, FEATURE_SCHEMA

spec = importlib.util.spec_from_file_location("dropout_model", Path(__file__).resolve().parents[1] / "ai_server/ml/word_model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

@pytest.mark.parametrize("rate", [None, 0.3])
def test_checkpoint_dropout_and_eval(tmp_path, rate):
    torch.set_num_threads(2)
    original = module.WordLSTM(3, hidden=8, layers=1, dropout=0.0)
    arch = {"n_in": N_FEAT, "hidden": 8, "layers": 1}
    if rate is not None:
        arch["dropout"] = rate
    path = tmp_path / "model.pt"
    torch.save({"model": original.state_dict(), "classes": ["a", "b", "c"], "arch": arch, "feature_schema": FEATURE_SCHEMA}, path)
    loaded, _ = module.load_checkpoint(path)
    assert loaded.inp[-1].p == (rate or 0.0)
    assert loaded.head[0].p == (rate or 0.0)
    assert loaded.lstm.dropout == 0.0
    assert not loaded.training
    x = torch.randn(2, 32, N_FEAT)
    with torch.no_grad():
        assert torch.equal(loaded(x), original.eval()(x))
        assert torch.equal(loaded(x), loaded(x))
    if rate:
        loaded.train()
        assert not torch.equal(loaded(x), loaded(x))
