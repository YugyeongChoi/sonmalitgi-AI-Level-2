from pathlib import Path
import sys
import numpy as np
import pytest
import torch

from hands_dataset import Config, N_FEAT, FEATURE_SCHEMA, make_features
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai_server"))
from app.landmarks import mediapipe_to_hands42
from app.recognizers import WordRecognizer


def sample():
    return np.load("data/word/clips/NIA_SL_WORD1290_REAL01_F.npy")


def test_body_changes_cannot_change_features():
    a = sample()
    expected = make_features(a, Config(augment=False))
    a[:, 42:] = np.nan
    np.testing.assert_array_equal(make_features(a, Config(augment=False)), expected)
    np.testing.assert_array_equal(make_features(a[:, :42], Config(augment=False)), expected)
    assert expected.shape == (32, N_FEAT)
    assert np.isfinite(expected).all()


def test_scale_translation_invariance():
    a = sample()[:, :42].copy()
    b = a.copy()
    b[..., :2] = b[..., :2] * 3 + [300, 100]
    np.testing.assert_allclose(make_features(a, Config()), make_features(b, Config()), atol=2e-5)


def test_missing_hands_and_augmentation():
    a = np.zeros((16, 42, 3), np.float32)
    assert not make_features(a, Config()).any()
    a = sample()
    a[:, :21, 2] = 0
    x = make_features(a, Config(), np.random.default_rng(4))
    assert np.isfinite(x).all()
    assert not x[:, -2].any()


def test_browser_conversion_matches_training_and_ignores_depth():
    a = sample()[0]
    hands = []
    for base, label in ((0, "Right"), (21, "Left")):
        if (a[base:base+21, 2] > .3).all():
            hands.append(dict(handedness=label, landmarks=[
                [float(x)/1920, float(y)/1080, -99] for x,y in a[base:base+21,:2]]))
    out = mediapipe_to_hands42(hands, 1920, 1080)
    assert out.shape == (42, 3)
    for h in hands:
        base = 0 if h["handedness"] == "Right" else 21
        np.testing.assert_allclose(out[base:base+21,:2], a[base:base+21,:2], atol=.001)
        assert (out[base:base+21,2] == 1).all()


def test_old_body_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="Hands-only checkpoint"):
        WordRecognizer("runs/20261030_043819/best.pt")


def test_training_and_serving_preprocessing_are_identical():
    root = Path(__file__).resolve().parents[1]
    assert (root / "hands_dataset.py").read_bytes() == (root / "ai_server/ml/hands_dataset.py").read_bytes()
