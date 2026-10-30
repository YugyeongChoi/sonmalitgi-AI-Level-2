"""
단어 인식 모델 정의 — 학습(train_word_lstm.py)과 AI 서버(ai_server)가 같은 파일을 공유한다.
구조를 바꾸면 기존 best.pt는 로드되지 않으므로 다시 학습해야 한다.
"""
import torch
import torch.nn as nn

from word_dataset import N_FEAT


class WordLSTM(nn.Module):
    """Linear(194→h) → 양방향 LSTM → 시간축 평균+최대 풀링 → Linear(클래스)."""

    def __init__(self, n_cls, n_in=N_FEAT, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, hidden), nn.GELU(),
                                 nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_cls))

    def forward(self, x):                      # x: (B, T, F)
        h, _ = self.lstm(self.inp(x))           # (B, T, 2h)
        z = torch.cat([h.mean(1), h.amax(1)], dim=-1)
        return self.head(z)


def load_checkpoint(path, device="cpu"):
    """best.pt → (model.eval(), ckpt dict). 체크포인트: model, classes, cfg, arch, epoch."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    arch = ckpt.get("arch", {})
    model = WordLSTM(len(ckpt["classes"]), n_in=arch.get("n_in", N_FEAT),
                     hidden=arch.get("hidden", 128), layers=arch.get("layers", 2), dropout=0.0)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
