#!/usr/bin/env python3
"""
4단계: 단어 인식 LSTM 학습 / 과적합 테스트 (word-pipeline-handoff.md 결정사항 8)

목적: 제공자 2명(단어당 train 1개)으로는 정확도가 의미 없음 → 파이프라인이 "학습 가능한지" 검증.
  1) 증강 끄고:  train 정확도가 100%에 도달해야 함 (못 하면 특징·라벨·모델 버그)
  2) 증강 켜고:  그래도 train을 맞춰야 함 (못 하면 증강이 너무 셈)
  val(다른 제공자) 정확도는 참고용.

모델: Linear(170→h) → 양방향 LSTM → 시간축 평균+최대 풀링 → Linear(클래스)
  - 단어 단위 인식이라 영상 전체를 보고 판단 → 양방향 사용 가능 (앱에서도 녹화 후 판정)

출력 (<out>/)
  best.pt      : 모델 가중치 + Config + 클래스 목록 (앱 변환·추론에 그대로 사용)
  classes.json : 인덱스 ↔ WORD ID ↔ 라벨
  log.csv      : epoch별 loss·정확도

사용 예:
  python train_word_lstm.py --labels 나 너 감사 친구 --no-augment --epochs 100
  python train_word_lstm.py --labels 나 너 감사 친구 --epochs 300
"""
import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from hands_dataset import N_FEAT, FEATURE_SCHEMA, Config, build_datasets, worker_init_fn


class WordLSTM(nn.Module):
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


def run_epoch(model, loader, opt, device, train):
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    lossf = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = lossf(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
    return loss_sum / max(tot, 1), correct / max(tot, 1)


@torch.no_grad()
def predict_report(model, ds, classes, names, device, top=3):
    """val 클립별 예측 (단어 수가 적을 때 눈으로 확인용)."""
    model.eval()
    rows = []
    for i in range(len(ds)):
        x, y = ds[i]
        p = model(x[None].to(device)).softmax(-1)[0].cpu()
        v, idx = p.topk(min(top, len(classes)))
        rows.append((names[classes[int(y)]], [(names[classes[int(j)]], round(float(s), 2))
                                              for s, j in zip(v, idx)]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/word/manifest.csv")
    ap.add_argument("--words", nargs="*", default=None, help="WORD ID 목록")
    ap.add_argument("--labels", nargs="*", default=None, help="라벨(한글) 목록")
    ap.add_argument("--no-augment", action="store_true", help="증강 끄기 (1차 과적합 테스트)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10,
                    help="epoch당 최소 스텝 수. 클립이 적어도 증강된 샘플을 충분히 보게 함")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0, help="Windows는 0 권장")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="출력 폴더 (기본 runs/hands_<시각>)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out or f"runs/hands_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config(augment=not args.no_augment)
    tr, va, classes = build_datasets(args.manifest, args.words, args.labels, cfg,
                                     out_classes=out / "classes.json")
    names = {e["word_id"]: f"{e['label']}({e['word_id']})"
             for e in json.loads((out / "classes.json").read_text(encoding="utf-8"))}
    print(f"device {device} | 클래스 {len(classes)} | train {len(tr)} / val {len(va)} | "
          f"증강 {'끔' if args.no_augment else '켬'}")
    if len(tr) == 0:
        raise SystemExit("train 클립이 없습니다 (단어 이름·manifest 경로 확인)")

    n_samples = max(len(tr), args.batch * args.steps)
    tl = DataLoader(tr, batch_size=args.batch,
                    sampler=RandomSampler(tr, replacement=True, num_samples=n_samples),
                    num_workers=args.workers,
                    worker_init_fn=worker_init_fn if args.workers > 0 else None)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=0)

    drop = 0.0 if args.no_augment else args.dropout
    model = WordLSTM(len(classes), hidden=args.hidden, layers=args.layers, dropout=drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"파라미터 {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K")

    best, log = -1.0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, tl, opt, device, True)
        sched.step()
        row = dict(epoch=ep, train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                   val_loss="", val_acc="")
        if ep % args.eval_every == 0 or ep == args.epochs:
            if len(va):
                va_loss, va_acc = run_epoch(model, vl, None, device, False)
                row.update(val_loss=round(va_loss, 4), val_acc=round(va_acc, 4))
            score = row["val_acc"] if row["val_acc"] != "" else tr_acc
            if score >= best:                       # 동점이면 나중 것(더 학습된 것) 저장
                best = score
                torch.save(dict(model=model.state_dict(), classes=classes, cfg=asdict(cfg),
                                feature_schema=FEATURE_SCHEMA,
                                arch=dict(hidden=args.hidden, layers=args.layers, n_in=N_FEAT),
                                epoch=ep), out / "best.pt")
            print(f"ep {ep:4d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {row['val_loss']} acc {row['val_acc']} | {time.time() - t0:.0f}s")
        log.append(row)

    with open(out / "log.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader(); w.writerows(log)

    # ---- 결과 판정
    final_tr = log[-1]["train_acc"]
    print("\n===== 결과 =====")
    print(f"마지막 train 정확도 {final_tr:.3f} (증강 {'끔' if args.no_augment else '켬'})")
    if args.no_augment:
        print("  → 1.0이면 1차 통과: 특징·라벨·모델이 정상. 다음은 증강 켜고 실행"
              if final_tr >= 0.999 else
              "  → 1.0 미달: 같은 데이터도 못 외움 = 버그 가능성. 특징·라벨 확인 필요")
    else:
        print("  → 0.9 이상이면 통과: 증강을 켜도 학습 가능"
              if final_tr >= 0.9 else
              "  → 낮음: 증강이 너무 세거나 epoch 부족. epoch 늘리거나 증강 약화")
    if len(va) and len(classes) <= 30:
        print("\nval 예측 (정답 → 상위 예측, 참고용):")
        for gt, preds in predict_report(model, va, classes, names, device):
            mark = "O" if preds[0][0] == gt else "X"
            print(f"  [{mark}] {gt} → {preds}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()