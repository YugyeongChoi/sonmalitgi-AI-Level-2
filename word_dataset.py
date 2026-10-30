#!/usr/bin/env python3
"""
3단계: 단어 인식 데이터로더 (word-pipeline-handoff.md 결정사항 4·5·7·9)

입력 : manifest.csv + clips/*.npy  (T, 54, 3) = 왼손21 + 오른손21 + 상체12 × (x, y, conf)
출력 : (32, 194) float32 특징 + WORD 클래스 인덱스

처리 순서 (train만 [증강])
  1. [프레임 솎아내기]  1/2·1/3 서브샘플링 → 저fps 클립(val sub 22%)·저사양 폰 흉내
  2. 결측: conf < thr → NaN, 짧은 공백(≤ 5프레임) 선형 보간, 긴 공백은 NaN 유지
  3. 정규화: 프레임별 목 원점, 클립 중앙값 어깨너비로 나눔 (단위 = 어깨너비)
  4. [공간 증강]  회전·스케일·이동(전체 동일) + 좌표 노이즈(점마다)
  5. [손 드롭아웃]  쉬는 손은 높은 확률로 통째로, 아무 손이나 짧은 구간 → MediaPipe 미검출 흉내
  6. 32프레임 리샘플링 [+ 앞뒤 랜덤 크롭 + 시간 워프]
  7. 특징: 몸 위치 + 손 위치 + 손 모양(손목 원점·손 크기 정규화) + 손 존재 플래그, NaN → 0

※ 균일한 속도 변화는 증강하지 않음: 어차피 32프레임으로 리샘플링되면 사라짐.
   대신 시간 워프(구간별로 빠르고 느리게)와 크롭(시작·끝 어긋남)을 씀.

특징 레이아웃 (F = 194)
  [  0: 24] 몸 12점 (x, y)
  [ 24: 66] 왼손 위치 21점   [ 66:108] 오른손 위치 21점      ← 몸 기준 좌표
  [108:150] 왼손 모양 21점   [150:192] 오른손 모양 21점      ← 손목 원점, 손 크기 1
  [192] 왼손 존재  [193] 오른손 존재

앱(MediaPipe) 추론에서는 같은 54점 배열을 만든 뒤 make_features(arr, cfg, rng=None)를 그대로 호출.

사용 예 (점검):
  python word_dataset.py --manifest data/word/manifest.csv --labels 나 너 감사 친구
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:          # numpy만으로도 동작 (학습에는 torch 필요)
    torch = None
    _TorchDataset = object

# ---------------------------------------------------------------- 배열 인덱스
LH = slice(0, 21)
RH = slice(21, 42)
BODY = slice(42, 54)
B_NECK, B_RSHO, B_LSHO = 42 + 1, 42 + 2, 42 + 5
B_RWRI, B_LWRI = 42 + 4, 42 + 7
H_WRIST, H_MMCP = 0, 9                     # 손 내부: 손목, 중지 MCP
N_FEAT = 194
USE_FLAGS_DEFAULT = ("ok", "sub", "near")


@dataclass
class Config:
    T: int = 32                    # 고정 길이 (결정사항 7)
    conf_thr: float = 0.3          # REAL은 conf가 전부 1이라 무의미, CROWD·MediaPipe용
    max_gap: int = 5               # 이 길이 이하 공백만 선형 보간 (결정사항 4)
    # ---- 증강 (train만)
    augment: bool = True
    p_decimate: float = 0.3        # 프레임 솎아내기 확률
    rot_deg: float = 10.0
    scale: float = 0.1             # ±10%
    shift: float = 0.05            # 어깨너비 단위
    noise: float = 0.02            # 어깨너비 단위, 점마다 가우시안 σ
    crop: float = 0.1              # 앞뒤 각각 최대 10% 크롭
    warp: float = 0.2              # 시간 워프 강도 (< 1/π 이어야 단조)
    rest_y: float = 1.0            # 손목 y > 이 값(목 아래, 어깨너비 단위)이면 쉬는 손
    p_drop_rest: float = 0.5       # 쉬는 손 구간을 결측 처리할 확률
    p_drop_seg: float = 0.2        # 아무 손이나 짧은 구간 결측 확률
    seg_len: tuple = (3, 8)        # 짧은 결측 구간 길이 (원본 프레임)


# ---------------------------------------------------------------- 단계별 함수
def mask_low_conf(arr, thr):
    """(T,54,3) → (T,54,2) xy, conf < thr 또는 NaN 은 NaN."""
    xy = arr[..., :2].astype(np.float32).copy()
    bad = ~(arr[..., 2] >= thr) | np.isnan(xy).any(-1)
    xy[bad] = np.nan
    return xy


def interp_short_gaps(xy, max_gap):
    """점마다 시간축으로 길이 ≤ max_gap 인 내부 공백만 선형 보간. 앞뒤 끝 공백은 그대로."""
    T = len(xy)
    out = xy.copy()
    miss = np.isnan(xy[..., 0])                       # (T, 54)
    t = np.arange(T)
    for j in np.where(miss.any(0) & ~miss.all(0))[0]:
        m = miss[:, j]
        valid = t[~m]
        # 공백 구간 찾기
        d = np.diff(np.r_[0, m.astype(np.int8), 0])
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]   # [s, e)
        for s, e in zip(starts, ends):
            if s == 0 or e == T or (e - s) > max_gap:
                continue
            for k in range(2):
                out[s:e, j, k] = np.interp(t[s:e], valid, xy[~m, j, k])
    return out


def normalize(xy):
    """목 원점(프레임별), 클립 중앙값 어깨너비로 스케일. 목 결측 프레임은 클립 중앙값 목."""
    neck = xy[:, B_NECK].copy()
    med_neck = np.nanmedian(neck, 0)
    if np.isnan(med_neck).any():                      # 목이 전부 없음 → 어깨 중점
        med_neck = np.nanmedian((xy[:, B_RSHO] + xy[:, B_LSHO]) / 2, 0)
    neck[np.isnan(neck).any(-1)] = med_neck
    sw = np.nanmedian(np.linalg.norm(xy[:, B_LSHO] - xy[:, B_RSHO], axis=-1))
    if not np.isfinite(sw) or sw <= 0:
        sw = 1.0
    return (xy - neck[:, None]) / sw


def aug_spatial(xy, cfg, rng):
    th = np.deg2rad(rng.uniform(-cfg.rot_deg, cfg.rot_deg))
    s = rng.uniform(1 - cfg.scale, 1 + cfg.scale)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32) * s
    out = xy @ R.T + rng.uniform(-cfg.shift, cfg.shift, 2).astype(np.float32)
    return out + rng.normal(0, cfg.noise, out.shape).astype(np.float32)


def aug_hand_dropout(xy, cfg, rng):
    """MediaPipe가 손을 못 잡는 상황 흉내. NaN으로 만들면 이후 존재 플래그 0, 값 0."""
    out = xy.copy()
    T = len(out)
    for hs, wri in [(LH, B_LWRI), (RH, B_RWRI)]:
        # (a) 쉬는 손: 손목이 rest_y 아래인 프레임들을 통째로
        if rng.random() < cfg.p_drop_rest:
            rest = out[:, wri, 1] > cfg.rest_y
            out[rest, hs] = np.nan
        # (b) 검출 깜빡임: 짧은 구간
        if rng.random() < cfg.p_drop_seg and T > cfg.seg_len[1]:
            L = rng.integers(cfg.seg_len[0], cfg.seg_len[1] + 1)
            s = rng.integers(0, T - L + 1)
            out[s:s + L, hs] = np.nan
    return out


def resample(xy, T_out, crop=0.0, warp=0.0, rng=None):
    """시간축 선형 리샘플링 (NaN 인지). 이웃 중 하나라도 NaN 이면 가까운 쪽 값 사용."""
    T = len(xy)
    lo_t, hi_t = 0.0, T - 1.0
    if rng is not None and crop > 0 and T > 4:
        lo_t = rng.uniform(0, crop) * (T - 1)
        hi_t = (T - 1) - rng.uniform(0, crop) * (T - 1)
    u = np.linspace(0, 1, T_out)
    if rng is not None and warp > 0:
        a = rng.uniform(-warp, warp)                  # |a| < 1/π → 단조 증가, 양 끝 고정
        u = u + a * np.sin(np.pi * u) / np.pi
    pos = lo_t + u * (hi_t - lo_t)
    i0 = np.clip(np.floor(pos).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    w = (pos - i0)[:, None, None].astype(np.float32)
    a0, a1 = xy[i0], xy[i1]
    out = (1 - w) * a0 + w * a1
    near = np.where(w < 0.5, a0, a1)
    bad = np.isnan(out)
    out[bad] = near[bad]
    return out


def to_features(xy):
    """(T,54,2) 정규화 좌표 → (T,194)."""
    T = len(xy)
    feats = [xy[:, BODY].reshape(T, -1), xy[:, LH].reshape(T, -1), xy[:, RH].reshape(T, -1)]
    pres = []
    for hs in (LH, RH):
        h = xy[:, hs]
        wrist = h[:, H_WRIST:H_WRIST + 1]
        size = np.nanmedian(np.linalg.norm(h[:, H_MMCP] - h[:, H_WRIST], axis=-1)) \
            if not np.isnan(h[:, H_MMCP, 0]).all() else np.nan
        if not np.isfinite(size) or size <= 1e-6:
            shape = np.full_like(h, np.nan)
        else:
            shape = (h - wrist) / size
        feats.append(shape.reshape(T, -1))
        pres.append((~np.isnan(h[:, H_WRIST, 0])).astype(np.float32)[:, None])
    x = np.concatenate(feats + pres, axis=1).astype(np.float32)
    return np.nan_to_num(x, nan=0.0)


def make_features(arr, cfg, rng=None):
    """원시 (T,54,3) → (cfg.T, 194). rng=None 또는 cfg.augment=False 이면 증강 없음 (val·앱 추론)."""
    aug = rng is not None and cfg.augment
    a = arr
    if aug and rng.random() < cfg.p_decimate and len(a) >= 12:
        k = int(rng.integers(2, 4))                   # 1/2 또는 1/3
        a = a[int(rng.integers(0, k))::k]
    xy = mask_low_conf(a, cfg.conf_thr)
    xy = interp_short_gaps(xy, cfg.max_gap)
    xy = normalize(xy)
    if aug:
        xy = aug_spatial(xy, cfg, rng)
        xy = aug_hand_dropout(xy, cfg, rng)
    xy = resample(xy, cfg.T, crop=cfg.crop if aug else 0, warp=cfg.warp if aug else 0,
                  rng=rng if aug else None)
    return to_features(xy)


# ---------------------------------------------------------------- 데이터셋
def read_manifest(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def resolve_words(rows, labels):
    """라벨(한글) → WORD ID. 같은 라벨에 WORD ID가 여러 개면 경고하고 전부 포함."""
    ids = []
    for lab in labels:
        hit = sorted({r["word_id"] for r in rows if r["label"] == lab})
        if not hit:
            print(f"[warn] 라벨 '{lab}' 없음", file=sys.stderr)
        elif len(hit) > 1:
            print(f"[warn] 라벨 '{lab}' WORD ID {len(hit)}개: {hit} → --words로 하나만 고르는 걸 권장",
                  file=sys.stderr)
        ids += hit
    return ids


class WordDataset(_TorchDataset):
    """split의 클립을 메모리에 올려두고(전체 ~200MB) __getitem__마다 특징 생성·증강."""

    def __init__(self, manifest, split, classes, cfg=None, use_flags=USE_FLAGS_DEFAULT,
                 train=None, seed=0):
        self.cfg = cfg or Config()
        self.train = (split == "train") if train is None else train
        root = Path(manifest).parent
        self.cls_index = {w: i for i, w in enumerate(classes)}
        rows = [r for r in read_manifest(manifest)
                if r["split"] == split and r.get("fps_flag", "ok") in use_flags
                and r["word_id"] in self.cls_index]
        self.items, self.arrs = [], []
        for r in rows:
            p = root / r["path"]
            if not p.exists():
                continue
            self.items.append(r)
            self.arrs.append(np.load(p))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = make_features(self.arrs[i], self.cfg, self.rng if self.train else None)
        y = self.cls_index[self.items[i]["word_id"]]
        if torch is not None:
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        return x, y


def worker_init_fn(wid):
    """num_workers > 0일 때 워커마다 다른 증강 난수."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32))


def build_datasets(manifest, words=None, labels=None, cfg=None, use_flags=USE_FLAGS_DEFAULT,
                   out_classes=None):
    """train/val 데이터셋 + 클래스 목록. words·labels 둘 다 없으면 train에 있는 전체 단어."""
    rows = read_manifest(manifest)
    ids = list(words or [])
    if labels:
        ids += resolve_words(rows, labels)
    if not ids:
        ids = sorted({r["word_id"] for r in rows if r["split"] == "train"})
    classes = sorted(set(ids))
    names = {r["word_id"]: r["label"] for r in rows}
    if out_classes:
        Path(out_classes).write_text(json.dumps(
            [{"index": i, "word_id": w, "label": names.get(w, "")} for i, w in enumerate(classes)],
            ensure_ascii=False, indent=1), encoding="utf-8")
    tr = WordDataset(manifest, "train", classes, cfg, use_flags)
    va = WordDataset(manifest, "val", classes, cfg, use_flags)
    return tr, va, classes


# ---------------------------------------------------------------- 점검
def _check(args):
    cfg = Config()
    tr, va, classes = build_datasets(args.manifest, args.words, args.labels, cfg,
                                     out_classes=Path(args.manifest).parent / "classes.json")
    print(f"클래스 {len(classes)}개 / train {len(tr)} / val {len(va)}")
    miss_val = set(classes) - {r['word_id'] for r in va.items}
    if miss_val:
        print(f"  val에 없는 단어 {len(miss_val)}개 (morpheme 없음·fps 제외 등)")

    def stats(ds, n_rep, name):
        xs, t0 = [], time.time()
        for _ in range(n_rep):
            for i in range(len(ds)):
                x, _ = ds[i]
                xs.append(np.asarray(x))
        if not xs:
            return
        X = np.stack(xs)
        dt = (time.time() - t0) / len(X) * 1000
        blocks = [("몸", 0, 24), ("왼손 위치", 24, 66), ("오른손 위치", 66, 108),
                  ("왼손 모양", 108, 150), ("오른손 모양", 150, 192)]
        print(f"\n[{name}] {X.shape}, 샘플당 {dt:.1f} ms, NaN/inf {int((~np.isfinite(X)).sum())}")
        for b, s, e in blocks:
            v = X[..., s:e]
            print(f"  {b:8s}: 평균 {v.mean():+.2f}, 표준편차 {v.std():.2f}, 최대|값| {np.abs(v).max():.1f}")
        print(f"  손 존재율: 왼 {X[..., 192].mean():.3f}, 오른 {X[..., 193].mean():.3f}")

    stats(tr, args.reps, "train (증강)")
    stats(va, 1, "val")

    # 쉬는 손 기준(rest_y) 보정용: 정규화 좌표 손목 y 분포
    ys = {"왼": [], "오른": []}
    for a in tr.arrs:
        xy = normalize(interp_short_gaps(mask_low_conf(a, cfg.conf_thr), cfg.max_gap))
        ys["왼"].append(xy[:, B_LWRI, 1]); ys["오른"].append(xy[:, B_RWRI, 1])
    print("\n손목 y (목 원점, 어깨너비 단위, 아래가 +) 분위수 5/25/50/75/95%:")
    for k, v in ys.items():
        v = np.concatenate(v)
        v = v[np.isfinite(v)]
        print(f"  {k}: {np.percentile(v, [5, 25, 50, 75, 95]).round(2)}")
    print(f"  현재 rest_y = {cfg.rest_y} (이보다 아래면 쉬는 손으로 보고 드롭아웃 대상)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/word/manifest.csv")
    ap.add_argument("--words", nargs="*", default=None, help="WORD ID 목록 (예: WORD1157)")
    ap.add_argument("--labels", nargs="*", default=None, help="라벨(한글) 목록 → WORD ID로 변환")
    ap.add_argument("--reps", type=int, default=3, help="train 증강 샘플 반복 횟수")
    _check(ap.parse_args())