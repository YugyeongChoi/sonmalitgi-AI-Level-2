#!/usr/bin/env python3
"""
QA: 추출된 단어 클립 스켈레톤 시각화 (word-pipeline-handoff.md 결정사항 10)

목적: REAL keypoint는 신뢰도가 전부 1이라, 쉬는 손·화면 밖 손 좌표가 믿을 만한지 눈으로 확인.

클립마다 출력 (<out>/<label>_<clip_id>.*)
- .gif : 스켈레톤 애니메이션 (상체 + 양손, 손목 궤적)
- .png : 요약 — 균등 샘플 6프레임 + 시간에 따른 손목 높이 + 손목 일치도
콘솔: 클립별 손목 일치도
- 손 keypoint의 손목(손 0번)과 몸 keypoint의 손목(BODY 4/7)은 같은 관절이므로 거의 겹쳐야 함.
  어깨너비 대비 거리가 크면(예: > 0.3) 그 손 좌표는 몸과 따로 노는 값 → 의심.

사용 예:
  python3 visualize_word_clips.py --labels 나 너 감사 친구
  python3 visualize_word_clips.py --clip-ids NIA_SL_WORD0001_REAL01_F
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter   # noqa: E402

for f in ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"]:
    if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# (T, 54, 3) 배열 내 위치
LH, RH, BODY = 0, 21, 42
# 상체 12점 로컬 순서: BODY_25 [0,1,2,3,4,5,6,7,15,16,17,18]
B_NOSE, B_NECK, B_RSHO, B_RELB, B_RWRI, B_LSHO, B_LELB, B_LWRI = range(8)
BODY_EDGES = [(1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
              (0, 8), (0, 9), (8, 10), (9, 11)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
C_BODY, C_L, C_R = "#888888", "#d1495b", "#2e86ab"   # 왼손(사람 기준)=빨강, 오른손=파랑


def wrist_check(a):
    """손목 일치도: |손0 - 몸손목| / 클립 중앙값 어깨너비. (왼, 오른) 각각 중앙값·최댓값."""
    xy = a[..., :2]
    sw = np.nanmedian(np.linalg.norm(xy[:, BODY + B_LSHO] - xy[:, BODY + B_RSHO], axis=-1))
    out = {}
    for name, h, w in [("L", LH, B_LWRI), ("R", RH, B_RWRI)]:
        d = np.linalg.norm(xy[:, h] - xy[:, BODY + w], axis=-1) / sw
        out[name] = (float(np.nanmedian(d)), float(np.nanmax(d)))
    return out, float(sw)


def limits(a):
    xy = a[..., :2].reshape(-1, 2)
    lo, hi = np.nanmin(xy, 0), np.nanmax(xy, 0)
    pad = (hi - lo).max() * 0.08
    return (lo[0] - pad, hi[0] + pad), (lo[1] - pad, hi[1] + pad)


def draw_frame(ax, fr, xlim, ylim):
    ax.clear()
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[1], ylim[0])          # 이미지 좌표: y 아래로 증가
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for i, j in BODY_EDGES:
        p, q = fr[BODY + i, :2], fr[BODY + j, :2]
        ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=C_BODY, lw=2)
    for h, c in [(LH, C_L), (RH, C_R)]:
        for i, j in HAND_EDGES:
            p, q = fr[h + i, :2], fr[h + j, :2]
            ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=c, lw=1.2)
        ax.plot(fr[h:h + 21, 0], fr[h:h + 21, 1], ".", color=c, ms=2.5)
    if np.isnan(fr[..., 0]).all():
        ax.text(0.5, 0.5, "결측 프레임", transform=ax.transAxes, ha="center", color="k")


def render(a, title, out_base, gif_fps):
    xlim, ylim = limits(a)
    T = len(a)

    # ---- GIF
    fig, ax = plt.subplots(figsize=(4, 4.5))
    def upd(t):
        draw_frame(ax, a[t], xlim, ylim)
        for h, c in [(LH, C_L), (RH, C_R)]:      # 손목(손 0번) 궤적
            ax.plot(a[: t + 1, h, 0], a[: t + 1, h, 1], "-", color=c, alpha=0.3, lw=1)
        ax.set_title(f"{title}\nframe {t + 1}/{T}", fontsize=9)
    anim = FuncAnimation(fig, upd, frames=T)
    anim.save(f"{out_base}.gif", writer=PillowWriter(fps=gif_fps))
    plt.close(fig)

    # ---- 요약 PNG
    fig = plt.figure(figsize=(12, 6.5))
    idx = np.linspace(0, T - 1, 6).round().astype(int)
    for k, t in enumerate(idx):
        ax = fig.add_subplot(2, 6, k + 1)
        draw_frame(ax, a[t], xlim, ylim)
        ax.set_title(f"t={t}", fontsize=8)
    ax = fig.add_subplot(2, 1, 2)
    for name, h, w, c in [("왼손", LH, B_LWRI, C_L), ("오른손", RH, B_RWRI, C_R)]:
        ax.plot(a[:, h, 1], "-", color=c, label=f"{name} 손목(손 kp)")
        ax.plot(a[:, BODY + w, 1], "--", color=c, alpha=0.6, label=f"{name} 손목(몸 kp)")
    ax.plot(a[:, BODY + B_RSHO, 1], ":", color="k", alpha=0.5, label="어깨 높이")
    ax.invert_yaxis()
    ax.set_xlabel("frame"); ax.set_ylabel("y (위로 갈수록 높음)")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/word/manifest.csv")
    ap.add_argument("--labels", nargs="*", default=[], help="라벨(한글)로 선택. 같은 라벨의 WORD ID가 여러 개면 전부")
    ap.add_argument("--clip-ids", nargs="*", default=[], help="clip_id로 직접 선택")
    ap.add_argument("--split", default="train", help="--labels 사용 시 split (기본 train)")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본: <manifest 폴더>/qa)")
    ap.add_argument("--gif-fps", type=int, default=10)
    args = ap.parse_args()

    man = Path(args.manifest)
    root = man.parent
    out = Path(args.out) if args.out else root / "qa"
    out.mkdir(parents=True, exist_ok=True)
    with open(man, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    sel = [r for r in rows if r["clip_id"] in set(args.clip_ids)]
    for lab in args.labels:
        hit = [r for r in rows if r["label"] == lab and r["split"] == args.split]
        if not hit:
            print(f"[skip] '{lab}' ({args.split}) 없음", file=sys.stderr)
        sel += hit
    if not sel:
        sys.exit("선택된 클립 없음: --labels 또는 --clip-ids를 지정하세요")

    print(f"{'label':8s} {'clip_id':28s} {'T':>4s}  손목 일치도(어깨너비 대비) 왼:중앙/최대  오른:중앙/최대")
    for r in sel:
        p = root / r["path"]
        if not p.exists():
            print(f"[skip] npy 없음: {p}", file=sys.stderr)
            continue
        a = np.load(p)
        wc, _ = wrist_check(a)
        flag = "  ← 의심" if max(wc["L"][0], wc["R"][0]) > 0.3 else ""
        print(f"{r['label']:8s} {r['clip_id']:28s} {len(a):4d}  "
              f"{wc['L'][0]:.2f}/{wc['L'][1]:.2f}   {wc['R'][0]:.2f}/{wc['R'][1]:.2f}{flag}")
        title = f"{r['label']} ({r['word_id']}, {r['signer']}, fps_flag={r.get('fps_flag', '')})"
        render(a, title, out / f"{r['label']}_{r['clip_id']}", args.gif_fps)
    print(f"\n출력: {out}")


if __name__ == "__main__":
    main()