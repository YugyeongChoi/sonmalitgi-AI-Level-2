"""Hands-only v1: (T,42,3) x/y/conf -> (32,170); no pose or depth features.
Original (T,54,3) clips are accepted, but sliced before any preprocessing.
"""
from pathlib import Path
import json
import numpy as np
from word_dataset import (Config, WordDataset as BodyDataset, read_manifest, resolve_words,
                          mask_low_conf, interp_short_gaps, aug_spatial, resample,
                          worker_init_fn, USE_FLAGS_DEFAULT, torch)

FEATURE_SCHEMA = "hands-xy-palm-v1"
N_FEAT = 170


def hands_xy(arr, cfg):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 3 or a.shape[1] not in (42, 54) or a.shape[2] != 3 or len(a) == 0:
        raise ValueError("Expected nonempty (T,42|54,3) x/y/conf")
    a = a[:, :42].copy()  # body is removed before masking/interpolation/normalization
    a[~np.isfinite(a)] = np.nan
    return interp_short_gaps(mask_low_conf(a, cfg.conf_thr), cfg.max_gap)


def normalize_hands(xy):
    # One fixed clip origin preserves wrist trajectories and distance between hands.
    wrists = xy[:, [0, 21]].reshape(-1, 2)
    valid = wrists[np.isfinite(wrists).all(-1)]
    origin = np.median(valid, axis=0) if len(valid) else np.zeros(2, np.float32)
    sizes = np.linalg.norm(xy[:, [9, 30]] - xy[:, [0, 21]], axis=-1).ravel()
    sizes = sizes[np.isfinite(sizes) & (sizes > 1e-6)]
    scale = float(np.median(sizes)) if len(sizes) else 1.0
    return (xy - origin) / scale


def make_features(arr, cfg, rng=None):
    aug = rng is not None and cfg.augment
    a = arr
    if aug and rng.random() < cfg.p_decimate and len(a) >= 12:
        k = int(rng.integers(2, 4))
        a = a[int(rng.integers(0, k))::k]
    xy = normalize_hands(hands_xy(a, cfg))
    if aug:
        xy = aug_spatial(xy, cfg, rng)
        # Body-relative resting-hand dropout is intentionally not used.
        for base in (0, 21):
            if rng.random() < cfg.p_drop_seg and len(xy) > cfg.seg_len[1]:
                length = int(rng.integers(cfg.seg_len[0], cfg.seg_len[1] + 1))
                start = int(rng.integers(0, len(xy) - length + 1))
                xy[start:start + length, base:base + 21] = np.nan
    xy = resample(xy, cfg.T, crop=cfg.crop if aug else 0,
                  warp=cfg.warp if aug else 0, rng=rng if aug else None)
    features = [xy.reshape(cfg.T, -1)]
    flags = []
    for base in (0, 21):
        hand = xy[:, base:base + 21]
        sizes = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=-1)
        sizes = sizes[np.isfinite(sizes) & (sizes > 1e-6)]
        shape = ((hand - hand[:, :1]) / np.median(sizes)
                 if len(sizes) else np.full_like(hand, np.nan))
        features.append(shape.reshape(cfg.T, -1))
        flags.append(np.isfinite(hand[:, 0]).all(-1).astype(np.float32)[:, None])
    return np.nan_to_num(np.concatenate(features + flags, axis=1), nan=0., posinf=0., neginf=0.).astype(np.float32)


class WordDataset(BodyDataset):
    def __getitem__(self, i):
        x = make_features(self.arrs[i], self.cfg, self.rng if self.train else None)
        y = self.cls_index[self.items[i]["word_id"]]
        if torch is not None:
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        return x, y


def build_datasets(manifest, words=None, labels=None, cfg=None,
                   use_flags=USE_FLAGS_DEFAULT, out_classes=None):
    rows = read_manifest(manifest)
    ids = list(words or [])
    if labels:
        ids += resolve_words(rows, labels)
    if not ids:
        ids = sorted({r["word_id"] for r in rows if r["split"] == "train"})
    classes = sorted(set(ids))
    names = {r["word_id"]: r["label"] for r in rows}
    if out_classes:
        Path(out_classes).write_text(json.dumps([
            dict(index=i, word_id=w, label=names.get(w, ""))
            for i, w in enumerate(classes)], ensure_ascii=False, indent=1), encoding="utf-8")
    return (WordDataset(manifest, "train", classes, cfg, use_flags),
            WordDataset(manifest, "val", classes, cfg, use_flags), classes)
