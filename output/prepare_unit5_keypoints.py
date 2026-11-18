"""Convert Unit 5 AIHub frame JSONs to the existing hands-only LSTM manifest."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

LABELS = {
    "SEN0009": "운전 / 천천히",
    "SEN0011": "운전 / 빨리",
    "SEN0012": "가다 / 빨리",
    "SEN0013": "시간 / 급하다",
    "SEN0017": "약속 / 늦다",
}
CLIP = re.compile(r"NIA_SL_(SEN\d+)_(REAL\d+)_([DFLRU])")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    parser.add_argument("output", type=Path)
    parser.add_argument("--all-train", action="store_true")
    parser.add_argument("--signer-splits", type=Path, help="JSON mapping val/test to signer IDs")
    args = parser.parse_args()
    if args.all_train and args.signer_splits:
        parser.error("--all-train and --signer-splits are mutually exclusive")
    split_by_signer = {}
    if args.signer_splits:
        groups = json.loads(args.signer_splits.read_text())
        if set(groups) != {"val", "test"}:
            parser.error("Signer splits must contain val and test")
        for split, signers in groups.items():
            for signer in signers:
                if signer in split_by_signer:
                    parser.error(f"Signer assigned twice: {signer}")
                split_by_signer[signer] = split
    rows = []
    clips = defaultdict(dict)
    seen = set()
    for source in args.archives:
        clips.clear()
        with ZipFile(source) as archive:
            for member in archive.namelist():
                path = Path(member)
                match = CLIP.fullmatch(path.parent.name)
                if not member.endswith("_keypoints.json") or not match or match[1] not in LABELS:
                    continue
                frame = int(path.name.removesuffix("_keypoints.json").rsplit("_", 1)[1])
                if frame in clips[path.parent.name]:
                    raise ValueError(f"Duplicate frame: {member}")
                clips[path.parent.name][frame] = member

            for clip_id, members in sorted(clips.items()):
                if clip_id in seen:
                    raise ValueError(f"Duplicate clip: {clip_id}")
                seen.add(clip_id)
                sentence, signer, angle = CLIP.fullmatch(clip_id).groups()
                frames = []
                for frame in sorted(members):
                    person = json.loads(archive.read(members[frame]))["people"]
                    hands = [person[f"hand_{side}_keypoints_2d"] for side in ("left", "right")]
                    if any(len(hand) != 63 for hand in hands):
                        raise ValueError(f"Expected 21 landmarks per hand: {members[frame]}")
                    frames.append(hands)
                arr = np.asarray(frames, dtype=np.float32).reshape(-1, 42, 3)
                if not np.isfinite(arr).all() or not (arr[..., 2] > 0).any():
                    raise ValueError(f"Invalid landmarks: {clip_id}")
                if args.signer_splits:
                    split = split_by_signer[signer]
                else:
                    split = "train" if args.all_train else {"F": "test", "U": "val"}.get(angle, "train")
                relative = Path("clips") / f"{clip_id}.npy"
                target = args.output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, arr, allow_pickle=False)
                rows.append(dict(clip_id=clip_id, path=str(relative), word_id=sentence,
                                 label=LABELS[sentence], signer=signer, angle=angle,
                                 split=split, n_frames=len(arr)))

    required_splits = ("val", "test") if args.signer_splits else ("train", "val", "test")
    for split in (("train",) if args.all_train else required_splits):
        if {row["word_id"] for row in rows if row["split"] == split} != LABELS.keys():
            raise ValueError(f"Missing sentence class in {split} split")
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} clips, {sum(row['n_frames'] for row in rows)} frames")


if __name__ == "__main__":
    main()
