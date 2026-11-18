import csv
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np

root = Path(__file__).resolve().parent
source = root.parent / "unit5_lstm_real01_to_real10/data"
archive = next((Path.home() / "Downloads").rglob("01_real_sen_morpheme.zip.part0"))
targets = {"SEN0009": ("운전", "천천히"), "SEN0011": ("운전", "빨리")}
rows = []
(root / "clips").mkdir(exist_ok=True)
with ZipFile(archive) as labels, (source / "manifest.csv").open() as manifest:
    names = {Path(n).name: n for n in labels.namelist() if n.endswith("_morpheme.json")}
    for row in csv.DictReader(manifest):
        if row["word_id"] not in targets:
            continue
        clip = row["clip_id"]
        annotation = json.loads(labels.read(names[f"{clip}_morpheme.json"]))
        segments = annotation["data"]
        words = tuple(segment["attributes"][0]["name"] for segment in segments)
        if row["signer"] == "REAL10" and row["word_id"] == "SEN0009" and words == ("운전", "느리다"):
            words = targets["SEN0009"]
        if words != targets[row["word_id"]]:
            raise ValueError((clip, words))
        frames = np.load(source / row["path"])
        fps = len(frames) / annotation["metaData"]["duration"]
        for segment, word in zip(segments, words):
            start = round(segment["start"] * fps)
            end = round(segment["end"] * fps)
            if not 0 <= start < end <= len(frames):
                raise ValueError((clip, start, end))
            relative = f"clips/{clip}_{word}.npy"
            np.save(root / relative, frames[start:end])
            rows.append(dict(path=relative, word_id=word, label=word, split="train", signer=row["signer"]))
with (root / "manifest.csv").open("w", newline="") as manifest:
    writer = csv.DictWriter(manifest, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} labeled word clips")
