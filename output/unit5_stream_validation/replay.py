"""Replay validation clips through production adapters and evaluation state machines."""
import csv
import hashlib
import json
from collections import deque
from pathlib import Path
import time

import numpy as np
from web_adapter import HandsWordAdapter
from ai_server.evaluation import Evaluator
from ai_server.sequence import WordSequence, SEQUENCES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
rows = [r for r in csv.DictReader((ROOT / "output/unit5_eval/manifest.csv").open())
        if r["split"] == "val"]
assert len(rows) == 50 and {r["signer"] for r in rows} == {"REAL13", "REAL17"}
models = {
    "sentence": HandsWordAdapter(ROOT / "unit5.pt"),
    "word": HandsWordAdapter(ROOT / "unit5_words.pt"),
}
targets = sorted({r["word_id"] for r in rows})
cache = {}


def predict(kind, clip, samples):
    key = (kind, clip, tuple(index for index, _ in samples))
    if key not in cache:
        cache[key] = models[kind].predict_landmarks([frame for _, frame in samples])
    return cache[key]


def replay(row, target, fps, threshold=.9, second_window=12, stable=2, swap=False):
    arr = np.load(ROOT / "output/unit5_eval" / row["path"])
    # Source timestamps are unavailable in the extracted ZIP. Explicit 30fps assumption.
    indices = np.unique(np.round(np.arange(0, len(arr), 30 / fps)).astype(int))
    indices = indices[indices < len(arr)]
    sequence = WordSequence(target) if target in SEQUENCES else None
    judge = Evaluator(target, threshold=threshold, stable=stable, require_calibrated=False)
    if sequence:
        sequence.judge = Evaluator(sequence.first, threshold=threshold, stable=stable,
                                   require_calibrated=False)
    frames = deque(maxlen=max(12, second_window) if sequence else 32)
    first_time = None
    history = []
    last_inferred = 0

    def infer():
        nonlocal first_time
        samples = list(frames)
        if sequence:
            samples = samples[-(second_window if sequence.stage else 8):]
        prediction = predict("word" if sequence else "sentence", row["clip_id"], samples) if not swap else models["word" if sequence else "sentence"].predict_landmarks([f for _, f in samples])
        stage = sequence.stage if sequence else None
        result = sequence.update(prediction, frames) if sequence else judge.update(prediction)
        if sequence and stage == 0 and sequence.stage == 1:
            first_time = samples[-1][0] / 30
            sequence.judge = Evaluator(sequence.second, threshold=threshold, stable=stable,
                                       require_calibrated=False)
        history.append(dict(source_frame=samples[-1][0], predicted=prediction.predicted_id,
                            confidence=prediction.confidence, result=result["result"]))
        return result

    minimum = 8 if sequence else 16
    for count, index in enumerate(indices, 1):
        hands = []
        for points in (arr[index, :21], arr[index, 21:]):
            hands.append([dict(x=float(x)/8192, y=float(y)/8192, z=0)
                          for x, y, _ in points])
        if swap:
            hands.reverse()
        frame = dict(schema="mediapipe-hands-v1", image_width=8192, image_height=8192,
                     left_hand=hands[0], right_hand=hands[1])
        frames.append((int(index), frame))
        if len(frames) >= minimum and (count - minimum) % 2 == 0:
            result = infer()
            last_inferred = count
            if result["result"] == "PASS":
                break
    else:
        # Same stop/final inference policy as the production WebSocket handler.
        if len(frames) >= minimum and len(indices) > last_inferred:
            result = infer()
    passed = bool(history and history[-1]["result"] == "PASS")
    return dict(clip=row["clip_id"], signer=row["signer"], angle=row["angle"],
                expected=row["word_id"], target=target, passed=passed,
                pass_seconds=history[-1]["source_frame"]/30 if passed else None,
                first_word_seconds=first_time, predictions=history)


def summarize(results):
    correct = [r for r in results if r["target"] == r["expected"]]
    wrong = [r for r in results if r["target"] != r["expected"]]
    def rate(items):
        return dict(passes=sum(r["passed"] for r in items), attempts=len(items))
    times = [r["pass_seconds"] for r in correct if r["passed"]]
    return dict(correct=rate(correct), wrong=rate(wrong),
                median_pass_seconds=float(np.median(times)) if times else None,
                front_correct=rate([r for r in correct if r["angle"] == "F"]),
                by_target={t:rate([r for r in correct if r["target"] == t]) for t in targets})

configs = [
    ("baseline_8fps", 8, .9, 12, 2),
    ("baseline_6fps", 6, .9, 12, 2),
    ("baseline_4fps", 4, .9, 12, 2),
    ("threshold_080_8fps", 8, .8, 12, 2),
    ("second_window_16_8fps", 8, .9, 16, 2),
    ("stable3_8fps", 8, .9, 12, 3),
]
report = {}
for name, fps, threshold, window, stable in configs:
    started = time.monotonic()
    results = [replay(row, target, fps, threshold, window, stable)
               for row in rows for target in targets]
    report[name] = dict(summary=summarize(results), results=results)
    print(name, json.dumps(report[name]["summary"]), round(time.monotonic()-started,1), flush=True)
(OUT / "results.json").write_text(json.dumps(dict(
    source_fps_assumption=30,
    checkpoint_sha256={kind:hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
                       for kind,file in (("sentence","unit5.pt"),("word","unit5_words.pt"))},
    experiments=report), ensure_ascii=False, indent=2)+"\n")
