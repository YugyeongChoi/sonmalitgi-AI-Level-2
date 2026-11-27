from .metrics import AttemptMetrics
from .sequence import SEQUENCES, WordSequence
import asyncio
from collections import deque
from contextlib import asynccontextmanager
from io import BytesIO
import json
import os
import time
from uuid import UUID

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError
from .evaluation import Evaluator, CumulativeEvaluator
from .model import load_model, load_word_model
from .landmarks import decode_landmarks

WINDOW = 16
SENTENCE_MIN_FRAMES = 16
SENTENCE_WINDOW = 32
STRIDE = 4
MAX_BYTES = 150_000
MAX_SECONDS = 30
ORIGINS = set(os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","))

@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()
    app.state.word_model = load_word_model()
    app.state.unit5_model = None
    if checkpoint := os.getenv("UNIT5_CHECKPOINT"):
        from web_adapter import HandsWordAdapter
        app.state.unit5_model = HandsWordAdapter(checkpoint)
    app.state.unit5_words = None
    if checkpoint := os.getenv("UNIT5_WORD_CHECKPOINT"):
        from web_adapter import HandsWordAdapter
        app.state.unit5_words = HandsWordAdapter(checkpoint)
    # Serialize inference for adapters that are not thread-safe.
    app.state.model_lock = asyncio.Lock()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_mode": app.state.model.mode,
            "word_model_mode": app.state.word_model.mode,
            "word_input_schema": getattr(app.state.word_model,"input_schema",None),
            "word_source_revision": "7ad3178e11f955205f0b9937660d625904dafab6",
            "unit5_supported_ids": sorted(getattr(app.state.unit5_model, "supported_ids", [])),
            "unit5_sequence_ids": sorted(SEQUENCES) if app.state.unit5_words else [],
            "word_supported_ids": sorted(getattr(app.state.word_model, "supported_ids", []))}

def decode_frame(data: bytes):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("프레임 크기 제한 초과")
    with Image.open(BytesIO(data)) as img:
        if img.format != "JPEG" or not (32 <= img.width <= 640 and 32 <= img.height <= 480):
            raise ValueError("640×480 이하 JPEG 프레임이 필요합니다")
        return img.convert("RGB")

@app.websocket("/ws/v1/word/evaluate")
@app.websocket("/ws/v1/evaluate")
async def evaluate(ws: WebSocket):
    if ws.headers.get("origin") not in ORIGINS:
        await ws.close(code=1008)
        return
    await ws.accept()
    word_session = ws.url.path == "/ws/v1/word/evaluate"
    model = app.state.word_model if word_session else app.state.model
    async def receive():
        event = await asyncio.wait_for(ws.receive(), timeout=10)
        if event["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(event.get("code", 1000))
        return event
    try:
        start = await receive()
        raw = start.get("text") or ""
        if len(raw) > 2048:
            raise ValueError("시작 메시지 크기 제한 초과")
        config = json.loads(raw)
        if not isinstance(config, dict) or config.get("type") != "start":
            raise ValueError("start 메시지가 필요합니다")
        attempt_id = str(UUID(config["attempt_id"]))
        target = config["target_id"]
        if not isinstance(target, str) or not 1 <= len(target) <= 64:
            raise ValueError("잘못된 target_id")
        input_mode = config.get("input_mode", "jpeg")
        if input_mode not in ("jpeg", "landmarks"):
            raise ValueError("Unknown input mode")
        unit5_model = getattr(app.state, "unit5_model", None)
        sentence_session = (
            word_session and unit5_model is not None and target in unit5_model.supported_ids
        )
        if sentence_session:
            model = unit5_model
        if word_session:
            if input_mode != "landmarks":
                raise ValueError("Word model requires landmarks")
            if model.mode == "demo":
                await ws.send_json({"type": "error", "code": "MODEL_NOT_READY",
                                    "message": "중급 손 전용 수어 모델을 준비 중입니다. 현재는 인식 평가를 제공하지 않습니다."})
                return
            if target not in model.supported_ids:
                await ws.send_json({"type": "error", "code": "UNSUPPORTED_TARGET",
                                    "message": "이 표현은 현재 중급 모델의 지원 단어에 포함되지 않습니다."})
                return
        sequence = (WordSequence(target)
                    if sentence_session and target in SEQUENCES and app.state.unit5_words else None)
        if sequence:
            model = app.state.unit5_words
        predict = getattr(model, "predict_landmarks", None) if input_mode == "landmarks" else model.predict
        if not callable(predict):
            await ws.send_json({"type": "error", "message": "서버 모델이 랜드마크 입력을 지원하지 않습니다. predict_landmarks 어댑터를 연결해 주세요."})
            return
        previous_timestamp = -1
        # User-enabled practice policy; it does not claim model calibration.
        judge = (CumulativeEvaluator(target, threshold=.5, required=3)
                 if word_session else Evaluator(target))
        if sentence_session:
            judge = Evaluator(target, threshold=.9, stable=3, require_calibrated=False)
        min_frames = 8 if sequence else SENTENCE_MIN_FRAMES if sentence_session else WINDOW
        frames = deque(maxlen=12 if sequence else SENTENCE_WINDOW if sentence_session else WINDOW)
        stride = 2 if sentence_session else STRIDE
        count = 0
        latest = dict(result="CAPTURE_ERROR", predicted_id=None, confidence=0,
                      feedback=("카메라가 켜진 뒤 문장을 끝까지 표현해 주세요. "
                                f"촬영 프레임이 {min_frames}개 이상 필요합니다."
                                if sentence_session else "분석할 프레임이 부족합니다."))
        metrics = AttemptMetrics()
        last_inferred = 0
        async def infer():
            samples = list(frames)
            if sequence:
                samples = samples[-sequence.window:]
            queued_at = time.monotonic()
            async with app.state.model_lock:
                inference_started = time.monotonic()
                prediction = await asyncio.to_thread(predict, samples)
            inference_finished = time.monotonic()
            previous_success = getattr(judge, "success_count", 0)
            outcome = sequence.update(prediction, frames) if sequence else judge.update(prediction)
            if isinstance(judge, CumulativeEvaluator) and judge.success_count > previous_success:
                # 같은 겹친 창으로 여러 번 적립하지 않고 새 좌표 구간을 받는다.
                frames.clear()
            if outcome["result"] == "PASS" and metrics.pass_seconds is None:
                metrics.pass_seconds = round(inference_finished - metrics.started, 3)
            window_seconds = (
                (samples[-1]["timestamp_ms"] - samples[0]["timestamp_ms"]) / 1000
                if input_mode == "landmarks" and len(samples) > 1 else None
            )
            outcome["metrics"] = dict(
                **metrics.snapshot(),
                window_frames=len(samples),
                window_seconds=round(window_seconds, 3) if window_seconds is not None else None,
                queue_ms=round((inference_started - queued_at) * 1000, 2),
                inference_ms=round((inference_finished - inference_started) * 1000, 2),
                raw_predicted_id=prediction.predicted_id,
                raw_confidence=prediction.confidence,
            )
            return outcome
        began = time.monotonic()
        await ws.send_json(dict(type="ready", attempt_id=attempt_id,
                                model_mode="unit5-experimental-v1" if sentence_session else model.mode,
                                input_mode=input_mode, fps=8, window=min_frames, stride=stride,
                                server_revision=os.getenv("RAILWAY_GIT_COMMIT_SHA", "local")))
        while True:
            event = await receive()
            elapsed = time.monotonic() - began
            if elapsed > MAX_SECONDS:
                if len(frames) >= min_frames and count > last_inferred:
                    latest = await infer()
                latest["metrics"] = dict(latest.get("metrics", {}), **metrics.snapshot())
                await ws.send_json(dict(type="final", attempt_id=attempt_id,
                                        target_id=target, model_mode=model.mode, **latest))
                break
            message = None
            if event.get("text") is not None:
                raw = event["text"]
                if len(raw.encode("utf-8")) > MAX_BYTES:
                    raise ValueError("Message too large")
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("Object required")
            if message is not None and message.get("type") == "stop":
                if len(frames) >= min_frames and count > last_inferred:
                    latest = await infer()
                latest["metrics"] = dict(latest.get("metrics", {}), **metrics.snapshot())
                await ws.send_json(dict(type="final", attempt_id=attempt_id,
                                        target_id=target, model_mode=model.mode, **latest))
                break
            if input_mode == "landmarks":
                if message is None:
                    raise ValueError("Landmark JSON required")
                if word_session and message.get("schema") != getattr(model, "input_schema", "mediapipe-hands-v1"):
                    raise ValueError("Hands-only schema required")
                frame = decode_landmarks(message, previous_timestamp)
                previous_timestamp = frame["timestamp_ms"]
            else:
                data = event.get("bytes")
                if data is None:
                    raise ValueError("JPEG binary required")
                frame = await asyncio.to_thread(decode_frame, data)
            frames.append(frame)
            metrics.receive_frame()
            count += 1
            # One client frame is in flight; server acknowledges after any inference.
            if len(frames) >= min_frames and (count - min_frames) % stride == 0:
                latest = await infer()
                last_inferred = count
                await ws.send_json(dict(type="prediction", attempt_id=attempt_id,
                                        target_id=target, frame_count=count,
                                        model_mode=model.mode, **latest))
                # PASS is terminal; FAIL is provisional until user stops.
                if latest["result"] == "PASS":
                    latest["metrics"] = dict(latest.get("metrics", {}), **metrics.snapshot())
                    await ws.send_json(dict(type="final", attempt_id=attempt_id,
                                            target_id=target, model_mode=model.mode, **latest))
                    break
            await ws.send_json(dict(type="ack", frame_count=count, metrics=metrics.snapshot()))
    except WebSocketDisconnect:
        return
    except (ValueError, KeyError, TypeError, UnidentifiedImageError, Image.DecompressionBombError):
        await ws.send_json({"type": "error", "message": "입력 형식 또는 프레임을 확인해 주세요."})
    except asyncio.TimeoutError:
        await ws.send_json({"type": "error", "message": "수신 시간이 초과됐습니다. 다시 시작해 주세요."})
    except Exception:
        import logging
        logging.exception("Evaluation failed")
        await ws.send_json({"type": "error", "message": "추론 처리에 실패했습니다."})
    finally:
        if ws.client_state.name != "DISCONNECTED":
            try:
                await ws.close()
            except RuntimeError:
                pass


@app.websocket("/ws/v1/dictionary/recognize")
async def recognize(ws: WebSocket):
    """Continuous, target-free recognition; never awards learning progress."""
    from .recognition import RecognitionState
    if ws.headers.get("origin") not in ORIGINS:
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        if len(raw.encode("utf-8")) > 2048:
            raise ValueError("Start too large")
        config = json.loads(raw)
        if not isinstance(config, dict) or config.get("type") != "start":
            raise ValueError("Start required")
        scope = config.get("scope", "intermediate")
        if scope not in ("intermediate", "unit5"):
            raise ValueError("Unknown scope")
        model = app.state.word_model if scope == "intermediate" else app.state.unit5_words
        if model is None or model.mode == "demo":
            await ws.send_json(dict(type="error", message="이 범위의 단어 모델을 준비 중입니다."))
            return
        supported = list(model.supported_ids)
        state = RecognitionState(supported)
        frames = deque(maxlen=16 if scope == "intermediate" else 12)
        minimum = 16 if scope == "intermediate" else 8
        count, previous = 0, -1
        await ws.send_json(dict(type="ready", supported_ids=supported, scope=scope))
        began = time.monotonic()
        while time.monotonic() - began < 120:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=15)
            if len(raw.encode("utf-8")) > MAX_BYTES:
                raise ValueError("Frame too large")
            message = json.loads(raw)
            if not isinstance(message, dict):
                raise ValueError("Object required")
            if message.get("type") == "stop":
                return
            if message.get("schema") != getattr(model, "input_schema", "mediapipe-hands-v1"):
                raise ValueError("Hands schema required")
            frame = decode_landmarks(message, previous)
            previous = frame["timestamp_ms"]
            if not frame["left_hand"] and not frame["right_hand"]:
                frames.clear()
                state.reset()
                count = 0
                await ws.send_json(dict(type="prediction", predicted_id=None, confidence=0,
                                        stable=False, message="양손을 카메라에 보여 주세요."))
            else:
                frames.append(frame)
                count += 1
                if len(frames) >= minimum and (count - minimum) % 2 == 0:
                    async with app.state.model_lock:
                        prediction = await asyncio.to_thread(model.predict_landmarks, list(frames))
                    await ws.send_json(dict(type="prediction", **state.update(prediction)))
            await ws.send_json(dict(type="ack"))
        await ws.send_json(dict(type="ended", message="2분이 지났습니다. 다시 시작해 주세요."))
    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        await ws.send_json(dict(type="error", message="연결 시간이 초과됐습니다. 다시 시작해 주세요."))
    except (ValueError, KeyError, TypeError):
        await ws.send_json(dict(type="error", message="입력 형식을 확인해 주세요."))
    except Exception:
        import logging
        logging.exception("Dictionary recognition failed")
        await ws.send_json(dict(type="error", message="인식 처리에 실패했습니다. 다시 시작해 주세요."))
    finally:
        if ws.client_state.name != "DISCONNECTED":
            try:
                await ws.close()
            except RuntimeError:
                pass
