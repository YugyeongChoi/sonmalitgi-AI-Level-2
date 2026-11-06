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
from .evaluation import Evaluator
from .model import load_model, load_word_model
from .landmarks import decode_landmarks

WINDOW = 16
STRIDE = 4
MAX_BYTES = 150_000
MAX_SECONDS = 30
ORIGINS = set(os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","))

@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()
    app.state.word_model = load_word_model()
    # Serialize inference for adapters that are not thread-safe.
    app.state.model_lock = asyncio.Lock()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_mode": app.state.model.mode,
            "word_model_mode": app.state.word_model.mode,
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
        predict = getattr(model, "predict_landmarks", None) if input_mode == "landmarks" else model.predict
        if not callable(predict):
            await ws.send_json({"type": "error", "message": "서버 모델이 랜드마크 입력을 지원하지 않습니다. predict_landmarks 어댑터를 연결해 주세요."})
            return
        previous_timestamp = -1
        # User-enabled practice policy; it does not claim model calibration.
        judge = (Evaluator(target, threshold=.5, stable=1, require_calibrated=False)
                 if word_session else Evaluator(target))
        frames = deque(maxlen=WINDOW)
        count = 0
        latest = dict(result="CAPTURE_ERROR", predicted_id=None, confidence=0,
                      feedback="분석할 프레임이 부족합니다.")
        last_inferred = 0
        async def infer():
            async with app.state.model_lock:
                prediction = await asyncio.to_thread(predict, list(frames))
            return judge.update(prediction)
        began = time.monotonic()
        await ws.send_json(dict(type="ready", attempt_id=attempt_id,
                                model_mode=model.mode, input_mode=input_mode, fps=8, window=WINDOW))
        while True:
            event = await receive()
            elapsed = time.monotonic() - began
            if elapsed > MAX_SECONDS:
                if len(frames) == WINDOW and count > last_inferred:
                    latest = await infer()
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
                if len(frames) == WINDOW and count > last_inferred:
                    latest = await infer()
                await ws.send_json(dict(type="final", attempt_id=attempt_id,
                                        target_id=target, model_mode=model.mode, **latest))
                break
            if input_mode == "landmarks":
                if message is None:
                    raise ValueError("Landmark JSON required")
                if word_session and message.get("schema") != "mediapipe-hands-v1":
                    raise ValueError("Hands-only schema required")
                frame = decode_landmarks(message, previous_timestamp)
                previous_timestamp = frame["timestamp_ms"]
            else:
                data = event.get("bytes")
                if data is None:
                    raise ValueError("JPEG binary required")
                frame = await asyncio.to_thread(decode_frame, data)
            frames.append(frame)
            count += 1
            # One client frame is in flight; server acknowledges after any inference.
            if len(frames) == WINDOW and (count - WINDOW) % STRIDE == 0:
                latest = await infer()
                last_inferred = count
                await ws.send_json(dict(type="prediction", attempt_id=attempt_id,
                                        target_id=target, frame_count=count,
                                        model_mode=model.mode, **latest))
                # PASS is terminal; FAIL is provisional until user stops.
                if latest["result"] == "PASS":
                    await ws.send_json(dict(type="final", attempt_id=attempt_id,
                                            target_id=target, model_mode=model.mode, **latest))
                    break
            await ws.send_json(dict(type="ack", frame_count=count))
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
