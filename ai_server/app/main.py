"""
KSL AI 서버 (FastAPI)

  GET  /health          상태 확인
  GET  /lessons         레슨별 클래스 목록 (앱이 목표 단어 고를 때)
  WS   /ws/practice     연습 세션 (프로토콜은 app/protocol.py 참고)
  GET  /demo/           브라우저 테스트 페이지 (웹캠 + MediaPipe)

실행:  uvicorn app.main:app --reload --port 8000
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter, ValidationError

from .config import ROOT, settings
from .protocol import CancelMsg, ClientMsg, EndMsg, FrameMsg, StartMsg
from .recognizers import load_registry
from .segmenter import make_segmenter
from .session import PracticeSession, SessionError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ksl")
_msg = TypeAdapter(ClientMsg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = load_registry(settings.models, settings.device)
    app.state.segmenter = make_segmenter(settings.segmenter)
    if not app.state.registry:
        log.warning("로드된 모델이 없습니다. KSL_MODELS 환경 변수나 models/word/best.pt를 확인하세요")
    yield


app = FastAPI(title="KSL AI Server", lifespan=lifespan)
# 개발 중에는 프론트엔드(다른 포트)에서 접근할 수 있게 전부 허용. 배포 시 도메인으로 좁힐 것
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "lessons": list(app.state.registry), "segmenter": settings.segmenter}


@app.get("/lessons")
def lessons():
    return {k: r.classes for k, r in app.state.registry.items()}


@app.websocket("/ws/practice")
async def practice(ws: WebSocket):
    await ws.accept()
    sess = PracticeSession(app.state.registry, app.state.segmenter, settings)
    await ws.send_json({"type": "ready",
                        "lessons": {k: len(r.classes) for k, r in app.state.registry.items()}})
    try:
        while True:
            raw = await ws.receive_json()
            try:
                msg = _msg.validate_python(raw)
                if isinstance(msg, FrameMsg):
                    out = sess.add_frame(msg)
                elif isinstance(msg, StartMsg):
                    out = sess.start(msg)
                elif isinstance(msg, EndMsg):
                    out = await asyncio.to_thread(sess.finish)     # 추론은 스레드에서 (루프 안 막음)
                elif isinstance(msg, CancelMsg):
                    sess.cancel()
                    out = {"type": "cancelled"}
                if out:
                    await ws.send_json(out)
            except ValidationError as e:
                await ws.send_json({"type": "error", "code": "bad_message",
                                    "message": e.errors(include_url=False, include_context=False, include_input=False)[:3]})
            except SessionError as e:
                await ws.send_json({"type": "error", "code": e.code, "message": e.message})
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("세션 오류")
        await ws.close(code=1011)


app.mount("/demo", StaticFiles(directory=ROOT / "static", html=True), name="demo")


@app.get("/")
def root():
    return RedirectResponse("/demo/")
