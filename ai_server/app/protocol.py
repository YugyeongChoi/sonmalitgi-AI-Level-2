"""
WebSocket 메시지 형식 (JSON). 엔드포인트: ws://<host>/ws/practice

클라이언트 → 서버
  {"type":"start", "lesson":"word", "target":"WORD1157", "width":1280, "height":720, "mirrored":false}
  {"type":"frame", "t":1234.5,
   "pose":[[x,y,z,visibility] × 33] | null,             ← MediaPipe PoseLandmarker 정규화 좌표 (0~1)
   "hands":[{"landmarks":[[x,y,z] × 21], "handedness":"Left", "score":0.97}, ...]}
  {"type":"end"}       ← 이 시점까지의 프레임으로 판정
  {"type":"cancel"}    ← 버퍼 비우고 대기 상태로

서버 → 클라이언트
  {"type":"ready", "lessons":{"word": 21}}
  {"type":"started", "lesson":"word", "target":"WORD1157"}
  {"type":"status", "frames":45, "body":true, "left_hand":false, "right_hand":true}
  {"type":"result", "lesson":"word", "target":"WORD1157", "correct":true, "target_prob":0.93,
   "top":[{"id":"WORD1157","label":"나","prob":0.93}, ...], "frames":72, "used_frames":[3, 70]}
  {"type":"error", "code":"not_started", "message":"..."}

※ mirrored: MediaPipe에 넣은 영상이 좌우반전된 영상이면 true.
   권장은 원본 영상으로 MediaPipe를 돌리고 화면 표시만 CSS로 반전(false).
"""
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class StartMsg(BaseModel):
    type: Literal["start"]
    lesson: str = "word"
    target: Optional[str] = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mirrored: bool = False


class HandMsg(BaseModel):
    landmarks: List[List[float]]
    handedness: Optional[str] = None
    score: Optional[float] = None


class FrameMsg(BaseModel):
    type: Literal["frame"]
    t: Optional[float] = None
    pose: Optional[List[List[float]]] = None
    hands: List[HandMsg] = []


class EndMsg(BaseModel):
    type: Literal["end"]


class CancelMsg(BaseModel):
    type: Literal["cancel"]


ClientMsg = Annotated[Union[StartMsg, FrameMsg, EndMsg, CancelMsg], Field(discriminator="type")]
