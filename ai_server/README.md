> **hands-only 브랜치:** 아래 내용은 원본 상체 모델 안내입니다. 현재 손 전용 모델의 실행·입력·제약은 루트 [README](../README.md)를 기준으로 하세요. 기존 best.pt는 사용할 수 없습니다.

# KSL AI 서버 (FastAPI)

브라우저가 웹캠 영상에서 MediaPipe로 좌표를 뽑아 WebSocket으로 보내면, 서버가 학습 모델로 판정해 돌려준다.

```
브라우저: 웹캠 → MediaPipe(Hand + Pose) → 좌표 JSON ──WebSocket──▶ 서버
서버: MediaPipe → 54점 변환 → 구간 처리 → 전처리(학습과 동일) → LSTM → 결과 JSON ──▶ 브라우저
```

## 결정 사항

| 항목 | 결정 |
| --- | --- |
| 클라이언트 | 웹 브라우저 |
| MediaPipe 위치 | 클라이언트 (서버에는 좌표만, 영상은 보내지 않음) |
| 통신 | WebSocket 스트리밍 (`/ws/practice`) |
| 수어 구간 | **미정** → `KSL_SEGMENTER`로 교체 가능. 지금은 클라이언트의 start~end 그대로 |
| MediaPipe→54점 변환 | 서버 (`app/landmarks.py`) |

## 폴더 구조

```
ai_server/
  app/
    main.py         엔드포인트 (/health, /lessons, /ws/practice, /demo)
    protocol.py     WebSocket 메시지 형식  ← 프론트엔드 개발 시 이 파일 기준
    landmarks.py    MediaPipe 좌표 → 학습 데이터와 같은 54점 배열
    session.py      연결별 프레임 버퍼, 판정
    recognizers.py  레슨 종류별 모델 (word, 나중에 fingerspell 등)
    segmenter.py    수어 구간 처리 (교체 가능)
    config.py       환경 변수 설정
  ml/
    word_dataset.py, word_model.py   ← 학습 코드와 같은 파일 (바뀌면 같이 복사)
  models/word/      best.pt, classes.json 두기 (기본 경로)
  static/index.html 브라우저 테스트 페이지
  tools/ws_smoke_test.py  웹캠 없이 npy 클립으로 서버 검증
```

## 실행 (Windows)

```cmd
cd ai_server
python -m pip install -r requirements.txt

:: 학습 결과 복사 (runs\<시각>\ 폴더의 두 파일)
mkdir models\word
copy ..\runs\<시각>\best.pt models\word\
copy ..\runs\<시각>\classes.json models\word\

python -m uvicorn app.main:app --reload --port 8000
```

- http://localhost:8000/health 에서 `"lessons": ["word"]`가 보이면 모델이 로드된 것
- http://localhost:8000/demo/ : 카메라 켜기 → 목표 단어 선택 → 녹화 시작 → 수어 → 끝·판정
- 웹캠은 `localhost` 또는 HTTPS에서만 동작한다. 다른 PC나 폰에서 접속하려면 HTTPS가 필요하다.

### 웹캠 없이 확인

서버를 켠 상태에서 학습 데이터 클립을 보내 본다. 21단어 모델이라면 val 클립 결과가 `train_word_lstm.py`의 val 예측과 거의 같아야 한다.

```cmd
python tools\ws_smoke_test.py --npy ..\data\word\clips\NIA_SL_WORD1157_REAL02_F.npy --target WORD1157
```

## 환경 변수

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `KSL_MODELS` | `word=models/word/best.pt` | `레슨=경로;레슨=경로` |
| `KSL_DEVICE` | `cpu` | `cuda` 가능 |
| `KSL_SEGMENTER` | `none` | `none` / `hands` (쉬는 손 구간을 앞뒤에서 잘라냄, 실험) |
| `KSL_TARGET_THRESHOLD` | `0.5` | 목표 단어가 1위이고 확률이 이 값 이상이면 정답 |
| `KSL_TOP_K` | `3` | 결과에 담을 상위 예측 수 |
| `KSL_MAX_FRAMES` / `KSL_MIN_FRAMES` | `300` / `8` | 한 번의 시도에서 받을 프레임 수 범위 |
| `KSL_STATUS_EVERY` | `5` | 녹화 중 상태(손 보임 여부)를 보내는 프레임 간격 |

## WebSocket 프로토콜

`app/protocol.py` 맨 위 주석에 전체 형식이 있다. 요약:

```
→ {"type":"start","lesson":"word","target":"WORD1157","width":1280,"height":720,"mirrored":false}
→ {"type":"frame","t":..., "pose":[[x,y,z,vis]×33], "hands":[{"landmarks":[[x,y,z]×21],"handedness":"Left"}]}  (반복)
← {"type":"status","frames":45,"body":true,"left_hand":false,"right_hand":true}                              (5프레임마다)
→ {"type":"end"}
← {"type":"result","correct":true,"target_prob":0.93,"top":[{"id":"WORD1157","label":"나","prob":0.93},...]}
```

## 주의

- **MediaPipe는 원본(좌우반전 안 된) 영상으로 돌리고**, 화면 표시만 CSS `transform: scaleX(-1)`로 반전한다. 반전 영상을 넣었다면 `mirrored: true`.
- 손 좌우는 MediaPipe handedness 라벨이 아니라 포즈의 손목 위치로 배정한다. 라벨은 셀카 기준이라 원본 영상에서는 반대로 나온다.
- `ml/`의 두 파일은 학습 코드와 같아야 한다. 전처리가 다르면 학습 때와 다른 입력이 들어가 정확도가 떨어진다.
- 팀원 모델(지문자·지숫자)은 `recognizers.py`에 `Recognizer`를 상속한 클래스를 만들고 `LOADERS`에 등록한다.

## 남은 결정

- 수어 구간: 카운트다운 고정 녹화 / 버튼 / 자동 감지 → `segmenter.py`에 구현 추가
- 배포 위치(학교 서버, 클라우드)와 HTTPS 설정
- 오답 분석용 좌표 저장 여부
