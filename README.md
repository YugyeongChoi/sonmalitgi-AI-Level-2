# 손말잇기 중급 — 손 전용 모델 실험

원본: https://github.com/sonmal-itgi/sonmalitgi-AI-Level-2
개인 포크의 `hands-only` 브랜치입니다. 상체 모델 원본은 main에 유지합니다.

## 변경 사항

- 입력은 사람 기준 왼손 21점 + 오른손 21점의 `(T,42,3)` x/y/conf입니다.
- 기존 `(T,54,3)` 학습 데이터는 첫 42점만 잘라 사용합니다. 몸 좌표는 정규화에도 사용하지 않습니다.
- 목 원점·어깨너비 대신 클립 내 손목 중앙값을 원점으로, 손목~중지 MCP 거리 중앙값을 크기로 사용합니다.
- 손 위치 84개 + 손목 기준 손 모양 84개 + 손 존재 2개 = 170개 특징입니다.
- MediaPipe 깊이 z는 학습 데이터의 신뢰도 conf와 다릅니다. 서버가 x/y를 픽셀로 복원하고 검출 손의 conf를 1로 지정합니다. 깊이는 사용하지 않습니다.
- 새 체크포인트에는 `feature_schema=hands-xy-palm-v1`을 저장합니다. 원본 상체 체크포인트는 명시적으로 거부합니다.
- 브라우저 데모에서 PoseLandmarker를 제거했습니다. 외부 웹사이트의 손 검출 코드는 변경하지 않았습니다.
- 몸 기준 쉬는 손 구간 자르기를 제거했습니다. `KSL_SEGMENTER=none`만 지원합니다.

원본 word_dataset.py는 데이터 로딩·시간 보간 등의 공용 함수와 원본 참고용으로 남겨두었습니다.
현재 학습/추론의 특징 생성은 hands_dataset.py만 사용합니다.
서버 ml/hands_dataset.py와 루트 hands_dataset.py는 동일해야 하며 테스트에서 비교합니다.

## 설치와 학습 (저장소 루트)

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r ai_server/requirements.txt
pip install pytest
python train_word_lstm.py --words WORD1290 WORD1201 WORD1589 WORD1191 WORD1185 WORD1342 WORD1343 WORD1345 WORD1149 WORD1295 WORD1274 --epochs 60 --eval-every 10 --batch 16 --steps 2 --hidden 64 --layers 1 --out runs/hands_intermediate_experiment
python -m pytest tests -q
```

## 시험 결과 및 남은 검증

11단어, train 11개 / val 11개, seed 0, 위 설정으로 CPU에서 학습했습니다.
마지막 학습 정확도 100%, 검증 top-1 9/11(81.8%)입니다.
검증 오인은 오다→달리다, 알다→달리다였습니다. 이는 동일한 검증셋으로
모델을 선택한 결과이며 독립 테스트셋 점수가 아닙니다.
데이터가 극히 적고 원본은 OpenPose 데이터이므로 MediaPipe 웹캠 정확도를 의미하지 않습니다.
추가 촬영자·미지원 단어·무동작·손 미검출을 포함한 웹캠 검증이 필요합니다.
모델 점수는 동작 정확도 퍼센트가 아닙니다.

몸 기준 위치를 버려 서로 다른 수어가 비슷해질 수 있습니다.
이 전처리는 손 동작을 보존하지만 프레임 밖 신체 부위를 기준으로 한 위치를 복원하지 못합니다.
손 크기가 작거나 두 손이 교차할 때의 좌우 라벨 흔들림도 별도 확인해야 합니다.

## 독립 데모 서버

```sh
KSL_MODELS=word="$PWD/runs/hands_intermediate_experiment/best.pt" .venv/bin/python -m uvicorn app.main:app --app-dir ai_server --host 127.0.0.1 --port 8001
```

http://localhost:8001/demo/ 에서 손 모델만 실행합니다.
/health와 /lessons로 모델/지원 단어를 확인합니다. 기존 기본 경로의 best.pt는
상체 모델이므로 새 체크포인트를 명시해야 합니다.
웹캠 없이 서버 변환·추론을 확인하려면:

```sh
.venv/bin/python ai_server/tools/ws_smoke_test.py --npy data/word/clips/NIA_SL_WORD1290_REAL02_F.npy --target WORD1290 --url ws://localhost:8001/ws/practice
```

## 웹사이트와의 연결

이 서버의 `/ws/practice`는 녹화 전체를 받아 end에서 결과를 반환합니다.
웹사이트의 `/ws/v1/word/evaluate`는 최근 16프레임 슬라이딩 평가이므로
프로토콜이 다르며 URL만 바꿔서는 연결되지 않습니다.
`web_adapter.py`는 앞서 준비한 WORD_MODEL_ADAPTER 계약을 구현합니다.
전체 동작으로 학습한 모델을 짧은 윈도우로 사용할 때의 성능은 별도 검증해야 합니다.
어댑터는 예측 ID·점수를 반환하되 calibrated=False로 두어 시험 모델로 진도 통과를 만들지 않습니다.

웹사이트 서버의 가상환경에 이 저장소의 추론 의존성을 설치하고,
이 저장소 루트를 PYTHONPATH에 추가합니다. 웹사이트 저장소 루트에서:

```sh
PYTHONPATH=/절대경로/word-model WORD_MODEL_ADAPTER=web_adapter:create_model HANDS_WORD_CHECKPOINT=/절대경로/word-model/runs/hands_intermediate_experiment/best.pt uvicorn ai_server.main:app --port 8000
```

웹사이트 현재 프레임에 영상 해상도가 없으므로 어댑터는 안전하게 촬영 규격 확인을 요구합니다.
정확한 실제 영상 종횡비를 `HANDS_VIDEO_ASPECT`로 설정해야 합니다(예: 640×480이면 1.3333333333).
가로·세로 비율이 달라지면 이 값도 바꿔야 합니다. 장기적으로 프레임 메시지에 실제 해상도를 전달하는 것이 적절합니다.

수정 코드는 로컬 word-model 폴더의 hands-only 브랜치에 커밋했습니다.
개인 포크는 생성했으며, 수정 코드의 공개 업로드는 사용자 승인 대기 상태입니다.
학습 데이터와 새 시험 가중치는 업로드 대상에서 제외합니다. 운영 웹사이트에는 배포하지 않았습니다.
