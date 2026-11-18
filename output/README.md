# 유닛 5 학습 및 평가 기록

AI Hub의 수어 영상 데이터셋을 활용해 손말잇기의 유닛 5 모델을 학습한 코드와 결과다. 데이터 출처는 [AI Hub 수어 영상](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)이다.

## 현재 운영 모델

| 파일 | 학습 사이너 | 용도 |
|---|---|---|
| 저장소 루트의 unit5.pt | REAL01~REAL12 | 다섯 문장 분류 |
| 저장소 루트의 unit5_words.pt | REAL01~REAL10 | 운전, 천천히, 빨리 단어 분류 |

단어 순서 판정은 deployment/ai_server/sequence.py에서 처리한다. REAL10의 SEN0009 두 번째 단어는 원본 형태소 라벨이 느리다이며, 이 실험에서는 천천히 범주로 통합했다. 해당 선택은 단어 준비 스크립트에 기록되어 있다.

## 파일 구성

- prepare_unit5_keypoints.py는 프레임 JSON ZIP을 기존 LSTM의 입력 배열과 manifest로 변환한다.
- unit5_lstm*/run은 문장 모델 버전별 가중치, 클래스, 학습 로그와 당시 수행한 검증 결과다.
- unit5_word_sequence*/prepare.py와 run은 단어별 시간 라벨을 이용한 구간 추출 코드와 단어 모델 학습 결과다. real01_to_real11 폴더의 준비 코드는 형태소 파일 부재로 학습 완료까지 이어지지 않은 시도다.
- unit5_eval에는 검증·테스트 클립의 식별자 및 배정 manifest와 문장 전체 검증 결과가 있다.
- unit5_eval_split.json에는 확보한 검증 및 테스트 사이너 배정이 있다.
- unit5_stream_validation/REPORT.md에는 운영 방식의 모의 평가와 설정 비교 결과가 있다.
- unit5_stream_validation/replay.py와 results.json에는 재현 코드 및 개별 판정 이력이 있다.

초기 실험은 카메라 각도로 학습·검증을 나눈 기록도 포함한다. 현재 배정은 사이너 기준으로 학습 REAL01~REAL12, 검증 REAL13·REAL17, 테스트 REAL14·REAL15·REAL16·REAL18이다. 초기 실험 점수와 최종 배정의 점수는 구분해서 해석해야 한다.

## 로컬 데이터 준비

키포인트 ZIP과 NPY 클립은 AI Hub 이용 조건에 따라 각 사용자가 직접 확보한다. [AI Hub FAQ](https://aihub.or.kr/aihubnews/faq/list.do)에서 원본 및 단순 가공 데이터의 재배포 조건을 확인할 수 있다. 가중치와 실험 결과를 공개하며 원본 키포인트를 새로 업로드하지 않는다.

이 폴더의 스크립트는 저장소 루트에서 실행한다. Python 환경에는 기존 프로젝트 의존성 외에 NumPy와 PyTorch가 필요하다.

문장 학습용 ZIP을 확보했다면 다음처럼 manifest를 만든다. ARCHIVE 인수에는 직접 확보한 각 사이너의 유닛 5 JSON ZIP 경로를 순서대로 넣는다.

```sh
python output/prepare_unit5_keypoints.py ARCHIVE01.zip ARCHIVE02.zip output/unit5_training/data --all-train
python train_word_lstm.py --manifest output/unit5_training/data/manifest.csv --out runs/unit5 --no-augment --epochs 80 --batch 25 --steps 12 --hidden 64 --layers 1
```

기존 문장 최종 학습 결과는 unit5_lstm_real01_to_real12/run에 보관했다. 재현 시 이전 출력 폴더와 다른 경로를 사용하면 기존 결과가 보존된다.

검증용 REAL13·REAL17과 테스트용 REAL14·REAL15·REAL16·REAL18의 유닛 5 JSON ZIP을 확보한 후 다음처럼 manifest와 로컬 배열을 재생성한다.

```sh
python output/prepare_unit5_keypoints.py REAL13.zip REAL17.zip REAL14.zip REAL15.zip REAL16.zip REAL18.zip output/unit5_eval --signer-splits output/unit5_eval_split.json
PYTHONPATH=deployment:. python output/unit5_stream_validation/replay.py
```

모의 평가 스크립트는 manifest의 val 행만 읽는다. 최종 테스트는 아직 수행 전이다. 모의 재생 시간은 원본 30fps 가정에 근거한다.

단어 구간 준비 스크립트는 Downloads의 01_real_sen_morpheme.zip.part0과 해당 버전의 로컬 문장 배열을 사용한다. 단일 ZIP으로 읽을 수 있는 형태소 파일이 필요하다. 서버가 사용하는 모델과 데이터 전처리 방식은 루트의 hands_dataset.py, word_dataset.py, web_adapter.py, ai_server/ml/word_model.py에서 확인한다.
