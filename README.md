# FastAPI Beans ResNet18 동기·비동기 평가 프로젝트

- GitHub: https://github.com/ariuck/fastapi_resnet18
- 모델 가중치는 저장소에 포함하지 않는다. 제출물 파일 `교과9_제출물.zip`의 `artifacts/` 폴더를 이 프로젝트 루트에 그대로 복사하면 4장 학습 없이 바로 실행할 수 있다.

콩잎 사진 1장을 `angular_leaf_spot` / `bean_rust` / `healthy`로 분류하는 ResNet18 모델을 Fine-tuning하고, FastAPI 동기 API와 Celery 비동기 작업 큐(Redis·PostgreSQL)로 서빙한다.

## 1. 환경

| 항목 | 사용 버전 |
|---|---|
| OS | Windows 11 + WSL2 Ubuntu 26.04 |
| Python | 3.11 (`.venv`) — WSL 기본 Python이 3.11이 아니면 [uv](https://docs.astral.sh/uv/)로 설치 |
| Docker | Docker Desktop (Compose v2) — Redis 6379, PostgreSQL 5432 |
| Python 프로세스 | FastAPI(uvicorn), Celery Worker — WSL에서 실행 |
| GPU | 선택 (학습만 사용, 서비스 추론은 CPU) |

## 2. 설치

```bash
cd ~/fastapi_resnet18
# Python 3.11이 없을 때
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt      # 또는 python -m pip install -r requirements.txt
cp .env.example .env                    # MODEL_PATH=artifacts/beans_resnet18
```

## 3. 데이터 준비

1. [Hugging Face AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans/tree/main/data)에서 `train.zip`, `validation.zip`, `test.zip`을 받는다 (라이선스 MIT, 공유 전 이용 조건 확인).
2. 원본은 별도 폴더(예: `~/beans_source/`)에 보존하고, 각 클래스 이미지를 아래처럼 **복사**한다.

| 원본 | 배치 위치 |
|---|---|
| `train/<class>/` | `dataset/train/<class>/` |
| `validation/<class>/` | **`dataset/val/<class>/`** (제공 학습 코드가 `val`을 읽음) |
| `test/<class>/` | `dataset/test/<class>/` |

3. `train/train/...`처럼 폴더가 중첩되지 않게 하고, 이미지가 아닌 파일(macOS 메타데이터 등)은 제외한다.
4. 기대 수량: train 1,034 / val 133 / test 128 (3클래스).

## 4. 모델 학습 (Head-only)

```bash
python -m training.finetune --data dataset --output artifacts/beans_resnet18 --epochs 3 --batch-size 8 --freeze-backbone
cat artifacts/beans_resnet18/training_report.json
```

생성 파일: `config.json`, `model.safetensors`, `preprocessor_config.json`, `training_report.json`

## 5. 구동 순서

```bash
source .venv/bin/activate
docker compose up -d redis postgres
docker compose exec redis redis-cli ping            # PONG
docker compose exec postgres pg_isready -U vision_user -d vision_db
```

별도 터미널 1 — Celery Worker:

```bash
cd ~/fastapi_resnet18 && source .venv/bin/activate
celery -A ai.celery_app:celery_app worker --loglevel=info --pool=solo --concurrency=1
```

별도 터미널 2 — FastAPI:

```bash
cd ~/fastapi_resnet18 && source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

| URL | 용도 |
|---|---|
| http://127.0.0.1:8000/vision/sync/ | 동기 분류 화면 |
| http://127.0.0.1:8000/vision/async/ | 비동기 분류 화면 |
| http://127.0.0.1:8000/docs | Swagger API 문서 |
| http://127.0.0.1:8000/health | 헬스 체크 |

> `MODEL_PATH`나 모델 파일을 바꾸면 FastAPI와 Worker를 **모두 재시작**한다.
> 02 문서의 `docker compose up -d --build`는 web·worker까지 컨테이너로 띄우므로 이 구조에서는 사용하지 않는다.

## 6. 확인할 응답

- `/health`: `{"status":"ok"}`
- 동기 POST: HTTP 200, `top1`, `all`, `inference_ms`, `cache_hit`
- 비동기 POST: HTTP 202, `job_id`, `check_url` → GET `check_url`: `status=done`
- PostgreSQL: `image_jobs`에 비동기 작업 기록

```bash
docker compose exec postgres psql -U vision_user -d vision_db -c \
  'SELECT id, status, model_id, top1_label, cache_hit, inference_ms FROM image_jobs ORDER BY id DESC LIMIT 10;'
```

## 7. 테스트·측정

```bash
pytest -q tests/test_api.py               # 모의 테스트 (실제 모델·Redis·DB 미사용)
python -m tools.measure_api --repeat 5    # 실서버 성능 측정 → reports/data/timings.csv
```

`tools/measure_api.py`는 평가용으로 추가한 측정 도구이며, 실행 시 Redis 캐시 DB(/1)를 비운다.

## 7-1. 정확도 개선 실험 (선택)

```bash
uv pip install -r experiments/requirements-exp.txt   # OpenCV, scikit-learn (실험 전용)
bash experiments/run_all.sh        # 전처리(CLAHE·잎 마스크)·증강 데이터 생성 + 실험 E1~E8 학습·val 평가
bash experiments/run_followup.sh   # 재현성(E7 2회차), OpenCV 특징 분류(R1·R2), 규칙 보류 분석
bash experiments/run_final_test.sh # 최종 모델(E7)·기준선(E0) Test 1회 평가
```

- 제공 `training/finetune.py`는 수정하지 않고 `--data`와 옵션만 바꾼다. 결과 모델은 `artifacts/exp/`, 기록은 `reports/data/experiments.csv`.
- 최종 선택 E7(Full Fine-tuning + 증강): val 0.9850 / Test 0.9766. 서비스에 적용하려면 `.env`의 `MODEL_PATH=artifacts/exp/E7_full3_aug`로 바꾸고 FastAPI·Worker를 재시작한다 (평가 제출본은 E0 유지).
- GPU 연산 비결정성 때문에 재실행 시 1~2장 차이가 날 수 있다.

## 8. 결과물 위치

| 내용 | 위치 |
|---|---|
| 산출물 1~8 | [`reports/01_FineTuning_결과보고서.md`](reports/01_FineTuning_결과보고서.md) ~ [`reports/08_최종기술보고서.md`](reports/08_최종기술보고서.md) |
| 측정 원본 | `reports/data/timings.csv` |
| 실행 증빙 (캡처·로그·응답 JSON) | `reports/evidence/` |
| 학습 결과 | `artifacts/beans_resnet18/training_report.json` |

## 9. 종료

```bash
# uvicorn·celery 터미널에서 Ctrl+C
docker compose stop        # 볼륨(데이터) 보존
```

`docker compose down -v`는 PostgreSQL·Redis 데이터를 삭제하므로 평가 중 사용하지 않는다.

## 10. 제출 및 보안

- `.env`, 비밀번호, API 키는 GitHub에 올리지 않는다 (`.gitignore`에 포함).
- `compose.yaml`의 DB 계정(`vision_user`/`vision_pass`)은 로컬 교육용 값이며 외부 공개 서버에 사용하지 않는다.
- `dataset/`, `artifacts/`, `media/jobs/`는 `.gitignore` 대상이다. 데이터셋은 위 3장 절차로 다시 받는다.
- 학습 가중치는 제출물 파일로 별도 전달한다: `artifacts/beans_resnet18/`(E0, 서비스 모델), `artifacts/exp/E7_full3_aug/`(E7, 정확도 개선 최종 모델). 4장 명령으로 다시 만들 수도 있다(GPU 비결정성으로 1~2장 차이 가능).
