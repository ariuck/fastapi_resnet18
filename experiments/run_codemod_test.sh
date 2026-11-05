#!/usr/bin/env bash
# [추가 실험 코드] 학습 코드 수정 후 최종 모델(C2) Test 평가 — Test 재사용(2회차)
# 선택 근거(Test 보기 전, val로만): C1·C2 모두 val 1.0000 → C2는 1.0을 3 에포크 연속 유지(C1은 1회)해 더 안정적이고,
# 코드 수정 3가지(랜덤 증강·코사인 스케줄러·조기 종료)를 모두 사용한 모델이므로 C2만 Test에 넣는다. C1은 Test에 쓰지 않는다.
set -e
cd ~/fastapi_resnet18
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_VERBOSITY=error
python -m experiments.evaluate --exp C2_full_rtaug_cos --model artifacts/exp/C2_full_rtaug_cos --data dataset --split test \
  --note "[코드수정] 최종 선택 Full+학습중랜덤증강+코사인 (Test 재사용 2회차)" \
  2>&1 | grep -v "slow image processor" | tee reports/evidence/experiments/codemod_test_eval.txt
