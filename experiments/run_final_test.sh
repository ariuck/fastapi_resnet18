#!/usr/bin/env bash
# [추가 실험 코드] 최종 Test 1회 평가 — val로 확정한 최종 모델(E7)과 개선 전 기준선(E0, 참고용)만 평가한다.
set -e
cd ~/fastapi_resnet18
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_VERBOSITY=error
{
  python -m experiments.evaluate --exp E7_full3_aug --model artifacts/exp/E7_full3_aug --data dataset --split test --note "최종 선택 Full+증강 (Test 1회)"
  python -m experiments.evaluate --exp E0 --model artifacts/beans_resnet18 --data dataset --split test --note "기준선 참고 (Test 1회)"
} 2>&1 | grep -v "slow image processor" | tee reports/evidence/experiments/final_test_eval.txt
