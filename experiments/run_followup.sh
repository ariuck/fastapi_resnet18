#!/usr/bin/env bash
# [추가 실험 코드] 후속 분석: 최고 설정 재현성 확인(E7 반복), OpenCV 특징 분류(R1·R2), CNN+규칙 보류 분석
set -e
cd ~/fastapi_resnet18
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_VERBOSITY=error
LOGDIR=reports/evidence/experiments
F="grep -v -E ^\s*$|slow image processor|Some weights|You should probably|- classifier|found shape"

# 1) 재현성: E7 설정 그대로 2회차 (seed 42 고정이지만 GPU 연산 비결정성으로 결과가 다를 수 있음)
if [ ! -f artifacts/exp/E7_full3_aug_r2/training_report.json ]; then
  { echo "\$ python -m training.finetune --data dataset_aug --output artifacts/exp/E7_full3_aug_r2 --epochs 3 --batch-size 8 --lr 0.00005"
    /usr/bin/time -f "# elapsed: %e s" python -m training.finetune --data dataset_aug --output artifacts/exp/E7_full3_aug_r2 --epochs 3 --batch-size 8 --lr 0.00005
  } 2>&1 | grep -v -E "^\s*$|slow image processor|Some weights|You should probably|- classifier|found shape" | tee "$LOGDIR/E7_full3_aug_r2_train.txt"
fi
python -m experiments.evaluate --exp E7_full3_aug_r2 --model artifacts/exp/E7_full3_aug_r2 --data dataset_aug --split val \
  --note "E7 재현성 확인 2회차" 2>&1 | grep -v "slow image processor" | tee "$LOGDIR/E7_full3_aug_r2_val_eval.txt"

# 2) 교과 9 OpenCV 특징만으로 분류 (CNN 미사용)
python -m experiments.opencv_features 2>&1 | tee "$LOGDIR/R_opencv_features.txt"

# 3) CNN + 규칙 보류 (val)
python -m experiments.review_rule --preds reports/data/preds/E0_val.csv reports/data/preds/E4_head10_leafmask_val.csv \
  reports/data/preds/E7_full3_aug_val.csv 2>&1 | tee "$LOGDIR/review_rule_val.txt"
