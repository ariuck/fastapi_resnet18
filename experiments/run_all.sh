#!/usr/bin/env bash
# [추가 실험 코드] 정확도 개선 실험 일괄 실행. 제공 training/finetune.py 는 수정 없이 --data/--output/옵션만 바꾼다.
# 모든 모델은 artifacts/exp/ 에 따로 저장 (서비스 모델 artifacts/beans_resnet18 은 건드리지 않음)
# 평가는 val 133장으로만 한다. test 는 최종 모델 확정 후 별도로 1회.
set -e
cd ~/fastapi_resnet18
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_VERBOSITY=error
LOGDIR=reports/evidence/experiments
mkdir -p "$LOGDIR" artifacts/exp

train() {  # $1=exp  $2=data  $3=note  rest=finetune 옵션
  local exp=$1 data=$2 note=$3; shift 3
  local out=artifacts/exp/$exp
  if [ ! -f "$out/training_report.json" ]; then
    { echo "\$ python -m training.finetune --data $data --output $out $*"
      echo "# started: $(date '+%F %T %Z')"
      /usr/bin/time -f "# elapsed: %e s" python -m training.finetune --data "$data" --output "$out" "$@"
    } 2>&1 | grep -v -E "^\s*$|slow image processor|Some weights|You should probably|- classifier|found shape" | tee "$LOGDIR/${exp}_train.txt"
  fi
  python -m experiments.evaluate --exp "$exp" --model "$out" --data "$data" --split val --note "$note" \
    2>&1 | grep -v "slow image processor" | tee "$LOGDIR/${exp}_val_eval.txt"
}

# 기준선 (이미 학습된 서비스 모델, 평가만)
python -m experiments.evaluate --exp E0 --model artifacts/beans_resnet18 --data dataset --split val \
  --note "기준선 Head-only 3ep lr2e-4 원본" 2>&1 | grep -v "slow image processor" | tee "$LOGDIR/E0_val_eval.txt"

# 1) 학습 전략 (옵션만 변경)
train E1_head10      dataset          "Head-only 10ep lr2e-4 원본"          --epochs 10 --batch-size 8 --freeze-backbone
train E2_full3       dataset          "Full 3ep lr5e-5 원본"                 --epochs 3  --batch-size 8 --lr 0.00005

# 2) 교과 9 OpenCV 전처리
[ -d dataset_clahe ]    || python -m experiments.preprocess --method clahe    --src dataset --dst dataset_clahe
[ -d dataset_leafmask ] || python -m experiments.preprocess --method leafmask --src dataset --dst dataset_leafmask
train E3_head10_clahe    dataset_clahe    "Head-only 10ep CLAHE"        --epochs 10 --batch-size 8 --freeze-backbone
train E4_head10_leafmask dataset_leafmask "Head-only 10ep 잎마스크"   --epochs 10 --batch-size 8 --freeze-backbone
train E5_full3_clahe     dataset_clahe    "Full 3ep lr5e-5 CLAHE"              --epochs 3  --batch-size 8 --lr 0.00005
train E6_full3_leafmask  dataset_leafmask "Full 3ep lr5e-5 잎마스크"           --epochs 3  --batch-size 8 --lr 0.00005

# 3) 데이터 증강
[ -d dataset_aug ] || python -m experiments.augment --src dataset --dst dataset_aug --copies 2
train E7_full3_aug       dataset_aug      "Full 3ep lr5e-5 증강×3"      --epochs 3  --batch-size 8 --lr 0.00005
train E8_head10_aug      dataset_aug      "Head-only 10ep 증강×3"       --epochs 10 --batch-size 8 --freeze-backbone

echo; echo "===== 실험 기록 (val) ====="
column -s, -t < reports/data/experiments.csv | cut -c1-200
