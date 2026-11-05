#!/usr/bin/env bash
# [추가 실험 코드] 학습 코드 수정 버전(training/finetune.py --augment/--scheduler/--patience)으로 풀 파인튜닝 재실험
# 비교 기준: E7_full3_aug (코드 수정 없이 옵션만, 미리 만든 증강 데이터 ×3, val 0.985)
# 평가는 val 133장으로만 한다.
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

# C1: 원본 데이터 + 학습 중 랜덤 증강 + 조기 종료 (최대 10 에포크, 3 에포크 연속 개선 없으면 멈춤)
train C1_full_rtaug     dataset "[코드수정] Full lr5e-5 학습중 랜덤증강 최대10ep 조기종료3" \
  --epochs 10 --batch-size 8 --lr 0.00005 --augment --patience 3
# C2: C1 + 코사인 학습률 스케줄러
train C2_full_rtaug_cos dataset "[코드수정] Full lr5e-5 학습중 랜덤증강 코사인 최대10ep 조기종료3" \
  --epochs 10 --batch-size 8 --lr 0.00005 --augment --scheduler cosine --patience 3
