"""[추가 실험 코드 · 제공 코드 아님] 교과 9 OpenCV 전처리를 데이터셋 전체에 적용해 새 데이터 폴더를 만든다.

예: python -m experiments.preprocess --method clahe    --src dataset --dst dataset_clahe
    python -m experiments.preprocess --method leafmask --src dataset --dst dataset_leafmask

method
- clahe    (Image Enhancement): BGR → LAB, L 채널에만 CLAHE(clipLimit 2.0, 8×8) → BGR
- leafmask (HSV + Morphology·Contour): HSV 잎 색 범위 inRange → Opening/Closing
           → 면적 2% 이상 Contour만 채워서 마스크 → 마스크 밖(흙·배경)을 검정으로
           마스크가 이미지의 15% 미만이면 '마스크 실패'로 보고 원본 유지(기록)

- 같은 처리를 train/val/test 모두에 적용한다(학습·평가 입력 조건 통일). 기준값은 고정 상수이며 test를 보고 조정하지 않는다.
- 제공 학습 코드(training/finetune.py)는 그대로 --data 만 바꿔 사용한다.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ["train", "val", "test"]

# HSV(OpenCV: H 0~179) 잎 색 범위 — 녹색~황록색(건강 잎·변색 잎 포함)
LEAF_LOW = np.array([22, 40, 30])
LEAF_HIGH = np.array([100, 255, 255])  # v1은 95 → 그늘진 청록색 잎이 빠져 100으로 확장
MIN_CONTOUR_RATIO = 0.015
MIN_MASK_RATIO = 0.15
# v1: 사각 25×25 Closing 2회 → 흙까지 메워지고 경계가 각짐. v2: 작은 타원 커널 + 외곽 Contour 채우기로 병반 구멍 처리
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def leaf_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LEAF_LOW, LEAF_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)
    # 병반(갈색·주황)은 녹색 범위 밖이라 잎 안의 구멍이 된다 → 외곽 Contour를 채우면 구멍이 메워진다
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = mask.shape[0] * mask.shape[1]
    keep = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_RATIO * area]
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, keep, -1, 255, thickness=cv2.FILLED)
    return filled


def apply_leafmask(bgr):
    mask = leaf_mask(bgr)
    ratio = float(mask.mean() / 255)
    if ratio < MIN_MASK_RATIO:
        return bgr, mask, ratio, False
    return cv2.bitwise_and(bgr, bgr, mask=mask), mask, ratio, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["clahe", "leafmask"])
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    stats = {}
    for split in SPLITS:
        ratios, fallback = [], 0
        for cls_dir in sorted(Path(args.src, split).iterdir()):
            out_dir = Path(args.dst, split, cls_dir.name)
            out_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(cls_dir.iterdir()):
                if path.suffix.lower() not in EXTS:
                    continue
                bgr = cv2.imread(str(path))
                if args.method == "clahe":
                    out = clahe(bgr)
                else:
                    out, _, ratio, ok = apply_leafmask(bgr)
                    ratios.append(ratio)
                    fallback += not ok
                cv2.imwrite(str(out_dir / (path.stem + ".png")), out)  # 무손실 저장
        if ratios:
            r = sorted(ratios)
            stats[split] = {"images": len(r), "mask_ratio_median": round(r[len(r) // 2], 3),
                            "mask_ratio_p10": round(r[len(r) // 10], 3), "fallback_original": fallback}
        print(f"{split}: done", stats.get(split, ""))
    Path(args.dst, "preprocess_info.json").write_text(json.dumps(
        {"method": args.method, "src": args.src, "leaf_hsv_low": LEAF_LOW.tolist(),
         "leaf_hsv_high": LEAF_HIGH.tolist(), "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
