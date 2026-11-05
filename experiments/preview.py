"""[추가 실험 코드] 전처리 결과를 눈으로 확인하는 예시 그림 생성 (val 이미지만 사용).

예: python -m experiments.preview --per-class 3
출력: reports/figures/preprocess_examples.png  (행: 이미지 / 열: 원본 · CLAHE · 잎 마스크 · 마스크 적용)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from experiments.preprocess import apply_leafmask, clahe

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="reports/figures/preprocess_examples.png")
    args = ap.parse_args()
    rows = []
    for cls in CLASSES:
        files = sorted(Path("dataset/val", cls).glob("*.jpg"))[args.start:args.start + args.per_class]
        for f in files:
            bgr = cv2.resize(cv2.imread(str(f)), (250, 250))
            masked, mask, ratio, ok = apply_leafmask(bgr)
            rows.append(np.hstack([
                label(bgr, f"{cls} / {f.name}"),
                label(clahe(bgr), "CLAHE"),
                label(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), f"mask {ratio:.2f}"),
                label(masked, "masked" if ok else "fallback(original)"),
            ]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.vstack(rows))
    print("saved", args.out)


if __name__ == "__main__":
    main()
