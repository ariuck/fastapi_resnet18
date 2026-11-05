"""[추가 실험 코드 · 제공 코드 아님] 교과 9 규칙 기반 특징만으로 3클래스를 분류해 본다 (CNN 미사용).

예: python -m experiments.opencv_features

교과 9 연결
- HSV / Morphology·Contour : 잎 마스크(experiments.preprocess.leaf_mask) 안에서 병반 후보 영역 추출
- Calibration                      : 특징 정규화·분류 기준을 train 으로만 학습 (val·test 로 조정하지 않음)
- Hybrid 비교                      : 결과를 CNN 실험과 같은 val 133장 기준으로 experiments.csv 에 기록

특징 (이미지 1장당 11개)
- leaf_ratio                 잎 마스크 면적 / 이미지
- brown_ratio, yellow_ratio  잎 안 갈색(H 0~21)·황색(H 22~34, S≥80) 픽셀 / 잎 픽셀
- spot_count, spot_mean_area 잎 안 병반 후보 Contour 개수, 평균 면적(잎 대비)
- h/s/v mean, h/s/v std      잎 픽셀의 HSV 평균·표준편차
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.preprocess import leaf_mask

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
NAMES = ["leaf_ratio", "brown_ratio", "yellow_ratio", "spot_count", "spot_mean_area",
         "h_mean", "s_mean", "v_mean", "h_std", "s_std", "v_std"]


def features(path):
    bgr = cv2.resize(cv2.imread(str(path)), (500, 500))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    leaf = leaf_mask(bgr)
    leaf_px = max(int((leaf > 0).sum()), 1)
    h, s, v = cv2.split(hsv)
    brown = cv2.inRange(hsv, (0, 60, 40), (21, 255, 230))
    yellow = cv2.inRange(hsv, (22, 80, 80), (34, 255, 255))
    lesion = cv2.bitwise_and(cv2.bitwise_or(brown, yellow), leaf)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    spots, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spots = [c for c in spots if cv2.contourArea(c) >= 4]
    sel = leaf > 0
    return [leaf_px / leaf.size,
            float((cv2.bitwise_and(brown, leaf) > 0).sum() / leaf_px),
            float((cv2.bitwise_and(yellow, leaf) > 0).sum() / leaf_px),
            len(spots),
            float(np.mean([cv2.contourArea(c) for c in spots]) / leaf_px) if spots else 0.0,
            *[float(ch[sel].mean()) if sel.any() else 0.0 for ch in (h, s, v)],
            *[float(ch[sel].std()) if sel.any() else 0.0 for ch in (h, s, v)]]


def load(split):
    X, y, names = [], [], []
    for i, c in enumerate(CLASSES):
        for p in sorted(Path("dataset", split, c).glob("*.jpg")):
            X.append(features(p))
            y.append(i)
            names.append(f"{c}/{p.name}")
    return np.array(X), np.array(y), names


def main():
    Xtr, ytr, _ = load("train")
    Xva, yva, _ = load("val")
    print("특징 클래스별 평균 (train)")
    for i, c in enumerate(CLASSES):
        m = Xtr[ytr == i].mean(0)
        print(f"  {c:<18} " + " ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, m)))
    models = {
        "R1_opencv_logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "R2_opencv_rf": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    log = Path("reports/data/experiments.csv")
    for exp, clf in models.items():
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        cm = [[int(((yva == a) & (pred == b)).sum()) for b in range(3)] for a in range(3)]
        acc = float((pred == yva).mean())
        class_acc = [round(cm[i][i] / sum(cm[i]), 4) for i in range(3)]
        print(f"[{exp}] val {int((pred == yva).sum())}/{len(yva)} = {acc:.4f} class_acc={class_acc} cm={cm}")
        if exp == "R2_opencv_rf":
            imp = sorted(zip(NAMES, clf.feature_importances_), key=lambda t: -t[1])
            print("  RF 특징 중요도:", ", ".join(f"{n} {v:.3f}" for n, v in imp))
        with log.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now().strftime("%F %T"), exp, type(clf).__name__, "dataset(OpenCV 특징 11개)",
                                    "val", round(acc, 4), int((pred == yva).sum()), len(yva), *class_acc,
                                    json.dumps(cm), "", "CNN 미사용, 교과9 HSV·Contour 특징 + " +
                                    ("로지스틱회귀" if "logreg" in exp else "랜덤포레스트")])


if __name__ == "__main__":
    main()
