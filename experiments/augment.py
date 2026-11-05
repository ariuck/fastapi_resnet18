"""[추가 실험 코드 · 제공 코드 아님] 학습 데이터에만 오프라인 증강 이미지를 추가한 데이터 폴더를 만든다. (교과 9 · 8일차 데이터 생성)

예: python -m experiments.augment --src dataset --dst dataset_aug --copies 2

- train: 원본 + 원본 1장당 증강 N장 (좌우·상하 반전, 90° 단위 회전, 밝기·대비 ±20%) — 병명(정답)이 바뀌지 않는 변형만 사용
- val, test: 원본 폴더를 심볼릭 링크로 그대로 사용 (증강 금지, 평가 조건 동일)
- 제공 학습 코드는 그대로 --data 만 바꿔 사용한다.
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def augment(bgr, rng):
    out = bgr
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.5:
        out = cv2.flip(out, 0)
    out = np.ascontiguousarray(np.rot90(out, rng.randint(0, 3)))
    alpha = rng.uniform(0.8, 1.2)   # 대비
    beta = rng.uniform(-20, 20)     # 밝기
    return cv2.convertScaleAbs(out, alpha=alpha, beta=beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--dst", default="dataset_aug")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    src, dst = Path(args.src).resolve(), Path(args.dst)
    counts = {}
    for cls_dir in sorted((src / "train").iterdir()):
        out_dir = dst / "train" / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for path in sorted(cls_dir.iterdir()):
            if path.suffix.lower() not in EXTS:
                continue
            shutil.copy2(path, out_dir / path.name)
            bgr = cv2.imread(str(path))
            for k in range(args.copies):
                cv2.imwrite(str(out_dir / f"{path.stem}_aug{k}.jpg"), augment(bgr, rng),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1 + args.copies
        counts[cls_dir.name] = n
    for split in ("val", "test"):
        link = dst / split
        if not link.exists():
            link.symlink_to(src / split, target_is_directory=True)
    print("train counts:", counts, "total", sum(counts.values()))


if __name__ == "__main__":
    main()
