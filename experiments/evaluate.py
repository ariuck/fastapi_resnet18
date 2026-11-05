"""[추가 실험 코드 · 제공 코드 아님] 학습된 모델을 지정 split으로 평가하고 실험 기록에 한 줄 추가한다. (교과 9 · 실험관리)

예: python -m experiments.evaluate --exp E0 --model artifacts/beans_resnet18 --data dataset --split val --note "Head-only 3ep"

- 정확도, 클래스별 정확도, 혼동행렬, 1·2위 점수 차(margin)를 계산한다.
- reports/data/experiments.csv 에 실험 1줄, reports/data/preds/<exp>_<split>.csv 에 이미지별 예측을 저장한다.
- 모델 선택은 val로만 한다. test는 최종 모델 확정 후 1회만 사용한다.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LOG = Path("reports/data/experiments.csv")
PRED_DIR = Path("reports/data/preds")


@torch.no_grad()
def predict(model_dir: str, split_root: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device).eval()
    classes = [model.config.id2label[i] for i in range(model.config.num_labels)]
    records = [(p, i) for i, c in enumerate(classes)
               for p in sorted((split_root / c).iterdir()) if p.suffix.lower() in EXTS]
    rows = []
    for start in range(0, len(records), 32):
        chunk = records[start:start + 32]
        images = []
        for path, _ in chunk:
            with Image.open(path) as raw:
                images.append(raw.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        probs = model(pixel_values=pixels).logits.softmax(-1).cpu()
        for (path, label), p in zip(chunk, probs):
            top2 = p.topk(2).values.tolist()
            rows.append({"image": f"{classes[label]}/{path.name}", "label": classes[label],
                         "pred": classes[int(p.argmax())], "margin": round(top2[0] - top2[1], 4),
                         **{f"p_{c}": round(float(v), 4) for c, v in zip(classes, p)}})
    return classes, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    classes, rows = predict(args.model, Path(args.data) / args.split)
    n = len(classes)
    cm = [[sum(1 for r in rows if r["label"] == a and r["pred"] == b) for b in classes] for a in classes]
    correct = sum(cm[i][i] for i in range(n))
    total = len(rows)
    class_acc = {c: round(cm[i][i] / sum(cm[i]), 4) for i, c in enumerate(classes)}
    wrong = [r for r in rows if r["label"] != r["pred"]]

    print(f"[{args.exp}] {args.model} on {args.data}/{args.split}: {correct}/{total} = {correct / total:.4f}")
    print("  정답/예측".ljust(22) + "".join(f"{c[:10]:>12}" for c in classes) + "   class_acc")
    for i, c in enumerate(classes):
        print(f"  {c:<20}" + "".join(f"{v:>12}" for v in cm[i]) + f"   {class_acc[c]:.4f}")
    print(f"  오답 {len(wrong)}건 margin 중앙값: "
          f"{sorted(r['margin'] for r in wrong)[len(wrong) // 2] if wrong else '-'} / "
          f"정답 margin 중앙값: {sorted(r['margin'] for r in rows if r['label'] == r['pred'])[(total - len(wrong)) // 2]}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with (PRED_DIR / f"{args.exp}_{args.split}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    report = Path(args.model) / "training_report.json"
    best = json.loads(report.read_text())["best_val_accuracy"] if report.exists() else None
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "exp", "model", "data", "split", "accuracy", "correct", "total",
                        *[f"acc_{c}" for c in classes], "confusion(row=정답)", "training_report_best_val", "note"])
        w.writerow([datetime.now().strftime("%F %T"), args.exp, args.model, args.data, args.split,
                    round(correct / total, 4), correct, total, *class_acc.values(),
                    json.dumps(cm), best, args.note])


if __name__ == "__main__":
    main()
