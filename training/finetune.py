"""예: python -m training.finetune --data dataset --epochs 3 --output artifacts/beans_resnet18

[수정] 풀 파인튜닝 개선용 옵션 3개 추가 (옵션을 주지 않으면 기존과 똑같이 동작)
  --augment          학습 중 랜덤 증강 (매 에포크마다 다른 변형: 좌우·상하 반전, 90° 회전, 밝기·대비 ±20%)
  --scheduler cosine 학습률을 배치마다 코사인 곡선으로 점점 줄임
  --patience N       검증 정확도가 N 에포크 연속 오르지 않으면 조기 종료
예: python -m training.finetune --data dataset --output artifacts/exp/C2 --epochs 10 --lr 0.00005
                                       --augment --scheduler cosine --patience 3
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_MODEL = "microsoft/resnet-18"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def random_augment(image: Image.Image) -> Image.Image:
    """[수정] 학습 이미지에만 쓰는 랜덤 증강. 병명(정답)이 바뀌지 않도록 색상(Hue)은 바꾸지 않는다."""
    if random.random() < 0.5:
        image = ImageOps.mirror(image)          # 좌우 반전
    if random.random() < 0.5:
        image = ImageOps.flip(image)            # 상하 반전
    k = random.randint(0, 3)
    if k:
        image = image.rotate(90 * k, expand=True)  # 90° 단위 회전
    alpha = random.uniform(0.8, 1.2)            # 대비
    beta = random.uniform(-20, 20)              # 밝기
    pixels = np.asarray(image, dtype=np.float32) * alpha + beta
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


class FolderImages(Dataset):
    def __init__(self, root: Path, classes: list[str], processor, augment: bool = False):
        self.records = []
        self.processor = processor
        self.augment = augment  # [수정] True면 불러올 때마다 랜덤 증강 (train 전용)
        for idx, name in enumerate(classes):
            folder = root / name
            if not folder.is_dir():
                raise ValueError(f"필요한 클래스 폴더가 없습니다: {folder}")
            self.records.extend((path, idx) for path in sorted(folder.iterdir())
                                if path.suffix.lower() in EXTS and path.is_file())
        if not self.records:
            raise ValueError(f"이미지가 없습니다: {root}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            if self.augment:
                image = random_augment(image)
            tensor = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": tensor, "labels": torch.tensor(label, dtype=torch.long)}

def run(args):
    random.seed(42)
    torch.manual_seed(42)

    data = Path(args.data)
    train_root = data / "train"
    if not train_root.is_dir():
        raise ValueError("dataset/train/ 폴더를 먼저 준비하세요.")

    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("클래스 폴더가 2개 이상 필요합니다.")

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    train_set = FolderImages(train_root, classes, processor, augment=args.augment)
    val_set = FolderImages(data / "val", classes, processor)  # 검증은 증강하지 않음

    train_counts = {n: sum(label == i for _, label in train_set.records) for i, n in enumerate(classes)}
    val_counts = {n: sum(label == i for _, label in val_set.records) for i, n in enumerate(classes)}

    if any(n < 2 for n in train_counts.values()) or any(n < 1 for n in val_counts.values()):
        raise ValueError("각 클래스에 학습 최소 2장, 검증 최소 1장이 필요합니다.")

    print(f"클래스: {classes} / Train: {len(train_set)}, Val: {len(val_set)}")

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(classes),
        id2label={i: name for i, name in enumerate(classes)},
        label2id={name: i for i, name in enumerate(classes)},
        ignore_mismatched_sizes=True,
    )

    # 기존 1000-class 분류 헤드는 크기가 맞지 않아 새롭게 초기화됩니다.
    # 초보자 첫 학습은 Backbone을 고정하고 마지막 분류 헤드만 학습합니다.
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # [수정] 학습률 스케줄러: 전체 학습 스텝에 걸쳐 lr → 0 으로 코사인 감소 (배치마다 갱신)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader))

    history = []
    best_acc = -1.0
    bad_epochs = 0       # [수정] 최고 기록이 갱신되지 않은 연속 에포크 수
    stopped_epoch = None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            result = model(**batch)
            result.loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(result.loss.item()))

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                pixels = batch["pixel_values"].to(device)
                logits = model(pixel_values=pixels).logits
                correct += (logits.argmax(dim=-1).cpu() == batch["labels"]).sum().item()
                total += len(batch["labels"])

        acc = correct / total
        row = {
            "epoch": epoch + 1,
            "loss": round(sum(losses) / len(losses), 4),
            "val_accuracy": round(acc, 4)
        }
        if scheduler is not None:
            row["lr_end"] = float(f"{optim.param_groups[0]['lr']:.3g}")  # [수정] 에포크 끝 학습률 기록
        history.append(row)
        print(row)

        if acc > best_acc:
            best_acc = acc
            bad_epochs = 0
            model.save_pretrained(str(output), safe_serialization=True)
            processor.save_pretrained(str(output))
        else:
            bad_epochs += 1
            # [수정] 조기 종료: patience 에포크 연속 개선이 없으면 멈춤 (저장된 모델은 최고 시점 그대로)
            if args.patience and bad_epochs >= args.patience:
                stopped_epoch = epoch + 1
                print(f"조기 종료: {args.patience} 에포크 연속 개선 없음 (epoch {stopped_epoch})")
                break

    (output / "training_report.json").write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "classes": classes,
                "train_count": len(train_set),
                "val_count": len(val_set),
                "best_val_accuracy": best_acc,
                "history": history,
                # [수정] 재현용 학습 설정 기록
                "options": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                            "freeze_backbone": args.freeze_backbone, "augment": args.augment,
                            "scheduler": args.scheduler, "patience": args.patience},
                "stopped_epoch": stopped_epoch
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"저장 완료: {output} / best val accuracy={best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--output", default="artifacts/beans_resnet18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    # [수정] 풀 파인튜닝 개선 옵션 (기본값은 모두 '사용 안 함' → 기존 결과 그대로 재현)
    parser.add_argument("--augment", action="store_true", help="학습 중 랜덤 증강")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none", help="학습률 스케줄러")
    parser.add_argument("--patience", type=int, default=0, help="조기 종료 기준 에포크 수 (0이면 사용 안 함)")
    run(parser.parse_args())
