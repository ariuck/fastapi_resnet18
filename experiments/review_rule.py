"""[추가 실험 코드 · 제공 코드 아님] CNN + 규칙 기반 '확인 필요(보류)' 분석. (교과 9 · 복합 Rule·실패 분석)

예: python -m experiments.review_rule --preds reports/data/preds/E0_val.csv

규칙: 1위와 2위 확률 차(margin)가 임계값 미만이면 자동 판정하지 않고 '확인 필요'로 보류한다.
- 3클래스 예측 자체는 바꾸지 않는다(라벨 강제 변경 없음).
- 보고: 임계값별 보류율, 자동 판정한 이미지의 정확도, 놓친 오답 수.
- 임계값은 미리 정한 후보(0.1/0.2/0.3)만 비교한다. 운영 기준 확정에는 별도 데이터가 필요하다.
"""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, nargs="+")
    ap.add_argument("--thresholds", default="0,0.1,0.2,0.3")
    args = ap.parse_args()
    ths = [float(t) for t in args.thresholds.split(",")]
    for path in args.preds:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        n = len(rows)
        print(f"== {path} (n={n})")
        print(f"  {'margin<':>8} {'보류':>9} {'자동판정':>8} {'자동 정확도':>10} {'자동 오답':>8} {'보류 중 오답':>11}")
        for t in ths:
            hold = [r for r in rows if float(r["margin"]) < t]
            auto = [r for r in rows if float(r["margin"]) >= t]
            ok = sum(r["label"] == r["pred"] for r in auto)
            wrong_hold = sum(r["label"] != r["pred"] for r in hold)
            acc = ok / len(auto) if auto else float("nan")
            print(f"  {t:>8.1f} {len(hold):>4} ({len(hold) / n:5.1%}) {len(auto):>8} {acc:>10.4f} {len(auto) - ok:>8} {wrong_hold:>11}")


if __name__ == "__main__":
    main()
