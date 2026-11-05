"""[추가 측정 도구 · 제공 코드 아님] 실행 중인 FastAPI에 실제 요청을 보내 동기/비동기 × 캐시 MISS/HIT 시간을 측정한다.

전제: Redis·PostgreSQL(docker compose), Celery Worker, uvicorn(127.0.0.1:8000)이 실행 중.
예: python -m tools.measure_api --repeat 5

측정 조건
- 이미지: dataset/test 의 서로 다른 파일(SHA-256 기록). Test 정확도 산출 용도가 아님.
- 시작 시 Redis 캐시 DB(/1)만 비워 MISS 조건을 보장한다(브로커 DB/2, PostgreSQL은 건드리지 않음).
- 워밍업: 첫 동기 요청(모델 최초 로드 포함)은 run_id=0 으로 따로 기록하고 비교에서 제외.
- MISS 그룹: 처음 보내는 이미지 / HIT 그룹: 같은 이미지를 즉시 재전송.
- http_ms: 클라이언트가 요청을 보내고 HTTP 응답을 받을 때까지.
- final_ms: 동기=http_ms, 비동기=접수 요청 시작 ~ 상태조회에서 done 확인까지(폴링 간격 0.1초 포함).
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import httpx

from ai.config import REDIS_CACHE_URL

BASE = "http://127.0.0.1:8000"
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
EVIDENCE = Path("reports/evidence/api")
POLL_SEC = 0.1


def test_images():
    """클래스를 번갈아 가며 Test 이미지를 순서대로 돌려준다."""
    per_class = {c: sorted(Path("dataset/test", c).glob("*.jpg")) for c in CLASSES}
    for i in range(max(map(len, per_class.values()))):
        for c in CLASSES:
            if i < len(per_class[c]):
                yield c, per_class[c][i]


def save(name, payload):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def post(client, url, path):
    started = time.perf_counter()
    r = client.post(url, files={"image": (path.name, path.read_bytes(), "image/jpeg")})
    return r, started, (time.perf_counter() - started) * 1000


def sync_call(client, path):
    r, _, http_ms = post(client, f"{BASE}/api/sync/classify/", path)
    r.raise_for_status()
    return r.json(), http_ms


def async_call(client, path):
    r, started, http_ms = post(client, f"{BASE}/api/async/classify/", path)
    assert r.status_code == 202, (r.status_code, r.text)
    accepted = r.json()
    while True:
        job = client.get(f"{BASE}{accepted['check_url']}").json()
        if job["status"] in ("done", "failed"):
            return accepted, job, http_ms, (time.perf_counter() - started) * 1000
        time.sleep(POLL_SEC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    from redis import Redis
    Redis.from_url(REDIS_CACHE_URL).flushdb()

    images = test_images()
    rows = []

    def row(run_id, path_kind, cache_status, http_ms, final_ms, result, job_id, cls, img):
        rows.append({
            "run_id": run_id, "path": path_kind, "cache_status": cache_status,
            "http_ms": round(http_ms, 1), "final_ms": round(final_ms, 1),
            "inference_ms": result.get("inference_ms"), "cache_hit": result.get("cache_hit"),
            "job_id": job_id or "", "image": f"{cls}/{img.name}", "folder_label": cls,
            "pred_label": (result.get("top1") or {}).get("label"),
            "score": round((result.get("top1") or {}).get("score") or 0, 4),
            "sha256_12": hashlib.sha256(img.read_bytes()).hexdigest()[:12],
        })

    with httpx.Client(timeout=120) as client:
        # 워밍업 (모델 최초 로드)
        cls, img = next(images)
        data, ms = sync_call(client, img)
        row(0, "sync", "MISS(warmup)", ms, ms, data, None, cls, img)
        save("sync_warmup.json", data)

        # 동기 MISS → 같은 이미지로 HIT
        sync_imgs = [next(images) for _ in range(args.repeat)]
        for i, (cls, img) in enumerate(sync_imgs, 1):
            data, ms = sync_call(client, img)
            row(i, "sync", "MISS", ms, ms, data, None, cls, img)
            save(f"sync_miss_{i}_{cls}.json", data)
        for i, (cls, img) in enumerate(sync_imgs, 1):
            data, ms = sync_call(client, img)
            row(i, "sync", "HIT", ms, ms, data, None, cls, img)
            save(f"sync_hit_{i}_{cls}.json", data)

        # 비동기 MISS → 같은 이미지로 HIT
        async_imgs = [next(images) for _ in range(args.repeat)]
        for status in ("MISS", "HIT"):
            for i, (cls, img) in enumerate(async_imgs, 1):
                accepted, job, http_ms, final_ms = async_call(client, img)
                row(i, "async", status, http_ms, final_ms, job, job["job_id"], cls, img)
                save(f"async_{status.lower()}_{i}_{cls}_202.json", accepted)
                save(f"async_{status.lower()}_{i}_{cls}_final.json", job)

    out = Path("reports/data/timings.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'path':<6} {'cache':<13} {'run':>3} {'http_ms':>8} {'final_ms':>9} {'infer_ms':>9} {'hit':>5} {'job':>4}  image -> pred(score)")
    for r in rows:
        print(f"{r['path']:<6} {r['cache_status']:<13} {r['run_id']:>3} {r['http_ms']:>8} {r['final_ms']:>9} "
              f"{str(r['inference_ms']):>9} {str(r['cache_hit']):>5} {str(r['job_id']):>4}  "
              f"{r['image']} -> {r['pred_label']}({r['score']})")

    def median(vals):
        s = sorted(vals)
        return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2

    print("\n[중앙값, 워밍업 제외]")
    for p in ("sync", "async"):
        for c in ("MISS", "HIT"):
            g = [r for r in rows if r["path"] == p and r["cache_status"] == c]
            inf = [r["inference_ms"] for r in g if r["inference_ms"] is not None]
            print(f"{p:<6} {c:<5} http={median([r['http_ms'] for r in g]):.1f}ms "
                  f"final={median([r['final_ms'] for r in g]):.1f}ms "
                  f"inference={median(inf) if inf else None}")
    print(f"\n저장: {out}, {EVIDENCE}/")


if __name__ == "__main__":
    main()
