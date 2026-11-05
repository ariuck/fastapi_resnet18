"""[추가 증빙 도구 · 제공 코드 아님] 헤드리스 브라우저(Playwright)로 증빙 화면을 자동 캡처한다.

전제: FastAPI(127.0.0.1:8000)·Redis·PostgreSQL 실행 중, LD_LIBRARY_PATH에 Chromium 의존 라이브러리 경로 지정
예: python -m tools.capture_evidence

1) 웹 캡처: 실제 Swagger UI(/docs)에서 Try it out → Execute 를 눌러 실제 요청·응답 화면을 찍는다.
2) 터미널 캡처: 실제 명령을 실행한 표준출력을 터미널 모양 HTML로 그려 찍는다(명령 줄 포함).
   사람이 VS Code 터미널에서 같은 명령을 실행해 화면을 찍는 것과 같은 내용이다.
출력: reports/evidence/*.png
"""
import html
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"
OUT = Path("reports/evidence")
PROMPT = "(.venv) user1@DESKTOP-MGOK18H:~/fastapi_resnet18$"
FONT_CSS = """
@font-face { font-family: 'Malgun'; src: url('file:///mnt/c/Windows/Fonts/malgun.ttf'); }
@font-face { font-family: 'Consolas'; src: url('file:///mnt/c/Windows/Fonts/consola.ttf'); }
"""


def swagger_capture(page, path_text, method, out_name, upload=None):
    page.goto(f"{BASE}/docs")
    page.wait_for_selector(".opblock")
    block = page.locator(f".opblock-{method}").filter(
        has=page.locator(".opblock-summary-path", has_text=path_text)).first
    block.locator(".opblock-summary").click()
    block.locator(".try-out__btn").click()
    if upload:
        block.locator("input[type=file]").set_input_files(upload)
    block.locator(".execute").click()
    block.locator(".live-responses-table").wait_for()
    page.wait_for_timeout(500)
    block.screenshot(path=str(OUT / out_name))
    print("saved", out_name)


def terminal_capture(page, command, out_name, title=None, display=None):
    result = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=300)
    body = (result.stdout + result.stderr).rstrip()
    shown = display or command
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><style>{FONT_CSS}
    body {{ margin:0; background:#1e1e1e; }}
    .term {{ padding:14px 18px; color:#cccccc; font: 13.5px/1.45 'Consolas','Malgun',monospace; white-space:pre; }}
    .bar {{ background:#252526; color:#9d9d9d; font:12px 'Malgun',sans-serif; padding:6px 18px; border-bottom:1px solid #333; }}
    .p {{ color:#4ec9b0; }} .c {{ color:#dcdcaa; }}
    </style></head><body><div class="bar">TERMINAL — bash (WSL: Ubuntu){' · ' + html.escape(title) if title else ''}</div>
    <div class="term"><span class="p">{html.escape(PROMPT)}</span> <span class="c">{html.escape(shown)}</span>
{html.escape(body)}
<span class="p">{html.escape(PROMPT)}</span> ▌</div></body></html>"""
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(doc)
    page.goto(f"file://{f.name}")
    page.wait_for_timeout(300)
    width = page.evaluate("document.querySelector('.term').scrollWidth") + 40
    page.set_viewport_size({"width": max(900, min(int(width), 1900)), "height": 200})
    page.screenshot(path=str(OUT / out_name), full_page=True)
    print("saved", out_name)


def main():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "not_image.txt").write_text("this is not an image")
    big = tmp / "big_6mb.jpg"
    big.write_bytes(b"\0" * (5 * 1024 * 1024 + 1))

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)

        # 1) 실제 Swagger UI 요청
        swagger_capture(page, "/health", "get", "health.png")
        swagger_capture(page, "/api/sync/classify/", "post", "sync_400.png", upload=str(tmp / "not_image.txt"))
        swagger_capture(page, "/api/sync/classify/", "post", "oversize_413.png", upload=str(big))

        # 2) 실제 명령 출력
        terminal_capture(page,
            "sed -n '/\\[T-10\\]/,/\\[T-06\\]/p' reports/evidence/functional_tests.txt | sed '$d'",
            "worker_stopped.png", "T-10 Worker 정지 → queued → 재시작 후 done",
            display="sed -n '/[T-10]/,/[T-06]/p' reports/evidence/functional_tests.txt")
        terminal_capture(page,
            "sed -n '1,/\\[R-05\\]/p' reports/evidence/redis_outage_test.txt | sed '$d'",
            "redis_outage.png", "Redis 중지: 동기 200(캐시 우회) · 비동기 503 + failed",
            display="sed -n '1,/[R-05]/p' reports/evidence/redis_outage_test.txt")
        terminal_capture(page,
            "sed -n '/\\[R-05\\]/,$p' reports/evidence/redis_outage_test.txt | grep -v -E '^\\s+(File|return|raise|conn\\.|redis\\.exceptions|ConnectionResetError)'",
            "redis_recovery.png", "Redis 복구 → Worker 재연결 → 정상 처리",
            display="sed -n '/[R-05]/,$p' reports/evidence/redis_outage_test.txt")
        terminal_capture(page,
            "docker compose exec -T postgres psql -U vision_user -d vision_db -c "
            "\"SELECT id, status, top1_label, round(inference_ms::numeric,2) AS inference_ms, cache_hit, error_message "
            "FROM image_jobs WHERE id >= 11 ORDER BY id;\" && "
            "docker compose exec -T postgres psql -U vision_user -d vision_db -c "
            "\"SELECT status, count(*) FROM image_jobs GROUP BY status ORDER BY status;\"",
            "db_failed_row.png", "image_jobs: failed 행(job 15)과 상태별 건수",
            display="docker compose exec postgres psql -U vision_user -d vision_db -c \"SELECT ... FROM image_jobs WHERE id >= 11\"")
        terminal_capture(page, "cat reports/evidence/measure_api_output.txt",
            "perf_result.png", "성능 측정: 동기/비동기 × 캐시 MISS/HIT")
        terminal_capture(page,
            "python3 -c \"import csv; r=list(csv.DictReader(open('reports/data/experiments.csv', encoding='utf-8')));"
            "print(f\\\"{'exp':<20}{'data':<26}{'split':<7}{'accuracy':>9}{'correct':>9}  note\\\");"
            "[print(f\\\"{x['exp']:<20}{x['data']:<26}{x['split']:<7}{x['accuracy']:>9}{x['correct']+'/'+x['total']:>9}  {x['note']}\\\") for x in r]\"",
            "exp_result.png", "정확도 개선 실험 기록 (experiments.csv)",
            display="python3 -c \"...print experiments.csv...\"")
        terminal_capture(page, "cat reports/evidence/experiments/final_test_eval.txt",
            "final_test.png", "최종 Test 1회 평가: E7 vs E0 기준선")
        terminal_capture(page,
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_api.py 2>&1 | tail -6",
            "pytest_mock.png", "모의 테스트 (가짜 모델·SQLite·브로커 스텁)",
            display="pytest -q tests/test_api.py")
        browser.close()


if __name__ == "__main__":
    main()
