"""
Milestone 6 – Requirement 6.2 (5 pts)
Behavioral & Adversarial Robustness Testing
--------------------------------------------
Tests the story-point estimator against edge cases, adversarial inputs,
and boundary conditions.  Works in two modes:

  1. API mode  – sends requests to the running FastAPI service (Milestone 5)
  2. Direct mode – calls the model/fallback directly (no server required)

Test categories:
  - Empty / null inputs
  - Boundary inputs (very short, very long)
  - Noisy / adversarial text
  - Typo injection
  - Non-English text
  - Code snippets as title/description
  - Numeric / gibberish inputs
  - Repeated content
  - SQL / prompt injection attempts

Outputs:
  results/robustness_results.json
  results/robustness_report.html
"""

import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import requests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API_BASE = "http://localhost:8000"   # Milestone 5 FastAPI


# ── Test case dataclass ───────────────────────────────────────────────────────

@dataclass
class RobustnessTest:
    name:        str
    category:    str
    title:       str
    description: str
    expected_behavior: str    # human-readable expectation
    # filled by runner
    prediction:  Optional[float] = None
    model_used:  Optional[str]   = None
    response_time_ms: Optional[float] = None
    status:      str = "PENDING"  # PASS / FAIL / WARN / ERROR
    notes:       str = ""


# ── Test suite definition ─────────────────────────────────────────────────────

def build_test_suite() -> list[RobustnessTest]:
    return [
        # ── Empty / null inputs ───────────────────────────────────────────
        RobustnessTest(
            name="empty_title_and_description",
            category="empty_input",
            title="", description="",
            expected_behavior="Returns a valid numeric prediction (fallback), no crash",
        ),
        RobustnessTest(
            name="whitespace_only",
            category="empty_input",
            title="   ", description="   \n\t  ",
            expected_behavior="Returns a valid numeric prediction, no crash",
        ),

        # ── Very short inputs ─────────────────────────────────────────────
        RobustnessTest(
            name="single_word_title",
            category="short_input",
            title="Fix", description="",
            expected_behavior="Returns prediction in [1, 13] range",
        ),
        RobustnessTest(
            name="two_word_story",
            category="short_input",
            title="Add button", description="Add a button",
            expected_behavior="Returns reasonable low prediction (1-3 SP)",
        ),

        # ── Very long inputs ──────────────────────────────────────────────
        RobustnessTest(
            name="very_long_description",
            category="long_input",
            title="Implement OAuth2 login",
            description=" ".join(["Implement OAuth2 login with Google and Facebook providers."] * 100),
            expected_behavior="Handles truncation, returns valid prediction, no OOM crash",
        ),
        RobustnessTest(
            name="extremely_long_title",
            category="long_input",
            title="A" * 5000,
            description="Normal description",
            expected_behavior="Returns valid prediction without crash",
        ),

        # ── Noisy / special characters ────────────────────────────────────
        RobustnessTest(
            name="emoji_input",
            category="noisy_input",
            title="🚀 Deploy feature 🎉",
            description="Ship the new 🌟 dashboard with charts 📊 and filters 🔍",
            expected_behavior="Returns valid numeric prediction, handles unicode",
        ),
        RobustnessTest(
            name="special_characters",
            category="noisy_input",
            title="Fix <script>alert('xss')</script> bug",
            description="The endpoint returns {\"error\": null} when input contains $pecial ch@racters & symbols!",
            expected_behavior="Returns valid prediction, no injection executed",
        ),
        RobustnessTest(
            name="all_punctuation",
            category="noisy_input",
            title="!!! ??? ### $$$ %%%",
            description="@@@@ **** ^^^^ ~~~~",
            expected_behavior="Returns valid numeric prediction, no crash",
        ),
        RobustnessTest(
            name="mixed_language",
            category="noisy_input",
            title="Corriger le bug d'authentification",
            description="Les utilisateurs ne peuvent pas se connecter après la réinitialisation du mot de passe.",
            expected_behavior="Returns a prediction (model may perform worse but should not crash)",
        ),

        # ── Typo injection ────────────────────────────────────────────────
        RobustnessTest(
            name="heavy_typos",
            category="typo",
            title="Fixx logi bugg in useer authentiication",
            description="Usres recieve an errror mesage aftar resetting pasword. Nede to fix ASAP.",
            expected_behavior="Returns reasonable prediction, typos should not crash service",
        ),
        RobustnessTest(
            name="leetspeak",
            category="typo",
            title="4dd 0Auth2 l0g1n",
            description="1mpl3m3nt G00gl3 4nd F4c3b00k l0g1n pr0v1d3rs",
            expected_behavior="Returns a prediction without crash",
        ),

        # ── Code as input ─────────────────────────────────────────────────
        RobustnessTest(
            name="code_snippet_as_description",
            category="code_input",
            title="Refactor authentication module",
            description="""def authenticate(user, password):\n    if user == 'admin' and password == 'secret':\n        return True\n    return False\n\n# TODO: use bcrypt hashing instead""",
            expected_behavior="Returns a prediction; code input is a common real JIRA pattern",
        ),
        RobustnessTest(
            name="sql_in_title",
            category="code_input",
            title="'; DROP TABLE issues; --",
            description="Fix SQL injection vulnerability in login form",
            expected_behavior="Returns valid prediction, SQL not executed",
        ),

        # ── Numeric / gibberish ───────────────────────────────────────────
        RobustnessTest(
            name="pure_numbers",
            category="gibberish",
            title="123456789",
            description="987654321 111 222 333",
            expected_behavior="Returns valid numeric prediction",
        ),
        RobustnessTest(
            name="random_gibberish",
            category="gibberish",
            title="qwerty asdfgh zxcvbn",
            description="Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod",
            expected_behavior="Returns a prediction in reasonable range",
        ),

        # ── Repeated content ──────────────────────────────────────────────
        RobustnessTest(
            name="repeated_single_word",
            category="repetition",
            title="bug bug bug bug bug bug",
            description="fix fix fix fix fix fix fix fix fix fix",
            expected_behavior="Returns valid prediction without looping / hanging",
        ),

        # ── Realistic JIRA stories (sanity checks) ────────────────────────
        RobustnessTest(
            name="realistic_small_story",
            category="sanity",
            title="Add loading spinner to login button",
            description="When the login form is submitted, show a spinner on the button to indicate loading.",
            expected_behavior="Predicts small value (1-3 SP)",
        ),
        RobustnessTest(
            name="realistic_large_story",
            category="sanity",
            title="Migrate database from MySQL to PostgreSQL",
            description=(
                "Migrate the entire production database from MySQL 5.7 to PostgreSQL 14. "
                "This includes schema migration, data migration, stored procedure rewrite, "
                "ORM compatibility updates, performance testing, rollback plan, and zero-downtime deployment."
            ),
            expected_behavior="Predicts large value (8-13+ SP)",
        ),
    ]


# ── Predictor (direct mode – no server) ──────────────────────────────────────

class DirectPredictor:
    """Calls model/fallback directly without going through FastAPI."""
    def __init__(self):
        self._model_loaded = False
        self._model = None
        self._try_load()

    def _try_load(self):
        import os
        if not os.getenv("HF_TOKEN"):
            log.info("Direct mode: using word-count fallback (no HF_TOKEN).")
            return
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Milestone 5"))
            from app.model import predict as m5_predict
            self._model = m5_predict
            self._model_loaded = True
        except Exception:
            pass

    def predict(self, title: str, description: str):
        text = f"Title: {title} Description: {description}"
        if self._model_loaded:
            return self._model(text)
        # word-count fallback (mirrors Milestone 5 logic)
        words = len(text.split())
        return max(1.0, min(13.0, words // 8 + 1))


# ── Test runner ───────────────────────────────────────────────────────────────

def _run_via_api(test: RobustnessTest, timeout: float = 10.0) -> RobustnessTest:
    payload = {"title": test.title, "description": test.description}
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=timeout)
        elapsed = (time.perf_counter() - t0) * 1000
        test.response_time_ms = round(elapsed, 2)

        if resp.status_code == 200:
            data = resp.json()
            test.prediction = data.get("story_points")
            test.model_used = data.get("model_used", "unknown")
            test.status = "PASS" if test.prediction is not None else "FAIL"
        else:
            test.status = "FAIL"
            test.notes  = f"HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.ConnectionError:
        test.status = "ERROR"
        test.notes  = "Connection refused — is the FastAPI server running on port 8000?"
    except requests.exceptions.Timeout:
        test.status = "FAIL"
        test.notes  = f"Timeout after {timeout}s"
    except Exception as exc:
        test.status = "ERROR"
        test.notes  = str(exc)
    return test


def _run_direct(test: RobustnessTest, predictor: DirectPredictor) -> RobustnessTest:
    t0 = time.perf_counter()
    try:
        pred = predictor.predict(test.title, test.description)
        elapsed = (time.perf_counter() - t0) * 1000
        test.response_time_ms = round(elapsed, 2)
        test.prediction  = round(float(pred), 4)
        test.model_used  = "fallback"
        test.status      = "PASS" if test.prediction is not None else "FAIL"
    except Exception as exc:
        test.status = "ERROR"
        test.notes  = str(exc)
    return test


def _evaluate_result(test: RobustnessTest) -> RobustnessTest:
    """Apply expectation checks beyond just 'no crash'."""
    if test.status != "PASS" or test.prediction is None:
        return test

    p = test.prediction

    # All predictions must be numeric and finite
    import math
    if not math.isfinite(p):
        test.status = "FAIL"
        test.notes += " Non-finite prediction."
        return test

    # Range check: story points should be roughly [0, 100]
    if p < 0 or p > 200:
        test.status = "WARN"
        test.notes += f" Prediction {p:.2f} outside expected range [0, 200]."

    # Sanity category: realistic stories
    if test.category == "sanity":
        if "small" in test.name and p > 5:
            test.status = "WARN"
            test.notes += f" Expected small SP (1-3), got {p:.2f}."
        elif "large" in test.name and p < 5:
            test.status = "WARN"
            test.notes += f" Expected large SP (8+), got {p:.2f}."

    return test


def run_robustness_tests(use_api: bool = True) -> dict:
    tests = build_test_suite()
    predictor = None if use_api else DirectPredictor()

    # Check if API is reachable
    if use_api:
        try:
            r = requests.get(f"{API_BASE}/health", timeout=3)
            if r.status_code != 200:
                raise ValueError()
            log.info("API is reachable — running in API mode.")
        except Exception:
            log.warning("API not reachable — falling back to direct mode.")
            use_api = False
            predictor = DirectPredictor()

    results = []
    for test in tests:
        log.info(f"  [{test.category}] {test.name} …")
        if use_api:
            test = _run_via_api(test)
        else:
            test = _run_direct(test, predictor)
        test = _evaluate_result(test)
        results.append(test)
        status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "ERROR": "🔴"}.get(test.status, "?")
        log.info(f"    {status_icon} {test.status}  pred={test.prediction}  "
                 f"time={test.response_time_ms}ms  {test.notes}")

    # ── Summary stats ──────────────────────────────────────────────────────
    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "ERROR": 0}
    for t in results:
        counts[t.status] = counts.get(t.status, 0) + 1

    by_category = {}
    for t in results:
        by_category.setdefault(t.category, []).append(
            {"name": t.name, "status": t.status, "prediction": t.prediction,
             "response_time_ms": t.response_time_ms, "notes": t.notes}
        )

    report = {
        "mode": "api" if use_api else "direct",
        "total_tests": len(results),
        "summary_counts": counts,
        "pass_rate": round(counts["PASS"] / len(results), 4) if results else 0,
        "by_category": by_category,
        "all_tests": [asdict(t) for t in results],
    }

    json_path = RESULTS_DIR / "robustness_results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Robustness results → {json_path}")

    _html_robustness_report(report)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  ROBUSTNESS TEST SUMMARY")
    print("=" * 68)
    print(f"  Mode        : {'API (FastAPI service)' if use_api else 'Direct (fallback)'}")
    print(f"  Total tests : {report['total_tests']}")
    print(f"  ✅ PASS     : {counts['PASS']}")
    print(f"  ⚠️  WARN     : {counts['WARN']}")
    print(f"  ❌ FAIL     : {counts['FAIL']}")
    print(f"  🔴 ERROR    : {counts['ERROR']}")
    print(f"  Pass rate   : {report['pass_rate']*100:.1f}%")
    print("=" * 68)

    return report


def _html_robustness_report(report: dict):
    status_colors = {
        "PASS":  "#d4edda", "FAIL": "#f8d7da",
        "WARN":  "#fff3cd", "ERROR": "#f5c6cb",
    }
    rows = ""
    for t in report["all_tests"]:
        bg = status_colors.get(t["status"], "#fff")
        rows += (
            f'<tr style="background:{bg}">'
            f'<td>{t["category"]}</td>'
            f'<td>{t["name"]}</td>'
            f'<td>{t["status"]}</td>'
            f'<td>{t["prediction"]}</td>'
            f'<td>{t["response_time_ms"]}</td>'
            f'<td style="font-size:12px">{t["expected_behavior"]}</td>'
            f'<td style="font-size:12px;color:#c00">{t["notes"]}</td>'
            f'</tr>\n'
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Robustness Test Report</title>
<style>
  body {{font-family:Arial,sans-serif;max-width:1200px;margin:40px auto;color:#333}}
  h1 {{color:#2c3e50;border-bottom:3px solid #2c3e50;padding-bottom:8px}}
  table {{border-collapse:collapse;width:100%;font-size:13px}}
  th,td {{border:1px solid #ccc;padding:6px 10px;text-align:left;vertical-align:top}}
  th {{background:#2c3e50;color:#fff}}
  .stats {{display:flex;gap:20px;margin:16px 0}}
  .stat {{background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:12px 20px;text-align:center}}
  .stat .val {{font-size:28px;font-weight:bold;color:#2c3e50}}
</style>
</head>
<body>
<h1>Milestone 6 – Robustness & Adversarial Testing Report</h1>
<div class="stats">
  <div class="stat"><div class="val">{report['total_tests']}</div>Total Tests</div>
  <div class="stat"><div class="val" style="color:#155724">{report['summary_counts']['PASS']}</div>PASS</div>
  <div class="stat"><div class="val" style="color:#856404">{report['summary_counts']['WARN']}</div>WARN</div>
  <div class="stat"><div class="val" style="color:#721c24">{report['summary_counts']['FAIL']}</div>FAIL</div>
  <div class="stat"><div class="val" style="color:#721c24">{report['summary_counts']['ERROR']}</div>ERROR</div>
  <div class="stat"><div class="val">{report['pass_rate']*100:.0f}%</div>Pass Rate</div>
</div>
<table>
<tr><th>Category</th><th>Test Name</th><th>Status</th><th>Prediction</th>
    <th>Time (ms)</th><th>Expected Behavior</th><th>Notes</th></tr>
{rows}
</table>
</body>
</html>"""
    html_path = RESULTS_DIR / "robustness_report.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Robustness HTML report → {html_path}")


if __name__ == "__main__":
    run_robustness_tests(use_api=True)
