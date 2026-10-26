"""
Reporting — findings recorded during play and the end-of-run report (GCA-011).

Findings come from the model's step output. Each is saved with the step, the
actions that led to it (repro) and the screenshot at that moment. When the run
ends, the model writes the role's report (bug report, mock review, run
summary...) from the findings, goal outcomes and run statistics.
"""

import json
import os
from datetime import datetime
from typing import List, Optional

from PIL import Image

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "Reports")

FINDING_TYPES = ("bug", "ux", "positive", "note")
SEVERITIES = ("low", "medium", "high")


class FindingsLog:
    def __init__(self, reports_dir: str = REPORTS_DIR):
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(reports_dir, self.session)
        os.makedirs(self.dir, exist_ok=True)
        self.findings: List[dict] = []

    def _is_duplicate(self, title: str) -> bool:
        t = title.strip().lower()
        return any(f["title"].strip().lower() == t for f in self.findings)

    def add(self, raw: dict, step: int, repro: List[str], screenshot: Optional[Image.Image]) -> Optional[dict]:
        """Store one finding from the model. Returns it, or None if invalid/duplicate."""
        if not isinstance(raw, dict):
            return None
        title = str(raw.get("title", "")).strip()
        if not title or self._is_duplicate(title):
            return None
        ftype = raw.get("type") if raw.get("type") in FINDING_TYPES else "note"
        severity = raw.get("severity") if raw.get("severity") in SEVERITIES else "low"
        n = len(self.findings) + 1
        shot = ""
        if screenshot is not None:
            shot = f"finding_{n:03d}_step{step}.png"
            try:
                screenshot.save(os.path.join(self.dir, shot))
            except Exception as e:
                print(f"[findings] could not save screenshot: {e}")
                shot = ""
        finding = {
            "id": n, "step": step, "type": ftype, "severity": severity, "title": title,
            "details": str(raw.get("details", "")).strip(),
            "confirmed": bool(raw.get("confirmed", False)),
            "repro": list(repro), "screenshot": shot,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        self.findings.append(finding)
        self._save()
        return finding

    def titles_for_prompt(self, limit: int = 30) -> str:
        """Titles already recorded, so the model doesn't report them again."""
        if not self.findings:
            return ""
        titles = [f"- {f['title']}" for f in self.findings[-limit:]]
        return "=== RECORDED FINDINGS (do not repeat) ===\n" + "\n".join(titles) + "\n=== END RECORDED FINDINGS ==="

    def _save(self):
        with open(os.path.join(self.dir, "findings.json"), "w", encoding="utf-8") as f:
            json.dump(self.findings, f, indent=2, ensure_ascii=False)

    def format_for_report(self) -> str:
        if not self.findings:
            return "FINDINGS: none recorded."
        lines = ["FINDINGS (in the order they were found):"]
        for f in self.findings:
            lines.append(
                f"- #{f['id']} step {f['step']} [{f['type']}/{f['severity']}] "
                f"{'CONFIRMED' if f['confirmed'] else 'unconfirmed'} — {f['title']}: {f['details']}"
                + (f" | repro: {' → '.join(f['repro'])}" if f['repro'] else "")
                + (f" | screenshot: {f['screenshot']}" if f['screenshot'] else ""))
        return "\n".join(lines)

    def save_report(self, text: str, role: str) -> str:
        path = os.path.join(self.dir, f"report_{role}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path


def latest_report() -> Optional[dict]:
    """Newest report file under Reports/, for the dashboard API."""
    if not os.path.isdir(REPORTS_DIR):
        return None
    best = None
    for session in os.listdir(REPORTS_DIR):
        d = os.path.join(REPORTS_DIR, session)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.startswith("report_") and name.endswith(".md"):
                p = os.path.join(d, name)
                if best is None or os.path.getmtime(p) > os.path.getmtime(best):
                    best = p
    if best is None:
        return None
    with open(best, encoding="utf-8") as f:
        return {"path": best, "text": f.read()}
