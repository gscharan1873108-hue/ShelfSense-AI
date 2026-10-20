"""Store-manager summary. Uses a local small LM via Ollama when SHELFSENSE_SLM is set
(e.g. `qwen2.5:3b`), otherwise a deterministic template."""
from __future__ import annotations

import json
import os
import urllib.request

from .models import Report

_ORDER = ["critical", "high", "medium", "low"]


def template_summary(report: Report) -> str:
    s = report.scores
    counts: dict[str, int] = {}
    for i in report.issues:
        counts[i.severity] = counts.get(i.severity, 0) + 1
    sev = ", ".join(f"{n} {k}" for k, n in sorted(counts.items(), key=lambda kv: _ORDER.index(kv[0])))
    text = (f"Planogram compliance {s['planogram_compliance']:.0f}%, availability {s['availability']:.0f}%"
            + (f", label accuracy {s['label_accuracy']:.0f}%" if "label_accuracy" in s else "")
            + f". Overall shelf health {s['overall']:.0f}/100. ")
    text += f"{len(report.issues)} issues ({sev})." if report.issues else "No issues detected."
    if report.issues:
        text += " Start with: " + " ".join(f"{k + 1}) {i.action}" for k, i in enumerate(report.issues[:3]))
    return text


def summarize(report: Report) -> str:
    base = template_summary(report)
    model = os.environ.get("SHELFSENSE_SLM")
    if not model:
        return base
    facts = {"scores": report.scores,
             "issues": [{"severity": i.severity, "shelf": i.shelf, "message": i.message, "action": i.action}
                        for i in report.issues[:12]]}
    prompt = ("You are a retail shelf-audit assistant. Using ONLY the JSON facts below, write a 3-4 sentence "
              "briefing for a store manager listing the most urgent actions first. Do not invent numbers.\n"
              + json.dumps(facts))
    req = urllib.request.Request(
        os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["response"].strip() or base
    except (OSError, KeyError, ValueError):
        return base
