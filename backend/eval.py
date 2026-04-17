"""Automated eval runner — regression suite for the EU compliance pipeline.

Loads ground_truth_eval_dataset.csv (50 labeled cases), runs each claim through
the five-gate rules engine, and scores against ground truth on three dimensions:
  1. Risk level correct
  2. Regulation cited correctly
  3. Issues identified

Gates 1-3 are pure Python (no API calls). Gates 4-5 use focused LLM calls.

Run as: python -m backend.eval
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from backend.analyze import analyze_claim, _parse_certifications
from backend.ingest import ensure_ingested

EVAL_CSV = Path(__file__).resolve().parent.parent / "ground_truth_eval_dataset.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"

GATE_LABELS = {1: "BLACKLIST", 2: "NEUTRALITY", 3: "CERTIFIED", 4: "COMPARATIVE", 5: "LLM"}


# ── Loading ──

def load_eval_dataset() -> list[dict]:
    """Load CSV, normalize 'None' certs to empty string."""
    rows = []
    with open(EVAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            certs = row.get("certifications", "").strip()
            if certs.lower() == "none":
                certs = ""
            row["certifications"] = certs
            rows.append(row)
    return rows


# ── Per-case evaluation ──

def evaluate_claim(claim: str, certs: str) -> dict:
    """Run claim through the five-gate rules engine directly."""
    certs_list = _parse_certifications(certs)
    return analyze_claim(claim, certs_list, "Eval Product")


# ── Scoring ──

REGULATION_KEYWORDS = {
    "ECGT": ["ecgt", "2024/825"],
    "UCPD": ["ucpd", "2005/29"],
    "ESPR": ["espr", "2024/1781"],
}


def _extract_regulation_key(text: str) -> str | None:
    """Extract regulation key (ECGT/UCPD/ESPR) from text via keyword matching."""
    text_lower = text.lower()
    for key, keywords in REGULATION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return key
    return None


def _extract_article_refs(text: str) -> set[str]:
    """Extract article references from text using regex patterns."""
    refs = set()
    patterns = [
        r"Annex\s+I\s+point\s+\d+[a-z]?",
        r"Article\s+\d+(?:\(\d+\))?(?:\([a-z]\))?",
        r"Recital\s+\d+",
        r"Article\s+\d+\(\d+\)\([a-z]\)",
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            refs.add(m.group(0).lower().strip())
    return refs


def score_case(result: dict, ground_truth: dict) -> dict:
    """Score a single case on 3 dimensions. Returns dict with pass/fail per dimension.

    Reads from rules engine output format:
      result["overall_risk"], result["findings"][*]["regulation"],
      result["findings"][*]["article"], result["findings"][*]["issue"]
    """
    scores = {}

    # 1. Risk level correct — exact case-insensitive match
    expected_risk = ground_truth["correct_risk"].strip().upper()
    got_risk = (result.get("overall_risk") or "").strip().upper()
    scores["risk_pass"] = got_risk == expected_risk
    scores["expected_risk"] = expected_risk
    scores["got_risk"] = got_risk

    # 2. Regulation cited correctly
    gt_reg_text = ground_truth.get("correct_regulation", "")
    expected_reg_key = _extract_regulation_key(gt_reg_text)

    # Collect all regulation text from findings
    tool_text_parts = []
    for f in result.get("findings", []):
        tool_text_parts.append(f.get("regulation", ""))
        tool_text_parts.append(f.get("article", ""))
        tool_text_parts.append(f.get("issue", "") or "")
        tool_text_parts.append(f.get("fix", "") or "")
    tool_combined = " ".join(tool_text_parts)
    got_reg_key = _extract_regulation_key(tool_combined)

    scores["reg_pass"] = (expected_reg_key is not None and got_reg_key == expected_reg_key)
    scores["expected_reg"] = gt_reg_text
    # Show first regulation from findings
    first_reg = result["findings"][0]["regulation"] if result.get("findings") else ""
    scores["got_reg"] = first_reg

    # 3. Issues identified
    correct_article = ground_truth.get("correct_article", "").strip()

    if correct_article.lower() == "none" or correct_article == "":
        # Compliant claim — pass if tool also says LOW
        scores["issues_pass"] = got_risk == "LOW"
    else:
        # Extract article refs from ground truth and tool output
        expected_refs = _extract_article_refs(correct_article)
        tool_issues_text = " ".join([
            f.get("article", "") + " " + (f.get("issue", "") or "")
            for f in result.get("findings", [])
        ])
        got_refs = _extract_article_refs(tool_issues_text)

        if not expected_refs:
            # No parseable refs in ground truth — pass if risk matches
            scores["issues_pass"] = scores["risk_pass"]
        else:
            # Check ≥50% of expected refs found
            matches = expected_refs & got_refs
            scores["issues_pass"] = len(matches) >= len(expected_refs) * 0.5

    scores["expected_article"] = correct_article
    scores["gate"] = result.get("gate_resolved_at", "?")
    scores["all_pass"] = scores["risk_pass"] and scores["reg_pass"] and scores["issues_pass"]

    return scores


# ── Orchestration ──

def run_eval():
    """Main eval loop: ensure_ingested → load CSV → evaluate all → score → report."""
    print("\n  Ensuring ChromaDB is populated...")
    ensure_ingested()

    dataset = load_eval_dataset()
    n = len(dataset)
    timestamp = datetime.now()

    print(f"""
======================================================================
  EU COMPLIANCE TOOL — REGRESSION EVAL (Rules Engine)
  {n} test cases | {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
======================================================================
""")

    results = []
    risk_correct = 0
    reg_correct = 0
    issues_correct = 0
    all_correct = 0
    failures = []
    gate_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for i, row in enumerate(dataset):
        case_id = row["id"]
        claim = row["claim"]
        certs = row["certifications"]
        short_claim = claim[:50] + "..." if len(claim) > 50 else claim

        result = evaluate_claim(claim, certs)
        scores = score_case(result, row)

        gate = scores["gate"]
        gate_label = GATE_LABELS.get(gate, "?")
        gate_counts[gate] = gate_counts.get(gate, 0) + 1

        risk_ok = "OK" if scores["risk_pass"] else "FAIL"
        reg_ok = "OK" if scores["reg_pass"] else "FAIL"
        issues_ok = "OK" if scores["issues_pass"] else "FAIL"
        status = "PASS" if scores["all_pass"] else "FAIL"

        print(f'  [{i+1:>2}/{n}] Case {case_id}: "{short_claim}"... {status} (G{gate}:{gate_label}) [risk:{risk_ok}, reg:{reg_ok}, issues:{issues_ok}]')

        if scores["risk_pass"]:
            risk_correct += 1
        if scores["reg_pass"]:
            reg_correct += 1
        if scores["issues_pass"]:
            issues_correct += 1
        if scores["all_pass"]:
            all_correct += 1
        else:
            failures.append({"case_id": case_id, "claim": claim, "certs": certs, "scores": scores})

        results.append({
            "case_id": case_id,
            "claim": claim,
            "certifications": certs,
            "result": result,
            "scores": scores,
        })

    # ── Score report ──
    print(f"""
======================================================================
  SCORE REPORT
======================================================================
  Risk level correct:     {risk_correct}/{n} ({100*risk_correct/n:.1f}%)
  Regulation cited:       {reg_correct}/{n} ({100*reg_correct/n:.1f}%)
  Issues identified:      {issues_correct}/{n} ({100*issues_correct/n:.1f}%)
  All three correct:      {all_correct}/{n} ({100*all_correct/n:.1f}%)
----------------------------------------------------------------------
  Gate distribution:""")
    for g in sorted(gate_counts):
        if gate_counts[g] > 0:
            print(f"    Gate {g} ({GATE_LABELS.get(g, '?'):>12}): {gate_counts[g]:>3} cases")
    print("======================================================================")

    if failures:
        print(f"""
  FAILURES ({len(failures)} cases):
  ------------------------------------------------------------------""")
        for f in failures:
            s = f["scores"]
            print(f'  Case {f["case_id"]}: "{f["claim"]}" (cert: {f["certs"] or "None"}) [Gate {s["gate"]}]')
            if not s["risk_pass"]:
                print(f'    Risk:       expected {s["expected_risk"]}, got {s["got_risk"]}')
            if not s["reg_pass"]:
                print(f'    Regulation: expected {s["expected_reg"]}')
                print(f'                got      {s["got_reg"]}')
            if not s["issues_pass"]:
                print(f'    Issues:     expected article {s["expected_article"]}')
            print()

    # ── Write JSON results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"eval_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    output = {
        "timestamp": timestamp.isoformat(),
        "total_cases": n,
        "scores": {
            "risk_correct": risk_correct,
            "regulation_correct": reg_correct,
            "issues_correct": issues_correct,
            "all_correct": all_correct,
        },
        "percentages": {
            "risk": round(100 * risk_correct / n, 1),
            "regulation": round(100 * reg_correct / n, 1),
            "issues": round(100 * issues_correct / n, 1),
            "all": round(100 * all_correct / n, 1),
        },
        "gate_distribution": {
            GATE_LABELS.get(g, str(g)): gate_counts[g]
            for g in sorted(gate_counts) if gate_counts[g] > 0
        },
        "results": results,
    }
    with open(out_file, "w", encoding="utf-8") as fout:
        json.dump(output, fout, indent=2, ensure_ascii=False)
    print(f"\n  Results written to {out_file}\n")

    sys.exit(0 if len(failures) == 0 else 1)


if __name__ == "__main__":
    run_eval()
