"""Evaluate retrieved evidence and preserve per-query traces."""
from typing import Any
import re
import unicodedata
from typing import Any


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).casefold()
    return re.sub(r"[\W_]+", " ", s).strip()


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and enough gold evidence; keep hit@1 and hit@3 trace.

    Accept any documented alternative gold evidence. Do not infer human grounding
    or chunk coherence labels from exact string match.
    """
    gold_source = question.get("gold_source_id")
    gold_page = question.get("gold_page")
    evidences = [question.get("evidence_quote", "")] + question.get("alt_evidence", [])
    valid_evidences = [_norm(e) for e in evidences if e]

    hit_results = []
    for hit in hits:
        hit_source = hit.get("document_id")
        hit_page = hit.get("page")
        content = hit.get("text", "")

        source_match = (hit_source == gold_source) if gold_source else True
        page_match = (hit_page == gold_page) if gold_page else True
        content_match = any(ev in _norm(content) for ev in valid_evidences) if valid_evidences else True

        hit_results.append(source_match and page_match and content_match)

    return {
        "question_id": question.get("id"),
        "question": question.get("question"),
        "split": question.get("split"),
        "strategy": question.get("strategy"),
        "hit_1": bool(hit_results[0]) if len(hit_results) > 0 else False,
        "hit_3": any(hit_results[:3]),
        "hits": hits
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    summary = {}
    for trace in traces:
        key = (trace.get("strategy"), trace.get("split"))
        if key not in summary:
            summary[key] = {"total": 0, "hit_1": 0, "hit_3": 0}
        
        summary[key]["total"] += 1
        summary[key]["hit_1"] += int(trace.get("hit_1", False))
        summary[key]["hit_3"] += int(trace.get("hit_3", False))

    result = {}
    for (strategy, split), metrics in summary.items():
        total = metrics["total"]
        key_str = f"{strategy}_{split}" if strategy and split else "overall"
        result[key_str] = {
            "total": total,
            "hit_1_rate": metrics["hit_1"] / total if total > 0 else 0.0,
            "hit_3_rate": metrics["hit_3"] / total if total > 0 else 0.0,
        }
        
    return result