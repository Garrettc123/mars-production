import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expected_calibration_error(confidence, correct, bins=10):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1 else (confidence <= upper))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def risk_coverage(confidence, correct):
    order = np.argsort(-np.asarray(confidence))
    correct = np.asarray(correct)[order]
    points = []
    for count in range(1, len(correct) + 1):
        coverage = count / len(correct)
        risk = 1.0 - float(correct[:count].mean())
        points.append({"coverage": coverage, "risk": risk})
    return points


def summarize(rows):
    usable = [r for r in rows if r.get("is_correct") is not None]
    if not usable:
        raise ValueError("No scored records: every record must include is_correct true or false.")
    correct = np.asarray([bool(r["is_correct"]) for r in usable], dtype=float)
    confidence = np.asarray([float(r["confidence"]) for r in usable], dtype=float)
    errors = 1.0 - correct
    metrics = {
        "n": int(len(usable)),
        "accuracy": float(correct.mean()),
        "brier_score": float(np.mean((confidence - correct) ** 2)),
        "ece_10": expected_calibration_error(confidence, correct, bins=10),
        "escalation_rate": float(np.mean([bool(r.get("routed_deep", False)) for r in usable])),
        "mean_cost_usd": float(np.nanmean([r.get("cost_usd", np.nan) for r in usable])),
        "mean_latency_ms": float(np.nanmean([r.get("latency_ms", np.nan) for r in usable])),
        "risk_coverage": risk_coverage(confidence, correct)
    }
    if len(np.unique(errors)) == 2:
        metrics["error_detection_auroc"] = float(roc_auc_score(errors, 1.0 - confidence))
    else:
        metrics["error_detection_auroc"] = None
    total_cost = np.nansum([r.get("cost_usd", np.nan) for r in usable])
    metrics["cost_per_correct_answer_usd"] = float(total_cost / correct.sum()) if correct.sum() else None
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute MARS benchmark metrics from JSONL predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = summarize(load_jsonl(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
