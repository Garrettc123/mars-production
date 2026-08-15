# MARS Research Package

This directory contains the reproducible research package for evaluating the MARS (Metacognitive Adaptive Reasoning System) routing hypothesis.

## Research question

Does uncertainty-aware routing improve the reliability and cost-quality tradeoff of a language-model workflow relative to a fixed one-pass baseline and an always-deep baseline?

## Conditions

- `baseline`: one standard reasoning pass.
- `self_reported_confidence`: one standard pass with model-reported confidence.
- `mars_score_only`: MARS confidence recorded; no escalation.
- `mars_adaptive`: route to deep reasoning when MARS uncertainty exceeds a pre-registered threshold.
- `always_deep`: all items receive the deep reasoning path.

## Primary outcomes

- Exact-match or verifier-based accuracy.
- Expected Calibration Error (ECE).
- Brier score.
- Error-detection AUROC.
- Risk-coverage curve and area under risk-coverage curve.
- Escalation rate, latency, token use, and cost per correct answer.

## Integrity rules

1. Freeze test data and configuration before evaluating the final test split.
2. Preserve the Git commit SHA, model identifier, prompt version, seed, timestamp, and package versions for every run.
3. Do not edit routing thresholds after examining held-out results.
4. Report failures, missing data, and negative results.
5. Do not interpret placeholder manuscript tables as empirical findings.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r research/src/requirements.txt
python research/src/run_eval.py --config research/configs/baseline.yaml --input data/example_predictions.jsonl --output results/baseline.jsonl
python research/src/metrics.py --input results/baseline.jsonl --output results/baseline_metrics.json
pytest research/tests
```

The runner consumes JSONL predictions. Production adapters should be implemented outside the metric layer so that the same scoring code evaluates every condition.

## Status

The package is scaffolding for reproducible evaluation. It contains no claimed MARS benchmark result.