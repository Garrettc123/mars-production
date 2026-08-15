# MARS reproducibility checklist

## Before running

- [ ] Select public or authorized datasets and record licenses.
- [ ] Define task verifier before test evaluation.
- [ ] Freeze train/dev/test split and input manifest.
- [ ] Fix model ID, provider/version, temperature, max tokens, and prompt version.
- [ ] Pre-register routing threshold and primary metric.
- [ ] Commit the code and record the Git SHA.

## During runs

- [ ] Save one JSONL row per prediction.
- [ ] Record confidence, uncertainty, route, tokens, cost, latency, seed, and timestamp.
- [ ] Record evaluator/verifier version and failure cases.
- [ ] Run each stochastic condition across multiple seeds.

## Before reporting

- [ ] Generate metrics directly from raw prediction files.
- [ ] Report confidence intervals or bootstrap intervals.
- [ ] Report results by task category and aggregate.
- [ ] Include all baseline conditions.
- [ ] Include limitations, negative results, and data exclusions.
- [ ] Confirm no target/example numbers remain presented as measured results.
