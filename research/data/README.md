# Data protocol

Store only dataset manifests and public/releasable examples in version control. Keep proprietary or customer data outside the repository and document its governance.

Each JSONL item must include:

```json
{
  "id": "unique-item-id",
  "split": "train|dev|test",
  "task_type": "math|multiple_choice|code_fix|debugging|ambiguity",
  "input": "prompt or task payload",
  "ground_truth": "answer, test reference, or verifier target",
  "source": "dataset name/version",
  "license": "license or usage note"
}
```

Recommended first benchmark mix:

- Objective math and reasoning items.
- Multiple-choice knowledge items.
- Code/debugging examples with deterministic unit tests.
- Ambiguous or out-of-distribution items for selective prediction evaluation.

Do not leak test answers into prompts, routing features, or calibration fitting.