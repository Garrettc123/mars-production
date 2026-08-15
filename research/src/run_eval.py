import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Normalize MARS benchmark prediction records.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="JSONL records from a condition adapter")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for raw in load_jsonl(Path(args.input)):
        confidence = float(raw.get("confidence", 1.0 - float(raw.get("uncertainty", 0.5))))
        confidence = max(0.0, min(1.0, confidence))
        uncertainty = float(raw.get("uncertainty", 1.0 - confidence))
        uncertainty = max(0.0, min(1.0, uncertainty))
        routed_deep = bool(raw.get("routed_deep", False))
        if config.get("routing", {}).get("enabled") and "routed_deep" not in raw:
            routed_deep = uncertainty >= float(config["routing"]["uncertainty_threshold"])
        rows.append({
            **raw,
            "condition": raw.get("condition", config["condition"]),
            "model_id": raw.get("model_id", config["model_id"]),
            "prompt_version": raw.get("prompt_version", config["prompt_version"]),
            "seed": raw.get("seed", config["seed"]),
            "confidence": confidence,
            "uncertainty": uncertainty,
            "routed_deep": routed_deep,
            "timestamp_utc": raw.get("timestamp_utc", datetime.now(timezone.utc).isoformat())
        })

    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"records_written": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
