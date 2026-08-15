from metrics import expected_calibration_error, summarize


def test_ece_perfectly_calibrated():
    assert expected_calibration_error([1.0, 0.0], [1.0, 0.0]) == 0.0


def test_summary_has_core_metrics():
    rows = [
        {"is_correct": True, "confidence": 0.9, "routed_deep": False, "cost_usd": 0.01, "latency_ms": 10},
        {"is_correct": False, "confidence": 0.2, "routed_deep": True, "cost_usd": 0.02, "latency_ms": 20},
    ]
    result = summarize(rows)
    assert result["accuracy"] == 0.5
    assert result["error_detection_auroc"] == 1.0
    assert result["escalation_rate"] == 0.5
