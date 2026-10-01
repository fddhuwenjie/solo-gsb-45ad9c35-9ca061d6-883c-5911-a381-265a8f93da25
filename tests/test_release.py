"""Batch release endpoint tests: exact scan matching, closed-loop and
open-chain rejection, spatial QC gate evaluation, 422/200 semantics and
determinism."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# 4 rows x 4 cols, 16 wells, two groups interleaved checkerboard-style.
LAYOUT_4X4 = {
    "gA1": "A1", "gA2": "A3", "gB1": "A2", "gB2": "A4",
    "gB3": "B1", "gB4": "B3", "gA3": "B2", "gA4": "B4",
    "gA5": "C1", "gA6": "C3", "gB5": "C2", "gB6": "C4",
    "gB7": "D1", "gB8": "D3", "gA7": "D2", "gA8": "D4",
}
SAMPLES_4X4 = [
    {"id": s, "group": "GA" if s.startswith("gA") else "GB"}
    for s in LAYOUT_4X4
]


def hash_deterministic(text: str) -> int:
    import hashlib

    return int(hashlib.md5(text.encode()).hexdigest()[:6], 16)


def _additive_readings():
    """Group baselines differ by 100 with small integer noise per well."""
    readings = {}
    for sid, well in LAYOUT_4X4.items():
        base = 100.0 if sid.startswith("gA") else 0.0
        noise = (hash_deterministic(sid) % 7) - 3  # -3..3
        readings[well] = base + noise
    return readings


def _payload(scan=None, **over):
    payload = {
        "plate": {"type": "custom", "rows": 4, "columns": 4},
        "layout": LAYOUT_4X4,
        "samples": SAMPLES_4X4,
        "scan": scan if scan is not None else {w: s for s, w in LAYOUT_4X4.items()},
        "blocked_wells": [],
        "readings": _additive_readings(),
        "seed": 7,
        "permutations": 100,
        "tolerance": 1e-6,
        "max_iterations": 100,
        "max_spatial_effect": 100.0,
        "min_p_value": 0.05,
    }
    payload.update(over)
    return payload


# --------------------------------------------------------------------- 200 released


def test_exact_match_and_passing_gates_is_released():
    r = client.post("/api/v1/arrangements/release", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "RELEASED"
    assert body["release"]["released"] is True
    assert body["release"]["reasons"] == []
    gates = {g["gate"]: g for g in body["release"]["gates"]}
    assert gates["max_abs_row_col_effect"]["passed"] is True
    assert gates["max_abs_row_col_effect"]["observed"] <= 100.0
    assert gates["permutation_p_value"]["passed"] is True
    assert gates["permutation_p_value"]["observed"] >= 0.05


def test_released_response_carries_scan_and_qc_summary():
    body = client.post("/api/v1/arrangements/release",
                       json=_payload()).json()
    assert set(body) == {"status", "release", "scan", "spatial_qc"}
    assert body["scan"] == {
        "matched": True,
        "occupied_wells": 16,
        "blocked_wells": [],
    }
    qc = body["spatial_qc"]
    assert set(qc) == {
        "seed", "plate", "parameters", "groups", "effects",
        "readings", "convergence", "permutation_test",
    }
    assert qc["convergence"]["converged"] is True
    assert qc["permutation_test"]["p_value"] == body["release"]["gates"][1]["observed"]


def test_release_is_deterministic():
    payload = _payload()
    b1 = client.post("/api/v1/arrangements/release", json=payload).json()
    b2 = client.post("/api/v1/arrangements/release", json=payload).json()
    assert b1 == b2


# ------------------------------------------------------------- 200 not released


def test_effect_gate_failure_is_not_released_without_summary():
    r = client.post("/api/v1/arrangements/release",
                    json=_payload(max_spatial_effect=0.0))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "NOT_RELEASED"
    assert body["release"]["released"] is False
    # No partial conclusion: neither the scan nor the QC summary is emitted.
    assert set(body) == {"status", "release"}
    gates = {g["gate"]: g for g in body["release"]["gates"]}
    assert gates["max_abs_row_col_effect"]["passed"] is False
    assert gates["max_abs_row_col_effect"]["observed"] > 0.0
    assert gates["permutation_p_value"]["passed"] is True
    assert any("spatial effect" in m for m in body["release"]["reasons"])


def test_p_value_gate_failure_is_not_released():
    r = client.post("/api/v1/arrangements/release",
                    json=_payload(min_p_value=0.999))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "NOT_RELEASED"
    gates = {g["gate"]: g for g in body["release"]["gates"]}
    assert gates["permutation_p_value"]["passed"] is False
    assert gates["permutation_p_value"]["observed"] < 0.999
    assert any("p-value" in m for m in body["release"]["reasons"])


def test_boundary_gate_values_pass():
    # Observed values are reused verbatim as the limits: <= / >= must pass.
    released = client.post("/api/v1/arrangements/release",
                           json=_payload()).json()
    effect = released["spatial_qc"]["effects"]["max_abs_row_col_effect"]
    p_value = released["spatial_qc"]["permutation_test"]["p_value"]
    r = client.post("/api/v1/arrangements/release", json=_payload(
        max_spatial_effect=effect, min_p_value=p_value))
    assert r.status_code == 200
    assert r.json()["status"] == "RELEASED"


def test_zero_permutations_refuses_release():
    r = client.post("/api/v1/arrangements/release",
                    json=_payload(permutations=0))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "NOT_RELEASED"
    gates = {g["gate"]: g for g in body["release"]["gates"]}
    assert gates["permutation_p_value"]["passed"] is False
    assert gates["permutation_p_value"]["observed"] is None
    assert any("permutation test was skipped" in m
               for m in body["release"]["reasons"])


# --------------------------------------------------------------------- 422 scan


def test_missing_sample_and_empty_target_well_422_lists_wells():
    scan = {w: s for s, w in LAYOUT_4X4.items()}
    del scan["A1"]  # gA1 missing; its target well is empty
    r = client.post("/api/v1/arrangements/release", json=_payload(scan=scan))
    assert r.status_code == 422
    body = r.json()
    assert "SAMPLES_MISSING" in body["infeasibility_categories"]
    assert "SCAN_NOT_MATCHED" in body["infeasibility_categories"]
    assert "status" not in body and "spatial_qc" not in body
    by_well = {m["well"]: m for m in body["misplaced_wells"]}
    assert by_well["A1"] == {"well": "A1", "expected": "gA1", "actual": None}


def test_unplanned_sample_422_lists_wells():
    scan = {w: s for s, w in LAYOUT_4X4.items()}
    scan["A1"] = "X9"  # gA1 replaced by a stranger
    r = client.post("/api/v1/arrangements/release", json=_payload(scan=scan))
    assert r.status_code == 422
    body = r.json()
    assert "UNPLANNED_SAMPLES" in body["infeasibility_categories"]
    assert "SAMPLES_MISSING" in body["infeasibility_categories"]
    assert "SCAN_NOT_MATCHED" in body["infeasibility_categories"]
    by_well = {m["well"]: m for m in body["misplaced_wells"]}
    assert by_well["A1"] == {"well": "A1", "expected": "gA1", "actual": "X9"}


def test_closed_loop_422_reports_cycle_and_misplaced_wells():
    scan = {w: s for s, w in LAYOUT_4X4.items()}
    scan["A1"], scan["A3"] = scan["A3"], scan["A1"]  # 2-cycle
    r = client.post("/api/v1/arrangements/release", json=_payload(scan=scan))
    assert r.status_code == 422
    body = r.json()
    cats = body["infeasibility_categories"]
    assert "SCAN_NOT_MATCHED" in cats
    assert "CLOSED_LOOP_MISPLACEMENT" in cats
    assert body["cycles"] == [["A1", "A3"]]
    by_well = {m["well"]: m for m in body["misplaced_wells"]}
    assert by_well["A1"]["actual"] == "gA2"
    assert by_well["A3"]["actual"] == "gA1"
    assert "status" not in body and "release" not in body


def test_open_chain_displacement_422_without_cycle():
    # 4x5 plate: column 5 is free space. Move gA1 from A1 onto the empty A5:
    # an open chain, not a cycle.
    scan = {w: s for s, w in LAYOUT_4X4.items()}
    scan["A5"] = scan.pop("A1")
    payload = _payload(
        scan=scan,
        plate={"type": "custom", "rows": 4, "columns": 5},
    )
    r = client.post("/api/v1/arrangements/release", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["infeasibility_categories"] == ["SCAN_NOT_MATCHED"]
    assert body["cycles"] == []
    by_well = {m["well"]: m for m in body["misplaced_wells"]}
    assert by_well["A1"] == {"well": "A1", "expected": "gA1", "actual": None}
    assert by_well["A5"] == {"well": "A5", "expected": None, "actual": "gA1"}


def test_blocked_well_occupied_422_gives_no_release_conclusion():
    r = client.post("/api/v1/arrangements/release",
                    json=_payload(blocked_wells=["A1"]))
    assert r.status_code == 422
    body = r.json()
    assert "BLOCKED_WELL_OCCUPIED" in body["infeasibility_categories"]
    assert "status" not in body and "release" not in body


def test_scan_duplicate_sample_422():
    scan = {w: s for s, w in LAYOUT_4X4.items()}
    scan["A2"] = scan["A1"]  # gA1 scanned twice; gB1 missing
    r = client.post("/api/v1/arrangements/release", json=_payload(scan=scan))
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert "SCAN_SAMPLES_DUPLICATED" in cats
    assert "SAMPLES_MISSING" in cats


def test_illegal_scan_well_422():
    scan = {w: s for s, w in LAYOUT_4X4.items()}
    scan["Z9"] = scan.pop("A1")
    r = client.post("/api/v1/arrangements/release", json=_payload(scan=scan))
    assert r.status_code == 422
    assert "WELL_REFERENCE_INVALID" in r.json()["infeasibility_categories"]


# --------------------------------------------------------------- 422 spatial QC


def test_non_convergence_is_422_without_verdict():
    r = client.post("/api/v1/arrangements/release", json=_payload(
        tolerance=1e-12, max_iterations=1))
    assert r.status_code == 422
    body = r.json()
    assert body["infeasibility_categories"] == ["SPATIAL_NOT_CONVERGED"]
    assert "status" not in body and "misplaced_wells" not in body


def test_spatial_group_insufficient_propagates_as_422():
    samples = [dict(s, group="GZ") if s["id"] == "gA8" else s
               for s in SAMPLES_4X4]
    r = client.post("/api/v1/arrangements/release",
                    json=_payload(samples=samples))
    assert r.status_code == 422
    assert "SPATIAL_GROUP_INSUFFICIENT" in r.json()["infeasibility_categories"]


# ------------------------------------------------------------- request errors


def test_request_validation_error_422():
    r = client.post("/api/v1/arrangements/release",
                    json={"plate": {"type": "96"}, "layout": {}})
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_gate_parameters_validated():
    # Missing required gates.
    payload = _payload()
    del payload["max_spatial_effect"]
    r = client.post("/api/v1/arrangements/release", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]

    # min_p_value outside (0, 1].
    r = client.post("/api/v1/arrangements/release",
                    json=_payload(min_p_value=1.5))
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]

    # max_spatial_effect negative.
    r = client.post("/api/v1/arrangements/release",
                    json=_payload(max_spatial_effect=-1.0))
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_existing_endpoints_remain_available():
    # Reconcile still generates correction instructions for the same cycle
    # that release refuses.
    scan = {w: s for s, w in LAYOUT_4X4.items()}
    scan["A1"], scan["A3"] = scan["A3"], scan["A1"]
    r = client.post("/api/v1/arrangements/reconcile", json={
        "plate": {"type": "custom", "rows": 4, "columns": 5},
        "target": LAYOUT_4X4,
        "scan": scan,
        "staging_well": "A5",
    })
    assert r.status_code == 200
    assert r.json()["move_count"] == 3
