"""Batch release verification for a fully scanned microplate.

Pipeline for ``POST /api/v1/arrangements/release``:

1. **Scan verification** — the submitted full scan (well → sample) is
   cross-checked against the planned layout (sample → well) with the same
   rules as the reconcile endpoint: duplicate/illegal well references,
   missing planned samples, unplanned samples, blocked-well occupancy and
   any positional disagreement (open chains *and* closed loops). Any such
   problem yields HTTP 422 with the problem categories and the misplaced
   well list; **no release conclusion is produced**.
2. **Spatial QC** — once the scan matches the plan exactly, the spatial
   QC pipeline (group-median centring, alternating median polish and the
   seed-controlled within-group permutation test) runs unchanged. Its own
   validity/non-convergence failures propagate as 422 without effects,
   p-value or any partial conclusion.
3. **Acceptance gates** — release is granted only when the polish
   converged (guaranteed by the spatial QC call), the observed maximal
   absolute row/column effect is no greater than ``max_spatial_effect`` and
   the permutation p-value is at least ``min_p_value``. Skipping the
   permutation test (``permutations = 0``) makes the p-value gate
   uncheckable, so release is refused. On every gate failure the service
   returns a non-released verdict with the failing reasons; no scan/QC
   summary is attached to a non-released response.
"""
from __future__ import annotations

from .plates import well_name
from .reconciler import ReconcileRequest, inspect_scan
from .schemas import (
    VIOLATION_CLOSED_LOOP_MISPLACEMENT,
    VIOLATION_SCAN_NOT_MATCHED,
    ReleaseRequest,
)
from .spatial_qc import SpatialQCInvalid, spatial_qc

RELEASED = "RELEASED"
NOT_RELEASED = "NOT_RELEASED"


class ReleaseInvalid(Exception):
    """The release request cannot proceed past verification (HTTP 422).

    Either the scan disagrees with the plan, or the spatial QC analysis
    itself is impossible. No release verdict and no partial conclusion are
    produced.
    """

    def __init__(self, categories: list[str], messages: list[str], **extra):
        self.categories = categories
        self.messages = messages
        # Optional, scan-specific payload (misplaced wells / cycles).
        self.extra = extra
        super().__init__("; ".join(messages) if messages else "invalid release request")


def _scan_summary(inspection) -> dict:
    return {
        "matched": True,
        "occupied_wells": len(inspection.scan_pos),
        "blocked_wells": [well_name(*w) for w in inspection.blocked],
    }


def release_verify(req: ReleaseRequest) -> dict:
    """Verify the scan, run spatial QC and decide the batch release gate."""
    # --- stage 1: scan must agree with the plan exactly --------------------
    reconcile_req = ReconcileRequest(
        plate=req.plate,
        target=req.layout,
        scan=req.scan,
        blocked_wells=req.blocked_wells,
    )
    inspection = inspect_scan(reconcile_req)
    cycles = [[well_name(*w) for w in cyc] for cyc in inspection.cycles]

    if inspection.categories or inspection.misplaced_wells:
        categories = list(inspection.categories)
        messages = list(inspection.messages)
        if inspection.misplaced_wells:
            categories.append(VIOLATION_SCAN_NOT_MATCHED)
            messages.append(
                f"scan disagrees with the planned layout at "
                f"{len(inspection.misplaced_wells)} well(s); release is not "
                "evaluated until the plate matches the plan"
            )
            if cycles:
                categories.append(VIOLATION_CLOSED_LOOP_MISPLACEMENT)
                messages.append(
                    f"closed-loop misplacement detected in {len(cycles)} cycle(s): "
                    "; ".join(" -> ".join(cyc) for cyc in cycles)
                )
        raise ReleaseInvalid(
            categories,
            messages,
            misplaced_wells=inspection.misplaced_wells,
            cycles=cycles,
        )

    # --- stage 2: spatial QC (its own gates raise 422) ---------------------
    try:
        qc = spatial_qc(req)
    except SpatialQCInvalid as exc:
        raise ReleaseInvalid(exc.categories, exc.messages) from exc

    # --- stage 3: acceptance gates -----------------------------------------
    observed_effect = qc["effects"]["max_abs_row_col_effect"]
    p_value = qc["permutation_test"]["p_value"]

    gates = []
    effect_passed = observed_effect <= req.max_spatial_effect
    gates.append({
        "gate": "max_abs_row_col_effect",
        "passed": effect_passed,
        "limit": req.max_spatial_effect,
        "observed": observed_effect,
        "rule": "observed max |row/column effect| must be <= limit",
    })

    reasons = []
    if not effect_passed:
        reasons.append(
            f"max absolute row/column effect {observed_effect:g} exceeds the "
            f"spatial effect limit {req.max_spatial_effect:g}"
        )

    if req.permutations == 0 or p_value is None:
        p_passed = False
        gates.append({
            "gate": "permutation_p_value",
            "passed": False,
            "threshold": req.min_p_value,
            "observed": None,
            "rule": "permutation p-value must be >= threshold",
        })
        reasons.append(
            "permutation test was skipped (permutations=0), so the minimum "
            "p-value gate cannot be satisfied"
        )
    else:
        p_passed = p_value >= req.min_p_value
        gates.append({
            "gate": "permutation_p_value",
            "passed": p_passed,
            "threshold": req.min_p_value,
            "observed": p_value,
            "rule": "permutation p-value must be >= threshold",
        })
        if not p_passed:
            reasons.append(
                f"permutation p-value {p_value:g} is below the minimum "
                f"required p-value {req.min_p_value:g}"
            )

    converged = qc["convergence"]["converged"]
    if not converged:
        # Defensive: spatial_qc already converts non-convergence to 422.
        reasons.append("median polish did not converge")

    all_passed = effect_passed and p_passed and converged

    if not all_passed:
        # Non-released verdict: no scan/QC summary, only the reasons and the
        # gate observations needed to justify the verdict.
        return {
            "status": NOT_RELEASED,
            "release": {
                "released": False,
                "gates": gates,
                "reasons": reasons,
            },
        }

    return {
        "status": RELEASED,
        "release": {
            "released": True,
            "gates": gates,
            "reasons": [],
        },
        "scan": _scan_summary(inspection),
        "spatial_qc": qc,
    }
