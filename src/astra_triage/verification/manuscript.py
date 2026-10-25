"""Layer one, part A: the arithmetic inside the article's own tables.

Every check here recomputes a printed number from other printed numbers. None reads a
record, so every one of them can be decided today, and a failure is a statement about
the manuscript rather than about the code.

The checks fall into three groups. Internal identities tie one column to another:
predictive values to sensitivity, specificity and prevalence, a change column to the
row it is measured against, a subgroup difference to the two areas it subtracts. Count
identities tie a percentage to its numerator and denominator. Interval identities
compare a printed interval with the closed-form standard error that the printed counts
support, which is the only part of the table a reader cannot recompute without the
records.
"""

from __future__ import annotations

import math

from astra_triage.metrics.discrimination import hanley_mcneil_variance
from astra_triage.metrics.heterogeneity import from_areas
from astra_triage.metrics.net_benefit import net_benefit
from astra_triage.metrics.operating import npv_from, ppv_from
from astra_triage.verification.checks import Check, verdict
from astra_triage.verification.transcription import (
    DERIVATION_EPISODES,
    DERIVATION_EVENTS,
    DERIVATION_PATIENTS,
    ECE_REDUCTION_PERCENT,
    EXTERNAL_EPISODES,
    EXTERNAL_EVENTS,
    EXTERNAL_PATIENTS,
    HETEROGENEITY_I2,
    INTERACTION_GAIN_HIGH_PP,
    INTERACTION_GAIN_LOW_PP,
    INTERACTION_GAIN_PP,
    NET_BENEFIT_ASTRA,
    NET_BENEFIT_CRP,
    PAIRED_DIFFERENCE_HIGH_PP,
    PAIRED_DIFFERENCE_LOW_PP,
    PAIRED_DIFFERENCE_P,
    PAIRED_DIFFERENCE_PP,
    PROSE_SITE_ADAPTER_COST_PP,
    PROSPECTIVE_OPEN_EPISODES,
    PROSPECTIVE_OPEN_EVENTS,
    PROSPECTIVE_SILENT_EPISODES,
    PROSPECTIVE_SILENT_EVENTS,
    SPARSITY_DROP_ASTRA_PP,
    SPARSITY_DROP_LATE_PP,
    SUPPORT_MINUS_MODEL_HIGH_PP,
    SUPPORT_MINUS_MODEL_LOW_PP,
    SUPPORT_MINUS_MODEL_PP,
    SUPPORT_MINUS_UNAIDED_HIGH_PP,
    SUPPORT_MINUS_UNAIDED_LOW_PP,
    SUPPORT_MINUS_UNAIDED_PP,
    TABLE2,
    TABLE3,
    TABLE4A,
    TABLE4B,
    TABLE5,
    TABLE6A,
    TABLE6B,
    TABLE7,
    TABLE9,
    TEMPORAL_EPISODES,
    TEMPORAL_EVENTS,
    TEMPORAL_PATIENTS,
    ModelRow,
)

PERCENT_TOLERANCE: float = 0.05
AREA_TOLERANCE: float = 0.0006
IDENTITY_TOLERANCE: float = 0.002
KAPPA_LOCATION: str = "Table 7 footnote"


def _model(name: str) -> ModelRow:
    for row in TABLE2:
        if row.name == name:
            return row
    raise KeyError(name)


def _prevalence() -> float:
    return TEMPORAL_EVENTS / TEMPORAL_EPISODES


def checks() -> list[Check]:
    rows: list[Check] = []
    rows.extend(_count_identities())
    rows.extend(_table2_identities())
    rows.extend(_table3_identities())
    rows.extend(_table4_identities())
    rows.extend(_table5_identities())
    rows.extend(_table6_identities())
    rows.extend(_table7_identities())
    rows.extend(_table9_identities())
    rows.extend(_abstract_identities())
    rows.extend(_prose_identities())
    return rows


def _count_identities() -> list[Check]:
    derivation = 100.0 * DERIVATION_EVENTS / DERIVATION_EPISODES
    temporal = 100.0 * TEMPORAL_EVENTS / TEMPORAL_EPISODES
    external = 100.0 * EXTERNAL_EVENTS / EXTERNAL_EPISODES
    return [
        Check(
            "M01",
            "manuscript",
            "the derivation event share is the ratio of its event count to its episode count",
            "Table 1, row 'Episodes reaching the primary endpoint'",
            verdict(derivation, 12.5, PERCENT_TOLERANCE),
            observed=round(derivation, 4),
            expected=12.5,
            tolerance=PERCENT_TOLERANCE,
        ),
        Check(
            "M02",
            "manuscript",
            "the temporal event share is the ratio of its event count to its episode count",
            "Table 1, row 'Episodes reaching the primary endpoint'",
            verdict(temporal, 12.9, PERCENT_TOLERANCE),
            observed=round(temporal, 4),
            expected=12.9,
            tolerance=PERCENT_TOLERANCE,
        ),
        Check(
            "M03",
            "manuscript",
            "the external event share is the ratio of its event count to its episode count",
            "Table 1, row 'Episodes reaching the primary endpoint'",
            verdict(external, 11.3, PERCENT_TOLERANCE),
            observed=round(external, 4),
            expected=11.3,
            tolerance=PERCENT_TOLERANCE,
        ),
        Check(
            "M04",
            "manuscript",
            "no cohort holds more patients than episodes",
            "Table 1, rows 'Episodes' and 'Patients'",
            (
                "PASS"
                if DERIVATION_PATIENTS <= DERIVATION_EPISODES
                and TEMPORAL_PATIENTS <= TEMPORAL_EPISODES
                and EXTERNAL_PATIENTS <= EXTERNAL_EPISODES
                else "FAIL"
            ),
            observed={
                "derivation": [DERIVATION_PATIENTS, DERIVATION_EPISODES],
                "temporal": [TEMPORAL_PATIENTS, TEMPORAL_EPISODES],
                "external": [EXTERNAL_PATIENTS, EXTERNAL_EPISODES],
            },
        ),
        Check(
            "M05",
            "manuscript",
            "the per-site counts sum to the derivation cohort and its event count",
            "Table 4 panel A",
            (
                verdict(float(sum(row.episodes for row in TABLE4A)), 3412.0, 0.5)
                if abs(sum(row.events for row in TABLE4A) - DERIVATION_EVENTS) <= 0
                else "FAIL"
            ),
            observed={
                "episodes": sum(row.episodes for row in TABLE4A),
                "events": sum(row.events for row in TABLE4A),
            },
            expected={"episodes": 3412, "events": 428},
            note="both sums must close on Table 1",
        ),
    ]


def _table2_identities() -> list[Check]:
    prevalence = _prevalence()
    rows: list[Check] = []

    ppv_rows = {}
    npv_rows = {}
    for row in TABLE2:
        implied_ppv = ppv_from(row.sensitivity, row.specificity, prevalence)
        implied_npv = npv_from(row.sensitivity, row.specificity, prevalence)
        ppv_rows[row.name] = {
            "printed": row.ppv,
            "implied": round(implied_ppv, 4),
            "gap": round(implied_ppv - row.ppv, 4),
        }
        npv_rows[row.name] = {
            "printed": row.npv,
            "implied": round(implied_npv, 4),
            "gap": round(implied_npv - row.npv, 4),
        }
    worst_ppv = max(ppv_rows.items(), key=lambda item: abs(item[1]["gap"]))
    worst_npv = max(npv_rows.items(), key=lambda item: abs(item[1]["gap"]))
    rows.append(
        Check(
            "M06",
            "manuscript",
            "positive predictive value follows from sensitivity, specificity and prevalence",
            "Table 2, columns PPV and Sensitivity/Specificity at the 0.20 threshold",
            "PASS" if abs(worst_ppv[1]["gap"]) <= IDENTITY_TOLERANCE else "FAIL",
            observed=ppv_rows,
            expected=f"every row within {IDENTITY_TOLERANCE}",
            tolerance=IDENTITY_TOLERANCE,
            note=f"largest gap on the {worst_ppv[0]} row",
        )
    )
    rows.append(
        Check(
            "M07",
            "manuscript",
            "negative predictive value follows from sensitivity, specificity and prevalence",
            "Table 2, columns NPV and Sensitivity/Specificity at the 0.20 threshold",
            "PASS" if abs(worst_npv[1]["gap"]) <= IDENTITY_TOLERANCE else "FAIL",
            observed=npv_rows,
            expected=f"every row within {IDENTITY_TOLERANCE}",
            tolerance=IDENTITY_TOLERANCE,
            note=f"largest gap on the {worst_npv[0]} row",
        )
    )

    width_rows = {}
    for row in TABLE2:
        standard_error = math.sqrt(
            hanley_mcneil_variance(row.auroc, TEMPORAL_EVENTS, TEMPORAL_EPISODES - TEMPORAL_EVENTS)
        )
        printed_half = 0.5 * (row.high - row.low)
        width_rows[row.name] = {
            "printed_half_width": round(printed_half, 5),
            "closed_form_half_width": round(1.959964 * standard_error, 5),
            "ratio": round(printed_half / (1.959964 * standard_error), 3),
        }
    ratios = [value["ratio"] for value in width_rows.values()]
    rows.append(
        Check(
            "M08",
            "manuscript",
            "each printed interval is the size a single-AUROC standard error supports at 79 events",
            "Table 2, bracketed intervals; footnote on the paired bootstrap",
            "PASS" if all(0.55 <= ratio <= 1.7 for ratio in ratios) else "FAIL",
            observed=width_rows,
            expected="printed half-width within 0.55 to 1.7 of the closed-form half-width",
            note=(
                "the Hanley-McNeil variance is a closed form in the area and the two class "
                "counts, so this comparison needs no records; the printed intervals are three "
                "to four times narrower"
            ),
        )
    )

    rows.append(
        Check(
            "M09",
            "manuscript",
            "every precision-recall area exceeds the event prevalence of the evaluation set",
            "Table 2, column AUPRC",
            "PASS" if all(row.auprc > prevalence for row in TABLE2) else "FAIL",
            observed={row.name: row.auprc for row in TABLE2},
            expected=round(prevalence, 4),
        )
    )

    rows.append(
        Check(
            "M10",
            "manuscript",
            "every row's Brier score is at least the square of its calibration error",
            "Table 2, columns Brier and ECE",
            "PASS" if all(row.brier >= row.ece**2 for row in TABLE2) else "FAIL",
            observed={row.name: round(row.brier - row.ece**2, 4) for row in TABLE2},
            expected="non-negative for every row",
        )
    )

    astra = _model("astra")
    late = _model("late_fusion")
    difference = 100.0 * (astra.auroc - late.auroc)
    rows.append(
        Check(
            "M11",
            "manuscript",
            "the paired difference is the difference of the two printed areas",
            "Table 2 footnote and abstract",
            verdict(difference, PAIRED_DIFFERENCE_PP, 0.05),
            observed=round(difference, 4),
            expected=PAIRED_DIFFERENCE_PP,
            tolerance=0.05,
        )
    )
    rows.append(
        Check(
            "M12",
            "manuscript",
            "the printed interval for the paired difference contains the printed difference",
            "Table 2 footnote",
            ("PASS" if PAIRED_DIFFERENCE_LOW_PP <= PAIRED_DIFFERENCE_PP <= PAIRED_DIFFERENCE_HIGH_PP else "FAIL"),
            observed=[PAIRED_DIFFERENCE_LOW_PP, PAIRED_DIFFERENCE_PP, PAIRED_DIFFERENCE_HIGH_PP],
        )
    )
    rows.append(
        Check(
            "M13",
            "manuscript",
            "the reported p-value is consistent with an interval that spans zero",
            "Table 2 footnote, p = 0.71",
            (
                "PASS"
                if PAIRED_DIFFERENCE_LOW_PP < 0.0 < PAIRED_DIFFERENCE_HIGH_PP and PAIRED_DIFFERENCE_P > 0.05
                else "FAIL"
            ),
            observed=PAIRED_DIFFERENCE_P,
            expected="above 0.05 with an interval containing zero",
        )
    )
    return rows


def _table3_identities() -> list[Check]:
    rows: list[Check] = []
    full = TABLE3[0]
    changes = {
        row.name: {
            "printed": row.change_pp,
            "recomputed": round(100.0 * (row.auroc - full.auroc), 3),
        }
        for row in TABLE3[1:]
    }
    worst = max(changes.items(), key=lambda item: abs(item[1]["printed"] - item[1]["recomputed"]))
    rows.append(
        Check(
            "M14",
            "manuscript",
            "each ablation change is the row's area minus the full model's area",
            "Table 3, column 'Change in AUROC vs full, pp'",
            "PASS" if abs(worst[1]["printed"] - worst[1]["recomputed"]) <= 0.06 else "FAIL",
            observed=changes,
            expected="every row within 0.1 percentage points",
            tolerance=0.06,
            note=f"largest gap on the {worst[0]} row",
        )
    )
    without_ultrasound = next(row for row in TABLE3 if row.name == "without_ultrasound")
    without_markers = next(row for row in TABLE3 if row.name == "without_markers")
    bound = next(row for row in TABLE3 if row.name == "additive_bound")
    mean_of_areas = 0.5 * (without_ultrasound.auroc + without_markers.auroc)
    rows.append(
        Check(
            "M15",
            "manuscript",
            "the additive bound equals the average of the two single-stream areas",
            "Table 3, last row and its footnote",
            verdict(bound.auroc, mean_of_areas, AREA_TOLERANCE),
            observed={
                "printed_bound": bound.auroc,
                "mean_of_single_stream_areas": round(mean_of_areas, 4),
            },
            expected="the average of the two single-stream areas",
            tolerance=AREA_TOLERANCE,
            note=(
                "the footnote defines the row as the average of the two scores; the printed "
                "value is the area of an averaged score, which is a different quantity and "
                "need not equal the average of the areas"
            ),
        )
    )
    gain = 100.0 * (full.auroc - bound.auroc)
    rows.append(
        Check(
            "M16",
            "manuscript",
            "the interaction gain is the full area minus the printed bound",
            "Table 3 footnote and Sec. 3.3",
            verdict(gain, INTERACTION_GAIN_PP, 0.05),
            observed=round(gain, 4),
            expected=INTERACTION_GAIN_PP,
            tolerance=0.05,
        )
    )
    rows.append(
        Check(
            "M17",
            "manuscript",
            "the printed interval for the interaction gain contains the gain",
            "Table 3 footnote, 95% CI 0.6 to 4.2",
            "PASS" if INTERACTION_GAIN_LOW_PP <= gain <= INTERACTION_GAIN_HIGH_PP else "FAIL",
            observed=round(gain, 4),
            expected=[INTERACTION_GAIN_LOW_PP, INTERACTION_GAIN_HIGH_PP],
        )
    )
    rows.append(
        Check(
            "M18",
            "manuscript",
            "the full model's calibration error is the one the ablation table repeats",
            "Table 2 row ASTRA, column ECE; Table 3 row 'Full model'",
            verdict(full.ece or 0.0, 0.026, 1e-9),
            observed=full.ece,
            expected=0.026,
        )
    )
    return rows


def _table4_identities() -> list[Check]:
    rows: list[Check] = []
    differences = {row.site: round(100.0 * (row.astra - row.late_fusion), 2) for row in TABLE4A}
    rows.append(
        Check(
            "M19",
            "manuscript",
            "the two smallest sites show the advantage the section reports",
            "Table 4 panel A and Sec. 3.4",
            "PASS" if differences["D"] > 5.0 and differences["E"] > 5.0 else "FAIL",
            observed=differences,
            expected="D and E above five percentage points",
        )
    )
    areas = {
        "astra": [row.astra for row in TABLE4A],
        "late_fusion": [row.late_fusion for row in TABLE4A],
        "logistic": [row.logistic for row in TABLE4A],
    }
    positives = [row.events for row in TABLE4A]
    negatives = [row.episodes - row.events for row in TABLE4A]
    computed = {name: round(from_areas(values, positives, negatives)["i_squared"], 1) for name, values in areas.items()}
    gaps = {name: round(computed[name] - HETEROGENEITY_I2[name], 1) for name in computed}
    rows.append(
        Check(
            "M20",
            "manuscript",
            "the printed I-squared values follow from the printed site areas and counts",
            "Table 4 panel A, heterogeneity sentence",
            "PASS" if all(abs(gap) <= 1.0 for gap in gaps.values()) else "FAIL",
            observed=computed,
            expected=HETEROGENEITY_I2,
            tolerance=1.0,
            note=(
                "DerSimonian-Laird on the logit scale with the closed-form per-site variance; "
                "the article does not state which variance estimator it used"
            ),
        )
    )
    lowest = min(TABLE4B, key=lambda row: row.auroc)
    rows.append(
        Check(
            "M21",
            "manuscript",
            "the lowest held-out-site area is the one obtained with the largest site removed",
            "Table 4 panel B and the sentence after it",
            "PASS" if lowest.site == "A" else "FAIL",
            observed={"lowest_site": lowest.site, "auroc": lowest.auroc},
            expected="Site A",
        )
    )
    widths = {}
    for row in TABLE4B:
        events = (
            152
            if row.site == "A"
            else (104 if row.site == "B" else 83 if row.site == "C" else 56 if row.site == "D" else 33)
        )
        standard_error = math.sqrt(hanley_mcneil_variance(row.auroc, events, row.episodes - events))
        printed_half = 0.5 * (row.high - row.low)
        widths[row.site] = {
            "printed_half_width": round(printed_half, 5),
            "closed_form_half_width": round(1.959964 * standard_error, 5),
            "ratio": round(printed_half / (1.959964 * standard_error), 3),
        }
    rows.append(
        Check(
            "M22",
            "manuscript",
            "each leave-one-site-out interval is the size its own held-out count supports",
            "Table 4 panel B, bracketed intervals",
            "PASS" if all(0.55 <= value["ratio"] <= 1.7 for value in widths.values()) else "FAIL",
            observed=widths,
            expected="printed half-width within 0.55 to 1.7 of the closed-form half-width",
            note=(
                "the printed half-widths sit between 0.0185 and 0.0205 for held-out sites of 264 "
                "to 1,186 episodes; at 33 events the closed form is four times wider"
            ),
        )
    )
    return rows


def _table5_identities() -> list[Check]:
    rows: list[Check] = []
    astra = TABLE5[0]
    internal = _model("astra")
    drop = 100.0 * (internal.auroc - astra.auroc)
    rows.append(
        Check(
            "M23",
            "manuscript",
            "the external drop is about three percentage points from the internal area",
            "Table 5 and Sec. 3.5",
            "PASS" if 2.0 <= drop <= 3.5 else "FAIL",
            observed=round(drop, 3),
            expected="about three percentage points",
        )
    )
    rows.append(
        Check(
            "M24",
            "manuscript",
            "the sensitivity and specificity of the transported model are the printed pair",
            "Table 5 and the sentence '0.788 ... at both sensitivity and specificity'",
            "FAIL" if astra.sensitivity != astra.specificity else "PASS",
            observed={"sensitivity": astra.sensitivity, "specificity": astra.specificity},
            expected="one shared value if the sentence is read literally",
            note="the table prints 0.788 and 0.771, so the sentence is a transcription slip",
        )
    )
    rows.append(
        Check(
            "M25",
            "manuscript",
            "every transported row loses discrimination against its internal counterpart",
            "Table 5 against Table 2",
            (
                "PASS"
                if all(
                    external.auroc < _model(internal_name).auroc
                    for external, internal_name in zip(
                        TABLE5, ("astra", "late_fusion", "grud", "logistic"), strict=True
                    )
                )
                else "FAIL"
            ),
            observed={row.name: row.auroc for row in TABLE5},
        )
    )
    return rows


def _table6_identities() -> list[Check]:
    rows: list[Check] = []
    differences = {
        row.name: {
            "printed": row.difference_pp,
            "recomputed": round(100.0 * (row.astra - row.late_fusion), 2),
        }
        for row in TABLE6A
    }
    worst = max(differences.items(), key=lambda item: abs(item[1]["printed"] - item[1]["recomputed"]))
    rows.append(
        Check(
            "M26",
            "manuscript",
            "each subgroup difference is the constrained area minus the unconstrained area",
            "Table 6 panel A, column 'Difference, pp'",
            "PASS" if abs(worst[1]["printed"] - worst[1]["recomputed"]) <= 0.06 else "FAIL",
            observed=differences,
            tolerance=0.06,
            note=f"largest gap on the {worst[0]} subgroup",
        )
    )
    age_sum = sum(row.episodes for row in TABLE6A[:4])
    rows.append(
        Check(
            "M27",
            "manuscript",
            "the four age groups partition the derivation cohort",
            "Table 6 panel A, rows 'Age ...'",
            verdict(float(age_sum), float(DERIVATION_EPISODES), 0.5),
            observed=age_sum,
            expected=DERIVATION_EPISODES,
        )
    )
    low = TABLE6B[0]
    very_high = TABLE6B[-1]
    late_drop = 100.0 * (low.late_auroc - very_high.late_auroc)
    astra_drop = 100.0 * (low.astra_auroc - very_high.astra_auroc)
    rows.append(
        Check(
            "M28",
            "manuscript",
            "the note reports the discrimination lost between the lowest and highest sparsity bands",
            "Table 6 panel B and the sentence after it",
            (
                "PASS"
                if abs(late_drop - SPARSITY_DROP_LATE_PP) <= 0.1 and abs(astra_drop - SPARSITY_DROP_ASTRA_PP) <= 0.1
                else "FAIL"
            ),
            observed={"late_fusion_drop": round(late_drop, 2), "astra_drop": round(astra_drop, 2)},
            expected={
                "late_fusion_drop": SPARSITY_DROP_LATE_PP,
                "astra_drop": SPARSITY_DROP_ASTRA_PP,
            },
            note="the same panel gives 8.5 and 3.5",
        )
    )
    reduction = 100.0 * (very_high.late_ece - very_high.astra_ece) / very_high.late_ece
    rows.append(
        Check(
            "M29",
            "manuscript",
            "the calibration-error reduction in the sparsest band is the printed share",
            "Abstract and Table 6 panel B",
            verdict(reduction, ECE_REDUCTION_PERCENT, 0.6),
            observed=round(reduction, 2),
            expected=ECE_REDUCTION_PERCENT,
            tolerance=0.6,
        )
    )
    rows.append(
        Check(
            "M30",
            "manuscript",
            "the slope advantage in the sparsest band is the printed 0.91 against 0.71",
            "Abstract and Table 6 panel B",
            "PASS" if very_high.astra_slope == 0.910 and very_high.late_slope == 0.710 else "FAIL",
            observed={"astra": very_high.astra_slope, "late_fusion": very_high.late_slope},
        )
    )
    return rows


def _table7_identities() -> list[Check]:
    rows: list[Check] = []
    alone, model, supported = TABLE7
    supported_gap = 100.0 * (supported.auroc - alone.auroc)
    model_gap = 100.0 * (supported.auroc - model.auroc)
    rows.append(
        Check(
            "M31",
            "manuscript",
            "the supported-minus-unaided gap is the difference of the two printed areas",
            "Table 7 and Sec. 3.7",
            verdict(supported_gap, SUPPORT_MINUS_UNAIDED_PP, 0.06),
            observed=round(supported_gap, 3),
            expected=SUPPORT_MINUS_UNAIDED_PP,
            tolerance=0.06,
        )
    )
    rows.append(
        Check(
            "M32",
            "manuscript",
            "the supported-minus-model gap is the difference of the two printed areas",
            "Table 7 and Sec. 3.7",
            verdict(model_gap, SUPPORT_MINUS_MODEL_PP, 0.06),
            observed=round(model_gap, 3),
            expected=SUPPORT_MINUS_MODEL_PP,
            tolerance=0.06,
        )
    )
    rows.append(
        Check(
            "M33",
            "manuscript",
            "both printed intervals for the reading gaps contain their point estimates",
            "Table 7 footnote",
            (
                "PASS"
                if SUPPORT_MINUS_UNAIDED_LOW_PP <= SUPPORT_MINUS_UNAIDED_PP <= SUPPORT_MINUS_UNAIDED_HIGH_PP
                and SUPPORT_MINUS_MODEL_LOW_PP <= SUPPORT_MINUS_MODEL_PP <= SUPPORT_MINUS_MODEL_HIGH_PP
                else "FAIL"
            ),
            observed={
                "supported_minus_unaided": [
                    SUPPORT_MINUS_UNAIDED_LOW_PP,
                    SUPPORT_MINUS_UNAIDED_PP,
                    SUPPORT_MINUS_UNAIDED_HIGH_PP,
                ],
                "supported_minus_model": [
                    SUPPORT_MINUS_MODEL_LOW_PP,
                    SUPPORT_MINUS_MODEL_PP,
                    SUPPORT_MINUS_MODEL_HIGH_PP,
                ],
            },
        )
    )
    rows.append(
        Check(
            "M34",
            "manuscript",
            "the supported arm gains on both operating columns at once",
            "Table 7, columns Sensitivity and Specificity",
            (
                "PASS"
                if supported.sensitivity > alone.sensitivity and supported.specificity > alone.specificity
                else "FAIL"
            ),
            observed={
                "sensitivity": [alone.sensitivity, supported.sensitivity],
                "specificity": [alone.specificity, supported.specificity],
            },
        )
    )
    return rows


def _table9_identities() -> list[Check]:
    silent = 100.0 * PROSPECTIVE_SILENT_EVENTS / PROSPECTIVE_SILENT_EPISODES
    open_label = 100.0 * PROSPECTIVE_OPEN_EVENTS / PROSPECTIVE_OPEN_EPISODES
    return [
        Check(
            "M35",
            "manuscript",
            "each prospective prevalence is the ratio of its event count to its episode count",
            "Table 9, rows 'Episodes (events)' and 'Endpoint prevalence, %'",
            (
                "PASS"
                if abs(silent - TABLE9[0].prevalence) <= PERCENT_TOLERANCE
                and abs(open_label - TABLE9[1].prevalence) <= PERCENT_TOLERANCE
                else "FAIL"
            ),
            observed={"silent": round(silent, 3), "open_label": round(open_label, 3)},
            expected={"silent": TABLE9[0].prevalence, "open_label": TABLE9[1].prevalence},
            tolerance=PERCENT_TOLERANCE,
        ),
        Check(
            "M36",
            "manuscript",
            "the two prospective windows hold different children",
            "Table 9 footnote",
            (
                "PASS"
                if PROSPECTIVE_SILENT_EPISODES + PROSPECTIVE_OPEN_EPISODES > 0
                and PROSPECTIVE_SILENT_EPISODES != PROSPECTIVE_OPEN_EPISODES
                else "FAIL"
            ),
            observed=[PROSPECTIVE_SILENT_EPISODES, PROSPECTIVE_OPEN_EPISODES],
            note=(
                "the counts differ, which is consistent with disjoint windows; identity of " "children needs the record"
            ),
        ),
    ]


def _abstract_identities() -> list[Check]:
    rows: list[Check] = []
    prevalence = _prevalence()
    astra = _model("astra")
    rule = _model("crp_rule")
    astra_value = net_benefit(astra.sensitivity, astra.specificity, prevalence, 0.20)
    rule_value = net_benefit(rule.sensitivity, rule.specificity, prevalence, 0.20)
    rows.append(
        Check(
            "M37",
            "manuscript",
            "the model's net benefit at 0.20 follows from its printed operating point",
            "Abstract and Table 2",
            verdict(astra_value, NET_BENEFIT_ASTRA, 0.002),
            observed=round(astra_value, 4),
            expected=NET_BENEFIT_ASTRA,
            tolerance=0.002,
        )
    )
    rows.append(
        Check(
            "M38",
            "manuscript",
            "the single-marker rule's net benefit at 0.20 follows from its printed operating point",
            "Abstract and Table 2 row 'C-reactive protein threshold rule'",
            verdict(rule_value, NET_BENEFIT_CRP, 0.002),
            observed=round(rule_value, 4),
            expected=NET_BENEFIT_CRP,
            tolerance=0.002,
            note="the printed 0.033 would need a sensitivity near 0.79 at this specificity",
        )
    )
    rows.append(
        Check(
            "M39",
            "manuscript",
            "the abstract's headline area and interval are the table's first row",
            "Abstract and Table 2",
            ("PASS" if astra.auroc == 0.871 and astra.low == 0.856 and astra.high == 0.885 else "FAIL"),
            observed=[astra.auroc, astra.low, astra.high],
            expected=[0.871, 0.856, 0.885],
        )
    )
    rows.append(
        Check(
            "M40",
            "manuscript",
            "the two smallest sites carry the stated 5.4 to 5.5 point advantage",
            "Abstract and Table 4 panel A",
            (
                "PASS"
                if abs(round(100.0 * (TABLE4A[4].astra - TABLE4A[4].late_fusion), 1) - 5.4) <= 0.05
                and abs(round(100.0 * (TABLE4A[3].astra - TABLE4A[3].late_fusion), 1) - 5.5) <= 0.05
                else "FAIL"
            ),
            observed={
                "D": round(100.0 * (TABLE4A[3].astra - TABLE4A[3].late_fusion), 2),
                "E": round(100.0 * (TABLE4A[4].astra - TABLE4A[4].late_fusion), 2),
            },
            expected=[5.4, 5.5],
        )
    )
    return rows


def _prose_identities() -> list[Check]:
    rows: list[Check] = []
    cross_attention = next(row for row in TABLE3 if row.name == "without_cross_attention")
    time_encoding = next(row for row in TABLE3 if row.name == "without_time_encoding")
    site_adapter = next(row for row in TABLE3 if row.name == "without_site_adapter")
    prose_consistent = (
        abs(cross_attention.change_pp) > abs(time_encoding.change_pp)
        and abs(site_adapter.change_pp) == PROSE_SITE_ADAPTER_COST_PP
    )
    rows.append(
        Check(
            "M41",
            "manuscript",
            "the ablation prose ranks the three component costs as the table does",
            "Sec. 3.3 against Table 3",
            "PASS" if prose_consistent else "FAIL",
            observed={
                "prose_cross_attention_cost": 2.9,
                "prose_time_encoding_cost": 3.5,
                "prose_site_adapter_cost": PROSE_SITE_ADAPTER_COST_PP,
                "table_site_adapter_cost": abs(site_adapter.change_pp),
            },
            note="the prose gives the site adapter 0.9 where the table gives 1.3",
        )
    )
    return rows
