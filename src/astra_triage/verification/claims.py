"""The claim-to-code map.

Each claim is a sentence the article makes, the symbol in this release that carries it,
and the identifiers of the checks that decide it. A claim's status is not written by
hand: it is read from the checks. A claim whose checks all pass is ``PASS``; a claim
with a failed check is ``FAIL``; a claim that needs the cohort is ``BLOCKED``; a claim
whose mechanism has no check is ``NOT_RUN`` and is a gap in the map rather than a
success.

The map is the artefact that answers the first question a reviewer asks of a code
release: where is this sentence in the code, and what would tell me that it is right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astra_triage.verification.checks import Check


@dataclass(frozen=True)
class Claim:
    identifier: str
    claim: str
    location: str
    kind: str
    symbols: tuple[str, ...]
    verified_by: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self, statuses: dict[str, str]) -> dict[str, Any]:
        verdicts = [statuses.get(check, "NOT_RUN") for check in self.verified_by]
        if "FAIL" in verdicts:
            status = "FAIL"
        elif verdicts and all(value == "PASS" for value in verdicts):
            status = "PASS"
        elif "BLOCKED" in verdicts:
            status = "BLOCKED"
        elif not self.verified_by:
            status = "BLOCKED" if self.kind == "result" else "NOT_RUN"
        else:
            status = "NOT_RUN"
        return {
            "id": self.identifier,
            "claim": self.claim,
            "location": self.location,
            "kind": self.kind,
            "symbols": list(self.symbols),
            "verified_by": list(self.verified_by),
            "status": status,
        }


M = "astra_triage."
MODEL = f"src/{M.replace('.', '/')}model/"
STREAMS = f"src/{M.replace('.', '/')}streams/"
OBJECTIVES = f"src/{M.replace('.', '/')}objectives/"
METRICS = f"src/{M.replace('.', '/')}metrics/"
EVALUATION = f"src/{M.replace('.', '/')}evaluation/"
CONSTRAINTS = f"src/{M.replace('.', '/')}constraints/"
PROPERTIES = f"src/{M.replace('.', '/')}properties/"
COMPARATORS = f"src/{M.replace('.', '/')}comparators/"
OBSERVATION = f"src/{M.replace('.', '/')}observation/"
ENDPOINTS = f"src/{M.replace('.', '/')}endpoints/"
TRAINING = f"src/{M.replace('.', '/')}training/"


def _claim(
    identifier: str,
    claim: str,
    location: str,
    kind: str,
    symbols: tuple[str, ...],
    verified_by: tuple[str, ...] = (),
) -> Claim:
    return Claim(identifier, claim, location, kind, symbols, verified_by)


CLAIMS: tuple[Claim, ...] = (
    _claim(
        "K01",
        "the endpoint is a surgical or percutaneous re-treatment within seven days of the index time",
        "Sec. 2.1",
        "definition",
        (f"{ENDPOINTS}definition.py:label", f"{ENDPOINTS}definition.py:PRIMARY_WINDOW_HOURS"),
        ("E21", "C01"),
    ),
    _claim(
        "K02",
        "a re-intervention after the window is a non-event for triage",
        "Sec. 2.1",
        "definition",
        (f"{ENDPOINTS}definition.py:label",),
        ("E21",),
    ),
    _claim(
        "K03",
        "the complement of an event is an escalated antibiotic course that improves without surgery",
        "Sec. 2.1",
        "definition",
        (f"{ENDPOINTS}definition.py:Outcome", f"{ENDPOINTS}adjudication.py:outcome_from"),
        ("E21",),
    ),
    _claim(
        "K04",
        "two blind adjudicators decide the outcome, with a third when they disagree",
        "Table 8 footnote",
        "method",
        (f"{ENDPOINTS}adjudication.py:adjudicate",),
        ("E21",),
    ),
    _claim(
        "K05",
        "the estimator reads only observations stamped at or before the index time",
        "Eq. (1)",
        "definition",
        (f"{ENDPOINTS}definition.py:index_visible", f"{STREAMS}case.py:Episode.visible"),
        ("E22",),
    ),
    _claim(
        "K06",
        "the model is r = f(U, M, s) in [0, 1] over the two stamped sequences and the static vector",
        "Eq. (1)",
        "method",
        (f"{MODEL}astra.py:Astra.forward",),
        ("E01",),
    ),
    _claim(
        "K07",
        "the estimator is count-blind: no count or monotone function of a count enters the model",
        "Sec. 2.3.0.1",
        "method",
        (f"{CONSTRAINTS}count_blind.py:audit", f"{STREAMS}case.py:_gaps"),
        ("E02", "E03"),
    ),
    _claim(
        "K08",
        "the estimator is approximately unchanged by the position of a reading",
        "Sec. 2.3.0.1",
        "method",
        (f"{CONSTRAINTS}count_blind.py:permutation_deviation",),
        ("E02",),
    ),
    _claim(
        "K09",
        "a learned site probe on the merged representation is the instrument for the position test",
        "Sec. 2.3.0.1",
        "method",
        (f"{CONSTRAINTS}site_probe.py:probe", f"{CONSTRAINTS}admissibility.py:assess"),
        ("E29",),
    ),
    _claim(
        "K10",
        "the elapsed time since the previous reading of a stream is an input, not noise",
        "Sec. 2.3.0.1",
        "method",
        (f"{STREAMS}case.py:_gaps", f"{MODEL}time_encoding.py:TimeEncoder.features"),
        ("E02",),
    ),
    _claim(
        "K11",
        "the absence of a reading is represented explicitly",
        "Sec. 2.3.0.1",
        "method",
        (f"{MODEL}missingness.py:MissingnessTokens",),
        ("E04", "E05"),
    ),
    _claim(
        "K12",
        "cross-stream attention lets each ultrasound reading query the marker sequence and the reverse",
        "Eq. (2)",
        "method",
        (f"{MODEL}cross_attention.py:CrossStreamAttention.forward",),
        ("C04", "E01"),
    ),
    _claim(
        "K13",
        "the two attention branches are concatenated into Z",
        "Eq. (2)",
        "method",
        (f"{MODEL}gating.py:StreamGate.forward",),
        ("E01",),
    ),
    _claim(
        "K14",
        "a gated pooling layer weights recency of observations and reliability of streams",
        "Sec. 2.3.0.2",
        "method",
        (f"{MODEL}gating.py:RecencyPooling.forward", f"{MODEL}gating.py:StreamGate"),
        ("E02",),
    ),
    _claim(
        "K15",
        "the pooled vector is concatenated with the static embedding and scored by a two-layer head",
        "Sec. 2.3.0.2",
        "method",
        (f"{MODEL}head.py:TriageHead", f"{MODEL}head.py:StaticEncoder"),
        ("E01",),
    ),
    _claim(
        "K16",
        "the static vector is age at surgery, procedure type and site",
        "Eq. (1), Sec. 2.3.0.1",
        "definition",
        (f"{STREAMS}spec.py:STATIC_CONTINUOUS", f"{STREAMS}spec.py:PROCEDURES"),
        ("E01",),
    ),
    _claim(
        "K17",
        "a learned missingness embedding is added to the newest reading when it is older than 24 hours",
        "Sec. 2.3.0.3",
        "method",
        (f"{MODEL}missingness.py:MissingnessTokens.forward",),
        ("E05",),
    ),
    _claim(
        "K18",
        "a structurally absent stream carries its own token rather than a stale one",
        "Table 1 footnote",
        "method",
        (f"{MODEL}missingness.py:MissingnessTokens.absent_embedding",),
        ("E04",),
    ),
    _claim(
        "K19",
        "a decay function sets the importance of a reading from its age",
        "Sec. 2.3.0.3",
        "method",
        (f"{OBSERVATION}decay.py:decay_weight", f"{MODEL}gating.py:RecencyPooling"),
        ("E02",),
    ),
    _claim(
        "K20",
        "a site adapter accounts for reporting practice and needs only the site of origin at inference",
        "Sec. 2.3.0.3",
        "method",
        (f"{MODEL}site_adapter.py:SiteAdapter",),
        ("C04", "E26"),
    ),
    _claim(
        "K21",
        "a gradient reversal term prevents the merged representation from encoding the site",
        "Sec. 2.3.0.3",
        "method",
        (f"{OBJECTIVES}site_loss.py:reverse_gradient", f"{MODEL}astra.py:Astra.forward"),
        ("E24", "E26"),
    ),
    _claim(
        "K22",
        "the objective is the class-weighted cross-entropy plus two weighted penalties",
        "Eq. (3)",
        "method",
        (f"{OBJECTIVES}composite.py:composite_loss",),
        ("E06", "E10"),
    ),
    _claim(
        "K23",
        "class weights follow the effective-number principle",
        "Sec. 2.3.0.4",
        "method",
        (f"{OBJECTIVES}effective_number.py:class_weights",),
        ("E10",),
    ),
    _claim(
        "K24",
        "the calibration penalty is the squared deviation of bin means from bin event rates, per batch",
        "Sec. 2.3.0.4",
        "method",
        (f"{OBJECTIVES}calibration.py:decile_penalty",),
        ("E11",),
    ),
    _claim(
        "K25",
        "the two penalty weights are fitted on the derivation cohort alone",
        "Sec. 2.3.0.4",
        "method",
        (f"{TRAINING}optim.py:TrainConfig",),
        ("C03",),
    ),
    _claim(
        "K26",
        "the two observation streams are the ultrasound descriptors and the blood markers",
        "Sec. 1",
        "definition",
        (f"{STREAMS}spec.py:ULTRASOUND_FIELDS", f"{STREAMS}spec.py:MARKER_FIELDS"),
        ("E01",),
    ),
    _claim(
        "K27",
        "the marker panel carries ratios derived from the reported counts",
        "Sec. 1",
        "definition",
        (f"{STREAMS}spec.py:derive_marker_ratios",),
        ("E01",),
    ),
    _claim(
        "K28",
        "the external cohort records no ultrasound stream, so that stream is absent rather than missing",
        "Table 1 footnote",
        "method",
        (f"{MODEL}missingness.py:MissingnessTokens.absent_embedding",),
        ("E04",),
    ),
    _claim(
        "K29",
        "the comparison table holds eight arms scored on identical records",
        "Table 2",
        "method",
        (f"{COMPARATORS}registry.py:build",),
        ("E28", "E01"),
    ),
    _claim(
        "K30",
        "the unconstrained baseline is a late-fusion transformer that may read the sampling counts",
        "Table 2; Sec. 3.2",
        "method",
        (
            f"{COMPARATORS}late_fusion.py:LateFusionTransformer",
            f"{COMPARATORS}late_fusion.py:_LateFusionNet._counts",
        ),
        ("E01", "E03"),
    ),
    _claim(
        "K31",
        "the sequence baselines are a temporal fusion transformer and a decaying recurrent network",
        "Table 2",
        "method",
        (f"{COMPARATORS}tft.py:TFT", f"{COMPARATORS}grud.py:GRUD"),
        ("E28",),
    ),
    _claim(
        "K32",
        "the last-value logistic arm discards the age of each reading",
        "Table 2; Sec. 1",
        "method",
        (f"{COMPARATORS}last_value.py:last_value_matrix",),
        ("E28",),
    ),
    _claim(
        "K33",
        "the two clinical rules are implemented as instruments and map to a probability through a fitted link",
        "Table 2",
        "method",
        (
            f"{COMPARATORS}pews.py:PaediatricEarlyWarning",
            f"{COMPARATORS}pas.py:PaediatricAppendicitisScore",
        ),
        ("E28",),
    ),
    _claim(
        "K34",
        "the single-marker rule reads the last C-reactive protein and nothing else",
        "Table 2; abstract",
        "method",
        (f"{COMPARATORS}crp_rule.py:last_crp",),
        ("E17",),
    ),
    _claim(
        "K35",
        "AUROC is computed on mid-ranks with the DeLong variance",
        "Table 2",
        "method",
        (f"{METRICS}discrimination.py:auroc", f"{METRICS}discrimination.py:auroc_variance"),
        ("E14", "E16"),
    ),
    _claim(
        "K36",
        "AUPRC is the step-wise average precision",
        "Table 2",
        "method",
        (f"{METRICS}discrimination.py:auprc",),
        ("E15",),
    ),
    _claim(
        "K37",
        "sensitivity, specificity and the predictive values are read at the pre-specified 0.20 threshold",
        "Table 2 footnote",
        "method",
        (f"{METRICS}operating.py:at_threshold", f"{METRICS}operating.py:ppv_from"),
        ("E17", "E18"),
    ),
    _claim(
        "K38",
        "calibration is reported as the expected calibration error, the Brier score and the recalibration slope",
        "Table 2",
        "method",
        (
            f"{METRICS}calibration.py:expected_calibration_error",
            f"{METRICS}calibration.py:calibration_slope",
        ),
        ("E12", "E13", "E19"),
    ),
    _claim(
        "K39",
        "differences are paired bootstrap estimates over 2,000 resamples",
        "Table 2 footnote",
        "method",
        (f"{METRICS}bootstrap.py:paired_difference",),
        ("E16",),
    ),
    _claim(
        "K40",
        "between-site heterogeneity is I-squared on site-specific logit areas",
        "Table 4",
        "method",
        (f"{METRICS}heterogeneity.py:from_areas",),
        ("M20",),
    ),
    _claim(
        "K41",
        "the sparsity strata are defined by the fraction of unobserved stream-days",
        "Table 6 panel B",
        "definition",
        (f"{OBSERVATION}sparsity.py:day_absence", f"{OBSERVATION}sparsity.py:stratum"),
        ("E23",),
    ),
    _claim(
        "K42",
        "the pre-specified subgroups are the four age bands and three recorded backgrounds",
        "Table 6 panel A",
        "definition",
        ("src/astra_triage/cohort/strata.py:STRATA",),
        ("E34",),
    ),
    _claim(
        "K43",
        "the high-sparsity stratum is the pooled unobserved share at or above 0.40",
        "Table 6 panel A",
        "definition",
        ("src/astra_triage/cohort/strata.py:HIGH_SPARSITY_EDGE",),
        ("E23",),
    ),
    _claim(
        "K44",
        "the reading study has three arms and is summarised per reader",
        "Table 7",
        "method",
        (f"{EVALUATION}reading.py:summarise",),
        ("E31",),
    ),
    _claim(
        "K45",
        "net benefit is reported at the 0.20 threshold with the treat-all reference",
        "Abstract",
        "method",
        (f"{METRICS}net_benefit.py:net_benefit", f"{EVALUATION}decision.py:table"),
        ("E17", "M37"),
    ),
    _claim(
        "K46",
        "the ablation removes a component at its input, not by zeroing a learned tensor",
        "Table 3",
        "method",
        (f"{EVALUATION}ablation.py:drop_stream",),
        ("E04",),
    ),
    _claim(
        "K47",
        "the additive bound is the rank average of the two single-stream scores",
        "Table 3 last row",
        "method",
        (f"{EVALUATION}ablation.py:additive_bound",),
        ("M15",),
    ),
    _claim(
        "K48",
        "leave-one-site-out refits on the remaining sites and scores the held-out one",
        "Table 4 panel B",
        "method",
        (f"{EVALUATION}sites.py:leave_one_site_out",),
        ("E30",),
    ),
    _claim(
        "K49",
        "the two prospective windows are consecutive and cover different children",
        "Table 9",
        "method",
        (f"{EVALUATION}deployment.py:DeploymentRecord",),
        ("M36",),
    ),
    _claim(
        "K50",
        "the appendix property: the last-observation estimator is biased when sampling is informative",
        "Sec. 2.3.0.1",
        "method",
        (f"{PROPERTIES}locf_bias.py:gap_information",),
        ("E32",),
    ),
    _claim(
        "K51",
        "the appendix property: the counting estimator is calibrated in the cohort and biased outside it",
        "Sec. 2.3.0.1",
        "method",
        (f"{PROPERTIES}counting_estimator.py:selection_bias",),
        ("E32",),
    ),
    _claim(
        "K52",
        "the appendix property: an unconstrained estimator cannot do worse than an admissible one",
        "Sec. 2.3.0.1",
        "method",
        (f"{PROPERTIES}dominance.py:dominance",),
        ("E32",),
    ),
    _claim(
        "K53",
        "the sampling process is modelled as state-dependent intensity",
        "Sec. 1",
        "method",
        (f"{OBSERVATION}process.py:SamplingProcess",),
        ("E32",),
    ),
    _claim(
        "K54",
        "the training schedule is not reported in the article and is released as a stated default",
        "Sec. 2.4",
        "method",
        (f"{TRAINING}optim.py:TrainConfig",),
        ("C02", "C03"),
    ),
    _claim(
        "K55",
        "the record is read from a caller-supplied table; no episode ships with the release",
        "Data statement",
        "method",
        ("src/astra_triage/cohort/reader.py:read_episodes",),
        ("E33",),
    ),
    _claim(
        "K56",
        "a patient never straddles two splits",
        "Table 1",
        "method",
        ("src/astra_triage/cohort/split.py:assert_patient_disjoint",),
        ("E33",),
    ),
    _claim(
        "K57",
        "the derivation cohort holds 3,412 episodes from five sites",
        "Table 1",
        "result",
        ("src/astra_triage/cohort/split.py:site_distribution",),
        ("R01",),
    ),
    _claim(
        "K58",
        "the model reaches an AUROC of 0.871 on the temporal hold-out",
        "Table 2",
        "result",
        (f"{EVALUATION}arms.py:evaluate",),
        ("R04",),
    ),
    _claim(
        "K59",
        "the paired difference against late fusion is -0.3 percentage points",
        "Table 2 footnote",
        "result",
        (f"{EVALUATION}arms.py:paired_delta",),
        ("R05", "M11"),
    ),
    _claim(
        "K60",
        "the constrained model's calibration error is 0.026",
        "Table 2",
        "result",
        (f"{METRICS}calibration.py:expected_calibration_error",),
        ("R04",),
    ),
    _claim(
        "K61",
        "removing the ultrasound stream costs 5.9 points",
        "Table 3",
        "result",
        (f"{EVALUATION}ablation.py:drop_stream",),
        ("R06", "M14"),
    ),
    _claim(
        "K62",
        "the additive bound of the two single-stream models is 0.847",
        "Table 3",
        "result",
        (f"{EVALUATION}ablation.py:additive_bound",),
        ("R07", "M15"),
    ),
    _claim(
        "K63",
        "the interaction gain is 2.4 percentage points",
        "Table 3",
        "result",
        (f"{EVALUATION}ablation.py:interaction_gain",),
        ("R08", "M16"),
    ),
    _claim(
        "K64",
        "the two smallest sites show a 5.4 to 5.5 point advantage",
        "Table 4; abstract",
        "result",
        (f"{EVALUATION}sites.py:per_site",),
        ("R09", "M40"),
    ),
    _claim(
        "K65",
        "the held-out-site area is 0.866 when the largest site is removed",
        "Table 4 panel B",
        "result",
        (f"{EVALUATION}sites.py:leave_one_site_out",),
        ("R10", "M21"),
    ),
    _claim(
        "K66",
        "the transported model reaches 0.842 on the external resource",
        "Table 5",
        "result",
        (f"{EVALUATION}arms.py:evaluate",),
        ("R11", "M23"),
    ),
    _claim(
        "K67",
        "the external calibration slope is 0.93",
        "Table 5",
        "result",
        (f"{METRICS}calibration.py:calibration_slope",),
        ("R11",),
    ),
    _claim(
        "K68",
        "infants under one year show a 2.3 point advantage",
        "Table 6",
        "result",
        (f"{EVALUATION}subgroups.py:subgroup_table",),
        ("R12", "M26"),
    ),
    _claim(
        "K69",
        "the very-high-sparsity band shows 0.038 against 0.094 calibration error",
        "Table 6; abstract",
        "result",
        (f"{EVALUATION}subgroups.py:sparsity_table",),
        ("R13", "M29"),
    ),
    _claim(
        "K70",
        "model support lifts clinician AUROC from 0.762 to 0.894",
        "Table 7",
        "result",
        (f"{EVALUATION}reading.py:summarise",),
        ("R14", "M31"),
    ),
    _claim(
        "K71",
        "inter-reader agreement in the unaided arm is 0.58",
        "Table 7 footnote",
        "result",
        (f"{METRICS}agreement.py:cohen_kappa",),
        ("R15", "E20"),
    ),
    _claim(
        "K72",
        "the silent window discriminates at 0.858 and the open-label window at 0.861",
        "Table 9",
        "result",
        (f"{EVALUATION}deployment.py:DeploymentPhase.discrimination",),
        ("R17",),
    ),
    _claim(
        "K73",
        "median time to re-intervention falls from 27.6 to 19.8 hours",
        "Table 9",
        "result",
        (f"{EVALUATION}deployment.py:DeploymentRecord.table",),
        ("R17",),
    ),
    _claim(
        "K74",
        "net benefit at 0.20 is 0.060 per episode",
        "Abstract",
        "result",
        (f"{METRICS}net_benefit.py:net_benefit",),
        ("M37",),
    ),
    _claim(
        "K75",
        "net benefit of the single-marker rule at 0.20 is 0.033",
        "Abstract",
        "result",
        (f"{METRICS}net_benefit.py:net_benefit",),
        ("M38",),
    ),
    _claim(
        "K76",
        "the eight adjudicated cases span every direction of disagreement",
        "Table 8",
        "result",
        (f"{ENDPOINTS}definition.py:band",),
        ("R16",),
    ),
    _claim(
        "K77",
        "the alert instrument cannot act on the record by itself",
        "Sec. 3.9",
        "method",
        (f"{EVALUATION}deployment.py:DeploymentPhase",),
        ("R17",),
    ),
    _claim(
        "K78",
        "the threshold is fixed before the prospective windows and not re-assessed",
        "Sec. 3.9",
        "method",
        (f"{ENDPOINTS}definition.py:TRIAGE_THRESHOLD",),
        ("C01", "R18"),
    ),
    _claim(
        "K79",
        "the release ships no episode and every command that needs records takes a path",
        "Data statement",
        "method",
        ("src/astra_triage/cli/train.py:main",),
        ("H10",),
    ),
    _claim(
        "K80",
        "the configuration carries no value equal to a reported metric",
        "Release contract",
        "method",
        ("src/astra_triage/verification/conformance.py:checks",),
        ("C06",),
    ),
)


def build(checks: list[Check]) -> list[dict[str, Any]]:
    statuses = {check.identifier: check.status for check in checks}
    return [claim.as_dict(statuses) for claim in CLAIMS]


def uncovered(entries: list[dict[str, Any]]) -> list[str]:
    return [entry["id"] for entry in entries if entry["status"] == "NOT_RUN"]
