"""The article's printed tables, transcribed as data.

Every number below is copied from the manuscript and is here only to be re-derived by
``manuscript.py``. Nothing in this module computes a result, and no number in it is
read by the training or evaluation code: the transcription exists so that the
arithmetic inside the tables can be checked against the text that carries it.
"""

from __future__ import annotations

from dataclasses import dataclass

DERIVATION_EPISODES: int = 3412
DERIVATION_PATIENTS: int = 3186
DERIVATION_EVENTS: int = 428
TEMPORAL_EPISODES: int = 612
TEMPORAL_PATIENTS: int = 592
TEMPORAL_EVENTS: int = 79
EXTERNAL_EPISODES: int = 1042
EXTERNAL_PATIENTS: int = 1015
EXTERNAL_EVENTS: int = 118
PROSPECTIVE_SILENT_EPISODES: int = 268
PROSPECTIVE_SILENT_EVENTS: int = 34
PROSPECTIVE_OPEN_EPISODES: int = 141
PROSPECTIVE_OPEN_EVENTS: int = 17
READING_EPISODES: int = 240
READING_CLINICIANS: int = 24


@dataclass(frozen=True)
class ModelRow:
    name: str
    auroc: float
    low: float
    high: float
    auprc: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    brier: float
    ece: float
    slope: float


TABLE2: tuple[ModelRow, ...] = (
    ModelRow("astra", 0.871, 0.856, 0.885, 0.548, 0.802, 0.803, 0.376, 0.965, 0.081, 0.026, 0.960),
    ModelRow("late_fusion", 0.874, 0.859, 0.888, 0.552, 0.814, 0.790, 0.377, 0.965, 0.079, 0.029, 0.940),
    ModelRow("tft", 0.851, 0.834, 0.867, 0.497, 0.789, 0.775, 0.348, 0.960, 0.086, 0.038, 0.910),
    ModelRow("grud", 0.836, 0.818, 0.853, 0.459, 0.771, 0.760, 0.328, 0.955, 0.091, 0.041, 0.890),
    ModelRow("logistic", 0.781, 0.761, 0.801, 0.352, 0.729, 0.702, 0.276, 0.944, 0.101, 0.052, 0.840),
    ModelRow("pews", 0.746, 0.724, 0.768, 0.301, 0.692, 0.679, 0.250, 0.937, 0.108, 0.061, 0.810),
    ModelRow("crp_rule", 0.712, 0.688, 0.736, 0.262, 0.664, 0.645, 0.224, 0.929, 0.114, 0.068, 0.770),
    ModelRow("appendicitis", 0.704, 0.679, 0.729, 0.248, 0.651, 0.638, 0.217, 0.927, 0.116, 0.072, 0.750),
)

PAIRED_DIFFERENCE_PP: float = -0.3
PAIRED_DIFFERENCE_LOW_PP: float = -1.9
PAIRED_DIFFERENCE_HIGH_PP: float = 1.3
PAIRED_DIFFERENCE_P: float = 0.71

NET_BENEFIT_ASTRA: float = 0.060
NET_BENEFIT_CRP: float = 0.033


@dataclass(frozen=True)
class AblationRow:
    name: str
    auroc: float
    change_pp: float
    ece: float | None


TABLE3: tuple[AblationRow, ...] = (
    AblationRow("full", 0.871, 0.0, 0.026),
    AblationRow("without_ultrasound", 0.812, -5.9, 0.041),
    AblationRow("without_markers", 0.795, -7.6, 0.046),
    AblationRow("without_time_encoding", 0.836, -3.5, 0.039),
    AblationRow("without_cross_attention", 0.842, -2.9, 0.036),
    AblationRow("without_missingness_token", 0.853, -1.8, 0.045),
    AblationRow("without_site_adapter", 0.858, -1.3, 0.040),
    AblationRow("additive_bound", 0.847, -2.4, None),
)

INTERACTION_GAIN_PP: float = 2.4
INTERACTION_GAIN_LOW_PP: float = 0.6
INTERACTION_GAIN_HIGH_PP: float = 4.2
PROSE_SITE_ADAPTER_COST_PP: float = 0.9


@dataclass(frozen=True)
class SiteRow:
    site: str
    episodes: int
    events: int
    astra: float
    late_fusion: float
    logistic: float


TABLE4A: tuple[SiteRow, ...] = (
    SiteRow("A", 1186, 152, 0.884, 0.881, 0.794),
    SiteRow("B", 842, 104, 0.869, 0.872, 0.779),
    SiteRow("C", 671, 83, 0.858, 0.861, 0.771),
    SiteRow("D", 449, 56, 0.841, 0.786, 0.752),
    SiteRow("E", 264, 33, 0.846, 0.792, 0.744),
)

HETEROGENEITY_I2: dict[str, float] = {"astra": 18.4, "late_fusion": 54.7, "logistic": 12.6}


@dataclass(frozen=True)
class LeaveOutRow:
    site: str
    auroc: float
    low: float
    high: float
    episodes: int


TABLE4B: tuple[LeaveOutRow, ...] = (
    LeaveOutRow("A", 0.866, 0.845, 0.886, 1186),
    LeaveOutRow("B", 0.872, 0.852, 0.891, 842),
    LeaveOutRow("C", 0.874, 0.855, 0.892, 671),
    LeaveOutRow("D", 0.873, 0.854, 0.891, 449),
    LeaveOutRow("E", 0.870, 0.851, 0.888, 264),
)


@dataclass(frozen=True)
class ExternalRow:
    name: str
    auroc: float
    low: float
    high: float
    auprc: float
    sensitivity: float
    specificity: float
    brier: float
    ece: float
    slope: float


TABLE5: tuple[ExternalRow, ...] = (
    ExternalRow("astra_laboratory", 0.842, 0.816, 0.867, 0.402, 0.788, 0.771, 0.089, 0.034, 0.930),
    ExternalRow("late_fusion", 0.826, 0.798, 0.852, 0.371, 0.766, 0.752, 0.094, 0.042, 0.880),
    ExternalRow("grud", 0.804, 0.775, 0.832, 0.328, 0.742, 0.731, 0.099, 0.047, 0.850),
    ExternalRow("logistic", 0.762, 0.731, 0.792, 0.276, 0.708, 0.697, 0.106, 0.055, 0.800),
)


@dataclass(frozen=True)
class SubgroupRow:
    name: str
    episodes: int
    events: int
    astra: float
    low: float
    high: float
    late_fusion: float
    difference_pp: float


TABLE6A: tuple[SubgroupRow, ...] = (
    SubgroupRow("age_under_1", 764, 96, 0.849, 0.803, 0.895, 0.826, 2.3),
    SubgroupRow("age_1_to_5", 1188, 151, 0.878, 0.855, 0.901, 0.879, -0.1),
    SubgroupRow("age_6_to_12", 982, 121, 0.879, 0.854, 0.904, 0.882, -0.3),
    SubgroupRow("age_13_to_17", 478, 60, 0.868, 0.826, 0.910, 0.872, -0.4),
    SubgroupRow("necrotising_enterocolitis", 635, 118, 0.833, 0.788, 0.878, 0.807, 2.6),
    SubgroupRow("complicated_appendicitis", 1167, 149, 0.889, 0.866, 0.912, 0.891, -0.2),
    SubgroupRow("enterostomy", 846, 108, 0.856, 0.820, 0.892, 0.831, 2.5),
    SubgroupRow("high_sparsity", 692, 87, 0.862, 0.828, 0.896, 0.842, 2.0),
)


@dataclass(frozen=True)
class StratumRow:
    name: str
    astra_auroc: float
    late_auroc: float
    astra_ece: float
    late_ece: float
    astra_slope: float
    late_slope: float


TABLE6B: tuple[StratumRow, ...] = (
    StratumRow("low", 0.884, 0.886, 0.021, 0.024, 0.980, 0.950),
    StratumRow("medium", 0.874, 0.879, 0.026, 0.031, 0.960, 0.920),
    StratumRow("high", 0.862, 0.842, 0.031, 0.068, 0.940, 0.790),
    StratumRow("very_high", 0.849, 0.801, 0.038, 0.094, 0.910, 0.710),
)

SPARSITY_DROP_LATE_PP: float = 7.3
SPARSITY_DROP_ASTRA_PP: float = 2.2
ECE_REDUCTION_PERCENT: float = 60.0


@dataclass(frozen=True)
class ReadingRow:
    condition: str
    auroc: float
    low: float
    high: float
    sensitivity: float
    specificity: float
    minutes: float


TABLE7: tuple[ReadingRow, ...] = (
    ReadingRow("clinician_alone", 0.762, 0.741, 0.783, 0.702, 0.694, 6.8),
    ReadingRow("model_alone", 0.871, 0.856, 0.885, 0.802, 0.803, 0.4),
    ReadingRow("clinician_with_support", 0.894, 0.880, 0.907, 0.834, 0.818, 7.6),
)

SUPPORT_MINUS_UNAIDED_PP: float = 13.2
SUPPORT_MINUS_UNAIDED_LOW_PP: float = 9.6
SUPPORT_MINUS_UNAIDED_HIGH_PP: float = 16.8
SUPPORT_MINUS_MODEL_PP: float = 2.3
SUPPORT_MINUS_MODEL_LOW_PP: float = 0.5
SUPPORT_MINUS_MODEL_HIGH_PP: float = 4.1
READER_KAPPA: float = 0.58


@dataclass(frozen=True)
class DeploymentColumn:
    label: str
    prevalence: float
    auroc: float
    low: float
    high: float
    alerts_per_hundred: float
    reviewed_same_shift: float
    time_to_reintervention: float
    delayed_beyond_48h: float
    exploration_without_infection: float


TABLE9: tuple[DeploymentColumn, ...] = (
    DeploymentColumn("silent", 12.7, 0.858, 0.798, 0.906, 28.3, 58.7, 27.6, 5.9, 3.4),
    DeploymentColumn("open_label", 12.1, 0.861, 0.801, 0.908, 27.9, 74.8, 19.8, 2.4, 3.5),
)

SPARSITY_STRATA_LABELS: tuple[str, ...] = (
    "Low (below 20%)",
    "Medium (20 to 40%)",
    "High (40 to 60%)",
    "Very high (above 60%)",
)
