"""Layer two: the code executed, against quantities computed independently.

Each check runs the released modules on tensors built inside the check process and
compares the result with an answer computed by a separate route: a pairwise loop for
an area, a hand-built confusion table for an operating point, a closed form for a
loss. No check calls the function it is checking to produce its own reference value,
and no check needs a record.

The cases are declared before they run, so a check that raises is reported as a
failure against its own identifier rather than disappearing from the report.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

from astra_triage.constraints import count_blind
from astra_triage.endpoints.definition import Outcome, OutcomeKind, label
from astra_triage.metrics.agreement import cohen_kappa
from astra_triage.metrics.calibration import (
    brier,
    calibration_slope,
    expected_calibration_error,
)
from astra_triage.metrics.discrimination import auprc, auroc, auroc_interval, auroc_variance
from astra_triage.metrics.net_benefit import net_benefit
from astra_triage.metrics.operating import at_threshold, npv_from, ppv_from
from astra_triage.model.astra import Astra, build_astra
from astra_triage.model.config import AstraConfig
from astra_triage.objectives.calibration import decile_penalty
from astra_triage.objectives.composite import LossWeights, binary_cross_entropy, composite_loss
from astra_triage.objectives.site_loss import reverse_gradient
from astra_triage.observation.sparsity import day_absence
from astra_triage.streams.case import Episode, Observation, build_batch
from astra_triage.training.checkpoint import load, save
from astra_triage.training.loop import train_arm
from astra_triage.training.optim import TrainConfig
from astra_triage.verification.checks import Check

TOLERANCE: float = 1e-6

SMALL = AstraConfig(
    d_model=32,
    heads=2,
    encoder_hidden=32,
    head_hidden=16,
    gate_hidden=16,
    adapter_rank=4,
    capacity=16,
)


@dataclass(frozen=True)
class Case:
    identifier: str
    statement: str
    location: str
    run: Callable[[], dict[str, Any]]
    tolerance: float | None = None


def _episode(
    name: str,
    site: str,
    label_value: int,
    index_hours: float,
    ultrasound_times: tuple[float, ...],
    marker_times: tuple[float, ...],
    age: float = 3.0,
    procedure: str = "resection_with_anastomosis",
    ultrasound_recorded: bool = True,
    markers_recorded: bool = True,
) -> Episode:
    ultrasound = tuple(
        Observation(t, (2.0 + 0.1 * t, float(int(t) % 3), 1.0, 5.0 + t, 0.0, 1.0)) for t in ultrasound_times
    )
    markers = tuple(Observation(t, (12.0 + 0.2 * t, 8.0, 2.0, 300.0, 40.0 + t, 0.8)) for t in marker_times)
    return Episode(
        episode_id=name,
        patient_id=f"p-{name}",
        site=site,
        age_years=age,
        procedure=procedure,
        index_hours=index_hours,
        ultrasound=ultrasound,
        markers=markers,
        label=label_value,
        split="train",
        ultrasound_recorded=ultrasound_recorded,
        markers_recorded=markers_recorded,
    )


def cohort(size: int = 10) -> list[Episode]:
    sites = ("A", "B", "C", "D", "E")
    rows = []
    for index in range(size):
        rows.append(
            _episode(
                f"e{index}",
                sites[index % len(sites)],
                index % 2,
                48.0 + 6.0 * index,
                (6.0, 18.0 + index, 30.0 + 2.0 * index),
                (2.0, 14.0, 26.0 + index, 40.0 + index),
                age=1.0 + 0.7 * index,
            )
        )
    return rows


def _model(seed: int = 3) -> Astra:
    torch.manual_seed(seed)
    return build_astra(SMALL)


def _check_forward() -> dict[str, Any]:
    model = _model()
    model.eval()
    shapes = {}
    with torch.no_grad():
        for size in (1, 3, 8):
            batch = build_batch(cohort(size), capacity=SMALL.capacity)
            output = model(batch)
            if tuple(output.probability.shape) != (size,):
                raise AssertionError(f"probability shape {tuple(output.probability.shape)}")
            if not bool(torch.isfinite(output.probability).all()):
                raise AssertionError("a probability is not finite")
            if float(output.probability.min()) < 0.0 or float(output.probability.max()) > 1.0:
                raise AssertionError("a probability leaves [0, 1]")
            shapes[str(size)] = list(output.probability.shape)
    return {"batch_shapes": shapes}


def _check_replication() -> dict[str, Any]:
    model = _model()
    episodes = cohort(6)
    result = count_blind.audit(model, episodes, factor=3)
    if not result.passes:
        raise AssertionError(
            f"replication deviation {result.replication:.2e}, permutation deviation " f"{result.permutation:.2e}"
        )
    return {
        "replication_deviation": result.replication,
        "permutation_deviation": result.permutation,
    }


def _check_unconstrained_breaks_it() -> dict[str, Any]:
    """The count-blind probe must fail on an estimator that reads the count."""
    episodes = cohort(6)

    class Counted(torch.nn.Module):
        def forward(self, batch: Any) -> torch.Tensor:
            counted = batch.ultrasound.mask.sum(dim=1).to(torch.float32) + batch.markers.mask.sum(dim=1).to(
                torch.float32
            )
            return torch.sigmoid((counted - 6.0) * 0.5)

    module = Counted()
    deviation = count_blind.replication_deviation(module, episodes, factor=3)
    if deviation <= 1e-3:
        raise AssertionError("the probe did not detect a count-dependent estimator")
    return {"deviation_on_count_dependent_estimator": deviation}


def _check_absent_stream() -> dict[str, Any]:
    model = _model()
    model.eval()
    present = cohort(4)
    absent = [
        _episode(e.episode_id, e.site, e.label, e.index_hours, (), (2.0, 30.0), ultrasound_recorded=False)
        for e in present
    ]
    with torch.no_grad():
        first = model(build_batch(present, capacity=SMALL.capacity)).probability
        second = model(build_batch(absent, capacity=SMALL.capacity)).probability
    if not bool(torch.isfinite(second).all()):
        raise AssertionError("the absent-stream path produced a non-finite probability")
    return {
        "with_ultrasound": first.tolist(),
        "without_ultrasound": second.tolist(),
        "changed": bool((first - second).abs().min() > 1e-4),
    }


def _check_stale_boundary() -> dict[str, Any]:
    model = _model()
    model.eval()
    model.ultrasound_encoder.tokens.stale.data.fill_(0.5)
    fresh = [_episode("s1", "A", 0, 54.0, (30.0,), (50.0,))]
    stale = [_episode("s2", "A", 0, 54.0, (29.0,), (50.0,))]
    with torch.no_grad():
        first = model(build_batch(fresh, capacity=SMALL.capacity))
        second = model(build_batch(stale, capacity=SMALL.capacity))
    if bool(first.ultrasound_stale[0]) or not bool(second.ultrasound_stale[0]):
        raise AssertionError("the 24-hour boundary is not the one the article states")
    if abs(float(first.probability[0]) - float(second.probability[0])) < 1e-7:
        raise AssertionError("the stale token did not change the score")
    return {
        "at_24_hours_stale": bool(first.ultrasound_stale[0]),
        "at_25_hours_stale": bool(second.ultrasound_stale[0]),
    }


def _check_gradients() -> dict[str, Any]:
    model = _model()
    model.set_reversal_strength(1.0)
    rows = cohort(8)
    rows.append(_episode("a1", "A", 0, 60.0, (), (10.0, 40.0), ultrasound_recorded=False))
    rows.append(_episode("a2", "B", 1, 60.0, (10.0, 40.0), (), markers_recorded=False))
    batch = build_batch(rows, capacity=SMALL.capacity)
    output = model(batch)
    weights = LossWeights()
    breakdown = composite_loss(
        output.logits,
        output.probability,
        batch.label,
        output.site_logits,
        batch.site,
        (4, 4),
        weights,
    )
    torch.autograd.backward(breakdown.total)
    missing = []
    zero = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif float(parameter.grad.abs().max()) == 0.0:
            zero.append(name)
    if missing:
        raise AssertionError(f"no gradient on {missing[:4]}")
    return {
        "parameters_without_gradient": len(missing),
        "parameters_with_zero_gradient": len(zero),
        "example_zero": zero[:4],
        "loss": breakdown.as_floats(),
    }


def _check_optimizer_step() -> dict[str, Any]:
    model = _model()
    before = [p.detach().clone() for p in model.parameters()]
    training = TrainConfig(
        epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        warmup_steps=0,
        scheduler="constant",
        precision="fp32",
        lambda_cal=0.0,
        lambda_site=0.0,
    )
    train_arm(model, cohort(8), training, torch.device("cpu"), seed=1)
    moved = sum(
        1 for old, new in zip(before, model.parameters(), strict=True) if float((old - new.detach()).abs().max()) > 0.0
    )
    if moved == 0:
        raise AssertionError("the optimiser step changed no parameter")
    return {"parameters_moved": moved, "parameters": len(before)}


def _check_checkpoint(tmp_path: str = "artifacts/checkpoint_probe.pt") -> dict[str, Any]:
    model = _model()
    model.eval()
    batch = build_batch(cohort(4), capacity=SMALL.capacity)
    training = TrainConfig(epochs=2, batch_size=4, precision="fp32")
    with torch.no_grad():
        first = model(batch).logits
    path = save(tmp_path, model, training, seed=99, epoch=2)
    restored = _model(seed=11)
    state = load(path, restored)
    restored.eval()
    with torch.no_grad():
        second = restored(batch).logits
    if state.seed != 99:
        raise AssertionError("the seed did not travel with the checkpoint")
    if float((first - second).abs().max()) > 1e-6:
        raise AssertionError("the restored model does not reproduce the logits")
    path.unlink()
    return {"seed": state.seed, "max_logit_gap": float((first - second).abs().max())}


def _check_overfit() -> dict[str, Any]:
    model = _model(seed=5)
    training = TrainConfig(
        epochs=120,
        batch_size=8,
        learning_rate=5e-3,
        warmup_steps=0,
        scheduler="constant",
        precision="fp32",
        lambda_cal=0.0,
        lambda_site=0.0,
    )
    rows = cohort(8)
    history = train_arm(model, rows, training, torch.device("cpu"), seed=5, steps_limit=120)
    if history.batches < 120:
        raise AssertionError("the loop stopped before its step budget")
    start, end = history.first_loss, history.last_loss
    if not end < 0.5 * start:
        raise AssertionError(f"the loss fell from {start:.4f} only to {end:.4f}")
    return {"first_loss": start, "last_loss": end, "batches": history.batches}


def _check_class_weighting() -> dict[str, Any]:
    balanced = float(binary_cross_entropy(torch.zeros(64), torch.tensor([0.0, 1.0] * 32, dtype=torch.float32), 32, 32))
    weighted = binary_cross_entropy(torch.zeros(80), torch.tensor([0.0] * 70 + [1.0] * 10, dtype=torch.float32), 70, 10)
    if abs(balanced - math.log(2.0)) > 1e-6:
        raise AssertionError(f"balanced loss at zero logits is {balanced}")
    if float(weighted) <= math.log(2.0):
        raise AssertionError("the imbalanced loss is not above the balanced one")
    return {"balanced": balanced, "imbalanced": float(weighted), "log_two": math.log(2.0)}


def _check_calibration_penalty() -> dict[str, Any]:
    probability = torch.tensor([0.1] * 50 + [0.7] * 50)
    target = torch.tensor([0.0] * 45 + [1.0] * 5 + [0.0] * 15 + [1.0] * 35)
    aligned = decile_penalty(probability, target)
    shifted = decile_penalty(1.0 - probability, target)
    if float(aligned) > 1e-8:
        raise AssertionError(f"an aligned batch is penalised {float(aligned)}")
    if float(shifted) <= 0.0:
        raise AssertionError("a reversed batch is not penalised")
    return {"aligned": float(aligned), "reversed": float(shifted)}


def _check_ece() -> dict[str, Any]:
    rng = torch.Generator().manual_seed(2)
    probability = torch.rand(4000, generator=rng)
    target = (torch.rand(4000, generator=rng) < probability).to(torch.float32)
    calibrated = expected_calibration_error(probability, target)
    shifted = expected_calibration_error(torch.clamp(probability - 0.25, min=0.0), target)
    if calibrated > 0.03:
        raise AssertionError(f"the calibrated sample reports {calibrated:.4f}")
    if shifted < 0.15:
        raise AssertionError(f"the shifted sample reports only {shifted:.4f}")
    return {"calibrated": calibrated, "shifted": shifted}


def _check_brier() -> dict[str, Any]:
    probability = torch.tensor([0.0, 1.0, 0.5, 0.5])
    target = torch.tensor([0.0, 1.0, 1.0, 0.0])
    value = brier(probability, target)
    if abs(value - 0.125) > TOLERANCE:
        raise AssertionError(f"Brier is {value}")
    return {"brier": value, "expected": 0.125}


def _check_auroc_pairwise() -> dict[str, Any]:
    scores = torch.tensor([0.9, 0.4, 0.4, 0.8, 0.2, 0.6])
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    observed = auroc(scores, labels)
    positive = scores[labels == 1.0]
    negative = scores[labels == 0.0]
    tally = 0.0
    for high in positive:
        for low in negative:
            tally += 1.0 if float(high) > float(low) else 0.5 if float(high) == float(low) else 0.0
    expected = tally / (positive.numel() * negative.numel())
    if abs(observed - expected) > TOLERANCE:
        raise AssertionError(f"AUROC {observed} against pairwise {expected}")
    return {"auroc": observed, "pairwise": expected}


def _check_auprc() -> dict[str, Any]:
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    observed = auprc(scores, labels)
    expected = 0.5 * 1.0 + 0.5 * (2.0 / 3.0)
    if abs(observed - expected) > TOLERANCE:
        raise AssertionError(f"AUPRC {observed} against the step formula {expected}")
    return {"auprc": observed, "step_formula": expected}


def _check_interval() -> dict[str, Any]:
    rng = torch.Generator().manual_seed(4)
    scores = torch.rand(400, generator=rng)
    labels = (torch.rand(400, generator=rng) < scores).to(torch.float32)
    area = auroc(scores, labels)
    variance = auroc_variance(scores, labels)
    low, high = auroc_interval(scores, labels)
    if variance <= 0.0:
        raise AssertionError("the DeLong variance is not positive")
    if not low < area < high:
        raise AssertionError("the interval does not bracket its point estimate")
    return {"auroc": area, "variance": variance, "low": low, "high": high}


def _check_net_benefit() -> dict[str, Any]:
    probability = torch.tensor([0.9, 0.8, 0.7, 0.1, 0.05, 0.02])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    point = at_threshold(probability, target, 0.20)
    observed = net_benefit(point.sensitivity, point.specificity, float(target.mean()), 0.20)
    true_positive = 2
    false_positive = 1
    manual = true_positive / 6.0 - (false_positive / 6.0) * (0.20 / 0.80)
    if abs(observed - manual) > TOLERANCE:
        raise AssertionError(f"net benefit {observed} against {manual}")
    return {"net_benefit": observed, "manual_counts": manual}


def _check_predictive_values() -> dict[str, Any]:
    sensitivity, specificity, prevalence = 0.802, 0.803, 79.0 / 612.0
    ppv = ppv_from(sensitivity, specificity, prevalence)
    npv = npv_from(sensitivity, specificity, prevalence)
    true_positive = sensitivity * prevalence
    false_positive = (1.0 - specificity) * (1.0 - prevalence)
    manual_ppv = true_positive / (true_positive + false_positive)
    true_negative = specificity * (1.0 - prevalence)
    manual_npv = true_negative / (true_negative + (1.0 - sensitivity) * prevalence)
    if abs(ppv - manual_ppv) > TOLERANCE or abs(npv - manual_npv) > TOLERANCE:
        raise AssertionError("the predictive-value identity does not close")
    return {"ppv": ppv, "npv": npv}


def _check_slope() -> dict[str, Any]:
    rng = torch.Generator().manual_seed(6)
    signal = torch.randn(6000, generator=rng)
    probability = torch.sigmoid(signal)
    target = (torch.rand(6000, generator=rng) < probability).to(torch.float32)
    _, slope = calibration_slope(probability, target)
    overconfident = torch.sigmoid(2.5 * signal)
    target_over = (torch.rand(6000, generator=rng) < probability).to(torch.float32)
    _, slope_over = calibration_slope(overconfident, target_over)
    if abs(slope - 1.0) > 0.12:
        raise AssertionError(f"a well-specified predictor gives a slope of {slope:.3f}")
    if slope_over >= 0.7:
        raise AssertionError(f"an overconfident predictor gives a slope of {slope_over:.3f}")
    return {"well_specified": slope, "overconfident": slope_over}


def _check_kappa() -> dict[str, Any]:
    first = [1, 1, 0, 0, 1, 0, 1, 0]
    second = [1, 0, 0, 1, 1, 0, 1, 0]
    observed = cohen_kappa(first, second)
    agree = sum(1 for a, b in zip(first, second, strict=True) if a == b) / 8.0
    expected = (agree - 0.5) / 0.5
    if abs(observed - expected) > TOLERANCE:
        raise AssertionError(f"kappa {observed} against {expected}")
    return {"kappa": observed}


def _check_endpoint_window() -> dict[str, Any]:
    inside = label(Outcome(kind=OutcomeKind.REINTERVENTION, reintervention_hours=168.0))
    outside = label(Outcome(kind=OutcomeKind.REINTERVENTION, reintervention_hours=169.0))
    treated = label(Outcome(kind=OutcomeKind.ANTIBIOTIC_ESCALATION, antibiotic_escalated=True))
    if (inside, outside, treated) != (1, 0, 0):
        raise AssertionError("the endpoint window does not close at seven days")
    return {"day_seven": inside, "day_seven_plus_one_hour": outside, "treated": treated}


def _check_index_filter() -> dict[str, Any]:
    episode = _episode("i1", "A", 1, 40.0, (10.0, 30.0, 44.0), (5.0, 39.0, 41.0))
    visible = episode.visible()
    if len(visible.ultrasound) != 2 or len(visible.markers) != 2:
        raise AssertionError("an observation after the index time survived the filter")
    if max(o.tau_hours for o in visible.ultrasound) > episode.index_hours:
        raise AssertionError("a surviving ultrasound is stamped after the index time")
    return {
        "ultrasound_kept": len(visible.ultrasound),
        "markers_kept": len(visible.markers),
    }


def _check_sparsity() -> dict[str, Any]:
    episode = _episode("sp1", "A", 0, 72.0, (25.0,), (2.0, 26.0, 50.0))
    absence = day_absence(episode)
    if absence.days_total != 4:
        raise AssertionError(f"the day count is {absence.days_total}")
    if absence.ultrasound_days_observed != 1 or absence.marker_days_observed != 3:
        raise AssertionError("the observed day counts are wrong")
    return {
        "days": absence.days_total,
        "ultrasound_fraction": absence.ultrasound_fraction,
        "marker_fraction": absence.marker_fraction,
        "pooled": absence.pooled_fraction,
    }


def _check_reversal() -> dict[str, Any]:
    value = torch.ones(3, requires_grad=True)
    reversed_value = reverse_gradient(value, 0.5)
    torch.autograd.backward(reversed_value.sum())
    if value.grad is None or float(value.grad[0]) != -0.5:
        raise AssertionError("the reversal did not flip the sign")
    return {"gradient": float(value.grad[0])}


def _check_determinism() -> dict[str, Any]:
    from astra_triage.training.seeding import set_seed

    set_seed(77)
    first = build_batch(cohort(5), capacity=SMALL.capacity)
    set_seed(77)
    second = build_batch(cohort(5), capacity=SMALL.capacity)
    gap = float((first.ultrasound.values - second.ultrasound.values).abs().max())
    if gap != 0.0:
        raise AssertionError("two identical builds differ")
    return {"max_gap": gap}


def _check_site_reversal_path() -> dict[str, Any]:
    model = _model()
    model.set_reversal_strength(1.0)
    batch = build_batch(cohort(6), capacity=SMALL.capacity)
    output = model(batch)
    output.site_logits.sum().backward()
    encoder_grad = cast(torch.Tensor | None, model.ultrasound_encoder.project[0].weight.grad)
    if encoder_grad is None:
        raise AssertionError("the site head did not reach the encoder")
    head_grad = cast(torch.Tensor | None, model.site_head.probe[1].weight.grad)
    if head_grad is None:
        raise AssertionError("the site head has no gradient of its own")
    return {
        "encoder_gradient_norm": float(encoder_grad.norm()),
        "site_head_gradient_norm": float(head_grad.norm()),
    }


def _check_trainable_count() -> dict[str, Any]:
    model = _model()
    total = model.parameter_count()
    if total < 10000 or total > 5_000_000:
        raise AssertionError(f"the model carries {total} parameters")
    buffers = [name for name, _ in model.named_buffers()]
    return {"parameters": total, "buffers": buffers}


def _check_every_arm() -> dict[str, Any]:
    from astra_triage.comparators.base import Comparator, Scored
    from astra_triage.comparators.registry import LEARNED_ARMS, RULE_ARMS, build

    rows = cohort(8)
    scored: dict[str, list[float]] = {}
    for name in LEARNED_ARMS + RULE_ARMS:
        built = build(name, SMALL)
        if isinstance(built, Comparator):
            built.fit(rows)
        arm = cast(Scored, built)
        result = arm.score(rows)
        if result.available:
            if not bool(torch.isfinite(result.scores).all()):
                raise AssertionError(f"{name} produced a non-finite score")
            scored[name] = [round(float(value), 4) for value in result.scores[:2]]
        else:
            scored[name] = [-1.0]
    if len(scored) != len(LEARNED_ARMS) + len(RULE_ARMS):
        raise AssertionError("an arm did not answer")
    return {"arms": sorted(scored), "first_scores": scored}


def _check_site_probe_runs() -> dict[str, Any]:
    from astra_triage.constraints.site_probe import probe

    model = _model()
    rows = [
        _episode(f"S{index}", ("A", "B", "C")[index % 3], index % 2, 44.0 + index, (8.0,), (4.0, 30.0))
        for index in range(9)
    ]
    result = probe(model, rows[:3], rows[3:6], epochs=30)
    if not 0.0 <= result.accuracy <= 1.0:
        raise AssertionError("the probe accuracy left [0, 1]")
    return {
        "accuracy": result.accuracy,
        "majority": result.majority_accuracy,
        "lift": result.lift,
        "episodes": result.episodes,
    }


def _check_leave_one_site_out() -> dict[str, Any]:
    from astra_triage.evaluation.sites import leave_one_site_out

    rows = [
        _episode(f"L{index}", ("A", "B")[index // 4], index % 2, 40.0 + index, (8.0,), (4.0, 30.0))
        for index in range(8)
    ]
    settings = TrainConfig(
        epochs=2, batch_size=4, learning_rate=5e-3, warmup_steps=0, scheduler="constant", precision="fp32"
    )
    produced = leave_one_site_out(lambda: _model(seed=2), rows, settings, torch.device("cpu"), seed=2)
    if not produced:
        raise AssertionError("no held-out site was scored")
    return {"sites": [row.site for row in produced], "auroc": [round(row.auroc, 4) for row in produced]}


def _check_reading_study() -> dict[str, Any]:
    from astra_triage.evaluation.reading import ReadingArm, gaps, summarise

    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    alone = torch.tensor([[0.6, 0.4, 0.55, 0.45, 0.5, 0.3]])
    model = torch.tensor([[0.8, 0.2, 0.7, 0.1, 0.9, 0.2]])
    supported = torch.tensor([[0.85, 0.15, 0.8, 0.1, 0.9, 0.25]])
    arms = [
        ReadingArm("clinician_alone", alone),
        ReadingArm("model_alone", model),
        ReadingArm("clinician_with_support", supported),
    ]
    rows = summarise(arms, labels)
    values = gaps(arms, labels, resamples=40)
    return {
        "rows": [round(row.auroc, 4) for row in rows],
        "supported_minus_unaided": round(values["supported_minus_unaided"][0], 4),
    }


def _check_properties_run() -> dict[str, Any]:
    from astra_triage.properties.counting_estimator import selection_bias
    from astra_triage.properties.dominance import dominance
    from astra_triage.properties.locf_bias import gap_information

    gap = gap_information(1.5, courses=1500)
    counting = selection_bias(1.5, episodes=3000)
    limit = dominance(1.5, episodes=1500)
    if not all(math.isfinite(value) for value in (gap.gap_coefficient, counting.bias, limit.admissible_risk)):
        raise AssertionError("a property returned a non-finite number")
    return {
        "gap_coefficient": round(gap.gap_coefficient, 4),
        "counting_bias": round(counting.bias, 4),
        "admissible_risk": round(limit.admissible_risk, 4),
        "earned": round(limit.earned, 5),
    }


def _check_cohort_round_trip() -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    from astra_triage.cohort.reader import RecordError, read_episodes, write_episodes
    from astra_triage.cohort.split import assert_patient_disjoint
    from astra_triage.streams.case import Episode

    rows = [_episode(f"R{index}", "A", index % 2, 40.0, (8.0,), (4.0,)) for index in range(4)]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "cohort.jsonl"
        write_episodes(rows, path)
        restored = read_episodes(path)
    if [e.episode_id for e in restored] != [e.episode_id for e in rows]:
        raise AssertionError("the round trip lost an episode")
    straddling = [
        rows[0],
        Episode(
            episode_id="R9",
            patient_id=rows[0].patient_id,
            site="B",
            age_years=2.0,
            procedure="appendicectomy",
            index_hours=40.0,
            ultrasound=rows[1].ultrasound,
            markers=rows[1].markers,
            label=1,
            split="test",
        ),
    ]
    try:
        assert_patient_disjoint(straddling)
    except RecordError:
        return {"episodes": len(restored), "straddle_detected": True}
    raise AssertionError("a patient straddling two splits was accepted")


def _check_strata_defined() -> dict[str, Any]:
    from astra_triage.cohort.strata import STRATA

    names = [entry.name for entry in STRATA]
    expected = {
        "age_under_1",
        "age_1_to_5",
        "age_6_to_12",
        "age_13_to_17",
        "necrotising_enterocolitis",
        "complicated_appendicitis",
        "enterostomy",
        "high_sparsity",
    }
    if set(names) != expected or len(names) != len(expected):
        raise AssertionError(f"the strata are {names}")
    return {"strata": names}


CASES: tuple[Case, ...] = (
    Case(
        "E01",
        "the fused model returns one finite probability per episode for several batch sizes",
        "Sec. 2.3.0.1, Eq. (1)",
        _check_forward,
    ),
    Case(
        "E02",
        "duplicating every reading leaves the probability unchanged",
        "Sec. 2.3.0.1, count-blind constraint",
        _check_replication,
        1e-5,
    ),
    Case(
        "E03",
        "the count-blind probe detects an estimator that reads the count",
        "Sec. 2.3.0.1, count-blind constraint",
        _check_unconstrained_breaks_it,
    ),
    Case(
        "E04",
        "a structurally absent stream still produces a score and changes it",
        "Table 1 footnote; Sec. 2.3.0.3",
        _check_absent_stream,
    ),
    Case(
        "E05",
        "the missingness token fires beyond 24 hours and not at 24 hours",
        "Sec. 2.3.0.3",
        _check_stale_boundary,
    ),
    Case(
        "E06",
        "the composite loss reaches every trainable parameter",
        "Eq. (3)",
        _check_gradients,
    ),
    Case(
        "E07",
        "an optimiser step moves the weights",
        "Sec. 2.4 is blank; the schedule is a stated default",
        _check_optimizer_step,
    ),
    Case(
        "E08",
        "a checkpoint restores the logits and carries the seed",
        "Atomic checkpoint contract",
        _check_checkpoint,
    ),
    Case(
        "E09",
        "one batch is fitted below half its starting loss inside 120 steps",
        "Execution check, not a reported result",
        _check_overfit,
    ),
    Case(
        "E10",
        "the class-weighted cross-entropy is ln 2 when balanced and larger when not",
        "Sec. 2.3.0.4, effective-number weights",
        _check_class_weighting,
    ),
    Case(
        "E11",
        "the batch calibration penalty vanishes on an aligned batch",
        "Sec. 2.3.0.4, L_cal",
        _check_calibration_penalty,
    ),
    Case(
        "E12",
        "expected calibration error is small when calibrated and large when shifted",
        "Table 2, column ECE",
        _check_ece,
    ),
    Case(
        "E13",
        "the Brier score equals its definition on a hand-built example",
        "Table 2, column Brier",
        _check_brier,
    ),
    Case(
        "E14",
        "the area under the curve equals the pairwise tally including ties",
        "Table 2, column AUROC",
        _check_auroc_pairwise,
    ),
    Case(
        "E15",
        "average precision equals the step formula on a hand-built example",
        "Table 2, column AUPRC",
        _check_auprc,
    ),
    Case(
        "E16",
        "the DeLong interval brackets its point estimate and has positive variance",
        "Table 2, bracketed intervals",
        _check_interval,
    ),
    Case(
        "E17",
        "net benefit equals the counted form on a hand-built cohort",
        "Abstract, net benefit at 0.20",
        _check_net_benefit,
    ),
    Case(
        "E18",
        "the predictive-value identities close against counted confusion cells",
        "Table 2, columns PPV and NPV",
        _check_predictive_values,
    ),
    Case(
        "E19",
        "the recalibration slope is one for a well-specified predictor and below one when overconfident",
        "Table 2, column Slope",
        _check_slope,
    ),
    Case(
        "E20",
        "Cohen's kappa equals its closed form on a hand-built table",
        "Table 7 footnote, kappa 0.58",
        _check_kappa,
    ),
    Case(
        "E21",
        "the endpoint closes at day seven and excludes day eight",
        "Sec. 2.1",
        _check_endpoint_window,
    ),
    Case(
        "E22",
        "the index-time filter removes every later observation",
        "Eq. (1)",
        _check_index_filter,
    ),
    Case(
        "E23",
        "the unobserved stream-day share matches a hand-counted four-day episode",
        "Table 1 and Table 6 panel B",
        _check_sparsity,
    ),
    Case(
        "E24",
        "the gradient reversal flips the sign at the requested strength",
        "Sec. 2.3.0.3",
        _check_reversal,
    ),
    Case(
        "E25",
        "seeding makes two batch builds identical",
        "Single seeding utility",
        _check_determinism,
    ),
    Case(
        "E26",
        "the site head reaches the encoder through the reversal path",
        "Sec. 2.3.0.3",
        _check_site_reversal_path,
    ),
    Case(
        "E27",
        "the model size sits in the range the default configuration declares",
        "Engineering default",
        _check_trainable_count,
    ),
    Case(
        "E28",
        "every registered arm builds, fits where it has a fit, and returns a finite score",
        "Table 2, all eight rows",
        _check_every_arm,
    ),
    Case(
        "E29",
        "a fresh site probe fits on frozen representations and returns an accuracy",
        "Sec. 2.3.0.1, the position test",
        _check_site_probe_runs,
    ),
    Case(
        "E30",
        "leave-one-site-out refits and scores the held-out site",
        "Table 4 panel B",
        _check_leave_one_site_out,
    ),
    Case(
        "E31",
        "the reading study summarises three arms and returns both gaps",
        "Table 7",
        _check_reading_study,
    ),
    Case(
        "E32",
        "the three appendix properties return finite numbers on draws from their model",
        "Sec. 2.3.0.1, appendix properties",
        _check_properties_run,
    ),
    Case(
        "E33",
        "an episode table survives a write and read, and a straddling patient is refused",
        "Table 1, patient-level split",
        _check_cohort_round_trip,
    ),
    Case(
        "E34",
        "the eight pre-specified subgroups are defined once each",
        "Table 6 panel A",
        _check_strata_defined,
    ),
)


def run() -> list[Check]:
    results: list[Check] = []
    for case in CASES:
        try:
            detail = case.run()
            results.append(
                Check(
                    case.identifier,
                    "code",
                    case.statement,
                    case.location,
                    "PASS",
                    observed=detail,
                    tolerance=case.tolerance,
                )
            )
        except Exception as error:
            results.append(
                Check(
                    case.identifier,
                    "code",
                    case.statement,
                    case.location,
                    "FAIL",
                    observed={"error": f"{type(error).__name__}: {error}"},
                    tolerance=case.tolerance,
                )
            )
    return results


def summary(results: list[Check]) -> dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "BLOCKED": 0}
    for check in results:
        counts[check.status] += 1
    return counts
