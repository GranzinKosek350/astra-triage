# Temporal Fusion of Postoperative Ultrasound and Inflammatory Markers for Intra-Abdominal Infection Triage

## Overview

A child recovering from bowel surgery is measured on two clocks that do not line up:
bedside ultrasound when someone looks, and a blood panel when the morning round orders
one. The decision the ward faces is whether to keep the antibiotics running or to go
back to theatre, and it is usually taken late, after the observations have already
stopped being reassuring.

This release implements the estimator the article calls ASTRA and the seven arms it is
compared against. The estimator reads the two stamped sequences as they stood at the
index time, not as they look in retrospect. Each reading carries its descriptor vector,
how long ago it was taken, and how long the stream had been silent. Each stream reads
the other through attention, each is pooled by recency, the two pools are mixed by a
reliability gate, and a two-layer head scores the case against a static vector of age,
procedure and site. Three mechanisms make the record's own shape part of the input: a
learned token for a stream that has gone quiet for more than a day, a separate token for
a stream the resource never records, and a site adapter held in check by a gradient
reversal.

What the article claims is a constrained representation: parity in discrimination with
an unconstrained late-fusion baseline, and an advantage in calibration and at the small
sites. The code is organised so that both halves of that claim can be examined — the
admissible-set audit in `constraints/`, the ablation surgery in `evaluation/ablation.py`,
and the three appendix properties in `properties/`.

No clinical record is shipped, rebuilt or inferred. The five-site derivation cohort, the
temporal hold-out, the reading study and the two prospective windows are private and
de-identified at patient and institution level; the external resource is credentialed.
Every command that reads records takes a table from the caller and stops without one.
Every record-level number in the article is marked `BLOCKED` in `claim_to_code.json`,
with the cohort it would need. The execution checks in the verification report use
tensors built inside the check process.

## What is in the tree

| path | contents |
| --- | --- |
| `claim_to_code.json` | eighty article statements, each with its location, the symbol that carries it and the checks that decide it |
| `verification_report.json` | both verification layers, the manuscript arithmetic, the configuration conformance, the hygiene scan and every blocked record-level check |
| `verification_summary.txt` | the same verdicts as plain text |
| `dataset_urls.txt` | the adjudicated external resource, its licence and what the probe returned |
| `integrity_manifest.json` | one SHA-256 per shipped file, written last |
| `THIRD_PARTY_NOTICES.txt` | the data attribution and the publications the comparator designs follow |
| `configs/` | endpoint, stream, model, optimisation and one file per reported experiment |
| `src/astra_triage/` | the package, laid out along the measurement chain rather than along a training script |

## Installation

```bash
python3 -m pip install -r requirements.txt -r requirements-dev.txt
export PYTHONPATH="$PWD/src"

conda env create -f environment.yml
conda activate astra-triage

docker build -t astra-triage .
```

The container definition is part of the release and runs the test suite as its build step;
it was not built on the authoring machine, which carries no container daemon, and
`verification_report.json` records that as `NOT_RUN` rather than as a success.

The package is not installed into the environment by default. `PYTHONPATH=src` is what
the scripts use and what `pyproject.toml` sets for the test runner.

## Data

The release ships no episode. Two things are needed before any command that reads
records will run.

**A staged episode table.** One JSON object per line, with the fields below. `index_hours`
and every timestamp are hours from admission; observations after `index_hours` are
dropped by the loader, so a table that contains them is safe to stage.

```json
{"episode_id": "…", "patient_id": "…", "site": "A", "age_years": 3.4,
 "procedure": "resection_with_anastomosis", "index_hours": 54.0, "split": "train",
 "label": 0, "ultrasound_recorded": true, "markers_recorded": true,
 "backgrounds": ["enterostomy"],
 "ultrasound": [[6.0, [2.1, 1, 0, 5.0, 0, 1]]],
 "markers": [[2.0, [12.0, 8.0, 2.0, 300.0, 45.0, 0.8]]],
 "vitals": null, "clinical": null}
```

`vitals` and `clinical` are optional and are read only by the two named clinical rules;
an arm whose block is absent reports itself unavailable rather than scoring a normal
value. A row may instead carry `reads` (the two blind adjudications) and `arbiter`,
in which case the label is derived through `endpoints/adjudication.py`.

```bash
bash scripts/prepare_cohort.sh /path/to/cohort.jsonl
```

That command prints the episode and event counts per partition and refuses a table in
which one patient straddles two splits.

**The external resource.** The independent cohort of Table 5 is the Paediatric Intensive
Care database v1.1.0 on PhysioNet, under the Credentialed Health Data License 1.5.0 and
its data use agreement; the landing page, DOI and access conditions are in
`dataset_urls.txt`. It is not redistributed here and no record from it is included. Its
2010 to 2018 window, its paediatric intensive-care setting and the absence of an
ultrasound stream are what identify it as the article's external cohort, and the
laboratory-pathway configuration (`supplementary_external.yaml`) drops the ultrasound
stream at the input for exactly that reason.

## Training

```bash
bash scripts/train.sh configs/experiments/main.yaml /path/to/cohort.jsonl artifacts/model.pt
```

One file per reported experiment:

| file | what it is |
| --- | --- |
| `main.yaml` | the constrained representation of Table 2 |
| `ablation_without_ultrasound.yaml`, `ablation_without_markers.yaml` | the two stream removals of Table 3 |
| `ablation_without_time_encoding.yaml`, `ablation_without_cross_attention.yaml`, `ablation_without_missingness_token.yaml`, `ablation_without_site_adapter.yaml` | the four component removals of Table 3 |
| `baseline_late_fusion.yaml`, `baseline_tft.yaml`, `baseline_grud.yaml`, `baseline_logistic_last_value.yaml` | the learned comparators |
| `baseline_pews.yaml`, `baseline_pas.yaml`, `baseline_crp_rule.yaml` | the three clinical rules |
| `supplementary_external.yaml`, `supplementary_subgroups.yaml`, `supplementary_leave_one_site.yaml`, `supplementary_reader_study.yaml`, `supplementary_silent_mode.yaml`, `supplementary_open_label.yaml` | the five supplementary tables |
| `_smoke.yaml` | for unit-test smoke only; do not use for reporting |

`configs/optimisation/adamw_cosine.yaml` holds the run settings. The article prints no
training schedule — Sec. 2.4 is a heading with no body, and no hyperparameter table
appears anywhere in the text or the supplementary tables — so every value in that file
is a stated default: 40 epochs, batch 32, one accumulation step, one process, AdamW at
3e-4 with weight decay 1e-2, 500 warm-up steps into a cosine decay, bf16, gradient
clipping at 1.0, five seeds. `verification_report.json` lists them under the
conformance checks as defaults, not as manuscript settings.

## Evaluation

```bash
# one arm on one partition
bash scripts/evaluate.sh configs/experiments/main.yaml /path/to/cohort.jsonl test artifacts/model.pt

# every arm on the same records, with the per-site, subgroup, sparsity and decision views
python -m astra_triage.cli.evaluate --experiment configs/experiments/main.yaml \
    --cohort /path/to/cohort.jsonl --split test --table --checkpoint artifacts/model.pt

# the reader matrix and the two prospective windows, from JSON tables
python -m astra_triage.cli.evaluate --experiment configs/experiments/main.yaml --reading reading.json
python -m astra_triage.cli.evaluate --experiment configs/experiments/main.yaml --deployment windows.json

# batch scoring, one line per episode
python -m astra_triage.cli.score --experiment configs/experiments/main.yaml \
    --cohort /path/to/cohort.jsonl --split test --out artifacts/scores.jsonl

# the admissible-set audit and the three appendix properties
python -m astra_triage.cli.audit --properties
python -m astra_triage.cli.audit --experiment configs/experiments/main.yaml \
    --cohort /path/to/cohort.jsonl --split validation --checkpoint artifacts/model.pt
```

`--table` reads one checkpoint per learned arm from `artifacts/<arm>.pt` (override the
directory with `--checkpoint-dir`), so train each arm first; an arm without a checkpoint
is reported unavailable rather than scored from its initial weights.

The evaluation command prints the row; it does not compare it against a printed value,
because no episode ships. The expected numbers and their tolerances are in the article,
and the mapping from each of those numbers to the symbol that would produce it is in
`claim_to_code.json`; the status of each is `BLOCKED` until the table is staged.

A `--reading` table is `{"labels": [...], "resamples": 2000, "arms": [{"name":
"clinician_alone", "probabilities": [[...] per reader], "median_minutes": 6.8}, ...]}`.
A `--deployment` log is `{"threshold_fixed_at": 0.2, "silent": {...}, "open_label":
{...}}` with the counts and medians of Table 9.

## Compute budget

The article does not report the training machine, so nothing here is a manuscript
figure. The defaults describe a model of 257k parameters: batch 32, 40 epochs, one
process, bf16 autocast, which fits on a single 8 GB accelerator and runs in minutes per
seed on the 3,412-episode cohort once staged. The peak footprint is dominated by the
padded sequence tensor, `batch × capacity × feature width`, not by the weights. The five
learned arms and the five seeds are the whole training cost of Table 2.

The verification pass needs no accelerator and no records: `pytest -q` finishes 160
tests in a few seconds on a laptop CPU, and `bash scripts/verify.sh` writes the four
artefacts in a few more.

## Verification

```bash
bash scripts/verify.sh   # writes claim_to_code.json, verification_report.json,
                         # verification_summary.txt, integrity_manifest.json
pytest -q
ruff check .
mypy --strict src/astra_triage
```

The first layer maps each article statement to code and, where the statement is
arithmetic, recomputes it from the article's own printed numbers. The second layer runs
the released modules on tensors built inside the check process — forward, loss,
backward, optimiser step, checkpoint round trip, single-batch overfit, the
count-blindness probes, the 24-hour boundary, the gradient reversal — and compares each
result with an answer computed by a separate route.

The two families are reported separately, because they answer different questions.
`code_status` is `PASS` when every execution, conformance and hygiene check holds.
`manuscript_status` records what the article's own tables do when their columns are
recomputed against each other; it is not `PASS`, and the report lists which rows and
why. `overall_status` is `PARTIALLY_VERIFIED`, which is the correct verdict for a
release whose record layer cannot run.

## Licence

MIT, see `LICENSE`. Third-party material, data attribution and the publications the
comparator designs follow are listed in `THIRD_PARTY_NOTICES.txt`.
