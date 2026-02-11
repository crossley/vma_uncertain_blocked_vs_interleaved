# vma_uncertain_blocked_vs_interleaved

This repository contains the code, data, and analysis for a
set of visuomotor adaptation (VMA) experiments examining how
**sensory uncertainty** and **trial structure** (blocked vs.
interleaved) influence motor adaptation.

Two closely related experiments are included:

1. **Visuomotor rotation (`_rot`)**
2. **Error clamp (`_clamp`)**

The experiments share the same task structure, data format,
and analysis pipeline. The only experimental difference is
the type of feedback manipulation applied during adaptation.

---

## Project Overview

Participants perform center-out reaching movements toward a
visual target. Across trials, the visual feedback associated
with the movement endpoint is manipulated while sensory
uncertainty varies either:

* **Blocked**: low and high uncertainty occur in contiguous
  blocks, or

* **Interleaved**: low and high uncertainty are randomly
  intermixed.

The project tests how uncertainty structure affects
trial-by-trial adaptation and learning dynamics.

---

## Experimental Variants

### 1. Visuomotor Rotation (`vma_uncertain_blocked_vs_interleaved_rot`)

Participants experience a **visuomotor rotation**, in which the visual cursor is rotated relative to the true hand movement.

* Rotation magnitude: **15°**
* Rotation applied during specific trial ranges

This version measures standard adaptation under uncertain visual feedback.

---

### 2. Error Clamp (`vma_uncertain_blocked_vs_interleaved_clamp`)

Participants experience an **error clamp**, where visual
feedback follows a predetermined trajectory independent of
the participant’s actual movement.

* Cursor feedback is clamped relative to the target direction
* Feedback error is fixed and not contingent on performance

Clamps are thought to isolate implicit learning processes by
removing explicit strategies.

All other aspects of the task (timing, uncertainty
manipulation, data recording, analysis) are matched to the
rotation version.

---

## Repository Structure

```
.
├── README.md
├── vma_uncertain_blocked_vs_interleaved_clamp/
└── vma_uncertain_blocked_vs_interleaved_rot/
```

Each experiment directory contains:

```
code/           Experiment scripts and analysis code
consent/        Participant information and consent materials
data/           Raw behavioral data (trial-level and movement-level)
data_summary/   Processed summary datasets (clamp only)
figures/        Generated figures
fits/           Model fits
```

---

## Running the Experiment

Each experiment includes a `run_exp.py` script:

```bash
cd vma_uncertain_blocked_vs_interleaved_rot/code
python run_exp.py
```

or

```bash
cd vma_uncertain_blocked_vs_interleaved_clamp/code
python run_exp.py
```

The script creates two output files per participant:

* `sub_<ID>_data.csv` — trial-level summary data
* `sub_<ID>_data_move.csv` — time-resolved movement data

---

## Data Files

Each session produces two CSV files per participant: a trial-level summary file and a movement-level time-series file.

---

### Trial-level file

`sub_<ID>_data.csv`

Each row corresponds to **one completed trial**.

| Column | Type | Description |
|---|---|---|
| `condition` | str | Experimental condition (`"blocked"` or `"interleaved"`), determining how sensory uncertainty is scheduled across trials. |
| `subject` | int | Participant ID. |
| `trial` | int | Trial index within the session. |
| `su` | float | Sensory uncertainty magnitude used on that trial (scaled in screen/pixel units). |
| `rotation` | float | Visuomotor rotation applied on that trial (rotation experiment) or equivalent manipulation parameter (clamp experiment). |
| `rt` | float | Reaction time in milliseconds (time from movement readiness to movement initiation). |
| `mt` | float | Movement time in milliseconds (time from movement initiation to reaching the target radius). |
| `ep` | float | Endpoint angle of the movement relative to the start position (radians). |

---

### Movement-level file

`sub_<ID>_data_move.csv`

Each row corresponds to a **time-sampled record during the experiment loop**, allowing reconstruction of movement trajectories.

| Column | Type | Description |
|---|---|---|
| `condition` | str | Experimental condition (`"blocked"` or `"interleaved"`). |
| `subject` | int | Participant ID. |
| `trial` | int | Trial index corresponding to the current movement sample. |
| `state` | str | Task state at the time of sampling (e.g., searching, holding, moving, feedback). |
| `t` | float | Experiment time in milliseconds since the start of the session. |
| `x` | float | Hand or cursor x-position in screen pixels. |
| `y` | float | Hand or cursor y-position in screen pixels. |

This file provides the full time-resolved trajectory for each trial and can be used to reconstruct movement kinematics and state transitions.

---

## Analysis

The primary analysis scripts arelocated in:

```
code/inspect_results.py
```

These scripts:

* load subject data
* compute trial-wise measures
* generate summary datasets
* produce per-subject and group-level figures
* prepare inputs for statistical analysis and model fitting

Generated outputs are written to:

* `data_summary/`
* `figures/`
* `fits/`

