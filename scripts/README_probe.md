# Fig.5 — Representational Analysis via Linear Probes

This document explains how to use `visualize_representational_probe.py` to generate the
"Representational Analysis via Linear Probes" figure (Fig.5 in the paper).

---

## Background

**Core hypothesis:** Our paired-frame dual-path training induces *emergent representational
specialization* — the cloud backbone learns to encode **high-level, time-invariant** task
information (what to do) while **discarding** low-level, timing-critical robot state
(how to do it right now).

A standard single-frame-trained VLA backbone, by contrast, encodes **everything** — both
high-level semantics and low-level motor details — into its hidden states.

**How to test this:** Train a simple linear regression probe (Ridge regression) on the
backbone's action-token hidden states to predict two kinds of ground-truth state:

| Probe Target | What it measures | Expected result |
|--------------|-----------------|-----------------|
| **High-level** (object positions + goal progress) | "What to do" — task-relevant semantic state | Both backbones: **high R²** |
| **Low-level** (joint angles, gripper pos, EEF pos) | "How to do it now" — timing-critical motor state | Standard VLA: **high R²**, Ours: **low R²** |

This gap in low-level probe accuracy is direct evidence that our backbone has been pushed
by $\mathcal{L}_{\text{stale}}$ to discard timing-critical information.

---

## Prerequisites

1. **Two checkpoints:**
   - Standard VLA checkpoint (single-frame trained, e.g. original OpenVLA-OFT fine-tuned
     on LIBERO with `L_fresh` only)
   - CloudEdgeVLA checkpoint (paired-frame trained, with VisionActionHead)

2. **LIBERO** installed (`pip install libero`)

3. **scikit-learn** installed (`pip install scikit-learn`) — used for Ridge regression
   linear probes.

---

## Quick Start

### Single suite (generates a one-suite, two-panel bar chart)

```bash
python scripts/visualize_representational_probe.py \
    --checkpoint_standard /path/to/openvla-oft-spatial \
    --checkpoint_ours     /path/to/cloudedgevla-spatial \
    --task_suite_name     libero_spatial \
    --output_path         results/fig5_probe_spatial.pdf
```

### All 4 LIBERO suites (generates the full Fig.5 with 4 groups per panel)

```bash
python scripts/visualize_representational_probe.py \
    --checkpoint_standard /path/to/openvla-oft-spatial-object-goal-10 \
    --checkpoint_ours     /path/to/cloudedgevla-spatial-object-goal-10 \
    --task_suite_name     all \
    --output_path         results/fig5_probe_all_suites.pdf
```

---

## All CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint_standard` | str | *(required)* | Path to Standard VLA checkpoint (single-frame trained) |
| `--checkpoint_ours` | str | *(required)* | Path to CloudEdgeVLA checkpoint (paired-frame trained) |
| `--task_suite_name` | str | `libero_spatial` | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, or `all` |
| `--num_episodes` | int | 5 | Number of episodes per task for data collection |
| `--num_frames` | int | 60 | Number of frames per episode to collect |
| `--lora_rank` | int | 32 | LoRA rank (must match both checkpoints) |
| `--test_ratio` | float | 0.3 | Fraction of data used for test set |
| `--output_path` | str | `results/fig5_probe.pdf` | Where to save the figure |
| `--seed` | int | 42 | Random seed for train/test split |

---

## Output

- **PDF figure** (two-panel bar chart):
  - Left panel: High-level probe R² (object positions + goal progress)
  - Right panel: Low-level probe R² (joint angles + gripper + EEF position)
  - Bars grouped by task suite, colored by model (grey = Standard VLA, blue = Ours)
- **Terminal summary table** with R² values

---

## Runtime Estimates

| Configuration | Approx. Time (1× A100) |
|---------------|------------------------|
| Single suite, 5 episodes × 60 frames | ~20–30 min |
| All 4 suites | ~80–120 min |

The bottleneck is hidden-state extraction from two VLA models (two full forward passes per
frame). To speed up: reduce `--num_episodes` or `--num_frames`.

---

## What the Probes Actually Measure

### High-Level State (object positions + goal progress)

Ground truth is constructed as:

```
high_level = [object_1_xyz, object_2_xyz, ..., goal_progress(0..1)]
```

- **Object positions:** extracted from MuJoCo via `env.sim.data.body_xpos`, filtering
  out robot links, table, floor, and other static bodies.
- **Goal progress:** normalised timestep `step / max_episode_steps` — a coarse proxy for
  how far along the task has progressed.

This target captures "what the scene looks like and how close we are to the goal" — purely
semantic information that should be encoded by any competent VLA backbone.

### Low-Level State (robot proprioception)

Ground truth is:

```
low_level = [eef_pos(3), eef_axisangle(3), gripper_qpos(2), gripper_open(1)]
```

This is the robot's instantaneous motor state at time $t$ — information that is
*timing-critical* (changes at every control step) and should be *discarded* by a
backbone trained to produce time-invariant features.

---

## Interpreting the Results

| Scenario | High-Level R² | Low-Level R² | Interpretation |
|----------|---------------|--------------|----------------|
| Standard VLA | ~0.8–0.95 | ~0.7–0.9 | Backbone encodes everything (expected) |
| CloudEdgeVLA (ours) | ~0.7–0.9 | ~0.1–0.3 | Backbone retains semantics, discards motor state ✓ |

The **gap in low-level R²** is the key evidence: our backbone's hidden states simply don't
contain precise robot joint/gripper information anymore. That information lives in the
edge-side vision encoder, which is the correct place for it (always fresh, latency-free).

---

## Technical Notes

**Why Ridge regression (not ordinary least squares)?**
Ridge adds L2 regularisation (`alpha=1.0`) which is important when the feature dimension
(L×D, typically 8×4096 = 32768) is large relative to the number of samples.

**Why R² (not classification accuracy)?**
The targets are continuous (positions, angles). R² measures how much variance in the
target is explained by the linear probe. R²=1 means perfect prediction; R²=0 means
the probe is no better than predicting the mean.

**Standardisation:** Both features and targets are standardised (zero mean, unit variance)
before fitting. This ensures multi-dimensional R² is meaningful when target dimensions
have different scales (e.g., metres for position vs. radians for angles).

**Why not a nonlinear probe (MLP)?**
A *linear* probe is specifically chosen because we want to measure what information is
*linearly decodable* from the hidden states. If a nonlinear probe could decode low-level
state but a linear one cannot, that would indicate the information is present but in a
highly compressed/entangled form — which is interesting but a different claim. Using
linear probes gives us a clean lower bound on what the backbone explicitly encodes.
