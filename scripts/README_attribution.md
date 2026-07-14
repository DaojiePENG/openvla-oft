# Fig.6 — Effective Input Attribution Visualization

This document explains how to use `visualize_action_head_attribution.py` to generate the
"Effective Input Attribution vs. Delay" figure (Fig.6 in the paper).

---

## Background

Our VisionActionHead uses a simple `concat → MLPResNet` architecture — there is no
explicit attention weight α. To show that the trained network implicitly learns a
"trust allocation" between cloud planning features and edge vision features, we extract
an **effective α(d)** using two complementary methods:

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| `gradient` (default) | $\alpha = \frac{\|\nabla_h \hat{a}\|}{\|\nabla_h \hat{a}\| + \|\nabla_z \hat{a}\|}$ | Theoretically grounded; sensitive to subtle reliance shifts | Requires backward pass (slower) |
| `ablation` | $\alpha = 1 - \frac{\|\hat{a}(h,\mathbf{0}) - \hat{a}(h,z)\|}{\|\hat{a}(h,z)\|}$ | Simple; no gradients needed; fast | Only captures coarse contribution |

**Expected result:** α is high (~0.7–0.9) for small delays (planning features are fresh and
informative) and drops to ~0.2–0.4 for large delays (action head shifts reliance to local
vision to compensate for stale planning features).

---

## Prerequisites

1. A trained checkpoint with VisionActionHead (not the base OpenVLA-OFT, but your
   CloudEdgeVLA fine-tuned checkpoint that includes the action_head `.pt` file).

2. LIBERO installed (`pip install libero`), since the script runs real LIBERO rollouts
   to collect observations.

3. All dependencies of the main evaluation pipeline (torch, transformers, timm, draccus,
   tensorflow, etc.).

---

## Quick Start

### Single suite (produces a line plot)

```bash
python scripts/visualize_action_head_attribution.py \
    --pretrained_checkpoint /path/to/your/checkpoint \
    --task_suite_name libero_spatial \
    --method gradient \
    --output_path results/fig6_attribution_spatial.pdf
```

### All 4 LIBERO suites (produces a grouped bar chart)

```bash
python scripts/visualize_action_head_attribution.py \
    --pretrained_checkpoint /path/to/your/checkpoint \
    --task_suite_name all \
    --method gradient \
    --output_path results/fig6_attribution_all_suites.pdf
```

### Using ablation method (faster, good for debugging)

```bash
python scripts/visualize_action_head_attribution.py \
    --pretrained_checkpoint /path/to/your/checkpoint \
    --task_suite_name libero_spatial \
    --method ablation \
    --output_path results/fig6_ablation_spatial.pdf
```

---

## All CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pretrained_checkpoint` | str | *(required)* | Path to trained VLA checkpoint directory |
| `--task_suite_name` | str | `libero_spatial` | LIBERO suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, or `all` |
| `--method` | str | `gradient` | Attribution method: `gradient` or `ablation` |
| `--delays` | int list | `0 1 3 5 8 10 15 20` | Delay values (in env steps) to evaluate |
| `--num_episodes` | int | 3 | Episodes per task for data collection |
| `--num_frames` | int | 40 | Frames per episode |
| `--max_samples_per_delay` | int | 30 | Max attribution samples per delay value per task |
| `--output_path` | str | `results/fig6_attribution.pdf` | Where to save the figure |
| `--lora_rank` | int | 32 | LoRA rank (must match training) |
| `--action_head_vision_encoder` | str | `siglip-base` | Vision encoder used in VisionActionHead |
| `--num_views` | int | 2 | Camera views (primary + wrist) |

---

## Outputs

- **PDF figure** at the specified `--output_path`:
  - Single suite → line plot with ±1 std shading
  - All suites → grouped bar chart with error bars
- **Terminal summary** table: mean ± std of α(d) for each delay value

---

## Runtime Estimates

| Configuration | Approx. Time (1× A100) |
|---------------|------------------------|
| Single suite, `gradient`, 3 episodes | ~15–25 min |
| Single suite, `ablation`, 3 episodes | ~8–12 min |
| All 4 suites, `gradient`, 3 episodes | ~60–100 min |

The bottleneck is the VLA forward pass on the delayed frame for each sample.
To speed up: reduce `--num_episodes`, `--num_frames`, or `--max_samples_per_delay`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'libero'`**
→ Install LIBERO: `pip install libero`. The script imports `libero.libero.benchmark`
to load task suites.

**`FileNotFoundError: action_head checkpoint not found`**
→ Ensure your checkpoint directory contains a file matching the pattern
`action_head*checkpoint*.pt`. This is produced by the training script.

**`AttributeError: 'OpenVLAForActionPrediction' has no attribute 'llm_dim'`**
→ Your checkpoint may be an older format. Make sure you're loading with the updated
`modeling_prismatic.py` that exposes `llm_dim`.

**Gradient attribution returns α ≈ 0.5 for all delays**
→ This usually means the action head was not trained with the dual-path loss
(paired-frame training). The α ≈ 0.5 is the expected value for a randomly initialized
fusion MLP. Make sure you're loading a checkpoint trained with `L_stale + L_fresh`.

---

## How It Works (Technical Details)

1. **Observation collection:** Runs real LIBERO episodes and stores raw images + proprio
   for each frame.

2. **Delay simulation:** For each delay *d*, pairs frame *t* (current) with frame *t-d*
   (delayed) from the same episode.

3. **Hidden state capture:** Processes the delayed frame through the full VLA pipeline
   (vision backbone → LLM → action tokens) to extract the action-token hidden states
   `h_stale ∈ ℝ^{L×D}`. This is captured via a context-manager monkey-patch on
   `VisionActionHead.predict_action` — it intercepts the `llm_hidden_states` argument
   without altering the forward pass.

4. **Attribution computation:**
   - *Gradient method:* Re-runs only the action head (not the full VLA) with
     `h_stale` and the current frame's pixels. Computes `∂â/∂h` and `∂â/∂z` via
     `torch.autograd.backward`. The ratio of gradient norms gives α.
   - *Ablation method:* Runs the action head twice — once normally, once with vision
     zeroed — and measures the output difference.

5. **Plotting:** Aggregates α values across all tasks and episodes per delay, plots
     mean ± std.
