# Latency-Tolerant Cloud-Edge Collaborative Vision-Language-Action Models via Emergent Representational Specialization

**Paper:** _(placeholder — arXiv link TBD)_

**Project website:** _(placeholder)_

**Summary video:** _(placeholder)_

---

## Overview

Deploying a 7B-parameter VLA on a robot requires either heavy on-board compute or offloading inference to the cloud. Cloud offloading introduces **variable network latency**: by the time the cloud's planning features arrive at the robot, the observation they were computed from is stale. Existing approaches (open-loop action-chunk replanning, synchronous fast-slow systems) assume bounded latency and a synchronized clock, and they stall or fail when those assumptions break.

**CloudEdgeVLA** splits the policy across the network boundary and trains it so that latency stops mattering:

- **Cloud — VLA backbone $f_\theta$** (7B, LoRA-adapted): processes a possibly-delayed observation $o_{t-d}$ and emits high-level *planning features* $h_{t-d}$. Runs asynchronously; the edge never blocks on it.
- **Edge — VisionActionHead $g_\phi$**: fuses the (stale) planning features $h_{t-d}$ with **real-time** local vision features $z_t$ from a frozen SigLIP encoder $v_\psi$, and outputs the action chunk $\hat{a}_t$. Runs every control step with whatever cloud features most recently arrived.

The key idea is **emergent representational specialization**: a paired-frame dual-path training objective pushes the backbone to encode only *time-invariant* task semantics ("what to do") while the action head learns to recover *time-critical* motor details ("how to do it right now") from fresh local vision. Because the backbone's output is delay-invariant by construction, a stale $h_{t-d}$ carries the same task-level information as a fresh $h_t$, and the edge fills in the rest.

<!-- placeholder: system architecture figure (see docs/architecture.md, Fig. 1) -->

---

## Method

### Paired-frame dual-path training

Each training step draws two frames from the **same episode window**: the current frame $o_t$ and a randomly delayed frame $o_{t-d}$, with $d \sim \mathcal{U}(1, W-1)$ for window size $W$. Both go through the **shared** backbone $f_\theta$, and both resulting planning features are fused with the **same** real-time vision features $z_t = v_\psi(o_t)$:

$$
\hat{a}^{\text{fresh}} = g_\phi(h_t,\; z_t), \qquad
\hat{a}^{\text{stale}} = g_\phi(h_{t-d},\; z_t)
$$

### Dual-path loss with curriculum

Both predictions are trained toward the same ground-truth action $a_t$:

$$
\mathcal{L} = (1-\lambda)\underbrace{\lVert \hat{a}^{\text{fresh}} - a_t \rVert_1}_{\mathcal{L}_{\text{fresh}}} + \lambda \underbrace{\lVert \hat{a}^{\text{stale}} - a_t \rVert_1}_{\mathcal{L}_{\text{stale}}}
$$

- $\mathcal{L}_{\text{fresh}}$ trains the action head for synchronous operation and gives the backbone direct action supervision.
- $\mathcal{L}_{\text{stale}}$ trains the action head to **compensate** for stale planning features using real-time vision: when $h_{t-d}$ is misaligned with the current state, the head must rely more on $z_t$.
- $\lambda$ follows a **curriculum**: it ramps linearly from $0$ to $\lambda_{\max}$ over the first half of training, letting the backbone first learn strong representations before being pressured to become delay-invariant.

The frozen edge vision encoder $v_\psi$ receives no gradient. Gradients from both losses flow to $\theta$ (backbone) and $\phi$ (action head).

<!-- placeholder: training pipeline figure (see docs/architecture.md, Fig. 2) -->

### Asynchronous deployment

At deployment the edge computes $z_t$ and executes an action **every** control step, always using the most recently received cloud features $h_{t-d}$ — no matter how stale. The cloud runs inference in the background and streams features whenever they are ready. See `docs/architecture.md` §5 for the deployment timeline.

---

## Results

_(placeholder — fill in with your numbers)_

Success rate vs. observation delay on LIBERO (delay in environment steps):

| Delay (steps) | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| OpenVLA | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| OpenVLA-OFT | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| UniVLA | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **CloudEdgeVLA (ours)** | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

<!-- placeholder: delay-vs-success-rate plot -->

---

## System Requirements

Inference:
* 1 GPU with ~16 GB VRAM for LIBERO sim benchmark tasks

Training:
* 1 GPU (A800 80 GB used for the reference runs). See [SETUP.md](SETUP.md) and the base OpenVLA-OFT [compute FAQ](https://openvla-oft.github.io/#train-compute) for other configurations.

---

## Installation

See [SETUP.md](SETUP.md) for setting up the conda environment (`openvla-oft`).

For LIBERO, additionally install the benchmark and its dependencies as described in [LIBERO.md](LIBERO.md).

---

## Training

CloudEdgeVLA is trained by fine-tuning an OpenVLA backbone with the frame-delay + VisionActionHead objective. The reference launch script for the `libero_goal` suite is [`hpc2_ft_libero_goal_frame_delay.sh`](hpc2_ft_libero_goal_frame_delay.sh):

```bash
sbatch hpc2_ft_libero_goal_frame_delay.sh
```

Key flags (passed to `vla-scripts/finetune.py`):

| Flag | Meaning |
|---|---|
| `--use_frame_delay true` | Sample a delayed frame from the dataset window each step |
| `--window_size 21` | Window of past frames; max delay = `window_size - 1` |
| `--use_vision_action_head true` | Use the fusion `VisionActionHead` instead of the plain L1 head |
| `--action_head_vision_encoder siglip-base` | Frozen edge vision encoder |
| `--freeze_action_head_vision true` | Keep the edge vision encoder frozen |
| `--action_head_num_views 2` | Primary + wrist camera |
| `--stale_loss_lambda_max 0.5` | $\lambda_{\max}$ for the dual-path loss |
| `--stale_loss_warmup_steps -1` | Curriculum ramp length; `-1` = `max_steps / 2` |

**Curriculum behaviour** ($\lambda$ over training, `max_steps = 200k`, warmup = `-1`):

```
step 0        λ = 0.0     loss = L_fresh only          (learn base task first)
step 100k     λ = 0.5     loss = 0.5·L_fresh + 0.5·L_stale   (full dual-path, held to end)
```

Watch the `loss_fresh` and `loss_stale` curves in W&B — both converging is the signal that the backbone/head split is being learned.

Data and paths are set at the top of the launch script:

```bash
VLA_PATH="openvla/openvla-7b"
DATA_ROOT_DIR="/path/to/modified_libero_rlds"     # placeholder
DATASET_NAME="libero_goal_no_noops"
RUN_ROOT_DIR="/path/to/runs"                       # placeholder
```

---

## Evaluation

Evaluate a trained checkpoint under simulated observation delay with [`hpc2_eval_frame_delay.sh`](hpc2_eval_frame_delay.sh) (wraps `experiments/robot/libero/run_libero_eval.py`):

```bash
sbatch hpc2_eval_frame_delay.sh
```

The eval harness holds a history of recent frames and, at each requery, feeds a delayed frame to the VLA backbone while passing the current frame to the VisionActionHead. Delay configs:

```
--use_frame_delay_eval false                           # baseline, no delay
--use_frame_delay_eval true  --max_delay_steps_eval 5  # uniform delay ≤ 5
--use_frame_delay_eval true  --max_delay_steps_eval 10 # uniform delay ≤ 10
--use_frame_delay_eval true  --max_delay_steps_eval 15 # uniform delay ≤ 15
```

---

## Analysis

Two scripts probe *why* the model is delay-tolerant. Both save the plotted figure and a companion `*_data.json` with the raw numbers.

### Representational probe (does the backbone discard motor state?)

Trains linear probes on backbone hidden states to predict high-level (object positions, goal progress) vs. low-level (joint/gripper/eef) state. Expectation: the standard VLA encodes both; ours encodes high-level but **discards** low-level. See [`scripts/README_probe.md`](scripts/README_probe.md).

```bash
sbatch hpc2_probe_libero_goal.sh
```

### Action-head attribution (does the head lean on vision as delay grows?)

Measures the effective reliance $\alpha(d)$ of the action head on cloud planning features vs. edge vision, via gradient- and ablation-based attribution across delays. See [`scripts/README_attribution.md`](scripts/README_attribution.md).

```bash
sbatch hpc2_attribution_libero_goal.sh
```

<!-- placeholder: representational specialization figure (see docs/architecture.md, Fig. 3) -->

---

## Repository Layout

| Path | Contents |
|---|---|
| `vla-scripts/finetune.py` | Training loop; dual-path forward pass and curriculum loss |
| `prismatic/models/action_heads.py` | `VisionActionHead`, `VisionEncoder`, `L1RegressionActionHead` |
| `prismatic/vla/datasets/datasets.py` | RLDS window + delayed-frame sampling |
| `prismatic/util/data_utils.py` | Collator (forwards `delayed_pixel_values`) |
| `experiments/robot/libero/run_libero_eval.py` | LIBERO eval with delay simulation |
| `scripts/visualize_representational_probe.py` | Linear-probe analysis |
| `scripts/visualize_action_head_attribution.py` | Attribution analysis |
| `docs/architecture.md` | Detailed architecture / figure spec |
| `hpc2_*.sh` | SLURM launch scripts (train / eval / probe / attribution) |

---

## Acknowledgements

This project is built on [OpenVLA-OFT](https://github.com/moojink/openvla-oft) (Kim et al., 2025), which is in turn built on [OpenVLA](https://openvla.github.io/). We thank the authors for releasing their code.

---

## Citation

_(placeholder — update once the paper is public)_

```bibtex
@article{cloudedgevla,
  title={Latency-Tolerant Cloud-Edge Collaborative Vision-Language-Action Models via Emergent Representational Specialization},
  author={TBD},
  journal={TBD},
  year={2026}
}
```
