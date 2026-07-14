"""
visualize_action_head_attribution.py

Computes and visualizes the effective input attribution α(d) for the VisionActionHead
under varying delay conditions.

For a concat + MLP architecture (no explicit attention), we measure how much the
action prediction relies on cloud planning features vs. edge vision features using
two complementary methods:

  Method A (default): Gradient-based attribution
      α(d) = ‖∂â/∂h‖ / (‖∂â/∂h‖ + ‖∂â/∂z‖)

  Method B: Input ablation
      α(d) = 1 - ‖â(h,0) - â(h,z)‖ / (‖â(h,z)‖ + ε)

Usage (single suite, gradient method):
    python scripts/visualize_action_head_attribution.py \
        --pretrained_checkpoint /path/to/checkpoint \
        --task_suite_name libero_spatial \
        --method gradient \
        --output_path results/fig6_attribution.pdf

Usage (all 4 suites, grouped bar chart):
    python scripts/visualize_action_head_attribution.py \
        --pretrained_checkpoint /path/to/checkpoint \
        --task_suite_name all \
        --method gradient \
        --output_path results/fig6_attribution_all_suites.pdf
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Append project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from prismatic.models.action_heads import VisionActionHead
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TASK_SUITE_NAMES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TASK_SUITE_LABELS = {
    "libero_spatial":  "Spatial",
    "libero_object":   "Object",
    "libero_goal":     "Goal",
    "libero_10":       "Long",
}

DEFAULT_DELAYS = [0, 1, 3, 5, 8, 10, 15, 20]


# ---------------------------------------------------------------------------
# Forward-hook helper to capture the LLM hidden states that flow into the
# VisionActionHead during a normal VLA forward pass.
# ---------------------------------------------------------------------------
class _HiddenStateCapture:
    """
    Context manager that registers a forward hook on ``action_head.predict_action``
    (the Python method, not an nn.Module) to intercept the ``llm_hidden_states``
    argument each time the action head is called.

    Usage::

        with _HiddenStateCapture(action_head) as cap:
            vla.predict_action(input_ids=..., pixel_values=..., action_head=action_head, ...)
        h_captured = cap.hidden_states  # tensor on DEVICE, detached
    """

    def __init__(self, action_head: VisionActionHead):
        self._action_head = action_head
        self._original_predict = action_head.__class__.predict_action
        self.hidden_states: Optional[torch.Tensor] = None

    def __enter__(self):
        _outer = self  # capture for closure

        def _intercepting_predict(self_head, llm_hidden_states, pixel_values=None):
            # Save a detached clone for later analysis
            _outer.hidden_states = llm_hidden_states.detach().clone()
            # Run the real action head so the VLA can finish its forward pass
            return _outer._original_predict(self_head, llm_hidden_states, pixel_values)

        # Monkey-patch at class level (affects all instances, but we only have one)
        self._action_head.__class__.predict_action = _intercepting_predict
        return self

    def __exit__(self, *exc):
        self._action_head.__class__.predict_action = self._original_predict
        return False


# ---------------------------------------------------------------------------
# Model loading (reuses existing infrastructure)
# ---------------------------------------------------------------------------
def load_model(
    pretrained_checkpoint: str,
    lora_rank: int = 32,
    action_head_vision_encoder: str = "siglip-base",
    num_views: int = 2,
) -> Tuple[nn.Module, VisionActionHead, any, "argparse.Namespace"]:
    """
    Returns:
        vla, action_head, processor, cfg
    """
    # Build minimal config namespace
    import types
    cfg = types.SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=pretrained_checkpoint,
        lora_rank=lora_rank,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=num_views,
        use_proprio=True,
        center_crop=True,
        load_in_8bit=False,
        load_in_4bit=False,
        use_vision_action_head=True,
        action_head_vision_encoder=action_head_vision_encoder,
        freeze_action_head_vision=True,
        action_head_num_views=num_views,
    )

    from experiments.robot.openvla_utils import get_action_head, get_processor, get_vla
    from experiments.robot.robot_utils import get_model

    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)

    return vla, action_head, processor, cfg


# ---------------------------------------------------------------------------
# Observation collection from LIBERO
# ---------------------------------------------------------------------------
def collect_observations(
    task_suite_name: str,
    num_episodes: int = 3,
    num_frames: int = 40,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """
    Collect raw observations from LIBERO environments.

    Returns:
        dict  task_description -> list of obs dicts
            Each dict: full_image (H,W,3), wrist_image (H,W,3), state (8,), task_label
    """
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action, get_libero_env, get_libero_image,
        get_libero_wrist_image, quat2axisangle,
    )
    from experiments.robot.openvla_utils import resize_image_for_policy

    np.random.seed(seed)

    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    all_obs = {}

    for task_id in range(min(task_suite.n_tasks, 5)):
        task = task_suite.get_task(task_id)
        env, task_desc = get_libero_env(task, "openvla", resolution=256)
        init_states = task_suite.get_task_init_states(task_id)

        task_obs = []
        for ep in range(min(num_episodes, len(init_states))):
            env.reset()
            obs = env.set_init_state(init_states[ep])

            for step in range(num_frames):
                if step < 10:  # let objects stabilise
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                    continue

                img  = get_libero_image(obs)
                wimg = get_libero_wrist_image(obs)
                img_r  = resize_image_for_policy(img,  224)
                wimg_r = resize_image_for_policy(wimg, 224)
                state  = np.concatenate((
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ))

                task_obs.append({
                    "full_image":   img_r,
                    "wrist_image":  wimg_r,
                    "state":        state,
                    "task_label":   task_desc,
                })

                obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                if done:
                    break
        all_obs[task_desc] = task_obs
        env.close()

    return all_obs


# ---------------------------------------------------------------------------
# Build pixel_values from raw observation (reuses VLA processor)
# ---------------------------------------------------------------------------
def obs_to_pixel_values(
    obs: dict,
    processor,
    cfg,
) -> torch.Tensor:
    """
    Convert a raw obs dict into the pixel_values tensor expected by the VLA.

    Returns:
        pixel_values: (1, C, H, W) tensor on CPU (will be moved to DEVICE later)
    """
    from PIL import Image
    from experiments.robot.openvla_utils import center_crop_image

    images = [obs["full_image"]]
    if cfg.num_images_in_input > 1 and "wrist_image" in obs:
        images.append(obs["wrist_image"])

    processed = []
    for img in images:
        pil = Image.fromarray(img).convert("RGB")
        if cfg.center_crop:
            pil = center_crop_image(pil)
        processed.append(pil)

    primary = processed.pop(0)
    prompt = f"In: What action should the robot take to {obs['task_label'].lower()}?\nOut:"
    inputs = processor(prompt, primary)
    pv = inputs["pixel_values"]  # (1, C, H, W)

    if processed:
        wrist_pvs = [processor(prompt, wimg)["pixel_values"] for wimg in processed]
        pv = torch.cat([pv] + wrist_pvs, dim=1)

    return pv  # (1, num_views*C, H, W) on CPU


# ---------------------------------------------------------------------------
# Attribution computation
# ---------------------------------------------------------------------------
def compute_gradient_attribution(
    action_head: VisionActionHead,
    h_stale: torch.Tensor,
    pixel_values_current: torch.Tensor,
) -> float:
    """
    Gradient-based effective attribution:

        α(d) = ‖∂â/∂h‖ / (‖∂â/∂h‖ + ‖∂â/∂z‖)

    High α → planning features dominate (small delay).
    Low  α → vision features dominate (large delay, planning stale).
    """
    was_training = action_head.training
    action_head.train()  # enable grad through fusion_mlp

    B = h_stale.shape[0]

    # --- LLM path: mean-pool over action_dim axis ---
    llm_features = (
        h_stale.reshape(B, NUM_ACTIONS_CHUNK, ACTION_DIM, -1)
        .mean(dim=2)                        # (B, chunk, D)
    )
    llm_features = llm_features.detach().requires_grad_(True)

    # --- Vision path ---
    with torch.no_grad():
        vision_raw = action_head.encode_vision(pixel_values_current)  # (B, vis_dim)
    vision_raw = vision_raw.detach().requires_grad_(True)
    vision_proj = action_head.vision_projector(vision_raw)             # (B, D)
    vision_proj = vision_proj.unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)

    # --- Fusion + prediction ---
    fused = torch.cat([llm_features, vision_proj], dim=-1)            # (B, chunk, 2D)
    actions = action_head.fusion_mlp(fused)                           # (B, chunk, act_dim)

    # Scalar target for backward
    actions.abs().sum().backward()

    g_h = llm_features.grad
    g_z = vision_raw.grad

    nh = g_h.norm().item() if g_h is not None else 0.0
    nz = g_z.norm().item() if g_z is not None else 0.0

    alpha = nh / (nh + nz + 1e-8)

    action_head.train(was_training)
    return alpha


def compute_ablation_attribution(
    action_head: VisionActionHead,
    h_stale: torch.Tensor,
    pixel_values_current: torch.Tensor,
) -> float:
    """
    Input-ablation effective attribution:

        α(d) = 1 - ‖â(h,0) - â(h,z)‖ / (‖â(h,z)‖ + ε)
    """
    action_head.eval()
    with torch.no_grad():
        a_vision  = action_head.predict_action(h_stale, pixel_values=pixel_values_current)
        a_no_vision = action_head.predict_action(h_stale, pixel_values=None)
        diff = (a_vision - a_no_vision).abs().sum().item()
        norm = a_vision.abs().sum().item() + 1e-8
        alpha = 1.0 - diff / norm
    return alpha


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_attribution(
    pretrained_checkpoint: str,
    task_suite_name: str = "libero_spatial",
    delays: Optional[List[int]] = None,
    method: str = "gradient",
    num_episodes: int = 3,
    num_frames: int = 40,
    max_samples_per_delay: int = 30,
    lora_rank: int = 32,
    action_head_vision_encoder: str = "siglip-base",
    num_views: int = 2,
) -> Dict[int, List[float]]:
    """
    Run the full attribution sweep over delays for one task suite.

    Returns:
        dict  delay -> list of α values
    """
    if delays is None:
        delays = DEFAULT_DELAYS

    print(f"[INFO] Loading model from {pretrained_checkpoint} ...")
    vla, action_head, processor, cfg = load_model(
        pretrained_checkpoint, lora_rank, action_head_vision_encoder, num_views,
    )

    print(f"[INFO] Collecting LIBERO observations ({task_suite_name}) ...")
    all_obs = collect_observations(task_suite_name, num_episodes, num_frames)

    # Pre-build a simple cfg namespace for obs_to_pixel_values
    img_cfg = types.SimpleNamespace(num_images_in_input=num_views, center_crop=cfg.center_crop)

    results = defaultdict(list)

    from transformers import LlamaTokenizerFast
    tokenizer = processor.tokenizer

    for task_desc, obs_list in all_obs.items():
        print(f"  [TASK] {task_desc}  ({len(obs_list)} frames)")
        n = len(obs_list)

        # Pre-compute pixel_values for every frame to avoid redundant processing
        pvs = [obs_to_pixel_values(o, processor, img_cfg) for o in obs_list]

        for delay in delays:
            count = 0
            # Slide a window over the episode
            for t in range(delay, n):
                if count >= max_samples_per_delay:
                    break
                count += 1

                pv_current = pvs[t].to(DEVICE, dtype=torch.bfloat16)
                pv_delayed = pvs[t - delay].to(DEVICE, dtype=torch.bfloat16)

                # Build prompt input_ids (same as VLA's predict_action)
                prompt = f"In: What action should the robot take to {obs_list[t]['task_label'].lower()}?\nOut:"
                input_ids = tokenizer(prompt, truncation=True, return_tensors="pt").input_ids.to(DEVICE)

                # --- Capture hidden states from delayed frame ---
                with _HiddenStateCapture(action_head) as cap:
                    with torch.inference_mode():
                        vla.predict_action(
                            input_ids=input_ids,
                            pixel_values=pv_delayed,
                            attention_mask=torch.ones_like(input_ids),
                            unnorm_key=None,
                            action_head=action_head,
                        )
                h_stale = cap.hidden_states  # (1, L, D), on DEVICE, bf16

                # --- Compute attribution ---
                if method == "gradient":
                    alpha = compute_gradient_attribution(action_head, h_stale, pv_current)
                else:
                    alpha = compute_ablation_attribution(action_head, h_stale, pv_current)

                results[delay].append(alpha)

    return dict(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(
    results: Dict[int, List[float]],
    output_path: str,
    method: str = "gradient",
    title_suffix: str = "",
) -> None:
    """Line plot: α(d) and 1-α(d) vs. delay with ±1 std shading."""
    delays = sorted(results.keys())
    means  = [np.mean(results[d]) for d in delays]
    stds   = [np.std(results[d])  for d in delays]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Planning attribution
    ax.plot(delays, means, "o-", color="#2563EB", lw=2.5, ms=8,
            label=r"Planning features $\alpha_{\mathrm{eff}}(d)$", zorder=3)
    ax.fill_between(delays,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.20, color="#2563EB", zorder=2)

    # Vision attribution (complement)
    vmeans = [1.0 - m for m in means]
    ax.plot(delays, vmeans, "s--", color="#DC2626", lw=2.0, ms=7,
            label=r"Vision features $1 - \alpha_{\mathrm{eff}}(d)$", zorder=3)
    ax.fill_between(delays,
                    [m - s for m, s in zip(vmeans, stds)],
                    [m + s for m, s in zip(vmeans, stds)],
                    alpha=0.15, color="#DC2626", zorder=2)

    ax.set_xlabel("Delay $d$ (environment steps)", fontsize=13)
    if method == "gradient":
        ax.set_ylabel(
            r"$\alpha_{\mathrm{eff}} = \|\nabla_h \hat{a}\|\,/\,"
            r"(\|\nabla_h \hat{a}\| + \|\nabla_z \hat{a}\|)$",
            fontsize=12,
        )
    else:
        ax.set_ylabel(
            r"$\alpha_{\mathrm{eff}} = 1 - \|\hat{a}(h,0) - \hat{a}(h,z)\|"
            r"\,/\,(\|\hat{a}(h,z)\| + \epsilon)$",
            fontsize=12,
        )

    ax.set_title(f"Effective Input Attribution vs. Delay{title_suffix}", fontsize=14)
    ax.legend(fontsize=12, loc="center right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[INFO] Saved → {output_path}")
    plt.close(fig)


def plot_grouped_bars(
    all_results: Dict[str, Dict[int, List[float]]],
    output_path: str,
    method: str = "gradient",
) -> None:
    """Grouped bar chart: one group per delay, bars coloured by task suite."""
    suites  = list(all_results.keys())
    delays  = sorted({d for r in all_results.values() for d in r})
    n_suites = len(suites)
    bw = 0.8 / n_suites
    colors = ["#2563EB", "#DC2626", "#16A34A", "#F59E0B"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, suite in enumerate(suites):
        r = all_results[suite]
        means = [np.mean(r.get(d, [0])) for d in delays]
        stds  = [np.std(r.get(d, [0]))  for d in delays]
        x = np.arange(len(delays)) + i * bw
        ax.bar(x, means, bw, yerr=stds,
               label=TASK_SUITE_LABELS.get(suite, suite),
               color=colors[i % len(colors)], alpha=0.85, capsize=3)

    ax.set_xlabel("Delay $d$ (environment steps)", fontsize=13)
    ax.set_ylabel(r"$\alpha_{\mathrm{eff}}(d)$ (planning attribution)", fontsize=12)
    ax.set_title("Effective Planning Attribution by Task Suite and Delay", fontsize=14)
    ax.set_xticks(np.arange(len(delays)) + bw * (n_suites - 1) / 2)
    ax.set_xticklabels(delays)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[INFO] Saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(results: Dict[int, List[float]], label: str = "") -> None:
    if label:
        print(f"\n{label}:")
    for d in sorted(results):
        v = results[d]
        print(f"  delay={d:3d}  α = {np.mean(v):.4f} ± {np.std(v):.4f}  (n={len(v)})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize effective input attribution α(d) for VisionActionHead",
    )
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        choices=TASK_SUITE_NAMES + ["all"])
    parser.add_argument("--method", type=str, default="gradient",
                        choices=["gradient", "ablation"])
    parser.add_argument("--delays", type=int, nargs="+", default=None)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=40)
    parser.add_argument("--max_samples_per_delay", type=int, default=30)
    parser.add_argument("--output_path", type=str, default="results/fig6_attribution.pdf")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--action_head_vision_encoder", type=str, default="siglip-base")
    parser.add_argument("--num_views", type=int, default=2)
    args = parser.parse_args()

    delays = args.delays if args.delays is not None else DEFAULT_DELAYS

    if args.task_suite_name == "all":
        all_res = {}
        for suite in TASK_SUITE_NAMES:
            print(f"\n{'='*60}\n[SUITE] {suite}\n{'='*60}")
            all_res[suite] = run_attribution(
                pretrained_checkpoint=args.pretrained_checkpoint,
                task_suite_name=suite,
                delays=delays,
                method=args.method,
                num_episodes=args.num_episodes,
                num_frames=args.num_frames,
                max_samples_per_delay=args.max_samples_per_delay,
                lora_rank=args.lora_rank,
                action_head_vision_encoder=args.action_head_vision_encoder,
                num_views=args.num_views,
            )
        plot_grouped_bars(all_res, args.output_path, args.method)
        for suite, r in all_res.items():
            print_summary(r, TASK_SUITE_LABELS.get(suite, suite))
    else:
        results = run_attribution(
            pretrained_checkpoint=args.pretrained_checkpoint,
            task_suite_name=args.task_suite_name,
            delays=delays,
            method=args.method,
            num_episodes=args.num_episodes,
            num_frames=args.num_frames,
            max_samples_per_delay=args.max_samples_per_delay,
            lora_rank=args.lora_rank,
            action_head_vision_encoder=args.action_head_vision_encoder,
            num_views=args.num_views,
        )
        plot_curves(results, args.output_path, args.method,
                     title_suffix=f" ({TASK_SUITE_LABELS.get(args.task_suite_name, args.task_suite_name)})")
        print_summary(results)


# Avoid name collision with the types import inside load_model
import types  # noqa: E402  (needed at module level for img_cfg in run_attribution)


if __name__ == "__main__":
    main()
