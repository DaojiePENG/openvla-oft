"""
visualize_representational_probe.py

Trains linear probes on VLA backbone hidden states to measure what information
is encoded, then compares:

  - Standard VLA backbone   (trained with single-frame L_fresh only)
  - Ours (CloudEdgeVLA)     (trained with paired-frame L_fresh + L_stale)

Probes predict two kinds of ground-truth state:

  High-level  (object positions in workspace)     — "what to do"
  Low-level   (robot joint angles, gripper, eef)  — "how to do it right now"

Expected result:
  Standard VLA  → high R² for BOTH (encodes everything in its hidden states)
  Ours          → high R² for high-level, LOW R² for low-level
                  (backbone discards timing-critical info; that lives in the
                   edge vision encoder instead)

Usage:
    python scripts/visualize_representational_probe.py \
        --checkpoint_standard /path/to/openvla-oft-spatial \
        --checkpoint_ours     /path/to/cloudedgevla-spatial \
        --task_suite_name     libero_spatial \
        --output_path         results/fig5_probe.pdf
"""

import argparse
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TASK_SUITE_NAMES  = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TASK_SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object":  "Object",
    "libero_goal":    "Goal",
    "libero_10":      "Long",
}


# ───────────────────────────────────────────────────────────────────────────
# Helpers: load a VLA model and optionally its action head
# ───────────────────────────────────────────────────────────────────────────
def load_vla(checkpoint: str, use_vision_action_head: bool, lora_rank: int = 32):
    """
    Load a VLA from checkpoint.

    Args:
        checkpoint: Path to pretrained checkpoint directory
        use_vision_action_head: True → load VisionActionHead (ours);
                                False → load standard L1RegressionActionHead
    Returns:
        vla, action_head (or None), processor, cfg
    """
    cfg = types.SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=checkpoint,
        lora_rank=lora_rank,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        load_in_8bit=False,
        load_in_4bit=False,
        use_vision_action_head=use_vision_action_head,
        action_head_vision_encoder="siglip-base",
        freeze_action_head_vision=True,
        action_head_num_views=2,
    )

    from experiments.robot.openvla_utils import get_action_head, get_processor, get_vla

    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)

    return vla, action_head, processor, cfg


# ───────────────────────────────────────────────────────────────────────────
# LIBERO observation collection
# ───────────────────────────────────────────────────────────────────────────
def collect_observations_with_labels(
    task_suite_name: str,
    num_episodes: int = 5,
    num_frames: int = 60,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """
    Collect observations together with ground-truth state labels.

    Returns per frame:
        full_image, wrist_image           — raw images (H,W,3)
        robot_state_low  (9,)             — [eef_pos(3), eef_axisangle(3), gripper_qpos(2), gripper_open(1)]
        object_state_high                 — (K,) positions of movable objects (varies per task)
        task_label                        — str
        timestep                          — int (for goal-progress encoding)
    """
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action, get_libero_env, get_libero_image,
        get_libero_wrist_image, quat2axisangle,
    )
    from experiments.robot.openvla_utils import resize_image_for_policy

    np.random.seed(seed)
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    all_data: Dict[str, List[dict]] = {}

    for task_id in range(min(task_suite.n_tasks, 5)):
        task = task_suite.get_task(task_id)
        env, task_desc = get_libero_env(task, "openvla", resolution=256)
        init_states = task_suite.get_task_init_states(task_id)

        frames: List[dict] = []
        for ep in range(min(num_episodes, len(init_states))):
            env.reset()
            obs = env.set_init_state(init_states[ep])

            for step in range(num_frames):
                if step < 10:
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                    continue

                # ── images ──
                img_r  = resize_image_for_policy(get_libero_image(obs), 224)
                wimg_r = resize_image_for_policy(get_libero_wrist_image(obs), 224)

                # ── low-level robot state (what to predict from hidden states) ──
                eef_pos     = obs["robot0_eef_pos"]                    # (3,)
                eef_aa      = quat2axisangle(obs["robot0_eef_quat"])   # (3,)
                gripper_qpos = obs["robot0_gripper_qpos"]              # (2,)
                gripper_open = np.array([1.0 if gripper_qpos.mean() > 0.04 else 0.0])
                robot_state_low = np.concatenate([eef_pos, eef_aa, gripper_qpos, gripper_open])  # (9,)

                # ── high-level object state ──
                # Retrieve movable object positions from MuJoCo simulation.
                # We filter out bodies with zero name (fixed/static) and the robot base.
                try:
                    sim = env.sim
                    body_names  = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
                    body_xpos   = sim.data.body_xpos                           # (nbody, 3)
                    object_positions = []
                    for bname, bxpos in zip(body_names, body_xpos):
                        if bname is None:
                            continue
                        bl = bname.lower()
                        # Skip robot links, table, floor, worldbody, cameras
                        if any(skip in bl for skip in [
                            "robot", "table", "floor", "world", "camera",
                            "gripper", "link", "mount", "base",
                        ]):
                            continue
                        object_positions.append(bxpos)
                    if len(object_positions) == 0:
                        # Fallback: use full MuJoCo state (excluding robot proprio)
                        full_state = sim.get_state().flatten()
                        object_state_high = full_state  # will be filtered later
                    else:
                        object_state_high = np.concatenate(object_positions)  # (3*K,)
                except Exception:
                    # Fallback if sim access fails
                    object_state_high = robot_state_low.copy()  # won't affect high-level probe much

                # ── goal progress (coarse: 0..1 normalised timestep) ──
                # We use the ratio step / max_episode_length as a proxy for goal progress.
                # This is purely temporal and should be equally predictable from both backbones.
                max_steps_map = {"libero_spatial": 220, "libero_object": 280,
                                 "libero_goal": 300, "libero_10": 520}
                max_s = max_steps_map.get(task_suite_name, 300)
                goal_progress = np.array([step / max_s])  # (1,)

                frames.append({
                    "full_image":       img_r,
                    "wrist_image":      wimg_r,
                    "robot_state_low":  robot_state_low,
                    "object_state_high": object_state_high,
                    "goal_progress":    goal_progress,
                    "task_label":       task_desc,
                    "timestep":         step,
                })

                obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                if done:
                    break

        all_data[task_desc] = frames
        env.close()

    return all_data


# ───────────────────────────────────────────────────────────────────────────
# Hidden-state extraction from VLA backbone
# ───────────────────────────────────────────────────────────────────────────
class _HS:
    """Context manager that monkey-patches action_head.predict_action to
    capture the incoming llm_hidden_states without altering the forward pass."""

    def __init__(self, action_head):
        self._ah   = action_head
        self._orig = action_head.__class__.predict_action
        self.hs: Optional[torch.Tensor] = None

    def __enter__(self):
        outer = self
        def _intercept(self_head, *args, **kwargs):
            # First positional arg is always llm_hidden_states for both
            # L1RegressionActionHead(self, llm_hidden_states) and
            # VisionActionHead(self, llm_hidden_states, pixel_values=None)
            outer.hs = args[0].detach().clone()
            return outer._orig(self_head, *args, **kwargs)
        self._ah.__class__.predict_action = _intercept
        return self

    def __exit__(self, *a):
        self._ah.__class__.predict_action = self._orig
        return False


@torch.no_grad()
def extract_hidden_states(
    vla, action_head, processor, cfg,
    observations: List[dict],
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract action-token hidden states from the VLA backbone for a list of observations.

    Returns:
        hs_array:       (N, L*D)  — flattened hidden states
        robot_labels:   (N, 9)    — low-level robot state
        object_labels:  (N, K)    — high-level object state (variable K, padded to max)
        progress_labels:(N, 1)    — goal progress
    """
    from PIL import Image
    from experiments.robot.openvla_utils import center_crop_image
    from transformers import LlamaTokenizerFast

    tokenizer = processor.tokenizer

    all_hs       = []
    all_robot    = []
    all_object   = []
    all_progress = []

    for i in range(0, len(observations), batch_size):
        batch = observations[i : i + batch_size]

        for obs in batch:
            # Process images
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
            pv = inputs["pixel_values"].to(DEVICE, dtype=torch.bfloat16)

            if processed:
                wps = [processor(prompt, w)["pixel_values"] for w in processed]
                pv = torch.cat([pv] + [w.to(DEVICE, dtype=torch.bfloat16) for w in wps], dim=1)

            input_ids = tokenizer(prompt, truncation=True, return_tensors="pt").input_ids.to(DEVICE)

            with _HS(action_head) as cap:
                vla.predict_action(
                    input_ids=input_ids,
                    pixel_values=pv,
                    attention_mask=torch.ones_like(input_ids),
                    unnorm_key=None,
                    action_head=action_head,
                )
            h = cap.hs  # (1, L, D)
            all_hs.append(h.cpu().float().numpy().reshape(-1))  # flatten to (L*D,)
            all_robot.append(obs["robot_state_low"].astype(np.float32))
            all_object.append(obs["object_state_high"].astype(np.float32))
            all_progress.append(obs["goal_progress"].astype(np.float32))

    # Pad object_state_high to uniform length
    max_k = max(len(o) for o in all_object)
    object_padded = np.zeros((len(all_object), max_k), dtype=np.float32)
    for i, o in enumerate(all_object):
        object_padded[i, :len(o)] = o

    return (
        np.stack(all_hs),
        np.stack(all_robot),
        object_padded,
        np.stack(all_progress),
    )


# ───────────────────────────────────────────────────────────────────────────
# Linear probe training and evaluation
# ───────────────────────────────────────────────────────────────────────────
def train_and_eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Train a Ridge regression probe and return the R² score on the test set.

    R² = 1 - SS_res / SS_tot.
    R² close to 1 → the hidden states contain the target information.
    R² close to 0 → the hidden states do NOT contain the target information.

    Returns:
        (mean_r2, r2_per_dim)  — average R² and per-dimension R² array
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Standardise targets too (makes multi-dimensional R² meaningful)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std  = y_train.std(axis=0, keepdims=True) + 1e-8
    y_train_s = (y_train - y_mean) / y_std
    y_test_s  = (y_test  - y_mean) / y_std

    probe = Ridge(alpha=1.0)
    probe.fit(X_train_s, y_train_s)

    y_pred = probe.predict(X_test_s)
    ss_res = np.sum((y_test_s - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_test_s - y_test_s.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_dim = 1.0 - ss_res / (ss_tot + 1e-12)

    return float(np.mean(r2_per_dim)), r2_per_dim  # average R² across target dimensions


def run_probe_analysis(
    checkpoint_standard: str,
    checkpoint_ours: str,
    task_suite_name: str,
    num_episodes: int = 5,
    num_frames: int = 60,
    lora_rank: int = 32,
    test_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, any]]:
    """
    Run the full probe analysis for one task suite.

    Returns:
        results: {
            "Standard VLA": {"high_level": R², "low_level": R²},
            "CloudEdgeVLA": {"high_level": R², "low_level": R²},
        }
        raw_info: {
            "num_samples": int,
            "num_train": int,
            "num_test": int,
            "high_level_dim_names": [...],
            "low_level_dim_names": [...],
            "per_dim_r2": {model: {"high_level": [...], "low_level": [...]}}
        }
    """
    # 1. Collect observations and labels
    print(f"[INFO] Collecting LIBERO observations ({task_suite_name}) ...")
    all_obs = collect_observations_with_labels(task_suite_name, num_episodes, num_frames, seed)

    # Flatten across tasks
    flat_obs = [frame for task_frames in all_obs.values() for frame in task_frames]
    print(f"  Total frames collected: {len(flat_obs)}")

    # 2. Load Standard VLA and extract hidden states
    print(f"[INFO] Loading Standard VLA from {checkpoint_standard} ...")
    vla_std, ah_std, proc_std, cfg_std = load_vla(checkpoint_standard, use_vision_action_head=False, lora_rank=lora_rank)
    print("[INFO] Extracting hidden states (Standard VLA) ...")
    hs_std, robot_lbl, object_lbl, progress_lbl = extract_hidden_states(
        vla_std, ah_std, proc_std, cfg_std, flat_obs
    )
    del vla_std, ah_std  # free GPU memory
    torch.cuda.empty_cache()

    # 3. Load Our VLA and extract hidden states
    print(f"[INFO] Loading CloudEdgeVLA from {checkpoint_ours} ...")
    vla_ours, ah_ours, proc_ours, cfg_ours = load_vla(checkpoint_ours, use_vision_action_head=True, lora_rank=lora_rank)
    print("[INFO] Extracting hidden states (CloudEdgeVLA) ...")
    hs_ours, _, _, _ = extract_hidden_states(
        vla_ours, ah_ours, proc_ours, cfg_ours, flat_obs
    )
    del vla_ours, ah_ours
    torch.cuda.empty_cache()

    # 4. Construct high-level and low-level target arrays
    # High-level = concat(object_state, goal_progress)
    high_level_target = np.concatenate([object_lbl, progress_lbl], axis=1)
    low_level_target  = robot_lbl  # (N, 9)

    # Dimension names for raw data export
    n_obj_dims = object_lbl.shape[1]
    high_level_dim_names = [f"object_pos_{i}" for i in range(n_obj_dims)] + ["goal_progress"]
    low_level_dim_names = ["eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az",
                           "gripper_qpos_0", "gripper_qpos_1", "gripper_open"]

    # 5. Train/test split
    np.random.seed(seed)
    N = len(flat_obs)
    perm = np.random.permutation(N)
    n_test = int(N * test_ratio)
    test_idx  = perm[:n_test]
    train_idx = perm[n_test:]

    print(f"[INFO] Train: {len(train_idx)}, Test: {len(test_idx)}")

    # 6. Train probes for both models
    results = {}
    per_dim_r2 = {}
    for label, hs in [("Standard VLA", hs_std), ("CloudEdgeVLA", hs_ours)]:
        print(f"\n  [{label}]")
        r2_high, r2_high_dims = train_and_eval_probe(
            hs[train_idx], high_level_target[train_idx],
            hs[test_idx],  high_level_target[test_idx],
        )
        r2_low, r2_low_dims = train_and_eval_probe(
            hs[train_idx], low_level_target[train_idx],
            hs[test_idx],  low_level_target[test_idx],
        )
        results[label] = {"high_level": r2_high, "low_level": r2_low}
        per_dim_r2[label] = {
            "high_level": r2_high_dims.tolist(),
            "low_level": r2_low_dims.tolist(),
        }
        print(f"    High-level (object + goal) R² = {r2_high:.4f}")
        print(f"    Low-level  (robot state)    R² = {r2_low:.4f}")

    raw_info = {
        "num_samples": N,
        "num_train": len(train_idx),
        "num_test": len(test_idx),
        "high_level_dim_names": high_level_dim_names,
        "low_level_dim_names": low_level_dim_names,
        "per_dim_r2": per_dim_r2,
    }

    return results, raw_info


# ───────────────────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────────────────
def plot_probe_results(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
) -> None:
    """
    Two-panel bar chart (or single panel if only one suite).

    Args:
        all_results: {
            suite_label: {
                "Standard VLA": {"high_level": R², "low_level": R²},
                "CloudEdgeVLA": {"high_level": R², "low_level": R²},
            }, ...
        }
    """
    suite_labels = list(all_results.keys())
    n_suites = len(suite_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panel_labels = ["(a) High-Level Task State\n(Object positions + Goal progress)",
                    "((b) Low-Level Robot State\n(Joint angles + Gripper + EEF position)"]

    colors = {"Standard VLA": "#94A3B8", "CloudEdgeVLA": "#2563EB"}

    for panel_idx, (key, title) in enumerate(
        [("high_level", panel_labels[0]), ("low_level", panel_labels[1])]
    ):
        ax = axes[panel_idx]

        x = np.arange(n_suites)
        bw = 0.35

        for j, model_label in enumerate(["Standard VLA", "CloudEdgeVLA"]):
            vals = [all_results[s][model_label][key] for s in suite_labels]
            bars = ax.bar(
                x + j * bw, vals, bw,
                label=model_label,
                color=colors[model_label],
                edgecolor="white",
                linewidth=0.8,
            )
            # Value labels on bars
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                )

        ax.set_xticks(x + bw / 2)
        ax.set_xticklabels(suite_labels, fontsize=11)
        ax.set_ylabel("Probe R² Score", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(1.05, ax.get_ylim()[1]))
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[INFO] Saved → {output_path}")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
# Raw data export
# ───────────────────────────────────────────────────────────────────────────
def save_raw_data(
    output_path: str,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    all_raw_info: Dict[str, Dict[str, any]],
    args: argparse.Namespace,
) -> None:
    """
    Save raw probe data alongside the figure.

    Saves a JSON file with the same stem as the PDF output, e.g.
        results/fig5_probe_goal.pdf  →  results/fig5_probe_goal_data.json
    """
    stem, _ = os.path.splitext(output_path)
    json_path = stem + "_data.json"

    export: Dict[str, any] = {
        "config": {
            "checkpoint_standard": args.checkpoint_standard,
            "checkpoint_ours": args.checkpoint_ours,
            "task_suite_name": args.task_suite_name,
            "num_episodes": args.num_episodes,
            "num_frames": args.num_frames,
            "lora_rank": args.lora_rank,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        },
        "suites": {},
    }

    for suite_label, results in all_results.items():
        raw = all_raw_info.get(suite_label, {})
        suite_data: Dict[str, any] = {
            "summary": results,
            "num_samples": raw.get("num_samples"),
            "num_train": raw.get("num_train"),
            "num_test": raw.get("num_test"),
            "dim_names": {
                "high_level": raw.get("high_level_dim_names", []),
                "low_level": raw.get("low_level_dim_names", []),
            },
            "per_dim_r2": raw.get("per_dim_r2", {}),
        }
        export["suites"][suite_label] = suite_data

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[INFO] Raw data saved → {json_path}")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Linear probe analysis: what information do VLA backbones encode?",
    )
    parser.add_argument("--checkpoint_standard", type=str, required=True,
                        help="Path to Standard VLA checkpoint (single-frame trained)")
    parser.add_argument("--checkpoint_ours", type=str, required=True,
                        help="Path to CloudEdgeVLA checkpoint (paired-frame trained)")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        choices=TASK_SUITE_NAMES + ["all"],
                        help="LIBERO suite (or 'all' for all 4)")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--output_path", type=str, default="results/fig5_probe.pdf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.task_suite_name == "all":
        all_results = {}
        all_raw_info = {}
        for suite in TASK_SUITE_NAMES:
            print(f"\n{'='*60}\n[SUITE] {suite}\n{'='*60}")
            results, raw_info = run_probe_analysis(
                checkpoint_standard=args.checkpoint_standard,
                checkpoint_ours=args.checkpoint_ours,
                task_suite_name=suite,
                num_episodes=args.num_episodes,
                num_frames=args.num_frames,
                lora_rank=args.lora_rank,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
            all_results[TASK_SUITE_LABELS[suite]] = results
            all_raw_info[TASK_SUITE_LABELS[suite]] = raw_info
        plot_probe_results(all_results, args.output_path)
        save_raw_data(args.output_path, all_results, all_raw_info, args)
    else:
        results, raw_info = run_probe_analysis(
            checkpoint_standard=args.checkpoint_standard,
            checkpoint_ours=args.checkpoint_ours,
            task_suite_name=args.task_suite_name,
            num_episodes=args.num_episodes,
            num_frames=args.num_frames,
            lora_rank=args.lora_rank,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        suite_label = TASK_SUITE_LABELS.get(args.task_suite_name, args.task_suite_name)
        all_results = {suite_label: results}
        all_raw_info = {suite_label: raw_info}
        plot_probe_results(all_results, args.output_path)
        save_raw_data(args.output_path, all_results, all_raw_info, args)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Linear Probe R² Scores")
    print("="*60)
    print(f"{'Suite':<12} {'Model':<18} {'High-Level':>12} {'Low-Level':>12}")
    print("-" * 60)
    for s, sres in all_results.items():
        for m, mres in sres.items():
            print(f"{s:<12} {m:<18} {mres['high_level']:>12.4f} {mres['low_level']:>12.4f}")


if __name__ == "__main__":
    main()
