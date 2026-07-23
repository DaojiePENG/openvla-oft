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
import torchvision.transforms.functional as _TVF

# Importing prismatic.vla pulls in TensorFlow through the RLDS package. Disable
# TensorFlow GPU discovery before that import so it cannot reserve H800 memory.
import tensorflow as _tf
try:
    _tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

# Monkey-patch to_tensor to work with numpy 1.26 + torch 2.3 ABI mismatch
# (torch.from_numpy and torch.tensor(numpy) both fail due to ABI incompatibility)
_orig_to_tensor = _TVF.to_tensor
def _patched_to_tensor(pic):
    import numpy as np
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    nptype = mode_to_nptype.get(pic.mode, np.uint8)
    img = np.asarray(pic, dtype=nptype)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    img = img.transpose((2, 0, 1))
    t = torch.tensor(img.tolist(), dtype=torch.float32 if nptype == np.uint8 else None)
    if nptype == np.uint8:
        t = t.div_(255)
    return t
_TVF.to_tensor = _patched_to_tensor

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


def configure_runtime(device: str) -> None:
    """Select the torch device and keep TensorFlow away from GPU memory."""
    global DEVICE
    DEVICE = torch.device(device)
    if DEVICE.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        torch.cuda.set_device(DEVICE)

    try:
        _tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass

    # openvla_utils has its own module-level DEVICE; keep it in sync.
    import experiments.robot.openvla_utils as openvla_utils
    openvla_utils.DEVICE = DEVICE
    print(f"[INFO] Torch device: {DEVICE}")


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

    # Monkey-patch _unnormalize_actions to use pure torch (bypass numpy ABI issues)
    def _torch_unnormalize_actions(self, normalized_actions, unnorm_key):
        action_norm_stats = self.get_action_stats(unnorm_key)
        device = normalized_actions.device if hasattr(normalized_actions, 'device') else torch.device("cuda:0")
        na = torch.tensor(normalized_actions.tolist(), dtype=torch.float32, device=device) if hasattr(normalized_actions, 'tolist') else torch.tensor(normalized_actions, dtype=torch.float32, device=device)
        action_low = torch.as_tensor(list(action_norm_stats["q01"]), dtype=torch.float32, device=device)
        action_high = torch.as_tensor(list(action_norm_stats["q99"]), dtype=torch.float32, device=device)
        mask = torch.as_tensor(list(action_norm_stats.get("mask", [1]*len(action_low))), dtype=torch.bool, device=device)
        actions = torch.where(
            mask,
            0.5 * (na + 1) * (action_high - action_low + 1e-8) + action_low,
            na,
        )
        return actions
    vla._unnormalize_actions = types.MethodType(_torch_unnormalize_actions, vla)

    return vla, action_head, processor, cfg


# ───────────────────────────────────────────────────────────────────────────
# LIBERO observation collection
# ───────────────────────────────────────────────────────────────────────────
def collect_observations_with_labels(
    task_suite_name: str,
    num_episodes: int = 5,
    num_frames: int = 60,
    seed: int = 42,
    max_tasks: int = 5,
    rollout_mode: str = "random",
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

    rng = np.random.default_rng(seed)
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    all_data: Dict[str, List[dict]] = {}

    num_tasks = min(task_suite.n_tasks, max_tasks)
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        env, task_desc = get_libero_env(task, "openvla", resolution=256)
        init_states = task_suite.get_task_init_states(task_id)

        frames: List[dict] = []
        for ep in range(min(num_episodes, len(init_states))):
            env.reset()
            obs = env.set_init_state(init_states[ep])
            random_action = np.zeros(ACTION_DIM, dtype=np.float32)

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
                # Direct object positions are cleaner than enumerating MuJoCo bodies,
                # which also includes fixtures and duplicate child bodies.
                object_pos_keys = sorted(
                    key for key in obs
                    if key.endswith("_pos") and not key.startswith("robot") and "_to_robot" not in key
                )
                if object_pos_keys:
                    object_state_high = np.concatenate([obs[key] for key in object_pos_keys])
                elif "object-state" in obs:
                    object_state_high = np.asarray(obs["object-state"])
                else:
                    raise RuntimeError("LIBERO observation has no object position labels")

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
                    "task_id":          task_id,
                    "episode_id":       ep,
                    "timestep":         step,
                })

                if rollout_mode == "random":
                    target = rng.normal(0.0, 0.35, size=ACTION_DIM)
                    target[3:6] *= 0.6
                    random_action = np.clip(0.75 * random_action + 0.25 * target, -1.0, 1.0)
                    random_action[-1] = -1.0 if ((step - 10) // 6) % 2 == 0 else 1.0
                    action = random_action
                else:
                    action = get_libero_dummy_action("openvla")
                obs, _, done, _ = env.step(action)
                if done:
                    break

        all_data[task_desc] = frames
        env.close()
        print(
            f"  [COLLECT] Task {task_id + 1}/{num_tasks}: {task_desc} "
            f"({len(frames)} frames)",
            flush=True,
        )

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
    unnorm_key: Optional[str] = None,
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
                    unnorm_key=unnorm_key,
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

    # Exclude dimensions that are effectively constant in either split. Their
    # R² denominator is near zero and used to dominate the mean with huge
    # negative numerical artifacts (the original cause of the empty panel).
    valid_dims = (y_train.std(axis=0) > 1e-5) & (y_test.std(axis=0) > 1e-5)
    if not np.any(valid_dims):
        raise ValueError("No target dimensions have enough variance for an R² probe")
    y_train = y_train[:, valid_dims]
    y_test = y_test[:, valid_dims]

    # Standardise targets too (makes multi-dimensional R² meaningful)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std  = y_train.std(axis=0, keepdims=True) + 1e-8
    y_train_s = (y_train - y_mean) / y_std
    y_test_s  = (y_test  - y_mean) / y_std

    probe = Ridge(alpha=1.0, solver="lsqr")
    probe.fit(X_train_s, y_train_s)

    y_pred = probe.predict(X_test_s)
    ss_res = np.sum((y_test_s - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_test_s - y_test_s.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_dim = 1.0 - ss_res / (ss_tot + 1e-12)

    full_r2 = np.full(valid_dims.shape, np.nan, dtype=np.float32)
    full_r2[valid_dims] = r2_per_dim
    return float(np.mean(r2_per_dim)), full_r2


def run_probe_analysis(
    checkpoint_standard: str,
    checkpoint_ours: str,
    task_suite_name: str,
    num_episodes: int = 5,
    num_frames: int = 60,
    lora_rank: int = 32,
    test_ratio: float = 0.3,
    seed: int = 42,
    max_tasks: int = 5,
    rollout_mode: str = "random",
    split_mode: str = "frame",
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
    all_obs = collect_observations_with_labels(
        task_suite_name, num_episodes, num_frames, seed, max_tasks, rollout_mode
    )

    # Flatten across tasks
    flat_obs = [frame for task_frames in all_obs.values() for frame in task_frames]
    print(f"  Total frames collected: {len(flat_obs)}")

    # unnorm_key for action unnormalization (probe only needs hidden states, but
    # predict_action requires it)
    unnorm_key = f"{task_suite_name}_no_noops"

    # 2. Load Standard VLA and extract hidden states
    print(f"[INFO] Loading Standard VLA from {checkpoint_standard} ...")
    vla_std, ah_std, proc_std, cfg_std = load_vla(checkpoint_standard, use_vision_action_head=False, lora_rank=lora_rank)
    print("[INFO] Extracting hidden states (Standard VLA) ...")
    hs_std, robot_lbl, object_lbl, progress_lbl = extract_hidden_states(
        vla_std, ah_std, proc_std, cfg_std, flat_obs, unnorm_key=unnorm_key
    )
    del vla_std, ah_std  # free GPU memory
    torch.cuda.empty_cache()

    # 3. Load Our VLA and extract hidden states
    print(f"[INFO] Loading CloudEdgeVLA from {checkpoint_ours} ...")
    vla_ours, ah_ours, proc_ours, cfg_ours = load_vla(checkpoint_ours, use_vision_action_head=True, lora_rank=lora_rank)
    print("[INFO] Extracting hidden states (CloudEdgeVLA) ...")
    hs_ours, _, _, _ = extract_hidden_states(
        vla_ours, ah_ours, proc_ours, cfg_ours, flat_obs, unnorm_key=unnorm_key
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
    rng = np.random.default_rng(seed)
    N = len(flat_obs)
    test_mask = np.zeros(N, dtype=bool)
    if split_mode == "trajectory":
        # Split whole trajectories, while retaining every task in both splits.
        for task_id in sorted({frame["task_id"] for frame in flat_obs}):
            episodes = sorted({
                frame["episode_id"] for frame in flat_obs if frame["task_id"] == task_id
            })
            if len(episodes) < 2:
                continue
            episodes = list(rng.permutation(episodes))
            n_test_eps = min(len(episodes) - 1, max(1, int(round(len(episodes) * test_ratio))))
            selected = set(episodes[:n_test_eps])
            for idx, frame in enumerate(flat_obs):
                if frame["task_id"] == task_id and frame["episode_id"] in selected:
                    test_mask[idx] = True
    if split_mode == "frame" or not test_mask.any():
        if split_mode == "trajectory":
            print("[WARN] Only one trajectory per task; falling back to a frame-level split")
        perm = rng.permutation(N)
        test_mask[perm[:max(1, int(N * test_ratio))]] = True
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)

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
                    "(b) Low-Level Robot State\n(EEF pose + Gripper state)"]

    colors = {"Standard VLA": "#94A3B8", "CloudEdgeVLA": "#2563EB"}
    all_panel_values = [
        all_results[suite][model][key]
        for suite in suite_labels
        for model in ("Standard VLA", "CloudEdgeVLA")
        for key in ("high_level", "low_level")
    ]
    global_lower = min(0.0, min(all_panel_values))
    global_upper = max(0.0, max(all_panel_values))
    global_margin = max(0.08, 0.12 * max(global_upper - global_lower, 0.1))

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
        ax.set_ylim(global_lower - global_margin, global_upper + global_margin)
        ax.axhline(0.0, color="#475569", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if Path(output_path).suffix.lower() != ".png":
        png_path = str(Path(output_path).with_suffix(".png"))
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved → {png_path}")
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
            "max_tasks": args.max_tasks,
            "rollout_mode": args.rollout_mode,
            "device": args.device,
            "split_mode": args.split_mode,
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
    def make_json_safe(value):
        if isinstance(value, dict):
            return {key: make_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [make_json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if hasattr(value, "tolist"):
            return make_json_safe(value.tolist())
        return value

    with open(json_path, "w") as f:
        json.dump(make_json_safe(export), f, indent=2, allow_nan=False)
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
    parser.add_argument("--max_tasks", type=int, default=5)
    parser.add_argument("--rollout_mode", choices=["random", "noop"], default="random")
    parser.add_argument("--split_mode", choices=["frame", "trajectory"], default="frame")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configure_runtime(args.device)

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
                max_tasks=args.max_tasks,
                rollout_mode=args.rollout_mode,
                split_mode=args.split_mode,
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
            max_tasks=args.max_tasks,
            rollout_mode=args.rollout_mode,
            split_mode=args.split_mode,
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
