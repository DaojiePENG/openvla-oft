#!/usr/bin/env python3
r"""Compress old OpenVLA-OFT checkpoints while keeping recent ones test-ready.

The training code stores both the learned components (LoRA adapter and heads)
and a ~14 GiB merged base model in every ``--<step>_chkpt`` directory.  This
script removes only the root-level merged Hugging Face model tensors from old
checkpoints.  It preserves the LoRA adapter, action/proprio/diffusion heads,
processor/config files, and a JSON manifest recording what was removed.

Checkpoints are grouped by their parent directory and the name before
``--<step>_chkpt``.  The newest two *complete merged* checkpoints in every
group are protected by default.  The newest two directory steps are also
protected even if a save is still incomplete, preventing a save-time race.

Safety properties:

* Dry-run is the default; files are deleted only with ``--apply``.
* A checkpoint is compressed only if its LoRA adapter and action head exist.
* Sharded models are considered complete only when their index and every
  referenced shard exist and are non-empty.
* Recently modified and incomplete checkpoints are skipped.
* Repeated runs are idempotent; already-compressed checkpoints are reported.
* An advisory lock prevents two apply processes from cleaning simultaneously.

Examples (run from the repository root):

    # Preview all decisions and estimated space savings (recommended first).
    python scripts/prune_merged_checkpoints.py

    # Apply the previewed cleanup.  Keep the newest two full checkpoints/group.
    python scripts/prune_merged_checkpoints.py --apply

    # Restrict cleanup to groups whose relative name contains "libero_goal".
    python scripts/prune_merged_checkpoints.py --group-regex libero_goal --apply

    # Use a different checkpoint root or retention count.
    python scripts/prune_merged_checkpoints.py \
        --runs-dir /path/to/openvla-oft/runs --keep-latest 2 --apply

To restore a compressed checkpoint, merge its preserved adapter into the same
base model used for training (also recorded in the compression manifest):

    python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint /path/to/the/original/base-model \
        --lora_finetuned_checkpoint_dir runs/...--10000_chkpt

The merge command recreates the removed ``model*.safetensors`` files in place.
It does not modify the separately saved action/proprio heads.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINT_RE = re.compile(r"^(?P<prefix>.+)--(?P<step>\d+)_chkpt$")
MANIFEST_NAME = "checkpoint_compression_manifest.json"
IN_PROGRESS_NAME = ".checkpoint_compression_in_progress.json"

# Only root-level Hugging Face model tensor payloads match these expressions.
# In particular, adapter_model.safetensors under lora_adapter/ cannot match.
MODEL_PAYLOAD_RES = (
    re.compile(r"^model\.safetensors$"),
    re.compile(r"^model\.safetensors\.index\.json$"),
    re.compile(r"^model-\d+-of-\d+\.safetensors$"),
    re.compile(r"^pytorch_model\.bin$"),
    re.compile(r"^pytorch_model\.bin\.index\.json$"),
    re.compile(r"^pytorch_model-\d+-of-\d+\.bin$"),
)
MODEL_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    prefix: str
    step: int
    group_key: str
    group_display: str


@dataclass(frozen=True)
class ModelState:
    kind: str
    payloads: tuple[Path, ...]
    reason: str


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def is_model_payload_name(name: str) -> bool:
    return any(pattern.fullmatch(name) for pattern in MODEL_PAYLOAD_RES)


def discover_checkpoints(runs_dir: Path) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    for path in runs_dir.rglob("*_chkpt"):
        if not path.is_dir() or path.is_symlink():
            continue
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            continue
        prefix = match.group("prefix")
        step = int(match.group("step"))
        parent = path.parent.resolve()
        group_key = f"{parent}::{prefix}"
        try:
            relative_parent = path.parent.relative_to(runs_dir)
            group_display = str(relative_parent / prefix)
        except ValueError:
            group_display = f"{path.parent}/{prefix}"
        checkpoints.append(Checkpoint(path, prefix, step, group_key, group_display))
    return checkpoints


def model_state(checkpoint_dir: Path) -> ModelState:
    payloads = tuple(
        sorted(
            (
                path
                for path in checkpoint_dir.iterdir()
                if (path.is_file() or path.is_symlink()) and is_model_payload_name(path.name)
            ),
            key=lambda path: path.name,
        )
    )
    manifest = checkpoint_dir / MANIFEST_NAME

    for index_name in MODEL_INDEX_NAMES:
        index_path = checkpoint_dir / index_name
        if not index_path.exists():
            continue
        try:
            index = json.loads(index_path.read_text())
            referenced_names = set(index["weight_map"].values())
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            return ModelState("incomplete", payloads, f"invalid {index_name}: {exc}")
        if not referenced_names:
            return ModelState("incomplete", payloads, f"empty {index_name}")
        unsafe_names = sorted(
            name for name in referenced_names if Path(name).name != name or not is_model_payload_name(name)
        )
        if unsafe_names:
            return ModelState(
                "incomplete",
                payloads,
                f"unsafe/unexpected shard names in {index_name}: {unsafe_names}",
            )
        missing = sorted(
            name
            for name in referenced_names
            if not (checkpoint_dir / name).is_file() or (checkpoint_dir / name).stat().st_size <= 0
        )
        if missing:
            return ModelState(
                "incomplete",
                payloads,
                f"{len(missing)} referenced model shard(s) missing or empty",
            )
        return ModelState("full", payloads, f"complete sharded model ({len(referenced_names)} shards)")

    for single_name in ("model.safetensors", "pytorch_model.bin"):
        single_path = checkpoint_dir / single_name
        if single_path.is_file() and single_path.stat().st_size > 0:
            return ModelState("full", payloads, f"complete single-file model ({single_name})")

    if payloads:
        return ModelState("incomplete", payloads, "model payload exists without a complete index/model")
    if manifest.is_file():
        return ModelState("compressed", (), f"manifest present ({MANIFEST_NAME})")
    return ModelState("adapter_only", (), "no merged model payload")


def recovery_payload_status(checkpoint_dir: Path) -> tuple[bool, str, Optional[str]]:
    adapter_dir = checkpoint_dir / "lora_adapter"
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = tuple(
        path
        for path in (
            adapter_dir / "adapter_model.safetensors",
            adapter_dir / "adapter_model.bin",
        )
        if path.is_file() and path.stat().st_size > 0
    )
    action_heads = tuple(checkpoint_dir.glob("action_head--*_checkpoint.pt"))

    missing: list[str] = []
    if not adapter_config.is_file():
        missing.append("lora_adapter/adapter_config.json")
    if not adapter_weights:
        missing.append("lora_adapter/adapter_model.{safetensors,bin}")
    if not any(path.is_file() and path.stat().st_size > 0 for path in action_heads):
        missing.append("action_head--*_checkpoint.pt")

    base_model: Optional[str] = None
    if adapter_config.is_file():
        try:
            config = json.loads(adapter_config.read_text())
            value = config.get("base_model_name_or_path")
            if isinstance(value, str) and value:
                base_model = value
        except (OSError, json.JSONDecodeError):
            missing.append("readable lora adapter_config.json")

    if missing:
        return False, "missing recovery payload: " + ", ".join(missing), base_model
    return True, "LoRA adapter and action head are present", base_model


def newest_mtime(checkpoint_dir: Path) -> float:
    newest = checkpoint_dir.stat().st_mtime
    for root, dirs, files in os.walk(checkpoint_dir, followlinks=False):
        for name in dirs + files:
            path = Path(root) / name
            try:
                newest = max(newest, path.lstat().st_mtime)
            except FileNotFoundError:
                # A concurrently written temporary file disappeared.  Treat the
                # directory as current so the age check will skip it.
                return datetime.now().timestamp()
    return newest


def atomic_write_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def compress_checkpoint(
    checkpoint: Checkpoint,
    state: ModelState,
    base_model: Optional[str],
) -> tuple[int, list[str]]:
    files = []
    total_bytes = 0
    for path in state.payloads:
        # Never follow or delete anything below a nested directory.
        if path.parent != checkpoint.path or not is_model_payload_name(path.name):
            raise RuntimeError(f"refusing unexpected payload path: {path}")
        stat = path.lstat()
        files.append({"name": path.name, "size_bytes": stat.st_size})
        total_bytes += stat.st_size

    now = datetime.now(timezone.utc).isoformat()
    common = {
        "format_version": 1,
        "checkpoint": str(checkpoint.path),
        "checkpoint_step": checkpoint.step,
        "group": checkpoint.group_display,
        "base_model_name_or_path": base_model,
        "removed_merged_model_files": files,
        "removed_bytes": total_bytes,
        "preserved": [
            "lora_adapter/",
            "action_head--*_checkpoint.pt",
            "proprio_projector--*_checkpoint.pt (when present)",
            "noisy_action_projector--*_checkpoint.pt (when present)",
            "vision_backbone--*_checkpoint.pt (when present)",
            "processor, tokenizer, model config, and dataset statistics",
        ],
        "remerge_command": (
            "python vla-scripts/merge_lora_weights_and_save.py "
            f"--base_checkpoint {base_model or '<BASE_CHECKPOINT>'} "
            f"--lora_finetuned_checkpoint_dir {checkpoint.path}"
        ),
    }
    in_progress_path = checkpoint.path / IN_PROGRESS_NAME
    manifest_path = checkpoint.path / MANIFEST_NAME
    atomic_write_json(
        in_progress_path,
        {**common, "status": "in_progress", "started_at_utc": now},
    )

    removed_names: list[str] = []
    try:
        for path in state.payloads:
            path.unlink()
            removed_names.append(path.name)
        atomic_write_json(
            manifest_path,
            {
                **common,
                "status": "compressed",
                "compressed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        in_progress_path.unlink(missing_ok=True)
    except Exception:
        # Leave the in-progress record in place so an interrupted cleanup is
        # visible and no later run silently assumes this is a normal checkpoint.
        raise
    return total_bytes, removed_names


def print_group_header(group: str, count: int) -> None:
    print(f"\n[{group}]  {count} checkpoint(s)")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Checkpoint tree to scan (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=2,
        help="Number of newest complete merged checkpoints to keep per group (default: 2)",
    )
    parser.add_argument(
        "--min-age-minutes",
        type=float,
        default=15.0,
        help="Never compress a checkpoint modified more recently than this (default: 15)",
    )
    parser.add_argument(
        "--group-regex",
        help="Only process checkpoint group names matching this regular expression",
    )
    parser.add_argument(
        "--verbose-files",
        action="store_true",
        help="List every merged model file that would be/is removed",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove merged model payloads; without this flag the run is a preview",
    )
    args = parser.parse_args(argv)
    if args.keep_latest < 1:
        parser.error("--keep-latest must be at least 1")
    if args.min_age_minutes < 0:
        parser.error("--min-age-minutes cannot be negative")
    if args.group_regex:
        try:
            re.compile(args.group_regex)
        except re.error as exc:
            parser.error(f"invalid --group-regex: {exc}")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    runs_dir = args.runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        print(f"ERROR: runs directory does not exist: {runs_dir}", file=sys.stderr)
        return 2

    lock_path = runs_dir / ".prune_merged_checkpoints.lock"
    lock_handle = lock_path.open("a+")
    try:
        lock_mode = fcntl.LOCK_EX | fcntl.LOCK_NB if args.apply else fcntl.LOCK_SH
        try:
            fcntl.flock(lock_handle.fileno(), lock_mode)
        except BlockingIOError:
            print(
                f"ERROR: another cleanup is applying changes under {runs_dir}",
                file=sys.stderr,
            )
            return 3

        checkpoints = discover_checkpoints(runs_dir)
        groups: dict[str, list[Checkpoint]] = defaultdict(list)
        for checkpoint in checkpoints:
            groups[checkpoint.group_key].append(checkpoint)

        group_filter = re.compile(args.group_regex) if args.group_regex else None
        selected_groups = [
            entries for entries in groups.values() if not group_filter or group_filter.search(entries[0].group_display)
        ]
        selected_groups.sort(key=lambda entries: entries[0].group_display)

        mode = "APPLY" if args.apply else "DRY-RUN"
        print(f"Mode: {mode}")
        print(f"Runs directory: {runs_dir}")
        print(f"Retention: newest {args.keep_latest} complete merged checkpoint(s) per group")
        print(f"Minimum age: {args.min_age_minutes:g} minutes")
        if not args.apply:
            print("No model files will be removed. Re-run with --apply after reviewing.")

        planned_bytes = 0
        removed_bytes = 0
        candidate_count = 0
        compressed_count = 0
        error_count = 0
        now = datetime.now().timestamp()

        for entries in selected_groups:
            entries.sort(key=lambda checkpoint: checkpoint.step, reverse=True)
            print_group_header(entries[0].group_display, len(entries))
            states = {entry.path: model_state(entry.path) for entry in entries}

            # Protect both the highest directory steps and the newest complete
            # models.  During an in-progress save this may temporarily protect
            # more than keep_latest checkpoints, which is intentionally safe.
            protected_paths = {entry.path for entry in entries[: args.keep_latest]}
            complete_entries = [entry for entry in entries if states[entry.path].kind == "full"]
            protected_paths.update(entry.path for entry in complete_entries[: args.keep_latest])

            for entry in entries:
                state = states[entry.path]
                label = f"step {entry.step:>8}  {entry.path.name}"
                if entry.path in protected_paths:
                    suffix = "full/test-ready" if state.kind == "full" else state.kind
                    print(f"  KEEP      {label}  [{suffix}]")
                    if state.kind != "full":
                        print(f"            WARNING: protected checkpoint is not full: {state.reason}")
                    continue
                if state.kind == "compressed":
                    print(f"  DONE      {label}  [already compressed]")
                    continue
                if state.kind != "full":
                    print(f"  SKIP      {label}  [{state.kind}: {state.reason}]")
                    continue

                recoverable, recovery_reason, base_model = recovery_payload_status(entry.path)
                if not recoverable:
                    print(f"  SKIP      {label}  [{recovery_reason}]")
                    continue

                age_minutes = (now - newest_mtime(entry.path)) / 60.0
                if age_minutes < args.min_age_minutes:
                    print(
                        f"  SKIP      {label}  [modified {age_minutes:.1f} min ago; "
                        f"minimum is {args.min_age_minutes:g} min]"
                    )
                    continue

                payload_bytes = sum(path.lstat().st_size for path in state.payloads)
                planned_bytes += payload_bytes
                candidate_count += 1
                verb = "COMPRESS" if args.apply else "WOULD DEL"
                print(f"  {verb:<9} {label}  [{human_bytes(payload_bytes)}; {recovery_reason}]")
                if args.verbose_files:
                    for path in state.payloads:
                        print(f"              {path.name} ({human_bytes(path.lstat().st_size)})")

                if args.apply:
                    try:
                        freed, _ = compress_checkpoint(entry, state, base_model)
                        removed_bytes += freed
                        compressed_count += 1
                    except Exception as exc:
                        error_count += 1
                        print(f"            ERROR: compression failed: {exc}", file=sys.stderr)

        print("\nSummary")
        print(f"  Groups scanned:            {len(selected_groups)}")
        print(f"  Old full checkpoints:      {candidate_count}")
        print(f"  Estimated removable size:  {human_bytes(planned_bytes)}")
        if args.apply:
            print(f"  Checkpoints compressed:    {compressed_count}")
            print(f"  Model payloads removed:    {human_bytes(removed_bytes)}")
            print(f"  Errors:                    {error_count}")
        else:
            print("  Files removed:             0 (dry-run)")

        return 1 if error_count else 0
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
