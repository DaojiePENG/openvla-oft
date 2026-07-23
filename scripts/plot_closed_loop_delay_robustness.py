"""Plot the LIBERO-Goal closed-loop delay-robustness comparison.

The aggregate success rates are kept here as the single source of truth and are
also exported as strict JSON next to the generated PNG.
"""

import argparse
import json
import os
from pathlib import Path

# Some container images mount ~/.config with a host UID that differs from the
# runtime user. Keep Matplotlib's writable cache inside the repository instead.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MPL_CONFIG_DIR = _PROJECT_ROOT / ".cache" / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DELAYS = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40], dtype=float)
SUCCESS_RATES = {
    "OpenVLA": np.array([77.0, 36.2, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "OpenVLA-OFT": np.array([97.2, 76.2, 26.2, 15.2, 4.0, 2.4, 0.0, 0.0, 0.0]),
    "UniVLA": np.array([94.6, 87.8, 48.2, 31.4, 21.8, 16.0, 11.8, 6.6, 3.0]),
    "CloudEdgeVLA": np.array([96.0, 95.4, 95.4, 94.8, 94.6, 94.4, 94.4, 94.2, 94.2]),
}

COLORS = {
    "OpenVLA": "#94A3B8",
    "OpenVLA-OFT": "#2A9D8F",
    "UniVLA": "#E76F51",
    "CloudEdgeVLA": "#2563EB",
}
MARKERS = {
    "OpenVLA": "D",
    "OpenVLA-OFT": "s",
    "UniVLA": "^",
    "CloudEdgeVLA": "o",
}


def normalized_aurc(values: np.ndarray) -> float:
    """Area under the success-retention curve, normalized to [0, 100]."""
    retention = values / values[0]
    return float(np.trapz(retention, DELAYS) / (DELAYS[-1] - DELAYS[0]) * 100.0)


def plot(output_path: Path) -> dict:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(15.2, 5.4), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[2.45, 1.0, 1.0], wspace=0.34)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_auc = fig.add_subplot(gs[0, 1])
    ax_retention = fig.add_subplot(gs[0, 2])

    # Panel (a): raw closed-loop success rates.
    ax_curve.axvspan(20, 40, color="#F1F5F9", alpha=0.9, zorder=0)
    ax_curve.axvline(20, color="#64748B", lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax_curve.text(
        30,
        7.5,
        "Beyond training\nwindow ($d>20$)",
        ha="center",
        va="center",
        color="#64748B",
        fontsize=10,
    )

    best_baseline = np.maximum.reduce(
        [SUCCESS_RATES[name] for name in ("OpenVLA", "OpenVLA-OFT", "UniVLA")]
    )
    cloudedge = SUCCESS_RATES["CloudEdgeVLA"]
    ax_curve.fill_between(
        DELAYS,
        best_baseline,
        cloudedge,
        where=cloudedge >= best_baseline,
        interpolate=True,
        color=COLORS["CloudEdgeVLA"],
        alpha=0.10,
        zorder=1,
    )

    for model in ("OpenVLA", "OpenVLA-OFT", "UniVLA", "CloudEdgeVLA"):
        is_ours = model == "CloudEdgeVLA"
        ax_curve.plot(
            DELAYS,
            SUCCESS_RATES[model],
            label=model,
            color=COLORS[model],
            marker=MARKERS[model],
            ms=7.5 if is_ours else 6.0,
            lw=3.2 if is_ours else 2.0,
            ls="-" if is_ours else "--",
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=4 if is_ours else 3,
        )

    ax_curve.annotate(
        "94.2%",
        xy=(40, cloudedge[-1]),
        xytext=(35.7, 85.0),
        color=COLORS["CloudEdgeVLA"],
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": COLORS["CloudEdgeVLA"], "lw": 1.2},
    )
    ax_curve.text(
        28.0,
        69.0,
        "+91.2 pp at $d=40$\nvs. best baseline",
        ha="center",
        va="center",
        color="#1E3A8A",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#93C5FD",
            "alpha": 0.92,
        },
        zorder=5,
    )
    ax_curve.set_title("(a) Closed-loop success under delay", loc="left", fontweight="bold")
    ax_curve.set_xlabel("Delay window $d$")
    ax_curve.set_ylabel("Task success rate (%)")
    ax_curve.set_xlim(-1, 43)
    ax_curve.set_ylim(-2, 103)
    ax_curve.set_xticks(DELAYS)
    ax_curve.set_yticks(np.arange(0, 101, 20))
    ax_curve.grid(axis="y", color="#CBD5E1", alpha=0.65, lw=0.8)
    ax_curve.legend(loc="center left", bbox_to_anchor=(0.02, 0.44), frameon=False)

    # Panels (b-c): compact robustness summaries.
    display_order = ["CloudEdgeVLA", "UniVLA", "OpenVLA-OFT", "OpenVLA"]
    y = np.arange(len(display_order))
    aurc = np.array([normalized_aurc(SUCCESS_RATES[name]) for name in display_order])
    d40_retention = np.array(
        [SUCCESS_RATES[name][-1] / SUCCESS_RATES[name][0] * 100.0 for name in display_order]
    )

    for ax, values, title in (
        (ax_auc, aurc, "(b) Normalized delay AUC"),
        (ax_retention, d40_retention, "(c) Success retained at $d=40$"),
    ):
        colors = [COLORS[name] for name in display_order]
        ax.barh(y, values, color=colors, height=0.58, alpha=0.94)
        ax.set_yticks(y)
        ax.set_yticklabels(display_order)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xlabel("Retention score (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#CBD5E1", alpha=0.55, lw=0.8)
        ax.set_axisbelow(True)
        for yi, value in zip(y, values):
            label_inside = value >= 85.0
            ax.text(
                value - 3.0 if label_inside else value + 2.0,
                yi,
                f"{value:.1f}",
                va="center",
                ha="right" if label_inside else "left",
                fontweight="bold" if yi == 0 else "normal",
                color="white" if label_inside else "#0F172A",
            )

    fig.suptitle(
        "CloudEdgeVLA Maintains Closed-Loop Performance Under Severe Delay",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.005,
        "LIBERO-Goal  •  Aggregate success rates  •  Higher is better",
        ha="center",
        color="#64748B",
        fontsize=10.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "benchmark": "libero_goal",
        "delay_window": DELAYS.astype(int).tolist(),
        "success_rate_percent": {
            model: values.tolist() for model, values in SUCCESS_RATES.items()
        },
        "derived_metrics": {
            model: {
                "normalized_delay_aurc_percent": normalized_aurc(values),
                "retention_at_d40_percent": float(values[-1] / values[0] * 100.0),
                "success_at_d40_percent": float(values[-1]),
            }
            for model, values in SUCCESS_RATES.items()
        },
        "uncertainty": "Not provided with the aggregate success rates; no error bars are plotted.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results/fig_closed_loop_delay_robustness.png"),
    )
    args = parser.parse_args()

    export = plot(args.output_path)
    json_path = args.output_path.with_name(f"{args.output_path.stem}_data.json")
    json_path.write_text(json.dumps(export, indent=2, allow_nan=False) + "\n")
    print(f"Saved figure: {args.output_path}")
    print(f"Saved data:   {json_path}")


if __name__ == "__main__":
    main()
