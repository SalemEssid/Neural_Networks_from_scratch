"""
Build the comparison figures used in README.md from saved experiment results.

Reads experiments/<name>/metrics.csv and summary.json (run run_experiments.py first)
and writes a light and a dark version of every figure to docs/figures/.

Usage:
    python make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


EXPERIMENTS_DIR = Path("experiments")
OUTPUT_DIR = Path("docs") / "figures"

# Validated categorical palette (first four slots) and chart chrome, per mode
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}

FIG_WIDTH = 8.3   # inches; saved at 200 dpi and shown at ~half size on GitHub
DPI = 200

# The run every other group is compared against
BASELINE = "opt_adam"

OPTIMIZER_SERIES = [
    ("Adam (lr 0.001)", "opt_adam"),
    ("SGD (lr 0.1)", "opt_sgd"),
    ("Momentum (lr 0.01)", "opt_momentum"),
    ("RMSprop (lr 0.001)", "opt_rmsprop"),
]

OVERFITTING_SERIES = [
    ("No regularization", "opt_adam"),
    ("L2, λ = 1", "reg_l2_1"),
    ("Dropout 0.3", "dropout_0.3"),
    ("Dropout 0.5", "dropout_0.5"),
]

TEST_ACCURACY_GROUPS = [
    ("Optimizer", [
        ("Adam (baseline)", "opt_adam"),
        ("SGD", "opt_sgd"),
        ("Momentum", "opt_momentum"),
        ("RMSprop", "opt_rmsprop"),
    ]),
    ("Weight penalty", [
        ("L2, λ = 0.01", "reg_l2_0.01"),
        ("L2, λ = 0.1", "reg_l2_0.1"),
        ("L2, λ = 1", "reg_l2_1"),
        ("L1, λ = 0.01", "reg_l1_0.01"),
    ]),
    ("Dropout", [
        ("Dropout 0.1", "dropout_0.1"),
        ("Dropout 0.3", "dropout_0.3"),
        ("Dropout 0.5", "dropout_0.5"),
    ]),
    ("Hidden layers", [
        ("128", "arch_128"),
        ("128-64", "arch_128_64"),
        ("512-256-128-64", "arch_512_256_128_64"),
    ]),
    ("Activation", [
        ("Sigmoid", "act_sigmoid"),
    ]),
]

# Runs that failed to train are left out of the dot plot (noted in its subtitle)
DIVERGED_BELOW = 0.5


def load_metrics(name):
    """Per-epoch metrics of an experiment as lists of floats, or None if it hasn't run."""
    path = EXPERIMENTS_DIR / name / "metrics.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    keys = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
    return {key: [float(row[key]) for row in rows] for key in keys}


def load_summary(name):
    """summary.json of an experiment, or None if it hasn't run."""
    path = EXPERIMENTS_DIR / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def new_figure(theme, height, title, subtitle, left=0.75, top_extra=0.0):
    """Figure with a left-aligned title and subtitle above a single plot area."""
    fig = plt.figure(figsize=(FIG_WIDTH, height), dpi=DPI, facecolor=theme["surface"])
    top = 0.85 + top_extra
    bottom = 0.55
    ax = fig.add_axes([left / FIG_WIDTH, bottom / height,
                       1 - (left + 0.35) / FIG_WIDTH, 1 - (top + bottom) / height])
    ax.set_facecolor(theme["surface"])

    fig.text(0.18 / FIG_WIDTH, 1 - 0.32 / height, title, color=theme["ink"],
             fontsize=13, fontweight="semibold", va="baseline")
    fig.text(0.18 / FIG_WIDTH, 1 - 0.58 / height, subtitle, color=theme["secondary"],
             fontsize=10, va="baseline")

    # Recessive chrome: hairline grid, only the baseline axis drawn
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(colors=theme["muted"], labelcolor=theme["secondary"], labelsize=9, length=0, pad=6)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name, mode):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}-{mode}.png"
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {path}")


def line_chart(mode, series, metric, title, subtitle, y_format, filename):
    """One line per experiment over epochs, legend on top, end-dots on every line."""
    theme = THEMES[mode]
    height = 4.2
    fig, ax = new_figure(theme, height, title, subtitle, top_extra=0.3)

    for i, (label, name) in enumerate(series):
        metrics = load_metrics(name)
        if metrics is None:
            continue
        x = [epoch + 1 for epoch in metrics["epoch"]]
        y = metrics[metric]
        color = theme["series"][i]
        ax.plot(x, y, color=color, linewidth=1.5, solid_joinstyle="round",
                solid_capstyle="round", label=label, zorder=3)
        ax.plot(x[-1], y[-1], "o", markersize=5.8, color=color,
                markeredgecolor=theme["surface"], markeredgewidth=1.4, zorder=4)

    ax.grid(axis="y", color=theme["grid"], linewidth=0.75)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(y_format))
    ax.set_xlabel("Epoch", color=theme["muted"], fontsize=9)

    legend = ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.02), ncol=len(series),
                       frameon=False, fontsize=9.5, handlelength=1.6, columnspacing=1.6,
                       borderaxespad=0)
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    save(fig, filename, mode)


def test_accuracy_chart(mode):
    """Dot plot of test accuracy for every run, grouped, with the baseline marked."""
    theme = THEMES[mode]

    rows = []        # (label, accuracy or None for a group header)
    diverged = []
    for group, items in TEST_ACCURACY_GROUPS:
        rows.append((group, None))
        for label, name in items:
            summary = load_summary(name)
            if summary is None:
                continue
            rows.append((label, summary["test_accuracy"]))
    for name in ("opt_adam_lr0.1",):
        summary = load_summary(name)
        if summary is not None and summary["test_accuracy"] < DIVERGED_BELOW:
            diverged.append(summary)

    shown = [acc for _, acc in rows if acc is not None and acc >= DIVERGED_BELOW]
    baseline = load_summary(BASELINE)

    height = 1.6 + 0.26 * len(rows)
    subtitle = "Each dot is one 15-epoch run on the 10,000-image test set"
    if diverged:
        subtitle += f"; Adam at lr 0.1 diverged ({diverged[0]['test_accuracy']:.1%}) and is not shown"
    fig, ax = new_figure(theme, height, "Test accuracy by experiment", subtitle, left=1.75)

    y_positions = list(range(len(rows)))[::-1]
    for (label, acc), y in zip(rows, y_positions):
        if acc is None or acc < DIVERGED_BELOW:
            continue
        ax.plot(acc, y, "o", markersize=6.5, color=theme["series"][0],
                markeredgecolor=theme["surface"], markeredgewidth=1.4, zorder=3)
        ax.text(acc, y, f"   {acc:.2%}", color=theme["secondary"], fontsize=8.5,
                va="center", ha="left")

    if baseline is not None:
        ax.axvline(baseline["test_accuracy"], color=theme["muted"], linewidth=0.75, zorder=2)
        ax.text(baseline["test_accuracy"], len(rows) - 0.35, " baseline", color=theme["muted"],
                fontsize=8.5, va="bottom", ha="left")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _ in rows])
    for tick, (_, acc) in zip(ax.get_yticklabels(), rows):
        if acc is None:
            tick.set_color(theme["ink"])
            tick.set_fontweight("semibold")
    ax.set_ylim(-0.6, len(rows) - 0.1)

    if shown:
        ax.set_xlim(min(shown) - 0.002, max(shown) + 0.004)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.75)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))

    save(fig, "test_accuracy", mode)


if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

    for mode in THEMES:
        line_chart(
            mode, OPTIMIZER_SERIES, "val_acc",
            "Validation accuracy by optimizer",
            "Same network ([784, 256, 128, 64, 10], ReLU), batch size 64, no regularization",
            lambda v, _: f"{v:.1%}", "optimizers"
        )
        line_chart(
            mode, OVERFITTING_SERIES, "val_loss",
            "Validation loss: overfitting and how regularization changes it",
            "Adam (lr 0.001), same network; a rising curve means the model is overfitting",
            lambda v, _: f"{v:.2f}", "overfitting"
        )
        test_accuracy_chart(mode)
