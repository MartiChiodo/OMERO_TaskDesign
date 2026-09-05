from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plot_results")


# Keys are numeric scenario IDs (used in CSVs and for grouping); values are the
# labels shown in tables and plots. Missing IDs fall back to their number.
SCENARIO_LABELS = {
    11: "S·so·lr", 12: "S·lo·lr", 13: "S·so·hr", 14: "S·lo·hr",
    31: "M·so·lr", 32: "M·lo·lr", 33: "M·so·hr", 34: "M·lo·hr",
    51: "L·so·lr", 52: "L·lo·lr", 53: "L·so·hr", 54: "L·lo·hr",
}


def scenario_label(scenario: int) -> str:
    """Label to display for a scenario, defaulting to its numeric ID."""
    return SCENARIO_LABELS.get(scenario, str(scenario))


@dataclass(frozen=True)
class Metric:
    key: str
    ylabel: str
    unit: str            # siunitx macro, "" if dimensionless
    better: str | None   # "max", "min" or None: which mean is bolded
    ylim: tuple | None = None
    scale: float = 1.0   # factor on raw values, e.g. seconds to minutes

    @property
    def folder(self) -> str:
        return self.key

    @property
    def files(self) -> dict[str, str]:
        return {mode: f"{mode}_{self.key}.csv" for mode in MODES}


MODES = ["Opt_False", "Opt_True"]
MODE_TITLES = {"Opt_False": "Without optimisation", "Opt_True": "With optimisation"}

METRICS = [
    Metric("throughput", "Throughput",
           unit="", better="max", ylim=(550, 900)),
    Metric("mean_flow_time", "Mean flow time (s)",
           unit=r"\second", better="min", ylim=(400, 1400)),
    Metric("average_pods", "Average number of pods moving",
           unit="", better=None, ylim=(15, 42)),
    Metric("computational_time", "Decision-making time (min)",
           unit=r"\minute", better="min", ylim=(0, 42), scale=1 / 60),
]

SCENARIO_GROUPS = {"1x": [11, 12, 13, 14], "3x": [31, 32, 33, 34], "5x": [51, 52, 53, 54]}


STYLE = {
    "Opt_False": {"color": "#1f77b4", "hatch": ""},
    "Opt_True": {"color": "#ff7f0e", "hatch": "///"},
}
BOX_WIDTH = 0.3
OFFSET = 0.17

MEAN_STYLE = {"marker": "D", "markerfacecolor": "white",
              "markeredgecolor": "black", "markersize": 4}
FLIER_STYLE = {"marker": "o", "markeredgecolor": "black",
               "markersize": 3.5, "linestyle": "none", "alpha": 0.75}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.linewidth": 0.8,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_metric(metric: Metric) -> dict[str, pd.DataFrame]:
    """Load a metric's CSVs for both configurations."""
    data = {}
    for mode, filename in metric.files.items():
        df = pd.read_csv(os.path.join(BASE_DIR, filename))

        # First column is always the scenario ID.
        if "Scenario" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Scenario"})

        # Flow time is reported per OrderSize; keep only the total.
        if metric.key == "mean_flow_time":
            df = df[df["OrderSize"].astype(str) == "Total"].drop(columns="OrderSize")

        df = df.set_index("Scenario").apply(pd.to_numeric, errors="coerce")
        df = df * metric.scale
        data[mode] = df
    return data


def header_cell(label: str, unit: str) -> str:
    """Bold header cell, with siunitx unit if given."""
    if unit:
        return rf"\textbf{{{label}}} [\unit{{{unit}}}]"
    return rf"\textbf{{{label}}}"


def winning_mode(mean_false: float, mean_true: float, better: str | None) -> str | None:
    """Config to highlight: 'false', 'true' or None."""
    if better not in ("max", "min") or mean_false == mean_true:
        return None
    false_wins = mean_false > mean_true if better == "max" else mean_false < mean_true
    return "false" if false_wins else "true"


def mean_cells(mean_false: float, mean_true: float, better: str | None) -> tuple[str, str]:
    """Format both means, bolding the winner."""
    text_false, text_true = f"{mean_false:.2f}", f"{mean_true:.2f}"
    winner = winning_mode(mean_false, mean_true, better)
    if winner == "false":
        text_false = rf"\textbf{{{text_false}}}"
    elif winner == "true":
        text_true = rf"\textbf{{{text_true}}}"
    return text_false, text_true


def build_latex_table(data: dict[str, pd.DataFrame], metric: Metric) -> str:
    """Build the .tex content for a metric."""
    unit, better = metric.unit, metric.better
    scenarios = sorted(set(data["Opt_False"].index) | set(data["Opt_True"].index))

    lines = [
        r"\begin{tabular}{@{}c cc c cc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Without optimisation}} & "
        r"& \multicolumn{2}{c}{\textbf{With optimisation}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}",
        rf"\textbf{{ID}} & {header_cell('Mean', unit)} & {header_cell('Std', unit)} & "
        rf"& {header_cell('Mean', unit)} & {header_cell('Std', unit)} \\",
        r"\midrule",
    ]

    previous_group = None
    for scenario in scenarios:
        group = scenario // 10
        if previous_group is not None and group != previous_group:
            lines.append(r"\addlinespace")  # gap between groups
        previous_group = group

        false_values = data["Opt_False"].loc[scenario].dropna()
        true_values = data["Opt_True"].loc[scenario].dropna()
        mean_false, std_false = false_values.mean(), false_values.std(ddof=1)
        mean_true, std_true = true_values.mean(), true_values.std(ddof=1)
        cell_false, cell_true = mean_cells(mean_false, mean_true, better)

        lines.append(
            rf"{scenario_label(scenario)} & {cell_false} & {std_false:.2f} & "
            rf"& {cell_true} & {std_true:.2f} \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def save_latex_table(data: dict[str, pd.DataFrame], metric: Metric) -> None:
    folder = os.path.join(OUTPUT_DIR, metric.folder)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "summary_statistics.tex")
    with open(path, "w", encoding="utf-8") as file:
        file.write(build_latex_table(data, metric))
    print(f"Saved table: {path}")


def legend_handles() -> list:
    """Legend entries: both configs."""
    return [
        Patch(facecolor=STYLE["Opt_False"]["color"], edgecolor="black",
              alpha=0.65, label=MODE_TITLES["Opt_False"]),
        Patch(facecolor=STYLE["Opt_True"]["color"], edgecolor="black",
              hatch="///", alpha=0.65, label=MODE_TITLES["Opt_True"]),
    ]


def draw_group_boxplot(ax, data: dict[str, pd.DataFrame], scenarios: list[int]) -> None:
    """Draw side by side boxes for both configs over a scenario group."""
    for idx, mode in enumerate(MODES):
        df = data[mode]
        present = [s for s in scenarios if s in df.index]
        values = [df.loc[s].dropna().values for s in present]
        offset = -OFFSET if idx == 0 else OFFSET
        positions = [scenarios.index(s) + offset for s in present]

        boxes = ax.boxplot(
            values,
            positions=positions,
            widths=BOX_WIDTH,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            meanprops=MEAN_STYLE,
            flierprops={**FLIER_STYLE, "markerfacecolor": STYLE[mode]["color"]},
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for box in boxes["boxes"]:
            box.set(facecolor=STYLE[mode]["color"], hatch=STYLE[mode]["hatch"],
                    edgecolor="black", linewidth=0.8, alpha=0.65)


def create_boxplots(data: dict[str, pd.DataFrame], metric: Metric) -> None:
    folder = os.path.join(OUTPUT_DIR, metric.folder)
    os.makedirs(folder, exist_ok=True)

    for group, scenarios in SCENARIO_GROUPS.items():
        fig, ax = plt.subplots(figsize=(6.3, 3.5))
        draw_group_boxplot(ax, data, scenarios)

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([scenario_label(s) for s in scenarios])
        ax.set_xlabel("Experimental Configuration")
        ax.set_ylabel(metric.ylabel)
        if metric.ylim is not None:
            ax.set_ylim(metric.ylim)

        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=legend_handles(), frameon=False, ncol=1)

        path = os.path.join(folder, f"boxplot_group_{group}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {path}")


def main() -> None:
    for metric in METRICS:
        print(f"\nProcessing {metric.key}")
        data = load_metric(metric)
        save_latex_table(data, metric)
        create_boxplots(data, metric)
    print("\nDone.")


if __name__ == "__main__":
    main()