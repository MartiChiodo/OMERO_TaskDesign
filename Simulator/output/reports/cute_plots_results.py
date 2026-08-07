import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

### Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plot_results")

### Metrics configuration
# "unit" is the siunitx unit macro shown in the table header ("" for a dimensionless count).
# "better" selects which strategy is shown in bold, row by row:
#   "max"  -> the higher mean wins (throughput)
#   "min"  -> the lower mean wins (flow time, computational time)
#   None   -> no value is highlighted (pod movement, where lower is not unambiguously better)
METRICS = {
    "throughput": {
        "files": {"Opt_False": "Opt_False_throughput.csv", "Opt_True": "Opt_True_throughput.csv"},
        "ylabel": "Throughput", "folder": "throughput", "ylim": (500, 900),
        "unit": "", "better": "max",
    },
    "mean_flow_time": {
        "files": {"Opt_False": "Opt_False_mean_flow_time.csv", "Opt_True": "Opt_True_mean_flow_time.csv"},
        "ylabel": "Mean flow time (s)", "folder": "mean_flow_time", "ylim": (400, 1400),
        "unit": "\\second", "better": "min",
    },
    "average_pods": {
        "files": {"Opt_False": "Opt_False_average_pods.csv", "Opt_True": "Opt_True_average_pods.csv"},
        "ylabel": "Average number of pods moving", "folder": "average_pods", "ylim": None,
        "unit": "", "better": None,
    },
    "computational_time": {
        "files": {"Opt_False": "Opt_False_computational_time.csv", "Opt_True": "Opt_True_computational_time.csv"},
        "ylabel": "Decision-making time (min)", "folder": "computational_time", "ylim": (0, 1600),
        "unit": "\\minute", "better": "min",
    },
}

SCENARIO_GROUPS = {"1x": [11, 12, 13, 14], "3x": [31, 32, 33, 34], "5x": [51, 52, 53, 54]}

### Plot style
STYLE = {
    "Opt_False": {"color": "#1f77b4", "hatch": ""},
    "Opt_True": {"color": "#ff7f0e", "hatch": "///"},
}
BOX_WIDTH = 0.3
OFFSET = 0.17

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.linewidth": 0.8,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

### Load results
def load_metric(metric):
    results = {}
    for mode, filename in METRICS[metric]["files"].items():
        df = pd.read_csv(os.path.join(BASE_DIR, filename))
        if "Scenario" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Scenario"})
        if metric == "mean_flow_time":
            df = df[df["OrderSize"].astype(str) == "Total"].drop(columns=["OrderSize"])
        df = df.set_index("Scenario").apply(pd.to_numeric, errors="coerce")
        if metric == "computational_time":  # conversion seconds -> minutes
            df = df / 60.0
        results[mode] = df
    return results

### LaTeX helpers
def header_cell(label, unit):
    # Bold header cell, appending the siunitx unit when present.
    return f"\\textbf{{{label}}} [\\unit{{{unit}}}]" if unit else f"\\textbf{{{label}}}"

def format_mean_pair(mean_false, mean_true, better):
    # Format the two means, wrapping the winning one in \textbf.
    text_false, text_true = f"{mean_false:.2f}", f"{mean_true:.2f}"
    if better == "max":
        winner = "false" if mean_false > mean_true else "true" if mean_true > mean_false else None
    elif better == "min":
        winner = "false" if mean_false < mean_true else "true" if mean_true < mean_false else None
    else:
        winner = None
    if winner == "false":
        text_false = f"\\textbf{{{text_false}}}"
    elif winner == "true":
        text_true = f"\\textbf{{{text_true}}}"
    return text_false, text_true

### Generate LaTeX tables
def create_latex_table(data, metric):
    unit, better = METRICS[metric]["unit"], METRICS[metric]["better"]
    scenarios = sorted(set(data["Opt_False"].index) | set(data["Opt_True"].index))
    lines = ["\\begin{tabular}{@{}c cc c cc@{}}", "\\toprule"]
    lines.append("& \\multicolumn{2}{c}{\\textbf{Without optimisation}} & & \\multicolumn{2}{c}{\\textbf{With optimisation}} \\\\")
    lines.append("\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}")
    lines.append(f"\\textbf{{ID}} & {header_cell('Mean', unit)} & {header_cell('Std', unit)} & "
                 f"& {header_cell('Mean', unit)} & {header_cell('Std', unit)} \\\\")
    lines.append("\\midrule")
    previous_group = None
    for scenario in scenarios:
        current_group = scenario // 10
        if previous_group is not None and current_group != previous_group:
            lines.append("\\addlinespace")
        false_values, true_values = data["Opt_False"].loc[scenario].dropna(), data["Opt_True"].loc[scenario].dropna()
        mean_false, std_false = false_values.mean(), false_values.std(ddof=1)
        mean_true, std_true = true_values.mean(), true_values.std(ddof=1)
        mean_false_str, mean_true_str = format_mean_pair(mean_false, mean_true, better)
        lines.append(f"{scenario} & {mean_false_str} & {std_false:.2f} & & {mean_true_str} & {std_true:.2f} \\\\")
        previous_group = current_group
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)

def save_latex_table(data, metric):
    folder = os.path.join(OUTPUT_DIR, METRICS[metric]["folder"])
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "summary_statistics.tex")
    with open(path, "w", encoding="utf-8") as file:
        file.write(create_latex_table(data, metric))
    print(f"Saved table: {path}")

### Generate boxplots
def create_boxplots(data, metric):
    folder = os.path.join(OUTPUT_DIR, METRICS[metric]["folder"])
    os.makedirs(folder, exist_ok=True)
    for group, scenarios in SCENARIO_GROUPS.items():
        fig, ax = plt.subplots(figsize=(6.3, 3.5))
        for idx, mode in enumerate(["Opt_False", "Opt_True"]):
            df = data[mode]
            values = [df.loc[s].dropna().values for s in scenarios if s in df.index]
            positions = [i - OFFSET if idx == 0 else i + OFFSET for i in range(len(values))]
            boxes = ax.boxplot(
                values, positions=positions, widths=BOX_WIDTH, patch_artist=True,
                showfliers=False, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
                medianprops={"color": "black", "linewidth": 1.2},
            )
            for box in boxes["boxes"]:
                box.set(facecolor=STYLE[mode]["color"], hatch=STYLE[mode]["hatch"],
                        edgecolor="black", linewidth=0.8, alpha=0.65)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios)
        ax.set_xlabel("Experiment ID")
        ax.set_ylabel(METRICS[metric]["ylabel"])
        if METRICS[metric]["ylim"] is not None:
            ax.set_ylim(METRICS[metric]["ylim"])
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=[
            Patch(facecolor=STYLE["Opt_False"]["color"], label="Without optimisation"),
            Patch(facecolor=STYLE["Opt_True"]["color"], hatch="///", label="With optimisation"),
        ], frameon=False)
        path = os.path.join(folder, f"boxplot_group_{group}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {path}")

### Main
if __name__ == "__main__":
    for metric in METRICS:
        print(f"\nProcessing {metric}")
        data = load_metric(metric)
        save_latex_table(data, metric)
        create_boxplots(data, metric)
    print("\nDone.")