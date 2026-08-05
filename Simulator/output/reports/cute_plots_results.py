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

METRICS = {
    "throughput": {
        "files": {
            "Opt_False": "Opt_False_throughput.csv",
            "Opt_True": "Opt_True_throughput.csv"
        },
        "ylabel": "Throughput",
        "folder": "throughput",
        "ylim": (500, 900)
    },

    "mean_flow_time": {
        "files": {
            "Opt_False": "Opt_False_mean_flow_time.csv",
            "Opt_True": "Opt_True_mean_flow_time.csv"
        },
        "ylabel": "Mean flow time (s)",
        "folder": "mean_flow_time",
        "ylim": (400, 1400)
    },

    "average_pods": {
        "files": {
            "Opt_False": "Opt_False_average_pods.csv",
            "Opt_True": "Opt_True_average_pods.csv"
        },
        "ylabel": "Average number of pods moving",
        "folder": "average_pods",
        "ylim": None
    },

    "computational_time": {
        "files": {
            "Opt_False": "Opt_False_computational_time.csv",
            "Opt_True": "Opt_True_computational_time.csv"
        },
        "ylabel": "Decision-making time (min)",
        "folder": "computational_time",
        "ylim": (0, 1600)
    }
}


SCENARIO_GROUPS = {
    "1x": [11, 12, 13, 14],
    "3x": [31, 32, 33, 34],
    "5x": [51, 52, 53, 54]
}


### Plot style

STYLE = {
    "Opt_False": {
        "color": "#1f77b4",
        "hatch": ""
    },

    "Opt_True": {
        "color": "#ff7f0e",
        "hatch": "///"
    }
}


BOX_WIDTH = 0.3
OFFSET = 0.17


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Latin Modern Roman",
        "CMU Serif",
        "Times New Roman",
        "DejaVu Serif"
    ],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


### Load results

def load_metric(metric):

    results = {}

    for mode, filename in METRICS[metric]["files"].items():

        path = os.path.join(BASE_DIR, filename)

        df = pd.read_csv(path)

        if "Scenario" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Scenario"})


        if metric == "mean_flow_time":

            df = df[df["OrderSize"].astype(str) == "Total"]
            df = df.drop(columns=["OrderSize"])


        df = df.set_index("Scenario")

        df = df.apply(
            pd.to_numeric,
            errors="coerce"
        )


        # conversion seconds -> minutes
        if metric == "computational_time":
            df = df / 60.0


        results[mode] = df


    return results



### Generate LaTeX tables

def create_latex_table(data):

    scenarios = sorted(
        set(data["Opt_False"].index)
        |
        set(data["Opt_True"].index)
    )

    lines = []

    lines.append("\\begin{tabular}{@{}c cc c cc@{}}")
    lines.append("\\toprule")

    lines.append(
        "& \\multicolumn{2}{c}{\\textbf{Without optimisation}} "
        "& & \\multicolumn{2}{c}{\\textbf{With optimisation}} \\\\"
    )

    lines.append(
        "\\cmidrule(lr){2-3}"
        "\\cmidrule(lr){5-6}"
    )

    lines.append(
        "\\textbf{ID}"
        " & \\textbf{Mean}"
        " & \\textbf{Std}"
        " & "
        " & \\textbf{Mean}"
        " & \\textbf{Std} \\\\"
    )

    lines.append("\\midrule")


    previous_group = None


    for scenario in scenarios:

        current_group = scenario // 10

        if previous_group is not None and current_group != previous_group:
            lines.append("\\addlinespace")


        row = [str(scenario)]


        for mode in ["Opt_False", "Opt_True"]:

            values = data[mode].loc[scenario].dropna()

            row.extend([
                f"{values.mean():.2f}",
                f"{values.std(ddof=1):.2f}"
            ])


        lines.append(
            f"{row[0]} & "
            f"{row[1]} & "
            f"{row[2]} & "
            f"& {row[3]} & "
            f"{row[4]} \\\\"
        )


        previous_group = current_group


    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)



def save_latex_table(data, metric):

    folder = os.path.join(
        OUTPUT_DIR,
        METRICS[metric]["folder"]
    )

    os.makedirs(folder, exist_ok=True)


    path = os.path.join(
        folder,
        "summary_statistics.tex"
    )


    with open(path, "w", encoding="utf-8") as file:

        file.write(
            create_latex_table(data)
        )


    print(f"Saved table: {path}")



### Generate boxplots

def create_boxplots(data, metric):

    folder = os.path.join(
        OUTPUT_DIR,
        METRICS[metric]["folder"]
    )

    os.makedirs(folder, exist_ok=True)


    for group, scenarios in SCENARIO_GROUPS.items():

        fig, ax = plt.subplots(
            figsize=(6.3, 3.5)
        )


        for idx, mode in enumerate(
            ["Opt_False", "Opt_True"]
        ):

            df = data[mode]


            values = [
                df.loc[s].dropna().values
                for s in scenarios
                if s in df.index
            ]


            positions = [
                i - OFFSET if idx == 0 else i + OFFSET
                for i in range(len(values))
            ]


            boxes = ax.boxplot(
                values,
                positions=positions,
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 4
                },
                medianprops={
                    "color": "black",
                    "linewidth": 1.2
                }
            )


            for box in boxes["boxes"]:

                box.set(
                    facecolor=STYLE[mode]["color"],
                    hatch=STYLE[mode]["hatch"],
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.65
                )


        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios)

        ax.set_xlabel("Experiment ID")
        ax.set_ylabel(METRICS[metric]["ylabel"])


        if METRICS[metric]["ylim"] is not None:

            ax.set_ylim(
                METRICS[metric]["ylim"]
            )


        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.6,
            alpha=0.6
        )


        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


        ax.legend(
            handles=[
                Patch(
                    facecolor=STYLE["Opt_False"]["color"],
                    label="Without optimisation"
                ),

                Patch(
                    facecolor=STYLE["Opt_True"]["color"],
                    hatch="///",
                    label="With optimisation"
                )
            ],
            frameon=False
        )


        path = os.path.join(
            folder,
            f"boxplot_group_{group}.pdf"
        )


        fig.savefig(
            path,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Saved plot: {path}")



### Main

if __name__ == "__main__":

    for metric in METRICS:

        print(f"\nProcessing {metric}")

        data = load_metric(metric)

        save_latex_table(
            data,
            metric
        )

        create_boxplots(
            data,
            metric
        )


    print("\nDone.")