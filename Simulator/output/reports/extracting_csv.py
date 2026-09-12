import os
import re
from glob import glob
import pandas as pd

import math
from scipy.stats import t

### Folders
REPORT_FOLDER = os.path.dirname(os.path.abspath(__file__))
# all processed CSVs go here, so they never mix with scripts or raw reports
CSV_FOLDER = os.path.join(REPORT_FOLDER, "csv")
# LaTeX appendix tables go here, derived from the same CSVs
TEX_FOLDER = os.path.join(REPORT_FOLDER, "tex")
os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs(TEX_FOLDER, exist_ok=True)

### Scenario short labels (same notation as the summary tables in the body)
SCENARIO_LABELS_TABLES = {
    11: "S·fl·lr", 12: "S·ml·lr", 13: "S·fl·hr", 14: "S·ml·hr",
    31: "M·fl·lr", 32: "M·ml·lr", 33: "M·fl·hr", 34: "M·ml·hr",
    51: "L·fl·lr", 52: "L·ml·lr", 53: "L·fl·hr", 54: "L·ml·hr",
}

def scenario_label(scenario) -> str:
    """Short label for a scenario, falling back to its raw id if unknown."""
    try:
        key = int(scenario)
    except (TypeError, ValueError):
        return str(scenario)
    return SCENARIO_LABELS_TABLES.get(key, str(scenario))


### Filename and table patterns
filename_pattern = re.compile(r"report_(.+?)_Opt.+?_Seed(\d+)\.txt")
# ORDERS BY SIZE rows: size | closed | avg_flow  (now capturing 'closed' too)
order_pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d\.]+)\s*$")
total_pattern = re.compile(r"^\s*Total\s+(\d+)\s+([\d\.]+)\s*$")

### Data extraction
def extract_reports(mode_folder):
    avg_pods, comp_time, throughput = {}, {}, {}
    flow_data, closed_data = {}, {}
    files = glob(os.path.join(mode_folder, "report_*_Opt*_Seed*.txt"))
    print(f"\nAnalyzing {mode_folder}")
    print(f"Reports found: {len(files)}")

    for filepath in files:
        filename = os.path.basename(filepath)
        m = filename_pattern.match(filename)
        if not m:
            print("Unrecognized filename:", filename)
            continue
        scenario, seed = m.group(1), int(m.group(2))
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Main metrics
        for line in lines:
            if "Total number of items picked" in line:  # throughput
                value = float(line.split("=")[1].strip().rstrip("."))
                throughput.setdefault(scenario, {})[seed] = value
            elif "Average number of pod moving simultaneously" in line:  # average pods moving
                value = float(line.split("=")[1].strip().rstrip("."))
                avg_pods.setdefault(scenario, {})[seed] = value
            elif "Computational time spent for making decisions" in line:  # computational time
                value = float(line.split("=")[1].replace("sec.", "").strip())
                comp_time.setdefault(scenario, {})[seed] = value

        # ORDERS BY SIZE table: flow time AND number of closed orders per size
        inside_table = False
        for line in lines:
            if "ORDERS BY SIZE" in line:
                inside_table = True
                continue
            if not inside_table:
                continue
            m_total = total_pattern.match(line)  # Total row: closed, avg_flow
            if m_total:
                closed_data.setdefault((scenario, "Total"), {})[seed] = int(m_total.group(1))
                flow_data.setdefault((scenario, "Total"), {})[seed]   = float(m_total.group(2))
                break
            m_order = order_pattern.match(line)  # per-size row: size, closed, avg_flow
            if m_order:
                order_size = int(m_order.group(1))
                closed     = int(m_order.group(2))
                avg_flow   = float(m_order.group(3))
                closed_data.setdefault((scenario, order_size), {})[seed] = closed
                flow_data.setdefault((scenario, order_size), {})[seed]   = avg_flow

    return avg_pods, comp_time, throughput, flow_data, closed_data

### CSV saving  (matrix scenario x seed)
def save_matrix_csv(data, filename):
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Scenario"
    df = df.sort_index().reindex(sorted(df.columns), axis=1)
    df.to_csv(os.path.join(CSV_FOLDER, filename))
    return df

def save_flow_csv(flow_data, filename):
    rows = []
    for (scenario, order_size), values in flow_data.items():
        row = {"Scenario": scenario, "OrderSize": order_size}
        row.update(values)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    seed_cols = sorted(c for c in df.columns if isinstance(c, int))
    df = df[["Scenario", "OrderSize"] + seed_cols]
    sort_key = lambda x: 9999 if x == "Total" else int(x)  # keep Total last
    df = df.sort_values(by=["Scenario", "OrderSize"],
                        key=lambda col: col.map(sort_key) if col.name == "OrderSize" else col)
    df.to_csv(os.path.join(CSV_FOLDER, filename), index=False)
    return df

### LaTeX appendix tables  (scenario x seed matrix)
def matrix_to_latex(df, filename, caption, label, value_fmt="{:.1f}"):
    seeds = list(df.columns)
    col_spec = "l" + "r" * len(seeds)
    header = " & ".join(["\\textbf{Config.}"] + [f"\\texttt{{{s}}}" for s in seeds])

    body_lines = []
    for scenario, row in df.iterrows():
        cells = [f"\\texttt{{{scenario_label(scenario)}}}"]
        for s in seeds:
            v = row[s]
            cells.append("--" if pd.isna(v) else value_fmt.format(v))
        body_lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(body_lines)

    tex = (
            "\\begin{table}[htb]\n\\centering\n"
            "\\small\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n\\toprule\n"
            f"{header} \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n"
            "\\end{tabular}%\n"
            "}\n"
            "\\end{table}\n"
        )
    with open(os.path.join(TEX_FOLDER, filename), "w", encoding="utf-8") as f:
        f.write(tex)

### LaTeX table for flow time: only the 'Total' row per configuration
def flow_total_to_latex(flow_data, filename, caption, label, value_fmt="{:.1f}"):
    totals = {}
    for (scenario, order_size), seed_values in flow_data.items():
        if order_size == "Total":
            totals[scenario] = seed_values
    if not totals:
        print(f"[flow_total_to_latex] no Total rows found, {filename} not written")
        return
    df = pd.DataFrame.from_dict(totals, orient="index")
    df.index.name = "Scenario"
    df = df.sort_index().reindex(sorted(df.columns), axis=1)

    seeds = list(df.columns)
    col_spec = "l" + "r" * len(seeds)
    header = " & ".join(["\\textbf{Config.}"] + [f"\\texttt{{{s}}}" for s in seeds])
    body_lines = []
    for scenario, row in df.iterrows():
        cells = [f"\\texttt{{{scenario_label(scenario)}}}"]
        for s in seeds:
            v = row[s]
            cells.append("--" if pd.isna(v) else value_fmt.format(v))
        body_lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(body_lines)

    tex = (
            "\\begin{table}[htb]\n\\centering\n"
            "\\small\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n\\toprule\n"
            f"{header} \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n"
            "\\end{tabular}%\n"
            "}\n"
            "\\end{table}\n"
        )
    with open(os.path.join(TEX_FOLDER, filename), "w", encoding="utf-8") as f:
        f.write(tex)

### CSV for closed-orders-by-size (scenario x size, mean over seeds)
def save_closed_by_size_csv(closed_data, filename, max_size=15):
    scenarios = sorted({s for (s, sz) in closed_data.keys()},
                       key=lambda x: int(x) if str(x).isdigit() else x)
    rows = []
    for scenario in scenarios:
        seeds = set()
        for (s, sz), sv in closed_data.items():
            if s == scenario:
                seeds.update(sv.keys())
        row = {"Scenario": scenario}
        for size in list(range(1, max_size + 1)) + ["Total"]:
            d = closed_data.get((scenario, size), {})
            row[str(size)] = round(sum(d.get(seed, 0) for seed in seeds) / len(seeds), 2) if seeds else None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CSV_FOLDER, filename), index=False)
    return df

### LaTeX table: mean number of closed orders by order size (rows=config, cols=size)
def closed_by_size_to_latex(closed_data, filename, caption, label, max_size=15):
    scenarios = sorted({s for (s, sz) in closed_data.keys()},
                       key=lambda x: int(x) if str(x).isdigit() else x)
    sizes = list(range(1, max_size + 1))

    def mean_closed(scenario, size):
        seeds = set()
        for (s, sz), sv in closed_data.items():
            if s == scenario:
                seeds.update(sv.keys())
        if not seeds:
            return None
        d = closed_data.get((scenario, size), {})
        return sum(d.get(seed, 0) for seed in seeds) / len(seeds)

    ncols = len(sizes) + 1
    col_spec = "l" + "r" * ncols
    header = " & ".join(["\\textbf{Config.}"] +
                        [f"\\textbf{{{s}}}" for s in sizes] + ["\\textbf{Total}"])

    body_lines = []
    for scenario in scenarios:
        cells = [f"\\texttt{{{scenario_label(scenario)}}}"]
        for s in sizes:
            v = mean_closed(scenario, s)
            cells.append("--" if v is None else f"{v:.1f}")
        vt = mean_closed(scenario, "Total")
        cells.append("--" if vt is None else f"{vt:.1f}")
        body_lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(body_lines)

    tex = (
        "\\begin{table}[htb]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n\\toprule\n"
        f"{header} \\\\\n"
        f"\\cmidrule(lr){{2-{ncols}}}\n"
        "\\midrule\n"
        f"{body}\n\\bottomrule\n"
        "\\end{tabular}%\n}\n"
        "\\end{table}\n"
    )
    with open(os.path.join(TEX_FOLDER, filename), "w", encoding="utf-8") as f:
        f.write(tex)

### Two-stage replication sizing (Law, Simulation Modeling & Analysis)
def two_stage_replications(throughput, filename,
                           alpha=0.05, gamma=0.05, beta=None):
    rows = []
    for scenario, seed_values in throughput.items():
        vals = list(seed_values.values())
        n0 = len(vals)
        if n0 < 2:
            continue
        mean_val = sum(vals) / n0
        var = sum((x - mean_val) ** 2 for x in vals) / (n0 - 1)
        std = math.sqrt(var)
        if beta is not None:
            target = beta
        else:
            target = (gamma / (1 + gamma)) * abs(mean_val)
        n = n0
        while True:
            t_crit = t.ppf(1 - alpha / 2, df=n - 1)
            half_width = t_crit * std / math.sqrt(n)
            if half_width <= target or n > 100:
                break
            n += 1
        t0 = t.ppf(1 - alpha / 2, df=n0 - 1)
        hw_pilot = t0 * std / math.sqrt(n0)
        rows.append({
            "Scenario": scenario, "n0": n0,
            "Mean": round(mean_val, 2), "StdDev": round(std, 2),
            "HalfWidth_pilot": round(hw_pilot, 2), "Target": round(target, 2),
            "N_required": n, "Extra_needed": max(0, n - n0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Scenario")
    df.to_csv(os.path.join(CSV_FOLDER, filename), index=False)
    return df

### Main
for mode in ["Opt_False", "Opt_True"]:
    mode_folder = os.path.join(REPORT_FOLDER, mode)
    if not os.path.exists(mode_folder):
        print(f"Folder {mode_folder} not found")
        continue

    avg_pods, comp_time, throughput, flow_data, closed_data = extract_reports(mode_folder)

    # matrices (scenario x seed)
    df_thr  = save_matrix_csv(throughput, f"{mode}_throughput.csv")
    df_pods = save_matrix_csv(avg_pods,   f"{mode}_average_pods.csv")
    df_time = save_matrix_csv(comp_time,  f"{mode}_computational_time.csv")
    save_flow_csv(flow_data, f"{mode}_mean_flow_time.csv")
    save_closed_by_size_csv(closed_data, f"{mode}_closed_by_size.csv")

    two_stage_replications(throughput, f"{mode}_replications_throughput.csv",
                           alpha=0.05, gamma=0.02)

    mode_label = "Optimizer" if mode == "Opt_True" else "Greedy policies"

    # per-replication appendix tables
    matrix_to_latex(
        df_thr, f"{mode}_throughput.tex",
        caption=f"{mode_label}: per-replication throughput (items picked).",
        label=f"tab:app_throughput_{mode.lower()}", value_fmt="{:.0f}")
    matrix_to_latex(
        df_pods, f"{mode}_average_pods.tex",
        caption=f"{mode_label}: per-replication average number of pods moving simultaneously.",
        label=f"tab:app_pods_{mode.lower()}", value_fmt="{:.2f}")
    flow_total_to_latex(
        flow_data, f"{mode}_flow_time.tex",
        caption=f"{mode_label}: per-replication average flow time (s), total over all order sizes.",
        label=f"tab:app_flowtime_{mode.lower()}", value_fmt="{:.1f}")

    # NEW: mean number of closed orders by order size
    closed_by_size_to_latex(
        closed_data, f"{mode}_closed_by_size.tex",
        caption=f"{mode_label}: mean number of closed orders by order size, averaged over seeds.",
        label=f"tab:app_closed_{mode.lower()}")

print(f"\nCSV files written to:   {CSV_FOLDER}")
print(f"LaTeX tables written to: {TEX_FOLDER}")
print("Done.")