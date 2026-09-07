import os
import re
from glob import glob
import pandas as pd

import math
from scipy.stats import t

### Folders
REPORT_FOLDER = os.path.dirname(os.path.abspath(__file__))

### Filename and table patterns
filename_pattern = re.compile(r"report_(.+?)_Opt.+?_Seed(\d+)\.txt")
order_pattern = re.compile(r"^\s*(\d+)\s+\d+\s+([\d\.]+)\s*$")
total_pattern = re.compile(r"^\s*Total\s+\d+\s+([\d\.]+)\s*$")

### Data extraction
def extract_reports(mode_folder):
    avg_pods, comp_time, throughput, flow_data = {}, {}, {}, {}
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

        # Mean flow time table
        inside_table = False
        for line in lines:
            if "ORDERS BY SIZE" in line:
                inside_table = True
                continue
            if not inside_table:
                continue
            m_total = total_pattern.match(line)  # Total row
            if m_total:
                flow_data.setdefault((scenario, "Total"), {})[seed] = float(m_total.group(1))
                break
            m_order = order_pattern.match(line)  # per-size row
            if m_order:
                order_size, avg_flow = int(m_order.group(1)), float(m_order.group(2))
                flow_data.setdefault((scenario, order_size), {})[seed] = avg_flow

    return avg_pods, comp_time, throughput, flow_data

### CSV saving
def save_matrix_csv(data, filename):
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Scenario"
    df = df.sort_index().reindex(sorted(df.columns), axis=1)
    df.to_csv(os.path.join(REPORT_FOLDER, filename))

def save_flow_csv(flow_data, filename):
    rows = []
    for (scenario, order_size), values in flow_data.items():
        row = {"Scenario": scenario, "OrderSize": order_size}
        row.update(values)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return
    seed_cols = sorted(c for c in df.columns if isinstance(c, int))
    df = df[["Scenario", "OrderSize"] + seed_cols]
    sort_key = lambda x: 9999 if x == "Total" else int(x)  # keep Total last
    df = df.sort_values(by=["Scenario", "OrderSize"],
                        key=lambda col: col.map(sort_key) if col.name == "OrderSize" else col)
    df.to_csv(os.path.join(REPORT_FOLDER, filename), index=False)


### Two-stage replication sizing (Law, Simulation Modeling & Analysis)
def two_stage_replications(throughput, filename,
                           alpha=0.05, gamma=0.05, beta=None):
    """
    For each scenario, use the pilot replications (the seeds already run)
    to estimate how many replications are needed.

    alpha  -> significance level (0.05 => 95% confidence)
    gamma  -> relative precision (e.g. 0.05 = 5%); used when beta is None
    beta   -> absolute precision (same units as throughput); overrides gamma
    """
    rows = []
    for scenario, seed_values in throughput.items():
        vals = list(seed_values.values())
        n0 = len(vals)
        if n0 < 2:
            continue  # need at least 2 reps to estimate variance

        mean_val = sum(vals) / n0
        var = sum((x - mean_val) ** 2 for x in vals) / (n0 - 1)
        std = math.sqrt(var)

        # target half-width
        if beta is not None:
            target = beta                                   # absolute
        else:
            target = (gamma / (1 + gamma)) * abs(mean_val)  # relative

        # iterative search: smallest n >= n0 whose half-width <= target
        n = n0
        while True:
            t_crit = t.ppf(1 - alpha / 2, df=n - 1)
            half_width = t_crit * std / math.sqrt(n)
            if half_width <= target or n > 100:
                break
            n += 1

        # half-width already achieved with the pilot
        t0 = t.ppf(1 - alpha / 2, df=n0 - 1)
        hw_pilot = t0 * std / math.sqrt(n0)

        rows.append({
            "Scenario": scenario,
            "n0": n0,
            "Mean": round(mean_val, 2),
            "StdDev": round(std, 2),
            "HalfWidth_pilot": round(hw_pilot, 2),
            "Target": round(target, 2),
            "N_required": n,
            "Extra_needed": max(0, n - n0),
        })

    df = pd.DataFrame(rows).sort_values("Scenario")
    df.to_csv(os.path.join(REPORT_FOLDER, filename), index=False)
    return df

### Main
for mode in ["Opt_False", "Opt_True"]:
    mode_folder = os.path.join(REPORT_FOLDER, mode)
    if not os.path.exists(mode_folder):
        print(f"Folder {mode_folder} not found")
        continue
    avg_pods, comp_time, throughput, flow_data = extract_reports(mode_folder)
    save_matrix_csv(avg_pods, f"{mode}_average_pods.csv")
    save_matrix_csv(comp_time, f"{mode}_computational_time.csv")
    save_matrix_csv(throughput, f"{mode}_throughput.csv")
    save_flow_csv(flow_data, f"{mode}_mean_flow_time.csv")
    two_stage_replications(throughput, f"{mode}_replications_throughput.csv",
                           alpha=0.05, gamma=0.02)  # 95% conf, 5% relative

print("\nCSV files created successfully.")