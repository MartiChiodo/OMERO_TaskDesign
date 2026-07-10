import os, sys, logging
import numpy as np
import numpy.random
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.core.warehouse import Warehouse
from scripts.sim.Simulator import Simulator, SimulatorConfig


def load_experiment(experiment_id: str) -> dict:
    csv_path = os.path.join(os.path.dirname(__file__), "experiments.csv")
    df = pd.read_csv(csv_path, dtype={"experiment_id": int})
    row = df[df["experiment_id"] == experiment_id]
    if row.empty:
        raise ValueError(f"Experiment '{experiment_id}' not found in experiments.csv")
    return row.iloc[0].to_dict()

def main():

    # EXPERIMENT TO SIMULATE
    # EXPERIMENT_IDS = [1,2,3,4] + [11,12,13,14] + [21,22,23,24] + [31,32,33,34]

    
    EXPERIMENT_IDS  = [41]
    SEED = 343310  
    OPTIM = True

    print(f"Usando SEED={SEED}, EXPERIMENT_IDS={EXPERIMENT_IDS}")

    base_dir = os.path.dirname(__file__)
    path_to_logs = os.path.join(base_dir, "plot_carini")
    path_to_reports = os.path.join(base_dir, "plot_carini")
    os.makedirs(path_to_logs, exist_ok=True)
    os.makedirs(path_to_reports, exist_ok=True)

    for EXPERIMENT_ID in EXPERIMENT_IDS:
        cfg = load_experiment(EXPERIMENT_ID)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=os.path.join(path_to_logs, f"logs_{EXPERIMENT_ID}_Opt{OPTIM}_Seed{SEED}.log"),
            encoding="utf-8",
            level=logging.DEBUG,
            datefmt='%Y-%m-%d %H:%M:%S',
            filemode="w",
            format="%(asctime)s %(levelname)s: %(message)s",
        )
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("gurobipy").setLevel(logging.WARNING)

        gen = numpy.random.default_rng(SEED)
        warehouse = Warehouse(
                random_generator          = gen,
                num_pods                  = int(cfg["num_pods"]),
                num_skus                  = int(cfg["num_skus"]),
                num_robots                = int(cfg["num_robots"]),
                num_workstations          = int(cfg["num_workstations"]),
                num_skus_per_pod          = int(cfg["num_skus_per_pod"]),
                grid_rows                 = int(cfg["grid_rows"]),
                grid_cols                 = int(cfg["grid_cols"]),
                ws_order_capacity         = int(cfg["ws_order_capacity"]),
                ws_released_task_capacity = int(cfg["ws_workload_capacity"]),
                robot_speed               = float(cfg["robot_speed"]),
                pod_process_time          = float(cfg["pod_process_time"]),
                item_process_time         = float(cfg["item_process_time"])
            )

        
        ### CALCOLO FREQUENZA SKU
        freq = Counter()

        for pod in warehouse.pods:
            for sku in pod.items:
                freq[sku] += 1

        # Ordina gli SKU per indice
        skus = sorted(freq.keys())
        frequencies = [freq[sku] for sku in skus]

        # Bar plot
        plt.figure(figsize=(12, 5))
        plt.bar(skus, frequencies)

        plt.xlabel("SKU ID")
        plt.ylabel("Number of pods containing SKU")
        plt.title("SKU popularity distribution across pods")

        plt.grid(axis="y", alpha=0.3)
        plt.show()

if __name__ == "__main__":
    main()