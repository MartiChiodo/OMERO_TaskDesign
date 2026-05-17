import os, sys, logging
import numpy as np
import numpy.random
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Simulator.scripts.core.warehouse import Warehouse
from Simulator.scripts.sim.Simulator import Simulator, SimulatorConfig


def load_experiment(experiment_id: str) -> dict:
    csv_path = os.path.join(os.path.dirname(__file__), "experiments.csv")
    df = pd.read_csv(csv_path, dtype={"experiment_id": int})
    row = df[df["experiment_id"] == experiment_id]
    if row.empty:
        raise ValueError(f"Experiment '{experiment_id}' not found in experiments.csv")
    return row.iloc[0].to_dict()

def main():

    # EXPERIMENT TO SIMULATE
    EXPERIMENT_IDS = [1,2,3,4,5,6] + [7,8,9,10,11,12] + [13,14,15,16] + [19,20,21,22]
    EXPERIMENT_IDS = [9,10]
    SEED = 343310
    OPTIM = True

    base_dir = os.path.dirname(__file__)
    path_to_logs = os.path.join(base_dir, "output", "logs", f"Opt_{OPTIM}")
    path_to_reports = os.path.join(base_dir, "output", "reports", f"Opt_{OPTIM}")
    os.makedirs(path_to_logs, exist_ok=True)
    os.makedirs(path_to_reports, exist_ok=True)

    for EXPERIMENT_ID in EXPERIMENT_IDS:
        cfg = load_experiment(EXPERIMENT_ID)
        # print(cfg.keys())

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=os.path.join(path_to_logs, f"logs_{EXPERIMENT_ID}_Opt{OPTIM}_Seed{SEED}.log"),
            encoding="utf-8",
            level=logging.INFO,
            datefmt="%H:%M:%S",
            filemode="w",
            format="%(asctime)s %(levelname)s: %(message)s",
        )
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("gurobipy").setLevel(logging.WARNING)

        gen = numpy.random.default_rng(SEED)

        sim = Simulator(
            random_generator=gen,
            config=SimulatorConfig(
                order_gen_config=[
                    float(cfg["interarrival_time"]),
                    float(cfg["prob_1_item_order"]),
                    float(cfg["geo_dist_param"])
                ],
                warm_up=float(cfg["warm_up"]),
                time_horizon=None,
                path_to_save_stat=os.path.join(path_to_reports, f"report_{EXPERIMENT_ID}_Opt{OPTIM}_Seed{SEED}.txt"),
                optimization_enabled=OPTIM,
                optimization_interval=float(cfg["delta_t_opt"])
            ),
            warehouse_factory=lambda: Warehouse(
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
        )

        sim.run(float(cfg["time_horizon"]))

if __name__ == "__main__":
    main()