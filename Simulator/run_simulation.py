import os, sys, logging
import numpy as np
import numpy.random
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.core.warehouse import Warehouse
from scripts.sim.Simulator import Simulator, SimulatorConfig


# The 12 configurations and the 10 seeds. Order matters: the SLURM array index
# is decoded into a (config, seed) pair using these two lists, so never reorder
# them without regenerating the array indices.
EXPERIMENT_IDS = [11, 12, 13, 14, 31, 32, 33, 34, 51, 52, 53, 54]
SEEDS = [343310, 293874, 301060, 300871, 30201,
         50102, 987034, 570183, 789124, 612937]
SEEDS = [343310]*4 + [293874]*4 + [789124]*4 + [987034]*4
EXPERIMENT_IDS = [0]
OPTIM = True


def load_experiment(experiment_id: int) -> dict:
    csv_path = os.path.join(os.path.dirname(__file__), "experiments.csv")
    df = pd.read_csv(csv_path, dtype={"experiment_id": int})
    row = df[df["experiment_id"] == experiment_id]
    if row.empty:
        raise ValueError(f"Experiment '{experiment_id}' not found in experiments.csv")
    return row.iloc[0].to_dict()


def decode_task(task_id: int) -> list[tuple[int, int]]:
    """
    Map a SLURM array index to the (experiment_id, seed) pairs it must run.

    With 12 configs x 10 seeds = 120 pairs, using --array=0-119 gives one pair
    per task (task_id // 10 -> config, task_id % 10 -> seed): short, backfillable
    tasks. The list return type keeps the caller uniform with the local case.
    """
    n_seeds = len(SEEDS)
    if not (0 <= task_id < len(EXPERIMENT_IDS) * n_seeds):
        raise ValueError(f"Task id {task_id} out of range for "
                         f"{len(EXPERIMENT_IDS)}x{n_seeds} = "
                         f"{len(EXPERIMENT_IDS)*n_seeds} pairs")
    exp = EXPERIMENT_IDS[task_id // n_seeds]
    seed = SEEDS[task_id % n_seeds]
    return [(exp, seed)]


def build_run_list() -> list[tuple[int, int]]:
    """
    Decide which (config, seed) pairs this process runs.

    On the cluster: one pair, decoded from SLURM_ARRAY_TASK_ID.
    Locally (no SLURM var): every config x every seed, in sequence.
    """
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_id is not None:
        return decode_task(int(slurm_id))
    # local: full sweep
    return [(exp, seed) for exp in EXPERIMENT_IDS for seed in SEEDS]


def run_one(experiment_id: int, seed: int) -> None:
    cfg = load_experiment(experiment_id)

    base_dir = os.path.dirname(__file__)
    path_to_logs = os.path.join(base_dir, "output", "logs", "exact_solver", f"Opt_{OPTIM}")
    path_to_reports = os.path.join(base_dir, "output", "reports", "exact_solver", f"Opt_{OPTIM}")
    os.makedirs(path_to_logs, exist_ok=True)
    os.makedirs(path_to_reports, exist_ok=True)

    # reset logging handlers so each run writes to its own file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=os.path.join(path_to_logs, f"logs_{experiment_id}_Opt{OPTIM}_Seed{seed}.log"),
        encoding="utf-8",
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode="w",
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("gurobipy").setLevel(logging.WARNING)

    print(f"\n\nRunning EXPERIMENT_ID={experiment_id}, SEED={seed}, OPTIM={OPTIM}", flush=True)

    gen = numpy.random.default_rng(seed)

    sim = Simulator(
        seed=seed,
        config=SimulatorConfig(
            order_gen_config=[
                float(cfg["interarrival_time"]),
                float(cfg["prob_1_item_order"]),
                float(cfg["geo_dist_param"])
            ],
            warm_up=float(cfg["warm_up"]),
            time_horizon=None,
            initial_backlog_size=int(cfg["initial_backlog_size"]),
            path_to_save_stat=os.path.join(
                path_to_reports, f"report_{experiment_id}_Opt{OPTIM}_Seed{seed}.txt"),
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

    del sim, gen


def main():
    runs = build_run_list()
    print(f"This process will run {len(runs)} (config, seed) pair(s): {runs}", flush=True)
    for experiment_id, seed in runs:
        run_one(experiment_id, seed)


if __name__ == "__main__":
    main()