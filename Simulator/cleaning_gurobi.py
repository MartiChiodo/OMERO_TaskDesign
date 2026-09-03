import signal
import gurobipy as gp

env = gp.Env()

def cleanup(signum, frame):
    """Ensure the Gurobi environment is disposed on SIGTERM."""
    print(f"Signal {signum} received — disposing Gurobi environment.")
    env.dispose()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, cleanup)