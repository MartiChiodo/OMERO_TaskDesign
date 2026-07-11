### SIMULATION CONFIGURATION
# All parameters are centralised here and imported by run_simulation.py


# Warehouse Layout 
NUM_PODS         = 100
NUM_SKUS         = 1000
NUM_ROBOTS       = 9
NUM_WORKSTATIONS = 4
NUM_SKUS_PER_POD = 20

GRID_ROWS = 10    # number of rows for the pod grid
GRID_COLS = 10    # number of columns for the pod grid

ROBOT_SPEED = 0.34       # cells per seconds (circa 1 m/s assuming cells are squares 1.5mx1.5m and L1 paths)


# Workstations
WS_ORDER_CAPACITY = 4   # M: max simultaneous open orders per workstation
WS_WORKLOAD_CAPACITY = 10   # max pods waiting in pod_queue per workstation 
                        # (more of an indication on how many active tasks reòlated to each ws simultaneosly)


# Parameters for order generation (Barnhart et al. 2024 approach)
PROB_1_ITEM_ORDER = 0.5
GEO_DIST_PARAM_ORDER = 0.65      # takes value in {0.25, 0.35, 0.45, 0.65}
INTERRARIVAL_TIME_ORDER = 3600/175       # equivalent of 175 orders per hour (unit time : sec)



# Simulation control 
TIME_HORIZON  = 2*60*60       # total simulation time in seconds 
DELTA_T_OPT = 15*60       # optimizer is called every DELTA_T_OPT seconds


# Pikcing time
POD_PROCESS_TIME = 5   # in seconds
ITEM_PROCESS_TIME = 5  # in seconds


