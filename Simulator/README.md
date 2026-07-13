# Tesi_LM Project – Simulator Folder Overview

## Folder Structure

```
Tesi_LM/Simulator/
├── run_simulation.py                      # Entry point: reads config, builds warehouse, runs simulation
├── experiments.csv                        # Numeric parameters for the simulation scenario
└── scripts/    
    ├── core/
    │   ├── enums.py                       # Enumerations: OrderStatus, RobotStatus, PodStatus, WorkstationPickingStatus, EventType
    │   ├── queues.py                      # Priorityqueue definition
    │   ├── entities.py                    # Domain entities: Visit, Task, Order, Robot, Pod, Workstation, Event
    │   └── warehouse.py                   # Phisical layout class and generation
    │
    ├── sim/
    │   ├── Simulator.py                   # Core DES engine: state, clock, event dispatch, simulation loop
    │   ├── event_handler.py               # Event logic: one function per event type
    │   └── utils.py       
    │
    ├── opt/
    │   ├── policies.py                    # Heuristic assignment policies (used when optimizer is disabled)
    |   ├── OptManager.py                  # Time-space network definition
    |   ├── local_search_stage1.py         # Implementation of a local search heuristic to solve the assignment problem
    |   ├── local_search_stafe2.py         # Implementation of a local search heuristic to solve the sequencing problem
    |   ├── stage2_data.py                 # Contain the main data structure to fed the sequencing problem
    |   ├── build_initial_x_stage2.py      # Contains the function to build a feasible initial solution for the stage 2 problem
    |   └── convert_OptSol_to_SimObj.py    # Contais a function to convert the decision variables into objects (`Task`) to fed the simulator
    │  
    └── stat/
        ├── Statanager.py                  # General KPIs collector and update method
        └── core.py                        # Sub-tracker definition: ResourceTracker, OrderFlowTracker, TimeWeightedMEanTracker


```

---

## `run_simulation.py`

Entry point of the simulation:

- Configures logging (writes to `output\logs\`)
- Seeds the random number generator (`numpy.random.default_rng`)
- Instantiates `Warehouse` from `config` parameters
- Optionally plots the layout via `warehouse.plot()`
- Creates a `Simulator` instance and calls `sim.run(TIME_HORIZON)`









