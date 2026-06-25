"""
sim.py
------
Core discrete-event simulation engine.

Calling ``sim.run()`` multiple times produces independent replicas automatically:
each call rebuilds the full mutable state from scratch via ``warehouse_factory``.

Architecture
------------
SimulatorConfig   Immutable run parameters (frozen dataclass).
SimulatorState    All mutable simulation state (plain dataclass).
Simulator         Orchestrator: owns config, RNG and warehouse_factory.
                  Every ``run()`` call starts from a clean state.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.random import Generator

from scripts.core.entities import Order, Task, Event
from scripts.core.warehouse import Warehouse
from scripts.core.queues import PriorityQueue
from scripts.sim import event_handler as eh
from scripts.core.enums import OrderStatus, EventType, PodStatus
from scripts.stat.StatManager import StatManager
from scripts.opt.OptManager import OptManager



# Immutable configuration

@dataclass
class SimulatorConfig:
    """
    Immutable run-level parameters shared across all replicas.

    order_gen_config    : list[float]   [orders_per_hour, prob_1_item_order, geo_dist_param].
    warm_up             : float         Seconds discarded from statistics at run start. Default 0.
    path_to_save_stat   : str           Path in where to save report file.s
    optimization_enabled: bool          Optimisation-based assignment if True. Default False.
    """
    order_gen_config:        list[float]
    time_horizon:            float
    warm_up:                 float = 0.0
    path_to_save_stat:       str = "Simulator/output/report.txt"
    optimization_enabled:    bool  = False
    optimization_interval :  float = 15*60


# Mutable simulation state

@dataclass
class SimulatorState:
    """
    All mutable state for one simulation run.

    Rebuilt from scratch at the start of every ``Simulator.run()`` call so that
    consecutive runs are fully independent replicas.

    current_time     : float                        Clock of the last processed event (seconds).
    warehouse        : Warehouse                    Live environment (pods, robots, workstations).
    future_events    : PriorityQueue[Event]         Min-heap; RELEASE_TASK breaks ties first.
    orders_in_system : PriorityQueue[Order]         All orders, ordered by status then arrival time.
    released_tasks   : PriorityQueue[Task]          Floor-ready tasks.
    active_tasks     : dict[int, Task]              Executing tasks keyed by task_id.
    orders_counter   : int                          Monotonic order-ID counter.
    task_counter     : int                          Monotonic task-ID counter.
    event_count      : int                          Total events processed (for logging).
    closed_orders    : int                          Completed orders (for logging).
    """
    current_time:     float
    warehouse:        Warehouse
    future_events:    PriorityQueue
    orders_in_system: PriorityQueue
    released_tasks:   PriorityQueue
    active_tasks:     dict[int, Task]
    orders_counter:   int = 0
    task_counter:     int = 0
    event_count:      int = 0
    closed_orders:    int = 0


def build_state(warehouse: Warehouse) -> SimulatorState:
    """Build a clean :class:`SimulatorState` from a fresh *warehouse*."""
    return SimulatorState(
        current_time=0.0,
        warehouse=warehouse,
        future_events=PriorityQueue(
            key=lambda e: (e.time, e.type != EventType.RELEASE_TASK)
        ),
        orders_in_system=PriorityQueue(
            key=lambda o: (o.status != OrderStatus.BACKLOG, o.arrival_time),
            id_attr="order_id",
        ),
        released_tasks=PriorityQueue(
            key=lambda t: (
                warehouse.pods[t.pod_id].status != PodStatus.IDLE,
                t.priority,
            ),
            id_attr="task_id",
        ),
        active_tasks={},
    )



# Simulator

class Simulator:
    """
    Discrete-event simulation engine.

    Call ``run()`` as many times as needed — each call is an independent replica.

    Parameters
    ----------
    random_generator  : Generator               Seeded numpy RNG for all stochastic draws.
    config            : SimulatorConfig         Immutable run parameters.
    warehouse_factory : Callable[[], Warehouse] Zero-arg callable returning a fresh Warehouse.
                                                Invoked once at the start of every ``run()`` call.

    """

    def __init__(
        self,
        random_generator:  Generator,
        config:            SimulatorConfig,
        warehouse_factory: Callable[[], Warehouse],
    ) -> None:
        self.RANDOM_GENERATOR:  Generator                = random_generator
        self.config:            SimulatorConfig          = config
        self._warehouse_factory: Callable[[], Warehouse] = warehouse_factory

        # state and STAT_MANAGER are placeholders; run() overwrites them.
        self.state:        SimulatorState | None = None
        self.STAT_MANAGER: StatManager    | None = None



    def run(self, time_horizon: float) -> None:
        """
        Execute one independent simulation run up to *time_horizon* seconds.

        Rebuilds :class:`SimulatorState` and :class:`StatManager` from scratch
        at the very start, so consecutive calls are fully independent replicas
        with no shared mutable state.

        Parameters
        ----------
        time_horizon : float
            Simulation end time in seconds (e.g. ``8 * 3600``).
        """
        # Reset: build a clean state for this replica 
        fresh_warehouse   = self._warehouse_factory()
        self.state        = build_state(fresh_warehouse)
        self.config.time_horizon = time_horizon

        assert self.config.warm_up < time_horizon, "Warm-up for KPIs collection exceeds simulation time horizon."
        self.STAT_MANAGER = StatManager(fresh_warehouse, self.config.warm_up)

        if self.config.optimization_enabled:
            self.OPT_MANAGER = OptManager(fresh_warehouse)

        # Event loop 
        _log_banner(time_horizon)
        state    = self.state
        dispatch = self.build_dispatch()

        state.future_events.push(Event(time=1e-8, type=EventType.ARRIVAL_ORDER, info = 100))
        if self.config.optimization_enabled:
            state.future_events.push(Event(time=1, type=EventType.RUN_OPTIMIZER))

        event_to_process = state.future_events.pop()
        while event_to_process.time < time_horizon:
            state.current_time = event_to_process.time
            state.event_count += 1

            for ws in state.warehouse.workstations:
                assert len(ws.opened_orders) <= ws.order_capacity, f'Workstation {ws.workstation_id} has {len(ws.opened_orders)} orders opened'

            logging.debug(
                "   EVENT NUM %-4d  current_time = %s   %-20s  [queue size = %d]",
                state.event_count,
                _fmt_time(state.current_time),
                event_to_process.type.name,
                len(state.future_events),
            )

            self._process_event(event_to_process, state, dispatch)

            event_to_process = state.future_events.pop()

        #  Statistics 
        logging.info("  END SIMULATION.\n")
        logging.info("Writing statistics report ...")
        self.STAT_MANAGER.return_statistics(self.config, self.state, output_path = self.config.path_to_save_stat)
        self.STAT_MANAGER.reset_statistics()


    # Private helpers

    def build_dispatch(self) -> dict:
        """
        Return the EventType → handler mapping.

        Handlers signature: ``(event, state, sim)``
        - *state* : :class:`SimulatorState`  — mutable queues and counters.
        - *sim*   : :class:`Simulator`       — config and RNG.
        """
        return {
            EventType.ARRIVAL_ORDER:   eh.arrival_order,
            EventType.RUN_OPTIMIZER:   eh.run_optimizer,
            EventType.RELEASE_TASK:    eh.release_task,
            EventType.START_TASK:      eh.start_task,
            EventType.ARRIVAL_POD_WST: eh.arrival_pod_wst,
            EventType.OPEN_ORDER:      eh.open_order,
            EventType.START_PICKING:   eh.start_picking,
            EventType.END_PICKING:     eh.end_picking,
            EventType.CLOSE_ORDER:     eh.close_order,
            EventType.RETURN_POD:      eh.return_pod,
        }

    def _process_event(
        self, event: Event, state: SimulatorState, dispatch: dict
    ) -> None:
        """
        Dispatch *event* to its registered handler.
        """
        handler = dispatch.get(event.type)
        if handler is None:
            raise ValueError(f"Unhandled event type: {event.type}")
        handler(event, state, self)


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS``."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def _log_banner(time_horizon: float) -> None:
    """Emit an INFO banner with the simulation time horizon."""
    logging.info("\n=============================================")
    logging.info("  START SIMULATION  {TIME HORIZON = %s}", _fmt_time(time_horizon))