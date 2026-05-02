from __future__ import annotations

import logging
from collections import defaultdict
from itertools import product

import numpy as np

from Simulator.scripts.core.warehouse import Warehouse
from Simulator.scripts.core.enums import OrderStatus


### CONSTANTS
OBATCH_SIZE = 70    # max orders pulled from the backlog per optimisation cycle
TIME_UNIT   = 30    # seconds per discrete time period
N_TIME      = 70    # number of discrete periods in the scheduling horizon


class OptManager:
    """
    Manages the MIP optimisation pipeline for the warehouse simulator.

    Static data (warehouse topology, time-space network) is computed once at
    construction time.  Simulation-dependent data (orders, tasks) is injected
    at each optimisation call via :meth:`solve_task_design_and_assignment`.

    Attributes
    ----------
    nodes            : list[tuple]             All (location, time) nodes.
    travelling_arcs  : list[list[tuple]]       Arcs for feasible pod movements.
    idle_arcs        : list[list[tuple]]       Arcs for pods staying in place.
    all_arcs         : list[list[tuple]]       travelling_arcs + idle_arcs.
    incoming_arc_idx : dict[tuple, list[int]]  Arc indices arriving at each node.
    outgoing_arc_idx : dict[tuple, list[int]]  Arc indices leaving each node.
    pod_indices_by_sku : dict[int, list[int]]  Pod indices that stock each SKU.
    """

    def __init__(self, warehouse: Warehouse) -> None:
        self._warehouse = warehouse

        self.n_skus         = warehouse.num_skus
        self.n_pods         = len(warehouse.pods)
        self.n_workstations = len(warehouse.workstations)

        # Pod storage locations and workstation positions (used in arc construction)
        self._L = [p.storage_location for p in warehouse.pods]
        self._W = [ws.position        for ws in warehouse.workstations]

        # Map each SKU to the pods that carry it (restricts x1/x2 variable domains)
        self.pod_indices_by_sku: dict[int, list[int]] = defaultdict(list)
        for ip, pod in enumerate(warehouse.pods):
            for sku in pod.items:
                self.pod_indices_by_sku[sku].append(ip)

        # Workstation parameters (assumed uniform across all stations)
        ws0 = warehouse.workstations[0]
        self.CAP_WS     = ws0.order_capacity
        self.DELTA_ITEM = ws0.item_process_time
        self.DELTA_POD  = ws0.pod_process_time
        self.N_TIME     = N_TIME
        self.TIME_UNIT  = TIME_UNIT

        logging.info("[OptManager] Building time-space network ...")
        self.nodes, self.travelling_arcs, self.idle_arcs = \
            self.build_network(warehouse, self._L, self._W)

        self.all_arcs = self.travelling_arcs + self.idle_arcs

        # Index arcs by destination and source node for O(1) constraint lookup
        self.incoming_arc_idx: dict[tuple, list[int]] = defaultdict(list)
        self.outgoing_arc_idx: dict[tuple, list[int]] = defaultdict(list)
        for idx, (src, dst) in enumerate(self.all_arcs):
            self.outgoing_arc_idx[src].append(idx)
            self.incoming_arc_idx[dst].append(idx)

        logging.info(
            "[OptManager] Network ready: %d nodes, %d travelling arcs, %d idle arcs.",
            len(self.nodes), len(self.travelling_arcs), len(self.idle_arcs),
        )


    ### Network construction

    def build_network(
        self,
        warehouse: Warehouse,
        L: list[int],
        W: list[int],
    ) -> tuple[list, list, list]:
        """
        Build the time-space network for pod routing.

        Nodes are (location, time) pairs.  Travelling arcs connect locations
        reachable within the horizon; idle arcs represent staying in place.
        Only pod↔workstation and workstation↔workstation movements are modelled
        (pod↔pod arcs are excluded by design).

        Parameters
        ----------
        L : list of pod storage cell ids.
        W : list of workstation cell ids.

        Returns
        -------
        nodes            : list[tuple]
        travelling_arcs  : list[list[tuple]]
        idle_arcs        : list[list[tuple]]
        """
        all_locations = L + W
        nodes = list(product(all_locations, range(N_TIME)))

        # Discretise pairwise travel times (ceiling to nearest time unit)
        travel_dt: dict[tuple, int] = {}
        for l1 in all_locations:
            for l2 in all_locations:
                if l1 == l2:
                    continue
                travel_dt[(l1, l2)] = int(np.ceil(
                    1.1 * warehouse.travel_time(
                        warehouse.cell2coord(l1),
                        warehouse.cell2coord(l2),
                        None,
                    ) / TIME_UNIT
                ))

        travelling_arcs: list = []

        def _add_arcs(sources: list, destinations: list) -> None:
            """Append all time-feasible arcs from each source to each destination."""
            for l1 in sources:
                for l2 in destinations:
                    if l1 == l2:
                        continue
                    dt = travel_dt[(l1, l2)]
                    if dt >= N_TIME:
                        continue
                    for t1 in range(N_TIME - dt):
                        travelling_arcs.append([(l1, t1), (l2, t1 + dt)])

        _add_arcs(L, W)   # pod storage → workstation
        _add_arcs(W, L)   # workstation → pod storage
        _add_arcs(W, W)   # workstation → workstation

        # Idle arcs: pod or workstation stays at the same cell each period
        idle_arcs = [
            [(loc, t), (loc, t + 1)]
            for (loc, t) in product(all_locations, range(N_TIME))
            if t + 1 < N_TIME
        ]

        return nodes, travelling_arcs, idle_arcs


    ### Order extraction

    def extract_orders(self, state) -> tuple[list, list]:
        """
        Collect the orders to optimise and their pending item lists.

        Combines a fresh backlog batch with orders already at workstations.
        For open orders, items already covered by active tasks are subtracted.

        Returns
        -------
        orders       : list[Order]
        orders_items : list[list[int]]   Pending SKU list per order (same index).
        """
        ws_orders       = []
        ws_orders_items = []

        for ws in state.warehouse.workstations:
            # Visits currently being processed at this workstation
            active_visits = [
                visit
                for task_id in ws.active_tasks
                for visit in state.active_tasks[task_id].stops
                if visit.workstation_id == ws.workstation_id
            ]

            # Buffered orders — all items still pending
            for order_id in ws.order_buffer:
                o = state.orders_in_system.get(order_id)
                if o is not None:
                    ws_orders.append(o)
                    ws_orders_items.append(list(o.items_required))

            # Open orders — exclude items already claimed by active task visits
            for order_id in ws.opened_orders:
                o = state.orders_in_system.get(order_id)
                if o is None:
                    continue
                covered = {
                    item
                    for visit in active_visits
                    if order_id in visit.orders
                    for item in visit.items
                }
                remaining = list(o.items_pending - covered)
                if remaining:
                    ws_orders.append(o)
                    ws_orders_items.append(remaining)

        # Backlog orders — pull up to OBATCH_SIZE - (already collected)
        backlog       = []
        backlog_items = []
        n_to_consider = min(OBATCH_SIZE - len(ws_orders), len(state.orders_in_system))
        l_to_push     = []

        while len(backlog) < n_to_consider and len(state.orders_in_system) > 0:
            o = state.orders_in_system.pop()
            l_to_push.append(o)
            if o.status == OrderStatus.BACKLOG:
                backlog.append(o)
                backlog_items.append(list(o.items_pending))

        # Restore the priority queue
        for o in l_to_push:
            state.orders_in_system.push(o)

        return ws_orders + backlog, ws_orders_items + backlog_items


    ### Task design and assignment

    def solve_task_design_and_assignment(self, sim, state):
        """
        Run the full optimisation pipeline and convert the solution into Tasks.

        Calls :func:`solve_by_decomposition` for Stage 1 (assignment) and
        either Stage 2 MIP or local search (scheduling), then converts the
        result into simulator Task objects via :func:`convert_OptSol_to_SimObj`.

        Returns
        -------
        orders              : list[Order]
        ordered_orders_by_w : dict[int, list[int]]   Order indices sorted by
                              start time, per workstation.
        tasks               : list[Task]
        """
        from Simulator.scripts.opt.opt_solver import optimizer_solver
        from Simulator.scripts.opt.utils import convert_OptSol_to_SimObj

        st2_data, x, v, y = optimizer_solver(self, state)

        orders, ordered_orders_by_w, tasks = convert_OptSol_to_SimObj(st2_data, x, v, y)

        return orders, ordered_orders_by_w, tasks