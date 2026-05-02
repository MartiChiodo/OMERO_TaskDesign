from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Stage2Data:
    """
    Bundles all static context required for Stage-2 scheduling.

    Built once after Stage 1 has fixed the assignment decisions (z1, x1)
    and passed as a single argument to every Stage-2 function, avoiding
    long argument lists across build_solution / check_constraints / local_search.

    Attributes
    ----------
    orders               : list[Order]
    orders_items         : list[list[int]]   Pending SKU list per order.
    relevant_pairs_for_x : list[tuple]       [(sku, order_idx), ...] — one entry per decision var im.
    items_of_order       : dict[int, list]   order_idx -> [im, ...]
    n_items_per_order    : np.ndarray        shape (n_orders,) — number of items per order.

    orders_by_workstation : list[set[int]]   ws_idx -> set of order indices.
    order_to_ws           : dict[int, int]   order_idx -> ws_idx.
    pod_of_item           : dict[int, int]   im -> pod_id.
    from_RelPod_to_PodId  : list[int]        rel_p -> pod_id.
    from_PodId_to_RelPod  : dict[int, int]   pod_id -> rel_p.

    current_time     : float
    arrival_times    : np.ndarray   shape (n_orders,) — order arrival timestamps.
    opened_order_ids : set[int]     order_ids already open at a workstation.

    OptManager : OptManager
    warehouse  : Warehouse
    state      : SimState

    ws_positions : list[int]        Derived — ws_idx -> grid cell position.
    earliest_t   : np.ndarray       Derived — im -> earliest feasible pick time.
    """

    # Orders 
    orders:                list
    orders_items:          list
    relevant_pairs_for_x:  list
    items_of_order:        dict
    n_items_per_order:     np.ndarray

    # Stage-1 assignment decisions 
    orders_by_workstation: list
    order_to_ws:           dict
    pod_of_item:           dict
    from_RelPod_to_PodId:  list
    from_PodId_to_RelPod:  dict

    # Simulation context
    current_time:      float
    arrival_times:     np.ndarray
    opened_order_ids:  set

    # References (pointers only — not duplicated)
    OptManager: object
    warehouse:  object
    state:      object

    # Derived fields — computed in __post_init__ 
    ws_positions: list       = field(init=False)
    earliest_t:   np.ndarray = field(init=False)

    def __post_init__(self):
        self.ws_positions = [
            self.state.warehouse.workstations[w].position
            for w in range(self.OptManager.n_workstations)
        ]
        self._compute_earliest_t()

    def _compute_earliest_t(self):
        """
        For each item im, find the earliest time step at which the pod carrying
        it can physically arrive at the assigned workstation, departing from
        storage at t=0. Used to seed the initial solution in local search.
        """
        n_travel = len(self.OptManager.travelling_arcs)
        T        = self.OptManager.N_TIME
        self.earliest_t = np.full(len(self.relevant_pairs_for_x), T - 1, dtype=int)

        for im, (_, m) in enumerate(self.relevant_pairs_for_x):
            w      = self.order_to_ws[m]
            ws_pos = self.ws_positions[w]
            p_id   = self.pod_of_item[im]
            stor   = self.warehouse.pods[p_id].storage_location

            # Scan all arcs departing from storage at t=0 towards this workstation
            for id_a in self.OptManager.outgoing_arc_idx.get((stor, 0), []):
                if id_a < n_travel:
                    arc = self.OptManager.all_arcs[id_a]
                    if arc[1][0] == ws_pos:
                        self.earliest_t[im] = min(self.earliest_t[im], arc[1][1])


def build_stage2_data(
    OptManager,
    state,
    orders:                list,
    orders_items:          list,
    relevant_pairs_for_x:  list,
    items_of_order:        dict,
    orders_by_workstation: list,
    order_to_ws_m:         dict,
    pod_of_item:           dict,
    from_RelPod_to_PodId:  list,
    from_PodId_to_RelPod:  dict,
) -> Stage2Data:
    """
    Convenience constructor — collects the few extra fields that need
    to be derived from state, then builds and returns a Stage2Data.
    Called at the boundary between Stage 1 and Stage 2.
    """
    n_orders = len(orders)

    # Gather all order_ids that are already open at some workstation
    opened_ids: set = set()
    for ws in state.warehouse.workstations:
        opened_ids |= set(ws.opened_orders)

    return Stage2Data(
        orders                = orders,
        orders_items          = orders_items,
        relevant_pairs_for_x  = relevant_pairs_for_x,
        items_of_order        = items_of_order,
        n_items_per_order     = np.array([len(orders_items[m]) for m in range(n_orders)]),
        orders_by_workstation = orders_by_workstation,
        order_to_ws           = order_to_ws_m,
        pod_of_item           = pod_of_item,
        from_RelPod_to_PodId  = from_RelPod_to_PodId,
        from_PodId_to_RelPod  = from_PodId_to_RelPod,
        current_time          = state.current_time,
        arrival_times         = np.array([o.arrival_time for o in orders]),
        opened_order_ids      = opened_ids,
        OptManager            = OptManager,
        warehouse             = state.warehouse,
        state                 = state,
    )