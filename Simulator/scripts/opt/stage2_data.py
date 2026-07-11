from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from bisect import bisect_left


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
    items_by_pod          : dict[int, list[int]] pod_id -> [im1, im2, ...]
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

    Reduced (call-scoped) network — PERFORMANCE
    ---------------------------------------------
    OptManager.nodes / outgoing_arc_idx / incoming_arc_idx are built once
    over the FULL warehouse (every pod, potentially thousands) and shared
    across every optimisation cycle. But within a single Stage-2 call only
    the pods in `from_RelPod_to_PodId` are ever routed — routes never
    target any location other than a pod's own storage cell or a
    workstation. In particular the arc lists anchored at workstation nodes
    (workstation -> pod storage, see OptManager._add_arcs(W, L)) scale
    with the TOTAL number of pods in the warehouse, not with the (much
    smaller) number of pods actually involved in this run.

    So, in addition to the raw OptManager-level structures, Stage2Data
    builds a *reduced* view scoped to `relevant_locations` = workstation
    positions ∪ storage locations of the selected pods only:

    relevant_locations : set[int]    Derived.
    nodes              : list[tuple] Derived — reduced (location, time) nodes.
    outgoing_arc_idx    : dict       Derived — reduced, node -> [arc_id, ...].
    incoming_arc_idx    : dict       Derived — reduced, node -> [arc_id, ...].
    idle_arc_id         : dict       Derived — node -> arc_id, O(1) self-loop lookup.
    arc_lookup          : dict       Derived — reduced (src_loc,dst_loc) -> sorted arc list.

    check_constraints / build_solution / _rebuild_pod_row in the local
    search module should use these `d.X` reduced structures (not
    `d.OptManager.X`) wherever they scan for routing-relevant arcs.
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
    items_by_pod:          dict
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
    ws_positions:        list       = field(init=False)
    earliest_t:          np.ndarray = field(init=False)
    relevant_locations:  set        = field(init=False)
    nodes:               list       = field(init=False)
    outgoing_arc_idx:    dict       = field(init=False)
    incoming_arc_idx:    dict       = field(init=False)
    idle_arc_id:         dict       = field(init=False)
    arc_lookup:          dict       = field(init=False)

    def __post_init__(self):
        self.ws_positions = [
            self.state.warehouse.workstations[w].position
            for w in range(self.OptManager.n_workstations)
        ]
        self._compute_earliest_t()
        self._build_reduced_network()

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

    def _build_reduced_network(self):
        """
        Build a call-scoped, reduced view of the time-space network
        restricted to `relevant_locations` = workstation positions ∪
        storage locations of the pods actually selected in Stage 1
        (from_RelPod_to_PodId) — instead of every pod storage location in
        the whole warehouse.

        This shrinks the per-node arc lists that the local search scans
        (outgoing_arc_idx / incoming_arc_idx at workstation nodes, and the
        EC13 node list) from O(n_pods_total) to O(n_pods_selected), and
        gives an O(1) idle-arc lookup on top (see `idle_arc_id`), removing
        a linear scan that used to be needed just to find the single
        "stay in place" arc at each node.
        """
        T = self.OptManager.N_TIME
        n_travel = len(self.OptManager.travelling_arcs)

        selected_storages = {
            self.warehouse.pods[p_id].storage_location
            for p_id in self.from_RelPod_to_PodId
        }
        self.relevant_locations = set(self.ws_positions) | selected_storages

        self.nodes = [(loc, t) for loc in self.relevant_locations for t in range(T)]

        self.outgoing_arc_idx = {}
        self.incoming_arc_idx = {}
        self.idle_arc_id      = {}

        for node in self.nodes:
            out_full = self.OptManager.outgoing_arc_idx.get(node, [])
            out_reduced = [
                a for a in out_full
                if self.OptManager.all_arcs[a][1][0] in self.relevant_locations
            ]
            self.outgoing_arc_idx[node] = out_reduced

            in_full = self.OptManager.incoming_arc_idx.get(node, [])
            self.incoming_arc_idx[node] = [
                a for a in in_full
                if self.OptManager.all_arcs[a][0][0] in self.relevant_locations
            ]

            # O(1) self-loop (idle) arc lookup for this node.
            for a in out_reduced:
                if a >= n_travel:
                    self.idle_arc_id[node] = a
                    break

        # arc_lookup: reuse OptManager's precomputed, static, sorted-by-
        # departure lookup if available (built once, independent of which
        # pods are selected — see OptManager.__init__), just filtering it
        # down to relevant_locations. Falls back to building it directly
        # (still filtered) if an older OptManager instance doesn't have it
        # precomputed yet.
        global_lookup = getattr(self.OptManager, "arc_lookup", None)
        if global_lookup is not None:
            self.arc_lookup = {
                key: arcs
                for key, arcs in global_lookup.items()
                if key[0] in self.relevant_locations and key[1] in self.relevant_locations
            }
        else:
            lookup: dict = {}
            for arc_id in range(n_travel):
                arc = self.OptManager.all_arcs[arc_id]
                src, dst = arc
                if src[0] not in self.relevant_locations or dst[0] not in self.relevant_locations:
                    continue
                key = (src[0], dst[0])
                lookup.setdefault(key, []).append((src[1], dst[1], arc_id, arc))
            for key in lookup:
                lookup[key].sort(key=lambda z: z[0])
            self.arc_lookup = lookup


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

    Note: the reduced network view (nodes / outgoing_arc_idx /
    incoming_arc_idx / idle_arc_id / arc_lookup) is built inside
    Stage2Data.__post_init__ — nothing to do here besides collecting the
    fields Stage2Data needs to compute it (from_RelPod_to_PodId, in
    particular, drives which pod storage locations stay "relevant").
    """
    n_orders = len(orders)

    # Gather all order_ids that are already open at some workstation
    opened_ids: set = set()
    for ws in state.warehouse.workstations:
        opened_ids |= set(ws.opened_orders)

    items_by_pod: dict[int, list[int]] = {}
    for im, _ in enumerate(relevant_pairs_for_x):
        p_id = pod_of_item[im]
        items_by_pod.setdefault(p_id, []).append(im)

    return Stage2Data(
        orders                = orders,
        orders_items          = orders_items,
        relevant_pairs_for_x  = relevant_pairs_for_x,
        items_of_order        = items_of_order,
        n_items_per_order     = np.array([len(orders_items[m]) for m in range(n_orders)]),
        orders_by_workstation = orders_by_workstation,
        order_to_ws           = order_to_ws_m,
        pod_of_item           = pod_of_item,
        items_by_pod          = items_by_pod,
        from_RelPod_to_PodId  = from_RelPod_to_PodId,
        from_PodId_to_RelPod  = from_PodId_to_RelPod,
        current_time          = state.current_time,
        arrival_times         = np.array([o.arrival_time for o in orders]),
        opened_order_ids      = opened_ids,
        OptManager            = OptManager,
        warehouse             = state.warehouse,
        state                 = state,
    )