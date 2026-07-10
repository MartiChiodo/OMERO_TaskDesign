from __future__ import annotations
import numpy as np
from bisect import bisect_left
import logging

from .stage2_data import Stage2Data
from .build_initial_x_v1 import build_initial_x

"""
Stage-2 local search: picking-time scheduling (x) and pod routing (y) over a
time-space network.

Decision variables
-------------------
x[im, t] : binary
    1 if item-order pair `im` (relevant_pairs_for_x[im] = (item, order)) has
    already been picked by time step t. x is non-decreasing in t (EC19):
    once an item is picked it stays picked for the rest of the horizon.
f[m, t] : binary
    1 if order m has started picking by time t (i.e. at least one of its
    items has been picked). Derived from x, never a free variable.
g[m, t] : binary
    1 if order m has been fully completed (all its items picked) by time t-1.
    Derived from x, never a free variable.
v[m, t] : binary
    v = f - g. 1 if order m is "in progress" at time t (started but not yet
    finished). Also derived from x.
y[p, a] : binary
    1 if pod p traverses arc `a` of the time-expanded transportation network
    (all_arcs), i.e. the routing decision for each pod over time.

Only x is truly free during the local search; f, g, v are always recomputed
deterministically from x, and y is (re)built from x only when a candidate
solution is about to be evaluated for full feasibility / accepted as the
new incumbent (build_solution / _rebuild_pod_row).
"""


### FAST HELPERS (no y needed)
# These mirror, on x only, the more expensive checks that build_solution /
# check_constraints perform once y is also available. They are used
# during the local search loop (cheap) and only the winning candidate is
# ever passed through the full, y-aware check_constraints.

def _build_fgv(x: np.ndarray, d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute f, g, v from x only, no pod routing.
    Identical logic to the corresponding block in build_solution.

    For each order m: t_start is the earliest time index at which any of
    its items is picked, t_end is one past the latest such time index.
    f[m] = 1 for t >= t_start ("order started"); g[m] = 1 for t >= t_end
    ("order finished"); v[m] = f[m] - g[m] ("order in progress").

    For orders that were already opened before the current horizon
    (order_id in d.opened_order_ids), v is forced to 1 for the whole
    interval up to t_end (they were already "in progress" at t = 0), and
    f is redefined as v + g to stay consistent with v = f - g.
    """
    T      = x.shape[1]
    M      = len(d.orders)
    all_ms = np.array([m for (_, m) in d.relevant_pairs_for_x])

    # For each item-order pair, index of the first time step where x == 1
    # (i.e. len(T) if the item is never picked within the horizon).
    first_one_idx = (x == 0).sum(axis=1)

    t_start = np.full(M, T, dtype=int)
    t_end   = np.zeros(M,  dtype=int)
    np.minimum.at(t_start, all_ms, first_one_idx)   # earliest pick per order
    np.maximum.at(t_end,   all_ms, first_one_idx)   # latest pick per order
    t_end += 1

    time_range = np.arange(T)
    f = (time_range[np.newaxis, :] >= t_start[:, np.newaxis]).astype(np.float64)
    g = (time_range[np.newaxis, :] >= t_end[:, np.newaxis]  ).astype(np.float64)
    v = f - g

    # Orders already open before the horizon: force "in progress" from t=0.
    if len(d.opened_order_ids) > 0:
        for m, order in enumerate(d.orders):
            if order.order_id in d.opened_order_ids:
                v[m, :t_end[m]] = 1
        f = v + g

    return f, g, v


def _check_x_fast(x: np.ndarray, f: np.ndarray, g: np.ndarray,
                  v: np.ndarray, d) -> bool:
    """
    Full x-only feasibility check (no y constraints).
    Returns True iff all x-only constraints hold.
    """
    T  = x.shape[1]
    M  = len(d.orders)
    dx = np.diff(x, axis=1)

    # x must never decrease over time (an item, once picked, stays picked).
    if (dx < -1e-6).any():
        return False
    # workstation SKU-throughput capacity, evaluated via v (order in progress).
    for order_ids in d.orders_by_workstation:
        if (v[list(order_ids), :].sum(axis=0) > d.OptManager.CAP_WS + 1e-6).any():
            return False
    # v must equal f - g.
    if (np.abs(v - (f - g)) > 1e-6).any():
        return False
    # f[m,t] >= x[im,t] for every item im of order m.
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        if (f[m] < x[im] - 1e-6).any():
            return False
    # g[m,t] <= x[im,t-1] for every item im of order m.
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        if (g[m, 1:] > x[im, :-1] + 1e-6).any():
            return False
    # f must be non-decreasing in t.
    if (np.diff(f, axis=1) < -1e-6).any():
        return False
    # g must be non-decreasing in t.
    if (np.diff(g, axis=1) < -1e-6).any():
        return False
    # g cannot be lower than what the picking progress already
    # guarantees (workload-consistency lower bound on g).
    for m in range(M):
        ims     = d.items_of_order[m]
        n_items = int(d.n_items_per_order[m])
        lb      = x[ims, :-1].sum(axis=0) - (n_items - 1)
        if (g[m, 1:] < lb - 1e-6).any():
            return False
    # initial condition for opened orders (v[m,0] = 1) or, for
    # not-yet-opened orders, f can only be active if items were picked.
    for m, order in enumerate(d.orders):
        if order.order_id in d.opened_order_ids:
            if not np.isclose(float(v[m, 0]), 1.0):
                return False
        else:
            ims = d.items_of_order[m]
            if (f[m] > x[ims, :].sum(axis=0) + 1e-6).any():
                return False
    return True


def _fast_update_fgv_from_move(
    x_cand: np.ndarray,
    f_curr: np.ndarray,
    g_curr: np.ndarray,
    v_curr: np.ndarray,
    move,
    im_by_order: dict[int, list[int]],
    first_one_idx: np.ndarray,
    d,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Incrementally update f, g, v after applying a move.

    Instead of recomputing f, g, v for every order (expensive), only the
    rows of the orders actually touched by the move are recomputed, using
    the same t_start/t_end logic as _build_fgv. All other rows are copied
    unchanged from the current solution.

    `move` types and which orders they affect:
      - 'item'/'rnd_item'   : a single item-order pair -> its order.
      - 'multi_item'        : several item-order pairs -> their orders.
      - 'order'              : a whole order -> itself.
      - 'swap'               : two orders -> both of them.
    """
    T = x_cand.shape[1]

    f_new = f_curr.copy()
    g_new = g_curr.copy()
    v_new = v_curr.copy()

    affected_orders: set[int] = set()

    if move[0] == "item":
        _, im, _ = move
        _, m = d.relevant_pairs_for_x[int(im)]
        affected_orders.add(m)
    elif move[0] == "rnd_item":
        _, im, _ = move
        _, m = d.relevant_pairs_for_x[int(im)]
        affected_orders.add(m)
    elif move[0] == "multi_item":
        _, ims, _ = move
        for im in ims:
            _, m = d.relevant_pairs_for_x[int(im)]
            affected_orders.add(m)
    elif move[0] == "order":
        _, m, _ = move
        affected_orders.add(int(m))
    elif move[0] == "swap":
        _, m1, m2 = move
        affected_orders.add(int(m1))
        affected_orders.add(int(m2))

    time_range = np.arange(T)

    # Recompute f, g, v only for the orders touched by this move.
    for m in affected_orders:
        ims             = d.items_of_order[m]
        item_pick_times = first_one_idx[ims]
        valid           = item_pick_times[item_pick_times < T]

        if len(valid) == 0:
            t_start = T
            t_end   = T
        else:
            t_start = valid.min()
            t_end   = valid.max() + 1

        f_row = (time_range >= t_start).astype(np.float64)
        g_row = (time_range >= t_end).astype(np.float64)
        v_row = f_row - g_row

        order = d.orders[m]
        if order.order_id in d.opened_order_ids:
            v_row[:t_end] = 1.0
            f_row = v_row + g_row

        f_new[m] = f_row
        g_new[m] = g_row
        v_new[m] = v_row

    return f_new, g_new, v_new


### SOLUTION BUILDER
# Expands a picking matrix x into the full solution (f, g, v, y) by
# deciding the actual pod routes over the time-expanded network. This is
# the expensive step of the pipeline: it is only ever called on the
# current incumbent, never on every neighbour explored during the search.

def build_solution(x: np.ndarray, d) -> tuple:
    """
    Derive the full solution (f, g, v, y) from the picking matrix x.
    Expensive — call only for the final best solution, not during search.

    For each pod, builds a chronological list of "events" (workstation,
    arrival time) it must visit, one per item it supplies, then threads a
    feasible path through the time-expanded network connecting consecutive
    events (going through the pod's storage location when convenient,
    otherwise moving there directly), padding the remaining time with idle
    arcs. If an event cannot be reached in time, the pod is left in place
    (a warning is logged) — this preserves flow-conservation (EC16) at the
    cost of a potential EC18 violation, to be caught by check_constraints.
    """
    n_travel = len(d.OptManager.travelling_arcs)
    T        = x.shape[1]

    f, g, v = _build_fgv(x, d)

    y = np.zeros(
        (len(d.from_RelPod_to_PodId), len(d.OptManager.all_arcs)),
        dtype=np.float64,
    )

    first_one_idx = (x == 0).sum(axis=1)

    def add_idle_arcs(y, p_rel: int, loc: int, t_from: int, t_to: int) -> None:
        """Fill the time window [t_from, t_to) at a fixed location with
        self-loop / waiting arcs for pod p_rel."""
        for t in range(t_from, t_to):
            for id_a in d.OptManager.outgoing_arc_idx.get((loc, t), []):
                if d.OptManager.all_arcs[id_a][1] == (loc, t + 1):
                    y[p_rel, id_a] = 1
                    break

    def find_arc_departing_after(src_loc, t_from, dst_loc, latest_arrival, d):
        """
        Return the latest-arriving feasible arc with arr <= latest_arrival.
        Returns (None, None) if no arc arrives in time — NO fallback,
        so that build_solution can handle the case without breaking EC16.
        """
        arcs = d.arc_lookup.get((src_loc, dst_loc), [])
        if not arcs:
            return None, None

        dep_times = [a[0] for a in arcs]
        idx  = bisect_left(dep_times, t_from)
        best = None

        while idx < len(arcs):
            dep_t, arr_t, arc_id, arc = arcs[idx]
            if arr_t > latest_arrival:
                break
            best = (arc_id, arc)
            idx += 1

        if best is None:
            return None, None
        return best

    for p_rel, p_id in enumerate(d.from_RelPod_to_PodId):
        storage_loc = d.warehouse.pods[p_id].storage_location

        # Build the chronological sequence of (workstation, arrival_time)
        # events this pod must satisfy, deduplicating consecutive repeats.
        items_for_pod = {
            im: int(first_one_idx[im])
            for im in d.items_by_pod[p_id]
            if int(first_one_idx[im]) < d.OptManager.N_TIME
        }
        events = []
        for im, t in sorted(items_for_pod.items(), key=lambda kv: kv[1]):
            _, m   = d.relevant_pairs_for_x[im]
            ws_loc = d.ws_positions[d.order_to_ws[m]]
            if not events or events[-1] != (ws_loc, t):
                events.append((ws_loc, t))

        prev_loc, prev_t = storage_loc, 0

        for ws_loc, arrive_t in events:

            # GUARD: se siamo già oltre arrive_t non possiamo tornare indietro
            if prev_t > arrive_t:
                logging.warning(
                    "build_solution: pod %d at %s t=%d > arrive_t=%d — "
                    "event skipped to preserve EC16",
                    p_id, prev_loc, prev_t, arrive_t
                )
                continue

            if prev_loc == ws_loc:
                # Pod già alla workstation: prova a passare per storage
                arc1, id_a1 = None, None
                for id_a in d.OptManager.outgoing_arc_idx.get((prev_loc, prev_t), []):
                    arc = d.OptManager.all_arcs[id_a]
                    if arc[1][0] == storage_loc:
                        arc1  = arc
                        id_a1 = id_a
                        break

                via_storage = False
                if arc1 is not None:
                    id_a2, arc2 = find_arc_departing_after(
                        storage_loc, arc1[1][1], ws_loc, arrive_t, d
                    )
                    if id_a2 is not None:
                        via_storage = True
                        y[p_rel, id_a1] = 1
                        add_idle_arcs(y, p_rel, storage_loc, arc1[1][1], arc2[0][1])
                        y[p_rel, id_a2] = 1
                        add_idle_arcs(y, p_rel, ws_loc, arc2[1][1], arrive_t)

                if not via_storage:
                    add_idle_arcs(y, p_rel, prev_loc, prev_t, arrive_t)

            else:
                # Pod non alla workstation: prova via storage, poi diretto
                arc1, id_a1 = None, None
                for id_a in d.OptManager.outgoing_arc_idx.get((prev_loc, prev_t), []):
                    arc = d.OptManager.all_arcs[id_a]
                    if arc[1][0] == storage_loc:
                        arc1  = arc
                        id_a1 = id_a
                        break

                via_storage = False
                if arc1 is not None:
                    id_a2, arc2 = find_arc_departing_after(
                        storage_loc, arc1[1][1], ws_loc, arrive_t, d
                    )
                    if id_a2 is not None:
                        via_storage = True
                        y[p_rel, id_a1] = 1
                        add_idle_arcs(y, p_rel, storage_loc, arc1[1][1], arc2[0][1])
                        y[p_rel, id_a2] = 1
                        add_idle_arcs(y, p_rel, ws_loc, arc2[1][1], arrive_t)

                if not via_storage:
                    arc_id, arc = find_arc_departing_after(
                        prev_loc, prev_t, ws_loc, arrive_t, d
                    )
                    if arc_id is not None:
                        add_idle_arcs(y, p_rel, prev_loc, prev_t, arc[0][1])
                        y[p_rel, arc_id] = 1
                    else:
                        # Evento irraggiungibile: resta fermo
                        logging.warning(
                            "build_solution: pod %d cannot reach ws %s "
                            "by t=%d from %s t=%d — staying put",
                            p_id, ws_loc, arrive_t, prev_loc, prev_t
                        )
                        add_idle_arcs(y, p_rel, prev_loc, prev_t, arrive_t + 1)
                        if arrive_t + 1 < d.OptManager.N_TIME:
                            prev_loc, prev_t = prev_loc, arrive_t + 1
                        else:
                            prev_loc, prev_t = prev_loc, arrive_t
                        continue

            if arrive_t + 1 < d.OptManager.N_TIME:
                add_idle_arcs(y, p_rel, ws_loc, arrive_t, arrive_t + 1)
                prev_loc, prev_t = ws_loc, arrive_t + 1
            else:
                prev_loc, prev_t = ws_loc, arrive_t

        # Return to storage at end of horizon
        if prev_loc != storage_loc:
            arc, id_arc = None, None
            for id_a in d.OptManager.outgoing_arc_idx.get((prev_loc, prev_t), []):
                a = d.OptManager.all_arcs[id_a]
                if a[1][0] == storage_loc:
                    arc, id_arc = a, id_a
                    break
            if arc is not None:
                y[p_rel, id_arc] = 1
                add_idle_arcs(y, p_rel, storage_loc, arc[1][1], T - 1)
            else:
                add_idle_arcs(y, p_rel, prev_loc, prev_t, T - 1)
        else:
            add_idle_arcs(y, p_rel, storage_loc, prev_t, T - 1)

    return x, f, g, v, y


def _rebuild_pod_row(p_rel: int, p_id: int, x: np.ndarray, d) -> np.ndarray:
    """
    Recompute the y row for a single pod without touching the rest of y.
    Mirrors the per-pod logic inside build_solution.

    Used as a cheap incremental update: when a move only touches a handful
    of orders, only the pods supplying those orders need their route
    recomputed, instead of rebuilding the whole y matrix from scratch.
    """
    n_travel    = len(d.OptManager.travelling_arcs)
    T           = d.OptManager.N_TIME
    storage_loc = d.warehouse.pods[p_id].storage_location
    first_one_idx = (x == 0).sum(axis=1)

    y_row = np.zeros(len(d.OptManager.all_arcs), dtype=np.float64)

    def add_idle_arcs(loc, t_from, t_to):
        for t in range(t_from, t_to):
            for id_a in d.OptManager.outgoing_arc_idx.get((loc, t), []):
                if d.OptManager.all_arcs[id_a][1] == (loc, t + 1):
                    y_row[id_a] = 1
                    break

    def find_arc(src_loc, t_from, dst_loc, latest):
        """Best (latest-arriving, within `latest`) direct arc from src_loc
        departing no earlier than t_from. Linear scan variant of
        find_arc_departing_after used above, kept separate on purpose."""
        best_arc, best_id = None, None
        for t_dep in range(t_from, latest):
            for id_a in d.OptManager.outgoing_arc_idx.get((src_loc, t_dep), []):
                if id_a >= n_travel:
                    continue
                arc = d.OptManager.all_arcs[id_a]
                if arc[1][0] == dst_loc and arc[1][1] <= latest:
                    if best_arc is None or arc[1][1] > best_arc[1][1]:
                        best_arc, best_id = arc, id_a
        return best_id, best_arc

    items_for_pod = {
        im: int(first_one_idx[im])
        for im in d.items_by_pod[p_id]
        if int(first_one_idx[im]) < T
    }
    events = []
    for im, t in sorted(items_for_pod.items(), key=lambda kv: kv[1]):
        _, m   = d.relevant_pairs_for_x[im]
        ws_loc = d.ws_positions[d.order_to_ws[m]]
        if not events or events[-1] != (ws_loc, t):
            events.append((ws_loc, t))

    prev_loc, prev_t = storage_loc, 0

    for ws_loc, arrive_t in events:

        # GUARD: evento nel passato, salta
        if prev_t > arrive_t:
            logging.warning(
                "_rebuild_pod_row: pod %d at %s t=%d > arrive_t=%d — skipping",
                p_id, prev_loc, prev_t, arrive_t
            )
            continue

        if prev_loc == ws_loc:
            via_storage = False
            for id_a in d.OptManager.outgoing_arc_idx.get((prev_loc, prev_t), []):
                arc = d.OptManager.all_arcs[id_a]
                if arc[1][0] == storage_loc:
                    id_a2, arc2 = find_arc(storage_loc, arc[1][1], ws_loc, arrive_t)
                    if id_a2 is not None:
                        via_storage = True
                        y_row[id_a] = 1
                        add_idle_arcs(storage_loc, arc[1][1], arc2[0][1])
                        y_row[id_a2] = 1
                        add_idle_arcs(ws_loc, arc2[1][1], arrive_t)
                    break
            if not via_storage:
                add_idle_arcs(prev_loc, prev_t, arrive_t)

        else:
            via_storage = False
            for id_a in d.OptManager.outgoing_arc_idx.get((prev_loc, prev_t), []):
                arc = d.OptManager.all_arcs[id_a]
                if arc[1][0] == storage_loc:
                    id_a2, arc2 = find_arc(storage_loc, arc[1][1], ws_loc, arrive_t)
                    if id_a2 is not None:
                        via_storage = True
                        y_row[id_a] = 1
                        add_idle_arcs(storage_loc, arc[1][1], arc2[0][1])
                        y_row[id_a2] = 1
                        add_idle_arcs(ws_loc, arc2[1][1], arrive_t)
                    break
            if not via_storage:
                arc_id, arc = find_arc(prev_loc, prev_t, ws_loc, arrive_t)
                if arc_id is not None:
                    add_idle_arcs(prev_loc, prev_t, arc[0][1])
                    y_row[arc_id] = 1
                else:
                    logging.warning(
                        "_rebuild_pod_row: pod %d cannot reach ws %s "
                        "by t=%d from %s t=%d — staying put",
                        p_id, ws_loc, arrive_t, prev_loc, prev_t
                    )
                    add_idle_arcs(prev_loc, prev_t, arrive_t + 1)
                    if arrive_t + 1 < d.OptManager.N_TIME:
                        prev_loc, prev_t = prev_loc, arrive_t + 1
                    else:
                        prev_loc, prev_t = prev_loc, arrive_t
                    continue

        if arrive_t + 1 < d.OptManager.N_TIME:
            add_idle_arcs(ws_loc, arrive_t, arrive_t + 1)
            prev_loc, prev_t = ws_loc, arrive_t + 1
        else:
            prev_loc, prev_t = ws_loc, arrive_t

    # Return to storage
    if prev_loc != storage_loc:
        arc, id_arc = None, None
        for id_a in d.OptManager.outgoing_arc_idx.get((prev_loc, prev_t), []):
            a = d.OptManager.all_arcs[id_a]
            if a[1][0] == storage_loc:
                arc, id_arc = a, id_a
                break
        if arc is not None:
            y_row[id_arc] = 1
            add_idle_arcs(storage_loc, arc[1][1], T - 1)
        else:
            add_idle_arcs(prev_loc, prev_t, T - 1)
    else:
        add_idle_arcs(storage_loc, prev_t, T - 1)

    return y_row


### OBJECTIVE AND CONSTRAINTS CHECKER
# compute_objective rewards fully-picked orders and penalises backlog
# (orders taking longer, relative to their arrival time, to complete).
# check_constraints is the authoritative, y-aware feasibility check: it
# verifies every EC and returns both a  boolean and a dict of violations 
# (empty dict <=> feasible), useful for # debugging which constraint(s) 
# failed.

def compute_objective(x: np.ndarray, f: np.ndarray, g: np.ndarray, d) -> float:
    """
    Stage-2 objective: reward completed picks minus a backlog penalty.

    picking_reward  = number of item-order pairs picked by the end of the
                       horizon (x[:, T-1].sum()).
    backlog_penalty = sum, over all orders and all time steps where the
                       order is not yet finished (g = 0), of the order's
                       "age" (elapsed time since arrival) — i.e. orders
                       that have been waiting longer are penalised more
                       for every time step they remain incomplete.

    The final objective is picking_reward minus a (small, normalised)
    weight on the backlog penalty.
    """
    T = x.shape[1]
    picking_reward  = x[:, T - 1].sum()
    backlog_penalty = float(sum(
        (d.current_time + t * d.OptManager.TIME_UNIT - d.arrival_times[m])
        / d.OptManager.TIME_UNIT
        * (1.0 - g[m, t])
        for m in range(len(d.orders))
        for t in range(T)
    ))
    return picking_reward - 0.1 * backlog_penalty / d.OptManager.N_TIME


def check_constraints(sol: tuple, d) -> tuple[bool, dict]:
    """
    Full constraint checker including y-based constraints.

    Returns (feasible, viols) where `feasible` is True iff `viols` is
    empty. `viols` maps each violated constraint's label to details about
    where/how it was violated, to make debugging infeasible candidates
    straightforward. 
    """
    x, f, g, v, y = sol
    T        = x.shape[1]
    n_travel = len(d.OptManager.travelling_arcs)
    viols: dict = {}

    num_constraints = 0

    # EC10: per-workstation SKU-throughput capacity, evaluated via v.
    for w, order_ids in enumerate(d.orders_by_workstation):
        num_constraints += 1
        cap = v[list(order_ids), :].sum(axis=0)
        bad = np.where(cap > d.OptManager.CAP_WS + 1e-6)[0]
        if bad.size:
            viols.setdefault('EC10', []).append(
                {'w': w, 'times': bad.tolist(), 'values': cap[bad].tolist()}
            )

    # EC11: combined item-picking + pod-arrival work rate per workstation
    # per time slot cannot exceed 2 * TIME_UNIT.
    ec11 = []
    for w, order_ids in enumerate(d.orders_by_workstation):
        ws_p  = d.ws_positions[w]
        ims_w = [im for im, (_, m) in enumerate(d.relevant_pairs_for_x) if m in order_ids]
        for t in range(1, T):
            num_constraints += 1
            item_work = d.OptManager.DELTA_ITEM * (x[ims_w, t] - x[ims_w, t - 1]).sum()
            travel_arrivals = [
                a for a in d.OptManager.incoming_arc_idx.get((ws_p, t), [])
                if a < n_travel
            ]
            pod_arrivals = d.OptManager.DELTA_POD * y[:, travel_arrivals].sum()
            total = float(item_work + pod_arrivals)
            if total > 2 * d.OptManager.TIME_UNIT + 1e-6:
                ec11.append({'w': w, 't': t, 'value': total})
    if ec11:
        viols['EC11'] = ec11

    # EC12: each pod must depart its storage location exactly once at t=0.
    ec12 = []
    for rel_p, p_id in enumerate(d.from_RelPod_to_PodId):
        num_constraints += 1
        stor     = d.warehouse.pods[p_id].storage_location
        out_arcs = d.OptManager.outgoing_arc_idx.get((stor, 0), [])
        flow_out = float(y[rel_p, out_arcs].sum())
        if not np.isclose(flow_out, 1.0):
            ec12.append({'pod': p_id, 'flow_out': flow_out})
    if ec12:
        viols['EC12'] = ec12

    # EC13: flow conservation for each pod at every intermediate node of
    # the time-expanded network (inflow == outflow).
    ec13 = []
    for rel_p in range(len(d.from_RelPod_to_PodId)):
        for node in d.OptManager.nodes:
            num_constraints += 1
            if node[1] in (0, d.OptManager.N_TIME - 1):
                continue
            in_f  = float(y[rel_p, d.OptManager.incoming_arc_idx.get(node, [])].sum())
            out_f = float(y[rel_p, d.OptManager.outgoing_arc_idx.get(node, [])].sum())
            if not np.isclose(in_f - out_f, 0.0):
                ec13.append({'pod': rel_p, 'node': node, 'imbalance': in_f - out_f})
    if ec13:
        viols['EC13'] = ec13

    # EC14: the pod supplying an item must actually be at the workstation
    # at the time that item is first picked.
    first_pick_time = (x == 0).sum(axis=1)
    ec14 = []
    for im, first_t in enumerate(first_pick_time):
        num_constraints += 1
        if first_t < d.OptManager.N_TIME:
            _, m   = d.relevant_pairs_for_x[im]
            ws_p   = d.ws_positions[d.order_to_ws[m]]
            rel_p  = d.from_PodId_to_RelPod[d.pod_of_item[im]]
            t_arcs = d.OptManager.incoming_arc_idx.get((ws_p, first_t), [])
            if float(y[rel_p, t_arcs].sum()) < 1e-6:
                ec14.append({'im': im, 't': first_t})
    if ec14:
        viols['EC18'] = ec14

    # EC15: x must be non-decreasing over time.
    num_constraints += x.shape[0]* (x.shape[1] - 1)
    dx  = np.diff(x, axis=1)
    bad = np.argwhere(dx < -1e-6)
    if bad.size:
        viols['EC15'] = bad.tolist()

    # EC16 ("pick_only_if_active"): x can only increase while the order is
    # in progress (v = 1); items can't be picked for an inactive order.
    poa = []
    num_constraints += len(d.relevant_pairs_for_x) * (x.shape[1] - 1)
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        bad_ts = np.where(dx[im] > v[m, 1:] + 1e-6)[0] + 1
        if bad_ts.size:
            poa.append({'im': im, 'times': bad_ts.tolist()})
    if poa:
        viols['pick_only_if_active'] = poa

    # EC17: v must equal f - g.
    num_constraints +=  v.shape[0]*v.shape[1]
    bad = np.argwhere(np.abs(v - (f - g)) > 1e-6)
    if bad.size:
        viols['EC17'] = bad.tolist()

    # EC18: f[m,t] >= x[im,t] for every item im of order m.
    ec18 = []
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        num_constraints +=  f.shape[1]
        bad = np.where(f[m] < x[im] - 1e-6)[0]
        if bad.size:
            ec18.append({'im': im, 'm': m, 'times': bad.tolist()})
    if ec18:
        viols['EC21'] = ec18

    # EC19: g[m,t] <= x[im,t-1] for every item im of order m.
    ec19 = []
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        num_constraints +=  g.shape[1] - 1
        bad = np.where(g[m, 1:] > x[im, :-1] + 1e-6)[0] + 1
        if bad.size:
            ec19.append({'im': im, 'm': m, 'times': bad.tolist()})
    if ec19:
        viols['EC19'] = ec19

    # EC20a / EC20b: f and g must each be non-decreasing over time.
    num_constraints += f.shape[0] * (f.shape[1] - 1)
    num_constraints += g.shape[0] * (g.shape[1] - 1)
    if (np.diff(f, axis=1) < -1e-6).any():
        viols['f_monotonicity'] = True
    if (np.diff(g, axis=1) < -1e-6).any():
        viols['g_monotonicity'] = True

    # EC21 ("continuity_v"): v cannot jump back up without passing through g.
    num_constraints += v.shape[0] * (v.shape[1] - 1)
    bad = np.argwhere(v[:, 1:] - (v[:, :-1] - g[:, 1:]) < -1e-6)
    if bad.size:
        viols['continuity_v'] = bad.tolist()

    # EC22 ("g_lower_bound"): g cannot be lower than what the picking
    # progress of the order already guarantees.
    g_lb = []
    num_constraints += g.shape[0] * (g.shape[1] - 1)
    for m in range(len(d.orders)):
        ims     = d.items_of_order[m]
        n_items = int(d.n_items_per_order[m])
        lb      = x[ims, :-1].sum(axis=0) - (n_items - 1)
        bad     = np.where(g[m, 1:] < lb - 1e-6)[0] + 1
        if bad.size:
            g_lb.append({'m': m, 'times': bad.tolist()})
    if g_lb:
        viols['g_lower_bound'] = g_lb

    # EC23 ("initial_cond" / "f_active_only_if_picked"): opened orders must
    # start in progress (v[m,0] = 1); not-yet-opened orders can only be
    # "active" (f = 1) once at least one item has actually been picked.
    for m, order in enumerate(d.orders):
        if order.order_id in d.opened_order_ids:
            num_constraints += 1
            if not np.isclose(float(v[m, 0]), 1.0):
                viols.setdefault('initial_cond', []).append(
                    {'m': m, 'v0': float(v[m, 0])}
                )
        else:
            ims = d.items_of_order[m]
            num_constraints += len(ims)
            bad = np.where(f[m] > x[ims, :].sum(axis=0) + 1e-6)[0]
            if bad.size:
                viols.setdefault('f_active_only_if_picked', []).append(
                    {'m': m, 'times': bad.tolist()}
                )

    # EC24 ("max_active_pods"): number of pods away from their storage
    # location at any given time cannot exceed the number of robots.
    # Un pod occupa un robot durante [src_t, dst_t) se parte da una
    # location che non e' il suo storage (cioe' e' fuori storage).
    n_pods = y.shape[0]
    active = np.zeros(T, dtype=int)
    num_constraints += T

    for rel_p in range(n_pods):
        pod_id      = d.from_RelPod_to_PodId[rel_p]
        pod_storage = d.warehouse.pods[pod_id].storage_location

        for a_idx in np.where(y[rel_p] > 0.5)[0]:
            src, dst       = d.OptManager.all_arcs[a_idx]
            src_loc, src_t = src
            dst_loc, dst_t = dst

            if src_loc != pod_storage:
                for t in range(src_t, dst_t):
                    if t < T:
                        active[t] += 1

    bad_t = np.where(active > len(d.warehouse.robots))[0]
    if bad_t.size:
        viols['max_active_pods'] = {
            'times':  bad_t.tolist(),
            'values': active[bad_t].tolist(),
        }

    return len(viols) == 0, viols, num_constraints


### NEIGHBOUR GENERATORS
### Each helper builds a candidate x by shifting the pick time of one or
### more items backwards in the horizon (a negative `variation`/`direction`
### moves the pick earlier). They return None when the move is infeasible
### purely on index-range grounds (out of [0, T) or [1, T)), letting the
### caller skip it cheaply before running the more expensive feasibility
### checks.

def _make_move_1(x, ims, variation, first_one_idx, T):
    """'item' / 'multi_item' move: shift the pick time of one or more items
    (ims) by `variation` steps, rewriting each affected row of x as a step
    function that turns on at the new time. Returns None if any new time
    falls outside [0, T)."""
    x_cand = x.copy()
    for im in ims:
        t_new = int(first_one_idx[im]) + variation
        if t_new < 0 or t_new >= T:
            return None
        x_cand[im, :] = 0
        x_cand[im, t_new:] = 1
    return x_cand


def _make_move_2(x, ims, variation, first_one_idx, T):
    """'order' move: same as _make_move_1 but disallows t_new = 0 (an order
    move cannot make an item picked at the very start of the horizon)."""
    x_cand = x.copy()
    for im in ims:
        t_new = int(first_one_idx[im]) + variation
        if t_new < 1 or t_new >= T:
            return None
        x_cand[im, :] = 0
        x_cand[im, t_new:] = 1
    return x_cand


def _make_move_3(x, ims1, ims2, first_one_idx, T):
    """'swap' move: exchange the (earliest) pick times of two orders'
    item sets, shifting all items of order 1 by the same delta needed to
    reach order 2's earliest pick time, and vice versa. Returns None if the
    orders already start at the same time (no-op) or if any resulting time
    falls outside [1, T)."""
    delta = (
        min(int(first_one_idx[im]) for im in ims2)
        - min(int(first_one_idx[im]) for im in ims1)
    )
    if delta == 0:
        return None
    new_t1 = [int(first_one_idx[im]) + delta for im in ims1]
    new_t2 = [int(first_one_idx[im]) - delta for im in ims2]
    if any(t < 1 or t >= T for t in new_t1 + new_t2):
        return None
    x_cand = x.copy()
    for im, t in zip(ims1, new_t1):
        x_cand[im, :] = 0
        x_cand[im, t:] = 1
    for im, t in zip(ims2, new_t2):
        x_cand[im, :] = 0
        x_cand[im, t:] = 1
    return x_cand


### LOCAL SEARCH

def local_search_stage2(d: Stage2Data) -> tuple:
    """
    Local search on x (picking matrix).

    Key design choices:
      1. build_solution (builds y) called only when a candidate x passes
         the x-only check AND improves the best objective.
      2. x_current tracks the current search point; best_x the global best.
      3. item_ids defined once before the main loop to avoid NameError.
    """
    rng = np.random.default_rng(seed=42)

    im_by_order: dict[int, list[int]] = {}
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        im_by_order.setdefault(m, []).append(im)

    ### INITIAL SOLUTION
    print("\n[ls_stage2] Building initial solution ...")
    logging.info("[ls_stage2] Building initial solution ...")

    x_current = build_initial_x(rng, d)
    _, f0, g0, v0, y0 = build_solution(x_current, d)
    feasible, viols, num_constraints = check_constraints((x_current, f0, g0, v0, y0), d)

    max_attempts = 10
    attempt = 1

    # Retry with a fresh initial x until a feasible solution is found or
    # the attempt budget is exhausted (mirrors the pattern used in Stage 1).
    while not feasible:
        print(f"[ls_stage2] violated = {list(viols.keys())} (attempt {attempt}/{max_attempts})")
        logging.info("[ls_stage2] violated = %s (attempt %d/%d)", list(viols.keys()), attempt, max_attempts)
        for k, vv in viols.items():
            print(f"  {k}: {vv[:3] if isinstance(vv, list) else vv}")

        if attempt >= max_attempts:
            msg = f"[ls_stage2] Failed to find a feasible initial solution after {max_attempts} attempts."
            print(msg)
            logging.error(msg)
            raise RuntimeError(msg)

        attempt += 1
        x_current = build_initial_x(rng, d)
        _, f0, g0, v0, y0 = build_solution(x_current, d)
        feasible, viols, num_constraints = check_constraints((x_current, f0, g0, v0, y0), d)

    best_x   = x_current.copy()
    best_sol = (best_x, f0, g0, v0, y0)
    best_obj = compute_objective(x_current, f0, g0, d)
    print(f"[ls_stage2] Feasible initial solution: obj = {best_obj:.4f}")
    logging.info("[ls_stage2] Feasible initial solution: obj = %.4f", best_obj)

    logging.warning(
        f"\Variables size:\n"
        f"  x shape = {best_sol[0].shape}\n"
        f"  f0 = {best_sol[1].shape}\n"
        f"  g0 = {best_sol[2].shape}\n"
        f"  v0 = {best_sol[3].shape}\n"
        f"  y shape = {best_sol[4].shape}\n"
        f"  num constraints satisfied = {num_constraints}\n"
    )
    
    T = x_current.shape[1]
    item_ids = list(range(x_current.shape[0]))

    ### MAIN LOOP
    am_I_stuck                   = False
    cont                         = 1
    iter_without_improvement     = 0
    max_iter_without_improvement = 5
    MAX_ITER  = 150
    MAX_NEIGH = 300

    print("[ls_stage2] Exploring neighbours ...")

    while not am_I_stuck and cont <= MAX_ITER:
        # first_one_idx: earliest pick-time index per item-order pair in
        # the current incumbent (used as the base for all move generators).
        first_one_idx = np.argmax(best_x > 0.5, axis=1)
        first_one_idx[best_x[:, -1] == 0] = T
        improved = False

        # Best and second-best candidate found in this iteration (the
        # second-best acts as a fallback if committing the best one turns
        # out to be y-infeasible after the full check).
        best_obj_in_iter        = -np.inf
        best_x_in_iter          = None
        best_move               = None
        best_f_in_iter          = None
        best_g_in_iter          = None
        best_v_in_iter          = None
        second_best_obj_in_iter = -np.inf
        second_best_x_in_iter   = None
        second_best_move        = None
        s_best_f_in_iter        = None
        s_best_g_in_iter        = None
        s_best_v_in_iter        = None

        # ---- Build move list ----------------------------------------- #
        # moves[0]: item-level moves ('item' / 'multi_item')
        # moves[1]: order-level moves ('order')
        # moves[2]: pairwise 'swap' moves between orders at the same
        #           workstation
        moves = [[], [], []]

        # When stuck for a while, use small, conservative shifts (-1, -2);
        # otherwise explore more aggressive shifts (-2 to -8) to escape
        # local optima faster.
        if iter_without_improvement > 1:
            for im in range(x_current.shape[0]):
                moves[0].append(('item', im, -1))
                moves[0].append(('item', im, -2))
            for direction in [-1, -2]:
                sampled = rng.choice(item_ids, size=min(len(item_ids), 20), replace=False)
                for i in range(0, len(sampled) - 1, 2):
                    moves[0].append(('multi_item', (sampled[i], sampled[i + 1]), direction))
                    if i + 2 < len(sampled) - 1:
                        moves[0].append((
                            'multi_item',
                            (sampled[i], sampled[i + 1], sampled[i + 2]),
                            direction
                        ))
        else:
            for im in range(x_current.shape[0]):
                moves[0].append(('item', im, -2))
                moves[0].append(('item', im, -4))
                moves[0].append(('item', im, -6))
                moves[0].append(('item', im, -8))
            for direction in [-2, -4, -5]:
                sampled = rng.choice(item_ids, size=min(len(item_ids), 20), replace=False)
                for i in range(0, len(sampled) - 1, 2):
                    moves[0].append(('multi_item', (sampled[i], sampled[i + 1]), direction))
                    if i + 2 < len(sampled) - 1:
                        moves[0].append((
                            'multi_item',
                            (sampled[i], sampled[i + 1], sampled[i + 2]),
                            direction
                        ))

        for m in range(len(d.orders)):
            moves[1].append(('order', m, -1))
            moves[1].append(('order', m, -2))
            moves[1].append(('order', m, -4))

        for order_ids in d.orders_by_workstation:
            order_list = list(order_ids)
            for i1, m1 in enumerate(order_list):
                for m2 in order_list[i1 + 1:]:
                    moves[2].append(('swap', m1, m2))

        # Cap the total neighbourhood size, subsampling each move category
        # proportionally (40% item, 40% order, 20% swap) rather than
        # truncating a single category entirely.
        total = sum(len(mv) for mv in moves)
        if total > MAX_NEIGH:
            for i, p in enumerate([0.4, 0.4, 0.2]):
                size = min(len(moves[i]), int(np.ceil(MAX_NEIGH * p)))
                if size and len(moves[i]) > size:
                    idxs     = rng.choice(len(moves[i]), size=size, replace=False)
                    moves[i] = [moves[i][j] for j in idxs]

        all_moves = moves[0] + moves[1] + moves[2]

        # ---- Evaluate moves ------------------------------------------ #
        for move in all_moves:
            first_one_idx_cand = first_one_idx.copy()

            if move[0] == 'item':
                _, im, direction = move
                first_one_idx_cand[im] += direction
                x_cand = _make_move_1(x_current, [im], int(direction), first_one_idx, T)
            elif move[0] == 'multi_item':
                _, ims, direction = move
                for im in ims:
                    first_one_idx_cand[im] += direction
                x_cand = _make_move_1(x_current, ims, int(direction), first_one_idx, T)
            elif move[0] == 'order':
                _, m, direction = move
                for im in im_by_order[m]:
                    first_one_idx_cand[im] += direction
                x_cand = _make_move_2(
                    x_current, im_by_order.get(int(m), []), int(direction), first_one_idx, T
                )
            else:  # swap
                _, m1, m2 = move
                x_cand = _make_move_3(
                    x_current,
                    im_by_order.get(int(m1), []),
                    im_by_order.get(int(m2), []),
                    first_one_idx, T,
                )
                delta = (
                    min(first_one_idx[im] for im in im_by_order[m2])
                    - min(first_one_idx[im] for im in im_by_order[m1])
                )
                for im in im_by_order[m1]:
                    first_one_idx_cand[im] += delta
                for im in im_by_order[m2]:
                    first_one_idx_cand[im] -= delta

            if x_cand is None:
                continue

            # Cheap incremental f/g/v update + x-only feasibility check
            # (EC10-EC13, EC17, EC19-EC23) before ever touching y.
            _, f_curr, g_curr, v_curr, _ = best_sol
            f_cand, g_cand, v_cand = _fast_update_fgv_from_move(
                x_cand, f_curr, g_curr, v_curr,
                move, im_by_order, first_one_idx_cand, d,
            )

            if _check_x_fast(x_cand, f_cand, g_cand, v_cand, d):
                obj = compute_objective(x_cand, f_cand, g_cand, d)
                if obj is not None and obj > best_obj_in_iter:
                    # New best in this iteration: demote the previous best
                    # to second-best (kept as a fallback candidate).
                    second_best_obj_in_iter = best_obj_in_iter
                    second_best_x_in_iter   = best_x_in_iter
                    second_best_move        = best_move
                    s_best_f_in_iter        = best_f_in_iter
                    s_best_g_in_iter        = best_g_in_iter
                    s_best_v_in_iter        = best_v_in_iter

                    best_obj_in_iter = obj
                    best_x_in_iter   = x_cand
                    best_move        = move
                    best_f_in_iter   = f_cand
                    best_g_in_iter   = g_cand
                    best_v_in_iter   = v_cand

                elif obj is not None and obj > second_best_obj_in_iter:
                    second_best_obj_in_iter = obj
                    second_best_x_in_iter   = x_cand
                    second_best_move        = move
                    s_best_f_in_iter        = f_cand
                    s_best_g_in_iter        = g_cand
                    s_best_v_in_iter        = v_cand

        # ---- Attempt to commit best (then second-best) --------------- #
        # A candidate that passes the x-only check may still violate a
        # y-based constraint (EC14-EC16, EC18, EC26...). Try the best
        # candidate first, and fall back to the second-best if the full
        # check_constraints rejects it.
        sol_num = 1
        while best_x_in_iter is not None and sol_num <= 2:
            x_current = best_x_in_iter

            if best_obj_in_iter > best_obj - 1e-10:
                if best_move[0] in ('item', 'order', 'multi_item', 'swap'):
                    # Only the pods supplying the items touched by this
                    # move need their y row rebuilt (cheap incremental
                    # update instead of a full build_solution call).
                    if best_move[0] == 'item':
                        _, best_im, _ = best_move
                        affected_pods = {d.pod_of_item[int(best_im)]}
                    elif best_move[0] == 'multi_item':
                        _, best_ims, _ = best_move
                        affected_pods = {d.pod_of_item[int(im)] for im in best_ims}
                    elif best_move[0] == 'order':
                        _, best_m, _ = best_move
                        affected_pods = {d.pod_of_item[im] for im in im_by_order[int(best_m)]}
                    else:  # swap
                        _, best_m1, best_m2 = best_move
                        affected_pods = (
                            {d.pod_of_item[im] for im in im_by_order[int(best_m1)]}
                            | {d.pod_of_item[im] for im in im_by_order[int(best_m2)]}
                        )

                    y_new = best_sol[4].copy()
                    for p_id in affected_pods:
                        p_rel = d.from_PodId_to_RelPod[p_id]
                        y_new[p_rel] = _rebuild_pod_row(p_rel, p_id, best_x_in_iter, d)

                    sol_curr = (
                        best_x_in_iter,
                        best_f_in_iter, best_g_in_iter, best_v_in_iter,
                        y_new
                    )
                else:
                    sol_curr = build_solution(best_x_in_iter, d)

                feasible, _, _ = check_constraints(sol_curr, d)
                if feasible:
                    improved = best_obj_in_iter >= best_obj
                    best_obj = best_obj_in_iter
                    best_x   = best_x_in_iter
                    best_sol = sol_curr
                    print(
                        f"[ls_stage2] Iter {cont}: improved "
                        f"move={best_move} obj={best_obj:.4f}"
                    )
                    logging.info(
                        "[ls_stage2] Iter %d: improved move=%s obj=%.4f",
                        cont, best_move, best_obj
                    )
                    sol_num = 3   # stop the fallback loop, committed successfully
                else:
                    # Best candidate is y-infeasible: retry with the
                    # second-best candidate from this iteration, if any.
                    if second_best_x_in_iter is None:
                        break
                    best_x_in_iter   = second_best_x_in_iter
                    best_f_in_iter   = s_best_f_in_iter
                    best_g_in_iter   = s_best_g_in_iter
                    best_v_in_iter   = s_best_v_in_iter
                    best_move        = second_best_move
                    best_obj_in_iter = second_best_obj_in_iter
                    sol_num += 1
            else:
                break

        ### Convergence check
        if improved:
            iter_without_improvement = 0
        else:
            iter_without_improvement += 1
            if iter_without_improvement >= max_iter_without_improvement:
                am_I_stuck = True
                print(
                    f"[ls_stage2] Converged after "
                    f"{max_iter_without_improvement} iters without improvement "
                    f"at obj={best_obj:.4f}"
                )
                logging.info(
                    "[ls_stage2] Converged after %d iters without improvement",
                    max_iter_without_improvement
                )
            else:
                print(
                    f"[ls_stage2] Iter {cont}: no improvement "
                    f"({iter_without_improvement}/{max_iter_without_improvement})"
                )
                logging.info("[ls_stage2] Iter %d: no improvement", cont)

        cont += 1

    print("[ls_stage2] Local search ended.")
    return best_sol