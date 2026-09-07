from __future__ import annotations
import heapq
import itertools
import numpy as np
from bisect import bisect_left
import logging

from .stage2_data import Stage2Data
from .build_initial_x_stage2 import build_initial_x

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

PERFORMANCE NOTES (optimisation pass)
--------------------------------------
This module is optimised relative to the original version in a few ways
that don't change behaviour/semantics, only speed:

  1. `_get_m_of_im(d)` caches, on `d`, the array mapping each item-order
     pair index to its order index (equivalent to iterating
     `d.relevant_pairs_for_x`). This lets `_check_x_fast` (called for
     every single candidate move, i.e. the hottest loop in the search)
     replace two Python-level loops with two vectorised NumPy comparisons.
  2. `_find_best_arc` is now a single shared helper (bisect-based lookup
     on `d.arc_lookup`) used by both `build_solution` and
     `_rebuild_pod_row`, instead of `_rebuild_pod_row` duplicating the
     same logic with a slower O(T) linear scan over `outgoing_arc_idx`.
  3. `_rebuild_pod_row` no longer recomputes `first_one_idx = (x == 0)
     .sum(axis=1)` for every pod it is asked to rebuild; the caller
     computes it once per candidate `x` and passes it in.
  4. The top-K_BEST candidate list in `local_search_stage2` is now
     maintained with a small heap (`heapq`) instead of a full
     sort-after-every-append, avoiding O(n log n) work on every single
     accepted neighbour.
  5. The 'order' and 'swap' move lists, which depend only on static
     problem data (`d.orders`, `d.orders_by_workstation`), are built once
     before the main loop instead of being rebuilt from scratch on every
     iteration.
"""


### FAST HELPERS (no y needed)
# These mirror, on x only, the more expensive checks that build_solution /
# check_constraints perform once y is also available. They are used
# during the local search loop (cheap) and only the winning candidate is
# ever passed through the full, y-aware check_constraints.

def _get_m_of_im(d) -> np.ndarray:
    """
    Cached array mapping each item-order pair index `im` to its order index
    `m` (i.e. the vectorised equivalent of iterating
    `d.relevant_pairs_for_x`). Cached on `d` since it never changes for a
    given Stage2Data instance, and is looked up on every single candidate
    move evaluated during the local search.
    """
    cached = getattr(d, "_m_of_im_cache", None)
    if cached is None:
        cached = np.array([m for (_, m) in d.relevant_pairs_for_x])
        d._m_of_im_cache = cached
    return cached


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
    all_ms = _get_m_of_im(d)

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

    # f[m,t] >= x[im,t] for every item im of order m, and
    # g[m,t] <= x[im,t-1] for every item im of order m.
    # Vectorised via the (cached) im -> m mapping instead of a Python loop
    # over relevant_pairs_for_x — this function is called once per
    # candidate move, i.e. up to MAX_NEIGH times per iteration.
    m_of_im = _get_m_of_im(d)
    if (f[m_of_im] < x - 1e-6).any():
        return False
    if (g[m_of_im, 1:] > x[:, :-1] + 1e-6).any():
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


### SHARED ARC-LOOKUP HELPER
# Used by both build_solution (per-pod route construction from scratch)
# and _rebuild_pod_row (incremental per-pod route rebuild). Factored out
# so the two call sites share one bisect-based implementation instead of
# _rebuild_pod_row keeping its own, slower O(T) linear-scan variant.

def _find_best_arc(d, src_loc, t_from, dst_loc, latest_arrival):
    """
    Return the latest-arriving feasible direct arc from src_loc to dst_loc,
    departing no earlier than t_from and arriving no later than
    latest_arrival. Returns (arc_id, arc), or (None, None) if no such arc
    exists — NO fallback, so callers can handle the case explicitly
    without breaking flow-conservation (EC16).
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


def _find_exact_departure_arc(d, src_loc, t_dep, dst_loc):
    """
    Return the (arc_id, arc) departing src_loc at EXACTLY t_dep and arriving
    at dst_loc, or (None, None) if no such arc exists. Bisect lookup on
    d.arc_lookup (O(log n)) — replaces the pattern, repeated 4x in
    build_solution/_rebuild_pod_row, of scanning
    outgoing_arc_idx[(src_loc, t_dep)] (a list whose size used to scale with
    the total number of pods in the warehouse) just to find the single arc
    heading to a specific destination location.
    """
    arcs = d.arc_lookup.get((src_loc, dst_loc), [])
    if not arcs:
        return None, None
    dep_times = [a[0] for a in arcs]
    idx = bisect_left(dep_times, t_dep)
    if idx < len(arcs) and arcs[idx][0] == t_dep:
        _, _, arc_id, arc = arcs[idx]
        return arc_id, arc
    return None, None


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
        self-loop / waiting arcs for pod p_rel. O(1) lookup per t via
        d.idle_arc_id instead of scanning outgoing_arc_idx for the match."""
        for t in range(t_from, t_to):
            id_a = d.idle_arc_id.get((loc, t))
            if id_a is not None:
                y[p_rel, id_a] = 1

    def find_arc_departing_after(src_loc, t_from, dst_loc, latest_arrival, d):
        """
        Return the latest-arriving feasible arc with arr <= latest_arrival.
        Returns (None, None) if no arc arrives in time — NO fallback,
        so that build_solution can handle the case without breaking EC16.
        """
        return _find_best_arc(d, src_loc, t_from, dst_loc, latest_arrival)

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
                id_a1, arc1 = _find_exact_departure_arc(d, prev_loc, prev_t, storage_loc)

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
                id_a1, arc1 = _find_exact_departure_arc(d, prev_loc, prev_t, storage_loc)

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
            id_arc, arc = _find_exact_departure_arc(d, prev_loc, prev_t, storage_loc)
            if arc is not None:
                y[p_rel, id_arc] = 1
                add_idle_arcs(y, p_rel, storage_loc, arc[1][1], T - 1)
            else:
                add_idle_arcs(y, p_rel, prev_loc, prev_t, T - 1)
        else:
            add_idle_arcs(y, p_rel, storage_loc, prev_t, T - 1)

    return x, f, g, v, y


def _rebuild_pod_row(p_rel: int, p_id: int, x: np.ndarray, d,
                      first_one_idx: np.ndarray | None = None) -> np.ndarray:
    """
    Recompute the y row for a single pod without touching the rest of y.
    Mirrors the per-pod logic inside build_solution.

    Used as a cheap incremental update: when a move only touches a handful
    of orders, only the pods supplying those orders need their route
    recomputed, instead of rebuilding the whole y matrix from scratch.

    `first_one_idx` (the earliest-pick-time index per item-order pair for
    the current `x`) can be passed in precomputed by the caller — this
    avoids recomputing the same O(N*T) array once per pod when several
    pods are rebuilt for the same candidate `x` (e.g. a 'swap' or 'order'
    move touching multiple pods).
    """
    T           = d.OptManager.N_TIME
    storage_loc = d.warehouse.pods[p_id].storage_location
    if first_one_idx is None:
        first_one_idx = (x == 0).sum(axis=1)

    y_row = np.zeros(len(d.OptManager.all_arcs), dtype=np.float64)

    def add_idle_arcs(loc, t_from, t_to):
        """O(1) lookup per t via d.idle_arc_id instead of scanning
        outgoing_arc_idx for the self-loop arc — this function is called
        very often (once per accepted candidate, per affected pod)."""
        for t in range(t_from, t_to):
            id_a = d.idle_arc_id.get((loc, t))
            if id_a is not None:
                y_row[id_a] = 1

    def find_arc(src_loc, t_from, dst_loc, latest):
        """Best (latest-arriving, within `latest`) direct arc from src_loc
        departing no earlier than t_from. Bisect-based lookup shared with
        build_solution's find_arc_departing_after (see _find_best_arc)."""
        return _find_best_arc(d, src_loc, t_from, dst_loc, latest)

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
            id_a, arc = _find_exact_departure_arc(d, prev_loc, prev_t, storage_loc)
            if arc is not None:
                id_a2, arc2 = find_arc(storage_loc, arc[1][1], ws_loc, arrive_t)
                if id_a2 is not None:
                    via_storage = True
                    y_row[id_a] = 1
                    add_idle_arcs(storage_loc, arc[1][1], arc2[0][1])
                    y_row[id_a2] = 1
                    add_idle_arcs(ws_loc, arc2[1][1], arrive_t)
            if not via_storage:
                add_idle_arcs(prev_loc, prev_t, arrive_t)

        else:
            via_storage = False
            id_a, arc_via = _find_exact_departure_arc(d, prev_loc, prev_t, storage_loc)
            if arc_via is not None:
                id_a2, arc2 = find_arc(storage_loc, arc_via[1][1], ws_loc, arrive_t)
                if id_a2 is not None:
                    via_storage = True
                    y_row[id_a] = 1
                    add_idle_arcs(storage_loc, arc_via[1][1], arc2[0][1])
                    y_row[id_a2] = 1
                    add_idle_arcs(ws_loc, arc2[1][1], arrive_t)
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
        id_arc, arc = _find_exact_departure_arc(d, prev_loc, prev_t, storage_loc)
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

    # Vectorised backlog penalty (equivalent to the original nested-sum
    # generator expression, but computed with NumPy broadcasting instead
    # of a Python-level double loop over orders x time).
    M = len(d.orders)
    arrival = np.asarray([d.arrival_times[m] for m in range(M)], dtype=np.float64)
    t_idx = np.arange(T, dtype=np.float64)
    age = (d.current_time + t_idx[np.newaxis, :] * d.OptManager.TIME_UNIT
           - arrival[:, np.newaxis]) / d.OptManager.TIME_UNIT
    backlog_penalty = float((age * (1.0 - g)).sum())

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
                a for a in d.incoming_arc_idx.get((ws_p, t), [])
                if a < n_travel
            ]
            pod_arrivals = d.OptManager.DELTA_POD * y[:, travel_arrivals].sum()
            total = float(item_work + pod_arrivals)
            if total > 2*d.OptManager.TIME_UNIT + 1e-6:
                ec11.append({'w': w, 't': t, 'value': total})
    if ec11:
        viols['EC11'] = ec11

    # EC12: each pod must depart its storage location exactly once at t=0.
    ec12 = []
    for rel_p, p_id in enumerate(d.from_RelPod_to_PodId):
        num_constraints += 1
        stor     = d.warehouse.pods[p_id].storage_location
        out_arcs = d.outgoing_arc_idx.get((stor, 0), [])
        flow_out = float(y[rel_p, out_arcs].sum())
        if not np.isclose(flow_out, 1.0):
            ec12.append({'pod': p_id, 'flow_out': flow_out})
    if ec12:
        viols['EC12'] = ec12

    # EC13: flow conservation for each pod at every intermediate node of
    # the time-expanded network (inflow == outflow).
    ec13 = []
    for rel_p in range(len(d.from_RelPod_to_PodId)):
        for node in d.nodes:
            num_constraints += 1
            if node[1] in (0, d.OptManager.N_TIME - 1):
                continue
            in_f  = float(y[rel_p, d.incoming_arc_idx.get(node, [])].sum())
            out_f = float(y[rel_p, d.outgoing_arc_idx.get(node, [])].sum())
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
            t_arcs = d.incoming_arc_idx.get((ws_p, first_t), [])
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
    m_of_im = _get_m_of_im(d)
    dx_gt_v = dx > v[m_of_im, 1:] + 1e-6   # vectorised: (n_items, T-1) mask
    for im in np.where(dx_gt_v.any(axis=1))[0]:
        bad_ts = np.where(dx_gt_v[im])[0] + 1
        poa.append({'im': int(im), 'times': bad_ts.tolist()})
    if poa:
        viols['pick_only_if_active'] = poa

    # EC17: v must equal f - g.
    num_constraints +=  v.shape[0]*v.shape[1]
    bad = np.argwhere(np.abs(v - (f - g)) > 1e-6)
    if bad.size:
        viols['EC17'] = bad.tolist()

    # EC18: f[m,t] >= x[im,t] for every item im of order m.
    # Vectorised mask via m_of_im; per-item detail list only built for the
    # (normally few) items that actually violate it.
    num_constraints += len(d.relevant_pairs_for_x) * f.shape[1]
    ec18_mask = f[m_of_im] < x - 1e-6
    ec18 = []
    for im in np.where(ec18_mask.any(axis=1))[0]:
        _, m = d.relevant_pairs_for_x[int(im)]
        ec18.append({'im': int(im), 'm': m, 'times': np.where(ec18_mask[im])[0].tolist()})
    if ec18:
        viols['EC21'] = ec18

    # EC19: g[m,t] <= x[im,t-1] for every item im of order m.
    num_constraints += len(d.relevant_pairs_for_x) * (g.shape[1] - 1)
    ec19_mask = g[m_of_im, 1:] > x[:, :-1] + 1e-6
    ec19 = []
    for im in np.where(ec19_mask.any(axis=1))[0]:
        _, m = d.relevant_pairs_for_x[int(im)]
        ec19.append({'im': int(im), 'm': m,
                     'times': (np.where(ec19_mask[im])[0] + 1).tolist()})
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

    # Warm the cached im -> m mapping once, up front (see _get_m_of_im);
    # every subsequent call in the hot loop (_check_x_fast, check_constraints)
    # reuses this cached array instead of rebuilding it.
    _get_m_of_im(d)

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
        print(f"[ls_stage2] violated = {list(viols)} (attempt {attempt}/{max_attempts})")
        logging.info("[ls_stage2] violated = %s (attempt %d/%d)", viols, attempt, max_attempts)
        for k, vv in viols.items():
            print(f"  {k}: {vv[:3] if isinstance(vv, list) else vv}")

        if attempt >= max_attempts:
            msg = f"[ls_stage2] Failed to find a feasible initial solution after {max_attempts} attempts."
            print(msg)
            logging.error(msg)
            raise RuntimeError(msg)

        x_current = build_initial_x(rng, d, attempt)
        attempt += 1
        _, f0, g0, v0, y0 = build_solution(x_current, d)
        feasible, viols, num_constraints = check_constraints((x_current, f0, g0, v0, y0), d)

    best_x   = x_current.copy()
    best_sol = (best_x, f0, g0, v0, y0)
    best_obj = compute_objective(x_current, f0, g0, d)
    print(f"[ls_stage2] Feasible initial solution: obj = {best_obj:.4f}")
    logging.info("[ls_stage2] Feasible initial solution: obj = %.4f", best_obj)
    logging.warning(
        f"\nVariables size:\n"
        f"  x shape = {best_sol[0].shape}\n"
        f"  f0 = {best_sol[1].shape}\n"
        f"  g0 = {best_sol[2].shape}\n"
        f"  v0 = {best_sol[3].shape}\n"
        f"  y shape = {best_sol[4].shape}\n"
        f"  num constraints satisfied = {num_constraints}\n"
    )
    
    T = x_current.shape[1]
    item_ids = list(range(x_current.shape[0]))

    # 'order' and 'swap' moves depend only on static problem data
    # (d.orders / d.orders_by_workstation), so they're built once here
    # instead of being reconstructed from scratch on every iteration of
    # the main loop below.
    static_order_moves = []
    for m in range(len(d.orders)):
        static_order_moves.append(('order', m, -1))
        static_order_moves.append(('order', m, -2))

    static_swap_moves = []
    for order_ids in d.orders_by_workstation:
        order_list = list(order_ids)
        for i1, m1 in enumerate(order_list):
            for m2 in order_list[i1 + 1:]:
                static_swap_moves.append(('swap', m1, m2))

    ### MAIN LOOP
    am_I_stuck                   = False
    cont                         = 1
    iter_without_improvement     = 0
    max_iter_without_improvement = 5
    MAX_ITER  = 100
    MAX_NEIGH = 400
    ACCEPT_WORSE_PROB = 0.4
    K_BEST = 4

    print("[ls_stage2] Exploring neighbours ...")
    current_sol = best_sol

    while not am_I_stuck and cont <= MAX_ITER:
        # first_one_idx: earliest pick-time index per item-order pair in
        # the current incumbent (used as the base for all move generators).
        first_one_idx = np.argmax(x_current > 0.5, axis=1)
        first_one_idx[x_current[:, -1] == 0] = T
        improved = False

        # ---- Build move list ----------------------------------------- #
        # moves[0]: item-level moves ('item' / 'multi_item')
        # moves[1]: order-level moves ('order')
        # moves[2]: pairwise 'swap' moves between orders at the same
        #           workstation
        moves = [[], [], [], []]

        # When stuck for a while, use small, conservative shifts (-1, -2);
        # otherwise explore more aggressive shifts (-2 to -8) to escape
        # local optima faster.
        if iter_without_improvement > 0:
            probs = [0.5,0.5, 0, 0]
            for im in range(x_current.shape[0]):
                moves[0].append(('item', im, -1))
                moves[0].append(('item', im, -2))
            for direction in [-1, -2]:
                sampled = rng.choice(item_ids, size=min(len(item_ids), 20), replace=False)
                for i in range(0, len(sampled) - 1, 2):
                    moves[1].append(('multi_item', (sampled[i], sampled[i + 1]), direction))
                    if i + 2 < len(sampled) - 1:
                        moves[1].append((
                            'multi_item',
                            (sampled[i], sampled[i + 1], sampled[i + 2]),
                            direction
                        ))
        else:
            probs = [0.4,0.3,0.1,0.2]
            for im in range(x_current.shape[0]):
                moves[0].append(('item', im, -2))
                moves[0].append(('item', im, -4))
                moves[0].append(('item', im, -6))
                moves[0].append(('item', im, -8))
            for direction in [-2, -4, -5]:
                sampled = rng.choice(item_ids, size=min(len(item_ids), 20), replace=False)
                for i in range(0, len(sampled) - 1, 2):
                    moves[1].append(('multi_item', (sampled[i], sampled[i + 1]), direction))
                    if i + 2 < len(sampled) - 1:
                        moves[1].append((
                            'multi_item',
                            (sampled[i], sampled[i + 1], sampled[i + 2]),
                            direction
                        ))

            # Reuse the precomputed static move lists instead of rebuilding
            # them (they don't depend on the current x_current / iteration).
            moves[2] = list(static_order_moves)
            moves[3] = list(static_swap_moves)

        # Checking probs is correct
        is_probability_vector = (
            len(probs) == len(moves) and
            all(p >= 0 for p in probs) and
            abs(sum(probs) - 1) < 1e-10
        )

        if is_probability_vector:
            pass

        elif len(probs) == len(moves):
            # normalizatin if possible
            total = sum(probs)

            if total > 0 and all(p >= 0 for p in probs):
                probs = [p / total for p in probs]
            else:
                # fallback
                probs = [0] * len(moves)
                probs[0] = 0.5
                probs[1] = 0.5

        else:
            # fallback
            probs = [0] * len(moves)
            probs[0] = 0.5
            probs[1] = 0.5

        # Cap the total neighbourhood size, subsampling each move category
        # proportionally rather than truncating a single category entirely.
        total = sum(len(mv) for mv in moves)
        if total > MAX_NEIGH:
            for i, p in enumerate(probs):
                size = min(len(moves[i]), int(np.ceil(MAX_NEIGH * p)))
                if size and len(moves[i]) > size:
                    idxs     = rng.choice(len(moves[i]), size=size, replace=False)
                    moves[i] = [moves[i][j] for j in idxs]

        all_moves = moves[0] + moves[1] + moves[2] + moves[3]

        # ---- Evaluate moves ------------------------------------------ #
        # Top-K_BEST candidates are kept with a bounded min-heap instead of
        # sorting the whole list after every single append: this avoids
        # O(n log n) work on every accepted neighbour when the pool of
        # feasible candidates in a given iteration is large.
        heap: list = []
        tie_breaker = itertools.count()

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
            _, f_curr, g_curr, v_curr, _ = current_sol
            f_cand, g_cand, v_cand = _fast_update_fgv_from_move(
                x_cand, f_curr, g_curr, v_curr,
                move, im_by_order, first_one_idx_cand, d,
            )

            if _check_x_fast(x_cand, f_cand, g_cand, v_cand, d):
                obj = compute_objective(x_cand, f_cand, g_cand, d)
                candidate = {
                    "obj": obj,
                    "x": x_cand,
                    "f": f_cand,
                    "g": g_cand,
                    "v": v_cand,
                    "move": move
                }

                entry = (obj, next(tie_breaker), candidate)
                if len(heap) < K_BEST:
                    heapq.heappush(heap, entry)
                elif obj > heap[0][0]:
                    heapq.heapreplace(heap, entry)

        best_candidates = [c for _, _, c in sorted(heap, key=lambda e: e[0], reverse=True)]


        # ---- Attempt to commit the best neighbour -------------------- #
        # The best x-only neighbour may still violate y-based constraints.
        # If it fails, try the second-best neighbour found in this iteration.
        #
        # A feasible worsening move can be accepted with a small probability
        # to escape local optima (diversification step). The global incumbent
        # is updated only when the objective improves.

        accepted_move = False

        for cand in best_candidates:
            x_cand = cand["x"]
            f_cand = cand["f"]
            g_cand = cand["g"]
            v_cand = cand["v"]
            obj_cand = cand["obj"]
            move_cand = cand["move"]

            if x_cand is None:
                continue

            # Rebuild only the affected pod routes.
            if move_cand[0] in ('item', 'order', 'multi_item', 'swap'):

                if move_cand[0] == 'item':
                    _, im, _ = move_cand
                    affected_pods = {d.pod_of_item[int(im)]}

                elif move_cand[0] == 'multi_item':
                    _, ims, _ = move_cand
                    affected_pods = {
                        d.pod_of_item[int(im)]
                        for im in ims
                    }

                elif move_cand[0] == 'order':
                    _, m, _ = move_cand
                    affected_pods = {
                        d.pod_of_item[im]
                        for im in im_by_order[int(m)]
                    }

                else:   # swap
                    _, m1, m2 = move_cand
                    affected_pods = (
                        {
                            d.pod_of_item[im]
                            for im in im_by_order[int(m1)]
                        }
                        |
                        {
                            d.pod_of_item[im]
                            for im in im_by_order[int(m2)]
                        }
                    )

                y_new = best_sol[4].copy()

                # first_one_idx for x_cand computed once here, and shared
                # across all affected pods below, instead of each call to
                # _rebuild_pod_row recomputing it independently.
                first_one_idx_x_cand = (x_cand == 0).sum(axis=1)

                for p_id in affected_pods:
                    p_rel = d.from_PodId_to_RelPod[p_id]
                    y_new[p_rel] = _rebuild_pod_row(
                        p_rel, p_id, x_cand, d, first_one_idx_x_cand)

                candidate_sol = (x_cand,f_cand,g_cand,v_cand,y_new)

            else:
                candidate_sol = build_solution(x_cand, d)


            feasible, viols, _ = check_constraints(candidate_sol, d)
            if not feasible:
                logging.info(
                    "[ls_stage2] Candidate rejected: "
                    "move=%s obj=%.4f violations=%s",
                    move_cand,
                    obj_cand,
                    list(viols.keys()),
                )
                continue


            # ----------------------------------------------------------
            # Improvement: always accept and update global incumbent.
            # ----------------------------------------------------------
            if obj_cand > best_obj + 1e-10:

                best_obj = obj_cand
                best_x = x_cand.copy()
                best_sol = candidate_sol

                x_current = x_cand.copy()
                current_sol = candidate_sol
                accepted_move = True

                print(
                    f"[ls_stage2] Iter {cont}: improved "
                    f"move={move_cand} obj={best_obj:.4f}"
                )

                logging.info(
                    "[ls_stage2] Iter %d: improved move=%s obj=%.4f",
                    cont,move_cand,best_obj,
                )

                break


            # ----------------------------------------------------------
            # Diversification: occasionally accept a worse solution.
            # The global best is preserved, but the current search point
            # moves away from the local optimum.
            # ----------------------------------------------------------
            if rng.random() < ACCEPT_WORSE_PROB:

                x_current = x_cand.copy()

                # keep f/g/v/y consistent with the accepted current point
                current_sol = candidate_sol
                accepted_move = True

                print(
                    f"[ls_stage2] Iter {cont}: accepted worse solution "
                    f"move={move_cand} obj={obj_cand:.4f} "
                    f"(best={best_obj:.4f})"
                )

                logging.info(
                    "[ls_stage2] Iter %d: accepted worse solution "
                    "move=%s obj=%.4f best=%.4f",
                    cont,move_cand,obj_cand,best_obj
                )

                break


        # ---- Convergence check -------------------------------------- #
        # Any accepted move (improving or worsening) means the search is
        # still exploring. Only iterations without accepted neighbours
        # increase the stagnation counter.

        if accepted_move:
            iter_without_improvement = 0

        else:
            iter_without_improvement += 1

            print(
                f"[ls_stage2] Iter {cont}: no accepted move "
                f"({iter_without_improvement}/"
                f"{max_iter_without_improvement})"
            )

            logging.info(
                "[ls_stage2] Iter %d: no accepted move (%d/%d)",
                cont,
                iter_without_improvement,
                max_iter_without_improvement,
            )

            if iter_without_improvement >= max_iter_without_improvement:
                am_I_stuck = True

                print(
                    f"[ls_stage2] Converged after "
                    f"{max_iter_without_improvement} iterations "
                    f"without accepted moves. "
                    f"Best obj={best_obj:.4f}"
                )

                logging.info(
                    "[ls_stage2] Converged after %d iterations "
                    "without accepted moves. Best obj=%.4f",
                    max_iter_without_improvement,best_obj
                )

        cont+=1

    print("[ls_stage2] Local search ended.")
    return best_sol