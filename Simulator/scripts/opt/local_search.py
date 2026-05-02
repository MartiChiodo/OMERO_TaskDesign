from __future__ import annotations

import logging
import numpy as np

from .stage2_data import Stage2Data


### Solution builder

def build_solution(x: np.ndarray, d: Stage2Data) -> tuple:
    """
    Derive the full solution (f, g, v, y) from the picking matrix x.

    Parameters
    ----------
    x : np.ndarray, shape (n_im, T)
        Cumulative picking matrix — x[im, t] = 1 if item im has been picked by t.
    d : Stage2Data

    Returns
    -------
    sol : tuple (x, f, g, v, y)
        x  — picking matrix (input, returned for convenience).
        f  — f[m, t] = 1 if at least one item of order m is picked by t.
        g  — g[m, t] = 1 if all items of order m are picked by t-1.
        v  — v[m, t] = f[m, t] - g[m, t]  (1 while order is active).
        y  — y[p, a] = 1 if pod p traverses arc a in the time-space network.
    """
    n_travel = len(d.OptManager.travelling_arcs)
    M  = len(d.orders)
    T  = x.shape[1]

    # Building f, g, v 
    # first_one_idx[im] = first t where x[im, t] flips from 0 to 1
    first_one_idx = (x == 0).sum(axis=1)                          # shape: (n_im,)

    all_ms = np.array([m for (_, m) in d.relevant_pairs_for_x])  # shape: (n_im,)
    all_ts = first_one_idx                                        # shape: (n_im,)

    # t_start[m] = earliest pick time among all items of order m
    # t_end[m]   = latest pick time + 1  (= first t where g can be 1)
    t_start = np.full(M, T, dtype=int)
    t_end  = np.zeros(M,  dtype=int)
    np.minimum.at(t_start, all_ms, all_ts)
    np.maximum.at(t_end, all_ms, all_ts)
    t_end += 1

    time_range = np.arange(T)                                     # shape: (T,)
    f = (time_range[np.newaxis, :] >= t_start[:, np.newaxis]).astype(np.float64)
    g = (time_range[np.newaxis, :] >= t_end[:, np.newaxis]  ).astype(np.float64)
    v = f - g

    # Enforcing initial conditions on v
    if len(d.opened_order_ids) > 0:
        for m, order in enumerate(d.orders):
            if order.order_id in d.opened_order_ids:
                # Orders already open must have v=1 from t=0 to completion
                v[m,:t_end[m]] = 1

        f = v+g

                
    # Building y (pod routing)
    y = np.zeros(
        (len(d.from_RelPod_to_PodId), len(d.OptManager.all_arcs)),
        dtype=np.float64,
    )

    def add_idle_arcs(y, p_rel: int, loc: int, t_from: int, t_to: int) -> None:
        """Set y=1 for idle arcs keeping pod p_rel at loc from t_from to t_to."""
        for t in range(t_from, t_to):
            for id_a in d.OptManager.outgoing_arc_idx.get((loc, t), []):
                if d.OptManager.all_arcs[id_a][1] == (loc, t + 1):
                    y[p_rel, id_a] = 1
                    break

    def find_arc_departing_after(src_loc: int, t_from: int, dst_loc: int, latest_arrival: int):
        """
        Find a travelling arc departing from src_loc at or after t_from and arriving
        at dst_loc no later than latest_arrival.

        Chooses the arc with the latest feasible arrival time, to minimise idle
        waiting at the destination.
        Returns (arc_id, arc) or (None, None).
        """
        best_arc = None
        best_id = None
        for t_dep in range(t_from, latest_arrival):
            for id_a in d.OptManager.outgoing_arc_idx.get((src_loc, t_dep), []):
                if id_a >= n_travel:
                    continue
                arc = d.OptManager.all_arcs[id_a]
                if arc[1][0] != dst_loc:
                    continue
                if arc[1][1] <= latest_arrival:
                    if best_arc is None or arc[1][1] > best_arc[1][1]:
                        best_arc = arc
                        best_id = id_a
        return best_id, best_arc or (None, None)

    infeasible_pods = []

    for p_rel, p_id in enumerate(d.from_RelPod_to_PodId):
        storage_loc = d.warehouse.pods[p_id].storage_location

        # Build sorted, deduplicated sequence of (ws_loc, t) events for this pod
        items_for_pod = {
            im: int(first_one_idx[im])
            for im, (_, _) in enumerate(d.relevant_pairs_for_x)
            if d.pod_of_item[im] == p_id and int(first_one_idx[im]) < d.OptManager.N_TIME 
        }
        events: list[tuple[int, int]] = []
        for im, t in sorted(items_for_pod.items(), key=lambda kv: kv[1]):
            _, m  = d.relevant_pairs_for_x[im]
            ws_loc = d.ws_positions[d.order_to_ws[m]]
            if not events or events[-1] != (ws_loc, t):
                events.append((ws_loc, t))

        prev_loc, prev_t = storage_loc, 0

        for ws_loc, arrive_t in events:
            if prev_loc == ws_loc:

                via_storage = False
                set_arc_outgoing = d.OptManager.outgoing_arc_idx.get((prev_loc, prev_t), [])
                for id_a in set_arc_outgoing:
                    arc = d.OptManager.all_arcs[id_a]
                    if arc[1][0] == storage_loc:
                       arc1 = arc
                       id_a1 = id_a
                       break
                
                id_a2, arc2 = find_arc_departing_after(storage_loc, arc1[1][1], ws_loc, arrive_t)
                if id_a2 is not None:
                    via_storage = True
                    y[p_rel, id_a1] = 1
                    add_idle_arcs(y, p_rel, storage_loc, arc1[1][1], arc2[0][1])
                    y[p_rel, id_a2] = 1
                    add_idle_arcs(y, p_rel, ws_loc, arc2[1][1], arrive_t)
                    via_storage = True

                if not via_storage:
                    add_idle_arcs(y, p_rel, prev_loc, prev_t, arrive_t)
            else:
                via_storage = False
                set_arc_outgoing = d.OptManager.outgoing_arc_idx.get((prev_loc, prev_t), [])
                for id_a in set_arc_outgoing:
                    arc = d.OptManager.all_arcs[id_a]
                    if arc[1][0] == storage_loc:
                       arc1 = arc
                       id_a1 = id_a
                       break
                
                id_a2, arc2 = find_arc_departing_after(storage_loc, arc1[1][1], ws_loc, arrive_t)
                if id_a2 is not None:
                    via_storage = True
                    y[p_rel, id_a1] = 1
                    add_idle_arcs(y, p_rel, storage_loc, arc1[1][1], arc2[0][1])
                    y[p_rel, id_a2] = 1
                    add_idle_arcs(y, p_rel, ws_loc, arc2[1][1], arrive_t)
                    via_storage = True
                    
                if not via_storage:
                    arc_id, arc = find_arc_departing_after(prev_loc, prev_t, ws_loc, arrive_t)
                    if arc_id is None:
                        infeasible_pods.append((p_rel, prev_loc, prev_t, ws_loc, arrive_t))
                    else:
                        add_idle_arcs(y, p_rel, prev_loc, prev_t, arc[0][1])
                        y[p_rel, arc_id] = 1

            prev_loc, prev_t = ws_loc, arrive_t

        # After the last event, send the pod back to its storage location
        if prev_loc != storage_loc:
            arc_id, arc = find_arc_departing_after(prev_loc, prev_t, storage_loc, d.OptManager.N_TIME - 1)
            if arc_id is None:
                infeasible_pods.append((p_rel, prev_loc, prev_t, storage_loc, d.OptManager.N_TIME - 1))
            else:
                add_idle_arcs(y, p_rel, prev_loc, prev_t, arc[0][1])
                y[p_rel, arc_id] = 1
                add_idle_arcs(y, p_rel, storage_loc, arc[1][1], d.OptManager.N_TIME - 1)
        else:
            add_idle_arcs(y, p_rel, storage_loc, prev_t, d.OptManager.N_TIME - 1)

    if infeasible_pods:
        logging.warning(
            "[build_solution] %d infeasible pod segments: %s",
            len(infeasible_pods), infeasible_pods,
        )

    return x, f, g, v, y


### Objective function 

def compute_objective(sol: tuple, d: Stage2Data) -> float:
    """
    Evaluate the Stage-2 objective (to be maximised):

        obj = sum(g) + 0.2 * sum(x[:, T-1]) - 0.5 * backlog_penalty

    where backlog_penalty penalises orders that are never opened, weighted
    by how long they have been waiting in the backlog.
    """
    x, f, g, v, y = sol
    T = x.shape[1]

    completion_reward = float(g.sum())
    pickup_bonus  = 0.5 * float(x[:, T - 1].sum())
    backlog_penalty = 0.5 * float(sum(
        (d.current_time - d.arrival_times[m]) / d.OptManager.TIME_UNIT
        * (1.0 -g[m, T - 1])
        for m in range(len(d.orders))
    ))

    return completion_reward + pickup_bonus - backlog_penalty


### Constraint checker

def check_constraints(sol: tuple, d: Stage2Data) -> tuple[bool, dict]:
    """
    Verify all Stage-2 constraints against a candidate solution.

    Returns
    -------
    feasible   : bool   True iff no constraint is violated.
    violations : dict   Maps constraint name to violation details (empty if feasible).
    """
    x, f, g, v, y = sol
    T = x.shape[1]
    n_travel = len(d.OptManager.travelling_arcs)
    viols: dict = {}

    # EC13: workstation throughput cap 
    # sum_{m at w} v[m, t] <= CAP_WS  for all w, t
    for w, order_ids in enumerate(d.orders_by_workstation):
        cap = v[list(order_ids), :].sum(axis=0)           # shape: (T,)
        bad = np.where(cap > d.OptManager.CAP_WS + 1e-6)[0]
        if bad.size:
            viols.setdefault('EC13', []).append(
                {'w': w, 'times': bad.tolist(), 'values': cap[bad].tolist()})

    # EC14: time capacity per (workstation, t)
    # DELTA_ITEM * new_picks + DELTA_POD * pod_at_ws <= TIME_UNIT
    ec14 = []
    for w, order_ids in enumerate(d.orders_by_workstation):
        ws_p  = d.ws_positions[w]
        ims_w = [im for im, (_, m) in enumerate(d.relevant_pairs_for_x) if m in order_ids]
        for t in range(1, T):
            item_work = d.OptManager.DELTA_ITEM * (x[ims_w, t] - x[ims_w, t - 1]).sum()
            travel_arrivals = [
                a for a in d.OptManager.incoming_arc_idx.get((ws_p, t), [])
                if a < n_travel   # idle arcs excluded
            ]
            pod_arrivals = d.OptManager.DELTA_POD * y[:, travel_arrivals].sum()
            total = float(item_work + pod_arrivals)
            if total > d.OptManager.TIME_UNIT + 1e-6:
                ec14.append({'w': w, 't': t, 'value': total})
    if ec14:
        viols['EC14'] = ec14

    # EC15: each pod departs from its storage location exactly once at t=0 
    ec15 = []
    for rel_p, p_id in enumerate(d.from_RelPod_to_PodId):
        stor = d.warehouse.pods[p_id].storage_location
        out_arcs = d.OptManager.outgoing_arc_idx.get((stor, 0), [])
        flow_out = float(y[rel_p, out_arcs].sum())
        if not np.isclose(flow_out, 1.0):
            ec15.append({'pod': p_id, 'flow_out': flow_out})
    if ec15:
        viols['EC15'] = ec15

    # EC16: flow conservation at intermediate nodes 
    # flow_in[p, node] == flow_out[p, node]  for all p, node with 0 < t < T-1
    ec16 = []
    for rel_p in range(len(d.from_RelPod_to_PodId)):
        for node in d.OptManager.nodes:
            if node[1] in (0, d.OptManager.N_TIME - 1):
                continue
            in_f = float(y[rel_p, d.OptManager.incoming_arc_idx.get(node, [])].sum())
            out_f = float(y[rel_p, d.OptManager.outgoing_arc_idx.get(node, [])].sum())
            if not np.isclose(in_f - out_f, 0.0):
                ec16.append({'pod': rel_p,
                             'node': node, 'imbalance': in_f - out_f})
    if ec16:
        viols['EC16'] = ec16

    # EC18: no pick at t=0; pick only when pod is present at workstation 
    first_pick_time = (x == 0).sum(axis=1)  # shape: (n_im,)
    ec18 = []
    for im, first_t in enumerate(first_pick_time):
        if first_t < d.OptManager.N_TIME:
            _, m = d.relevant_pairs_for_x[im]
            ws_p = d.ws_positions[d.order_to_ws[m]]
            rel_p = d.from_PodId_to_RelPod[d.pod_of_item[im]] 
            t_arcs = d.OptManager.outgoing_arc_idx.get((ws_p, first_t), [])
            pod_here = float(y[rel_p, t_arcs].sum())
            if pod_here < 1e-6:
                ec18.append({'im': im, 't': first_t, 'pod_here': pod_here})   
    if ec18:
        viols['EC18'] = ec18

    # EC19: x is non-decreasing (once picked, stays picked)
    dx  = np.diff(x, axis=1)                              # shape: (n_im, T-1)
    bad = np.argwhere(dx < -1e-6)
    if bad.size:
        viols['EC19'] = bad.tolist()

    # pick_only_if_active: x[im,t] - x[im,t-1] <= v[m,t]
    poa = []
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        bad_ts = np.where(dx[im] > v[m, 1:] + 1e-6)[0] + 1
        if bad_ts.size:
            poa.append({'im': im, 'times': bad_ts.tolist()})
    if poa:
        viols['pick_only_if_active'] = poa

    # EC20: v == f - g
    bad = np.argwhere(np.abs(v - (f - g)) > 1e-6)
    if bad.size:
        viols['EC20'] = bad.tolist()

    # EC21: f[m,t] >= x[im,t]  for all im of order m 
    ec21 = []
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        bad = np.where(f[m] < x[im] - 1e-6)[0]
        if bad.size:
            ec21.append({'im': im, 'm': m, 'times': bad.tolist()})
    if ec21:
        viols['EC21'] = ec21

    # EC22: g[m,t] <= x[im,t-1]  for all im of order m 
    ec22 = []
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        bad = np.where(g[m, 1:] > x[im, :-1] + 1e-6)[0] + 1
        if bad.size:
            ec22.append({'im': im, 'm': m, 'times': bad.tolist()})
    if ec22:
        viols['EC22'] = ec22

    # Monotonicity of f and g 
    if (np.diff(f, axis=1) < -1e-6).any():
        viols['f_monotonicity'] = True
    if (np.diff(g, axis=1) < -1e-6).any():
        viols['g_monotonicity'] = True

    # Continuity of v: v[m,t] >= v[m,t-1] - g[m,t]
    bad = np.argwhere(v[:, 1:] - (v[:, :-1] - g[:, 1:]) < -1e-6)
    if bad.size:
        viols['continuity_v'] = bad.tolist()

    # g lower bound: g[m,t+1] >= sum_im x[im,t] - (n_items_m - 1)
    g_lb = []
    for m in range(len(d.orders)):
        ims = d.items_of_order[m]
        n_items = int(d.n_items_per_order[m])
        lb = x[ims, :-1].sum(axis=0) - (n_items - 1)   # shape: (T-1,)
        bad = np.where(g[m, 1:] < lb - 1e-6)[0] + 1
        if bad.size:
            g_lb.append({'m': m, 'times': bad.tolist()})
    if g_lb:
        viols['g_lower_bound'] = g_lb

    # Initial conditions
    for m, order in enumerate(d.orders):
        if order.order_id in d.opened_order_ids:
            # Orders already open must have v=1 at t=0
            if not np.isclose(float(v[m, 0]), 1.0):
                viols.setdefault('initial_cond', []).append(
                    {'m': m, 'v0': float(v[m, 0])})
        else:
            # f[m,t] can only be 1 if at least one item has been picked by t
            ims = d.items_of_order[m]
            bad = np.where(f[m] > x[ims, :].sum(axis=0) + 1e-6)[0]
            if bad.size:
                viols.setdefault('f_active_only_if_picked', []).append(
                    {'m': m, 'times': bad.tolist()})

    feasible = len(viols) == 0
    return feasible, viols


### Initial solution 

def build_initial_x(rng: np.random.Generator, d: Stage2Data) -> np.ndarray:
    T    = d.OptManager.N_TIME
    n_im = len(d.relevant_pairs_for_x)
    x    = np.zeros((n_im, T), dtype=np.float64)
    scheduled = np.zeros(n_im, dtype=bool)

    # Step 1: already -> open orders 
    for m, order in enumerate(d.orders):
        if order.order_id in d.opened_order_ids:
            for im in d.items_of_order[m]:
                t0 = max(int(d.earliest_t[im]), 1)
                x[im, t0:] = 1
                scheduled[im] = True

    # Step 2: remaining orders -> batch by CAP_WS, stagger pods within batch 
    for w, order_ids in enumerate(d.orders_by_workstation):

        # Only unscheduled orders
        pending_orders = [
            m for m in order_ids
            if not all(scheduled[im] for im in d.items_of_order[m])
        ]
        if not pending_orders:
            continue

        # Sort orders by the earliest time any of their pods can reach this ws
        def order_earliest(m):
            return max(
                max(int(d.earliest_t[im]) for im in d.items_of_order[m]),
                1
            )

        pending_orders.sort(key=order_earliest)

        # Split into batches of CAP_WS — orders in the same batch run in parallel,
        # batches run sequentially so at most CAP_WS orders are active at any t.
        batches = [
            pending_orders[i : i + d.OptManager.CAP_WS]
            for i in range(0, len(pending_orders), d.OptManager.CAP_WS)
        ]

        batch_start_t = 0   # earliest slot available for the current batch

        for batch in batches:

            # Collect all pods needed by this batch
            pod_to_ims: dict[int, list[int]] = {}
            for m in batch:
                for im in d.items_of_order[m]:
                    if not scheduled[im]:
                        pod_to_ims.setdefault(d.pod_of_item[im], []).append(im)

            # Earliest pick time per pod (must respect both travel time and batch_start_t)
            pod_earliest_in_batch = {
                p_id: max(
                    max(int(d.earliest_t[im]) for im in ims),
                    batch_start_t,
                    1,
                )
                for p_id, ims in pod_to_ims.items()
            }

            # Within the batch, stagger pods so no two travel-arcs arrive at same t
            # (EC14 compliance) — sort by earliest and assign sequential slots
            next_free_t = batch_start_t
            pod_pick_t: dict[int, int] = {}

            for p_id in sorted(pod_to_ims, key=lambda p: pod_earliest_in_batch[p]):
                t_pick = max(pod_earliest_in_batch[p_id], next_free_t)
                if t_pick >= T:
                    logging.warning(
                        "[build_initial_x] Pod %d at ws %d exceeds horizon "
                        "(t_pick=%d >= T=%d).", p_id, w, t_pick, T
                    )
                    continue
                pod_pick_t[p_id] = t_pick
                next_free_t = t_pick + 1

            # Assign x and mark scheduled
            for p_id, ims in pod_to_ims.items():
                if p_id not in pod_pick_t:
                    continue
                t_pick = pod_pick_t[p_id]
                for im in ims:
                    x[im, t_pick:] = 1
                    scheduled[im]  = True

            # Batch k+1 starts only after all orders in batch k are closed.
            # t_end of an order = last pod pick in that order + 1.
            # We take the max over all pods in this batch.
            if pod_pick_t:
                batch_start_t = max(pod_pick_t.values()) + 1

    return x

### Local search


def _check_x_only(x_cand: np.ndarray, d: Stage2Data, im_by_order: dict) -> bool:
    """
    Cheap feasibility pre-filter on x alone, before the expensive build_solution.
    Checks only constraints that depend solely on x (not y).
    Returns False immediately on the first violation found.
    """
    T     = x_cand.shape[1]
    dx    = np.diff(x_cand, axis=1)

    # EC19: x non-decreasing
    if (dx < -1e-6).any():
        return False

    # EC18: no pick at t=0
    if x_cand[:, 0].any():
        return False

    # pick_only_if_active and EC13: check per workstation
    all_ms = np.array([m for (_, m) in d.relevant_pairs_for_x])
    M      = len(d.orders)

    # Reconstruct v cheaply from x to check EC13
    first_one = (x_cand == 0).sum(axis=1)
    t_start   = np.full(M, T, dtype=int)
    t_end     = np.zeros(M, dtype=int)
    np.minimum.at(t_start, all_ms, first_one)
    np.maximum.at(t_end,   all_ms, first_one)
    t_end += 1
    time_range = np.arange(T)
    v = ((time_range[np.newaxis, :] >= t_start[:, np.newaxis]) &
         (time_range[np.newaxis, :] <  t_end[:, np.newaxis]  )).astype(np.float64)

    for w, order_ids in enumerate(d.orders_by_workstation):
        cap = v[list(order_ids), :].sum(axis=0)
        if (cap > d.OptManager.CAP_WS + 1e-6).any():
            return False

    return True


def _make_move_1(x: np.ndarray, im: int, variation: int, first_one_idx: np.ndarray, T: int):
    """
    Shift item im pick time by variation.
    Returns new x or None if out of bounds.
    """
    t0 = int(first_one_idx[im])
    t_new = t0 + variation
    if t_new < 1 or t_new >= T:
        return None
    x_cand = x.copy()
    x_cand[im, t_new:] = 1
    x_cand[im, :t_new] = 0

    return x_cand


def _make_move_2(x: np.ndarray, ims: list, variation: int, first_one_idx: np.ndarray, T: int):
    """
    Shift all items of an order by variation.
    Returns new x or None if any item goes out of bounds.
    """
    x_cand = x.copy()
    for im in ims:
        t0    = int(first_one_idx[im])
        t_new = t0 + variation
        if t_new < 1 or t_new >= T:
            return None
        x_cand[im, :t_new] = 0
        x_cand[im, t_new:] = 1
    return x_cand


def _make_move_3(x: np.ndarray, ims1: list, ims2: list, first_one_idx: np.ndarray, T: int):
    """
    Swap the pick-time slots of two orders.
    Returns new x or None if out of bounds.
    """
    delta = min(int(first_one_idx[im]) for im in ims2) \
          - min(int(first_one_idx[im]) for im in ims1)
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


def evaluate(x_cand, d):
    """Build solution and check constraints. Returns (obj, sol) or (None, None)."""
    sol = build_solution(x_cand, d)
    ok, _ = check_constraints(sol, d)
    if ok:
        return compute_objective(sol, d), sol
    return None, None


def local_search(d: Stage2Data) -> tuple:
    """
    First-improvement local search.

    For each iteration, neighbours are visited in randomised order and the
    first feasible improving one is accepted immediately — no need to scan
    the entire neighbourhood.

    Pipeline per candidate:
      1. Build x_cand          (O(1) — just flip a few entries)
      2. _check_x_only          (cheap — no y, just x-based constraints)
      3. build_solution + check_constraints  (expensive — only if step 2 passes)

    Moves tried (in randomised order each iteration):
      Move 1 — shift single item ±1
      Move 2 — shift entire order ±1
      Move 3 — swap two orders at the same workstation
    """
    rng = np.random.default_rng(seed=42)

    im_by_order: dict[int, list[int]] = {}
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        im_by_order.setdefault(m, []).append(im)

    # Initial solution 
    print("\n\n[local_search] Building initial solution ...")
    x_current = build_initial_x(rng, d)
    sol_current = build_solution(x_current, d)
    feasible, viols = check_constraints(sol_current, d)
    while not feasible:
        x_current = build_initial_x(rng, d)
        sol_current = build_solution(x_current, d)
        feasible, viols = check_constraints(sol_current, d)
    print(f"[local_search] Feasible initial solution found with obj {compute_objective(sol_current, d):.4f}.")

    best_sol = sol_current
    best_obj = compute_objective(sol_current, d) if feasible else -np.inf
    T = x_current.shape[1]

    # Track visited solutions to avoid cycles
    visited_x = set()
    visited_x.add(hash(x_current.tobytes()))

    am_I_stuck = False
    cont = 1
    iter_without_improvement = 0
    max_iter_without_improvement = 3  # Stop after 3 iterations without improvement

    while not am_I_stuck or cont <= 20:
        first_one_idx = (x_current == 0).sum(axis=1)
        improved  = False
        print(f"[local_search] Exploring neighbors (iteration {cont}).")
        best_obj_in_iter = - np.inf
        best_sol_in_iter = None
        best_move_in_iter = None

        # ── Build candidate list (cheap — just metadata, no x copy yet) ───────
        # Each entry: (move_type, args...)
        moves = [[] for _ in range(3)] 

        if iter_without_improvement > 0:
            # Moves will be focused on smaller adjustments 
            for im in range(x_current.shape[0]):
                moves[0].append(['item', im, -1])
                moves[0].append(['item', im, -2])
        else:
            for im in range(x_current.shape[0]):
                var = rng.integers(1, 4)
                moves[0].append(['item', im, -var])

            for m in range(len(d.orders)):
                var = rng.integers(1, 4)
                moves[1].append(['order', m, -var])

            for w, order_ids in enumerate(d.orders_by_workstation):
                order_list = list(order_ids)
                for i1, m1 in enumerate(order_list):
                    for m2 in order_list[i1 + 1:]:
                        moves[2].append(['swap', m1, m2])

        # Sample up to 30 random moves to speed up
        max_neigh = 30
        if len(moves[0]) + len(moves[1]) + len(moves[2])  > max_neigh:
            p0 = 0.3 if iter_without_improvement > 0 else 1
            moves[0] = rng.choice(moves[0], size=min(len(moves[0]), int(np.ceil(max_neigh * p0))), replace=False).tolist()
            moves[1] = rng.choice(moves[1], size=min(len(moves[1]), int(np.ceil(max_neigh * 0.3))), replace=False).tolist()
            moves[2] = rng.choice(moves[2], size=min(len(moves[2]), int(np.ceil(max_neigh * 0.4))), replace=False).tolist()

        moves = moves[0] + moves[1] + moves[2]
        for move in moves:

            #  Build candidate x 
            if move[0] == 'item':
                _, im, direction = move
                im, direction = int(im), int(direction)
                x_cand = _make_move_1(x_current, im, direction, first_one_idx, T)

            elif move[0] == 'order':
                _, m, direction = move
                m, direction = int(m), int(direction)
                x_cand = _make_move_2(
                    x_current, im_by_order.get(m, []), direction, first_one_idx, T
                )

            else:  # swap
                _, m1, m2 = move
                m1, m2 = int(m1), int(m2)
                x_cand = _make_move_3(
                    x_current,
                    im_by_order.get(m1, []),
                    im_by_order.get(m2, []),
                    first_one_idx, T,
                )

            if x_cand is None:
                continue   

            # Skip if already visited
            x_hash = hash(x_cand.tobytes())
            if x_hash in visited_x:
                continue

            # Pre-filter: x-only constraints (no y needed) 
            if not _check_x_only(x_cand, d, im_by_order):
                continue

            # Full evaluation: build y + check all constraints 
            obj, sol = evaluate(x_cand, d)
            if obj is not None and obj > best_obj_in_iter:
                best_obj_in_iter = obj
                best_sol_in_iter = sol
                best_move_in_iter = move
                x_current = x_cand
                visited_x.add(x_hash)  # Mark as visited
                # Limit set size to prevent memory issues
                if len(visited_x) > 1000:
                    visited_x.pop()

        if best_sol_in_iter is not None and best_obj_in_iter > best_obj:
            best_sol = best_sol_in_iter
            best_obj = best_obj_in_iter  
            improved  = True
            print(f"[local_search] Improved by move ({best_move_in_iter[0]}) → {best_obj:.4f}")
            

        if improved:
            iter_without_improvement = 0
        else:
            iter_without_improvement += 1
            if iter_without_improvement >= max_iter_without_improvement:
                am_I_stuck = True
                print(f"[local_search] Converged after {max_iter_without_improvement} iterations without improvement at {best_obj:.4f}")
            else:
                print(f"[local_search] No improvement in this iteration ({iter_without_improvement}/{max_iter_without_improvement})")

        cont += 1

    return best_sol