from __future__ import annotations
import numpy as np
from bisect import bisect_left
import logging

from .stage2_data import Stage2Data
from .build_initial_x import build_initial_x



### Fast helpers (no y needed)

def _build_fgv(x: np.ndarray, d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute f, g, v from x only, no pod routing.
    Identical logic to the corresponding block in build_solution.
    """
    T   = x.shape[1]
    M   = len(d.orders)
    all_ms = np.array([m for (_, m) in d.relevant_pairs_for_x])

    first_one_idx = (x == 0).sum(axis=1)

    t_start = np.full(M, T, dtype=int)
    t_end   = np.zeros(M,  dtype=int)
    np.minimum.at(t_start, all_ms, first_one_idx)
    np.maximum.at(t_end,   all_ms, first_one_idx)
    t_end += 1

    time_range = np.arange(T)
    f = (time_range[np.newaxis, :] >= t_start[:, np.newaxis]).astype(np.float64)
    g = (time_range[np.newaxis, :] >= t_end[:, np.newaxis]  ).astype(np.float64)
    v = f - g

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
    Replaces the cheaper _check_x_only used as a pre-filter; call this once
    _check_x_only passes to avoid building y unnecessarily.
    Returns True iff all x-only constraints hold.
    """
    T   = x.shape[1]
    M   = len(d.orders)
    dx  = np.diff(x, axis=1)

    # EC19: x non-decreasing
    if (dx < -1e-6).any():
        return False

    # EC18: no pick at t=0
    if x[:, 0].any():
        return False

    # EC13: workstation throughput cap
    for order_ids in d.orders_by_workstation:
        if (v[list(order_ids), :].sum(axis=0) > d.OptManager.CAP_WS + 1e-6).any():
            return False

    # EC20: v == f - g
    if (np.abs(v - (f - g)) > 1e-6).any():
        return False

    # EC21: f[m,t] >= x[im,t]
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        if (f[m] < x[im] - 1e-6).any():
            return False

    # EC22: g[m,t] <= x[im,t-1]
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        if (g[m, 1:] > x[im, :-1] + 1e-6).any():
            return False

    # Monotonicity of f and g
    if (np.diff(f, axis=1) < -1e-6).any():
        return False
    if (np.diff(g, axis=1) < -1e-6).any():
        return False

    # g lower bound: g[m,t+1] >= sum_im x[im,t] - (n_items_m - 1)
    for m in range(M):
        ims     = d.items_of_order[m]
        n_items = int(d.n_items_per_order[m])
        lb      = x[ims, :-1].sum(axis=0) - (n_items - 1)
        if (g[m, 1:] < lb - 1e-6).any():
            return False

    # Initial conditions
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

    Recomputes only rows corresponding to affected orders.
    """
    T = x_cand.shape[1]

    f_new = f_curr.copy()
    g_new = g_curr.copy()
    v_new = v_curr.copy()

    affected_orders: set[int] = set()

    # Identify affected orders
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


    # Recompute only affected rows
    time_range = np.arange(T)

    for m in affected_orders:
        ims = d.items_of_order[m]

        # first pick time of each item in order
        item_pick_times = first_one_idx[ims]

        # ignore unscheduled items (T means never picked)
        valid = item_pick_times[item_pick_times < T]

        if len(valid) == 0:
            t_start = T
            t_end = T
        else:
            t_start = valid.min()
            t_end = valid.max() + 1

        f_row = (time_range >= t_start).astype(np.float64)
        g_row = (time_range >= t_end).astype(np.float64)
        v_row = f_row - g_row

        # handle already opened orders
        order = d.orders[m]
        if order.order_id in d.opened_order_ids:
            v_row[:t_end] = 1.0
            f_row = v_row + g_row

        f_new[m] = f_row
        g_new[m] = g_row
        v_new[m] = v_row

    return f_new, g_new, v_new

### SOLUTION BUILDER 

def build_solution(x: np.ndarray, d) -> tuple:
    """
    Derive the full solution (f, g, v, y) from the picking matrix x.
    Expensive — call only for the final best solution, not during search.
    """
    n_travel = len(d.OptManager.travelling_arcs)
    M  = len(d.orders)
    T  = x.shape[1]

    f, g, v = _build_fgv(x, d)

    y = np.zeros(
        (len(d.from_RelPod_to_PodId), len(d.OptManager.all_arcs)),
        dtype=np.float64,
    )

    first_one_idx = (x == 0).sum(axis=1)

    # Helpers
    def add_idle_arcs(y, p_rel: int, loc: int, t_from: int, t_to: int) -> None:
        for t in range(t_from, t_to):
            for id_a in d.OptManager.outgoing_arc_idx.get((loc, t), []):
                if d.OptManager.all_arcs[id_a][1] == (loc, t + 1):
                    y[p_rel, id_a] = 1
                    break


    def find_arc_departing_after(src_loc, t_from, dst_loc, latest_arrival, d):
        """
        Return latest-arriving feasible direct arc:
        dep >= t_from
        arr <= latest_arrival
        """
        arcs = d.arc_lookup.get((src_loc, dst_loc), [])
        if not arcs:
            return None, None

        dep_times = [a[0] for a in arcs]
        idx = bisect_left(dep_times, t_from)
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

    infeasible_pods = []

    for p_rel, p_id in enumerate(d.from_RelPod_to_PodId):
        storage_loc = d.warehouse.pods[p_id].storage_location

        items_for_pod = {
            im: int(first_one_idx[im])
            for im in d.items_by_pod[p_id]
            if int(first_one_idx[im]) < d.OptManager.N_TIME
        }
        events = [] # events = [(ws_loc, t_target), ...]
        for im, t in sorted(items_for_pod.items(), key=lambda kv: kv[1]):
            _, m   = d.relevant_pairs_for_x[im]
            ws_loc = d.ws_positions[d.order_to_ws[m]]
            if not events or events[-1] != (ws_loc, t):
                events.append((ws_loc, t))

        # Pod departs from its storage location at time 0
        prev_loc, prev_t = storage_loc, 0

        for ws_loc, arrive_t in events:
            if prev_loc == ws_loc:
                # Pod already at workstation 
                # To avoid congestions at workstation, pod is sent to storage location if possible
                via_storage = False
                for id_a in d.OptManager.outgoing_arc_idx.get((prev_loc, prev_t), []):
                    arc = d.OptManager.all_arcs[id_a]
                    if arc[1][0] == storage_loc:
                        arc1   = arc
                        id_a1  = id_a
                        break
                id_a2, arc2 = find_arc_departing_after(storage_loc, arc1[1][1], ws_loc, arrive_t, d)
                if id_a2 is not None:
                    via_storage = True
                    y[p_rel, id_a1] = 1
                    add_idle_arcs(y, p_rel, storage_loc, arc1[1][1], arc2[0][1])
                    y[p_rel, id_a2] = 1
                    add_idle_arcs(y, p_rel, ws_loc, arc2[1][1], arrive_t)
                if not via_storage:
                    add_idle_arcs(y, p_rel, prev_loc, prev_t, arrive_t)
            else:
                # Pod not at workstation
                via_storage = False
                for id_a in d.OptManager.outgoing_arc_idx.get((prev_loc, prev_t), []):
                    arc = d.OptManager.all_arcs[id_a]
                    if arc[1][0] == storage_loc:
                        arc1  = arc
                        id_a1 = id_a
                        break
                id_a2, arc2 = find_arc_departing_after(storage_loc, arc1[1][1], ws_loc, arrive_t, d)
                if id_a2 is not None:
                    via_storage = True
                    y[p_rel, id_a1] = 1
                    add_idle_arcs(y, p_rel, storage_loc, arc1[1][1], arc2[0][1])
                    y[p_rel, id_a2] = 1
                    add_idle_arcs(y, p_rel, ws_loc, arc2[1][1], arrive_t)
                if not via_storage:
                    arc_id, arc = find_arc_departing_after(prev_loc, prev_t, ws_loc, arrive_t, d)
                    if arc_id is None:
                        infeasible_pods.append((p_rel, prev_loc, prev_t, ws_loc, arrive_t))
                    else:
                        add_idle_arcs(y, p_rel, prev_loc, prev_t, arc[0][1])
                        y[p_rel, arc_id] = 1

            if arrive_t + 1 < d.OptManager.N_TIME:
                add_idle_arcs(y, p_rel, ws_loc, arrive_t, arrive_t + 1)
                prev_loc, prev_t = ws_loc, arrive_t + 1
            else:
                prev_loc, prev_t = ws_loc, arrive_t

        # Path is ended with idle arcs at storage location
        # Return to storage
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

    # Build events for this pod only
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

def compute_objective(x: np.ndarray, f: np.ndarray, g: np.ndarray, d) -> float:
    T = x.shape[1]
    picking_reward = x[:, T-1].sum() 
    backlog_penalty   = float(sum(
        (d.current_time + t * d.OptManager.TIME_UNIT - d.arrival_times[m]) / d.OptManager.TIME_UNIT 
        * (1.0 - g[m, t])
        for m in range(len(d.orders))
        for t in range(T)
    ))
    return picking_reward - backlog_penalty/d.OptManager.N_TIME


def check_constraints(sol: tuple, d) -> tuple[bool, dict]:
    """Full constraint checker including y-based constraints."""
    x, f, g, v, y = sol
    T = x.shape[1]
    n_travel = len(d.OptManager.travelling_arcs)
    viols: dict = {}

    for w, order_ids in enumerate(d.orders_by_workstation):
        cap = v[list(order_ids), :].sum(axis=0)
        bad = np.where(cap > d.OptManager.CAP_WS + 1e-6)[0]
        if bad.size:
            viols.setdefault('EC13', []).append(
                {'w': w, 'times': bad.tolist(), 'values': cap[bad].tolist()})

    ec14 = []
    for w, order_ids in enumerate(d.orders_by_workstation):
        ws_p  = d.ws_positions[w]
        ims_w = [im for im, (_, m) in enumerate(d.relevant_pairs_for_x) if m in order_ids]
        for t in range(1, T):
            item_work = d.OptManager.DELTA_ITEM * (x[ims_w, t] - x[ims_w, t - 1]).sum()
            travel_arrivals = [
                a for a in d.OptManager.incoming_arc_idx.get((ws_p, t), [])
                if a < n_travel
            ]
            pod_arrivals = d.OptManager.DELTA_POD * y[:, travel_arrivals].sum()
            total = float(item_work + pod_arrivals)
            if total > 2*d.OptManager.TIME_UNIT + 1e-6:
                ec14.append({'w': w, 't': t, 'value': total})
    if ec14:
        viols['EC14'] = ec14

    ec15 = []
    for rel_p, p_id in enumerate(d.from_RelPod_to_PodId):
        stor     = d.warehouse.pods[p_id].storage_location
        out_arcs = d.OptManager.outgoing_arc_idx.get((stor, 0), [])
        flow_out = float(y[rel_p, out_arcs].sum())
        if not np.isclose(flow_out, 1.0):
            ec15.append({'pod': p_id, 'flow_out': flow_out})
    if ec15:
        viols['EC15'] = ec15

    ec16 = []
    for rel_p in range(len(d.from_RelPod_to_PodId)):
        for node in d.OptManager.nodes:
            if node[1] in (0, d.OptManager.N_TIME - 1):
                continue
            in_f  = float(y[rel_p, d.OptManager.incoming_arc_idx.get(node, [])].sum())
            out_f = float(y[rel_p, d.OptManager.outgoing_arc_idx.get(node, [])].sum())
            if not np.isclose(in_f - out_f, 0.0):
                ec16.append({'pod': rel_p, 'node': node, 'imbalance': in_f - out_f})
    if ec16:
        viols['EC16'] = ec16

    first_pick_time = (x == 0).sum(axis=1)
    ec18 = []
    for im, first_t in enumerate(first_pick_time):
        if first_t < d.OptManager.N_TIME:
            _, m   = d.relevant_pairs_for_x[im]
            ws_p   = d.ws_positions[d.order_to_ws[m]]
            rel_p  = d.from_PodId_to_RelPod[d.pod_of_item[im]]
            t_arcs = d.OptManager.incoming_arc_idx.get((ws_p, first_t), [])
            if float(y[rel_p, t_arcs].sum()) < 1e-6:
                ec18.append({'im': im, 't': first_t})
    if ec18:
        viols['EC18'] = ec18

    dx  = np.diff(x, axis=1)
    bad = np.argwhere(dx < -1e-6)
    if bad.size:
        viols['EC19'] = bad.tolist()

    poa = []
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        bad_ts = np.where(dx[im] > v[m, 1:] + 1e-6)[0] + 1
        if bad_ts.size:
            poa.append({'im': im, 'times': bad_ts.tolist()})
    if poa:
        viols['pick_only_if_active'] = poa

    bad = np.argwhere(np.abs(v - (f - g)) > 1e-6)
    if bad.size:
        viols['EC20'] = bad.tolist()

    ec21 = []
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        bad = np.where(f[m] < x[im] - 1e-6)[0]
        if bad.size:
            ec21.append({'im': im, 'm': m, 'times': bad.tolist()})
    if ec21:
        viols['EC21'] = ec21

    ec22 = [] ### DA COONTROLLARE
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        bad = np.where(g[m, 1:] > x[im, :-1] + 1e-6)[0] + 1
        if bad.size:
            ec22.append({'im': im, 'm': m, 'times': bad.tolist()})
    if ec22:
        viols['EC22'] = ec22

    if (np.diff(f, axis=1) < -1e-6).any():
        viols['f_monotonicity'] = True
    if (np.diff(g, axis=1) < -1e-6).any():
        viols['g_monotonicity'] = True

    bad = np.argwhere(v[:, 1:] - (v[:, :-1] - g[:, 1:]) < -1e-6)
    if bad.size:
        viols['continuity_v'] = bad.tolist()

    g_lb = [] ### DA CONTROLLARE
    for m in range(len(d.orders)):
        ims    = d.items_of_order[m]
        n_items = int(d.n_items_per_order[m])
        lb     = x[ims, :-1].sum(axis=0) - (n_items - 1)
        bad    = np.where(g[m, 1:] < lb - 1e-6)[0] + 1
        if bad.size:
            g_lb.append({'m': m, 'times': bad.tolist()})
    if g_lb:
        viols['g_lower_bound'] = g_lb

    for m, order in enumerate(d.orders):
        if order.order_id in d.opened_order_ids:
            if not np.isclose(float(v[m, 0]), 1.0):
                viols.setdefault('initial_cond', []).append({'m': m, 'v0': float(v[m, 0])})
        else:
            ims = d.items_of_order[m]
            bad = np.where(f[m] > x[ims, :].sum(axis=0) + 1e-6)[0]
            if bad.size:
                viols.setdefault('f_active_only_if_picked', []).append(
                    {'m': m, 'times': bad.tolist()})

    
    ### Computing simultaneosly active pods 
    T = d.OptManager.N_TIME
    n_pods = y.shape[0]
    active = np.zeros(T, dtype=int)

    storage_positions = set(d.OptManager._L)

    for rel_p in range(n_pods):
        pod_id = d.from_RelPod_to_PodId[rel_p]

        for a_idx, val in enumerate(y[rel_p]):
            if val < 0.5:
                continue

            src, dst = d.OptManager.all_arcs[a_idx]
            _, src_t = src
            dst_loc, dst_t = dst

            if not dst_loc == d.warehouse.pods[pod_id].storage_location:
                active[dst_t] += 1
                # intermediate timestep
                for t in range(src_t + 1, dst_t):
                    active[t] += 1

    bad_t = np.where(active > len(d.warehouse.robots))[0]
    if bad_t.size:
        viols['max_active_pods'] = {
            'times': bad_t.tolist(),
            'values': active[bad_t].tolist()
        }


    return len(viols) == 0, viols



""" ### INITIAL SOLUTION BUILDER

# Helpers

def _estimate_travel_time(storage_loc: int, ws_loc: int, d) -> int:
    n_travel = len(d.OptManager.travelling_arcs)
    for id_a in d.OptManager.outgoing_arc_idx.get((storage_loc, 0), []):
        if id_a >= n_travel:
            continue
        arc = d.OptManager.all_arcs[id_a]
        if arc[1][0] == ws_loc:
            return arc[1][1]
    return 1
 
 
def find_feasible_pick_time(
    candidate_t: int,
    p_rel: int,
    pod_busy: np.ndarray,
    robot_load: np.ndarray,
    n_robots: int,
    active_window: int,
    T: int,
    max_shift: int = 20,
) -> int | None:
    for shift in range(max_shift + 1):
        t = candidate_t + shift
        if t < 1 or t >= T:
            continue
        if pod_busy[p_rel, max(0, t - 1):min(T, t + 2)].any():
            continue
        
        t0 = max(0, t - active_window)
        t1 = min(T, t + active_window + 1)
        if robot_load[t0:t1].max() < n_robots:
            return t
    return None
 
 
def build_initial_x_v0(rng: np.random.Generator, d) -> np.ndarray:
    
    # Builds a feasible initial picking matrix x for the Stage-2 local search.
 
    # Guaranteed invariants
    # ---------------------
    # EC13  - at every time slot t, at most CAP_WS orders are active per
    #         workstation.  Enforced by non-overlapping time windows between
    #         consecutive batches.
 
    # Order atomicity - an order is written to x only if ALL its pods can be
    #         scheduled (commit-or-skip).  This prevents the "f[m]=1 but
    #         g[m]=0 forever" situation that caused cascading EC13 violations
    #         in the previous version.
 
    # EC14 proxy - pod_busy blocks consecutive picks on the same pod;
    #         robot_load proxies max_active_pods (EC14 / EC15 / EC16 are
    #         then refined by the local search).

    
    T    = d.OptManager.N_TIME
    n_im = len(d.relevant_pairs_for_x)
    CAP_WS = d.OptManager.CAP_WS
 
    x          = np.zeros((n_im, T), dtype=np.float64)
    n_robots   = len(d.warehouse.robots)
    robot_load = np.zeros(T, dtype=int)
    n_pods     = len(d.from_RelPod_to_PodId)
    pod_busy   = np.zeros((n_pods, T), dtype=bool)
 
    # Pre-compute one-way travel time for every (pod, workstation) pair
    pod_ws_travel: dict[tuple[int, int], int] = {}
    for p_id in d.from_RelPod_to_PodId:
        storage_loc = d.warehouse.pods[p_id].storage_location
        for w_id in range(len(d.warehouse.workstations)):
            ws_loc = d.ws_positions[w_id]
            pod_ws_travel[(p_id, w_id)] = _estimate_travel_time(storage_loc, ws_loc, d)
 
    for w_id, order_ids in enumerate(d.orders_by_workstation):
        ws = d.warehouse.workstations[w_id]

        if not order_ids:
            continue
    
        order_pods: dict[int, set] = {
            m: {d.pod_of_item[im] for im in d.items_of_order[m]}
            for m in order_ids
        }
    
        # Count how many other orders share at least one pod with each order.
        # Orders with more pod-sharing partners produce denser batches (fewer robot trips).
        pod_to_orders: dict = {}
        for m, pods in order_pods.items():
            for p in pods:
                pod_to_orders.setdefault(p, set()).add(m)
    
        shared_count: dict[int, int] = {
            m: len({o for p in order_pods[m] for o in pod_to_orders[p]} - {m})
            for m in order_ids
        }
    
        # Seed order: most-shared first -> denser batches.
        # Tie-break by earliest_t so time-urgent orders seed early batches.
        seed_order = sorted(
            order_ids,
            key=lambda m: (
                -shared_count[m],
                max((int(d.earliest_t[im]) for im in d.items_of_order[m]), default=0),
                rng.random(),
            ),
        )
    
        # Fill order: earliest_t first so each batch is time-feasible.
        fill_order = sorted(
            order_ids,
            key=lambda m: (
                - d.orders[m].arrival_time + rng.integers(-30,30)
            ),
        )
    
        assigned: set[int] = set()
        batches: list[list[int]] = []
        for seed in seed_order:
            if seed in assigned:
                continue

            batch = [seed]
            assigned.add(seed)
            batch_pods = set(order_pods[seed])

            for m in fill_order:
                if m in assigned or len(batch) >= CAP_WS:
                    continue

                if order_pods[m] & batch_pods:
                    batch.append(m)
                    batch_pods |= order_pods[m]
                    assigned.add(m)
            batches.append(batch)
    

        # Merge underfull batches: fill existing ones before opening new batches
        pool = [m for b in batches if len(b) < CAP_WS for m in b]
        batches = [b for b in batches if len(b) == CAP_WS] 
        rng.shuffle(pool)
        for m in pool:
            placed = False
            for idx in rng.permutation(len(batches)):
                if len(batches[idx]) < CAP_WS :
                    batches[idx].append(m)
                    placed = True
                    break
            if not placed:
                batches.append([m])


        # Build the batch list:
        #   1) Opened orders: hard chunks of CAP_WS (already active at t=0)
        #   2) New orders:    pod-sharing greedy batches, <= CAP_WS each
        opened  = [m for m in order_ids if d.orders[m].order_id in ws.opened_orders]
        if len(opened) > 0:
            batches.insert(0, opened)
 
 
        not_before_t = 0   # first pick of the next batch must be >= this
        not_commited = []
        for id_b, batch in enumerate(batches):
            if not batch:
                continue

            # Collect unscheduled items grouped by pod, per order 
            order_pod_items: dict[int, dict[int, list[int]]] = {}
            for m in batch:
                pod_map: dict[int, list[int]] = {}
                for im in d.items_of_order[m]:
                    if x[im, -1] < 0.5:                    # not yet scheduled
                        pod_map.setdefault(d.pod_of_item[im], []).append(im)
                if pod_map:
                    order_pod_items[m] = pod_map
        
            if not order_pod_items:
                continue
        
            # Earliest feasible pick slot per pod (max across all its items)
            # Also count how many committable orders depend on each pod: schedule
            # the most-critical pods first so a failure skips fewer orders.
            pod_earliest: dict[int, int] = {}
            pod_order_count: dict[int, int] = {}
            for m, pod_map in order_pod_items.items():
                for p_id, ims in pod_map.items():
                    e = max(max(int(d.earliest_t[im]) for im in ims), 1)
                    pod_earliest[p_id] = max(pod_earliest.get(p_id, 0), e)
                    pod_order_count[p_id] = pod_order_count.get(p_id, 0) + 1
        
            # Tentative scheduling on temporary copies
            tentative_picks: dict[int, int] = {}
            t_pod_busy = pod_busy.copy()
            t_robot_load = robot_load.copy()
            next_t = not_before_t
        
            # Sort: most-shared pods first (failure costs more orders), then by earliest_t
            for p_id in sorted(pod_earliest, key=lambda p: (-pod_order_count[p], pod_earliest[p])):
                travel = pod_ws_travel.get((p_id, w_id), 1)
                # CRITICAL: candidate_t >= not_before_t ensures no pick from this
                # batch ever falls inside the previous batch's time window -> EC13 invariant
                candidate_t = max(pod_earliest[p_id], next_t, not_before_t)
                p_rel = d.from_PodId_to_RelPod[p_id]
        
                t_pick = find_feasible_pick_time(
                    candidate_t, p_rel, t_pod_busy, t_robot_load,
                    n_robots, travel, T)
                if t_pick is None:
                    continue
        
                tentative_picks[p_id] = t_pick
                t_pod_busy[p_rel, t_pick] = True
                t0, t1 = max(0, t_pick - travel), min(T, t_pick + travel + 1)
                t_robot_load[t0:t1] += 1
                next_t = t_pick + 1
        
            # Commit-or-skip per order
            # An order is committable only if ALL its pods found a slot.
            # Pods shared between committable orders are committed only once.
            if id_b < len(batches):
                committable = [
                    m for m, pod_map in order_pod_items.items()
                    if all(p_id in tentative_picks for p_id in pod_map)
                ]
                not_commited += [m for m in order_pod_items.keys() if m not in committable]

            if not committable:
                continue     # no progress, not_before_t unchanged
        
            committed_pods: set[int] = set()
            for m in committable:
                committed_pods |= set(order_pod_items[m].keys())
        
            batch_last_pick = not_before_t
        
            for p_id in committed_pods:
                t_pick = tentative_picks.get(p_id)
                if t_pick:
                    travel = pod_ws_travel.get((p_id, w_id), 1)
                    p_rel = d.from_PodId_to_RelPod[p_id]
            
                    # Commit to the real tracking structures
                    pod_busy[p_rel, t_pick] = True
                    t0, t1 = max(0, t_pick - travel), min(T, t_pick + travel + 1)
                    robot_load[t0:t1] += 1
                    batch_last_pick = max(batch_last_pick, t_pick)
            
                    # Write x only for items belonging to committable orders
                    for m in committable:
                        for im in order_pod_items[m].get(p_id, []):
                            x[im, t_pick:] = 1.0
        
            # Update not_before_t
            batch_end_t = max(
                max(tentative_picks.get(p_id, 0) for p_id in order_pod_items[m])
                for m in committable
            )
            not_before_t = batch_end_t + 1


        ### RETRY SCHEDULING NOT_COMMITED ORDER

        for m in not_commited:
            pod_map = {}
            for im in d.items_of_order[m]:
                if x[im, -1] < 0.5:
                    pod_map.setdefault(d.pod_of_item[im], []).append(im)

            if not pod_map:
                continue

            tentative = {}
            next_t = not_before_t
            feasible = True

            # try scheduling pods of this order sequentially
            for p_id, ims in sorted(
                pod_map.items(),
                key=lambda kv: max(int(d.earliest_t[im]) for im in kv[1])
            ):
                earliest = max(
                    max(int(d.earliest_t[im]) for im in ims),
                    next_t,
                    not_before_t,
                    1,
                )

                travel = pod_ws_travel.get((p_id, w_id), 1)
                p_rel = d.from_PodId_to_RelPod[p_id]
                t_pick = find_feasible_pick_time(
                    earliest, p_rel, pod_busy, 
                    robot_load, n_robots, travel, T)

                if t_pick is None:
                    feasible = False
                    break

                tentative[p_id] = t_pick
                next_t = t_pick + 1

            if not feasible:
                continue

            # commit order
            last_pick = not_before_t - 1
            for p_id, t_pick in tentative.items():
                travel = pod_ws_travel.get((p_id, w_id), 1)
                p_rel = d.from_PodId_to_RelPod[p_id]
                pod_busy[p_rel, t_pick] = True
                t0, t1 = max(0, t_pick - travel), min(T, t_pick + travel + 1)
                robot_load[t0:t1] += 1
                last_pick = max(last_pick, t_pick)
                for im in pod_map[p_id]:
                    x[im, t_pick:] = 1.0

            not_before_t = last_pick + 2

    return x """


### NEIGHBHOURS GENERATORS

def _make_move_1(x, ims, variation, first_one_idx, T):
    """Shift k item simultaneamente."""
    x_cand = x.copy()
    for im in ims:
        t_new = int(first_one_idx[im]) + variation
        if t_new < 0 or t_new >= T:
            return None
        x_cand[im, :] = 0
        x_cand[im, t_new:] = 1
    return x_cand


def _make_move_2(x, ims, variation, first_one_idx, T):
    x_cand = x.copy()
    for im in ims:
        t_new = int(first_one_idx[im]) + variation
        if t_new < 1 or t_new >= T:
            return None
        x_cand[im, :] = 0
        x_cand[im, t_new:] = 1
    return x_cand


def _make_move_3(x, ims1, ims2, first_one_idx, T):
    delta = (min(int(first_one_idx[im]) for im in ims2)
             - min(int(first_one_idx[im]) for im in ims1))
    if delta == 0:
        return None
    new_t1 = [int(first_one_idx[im]) + delta for im in ims1]
    new_t2 = [int(first_one_idx[im]) - delta for im in ims2]
    if any(t < 1 or t >= T for t in new_t1 + new_t2):
        return None
    x_cand = x.copy()
    for im, t in zip(ims1, new_t1):
        x_cand[im, :] = 0; x_cand[im, t:] = 1
    for im, t in zip(ims2, new_t2):
        x_cand[im, :] = 0; x_cand[im, t:] = 1
    return x_cand


### LOCAL SEARCH

def local_search_stage2(d: Stage2Data) -> tuple:
    """
    Faster local search:
      1. During search, NEVER call build_solution (builds y — expensive).
         Instead use _fast_evaluate: build f/g/v from x + check x-only
         constraints + compute objective. All pure numpy.
      2. build_solution(y) is called exactly ONCE at the end on the best x.
      3. x_current is updated only at the END of each iteration (best-in-iter),
         not inside the move loop (was a latent bug causing inconsistent
         first_one_idx across moves in the same iteration).
      4. visited_x uses a fixed-size deque to bound memory without the
         arbitrary dict.pop() that could remove the current solution.
    """
    from collections import deque

    rng = np.random.default_rng(seed=42)

    im_by_order: dict[int, list[int]] = {}
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        im_by_order.setdefault(m, []).append(im)

    # Initial solution 
    print("\n[ls_stage2] Building initial solution ...")
    logging.info("\n[ls_stage2] Building initial solution ...")
    x_current = build_initial_x(rng, d)
    _, f0, g0, v0, y0 = build_solution(x_current, d)
    feasible, viols = check_constraints((x_current, f0, g0, v0, y0), d)
    while not feasible:
        print(f"[ls_stage2] violated = {list(viols.keys())}")
        logging.info("[ls_stage2] violated = %s", list(viols.keys()))
        for k, v in viols.items():
            print(f"  {k}: {v[:3] if isinstance(v, list) else v}")
        x_current = build_initial_x(rng, d)
        _, f0, g0, v0, y0 = build_solution(x_current, d)
        feasible, viols = check_constraints((x_current, f0, g0, v0, y0), d)

    best_x   = x_current.copy()
    best_sol = (best_x, f0, g0, v0, y0)
    best_obj = compute_objective(x_current, f0, g0, d)
    print(f"[ls_stage2] Feasible initial solution: obj = {best_obj:.4f}")
    logging.info("[ls_stage2] Feasible initial solution: obj = %.4f", best_obj)

    T = x_current.shape[1]


    ### MAIN LOOP

    am_I_stuck                 = False
    cont                       = 1
    iter_without_improvement   = 0
    max_iter_without_improvement = 3
    MAX_ITER = 45
    MAX_NEIGH = 150

    print("[ls_stage2] Exploring neighbours ...")

    while not am_I_stuck and cont <= MAX_ITER:
        first_one_idx = np.argmax(best_x > 0.5, axis=1)
        first_one_idx[best_x[:, -1] == 0] = T   # recompute once per iter
        improved      = False

        best_obj_in_iter = -np.inf
        best_x_in_iter   = None
        second_best_obj_in_iter = -np.inf
        second_best_x_in_iter   = None

        # Build move list 
        moves = [[], [], []]

        if iter_without_improvement > 0:
            # Smaller moves
            for im in range(x_current.shape[0]):
                moves[0].append(('item',  im, -1))
                moves[0].append(('item',  im, -2))
                for direction in [-1,-2]:
                    sampled = rng.choice(item_ids, size=min(len(item_ids), 20), replace=False)
                    for i in range(0, len(sampled) - 1, 2):
                        moves[0].append(('multi_item', (sampled[i], sampled[i+1]), direction))
                        if i+2 < len(sampled)-1:
                            moves[0].append(('multi_item', (sampled[i], sampled[i+1], sampled[i+2]), direction))
        else:
            for im in range(x_current.shape[0]):
                moves[0].append(('item', im, -2))
                moves[0].append(('item', im, -4))
                moves[0].append(('item', im, -6))
                moves[0].append(('item', im, -8))

            item_ids = list(range(x_current.shape[0]))
            for direction in [-2, -4, -5]:
                sampled = rng.choice(item_ids, size=min(len(item_ids), 20), replace=False)
                for i in range(0, len(sampled) - 1, 2):
                    moves[0].append(('multi_item', (sampled[i], sampled[i+1]), direction))
                    if i+2 < len(sampled)-1:
                        moves[0].append(('multi_item', (sampled[i], sampled[i+1], sampled[i+2]), direction))

        for m in range(len(d.orders)):
            moves[1].append(('order', m, -1))
            moves[1].append(('order', m, -2))
            moves[1].append(('order', m, -4))
        for order_ids in d.orders_by_workstation:
            order_list = list(order_ids)
            for i1, m1 in enumerate(order_list):
                for m2 in order_list[i1 + 1:]:
                    moves[2].append(('swap', m1, m2))

        # Randomly reducing the neighborhood
        total = sum(len(m) for m in moves)
        if total > MAX_NEIGH:
            for i, p in enumerate([0.5, 0.3, 0.2]):
                size = min(len(moves[i]), int(np.ceil(MAX_NEIGH * p)))
                if size and len(moves[i]) > size:
                    idxs = rng.choice(len(moves[i]), size=size, replace=False)
                    moves[i] = [moves[i][j] for j in idxs]

        all_moves = moves[0] + moves[1] + moves[2]

        for move in all_moves:
            first_one_idx_cand = first_one_idx.copy()

            # Build candidate x
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
                    x_current, im_by_order.get(int(m), []), int(direction), first_one_idx, T)
            else:
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

            _, f_curr, g_curr, v_curr, _ = best_sol
            f_cand, g_cand, v_cand = _fast_update_fgv_from_move(
                                            x_cand,
                                            f_curr,
                                            g_curr,
                                            v_curr,
                                            move,
                                            im_by_order,
                                            first_one_idx_cand,
                                            d,
                                        )

            # Full x-only evaluation (no y built here —> speedup)
            if _check_x_fast(x_cand, f_cand, g_cand, v_cand, d):
                obj = compute_objective(x_cand, f_cand, g_cand, d)
                if obj is not None and obj > best_obj_in_iter:
                    best_obj_in_iter = obj
                    best_x_in_iter = x_cand
                    best_move  = move   
                    best_f_in_iter, best_g_in_iter, best_v_in_iter = f_cand, g_cand, v_cand
                elif obj is not None and obj > second_best_obj_in_iter:
                    second_best_obj_in_iter = obj
                    second_best_x_in_iter = x_cand
                    second_best_move  = move   
                    s_best_f_in_iter, s_best_g_in_iter, s_best_v_in_iter = f_cand, g_cand, v_cand
                

        sol_num = 1
        while best_x_in_iter is not None and sol_num <= 2:
            x_current = best_x_in_iter
            if best_obj_in_iter > best_obj -1e-10:

                # I check the full feasibility
                if best_move[0] in ('item', 'order', 'multi_item', 'swap'):
                    # Identify only the affected pods
                    if best_move[0] == 'item':
                        _, best_im, _ = best_move
                        affected_pods = {d.pod_of_item[int(best_im)]}
                    elif best_move[0] == 'multi_item':
                        _, best_ims, _ = best_move
                        affected_pods = {d.pod_of_item[int(im)] for im in best_ims}
                    elif best_move[0] == 'order':
                        _, best_m, _ = best_move
                        affected_pods = {d.pod_of_item[im] for im in im_by_order[int(best_m)]}
                    elif best_move[0] == 'swap':
                        _, best_m1, best_m2 = best_move
                        affected_pods = {d.pod_of_item[im] for im in im_by_order[int(best_m1)]}.union({d.pod_of_item[im] for im in im_by_order[int(best_m2)]})

                    # Rebuild only the affected pod rows
                    y_new = best_sol[4].copy()
                    for p_id in affected_pods:
                        p_rel = d.from_PodId_to_RelPod[p_id]
                        y_new[p_rel] = _rebuild_pod_row(p_rel, p_id, best_x_in_iter, d)

                    sol_curr = (best_x_in_iter, best_f_in_iter, best_g_in_iter, best_v_in_iter, y_new)
                
                else:
                    # Fallback -> y is fully rebuilt
                    sol_curr = build_solution(best_x_in_iter, d)   

                feasible, _ = check_constraints(sol_curr, d)
                if feasible:
                    improved = best_obj_in_iter > best_obj
                    best_obj = best_obj_in_iter
                    best_x = best_x_in_iter
                    best_sol = sol_curr
                    print(f"[ls_stage2] Iter {cont} : Improved with move {best_move} → {best_obj:.4f}")
                    logging.info("[ls_stage2] Iter %i : Improved with move %s → %.4f",
                                 cont, best_move, best_obj)
                    sol_num = 3
                else:
                    if second_best_x_in_iter is None:
                        break
                    best_x_in_iter = second_best_x_in_iter
                    best_f_in_iter = s_best_f_in_iter
                    best_g_in_iter = s_best_g_in_iter
                    best_v_in_iter = s_best_v_in_iter
                    best_move = second_best_move
                    best_obj_in_iter = second_best_obj_in_iter
                    sol_num += 1
            else:
                break


        if improved:
            iter_without_improvement = 0
        else:
            iter_without_improvement += 1
            if iter_without_improvement >= max_iter_without_improvement:
                am_I_stuck = True
                print(f"[ls_stage2] Converged after "
                      f"{max_iter_without_improvement} iters without improvement "
                      f"at {best_obj:.4f}")
                logging.info("[ls_stage2] Converged after %i iters without improvement", max_iter_without_improvement)
            else:
                print(f"[ls_stage2] Iter {cont} : No improvement "
                      f"({iter_without_improvement}/{max_iter_without_improvement})")
                logging.info("[ls_stage2] Iter %i No improvement", cont)

        cont += 1

    print("[ls_stage2] Local serch ended.")
    
    return best_sol