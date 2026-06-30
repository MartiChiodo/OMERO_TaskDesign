from __future__ import annotations
import numpy as np
from bisect import bisect_left
import logging

from .stage2_data import Stage2Data
from .build_initial_x_v1 import build_initial_x


### Fast helpers (no y needed)

def _build_fgv(x: np.ndarray, d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute f, g, v from x only, no pod routing.
    Identical logic to the corresponding block in build_solution.
    """
    T      = x.shape[1]
    M      = len(d.orders)
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
    Returns True iff all x-only constraints hold.
    """
    T  = x.shape[1]
    M  = len(d.orders)
    dx = np.diff(x, axis=1)

    if (dx < -1e-6).any():
        return False
    if x[:, 0].any():
        return False
    for order_ids in d.orders_by_workstation:
        if (v[list(order_ids), :].sum(axis=0) > d.OptManager.CAP_WS + 1e-6).any():
            return False
    if (np.abs(v - (f - g)) > 1e-6).any():
        return False
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        if (f[m] < x[im] - 1e-6).any():
            return False
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        if (g[m, 1:] > x[im, :-1] + 1e-6).any():
            return False
    if (np.diff(f, axis=1) < -1e-6).any():
        return False
    if (np.diff(g, axis=1) < -1e-6).any():
        return False
    for m in range(M):
        ims     = d.items_of_order[m]
        n_items = int(d.n_items_per_order[m])
        lb      = x[ims, :-1].sum(axis=0) - (n_items - 1)
        if (g[m, 1:] < lb - 1e-6).any():
            return False
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

def build_solution(x: np.ndarray, d) -> tuple:
    """
    Derive the full solution (f, g, v, y) from the picking matrix x.
    Expensive — call only for the final best solution, not during search.
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
                        # Evento irraggiungibile: resta fermo, EC18 violato
                        # ma EC16 preservato (nessun arco inserito)
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

def compute_objective(x: np.ndarray, f: np.ndarray, g: np.ndarray, d) -> float:
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
    """Full constraint checker including y-based constraints."""
    x, f, g, v, y = sol
    T        = x.shape[1]
    n_travel = len(d.OptManager.travelling_arcs)
    viols: dict = {}

    # EC13
    for w, order_ids in enumerate(d.orders_by_workstation):
        cap = v[list(order_ids), :].sum(axis=0)
        bad = np.where(cap > d.OptManager.CAP_WS + 1e-6)[0]
        if bad.size:
            viols.setdefault('EC13', []).append(
                {'w': w, 'times': bad.tolist(), 'values': cap[bad].tolist()}
            )

    # EC14
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
            if total > 2 * d.OptManager.TIME_UNIT + 1e-6:
                ec14.append({'w': w, 't': t, 'value': total})
    if ec14:
        viols['EC14'] = ec14

    # EC15
    ec15 = []
    for rel_p, p_id in enumerate(d.from_RelPod_to_PodId):
        stor     = d.warehouse.pods[p_id].storage_location
        out_arcs = d.OptManager.outgoing_arc_idx.get((stor, 0), [])
        flow_out = float(y[rel_p, out_arcs].sum())
        if not np.isclose(flow_out, 1.0):
            ec15.append({'pod': p_id, 'flow_out': flow_out})
    if ec15:
        viols['EC15'] = ec15

    # EC16
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

    # EC18
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

    # EC19
    dx  = np.diff(x, axis=1)
    bad = np.argwhere(dx < -1e-6)
    if bad.size:
        viols['EC19'] = bad.tolist()

    # pick_only_if_active
    poa = []
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        bad_ts = np.where(dx[im] > v[m, 1:] + 1e-6)[0] + 1
        if bad_ts.size:
            poa.append({'im': im, 'times': bad_ts.tolist()})
    if poa:
        viols['pick_only_if_active'] = poa

    # EC20
    bad = np.argwhere(np.abs(v - (f - g)) > 1e-6)
    if bad.size:
        viols['EC20'] = bad.tolist()

    # EC21
    ec21 = []
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        bad = np.where(f[m] < x[im] - 1e-6)[0]
        if bad.size:
            ec21.append({'im': im, 'm': m, 'times': bad.tolist()})
    if ec21:
        viols['EC21'] = ec21

    # EC22
    ec22 = []
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        bad = np.where(g[m, 1:] > x[im, :-1] + 1e-6)[0] + 1
        if bad.size:
            ec22.append({'im': im, 'm': m, 'times': bad.tolist()})
    if ec22:
        viols['EC22'] = ec22

    # f/g monotonicity
    if (np.diff(f, axis=1) < -1e-6).any():
        viols['f_monotonicity'] = True
    if (np.diff(g, axis=1) < -1e-6).any():
        viols['g_monotonicity'] = True

    # continuity_v
    bad = np.argwhere(v[:, 1:] - (v[:, :-1] - g[:, 1:]) < -1e-6)
    if bad.size:
        viols['continuity_v'] = bad.tolist()

    # g_lower_bound
    g_lb = []
    for m in range(len(d.orders)):
        ims     = d.items_of_order[m]
        n_items = int(d.n_items_per_order[m])
        lb      = x[ims, :-1].sum(axis=0) - (n_items - 1)
        bad     = np.where(g[m, 1:] < lb - 1e-6)[0] + 1
        if bad.size:
            g_lb.append({'m': m, 'times': bad.tolist()})
    if g_lb:
        viols['g_lower_bound'] = g_lb

    # initial_cond / f_active_only_if_picked
    for m, order in enumerate(d.orders):
        if order.order_id in d.opened_order_ids:
            if not np.isclose(float(v[m, 0]), 1.0):
                viols.setdefault('initial_cond', []).append(
                    {'m': m, 'v0': float(v[m, 0])}
                )
        else:
            ims = d.items_of_order[m]
            bad = np.where(f[m] > x[ims, :].sum(axis=0) + 1e-6)[0]
            if bad.size:
                viols.setdefault('f_active_only_if_picked', []).append(
                    {'m': m, 'times': bad.tolist()}
                )

    # max_active_pods
    # Un pod occupa un robot durante [src_t, dst_t) se parte da una
    # location che non e' il suo storage (cioe' e' fuori storage).
    n_pods = y.shape[0]
    active = np.zeros(T, dtype=int)

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

    return len(viols) == 0, viols


### NEIGHBOUR GENERATORS

def _make_move_1(x, ims, variation, first_one_idx, T):
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

    # ------------------------------------------------------------------ #
    # Initial solution
    # ------------------------------------------------------------------ #
    print("\n[ls_stage2] Building initial solution ...")
    logging.info("[ls_stage2] Building initial solution ...")

    x_current = build_initial_x(rng, d)
    _, f0, g0, v0, y0 = build_solution(x_current, d)
    feasible, viols = check_constraints((x_current, f0, g0, v0, y0), d)

    max_attempts = 10
    attempt = 1

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
        feasible, viols = check_constraints((x_current, f0, g0, v0, y0), d)

    best_x   = x_current.copy()
    best_sol = (best_x, f0, g0, v0, y0)
    best_obj = compute_objective(x_current, f0, g0, d)
    print(f"[ls_stage2] Feasible initial solution: obj = {best_obj:.4f}")
    logging.info("[ls_stage2] Feasible initial solution: obj = %.4f", best_obj)
    
    T = x_current.shape[1]
    item_ids = list(range(x_current.shape[0]))

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    am_I_stuck                   = False
    cont                         = 1
    iter_without_improvement     = 0
    max_iter_without_improvement = 5
    MAX_ITER  = 150
    MAX_NEIGH = 300

    print("[ls_stage2] Exploring neighbours ...")

    while not am_I_stuck and cont <= MAX_ITER:
        first_one_idx = np.argmax(best_x > 0.5, axis=1)
        first_one_idx[best_x[:, -1] == 0] = T
        improved = False

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
        moves = [[], [], []]

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

            _, f_curr, g_curr, v_curr, _ = best_sol
            f_cand, g_cand, v_cand = _fast_update_fgv_from_move(
                x_cand, f_curr, g_curr, v_curr,
                move, im_by_order, first_one_idx_cand, d,
            )

            if _check_x_fast(x_cand, f_cand, g_cand, v_cand, d):
                obj = compute_objective(x_cand, f_cand, g_cand, d)
                if obj is not None and obj > best_obj_in_iter:
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
        sol_num = 1
        while best_x_in_iter is not None and sol_num <= 2:
            x_current = best_x_in_iter

            if best_obj_in_iter > best_obj - 1e-10:
                if best_move[0] in ('item', 'order', 'multi_item', 'swap'):
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

                feasible, _ = check_constraints(sol_curr, d)
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
                    sol_num = 3
                else:
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

        # ---- Convergence check --------------------------------------- #
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