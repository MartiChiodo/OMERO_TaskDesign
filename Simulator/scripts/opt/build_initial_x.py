from __future__ import annotations
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_travel_time(storage_loc: int, ws_loc: int, d) -> int:
    n_travel = len(d.OptManager.travelling_arcs)
    for id_a in d.OptManager.outgoing_arc_idx.get((storage_loc, 0), []):
        if id_a >= n_travel:
            continue
        arc = d.OptManager.all_arcs[id_a]
        if arc[1][0] == ws_loc:
            return arc[1][1]
    return 1


def _travel_time_between(src_loc: int, dst_loc: int, d) -> int:
    """Travel time between any two locations using arc_lookup."""
    if src_loc == dst_loc:
        return 0
    arcs = d.arc_lookup.get((src_loc, dst_loc), [])
    if arcs:
        dep_t, arr_t, _, _ = arcs[0]
        return max(1, arr_t - dep_t)
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def build_initial_x(rng: np.random.Generator, d) -> np.ndarray:
    """
    Builds a feasible initial picking matrix x.

    Key improvements over previous version
    ───────────────────────────────────────
    1. NO pod_busy copy per batch.
       Tentative picks are tracked in a lightweight dict[p_rel -> set[t]].
       Only robot_load is snapshotted (O(T)) and restored on rollback.

    2. Better batching.
       Candidates are scored at each fill step by:
           (intersection DESC, new_pods ASC)
       using numpy boolean matrix ops, so we always add the order that
       maximises actual pod sharing with the current batch.

    3. Pod chaining.
       candidate_t and robot_load window use the pod's last known position
       (workstation or storage), not always storage.
    """
    T        = d.OptManager.N_TIME
    n_im     = len(d.relevant_pairs_for_x)
    CAP_WS   = d.OptManager.CAP_WS
    n_pods   = len(d.from_RelPod_to_PodId)
    n_robots = len(d.warehouse.robots)
    n_ws     = len(d.warehouse.workstations)
    MAX_SHIFT = 20

    x          = np.zeros((n_im, T), dtype=np.float64)
    robot_load = np.zeros(T, dtype=np.int32)
    pod_busy   = np.full((n_pods, T), -1, dtype=np.int32)

    # pod_state[p_rel] = (ws_id | None, last_pick_t)
    # None  → pod is at storage (or not yet used)
    pod_state_loc = [None] * n_pods   # int workstation index, or None
    pod_state_t   = [0]    * n_pods   # time of last committed pick

    # ── Travel time tables ────────────────────────────────────────────────────
    pod_ws_travel: dict[tuple, int] = {}
    for p_id in d.from_RelPod_to_PodId:
        sloc = d.warehouse.pods[p_id].storage_location
        for w in range(n_ws):
            pod_ws_travel[p_id, w] = _estimate_travel_time(sloc, d.ws_positions[w], d)

    ws_ws_travel = np.zeros((n_ws, n_ws), dtype=np.int32)
    for wf in range(n_ws):
        for wt in range(n_ws):
            if wf != wt:
                ws_ws_travel[wf, wt] = _travel_time_between(
                    d.ws_positions[wf], d.ws_positions[wt], d
                )

    # ── Closures for pod position ─────────────────────────────────────────────
    def travel_to(p_id: int, p_rel: int, w_id: int) -> int:
        loc = pod_state_loc[p_rel]
        if loc is None:  return pod_ws_travel.get((p_id, w_id), 1) + 1
        if loc == w_id:  return 1
        return int(ws_ws_travel[loc, w_id]) + 1

    def min_candidate(p_id: int, p_rel: int, w_id: int, pod_e: int, floor: int) -> int:
        loc    = pod_state_loc[p_rel]
        travel = travel_to(p_id, p_rel, w_id)
        # earliest the pod can physically arrive at this ws
        arrival = (pod_state_t[p_rel] + travel) if loc is not None else travel
        return max(pod_e, arrival, floor, 1)

    # ── Slot finder — NO array copy ───────────────────────────────────────────
    def find_slot(p_rel: int, start: int, travel: int,
                  locked: dict[int, set]) -> int | None:
        """
        Find first t >= start where:
          - pod_busy[p_rel, t-1:t+2] is all free (committed + tentative)
          - robot_load[t-travel : t+travel+1].max() < n_robots
        """
        ts = locked.get(p_rel, frozenset())
        pb = pod_busy[p_rel]           # view, no copy
        rl = robot_load

        for shift in range(MAX_SHIFT + 1):
            t = start + shift
            if t < 1 or t >= T:
                continue

            # pod-free window check (committed array + tentative set)
            lo, hi = max(0, t - 1), min(T - 1, t + 1)
            if pb[lo:hi + 1].max() >= 0:   # any committed pick nearby
                continue
            # check tentative locks in the same window
            if any(tt in ts for tt in range(lo, hi + 1)):
                continue

            # robot load window
            if rl[max(0, t - travel):min(T, t + travel + 1)].max() < n_robots:
                return t
        return None

    # =========================================================================
    # Main loop
    # =========================================================================
    for w_id, order_ids in enumerate(d.orders_by_workstation):
        ws = d.warehouse.workstations[w_id]
        if not order_ids:
            continue

        orders_list = list(order_ids)
        n_ord       = len(orders_list)
        ord_row     = {m: i for i, m in enumerate(orders_list)}

        order_pods = {
            m: frozenset(d.pod_of_item[im] for im in d.items_of_order[m])
            for m in orders_list
        }

        # ── Binary matrix: orders × pods (ws-local pod universe) ─────────────
        ws_pod_list = sorted({p for pods in order_pods.values() for p in pods})
        n_wp        = len(ws_pod_list)
        pod_col     = {p: j for j, p in enumerate(ws_pod_list)}

        A = np.zeros((n_ord, n_wp), dtype=np.bool_)
        for m, pods in order_pods.items():
            for p in pods:
                A[ord_row[m], pod_col[p]] = True

        # ── Seed order: most inter-connected orders first ─────────────────────
        # (more shared pods → denser batches as seeds)
        pod_to_ords: dict[int, list] = defaultdict(list)
        for m, pods in order_pods.items():
            for p in pods:
                pod_to_ords[p].append(m)

        shared_count = {
            m: len({o for p in order_pods[m] for o in pod_to_ords[p]} - {m})
            for m in orders_list
        }
        seed_order = sorted(
            orders_list,
            key=lambda m: (-shared_count[m], -d.orders[m].arrival_time, rng.random()),
        )

        # ── Greedy batching — score by (intersection DESC, new_pods ASC) ─────
        #
        # At each fill step we compute, for ALL remaining orders at once:
        #   inter   = (A_free & batch_vec).sum(axis=1)
        #   new_p   = (A_free & ~batch_vec).sum(axis=1)
        # and pick the candidate with the best (inter DESC, new_p ASC).
        # This is a full numpy op, no Python per-order loop.
        assigned   = np.zeros(n_ord, dtype=bool)
        batches:   list[list[int]] = []
        batch_vecs: list[np.ndarray] = []   # kept in sync for fast merge step

        for seed in seed_order:
            si = ord_row[seed]
            if assigned[si]:
                continue

            batch     = [seed]
            assigned[si] = True
            bvec      = A[si].copy()

            while len(batch) < CAP_WS:
                free_idx = np.where(~assigned)[0]
                if free_idx.size == 0:
                    break

                free_A = A[free_idx]                       # (n_free, n_wp)
                inter  = (free_A & bvec).sum(axis=1)       # (n_free,)

                # Only consider orders that share at least one pod
                share_mask = inter > 0
                if not share_mask.any():
                    break

                new_p = (free_A[share_mask] & ~bvec).sum(axis=1)
                # lexsort: primary = new_p ASC, secondary = inter DESC
                share_free  = free_idx[share_mask]
                best_local  = np.lexsort((new_p, -inter[share_mask]))[0]
                best_global = share_free[best_local]

                batch.append(orders_list[best_global])
                assigned[best_global] = True
                bvec |= A[best_global]

            batches.append(batch)
            batch_vecs.append(bvec)

        # ── Merge underfull batches (best-fit by intersection) ────────────────
        pool    = [m for b in batches if len(b) < CAP_WS for m in b]
        keep    = [(b, bv) for b, bv in zip(batches, batch_vecs) if len(b) == CAP_WS]
        batches     = [x for x, _ in keep]
        batch_vecs  = [v for _, v in keep]

        for m in (pool[i] for i in rng.permutation(len(pool))):
            mi = ord_row[m]
            best_idx, best_score = None, (-1, float('inf'))
            for bi, (b, bv) in enumerate(zip(batches, batch_vecs)):
                if len(b) >= CAP_WS:
                    continue
                inter = int((A[mi] & bv).sum())
                new_p = int((A[mi] & ~bv).sum())
                score = (inter, -new_p)
                if score > best_score:
                    best_score, best_idx = score, bi
            if best_idx is not None:
                batches[best_idx].append(m)
                batch_vecs[best_idx] |= A[mi]
            else:
                batches.append([m])
                batch_vecs.append(A[mi].copy())

        # ── Opened orders: forced first batch ────────────────────────────────
        opened = [m for m in orders_list if d.orders[m].order_id in ws.opened_orders]
        if opened:
            batches.insert(0, opened)

        not_before_t  = 0
        not_committed: list[int] = []

        # ── Schedule each batch ───────────────────────────────────────────────
        for batch in batches:
            if not batch:
                continue

            order_pod_items: dict[int, dict[int, list]] = {}
            for m in batch:
                pm: dict[int, list] = {}
                for im in d.items_of_order[m]:
                    if x[im, -1] < 0.5:
                        pm.setdefault(d.pod_of_item[im], []).append(im)
                if pm:
                    order_pod_items[m] = pm

            if not order_pod_items:
                continue

            pod_earliest: dict[int, int] = {}
            pod_ord_cnt:  dict[int, int] = {}
            for m, pm in order_pod_items.items():
                for p_id, ims in pm.items():
                    e = max(max(int(d.earliest_t[im]) for im in ims), 1)
                    pod_earliest[p_id] = max(pod_earliest.get(p_id, 0), e)
                    pod_ord_cnt[p_id]  = pod_ord_cnt.get(p_id, 0) + 1

            # Tentative scheduling
            # ─ NO pod_busy copy: tentative picks tracked in `locked` dict ────
            # ─ robot_load snapshot: O(T) instead of O(n_pods × T) ─────────────
            tentative_picks: dict[int, int] = {}
            locked:  dict[int, set] = {}     # p_rel → set of tentatively locked t
            rl_snap  = robot_load.copy()     # only O(T)
            next_t   = not_before_t

            for p_id in sorted(pod_earliest,
                               key=lambda p: (-pod_ord_cnt[p], pod_earliest[p])):
                p_rel  = d.from_PodId_to_RelPod[p_id]
                travel = travel_to(p_id, p_rel, w_id)
                cand   = max(min_candidate(p_id, p_rel, w_id, pod_earliest[p_id],
                                          not_before_t), next_t)

                t_pick = find_slot(p_rel, cand, travel, locked)
                if t_pick is None:
                    continue

                tentative_picks[p_id] = t_pick
                locked.setdefault(p_rel, set()).add(t_pick)
                t0, t1 = max(0, t_pick - travel), min(T, t_pick + travel + 1)
                robot_load[t0:t1] += 1
                next_t = t_pick + 1

            # Commit-or-skip
            committable = [
                m for m, pm in order_pod_items.items()
                if all(p in tentative_picks for p in pm)
            ]
            not_committed += [m for m in order_pod_items if m not in committable]

            # Always restore robot_load; re-apply only for committed pods
            robot_load[:] = rl_snap

            if not committable:
                continue

            committed_pods: set[int] = set()
            for m in committable:
                committed_pods |= set(order_pod_items[m].keys())

            batch_end_t = not_before_t
            for p_id in committed_pods:
                t_pick = tentative_picks[p_id]
                p_rel  = d.from_PodId_to_RelPod[p_id]
                travel = travel_to(p_id, p_rel, w_id)

                pod_busy[p_rel, t_pick] = w_id
                t0, t1 = max(0, t_pick - travel), min(T, t_pick + travel + 1)
                robot_load[t0:t1] += 1
                pod_state_loc[p_rel] = w_id
                pod_state_t[p_rel]   = t_pick
                batch_end_t = max(batch_end_t, t_pick)

                for m in committable:
                    for im in order_pod_items[m].get(p_id, []):
                        x[im, t_pick:] = 1.0

            not_before_t = batch_end_t + 1

        # ── Retry uncommitted orders ──────────────────────────────────────────
        for m in not_committed:
            pm: dict[int, list] = {}
            for im in d.items_of_order[m]:
                if x[im, -1] < 0.5:
                    pm.setdefault(d.pod_of_item[im], []).append(im)
            if not pm:
                continue

            tentative: dict[int, int] = {}
            retry_locked: dict[int, set] = {}
            next_t = not_before_t
            ok     = True

            for p_id, ims in sorted(pm.items(),
                                    key=lambda kv: max(int(d.earliest_t[i]) for i in kv[1])):
                p_rel  = d.from_PodId_to_RelPod[p_id]
                travel = travel_to(p_id, p_rel, w_id)
                cand   = max(min_candidate(p_id, p_rel, w_id,
                                          max(int(d.earliest_t[i]) for i in ims),
                                          not_before_t), next_t)

                t_pick = find_slot(p_rel, cand, travel, retry_locked)
                if t_pick is None:
                    ok = False
                    break

                tentative[p_id] = t_pick
                retry_locked.setdefault(p_rel, set()).add(t_pick)
                next_t = t_pick + 1

            if not ok:
                continue

            last = not_before_t
            for p_id, t_pick in tentative.items():
                p_rel  = d.from_PodId_to_RelPod[p_id]
                travel = travel_to(p_id, p_rel, w_id)
                pod_busy[p_rel, t_pick] = w_id
                t0, t1 = max(0, t_pick - travel), min(T, t_pick + travel + 1)
                robot_load[t0:t1] += 1
                pod_state_loc[p_rel] = w_id
                pod_state_t[p_rel]   = t_pick
                last = max(last, t_pick)
                for im in pm[p_id]:
                    x[im, t_pick:] = 1.0

            not_before_t = last + 2

    return x