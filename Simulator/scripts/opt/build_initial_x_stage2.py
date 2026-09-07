from __future__ import annotations
import numpy as np
from collections import defaultdict


# Travel time from a pod's storage location to a workstation, read off the
# outgoing arcs at t = 0.
def _estimate_travel_time(storage_loc: int, ws_loc: int, d) -> int:
    """Travel time from a pod's storage location to workstation ws_loc."""
    n_travel = len(d.OptManager.travelling_arcs)
    for id_a in d.OptManager.outgoing_arc_idx.get((storage_loc, 0), []):
        if id_a >= n_travel:
            continue
        arc = d.OptManager.all_arcs[id_a]
        if arc[1][0] == ws_loc:
            return arc[1][1]
    return 1


# General hop between two locations, e.g. a pod moving on without returning
# to storage.
def _travel_time_between(src_loc: int, dst_loc: int, d) -> int:
    """Travel time between any two locations, looked up from arc_lookup."""
    if src_loc == dst_loc:
        return 0
    arcs = d.arc_lookup.get((src_loc, dst_loc), [])
    if arcs:
        dep_t, arr_t, _, _ = arcs[0]
        return max(1, arr_t - dep_t)
    return 1


# Order age, used only as a tie break in favour of older orders.
def _order_age(m, d) -> float:
    """Elapsed time since order m arrived, in time units."""
    return max(0.0, (d.current_time - d.orders[m].arrival_time) / d.OptManager.TIME_UNIT)


def build_initial_x(rng: np.random.Generator, d, attempt_idx: int = 0) -> np.ndarray:
    """Build an initial picking matrix x of shape (n_im, T), batching orders per
    workstation to maximise pod sharing."""
    T        = d.OptManager.N_TIME
    n_im     = len(d.relevant_pairs_for_x)
    n_pods   = len(d.from_RelPod_to_PodId)
    n_robots = len(d.warehouse.robots)
    n_ws     = len(d.warehouse.workstations)

    # Retries shrink the batches, easing contention on capacity, pods and robots.
    CAP_WS = max(1, d.OptManager.CAP_WS - (attempt_idx > 1) - (attempt_idx > 3))

    # How far a retried order may slide right, and how many times.
    RETRY_STEP     = 10
    RETRY_ATTEMPTS = 40

    x          = np.zeros((n_im, T), dtype=np.float64)
    robot_load = np.zeros(T, dtype=np.int32)
    pod_busy   = np.full((n_pods, T), -1, dtype=np.int32)

    # Where each pod sits (None = still in storage) and the time of its last
    # committed pick; this is what makes chaining possible.
    pod_state_loc = [None] * n_pods
    pod_state_t   = [0]    * n_pods

    # Items per pick slot. The first slot of a visit also pays the pod arrival
    # cost DELTA_POD; later slots only pay DELTA_ITEM.
    TU     = d.OptManager.TIME_UNIT
    D_ITEM = d.OptManager.DELTA_ITEM
    D_POD  = d.OptManager.DELTA_POD
    budget = TU
    if D_ITEM > 0:
        max_first = max(1, int((budget - D_POD) // D_ITEM))
        max_later = max(1, int(budget // D_ITEM))
    else:
        max_first = max_later = 10 ** 9  # no per-item cost: no limit

    def n_slots_for(n_items: int) -> int:
        """Consecutive slots one visit needs to serve n_items within the budget."""
        if n_items <= max_first:
            return 1
        rem = n_items - max_first
        return 1 + (rem + max_later - 1) // max_later

    # Travel lookups built once, so the closures below are plain O(1) reads.
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

    def travel_to(p_id: int, p_rel: int, w_id: int) -> int:
        """Time to bring the pod to w_id: from storage if unused, else from its
        last stop. A pod already at w_id costs one step, which chaining exploits."""
        loc = pod_state_loc[p_rel]
        if loc is None:
            return pod_ws_travel.get((p_id, w_id), 1) + 1
        if loc == w_id:
            return 1
        return int(ws_ws_travel[loc, w_id]) + 1

    def min_candidate(p_id: int, p_rel: int, w_id: int,
                      pod_e: int, floor: int) -> int:
        """Earliest the pod could reach w_id, not before its items are available
        (pod_e) nor before the caller's floor."""
        loc     = pod_state_loc[p_rel]
        travel  = travel_to(p_id, p_rel, w_id)
        arrival = (pod_state_t[p_rel] + travel) if loc is not None else travel
        return max(pod_e, arrival, floor, 1)

    # Main loop: one workstation at a time. All per-workstation state lives here.
    for w_id, order_ids in enumerate(d.orders_by_workstation):
        if not order_ids:
            continue

        orders_list = list(order_ids)

        # ws_active[t]: orders in progress at t (first pick to last; opened
        # orders from t = 0). Enforces capacity exactly, letting orders overlap.
        ws_active = np.zeros(T, dtype=np.int32)

        # ws_used[t]: True if some pod already picks here at t. With the per-visit
        # chunking this keeps one pod per slot, so the per-slot budget holds.
        ws_used = np.zeros(T, dtype=bool)

        opened = [m for m in orders_list
                  if d.orders[m].order_id in d.opened_order_ids]
        opened_set = set(opened)
        others = [m for m in orders_list if m not in opened_set]

        # Each opened order preloads a slot over the whole horizon until proven
        # otherwise; if one is never scheduled the preload just stays.
        if opened:
            ws_active[:] += len(opened)

        def find_slot(p_rel: int, start: int, travel: int,
                      locked: dict[int, set], ws_locked: set,
                      k: int) -> int | None:
            """Earliest start t >= start for a k-slot visit with the pod free,
            no other pod picking here, and the robot load under the count."""
            ts = locked.get(p_rel, frozenset())
            pb = pod_busy[p_rel]

            for t in range(max(1, start), T - k + 1):
                lo, hi = max(0, t - 1), min(T - 1, t + k)
                if pb[lo:hi + 1].max() >= 0:
                    continue
                if any(lo <= tt <= hi for tt in ts):
                    continue
                if ws_used[t:t + k].any():
                    continue
                if any(t <= tt < t + k for tt in ws_locked):
                    continue
                r0 = max(0, t - travel)
                r1 = min(T, t + k - 1 + travel + 1)
                if robot_load[r0:r1].max() < n_robots:
                    return t
            return None

        def schedule_batch(batch_orders: list[int],
                           floor_t: int) -> tuple[list[int], list[int]]:
            """Schedule a batch (picks at t >= floor_t), committing each order
            atomically. Returns (committed, failed)."""
            # Remaining items per order, grouped by pod. Opened orders finished
            # in an earlier batch drop out here.
            order_pod_items: dict[int, dict[int, list]] = {}
            trivially_done: list[int] = []
            for m in batch_orders:
                pm: dict[int, list] = {}
                for im in d.items_of_order[m]:
                    if x[im, -1] < 0.5:
                        pm.setdefault(d.pod_of_item[im], []).append(im)
                if pm:
                    order_pod_items[m] = pm
                else:
                    # Nothing left: an opened order closes at once, release its
                    # preload from t = 1 on.
                    trivially_done.append(m)
                    if m in opened_set:
                        ws_active[1:] -= 1

            if not order_pod_items:
                return trivially_done, []

            # One record per pod: earliest availability, count (orders served,
            # primary key), max_age (tie break), n_items (visit size).
            pod_info: dict[int, dict] = {}
            for m, pm in order_pod_items.items():
                age_m = _order_age(m, d)
                for p_id, ims in pm.items():
                    e    = max(max(int(d.earliest_t[im]) for im in ims), 1)
                    info = pod_info.setdefault(
                        p_id,
                        {"earliest": 0, "count": 0, "max_age": 0.0, "n_items": 0})
                    info["earliest"] = max(info["earliest"], e)
                    info["count"]   += 1
                    info["max_age"]  = max(info["max_age"], age_m)
                    info["n_items"] += len(ims)

            # Phase 1: tentative reservations. Advancing next_t keeps a
            # batch's pods close in time, and its windows tight.
            tentative: dict[int, tuple[int, int]] = {}
            locked:    dict[int, set] = {}
            ws_locked: set = set()
            rl_snap = robot_load.copy()
            next_t  = floor_t

            for p_id in sorted(pod_info,
                               key=lambda p: (-pod_info[p]["count"],
                                              -pod_info[p]["max_age"],
                                              pod_info[p]["earliest"])):
                p_rel  = d.from_PodId_to_RelPod[p_id]
                travel = travel_to(p_id, p_rel, w_id)
                k      = n_slots_for(pod_info[p_id]["n_items"])
                cand   = max(min_candidate(p_id, p_rel, w_id,
                                           pod_info[p_id]["earliest"], floor_t),
                             next_t)

                t0 = find_slot(p_rel, cand, travel, locked, ws_locked, k)
                if t0 is None:
                    continue

                tentative[p_id] = (t0, k)
                locked.setdefault(p_rel, set()).update(range(t0, t0 + k))
                ws_locked.update(range(t0, t0 + k))
                robot_load[max(0, t0 - travel):min(T, t0 + k - 1 + travel + 1)] += 1
                next_t = t0 + k

            # Phase 2: commit decisions, oldest first. Commit iff all pods got
            # a slot and the counter stays within CAP_WS over the window.
            committed: list[int] = []
            failed:    list[int] = []
            charged:   dict[int, tuple[int, int]] = {}

            for m in sorted(order_pod_items,
                            key=lambda m: -_order_age(m, d)):
                pm = order_pod_items[m]
                if not all(p in tentative for p in pm):
                    failed.append(m)
                    continue

                a = min(tentative[p][0] for p in pm)
                b = max(tentative[p][0] + tentative[p][1] - 1 for p in pm)

                if m in opened_set:
                    # Opened orders already hold their slot via the preload, so
                    # committing only releases it (phase 3). No check needed.
                    committed.append(m)
                elif ws_active[a:b + 1].max() + 1 <= CAP_WS:
                    ws_active[a:b + 1] += 1
                    charged[m] = (a, b)
                    committed.append(m)
                else:
                    failed.append(m)

            # Phase 3: undo tentative robot reservations, then commit the pods of
            # the committed orders only.
            robot_load[:] = rl_snap
            if not committed:
                return trivially_done, failed

            committed_pods: set[int] = set()
            for m in committed:
                committed_pods |= set(order_pod_items[m].keys())

            order_first: dict[int, int] = {}
            order_last:  dict[int, int] = {}

            for p_id in committed_pods:
                t0, _  = tentative[p_id]
                p_rel  = d.from_PodId_to_RelPod[p_id]
                travel = travel_to(p_id, p_rel, w_id)

                # Items of committed orders only, tagged with their order so the
                # true per-order window can be tracked.
                items = [(im, m) for m in committed
                         for im in order_pod_items[m].get(p_id, [])]

                # Split the visit into per-slot chunks: first slot pays the pod
                # arrival, later ones only the item cost.
                chunks, i, cap = [], 0, max_first
                while i < len(items):
                    chunks.append(items[i:i + cap])
                    i  += cap
                    cap = max_later

                t_last = t0 + len(chunks) - 1
                for off, chunk in enumerate(chunks):
                    t = t0 + off
                    pod_busy[p_rel, t] = w_id
                    ws_used[t] = True
                    for im, m in chunk:
                        x[im, t:] = 1.0
                        order_first[m] = min(order_first.get(m, T), t)
                        order_last[m]  = max(order_last.get(m, 0), t)

                robot_load[max(0, t0 - travel):min(T, t_last + travel + 1)] += 1
                pod_state_loc[p_rel] = w_id
                pod_state_t[p_rel]   = t_last

            # Finalise the counter to the exact first-to-last-pick interval
            # (opened orders: release the preload after their last pick).
            for m in committed:
                first, last = order_first[m], order_last[m]
                if m in opened_set:
                    if last + 1 < T:
                        ws_active[last + 1:] -= 1
                else:
                    a, b = charged[m]
                    if a < first:
                        ws_active[a:first] -= 1
                    if last + 1 <= b:
                        ws_active[last + 1:b + 1] -= 1

            return trivially_done + committed, failed

        def schedule_with_retries(m: int) -> bool:
            """Retry one order as its own batch, sliding the floor right until a
            free region is found. True iff committed."""
            floor = 1
            for _ in range(RETRY_ATTEMPTS):
                if floor >= T:
                    return False
                committed, _ = schedule_batch([m], floor)
                if committed:
                    return True
                floor += RETRY_STEP
            return False

        # Step 1: opened orders first. They are in progress from t = 0 anyway, so
        # finishing them early frees capacity and bites into the backlog penalty.
        pending: list[int] = []
        if opened:
            _, failed = schedule_batch(opened, 1)
            for m in sorted(failed, key=lambda m: -_order_age(m, d)):
                schedule_with_retries(m)

        if not others:
            continue

        # Step 2: batch the rest by pod sharing, on a boolean orders-by-pods
        # matrix A local to this workstation (A[i, j] iff order i needs pod j).
        order_pods = {
            m: frozenset(d.pod_of_item[im] for im in d.items_of_order[m])
            for m in others
        }
        ws_pod_list = sorted({p for ps in order_pods.values() for p in ps})
        pod_col     = {p: j for j, p in enumerate(ws_pod_list)}
        n_ord       = len(others)
        ord_row     = {m: i for i, m in enumerate(others)}

        A = np.zeros((n_ord, max(1, len(ws_pod_list))), dtype=np.bool_)
        for m, pods in order_pods.items():
            for p in pods:
                A[ord_row[m], pod_col[p]] = True

        # Seed: an order sharing pods with many others, since the batch around it
        # has more room to grow. Age breaks ties.
        pod_to_ords: dict[int, list] = defaultdict(list)
        for m, pods in order_pods.items():
            for p in pods:
                pod_to_ords[p].append(m)
        shared_count = {
            m: len({o for p in order_pods[m] for o in pod_to_ords[p]} - {m})
            for m in others
        }
        seed_order = sorted(
            others,
            key=lambda m: (-shared_count[m], -_order_age(m, d), rng.random()),
        )

        # Grow each batch by the order sharing the most pods with it and, on
        # ties, needing the fewest new pods. Stop at CAP_WS or no shared pod.
        assigned   = np.zeros(n_ord, dtype=bool)
        batches:    list[list[int]] = []
        batch_vecs: list[np.ndarray] = []

        for seed in seed_order:
            si = ord_row[seed]
            if assigned[si]:
                continue
            batch, bvec  = [seed], A[si].copy()
            assigned[si] = True

            while len(batch) < CAP_WS:
                free_idx = np.where(~assigned)[0]
                if free_idx.size == 0:
                    break
                free_A = A[free_idx]
                inter  = (free_A & bvec).sum(axis=1)
                mask   = inter > 0
                if not mask.any():
                    break
                new_p  = (free_A[mask] & ~bvec).sum(axis=1)
                cand   = free_idx[mask]
                best   = cand[np.lexsort((new_p, -inter[mask]))[0]]
                batch.append(others[best])
                assigned[best] = True
                bvec |= A[best]

            batches.append(batch)
            batch_vecs.append(bvec)

        # Merge underfull batches into whichever full batch fits best (same
        # scoring as growth); an order fitting nowhere becomes a singleton.
        pool = [m for b in batches if len(b) < CAP_WS for m in b]
        keep = [(b, bv) for b, bv in zip(batches, batch_vecs) if len(b) == CAP_WS]
        batches, batch_vecs = [b for b, _ in keep], [bv for _, bv in keep]

        for m in sorted(pool, key=lambda m: (-_order_age(m, d), rng.random())):
            mi = ord_row[m]
            best_idx, best_score = None, (-1, float('inf'))
            for bi, (b, bv) in enumerate(zip(batches, batch_vecs)):
                if len(b) >= CAP_WS:
                    continue
                score = (int((A[mi] & bv).sum()), -int((A[mi] & ~bv).sum()))
                if score > best_score:
                    best_score, best_idx = score, bi
            if best_idx is not None:
                batches[best_idx].append(m)
                batch_vecs[best_idx] |= A[mi]
            else:
                batches.append([m])
                batch_vecs.append(A[mi].copy())

        # Order the batches by chaining on shared pods: densest first, then
        # always the batch sharing the most pods with the last (already parked).
        def _density(i: int) -> float:
            return len(batches[i]) / max(1, int(batch_vecs[i].sum()))

        def _oldest(i: int) -> float:
            return max(_order_age(m, d) for m in batches[i])

        remaining = list(range(len(batches)))
        chain: list[int] = []
        if remaining:
            start = max(remaining, key=lambda i: (_density(i), _oldest(i)))
            chain.append(start)
            remaining.remove(start)
            prev_vec = batch_vecs[start]
            while remaining:
                nxt = max(
                    remaining,
                    key=lambda i: (int((batch_vecs[i] & prev_vec).sum()),
                                   _density(i),
                                   _oldest(i)),
                )
                chain.append(nxt)
                remaining.remove(nxt)
                prev_vec = batch_vecs[nxt]
        batches = [batches[i] for i in chain]

        # Step 3: schedule the batches. Each searches from t = 1, filling gaps
        # left by earlier ones; the counter alone decides how much overlap fits.
        for batch in batches:
            _, failed = schedule_batch(batch, 1)
            pending += failed

        # Step 4: retry pass, oldest first, each order alone with the floor
        # pushed later.
        for m in sorted(pending, key=lambda m: -_order_age(m, d)):
            schedule_with_retries(m)

    return x