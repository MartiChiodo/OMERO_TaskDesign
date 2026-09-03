from __future__ import annotations
import numpy as np
from collections import defaultdict


### Travel time helpers
###
### _estimate_travel_time reads the travel time from a pod's storage
### location to a workstation by scanning the outgoing travelling arcs
### at t equal to 0 (the time expanded network repeats identically at
### every departure instant from storage, so t equal to 0 is a valid
### representative of them all).
### _travel_time_between is the general case, used for hops between two
### workstations when a pod moves on without passing through storage.

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


def _travel_time_between(src_loc: int, dst_loc: int, d) -> int:
    """Travel time between any two locations, looked up from arc_lookup."""
    if src_loc == dst_loc:
        return 0
    arcs = d.arc_lookup.get((src_loc, dst_loc), [])
    if arcs:
        dep_t, arr_t, _, _ = arcs[0]
        return max(1, arr_t - dep_t)
    return 1


### Order age
###
### Mirrors the age term inside the backlog penalty of compute_objective.
### It is used ONLY as a tie break signal throughout this file: pod
### sharing decides feasibility and batch structure, and whenever two
### choices are equally good on that front, the older order wins. This
### aligns the heuristic's tie breaks with what the true objective
### penalises, instead of leaving them to arrival order or randomness.

def _order_age(m, d) -> float:
    """Age of order m, i.e. elapsed time since its arrival, in time units."""
    return max(0.0, (d.current_time - d.orders[m].arrival_time) / d.OptManager.TIME_UNIT)


### Main builder

def build_initial_x(rng: np.random.Generator, d, attempt_idx: int = 0) -> np.ndarray:
    """
    Build an initial picking matrix x of shape (n_im, T), batching orders
    per workstation to maximise pod sharing, with EC10 enforced EXACTLY
    through a per time step capacity registry instead of the strictly
    serialized waves of version 2.

    ### The EC10 semantics this file is built around
    An order is in progress from its FIRST pick until ONE STEP AFTER its
    LAST pick: t_end equals last_pick plus 1, hence v equals 1 on the
    closed interval from first_pick to last_pick, and 0 from last_pick
    plus 1 onwards. Already opened orders (order_id in
    d.opened_order_ids) additionally have v FORCED to 1 from t equal to
    0 until completion, and an opened order that is never completed
    keeps v equal to 1 for the WHOLE horizon.

    ### The capacity registry (what changed versus version 2)
    Version 2 guaranteed EC10 by never letting two waves of orders
    overlap in time. Sufficient, but far more restrictive than the
    constraint itself: half of a wave could be finished, and its freed
    slots stayed unusable until the LAST order of the wave closed. That
    conservatism is what degraded solution quality.

    Version 3 keeps, per workstation, an integer array ws_active of
    length T counting how many orders are in progress at every time
    step. The rules are simple and exact:

      1. Every opened order preloads a plus 1 on the entire horizon,
         because its v is forced to 1 from t equal to 0. When it is
         committed and closes at its last pick L, the occupancy is
         RELEASED on the interval from L plus 1 to the end. An opened
         order that is never scheduled keeps its slot forever, which is
         exactly what the constraint checker will see.
      2. A regular order about to be committed knows its picking window
         (from its earliest to its latest reserved pick). It is accepted
         only if, over that whole window, ws_active plus 1 stays within
         CAP_WS; on acceptance the registry is incremented there.

    Orders therefore overlap freely, exactly like the real constraint
    allows: the moment one closes, its slot is reusable one step later.
    Feasibility is checked against the truth, not against a proxy.

    ### Everything that stays from version 2
    Pod sharing batching (vectorised orders by pods boolean matrix, seed
    by connectivity, growth by maximum intersection and minimum new
    pods, merge of underfull batches), chained batch ordering on shared
    pods so that pods parked at the workstation by one batch are reused
    immediately by the next, atomic commit per order (an order is never
    partially picked), opened orders scheduled first, failed orders
    retried oldest first, and the EC11 guard: a pod visit serving more
    items than the per slot work budget allows is split over consecutive
    slots, and at most one pod occupies any pick slot at a workstation.
    """
    T        = d.OptManager.N_TIME
    n_im     = len(d.relevant_pairs_for_x)
    n_pods   = len(d.from_RelPod_to_PodId)
    n_robots = len(d.warehouse.robots)
    n_ws     = len(d.warehouse.workstations)

    ### Retry attempts shrink the batches, lowering the concurrency the
    ### registry has to accommodate and the contention on pods and robots.
    CAP_WS = max(1, d.OptManager.CAP_WS - (attempt_idx > 1) - (attempt_idx > 3))

    ### How far a retried order may be pushed to the right, per attempt,
    ### while looking for a time region with residual EC10 capacity.
    RETRY_STEP     = 10
    RETRY_ATTEMPTS = 40

    x          = np.zeros((n_im, T), dtype=np.float64)
    robot_load = np.zeros(T, dtype=np.int32)
    pod_busy   = np.full((n_pods, T), -1, dtype=np.int32)

    ### Pod state, the key to chaining trips: pod_state_loc[p_rel] is the
    ### workstation the pod currently sits at, or None if it never left
    ### storage; pod_state_t[p_rel] is the time of its last committed
    ### pick, i.e. the earliest departure time for its next trip.
    pod_state_loc = [None] * n_pods
    pod_state_t   = [0]    * n_pods

    ### EC11 budget: how many items fit in a single pick slot. The first
    ### slot of a visit also pays the pod arrival cost DELTA_POD; later
    ### slots (pod already parked at the workstation) only pay the per
    ### item cost DELTA_ITEM.
    TU     = d.OptManager.TIME_UNIT
    D_ITEM = d.OptManager.DELTA_ITEM
    D_POD  = d.OptManager.DELTA_POD
    budget = TU
    if D_ITEM > 0:
        max_first = max(1, int((budget - D_POD) // D_ITEM))
        max_later = max(1, int(budget // D_ITEM))
    else:
        ### Degenerate configuration with no per item cost: no limit.
        max_first = max_later = 10 ** 9

    def n_slots_for(n_items: int) -> int:
        """Number of consecutive pick slots one pod visit needs to serve
        n_items without breaking the EC11 per slot work budget."""
        if n_items <= max_first:
            return 1
        rem = n_items - max_first
        return 1 + (rem + max_later - 1) // max_later

    ### Travel time lookup tables, built once up front so that the per
    ### pod closures below are O(1) dictionary or array lookups instead
    ### of rescanning the arc network on every call.
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
        """Travel time from wherever the pod currently is to workstation
        w_id: from storage if it was never used, otherwise from its last
        known position. A pod already parked at w_id costs a single step,
        which is exactly what the chained batch ordering exploits."""
        loc = pod_state_loc[p_rel]
        if loc is None:
            return pod_ws_travel.get((p_id, w_id), 1) + 1
        if loc == w_id:
            return 1
        return int(ws_ws_travel[loc, w_id]) + 1

    def min_candidate(p_id: int, p_rel: int, w_id: int,
                      pod_e: int, floor: int) -> int:
        """Earliest time the pod could physically arrive at w_id, bounded
        below by the items' own earliest availability (pod_e) and by a
        caller supplied floor."""
        loc     = pod_state_loc[p_rel]
        travel  = travel_to(p_id, p_rel, w_id)
        arrival = (pod_state_t[p_rel] + travel) if loc is not None else travel
        return max(pod_e, arrival, floor, 1)

    ### Main loop: one workstation at a time. All the per workstation
    ### state (the EC10 capacity registry, the pick slot occupancy used
    ### for EC11, the scheduling closures) lives inside this loop.
    for w_id, order_ids in enumerate(d.orders_by_workstation):
        if not order_ids:
            continue

        orders_list = list(order_ids)

        ### EC10 capacity registry: ws_active[t] counts how many orders
        ### of THIS workstation are in progress at time t, under exactly
        ### the same semantics as the constraint checker.
        ws_active = np.zeros(T, dtype=np.int32)

        ### EC11 pick slot occupancy: ws_used[t] is True iff some pod
        ### already picks at this workstation at time t. Together with
        ### the per visit chunking this guarantees at most one pod per
        ### slot, so the per slot work budget can never be exceeded.
        ws_used = np.zeros(T, dtype=bool)

        ### Use d.opened_order_ids, the SAME source of truth as the
        ### constraint checker, not ws.opened_orders.
        opened = [m for m in orders_list
                  if d.orders[m].order_id in d.opened_order_ids]
        opened_set = set(opened)
        others = [m for m in orders_list if m not in opened_set]

        ### Preload the registry: every opened order occupies one slot on
        ### the WHOLE horizon until proven otherwise (its v is forced to
        ### 1 from t equal to 0, and only completing it releases the
        ### slot). If an opened order is never scheduled, the preload
        ### simply stays, which matches what the checker will compute.
        if opened:
            ws_active[:] += len(opened)

        def find_slot(p_rel: int, start: int, travel: int,
                      locked: dict[int, set], ws_locked: set,
                      k: int) -> int | None:
            """
            First feasible start time t at or after `start` for a pod
            visit spanning the k consecutive slots from t to t plus k
            minus 1, such that:
              1. the pod is free on the visit window padded by one step
                 on each side, counting both committed picks (pod_busy)
                 and picks tentatively reserved earlier in THIS batch
                 (locked);
              2. no other pod, committed (ws_used) or tentatively
                 reserved in this batch (ws_locked), picks at this
                 workstation during those k slots (EC11 guard);
              3. the robot load window covering the travel plus the
                 whole visit stays strictly under the robot count.
            The scan runs to the end of the horizon (no artificial shift
            bound): if capacity exists anywhere, it will be found.
            """
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
            """
            Try to schedule the orders of one batch with all picks at t
            at or after floor_t. Commit or skip PER ORDER, atomically.

            Phase 1 reserves tentative pick slots pod by pod (pods
            serving more orders of the batch go first, since one visit
            then captures more sharing). Phase 2 walks the orders and
            commits each one only if ALL its pods got a slot AND the
            EC10 registry has room over the order's whole picking
            window. Phase 3 writes x, pod_busy, robot_load, ws_used and
            the pod state for the pods of the committed orders only,
            after rolling back every tentative robot reservation.

            Returns (committed, failed).
            """
            ### Remaining, not yet picked items per order, grouped by
            ### pod. Opened orders already completed in an earlier batch
            ### simply drop out here.
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
                    ### No remaining items: nothing to schedule. For an
                    ### opened order this means it closes immediately,
                    ### so its preloaded occupancy is released from t
                    ### equal to 1 onwards.
                    trivially_done.append(m)
                    if m in opened_set:
                        ws_active[1:] -= 1

            if not order_pod_items:
                return trivially_done, []

            ### One combined record per pod: earliest (its items'
            ### earliest availability), count (how many orders of this
            ### batch it serves, the primary key, since a higher count
            ### means more pod sharing captured in a single visit),
            ### max_age (the oldest order it touches, the tie break) and
            ### n_items (to size the visit under the EC11 budget).
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

            ### Phase 1: tentative reservations. Only the O(T) robot
            ### load array is snapshotted for rollback; tentative pod
            ### picks live in `locked` and tentative workstation slots
            ### in `ws_locked`, so no large array is ever copied.
            ### Advancing next_t to the end of the previous visit keeps
            ### the pods of one batch packed close together in time,
            ### which keeps the orders' picking windows (and hence their
            ### EC10 charge) tight.
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

            ### Phase 2: per order commit decision, oldest orders first
            ### so that whenever capacity is scarce it goes to the
            ### orders the backlog penalty cares most about. An order is
            ### committed iff ALL its pods got a slot AND the EC10
            ### registry stays within CAP_WS over its whole (slightly
            ### conservative) picking window; the exact window is
            ### refunded in phase 3 once the items are placed.
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
                    ### Opened orders already occupy their slot on the
                    ### whole horizon via the preload, so committing
                    ### them never ADDS occupancy; it only releases it
                    ### (phase 3). No capacity check needed.
                    committed.append(m)
                elif ws_active[a:b + 1].max() + 1 <= CAP_WS:
                    ws_active[a:b + 1] += 1
                    charged[m] = (a, b)
                    committed.append(m)
                else:
                    failed.append(m)

            ### Phase 3: rollback of every tentative robot reservation,
            ### then real commit for the pods of the committed orders
            ### only, so nothing reserved for excluded orders lingers.
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

                ### Items of committed orders only, each tagged with its
                ### order so the true per order picking window can be
                ### tracked below. Fewer items than tentatively sized
                ### can only mean fewer chunks, never more.
                items = [(im, m) for m in committed
                         for im in order_pod_items[m].get(p_id, [])]

                ### Split the visit into consecutive per slot chunks
                ### respecting the EC11 budget: the first slot also pays
                ### the pod arrival cost, later slots only the item cost.
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

            ### Registry finalisation per committed order. Opened
            ### orders: release the preloaded occupancy from one step
            ### after their last pick (that is when they close). Regular
            ### orders: refund the edges of the conservative window
            ### charged in phase 2, so the registry stores the EXACT in
            ### progress interval from first pick to last pick.
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
            """Retry a single order as its own batch, pushing the floor
            progressively to the right: if the earliest region has no
            residual EC10 capacity or no free slots, a later region may.
            Returns True iff the order was eventually committed."""
            floor = 1
            for _ in range(RETRY_ATTEMPTS):
                if floor >= T:
                    return False
                committed, _ = schedule_batch([m], floor)
                if committed:
                    return True
                floor += RETRY_STEP
            return False

        ### Step 1: opened orders first. They are in progress from t
        ### equal to 0 no matter what, so completing them as early as
        ### possible both frees registry capacity for everything else
        ### and directly attacks the backlog penalty (they are almost
        ### always the oldest orders).
        pending: list[int] = []
        if opened:
            _, failed = schedule_batch(opened, 1)
            for m in sorted(failed, key=lambda m: -_order_age(m, d)):
                schedule_with_retries(m)

        if not others:
            continue

        ### Step 2: pod sharing batching of the remaining orders, on a
        ### binary orders by pods matrix A local to this workstation:
        ### A[i, j] is True iff order i needs pod j.
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

        ### Seed choice: an order sharing pods with many OTHER orders
        ### makes a good seed, since the batch built around it has more
        ### candidates to grow into. Age breaks ties among equally
        ### connected candidates; pod sharing stays the primary key.
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

        ### Greedy batch growth: starting from each unassigned seed,
        ### repeatedly add the unassigned order that shares the MOST
        ### pods with the batch built so far (inter, descending) and,
        ### among ties, needs the FEWEST brand new pods (new_p,
        ### ascending), computed for every remaining candidate at once
        ### via vectorised numpy boolean operations. Growth stops when
        ### the batch reaches CAP_WS orders or nothing shares a pod.
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

        ### Merge underfull batches: any batch below CAP_WS gets its
        ### orders redistributed, oldest first, into whichever remaining
        ### batch fits them best (same intersection and new pods scoring
        ### as the growth step). An order fitting nowhere becomes its
        ### own singleton batch.
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

        ### Batch scheduling order: greedy CHAINING on shared pods. The
        ### first batch is the densest one (orders per pod needed,
        ### higher means a more pod efficient group). Every following
        ### batch is the remaining one sharing the MOST pods with the
        ### batch just scheduled: those pods are already parked at the
        ### workstation, so travel_to returns a single step for them and
        ### their picks land earlier. Density and age are tie breaks.
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

        ### Step 3: schedule the batches. No serialization floor here:
        ### every batch searches from t equal to 1, gaps left by earlier
        ### batches get filled, and the capacity registry alone decides
        ### how much concurrency each time region can still absorb.
        for batch in batches:
            _, failed = schedule_batch(batch, 1)
            pending += failed

        ### Step 4: retry pass, oldest orders first, each order alone
        ### and with a floor pushed progressively later, so it lands in
        ### the first time region with both free pick slots and residual
        ### EC10 capacity.
        for m in sorted(pending, key=lambda m: -_order_age(m, d)):
            schedule_with_retries(m)

    return x