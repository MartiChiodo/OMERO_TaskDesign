
import numpy as np
### INITIAL SOLUTION BUILDER

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

    return x 
