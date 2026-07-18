from __future__ import annotations
from collections import defaultdict
import numpy as np
import logging

"""
Stage-1 local search for the order picking / pod assignment problem.

Decision variables
-------------------
x[im, p] : binary
    1 if item-order pair `im` (referring to relevant_pairs_for_x[im] = (item, order))
    is retrieved from pod `p`, 0 otherwise.
y[w, p] : binary
    1 if pod `p` is visited at workstation `w` (i.e. at least one item assigned to
    workstation `w` is retrieved from pod `p`), 0 otherwise.
z[m, w] : binary
    1 if order `m` is assigned to workstation `w`, 0 otherwise.
"""

def _get_beta(state, k_threshold: int = 3) -> float:
    """
    Distance weight for the Stage-1 cost, calibrated on the warehouse layout.
    Since serving k SKUs from a single pod saves k-1 visits, this weight makes
    pod sharing always win from `k_threshold` SKUs upwards, while below that
    threshold visits and travel distance trade off against each other.
    """
    d = _get_ws_pod_distance_matrix(state)
    span = float(d.max() - d.min())
    return (k_threshold - 1.5) / span if span > 0 else 0.0


def check_constraints(
    orders,
    orders_items,
    OptManager,
    relevant_pairs_for_x,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fixed_orders: dict[int, int] | None = None,
    ) -> bool:
    """
    Verify that a candidate (x, y, z) solution satisfies all Stage-1 constraints.<
    """
    n_w = OptManager.n_workstations

    num_constraints = 0

    # EC1: each order assigned to exactly one workstation
    num_constraints += z.shape[0]
    if (z.sum(axis=1) != 1).any():
        return False, 0

    # EC2: each item retrieved from exactly one pod (among those stocking its sku)
    for im, (i, _) in enumerate(relevant_pairs_for_x):
        num_constraints += 1
        if x[im, OptManager.pod_indices_by_sku[i]].sum() != 1:
            return False, 0

    # EC3: orders already opened at a workstation must stay pinned to it
    if fixed_orders:
        num_constraints += len(fixed_orders)
        for m, w_fixed in fixed_orders.items():
            if z[m, w_fixed] != 1:
                return False, 0

    # EC4: y[w,p] >= x[im,p] + z[m,w] - 1  for all w, im, p
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        num_constraints += y.shape[0] * y.shape[1]
        required = np.outer(z[m], x[im])   # shape (n_w, n_p)
        if (y < required - 1e-6).any():
            return False, 0

    # EC5: workload balance (in terms of number of SKUs per workstation)
    sku_per_order = np.array([len(orders_items[m]) for m in range(len(orders))])
    total_skus = sku_per_order.sum()

    lower_I = np.floor(total_skus / n_w * 0.8)
    upper_I = np.ceil(total_skus / n_w * 1.2)

    ws_loads = sku_per_order @ z          # shape (n_w,)
    num_constraints += 2*ws_loads.shape[0]
    if (ws_loads > upper_I + 1e-6).any() or (ws_loads < lower_I - 1e-6).any():
        return False, 0


    return True, num_constraints


def _get_ws_pod_distance_matrix(state) -> np.ndarray:
    """
    Cached Manhattan-distance matrix between every workstation and every pod
    storage location, shape (n_workstations, n_pods).

    Computed once per `state` (positions never change during the local
    search), instead of calling cell2coord() and recomputing the distance
    for every (w_id, p_id) pair on every single call to compute_objective —
    which, for n_ws x n_pods = 4 x 300 = 1200 pairs, is called many times
    per local-search run.
    """
    cached = getattr(state, "_ws_pod_dist_matrix", None)
    if cached is not None:
        return cached

    ws_coords = np.array([
        state.warehouse.cell2coord(ws.position)
        for ws in state.warehouse.workstations
    ])
    pod_coords = np.array([
        state.warehouse.cell2coord(p.storage_location)
        for p in state.warehouse.pods
    ])

    # Manhattan distance via broadcasting:
    # ws_coords[:, None, :] has shape (n_ws, 1, 2)
    # pod_coords[None, :, :] has shape (1, n_pods, 2)
    # -> abs-diff-sum over the last axis gives (n_ws, n_pods)
    dist_matrix = np.abs(
        ws_coords[:, None, :] - pod_coords[None, :, :]
    ).sum(axis=2)

    state._ws_pod_dist_matrix = dist_matrix
    return dist_matrix


def compute_objective(y: np.ndarray, state) -> float:
    """
    Stage-1 objective: number of (workstation, pod) visits, plus a distance
    penalty weighted according to the layout (see _get_beta).
    """
    dist = _get_ws_pod_distance_matrix(state)
    return float(y.sum()) + _get_beta(state) * float((y * dist).sum())


def get_fixed_orders(orders, state, n_w: int) -> dict[int, int]:
    """
    Build the {order_index -> workstation_index} mapping for orders that are
    already "opened" (picking started) at a given workstation.

    These orders are hard-fixed: they cannot be reassigned to another
    workstation during either the construction of the initial solution or
    the local search (EC3).
    """
    fixed_orders: dict[int, int] = {}
    for w in range(n_w):
        opened_ids = state.warehouse.workstations[w].opened_orders
        for m in range(len(orders)):
            if orders[m].order_id in opened_ids:
                fixed_orders[m] = w
    return fixed_orders


def build_initial_solution(orders, orders_items, relevant_pairs_for_x,
                           OptManager, state, rng,
                           w_new_pod=1.0, w_dist=None, cand_cap=60):
    """
    Seed and grow construction: cluster on the same workstation the orders
    that share SKUs, picking for each SKU the closest already open pod, or,
    when none is open, the closest pod that stocks it.
    """
    n_orders = len(orders)
    n_w      = OptManager.n_workstations
    n_p      = len(state.warehouse.pods)
    n_im     = len(relevant_pairs_for_x)

    z = np.zeros((n_orders, n_w), dtype=np.float64)
    x = np.zeros((n_im,    n_p), dtype=np.float64)
    y = np.zeros((n_w,     n_p), dtype=np.float64)

    dist = _get_ws_pod_distance_matrix(state)          # shape (n_w, n_p)
    if w_dist is None:
        w_dist = _get_beta(state)                      # same weight as the objective

    # Lookup tables
    items_of_order = defaultdict(list)                 # m maps to list of im
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        items_of_order[m].append(im)

    skus_of_order = {m: {relevant_pairs_for_x[im][0] for im in ims}
                     for m, ims in items_of_order.items()}

    orders_by_sku = defaultdict(set)                   # i maps to set of m
    for m, skus in skus_of_order.items():
        for i in skus:
            orders_by_sku[i].add(m)

    pods_by_sku = {i: np.asarray(OptManager.pod_indices_by_sku[i])
                   for i in orders_by_sku}

    # Consistency: count SKUs exactly the way the cost function counts them.
    sku_per_order = np.array([len(skus_of_order.get(m, ())) for m in range(n_orders)])
    total_skus = sku_per_order.sum()
    lower_I    = int(np.ceil(np.floor(total_skus / n_w * 0.8)))
    upper_I    = np.ceil(total_skus / n_w * 1.2)
    ws_load    = np.zeros(n_w, dtype=int)
    covered    = [set() for _ in range(n_w)]           # SKUs covered by open pods at w

    def _plan(m, w):
        """Return (plan, cost) where plan is a list of (im, p) and cost counts
        newly opened pods plus their weighted distance. Pods opened while
        planning this very order are tracked, so two SKUs sitting on the same
        pod are charged a single opening."""
        opened, plan, cost = set(), [], 0.0
        # SKUs with many candidate pods go last: the constrained ones choose first.
        ims = sorted(items_of_order.get(m, []),
                     key=lambda im: len(pods_by_sku[relevant_pairs_for_x[im][0]]))
        for im in ims:
            i    = relevant_pairs_for_x[im][0]
            pods = pods_by_sku[i]
            mask = y[w, pods] == 1
            if opened:
                mask = mask | np.isin(pods, list(opened))
            if mask.any():                              # reuse an open pod, zero cost
                cand = pods[mask]
                p = int(cand[np.argmin(dist[w, cand])])
            else:                                       # open a fresh pod
                p = int(pods[np.argmin(dist[w, pods])])
                opened.add(p)
                cost += w_new_pod + w_dist * dist[w, p]
            plan.append((im, p))
        return plan, cost

    def _apply(m, w):
        plan, _ = _plan(m, w)
        z[m, w] = 1
        ws_load[w] += sku_per_order[m]
        for im, p in plan:
            x[im, p] = 1
            y[w, p]  = 1
            covered[w].add(relevant_pairs_for_x[im][0])
        free.discard(m)

    # Step 1: pin the orders already opened at a workstation (EC3)
    fixed_orders = get_fixed_orders(orders, state, n_w)
    free = set(range(n_orders))
    for m, w in fixed_orders.items():
        if z[m].sum() == 0:
            _apply(m, w)

    # Step 2: seed and grow until the lower bound is met
    def _seed_for(w):
        """Cheapest free order to serve from w in terms of nearby pods.
        Ties are broken in favour of the larger orders."""
        pool = list(free)
        if len(pool) > cand_cap:
            pool = [pool[k] for k in rng.choice(len(pool), cand_cap, replace=False)]
        return min(pool, key=lambda m: (_plan(m, w)[1] / max(sku_per_order[m], 1),
                                        -sku_per_order[m]))

    def _grow_candidates(w):
        """Only the orders sharing at least one SKU with the pods open at w."""
        cand = set()
        for i in covered[w]:
            cand |= orders_by_sku[i]
        cand &= free
        if not cand:
            return None
        cand = list(cand)
        if len(cand) > cand_cap:
            cand = [cand[k] for k in rng.choice(len(cand), cand_cap, replace=False)]
        return cand

    for w in range(n_w):
        while ws_load[w] < lower_I and free:
            cand = _grow_candidates(w) if covered[w] else None
            if cand:
                # Incremental cost normalised by the SKUs the order brings along.
                m = min(cand, key=lambda m: _plan(m, w)[1] / max(sku_per_order[m], 1))
            else:
                m = _seed_for(w)
            _apply(m, w)

    # Step 3: leftovers, greedy assignment under the upper bound
    for m in list(free):
        best_w, best_cost = None, float("inf")
        for w in range(n_w):
            if ws_load[w] + sku_per_order[m] > upper_I + 1e-6:
                continue
            c = _plan(m, w)[1]
            if c < best_cost:
                best_cost, best_w = c, w
        if best_w is None:                              # no admissible workstation
            best_w = int(np.argmin(ws_load))
        _apply(m, best_w)

    return z, x, y


def local_search_stage1(
    orders, orders_items, relevant_pairs_for_x,
    OptManager, state, n_w: int,
    ):
    n_orders = len(orders)
    rng = np.random.default_rng(seed=42)

    fixed_orders = get_fixed_orders(orders, state, n_w)
    fixed_z = set(fixed_orders.keys())

    print("\n[ls_stage1] Building initial solution ...")
    logging.info("[ls_stage1] Building initial solution ...")

    MAX_INIT_ATTEMPTS = 10

    attempt = 1
    while True:
        z0, x0, y0 = build_initial_solution(
            orders, orders_items, relevant_pairs_for_x,
            OptManager, state, rng
        )

        ok, num_constraints = check_constraints(
            orders, orders_items, OptManager,
            relevant_pairs_for_x, x0, y0, z0, fixed_orders
        )

        if ok:
            break

        if attempt >= MAX_INIT_ATTEMPTS:
            raise RuntimeError("[ls_stage1] Failed to build feasible initial solution")

        attempt += 1
        logging.info("[ls_stage1] Retry initial solution (%d/%d)", attempt, MAX_INIT_ATTEMPTS)

    best_sol = (x0, z0, y0)
    best_obj = compute_objective(y0, state)

    current_sol = best_sol
    current_obj = best_obj

    logging.info("[ls_stage1] Initial objective = %.4f", best_obj)
    logging.warning(
            f"\nVariables size:\n"
            f"  shape x = {x0.shape}\n"
            f"  shape y = {y0.shape}\n"
            f"  shape z = {z0.shape}\n"
            f"  constraints_satisfied = {num_constraints}\n"
        )

    ### MAIN LOOP

    MAX_ITER = 200
    MAX_NEIGH = 600
    max_no_improve = 10
    ACCEPT_WORSE_PROB = 0.4

    iter_without_improvement = 0
    cont = 1
    am_I_stuck = False

    while cont <= MAX_ITER and not am_I_stuck:

        best_iter_sol = None
        best_iter_move = None
        best_iter_obj = np.inf

        # Generating a neighborhoud

        moves = []

        order_list = [m for m in range(n_orders) if m not in fixed_z]
        pairs = [(order_list[i], order_list[j])
                 for i in range(len(order_list))
                 for j in range(i + 1, len(order_list))]
        moves += [('swap', a, b) for a, b in pairs]

        for m in order_list:
            for w in range(n_w):
                moves.append(('moveto', m, w))

        for im, (i, _) in enumerate(relevant_pairs_for_x):
            for p in OptManager.pod_indices_by_sku[i]:
                moves.append(('repod', im, p))

        rng.shuffle(moves)
        moves = moves[:min(MAX_NEIGH, len(moves))]

        for move in moves:
            if move[0] == "swap":
                cand = _make_swap(current_sol, move[1], move[2], relevant_pairs_for_x)
            elif move[0] == "moveto":
                cand = _make_moveto(current_sol, move[1], move[2], relevant_pairs_for_x)
            else:
                cand = _make_repod(current_sol, move[1], move[2], relevant_pairs_for_x)

            x, z, y = cand
            ok, _ = check_constraints(
                orders, orders_items, OptManager,
                relevant_pairs_for_x, x, y, z, fixed_orders
            )

            if not ok:
                continue

            obj = compute_objective(y, state)
            if obj < best_iter_obj:
                best_iter_obj = obj
                best_iter_sol = cand
                best_iter_move = move

        if best_iter_sol is None:
            logging.info("[ls_stage1] Iter %d: NO FEASIBLE NEIGHBOUR", cont)
            print(f"[ls_stage1] Iter {cont}: NO FEASIBLE NEIGHBOUR")
            iter_without_improvement += 1

        elif best_iter_obj < best_obj:
            current_sol = best_iter_sol
            current_obj = best_iter_obj
            best_sol = best_iter_sol
            best_obj = best_iter_obj
            iter_without_improvement = 0
            logging.info("[ls_stage1] Iter %d: IMPROVEMENT move=%s obj=%.4f",
                         cont, best_iter_move, best_obj)
            print(f"[ls_stage1] Iter {cont}: IMPROVEMENT move={best_iter_move} obj={best_obj}")

        elif best_iter_obj == current_obj:
            current_sol = best_iter_sol
            current_obj = best_iter_obj
            iter_without_improvement += 1
            logging.info("[ls_stage1] Iter %d: PLATEAU move=%s obj=%.4f",
                         cont, best_iter_move, current_obj)
            print(f"[ls_stage1] Iter {cont}: PLATEAU move={best_iter_move} obj={current_obj}")

        else:
            if rng.random() < ACCEPT_WORSE_PROB:
                current_sol = best_iter_sol
                current_obj = best_iter_obj
                logging.info("[ls_stage1] Iter %d: WORSE ACCEPTED move=%s obj=%.4f best=%.4f",
                             cont, best_iter_move, current_obj, best_obj)
                print(f"[ls_stage1] Iter {cont}: WORSE ACCEPTED move={best_iter_move} obj={current_obj} best={best_obj}")
            else:
                logging.info("[ls_stage1] Iter %d: WORSE REJECTED neigh=%.4f current=%.4f best=%.4f",
                             cont, best_iter_obj, current_obj, best_obj)
                print(f"[ls_stage1] Iter {cont}: WORSE REJECTED move={best_iter_move} obj={current_obj} best={best_obj}")
            iter_without_improvement += 1

        if iter_without_improvement >= max_no_improve:
            logging.info("[ls_stage1] Converged after %d iterations without imprevement.", iter_without_improvement)
            print(f"[ls_stage1] Converged after {iter_without_improvement} iterations without imprevement.")
            am_I_stuck = True

        cont += 1

    x, z, y = best_sol
    logging.info("[ls_stage1] Final objective = %.4f", best_obj)
    return x, z



# ---------------------------------------------------------------------------
# Move helpers: each takes the current solution and returns a *new* candidate
# solution (x, z, y) reflecting a single local move. y is always rebuilt from
# scratch from (x, z) to keep it consistent (EC4).
# ---------------------------------------------------------------------------

def _make_swap(sol, m1, m2, relevant_pairs_for_x):
    """Swap the workstations of two orders m1, m2 (no-op if already equal)."""
    x, z, y = [arr.copy() for arr in sol]
    w1, w2 = z[m1].argmax(), z[m2].argmax()
    if w1 == w2:
        return x, z, y          # no-op: same workstation
    z[m1, w1], z[m2, w2] = 0, 0
    z[m1, w2], z[m2, w1] = 1, 1
    y = _rebuild_y(x, z, relevant_pairs_for_x)
    return x, z, y


def _make_moveto(sol, m, w_new, relevant_pairs_for_x):
    """Move order m to workstation w_new (no-op if already assigned there)."""
    x, z, y = [arr.copy() for arr in sol]
    w_old = z[m].argmax()
    if w_old == w_new:
        return x, z, y          # no-op
    z[m, w_old] = 0
    z[m, w_new] = 1
    y = _rebuild_y(x, z, relevant_pairs_for_x)
    return x, z, y


def _make_repod(sol, im, p_new, relevant_pairs_for_x):
    """Change the pod supplying item-order pair im to p_new (no-op if unchanged)."""
    x, z, y = [arr.copy() for arr in sol]
    p_old = x[im].argmax()
    if p_old == p_new:
        return x, z, y          # no-op
    x[im, p_old] = 0
    x[im, p_new] = 1
    y = _rebuild_y(x, z, relevant_pairs_for_x)
    return x, z, y


def _rebuild_y(x, z, relevant_pairs_for_x):
    """Recompute y[w, p] from scratch given x and z, enforcing EC4:
    y[w, p] = 1 iff some item-order pair uses pod p while its order is
    assigned to workstation w."""
    n_w = z.shape[1]
    n_p = x.shape[1]
    y   = np.zeros((n_w, n_p))
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        w = z[m].argmax()
        p = x[im].argmax()
        y[w, p] = 1
    return y