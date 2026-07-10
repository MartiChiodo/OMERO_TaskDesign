from __future__ import annotations
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


def compute_objective(y: np.ndarray):
    """Stage-1 objective: total number of (workstation, pod) visits, i.e. sum(y)."""
    return y.sum()


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


def build_initial_solution(orders, orders_items, relevant_pairs_for_x, OptManager, state, rng):
    """
    Build a feasible initial solution for Stage-1.

    Strategy
    --------
    1. Pin orders that are already opened at a workstation (EC3) to that
       workstation.
    2. Assign the remaining ("free") orders in a first pass to satisfy the
       lower bound on SKU load per workstation (EC5).
    3. Assign any still-free orders in a second, greedy pass that tries to
       minimise the number of newly activated (workstation, pod) pairs,
       while respecting the upper SKU-load bound.
    """
    n_orders = len(orders)
    n_w      = OptManager.n_workstations
    n_p      = len(state.warehouse.pods)
    n_im     = len(relevant_pairs_for_x)

    z = np.zeros((n_orders, n_w), dtype=np.float64)
    x = np.zeros((n_im,    n_p), dtype=np.float64)
    y = np.zeros((n_w,     n_p), dtype=np.float64)

    # Use SKU-based bounds, consistent with check_constraints / EC5
    sku_per_order = np.array([len(orders_items[m]) for m in range(n_orders)])
    total_skus    = sku_per_order.sum()
    lower_I       = np.floor(total_skus / n_w * 0.8)
    upper_I       = np.ceil(total_skus / n_w * 1.2)
    ws_load_skus  = np.zeros(n_w, dtype=int)   # tracks SKUs, not orders

    # Pre-compute order -> items lookup (order_index -> list of x-row indices)
    items_of_order: dict[int, list[int]] = {}
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        items_of_order.setdefault(m, []).append(im)

    def _assign_order(m, w):
        """Assign order m to workstation w, updating z, x, y and ws_load_skus.
        For each item of the order, reuse an already-open pod at that
        workstation if one stocks the SKU, otherwise open a new one."""
        z[m, w] = 1
        ws_load_skus[w] += sku_per_order[m]
        for im in items_of_order.get(m, []):
            i = relevant_pairs_for_x[im][0]
            pods = OptManager.pod_indices_by_sku[i]
            p = next((p for p in pods if y[w, p] == 1), pods[0])
            x[im, p] = 1
            y[w, p]  = 1

    ### Step 1: fix already-opened orders to their current workstation (EC3)
    fixed_orders = get_fixed_orders(orders, state, n_w)
    fixed = set(fixed_orders.keys())
    for m, w in fixed_orders.items():
        if z[m].sum() == 0:          # avoid double-assigning
            _assign_order(m, w)

    # Remaining orders in random order
    free_orders = [m for m in range(n_orders) if m not in fixed]
    rng.shuffle(free_orders)

    ### Step 2: first pass, satisfy the lower bound on each workstation 
    min_load_skus = int(np.ceil(lower_I))
    remaining     = free_orders.copy()
    still_free    = []

    for w in range(n_w):
        while ws_load_skus[w] < min_load_skus and remaining:
            m = remaining.pop()
            _assign_order(m, w)
        still_free = remaining   # orders left after satisfying lower bounds

    #### Step 3: second pass, greedily minimise new pod activations 
    for m in still_free:
        best_w, best_cost = None, float("inf")

        for w in range(n_w):
            if ws_load_skus[w] + sku_per_order[m] > upper_I + 1e-6:
                continue

            # Cost = number of items in order m that would require opening
            # a brand-new (workstation, pod) pair at workstation w.
            cost = sum(
                0 if y[w, OptManager.pod_indices_by_sku[
                    relevant_pairs_for_x[im][0]]].any() else 1
                for im in items_of_order.get(m, [])
            )

            if cost < best_cost:
                best_cost = cost
                best_w = w

        if best_w is None:
            # No workstation can accept the order without breaking the
            # upper bound: fall back to the least-loaded one.
            best_w = int(np.argmin(ws_load_skus))

        _assign_order(m, best_w)

    return z, x, y


def local_search_stage1(
    orders, orders_items, relevant_pairs_for_x,
    OptManager, state, n_w: int,
) -> tuple[dict, dict]:
    """
    Local search for Stage-1: order-to-workstation and item-to-pod assignment.

    Minimises the number of distinct (workstation, pod) pairs — equivalent
    to minimising sum(y), the Stage-1 objective — while respecting all
    constraints, including EC9 (orders already opened at a workstation stay
    pinned there).

    Neighbourhood moves explored at each iteration:
    - 'swap'   : exchange the workstations of two (non-fixed) orders.
    - 'moveto' : move a single (non-fixed) order to a different workstation.
    - 'repod'  : change the pod used to retrieve a given item-order pair.
    """
    n_orders = len(orders)
    rng      = np.random.default_rng(seed=42)

    # Orders already opened at a workstation (EC3): fixed, cannot be moved
    fixed_orders = get_fixed_orders(orders, state, n_w)
    fixed_z = set(fixed_orders.keys())

    ### INITIAL SOLUTION
    print("\n[ls_stage1] Building initial solution ...")
    logging.info("\n[ls_stage1] Building initial solution ...")

    MAX_INIT_ATTEMPTS = 10

    z0, x0, y0 = build_initial_solution(orders, orders_items, relevant_pairs_for_x,
                                        OptManager, state, rng)
    attempt = 1

    ok, num_constraints = check_constraints(orders, orders_items, OptManager, relevant_pairs_for_x,
                                 x0, y0, z0, fixed_orders)

    while not ok:
        if attempt >= MAX_INIT_ATTEMPTS:
            msg = (f"[ls_stage1] Failed to find a feasible initial solution "
                   f"after {MAX_INIT_ATTEMPTS} attempts.")
            print(msg)
            logging.error(msg)
            raise RuntimeError(msg)

        attempt += 1
        print(f"[ls_stage1] Infeasible initial solution, retrying "
              f"(attempt {attempt}/{MAX_INIT_ATTEMPTS}) ...")
        logging.info("[ls_stage1] Infeasible initial solution, retrying (attempt %d/%d)",
                     attempt, MAX_INIT_ATTEMPTS)
        z0, x0, y0 = build_initial_solution(orders, orders_items, relevant_pairs_for_x,
                                            OptManager, state, rng)

        ok, num_constraints = check_constraints(orders, orders_items, OptManager, relevant_pairs_for_x,
                                 x0, y0, z0, fixed_orders)


    best_sol = (x0, z0, y0)
    best_obj = compute_objective(y0)
    print(f"[ls_stage1] Feasible initial solution found at attempt "
          f"{attempt}/{MAX_INIT_ATTEMPTS}: obj = {best_obj:.4f}")
    logging.info("[ls_stage1] Feasible initial solution: obj = %.4f", best_obj)

    logging.warning(
            f"\nVariables size:\n"
            f"  shape x = {x0.shape}\n"
            f"  shape y = {y0.shape}\n"
            f"  shape z = {z0.shape}\n"
            f"  constraints_satisfied = {num_constraints}\n"
        )

    ### MAIN LOOP
    MAX_ITER           = 50
    MAX_NEIGH          = 600
    max_no_improve     = 3
    iter_without_improvement = 0
    am_I_stuck         = False
    cont               = 1

    print("[ls_stage1] Exploring neighbours ...")

    while cont <= MAX_ITER and not am_I_stuck:
        best_iter_obj  = np.inf
        best_iter_move = None
        best_iter_sol  = None

        # Generate candidate moves. Only non-fixed orders (not in fixed_z)
        # are eligible for 'swap' and 'moveto' moves, so that EC9 orders
        # never leave their pinned workstation.
        moves = []

        # Swap workstations between 2 (non-fixed) orders
        order_list = [m for m in range(n_orders) if m not in fixed_z]
        pairs = [(order_list[i], order_list[j])
                 for i in range(len(order_list))
                 for j in range(i + 1, len(order_list))]
        moves += [('swap', m1, m2) for m1, m2 in pairs]

        # Change workstation of a (non-fixed) order
        for m in order_list:
            for w in range(n_w):
                moves.append(('moveto', m, w))

        # Re-pod moves: change which pod supplies a given item-order pair
        for im, (i, _) in enumerate(relevant_pairs_for_x):
            pods = OptManager.pod_indices_by_sku[i]
            for p in pods:
                moves.append(('repod', im, p))

        rng.shuffle(moves)
        moves = moves[:min(MAX_NEIGH + 1, len(moves))]

        for move in moves:
            if move[0] == 'swap':
                sol_cand = _make_swap(best_sol, move[1], move[2], relevant_pairs_for_x)
            elif move[0] == 'moveto':
                sol_cand = _make_moveto(best_sol, move[1], move[2], relevant_pairs_for_x)
            elif move[0] == 'repod':
                sol_cand = _make_repod(best_sol, move[1], move[2], relevant_pairs_for_x)
            else:
                continue

            x, z, y = sol_cand
            ok, _ = check_constraints(orders, orders_items, OptManager, relevant_pairs_for_x,
                                  x, y, z, fixed_orders)
            if ok:
                obj = compute_objective(y)
                if obj < best_iter_obj:
                    best_iter_obj  = obj
                    best_iter_sol  = sol_cand
                    best_iter_move = move

        if best_iter_sol is not None:
            if best_iter_obj < best_obj:
                best_sol = best_iter_sol
                best_obj = best_iter_obj
                print(f"[ls_stage1] Iter {cont}: Improved with move "
                      f"{best_iter_move} → {best_obj:.4f}")
                logging.info("[ls_stage1] Iter %i: Improved with move %s → %.4f",
                             cont, best_iter_move, best_obj)
                iter_without_improvement = 0
            else:
                if best_iter_obj == best_obj:
                    # Moving along a plateau (same objective, different solution)
                    best_sol = best_iter_sol
                    best_obj = best_iter_obj

                iter_without_improvement += 1
                if iter_without_improvement >= max_no_improve:
                    am_I_stuck = True
                    print(f"[ls_stage1] Converged after {max_no_improve} "
                          f"iters without improvement at {best_obj:.4f}")
                    logging.info("[ls_stage1] Converged after %i iters without improvement",
                                 max_no_improve)
                else:
                    print(f"[ls_stage1] Iter {cont}: No improvement "
                          f"({iter_without_improvement}/{max_no_improve})")
                    logging.info("[ls_stage1] Iter %i: No improvement", cont)

        cont += 1

    print(f"[ls_stage1] Final obj = {best_obj}")
    x, z, y = best_sol
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