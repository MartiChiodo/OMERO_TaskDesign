from __future__ import annotations
import numpy as np
import logging


def check_constraints(orders, orders_items, OptManager, relevant_pairs_for_x,
                      x: np.ndarray, y: np.ndarray, z: np.ndarray) -> bool:
    """
    x[im, p] = 1 if pod p retrieves item im
    y[w,  p] = 1 if pod p visits workstation w
    z[m,  w] = 1 if order m is assigned to workstation w
    """
    n_w = OptManager.n_workstations

    # EC7: each order assigned to exactly one workstation
    if (z.sum(axis=1) != 1).any():
        return False

    # EC8: each item retrieved from exactly one pod (among those stocking its sku)
    for im, (i, _) in enumerate(relevant_pairs_for_x):
        if x[im, OptManager.pod_indices_by_sku[i]].sum() != 1:
            return False

    # EC10: y[w,p] >= x[im,p] + z[m,w] - 1  for all w, im, p
    # Equivalent: y[w,p] must be 1 whenever both x[im,p]=1 and z[m,w]=1
    # Vectorised: for each im, the outer product z[m,:] (shape n_w) and x[im,:] (shape n_p)
    # gives a (n_w, n_p) matrix of required y activations
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        required = np.outer(z[m], x[im])   # shape (n_w, n_p)
        if (y < required - 1e-6).any():
            return False

    # EC11: workload balance
    # EC11: workload balance (in termini di SKU)
    sku_per_order = np.array([len(orders_items[m]) for m in range(len(orders))])  # shape (n_orders,)
    total_skus = sku_per_order.sum()

    lower_I = total_skus / n_w * 0.75
    upper_I = total_skus / n_w * 1.25

    ws_loads = sku_per_order @ z          # shape (n_w,)  —  SKU totali per worker
    if (ws_loads > upper_I + 1e-6).any() or (ws_loads < lower_I - 1e-6).any():
        return False

    return True


def compute_objective(y: np.ndarray):
    return y.sum()


def build_initial_solution(orders, relevant_pairs_for_x, OptManager, state, rng):
    """
    Build a feasible initial solution for Stage-1.
    """
    n_orders = len(orders)
    n_w      = OptManager.n_workstations
    n_p      = len(state.warehouse.pods)
    n_im     = len(relevant_pairs_for_x)

    z = np.zeros((n_orders, n_w), dtype=np.float64)
    x = np.zeros((n_im,    n_p), dtype=np.float64)
    y = np.zeros((n_w,     n_p), dtype=np.float64)

    lower_I = n_orders / n_w * 0.5
    upper_I = n_orders / n_w * 1.5
    ws_load = np.zeros(n_w, dtype=int)

    # Pre-compute order -> items lookup
    items_of_order: dict[int, list[int]] = {}
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        items_of_order.setdefault(m, []).append(im)

    # Fix already-open orders
    fixed = set()
    for w in range(n_w):
        for m in range(n_orders):
            if orders[m].order_id in state.warehouse.workstations[w].opened_orders:
                z[m, w] = 1
                ws_load[w] += 1
                fixed.add(m)

    # Assign pods to fixed orders: prefer pods already active at that workstation
    for m in fixed:
        w = int(np.argmax(z[m]))
        for im in items_of_order.get(m, []):
            i = relevant_pairs_for_x[im][0]
            pods = OptManager.pod_indices_by_sku[i]
            # Pick pod already visiting w if any, else first available
            p = next((p for p in pods if y[w, p] == 1), pods[0])
            x[im,p] = 1
            y[w,p] = 1

    # Assign remaining orders in random order
    free_orders = [m for m in range(n_orders) if m not in fixed]
    rng.shuffle(free_orders)

    min_load = int(np.ceil(lower_I))

    # First ensuring lowerbound in EC11 is satisfied
    remaining = free_orders.copy()
    free_orders = []

    for w in range(n_w):
        while ws_load[w] < min_load and remaining:
            m = remaining.pop()

            z[m, w] = 1
            ws_load[w] += 1

            # assign pods for this order
            for im in items_of_order.get(m, []):
                i = relevant_pairs_for_x[im][0]
                pods = OptManager.pod_indices_by_sku[i]
                p = next((p for p in pods if y[w, p] == 1), pods[0])
                x[im, p] = 1
                y[w, p] = 1

    # orders still unassigned
    free_orders = remaining


    # Assignments minimizing costs
    for m in free_orders:
        best_w, best_cost = None, float("inf")

        for w in range(n_w):
            if ws_load[w] >= upper_I:
                continue

            cost = sum(
                0 if y[w, pods].any() else 1
                for im in items_of_order.get(m, [])
                for pods in [OptManager.pod_indices_by_sku[
                    relevant_pairs_for_x[im][0]
                ]]
            )

            if cost < best_cost:
                best_cost = cost
                best_w = w

        if best_w is None:
            best_w = int(np.argmin(ws_load))

        z[m, best_w] = 1
        ws_load[best_w] += 1

        for im in items_of_order.get(m, []):
            i = relevant_pairs_for_x[im][0]
            pods = OptManager.pod_indices_by_sku[i]
            p = next((p for p in pods if y[best_w, p] == 1), pods[0])
            x[im, p] = 1
            y[best_w, p] = 1

    return z, x, y


def local_search_stage1(
    orders, orders_items, relevant_pairs_for_x,
    OptManager, state, n_w: int,
) -> tuple[dict, dict]:
    """
    Local search for Stage-1: order-workstation and item-pod assignment.

    Minimises the number of distinct (workstation, pod) pairs — equivalent
    to minimising sum(y1), the Stage-1 objective.
    """
    n_orders = len(orders)
    rng      = np.random.default_rng(seed=42)


    fixed_z = set()
    for w in range(n_w):
        for m in range(n_orders):
            if orders[m].order_id in state.warehouse.workstations[w].opened_orders:
                fixed_z.add(m)

    # Initial solution
    print("\n[ls_stage1] Building initial solution …")
    z0, x0, y0 = build_initial_solution(orders, relevant_pairs_for_x, OptManager, state, rng)
    while not check_constraints(orders, orders_items, OptManager, relevant_pairs_for_x, x0, y0, z0):
        z0, x0, y0 = build_initial_solution(orders, relevant_pairs_for_x, OptManager, state, rng)

    best_sol = (x0, z0, y0)
    best_obj = compute_objective(y0)
    print(f"[ls_stage1] Feasible initial solution: obj = {best_obj:.4f}")

    # Main loop 
    MAX_ITER  = 30
    MAX_NEIGH = 600
    max_no_improve = 5
    iter_without_improvement = 0
    am_I_stuck = False
    cont = 1

    print(f"[ls_stage1] Exploring neighbours ...")

    while cont <= MAX_ITER and not am_I_stuck:
        best_iter_obj = np.inf
        best_iter_move = None
        best_iter_sol = None

        # Generate candidate moves
        moves = []

        # Swap workstations between 2 orders
        order_list = [m for m in range(n_orders) if m not in fixed_z]
        pairs = [(order_list[i], order_list[j])
                 for i in range(len(order_list))
                 for j in range(i + 1, len(order_list))]
        moves += [('swap', m1, m2) for m1, m2 in pairs]

        # Change workstation
        for m in order_list:
            for w in range(n_w):
                moves.append(('moveto', m, w))

        # Re-pod moves
        for im, (i,_) in enumerate(relevant_pairs_for_x):
            pods = OptManager.pod_indices_by_sku[i]
            for p in pods:
                moves.append(('repod', im, p))

        # Capping the number of moves explored
        rng.shuffle(moves)
        moves = moves[:min(MAX_NEIGH+1, len(moves))]

        for move in moves:
            if move[0] == 'swap':
                sol_cand = _make_swap(best_sol, move[1], move[2], relevant_pairs_for_x)
            elif move[0] == 'moveto':
                sol_cand = _make_moveto(best_sol, move[1], move[2], relevant_pairs_for_x)
            elif move[0] == 'repod':
                sol_cand = _make_repod(best_sol, move[1], move[2], relevant_pairs_for_x)

            x, z, y = sol_cand
            if check_constraints(orders, orders_items, OptManager, relevant_pairs_for_x, x, y, z):
                obj = compute_objective(y)

                if obj < best_iter_obj:
                    best_iter_obj = obj
                    best_iter_sol = sol_cand
                    best_iter_move = move


        if not best_iter_sol is None:
            if best_iter_obj < best_obj:
                best_sol = best_iter_sol
                best_obj = best_iter_obj
                print(f"[ls_stage1] Iter {cont} : Improved with move {best_iter_move} → {best_obj:.4f}")
                iter_without_improvement = 0
            else:
                if best_iter_obj == best_obj:
                    # Moving along a plateau
                    best_sol = best_iter_sol
                    best_obj = best_iter_obj                
                
                iter_without_improvement += 1
                if iter_without_improvement >= max_no_improve:
                    am_I_stuck = True
                    print(f"[ls_stage1] Converged after "
                        f"{max_no_improve} iters without improvement "
                        f"at {best_obj:.4f}")
                else:
                    print(f"[ls_stage1] Iter {cont} : No improvement "
                        f"({iter_without_improvement}/{max_no_improve})")

        cont += 1

    print(f"[ls_stage1] Final obj = {best_obj}")
    x, z, y = best_sol
    return x, z


def _make_swap(sol, m1, m2, relevant_pairs_for_x):
    x, z, y = [arr.copy() for arr in sol]
    w1, w2 = z[m1,:].argmax(), z[m2,:].argmax()
    pods1, pods2 = [], []
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        if m == m1:
            pods1.append(x[im,:].argmax())
        elif m == m2:
            pods2.append(x[im,:].argmax())

    # Update z and y
    z[m1, w1], z[m2, w2] = 0, 0
    z[m1, w2], z[m2, w1] = 1, 1
    y = _rebuild_y(x,z,relevant_pairs_for_x)
    return x, z, y


def _make_moveto(sol, m, w_new, relevant_pairs_for_x):
    x, z, y = [arr.copy() for arr in sol]
    w_old = z[m,:].argmax()
    pods = []
    for im, (_, id_m) in enumerate(relevant_pairs_for_x):
        if id_m == m:
            pods.append(x[im,:].argmax())

    # Update z and y
    z[m, w_old], z[m, w_new] = 0, 1
    y = _rebuild_y(x,z,relevant_pairs_for_x)
    return x, z, y


def _make_repod(sol, im, p_new, relevant_pairs_for_x):
    x, z, y = [arr.copy() for arr in sol]
    p_old = x[im,:].argmax()
    for id_im, (_,m) in enumerate(relevant_pairs_for_x):
        if id_im == im:
            w = z[m,:].argmax()
    
    # Update x and y
    x[im,p_old], x[im, p_new] = 0, 1
    y = _rebuild_y(x,z,relevant_pairs_for_x)
    return x, z, y


def _rebuild_y(x, z, relevant_pairs_for_x):
    n_w = z.shape[1]
    n_p = x.shape[1]
    y = np.zeros((n_w, n_p))
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        w = z[m].argmax()
        p = x[im].argmax()
        y[w, p] = 1
    return y



