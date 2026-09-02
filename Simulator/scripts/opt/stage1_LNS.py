from __future__ import annotations
import time
import numpy as np
import logging
import gurobipy as gp
from gurobipy import GRB

from .local_search_stage1 import (
    build_initial_solution, check_constraints, compute_objective,
    get_fixed_orders, _get_beta, _get_ws_pod_distance_matrix,
)

# config
ORDERS_INIT   = 10      # orders freed per destroy at the start
ORDERS_MAX    = 30      # max orders freed
TIME_BUDGET   = 60.0    # total seconds
MIP_TIME      = 5.0     # seconds per repair solve
MIP_GAP       = 0.02
PATIENCE      = 3       # stalls before growing the hole
SMALL_GAIN    = 0.5     # a gain below this counts as marginal
MAX_VALID     = 5       # full checks before trusting the MILP
RECHECK_EVERY = 25
HOLE_FRAC     = 0.35
STALL_STOP    = 10      # stop after this many stalls once at max size
SEED          = 42


def destroy_orders(rng, k, free_orders, items_of_order):
    # free k random non fixed orders and their items
    chosen = rng.choice(free_orders, size=min(k, len(free_orders)), replace=False)
    freed_orders = {int(m) for m in np.atleast_1d(chosen)}
    freed_items = set()
    for m in freed_orders:
        for im in items_of_order[m]:
            freed_items.add(int(im))
    return freed_orders, freed_items


def repair(orders, orders_items, relevant_pairs_for_x, OptManager, state, n_w,
           x_inc, z_inc, y_inc, freed_orders, freed_items, sku_per_order,
           total_load, c_mat, time_limit=MIP_TIME, mip_gap=MIP_GAP):
    # admissible pods for each freed item, and the pods the hole touches
    adm = {im: list(OptManager.pod_indices_by_sku[relevant_pairs_for_x[im][0]])
           for im in freed_items}
    P = sorted({int(p) for ims in adm.values() for p in ims})
    Pset = set(P)

    mdl = gp.Model("repair1")
    mdl.Params.OutputFlag = 0
    mdl.Params.TimeLimit = time_limit
    mdl.Params.MIPGap = mip_gap

    z = mdl.addVars(sorted(freed_orders), range(n_w), vtype=GRB.BINARY, name="z")
    x = {}
    for im in freed_items:
        for p in adm[im]:
            x[im, p] = mdl.addVar(vtype=GRB.BINARY)
            x[im, p].Start = float(x_inc[im, p])
    for m in freed_orders:
        for w in range(n_w):
            z[m, w].Start = float(z_inc[m, w])

    y = {}
    for p in P:
        for w in range(n_w):
            y[w, p] = mdl.addVar(lb=0.0, ub=1.0)
            y[w, p].Start = float(y_inc[w, p])

    # a fixed item using a touched pod pins that y to one
    for im in range(len(relevant_pairs_for_x)):
        if im in freed_items:
            continue
        p_used = int(np.argmax(x_inc[im]))
        if x_inc[im, p_used] > 0.5 and p_used in Pset:
            _, m = relevant_pairs_for_x[im]
            y[int(np.argmax(z_inc[m])), p_used].lb = 1.0

    def xv(im, p):
        return x[im, p] if (im, p) in x else float(x_inc[im, p])

    # EC1 one workstation per order
    for m in freed_orders:
        mdl.addConstr(gp.quicksum(z[m, w] for w in range(n_w)) == 1.0)
    # EC2 one pod per item
    for im in freed_items:
        mdl.addConstr(gp.quicksum(x[im, p] for p in adm[im]) == 1.0)
    # EC4 visit link
    for im in freed_items:
        _, m = relevant_pairs_for_x[im]
        for p in adm[im]:
            for w in range(n_w):
                mdl.addConstr(y[w, p] >= x[im, p] + z[m, w] - 1.0)
    # EC5 load balance
    sku_total = int(sku_per_order.sum())
    lo = np.floor(sku_total / n_w * 0.8)
    hi = np.ceil(sku_total / n_w * 1.2)
    const_load = total_load.copy()
    for m in freed_orders:
        const_load[int(np.argmax(z_inc[m]))] -= sku_per_order[m]
    for w in range(n_w):
        load = const_load[w] + gp.quicksum(sku_per_order[m] * z[m, w] for m in freed_orders)
        mdl.addConstr(load <= hi)
        mdl.addConstr(load >= lo)

    # minimise visits plus weighted distance
    mdl.setObjective(gp.quicksum(c_mat[w, p] * y[w, p] for p in P for w in range(n_w)),
                     GRB.MINIMIZE)

    mdl.optimize()
    if mdl.SolCount == 0:
        return None

    x_new = x_inc.copy()
    for im in freed_items:
        x_new[im, :] = 0.0
        for p in adm[im]:
            if x[im, p].X > 0.5:
                x_new[im, p] = 1.0
    z_new = z_inc.copy()
    for m in freed_orders:
        z_new[m, :] = 0.0
        for w in range(n_w):
            if z[m, w].X > 0.5:
                z_new[m, w] = 1.0
    y_new = y_inc.copy()
    for p in P:
        for w in range(n_w):
            y_new[w, p] = 1.0 if y[w, p].X > 0.5 else 0.0
    return x_new, z_new, y_new


def lns_stage1(orders, orders_items, relevant_pairs_for_x, OptManager, state, n_w,
               time_budget=TIME_BUDGET, seed=SEED):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    n_orders = len(orders)
    n_pairs = len(relevant_pairs_for_x)
    dist = _get_ws_pod_distance_matrix(state)
    c_mat = 1.0 + _get_beta(state) * dist
    hole_cap = max(20, int(HOLE_FRAC * n_pairs))

    items_of_order = {m: [] for m in range(n_orders)}
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        items_of_order[m].append(im)
    sku_per_order = np.array([len(orders_items[m]) for m in range(n_orders)])

    fixed_orders = get_fixed_orders(orders, state, n_w)
    free_orders = np.array([m for m in range(n_orders) if m not in fixed_orders])

    print(f"\n[lns_stage1] start | {n_orders} orders, {len(state.warehouse.pods)} pods, "
          f"{n_w} workstations ({len(free_orders)} free to move) | budget {time_budget:.0f}s")
    logging.info("\n[lns_stage1] start | %d orders, %d pods, %d workstations "
                 "(%d free to move) | budget %.0fs",
                 n_orders, len(state.warehouse.pods), n_w, len(free_orders), time_budget)

    # initial solution from the existing construction
    for _ in range(10):
        z0, x0, y0 = build_initial_solution(
            orders, orders_items, relevant_pairs_for_x, OptManager, state, rng)
        ok, _ = check_constraints(orders, orders_items, OptManager,
                                  relevant_pairs_for_x, x0, y0, z0, fixed_orders)
        if ok:
            break
    else:
        raise RuntimeError("[lns_stage1] could not build a feasible initial solution")

    x_best, z_best, y_best = x0, z0, y0
    best = compute_objective(y_best, state)
    print(f"[lns_stage1] initial objective {best:.4f}")
    logging.info("[lns_stage1] initial objective %.4f", best)

    total_load = sku_per_order @ z_best
    k, stall, it = ORDERS_INIT, 0, 0
    verified, since_check, trust = 0, 0, False

    while time.perf_counter() - t0 < time_budget:
        it += 1

        # grow when stalling, here so no gain iterations count too
        if stall >= PATIENCE and k < ORDERS_MAX:
            k += 1
            stall = 0
        # stop if stuck at the biggest hole
        if stall >= STALL_STOP and k >= ORDERS_MAX:
            print(f"[lns_stage1] stopping | no improvement after {STALL_STOP} tries "
                  f"at the largest hole ({k} orders)")
            logging.info("[lns_stage1] stopping | no improvement after %d tries "
                         "at the largest hole (%d orders)", STALL_STOP, k)
            break

        freed_orders, freed_items = destroy_orders(rng, k, free_orders, items_of_order)
        if len(freed_items) > hole_cap:
            k = max(1, k - 1)
            continue

        res = repair(orders, orders_items, relevant_pairs_for_x, OptManager, state, n_w,
                     x_best, z_best, y_best, freed_orders, freed_items,
                     sku_per_order, total_load, c_mat)
        if res is None:
            stall += 1
            continue

        x_new, z_new, y_new = res
        obj = compute_objective(y_new, state)
        delta = best - obj

        if delta <= 1e-9:
            stall += 1
            continue

        # check the first few accepted moves, then trust the MILP
        need = (not trust) or (since_check >= RECHECK_EVERY)
        if need:
            ok, _ = check_constraints(orders, orders_items, OptManager,
                                      relevant_pairs_for_x, x_new, y_new, z_new,
                                      fixed_orders)
            if not ok:
                verified, trust = 0, False
                print(f"[lns_stage1] iter {it} | rejected, the MILP result "
                      f"does not pass the checker")
                logging.warning("[lns_stage1] iter %d | rejected, the MILP result "
                                "does not pass the checker", it)
                stall += 1
                continue
            verified += 1
            since_check = 0
            if verified >= MAX_VALID:
                trust = True

        best = obj
        x_best, z_best, y_best = x_new, z_new, y_new
        total_load = sku_per_order @ z_best
        since_check += 1
        stall = 0
        logging.info("[lns_stage1] iter %d | obj = %.4f | delta = - %.4f "
                     "| hole %d orders | %.0fs elapsed",
                     it, best, delta, k, time.perf_counter() - t0)
        if delta < SMALL_GAIN and k < ORDERS_MAX:
            k += 1

    ok, _ = check_constraints(orders, orders_items, OptManager,
                              relevant_pairs_for_x, x_best, y_best, z_best, fixed_orders)
    print(f"[lns_stage1] done | {time.perf_counter() - t0:.1f}s, {it} iterations "
          f"| final objective {best:.4f} | feasible {ok}")
    logging.info("[lns_stage1] done | %.1fs, %d iterations | final objective %.4f "
                 "| feasible %s", time.perf_counter() - t0, it, best, ok)
    return x_best, z_best