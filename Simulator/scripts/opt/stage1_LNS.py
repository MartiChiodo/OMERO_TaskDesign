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
ORDERS_INIT   = 30      # orders freed per destroy at the start
ORDERS_MAX    = 80     # max orders freed
TIME_BUDGET   = 30.0    # total seconds
MIP_TIME      = 5.0     # seconds per repair solve
MIP_GAP       = 0.02
PATIENCE      = 3       # stalls before growing the hole
SMALL_GAIN    = 0.3     # a gain below this counts as marginal
MAX_VALID     = 5       # full checks before trusting the MILP
RECHECK_EVERY = 25
HOLE_FRAC     = 0.5
STALL_STOP    = 10      # stop after this many stalls once at max size
SEED          = 42

GROW_STALL    = 10      # grow the hole by this many orders when stalling
GROW_MARGINAL = 5       # grow by this many when the gain is only marginal
SHRINK_OVER   = 5       # shrink by this many when the hole overflows the cap


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
           x_incumbent, z_incumbent, y_incumbent, freed_orders, freed_items,
           sku_per_order, total_load, cost_matrix, env,
           time_limit=MIP_TIME, mip_gap=MIP_GAP):

    # reduced problem: only the freed part is variable, everything else enters
    # as a constant read from the incumbent, so the model scales with the hole
    # pods each freed item may use, and the set of pods the hole touches
    admissible_pods = {
        item: list(OptManager.pod_indices_by_sku[relevant_pairs_for_x[item][0]])
        for item in freed_items
    }
    touched_pods = sorted({int(p) for pods in admissible_pods.values() for p in pods})
    touched_pods_set = set(touched_pods)

    mdl = gp.Model("repair1", env=env)
    mdl.Params.OutputFlag = 0
    mdl.Params.TimeLimit = time_limit
    mdl.Params.MIPGap = mip_gap

    # decision variables, only for the freed part
    assign_var = mdl.addVars(sorted(freed_orders), range(n_w),
                             vtype=GRB.BINARY, name="assign")   # order -> workstation
    pick_var = {}                                               # item  -> pod
    for item in freed_items:
        for pod in admissible_pods[item]:
            pick_var[item, pod] = mdl.addVar(vtype=GRB.BINARY)
            pick_var[item, pod].Start = float(x_incumbent[item, pod])
    for order in freed_orders:
        for w in range(n_w):
            assign_var[order, w].Start = float(z_incumbent[order, w])

    visit_var = {}                                             # pod visited at ws
    for pod in touched_pods:
        for w in range(n_w):
            visit_var[w, pod] = mdl.addVar(lb=0.0, ub=1.0)
            visit_var[w, pod].Start = float(y_incumbent[w, pod])

    # CONSTANT injection 1: a FIXED item that uses a touched pod still forces its
    # visit. Since that item's pick/assign are constants (not variables), EC4
    # cannot force the visit for us, so we pin the visit lower bound by hand.
    for item in range(len(relevant_pairs_for_x)):
        if item in freed_items:
            continue
        pod_used = int(np.argmax(x_incumbent[item]))
        if x_incumbent[item, pod_used] > 0.5 and pod_used in touched_pods_set:
            _, order = relevant_pairs_for_x[item]
            ws_fixed = int(np.argmax(z_incumbent[order]))
            visit_var[ws_fixed, pod_used].lb = 1.0

    def pick_value(item, pod):
        # a variable if the item is freed, otherwise the incumbent constant
        return pick_var[item, pod] if (item, pod) in pick_var \
            else float(x_incumbent[item, pod])

    # EC1 each freed order goes to exactly one workstation
    for order in freed_orders:
        mdl.addConstr(gp.quicksum(assign_var[order, w] for w in range(n_w)) == 1.0)

    # EC2 each freed item is picked from exactly one admissible pod
    for item in freed_items:
        mdl.addConstr(gp.quicksum(pick_var[item, pod]
                                  for pod in admissible_pods[item]) == 1.0)

    # EC4 if a freed item is picked from pod at a workstation, that pod is visited
    for item in freed_items:
        _, order = relevant_pairs_for_x[item]
        for pod in admissible_pods[item]:
            for w in range(n_w):
                mdl.addConstr(
                    visit_var[w, pod] >= pick_var[item, pod] + assign_var[order, w] - 1.0)

    # EC5 workload balance. CONSTANT injection 2: the fixed orders contribute a
    # constant load per workstation (total minus the freed orders' incumbent
    # load); only the freed orders add variable terms.
    sku_total = int(sku_per_order.sum())
    load_lo = np.floor(sku_total / n_w * 0.8)
    load_hi = np.ceil(sku_total / n_w * 1.2)
    fixed_load = total_load.copy()
    for order in freed_orders:
        fixed_load[int(np.argmax(z_incumbent[order]))] -= sku_per_order[order]
    for w in range(n_w):
        load = fixed_load[w] + gp.quicksum(
            sku_per_order[order] * assign_var[order, w] for order in freed_orders)
        mdl.addConstr(load <= load_hi)
        mdl.addConstr(load >= load_lo)

    # objective: minimise pod visits weighted by distance, over the freed visits
    mdl.setObjective(
        gp.quicksum(cost_matrix[w, pod] * visit_var[w, pod]
                    for pod in touched_pods for w in range(n_w)),
        GRB.MINIMIZE)

    try:
        mdl.optimize()
        if mdl.SolCount == 0:
            return None
        # splice the freed variables back onto a copy of the incumbent
        x_new = x_incumbent.copy()
        for item in freed_items:
            x_new[item, :] = 0.0
            for pod in admissible_pods[item]:
                if pick_var[item, pod].X > 0.5:
                    x_new[item, pod] = 1.0
        z_new = z_incumbent.copy()
        for order in freed_orders:
            z_new[order, :] = 0.0
            for w in range(n_w):
                if assign_var[order, w].X > 0.5:
                    z_new[order, w] = 1.0
        y_new = y_incumbent.copy()
        for pod in touched_pods:
            for w in range(n_w):
                y_new[w, pod] = 1.0 if visit_var[w, pod].X > 0.5 else 0.0
        return x_new, z_new, y_new
    finally:
        mdl.dispose()          # release the model (and its WLS session) each iteration


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

    # one shared environment for the whole run, so the job keeps a single WLS
    # session instead of opening one per repair
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
      while time.perf_counter() - t0 < time_budget:
          it += 1

          # grow when stalling, here so no gain iterations count too
          if stall >= PATIENCE and k < ORDERS_MAX:
              k = min(ORDERS_MAX, k + GROW_STALL)
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
              k = max(1, k - SHRINK_OVER)
              continue

          res = repair(orders, orders_items, relevant_pairs_for_x, OptManager, state, n_w,
                       x_best, z_best, y_best, freed_orders, freed_items,
                       sku_per_order, total_load, c_mat, env)
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
          logging.info("[lns_stage1] iter %d | objective %.4f | improved by %.4f "
                       "| hole %d orders | %.0fs elapsed",
                       it, best, delta, k, time.perf_counter() - t0)
          if delta < SMALL_GAIN and k < ORDERS_MAX:
              k = min(ORDERS_MAX, k + GROW_MARGINAL)

      ok, _ = check_constraints(orders, orders_items, OptManager,
                                relevant_pairs_for_x, x_best, y_best, z_best, fixed_orders)
      print(f"[lns_stage1] done | {time.perf_counter() - t0:.1f}s, {it} iterations "
            f"| final objective {best:.4f} | feasible {ok}")
      logging.info("[lns_stage1] done | %.1fs, %d iterations | final objective %.4f "
                   "| feasible %s", time.perf_counter() - t0, it, best, ok)
    finally:
        env.dispose()   # close the shared WLS session
    return x_best, z_best