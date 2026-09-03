from __future__ import annotations
import time
import numpy as np
import logging
import gurobipy as gp
from gurobipy import GRB

from .local_search_stage2 import (
    _build_fgv, compute_objective, check_constraints, build_solution,
)
from .build_initial_x_stage2 import build_initial_x

# config
PODS_INIT     = 10      # pods freed per destroy at the start
PODS_MAX      = 50      # max pods freed
TIME_BUDGET   = 400.0   # total seconds
MIP_TIME      = 30.0    # seconds per repair solve
MIP_GAP       = 0.02
PATIENCE      = 3       # stalls before growing the hole
SMALL_GAIN    = 0.3     # a gain below this counts as marginal
MAX_VALID     = 5       # full checks before trusting the MILP
RECHECK_EVERY = 25
HOLE_FRAC     = 0.35
STALL_STOP    = 10      # stop after this many stalls once at max size
SEED          = 42

GROW_STALL    = 5       # grow the hole by this many pods when stalling
GROW_MARGINAL = 3       # grow by this many when the gain is only marginal
SHRINK_OVER   = 3       # shrink by this many when the hole overflows the cap


def _pod_subnetwork(d):
    # cache, per pod, the arcs and nodes of its small subnetwork
    cache = getattr(d, "_pod_subnet_cache", None)
    if cache is not None:
        return cache
    all_arcs = d.OptManager.all_arcs
    pod_arcs, pod_arcs_set, pod_nodes = {}, {}, {}
    for p_id in d.from_RelPod_to_PodId:
        storage = d.warehouse.pods[p_id].storage_location
        locs = {storage}
        for im in d.items_by_pod[p_id]:
            _, m = d.relevant_pairs_for_x[int(im)]
            locs.add(d.ws_positions[d.order_to_ws[m]])
        arcs = [a for a in range(len(all_arcs))
                if all_arcs[a][0][0] in locs and all_arcs[a][1][0] in locs]
        pod_arcs[p_id] = arcs
        pod_arcs_set[p_id] = set(arcs)
        pod_nodes[p_id] = [nd for nd in d.nodes if nd[0] in locs]
    cache = (pod_arcs, pod_arcs_set, pod_nodes)
    d._pod_subnet_cache = cache
    return cache


def _loads(d, y, pods_rel, ws_loc_to_w, n_travel, T):
    # robot use per t and pod arrivals per (w, t) for a set of pods
    active = np.zeros(T, dtype=float)
    arr = {}
    all_arcs = d.OptManager.all_arcs
    for p_rel in pods_rel:
        p_id = d.from_RelPod_to_PodId[p_rel]
        storage = d.warehouse.pods[p_id].storage_location
        for a in np.where(y[p_rel] > 0.5)[0]:
            (sl, st), (dl, dt) = all_arcs[a]
            if sl != storage:
                for t in range(st, min(dt, T)):
                    active[t] += 1
            if a < n_travel and dl in ws_loc_to_w:
                key = (ws_loc_to_w[dl], dt)
                arr[key] = arr.get(key, 0) + 1
    return active, arr


def destroy_pods(d, rng, k):
    # free k random pods, their items and the orders those items touch
    n_pods = len(d.from_RelPod_to_PodId)
    seed_rel = rng.choice(n_pods, size=min(k, n_pods), replace=False)
    freed_pods = {d.from_RelPod_to_PodId[int(r)] for r in np.atleast_1d(seed_rel)}
    freed_items = set()
    for p in freed_pods:
        for im in d.items_by_pod[p]:
            freed_items.add(int(im))
    affected = {int(d.relevant_pairs_for_x[im][1]) for im in freed_items}
    return freed_items, freed_pods, affected


def repair(d, x_incumbent, y_incumbent, f_incumbent, g_incumbent, v_incumbent,
           freed_items, freed_pods, affected, total_active, total_arr,
           ws_loc_to_w, precomp, env, time_limit=MIP_TIME, mip_gap=MIP_GAP):

    # reduced problem: only the freed items, their orders and the freed pods are
    # variables, everything else enters as a constant read from the incumbent,
    # so the model scales with the hole
    x_inc, y_inc = x_incumbent, y_incumbent
    f_inc, g_inc, v_inc = f_incumbent, g_incumbent, v_incumbent
    T = d.OptManager.N_TIME
    n_travel = len(d.OptManager.travelling_arcs)
    n_robots = len(d.warehouse.robots)
    pod_arcs, pod_arcs_set, pod_nodes = precomp

    mdl = gp.Model("repair2", env=env)
    mdl.Params.OutputFlag = 0
    mdl.Params.TimeLimit = time_limit
    mdl.Params.MIPGap = mip_gap

    # decision variables, only for the freed part
    x = mdl.addVars(sorted(freed_items), range(T), vtype=GRB.BINARY, name="x")
    f = mdl.addVars(sorted(affected), range(T), lb=0, ub=1, name="f")
    g = mdl.addVars(sorted(affected), range(T), lb=0, ub=1, name="g")
    v = mdl.addVars(sorted(affected), range(T), lb=0, ub=1, name="v")

    y = {}
    for p in freed_pods:
        p_rel = d.from_PodId_to_RelPod[p]
        for a in pod_arcs[p]:
            y[(p, a)] = mdl.addVar(vtype=GRB.BINARY)
            y[(p, a)].Start = float(y_inc[p_rel, a])
    for im in freed_items:
        for t in range(T):
            x[im, t].Start = float(x_inc[im, t])

    def x_val(im, t):
        # a variable if the item is freed, otherwise the incumbent constant
        return x[im, t] if im in freed_items else float(x_inc[im, t])

    def v_val(m, t):
        # a variable if the order is affected, otherwise the incumbent constant
        return v[m, t] if m in affected else float(v_inc[m, t])

    for im in freed_items:                                     # EC15 x non decreasing
        for t in range(1, T):
            mdl.addConstr(x[im, t] >= x[im, t-1])
    for m in affected:
        ims = list(d.items_of_order[m])
        n_it = int(d.n_items_per_order[m])
        for t in range(T):
            mdl.addConstr(v[m, t] == f[m, t] - g[m, t])        # EC17
        for t in range(1, T):
            mdl.addConstr(f[m, t] >= f[m, t-1])                # EC20 f
            mdl.addConstr(g[m, t] >= g[m, t-1])                # EC20 g
            mdl.addConstr(v[m, t] >= v[m, t-1] - g[m, t])      # EC21
            mdl.addConstr(g[m, t] >= gp.quicksum(x_val(im, t-1) for im in ims) - (n_it - 1))  # EC22
        for im in ims:                                         # EC16 EC18 EC19
            for t in range(T):
                mdl.addConstr(f[m, t] >= x_val(im, t))
            for t in range(1, T):
                mdl.addConstr(x_val(im, t) - x_val(im, t-1) <= v[m, t])
                mdl.addConstr(g[m, t] <= x_val(im, t-1))
        if d.orders[m].order_id in d.opened_order_ids:         # EC23
            mdl.addConstr(v[m, 0] == 1.0)
        else:
            for t in range(T):
                mdl.addConstr(f[m, t] <= gp.quicksum(x_val(im, t) for im in ims))

    # EC10 workstation capacity, only where an affected order sits
    for w, order_ids in enumerate(d.orders_by_workstation):
        ids = list(order_ids)
        if not any(m in affected for m in ids):
            continue
        for t in range(T):
            mdl.addConstr(gp.quicksum(v_val(m, t) for m in ids) <= d.OptManager.CAP_WS)

    # routing for freed pods inside their subnetwork
    for p in freed_pods:
        storage = d.warehouse.pods[p].storage_location
        aset = pod_arcs_set[p]
        out0 = [a for a in d.outgoing_arc_idx.get((storage, 0), []) if a in aset]
        mdl.addConstr(gp.quicksum(y[(p, a)] for a in out0) == 1.0)   # EC12
        for node in pod_nodes[p]:                                    # EC13
            if node[1] in (0, T - 1):
                continue
            inc = [a for a in d.incoming_arc_idx.get(node, []) if a in aset]
            out = [a for a in d.outgoing_arc_idx.get(node, []) if a in aset]
            if inc or out:
                mdl.addConstr(gp.quicksum(y[(p, a)] for a in inc)
                              - gp.quicksum(y[(p, a)] for a in out) == 0.0)

    # EC14 the pod is at the workstation when its item is picked
    for im in freed_items:
        p = d.pod_of_item[im]
        aset = pod_arcs_set[p]
        _, m = d.relevant_pairs_for_x[int(im)]
        ws_p = d.ws_positions[d.order_to_ws[m]]
        for t in range(T):
            inc = [a for a in d.incoming_arc_idx.get((ws_p, t), []) if a in aset]
            dx = x_val(im, t) if t == 0 else (x_val(im, t) - x_val(im, t-1))
            mdl.addConstr(gp.quicksum(y[(p, a)] for a in inc) >= dx)

    # constant injection: the fixed pods contribute constant loads (total minus
    # the freed pods), used in EC11 and EC24 below
    freed_rel = [d.from_PodId_to_RelPod[p] for p in freed_pods]
    freed_active, freed_arr = _loads(d, y_inc, freed_rel, ws_loc_to_w, n_travel, T)
    const_active = total_active - freed_active

    # EC11 item work plus pod arrivals per (w, t)
    for w, order_ids in enumerate(d.orders_by_workstation):
        ws_p = d.ws_positions[w]
        ims_w = [im for im, (_, m) in enumerate(d.relevant_pairs_for_x)
                 if m in set(order_ids)]
        for t in range(1, T):
            item_work = d.OptManager.DELTA_ITEM * gp.quicksum(
                x_val(im, t) - x_val(im, t-1) for im in ims_w)
            travel_in = [a for a in d.incoming_arc_idx.get((ws_p, t), []) if a < n_travel]
            var_arr = gp.quicksum(y[(p, a)] for p in freed_pods for a in travel_in
                                  if a in pod_arcs_set[p])
            const_cnt = total_arr.get((w, t), 0) - freed_arr.get((w, t), 0)
            mdl.addConstr(item_work + d.OptManager.DELTA_POD * (var_arr + const_cnt)
                          <= 2 * d.OptManager.TIME_UNIT)

    # EC24 robots in use per t
    for t in range(T):
        occ = []
        for p in freed_pods:
            storage = d.warehouse.pods[p].storage_location
            for a in pod_arcs[p]:
                (sl, st), (dl, dt) = d.OptManager.all_arcs[a]
                if sl != storage and st <= t < dt:
                    occ.append(y[(p, a)])
        mdl.addConstr(gp.quicksum(occ) + const_active[t] <= n_robots)

    # objective, variable part only
    picking = gp.quicksum(x[im, T-1] for im in freed_items)
    terms = []
    for m in sorted(affected):
        a_m = float(d.arrival_times[m])
        for t in range(T):
            age = (d.current_time + t * d.OptManager.TIME_UNIT - a_m) / d.OptManager.TIME_UNIT
            terms.append(age * (1.0 - g[m, t]))
    mdl.setObjective(picking - 0.1 * gp.quicksum(terms) / d.OptManager.N_TIME, GRB.MAXIMIZE)

    try:
        mdl.optimize()
        if mdl.SolCount == 0:
            return None
        x_new = x_inc.copy()
        for im in freed_items:
            for t in range(T):
                x_new[im, t] = 1.0 if x[im, t].X > 0.5 else 0.0
        y_new = y_inc.copy()
        for p in freed_pods:
            p_rel = d.from_PodId_to_RelPod[p]
            y_new[p_rel, :] = 0.0
            for a in pod_arcs[p]:
                y_new[p_rel, a] = 1.0 if y[(p, a)].X > 0.5 else 0.0
        return x_new, y_new
    finally:
        mdl.dispose()          # release the model (and its WLS session) each iteration


def lns_stage2(d, time_budget=TIME_BUDGET, seed=SEED):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    n_pairs = len(d.relevant_pairs_for_x)
    n_pods = len(d.from_RelPod_to_PodId)
    n_travel = len(d.OptManager.travelling_arcs)
    T = d.OptManager.N_TIME
    ws_loc_to_w = {loc: w for w, loc in enumerate(d.ws_positions)}
    hole_cap = max(40, int(HOLE_FRAC * n_pairs))

    print(f"\n[lns_stage2] start | {n_pairs} item order pairs, {n_pods} pods, "
          f"{len(d.ws_positions)} workstations, {T} time steps | budget {time_budget:.0f}s")
    logging.info("\n[lns_stage2] start | %d item order pairs, %d pods, %d workstations, "
                 "%d time steps | budget %.0fs",
                 n_pairs, n_pods, len(d.ws_positions), T, time_budget)

    precomp = _pod_subnetwork(d)

    # initial solution from the existing pipeline
    x_best = build_initial_x(rng, d)
    _, f_best, g_best, v_best, y_best = build_solution(x_best, d)
    ok, viols, _ = check_constraints((x_best, f_best, g_best, v_best, y_best), d)
    if not ok:
        print(f"[lns_stage2] initial solution is infeasible: {list(viols)}")
        logging.warning("[lns_stage2] initial solution is infeasible: %s", list(viols))
    best = compute_objective(x_best, f_best, g_best, d)
    print(f"[lns_stage2] initial objective {best:.4f}")
    logging.info("[lns_stage2] initial objective %.4f", best)

    total_active, total_arr = _loads(d, y_best, range(n_pods), ws_loc_to_w, n_travel, T)
    k, stall, it = PODS_INIT, 0, 0
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
          if stall >= PATIENCE and k < PODS_MAX:
              k = min(PODS_MAX, k + GROW_STALL)
              stall = 0
          # stop if stuck at the biggest hole
          if stall >= STALL_STOP and k >= PODS_MAX:
              print(f"[lns_stage2] stopping | no improvement after {STALL_STOP} tries "
                    f"at the largest hole ({k} pods)")
              logging.info("[lns_stage2] stopping | no improvement after %d tries "
                           "at the largest hole (%d pods)", STALL_STOP, k)
              break

          freed_items, freed_pods, affected = destroy_pods(d, rng, k)
          if len(freed_items) > hole_cap:
              k = max(1, k - SHRINK_OVER)
              continue

          res = repair(d, x_best, y_best, f_best, g_best, v_best,
                       freed_items, freed_pods, affected,
                       total_active, total_arr, ws_loc_to_w, precomp, env)
          if res is None:
              stall += 1
              continue

          x_new, y_new = res
          f_new, g_new, v_new = _build_fgv(x_new, d)
          obj = compute_objective(x_new, f_new, g_new, d)
          delta = obj - best

          if delta <= 1e-9:
              stall += 1
              continue

          # check the first few accepted moves, then trust the MILP
          need = (not trust) or (since_check >= RECHECK_EVERY)
          if need:
              ok, viols, _ = check_constraints((x_new, f_new, g_new, v_new, y_new), d)
              if not ok:
                  verified, trust = 0, False
                  print(f"[lns_stage2] iter {it} | rejected, the MILP result "
                        f"does not pass the checker: {list(viols)}")
                  logging.warning("[lns_stage2] iter %d | rejected, the MILP result "
                                  "does not pass the checker: %s", it, list(viols))
                  stall += 1
                  continue
              verified += 1
              since_check = 0
              if verified >= MAX_VALID:
                  trust = True

          best = obj
          x_best, f_best, g_best, v_best, y_best = x_new, f_new, g_new, v_new, y_new
          total_active, total_arr = _loads(d, y_best, range(n_pods), ws_loc_to_w, n_travel, T)
          since_check += 1
          stall = 0
          logging.info("[lns_stage2] iter %d | objective %.4f | improved by %.4f "
                       "| hole %d pods | %.0fs elapsed",
                       it, best, delta, k, time.perf_counter() - t0)
          if delta < SMALL_GAIN and k < PODS_MAX:
              k = min(PODS_MAX, k + GROW_MARGINAL)

      ok, viols, _ = check_constraints((x_best, f_best, g_best, v_best, y_best), d)
      print(f"[lns_stage2] done | {time.perf_counter() - t0:.1f}s, {it} iterations "
            f"| final objective {best:.4f} | feasible {ok}")
      logging.info("[lns_stage2] done | %.1fs, %d iterations | final objective %.4f "
                   "| feasible %s", time.perf_counter() - t0, it, best, ok)
    finally:
        env.dispose()   # close the shared WLS session
    return x_best, f_best, g_best, v_best, y_best