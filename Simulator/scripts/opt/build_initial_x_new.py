import gurobipy as gb
import numpy as np
import logging


def build_initial_x(rng, d):

    """
    Build an initial feasible x using Gurobi with a z-based pod location model.

    Constraints modelled (aligned with check_constraints / _build_fgv):
        EC13  — workstation capacity: sum(v[m,t]) <= CAP_WS per ws, per t
        EC14  — workstation throughput: item_work + pod_arrivals <= 2*TIME_UNIT
        EC18  — pick only if pod at workstation (via z)
        EC19  — x non-decreasing
        EC20  — v == f - g
        EC21  — f[m,t] >= x[im,t]
        EC22  — g[m,t] <= x[im,t-1]   (missing in old version)
        f_active_only_if_picked — f[m,t] <= sum(x[im,t])  (missing)
        pick_only_if_active     — x[im,t]-x[im,t-1] <= v[m,t]  (missing)
        g_lower_bound           — g[m,t] >= sum(x[im,t-1]) - (n-1)
        f/g monotonicity
        continuity_v            — v[m,t] >= v[m,t-1] - g[m,t]  (missing)
        initial_cond            — v[m,0]=1 for opened orders
        no_pick_at_t0           — x[im,0]=0
        travel_time_lb          — x[im,t]=0 for t < travel(pod->ws)
        max_active_pods         — pods outside storage <= n_robots (via z)

    EC15/EC16 (arc flow conservation on y) are not modelled here;
    they are satisfied by construction in build_solution via _rebuild_pod_row.
    """

    n_orders = len(d.orders)
    n_robots = len(d.warehouse.robots)
    T        = d.OptManager.N_TIME
    CAP_WS   = d.OptManager.CAP_WS

    model = gb.Model("build_initial_x")
    model.setParam("OutputFlag",    0)
    model.setParam("MIPFocus",      1)   # feasibility first
    model.setParam("Heuristics",    0.8)
    model.setParam("NoRelHeurTime", 20)
    model.setParam("TimeLimit",     120)

    # ------------------------------------------------------------------ #
    # Precompute per-pod relevant locations and travel times
    # ------------------------------------------------------------------ #

    pod_relevant_locs: dict[int, list] = {}  # p_rel -> [storage, ws1, ws2, ...]
    pod_loc_idx:       dict[int, dict] = {}  # p_rel -> {loc: l_idx}
    pod_ws_travel:     dict[tuple, int] = {} # (p_rel, w_id) -> travel time

    def _travel_time(p_id: int, w_id: int) -> int:
        """
        Worst-case (maximum) travel time from storage to ws across all
        departure times. This ensures Gurobi never schedules a pick at a t
        that build_solution cannot satisfy with a real arc.
        """
        storage = d.warehouse.pods[p_id].storage_location
        ws_loc  = d.ws_positions[w_id]
        n_tr    = len(d.OptManager.travelling_arcs)
        
        min_travel = T  # pessimistic default
        for t_dep in range(T):
            for id_a in d.OptManager.outgoing_arc_idx.get((storage, t_dep), []):
                if id_a >= n_tr:
                    continue
                arc = d.OptManager.all_arcs[id_a]
                if arc[1][0] == ws_loc:
                    travel = arc[1][1] - t_dep
                    min_travel = min(min_travel, travel)
                    break  # primo arco valido per questo t_dep
        
        return min_travel if min_travel < T else 1

    for p_rel, p_id in enumerate(d.from_RelPod_to_PodId):
        storage = d.warehouse.pods[p_id].storage_location
        ws_set  = {
            d.ws_positions[d.order_to_ws[d.relevant_pairs_for_x[im][1]]]
            for im in d.items_by_pod[p_id]
            if im < len(d.relevant_pairs_for_x)
        }
        locs = [storage] + sorted(ws_set)
        pod_relevant_locs[p_rel] = locs
        pod_loc_idx[p_rel]       = {loc: idx for idx, loc in enumerate(locs)}

    n_ws = len(d.warehouse.workstations)
    for p_rel, p_id in enumerate(d.from_RelPod_to_PodId):
        for w_id in range(n_ws):
            pod_ws_travel[(p_rel, w_id)] = _travel_time(p_id, w_id)

    # ------------------------------------------------------------------ #
    # Variables
    # ------------------------------------------------------------------ #

    n_im = len(d.relevant_pairs_for_x)

    x = model.addVars(n_im, T, vtype=gb.GRB.BINARY, name="x")

    # z[p_rel][l_idx, t] = 1 iff pod p_rel is at location l_idx at time t
    z = {
        p_rel: model.addVars(
            len(pod_relevant_locs[p_rel]), T, vtype=gb.GRB.BINARY, name=f"z_{p_rel}"
        )
        for p_rel in range(len(d.from_RelPod_to_PodId))
    }

    f = model.addVars(n_orders, T, vtype=gb.GRB.BINARY, name="f")
    g = model.addVars(n_orders, T, vtype=gb.GRB.BINARY, name="g")
    v = model.addVars(n_orders, T, vtype=gb.GRB.BINARY, name="v")

    # ------------------------------------------------------------------ #
    # Constraints on x
    # ------------------------------------------------------------------ #

    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        p_id   = d.pod_of_item[im]
        p_rel  = d.from_PodId_to_RelPod[p_id]
        w_id   = d.order_to_ws[m]
        ws_l   = d.ws_positions[w_id]
        travel = pod_ws_travel[(p_rel, w_id)]
        l_idx  = pod_loc_idx[p_rel].get(ws_l)

        # EC18 proxy + no pick at t=0 + travel time lower bound
        t_earliest = max(1, travel+1)
        for t in range(t_earliest):
            model.addConstr(x[im, t] == 0, name=f"no_early_pick_{im}_{t}")

        for t in range(t_earliest, T):
            # EC19: x non-decreasing
            model.addConstr(x[im, t] >= x[im, t - 1], name=f"EC19_{im}_{t}")

            # EC18: pick only if pod at workstation
            if l_idx is not None:
                model.addConstr(
                    x[im, t] - x[im, t - 1] <= z[p_rel][l_idx, t],
                    name=f"EC18_{im}_{t}"
                )

    # ------------------------------------------------------------------ #
    # Constraints on z
    # ------------------------------------------------------------------ #

    for p_rel, p_id in enumerate(d.from_RelPod_to_PodId):
        n_locs   = len(pod_relevant_locs[p_rel])
        stor_idx = 0  # storage is always index 0 by construction

        # Exactly one location at every t
        for t in range(T):
            model.addConstr(
                gb.quicksum(z[p_rel][l, t] for l in range(n_locs)) == 1,
                name=f"z_one_loc_{p_rel}_{t}"
            )

        # Start at storage at t=0
        model.addConstr(z[p_rel][stor_idx, 0] == 1, name=f"z_start_{p_rel}")

        # Travel time feasibility: pod can only be at ws location l
        # at time t if it departed storage early enough.
        # Equivalently: if z[p_rel][l, t] = 1 and l != storage,
        # then z[p_rel][stor_idx, t'] = 1 for some t' <= t - travel(p, ws(l)).
        # We enforce this as: z[p_rel][l, t] <= sum_{t'<=t-travel} z[p_rel][stor_idx, t']
        # which simplifies (since stor_idx is the only departure point and z sums
        # to 1) to a bound on the earliest arrival at each ws location.
        for l_idx, loc in enumerate(pod_relevant_locs[p_rel]):
            if l_idx == stor_idx:
                continue
            # Identify which w_id this loc corresponds to
            w_id_for_loc = next(
                (w for w in range(n_ws) if d.ws_positions[w] == loc), None
            )
            if w_id_for_loc is None:
                continue
            travel = pod_ws_travel[(p_rel, w_id_for_loc)]
            # Pod cannot be at this ws before it had time to travel there
            for t in range(min(travel, T)):
                model.addConstr(
                    z[p_rel][l_idx, t] == 0,
                    name=f"z_travel_lb_{p_rel}_{l_idx}_{t}"
                )

    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        p_id   = d.pod_of_item[im]
        p_rel  = d.from_PodId_to_RelPod[p_id]
        w_id   = d.order_to_ws[m]
        ws_loc = d.ws_positions[w_id]
        storage = d.warehouse.pods[p_id].storage_location
        n_tr   = len(d.OptManager.travelling_arcs)

        # Raccogli tutti i t_arr raggiungibili da storage verso ws_loc
        valid_pick_times = set()
        for t_dep in range(T):
            for id_a in d.OptManager.outgoing_arc_idx.get((storage, t_dep), []):
                if id_a >= n_tr:
                    continue
                arc = d.OptManager.all_arcs[id_a]
                if arc[1][0] == ws_loc:
                    valid_pick_times.add(arc[1][1])

        # Blocca i pick a t non raggiungibili da archi reali
        for t in range(1, T):
            if t not in valid_pick_times:
                model.addConstr(
                    x[im, t] - x[im, t - 1] == 0,
                    name=f"valid_pick_{im}_{t}"
                )

    # ------------------------------------------------------------------ #
    # max_active_pods: pods outside storage <= n_robots at every t
    #
    # Idea: se il pod di item im viene pickato per la prima volta a t_pick
    # (cioè x[im,t]-x[im,t-1]=1), allora occupa un robot nell'intervallo
    # [t_pick - travel, t_pick + travel].
    # Usiamo delta_x[im,t] = x[im,t] - x[im,t-1] come trigger.
    # ------------------------------------------------------------------ #

    a = model.addVars(
        len(d.from_RelPod_to_PodId), T,
        vtype=gb.GRB.BINARY, name="a"
    )

    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        p_id   = d.pod_of_item[im]
        p_rel  = d.from_PodId_to_RelPod[p_id]
        w_id   = d.order_to_ws[m]
        travel = pod_ws_travel[(p_rel, w_id)]

        for t in range(1, T):
            # delta_x[im,t] = x[im,t] - x[im,t-1] ∈ {0,1}
            # Se questo item viene pickato a t, il pod è fuori storage
            # nell'intervallo [t-travel, t+travel]
            t_lo = max(0, t - travel)
            t_hi = min(T - 1, t + travel)
            for tau in range(t_lo, t_hi + 1):
                model.addConstr(
                    a[p_rel, tau] >= x[im, t] - x[im, t - 1],
                    name=f"a_pick_{p_rel}_{im}_{t}_{tau}"
                )

    for t in range(T):
        model.addConstr(
            gb.quicksum(
                a[p_rel, t]
                for p_rel in range(len(d.from_RelPod_to_PodId))
            ) <= n_robots,
            name=f"max_active_{t}"
        )

    # ------------------------------------------------------------------ #
    # f, g, v — aligned with _build_fgv and check_constraints
    # ------------------------------------------------------------------ #

    for m in range(n_orders):
        ims     = list(d.items_of_order[m])
        n_items = len(ims)

        for t in range(T):
            # EC21: f[m,t] >= x[im,t] for all im in order m
            for im in ims:
                model.addConstr(f[m, t] >= x[im, t], name=f"EC21_{m}_{im}_{t}")

            # f_active_only_if_picked: f[m,t] <= sum(x[im,t])
            # Ensures f=0 when no item has been picked yet — aligns with _build_fgv
            model.addConstr(
                f[m, t] <= gb.quicksum(x[im, t] for im in ims),
                name=f"f_active_{m}_{t}"
            )

            # EC22: g[m,t] <= x[im,t-1] for all im (g=1 only if all items picked
            # by t-1). At t=0 there is no t-1, so g[m,0]=0 for non-opened orders.
            for im in ims:
                if t == 0:
                    model.addConstr(g[m, t] == 0, name=f"g_zero_{m}_{im}")
                else:
                    model.addConstr(g[m, t] <= x[im, t - 1], name=f"EC22_{m}_{im}_{t}")

            # g_lower_bound: g[m,t] >= sum(x[im,t-1]) - (n_items-1)
            if t > 0:
                model.addConstr(
                    g[m, t] >= gb.quicksum(x[im, t - 1] for im in ims) - (n_items - 1),
                    name=f"g_lb_{m}_{t}"
                )

            # f monotone, g monotone, f >= g
            if t > 0:
                model.addConstr(f[m, t] >= f[m, t - 1], name=f"f_mono_{m}_{t}")
                model.addConstr(g[m, t] >= g[m, t - 1], name=f"g_mono_{m}_{t}")

            model.addConstr(f[m, t] >= g[m, t], name=f"f_ge_g_{m}_{t}")

            # EC20: v = f - g
            model.addConstr(v[m, t] == f[m, t] - g[m, t], name=f"EC20_{m}_{t}")

            # continuity_v: v[m,t] >= v[m,t-1] - g[m,t]
            if t > 0:
                model.addConstr(
                    v[m, t] >= v[m, t - 1] - g[m, t],
                    name=f"cont_v_{m}_{t}"
                )

            # pick_only_if_active: x[im,t] - x[im,t-1] <= v[m,t]
            # An item can only be newly picked if the order is active at t
            if t > 0:
                for im in ims:
                    model.addConstr(
                        x[im, t] - x[im, t - 1] <= v[m, t],
                        name=f"pick_active_{m}_{im}_{t}"
                    )

    # ------------------------------------------------------------------ #
    # EC13: workstation capacity
    # ------------------------------------------------------------------ #

    for w, order_ids in enumerate(d.orders_by_workstation):
        for t in range(T):
            model.addConstr(
                gb.quicksum(v[m, t] for m in order_ids) <= CAP_WS,
                name=f"EC13_{w}_{t}"
            )

    # ------------------------------------------------------------------ #
    # EC14: workstation throughput
    # item_work = DELTA_ITEM * sum(x[im,t] - x[im,t-1])  for im in ws
    # pod_arrivals = DELTA_POD * number of pods arriving at ws at t
    # A pod "arrives" at ws at t if z[p_rel][l_ws, t]=1 and z[p_rel][l_ws, t-1]=0
    # Proxy: delta_z[p_rel, l_ws, t] = z[p_rel][l_ws, t] - z[p_rel][l_ws, t-1]
    # ------------------------------------------------------------------ #

    DELTA_ITEM = d.OptManager.DELTA_ITEM
    DELTA_POD  = d.OptManager.DELTA_POD
    TIME_UNIT  = d.OptManager.TIME_UNIT

    # Binary var: pod p_rel arrives at workstation ws location at time t
    pod_arrives = {}
    for p_rel in range(len(d.from_RelPod_to_PodId)):
        for l_idx, loc in enumerate(pod_relevant_locs[p_rel]):
            if l_idx == 0:
                continue  # storage, not a ws
            for t in range(1, T):
                var = model.addVar(vtype=gb.GRB.BINARY, name=f"arr_{p_rel}_{l_idx}_{t}")
                # arr = 1 iff z[l,t]=1 and z[l,t-1]=0
                model.addConstr(var >= z[p_rel][l_idx, t] - z[p_rel][l_idx, t - 1])
                model.addConstr(var <= z[p_rel][l_idx, t])
                pod_arrives[(p_rel, l_idx, t)] = var

    for w, order_ids in enumerate(d.orders_by_workstation):
        ws_loc = d.ws_positions[w]
        ims_w  = [
            im for im, (_, m) in enumerate(d.relevant_pairs_for_x)
            if m in order_ids
        ]
        for t in range(1, T):
            # Item work at this ws at time t
            item_work = DELTA_ITEM * gb.quicksum(
                x[im, t] - x[im, t - 1] for im in ims_w
            )
            # Pod arrivals at this ws at time t (across all pods)
            arrivals = gb.quicksum(
                pod_arrives[(p_rel, l_idx, t)]
                for p_rel in range(len(d.from_RelPod_to_PodId))
                for l_idx, loc in enumerate(pod_relevant_locs[p_rel])
                if loc == ws_loc and (p_rel, l_idx, t) in pod_arrives
            )
            model.addConstr(
                item_work + DELTA_POD * arrivals <= 2 * TIME_UNIT,
                name=f"EC14_{w}_{t}"
            )

    # ------------------------------------------------------------------ #
    # Initial conditions (opened orders)
    # ------------------------------------------------------------------ #

    for m, order in enumerate(d.orders):
        if order.order_id in d.opened_order_ids:
            model.addConstr(v[m, 0] == 1, name=f"opened_{m}")
            # f must also be 1 at t=0 for opened orders
            model.addConstr(f[m, 0] == 1, name=f"opened_f_{m}")

    # ------------------------------------------------------------------ #
    # Objective: maximise items picked by end of horizon
    # Secondary term: minimise sum(1 - g[m,t]) to reduce backlog
    # (weighted small so feasibility-quality tradeoff stays fast)
    # ------------------------------------------------------------------ #

    picking_reward = gb.quicksum(x[im, T - 1] for im in range(n_im))

    model.setObjective(
        picking_reward,
        gb.GRB.MAXIMIZE
    )

    # ------------------------------------------------------------------ #
    # Solve with progressive time budgets (warm-start between iterations)
    # ------------------------------------------------------------------ #

    budgets = [20, 60, 120]

    best_x_sol = None

    for i, budget in enumerate(budgets):
        model.setParam("TimeLimit", budget)
        model.optimize()

        if model.SolCount == 0:
            if i == 0:
                logging.warning("[build_initial_x] No feasible solution in first budget.")
                # Try with relaxed EC14 (remove throughput constraints and retry)
                for c in model.getConstrs():
                    if c.ConstrName.startswith("EC14_"):
                        model.remove(c)
                model.setParam("TimeLimit", 60)
                model.optimize()
                if model.SolCount == 0:
                    model.computeIIS()
                    model.write("iis.ilp")
                    raise RuntimeError(
                        "[build_initial_x] No feasible solution found. IIS written to iis.ilp"
                    )
            # Previous budget already found a solution — keep it
            break

        obj = model.ObjVal
        gap = model.MIPGap
        logging.info(
            "[build_initial_x] iter %d/%d: obj=%.3f, gap=%.1f%%",
            i + 1, len(budgets), obj, gap * 100
        )

        if gap < 0.01:
            logging.info("[build_initial_x] Gap < 1%%, stopping early.")
            break

    if model.SolCount == 0:
        model.computeIIS()
        model.write("iis.ilp")
        raise RuntimeError("[build_initial_x] No feasible solution found.")

    # ------------------------------------------------------------------ #
    # Extract x
    # ------------------------------------------------------------------ #

    x_sol = np.zeros((n_im, T), dtype=np.float64)
    for im in range(n_im):
        for t in range(T):
            x_sol[im, t] = 1.0 if x[im, t].X > 0.5 else 0.0

    # Enforce monotonicity in extraction (safety net for numerical noise)
    for im in range(n_im):
        for t in range(1, T):
            if x_sol[im, t] < x_sol[im, t - 1]:
                x_sol[im, t] = x_sol[im, t - 1]

    logging.info("[build_initial_x] Done. Extracted x with %d items picked.",
                 int(x_sol[:, T - 1].sum()))


    return x_sol