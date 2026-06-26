import gurobipy as gb
import numpy as np
import logging


def build_initial_x(rng, d):
    """
    Build an initial feasible x using Gurobi with a z-based pod location model.

    Constraints modelled (aligned with check_constraints / _build_fgv):
        EC13  - workstation capacity: sum(v[m,t]) <= CAP_WS per ws, per t
        EC14  - workstation throughput: item_work + pod_arrivals <= 2*TIME_UNIT
        EC18  - pick only if pod at workstation (via z)
        EC19  - x non-decreasing
        EC20  - v == f - g
        EC21  - f[m,t] >= x[im,t]
        EC22  - g[m,t] <= x[im,t-1]
        f_active_only_if_picked - f[m,t] <= sum(x[im,t])
        pick_only_if_active     - x[im,t]-x[im,t-1] <= v[m,t]
        g_lower_bound           - g[m,t] >= sum(x[im,t-1]) - (n-1)
        f/g monotonicity
        continuity_v            - v[m,t] >= v[m,t-1] - g[m,t]
        initial_cond            - v[m,0]=1 for opened orders
        no_pick_at_t0           - x[im,0]=0
        travel_time_lb          - x[im,t]=0 for t < travel(pod->ws)
        valid_pick_times        - pick only at t reachable by real arcs
        z_transition            - pod movement physically consistent across
                                  all location pairs (prevents teleportation)
        ws_gap                  - picks on different ws of the same pod must
                                  be separated by at least travel(ws1->ws2) slots
        max_active_pods         - pods outside storage <= n_robots
    """

    n_orders   = len(d.orders)
    n_robots   = len(d.warehouse.robots)
    T          = d.OptManager.N_TIME
    CAP_WS     = d.OptManager.CAP_WS
    DELTA_ITEM = d.OptManager.DELTA_ITEM
    DELTA_POD  = d.OptManager.DELTA_POD
    TIME_UNIT  = d.OptManager.TIME_UNIT
    n_ws       = len(d.warehouse.workstations)
    n_rels     = len(d.from_RelPod_to_PodId)
    n_im       = len(d.relevant_pairs_for_x)
    n_tr       = len(d.OptManager.travelling_arcs)

    model = gb.Model("build_initial_x")
    model.setParam("OutputFlag",    0)
    model.setParam("MIPFocus",      1)
    model.setParam("Heuristics",    0.8)
    model.setParam("NoRelHeurTime", 20)

    # ================================================================== #
    # PRECOMPUTATIONS
    # ================================================================== #

    def _travel_time_storage_to_ws(p_id, w_id):
        """Minimum travel time from pod storage to ws over all real arcs."""
        storage    = d.warehouse.pods[p_id].storage_location
        ws_loc     = d.ws_positions[w_id]
        min_travel = T
        for t_dep in range(T):
            for id_a in d.OptManager.outgoing_arc_idx.get((storage, t_dep), []):
                if id_a >= n_tr:
                    continue
                arc = d.OptManager.all_arcs[id_a]
                if arc[1][0] == ws_loc:
                    min_travel = min(min_travel, arc[1][1] - t_dep)
                    break
        return min_travel if min_travel < T else 1

    def _travel_time_between(src_loc, dst_loc):
        """
        Minimum travel time between two locations over real arcs.
        Returns None if no direct arc exists.
        """
        if src_loc == dst_loc:
            return 0
        min_travel = T
        found = False
        for t_dep in range(T):
            for id_a in d.OptManager.outgoing_arc_idx.get((src_loc, t_dep), []):
                if id_a >= n_tr:
                    continue
                arc = d.OptManager.all_arcs[id_a]
                if arc[1][0] == dst_loc:
                    min_travel = min(min_travel, arc[1][1] - t_dep)
                    found = True
                    break
        return min_travel if found else None

    # Per-pod relevant locations: [storage, ws1, ws2, ...]
    pod_relevant_locs = {}   # p_rel -> [storage_loc, ws_loc1, ...]
    pod_loc_idx       = {}   # p_rel -> {loc: l_idx}
    pod_ws_travel     = {}   # (p_rel, w_id) -> int

    for p_rel, p_id in enumerate(d.from_RelPod_to_PodId):
        storage = d.warehouse.pods[p_id].storage_location
        ws_set  = {
            d.ws_positions[d.order_to_ws[d.relevant_pairs_for_x[im][1]]]
            for im in d.items_by_pod[p_id]
            if im < n_im
        }
        locs = [storage] + sorted(ws_set)
        pod_relevant_locs[p_rel] = locs
        pod_loc_idx[p_rel]       = {loc: idx for idx, loc in enumerate(locs)}

    for p_rel, p_id in enumerate(d.from_RelPod_to_PodId):
        for w_id in range(n_ws):
            pod_ws_travel[(p_rel, w_id)] = _travel_time_storage_to_ws(p_id, w_id)

    # Travel time between every pair of locations for each pod.
    # Used by z_transition and ws_gap constraints.
    inter_loc_travel = {}   # (p_rel, l_src, l_dst) -> int or None
    for p_rel in range(n_rels):
        locs   = pod_relevant_locs[p_rel]
        n_locs = len(locs)
        for l_src in range(n_locs):
            for l_dst in range(n_locs):
                if l_src == l_dst:
                    inter_loc_travel[(p_rel, l_src, l_dst)] = 0
                else:
                    inter_loc_travel[(p_rel, l_src, l_dst)] = \
                        _travel_time_between(locs[l_src], locs[l_dst])

    # Valid pick times per item: t values at which a real arc arrives at ws.
    # Prevents EC18 violations caused by picks at unreachable timestamps.
    valid_pick_times_per_im = {}
    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        p_id    = d.pod_of_item[im]
        storage = d.warehouse.pods[p_id].storage_location
        ws_loc  = d.ws_positions[d.order_to_ws[m]]
        vpt     = set()
        for t_dep in range(T):
            for id_a in d.OptManager.outgoing_arc_idx.get((storage, t_dep), []):
                if id_a >= n_tr:
                    continue
                arc = d.OptManager.all_arcs[id_a]
                if arc[1][0] == ws_loc:
                    vpt.add(arc[1][1])
        valid_pick_times_per_im[im] = vpt

    logging.info(
        "[build_initial_x] DELTA_ITEM=%.2f DELTA_POD=%.2f "
        "TIME_UNIT=%.2f EC14_limit=%.2f",
        DELTA_ITEM, DELTA_POD, TIME_UNIT, 2 * TIME_UNIT
    )


    # ================================================================== #
    # VARIABLES
    # ================================================================== #

    x = model.addVars(n_im, T,     vtype=gb.GRB.BINARY, name="x")
    f = model.addVars(n_orders, T, vtype=gb.GRB.BINARY, name="f")
    g = model.addVars(n_orders, T, vtype=gb.GRB.BINARY, name="g")
    v = model.addVars(n_orders, T, vtype=gb.GRB.BINARY, name="v")
    a = model.addVars(n_rels, T,   vtype=gb.GRB.BINARY, name="a")

    z = {
        p_rel: model.addVars(
            len(pod_relevant_locs[p_rel]), T,
            vtype=gb.GRB.BINARY, name=f"z_{p_rel}"
        )
        for p_rel in range(n_rels)
    }

    # ================================================================== #
    # CONSTRAINTS ON x
    # ================================================================== #

    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        p_id   = d.pod_of_item[im]
        p_rel  = d.from_PodId_to_RelPod[p_id]
        w_id   = d.order_to_ws[m]
        ws_l   = d.ws_positions[w_id]
        travel = pod_ws_travel[(p_rel, w_id)]
        l_idx  = pod_loc_idx[p_rel].get(ws_l)
        vpt    = valid_pick_times_per_im[im]

        # No pick at t=0 or before pod can physically arrive
        t_earliest = max(1, travel + 1)
        for t in range(t_earliest):
            model.addConstr(x[im, t] == 0, name=f"no_early_{im}_{t}")

        for t in range(t_earliest, T):
            # EC19: x non-decreasing
            model.addConstr(x[im, t] >= x[im, t - 1], name=f"EC19_{im}_{t}")

            # EC18 proxy: new pick requires pod at ws
            if l_idx is not None:
                model.addConstr(
                    x[im, t] - x[im, t - 1] <= z[p_rel][l_idx, t],
                    name=f"EC18_{im}_{t}"
                )

            # valid_pick_times: block picks at t with no real arc from storage
            if t not in vpt:
                model.addConstr(
                    x[im, t] - x[im, t - 1] == 0,
                    name=f"vpt_{im}_{t}"
                )

    # ================================================================== #
    # WS_GAP: picks on different workstations of the same pod must be
    # separated by at least travel(ws1->ws2) slots in both directions.
    #
    # This works together with z_transition: z_transition constrains the
    # pod position variable z, while ws_gap directly constrains x picks.
    # Both are needed because z and x are linked only via the EC18 proxy,
    # which is a one-way implication (pick => pod at ws, not vice versa).
    # ================================================================== #

    for p_rel, p_id in enumerate(d.from_RelPod_to_PodId):
        items_of_pod = [im for im in d.items_by_pod[p_id] if im < n_im]

        ws_items: dict[int, list[int]] = {}
        for im in items_of_pod:
            _, m = d.relevant_pairs_for_x[im]
            w_id = d.order_to_ws[m]
            ws_items.setdefault(w_id, []).append(im)

        ws_list = list(ws_items.keys())
        for i, w1 in enumerate(ws_list):
            for w2 in ws_list[i + 1:]:
                l1 = pod_loc_idx[p_rel].get(d.ws_positions[w1])
                l2 = pod_loc_idx[p_rel].get(d.ws_positions[w2])

                # Use direct ws->ws travel if available, else round trip via storage
                tt_direct = inter_loc_travel.get((p_rel, l1, l2)) \
                    if l1 is not None and l2 is not None else None
                min_gap = tt_direct if tt_direct is not None else (
                    pod_ws_travel[(p_rel, w1)] + pod_ws_travel[(p_rel, w2)]
                )
                min_gap = max(min_gap, 1)  # at least 1 slot apart

                ims1 = ws_items[w1]
                ims2 = ws_items[w2]

                for t in range(1, T):
                    for im1 in ims1:
                        for im2 in ims2:
                            # im1 picked at t => im2 not pickable in [t, t+min_gap]
                            for t2 in range(t, min(t + min_gap + 1, T)):
                                model.addConstr(
                                    (x[im1, t] - x[im1, t - 1])
                                    + (x[im2, t2] - x[im2, t2 - 1]) <= 1,
                                    name=f"ws_gap_{p_rel}_{im1}_{im2}_{t}_{t2}"
                                )
                            # symmetric: im2 picked at t => im1 not in [t, t+min_gap]
                            for t2 in range(t, min(t + min_gap + 1, T)):
                                model.addConstr(
                                    (x[im2, t] - x[im2, t - 1])
                                    + (x[im1, t2] - x[im1, t2 - 1]) <= 1,
                                    name=f"ws_gap_sym_{p_rel}_{im2}_{im1}_{t}_{t2}"
                                )

    # ================================================================== #
    # CONSTRAINTS ON z
    # ================================================================== #

    for p_rel, p_id in enumerate(d.from_RelPod_to_PodId):
        locs     = pod_relevant_locs[p_rel]
        n_locs   = len(locs)
        stor_idx = 0  # storage is always index 0

        # Exactly one location at every t
        for t in range(T):
            model.addConstr(
                gb.quicksum(z[p_rel][l, t] for l in range(n_locs)) == 1,
                name=f"z_one_{p_rel}_{t}"
            )

        # Pod starts at storage at t=0
        model.addConstr(z[p_rel][stor_idx, 0] == 1, name=f"z_start_{p_rel}")

        # Lower bound on earliest arrival at each ws
        for l_idx, loc in enumerate(locs):
            if l_idx == stor_idx:
                continue
            w_id_for_loc = next(
                (w for w in range(n_ws) if d.ws_positions[w] == loc), None
            )
            if w_id_for_loc is None:
                continue
            travel = pod_ws_travel[(p_rel, w_id_for_loc)]
            for t in range(min(travel, T)):
                model.addConstr(
                    z[p_rel][l_idx, t] == 0,
                    name=f"z_tlb_{p_rel}_{l_idx}_{t}"
                )

        # Transition constraint: pod at l_dst at t only if it was at some
        # l_src at t - travel(l_src->l_dst). Covers all location pairs,
        # including ws->ws, preventing teleportation between workstations.
        for l_dst in range(n_locs):
            if l_dst == stor_idx:
                continue
            for t in range(1, T):
                rhs_terms = []
                for l_src in range(n_locs):
                    tt = inter_loc_travel.get((p_rel, l_src, l_dst))
                    if tt is None:
                        continue  # not directly reachable
                    t_dep = t - tt
                    if t_dep >= 0:
                        rhs_terms.append(z[p_rel][l_src, t_dep])

                if rhs_terms:
                    model.addConstr(
                        z[p_rel][l_dst, t] <= gb.quicksum(rhs_terms),
                        name=f"z_trans_{p_rel}_{l_dst}_{t}"
                    )
                else:
                    model.addConstr(
                        z[p_rel][l_dst, t] == 0,
                        name=f"z_trans_{p_rel}_{l_dst}_{t}"
                    )

    # ================================================================== #
    # MAX ACTIVE PODS
    # A new pick at t means the pod occupies a robot in [t-travel, t+travel].
    # ================================================================== #

    for im, (_, m) in enumerate(d.relevant_pairs_for_x):
        p_id   = d.pod_of_item[im]
        p_rel  = d.from_PodId_to_RelPod[p_id]
        w_id   = d.order_to_ws[m]
        travel = pod_ws_travel[(p_rel, w_id)]

        for t in range(1, T):
            t_lo = max(0, t - travel)
            t_hi = min(T - 1, t + travel)
            for tau in range(t_lo, t_hi + 1):
                model.addConstr(
                    a[p_rel, tau] >= x[im, t] - x[im, t - 1],
                    name=f"a_{p_rel}_{im}_{t}_{tau}"
                )

    for t in range(T):
        model.addConstr(
            gb.quicksum(a[p_rel, t] for p_rel in range(n_rels)) <= n_robots,
            name=f"max_active_{t}"
        )

    # ================================================================== #
    # EC14: WORKSTATION THROUGHPUT
    # item_work + pod_arrivals <= 1.7 * TIME_UNIT per ws per t 
    #  (more conservative than original constraint)
    # ================================================================== #

    EC_BUDGET = 1.7

    pod_arrives = {}
    for p_rel in range(n_rels):
        for l_idx, loc in enumerate(pod_relevant_locs[p_rel]):
            if l_idx == 0:
                continue
            for t in range(1, T):
                var = model.addVar(
                    vtype=gb.GRB.BINARY, name=f"arr_{p_rel}_{l_idx}_{t}"
                )
                # arr=1 iff z[l,t]=1 AND z[l,t-1]=0
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
            item_work = DELTA_ITEM * gb.quicksum(
                x[im, t] - x[im, t - 1] for im in ims_w
            )
            arrivals = gb.quicksum(
                pod_arrives[(p_rel, l_idx, t)]
                for p_rel in range(n_rels)
                for l_idx, loc in enumerate(pod_relevant_locs[p_rel])
                if loc == ws_loc and (p_rel, l_idx, t) in pod_arrives
            )
            model.addConstr(
                item_work + DELTA_POD * arrivals <= EC_BUDGET * TIME_UNIT,
                name=f"EC14_{w}_{t}"
            )

    # ================================================================== #
    # EC13: WORKSTATION CAPACITY
    # sum(v[m,t]) <= CAP_WS per ws per t
    # ================================================================== #

    for w, order_ids in enumerate(d.orders_by_workstation):
        for t in range(T):
            model.addConstr(
                gb.quicksum(v[m, t] for m in order_ids) <= CAP_WS,
                name=f"EC13_{w}_{t}"
            )

    # ================================================================== #
    # f, g, v DEFINITION (aligned with _build_fgv and check_constraints)
    # ================================================================== #

    for m in range(n_orders):
        ims     = list(d.items_of_order[m])
        n_items = len(ims)

        for t in range(T):
            # EC21: f >= x[im,t]
            for im in ims:
                model.addConstr(f[m, t] >= x[im, t], name=f"EC21_{m}_{im}_{t}")

            # f_active_only_if_picked: f <= sum(x[im,t])
            model.addConstr(
                f[m, t] <= gb.quicksum(x[im, t] for im in ims),
                name=f"f_act_{m}_{t}"
            )

            # EC22: g[m,t] <= x[im,t-1]; g[m,0]=0
            for im in ims:
                if t == 0:
                    model.addConstr(g[m, 0] == 0, name=f"g0_{m}_{im}")
                else:
                    model.addConstr(
                        g[m, t] <= x[im, t - 1], name=f"EC22_{m}_{im}_{t}"
                    )

            # g_lower_bound
            if t > 0:
                model.addConstr(
                    g[m, t] >= gb.quicksum(x[im, t - 1] for im in ims) - (n_items - 1),
                    name=f"g_lb_{m}_{t}"
                )

            # f, g monotone; f >= g
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

            # pick_only_if_active: delta_x <= v[m,t]
            if t > 0:
                for im in ims:
                    model.addConstr(
                        x[im, t] - x[im, t - 1] <= v[m, t],
                        name=f"pick_act_{m}_{im}_{t}"
                    )

    # ================================================================== #
    # INITIAL CONDITIONS (opened orders)
    # ================================================================== #

    for m, order in enumerate(d.orders):
        if order.order_id in d.opened_order_ids:
            model.addConstr(v[m, 0] == 1, name=f"opened_v_{m}")
            model.addConstr(f[m, 0] == 1, name=f"opened_f_{m}")

    # ================================================================== #
    # OBJECTIVE: maximise items picked by end of horizon
    # ================================================================== #

    model.setObjective(
        gb.quicksum(x[im, T - 1] for im in range(n_im)),
        gb.GRB.MAXIMIZE
    )

    # ================================================================== #
    # SOLVE with progressive time budgets (warm-start between calls)
    # ================================================================== #

    budgets = [20, 60]

    for i, budget in enumerate(budgets):
        model.setParam("TimeLimit", budget)
        model.optimize()

        obj = model.ObjVal
        gap = model.MIPGap
        logging.info(
            "[build_initial_x] iter %d/%d: obj=%.3f gap=%.1f%%",
            i + 1, len(budgets), obj, gap * 100
        )
        if gap < 0.01:
            break

    if model.SolCount == 0:
        model.computeIIS()
        model.write("iis.ilp")
        raise RuntimeError("[build_initial_x] No feasible solution found.")

    # ================================================================== #
    # EXTRACT x
    # ================================================================== #

    x_sol = np.zeros((n_im, T), dtype=np.float64)
    for im in range(n_im):
        for t in range(T):
            x_sol[im, t] = 1.0 if x[im, t].X > 0.5 else 0.0

    # Safety net: enforce monotonicity against numerical noise
    for im in range(n_im):
        for t in range(1, T):
            if x_sol[im, t] < x_sol[im, t - 1]:
                x_sol[im, t] = x_sol[im, t - 1]

    logging.info(
        "[build_initial_x] Done. Items picked at T-1: %d / %d",
        int(x_sol[:, T - 1].sum()), n_im
    )

    return x_sol