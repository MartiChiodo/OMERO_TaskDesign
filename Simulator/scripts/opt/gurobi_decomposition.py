import logging
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scripts.opt.stage2_data import build_stage2_data

# same cost function used by the LNS stage 1, so the benchmark optimises the
# exact same objective and the gap is meaningful
from scripts.opt.local_search_stage1 import _get_beta, _get_ws_pod_distance_matrix
from scripts.opt.convert_OptSol_to_SimObj import convert_OptSol_to_SimObj


def gurobi_benchmark(OptManager, sim, state,
                     time_limit=15*60, mip_gap=0.01):
    """
    Full two-stage decomposition solved with everything variable, used as a
    reference to measure the LNS optimality gap.

    Gurobi session handling (important with a WLS licence): pass an `env` from
    the caller to reuse a single session across many calls. If none is given,
    one is created here and disposed at the end. Both models are always
    disposed in a finally block, so no session or model is left dangling even
    if an exception is raised mid-solve.
    """

    t_start = time.perf_counter()

    # models and env tracked here so the finally block can always release them
    model1 = None
    model2 = None
    env = gp.Env()              # one WLS session, released in finally

    rng = np.random.default_rng()   

    try:
        ###- setup
        orders, orders_items = OptManager.extract_orders(state)
        n_orders = len(orders)

        if n_orders == 0:
            logging.debug("[benchmark] No orders to optimise - skipping model build.")
            return

        relevant_pairs_for_x = [(i, m) for m in range(n_orders) for i in orders_items[m]]
        items_of_order: dict[int, list[int]] = {m: [] for m in range(n_orders)}
        for im, (_, m) in enumerate(relevant_pairs_for_x):
            items_of_order[m].append(im)

        n_p = OptManager.n_pods
        n_w = OptManager.n_workstations
        n_a = len(OptManager.all_arcs)
        T = OptManager.N_TIME
        n_travel = len(OptManager.travelling_arcs)
        n_robots = len(state.warehouse.robots)

        logging.info("[benchmark] instance size: %d orders, %d item-order pairs, "
                     "%d pods, %d workstations, %d arcs, T=%d",
                     n_orders, len(relevant_pairs_for_x), n_p, n_w, n_a, T)

        # SKUs per order - used by the workload balance (repair1 EC5)
        sku_per_order = np.array([len(orders_items[m]) for m in range(n_orders)], dtype=float)
        sku_total = int(sku_per_order.sum())

        # distance-weighted cost, rebuilt exactly as in local_search_stage1:
        #   cost = 1 + beta(state) * dist[w, pod]
        dist = _get_ws_pod_distance_matrix(state)
        cost_matrix = 1.0 + _get_beta(state) * dist


        ###  STAGE 1  - ORDER->WS and ITEM->POD assignment

        logging.info("[benchmark] Building stage-1 model ...")
        print("[benchmark] Building stage-1 model ...")
        t_build1 = time.perf_counter()
        model1 = gp.Model('benchmark_stage1', env=env)
        model1.Params.OutputFlag = 0
        if time_limit is not None:
            model1.Params.TimeLimit = time_limit
        model1.Params.MIPGap = mip_gap
        model1.Params.Seed = rng.integers(0, 10**3)

        # z1[m,w] order->ws ; x1[im,p] item->pod ; y1[w,p] pod visits ws
        z1 = model1.addVars(n_orders, n_w, vtype=GRB.BINARY, name="assign")
        x1 = {}
        for im, (i, m) in enumerate(relevant_pairs_for_x):
            for p in OptManager.pod_indices_by_sku[i]:
                x1[im, p] = model1.addVar(vtype=GRB.BINARY)

        admissible_pods = {
            im: list(OptManager.pod_indices_by_sku[relevant_pairs_for_x[im][0]])
            for im in range(len(relevant_pairs_for_x))
        }
        all_pods = sorted({int(p) for pods in admissible_pods.values() for p in pods})
        # y1 continuous in [0,1], pinned up by EC4 (as in repair1)
        y1 = {}
        for p in all_pods:
            for w in range(n_w):
                y1[w, p] = model1.addVar(lb=0.0, ub=1.0)

        # EC1: each order to exactly one workstation
        for m in range(n_orders):
            model1.addConstr(gp.quicksum(z1[m, w] for w in range(n_w)) == 1.0, name="EC1")

        # EC2: each item picked from exactly one admissible pod
        for im in range(len(relevant_pairs_for_x)):
            model1.addConstr(
                gp.quicksum(x1[im, p] for p in admissible_pods[im]) == 1.0, name="EC2")

        # EC4: if item im picked from pod p and its order m at ws w, then pod p visits w
        for im, (i, m) in enumerate(relevant_pairs_for_x):
            for p in admissible_pods[im]:
                for w in range(n_w):
                    model1.addConstr(
                        y1[w, p] >= x1[im, p] + z1[m, w] - 1.0, name="EC4")

        # EC5: workload balance on SKUs (repair1: sku_total, floor/ceil, 0.8..1.2)
        load_lo = np.floor(sku_total / n_w * 0.8)
        load_hi = np.ceil(sku_total / n_w * 1.2)
        for w in range(n_w):
            load = gp.quicksum(sku_per_order[m] * z1[m, w] for m in range(n_orders))
            model1.addConstr(load <= load_hi, name="EC5_upper")
            model1.addConstr(load >= load_lo, name="EC5_lower")

        # Fix assignment for orders already open at each workstation
        for w in range(n_w):
            for m in range(n_orders):
                if orders[m].order_id in state.warehouse.workstations[w].opened_orders:
                    model1.addConstr(z1[m, w] == 1.0, name="InitialCond1")

        # Objective (repair1): minimise distance-weighted pod->ws visits
        model1.setObjective(
            gp.quicksum(cost_matrix[w, p] * y1[w, p] for p in all_pods for w in range(n_w)),
            GRB.MINIMIZE)

        build1_time = time.perf_counter() - t_build1
        logging.info("[benchmark] Stage-1 built in %.2f s. Solving ...", build1_time)
        print(f"[benchmark] Stage-1 built in {build1_time} s. Solving ...")

        t_solve1 = time.perf_counter()
        model1.optimize()
        solve1_time = time.perf_counter() - t_solve1
        logging.info("[benchmark] Stage-1 status %s   [2:OPT 3:INFEAS 9:TIME]", model1.Status)
        print(f"[benchmark] Stage-1 status {model1.Status}   [2:OPT 3:INFEAS 9:TIME]")

        if model1.Status == GRB.INFEASIBLE:
            model1.computeIIS()
            model1.write("Simulator/scripts/opt/iis_benchmark1.ilp")
            logging.warning("[benchmark] Stage-1 INFEASIBLE after %.2f s (build %.2f s).",
                            solve1_time, build1_time)
            return None

        logging.info("[benchmark] Stage-1 solved in %.2f s  (build %.2f s, solve %.2f s)  "
                     "obj = %.4f, bound = %.4f, gap = %.2f%%",
                     build1_time + solve1_time, build1_time, solve1_time,
                     model1.ObjVal, model1.ObjBound, 100.0 * model1.MIPGap)
        print(f"[benchmark] Stage-1 solved in {build1_time + solve1_time:.2f} s  "
                    f"(build {build1_time:.2f} s, solve {solve1_time:.2f} s)  "
                    f"obj = {model1.ObjVal:.4f}, bound = {model1.ObjBound:.4f}, "
                    f"gap = {100.0 * model1.MIPGap:.2f}%")

        #  extract stage-1 solution
        z1_sol = {(m, w): z1[m, w].X for m in range(n_orders) for w in range(n_w)}
        x1_sol = {(im, p): x1[im, p].X for (im, p) in x1}

        orders_by_workstation = [set() for _ in range(n_w)]
        order_to_ws_m: dict[int, int] = {}
        pod_of_item: dict[int, int] = {}

        for im, (i, m) in enumerate(relevant_pairs_for_x):
            for w in range(n_w):
                if z1_sol[m, w] > 0.5:
                    orders_by_workstation[w].add(m)
                    order_to_ws_m[m] = w
                    break
            for p in admissible_pods[im]:
                if x1_sol[im, p] > 0.5:
                    pod_of_item[im] = p
                    break

        from_RelPod_to_PodId = sorted(set(pod_of_item.values()))
        from_PodId_to_RelPod = {id_p: rel_p for rel_p, id_p in enumerate(from_RelPod_to_PodId)}
        n_rel_pods = len(from_RelPod_to_PodId)


        ###  STAGE 2  - SCHEDULING   (from repair2, everything variable)

        logging.info("[benchmark] Building stage-2 model ...")
        print(f"\n[benchmark] Building stage-2 model ...")
        t_build2 = time.perf_counter()
        model2 = gp.Model('benchmark_stage2', env=env)
        model2.Params.OutputFlag = 0
        if time_limit is not None:
            model2.Params.TimeLimit = time_limit
        model2.Params.MIPGap = mip_gap
        model2.Params.Seed = rng.integers(0, 10**3)

        x2 = model2.addVars(len(relevant_pairs_for_x), T, vtype=GRB.BINARY, name="x2")
        y2 = model2.addVars(n_rel_pods, n_a, vtype=GRB.BINARY, name="y2")
        f2 = model2.addVars(n_orders, T, lb=0, ub=1, name="f2")
        g2 = model2.addVars(n_orders, T, lb=0, ub=1, name="g2")
        v2 = model2.addVars(n_orders, T, lb=0, ub=1, name="v2")

        logging.info("[benchmark] Stage-2 variables: x2=%d, y2=%d (%d pods x %d arcs), "
                     "f2/g2/v2=%d each",
                     len(relevant_pairs_for_x) * T, n_rel_pods * n_a, n_rel_pods, n_a,
                     n_orders * T)

        ws_positions = [state.warehouse.workstations[w].position for w in range(n_w)]

        # order-status constraints (EC15..EC22, EC17)
        for m in range(n_orders):
            ims = items_of_order[m]
            n_it = len(orders_items[m])
            for t in range(T):
                model2.addConstr(v2[m, t] == f2[m, t] - g2[m, t], name="EC17")
            for t in range(1, T):
                model2.addConstr(f2[m, t] >= f2[m, t - 1], name="EC20f")
                model2.addConstr(g2[m, t] >= g2[m, t - 1], name="EC20g")
                model2.addConstr(v2[m, t] >= v2[m, t - 1] - g2[m, t], name="EC21")
                model2.addConstr(
                    g2[m, t] >= gp.quicksum(x2[im, t - 1] for im in ims) - (n_it - 1),
                    name="EC22")
            for im in ims:
                for t in range(T):
                    model2.addConstr(f2[m, t] >= x2[im, t], name="EC16")
                for t in range(1, T):
                    model2.addConstr(x2[im, t] >= x2[im, t - 1], name="EC15")
                    model2.addConstr(x2[im, t] - x2[im, t - 1] <= v2[m, t], name="EC18")
                    model2.addConstr(g2[m, t] <= x2[im, t - 1], name="EC19")
                model2.addConstr(x2[im, 0] == 0.0, name="x0")

            # EC23: initial condition on already-open orders
            if orders[m].order_id in state.warehouse.workstations[order_to_ws_m[m]].opened_orders:
                model2.addConstr(v2[m, 0] == 1.0, name="EC23")
            else:
                for t in range(T):
                    model2.addConstr(
                        f2[m, t] <= gp.quicksum(x2[im, t] for im in ims),
                        name="f2_needs_a_pick")

        # workstation capacity (EC10)
        for w in range(n_w):
            ids = list(orders_by_workstation[w])
            for t in range(T):
                model2.addConstr(
                    gp.quicksum(v2[m, t] for m in ids) <= OptManager.CAP_WS, name="EC10")

        # routing / flow per pod (EC12 depart, EC13 conservation)
        for p in from_RelPod_to_PodId:
            rel_p = from_PodId_to_RelPod[p]
            storage = state.warehouse.pods[p].storage_location
            out0 = OptManager.outgoing_arc_idx.get((storage, 0), [])
            model2.addConstr(gp.quicksum(y2[rel_p, a] for a in out0) == 1.0, name="EC12")
            for node in OptManager.nodes:
                if node[1] in (0, T - 1):
                    continue
                inc = OptManager.incoming_arc_idx.get(node, [])
                out = OptManager.outgoing_arc_idx.get(node, [])
                if inc or out:
                    model2.addConstr(
                        gp.quicksum(y2[rel_p, a] for a in inc)
                        - gp.quicksum(y2[rel_p, a] for a in out) == 0.0, name="EC13")

        # pod at the workstation when its item is picked (EC14)
        for im, (i, m) in enumerate(relevant_pairs_for_x):
            p = pod_of_item[im]
            rel_p = from_PodId_to_RelPod[p]
            ws_p = ws_positions[order_to_ws_m[m]]
            for t in range(T):
                inc = OptManager.incoming_arc_idx.get((ws_p, t), [])
                dx = x2[im, t] if t == 0 else (x2[im, t] - x2[im, t - 1])
                model2.addConstr(gp.quicksum(y2[rel_p, a] for a in inc) >= dx, name="EC14")

        # time capacity per (w,t): item work + pod arrivals (EC11)
        for w in range(n_w):
            ws_p = ws_positions[w]
            ims_w = [im for im, (_, m) in enumerate(relevant_pairs_for_x)
                     if m in orders_by_workstation[w]]
            for t in range(1, T):
                item_work = OptManager.DELTA_ITEM * gp.quicksum(
                    x2[im, t] - x2[im, t - 1] for im in ims_w)
                travel_in = [a for a in OptManager.incoming_arc_idx.get((ws_p, t), [])
                             if a < n_travel]
                pod_arrivals = gp.quicksum(
                    y2[from_PodId_to_RelPod[p], a]
                    for p in from_RelPod_to_PodId for a in travel_in)
                model2.addConstr(
                    item_work + OptManager.DELTA_POD * pod_arrivals
                    <= 1.5 * OptManager.TIME_UNIT, name="EC11")

        # robots in use per t (EC24)
        for t in range(T):
            occ = []
            for p in from_RelPod_to_PodId:
                rel_p = from_PodId_to_RelPod[p]
                storage = state.warehouse.pods[p].storage_location
                for a in range(n_a):
                    (sl, st), (dl, dt) = OptManager.all_arcs[a]
                    if sl != storage and st <= t < dt:
                        occ.append(y2[rel_p, a])
            model2.addConstr(gp.quicksum(occ) <= n_robots, name="EC24")

        # objective: maximise picking - 0.1 * mean age-penalty
        picking = gp.quicksum(x2[im, T - 1] for im in range(len(relevant_pairs_for_x)))
        terms = []
        for m in range(n_orders):
            a_m = float(orders[m].arrival_time)
            for t in range(T):
                age = (state.current_time + t * OptManager.TIME_UNIT - a_m) / OptManager.TIME_UNIT
                terms.append(age * (1.0 - g2[m, t]))
        model2.setObjective(
            picking - 0.1 * gp.quicksum(terms) / OptManager.N_TIME, GRB.MAXIMIZE)

        build2_time = time.perf_counter() - t_build2
        logging.info("[benchmark] Stage-2 built in %.2f s. Solving ...", build2_time)
        print(f"[benchmark] Stage-2 built in {build2_time} s. Solving ...")

        t_solve2 = time.perf_counter()
        model2.optimize()
        solve2_time = time.perf_counter() - t_solve2
        logging.info("[benchmark] Stage-2 status %s   [2:OPT 3:INFEAS 9:TIME]", model2.Status)
        print(f"[benchmark] Stage-2 status {model2.Status}   [2:OPT 3:INFEAS 9:TIME]")

        if model2.Status == GRB.INFEASIBLE:
            model2.computeIIS()
            model2.write("Simulator/scripts/opt/iis_benchmark2.ilp")
            logging.warning("[benchmark] Stage-2 INFEASIBLE after %.2f s (build %.2f s).",
                            solve2_time, build2_time)
            return None

        benchmark_obj = model2.ObjVal
        benchmark_bound = model2.ObjBound
        logging.info("[benchmark] Stage-2 solved in %.2f s  (build %.2f s, solve %.2f s)  "
                     "obj = %.4f, bound = %.4f, gap = %.2f%%",
                     build2_time + solve2_time, build2_time, solve2_time,
                     benchmark_obj, benchmark_bound, 100.0 * model2.MIPGap)
        print(f"[benchmark] Stage-2 solved in {build2_time + solve2_time:.2f} s  "
            f"(build {build2_time:.2f} s, solve {solve2_time:.2f} s)  "
            f"obj = {model2.ObjVal:.4f}, bound = {model2.ObjBound:.4f}, "
            f"gap = {100.0 * model2.MIPGap:.2f}%")

        # extract solution while the model is still alive
        x2_sol = model2.getAttr(GRB.Attr.X, x2)
        v2_sol = model2.getAttr(GRB.Attr.X, v2)
        y2_sol = model2.getAttr(GRB.Attr.X, y2)

        st2_data = build_stage2_data(
                OptManager=OptManager,
                state=state,
                orders=orders,
                orders_items=orders_items,
                relevant_pairs_for_x=relevant_pairs_for_x,
                items_of_order=items_of_order,
                orders_by_workstation=orders_by_workstation,
                order_to_ws_m=order_to_ws_m,
                pod_of_item=pod_of_item,
                from_RelPod_to_PodId=from_RelPod_to_PodId,
                from_PodId_to_RelPod=from_PodId_to_RelPod
            )

        orders, ordered_orders_by_w, tasks = convert_OptSol_to_SimObj(
            st2_data, x2_sol, v2_sol, y2_sol)

        total_time = time.perf_counter() - t_start
        logging.info("[benchmark] DONE in %.2f s total  "
                     "(stage-1 %.2f s + stage-2 %.2f s + overhead %.2f s)",
                     total_time,
                     build1_time + solve1_time,
                     build2_time + solve2_time,
                     total_time - build1_time - solve1_time - build2_time - solve2_time)
        print(f"[benchmark] DONE in {total_time:.2f} s total  "
                f"(stage-1 {build1_time + solve1_time:.2f} s + "
                f"stage-2 {build2_time + solve2_time:.2f} s + "
                f"overhead {total_time - build1_time - solve1_time - build2_time - solve2_time:.2f} s)")

        return orders, ordered_orders_by_w, tasks

    finally:
        # always release the models and the env, so a WLS session is never
        # left open across calls even on error
        if model1 is not None:
            model1.dispose()
        if model2 is not None:
            model2.dispose()
        env.dispose()