import logging
import gurobipy as gb
import math


def solve_by_decomposition(OptManager, sim, state):
    """
    Solve the warehouse optimization via two-stage MIP decomposition.

    Stage 1 — Assignment: assign orders to workstations (z) and items to pods (x/y).
    Stage 2 — Scheduling: route pods over time (y2) and schedule item picks (x2/v2).
    """

    orders, orders_items = OptManager.extract_orders(state)
    n_orders = len(orders)
    relevant_pairs_for_x = [(i, m) for m in range(n_orders) for i in orders_items[m]]
    items_of_order: dict[int, list[int]] = {m: [] for m in range(n_orders)}
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        items_of_order[m].append(im)

    if n_orders == 0:
        logging.debug("[OptManager] No orders to optimise — skipping model build.")
        return

    n_p = OptManager.n_pods
    n_w = OptManager.n_workstations
    n_a = len(OptManager.all_arcs)


    ### STAGE 1: ORDER-WORKSTATION AND ORDER-ITEM-POD ASSIGNMENT

    logging.info("Building first stage problem ...")
    model1 = gb.Model('OW_OIP_Assignments')

    # z1[m,w]   = 1 if order m is assigned to workstation w
    # x1[i,m,p] = 1 if item i of order m is retrieved from pod p
    # y1[w,p]   = 1 if pod p must visit workstation w (derived from x1 and z1)
    z1 = model1.addVars(n_orders, n_w, vtype=gb.GRB.BINARY)
    x1 = model1.addVars(len(relevant_pairs_for_x), n_p, vtype=gb.GRB.BINARY)
    y1 = model1.addVars(n_w, n_p, vtype=gb.GRB.BINARY)

    # EC7: each order is assigned to exactly one workstation
    for m in range(n_orders):
        model1.addLConstr(
            gb.quicksum(z1[m, w] for w in range(n_w)),
            gb.GRB.EQUAL, 1, name='EC7')

    for im, (i,m) in enumerate(relevant_pairs_for_x):
        # EC8: each item of the order is retrieved from exactly one pod that stocks it
        model1.addLConstr(
            gb.quicksum(x1[im, p] for p in OptManager.pod_indices_by_sku[i]),
            gb.GRB.EQUAL, 1, name='EC8')

        # EC10: y1[w,p] is forced to 1 when both x1[i,m,p] and z1[m,w] are 1
        for w in range(n_w):
            for p in OptManager.pod_indices_by_sku[i]:
                model1.addLConstr(
                    y1[w, p], gb.GRB.GREATER_EQUAL,
                    x1[im, p] + z1[m, w] - 1, name='EC10')

    # EC11: workload balancing — each workstation handles between 1% and 9% of total items
    total_items = sum(len(i) for i in orders_items)
    lower_I = n_orders / n_w * 0.5
    upper_I = n_orders / n_w * 1.5
    for w in range(n_w):
        orders_at_w = gb.quicksum(z1[m, w] for m in range(n_orders))
        model1.addLConstr(orders_at_w, gb.GRB.LESS_EQUAL,    upper_I, name='EC11_upper')
        model1.addLConstr(orders_at_w, gb.GRB.GREATER_EQUAL, lower_I, name='EC11_lower')

    # Fix assignment for orders already open at each workstation
    for w in range(n_w):
        for m in range(n_orders):
            if orders[m].order_id in state.warehouse.workstations[w].opened_orders:
                model1.addLConstr(z1[m, w] == 1, name='InitialCond')

    # Minimize total pod-workstation visits (proxy for travel distance)
    model1.setObjective(
        gb.quicksum(y1[w, p] for w in range(n_w) for p in range(n_p)),
        sense=gb.GRB.MINIMIZE)

    logging.info("Model1 built. Solving ...")
    model1.optimize()
    logging.info("Model1 solved. Status %s   [2: OPTIMAL, 3: INFEASIBLE, 9: TIME LIMIT]",
                 model1.Status)

    if model1.Status == gb.GRB.INFEASIBLE:
        model1.computeIIS()
        model1.write("Simulator/scripts/opt/iis.ilp")

    # Extract stage-1 solution: map each order to its workstation and each (item, order) to its pod
    z1_sol = model1.getAttr(gb.GRB.Attr.X, z1)
    x1_sol = model1.getAttr(gb.GRB.Attr.X, x1)

    orders_by_workstation = [set() for _ in range(n_w)] # workstation index w → order index m
    order_to_ws_m: dict[int, int] = {}   # order index m → workstation index w
    pod_of_item = {}  # (sku, order_idx) -> pod_idx

    for im, (i,m) in enumerate(relevant_pairs_for_x):
        for w in range(n_w):
            if z1_sol[m, w] > 0.5:
                orders_by_workstation[w].add(m)
                order_to_ws_m[m] = w
                break
        for p in OptManager.pod_indices_by_sku[i]:
            if x1_sol[im, p] > 0.5:
                pod_of_item[im] = p
                break

    from_RelPod_to_PodId = list(set(pod_of_item.values()))
    from_PodId_to_RelPod = {id_p:rel_p for rel_p, id_p in enumerate(from_RelPod_to_PodId)}
    

    ### STAGE 2: SCHEDULING 

    logging.info("Building second stage model ...")
    model2 = gb.Model('Scheduling')

    # x2[i,m,t] = 1 if item i of order m has been picked by time t (cumulative)
    # y2[p,a]   = 1 if pod p traverses arc a in the time-space network
    # v2[m,t]   = 1 if order m is actively being picked at time t
    # f2[m,t]   = 1 if at least one item of order m is picked by time t (triggers opening)
    # g2[m,t]   = 1 if all items of order m are picked by time t-1 (triggers closing)
    x2 = model2.addVars(len(relevant_pairs_for_x), OptManager.N_TIME, vtype=gb.GRB.BINARY)
    y2 = model2.addVars(len(from_RelPod_to_PodId), n_a, vtype=gb.GRB.BINARY)
    v2 = model2.addVars(n_orders, OptManager.N_TIME, vtype=gb.GRB.BINARY)
    f2 = model2.addVars(n_orders, OptManager.N_TIME, vtype=gb.GRB.BINARY)
    g2 = model2.addVars(n_orders, OptManager.N_TIME, vtype=gb.GRB.BINARY)

    for w in range(n_w):
        w_pos = state.warehouse.workstations[w].position
        for t in range(OptManager.N_TIME):
            # EC13: workstation throughput cap — at most CAP_WS orders picked simultaneously
            model2.addLConstr(
                gb.quicksum(v2[m, t] for m in orders_by_workstation[w]),
                gb.GRB.LESS_EQUAL, OptManager.CAP_WS, name='EC13')
            
            # EC14: time capacity — item picks and pod arrivals must fit within TIME_UNIT
            if t > 0:
                item_work = gb.quicksum(
                    OptManager.DELTA_ITEM * (x2[im, t] - x2[im, t - 1])
                    for im, (_,m) in enumerate(relevant_pairs_for_x)
                    if m in orders_by_workstation[w])
                pod_arrivals = gb.quicksum(
                    OptManager.DELTA_POD * y2[rel_p, a]
                    for rel_p in range(len(from_RelPod_to_PodId))
                    for a in OptManager.outgoing_arc_idx[(w_pos, t)])
                model2.addLConstr(
                    item_work + pod_arrivals,
                    gb.GRB.LESS_EQUAL, OptManager.TIME_UNIT, name='EC14')

    for rel_p in range(len(from_RelPod_to_PodId)):
        # EC15: flow conservation at t=0 — each pod departs from its storage location
        model2.addLConstr(
            gb.quicksum(y2[rel_p, a] for a in OptManager.outgoing_arc_idx[
                (state.warehouse.pods[from_RelPod_to_PodId[rel_p]].storage_location, 0)]),
            gb.GRB.EQUAL, 1, name=f'EC15_pod{p}_iniz')

        # EC16: flow conservation at intermediate nodes (pod cannot appear or vanish)
        for node in OptManager.nodes:
            if node[1] in (0, OptManager.N_TIME - 1):
                continue
            model2.addLConstr(
                gb.quicksum(y2[rel_p, a] for a in OptManager.incoming_arc_idx[node])
                - gb.quicksum(y2[rel_p, a] for a in OptManager.outgoing_arc_idx[node]),
                gb.GRB.EQUAL, 0, name=f'EC16_pod{from_RelPod_to_PodId[rel_p]}_nodo{node}')

    # TODO: add congestion constraints

    
    for im, (i,m) in enumerate(relevant_pairs_for_x):
        w_pos = state.warehouse.workstations[order_to_ws_m[m]].position
        p_id = pod_of_item[im]
        rel_p = from_PodId_to_RelPod[p_id]
        # EC18: item cannot be picked at t=0; pick only after pod arrives at workstation
        model2.addLConstr(x2[im, 0], gb.GRB.EQUAL, 0, name='EC18')
        for t in range(1, OptManager.N_TIME):
            model2.addLConstr(
                x2[im, t] - x2[im, t - 1],
                gb.GRB.LESS_EQUAL,
                gb.quicksum(y2[rel_p, a] for a in OptManager.outgoing_arc_idx[(w_pos, t)]
                            if a < len(OptManager.travelling_arcs)),
                name='EC18')

    for im, (i,m) in enumerate(relevant_pairs_for_x):
        for t in range(OptManager.N_TIME):
            if t > 0:
                # EC19: x2 is non-decreasing (once picked, stays picked)
                model2.addLConstr(x2[im, t] >= x2[im, t - 1], name='EC19')
                model2.addLConstr(x2[im, t] - x2[im, t - 1] <= v2[m, t], name='pick_only_if_active')
            
            # EC21/EC22: link f2 and g2 to item completion status
            model2.addLConstr(f2[m, t] >= x2[im, t], name='EC21')
            if t > 0:
                model2.addLConstr(g2[m, t] <= x2[im, t-1], name='EC22')
                # Why? I have somehow to allow 1-item order to stay open for one at least one period


    for m in range(n_orders):
        for t in range(OptManager.N_TIME):
            if t > 0:
                # f2 and g2 are non-decreasing
                model2.addLConstr(f2[m, t] >= f2[m, t-1], name='f_monocity') 
                model2.addLConstr(g2[m, t] >= g2[m, t-1], name='g_monocity') 
                model2.addLConstr(v2[m, t] >= v2[m, t - 1] - g2[m, t], name='continuity_of_v2')

            # EC20: v2 = f2 - g2 (order is active iff started but not yet complete)
            model2.addLConstr(v2[m, t] == f2[m, t] - g2[m, t], name='EC20')


    for m in range(n_orders):
        for t in range(OptManager.N_TIME-1):
            # Lower bound on g2: order completes only when all items are picked
            model2.addLConstr(
                g2[m, t+1] >= gb.quicksum(x2[im, t] for im in items_of_order[m]) - (len(orders_items[m]) - 1),
                name='g_LowerB')

        # Fix v2[m,0]=1 for orders already open (not yet closed by active tasks)
        if orders[m].order_id in state.warehouse.workstations[order_to_ws_m[m]].opened_orders:
            model2.addLConstr(v2[m, 0] == 1, name='InitialCond')
        else:
            for t in range(OptManager.N_TIME):
                model2.addLConstr(f2[m, t] <= gb.quicksum(x2[im, t] for im in items_of_order[m]), name='f2_active_onlyif_at_least_1_item_is_picked')


    """ # Maximize items picked by the end of the horizon
    model2.setObjective(
        gb.quicksum(x2[im, OptManager.N_TIME - 1]
                    for im in range(len(relevant_pairs_for_x))),
        sense=gb.GRB.MAXIMIZE) """
    
    # Minimize sum of completion times = Maximize sum of g2
    # (equivalently: reward completing orders as early as possible)
    model2.setObjective(
        gb.quicksum(
            g2[m, t]
            for m in range(n_orders)
            for t in range(OptManager.N_TIME)
        ) +
        0.2 * gb.quicksum(x2[im, OptManager.N_TIME-1] 
                          for im in range(len(relevant_pairs_for_x)))
        - 0.5*gb.quicksum((state.current_time - orders[m].arrival_time)/OptManager.TIME_UNIT*(1-f2[m,OptManager.N_TIME - 1])
                          for m in range(n_orders)), # penalty for orders I won't even open based on how long they have been in backlog
        sense=gb.GRB.MAXIMIZE
    )

    model2.setParam('MIPGap', 0.10)

    logging.info('Model2 built. Solving ...')
    model2.optimize()
    logging.info("Model2 solved. Status %s   [2: OPTIMAL, 3: INFEASIBLE, 9: TIME LIMIT]",
                 model2.Status)

    if model2.Status == gb.GRB.INFEASIBLE:
        model2.computeIIS()
        model2.write("is.ilp")

    x2_sol = model2.getAttr(gb.GRB.Attr.X, x2)
    v2_sol = model2.getAttr(gb.GRB.Attr.X, v2)
    y2_sol = model2.getAttr(gb.GRB.Attr.X, y2)
    return orders, orders_items, z1_sol, x1_sol, x2_sol, v2_sol, y2_sol, from_RelPod_to_PodId