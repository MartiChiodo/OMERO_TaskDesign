import gurobipy as gb
import logging

from Simulator.scripts.opt.stage2_data import build_stage2_data
from Simulator.scripts.opt.local_search import local_search

def optimizer_solver(OptManager, state):

    orders, orders_items = OptManager.extract_orders(state)
    n_orders = len(orders)
    relevant_pairs_for_x = [(i, m) for m in range(n_orders) for i in orders_items[m]]
    items_of_order: dict[int, list[int]] = {m: [] for m in range(n_orders)}
    for im, (_, m) in enumerate(relevant_pairs_for_x):
        items_of_order[m].append(im)

    if n_orders == 0:
        return

    n_p = OptManager.n_pods
    n_w = OptManager.n_workstations
    n_a = len(OptManager.all_arcs)


    ### STAGE 1: ORDER-WORKSTATION AND ORDER-ITEM-POD ASSIGNMENT

    logging.info("Building first stage problem ...")
    model1 = gb.Model('OW_OIP_Assignments')

    # z1[m,w] = 1 if order m is assigned to workstation w
    # x1[im,p] = 1 if item i of order m is retrieved from pod p
    # y1[w,p] = 1 if pod p must visit workstation w (derived from x1 and z1)
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
    st2_data =  build_stage2_data(
            OptManager = OptManager,
            state = state,
            orders = orders,
            orders_items= orders_items,
            relevant_pairs_for_x = relevant_pairs_for_x,
            items_of_order = items_of_order, 
            orders_by_workstation= orders_by_workstation,
            order_to_ws_m = order_to_ws_m,
            pod_of_item = pod_of_item,
            from_RelPod_to_PodId = from_RelPod_to_PodId,
            from_PodId_to_RelPod = from_PodId_to_RelPod
        )
    
    sol =  local_search(st2_data)
    (x, f, g, v, y) = sol

    return st2_data, x, v, y