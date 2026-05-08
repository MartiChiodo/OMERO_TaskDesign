from collections import defaultdict
from Simulator.scripts.core.entities import Task, Visit

def convert_OptSol_to_SimObj(data, x_sol, v_sol, y_sol):

    n_orders = len(data.orders)
    relevant_pairs_for_x = data.relevant_pairs_for_x


    ### Step 1: extract workstation assignments and order start times 

    orders_by_workstation =  [lst.copy() for lst in data.orders_by_workstation]
    order_start_time: dict[int, int] = {}

    for m in range(n_orders):
        w = data.order_to_ws[m]
        # Already-open orders start immediately; others use first active v2 period
        if data.orders[m].order_id in data.state.warehouse.workstations[w].opened_orders:
            start_t = 0
        else:
            start_t = next(
                (t for t in range(data.OptManager.N_TIME) if v_sol[m, t] > 0.5),
                None,   # fallback: do not consider if never active
            )
        if start_t is None:
            orders_by_workstation[w].remove(m)
        else:
            order_start_time[m] = start_t

    # Sort orders within each workstation by start time for downstream processing
    ordered_orders_by_w = {
        w: sorted(idxs, key=lambda m: order_start_time[m])
        for w, idxs in enumerate(orders_by_workstation)
    }


    ### Step 2: Retrieving lookups

    order_to_ws = data.order_to_ws    # map order_idx -> ws_idx
    item_to_pod = data.pod_of_item    # map  im -> pod_id 

    # (sku, order_idx) → timestep at which item is first picked
    item_to_time: dict[tuple[int, int], int] = {}
    for im, (i, m) in enumerate(relevant_pairs_for_x):
        if x_sol[im, 0] > 0.5:
            item_to_time[(i, m)] = 0
        else:
            item_to_time[(i, m)] = next(
                (t for t in range(1, data.OptManager.N_TIME)
                    if x_sol[im, t] > 0.5 and x_sol[im, t - 1] < 0.5),
                None
            )

    ### Step 3 & 4: reconstruct pod trajectories from y2_sol → Tasks

    workstation_positions = set(data.OptManager._W)
    storage_positions = set(data.OptManager._L)
    pos_to_ws = {
        data.state.warehouse.workstations[w].position: w
        for w in range(data.OptManager.n_workstations)
    }

    # Precompute: for each (pod_id, t_arrival, w_idx) → items and orders to pick
    pick_at: dict[tuple[int, int, int], dict] = defaultdict(
        lambda: {"items": set(), "orders": set()}
    )
    for im, p_id in item_to_pod.items():
        i, m = relevant_pairs_for_x[im]
        t = item_to_time.get((i,m))
        if t is None:
            continue
        w = order_to_ws.get(m)
        if w is None:
            continue
        pick_at[(p_id, t, w)]["items"].add(i)
        pick_at[(p_id, t, w)]["orders"].add(data.orders[m].order_id)


    tasks: list[Task] = []

    for rel_p, p_id in enumerate(data.from_RelPod_to_PodId):

        # Collect all arcs traversed by this pod, sorted by departure time
        traversed = sorted(
            [(src[1], src[0], dst[1], dst[0])          # (t_src, loc_src, t_dst, loc_dst)
            for a_idx, (src, dst) in enumerate(data.OptManager.all_arcs)
            if y_sol[rel_p, a_idx] > 0.5],
            key= lambda k : k[0]
        )

        # Walk the trajectory and split into trips.
        # A trip = sequence of workstation visits between two storage stays.
        # A new trip starts when the pod departs from storage after returning.
        current_stops: list[tuple[int, int]] = []   # (t_arrival, w_idx)
        in_trip = False

        for id, (t_src, _, t_dst, loc_dst) in enumerate(traversed):

            if loc_dst in workstation_positions:
                # Pod arrives at a workstation
                w_idx = pos_to_ws[loc_dst]
                current_stops.append((t_dst, w_idx))
                in_trip = True

            elif loc_dst in storage_positions and in_trip:
                # Pod returns to storage to stay → close current trip as a Task
                stops = []
                for t_arr, w_idx in current_stops:
                    pick_data = pick_at.get((p_id, t_arr, w_idx))
                    if pick_data and pick_data["items"]:
                        stops.append(Visit(
                            workstation_id=w_idx,
                            orders=pick_data["orders"],
                            items=pick_data["items"],
                        ))
                if stops:
                    pr = None
                    for i,m in [(i,m) for i,m in relevant_pairs_for_x if i in stops[0].items and data.orders[m].order_id in stops[0].orders]:
                        pr = item_to_time.get((i, m))
                        if not pr == None:
                            break 
                    tasks.append(Task(
                        task_id=None,
                        pod_id=p_id,
                        robot_id=None,
                        stops=stops,
                        priority=pr,
                    ))
                current_stops = []
                in_trip = False

        # Handle trip still open at end of horizon (pod never returned to storage)
        if current_stops:
            stops = []
            for t_arr, w_idx in current_stops:
                pick_data = pick_at.get((p_id, t_arr, w_idx))
                if pick_data and pick_data["items"]:
                    stops.append(Visit(
                        workstation_id=w_idx,
                        orders=pick_data["orders"],
                        items=pick_data["items"],
                    ))
            if stops:
                pr = None
                for i,m in [(i,m) for i,m in relevant_pairs_for_x if i in stops[0].items and data.orders[m].order_id in stops[0].orders]:
                    pr = item_to_time.get((i, m))
                    if not pr == None:
                        break 
                tasks.append(Task(
                    task_id=None,
                    pod_id=p_id,
                    robot_id=None,
                    stops=stops,
                    priority=pr,
                ))

    # Assigning priority to tasks
    for task in tasks:
        t_firstpicking =  task.priority
        ws = data.state.warehouse.workstations[task.stops[0].workstation_id]
        pod = data.state.warehouse.pods[task.pod_id]
        pr = (t_firstpicking * data.OptManager.TIME_UNIT - 0.25*data.state.warehouse.travel_time(
                data.state.warehouse.cell2coord(ws.position),
                data.state.warehouse.cell2coord(pod.storage_location)
            ))
        task.priority = pr

    # Sorting tasks according to priority + mapping priority in a better interval
    tasks.sort(key=lambda t: t.priority)
    for new_id, task in enumerate(tasks):
        task.task_id = data.state.task_counter + new_id
        task.priority = new_id / (len(tasks) -1) * 500

    return data.orders, ordered_orders_by_w, tasks