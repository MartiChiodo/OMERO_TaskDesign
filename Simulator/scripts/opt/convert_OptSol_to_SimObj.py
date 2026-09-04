from scripts.core.entities import Task, Visit


def _first_pick_time(x_sol, im, N_TIME):
    # x_sol is cumulative (0001111), so the pick is the first t that turns to 1
    if x_sol[im, 0] > 0.5:
        return 0
    for t in range(1, N_TIME):
        if x_sol[im, t] > 0.5 and x_sol[im, t - 1] < 0.5:
            return t
    return None


def convert_OptSol_to_SimObj(data, x_sol, v_sol, y_sol=None):
    """Turn the Stage 2 solution into orders, an opening order per workstation,
    and tasks. Tasks say which pod brings which items where; routing is left to
    the simulator. y_sol is unused, kept only for a stable signature."""
    n_orders             = len(data.orders)
    relevant_pairs_for_x = data.relevant_pairs_for_x
    N_TIME               = data.OptManager.N_TIME
    warehouse            = data.state.warehouse

    # Step 1: find when each order opens (its pick window), drop the ones that
    # never open.
    orders_by_workstation = [lst.copy() for lst in data.orders_by_workstation]
    order_start_time = {}
    for m in range(n_orders):
        w = data.order_to_ws[m]

        if data.orders[m].order_id in warehouse.workstations[w].opened_orders:
            start_t = 0                      # already open
        else:
            # first timestep where the order is active
            start_t = None
            for t in range(N_TIME):
                if v_sol[m, t] > 0.5:
                    start_t = t
                    break

        if start_t is None:
            orders_by_workstation[w].remove(m)
        else:
            order_start_time[m] = start_t

    # Step 2: first pick time of each (item, order) pair.
    order_to_ws = data.order_to_ws
    item_to_pod = data.pod_of_item
    item_to_time = {}
    for im, (i, m) in enumerate(relevant_pairs_for_x):
        t = _first_pick_time(x_sol, im, N_TIME)
        if t is not None:
            item_to_time[(i, m)] = t

    # Step 3: collect the work into groups keyed by (pod, window, workstation).
    # Each group holds the items and orders picked there; first_pick only orders
    # the stops later.
    groups = {}
    for im, (i, m) in enumerate(relevant_pairs_for_x):
        if m not in order_start_time:
            continue                         # order was dropped
        t_pick = item_to_time.get((i, m))
        if t_pick is None:
            continue                         # item never picked

        w      = order_to_ws[m]
        p_id   = item_to_pod[im]
        window = order_start_time[m]
        key    = (p_id, window, w)

        if key not in groups:
            groups[key] = {"items": set(), "orders": set(), "first_pick": N_TIME + 2}
        group = groups[key]
        group["items"].add(i)
        group["orders"].add(data.orders[m].order_id)
        group["first_pick"] = min(group["first_pick"], t_pick)

    # Build one task from a chunk of groups (same pod, nearby windows). Groups on
    # the same workstation are merged into one stop, stops are ordered by pick.
    def _make_task(p_id, chunk):
        by_w = {}
        for window, w, group in chunk:
            if w not in by_w:
                by_w[w] = {"items": set(), "orders": set(), "first_pick": N_TIME + 2}
            by_w[w]["items"]      |= group["items"]
            by_w[w]["orders"]     |= group["orders"]
            by_w[w]["first_pick"]  = min(by_w[w]["first_pick"], group["first_pick"])

        stops_sorted = sorted(by_w.items(), key=lambda item: item[1]["first_pick"])
        stops = []
        for w, d in stops_sorted:
            stops.append(Visit(workstation_id=w, orders=d["orders"], items=d["items"]))

        # priority comes from the first stop of the task
        window0 = min(window for window, _, _ in chunk)
        first0  = min(d["first_pick"] for d in by_w.values())
        task = Task(task_id=None, pod_id=p_id, robot_id=None,
                    stops=stops, priority=window0)
        return (window0, first0, task)

    # Step 4: for each pod, sort its groups by window and cut a new task when the
    # gap between two consecutive windows is longer than the time the pod would
    # need to travel back to storage and out again. If the gap is shorter, the pod
    # is kept out in a single task; if longer, splitting into two trips is feasible.
    def round_trip(p_id, w_idx):
        # storage(pod) -> workstation -> storage, in timesteps
        pod = warehouse.pods[p_id]
        ws  = warehouse.workstations[w_idx]
        one_way = warehouse.travel_time(
            warehouse.cell2coord(pod.storage_location),
            warehouse.cell2coord(ws.position),
        )
        return 2.0 * one_way / data.OptManager.TIME_UNIT
        
    by_pod = {}
    for (p_id, window, w), group in groups.items():
        by_pod.setdefault(p_id, []).append((window, w, group))

    tasks_with_key = []
    for p_id, entries in by_pod.items():
        entries.sort(key=lambda e: (e[0], e[2]["first_pick"]))

        chunk = [entries[0]]
        for idx in range(1, len(entries)):
            prev_window = entries[idx - 1][0]
            curr_window, curr_w, _ = entries[idx]
            if curr_window - prev_window > round_trip(p_id, curr_w):
                tasks_with_key.append(_make_task(p_id, chunk))
                chunk = []
            chunk.append(entries[idx])
        tasks_with_key.append(_make_task(p_id, chunk))


    # Step 5: refine task priorities with a travel-time lead, then sort and
    # remap to [0, 300]. The base priority is the task's earliest first-pick
    # time (first0); a pod far from its first workstation is nudged earlier so
    # it can depart in time to arrive as planned.
    tasks = []
    for window0, first0, task in tasks_with_key:
        pod = warehouse.pods[task.pod_id]
        ws  = warehouse.workstations[task.stops[0].workstation_id]
        travel_lead = (
            1.1 * warehouse.travel_time(
                warehouse.cell2coord(pod.storage_location),
                warehouse.cell2coord(ws.position),
            )
        ) / data.OptManager.TIME_UNIT
        task.priority = first0 - 0.25 * travel_lead
        tasks.append(task)

    # Sort by adjusted priority (ascending = more urgent first)
    tasks.sort(key=lambda t: t.priority)

    # Assign final task IDs and remap priorities to [0, 300]
    n_tasks = len(tasks)
    order_first_task = [N_TIME] * n_orders
    for new_id, task in enumerate(tasks):
        task.task_id  = data.state.task_counter + new_id
        task.priority = (new_id / (n_tasks - 1) * 300) if n_tasks > 1 else 0

        involved = {o_id for v in task.stops for o_id in v.orders}
        for m, o in enumerate(data.orders):
            if o.order_id in involved:
                order_first_task[m] = min(order_first_task[m], task.priority)

    # Step 6: at each workstation, open orders by opening time first so the
    # opening sequence matches the windows the tasks were built on; task
    # priority only breaks ties.
    ordered_orders_by_w = {}
    for w, idxs in enumerate(orders_by_workstation):
        ordered_orders_by_w[w] = sorted(
            idxs,
            key=lambda m: (order_start_time.get(m, N_TIME), order_first_task[m]),
        )

    return data.orders, ordered_orders_by_w, tasks