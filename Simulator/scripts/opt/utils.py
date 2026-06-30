from collections import defaultdict
from scripts.core.entities import Task, Visit


def _first_pick_time(x_sol, im, N_TIME):
    """
    Return the timestep at which item `im` is first picked.
    x_sol[im, t] is cumulative (0001111), so the pick happens at the first t
    where x_sol[im, t] transitions from 0 to 1.
    Returns None if the item is never picked within the horizon.
    """
    if x_sol[im, 0] > 0.5:
        return 0
    for t in range(1, N_TIME):
        if x_sol[im, t] > 0.5 and x_sol[im, t - 1] < 0.5:
            return t
    return None


def _compute_priority(stops, relevant_pairs_for_x, orders, item_to_time, N_TIME):
    """
    Return the earliest pick time among all items in the first stop of a task.
    Falls back to N_TIME + 2 if no valid time is found.
    """
    pr = N_TIME + 2
    first_stop = stops[0]
    for i, m in relevant_pairs_for_x:
        if i in first_stop.items and orders[m].order_id in first_stop.orders:
            t = item_to_time.get((i, m))
            if t is not None:
                pr = min(pr, t)
    return pr


def _build_task_stops(current_stops, pick_at):
    """
    Convert a list of (t_arrival, w_idx) pod stops into Visit objects,
    keeping only stops where items are actually picked.
    """
    stops = []
    for t_arr, w_idx in current_stops:
        pick_data = pick_at.get((t_arr, w_idx))  # keyed by (t, w) after pod loop
        if pick_data and pick_data["items"]:
            stops.append(Visit(
                workstation_id=w_idx,
                orders=pick_data["orders"],
                items=pick_data["items"],
            ))
    return stops


def convert_OptSol_to_SimObj(data, x_sol, v_sol, y_sol):
    """
    Convert Stage-2 binary decision variables into simulator objects.

    Variables (all indexed over the optimisation horizon):
      x_sol[im, t]    : cumulative; 1 if item `im` has been picked by time t
      v_sol[m,  t]    : 1 if order m is open at time t
      y_sol[rel_p, a] : 1 if pod rel_p traverses arc a

    Returns:
      orders               : list of Order objects
      ordered_orders_by_w  : {ws_idx: [order_idx, ...]} sorted by start time
      tasks                : list of Task objects ready for the simulator
    """
    n_orders             = len(data.orders)
    relevant_pairs_for_x = data.relevant_pairs_for_x
    N_TIME               = data.OptManager.N_TIME
    warehouse            = data.state.warehouse   # single source of truth

    # ------------------------------------------------------------------
    # Step 1: workstation assignments and order start times
    # ------------------------------------------------------------------
    orders_by_workstation = [lst.copy() for lst in data.orders_by_workstation]
    order_start_time: dict[int, int] = {}

    for m in range(n_orders):
        w = data.order_to_ws[m]
        if data.orders[m].order_id in warehouse.workstations[w].opened_orders:
            # Already-open orders are available immediately
            start_t = 0
        else:
            # First timestep where the order becomes active
            start_t = next(
                (t for t in range(N_TIME) if v_sol[m, t] > 0.5),
                None,
            )

        if start_t is None:
            # Order never activated — drop it from the workstation list
            orders_by_workstation[w].remove(m)
        else:
            order_start_time[m] = start_t

    # ------------------------------------------------------------------
    # Step 2: item → pod and item → pick-time lookups
    # ------------------------------------------------------------------
    order_to_ws = data.order_to_ws   # order_idx → ws_idx
    item_to_pod = data.pod_of_item   # im → pod_id

    # (sku_idx, order_idx) → timestep of first pick
    item_to_time: dict[tuple[int, int], int] = {}
    for im, (i, m) in enumerate(relevant_pairs_for_x):
        t = _first_pick_time(x_sol, im, N_TIME)
        if t is not None:
            item_to_time[(i, m)] = t

    # ------------------------------------------------------------------
    # Step 3: build pick_at index
    # pick_at[(pod_id, t, w_idx)] → {items, orders} to service at that stop
    # ------------------------------------------------------------------
    pick_at: dict[tuple[int, int, int], dict] = defaultdict(
        lambda: {"items": set(), "orders": set()}
    )
    for im, (i, m) in enumerate(relevant_pairs_for_x):
        t = item_to_time.get((i, m))
        w = order_to_ws.get(m)
        if t is None or w is None:
            continue
        p_id = item_to_pod[im]
        pick_at[(p_id, t, w)]["items"].add(i)
        pick_at[(p_id, t, w)]["orders"].add(data.orders[m].order_id)

    # ------------------------------------------------------------------
    # Step 4: reconstruct pod trajectories from y_sol → Tasks
    # ------------------------------------------------------------------
    workstation_positions = set(data.OptManager._W)
    storage_positions     = set(data.OptManager._L)
    pos_to_ws = {
        warehouse.workstations[w].position: w
        for w in range(data.OptManager.n_workstations)
    }

    tasks: list[Task] = []

    for rel_p, p_id in enumerate(data.from_RelPod_to_PodId):

        # Collect all arcs traversed by this pod, sorted by departure time
        traversed = sorted(
            [
                (src[1], src[0], dst[1], dst[0])   # (t_src, loc_src, t_dst, loc_dst)
                for a_idx, (src, dst) in enumerate(data.OptManager.all_arcs)
                if y_sol[rel_p, a_idx] > 0.5
            ],
            key=lambda k: k[0],
        )

        # Walk the trajectory and emit a Task each time the pod returns to storage.
        # A "trip" is the sequence of workstation visits between two storage stays.
        current_stops: list[tuple[int, int]] = []   # (t_arrival, w_idx)
        in_trip = False

        for t_src, loc_src, t_dst, loc_dst in traversed:

            if loc_dst in workstation_positions:
                # Pod arrives at a workstation — record the stop
                w_idx = pos_to_ws[loc_dst]
                current_stops.append((t_dst, w_idx))
                in_trip = True

            elif loc_dst in storage_positions and in_trip:
                # Pod returns to storage — close the current trip as a Task
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
                    pr = _compute_priority(
                        stops, relevant_pairs_for_x, data.orders, item_to_time, N_TIME
                    )
                    tasks.append(Task(
                        task_id=None,
                        pod_id=p_id,
                        robot_id=None,
                        stops=stops,
                        priority=pr,
                    ))
                current_stops = []
                in_trip = False

        # Handle a trip still open at end of horizon (pod never returned to storage)
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
                pr = _compute_priority(
                    stops, relevant_pairs_for_x, data.orders, item_to_time, N_TIME
                )
                tasks.append(Task(
                    task_id=None,
                    pod_id=p_id,
                    robot_id=None,
                    stops=stops,
                    priority=pr,
                ))

    # ------------------------------------------------------------------
    # Step 5: refine priorities using travel time and sort tasks
    # ------------------------------------------------------------------
    for task in tasks:
        t_firstpick = task.priority
        pod = warehouse.pods[task.pod_id]
        ws  = warehouse.workstations[task.stops[0].workstation_id]
        # Adjust: earlier pick time → higher priority; subtract travel lead time
        travel_lead = (
            1.1 * warehouse.travel_time(
                warehouse.cell2coord(pod.storage_location),
                warehouse.cell2coord(ws.position),
            )
        ) / data.OptManager.TIME_UNIT
        task.priority = t_firstpick - 0.25 * travel_lead

    # Sort tasks by adjusted priority (ascending = more urgent first)
    tasks.sort(key=lambda t: t.priority)

    # Assign final integer task IDs and remap priorities to [0, 300]
    n_tasks = len(tasks)
    order_first_task = [N_TIME] * n_orders

    for new_id, task in enumerate(tasks):
        task.task_id = data.state.task_counter + new_id
        task.priority = (new_id / (n_tasks - 1) * 300) if n_tasks > 1 else 0

        # Track the earliest task priority seen for each order
        involved_orders = {o_id for v in task.stops for o_id in v.orders}
        for m, o in enumerate(data.orders):
            if o.order_id in involved_orders:
                order_first_task[m] = min(order_first_task[m], task.priority)

    # ------------------------------------------------------------------
    # Step 6: sort orders within each workstation by (start_time, first_task)
    # ------------------------------------------------------------------
    ordered_orders_by_w = {
        w: sorted(
            idxs,
            key=lambda m: (
                order_start_time.get(m, N_TIME),   # safe fallback if missing
                order_first_task[m],
            ),
        )
        for w, idxs in enumerate(orders_by_workstation)
    }

    return data.orders, ordered_orders_by_w, tasks