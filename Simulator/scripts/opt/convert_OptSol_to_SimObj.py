from dataclasses import dataclass, field

from scripts.core.entities import Task, Visit


@dataclass
class _Group:
    """The items and orders one pod picks at one workstation within one opening
    window. first_pick is the earliest pick timestep, kept to order stops and set
    a task's priority."""
    items:      set   = field(default_factory=set)
    orders:     set   = field(default_factory=set)
    first_pick: float = float("inf")

    def add(self, item, order_id, t_pick):
        self.items.add(item)
        self.orders.add(order_id)
        self.first_pick = min(self.first_pick, t_pick)

    def merge(self, other):
        self.items      |= other.items
        self.orders     |= other.orders
        self.first_pick  = min(self.first_pick, other.first_pick)


def _first_pick_time(x_sol, im, N_TIME):
    """x_sol is cumulative (0001111), so the pick is simply the first t at 1."""
    for t in range(N_TIME):
        if x_sol[im, t] > 0.5:
            return t
    return None


def convert_OptSol_to_SimObj(data, x_sol, v_sol, y_sol=None, gap_factor=1.0):
    """Turn the Stage 2 solution into orders, an opening order per workstation,
    and tasks. Tasks say which pod brings which items where; routing is left to
    the simulator. y_sol is unused, kept only for a stable signature.

    A task is one pod's run through orders that open close together in time. The
    cut is driven by the order opening windows, not by pick timesteps: opening
    order is the robust part of the plan, and cutting on it never splits a single
    order (however many SKUs it spans) across two tasks of the same pod. For each
    pod its groups are split into a new task whenever the gap between consecutive
    opening windows exceeds gap_factor times the pod's round trip home: larger
    keeps the pod out across bigger gaps, smaller sends it home sooner, and
    because the threshold scales with distance a far pod (expensive to shuttle) is
    kept out longer than a near one. Task priority is the earliest pick time of
    the task, remapped to [0, 300] at the end."""
    orders    = data.orders
    N_TIME    = data.OptManager.N_TIME
    TIME_UNIT = data.OptManager.TIME_UNIT
    warehouse = data.state.warehouse

    def travel_steps(p_id, w_idx):
        """One-way storage(pod) -> workstation travel time, in timesteps."""
        pod = warehouse.pods[p_id]
        ws  = warehouse.workstations[w_idx]
        return warehouse.travel_time(
            warehouse.cell2coord(pod.storage_location),
            warehouse.cell2coord(ws.position),
        ) / TIME_UNIT

    def build_task(p_id, chunk):
        """Merge a chunk of same-pod groups into one Task: groups at the same
        workstation collapse into one stop, stops are ordered by first pick, and
        the priority is the task's earliest pick time."""
        by_w = {}
        for _window, w, group in chunk:
            by_w.setdefault(w, _Group()).merge(group)

        stops_sorted = sorted(by_w.items(), key=lambda kv: kv[1].first_pick)
        stops = [Visit(workstation_id=w, orders=g.orders, items=g.items)
                 for w, g in stops_sorted]

        earliest_pick = stops_sorted[0][1].first_pick
        return Task(task_id=None, pod_id=p_id, robot_id=None,
                    stops=stops, priority=earliest_pick)

    # 1. When does each order open (its window)? Drop orders that never open.
    orders_by_workstation = [lst.copy() for lst in data.orders_by_workstation]
    order_start_time = {}
    for m in range(len(orders)):
        w = data.order_to_ws[m]
        if orders[m].order_id in warehouse.workstations[w].opened_orders:
            order_start_time[m] = 0                       # already open
            continue
        for t in range(N_TIME):                           # first active timestep
            if v_sol[m, t] > 0.5:
                order_start_time[m] = t
                break
        else:
            orders_by_workstation[w].remove(m)            # never opens

    # 2. Group every pick by (pod, opening window, workstation).
    groups = {}
    for im, (i, m) in enumerate(data.relevant_pairs_for_x):
        if m not in order_start_time:
            continue                                      # order was dropped
        t_pick = _first_pick_time(x_sol, im, N_TIME)
        if t_pick is None:
            continue                                      # item never picked
        key = (data.pod_of_item[im], order_start_time[m], data.order_to_ws[m])
        groups.setdefault(key, _Group()).add(i, orders[m].order_id, t_pick)

    # 3. For each pod, walk its groups in opening-window order and split off a new
    #    task when the gap between consecutive windows exceeds gap_factor times the
    #    pod's round trip home. Orders that open close together stay in one run; a
    #    big jump means the pod would sit out too long between them, so it goes
    #    home and comes back as a new task.
    by_pod = {}
    for (p_id, window, w), group in groups.items():
        by_pod.setdefault(p_id, []).append((window, w, group))

    tasks = []
    for p_id, entries in by_pod.items():
        entries.sort(key=lambda e: (e[0], e[2].first_pick))   # by opening window

        chunk = [entries[0]]
        for prev, curr in zip(entries, entries[1:]):
            window_gap = curr[0] - prev[0]
            round_trip = travel_steps(p_id, prev[1]) + travel_steps(p_id, curr[1])
            if window_gap > gap_factor * round_trip:
                tasks.append(build_task(p_id, chunk))
                chunk = []
            chunk.append(curr)
        tasks.append(build_task(p_id, chunk))

    # 4. Sort tasks by priority (earliest pick first), assign ids and remap
    #    priorities to [0, 300]. Track the first task each order appears in.
    tasks.sort(key=lambda t: t.priority)
    order_first_task = [N_TIME] * len(orders)
    for new_id, task in enumerate(tasks):
        task.task_id  = data.state.task_counter + new_id
        task.priority = new_id / (len(tasks) - 1) * 300 if len(tasks) > 1 else 0
        involved = {o_id for v in task.stops for o_id in v.orders}
        for m, o in enumerate(orders):
            if o.order_id in involved:
                order_first_task[m] = min(order_first_task[m], task.priority)

    # 5. At each workstation, open orders by opening time; task priority only
    #    breaks ties.
    ordered_orders_by_w = {
        w: sorted(idxs, key=lambda m: (order_start_time.get(m, N_TIME),
                                       order_first_task[m]))
        for w, idxs in enumerate(orders_by_workstation)
    }

    return orders, ordered_orders_by_w, tasks