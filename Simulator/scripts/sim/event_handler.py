"""
Event handlers for the warehouse simulator.

Each handler has signature ``(event, state, sim)`` where:
- *state* : SimulatorState  — all mutable queues, counters, and warehouse.
- *sim*   : Simulator       — immutable config (sim.config) and RNG (sim.RANDOM_GENERATOR).
"""

import logging, time, copy
from Simulator.scripts.core.entities import Order, Event, Task, Visit
from Simulator.scripts.core.warehouse import Warehouse
from Simulator.scripts.core.enums import OrderStatus, RobotStatus, PodStatus, WorkstationPickingStatus, EventType
from Simulator.scripts.opt.policies import assign_order_to_workstation_policy, design_tasks_for_ws, get_nearest_idle_robot
from Simulator.scripts.core.queues import PriorityQueue

TIME_LIMIT_AT_WS = 300

def arrival_order(event: Event, state, sim) -> None:
    """
    Handle a new customer order arrival.

    Generates an order following Barnhart et al. 2024: single-item with
    probability p, otherwise geometric(p) + 2 items. Adds the order to
    the system backlog and schedules the next order arrival.

    If optimization is disabled, immediately assigns the order to the
    least-loaded workstation and opens it if a slot is available.
    """
    assert len(sim.config.order_gen_config) == 3, (
        f"order_gen_config must have 3 elements (interarrival, p_single, p_geo), "
        f"got {len(sim.config.order_gen_config)}"
    )

    n_order_to_generate = event.info 

    for _ in range(n_order_to_generate):

        order_id = state.orders_counter
        state.orders_counter += 1

        rnd = sim.RANDOM_GENERATOR.random()
        if rnd < sim.config.order_gen_config[1]:
            order_size = 1
        else:
            order_size = sim.RANDOM_GENERATOR.geometric(p=sim.config.order_gen_config[2]) + 2

        sku_list = [
            _sample_sku(sim.RANDOM_GENERATOR, state.warehouse.num_skus)
            for _ in range(order_size)
        ]

        o = Order(
            order_id=order_id,
            arrival_time=state.current_time,
            order_size=order_size,
            items_required=set(sku_list),
            items_pending=set(sku_list),
            workstation_id=None,
            status=OrderStatus.BACKLOG
        )
        state.orders_in_system.push(o)

        logging.debug("Order %i arrived: items_required = %s.     [orders_in_system = %i]",
                    order_id, sku_list, state.orders_counter - _count_closed(state))
        
        if not sim.config.optimization_enabled:
            st = time.time()
            workstation_id = assign_order_to_workstation_policy(
                o,
                state.warehouse.workstations
            )
            sim.STAT_MANAGER.decisions_computing_time += time.time() - st
            workstation = state.warehouse.get_workstation(workstation_id)
            o.workstation_id = workstation_id
            o.status = OrderStatus.WAITING

            if workstation.has_open_slot():
                state.future_events.push(Event(
                        time=state.current_time,
                        type=EventType.OPEN_ORDER,
                        info=o
                    ))
            else:
                workstation.order_buffer.append(order_id)
                logging.debug("Order queued at workstation %i.   [order_queue len = %i]",
                            workstation_id, len(workstation.order_buffer))

    # Schedule next arrival
    interarrival_time = sim.config.order_gen_config[0]
    state.future_events.push(Event(
        time=state.current_time + interarrival_time,
        type=EventType.ARRIVAL_ORDER, 
        info = 1
    ))



def open_order(event: Event, state, sim) -> None:
    """
    Open an order at its assigned workstation.

    Transitions the order from WAITING to OPEN status and registers it
    in the workstation's active orders. If optimization is disabled,
    immediately designs tasks to fetch pods matching the order SKUs.
    """
    o = event.info

    if o.workstation_id is None:
        raise ValueError(f"Cannot open order {o.order_id}: no workstation assigned")

    assert o.status == OrderStatus.WAITING, (
        f"Order {o.order_id} state transition error: expected WAITING, got {o.status.name}"
    )

    workstation = state.warehouse.get_workstation(o.workstation_id)

    if len(workstation.opened_orders) > workstation.order_capacity -1:
        workstation.order_buffer.append(o.order_id)
        # Can happen when multiple orders arrive at once
        logging.debug("Order %i queued at workstation %i.   [order_queue len = %i]",
                            o.order_id, o.workstation_id, len(workstation.order_buffer))
        return

    o.status = OrderStatus.OPEN
    workstation.opened_orders.add(o.order_id)

    sim.STAT_MANAGER.update_statistic(
        type='WS_AVG_OO',
        info=[workstation.workstation_id, +1, state.current_time]
    )

    logging.debug("Order %i (skus required = %s) opened at workstation %i.   [open_orders = %i/%i]",
                  o.order_id, o.items_required, o.workstation_id,
                  len(workstation.opened_orders), workstation.order_capacity)

    if not sim.config.optimization_enabled:
        st = time.time()
        tasks, state.task_counter = design_tasks_for_ws(
            workstation=workstation,
            warehouse=state.warehouse,
            orders_in_system=state.orders_in_system,
            task_counter=state.task_counter,
            active_tasks=state.active_tasks
        )
        sim.STAT_MANAGER.decisions_computing_time += time.time() - st

        if tasks:
            total_items = sum(len(t.stops[0].items) for t in tasks)
            logging.debug("%i task(s) designed covering %i sku(s) required", len(tasks), total_items)
            for t in tasks:
                state.future_events.push(Event(
                    time=state.current_time,
                    type=EventType.RELEASE_TASK,
                    info=t
                ))
        else:
            logging.debug("No tasks designed (all SKUs already covered)")

    elif len(state.released_tasks) > 0:
        state.future_events.push(Event(
                    time=state.current_time,
                    type=EventType.START_TASK
                ))



def release_task(event: Event, state, sim) -> None:
    """
    Release a task to the execution queue.

    Moves a task from scheduled_tasks or event payload to released_tasks,
    making it eligible for execution by an idle robot. Immediately triggers
    START_TASK if an idle robot is available.
    """
    if sim.config.optimization_enabled:
        task = event.info
    else:
        task = event.info

    assert task is not None, "release_task: task is None after retrieval"

    if state.released_tasks.get(task.task_id) is not None:
        state.released_tasks.update(task)
    else:
        state.released_tasks.push(task)

    ws_list = []
    for visit in task.stops:
        ws = state.warehouse.get_workstation(visit.workstation_id)
        ws.released_tasks.add(task.task_id)
        ws_list.append(visit.workstation_id)

    logging.debug("Task %i released: pod %i required by workstation(s) %s.  [released tasks = %i]",
                  task.task_id, task.pod_id, ws_list, len(state.released_tasks))

    has_idle_robot = any(r.status == RobotStatus.IDLE for r in state.warehouse.robots)
    if has_idle_robot:
        state.future_events.push(Event(time=state.current_time, type=EventType.START_TASK))


def start_task(event: Event, state, sim) -> None:
    """
    Start executing a task with the nearest idle robot.

    Assigns the highest-priority released task to the nearest idle robot.
    Skips tasks whose pod is currently BUSY, re-queuing them for later.
    Updates pod and robot status to BUSY, registers visits as active,
    and schedules pod arrival at the first workstation.
    """
    if state.released_tasks.is_empty():
        return
    
    skipped_t = []
    task = None

    while not state.released_tasks.is_empty():
        candidate = state.released_tasks.pop()

        if sim.config.optimization_enabled:
            valid = any(
                o in state.warehouse.workstations[v.workstation_id].opened_orders.union(state.warehouse.workstations[v.workstation_id].order_buffer[:2])
                for v in candidate.stops
                for o in v.orders
            )
            if not valid:
                logging.debug("Task %i blocked: no orders in %s is open yet.  [released tasks = %i]",
                            candidate.task_id, {v.workstation_id : [o for o in v.orders] for v in candidate.stops}, len(state.released_tasks))
                skipped_t.append(candidate)
                continue   # passa al prossimo candidato 

        pod = state.warehouse.get_pod(candidate.pod_id)
        if pod.status == PodStatus.IDLE:
            task = candidate
            break

        logging.debug("Task %i blocked: pod %i not idle.  [released tasks = %i]",
                    candidate.task_id, candidate.pod_id, len(state.released_tasks))
        skipped_t.append(candidate)

    for t in skipped_t:
        state.released_tasks.push(t)

    if task is None:
        return

    pod = state.warehouse.get_pod(task.pod_id)
    assert pod.status == PodStatus.IDLE, f"Pod {task.pod_id} should be IDLE before task start"

    robot_id = get_nearest_idle_robot(pod, state.warehouse)
    if robot_id is None:
        logging.debug("Task %i blocked: no idle robots.", task.task_id)
        state.released_tasks.push(task)
        return

    robot = state.warehouse.get_robot(robot_id)
    assert robot.status == RobotStatus.IDLE, f"Robot {robot_id} should be IDLE before task start"

    state.active_tasks[task.task_id] = task
    pod.status   = PodStatus.BUSY
    robot.status = RobotStatus.BUSY
    task.robot_id = robot_id

    sim.STAT_MANAGER.update_statistic(
        type='RB_FREQ',
        info=[robot.robot_id, RobotStatus.BUSY, state.current_time]
    )

    for visit in task.stops:
        ws = state.warehouse.get_workstation(visit.workstation_id)
        ws.active_tasks.add(task.task_id)
        ws.released_tasks.discard(task.task_id)

    first_visit = task.stops[0]
    first_workstation = state.warehouse.get_workstation(first_visit.workstation_id)
    travel_time = state.warehouse.travel_time(
        state.warehouse.cell2coord(pod.storage_location),
        state.warehouse.cell2coord(first_workstation.position),
        sim.RANDOM_GENERATOR
    )
    state.future_events.push(Event(
        time=state.current_time + travel_time,
        type=EventType.ARRIVAL_POD_WST,
        info=task
    ))

    idle_robots = sum(1 for r in state.warehouse.robots if r.status == RobotStatus.IDLE)
    logging.debug(
        "Task %i started: robot %i → pod %i → workstation %i for orders %s (arrival = %.1f s).   [idle robots = %i/%i]",
        task.task_id, task.robot_id, task.pod_id, first_visit.workstation_id,
        first_visit.orders, state.current_time + travel_time,
        idle_robots, len(state.warehouse.robots)
    )

    # Updating stats
    sim.STAT_MANAGER.update_statistic(
        type='POD_AVG_MOVING',
        info=[+1, state.current_time]
    )
    


def arrival_pod_wst(event: Event, state, sim) -> None:
    """
    Handle pod arrival at a workstation.

    Updates robot position and checks if the workstation is idle.
    If idle, immediately starts picking; otherwise, queues the pod.
    """
    task = event.info
    current_visit = task.stops[0]
    workstation = state.warehouse.get_workstation(current_visit.workstation_id)
    robot = state.warehouse.get_robot(task.robot_id)

    assert robot.status == RobotStatus.BUSY, (
        f"Robot {task.robot_id} should be BUSY on pod arrival, got {robot.status.name}"
    )

    robot.position = workstation.position

    logging.debug("Pod %i arrived at workstation %i - status = %s",
                  task.pod_id, current_visit.workstation_id, workstation.status.name)

    if workstation.status == WorkstationPickingStatus.IDLE \
        and any(o in workstation.opened_orders for o in task.stops[0].orders):
        state.future_events.push(Event(
            time=state.current_time,
            type=EventType.START_PICKING,
            info=task
        ))
    else:
        workstation.picking_buffer[(task.task_id)] = state.current_time
        logging.debug("Pod queued at workstation.    [picking_buffer = %s]",
                      workstation.picking_buffer)


def start_picking(event: Event, state, sim) -> None:
    """
    Start picking items from an arrived pod.

    Sets workstation status to BUSY and schedules picking completion
    based on the number of items at this visit.
    """
    task        = event.info
    visit       = task.stops[0]
    workstation = state.warehouse.get_workstation(visit.workstation_id)

    assert workstation.status == WorkstationPickingStatus.IDLE, (
        f"Workstation {visit.workstation_id} should be IDLE before picking, "
        f"got {workstation.status.name}"
    )
    assert task.task_id in state.active_tasks, (
        f"Task {task.task_id} not found in active_tasks at start_picking"
    )

    workstation.status = WorkstationPickingStatus.BUSY
    sim.STAT_MANAGER.update_statistic(
        type='WS_FREQ',
        info=[workstation.workstation_id, WorkstationPickingStatus.BUSY, state.current_time]
    )

    logging.debug("Processing task %i at workstation %i: picking items %s for orders %s",
                  task.task_id, visit.workstation_id, visit.items, visit.orders)

    picking_time = workstation.estimated_picking_time(len(visit.items))
    state.future_events.push(Event(
        time=state.current_time + picking_time,
        type=EventType.END_PICKING,
        info=task
    ))


def end_picking(event: Event, state, sim) -> None:
    """
    Handle picking completion at a workstation.

    Drains the picking buffer first (unconditionally), then schedules the
    pod's next stop or return to storage. Finally updates order states and
    closes any completed orders. Task redesign is skipped if any order closed
    (the close_order handler will open a new one and trigger redesign).
    """
    task             = event.info
    completed_visit  = task.stops[0]
    workstation      = state.warehouse.get_workstation(completed_visit.workstation_id)

    assert workstation.status == WorkstationPickingStatus.BUSY, (
        f"Workstation {completed_visit.workstation_id} should be BUSY at end_picking, "
        f"got {workstation.status.name}"
    )

    workstation.status = WorkstationPickingStatus.IDLE
    workstation.active_tasks.discard(task.task_id)

    sim.STAT_MANAGER.update_statistic(
        type='WS_FREQ',
        info=[workstation.workstation_id, WorkstationPickingStatus.IDLE, state.current_time]
    )

    logging.debug("Ended picking at workstation %i.    [picking_buffer len = %s]",
                   completed_visit.workstation_id, workstation.picking_buffer)
    logging.debug("Task should have served orders %s, found open orders %s.",
                  completed_visit.orders, workstation.opened_orders)

    # Drain picking buffer 
    if workstation.picking_buffer:
        
        next_task_id = None
        for id_t, t_arr in sorted(workstation.picking_buffer.items(), key=lambda x: x[1]):
            if next_task_id is None:
                t = state.active_tasks.get(id_t)

                if any(o in workstation.opened_orders for o in t.stops[0].orders):
                    next_task_id = id_t
                    workstation.picking_buffer.pop(id_t)
                    break

        # Check if some pod got stuck in the buffer
        for id_t, t_arr in list(workstation.picking_buffer.items()):
            t = state.active_tasks.get(id_t)
            if state.current_time - t_arr > TIME_LIMIT_AT_WS:

                # Performing pop/remove I would do if Visit was done
                t.stops.pop(0) # would be done in end_picking
                workstation.picking_buffer.pop(id_t) # would be done in start_picking
                workstation.active_tasks.discard(id_t) # would be done in end_picking

                if len(t.stops) == 0: 
                    pod_t = state.warehouse.pods[t.pod_id]
                    travel_time = state.warehouse.travel_time(
                        state.warehouse.cell2coord(workstation.position),
                        state.warehouse.cell2coord(pod_t.storage_location),
                        sim.RANDOM_GENERATOR
                    )

                    state.future_events.push(Event(
                        time=state.current_time + travel_time,
                        type=EventType.RETURN_POD,
                        info=t
                    ))
                    logging.info("Task %i got stuck at workstation %i -> heading to the pod storage location.",
                             id_t, workstation.workstation_id)
                else:
                    next_ws = state.warehouse.workstations[t.stops[0].workstation_id]
                    travel_time = state.warehouse.travel_time(
                        state.warehouse.cell2coord(workstation.position),
                        state.warehouse.cell2coord(next_ws.position),
                        sim.RANDOM_GENERATOR
                    )

                    state.future_events.push(Event(
                        time=state.current_time + travel_time,
                        type=EventType.ARRIVAL_POD_WST,
                        info=t
                    ))
                    logging.info("Task %i got stuck at workstation %i -> heading to next workstation.",
                             id_t, next_ws.workstation_id)



        if next_task_id is not None:                
            next_task = state.active_tasks.get(next_task_id)
            assert next_task is not None, (
                f"Task {next_task_id} in picking_buffer but not in active_tasks"
            )
            state.future_events.push(Event(
                time=state.current_time,
                type=EventType.START_PICKING,
                info=next_task
            ))
            logging.debug("Dequeued task %i from picking buffer at workstation %i.   [buffer len = %i]",
                        next_task_id, workstation.workstation_id, len(workstation.picking_buffer))


    

    # Update order states 
    completed_orders = []
    list_o = list(completed_visit.orders)
    for order_id in list_o:
        if order_id in workstation.opened_orders:
            completed_visit.orders.remove(order_id)
            order = state.orders_in_system.get(order_id)
            if order is None:
                continue

            # Updating statistics
            if sim.STAT_MANAGER.WARM_UP <= state.current_time:
                sim.STAT_MANAGER.throughput += len(order.items_pending and completed_visit.items)

            order.items_pending -= completed_visit.items
            assert len(order.items_pending) >= 0, (
                f"Order {order_id} has negative pending items after picking"
            )
            if len(order.items_pending) == 0:
                completed_orders.append(order_id)
                state.future_events.push(Event(
                    time=state.current_time,
                    type=EventType.CLOSE_ORDER,
                    info=order
                ))

    # If some orders that should be served was not I re-schedule the Visit
    task.stops[0] = completed_visit
    if len(task.stops[0].orders) == 0 :
        task.stops.pop(0)
 
    # Schedule next stop or pod return
    if len(task.stops) == 0:
        pod = state.warehouse.get_pod(task.pod_id)
        return_travel_time = state.warehouse.travel_time(
            state.warehouse.cell2coord(workstation.position),
            state.warehouse.cell2coord(pod.storage_location),
            sim.RANDOM_GENERATOR
        )
        state.future_events.push(Event(
            time=state.current_time + return_travel_time,
            type=EventType.RETURN_POD,
            info=task
        ))
        logging.debug("Task %i completed: pod %i returning to storage.", task.task_id, task.pod_id)
    else:
        next_visit = task.stops[0]
        next_workstation = state.warehouse.get_workstation(next_visit.workstation_id)
        travel_time = state.warehouse.travel_time(
            state.warehouse.cell2coord(workstation.position),
            state.warehouse.cell2coord(next_workstation.position),
            sim.RANDOM_GENERATOR
        )
        state.future_events.push(Event(
            time=state.current_time + travel_time,
            type=EventType.ARRIVAL_POD_WST,
            info=task
        ))
        logging.debug("Task %i heading to workstation %i.", task.task_id, next_visit.workstation_id)


    # Skip redesign if any order closed: close_order will handle it
    if completed_orders:
        return

    if not sim.config.optimization_enabled:
        st = time.time()
        new_tasks, state.task_counter = design_tasks_for_ws(
            workstation=workstation,
            warehouse=state.warehouse,
            orders_in_system=state.orders_in_system,
            task_counter=state.task_counter,
            active_tasks=state.active_tasks
        )
        sim.STAT_MANAGER.decisions_computing_time += time.time() - st
        if new_tasks:
            for new_task in new_tasks:
                state.future_events.push(Event(
                    time=state.current_time,
                    type=EventType.RELEASE_TASK,
                    info=new_task
                ))
            logging.debug("Redesigned %i task(s) for workstation %i",
                          len(new_tasks), workstation.workstation_id)


def return_pod(event: Event, state, sim) -> None:
    """
    Return a pod to its storage location after task completion.

    Releases the robot and pod, marking them as IDLE. Triggers execution
    of the next available task if any exist.
    """
    task  = event.info
    pod   = state.warehouse.get_pod(task.pod_id)
    robot = state.warehouse.get_robot(task.robot_id)

    assert len(task.stops) == 0, (
        f"Task {task.task_id} has remaining stops at return: "
        f"{[v.workstation_id for v in task.stops]}"
    )
    assert pod.status   == PodStatus.BUSY,   f"Pod {task.pod_id} should be BUSY at return"
    assert robot.status == RobotStatus.BUSY, f"Robot {task.robot_id} should be BUSY at return"

    robot.status = RobotStatus.IDLE
    robot.position = pod.storage_location
    pod.status = PodStatus.IDLE
    state.active_tasks.pop(task.task_id)

    sim.STAT_MANAGER.update_statistic(
        type='RB_FREQ',
        info=[robot.robot_id, RobotStatus.IDLE, state.current_time]
    )

    idle_robots = sum(1 for r in state.warehouse.robots if r.status == RobotStatus.IDLE)
    logging.debug(
        "Pod %i returned. Robot %i idle.   [idle robots = %i/%i, released tasks = %i]",
        pod.pod_id, robot.robot_id,
        idle_robots, len(state.warehouse.robots), len(state.released_tasks)
    )

    if not state.released_tasks.is_empty():
        state.future_events.push(Event(time=state.current_time, type=EventType.START_TASK))

     # Updating stats
    sim.STAT_MANAGER.update_statistic(
        type='POD_AVG_MOVING',
        info=[-1, state.current_time]
    )


def close_order(event: Event, state, sim) -> None:
    """
    Close a completed order and attempt to open the next queued order.

    Transitions order to CLOSED status and removes it from the workstation.
    If there are pending orders in the workstation queue, opens the first one.
    """
    order = event.info

    assert len(order.items_pending) == 0, (
        f"Cannot close order {order.order_id}: {len(order.items_pending)} items still pending"
    )
    assert order.status == OrderStatus.OPEN, (
        f"Order {order.order_id} expected OPEN at close, got {order.status.name}"
    )

    workstation  = state.warehouse.get_workstation(order.workstation_id)
    order.status  = OrderStatus.CLOSED
    workstation.opened_orders.discard(order.order_id)

    sim.STAT_MANAGER.update_statistic(
        type='WS_AVG_OO',
        info=[workstation.workstation_id, -1, state.current_time]
    )
    
    sim.STAT_MANAGER.update_statistic(type='OFT', info=[order, state.current_time])

    flow_time = state.current_time - order.arrival_time
    logging.debug(
        "Order %i closed at workstation %i: flow_time = %.1f s.     "
        "[order_buffer = %i] [open_orders = %i/%i]",
        order.order_id, workstation.workstation_id, flow_time,
        len(workstation.order_buffer),
        len(workstation.opened_orders), workstation.order_capacity
    )

    if workstation.order_buffer:
        next_order_id = workstation.order_buffer.pop(0)
        next_order    = state.orders_in_system.get(next_order_id)
        if next_order is not None:
            state.future_events.push(Event(
                time=state.current_time,
                type=EventType.OPEN_ORDER,
                info=next_order
            ))
            logging.debug("Next order in buffer: %i", next_order_id)


def run_optimizer(event: Event, state, sim) -> None:
    """
    Execute one optimization cycle: design tasks, assign orders to workstations,
    schedule release events, and queue the next optimization run.
    """

    # Release task stuck at picking buffer
    for ws in state.warehouse.workstations:
        for id_t in list(ws.picking_buffer.keys()):
            t = state.active_tasks.get(id_t)
            # Deleting next stops (for the sake of simplicity)
            for v in t.stops:
                state.warehouse.workstations[v.workstation_id].active_tasks.discard(id_t)

            t.stops = [t.stops[0]]
            available_order = t.stops[0].orders & ws.opened_orders
            if len(available_order) == 0:
                t.stops.pop(0)
                ws.picking_buffer.pop(id_t)

                pod_t = state.warehouse.pods[t.pod_id]
                travel_time = state.warehouse.travel_time(
                    state.warehouse.cell2coord(ws.position),
                    state.warehouse.cell2coord(pod_t.storage_location),
                    sim.RANDOM_GENERATOR
                )

                state.future_events.push(Event(
                    time=state.current_time + travel_time,
                    type=EventType.RETURN_POD,
                    info=t
                ))
            else:
                t.stops[0].orders = available_order
                

    st = time.time()
    orders, ordered_orders_by_w, tasks = sim.OPT_MANAGER.solve_task_design_and_assignment(sim, state)
    sim.STAT_MANAGER.decisions_computing_time += time.time() - st

    # Reset released_tasks queue
    state.released_tasks = PriorityQueue(
        key=lambda t: (
            state.warehouse.pods[t.pod_id].status != PodStatus.IDLE,
            t.priority,
        ),
        id_attr="task_id",
    )

    # Flush RELEASE_TASK events from the future queue, preserving all other event types
    ev_l = []
    while len(state.future_events) > 0:
        e = state.future_events.pop()
        if e.type != EventType.RELEASE_TASK:
            ev_l.append(e)
    for ev in ev_l:
        state.future_events.push(ev)

    # Assign orders to workstations.
    # `ordered_orders_by_w` uses indices into `orders` (not order_id),
    for w, elem in ordered_orders_by_w.items():
        state.warehouse.workstations[w].order_buffer = []
        ability_to_open = (
            state.warehouse.workstations[w].order_capacity
            - len(state.warehouse.workstations[w].opened_orders)
        )

        for m in elem:
            o = orders[m]

            if o.order_id not in state.warehouse.workstations[w].opened_orders:
                o.status = OrderStatus.WAITING
                o.workstation_id = w

                if ability_to_open > 0 and o.status != OrderStatus.OPEN:
                    # Workstation has capacity: open the order immediately
                    state.future_events.push(Event(
                        time=state.current_time,
                        type=EventType.OPEN_ORDER,
                        info=o,
                    ))
                    ability_to_open -= 1
                    logging.debug(
                        "Order %i will be opened at workstation %i. [opened orders = %i / %i]",
                        o.order_id, w,
                        len(state.warehouse.workstations[w].opened_orders),
                        state.warehouse.workstations[w].order_capacity,
                    )
                else:
                    # Workstation at capacity: buffer the order
                    state.warehouse.workstations[w].order_buffer.append(o.order_id)
                    logging.debug(
                        "Order %i queued at workstation %i. [order_queue len = %i]",
                        o.order_id, w,
                        len(state.warehouse.workstations[w].order_buffer),
                    )

    # Schedule task release events; offset by priority to stagger execution
    logging.warning("Task designing ended. Optimizer designed %i tasks.", len(tasks))
    for t in tasks:
        state.future_events.push(Event(
            time=state.current_time + t.priority,
            type=EventType.RELEASE_TASK,
            info=t,
        ))
        state.task_counter += 1

    # Enqueue the next optimization run after the configured interval
    state.future_events.push(Event(
        time=state.current_time + sim.config.optimization_interval,
        type=EventType.RUN_OPTIMIZER,
    ))




    


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _count_closed(state) -> int:
    """Count closed orders in the system. O(n) — consider a dedicated counter."""
    return sum(1 for o in state.orders_in_system if o.status == OrderStatus.CLOSED)

def _sample_sku(gen, N):
    """
    Sample a SKU index from a truncated normal distribution over [0, N).
    """
    while True:
        id_s = int(gen.normal(0.5 * N, N/6))
        if 0 <= id_s < N:
            return id_s
        