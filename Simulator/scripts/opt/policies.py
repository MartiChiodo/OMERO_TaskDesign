"""
Scheduling and assignment policies for the warehouse simulator.

This module implements heuristic policies for order-to-workstation assignment,
pod selection for task design, and robot allocation.
"""

import logging
from scripts.core.entities import Task, Visit
from scripts.core.enums import PodStatus, RobotStatus


def assign_order_to_workstation_policy(order, workstations) -> int:
    """
    Assign an order to the workstation with the shortest queue.

    Uses a greedy policy: selects the workstation with the minimum
    combined size of opened orders and pending orders in buffer.
    Ties are broken by workstation ID (lower ID preferred).

    Parameters
    ----------
    order : Order                      Order to assign (used only for logging, not modified).
    workstations : list[Workstation]   Available workstations to choose from.
    """
    
    if not workstations:
        raise ValueError("Cannot assign order: no workstations available")
    
    selected_workstation = min(
        workstations,
        key=lambda ws: len(ws.order_buffer) + len(ws.opened_orders)
    )
    
    return selected_workstation.workstation_id


def design_tasks_for_ws(
    workstation,
    warehouse,
    orders_in_system,
    task_counter,
    active_tasks
) -> list[Task]:
    """
    Design tasks for a workstation using greedy set cover.

    Iteratively selects idle pods that maximize SKU coverage for open
    orders at the workstation (pile-on maximization). Each pod is converted
    to a Task with a single Visit grouping all contributing orders.

    Stops when either:
    1. Workload capacity is reached
    2. All open order SKUs are covered
    3. No more pods can contribute

    Parameters
    ----------
    workstation : Workstation                  Target workstation with opened_orders containing active orders.
    warehouse : Warehouse                      Warehouse instance for pod and distance queries.
    orders_in_system : PriorityQueue[Order]    All orders in the system (for SKU lookup).
    task_counter : int                         Current task ID counter (auto-incremented for new tasks).
    active_tasks : list[Task]                  Tasks currently allocated to robots relatid to this workstation
    """
    
    # Collect all uncovered SKUs across open orders at target workstation
    uncovered_skus = set(
        [item
        for o_id in workstation.opened_orders
        if orders_in_system.get(o_id) is not None
        for item in orders_in_system.get(o_id).items_pending])

    uncovered_skus -= set([
        item
        for id_t in workstation.active_tasks
        for v in active_tasks[id_t].stops
        for item in v.items]
    )

    if not uncovered_skus:
        return [], task_counter
    
    # Track which pods are already selected in active tasks
    already_selected = set([
        visit.pod_id if hasattr(visit, 'pod_id') else id(visit)
        for visit in workstation.active_tasks]
    )
    
    tasks = []
    num_capacity = workstation.released_task_capacity - len(workstation.active_tasks)
    set_task_id_to_be_rewritten = workstation.released_tasks.copy()
    
    # Greedy selection loop
    while len(tasks) < num_capacity and uncovered_skus:
        best_pod = None
        best_pile_on = 0
        best_distance = float('inf')
        
        # Find pod with best pile-on score
        for pod in warehouse.pods:

            if pod.pod_id in already_selected:
                continue
            
            # Compute pile-on: how many uncovered SKUs this pod has
            pile_on = len(set(pod.items) & uncovered_skus)
            if pile_on == 0:
                continue
            
            # Compute distance as tiebreaker
            distance = warehouse.manhattan_distance(
                warehouse.cell2coord(pod.storage_location),
                warehouse.cell2coord(workstation.position)
            )
            
            # Update best pod if this one is better
            if (pile_on > best_pile_on or 
                (pile_on == best_pile_on and distance < best_distance)):
                best_pod = pod
                best_pile_on = pile_on
                best_distance = distance
        
        # If no more pods can contribute, stop
        if best_pod is None:
            break
        
        # Build Visit: group all orders that benefit from this pod
        contributing_order_ids = []
        skus_in_visit = []
        
        for order_id in workstation.opened_orders:
            order = orders_in_system.get(order_id)
            if order is None:
                continue
            
            # Find SKUs from this pod that match order requirements
            order_matching_skus = [
                sku for sku in best_pod.items
                if sku in order.items_pending and sku in uncovered_skus
            ]
            
            if order_matching_skus:
                contributing_order_ids.append(order_id)
                skus_in_visit.extend(order_matching_skus)
        
        # Create task with single visit
        if len(set_task_id_to_be_rewritten) > 0:
            t_id = set_task_id_to_be_rewritten.pop()
        else:
            t_id = task_counter 
            task_counter += 1 

        t = Task(
            task_id=t_id,
            pod_id=best_pod.pod_id,
            robot_id=None,
            stops=[Visit(
                workstation_id=workstation.workstation_id,
                orders=set(contributing_order_ids),
                items=set(skus_in_visit)
            )],
            priority=task_counter + len(tasks),
        )
        
        tasks.append(t)
        already_selected.add(best_pod.pod_id)
        uncovered_skus -= set(best_pod.items)
    
    return tasks, task_counter


def get_nearest_idle_robot(pod, warehouse) -> int | None:
    """
    Find the nearest idle robot to a given pod.

    Searches all robots, selects the IDLE robot with minimum Manhattan
    distance to the pod's location. Ties are broken by robot ID (lower ID
    preferred for determinism).

    Parameters
    ----------
    pod : Pod               Target pod at storage_location.
    warehouse : Warehouse   Warehouse instance for robot and distance queries.
    """
    
    best_robot = None
    best_distance = float('inf')
    
    for robot in warehouse.robots:
        # Only consider idle robots
        if robot.status != RobotStatus.IDLE:
            continue
        
        distance = warehouse.manhattan_distance(
            warehouse.cell2coord(robot.position),
            warehouse.cell2coord(pod.storage_location)
        )
        
        # Update if this robot is closer (or tie-break by ID)
        if (distance < best_distance or 
            (distance == best_distance and 
             (best_robot is None or robot.robot_id < best_robot.robot_id))):
            best_robot = robot
            best_distance = distance
    
    if best_robot is not None:
        return best_robot.robot_id
    else:
        return None