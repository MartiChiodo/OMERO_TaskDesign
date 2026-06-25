from enum import Enum, auto
from typing import Any
from dataclasses import dataclass, field
from scripts.core.enums import OrderStatus, RobotStatus, PodStatus, WorkstationPickingStatus, EventType


### MISSIONS-RELATED CLASSES

@dataclass
class Visit:
    """
    Represents a single stop within a task.

    A pod must travel to a specific workstation and pick a set of items
    associated with a given order.

    Parameters
    ----------
    workstation_id : int   Identifier of the destination workstation.
    orders : set[int]      Order IDs being served at this stop.
    items : set[int]       SKU IDs to be picked at this stop.
    """
    workstation_id: int
    orders: set[int]
    items: set[int]


@dataclass
class Task:
    """
    Represents a full mission assigned to a pod-robot pair.

    A task consists of an ordered sequence of stops that define
    the route and picking operations for a pod.

    Parameters
    ----------
    task_id : int           Unique identifier of the task.
    pod_id : int            Identifier of the pod executing the task.
    robot_id : int | None   Identifier of the robot allocated to the task.
    stops : list[Visit]     Ordered list of stops to be executed. Stops are popped during END_PICKING.
    priority : float        Scheduling priority of the task (lower = higher priority).
    """
    task_id: int
    pod_id: int
    robot_id: int | None
    stops: list[Visit]
    priority: float


### OUTER SIMULATOR RELATED CLASSES
# Persistent across the entire simulation.
# Updated by both optimizer and emulator.

@dataclass
class Order:
    """
    Represents a customer order tracked by the outer simulator.

    Parameters
    ----------
    order_id : int                Unique identifier of the order.
    arrival_time : float          Simulation time at which the order arrived.
    order_size : int              Total number of items required by the order.
    items_required : set[int]     Set of SKUs required to complete the order (immutable).
    items_pending : set[int]      SKUs not yet picked (decremented during execution).
    workstation_id : int | None   Workstation assigned by the optimizer (None if not yet assigned).
    status : OrderStatus          Current lifecycle status of the order.
    """
    order_id: int
    arrival_time: float
    order_size: int
    items_required: set[int]
    items_pending: set[int]
    workstation_id: int | None
    status: OrderStatus = OrderStatus.BACKLOG


### EMULATOR RELATED CLASSES
# Tracks detailed execution state.

@dataclass
class Robot:
    """
    Represents the fine-grained state of a robot.

    Parameters
    ----------
    robot_id : int               Unique identifier of the robot.
    position : int               Identifier of the cell of the Current grid position of the robot.
    status : RobotStatus         Current operational status of the robot.
    """
    robot_id: int
    position: int
    status: RobotStatus = RobotStatus.IDLE


@dataclass
class Pod:
    """
    Represents the fine-grained state of a pod.

    Parameters
    ----------
    pod_id : int                          Unique identifier of the pod.
    storage_location : int                Default grid cell where the pod rests when idle.
    items : set[int]                      SKU IDs currently stored in the pod.
    status : PodStatus                    Current operational status of the pod.
    """
    pod_id: int
    storage_location: int
    items: set[int] = field(default_factory=set)
    status: PodStatus = PodStatus.IDLE


@dataclass
class Workstation:
    """
    Represents the fine-grained state of a workstation.

    A workstation is a picking station on the warehouse perimeter where orders
    are processed. It manages both the orders being processed and the tasks
    (pod assignments) required to fulfill them.

    Attributes
    ----------
    workstation_id : int            Unique identifier of the workstation.
    position : int                  Cell id of the workstation on the warehouse perimeter.

    order_capacity : int            Maximum number of simultaneously open (OPEN status) orders.
    opened_orders : set[int]        Order IDs currently in OPEN status.

    released_task_capacity : int    Maximum number of tasks simultaneously in RELEASED status (waiting for robot).
                                    Once a task is assigned to a robot (START_TASK), it no longer counts toward
                                    this limit.
    active_tasks : set[int]         Task IDs currently allocated to robots (in progress at this workstation).
    released_tasks: set[int]        Task IDs not yet allocated to robots.
    

    pod_process_time : float        Time in minutes to process (prepare) a pod at this workstation.
    item_process_time : float       Time in minutes to pick a single item from a pod.

    status : WorkstationPickingStatus   Whether the workstation is IDLE or currently BUSY picking items.
    order_buffer : list[int]            Order IDs queued to be opened when a slot becomes available.
    picking_buffer : dict[int, float]   Task ID queued here when workstation is busy (BUSY status) with their arrival time.


    Notes
    -----
    Task lifecycle at workstation:
    1. Task released: counted in counter_released_task and added to sim.released_tasks
    2. Task assigned to robot: moved to active_tasks
    3. Task completes picking: removed from active_tasks and counter is decremented
    4. Task has more stops: sent to next workstation
    5. Task done: robot returns pod to storage
    """
    workstation_id: int
    order_capacity: int
    released_task_capacity: int
    position: int
    pod_process_time: float
    item_process_time: float

    opened_orders: set[int] = field(default_factory=set)
    order_buffer: list[int] = field(default_factory=list)
    picking_buffer: dict[int, float] = field(default_factory=dict)
    active_tasks: set[int] = field(default_factory=set)
    released_tasks: set[int] = field(default_factory=set)
    status: WorkstationPickingStatus = WorkstationPickingStatus.IDLE


    ### UTILITY METHODS

    def has_open_slot(self) -> bool:
        """
        Check if the workstation can accept a new order.
        """
        return len(self.opened_orders) < self.order_capacity

    def can_release_task(self) -> bool:
        """
        Check if the workstation can release another task.

        A task can be released if the total number of pending + active tasks
        is below the released_task_capacity threshold.
        """
        return len(self.released_tasks) + len(self.active_tasks) < self.released_task_capacity

    def estimated_picking_time(self, num_items: int) -> float:
        """
        Estimate the time required to pick items from a pod.

        Computed as: pod_process_time + num_items x item_process_time
        """
        return self.pod_process_time + num_items * self.item_process_time


### EVENT CONTAINER

@dataclass
class Event:
    """
    Discrete-event simulation (DES) event.

    Parameters
    ----------
    time : float           Simulation time at which the event is scheduled.
    type : EventType       Type of event (determines handler function).
    info : Any, optional   Event-specific payload carrying additional data (e.g., Order, Task).
    """ 
    time: float
    type: EventType
    info: Any = None