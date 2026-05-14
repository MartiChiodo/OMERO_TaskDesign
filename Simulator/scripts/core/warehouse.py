from __future__ import annotations
from numpy.random import Generator
from collections import defaultdict
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import math

from Simulator.scripts.core.enums import PodStatus, WorkstationPickingStatus, RobotStatus
from Simulator.scripts.core.entities import Pod, Workstation, Robot


# Layout constants
MARGIN = 3
MIN_SPACING = 2


class Warehouse:
    """
    Physical warehouse representation with optimized lookups.

    Attributes
    ----------
    grid_rows, grid_cols : int         Specifify the number of rows/cols which make the pod grid
    X, Y : int                         Total dimensions of the warehouse (includes roads and margins)
    num_skus : int                     Number of unique skus in the warehouse
    
    robot_speed : float                Speed at which robot moves (in cell per minute)

    pods : list[Pod]                              List of all the pods of the warehouse
    pods_by_id : dict[int, Pod]                   Fast O(1) lookup of pods by ID.
    robots : list [Robot]                         List of all the robots of the warehouse
    robots_by_id : dict[int, Robot]               Fast O(1) lookup of robots by ID.
    workstations : list[Workstation]              List of all the workstations of the warehouse
    workstations_by_id : dict[int, Workstation]   Fast O(1) lookup of workstations by ID.
 
    pods_by_sku : dict[int, list[int]]            Reverse index: SKU → list of pod IDs containing that SKU.
                                                  Useful for optimize policies that search pods by SKU requirements.
    """

    def __init__(
        self,
        random_generator: Generator,
        num_pods: int,
        num_skus: int,
        num_robots: int,
        num_workstations: int,
        num_skus_per_pod: int,
        grid_rows: int,
        grid_cols: int,
        ws_order_capacity: int,
        ws_released_task_capacity: int,
        robot_speed: float = 30.0,
        pod_process_time: float = 5/60,
        item_process_time: float = 5/60,
    ) -> None:
        """
        Initialize warehouse.
        """

        # Validation
        if num_pods != grid_rows * grid_cols:
            raise ValueError(
                f"Warehouse layout mismatch: num_pods ({num_pods}) must equal "
                f"grid_rows x grid_cols ({grid_rows} x {grid_cols})"
            )

        if num_pods <= 0 or num_robots <= 0 or num_workstations <= 0:
            raise ValueError("All entity counts must be > 0")

        if robot_speed <= 0 or ws_order_capacity <= 0 or ws_released_task_capacity <= 0:
            raise ValueError("All capacities and speeds must be > 0")

        # Store configuration
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.robot_speed = robot_speed
        self.num_skus = num_skus

        # Compute physical dimensions
        self.X = grid_rows + 2 * ((grid_rows - 1) // 2) + 2 * MARGIN - 1
        self.Y = grid_cols + 2 * MARGIN - 1

        logging.info(
            "Warehouse initialization | grid=%dx%d, physical=%dx%d, "
            "num_skus=%d, num_robots=%d, num_ws=%d",
            grid_rows, grid_cols, self.X, self.Y, num_skus, num_robots, num_workstations
        )

        # Generate entities
        self.pods = self._generate_pods(
            random_generator, num_pods, num_skus, grid_rows, grid_cols, num_skus_per_pod
        )

        self.workstations = self._generate_workstations(
            num_workstations,
            ws_order_capacity,
            ws_released_task_capacity,
            pod_process_time,
            item_process_time
        )
        
        self.robots = self._generate_robots(random_generator, num_robots)

        # Build fast lookup indices
        self._build_indices()

        logging.info(
            "Warehouse initialized | %d pods  (average number of skus per pod = %3.2f), %d workstations, %d robots",
            len(self.pods), sum([len(p.items) for p in self.pods])/len(self.pods), len(self.workstations), len(self.robots)
        )

    
    ### ENTITY GENERATION

    def _generate_pods(
        self,
        random_generator: Generator,
        num_pods: int,
        num_skus: int,
        grid_rows: int,
        grid_cols: int,
        num_skus_per_pod: int
    ) -> list[Pod]:
        """
        Generate pods.

        Uses truncated-normal distribution for more realistic SKU distribution:
        - Some SKUs appear in many pods (popular items)
        - Other SKUs appear in few pods (niche items)
        """

        # Pre-allocate list
        pods = [None] * num_pods

        # Assign SKUs to pods
        for col in range(grid_cols):
            for row in range(grid_rows):
                pod_id = col * grid_rows + row

                # Compute grid position
                x_position = MARGIN + row + 2 * (row // 2)
                y_position = self.Y - MARGIN - col

                # Sample SKUs for this pod 
                samples = random_generator.normal(loc=num_skus/2, scale=0.8*num_skus, size=num_skus_per_pod)
                samples = np.round(samples).astype(int)
                pod_skus = samples[(samples >= 0) & (samples <= num_skus-1)]

                pods[pod_id] = Pod(
                    pod_id=pod_id,
                    storage_location=self.coord2cell(x_position, y_position),
                    items=set(pod_skus),
                    status=PodStatus.IDLE
                )

        # Verify all SKUs appear at least once (coverage check)
        all_skus = set()
        for pod in pods:
            all_skus.update(pod.items)

        missing_skus = set(range(num_skus)) - all_skus
        for sku_id in missing_skus:
            pod = min(pods, key= lambda p: len(p.items))
            pods[pod.pod_id].items.add(sku_id)

        all_skus = set()
        for pod in pods:
            all_skus.update(pod.items)
        assert len(all_skus) == num_skus, f"Only {len(all_skus)} skus have been assigned to at least one pod."

        return pods

    def _generate_workstations(
        self,
        num_workstations: int,
        ws_order_capacity: int,
        ws_released_task_capacity: int,
        pod_process_time: float,
        item_process_time: float
    ) -> list[Workstation]:
        """
        Generate workstations.
        """

        workstations = [None] * num_workstations

        # Check if symmetric bottom placement is possible
        max_bottom_slots = (self.X - 2) // MIN_SPACING + 1

        if num_workstations <= max_bottom_slots:
            # Symmetric placement on bottom edge
            center_x = self.X // 2
            start_offset = -(num_workstations // 2) * MIN_SPACING

            for ws_id in range(num_workstations):
                x_position = center_x + start_offset + ws_id * MIN_SPACING
                workstations[ws_id] = Workstation(
                    workstation_id=ws_id,
                    order_capacity=ws_order_capacity,
                    released_task_capacity=ws_released_task_capacity,
                    position= self.coord2cell(x_position, 0),
                    pod_process_time=pod_process_time,
                    item_process_time=item_process_time
                )

            return workstations

        # Perimeter placement (anti-clockwise)
        x_position, y_position = self.X // 2, 0

        for ws_id in range(num_workstations):
            workstations[ws_id] = Workstation(
                workstation_id=ws_id,
                order_capacity=ws_order_capacity,
                released_task_capacity=ws_released_task_capacity,
                position= self.coord2cell(x_position, y_position),
                pod_process_time=pod_process_time,
                item_process_time=item_process_time
            )

            # Advance anti-clockwise
            if y_position == 0:
                x_position = x_position + MIN_SPACING if x_position + MIN_SPACING < self.X else self.X
                y_position = 0 if x_position != self.X else MIN_SPACING
            elif x_position == self.X:
                y_position = y_position + MIN_SPACING if y_position + MIN_SPACING < self.Y else self.Y
                x_position = self.X if y_position != self.Y else (self.X - MIN_SPACING)
            elif y_position == self.Y:
                x_position = x_position - MIN_SPACING if x_position - MIN_SPACING > 0 else 0
                y_position = self.Y if x_position != 0 else (self.Y - MIN_SPACING)
            elif x_position == 0:
                y_position = y_position - MIN_SPACING if y_position - MIN_SPACING > 0 else 0
                x_position = 0 if y_position != 0 else MIN_SPACING

        return workstations

    def _generate_robots(
        self,
        random_generator: Generator,
        num_robots: int
    ) -> list[Robot]:
        """
        Generate robots with collision avoidance (same as before).
        """

        robots = [None] * num_robots
        assigned_positions = set()

        for robot_id in range(num_robots):
            while True:
                x_position = random_generator.integers(1, self.X - 1)
                y_position = random_generator.integers(1, self.Y - 1)
                if (x_position, y_position) not in assigned_positions:
                    break

            assigned_positions.add((x_position, y_position))

            robots[robot_id] = Robot(
                robot_id=robot_id,
                position=self.coord2cell(x_position, y_position),
                status=RobotStatus.IDLE
            )

        return robots


    ### BUILD INDICES

    def _build_indices(self) -> None:
        """
        Build fast-lookup indices.

        Creates O(1) lookup dictionaries instead of O(n) linear search.
        Also builds SKU reverse index for optimized task design.
        """

        # O(1) entity lookups
        self.pods_by_id = {pod.pod_id: pod for pod in self.pods}
        self.robots_by_id = {robot.robot_id: robot for robot in self.robots}
        self.workstations_by_id = {ws.workstation_id: ws for ws in self.workstations}

        # SKU reverse index: SKU → list of pod IDs
        self.pods_by_sku: dict[int, list[int]] = defaultdict(list)
        for pod in self.pods:
            for sku in pod.items:
                self.pods_by_sku[sku].append(pod.pod_id)



    ### DISTANCE AND TRAVEL TIME

    def coord2cell(self, position_x: int, position_y: int) -> int:
        return position_x + self.X * position_y
    
    def cell2coord(self, cell_id: int) -> int:
        return (cell_id % self.X, math.floor(cell_id/self.X))

    @staticmethod
    def manhattan_distance(
        position_a: int,
        position_b: int,
    ) -> int:
        """Compute Manhattan distance between two positions."""
        return abs(position_a[0] - position_b[0]) + abs(position_a[1] - position_b[1])

    def travel_time(
        self,
        position_a,
        position_b,
        random_generator: Generator | None = None,
    ) -> float:
        """
        Estimate travel time between two positions.

        Computed as Manhattan distance divided by robot speed.
        Optional noise for realism.
        """

        nominal_time = self.manhattan_distance(position_a, position_b) / self.robot_speed

        if random_generator is not None:
            # Noise is drawn from a Beta(2,10) with support [0, 0.5*nominal_time]
            # Mean is 2/(2+10)*0.5*nominal_time = 0.0833*nominal_time
            noise = random_generator.beta(a=2, b = 10)*0.5*nominal_time
            return nominal_time + noise

        return nominal_time


    ### FAST ENTITY LOOKUP 

    def get_pod(self, pod_id: int) -> Pod:
        """
        Retrieve a pod by ID - O(1) with index.
        """
        pod = self.pods_by_id.get(pod_id)
        if pod is None:
            raise KeyError(f"Pod {pod_id} not found")
        return pod

    def get_workstation(self, workstation_id: int) -> Workstation:
        """
        Retrieve a workstation by ID - O(1) with index.
        """
        workstation = self.workstations_by_id.get(workstation_id)
        if workstation is None:
            raise KeyError(f"Workstation {workstation_id} not found")
        return workstation

    def get_robot(self, robot_id: int) -> Robot:
        """
        Retrieve a robot by ID - O(1) with index.
        """
        robot = self.robots_by_id.get(robot_id)
        if robot is None:
            raise KeyError(f"Robot {robot_id} not found")
        return robot

    def get_pods_containing_sku(self, sku_id: int) -> list[Pod]:
        """
        Get all pods containing a specific SKU - O(k) where k = pods with SKU.
        """
        pod_ids = self.pods_by_sku.get(sku_id, [])
        return [self.pods_by_id[pod_id] for pod_id in pod_ids]
    

    ### VISUALIZATION

    def plot(
        self,
        save: bool = True,
        folder: str = r"Simulator\output\plots",
    ) -> None:
        """Plot warehouse layout."""

        scale = 0.8
        fig, ax = plt.subplots(figsize=(self.X * scale, self.Y * scale))
        ax.set_aspect('equal')

        for pod in self.pods:
            x, y = self.cell2coord(pod.storage_location)
            ax.add_patch(plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                       fill=False, color='black', linewidth=0.5))
            ax.text(x, y, str(pod.pod_id), ha='center', va='center',
                   fontsize=6, color='black')

        for workstation in self.workstations:
            x, y = self.cell2coord(workstation.position)
            ax.add_patch(plt.Circle((x, y), 0.5, fill=False, color='red', linewidth=1))
            ax.text(x, y, str(workstation.workstation_id), ha='center', va='center',
                   fontsize=8, color='red', fontweight='bold')

        for robot in self.robots:
            x, y = self.cell2coord(robot.position)
            ax.add_patch(plt.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5,
                                       fill=False, color='blue', linewidth=0.5))
            ax.text(x, y, str(robot.robot_id), ha='center', va='center',
                   fontsize=6, color='blue')

        ax.add_patch(plt.Rectangle((0, 0), self.X, self.Y, fill=False,
                                  edgecolor='red', linewidth=2.5))

        ax.set_xlim(-2, self.X + 2)
        ax.set_ylim(-2, self.Y + 2)
        ax.set_xticks(range(0, self.X + 3))
        ax.set_yticks(range(0, self.Y + 3))
        ax.grid(True, alpha=0.3)
        plt.title("Warehouse Layout", fontsize=14, fontweight='bold')

        if save:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, "warehouse_layout.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Warehouse layout saved to {filepath}")
        else:
            plt.show()

    def __repr__(self) -> str:
        return (
            f"Warehouse("
            f"grid={self.grid_rows}x{self.grid_cols}, "
            f"physical={self.X}x{self.Y}, "
            f"pods={len(self.pods)}, "
            f"ws={len(self.workstations)}, "
            f"robots={len(self.robots)})"
        )