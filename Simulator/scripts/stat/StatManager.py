import logging
import os

from Simulator.scripts.core.enums import WorkstationPickingStatus, RobotStatus, OrderStatus
from Simulator.scripts.stat.core import *


#  Main coordinator 

class StatManager:
    """
    Collects and manages performance statistics for the warehouse simulation.

    Delegates measurement to specialised sub-trackers:
    - ``ResourceTracker``       workstation and robot utilization
    - ``OrderFlowTracker``      per-size order flow times
    - ``TimeWeightedMeanTracker``  average open orders per workstation

    The public ``update_statistic`` interface is preserved for compatibility
    with the event handlers. Internally, the *type* argument is coerced to
    ``StatType``.

    Parameters
    ----------
    warehouse : Warehouse        Warehouse instance providing resource counts.
    warm_up : float, optional    Simulation time before which statistics are not collected.
    """

    def __init__(self, warehouse, warm_up: float = 0.0) -> None:
        
        self.WARM_UP = warm_up

        n_ws = len(warehouse.workstations)
        n_rb = len(warehouse.robots)

        self.ws_tracker  = ResourceTracker(n_ws, 2, WorkstationPickingStatus.IDLE.value)
        self.rb_tracker  = ResourceTracker(n_rb, 2, RobotStatus.IDLE.value)
        self.oft_tracker = OrderFlowTracker()
        self.oo_tracker  = TimeWeightedMeanTracker(n_ws, warm_up)

        # Time spent in computation for decision-making
        self.decisions_computing_time = 0

        # Pod Moved and throughput
        self.avg_number_pod_moving = TimeWeightedMeanTracker(1, warm_up)
        self.throughput = 0

     

    def update_statistic(self, type: str, info: list) -> None:
        """
        Dispatch a statistic update to the appropriate sub-tracker.

        Parameters
        ----------
        type : str | StatType
            One of "OFT", "WS_FREQ", "RB_FREQ", "WS_AVG_OO".
        info : list
            Payload whose structure depends on *type*:
            - OFT        : [order, completion_time]
            - WS_FREQ    : [ws_id, WorkstationPickingStatus, clock]
            - RB_FREQ    : [robot_id, RobotStatus, clock]
            - WS_AVG_OO  : [ws_id, new_open_order_count, clock]
            - POD_AVG_MOVING : [variation, clock]
        """
        stat = StatType(type)

        match stat:

            case StatType.ORDER_FLOW_TIME:
                order, completion_time = info[0], info[1]
                if completion_time < self.WARM_UP:
                    return
                flow_time = completion_time - order.arrival_time
                self.oft_tracker.record(order.order_size, flow_time)

            case StatType.WS_UTILIZATION:
                ws_id, new_state, clock = info[0], info[1], info[2]
                if clock < self.WARM_UP:
                    self.ws_tracker.seed_state(ws_id, new_state.value)
                else:
                    self.ws_tracker.record(ws_id, new_state.value, clock)

            case StatType.ROBOT_UTILIZATION:
                rb_id, new_state, clock = info[0], info[1], info[2]
                if clock < self.WARM_UP:
                    self.rb_tracker.seed_state(rb_id, new_state.value)
                else:
                    self.rb_tracker.record(rb_id, new_state.value, clock)

            case StatType.WS_AVG_OPEN_ORDER:
                ws_id, variation, clock = info[0], info[1], info[2]
                new_value = self.oo_tracker.last_val[ws_id] + variation
                effective_clock = max(clock, self.WARM_UP)       
                self.oo_tracker.record(ws_id, new_value, effective_clock)

            case StatType.POD_AVG_MOVING:
                variation, clock = info[0], info[1]
                new_value = self.avg_number_pod_moving.last_val[0] + variation
                effective_clock = max(clock, self.WARM_UP)
                self.avg_number_pod_moving.record(0, new_value, effective_clock)


    #  Report 

    def return_statistics(self, sim_config, state, output_path: str) -> None:
        """Compute, print, and save a summary report."""

        end_time = state.current_time

        report = self.build_report(sim_config, end_time, state)
        print(report)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logging.info("Report saved to: %s", output_path)

    def build_report(self, config, end_time, state) -> str:
        """Return the full report as a string."""
        lines: list[str] = []
        lines.append(f"Simulation with time-horizon = {config.time_horizon} sec and warm-up = {config.warm_up} sec.")
        lines.append(f"Optimization enabled = {config.optimization_enabled}")

        lines.append(f"\nTotal number of items picked (throughtput) = {self.throughput}.\nAverage number of pod moving simultaneously = {self.avg_number_pod_moving.mean(0, end_time)}.")
        lines.append(f"Computational time spent for making decisions = {self.decisions_computing_time} sec.")

        lines += self.format_closed_orders_table()
        lines.append(f"\nIn the system there are still {len([o for o in state.orders_in_system if o.status == OrderStatus.BACKLOG])} order(s) in backlog, {len([o for o in state.orders_in_system if o.status == OrderStatus.WAITING])} order(s) enqueued at a workstation and {len([o for o in state.orders_in_system if o.status == OrderStatus.OPEN])} order(s) open at a workstation.")
        if len([o for o in state.orders_in_system if o.status == OrderStatus.BACKLOG]) > 0:
            lines+=self.format_backlog_orders_table(state.orders_in_system, end_time)
        

        lines += self.format_resource_table("ROBOTS", self.rb_tracker, with_avg_oo=False, time_horizon=end_time)
        lines += self.format_resource_table("WORKSTATIONS", self.ws_tracker, with_avg_oo=True, time_horizon=end_time)
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def format_resource_table(
        self, name: str, tracker: ResourceTracker, with_avg_oo: bool, time_horizon : float
    ) -> list[str]:
        lines = []
        lines.append(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")

        util = tracker.utilization()
        usage = tracker.usage

        if with_avg_oo:
            lines.append(f"  {'ID':<6} {'Idle':>10} {'Busy':>10} {'Util':>8} {'Avg OO':>10}")
        else:
            lines.append(f"  {'ID':<6} {'Idle':>10} {'Busy':>10} {'Util':>8}")

        lines.append("-" * 60)

        for i in range(len(usage)):
            if with_avg_oo:
                # Use a placeholder end_clock that matches last recorded clock
                avg_oo = self.oo_tracker.mean(i, time_horizon)
                lines.append(
                    f"  {i:<6} {usage[i,0]:>10.2f} {usage[i,1]:>10.2f} "
                    f"{util[i]:>7.1%} {avg_oo:>10.2f}"
                )
            else:
                lines.append(
                    f"  {i:<6} {usage[i,0]:>10.2f} {usage[i,1]:>10.2f} {util[i]:>7.1%}"
                )

        lines.append("-" * 60)

        total_time = usage.sum()
        total_busy = usage[:, 1].sum()
        global_util = total_busy / total_time if total_time > 0 else 0.0

        lines.append(f"  {'Mean':<6} {usage[:,0].mean():>10.2f} {usage[:,1].mean():>10.2f} {util.mean():>7.1%}")
        lines.append(f"  {'Std':<5}  {usage[:,0].std():>10.2f} {usage[:,1].std():>10.2f} {util.std():>7.1%}")
        lines.append(f"  {'Global':<6} {'':>10} {'':>10} {global_util:>7.1%}")

        return lines

    def format_closed_orders_table(self) -> list[str]:
        lines = []
        lines.append(f"\n{'=' * 60}\n  ORDERS BY SIZE\n{'=' * 60}")
        lines.append(f"  {'Size':<8} {'Closed':>8} {'Avg Flow (sec)':>17}")
        lines.append("-" * 60)

        tot_closed = 0
        sizes = self.oft_tracker.count.keys()
        for size in sizes:
            n   = self.oft_tracker.count[size]
            tot_closed += n
            avg = self.oft_tracker.mean_flow_time(size)
            lines.append(f"  {size:<8} {n:>8} {avg:>12.2f}")

        lines.append("-" * 60)
        lines.append(
            f"  {'Total':<8} {tot_closed:>8} "
            f"{self.oft_tracker.global_mean_flow_time():>12.2f}"
        )
        return lines
    
    def format_backlog_orders_table(self,orders_in_system, end_time) -> list[str]:
        lines = []
        lines.append(f"\n{'=' * 60}\n  BACKLOG ORDERS \n{'=' * 60}")
        lines.append(f"  {'Size':<8} {'Number':>8} {'Avg waiting time (sec)':>25}")
        lines.append("-" * 60)

        tot_backlog = 0
        backlog = {}
        for o in orders_in_system:
            if o.status == OrderStatus.BACKLOG:
                if o.order_size in backlog.keys():
                    backlog[o.order_size][0] += 1
                    backlog[o.order_size][1] += end_time - o.arrival_time
                else:
                    backlog[o.order_size] = [1, end_time - o.arrival_time]

        sizes = backlog.keys()
        for size in sizes:
            n   = backlog[size][0]
            tot_backlog += n
            avg = backlog[size][1]/n
            lines.append(f"  {size:<8} {n:>8} {avg:>15.2f}")

        lines.append("-" * 60)
        return lines

    #  Reset 

    def reset_statistics(self) -> None:
        """
        Reset all collected statistics.

        Safe to call between replications. Resets all sub-trackers,
        including WS_AVG_OO which was previously omitted.
        """
        self.ws_tracker.reset()
        self.rb_tracker.reset()
        self.oft_tracker.reset()
        self.oo_tracker.reset()
        self.decisions_computing_time = 0
        self.avg_number_pod_moving.reset()
        self.throughput = 0