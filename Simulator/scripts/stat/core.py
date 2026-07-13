from enum import Enum
import numpy as np

# Stat type enum 

class StatType(str, Enum):
    ORDER_FLOW_TIME   = "OFT"
    WS_UTILIZATION    = "WS_FREQ"
    ROBOT_UTILIZATION = "RB_FREQ"
    WS_AVG_OPEN_ORDER = "WS_AVG_OO"
    POD_AVG_MOVING = "POD_AVG_MOVING"


# Sub-trackers 

class ResourceTracker:
    """
    Time-weighted utilization tracker for a pool of homogeneous resources
    (workstations or robots).

    Accumulates time spent in each discrete state using a last-event clock.
    State indices must match the .value of the corresponding enum.
    """

    def __init__(self, n: int, n_states: int, default_idle_state: int) -> None:
        self.usage       = np.zeros((n, n_states))   # accumulated time per state
        self.last_clock  = np.full(n, -1.0)
        self.last_state  = np.full(n, -1, dtype=int)
        self.default    = default_idle_state

    def record(self, resource_id: int, new_state: int, clock: float) -> None:
        """
        Register a state transition for *resource_id* at *clock*.

        On first call after warm-up, initialises the baseline without
        accumulating spurious idle time.
        """
        prev_clock = self.last_clock[resource_id]
        prev_state = self.last_state[resource_id]

        if prev_clock >= 0.0 and prev_state >= 0:
            self.usage[resource_id, prev_state] += clock - prev_clock
        else:
            # First post-warm-up event: seed state, no time to accumulate
            if prev_state < 0:
                self.last_state[resource_id] = self.default

        self.last_clock[resource_id] = clock
        self.last_state[resource_id] = new_state

    def seed_state(self, resource_id: int, state: int) -> None:
        """Record a pre-warm-up state without accumulating time."""
        self.last_state[resource_id] = state

    def utilization(self) -> np.ndarray:
        """Return per-resource utilization ratio (busy / total). Shape: (n,)."""
        total = self.usage.sum(axis=1)
        busy  = self.usage[:, 1]
        util  = np.zeros_like(busy)
        mask  = total > 0
        util[mask] = busy[mask] / total[mask]
        return util

    def reset(self) -> None:
        n = self.usage.shape[0]
        self.usage       = np.zeros_like(self.usage)
        self.last_clock = np.full(n, -1.0)
        self.last_state = np.full(n, -1, dtype=int)


class OrderFlowTracker:
    """
    Tracks completed-order counts and cumulative flow times, broken down
    by order size.
    """

    def __init__(self) -> None:
        self.count:    dict[int, int]   = {}
        self.cum_time: dict[int, float] = {}

    def record(self, order_size: int, flow_time: float) -> None:
        """Register a completed order of *order_size* with *flow_time*."""
        if order_size in self.count:
            self.count[order_size]    += 1
            self.cum_time[order_size] += flow_time
        else:
            self.count[order_size]    = 1
            self.cum_time[order_size] = flow_time

    def mean_flow_time(self, order_size: int) -> float:
        """Return mean flow time for *order_size*, or 0.0 if no data."""
        n = self.count.get(order_size, 0)
        return self.cum_time.get(order_size, 0.0) / n if n > 0 else 0.0

    def global_mean_flow_time(self) -> float:
        total_n   = sum(self.count.values())
        total_cum = sum(self.cum_time.values())
        return total_cum / total_n if total_n > 0 else 0.0

    def reset(self) -> None:
        self.count.clear()
        self.cum_time.clear()


class TimeWeightedMeanTracker:
    """
    Tracks the time-weighted mean of an integer-valued signal for each of
    *n* resources (e.g. number of open orders per workstation).

    Uses area accumulation: integral of signal over time, divided by
    elapsed time, gives the time-average.
    """

    def __init__(self, n: int, warm_up: float) -> None:
        self.n        = n
        self.warm_up  = warm_up
        self.area     = np.zeros(n)
        self.last_val = np.zeros(n, dtype=int)
        self.last_clk = np.full(n, float(warm_up)) 

    def record(self, resource_id: int, new_value: int, clock: float) -> None:
        """Register a value change for *resource_id* at *clock*."""
        elapsed = clock - self.last_clk[resource_id]
        if elapsed > 0:
            self.area[resource_id] += elapsed * self.last_val[resource_id]
        self.last_clk[resource_id] = clock
        self.last_val[resource_id] = new_value

    def mean(self, resource_id: int, end_clock: float) -> float:
        """
        Return the time-weighted mean for *resource_id* up to *end_clock*.
        Flushes the open interval without modifying internal state.
        """
        elapsed_total = end_clock - self.warm_up
        if elapsed_total <= 0:
            return 0.0
        pending = (end_clock - self.last_clk[resource_id]) * self.last_val[resource_id] if (end_clock - self.last_clk[resource_id])>= 0 else 0
        return (self.area[resource_id] + pending) / elapsed_total

    def reset(self) -> None:
        self.area     = np.zeros(self.n)
        self.last_val = np.zeros(self.n, dtype=int)
        self.last_clk = np.full(self.n, self.warm_up)

