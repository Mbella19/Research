"""Live paper-trading bridge (D-032).

Mac-side brain: the exact frozen inference path (features → booster+qmap →
EV gate → engine-order guards) driven by fresh bars from a thin Wine-side
MT5 executor, communicating via atomic files. All trading intelligence
lives here; the executor only exports bars/status and executes orders.
"""
