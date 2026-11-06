"""Optional order layer and order scheduling.

* order layer (``orders:`` scenario key): orders, packages and inventory, dependency-aware release,
  extended task lifecycle with failures, retries and cancellation (``OrderManager``);
* ``OrderScheduler(rule)``: one scheduling pipeline with the rules FIFO, EDD, LWF, WSP (+ WSP_eq
  ablation), SPT, CR, NSP, ZLP_2D and PLB, and a decision log;
* ``TraceRecorder`` / ``evaluate``: trace, five-level metrics, bottleneck attribution, diagnosis;
* ``write_dashboard``: offline HTML replay; ``run_episode`` / ``evaluation``: paired experiments.

Without the ``orders:`` key the simulator runs exactly as before."""
from battery_tarware.orders.dashboard import write_dashboard
from battery_tarware.orders.inventory import Inventory, InventoryError, Package, RackLocation, Shelf
from battery_tarware.orders.lifecycle import TRANSITIONS, InvalidOTransition, OTaskState
from battery_tarware.orders.manager import FailureRecord, Order, OrderManager, OrderTask, TaskGroup
from battery_tarware.orders.metrics import evaluate
from battery_tarware.orders.monitor import ProgressMonitor, StallReport
from battery_tarware.orders.run import run_episode
from battery_tarware.orders.scenarios import order_scenario
from battery_tarware.orders.scheduler import RULE_NAMES, OrderScheduler
from battery_tarware.orders.spec import OrderSpec
from battery_tarware.orders.trace import TraceRecorder

__all__ = ["OrderManager", "Order", "OrderTask", "TaskGroup", "FailureRecord", "OTaskState", "TRANSITIONS",
           "InvalidOTransition", "Inventory", "InventoryError", "Package", "Shelf", "RackLocation",
           "ProgressMonitor", "StallReport", "OrderSpec", "OrderScheduler", "RULE_NAMES", "TraceRecorder",
           "evaluate", "write_dashboard", "run_episode", "order_scenario"]
