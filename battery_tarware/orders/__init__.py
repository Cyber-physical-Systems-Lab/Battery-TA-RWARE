"""Optional order layer: orders, packages and inventory, dependency-aware task release, an extended
task lifecycle with failures and retries, and order-level metrics. Enabled by the ``orders:``
scenario key; without it the simulator runs exactly as before."""
from battery_tarware.orders.inventory import Inventory, InventoryError, Package, RackLocation, Shelf
from battery_tarware.orders.lifecycle import TRANSITIONS, InvalidOTransition, OTaskState
from battery_tarware.orders.manager import FailureRecord, Order, OrderManager, OrderTask, TaskGroup
from battery_tarware.orders.monitor import ProgressMonitor, StallReport
from battery_tarware.orders.spec import OrderSpec

__all__ = ["OrderManager", "Order", "OrderTask", "TaskGroup", "FailureRecord", "OTaskState", "TRANSITIONS",
           "InvalidOTransition", "Inventory", "InventoryError", "Package", "Shelf", "RackLocation",
           "ProgressMonitor", "StallReport", "OrderSpec"]
