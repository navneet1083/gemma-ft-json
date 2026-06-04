"""Shared utilities: device selection, logging, seeding, and tensor guards."""
from .device import get_device, describe_device
from .seed import set_seed
from .logging_utils import setup_logging, JsonlMetricLogger
from .checks import check_shape, assert_finite, safe_softmax_dim

__all__ = [
    "get_device", "describe_device", "set_seed",
    "setup_logging", "JsonlMetricLogger",
    "check_shape", "assert_finite", "safe_softmax_dim",
]
