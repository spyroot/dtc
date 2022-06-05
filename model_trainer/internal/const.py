from enum import Enum


class ReduceMode(Enum):
    MIN = "min"
    MAX = "max"


class MetricType(Enum):
    MEAN = "mean"
    SUM = "sum"
