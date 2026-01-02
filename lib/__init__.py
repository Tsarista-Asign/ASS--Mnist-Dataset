"""Thư viện hỗ trợ làm việc với MNIST dataset (OOP)."""

from lib.dataset import MNISTDataLoader
from lib.model import MNISTClassifier
from lib.reduction import (
    DimensionalityReducer,
    ChiSquareReducer,
    create_reducer,
    UnifiedReducer,
    REDUCTION_METHODS,
    get_default_n_components_trials,
)
from lib.utils import (
    plot_samples,
    plot_training_history,
    plot_confusion_matrix,
    print_classification_report,
    measure_array_memory_mb,
    run_and_measure_seconds,
    plot_comparison_reduction,
    print_comparison_table,
)

__all__ = [
    "MNISTDataLoader",
    "MNISTClassifier",
    "DimensionalityReducer",
    "ChiSquareReducer",
    "create_reducer",
    "UnifiedReducer",
    "REDUCTION_METHODS",
    "get_default_n_components_trials",
    "plot_samples",
    "plot_training_history",
    "plot_confusion_matrix",
    "print_classification_report",
    "measure_array_memory_mb",
    "run_and_measure_seconds",
    "plot_comparison_reduction",
    "print_comparison_table",
]
