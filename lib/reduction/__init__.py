"""Package giảm chiều: PCA, Chi-Square và interface thống nhất."""

from lib.reduction.pca import DimensionalityReducer
from lib.reduction.chi2 import ChiSquareReducer
from lib.reduction.reducer import (
    create_reducer,
    UnifiedReducer,
    REDUCTION_METHODS,
    get_default_n_components_trials,
)

__all__ = [
    "DimensionalityReducer",
    "ChiSquareReducer",
    "create_reducer",
    "UnifiedReducer",
    "REDUCTION_METHODS",
    "get_default_n_components_trials",
]
