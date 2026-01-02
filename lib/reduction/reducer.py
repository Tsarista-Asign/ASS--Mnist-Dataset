"""
Interface thống nhất cho giảm chiều. Phân nhánh phương pháp (PCA / Chi-Square) chỉ trong module này.
Notebook chỉ cần gọi create_reducer(method, ...) và dùng chung một API.
"""

import numpy as np
from typing import Union, Optional, Any

from lib.reduction.pca import DimensionalityReducer as PCAReducer
from lib.reduction.chi2 import ChiSquareReducer as Chi2Reducer


REDUCTION_METHODS = ("pca", "chi2")


def get_default_n_components_trials(method: str) -> list:
    """
    Danh sách n_components mặc định để thử (phân nhánh trong module).
    PCA: tỉ lệ phương sai (float); Chi2: số đặc trưng (int).
    """
    method_lower = method.strip().lower()
    if method_lower not in REDUCTION_METHODS:
        raise ValueError(f"method phải là một trong {REDUCTION_METHODS}, nhận: {method!r}")
    if method_lower == "pca":
        return [0.99, 0.95, 0.90, 0.80]
    return [300, 154, 100, 50]


def create_reducer(
    method: str,
    n_components: Union[int, float],
    random_state: Optional[int] = None,
) -> "UnifiedReducer":
    """
    Tạo reducer theo phương pháp (phân nhánh nội bộ).

    Args:
        method: "pca" hoặc "chi2".
        n_components: Với PCA: số thành phần (int) hoặc tỉ lệ phương sai (float).
            Với Chi2: số đặc trưng (int) hoặc độ chính xác tối thiểu (float).
        random_state: Seed.

    Returns:
        UnifiedReducer với API thống nhất.
    """
    method_lower = method.strip().lower()
    if method_lower not in REDUCTION_METHODS:
        raise ValueError(
            f"method phải là một trong {REDUCTION_METHODS}, nhận: {method!r}"
        )
    return UnifiedReducer(
        method=method_lower,
        n_components=n_components,
        random_state=random_state,
    )


class UnifiedReducer:
    """
    Wrapper thống nhất: nhận method và ủy quyền cho PCA hoặc Chi2.
    Notebook chỉ gọi fit/fit_transform/transform và thuộc tính chung.
    """

    def __init__(
        self,
        method: str,
        n_components: Union[int, float],
        random_state: Optional[int] = None,
    ) -> None:
        self._method = method.strip().lower()
        if self._method not in REDUCTION_METHODS:
            raise ValueError(
                f"method phải là một trong {REDUCTION_METHODS}, nhận: {method!r}"
            )
        self._n_components = n_components
        self._random_state = random_state
        self._impl: Any = None
        self._build_impl()

    def _build_impl(self) -> None:
        if self._method == "pca":
            self._impl = PCAReducer(
                n_components=self._n_components,
                random_state=self._random_state,
            )
        else:
            self._impl = Chi2Reducer(
                n_components=self._n_components,
                random_state=self._random_state,
            )

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "UnifiedReducer":
        """Fit reducer. PCA chỉ dùng X; Chi2 dùng X, y và tùy chọn X_val, y_val."""
        if self._method == "pca":
            self._impl.fit(X)
        else:
            if y is None:
                raise ValueError("Chi-Square reducer cần y khi gọi fit().")
            self._impl.fit(X, y, X_val=X_val, y_val=y_val)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X sang không gian giảm chiều."""
        return self._impl.transform(X)

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit và transform trong một bước."""
        if self._method == "pca":
            return self._impl.fit_transform(X)
        if y is None:
            raise ValueError("Chi-Square reducer cần y khi gọi fit_transform().")
        return self._impl.fit_transform(X, y, X_val=X_val, y_val=y_val)

    @property
    def n_components_(self) -> int:
        """Số chiều sau khi giảm."""
        return self._impl.n_components_

    def total_explained_variance_ratio(self) -> Optional[float]:
        """Chỉ có ý nghĩa với PCA; Chi2 trả về None."""
        if self._method == "pca":
            return self._impl.total_explained_variance_ratio()
        return None

    def min_accuracy_reached(self) -> Optional[float]:
        """Chỉ có ý nghĩa với Chi2 (chế độ float); PCA trả về None."""
        if self._method == "chi2":
            return self._impl.min_accuracy_reached()
        return None

    @property
    def method(self) -> str:
        """Tên phương pháp đang dùng."""
        return self._method

    @property
    def method_label(self) -> str:
        """Nhãn hiển thị (PCA / Chi-Square) để dùng trong báo cáo."""
        if self._method == "pca":
            return "PCA"
        return "Chi-Square"
