"""Giảm chiều bằng chọn đặc trưng Chi-Square (SelectKBest + chi2)."""

import numpy as np
from typing import Union, Optional


class ChiSquareReducer:
    """Lớp giảm chiều bằng chọn đặc trưng Chi-Square (SelectKBest + chi2)."""

    def __init__(
        self,
        n_components: Union[int, float],
        random_state: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_components: Số đặc trưng (int) hoặc độ chính xác tối thiểu (float trong (0, 1]).
            random_state: Seed cho classifier khi n_components là float.
        """
        if isinstance(n_components, int) and n_components <= 0:
            raise ValueError("n_components (int) phải là số nguyên dương.")
        if isinstance(n_components, float) and not (0 < n_components <= 1):
            raise ValueError("n_components (float) phải trong khoảng (0, 1].")
        self._n_components = n_components
        self._random_state = random_state
        self._selector = None
        self._fitted = False
        self._min_accuracy_actual: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ChiSquareReducer":
        """Fit selector Chi-Square trên (X, y). X phải không âm."""
        try:
            from sklearn.feature_selection import SelectKBest, chi2
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError("Cần cài đặt: pip install scikit-learn") from None

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        n_features = X.shape[1]

        if isinstance(self._n_components, int):
            k = min(self._n_components, n_features)
            self._selector = SelectKBest(score_func=chi2, k=k)
            self._selector.fit(X, y)
            self._fitted = True
            return self

        target_acc = float(self._n_components)
        use_val = X_val is not None and y_val is not None
        if use_val and X_val.ndim == 3:
            X_val = X_val.reshape(X_val.shape[0], -1)

        sel_all = SelectKBest(score_func=chi2, k=min(n_features, 784))
        sel_all.fit(X, y)
        scores = sel_all.scores_
        rank = np.argsort(scores)[::-1]

        def accuracy_at_k(k: int) -> float:
            cols = rank[:k]
            Xk = X[:, cols]
            clf = LogisticRegression(max_iter=100, random_state=self._random_state)
            clf.fit(Xk, y)
            if use_val:
                Xv = X_val[:, cols]
                return float(clf.score(Xv, y_val))
            return float(clf.score(Xk, y))

        lo, hi = 1, n_features
        best_k = n_features
        while lo <= hi:
            mid = (lo + hi) // 2
            acc = accuracy_at_k(mid)
            if acc >= target_acc:
                best_k = mid
                hi = mid - 1
            else:
                lo = mid + 1

        self._min_accuracy_actual = accuracy_at_k(best_k)
        self._selector = SelectKBest(score_func=chi2, k=best_k)
        self._selector.fit(X, y)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Chọn k đặc trưng từ X."""
        if not self._fitted or self._selector is None:
            raise RuntimeError("Chưa gọi fit(X, y).")
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self._selector.transform(X)

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit và transform trong một bước."""
        return self.fit(X, y, X_val=X_val, y_val=y_val).transform(X)

    @property
    def n_components_(self) -> int:
        """Số chiều sau khi giảm."""
        if self._selector is None:
            raise RuntimeError("Chưa gọi fit(X, y).")
        return int(self._selector.get_support().sum())

    @property
    def scores_(self) -> np.ndarray:
        """Điểm Chi-Square theo từng đặc trưng (sau khi fit)."""
        if self._selector is None:
            raise RuntimeError("Chưa gọi fit(X, y).")
        return self._selector.scores_

    def min_accuracy_reached(self) -> Optional[float]:
        """Độ chính xác đạt được khi dùng chế độ float."""
        return self._min_accuracy_actual
