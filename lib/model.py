"""Module mô hình phân loại MNIST (OOP wrapper)."""

import numpy as np
from typing import Any, Optional


class MNISTClassifier:
    """Lớp bọc (wrapper) mô hình phân loại cho MNIST."""

    def __init__(
        self,
        model_type: str = "logistic",
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_type: 'logistic' hoặc 'forest'.
            random_state: Seed cho reproducibility.
            **kwargs: Tham số bổ sung cho model sklearn.
        """
        self._model_type = model_type
        self._random_state = random_state
        self._kwargs = kwargs
        self._model: Any = None
        self._history: dict[str, list[float]] = {}

    def _create_model(self) -> Any:
        """Tạo instance model sklearn."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            raise ImportError("Cần cài đặt: pip install scikit-learn") from None

        common = {"random_state": self._random_state, **self._kwargs}
        if self._model_type == "logistic":
            return LogisticRegression(max_iter=100, **common)
        if self._model_type == "forest":
            return RandomForestClassifier(n_estimators=100, **common)
        raise ValueError(
            f"model_type phải là 'logistic' hoặc 'forest', nhận: {self._model_type}"
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MNISTClassifier":
        """
        Huấn luyện mô hình.

        Args:
            X: Dữ liệu train (N, features).
            y: Nhãn train (N,).
            X_val, y_val: Validation (tùy chọn, dùng để ghi history).

        Returns:
            self (để chain).
        """
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        if X_val is not None and X_val.ndim == 3:
            X_val = X_val.reshape(X_val.shape[0], -1)

        self._model = self._create_model()
        self._model.fit(X, y)

        self._history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        train_acc = self._model.score(X, y)
        self._history["accuracy"].append(float(train_acc))
        if X_val is not None and y_val is not None:
            val_acc = self._model.score(X_val, y_val)
            self._history["val_accuracy"].append(float(val_acc))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn."""
        if self._model is None:
            raise RuntimeError("Chưa gọi fit().")
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self._model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Độ chính xác trên (X, y)."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return float(self._model.score(X, y))

    @property
    def history(self) -> dict[str, list[float]]:
        """Lịch sử train (accuracy/val_accuracy nếu có)."""
        return self._history
