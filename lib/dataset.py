"""Module tải và tiền xử lý MNIST dataset (OOP)."""

import numpy as np
from typing import Tuple, Optional


class MNISTDataLoader:
    """Lớp tải và tiền xử lý dữ liệu MNIST."""

    def __init__(
        self,
        normalize: bool = True,
        flatten: bool = False,
        data_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            normalize: Chuẩn hóa pixel về [0, 1] nếu True.
            flatten: Trải ảnh 28x28 thành vector 784 nếu True.
            data_dir: Thư mục lưu/tải file MNIST (tùy chọn).
        """
        self._normalize = normalize
        self._flatten = flatten
        self._data_dir = data_dir
        self._train_images: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        self._test_images: Optional[np.ndarray] = None
        self._test_labels: Optional[np.ndarray] = None

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Tải dữ liệu train và test từ nguồn (mnist_datasets).

        Returns:
            (X_train, y_train, X_test, y_test) đã áp dụng preprocess.
        """
        try:
            from mnist_datasets import MNISTLoader as _MNISTLoader
        except ImportError:
            raise ImportError(
                "Cần cài đặt: pip install mnist-datasets"
            ) from None

        loader = _MNISTLoader(
            folder=self._data_dir if self._data_dir is not None else "mnist_data"
        )
        train_images, train_labels = loader.load(train=True)
        test_images, test_labels = loader.load(train=False)

        self._train_images = self._preprocess(train_images)
        self._train_labels = np.array(train_labels, dtype=np.int64)
        self._test_images = self._preprocess(test_images)
        self._test_labels = np.array(test_labels, dtype=np.int64)

        return (
            self._train_images,
            self._train_labels,
            self._test_images,
            self._test_labels,
        )

    def _preprocess(self, images: np.ndarray) -> np.ndarray:
        """Chuẩn hóa và (tùy chọn) flatten ảnh."""
        out = np.array(images, dtype=np.float64)
        if self._normalize:
            out = out / 255.0
        if self._flatten and out.ndim == 3:
            out = out.reshape(out.shape[0], -1)
        return out

    def get_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (X_train, y_train). Phải gọi load() trước."""
        if self._train_images is None:
            raise RuntimeError("Chưa gọi load(). Gọi load() trước khi get_train().")
        return self._train_images, self._train_labels

    def get_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (X_test, y_test). Phải gọi load() trước."""
        if self._test_images is None:
            raise RuntimeError("Chưa gọi load(). Gọi load() trước khi get_test().")
        return self._test_images, self._test_labels

    def get_class_names(self) -> list[str]:
        """Tên các lớp (chữ số 0–9)."""
        return [str(i) for i in range(10)]

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Shape của một mẫu đầu vào (sau preprocess)."""
        if self._train_images is None:
            return (28, 28) if not self._flatten else (784,)
        return self._train_images.shape[1:]
