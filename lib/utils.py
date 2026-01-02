"""Các hàm tiện ích: vẽ ảnh, đồ thị, ma trận nhầm lẫn, báo cáo phân lớp, đo bộ nhớ/thời gian."""

import time
import numpy as np
from typing import Optional, List, Callable, TypeVar
import matplotlib.pyplot as plt

T = TypeVar("T")


def plot_samples(
    images: np.ndarray,
    labels: np.ndarray,
    n_rows: int = 2,
    n_cols: int = 5,
    title: str = "MNIST samples",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Hiển thị lưới ảnh mẫu kèm nhãn.

    Args:
        images: Mảng ảnh (N, H, W) hoặc (N, 784).
        labels: Nhãn tương ứng (N,).
        n_rows: Số hàng.
        n_cols: Số cột.
        title: Tiêu đề figure.
        figsize: Kích thước figure (width, height).

    Returns:
        Figure matplotlib.
    """
    n = min(n_rows * n_cols, len(images), len(labels))
    if n == 0:
        raise ValueError("images và labels không được rỗng.")

    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)

    if figsize is None:
        figsize = (2 * n_cols, 2 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for i in range(n):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        ax.imshow(images[i], cmap="gray")
        ax.set_title(str(int(labels[i])), fontsize=12)
        ax.axis("off")
    for i in range(n, axes.size):
        r, c = i // n_cols, i % n_cols
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_training_history(
    history: dict[str, List[float]],
    title: str = "Training history",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Vẽ đồ thị loss/accuracy theo epoch (nếu có trong history).

    Args:
        history: Dict với key như 'loss', 'accuracy', 'val_loss', 'val_accuracy'.
        title: Tiêu đề figure.
        figsize: Kích thước figure.

    Returns:
        Figure matplotlib.
    """
    if figsize is None:
        figsize = (10, 4)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    n_epochs = len(history.get("loss", history.get("accuracy", [0])))
    epochs = range(1, n_epochs + 1)

    if "loss" in history:
        axes[0].plot(epochs, history["loss"], "b-", label="Train loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "r-", label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "accuracy" in history:
        axes[1].plot(epochs, history["accuracy"], "b-", label="Train accuracy")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], "r-", label="Val accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion matrix",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Vẽ ma trận nhầm lẫn.

    Args:
        y_true: Nhãn thật.
        y_pred: Nhãn dự đoán.
        class_names: Tên lớp (mặc định 0–9).
        title: Tiêu đề figure.
        figsize: Kích thước figure.

    Returns:
        Figure matplotlib.
    """
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        raise ImportError("Cần cài đặt: pip install scikit-learn") from None

    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    if figsize is None:
        figsize = (8, 6)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        xlabel="Predicted",
        ylabel="True",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    In báo cáo phân lớp (precision, recall, f1) ra console.

    Args:
        y_true: Nhãn thật.
        y_pred: Nhãn dự đoán.
        class_names: Tên lớp (mặc định 0–9).
    """
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        raise ImportError("Cần cài đặt: pip install scikit-learn") from None

    if class_names is None:
        class_names = [str(i) for i in range(10)]
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print(report)


def measure_array_memory_mb(arr: np.ndarray) -> float:
    """
    Ước lượng bộ nhớ (MB) của mảng NumPy.

    Args:
        arr: Mảng cần đo.

    Returns:
        Bộ nhớ tính bằng MB.
    """
    return float(arr.nbytes) / (1024.0 * 1024.0)


def run_and_measure_seconds(fn: Callable[[], T]) -> tuple[T, float]:
    """
    Chạy hàm không tham số và đo thời gian (giây).

    Args:
        fn: Hàm gọi được không tham số (ví dụ: lambda: model.fit(X, y)).

    Returns:
        (kết quả của fn(), thời gian chạy tính bằng giây).
    """
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def plot_comparison_reduction(
    baseline: dict,
    reduced: dict,
    title: str = "So sánh Baseline vs Sau giảm chiều",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Vẽ biểu đồ cột so sánh: bộ nhớ (MB), thời gian fit (s), thời gian predict (s), độ chính xác (%).

    Args:
        baseline: Dict với key: memory_mb, time_fit_s, time_predict_s, accuracy.
        reduced: Cùng format.
        title: Tiêu đề figure.
        figsize: Kích thước figure.

    Returns:
        Figure matplotlib.
    """
    if figsize is None:
        figsize = (12, 5)

    metrics = ["memory_mb", "time_fit_s", "time_predict_s", "accuracy"]
    labels = ["Bộ nhớ (MB)", "Thời gian fit (s)", "Thời gian predict (s)", "Độ chính xác (%)"]
    baseline_vals = [baseline.get(k, 0) for k in metrics]
    reduced_vals = [reduced.get(k, 0) for k in metrics]
    for i in range(4):
        if metrics[i] == "accuracy":
            baseline_vals[i] = baseline_vals[i] * 100.0
            reduced_vals[i] = reduced_vals[i] * 100.0

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    x = np.arange(2)
    width = 0.35

    for i, (ax, label) in enumerate(zip(axes, labels)):
        vals = [baseline_vals[i], reduced_vals[i]]
        ax.bar(x - width / 2, [baseline_vals[i], reduced_vals[i]], width, label=["Baseline", "Sau giảm chiều"])
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Sau giảm chiều"])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for j, v in enumerate(vals):
            ax.text(j - width / 2, v + (max(vals) * 0.02 if max(vals) > 0 else 0.1), f"{v:.3g}", ha="center", fontsize=9)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def print_comparison_table(
    baseline: dict,
    reduced: dict,
    reduced_name: str = "Sau giảm chiều",
) -> None:
    """
    In bảng so sánh baseline và sau giảm chiều (bộ nhớ, thời gian, độ chính xác, hao hụt).

    Args:
        baseline: Dict với memory_mb, time_fit_s, time_predict_s, accuracy.
        reduced: Cùng format.
        reduced_name: Tên cột cho dữ liệu reduced.
    """
    acc_b = baseline.get("accuracy", 0)
    acc_r = reduced.get("accuracy", 0)
    loss_pct = (acc_b - acc_r) / acc_b * 100.0 if acc_b > 0 else 0.0

    mem_b = baseline.get("memory_mb", 0)
    mem_r = reduced.get("memory_mb", 0)
    mem_save_pct = (mem_b - mem_r) / mem_b * 100.0 if mem_b > 0 else 0.0

    print("=" * 60)
    print("BẢNG SO SÁNH: BASELINE vs SAU GIẢM CHIỀU")
    print("=" * 60)
    print(f"{'Chỉ số':<25} {'Baseline':>12} {reduced_name:>12} {'Chênh lệch':>12}")
    print("-" * 60)
    print(f"{'Bộ nhớ (MB)':<25} {mem_b:>12.4f} {mem_r:>12.4f} {mem_save_pct:>+11.2f}%")
    print(f"{'Thời gian fit (s)':<25} {baseline.get('time_fit_s', 0):>12.4f} {reduced.get('time_fit_s', 0):>12.4f}")
    print(f"{'Thời gian predict (s)':<25} {baseline.get('time_predict_s', 0):>12.4f} {reduced.get('time_predict_s', 0):>12.4f}")
    print(f"{'Độ chính xác (%)':<25} {acc_b*100:>12.2f} {(acc_r*100):>12.2f} {loss_pct:>+11.2f}% (hao hụt)")
    print("=" * 60)
