# MNIST Dataset – OOP & Giảm chiều

Dự án Python OOP làm việc với bộ dữ liệu MNIST (phong cách [hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)): thư viện hỗ trợ trong `lib`, pipeline cơ bản và bài toán giảm chiều (PCA) kèm so sánh baseline.

## Yêu cầu

- Python 3.10–3.13 (khuyến nghị 3.10–3.12 do giới hạn một số package)
- Pip

## Cài đặt

Thư mục `.venv` không được đưa lên git. Sau khi clone, tạo môi trường ảo và cài dependency:

```bash
python -m venv .venv
```

**Kích hoạt .venv:**

| Môi trường        | Lệnh |
|-------------------|------|
| Windows PowerShell | `.\\.venv\\Scripts\\Activate.ps1` |
| Windows CMD       | `.venv\\Scripts\\activate.bat` |
| Linux / macOS     | `source .venv/bin/activate` |

**Cài package:**

```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
PY--MNIST-Dataset/
├── lib/                    # Thư viện OOP
│   ├── __init__.py
│   ├── dataset.py          # MNISTDataLoader – tải & tiền xử lý
│   ├── model.py            # MNISTClassifier – wrapper sklearn
│   ├── reduction/          # Giảm chiều (tách file, phân nhánh trong module)
│   │   ├── __init__.py
│   │   ├── pca.py          # DimensionalityReducer (PCA)
│   │   ├── chi2.py         # ChiSquareReducer (Chi-Square)
│   │   └── reducer.py     # create_reducer, UnifiedReducer – interface chung
│   └── utils.py            # Vẽ, đo bộ nhớ/thời gian, so sánh
├── mnist_pipeline.ipynb    # Pipeline: tải → train → đánh giá
├── mnist_dimensionality_reduction.ipynb  # Notebook chung: biến METHOD (pca/chi2), không phân nhánh
├── mnist_dimensionality_reduction_pca.ipynb   # Riêng PCA (tham khảo)
├── mnist_dimensionality_reduction_chi2.ipynb  # Riêng Chi-Square (tham khảo)
├── requirements.txt
└── README.md
```

## Thư viện `lib`

- **MNISTDataLoader**: tải MNIST (package `mnist-datasets`), chuẩn hóa, flatten.
- **MNISTClassifier**: phân loại (logistic / random forest).
- **lib.reduction** (package): **pca.py** (DimensionalityReducer), **chi2.py** (ChiSquareReducer), **reducer.py** (create_reducer, UnifiedReducer, get_default_n_components_trials). Phân nhánh phương pháp chỉ trong reducer; notebook gọi chung một API.
- **Utils**: `plot_samples`, `plot_confusion_matrix`, `print_classification_report`, `measure_array_memory_mb`, `run_and_measure_seconds`, `plot_comparison_reduction`, `print_comparison_table`.

## Notebook

| Notebook | Nội dung |
|----------|----------|
| **mnist_pipeline.ipynb** | Tải MNIST, xem mẫu, train classifier, báo cáo phân lớp và confusion matrix. |
| **mnist_dimensionality_reduction.ipynb** | **Notebook chung:** đặt `METHOD = "pca"` hoặc `"chi2"`, gọi `create_reducer(METHOD, ...)`; không phân nhánh, so sánh baseline vs sau giảm chiều. |
| **mnist_dimensionality_reduction_pca.ipynb** | Riêng PCA (tham khảo). |
| **mnist_dimensionality_reduction_chi2.ipynb** | Riêng Chi-Square (tham khảo). |

Chạy notebook: mở file trong Jupyter/Cursor, chọn kernel trỏ tới `.venv\\Scripts\\python.exe` (hoặc Python 3.x ('.venv')).

## Giấy phép & tham chiếu

- MNIST: [Yann LeCun](http://yann.lecun.com/exdb/mnist/).
- Dataset tham khảo: [hojjatk/mnist-dataset (Kaggle)](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
- Load dữ liệu: [mnist-datasets](https://pypi.org/project/mnist-datasets/) (PyPI).
