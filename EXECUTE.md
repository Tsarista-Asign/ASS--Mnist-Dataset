# Báo cáo nghiên cứu: Giảm chiều dữ liệu trên MNIST

Báo cáo dựa trên các phương pháp, thuật toán được xây dựng và kết quả thực nghiệm trong dự án **PY--MNIST-Dataset** (OOP & Giảm chiều).

---

## 1. Khái niệm

### 1.1 Giảm chiều dữ liệu là gì?

**Giảm chiều dữ liệu (Dimensionality Reduction)** là kỹ thuật biến đổi dữ liệu từ không gian có nhiều chiều (nhiều đặc trưng) sang không gian có ít chiều hơn, sao cho vẫn giữ được phần lớn thông tin quan trọng phục vụ cho học máy hoặc phân tích.

Trong dự án này:

- **Dữ liệu gốc**: Mỗi ảnh MNIST 28×28 pixel được trải thành vector **784 chiều** (flatten).
- **Sau giảm chiều**: Dữ liệu được biến đổi sang không gian ít chiều hơn (ví dụ **154 chiều** khi dùng PCA với 95% phương sai), hoặc chỉ giữ lại một tập con các đặc trưng (khi dùng Chi-Square).

### 1.2 Hai phương pháp được triển khai

| Phương pháp | Mô tả ngắn |
|-------------|------------|
| **PCA (Principal Component Analysis)** | Phân tích thành phần chính: chiếu dữ liệu lên các trục phương sai lớn nhất. Không cần nhãn; tham số chính là số thành phần hoặc **tỉ lệ phương sai** cần giữ lại (ví dụ 0.95 = 95%). |
| **Chi-Square (SelectKBest + χ²)** | Chọn đặc trưng dựa trên điểm χ² giữa từng đặc trưng và nhãn. Cần nhãn khi huấn luyện; tham số là **số đặc trưng** (int) hoặc **độ chính xác tối thiểu** (float) để tìm k tối ưu. |

Cả hai đều được gói trong **interface thống nhất** (`UnifiedReducer`, `create_reducer`) trong package `lib.reduction`, giúp notebook chỉ cần đổi biến `METHOD` (pca / chi2) và tham số `n_components` mà không phân nhánh code.

---

## 2. Mục đích

- **Giảm bộ nhớ lưu trữ**: Ma trận đặc trưng nhỏ hơn (ít chiều hơn) tốn ít RAM hơn khi train và predict.
- **Rút ngắn thời gian huấn luyện và dự đoán**: Mô hình (ví dụ Logistic Regression) làm việc trên ít biến hơn nên fit và predict nhanh hơn.
- **Giảm nhiễu và overfitting**: Bớt chiều vô ích hoặc tương quan cao giúp mô hình ổn định hơn, đôi khi cải thiện khả năng tổng quát.
- **Dễ trực quan hóa và phân tích**: Dữ liệu ít chiều (đặc biệt sau PCA) có thể dùng cho visualization (2D/3D) hoặc phân tích tiếp theo.

Trong bối cảnh MNIST, mục tiêu cụ thể là: **giữ độ chính xác phân loại gần với baseline** trong khi **giảm mạnh bộ nhớ và thời gian** (fit, predict).

---

## 3. Hiệu quả

### 3.1 Chỉ số định lượng (minh chứng từ notebook)

Thí nghiệm: **METHOD = "pca"**, **n_components = 0.95** (giữ 95% phương sai), classifier: **Logistic Regression**, **random_state = 42**.

| Chỉ số | Baseline (784 chiều) | Sau giảm chiều (PCA, 154 chiều) | Nhận xét |
|--------|----------------------|----------------------------------|----------|
| **Bộ nhớ X_train (MB)** | 358,89 | 70,50 | Giảm ~**80,36%** |
| **Thời gian fit (s)** | 19,69 | 3,60 | Giảm ~**81,7%** |
| **Thời gian predict (s)** | 0,0256 | 0,0052 | Giảm ~**79,7%** |
| **Độ chính xác (%)** | 92,57 | 92,29 | Hao hụt ~**0,28%** |

- **Số chiều**: 784 → **154** (tỉ lệ phương sai tích lũy đạt ~0,9502).
- Kết luận định lượng: giảm chiều bằng PCA (95% phương sai) **tiết kiệm rất lớn** bộ nhớ và thời gian, trong khi **độ chính xác gần như giữ nguyên**.

### 3.2 Hiệu quả về mặt thiết kế

- **API thống nhất**: Một luồng code cho cả PCA và Chi-Square (fit → transform → train → đánh giá), dễ so sánh và mở rộng.
- **Tham số linh hoạt**: PCA dùng tỉ lệ phương sai (0,99; 0,95; 0,90; 0,80); Chi-Square dùng số đặc trưng (300, 154, 100, 50) hoặc độ chính xác mục tiêu (float).
- **Có thể thử nhiều mức** `n_components` qua `get_default_n_components_trials(METHOD)` và so sánh bộ nhớ / độ chính xác theo từng mức.

---

## 4. Minh chứng

Minh chứng dưới đây là **số liệu thực tế** đo được trong thí nghiệm (chạy trên bộ MNIST, classifier Logistic Regression, random_state = 42), không đề cập đến phần triển khai code.

### 4.1 Cấu hình thí nghiệm

- **Dữ liệu**: MNIST — X_train (60 000 mẫu, 784 chiều), X_test (10 000 mẫu).
- **Phương pháp giảm chiều**: PCA.
- **Tham số**: n_components = 0,95 (giữ 95% phương sai).

### 4.2 Số liệu đo được

**Số chiều và phương sai:**

| Chỉ số | Giá trị |
|--------|--------|
| Số chiều gốc | 784 |
| Số chiều sau giảm (PCA 95%) | 154 |
| Tổng phương sai tích lũy giữ lại | 0,9502 (≈95,02%) |

**Bộ nhớ X_train:**

| Điều kiện | Bộ nhớ (MB) | Chênh lệch |
|-----------|-------------|------------|
| Baseline (784 chiều) | 358,8867 | — |
| Sau giảm chiều (154 chiều) | 70,4956 | Tiết kiệm 80,36% |

**Thời gian và độ chính xác:**

| Chỉ số | Baseline | Sau giảm chiều |
|--------|----------|----------------|
| Thời gian fit (s) | 19,6934 | 3,5998 |
| Thời gian predict (s) | 0,0256 | 0,0052 |
| Độ chính xác (accuracy) | 0,9257 (92,57%) | 0,9229 (92,29%) |

**Báo cáo phân lớp (sau giảm chiều):** 
*accuracy tổng thể 0,9229 trên 10 000 mẫu test đã được ghi lại trong output của thí nghiệm.*

### 4.3 Thử nhiều mức n_components (PCA)

Khi thay đổi n_components (tỉ lệ phương sai), một lần chạy thí nghiệm cho kết quả ví dụ:

| n_components | Số chiều | Bộ nhớ (MB) | Accuracy |
|--------------|----------|-------------|----------|
| 0,9500 | 154 | 70,4956 | 0,9229 |

Các mức khác (0,99; 0,90; 0,80) có thể chạy tương tự để so sánh trade-off giữa số chiều, bộ nhớ và độ chính xác.

---

Toàn bộ số liệu trên là **kết quả đo trực tiếp** từ thí nghiệm (output notebook), dùng làm minh chứng định lượng cho hiệu quả giảm chiều.

---

## 5. Kết luận

- **Khái niệm**: Giảm chiều dữ liệu là việc biến đổi dữ liệu từ không gian nhiều chiều (ở đây 784) sang ít chiều hơn (ví dụ 154) nhằm giữ lại phần lớn thông tin hữu ích. Dự án triển khai hai phương pháp: **PCA** (không cần nhãn, dựa trên phương sai) và **Chi-Square** (cần nhãn, chọn đặc trưng theo điểm χ²), với API thống nhất trong `lib.reduction`.

- **Mục đích**: Giảm bộ nhớ, rút ngắn thời gian train/predict, và nếu có thể giảm nhiễu/overfitting, trong khi vẫn duy trì độ chính xác phân loại gần với baseline.

- **Hiệu quả**: Với PCA (95% phương sai) trên MNIST, bộ nhớ và thời gian fit/predict giảm khoảng **80%**, còn độ chính xác chỉ hao hụt khoảng **0,28%** (92,57% → 92,29%), cho thấy giảm chiều trong bài toán này **rất hiệu quả** về trade-off tài nguyên – chất lượng.

- **Minh chứng**: Là số liệu thực tế : số chiều (784 → 154), tỉ lệ phương sai giữ lại (~95%), bộ nhớ (358,89 MB → 70,50 MB, tiết kiệm ~80%), thời gian fit/predict giảm mạnh, độ chính xác 92,57% → 92,29% (hao hụt ~0,28%). Các giá trị này được ghi lại trực tiếp từ output của thí nghiệm.

**Kết luận chung**: Nghiên cứu giảm chiều trong dự án MNIST đạt được mục tiêu: khái niệm (PCA, Chi-Square), mục đích (tiết kiệm tài nguyên, giữ chất lượng) được đáp ứng, hiệu quả được định lượng bằng bộ nhớ và thời gian giảm mạnh với hao hụt độ chính xác rất nhỏ. Minh chứng dựa hoàn toàn trên **số liệu thực tế** từ thí nghiệm (bộ nhớ, thời gian, accuracy, số chiều, phương sai). Giảm chiều dữ liệu trong bối cảnh này là một bước tiền xử lý có lợi rõ ràng cho bài toán phân loại MNIST.
