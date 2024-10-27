# Dự Án Dự Đoán Đột Quỵ
Chỉ để phục vụ học tập với các thuật toán

## Mô Tả
Dự án này sử dụng mô hình máy học để dự đoán khả năng đột quỵ ở bệnh nhân dựa trên các đặc điểm như tuổi tác, giới tính, huyết áp, tiểu đường và các thông tin sức khỏe khác. Mô hình sử dụng thuật toán CART, PCA và Random Forest để thực hiện dự đoán.

## Công Nghệ Sử Dụng
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
## Dữ Liệu
Dữ liệu được sử dụng trong dự án được lấy từ [tên nguồn dữ liệu]. Dưới đây là thông tin chi tiết về các trường trong tập dữ liệu:

| Tên trường            | Mô tả                                         |
|----------------------|------------------------------------------------|
| `id`                 | Định danh bệnh nhân                            |
| `gender`             | Giới tính (0: Nữ, 1: Nam)                      |
| `age`                | Tuổi                                           |
| `hypertension`       | Huyết áp cao (0: Không, 1: Có)                 |
| `heart_disease`      | Bệnh tim (0: Không, 1: Có)                     |
| `ever_married`       | Tình trạng kết hôn (0: Chưa, 1: Đã kết hôn)    |
| `work_type`          | Loại công việc                                 |
| `Residence_type`     | Loại nơi cư trú                                | 
| `avg_glucose_level`  | Mức đường huyết trung bình                     |
| `bmi`                | Chỉ số khối cơ thể                             |
| `smoking_status`     | Tình trạng hút thuốc                           |
| `stroke`             | Kết quả dự đoán (0: Không đột quỵ, 1: Đột quỵ) |

## Cài Đặt
Để cài đặt các thư viện cần thiết, bạn có thể sử dụng pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```
## Cách Sử Dụng
Có thể dùng model đã được train ở trong project hoặc tự train


