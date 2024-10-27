import pandas as pd
import numpy as np

def generate_additional_stroke_data(input_file_path, output_file_path, num_additional_records):
    # Đọc dữ liệu từ file CSV hiện có
    data = pd.read_csv(input_file_path, delimiter=',')

    # In ra tên các cột để kiểm tra
    print("Tên cột trong DataFrame:", data.columns)

    # Xóa khoảng trắng nếu có
    data.columns = data.columns.str.strip()

    # Tạo id mới cho các bản ghi bổ sung
    existing_ids = data['id'].unique()
    new_ids = np.arange(existing_ids.max() + 1, existing_ids.max() + 1 + num_additional_records)

    # Tạo dữ liệu ngẫu nhiên cho các cột cần thiết
    additional_data = {
        'id': new_ids,
        'gender': np.random.choice(data['gender'].unique(), size=num_additional_records),
        'age': np.random.randint(30, 80, size=num_additional_records),  # Giả định độ tuổi từ 30 đến 80
        'hypertension': np.random.choice([0, 1], size=num_additional_records),
        'heart_disease': np.random.choice([0, 1], size=num_additional_records),
        'ever_married': np.random.choice(['Yes', 'No'], size=num_additional_records),
        'work_type': np.random.choice(data['work_type'].unique(), size=num_additional_records),
        'Residence_type': np.random.choice(data['Residence_type'].unique(), size=num_additional_records),
        'avg_glucose_level': np.random.uniform(60, 300, size=num_additional_records),  # Giả định mức glucose
        'bmi': np.random.uniform(15, 40, size=num_additional_records),  # Giả định BMI
        'smoking_status': np.random.choice(data['smoking_status'].unique(), size=num_additional_records),
        'stroke': [1] * num_additional_records  # Đặt giá trị stroke = 1
    }

    # Tạo DataFrame từ dữ liệu bổ sung
    additional_df = pd.DataFrame(additional_data)

    # Kết hợp dữ liệu gốc với dữ liệu bổ sung
    combined_data = pd.concat([data, additional_df], ignore_index=True)

    # Lưu dữ liệu đã kết hợp vào một file CSV mới
    combined_data.to_csv(output_file_path, sep=',', index=False)

if __name__ == "__main__":
    input_file = 'testData.csv'  # Thay đổi đường dẫn đến file đầu vào
    output_file = 'testData.csv'  # Thay đổi đường dẫn đến file đầu ra
    additional_records = 3000  # Số lượng bản ghi cần tạo thêm

    generate_additional_stroke_data(input_file, output_file, additional_records)
    print(f"Đã tạo thêm {additional_records} bản ghi stroke = 1 và lưu vào {output_file}.")
