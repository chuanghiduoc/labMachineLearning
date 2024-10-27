import pandas as pd
import numpy as np # lam viec voi ma tran & array
from sklearn.model_selection import train_test_split # chia tap dl thanh 2 phan traning data & test data
from sklearn import decomposition # giam chieu dl
from sklearn.tree import DecisionTreeClassifier # cay phan lop
from sklearn.metrics import confusion_matrix # ma tran nham lan
from sklearn.metrics import accuracy_score # do chinh xac
import tkinter as inp #giao dien
from tkinter.ttk import *
from sklearn.impute import SimpleImputer
from tkinter import messagebox
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle

# Đọc file
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Map các cột phân loại thành các giá trị số
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
df['work_type'] = df['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
df['Residence_type'] = df['Residence_type'].map({'Urban': 0, 'Rural': 1})
df['smoking_status'] = df['smoking_status'].map({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3})

# Ma trận dữ liệu X
X = np.array(df.drop(columns=['stroke', 'id']))
# Ma trận nhãn lớp Y
y = np.array([df["stroke"]]).T
print(X.shape)
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Chọn tập có n thuộc tính tốt nhất bằng phương pháp PCA
train_scores = []  # Danh sách lưu trữ độ chính xác trên tập huấn luyện
test_scores = []   # Danh sách lưu trữ độ chính xác trên tập kiểm tra
n = 0
score = 0

for i in range(1, X.shape[1] + 1):
    print("Lần:", i)
    pca = decomposition.PCA(n_components=i)
    pca.fit(X)
    Xbar = pca.transform(X)  # Áp dụng giảm kích thước cho X.

    # Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(Xbar, y, test_size=0.3, shuffle=True)

    # Tạo mô hình cây quyết định
    # model = RandomForestClassifier(n_estimators=100, max_depth=3)
    # model = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=50, min_samples_leaf=15, ccp_alpha=0.01)
    model = DecisionTreeClassifier(criterion="gini")
    # noise = np.random.normal(0, 0.1, X_train.shape)  # Tạo nhiễu
    # X_train_noisy = X_train + noise 
    model.fit(X_train, y_train)

    # Dự đoán trên tập huấn luyện và kiểm tra
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Tính toán độ chính xác
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"Độ chính xác trên tập huấn luyện: {train_score:.2f}")
    print(f"Độ chính xác trên tập kiểm tra: {test_score:.2f}")
    print(f"Độ chính xác trung bình (Cross-Validation): {cv_scores.mean():.2f}")

    # Lưu trữ độ chính xác
    train_scores.append(train_score)
    test_scores.append(test_score)

    if test_score >= score:
        score = test_score
        n = i
    

# Sử dụng tập n thuộc tính tốt đã chọn để tạo ra tập huấn luyện (train) và tập kiểm tra (test) mới
print("N_components:", n)
main_pca = decomposition.PCA(n_components=n)
main_pca.fit(X)
Xbar1 = main_pca.transform(X)
X_train1, X_test1, y_train1, y_test1 = train_test_split(Xbar1, y, test_size=0.3, shuffle=True)

def plot_accuracy_scores(train_scores, test_scores, max_components):
    """
    Vẽ biểu đồ độ chính xác của mô hình theo số lượng thành phần chính.

    Args:
    - train_scores: Danh sách chứa độ chính xác trên tập huấn luyện.
    - test_scores: Danh sách chứa độ chính xác trên tập kiểm tra.
    - max_components: Số lượng thành phần chính lớn nhất đã sử dụng trong PCA.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), train_scores, label='Độ chính xác trên tập huấn luyện', marker='o')
    plt.plot(range(1, max_components + 1), test_scores, label='Độ chính xác trên tập kiểm tra', marker='o')
    plt.xlabel('Số lượng thành phần chính (n_components)')
    plt.ylabel('Độ chính xác')
    plt.title('Độ chính xác của mô hình theo số lượng thành phần chính')
    plt.legend()
    plt.grid()
    plt.show()

# Gọi hàm sau khi hoàn thành vòng lặp
# plot_accuracy_scores(train_scores, test_scores, X.shape[1])

# dung CART (cay phan lop) de xd mo hinh
mainModel = DecisionTreeClassifier(criterion = "gini")
print("Số đặc trưng trong tập huấn luyện:", X_train1.shape[1])

# AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=50)

# Huấn luyện mô hình
mainModel.fit(X_train1, y_train1)

with open('model_stroke.pkl', 'wb') as model_file:
    pickle.dump(mainModel, model_file)

print("Mô hình đã được lưu vào 'model_stroke.pkl'.")
with open('model_stroke.pkl', 'rb') as feature_file:
    feature_names = pickle.load(feature_file)

print("Các đặc trưng đã được sử dụng trong mô hình:", feature_names)
y_pred1 = mainModel.predict(X_test1)

# ma tran nham lan
#                                        predict
#                      |     positive        |    negative
#    ------------------|---------------------|--------------
#      true | positive |  True positive (TP) | False Negative (FN)
#           | negative |  False positive (FP)| True Negative (TN)

cnf_matrix = confusion_matrix(y_test1, y_pred1)
print('Ma trận nhầm lẫn:')
print(cnf_matrix)

# ham tinh precision va recall
def cm2pr_binary(cm):
    p = cm[0,0]/np.sum(cm[:,0])
    r = cm[0,0]/np.sum(cm[0])
    return (p, r)

# danh gia mo hinh
acc = accuracy_score(y_test1, y_pred1) #do cxac
# precision = là tỷ lệ số điểm true positive (TP) trong tổng số điểm được phân loại là positive (TP + FP)
# recall = là tỷ lệ số điểm true positive (TP) trong tổng số điểm thực sự là positive (TP + FN)
precision,recall = cm2pr_binary(cnf_matrix)
# f1-score là kết hợp của precision & recall
f1_score = (2 * precision * recall) / (precision + recall)

print('Accuracy (Sự chính xác) = {0:.2f}'.format(acc))
print('Precision = {0:.2f}'.format(precision))
print('Recall = {0:.2f}'.format(recall))
print('F1-score = {0:.2f}'.format(f1_score))

def show_confusion_matrix():
    plt.figure(figsize=(6, 4))
    sns.heatmap(cnf_matrix, annot=True, fmt="d", cmap="viridis", cbar=False)
    plt.title('Ma trận nhầm lẫn')
    plt.xlabel('Giá trị dự đoán')
    plt.ylabel('Giá trị thực')
    plt.show()

# Phân bố số lượng đột quỵ
def plot_stroke_distribution(data):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x='stroke', palette='viridis')
    plt.title('Phân bố nguy cơ đột quỵ')
    plt.xlabel('Nguy cơ đột quỵ (0-Không, 1-Có)')
    plt.ylabel('Số lượng')
    plt.show()

#Biểu đồ phân phối tuổi theo nguy cơ đột quỵ

def plot_age_distribution_by_stroke(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='age', hue='stroke', multiple='stack', kde=True, palette='viridis')
    plt.title('Phân phối tuổi theo nguy cơ đột quỵ')
    plt.xlabel('Tuổi')
    plt.ylabel('Số lượng')
    plt.show()

# Biểu đồ nguy cơ đột quỵ theo hypertension và heart_disease
def plot_stroke_by_hypertension_heart_disease(data):
    plt.figure(figsize=(10, 4))
    
    # Cao huyết áp
    plt.subplot(1, 2, 1)
    sns.countplot(data=data, x='hypertension', hue='stroke', palette='viridis')
    plt.title('Nguy cơ đột quỵ theo cao huyết áp')
    plt.xlabel('Cao huyết áp (0-Không, 1-Có)')
    plt.ylabel('Số lượng')

    # Bệnh tim
    plt.subplot(1, 2, 2)
    sns.countplot(data=data, x='heart_disease', hue='stroke', palette='viridis')
    plt.title('Nguy cơ đột quỵ theo bệnh tim')
    plt.xlabel('Bệnh tim (0-Không, 1-Có)')
    plt.ylabel('Số lượng')

    plt.tight_layout()
    plt.show()

# Phân phối avg_glucose_level và bmi theo nguy cơ đột quỵ
def plot_glucose_bmi_distribution_by_stroke(data):
    plt.figure(figsize=(12, 5))

    # Đường huyết
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=data, x="avg_glucose_level", hue="stroke", fill=True, palette="viridis")
    plt.title('Phân phối mức đường huyết theo nguy cơ đột quỵ')
    plt.xlabel('Mức đường huyết trung bình (mg/dL)')

    # BMI
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=data, x="bmi", hue="stroke", fill=True, palette="viridis")
    plt.title('Phân phối BMI theo nguy cơ đột quỵ')
    plt.xlabel('Chỉ số BMI')

    plt.tight_layout()
    plt.show()

def show_input_interface():
    # giao dien
    master = inp.Tk()
    master.title('Nhập thông tin')

    labels = [
        "ID", 
        "Giới tính (0-Nam, 1-Nữ)",
        "Tuổi", 
        "Cao huyết áp (0-Không, 1-Có)", 
        "Bệnh tim (0-Không, 1-Có)", 
        "Đã kết hôn (0-Không, 1-Có)", 
        "Loại công việc (0-Tư nhân, 1-Tự làm, 2-Công chức, 3-Trẻ em, 4-Chưa làm việc)", 
        "Loại cư trú (0-Đô thị, 1-Nông thôn)", 
        "Mức đường huyết trung bình (mg/dL)", 
        "Cân nặng (kg)",  # New field for weight
        "Chiều cao (m)",   # New field for height
        "BMI",          # BMI field
        "Trạng thái hút thuốc (0-Chưa bao giờ, 1-Trước đây đã hút, 2-Hút, 3-Không rõ)"
    ]

    entries = []
    for i, label_text in enumerate(labels):
        # inp.Label(master, text=label_text).grid(row=i)
        inp.Label(master, text=label_text, anchor='w').grid(row=i, sticky='w')
        entry = Entry(master, width=30)
        entry.grid(row=i, column=1)
        entries.append(entry)

    inp.Label(master, text="Nguy cơ đột quỵ được dự đoán (0-Không, 1-Có)").grid(row=len(labels)+1)
    result_entry = Entry(master, width=30)
    result_entry.grid(row=len(labels)+1, column=1)

    def toggle_weight_height_fields():
        """Bật hoặc tắt các trường cân nặng và chiều cao dựa trên thông tin đầu vào BMI."""
        bmi_value = entries[11].get().strip()
        if bmi_value:
            entries[9].config(state='disabled')
            entries[10].config(state='disabled')
        else:
            entries[9].config(state='normal')
            entries[10].config(state='normal')

    def predict():

        try:
            bmi_entry = entries[11].get().strip() 
            if bmi_entry == "":
                weight = float(entries[9].get())  
                height = float(entries[10].get())  
                
                # Tính chỉ số BMI nếu trường BMI trống
                if height > 0:  
                    bmi = weight / (height ** 2)
                    entries[11].delete(0, inp.END) 
                    entries[11].insert(0, f"{bmi:.2f}")  
                else:
                    messagebox.showwarning("Lỗi đầu vào", "Chiều cao phải lớn hơn 0 để tính chỉ số BMI.")
                    return
            else:
                bmi = float(bmi_entry)
            result_entry.delete(0, 'end')
            data_new = np.array([[
                int(entries[0].get()), # ID (không dùng để dự đoán)
                int(entries[1].get()), # Giới tính
                int(entries[2].get()), # Tuổi
                int(entries[3].get()), # Tăng huyết áp
                int(entries[4].get()), # Bệnh tim
                int(entries[5].get()), # Đã từng kết hôn
                int(entries[6].get()), # Loại công việc
                int(entries[7].get()), # Loại nơi cư trú
                float(entries[8].get()), # Mức đường huyết trung bình
                bmi, # Sử dụng chỉ số BMI được tính toán hoặc cung cấp
                int(entries[12].get()) # Trạng thái hút thuốc
            ]])

            # Transform data using PCA
            data_new_pca = main_pca.transform(data_new[:, 1:])  # Exclude ID for prediction
            prediction = mainModel.predict(data_new_pca)[0]
            result_entry.insert(0, prediction)

        except ValueError:
            messagebox.showwarning("Lỗi đầu vào", "Vui lòng nhập các giá trị số hợp lệ.")

    # Bind the toggle function to the BMI entry
    entries[11].bind("<KeyRelease>", lambda event: toggle_weight_height_fields())

    inp.Button(master, text="Dự đoán", command=predict, activebackground='green').grid(row=len(labels), column=1)
    master.mainloop()

if __name__ == "__main__":

    # Vẽ tất cả các biểu đồ trong luồng chính
    # plot_stroke_distribution(df)
    # plot_age_distribution_by_stroke(df)
    # plot_stroke_by_hypertension_heart_disease(df)
    # plot_glucose_bmi_distribution_by_stroke(df)
    # show_confusion_matrix()

    # Giao diện nhập liệu
    show_input_interface()
