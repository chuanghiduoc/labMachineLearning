import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open('model_stroke.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the PCA
with open('pca_model.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)

# Load the CSV data
data = pd.read_csv('stroke_data.csv', delimiter=';')

# Preprocessing
# Encode categorical variables
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
data['ever_married'] = data['ever_married'].map({'No': 0, 'Yes': 1})
data['work_type'] = data['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
data['Residence_type'] = data['Residence_type'].map({'Urban': 0, 'Rural': 1})
data['smoking_status'] = data['smoking_status'].map({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3})

# Handle missing values
imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(imputer.fit_transform(data.drop(columns=['id', 'stroke'])), columns=data.columns[1:-1])

# Apply PCA
data_pca = pca.transform(data_imputed)

# Make predictions
predictions = model.predict(data_pca)

# Calculate correct and incorrect predictions
true_labels = data['stroke'].values  # Assuming 'stroke' is the column with true labels
correct_predictions = (predictions == true_labels)
incorrect_predictions = ~correct_predictions

# Count the number of correct and incorrect predictions
num_correct = np.sum(correct_predictions)
num_incorrect = np.sum(incorrect_predictions)

# Output the predictions
for i, prediction in enumerate(predictions):
    print(f"ID: {data['id'].iloc[i]}, Dự đoán nguy cơ đột quỵ: {prediction}")

# Generate and print classification report
report = classification_report(true_labels, predictions)
print(report)

# Generate and print confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Plotting confusion matrix as heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Dự đoán Không Đột Quỵ', 'Dự đoán Đột Quỵ'], yticklabels=['Thực tế Không Đột Quỵ', 'Thực tế Đột Quỵ'])
plt.title('Biểu Đồ Ma Trận Nhầm Lẫn')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()

# Vẽ biểu đồ dự đoán đúng và sai
labels = ['Dự đoán Đúng', 'Dự đoán Sai']
counts = [num_correct, num_incorrect]

plt.bar(labels, counts, color=['green', 'red'])
plt.title('Số Lượng Dự Đoán Đúng và Sai')
plt.xlabel('Loại Dự Đoán')
plt.ylabel('Số Lượng')
plt.grid(axis='y')

# Thêm số lượng cụ thể vào các cột
for i, count in enumerate(counts):
    plt.text(i, count + 0.5, str(count), ha='center', va='bottom')

plt.show()