import pandas as pd #doc file
import numpy as np # lam viec voi ma tran & array
from sklearn.model_selection import train_test_split # chia tap dl thanh 2 phan traning data & test data
from sklearn import decomposition # giam chieu dl
from sklearn.tree import DecisionTreeClassifier # cay phan lop
from sklearn.metrics import confusion_matrix # ma tran nham lan
from sklearn.metrics import accuracy_score # do chinh xac
import tkinter as inp #giao dien
from tkinter.ttk import *

# doc file
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
# ma tran du lieu X
X = np.array(df.drop(columns=['DEATH_EVENT']))
# ma tran nhan lop Y
y = np.array([df["DEATH_EVENT"]]).T

# chon tap co n thuoc tinh tot nhat = phg phap PCA
n = 0
score = 0
for i in range(1,13):
	print("Lần:", i)
	pca = decomposition.PCA(n_components=i)
	pca.fit(X)
	# print(pca)
	Xbar = pca.transform(X)  # ap dung giam kich thuoc cho X.
	X_train, X_test, y_train, y_test = train_test_split(Xbar, y, test_size=0.3, shuffle=False)
	model = DecisionTreeClassifier(criterion = "gini")
	model = model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	d = 0
	trained_score = accuracy_score(y_test, y_pred)
	print("Điểm:", trained_score)
	if(trained_score >= score):
		score = trained_score
		n = i

# dung tap n thuoc tinh tot da chon de tao ra tap tran & test moi
print("N_components:", n)
main_pca = decomposition.PCA(n_components=n)
main_pca.fit(X)
Xbar1 = main_pca.transform(X)
X_train1, X_test1, y_train1, y_test1 = train_test_split(Xbar1, y, test_size=0.3, shuffle=False)

# dung CART (cay phan lop) de xd mo hinh
mainModel = DecisionTreeClassifier(criterion = "gini")
mainModel.fit(X_train1, y_train1)
y_pred1 = mainModel.predict(X_test1)

# ma tran nham lan
#                                        predict
#                      |     positive        |    negative
#    ------------------|---------------------|--------------
#      true | positive |  True positive (TP) | False Negative (FN)
#           | negative |  False positive (FP)| True Negative (TN)

cnf_matrix = confusion_matrix(y_test1, y_pred1)
print('Confusion matrix:')
print(cnf_matrix)

# ham tinh precision va recall
def cm2pr_binary(cm):
    p = cm[0,0]/np.sum(cm[:,0])
    r = cm[0,0]/np.sum(cm[0])
    return (p, r)

# danh gia mo hinh
acc = accuracy_score(y_test1, y_pred1) #do cxac
# precision = ti le so diem true positive (TP) trong nhung diem duoc phan loai positive (TP+FP)
# recall = ti le so diem true positive (TP) trong nhung diem thuc su la positive (TP+FN)
precision,recall = cm2pr_binary(cnf_matrix)
# f1-score la ket hop cua precision & recall
f1_score = (2 * precision * recall) / (precision + recall)

print('Accuracy = {0:.2f}'.format(acc))
print('Precision = {0:.2f}'.format(precision))
print('Recall = {0:.2f}'.format(recall))
print('F1-score = {0:.2f}'.format(f1_score))

# giao dien
master = inp.Tk()
master.title('Nhập thông tin cho bệnh nhân')

inp.Label(master, text="Tuổi:").grid(row=0)
e1 = Entry(master, width=30)
e1.grid(row=0, column=1)

inp.Label(master, text="Thiếu máu (0-không, 1-có):").grid(row=1)
e2 = Entry(master, width=30)
e2.grid(row=1, column=1)

inp.Label(master, text="Enzym CPK trong máu (mcg/l):").grid(row=2)
e3 = Entry(master, width=30)
e3.grid(row=2, column=1)

inp.Label(master, text="Tiểu đường (0-không, 1-có):").grid(row=3)
e4 = Entry(master, width=30)
e4.grid(row=3, column=1)

inp.Label(master, text="Lượng máu rời khỏi tim mỗi lần co bóp (%):").grid(row=4)
e5 = Entry(master, width=30)
e5.grid(row=4, column=1)

inp.Label(master, text="Cao huyết áp (0-không, 1-có):").grid(row=5)
e6 = Entry(master, width=30)
e6.grid(row=5, column=1)

inp.Label(master, text="Tiểu cầu trong máu (kilophat tiểu cầu / ml):").grid(row=6)
e7 = Entry(master, width=30)
e7.grid(row=6, column=1)

inp.Label(master, text="Huyết thanh creatinine trong máu (mg/dl):").grid(row=7)
e8 = Entry(master, width=30)
e8.grid(row=7, column=1)

inp.Label(master, text="Huyết thanh sodium trong máu (mEq/l):").grid(row=8)
e9 = Entry(master, width=30)
e9.grid(row=8, column=1)

inp.Label(master, text="Giới tính (0-nam, 1-nữ):").grid(row=9)
e10 = Entry(master, width=30)
e10.grid(row=9, column=1)

inp.Label(master, text="Hút thuốc (0-không, 1-có):").grid(row=10)
e11 = Entry(master, width=30)
e11.grid(row=10, column=1)

inp.Label(master, text="Thời gian theo dõi (ngày):").grid(row=11)
e12 = Entry(master, width=30)
e12.grid(row=11, column=1)

inp.Label(master, text="Dự đoán tử vong (0-không, 1-có)").grid(row=13)
e13 = Entry(master, width=30)
e13.grid(row=13, column=1)

def predict():
	age = int(e1.get())
	anaemia = int(e2.get())
	creatinine_phosphokinase = int(e3.get())
	diabetes = int(e4.get())
	ejection_fraction = int(e5.get())
	high_blood_pressure = int(e6.get())
	platelets = int(e7.get())
	serum_creatinine = float(e8.get())
	serum_sodium = int(e9.get())
	sex = int(e10.get())
	smoking = int(e11.get())
	time = int(e12.get())
	data_new = np.array([[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,
                                   high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])
	data_new_pca = main_pca.transform(data_new)
	kq = mainModel.predict(data_new_pca)[0]
	e13.insert(0, kq)

inp.Button(master,text ="Dự đoán", command = predict,activebackground='green',
          justify='center').grid(row=12, column=1)
master.mainloop()

