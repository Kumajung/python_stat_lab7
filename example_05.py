# Example5 => ศึกษาความสัมพันธ์ระหว่าง "จำนวนชั่วโมงการอ่านหนังสือ (Study Hours)" กับ "คะแนนสอบ (Exam Score)"
# ข้อมูลจากไฟล์ study_hours_vs_exam_scores.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# อ่านข้อมูลจากไฟล์ CSV
data = pd.read_csv('/kaggle/input/study-hours-vs-exam-scores/study_hours_vs_exam_scores.csv')
X = data[['Study Hours']].values  # ตัวแปรอิสระ (Independent Variable)
Y = data['Exam Score'].values  # ตัวแปรตาม (Dependent Variable)

# สร้างโมเดล Simple Linear Regression
model = LinearRegression()
model.fit(X, Y)  # ฝึกสอนโมเดล (Training the model)

# คำนวณค่าคาดการณ์ (Prediction)
Y_pred = model.predict(X)

# แสดงค่าค่าสัมประสิทธิ์และค่า Intercept
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope (ค่าสัมประสิทธิ์ของ X): {slope:.2f}")
print(f"Intercept (จุดตัดแกน Y): {intercept:.2f}")
print(f"สมการการถดถอยโดยประมาณ: Exam Score = {intercept:.0f} + {slope:.1f} * Study Hours")

# คำนวณค่าความถูกต้องของโมเดล
mse = mean_squared_error(Y, Y_pred)  # Mean Squared Error
r2 = r2_score(Y, Y_pred)  # R-squared
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"R-squared (ค่าความแม่นยำของโมเดล): {r2:.2f}")

# การสร้างกราฟแสดงผล
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.flatten(), y=Y, color='blue', label='Actual Data')  # จุดข้อมูลจริง
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')  # เส้นสมการเชิงเส้น
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Simple Linear Regression: Study Hours vs. Exam Score")
plt.legend()
plt.show()
