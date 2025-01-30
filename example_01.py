# Example1 => หน้าที่ 5
# เป้าหมายหลักของการวิเคราะห์เชิงปริมาณคือการใช้ข้อมูลปัจจุบันเกี่ยวกับปรากฏการณ์
# เพื่อทํานายพฤติกรรมในอนาคต 
# •ข้อมูลปัจจุบันมักจะอยู่ในรูปแบบของชุดข้อมูล 
# •ในกรณีง่ายๆเมื่อข้อมูลสร้างชุดของคู่ของตัวเลขเราอาจตีความว่าเป็นตัวแทนของค่าที่สังเกตได้ของตัวแปรอิสระ 
# (หรือตัวทํานาย) X และตัวแปรตาม (หรือการตอบสนอง) Y
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ข้อมูล Lot Size (X) และ Man-hours (Y)
X = np.array([30, 20, 60, 80, 40, 50, 60, 30, 70, 60]).reshape(-1, 1)  # ตัวแปรอิสระ (Independent Variable)
Y = np.array([73, 50, 128, 170, 87, 108, 135, 69, 148, 132])  # ตัวแปรตาม (Dependent Variable)

# สร้างโมเดล Simple Linear Regression
model = LinearRegression()
model.fit(X, Y)  # ฝึกสอนโมเดล (Training the model)

# คำนวณค่าคาดการณ์ (Prediction)
Y_pred = model.predict(X)

# แสดงค่าค่าสัมประสิทธิ์และค่า Intercept
print(f"Slope (ค่าสัมประสิทธิ์ของ X): {model.coef_[0]}")
print(f"Intercept (จุดตัดแกน Y): {model.intercept_}")

# คำนวณค่าความถูกต้องของโมเดล
mse = mean_squared_error(Y, Y_pred)  # คำนวณ Mean Squared Error
r2 = r2_score(Y, Y_pred)  # คำนวณค่า R-squared
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (ค่าความแม่นยำของโมเดล): {r2:.2f}")

# แสดงสมการการถดถอยโดยประมาณ
slope = model.coef_[0]
intercept = model.intercept_
print(f"สมการการถดถอยโดยประมาณ: Y = {intercept:.0f}+{slope:.1f}X")

# การสร้างกราฟแสดงผล
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.flatten(), y=Y, color='blue', label='Actual Data')  # จุดข้อมูลจริง
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')  # เส้นสมการเชิงเส้น
plt.xlabel("Lot Size")
plt.ylabel("Man-hours")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()


