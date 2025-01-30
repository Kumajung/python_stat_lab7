# Example 2 => หน้า 19 ตัวอย่าง: ค่าใช้จ่ายในการโฆษณารายสัปดาห์ (x) และยอดขายรายสัปดาห์ (y) แสดงในตารางต่อไปนี้
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ข้อมูล Weekly Advertising Expenditure (X) และ Weekly Sales (Y)
X = np.array([41, 54, 63, 54, 48, 46, 62, 61, 64, 71]).reshape(-1, 1)  # ตัวแปรอิสระ (Independent Variable)
Y = np.array([1250, 1380, 1425, 1425, 1450, 1300, 1400, 1510, 1575, 1650])  # ตัวแปรตาม (Dependent Variable)

# สร้างโมเดล Simple Linear Regression
model = LinearRegression()
model.fit(X, Y)  # ฝึกสอนโมเดล (Training the model)

# คำนวณค่าคาดการณ์ (Prediction)
Y_pred = model.predict(X)

# แสดงค่าค่าสัมประสิทธิ์และค่า Intercept
print(f"Slope (ค่าสัมประสิทธิ์ของ X): {model.coef_[0]:.2f}")
print(f"Intercept (จุดตัดแกน Y): {model.intercept_:.2f}")

# คำนวณค่าความถูกต้องของโมเดล
mse = mean_squared_error(Y, Y_pred)  # คำนวณ Mean Squared Error
r2 = r2_score(Y, Y_pred)  # คำนวณค่า R-squared
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"R-squared (ค่าความแม่นยำของโมเดล): {r2}")

# แสดงสมการการถดถอยโดยประมาณ
slope = model.coef_[0]
intercept = model.intercept_
print(f"สมการการถดถอยโดยประมาณ: Y = {intercept:.0f}+{slope:.1f}X")

# การสร้างกราฟแสดงผล
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.flatten(), y=Y, color='blue', label='Actual Data')  # จุดข้อมูลจริง
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')  # เส้นสมการเชิงเส้น
plt.xlabel("Advertising Expenditure")
plt.ylabel("Weekly Sales")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()