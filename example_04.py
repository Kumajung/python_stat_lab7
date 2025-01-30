# Example4 => 
# ตัวอย่าง: การศึกษาความสัมพันธ์ระหว่างเวลาการออกกำลังกายกับปริมาณแคลอรี่ที่เผาผลาญ
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ข้อมูลระยะเวลาการออกกำลังกาย (X) และปริมาณแคลอรี่ที่เผาผลาญ (Y)
X = np.array([35, 50, 65, 45, 40, 55, 70, 60, 75, 80]).reshape(-1, 1)  # เวลาการออกกำลังกาย (นาที)
Y = np.array([2100, 2500, 2900, 2300, 2200, 2700, 3100, 2800, 3300, 3500])  # แคลอรี่ที่เผาผลาญ

# สร้างโมเดล Simple Linear Regression
model = LinearRegression()
model.fit(X, Y)  # ฝึกสอนโมเดล

# คำนวณค่าคาดการณ์ (Prediction)
Y_pred = model.predict(X)

# แสดงค่าค่าสัมประสิทธิ์และค่า Intercept
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope (B1 - อัตราการเปลี่ยนแปลง): {slope:.2f}")
print(f"Intercept (B0 - จุดตัดแกน Y): {intercept:.2f}")
print(f"สมการการถดถอยโดยประมาณ: Calories Burned = {intercept:.2f} + {slope:.2f} * Exercise Time")

# คำนวณค่า Residuals (ความคลาดเคลื่อน)
residuals = Y - Y_pred

# คำนวณค่าความถูกต้องของโมเดล
mse = mean_squared_error(Y, Y_pred)  # Mean Squared Error
r2 = r2_score(Y, Y_pred)  # R-squared
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (ค่าความแม่นยำของโมเดล): {r2:.2f}")

# การสร้างกราฟแสดงผล
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.flatten(), y=Y, color='blue', label='Actual Data')  # จุดข้อมูลจริง
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')  # เส้นสมการเชิงเส้น
plt.xlabel("Exercise Time (minutes)")
plt.ylabel("Calories Burned")
plt.title("Simple Linear Regression: Exercise Time vs. Calories Burned")
plt.legend()
plt.show()
