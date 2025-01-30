# Example 3 => หน้า 25
# ตัวอย่าง: ยอดค้าปลีกและพื้นที่ เป็นเรื่องปกติในการดําเนินธุรกิจค้าปลีกที่จะประเมินประสิทธิภาพของร้านค้า
# ส่วนหนึ่งในแง่ของยอดขายประจําปีที่สัมพันธ์กับพื้นที่ (ตารางฟุต) เราอาจคาดหวังว่ายอดขายจะเพิ่มขึ้น
# เป็นเส้นตรงเมื่อร้านค้ามีขนาดใหญ่ขึ้น แบบจําลองการถดถอยสําหรับประชากรร้านค้ากล่าวว่า 
# SALES = B0 + B1 x AREA + e
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ข้อมูล Floor Space (X) และ Sales (Y)
X = np.array([1500, 2000, 2500, 1800, 2200, 2700, 3000, 3200, 3500, 4000]).reshape(-1, 1)  # พื้นที่ร้านค้า (ตารางฟุต)
Y = np.array([300, 410, 520, 380, 450, 570, 600, 650, 700, 800])  # ยอดขายประจำปี (พันดอลลาร์)

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
print(f"สมการการถดถอยโดยประมาณ: SALES = {intercept:.2f} + {slope:.2f} * AREA")

# คำนวณค่าความถูกต้องของโมเดล
mse = mean_squared_error(Y, Y_pred)  # คำนวณ Mean Squared Error
r2 = r2_score(Y, Y_pred)  # คำนวณค่า R-squared
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (ค่าความแม่นยำของโมเดล): {r2:.2f}")

# การสร้างกราฟแสดงผล
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.flatten(), y=Y, color='blue', label='Actual Data')  # จุดข้อมูลจริง
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')  # เส้นสมการเชิงเส้น
plt.xlabel("Floor Space (sq. ft.)")
plt.ylabel("Annual Sales (000s USD)")
plt.title("Simple Linear Regression: Floor Space vs. Sales")
plt.legend()
plt.show()