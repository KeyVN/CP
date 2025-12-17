import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier # Thư viện thay thế hoàn hảo

# 1. Tải dữ liệu từ folder data/
try:
    df = pd.read_csv('data/Dataset Project 404.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file CSV trong thư mục data/")
    exit()

# 2. Làm sạch dữ liệu
df = df.dropna(subset=['Job profession'])
df['Job profession'] = df['Job profession'].str.strip()

# 8 loại trí thông minh Gardner
features = ['Linguistic', 'Musical', 'Bodily', 'Logical - Mathematical', 
            'Spatial-Visualization', 'Interpersonal', 'Intrapersonal', 'Naturalist']
X = df[features]
y = df['Job profession']

# 3. Tiền xử lý
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Huấn luyện mô hình (Gradient Boosting)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

print("Đang huấn luyện mô hình... Vui lòng đợi trong giây lát.")
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 5. Lưu kết quả vào folder models/
if not os.path.exists('models'): os.makedirs('models')

with open('models/career_model_gb.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Đã lưu mô hình với tên: career_model_gb.pkl")
print(f"Huấn luyện thành công!")
print(f"Độ chính xác: {model.score(X_test, y_test)*100:.2f}%")