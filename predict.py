import pickle
import numpy as np
import pandas as pd # Thêm pandas để đặt tên cột

def predict_career(scores):
    # Danh sách tên cột giống hệt lúc huấn luyện
    feature_names = ['Linguistic', 'Musical', 'Bodily', 'Logical - Mathematical', 
                     'Spatial-Visualization', 'Interpersonal', 'Intrapersonal', 'Naturalist']
    
    try:
        with open('models/career_model_lgbm.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)

        # Tạo DataFrame thay vì numpy array để có tên cột
        input_df = pd.DataFrame([scores], columns=feature_names)
        
        # Chuẩn hóa và dự đoán
        input_scaled = scaler.transform(input_df)
        prediction_numeric = model.predict(input_scaled)
        career_name = le.inverse_transform(prediction_numeric)
        
        return career_name[0]
    except Exception as e:
        return f"Lỗi: {str(e)}"

if __name__ == "__main__":
    print("--- HỆ THỐNG GỢI Ý NGHỀ NGHIỆP THÔNG MINH ---")
    # Nhập điểm trực tiếp để kiểm tra các trường hợp khác nhau
    try:
        user_input = []
        labels = ['Ngôn ngữ', 'Âm nhạc', 'Vận động', 'Logic/Toán', 'Không gian', 'Giao tiếp', 'Nội tâm', 'Tự nhiên']
        for label in labels:
            score = float(input(f"Nhập điểm {label} (5-20): "))
            user_input.append(score)
            
        result = predict_career(user_input)
        print("\n" + "="*40)
        print(f" KẾT QUẢ GỢI Ý: {result.upper()} ")
        print("="*40)
    except ValueError:
        print("Vui lòng chỉ nhập số!")