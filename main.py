from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import uuid
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# โหลดโมเดล
model_diag = joblib.load('modeldiag/rf_model.joblib')
scaler_diag = joblib.load('modeldiag/scaler.joblib')
columns_diag = joblib.load('modeldiag/columns_to_use.joblib')

model_brca = joblib.load('modelbrca/svm_model.joblib')
scaler_brca = joblib.load('modelbrca/scaler.joblib')
columns_brca = joblib.load('modelbrca/columns_to_use.joblib')

# ตั้งค่า MongoDB
client = MongoClient("mongodb+srv://busduan:busduan123@cluster0.ui6tr.mongodb.net/")
personal_db = client["PersonalInfoDB"]
brca_db = client["BrcaDB"]
diag_db = client["DiagDB"]
formodel_db = client["formodelDB"]
screening_db = client["ScreeningDB"] 

app = FastAPI(
    title="Breast Cancer Risk Assessment API",
    description="API สำหรับประเมินความเสี่ยงมะเร็งเต้านมโดยใช้โมเดล Machine Learning",
    version="1.0.0"
)

# อนุญาต origin ที่ต้องการ
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    FORM_TYPE: str  # 'form1' หรือ 'form2'
    BRCA: Optional[str] = None
    BMI_GROUP: dict[str, float]  # dict ที่ประกอบด้วย 'weight' และ 'height'
    AGE_GROUP: int  # อายุจริงของผู้ใช้
    PROVINCE_GROUP: list[str]  # รายการจังหวัด
    consent: bool = None

class ScreeningData(BaseModel):
    fullName: str
    idCard: str
    phoneNumber: str
    birthDate: str
    address: str
    province: str
    weight: float
    height: float
    BMI: float
    BRCA: str
    result: str
    probability: float
    consent: bool

def bmi_group(weight: float, height: float) -> str:
    """คำนวณค่า BMI และจัดกลุ่มเป็น '1:Normal' หรือ '2:Abnormal'"""
    if height == 0:
        return "ไม่สามารถคำนวณได้"
    bmi = weight / (height / 100) ** 2
    return '1:Normal' if 19 <= bmi <= 24.9 else '2:Abnormal'

def age_group(age: int) -> str:
    """จัดกลุ่มอายุ"""
    if 25 <= age <= 34:
        return '1:<35'
    elif 35 <= age <= 39:
        return '2:<40'
    elif 40 <= age <= 44:
        return '3:<45'
    elif 45 <= age <= 49:
        return '4:<50'
    else:
        return '5:<90'

def province_group(province: str) -> int:
    """จัดกลุ่มจังหวัด"""
    if province in ['Yala', 'Pattani', 'Narathiwat']:
        return 1
    elif province in ['Songkhla', 'Satun', 'Trang', 'Phatthalung']:
        return 2
    else:
        return 3

def brca(brca: str) -> str:
    """แปลงค่า BRCA"""
    return '1:N' if brca == 'Negative' else '2:P'

def risk_level(prob):
    if prob <= 0.20:
        return 'เสี่ยงต่ำมาก', round(prob, 2)
    elif prob <= 0.40:
        return 'เสี่ยงต่ำ', round(prob, 2)
    elif prob <= 0.60:
        return 'เสี่ยงปานกลาง', round(prob, 2)
    elif prob <= 0.80:
        return 'เสี่ยงสูง', round(prob, 2)
    else:
        return 'เสี่ยงสูงมาก', round(prob, 2)

@app.post("/predict/")
def predict(data: InputData):
    try:
        
        weight = data.BMI_GROUP.get("weight", 0)
        height = data.BMI_GROUP.get("height", 0)

        # หากน้ำหนักหรือส่วนสูงเป็นค่าติดลบหรือ 0 ให้คืนค่า HTTPException
        if weight <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="น้ำหนักและส่วนสูงต้องมีค่ามากกว่า 0")
        
        
        user_id = str(uuid.uuid4())
        bmi_value = weight / (height / 100) ** 2

        # เก็บข้อมูลส่วนตัว
        personal_info = {
            "_id": user_id,
            "weight": weight,
            "height": height,
            "bmi": round(bmi_value, 2),
            "age": data.AGE_GROUP,
            "province": data.PROVINCE_GROUP[0],
            "consent": data.consent,
            "timestamp": datetime.now()
        }
        personal_db.personal_info.insert_one(personal_info)

        if data.FORM_TYPE == "form1":
            age_group_value = age_group(data.AGE_GROUP)
            bmi_group_value = bmi_group(weight, height)
            province_group_value = province_group(data.PROVINCE_GROUP[0])
            brca_result = brca(data.BRCA)

            brca_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "brca_value": 'Positive' if brca_result == '2:P' else 'Negative',
                "source": "manual",
                "timestamp": datetime.now()
            }
            brca_db.brca_info.insert_one(brca_data)

            formodel_data = {
                "user_id": user_id,
                "bmi_group": bmi_group_value,
                "age_group": age_group_value,
                "province_group": province_group_value,
                "brca_value": brca_result,
                "timestamp": datetime.now()
            }
            formodel_db.formodel_info.insert_one(formodel_data)

            df_diag = pd.DataFrame([{
                'BRCA': brca_result,
                'BMI_GROUP': bmi_group_value,
                'AGE_GROUP': age_group_value,
                'PROVINCE_GROUP': province_group_value,
            }])
            print("011", df_diag.to_string())
            df_diag = pd.get_dummies(df_diag).reindex(columns=columns_diag, fill_value=0)
            print("012", df_diag.to_string())
            df_diag_scaled = scaler_diag.transform(df_diag)
            diag_proba = model_diag.predict_proba(df_diag_scaled)[:, 1]
            diag_risk_level, prob = risk_level(diag_proba[0])

            diag_data = {
                "user_id": user_id,
                "diag_prediction": diag_risk_level,
                "diag_probability": round(prob * 100, 2),
                "timestamp": datetime.now()
            }
            diag_db.diag_results.insert_one(diag_data)

            return {
                "prediction_diag": diag_risk_level,
                "probability_diag": round(prob * 100, 2)
            }

        elif data.FORM_TYPE == "form2":
            age_group_value = age_group(data.AGE_GROUP)
            bmi_group_value = bmi_group(weight, height)
            province_group_value = province_group(data.PROVINCE_GROUP[0])

            df_brca = pd.DataFrame([{
                'BMI_GROUP': bmi_group_value,
                'AGE_GROUP': age_group_value,
                'PROVINCE_GROUP': province_group_value,
            }])
            print("013", df_brca.to_string())
            df_brca = pd.get_dummies(df_brca).reindex(columns=columns_brca, fill_value=0)
            print("014", df_brca.to_string())
            df_brca_scaled = scaler_brca.transform(df_brca)
            brca_proba = model_brca.predict_proba(df_brca_scaled)[:, 1]

            def brca_level(prob):
                return ('Negative', prob) if prob <= 0.5 else ('Positive', prob)
                
            brca_level_result, brca_prob = brca_level(brca_proba[0])

            brca_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "brca_value": brca_level_result,
                "source": "prediction",
                "probability": round(brca_prob * 100, 2),
                "timestamp": datetime.now()
            }
            brca_db.brca_info.insert_one(brca_data)

            brca_value = "2:P" if brca_level_result == "Positive" else "1:N"
            formodel_data = {
                "user_id": user_id,
                "bmi_group": bmi_group_value,
                "age_group": age_group_value,
                "province_group": province_group_value,
                "brca_value": brca_value,
                "timestamp": datetime.now()
            }
            formodel_db.formodel_info.insert_one(formodel_data)

            df_diag = pd.DataFrame([{
                'BRCA': brca_value,
                'BMI_GROUP': bmi_group_value,
                'AGE_GROUP': age_group_value,
                'PROVINCE_GROUP': province_group_value,
            }])
            print("015", df_diag.to_string())
            df_diag = pd.get_dummies(df_diag).reindex(columns=columns_diag, fill_value=0)
            print("016", df_diag.to_string())
            df_diag_scaled = scaler_diag.transform(df_diag)
            diag_proba = model_diag.predict_proba(df_diag_scaled)[:, 1]
            diag_risk_level, prob = risk_level(diag_proba[0])

            diag_data = {
                "user_id": user_id,
                "diag_prediction": diag_risk_level,
                "diag_probability": round(prob * 100, 2),
                "timestamp": datetime.now()
            }
            diag_db.diag_results.insert_one(diag_data)

            return {
                "prediction_brca": brca_level_result,
                "probability_brca": round(brca_prob * 100, 2),
                "prediction_diag": diag_risk_level,
                "probability_diag": round(prob * 100, 2)
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/screening/")
def screening(data: ScreeningData):
    try:
        # คำนวณค่า BMI หากจำเป็น
        if data.weight and data.height:
            data.BMI = data.weight / (data.height / 100) ** 2
        
        # สร้างข้อมูลที่ต้องการบันทึก
        screening_data = {
            "fullName": data.fullName,
            "idCard": data.idCard,
            "phoneNumber": data.phoneNumber,
            "birthDate": data.birthDate,
            "address": data.address,
            "province": data.province,
            "weight": data.weight,
            "height": data.height,
            "BMI": round(data.BMI, 2),
            "BRCA": data.BRCA,
            "result": data.result,
            "probability": data.probability,
            "consent": data.consent,
            "timestamp": datetime.now()
        }
        
        # บันทึกข้อมูลลงใน ScreeningDB
        screening_db.screening_info.insert_one(screening_data)

        return {"message": "Screening data saved successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
