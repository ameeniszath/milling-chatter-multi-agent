import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import uvicorn

app = FastAPI()

def load_and_train():
    # 1. Load Data (Ensuring we use your specific file names)
    df_s = pd.read_excel('stable.xlsx')
    df_s = df_s[['rpm', 'doc']].dropna()
    df_s['Target'] = 'Stable'
    
    df_u = pd.read_excel('unstable.xlsx')
    df_u = df_u[['rpm', 'doc']].dropna()
    df_u['Target'] = 'Unstable'
    
    # 2. Combine and prepare features
    df = pd.concat([df_s, df_u], axis=0)
    X = df[['rpm', 'doc']].values
    y = df['Target'].values
    
    # 3. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train SVM with Probability enabled
    # We use probability=True so we can call predict_proba later
    model = SVC(kernel='rbf', C=100, gamma='auto', probability=True)
    model.fit(X_scaled, y)
    
    return model, scaler, model.classes_

# Initialize
model, scaler, class_labels = load_and_train()

class StabilityInput(BaseModel):
    rpm: float
    doc: float  # User now inputs this in METERS (e.g., 0.0015)

@app.post("/predict")
async def predict(data: StabilityInput):
    # No conversion needed as user provides meters directly
    input_data = np.array([[data.rpm, data.doc]])
    input_scaled = scaler.transform(input_data)
    
    # Get Class Prediction
    prediction = model.predict(input_scaled)[0]
    
    # Get Probabilities
    probs = model.predict_proba(input_scaled)[0]
    prob_dict = {label: round(float(prob) * 100, 2) for label, prob in zip(class_labels, probs)}
    
    return {
        "input_rpm": data.rpm,
        "input_doc_m": data.doc,
        "status": prediction,
        "confidence_metrics": prob_dict,
        "probability_stable": f"{prob_dict.get('Stable')}%",
        "probability_unstable": f"{prob_dict.get('Unstable')}%"
    }

if __name__ == "__main__":
    # Ensure this port matches your n8n HTTP Request node
    uvicorn.run(app, host="0.0.0.0", port=8005)