import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from catboost import CatBoostRegressor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. ELEMENTAL DATA (Required for Physics Features)
elements_data = {
    'H': [1.00, 2.20], 'Li': [6.94, 0.98], 'C': [12.01, 2.55], 'N': [14.01, 3.04],
    'O': [16.00, 3.44], 'F': [19.00, 3.98], 'Mg': [24.31, 1.31], 'Al': [26.98, 1.61],
    'Si': [28.09, 1.90], 'P': [30.97, 2.19], 'S': [32.06, 2.58], 'Ti': [47.87, 1.54],
    'V': [50.94, 1.63], 'Fe': [55.85, 1.83], 'Ni': [58.69, 1.91], 'Cu': [63.55, 1.90],
    'Zn': [65.38, 1.65], 'Ga': [69.72, 1.81], 'Ge': [72.63, 2.01], 'As': [74.92, 2.18],
    'Se': [78.96, 2.55], 'Mo': [95.95, 2.16], 'Cd': [112.41, 1.69], 'In': [114.82, 1.78],
    'Sn': [118.71, 1.96], 'Te': [127.60, 2.10], 'Cs': [132.91, 0.79], 'Ba': [137.33, 0.89],
    'W': [183.84, 2.36], 'Pt': [195.08, 2.28], 'Au': [196.97, 2.54], 'Pb': [207.2, 2.33],
    'Sr': [87.62, 0.95], 'Y': [88.91, 1.22], 'Zr': [91.22, 1.33]
}

def extract_features(formula):
    parts = re.findall(r'([A-Z][a-z]*)(\d*)', str(formula))
    w, ens = [], []
    for el, c in parts:
        c = int(c) if c else 1
        if el in elements_data:
            w.extend([elements_data[el][0]] * c)
            ens.extend([elements_data[el][1]] * c)
    if not ens: return 50.0, 2.0, 0.0, 0.0
    return np.mean(w), np.mean(ens), np.max(ens) - np.min(ens), np.std(ens)

# 2. LOAD 4 PRECISION MODELS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models = [CatBoostRegressor() for _ in range(4)]
for i in range(4):
    models[i].load_model(os.path.join(BASE_DIR, f"model_v3_{i}.cbm"))

class NanoInput(BaseModel):
    formula: str
    size_nm: float
    crystal_structure: str
    material_class: str
    shape: str

@app.post("/predict")
def predict(data: NanoInput):
    # Map to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Feature Engineering (Must match Training Script exactly)
    w, ens, en_diff, en_std = extract_features(input_df['formula'][0])
    input_df['avg_w'], input_df['avg_en'] = w, ens
    input_df['en_diff'], input_df['en_std'] = en_diff, en_std
    input_df['inv_size'] = 1.0 / (input_df['size_nm'] + 1e-5)
    
    # Define columns to match training order
    model_features = ['avg_w', 'avg_en', 'en_diff', 'en_std', 'crystal_structure', 'material_class', 'size_nm', 'inv_size', 'shape']
    X = input_df[model_features].copy()
    
    # Cast categories to string to avoid 2.0 error
    cat_cols = ['crystal_structure', 'material_class', 'shape']
    for col in cat_cols:
        X[col] = X[col].astype(str)
        
    # Get predictions from all 4 models
    preds = [model.predict(X)[0] for model in models]
    
    return {
        "bandgap": f"{preds[0]:.2f} eV",
        "density": f"{preds[1]:.2f} g/cm³",
        "formation_energy": f"{preds[2]:.2f} eV/atom",
        "specific_heat": f"{preds[3]:.4f} J/gK"
    }
