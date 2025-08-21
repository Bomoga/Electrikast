from pathlib import Path
import json
import numpy as np
import joblib

class EnergyPredictor:
    def __init__(self, model_path: str, feature_order_path: str):
        self.model_path = Path(model_path)
        self.feature_order_path = Path(feature_order_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not self.feature_order_path.exists():
            raise FileNotFoundError(f"Feature order file not found at {self.feature_order_path}")
        
        self.model = joblib.load(self.model_path)
        self.feature_order_path = joblib.load(self.feature_order_path.read_text)

    def validate(self, inputs: dict):
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be provided as a dictionary of {feature_name: value}")
        
        missing = [f for f in self.features if f not in inputs]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        return np.array([[inputs[f] for f in self.features]], dtype = float)
    
    def predict(self, inputs: dict) -> float:
        x = self.validate(inputs)
        y = self.model.predict(x)
        return float(y[0])