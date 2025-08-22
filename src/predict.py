from pathlib import Path
import json
import numpy as np
import joblib
import json

class EnergyPredictor:
    def __init__(self, model_path: str, feature_order_path: str):
        self.model_path = Path(model_path)
        self.feature_order_path = Path(feature_order_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not self.feature_order_path.exists():
            raise FileNotFoundError(f"Feature order file not found at {self.feature_order_path}")
        
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError("Failed to load model at {self.model_path}: {e}") from e
        
        try:
            raw = self.feature_order_path.read_text(encoding="utf-8")
            self.features = json.loads(raw)
        except Exception as e:
            raise RuntimeError("Failed to parse JSON from {self.feature_order_path}: {e}") from e
        
        if not isinstance(self.features, list) or not all(isinstance(f, str) for f in self.features):
            raise ValueError(f"feature_order.json must be a JSON array of strings. Caught: {type(self.features)}")
        if len(self.features == 0):
            raise ValueError("feature_order.json is empty. At minimum one feature is expected.")
        
    def validate(self, inputs: dict):
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be provided as a dictionary of {feature_name: value}")
        
        missing = [f for f in self.features if f not in inputs]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        try:
            row = [float(inputs[i] for i in self.features)]
        except Exception as e:
            raise ValueError("All values must be numeric. Failed to convert one or more values: {e}")
        
        return np.array([[row]], dtype = float)
    
    def predict(self, inputs: dict) -> float:
        x = self.validate(inputs)
        y = self.model.predict(x)
        return float(y[0])