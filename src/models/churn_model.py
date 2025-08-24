from abc import ABC, abstractmethod
import xgboost as xgb
from typing import Any  

class ChurnModel(ABC):
    """Clase base abstracta para todos los modelos de predicción de churn."""

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

class XGBoostModel(ChurnModel):
    """Implementación concreta del modelo XGBoost."""

    def __init__(self, params: dict = None):
        self.model = xgb.XGBClassifier(**(params or {}))

    def train(self, X_train, y_train):
        print("Entrenando el modelo XGBoost...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)