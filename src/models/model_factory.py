# src/models/model_factory.py
# Clase para crear instancias de modelos y preparar GridSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from imblearn.pipeline import Pipeline

class ModelFactory:
    """Clase que crea instancias de modelos y sus parámetros para la optimización."""

    @staticmethod
    def create_model_and_params(model_name: str) -> tuple:
        """
        Crea la instancia base de un modelo y sus parámetros de búsqueda.
        """
        models_and_params = {
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42),
                'params': {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7, 9]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'model__C': [0.01, 0.1, 1, 10, 100],
                    'model__solver': ['liblinear', 'lbfgs']
                }
            },
            'svc': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'model__C': [0.1, 1, 10, 100],
                    'model__kernel': ['rbf', 'linear']
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'model__n_neighbors': [3, 5, 7, 9, 11, 13]
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'model__max_depth': [3, 5, 7, 9, None],
                    'model__min_samples_split': [2, 5, 10, 15]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'model__n_estimators': [100, 200, 300, 400],
                    'model__max_depth': [5, 10, 15, None]
                }
            },
            'adaboost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'model__n_estimators': [50, 100, 200, 300],
                    'model__learning_rate': [0.01, 0.1, 0.5, 1.0]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.05, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42),
                'params': {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'model__num_leaves': [31, 62, 127]
                }
            }
        }

        if model_name in models_and_params:
            config = models_and_params[model_name]
            return config['model'], config['params']
        
        raise ValueError(f"Modelo '{model_name}' no soportado.")