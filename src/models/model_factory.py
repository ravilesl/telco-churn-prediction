from src.models.churn_model import XGBoostModel
from sklearn.model_selection import GridSearchCV

class ModelFactory:
    """Clase que crea instancias de modelos y las prepara para la optimización."""


    @staticmethod
    def create_model_base(model_name: str):
        """
        Crea solo la instancia base de un modelo.
        """
        if model_name == 'xgboost':
            return XGBoostModel().model
        
        raise ValueError(f"Modelo '{model_name}' no soportado.")
    
    @staticmethod
    def create_grid_search(model_name: str, search_params: dict):
        """
        Crea un objeto GridSearchCV para la búsqueda de hiperparámetros.
        (Este método se mantiene si lo necesitas en otra parte del código)
        """
        base_model = ModelFactory.create_model_base(model_name)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=search_params['params_grid'],
            cv=search_params['cv'],
            scoring=search_params['scoring'],
            n_jobs=-1,
            verbose=1
        )
        return grid_search