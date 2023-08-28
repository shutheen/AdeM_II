import os
import pandas as pd
import numpy as np
from sklearn import joblib
from sklearn.ensemble import VotingRegressor

class ModelInferencePipeline:
    def __init__(self, model_folder):
        self.model_folder = model_folder

    def load_models(self):
        models = {}
        
        # Cargar modelos individuales
        rf_model = joblib.load(os.path.join(self.model_folder, "random_forest_model.joblib"))
        svm_model = joblib.load(os.path.join(self.model_folder, "svm_model.joblib"))
        nn_model = joblib.load(os.path.join(self.model_folder, "neural_network_model.joblib"))
        linear_model = joblib.load(os.path.join(self.model_folder, "linear_regression_model.joblib"))
        
        models["rf"] = rf_model
        models["svm"] = svm_model
        models["nn"] = nn_model
        models["linear"] = linear_model
        
        # Cargar modelo de votación
        voting_model = joblib.load(os.path.join(self.model_folder, "trained_model.pkl"))
        models["voting"] = voting_model
        
        return models
    
    def make_predictions(self, models, data):
        predictions = {}
        
        # Realizar predicciones individuales
        for name, model in models.items():
            predictions[name] = model.predict(data)
        
        # Realizar predicciones con el modelo de votación
        voting_predictions = np.column_stack([predictions[name] for name in ["rf", "svm", "nn", "linear"]])
        final_predictions = models["voting"].predict(voting_predictions)
        predictions["voting"] = final_predictions
        
        return predictions


if __name__ == "__main__":
    model_folder = os.path.join("models")
    inference_pipeline = ModelInferencePipeline(model_folder)
    
    # Cargar modelos
    loaded_models = inference_pipeline.load_models()
    
    # Cargar datos para hacer predicciones
    test_data_path = "H:\\Users\\Martin\\Desktop\\Especialización  IA\\1 - Bimestres y Cursos\\3B\\3. Aprendizaje de Máquina II\\TP - MartinCaceres_AMq2\MartinCaceres_AMq2_TPFinal\\Notebook\\test_final.csv"
    df_test = pd.read_csv(test_data_path)
    X_test_imputed = df_test
    
    # Hacer predicciones
    predictions = inference_pipeline.make_predictions(loaded_models, X_test_imputed)
    
    # Guardar predicciones en un archivo CSV
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv("predictions.csv", index=False)
