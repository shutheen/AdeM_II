# Imports
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import joblib
import os

class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        # Lee los datos desde el archivo de entrada y devuelve un DataFrame
        pandas_df = pd.read_csv(self.input_path)
        return pandas_df
    
    def run(self):
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)

def model_training(self, df: pd.DataFrame) -> VotingRegressor:
    """
    Entrena modelos individuales y crea un Voting Regressor con todos los modelos.
    """
    X_train = df.drop("Item_Outlet_Sales", axis=1)
    y_train = df["Item_Outlet_Sales"]

    rf_model = joblib.load(os.path.join(self.model_path, "random_forest_model.joblib"))
    svm_model = joblib.load(os.path.join(self.model_path, "svm_model.joblib"))
    nn_model = joblib.load(os.path.join(self.model_path, "neural_network_model.joblib"))
    simple_regressor_model = joblib.load(os.path.join(self.model_path, "linear_regression_model.joblib"))

    # Crear un Voting Regressor con hard voting
    voting_model = VotingRegressor(
        estimators=[
            ("rf", rf_model),
            ("svm", svm_model),
            ("nn", nn_model),
            ("simple_regressor", simple_regressor_model)
        ]
    )

    return voting_model



    def model_dump(self, model_trained) -> None:
        joblib.dump(model_trained, os.path.join(self.model_path, "trained_model.pkl"))
        return None

    def run(self):
        df = self.read_data()
        model_trained = self.model_training(df)
    
        # Entrenar el modelo de votación
        voting_model = model_trained.fit(X_train_imputed, y_train)
    
        # Guardar el modelo de votación
        model_path_voting = os.path.join(self.model_path, "voting_model.joblib")
        joblib.dump(voting_model, model_path_voting)
    
        return voting_model



if __name__ == "__main__":
    notebook_path = os.path.abspath("train.py")
    project_folder = os.path.dirname(os.path.dirname(notebook_path))
    
    model_path = os.path.join(project_folder, "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    input_path = os.path.join(project_folder, "data", "Train_BigMart_transformed.csv")
    
    ModelTrainingPipeline(input_path=input_path, model_path=model_path).run()
