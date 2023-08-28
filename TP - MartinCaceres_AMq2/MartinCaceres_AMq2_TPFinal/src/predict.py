"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pandas as pd  # Agrega la importación de pandas u otras bibliotecas necesarias
from sklearn.externals import joblib  # Agrega la importación correcta para cargar el modelo

class MakePredictionPipeline(object):
    
    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
                
                
    def load_data(self) -> pd.DataFrame:
        """
        Carga los datos de entrada para hacer predicciones.

        :return data: DataFrame con los datos de entrada
        :rtype: pd.DataFrame
        """
        data = pd.read_csv(self.input_path)  # Carga los datos desde el archivo de entrada
        return data

    def load_model(self) -> None:
        """
        Carga el modelo entrenado para hacer predicciones.
        """    
        self.model = joblib.load(self.model_path)  # Carga el modelo utilizando joblib u otra biblioteca
        
        return None


    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza predicciones en los datos de entrada utilizando el modelo cargado.

        :param data: DataFrame con los datos de entrada
        :type data: pd.DataFrame
        :return predictions: DataFrame con las predicciones realizadas
        :rtype: pd.DataFrame
        """
        predictions = self.model.predict(data)  # Utiliza el modelo para hacer predicciones en los datos
        return predictions


    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Escribe las predicciones en el archivo de salida.

        :param predicted_data: DataFrame con las predicciones
        :type predicted_data: pd.DataFrame
        """
        predicted_data.to_csv(self.output_path, index=False)  # Guarda las predicciones en un archivo CSV
        
        return None


    def run(self):

        data = self.load_data()
        self.load_model()
        predictions = self.make_predictions(data)
        self.write_predictions(predictions)


if __name__ == "__main__":
    
    # Aquí podrías configurar el entorno o bibliotecas necesarias, como Spark en tu caso
    # spark = Spark()  # Esto es un ejemplo, ajusta según tus necesidades
    
    pipeline = MakePredictionPipeline(input_path='Ruta/De/Donde/Voy/A/Leer/Mis/Datos',
                                      output_path='Ruta/Donde/Voy/A/Escribir/Mis/Predicciones',
                                      model_path='Ruta/De/Donde/Voy/A/Leer/Mi/Modelo.pkl')  # Asegúrate de tener la extensión correcta del archivo del modelo
    pipeline.run()
