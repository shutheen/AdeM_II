"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: Realizamos la transformación de datos en un dataSet y guardamos el resultado en un archivo CSV.
AUTOR:
FECHA:
"""

# Imports
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import os

class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:

        data_path = os.path.join(self.input_path, "Notebook", "train_final.csv")
        
        pandas_df = pd.read_csv(data_path)

        return pandas_df

    
    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza la transformación de datos en el DataFrame de entrada.

        :param df: DataFrame de entrada
        :type df: pd.DataFrame
        :return df_transformed: DataFrame transformado
        :rtype: pd.DataFrame
        """

        # Obtener las columnas categóricas del conjunto de datos
        categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

        # Realizar la codificación one-hot de las columnas categóricas
        encoder = OneHotEncoder(drop="first", sparse=False)
        df_encoded = encoder.fit_transform(df[categorical_columns])

        # Crear un nuevo DataFrame con las columnas codificadas
        df_encoded_df = pd.DataFrame(df_encoded, columns=encoder.get_feature_names(categorical_columns))

        # Concatenar las columnas codificadas con las columnas numéricas originales
        df_processed = pd.concat([df.drop(categorical_columns, axis=1), df_encoded_df], axis=1)

        # Crear un imputador y ajustarlo al conjunto de procesado
        imputer = SimpleImputer(strategy="mean")
        df_imputed = imputer.fit_transform(df_processed)

        # Crear un DataFrame con los datos imputados
        df_transformed = pd.DataFrame(df_imputed, columns=df_processed.columns)

        return df_transformed
    
    

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Escribe los datos preparados en el archivo de salida.
    
        :param transformed_dataframe: DataFrame con los datos transformados
        :type transformed_dataframe: pd.DataFrame
        """
    
        # Guardar el DataFrame en un archivo CSV en la ruta de salida
        transformed_dataframe.to_csv(self.output_path, index=False)
    
        return None


    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    project_folder = os.path.join("H:\\Users\\Martin\\Desktop\\Especialización  IA\\1 - Bimestres y Cursos\\3B\\3. Aprendizaje de Máquina II\\TP - MartinCaceres_AMq2\\MartinCaceres_AMq2_TPFinal")
    
    input_path = os.path.join(project_folder, "data", "Train_BigMart.csv")
    output_path = os.path.join(project_folder, "data", "train_final.csv")
    
    FeatureEngineeringPipeline(input_path=input_path, output_path=output_path).run()
