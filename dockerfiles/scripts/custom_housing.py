from inspect import signature
from sys import version_info
import os
import io
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import dvc.api

from custom_data_transformers.utils import HouseColumnTransformerFunc

backend_uri = os.environ['MLFLOW_TRACKING_URI']
artifact_uri = os.environ['MLFLOW_ARTIFACT_STORE']
mlflow.set_tracking_uri(backend_uri)

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
pipeline_path = "pipe.pkl"
model_path = "model.pkl"

artifacts = {
    "pipeline": pipeline_path,
    "model": model_path
}

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pipeline = joblib.load(context.artifacts["pipeline"])
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        model_input.columns = ["housing_median_age", "total_rooms", "total_bedrooms", 
                   "population", "households", "median_income", "ocean_proximity"]
        input_matrix = self.pipeline.transform(model_input)
        return self.model.predict(input_matrix)


def fetch_data_from_s3(path, repo, version):
    data = dvc.api.read(
            path = path,
            repo = repo,
            rev = version
        )
    return pd.read_csv(io.StringIO(data), sep=',')


def fetch_data_from_fs(url):
    return pd.read_csv(url, sep=',')

def get_pipeline(catCols, numCols, txCat = True):
    customColTrans = FunctionTransformer(HouseColumnTransformerFunc, 
                                     kw_args={"add_bedrooms_per_room": False})

    num_pipeline = Pipeline([
                        ("Imputer", SimpleImputer()),
                        ("Scaler", StandardScaler()),
                        ("Transform", customColTrans)
                    ])

    if txCat:
        full_pipeline = ColumnTransformer([
                            ("Numerical_Pipeline", num_pipeline, numCols),
                            ("OneHot", OneHotEncoder(), catCols)
                        ])
    else:
        full_pipeline = ColumnTransformer([
                            ("Numerical_Pipeline", num_pipeline, numCols)
                        ])
    return full_pipeline

def get_trained_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model

def get_model_signature(numCols, catCols):
    input_schema = []
    for col in numCols:
        input_schema.append(ColSpec("double", col))
    for col in catCols:
        input_schema.append(ColSpec("string", col))
    input_schema = Schema(input_schema)
    output_schema = Schema([ColSpec('double')])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    return signature

def log_experiment(experiment_name, x, y, model, data_version, sign, feature_names):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment.name)

    mlflow_pyfunc_model_path = experiment_name
    with mlflow.start_run():
        mlflow.log_param('data_url', data_url)
        mlflow.log_param('data_version', data_version)
        mlflow.log_param('input_rows', x.shape[0])
        mlflow.log_param('input_columns', x.shape[1])
        
        predictions = model.predict(x)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        mlflow.log_metric("rmse", rmse)
        mlflow.pyfunc.log_model(mlflow_pyfunc_model_path, 
                                python_model=ModelWrapper(), 
                                artifacts=artifacts,
                                signature=sign,
                                registered_model_name='regres')
        mlflow.shap.log_explanation(model.predict, 
                                pd.DataFrame(data = x[:20], columns = feature_names))
        
    mlflow.end_run()

if __name__ == "__main__":
    targetCol = "median_house_value"
    catCols = ["ocean_proximity"]
    numCols = ["housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    data_path = 'dataset/housing.csv'
    repo = '.git'
    versions = ['v1', 'v2'] #Git tag
    
    for v in versions:
        data_url = dvc.api.get_url(path = data_path, repo = repo, rev = v)
        storage_type = data_url.split(":")[0]
        if storage_type.upper() == "S3":
            housing = fetch_data_from_s3(data_path, repo, v)
        else:
            housing = fetch_data_from_fs(data_url)

        x = housing[numCols + catCols]
        y = housing[targetCol]
        
        pipe = get_pipeline(catCols, numCols).fit(x)
        x_transformed = pipe.transform(x)
        model = get_trained_model(x_transformed, y)
        sign = get_model_signature(numCols, catCols)

        categories = list(pipe.named_transformers_['OneHot'].categories_[0])
        feature_names = ["housing_median_age", "total_bedrooms", "median_income", 
                         "rooms_per_household", "population_per_household"] + categories

        joblib.dump(model, model_path)
        joblib.dump(pipe, pipeline_path)
        log_experiment('regres_custom_housing', x_transformed, y, model, v, sign, feature_names)