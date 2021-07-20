import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient

model_uri = os.environ['MLFLOW_REGISTERED_MODEL_URI']
experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
backend_uri = os.environ['MLFLOW_TRACKING_URI']
artifact_uri = os.environ['MLFLOW_ARTIFACT_STORE']
mlflow.set_tracking_uri(backend_uri)


class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pipeline = joblib.load(context.artifacts["pipeline"])
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        model_input.columns = ["housing_median_age", "total_rooms", "total_bedrooms", 
                   "population", "households", "median_income", "ocean_proximity"]
        input_matrix = self.pipeline.transform(model_input)
        return self.model.predict(input_matrix)


def export_registered_model() -> str:
    client = MlflowClient()
    model = mlflow.pyfunc.load_model(model_uri)
    client.download_artifacts(model.metadata.run_id, model.metadata.artifact_path, '.')
    return model.metadata.artifact_path


def import_model(artifact_path: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment.name)

    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path = artifact_path, python_model = ModelWrapper())


if __name__ == "__main__":
    artifact_path = export_registered_model()
    import_model(artifact_path)