docker build -t mlflow_sk -f Dockerfile.logreg_serving .
docker build -t mlflow_server -f Dockerfile.mlflow_server .