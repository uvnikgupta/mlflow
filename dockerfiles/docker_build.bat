docker build -t mlflow_server -f Dockerfile.mlflow.server .
docker build -t mlflow_logreg_infer -f Dockerfile.logreg.inference .
docker build -t mlflow_linreg_infer --build-arg USER=%PACKAGR_USER% --build-arg PASS=%PACKAGR_PASS% -f Dockerfile.linreg.inference .
docker build -t mltoolkit -f Dockerfile.mltoolkit .
docker build -t minioclient -f Dockerfile.minioclient .