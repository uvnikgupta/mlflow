docker build -t mlflow_server -f Dockerfile.mlflow.server .
docker build -t mlflow_clf_infer:1.0 -f Dockerfile.clf.inference .
docker build -t mlflow_regres_infer:1.0 --build-arg USER=%PACKAGR_USER% --build-arg PASS=%PACKAGR_PASS% -f Dockerfile.regres.inference .
docker build -t mlflow_regres_infer:2.0 --build-arg USER=%PACKAGR_USER% --build-arg PASS=%PACKAGR_PASS% -f Dockerfile.regres.v2.inference .
docker build -t mltoolkit -f Dockerfile.mltoolkit .
docker build -t minioclient -f Dockerfile.minioclient .