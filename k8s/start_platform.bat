kubectl apply -f pg_dep.yml
kubectl apply -f minio_dep.yml
kubectl apply -f mlflow_server.yml
@echo off
@timeout /t 30 /nobreak
@echo on
kubectl apply -f minioclient_job.yml
kubectl apply -f mltoolkit_dep.yml