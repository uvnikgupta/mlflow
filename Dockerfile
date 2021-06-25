FROM python:3.8
RUN pip install -U mlflow \
    && pip install cloudpickle==1.5.0 \
    && pip install scikit-learn==0.24.2 \
    && pip install psycopg2

ENV AWS_ACCESS_KEY_ID minioadmin
ENV AWS_SECRET_ACCESS_KEY minioadmin
ENV MLFLOW_S3_ENDPOINT_URL http://127.0.0.1:9000

ENV MLFLOW_TRACKING_URI postgresql+psycopg2://postgres:password@127.0.0.1:5432
ENV MLFLOW_ARTIFACT_STORE s3://mlruns

ENV MODEL logreg
ENV VERSION Staging
ENV PORT 1235

EXPOSE 1235
CMD ["mlflow models serve --no-conda --model-uri models:/logreg/Staging -p 1235"]