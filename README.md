### To run MLFlow in a sandbox cloud environment follow the below steps:
1. <code>docker pull postgres</code>
2. <code>docker pull minio/minio</code>
3. Create a local folder for it to be mounted as persistent volume in the containers and add this as "file sharing" resource under Docker desktop --> Settings --> Resources --> File Sharing
4. <code>docker run -d --rm -p 5432:5432 -e POSTGRES_PASSWORD=password --name postgres postgres</code>
5. <code>docker run -d --rm -p 9000:9000 --name minio minio/minio server /data</code><br>
  The above will start the containers with ephemeral storage. To start them with persistent volume, use the -v option with docker run command:<br>
		<code>docker run -d --rm -p 9000:9000 -v [local path]:/data--name minio minio/minio server /data</code><br>
		<ul><li>Running the postgres container with persistent volume on windows is a bit more involved and needs a docker compose file. The container path that needs to be mounted is <code>/var/lib/postgresql/data</code></li></ul>
6. Logon to minio UI (http://localhost:9000) and add a bucket named <code>mlruns</code>
7. Set the following environment variables:
		<code>MLFLOW_TRACKING_URI=postgresql+psycopg2://postgres:password@localhost:5432</code><br>
		<code>MLFLOW_ARTIFACT_STORE=s3://mlruns</code><br>
		<code>AWS_ACCESS_KEY_ID=minioadmin</code><br>
		<code>AWS_SECRET_ACCESS_KEY=minioadmin</code><br>
		<code>MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000</code><br>
8. Optional : <br>
	 <ul><li>Install pgAdmin and connect to the postgres server running in the container</li><ul>

### MLFlow UI
  <code>mlflow ui --backend-store-uri  %MLFLOW_TRACKING_URI%</code><br>
  OR<br>
  <code>mlflow server --backend-store-uri  %MLFLOW_TRACKING_URI% --default-artifact-root %MLFLOW_ARTIFACT_STORE%</code><br>

### Run the code
1. Clone this get repo
2. Download Kaggle Credit card fraud dataset from https://www.kaggle.com/mlg-ulb/creditcardfraud and put it under <code>./datasets</code>
3. Run the mlflow_sk for sklearn LogisticRegression Model
4. Run the mlflow_tf for MNIST dataset using Tensorflow
	
### Serve ML model
  <code>mlflow models serve --model-uri models:/REGISTERED_MODEL_NAME/MODEL_VERSION -p PORT</code>
  
### To run it in local PC environment follow the below steps:
1. Set the following environment variables: <br>
		<code>MLFLOW_TRACKING_URI=sqlite:///[Path/to/sqlite db file]</code><br>
		<code>MLFLOW_ARTIFACT_STORE=file:/[Path/to/local/folder]</code><br>

