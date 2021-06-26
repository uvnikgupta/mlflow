# To run MLFlow follow the below steps:
1. Clone this get repo
2. Download Kaggle Credit card fraud dataset from https://www.kaggle.com/mlg-ulb/creditcardfraud and put it under <code>./dataset</code>
3. Install Docker Desktop and enable Kubernetes from settings
4. Install pgAdmin
5. pip install all the required python libraries including mlflow and miniconda
6. <code>docker pull postgres</code>
7. <code>docker pull minio/minio</code>
8. Build the docker images for mlflow server and logreg serving using <code>docker_build.bat</code>.
	* The mlflow server is for the mlflow UI and logreg serving is to serve the model registered as "Staging"
	* <b>Note:</b> If you are running this in pure Docker environment without using Kubernetes you may have to change the IP addresses of the postgres and minio server in the docker files before building them
### Docker environment
1. <code>docker run -d --rm -p 5432:5432 -e POSTGRES_PASSWORD=password --name postgres postgres</code>
	* Open pgAdmin and connect to the postgres server running in the container (host=localhost, port=5432)
2. <code>docker run -d --rm -p 9000:9000 --name minio minio/minio server /data</code><br>
  The above will start the containers with ephemeral storage. To start them with persistent volume, do the following:
	  <ul>
		<li>Create a local folder for it to be mounted as persistent volume in the containers and add this as "file sharing" resource under Docker desktop --> Settings --> Resources --> File Sharing</li>
		<li>use the -v option with docker run command:<br>
			<code>docker run -d --rm -p 9000:9000 -v [local path]:/data--name minio minio/minio server /data</code></li>
			<ul><li>Running the postgres container with persistent volume on windows is a bit more involved and needs a docker compose file. The container path that needs to be mounted is <code>/var/lib/postgresql/data</code></li></ul></ul>
3. <code>docker run -d --rm -p 5000:5000 --name mlflow_server mlflow_server</code><br>
4. Logon to minio UI (http://localhost:9000) and add a bucket named <code>mlruns</code>
5. Logon to MLFlow UI using http://localhost:5000
6. Run the <code>mlflow_sk.pynb</code> for sklearn LogisticRegression Model
7. Use the MLFlow UI to register one of the models as "logreg" and promote it to "Staging"
8. <code>docker run -d --rm -p 1235:1235 --name mlflow_sk mlflow_sk</code> to start model serving<br>
9. Change the port in <code>mlflow_serving.ipynb</code> to <code>1235</code> and run it. Check the returned inference values

### Kubernetes environment
1. Run <code> kubectl apply -f . </code>
2. Logon to minio UI (http://localhost:30009) and add a bucket named <code>mlruns</code>
3. Logon to MLFlow UI using http://localhost:30005
4. Open pgAdmin and connect to the postgres server running in kubernetes (host=localhost, port=32345)
5. Run the <code>mlflow_sk.pynb</code> for sklearn LogisticRegression Model
6. Use the MLFlow UI to register one of the models as "logreg" and promote it to "Staging"
7. Change the port in <code>mlflow_serving.ipynb</code> to <code>31235</code> and run it. Check the returned inference values

### Local Environment
1. Install sqlite3 if it does not already exist (Normally this comes by default with Python installation)
2. Create a folder for the sqlite db
3. Set the following environment variables:<br>
		<code>MLFLOW_TRACKING_URI=sqlite:///[Path/to/sqlite db file]</code><br>
		<code>MLFLOW_ARTIFACT_STORE=file:/[Path/to/local/folder]</code>
4. Run <code>mlflow ui --backend-store-uri  %MLFLOW_TRACKING_URI%</code> OR <code>mlflow server --backend-store-uri  %MLFLOW_TRACKING_URI% --default-artifact-root %MLFLOW_ARTIFACT_STORE%</code> to start the MLFlow server
5. Logon to MLFlow UI using http://localhost:5000
5. Run the <code>mlflow_sk.pynb</code> for sklearn LogisticRegression Model
6. Use the MLFlow UI to register one of the models as "logreg" and promote it to "Staging"
7. Run <code>mlflow models serve --model-uri models:/logreg/Staging -p 1235</code> to start serving the model
8. Change the port in <code>mlflow_serving.ipynb</code> to <code>1235</code> and run it. Check the returned inference values

## Further trials
1. Try to serve the Tensorflow model for MNIST dataset in mlflow_tf.ipynb