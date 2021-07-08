import os
from minio import Minio
from minio.error import S3Error

s3_url = os.environ['MLFLOW_URL_FOR_CLIENT']
s3_access_key = os.environ['AWS_ACCESS_KEY_ID']
s3_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

def main():
    client = Minio(
        s3_url,
        access_key=s3_access_key,
        secret_key=s3_secret_key,
        secure=False
    )

    found = client.bucket_exists("mlruns")
    if not found:
        client.make_bucket("mlruns")
    else:
        print("Bucket 'mlruns' already exists")
    
    found = client.bucket_exists("traindata")
    if not found:
        client.make_bucket("traindata")
    else:
        print("Bucket 'traindata' already exists")

if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print(f'Error occured : {exc}')