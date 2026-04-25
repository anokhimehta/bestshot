FROM ghcr.io/mlflow/mlflow:v2.10.0

# Include DB and S3 client dependencies at build time so pods do not need
# outbound internet access during startup.
RUN pip install --no-cache-dir psycopg2-binary boto3
