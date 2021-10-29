FROM python:latest

RUN pip install psycopg2-binary mlflow boto3

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri "$DB_CON_URL" \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root s3://mlflow/