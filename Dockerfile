FROM python:3.8.2-slim

ARG DB_URI
ENV DB_URI=${DB_URI}
ARG VIRTUAL_PORT
ENV VIRTUAL_PORT=${VIRTUAL_PORT}
ARG ARTIFACT_PATH
ENV ARTIFACT_PATH=${ARTIFACT_PATH}

RUN pip install psycopg2-binary mlflow

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri "$DB_URI" \
    --host 0.0.0.0 \
    --port "$VIRTUAL_PORT" \
    --default-artifact-root "$ARTIFACT_PATH"