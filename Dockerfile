FROM apache/airflow:2.9.2


USER root
# Install additional system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*


USER airflow

WORKDIR /opt/airflow

COPY ./backend /opt/airflow/backend
COPY ./backend/requirements.txt /opt/airflow/backend/requirements.txt

RUN pip install --no-cache-dir -r /opt/airflow/backend/requirements.txt


# Set environment variables for MLflow and MinIO
ENV MLFLOW_TRACKING_URI=http://mlflow:5001
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV AWS_DEFAULT_REGION=us-east-1

ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/backend/dags
ENV PYTHONPATH=/opt/airflow/backend