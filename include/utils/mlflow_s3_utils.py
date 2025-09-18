import os
import mlflow
import boto3
import shutil
import traceback

from typing import Optional, List
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
from include.utils.service_discovery import get_minio_endpoint

import logging
from include.logger import logger


logger = logging.getLogger(__name__)


class MLflowS3Manager:
    """
    Manager to handle MLflow artifact operations with S3-compatible storage (e.g., MinIO)
    """

    def __init__(self, bucket_name: str = "mlflow-artifacts"):
        self.bucket_name = bucket_name

        self.s3_client = boto3.client(
            's3',
            endpoint_url=get_minio_endpoint(),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin'),
            config=Config(signature_version='s3v4'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )

    def _build_s3_key(self, run_id: str, local_path: str, artifact_path: Optional[str] = None) -> str:
        """
        Construct S3 object key from run ID and local file path
        """
        base = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts"
        rel_path = f"{artifact_path}/{os.path.basename(local_path)}" if artifact_path else os.path.basename(local_path)
        return f"{base}/{rel_path}".replace("//", "/")

    def upload_artifact_to_s3(self, local_path: str, run_id: str, artifact_path: Optional[str] = None) -> str:
        """
        Upload a single file to S3
        """
        try:
            s3_key = self._build_s3_key(run_id, local_path, artifact_path)
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return s3_key
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Upload failed for {local_path}: {e}\n{traceback.format_exc()}")
            raise

    def log_artifact_with_s3(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log to MLflow and then upload to S3 explicitly
        """
        try:
            if artifact_path:
                mlflow.log_artifact(local_path, artifact_path)
            else:
                mlflow.log_artifact(local_path)

            run = mlflow.active_run()
            if run:
                self.upload_artifact_to_s3(local_path, run.info.run_id, artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifact and sync: {e}\n{traceback.format_exc()}")

    def sync_mlflow_artifacts_to_s3(self, run_id: str):
        """
        Recursively upload all local artifacts from a given MLflow run to S3
        """
        try:
            client = mlflow.tracking.MlflowClient()
            local_dir = f"/tmp/mlflow_sync/{run_id}"
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)

            logger.info(f"Downloading artifacts for run {run_id}")
            artifacts_dir = client.download_artifacts(run_id, "", dst_path=local_dir)

            count = 0
            for root, _, files in os.walk(artifacts_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file, artifacts_dir)
                    s3_key = self._build_s3_key(run_id, relative_path)
                    self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
                    logger.debug(f"Synced {relative_path} -> s3://{self.bucket_name}/{s3_key}")
                    count += 1

            shutil.rmtree(local_dir)
            logger.info(f"Successfully synced {count} artifact(s) for run {run_id} to S3")
        except Exception as e:
            logger.error(f"Artifact sync failed for run {run_id}: {e}\n{traceback.format_exc()}")
            raise

    def list_s3_artifacts(self, run_id: str) -> List[str]:
        """
        List all artifacts uploaded to S3 for a given run
        """
        try:
            prefix = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            keys = []
            for page in page_iterator:
                if "Contents" in page:
                    keys.extend(obj["Key"] for obj in page["Contents"])

            logger.info(f"Found {len(keys)} artifact(s) for run {run_id} in S3")
            return keys

        except Exception as e:
            logger.error(f"Failed to list artifacts for run {run_id}: {e}\n{traceback.format_exc()}")
            return []
