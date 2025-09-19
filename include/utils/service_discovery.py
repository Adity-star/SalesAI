"""
Simple service discovery for MLflow and MinIO endpoints
"""

import os
import logging
import time
from typing import Optional
import urllib.request

logger = logging.getLogger(__name__)

def get_mlflow_endpoint() -> Optional[str]:
    """Try multiple endpoints to find MLflow"""
    # Check if explicitly set in environment
    env_uri = os.getenv('MLFLOW_TRACKING_URI')
    if env_uri:
        logger.info(f"Using MLFLOW_TRACKING_URI from env: {env_uri}")
        return env_uri
    
    # Check if we're in a container by looking for common container indicators
    in_container = os.path.exists('/.dockerenv') or os.environ.get('AIRFLOW__CORE__EXECUTOR')
    
    # Order endpoints based on environment
    if in_container:
        # In container, prioritize service names
        endpoints = [
            'http://mlflow:5001',
            'http://host.docker.internal:5001',
            'http://172.17.0.1:5001',  # Default Docker bridge
            'http://localhost:5001'
        ]
    else:
        # Outside container, prioritize localhost
        endpoints = [
            'http://localhost:5001',
            'http://127.0.0.1:5001',
            'http://host.docker.internal:5001'
        ]

    import urllib.request
    for endpoint in endpoints:
        try:
            start_time = time.time()
            url = f"{endpoint}/api/2.0/mlflow/experiments/list"

            logger.debug(f"Trying MLflow endpoint: {url}")
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as response:
                elapsed = time.time() - start_time
                if response.getcode() == 200:
                  logger.info(f"MLflow is accessible at: {endpoint} (response time: {elapsed:.2f}s)")

                  return endpoint
                else:
                    logger.debug(f"MLflow endpoint {endpoint} returned HTTP {response.getcode()} (response time: {elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.debug(f"MLflow not accessible at {endpoint} after {elapsed:.2f}s: {e}")


    # If nothing works, return the most likely default based on environment
    default = 'http://mlflow:5001' if in_container else 'http://localhost:5001'
    logger.warning(f"Could not connect to MLflow, using default: {default}")
    return default

def get_minio_endpoint() -> Optional[str]:
    """Try multiple endpoints to find MinIO"""
    # Check if explicitly set in environment
    env_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')
    if env_url:
        logger.info(f"Using MLFLOW_S3_ENDPOINT_URL from env: {env_url}")
        return env_url
    
    # Check if we're in a container
    in_container = os.path.exists('/.dockerenv') or os.environ.get('AIRFLOW__CORE__EXECUTOR')
    
    # Order endpoints based on environment
    if in_container:
        # In container, prioritize service names
        endpoints = [
            'http://minio:9000',
            'http://host.docker.internal:9000',
            'http://172.17.0.1:9000',  # Default Docker bridge
            'http://localhost:9000'
        ]
    else:
        # Outside container, prioritize localhost
        endpoints = [
            'http://localhost:9000',
            'http://127.0.0.1:9000',
            'http://host.docker.internal:9000'
        ]

    import urllib.request
    for endpoint in endpoints:
        try:
            start_time = time.time()
            url = f"{endpoint}/minio/health/live"
            logger.debug(f"Trying MinIO endpoint: {url}")
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as response:
                elapsed = time.time() - start_time
                if response.getcode() == 200:
                    logger.info(f"MinIO is accessible at: {endpoint} (response time: {elapsed:.2f}s)")
                    return endpoint
                else:
                    logger.debug(f"MinIO endpoint {endpoint} returned HTTP {response.getcode()} (response time: {elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.debug(f"MinIO not accessible at {endpoint} after {elapsed:.2f}s: {e}")


    # If nothing works, return the most likely default based on environment
    default = 'http://minio:9000' if in_container else 'http://localhost:9000'
    logger.warning(f"Could not connect to MinIO, using default: {default}")
    return default


# Backward compatibility
def get_mlflow_uri() -> str:
    """Get MLflow URI (backward compatibility)"""
    return get_mlflow_endpoint()


def get_minio_url() -> str:
    """Get MinIO URL (backward compatibility)"""
    return get_minio_endpoint()