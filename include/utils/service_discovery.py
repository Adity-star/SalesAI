"""
Service discovery for MLflow and MinIO.
Includes retry logic, dynamic config, and better logging.
"""

import os
import time
import urllib.request
from typing import Optional, List
from include.logger import Logger
from urllib.error import URLError, HTTPError


logger = Logger.getLogger(__name__)
logger.setLevel(Logger.INFO)


def _is_in_container() -> bool:
    """Detect if running inside a container (Docker or Kubernetes)"""
    return (
        os.path.exists('/.dockerenv') or
        os.environ.get('AIRFLOW__CORE__EXECUTOR') or
        os.environ.get('KUBERNETES_SERVICE_HOST')
    )


def _probe_endpoint(
    url: str,
    health_path: str,
    timeout: int = 2,
    retries: int = 3,
    backoff: float = 1.0
) -> bool:
    """Attempt to access an endpoint with retries"""
    full_url = f"{url.rstrip('/')}/{health_path.lstrip('/')}"
    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"Probing {full_url} (attempt {attempt})")
            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.getcode() == 200:
                    return True
        except (URLError, HTTPError) as e:
            logger.debug(f"Attempt {attempt}: Failed to reach {full_url}: {e}")
            time.sleep(backoff * attempt)
    return False

def _get_configured_value(key: str, default: str) -> str:
    """Get value from env or fallback to default"""
    return os.getenv(key, default)


def discover_service_endpoint(
    name: str,
    env_var: str,
    default_endpoints: List[str],
    health_path: str,
    default_port: int,
    timeout: int = 2,
    retries: int = 3,
    backoff: float = 1.0
) -> str:
    """General service discovery function"""
    # 1. Check environment variable
    env_value = os.getenv(env_var)
    if env_value:
        logger.info(f"{name} endpoint set via environment: {env_value}")
        return env_value

    # 2. Detect container or not
    in_container = _is_in_container()
    logger.debug(f"{name}: Detected container = {in_container}")

    # 3. Select endpoints to try
    if in_container:
        endpoints = [
            f"http://{name.lower()}:{default_port}",
            f"http://host.docker.internal:{default_port}",
            f"http://172.17.0.1:{default_port}",  # Docker bridge
            f"http://localhost:{default_port}"
        ]
    else:
        endpoints = [
            f"http://localhost:{default_port}",
            f"http://127.0.0.1:{default_port}",
            f"http://host.docker.internal:{default_port}"
        ]
    
    # Allow override/additional endpoints
    endpoints = default_endpoints or endpoints

    # 4. Try endpoints
    for endpoint in endpoints:
        if _probe_endpoint(endpoint, health_path, timeout, retries, backoff):
            logger.info(f"{name} is accessible at: {endpoint}")
            return endpoint

    # 5. Fallback
    default = f"http://{name.lower()}:{default_port}" if in_container else f"http://localhost:{default_port}"
    logger.warning(f"Could not reach {name}. Falling back to default: {default}")
    return default

# ----- MLflow-specific -----

def get_mlflow_endpoint(
    timeout: int = 2,
    retries: int = 3,
    backoff: float = 1.0,
    health_path: str = "/health"
) -> str:
    return discover_service_endpoint(
        name="MLflow",
        env_var="MLFLOW_TRACKING_URI",
        default_endpoints=[],
        health_path=health_path,
        default_port=5001,
        timeout=timeout,
        retries=retries,
        backoff=backoff
    )


def get_mlflow_uri() -> str:
    return get_mlflow_endpoint()


# ----- MinIO-specific -----

def get_minio_endpoint(
    timeout: int = 2,
    retries: int = 3,
    backoff: float = 1.0,
    health_path: str = "/minio/health/live"
) -> str:
    return discover_service_endpoint(
        name="MinIO",
        env_var="MLFLOW_S3_ENDPOINT_URL",
        default_endpoints=[],
        health_path=health_path,
        default_port=9000,
        timeout=timeout,
        retries=retries,
        backoff=backoff
    )

def get_minio_url() -> str:
    return get_minio_endpoint()
