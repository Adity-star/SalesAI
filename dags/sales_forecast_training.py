import os
import sys
import pandas as pd

from datetime import datetime, timedelta
from airflow.decorators import dag, task


# Add include path
sys.path.append("/usr/local/airflow/include")


default_args = {
    "owner": "PeaceAi",
    "depends_on_past": False,
    "start_date": datetime(2025, 7, 24),
    "email_on_failure": True,
    "email_on_retry": False,
    "email": ["aakuskar.980@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}