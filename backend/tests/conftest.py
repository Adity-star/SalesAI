import pytest
import shutil
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipelines.data_generator import SyntheticDataGenerator

@pytest.fixture(scope="session")
def output_dir():
    path = "/tmp/sales_data"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path

@pytest.fixture(scope="session")
def data_generator(output_dir):
    generator = SyntheticDataGenerator(start_date="2023-01-01", end_date="2023-01-10")
    generator.generate_sales_data(output_dir=output_dir)
    return output_dir
