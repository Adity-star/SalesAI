import pytest
import os

from src.data_pipelines.data_generator import SyntheticDataGenerator

@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("sales_data")
    return path

@pytest.fixture(scope="session")
def data_generator(output_dir):
    generator = SyntheticDataGenerator(start_date="2023-01-01", end_date="2023-01-10")
    generator.generate_sales_data(output_dir=output_dir)
    return output_dir