import os
import pytest
import pandas as pd
from pathlib import Path

@pytest.mark.parametrize("folder, required_columns", [
    ("sales", ["date", "store_id", "product_id", "quantity_sold", "revenue", "profit"]),
    ("inventory", ["date", "store_id", "product_id", "inventory_level", "reorder_point"]),
    ("customer_traffic", ["date", "store_id", "customer_traffic", "weather_impact"]),
    ("promotions", ["date", "product_id", "promotion_type", "discount_percent"]),
    ("store_events", ["store_id", "date", "event_type", "impact"]),
])
def test_parquet_files_exist_and_have_columns(data_generator, folder, required_columns):
    base_dir = Path(data_generator)
    folder_path = base_dir / folder
    assert folder_path.exists(), f"{folder} folder is missing."

    found_file = False
    for file in folder_path.rglob("*.parquet"):
        df = pd.read_parquet(file)
        assert not df.empty, f"{file} is empty."
        for col in required_columns:
            assert col in df.columns, f"{col} missing in {file}"
        found_file = True

    assert found_file, f"No .parquet files found in {folder}"

def test_metadata_file_exists_and_valid(data_generator):
    metadata_path = Path(data_generator) / "metadata" / "generation_metadata.parquet"
    assert metadata_path.exists(), "Metadata file missing."
    df = pd.read_parquet(metadata_path)
    assert not df.empty, "Metadata is empty."
    required_keys = ['generation_date', 'start_date', 'end_date', 'n_stores', 'n_products']
    for key in required_keys:
        assert key in df.columns, f"{key} missing in metadata"

def test_data_within_date_range(data_generator):
    meta_path = Path(data_generator) / "metadata/generation_metadata.parquet"
    metadata = pd.read_parquet(meta_path).iloc[0]
    start_date = pd.to_datetime(metadata['start_date'])
    end_date = pd.to_datetime(metadata['end_date'])

    for folder in ["sales", "customer_traffic", "inventory"]:
        for file in Path(data_generator, folder).rglob("*.parquet"):
            df = pd.read_parquet(file)
            if 'date' in df.columns:
                assert df['date'].min() >= start_date, f"{file}: date out of range"
                assert df['date'].max() <= end_date, f"{file}: date out of range"

def test_positive_quantities_and_revenue(data_generator):
    sales_dir = Path(data_generator) / "sales"
    for file in sales_dir.rglob("*.parquet"):
        df = pd.read_parquet(file)
        assert (df["quantity_sold"] >= 0).all(), f"Negative quantity in {file}"
        assert (df["revenue"] >= 0).all(), f"Negative revenue in {file}"
        assert (df["profit"] >= 0).all(), f"Negative profit in {file}"
