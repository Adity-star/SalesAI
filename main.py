from include.utils.data_generator import SyntheticDataGenerator
import sys
from include.logger import logger

import os
print(os.__file__)
print(hasattr(os, "register_at_fork"))



def extract_data_task():
        logger.info("Starting synthetic sales data generation")

        from include.utils.data_generator import SyntheticDataGenerator

        data_output_dir = "tmp/sales_data"
        generator = SyntheticDataGenerator(
            start_date="2023-01-01", end_date="2023-12-31"
        )
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        total_files = sum(len(paths) for paths in file_paths.values())
        
        logger.info(f"Generated {total_files} files:")
        for data_type, paths in file_paths.items():
            logger.info(f"  - {data_type}: {len(paths)} files")
        
        return {
            "data_output_dir": data_output_dir,
            "file_paths": file_paths,
            "total_files": total_files,
        }


extract_data_task()


