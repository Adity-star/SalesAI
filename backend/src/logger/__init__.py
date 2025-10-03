# Logging framework

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
import sys
import re


def from_root(path: str = "") -> str:
    """Return absolute path from project root."""
    return str(Path(__file__).resolve().parents[2] / path)


# Constants
LOG_DIR = 'logs'
LOG_FILE_TIMESTAMP = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
MAX_LOG_SIZE = 5 * 1024 * 1024 
BACKUP_COUNT = 3  

# Ensure log directory exists
log_dir_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)

# Force stdout and stderr to UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# helper function to get log file path
def get_log_file_path(filename_prefix: str = "app", use_timestamp: bool = True) -> str:
    """
    Constructs a log file path with optional timestamp.
    """
    filename = f"{filename_prefix}_{LOG_FILE_TIMESTAMP}.log" if use_timestamp else f"{filename_prefix}.log"
    return os.path.join(log_dir_path, filename)

# ANSI escape sequences for colors
class LogColors:
    RESET = "\x1b[0m"
    BLACK = "\x1b[30m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[37m"

# Mapping logging levels to colors
LOG_LEVEL_COLORS = {
    logging.DEBUG: LogColors.CYAN,
    logging.INFO: LogColors.GREEN,
    logging.WARNING: LogColors.YELLOW,
    logging.ERROR: LogColors.RED,
    logging.CRITICAL: LogColors.MAGENTA,
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_LEVEL_COLORS.get(record.levelno, LogColors.WHITE)
        record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
        record.msg = f"{color}{record.msg}{LogColors.RESET}"
        return super().format(record)

def configure_logger(
    logger_name: str = "",
    level: int = logging.DEBUG,
    log_filename: str = None
) -> logging.Logger:
    """
    Configure a logger with both file and console handlers.
    Each run creates a new timestamped log file.
    
    Args:
        logger_name (str): Name of the logger
        level (int): Logging level
        log_filename (str, optional): Custom log filename. If None, generates timestamped name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(logger_name)

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)

    # Create formatters
    file_formatter = logging.Formatter(
        '[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s'
    )
    colored_formatter = ColoredFormatter(
        '[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s'
    )

    # Setup file handler with timestamped filename
    if log_filename is None:
        log_filename = get_log_file_path(logger_name if logger_name else "app")
    
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Setup console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(colored_formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Create default logger
logger = configure_logger()
logger.info("Logger is configured and ready.")