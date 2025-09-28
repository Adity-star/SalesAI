import sys
import logging
import traceback
from types import ModuleType


def error_message_detail(error: Exception, error_detail: ModuleType) -> str:
    """
    Extracts detailed error information including file name, line number, and the error message.

    :param error: The exception that occurred.
    :param error_detail: The sys module to access traceback details.
    :return: A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"❗ Error occurred in python script: [{file_name}] "
            f"at line number [{line_number}]: {str(error)}"
        )
    else:
        error_message = f"❗ Exception: {str(error)}"

    logging.error(error_message)
    return error_message


class CustomException(Exception):
    """
    Custom exception class that includes traceback details.
    Accepts the exception object and the `sys` module.
    """

    def __init__(self, error: Exception, error_detail: ModuleType):
        """
        Initialize the exception with error object and sys module for traceback extraction.
        """
        self.error_message = error_message_detail(error, error_detail)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message
