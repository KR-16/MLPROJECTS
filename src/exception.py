# Module provides various function and variables that are used to manipulate different parts of the Python runtime environment

import sys
from src.logger import logging
"""
error_message_details(error, error_detail:sys):
    return type will be sys
    error_detail.exc_info() will return 3 outputs
    exc_tb: will give the error detail, line number, filename
    file_name:Gives the filename or script in which error occured
    exc_tb.tb_lineno: returns the line number
    str(error): error message
"""

def error_message_details(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_str = str(error)
    error_message = "Error occured in Python Script name:[{0}] on line number [{1}] error message [{2}]".format(
    file_name, line_no, error_str
    )
    return error_message

"""
CustomException(Exception):
    __init__: function is used to initialize the instance of the exception with an error message and detail
    super.__init__: is used to call the parent's constructor, which initializes an instance of Exception with the same error message and detail as before
    __str__: returns the error message of the exception
    error_detail is tracked by sys
"""

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail = error_detail)
    
    def __str__(self) -> str:
        return self.error_message