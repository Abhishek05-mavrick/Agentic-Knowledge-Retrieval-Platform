import sys
from .logger import logger

def error_details(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    filename= exc_tb.tb_frame.f_code.co_filename
    line_number= exc_tb.tb_lineno
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(filename,line_number,str(error))
    return error_message

class customException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_details(error_message, error_detail=error_detail)
        logger.error(self.error_message)
    
    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        raise customException(e, sys)