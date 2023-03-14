import logging
import os
from datetime import datetime

"""
LOG_FILE: It is used give the time in the datetime format with extension ".log"
logs_path: It creates the logs file in the current working directory
makedirs: is used to create a new directory for the logs
"""

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok = True)


LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


"""
basicConfig: it gives the filename, format and level indicates the urgency of the entry into the file
level = INFO = 20
"""

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging has started")