import logging
from logging.handlers import RotatingFileHandler

# Define log file path
LOG_FILE_PATH = "app.log"

with open(LOG_FILE_PATH, "w") as file:
    pass  # Create an empty log file

# Create a logger for your module
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # Capture all levels

# Prevent adding multiple handlers if this setup is run multiple times
if not logger.hasHandlers():
    # Create a console handler for standard output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # INFO and above for console

    # Create a rotating file handler to save logs to a file
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=10**6,  # 1 MB per log file
        backupCount=5,   # Keep up to 5 backup files
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # DEBUG and above for file

    # Create a logging format with more contextual information
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Attach the formatter to both handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Prevent log messages from being propagated to the root logger
    logger.propagate = False

# Optional: Suppress overly verbose logs from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("motor").setLevel(logging.WARNING)
