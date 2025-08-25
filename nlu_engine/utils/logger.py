import logging
import pathlib
from logging.handlers import TimedRotatingFileHandler

from decouple import config


def create_logger(name: str = "executionlog", level: str = "INFO", file: str = "logs/execution.log") -> logging.Logger:
    """
    Create and configure a logger with console and timed rotating file handlers.

    Args:
        name: Logger name.
        level: Logging level (e.g., "DEBUG", "INFO").
        file: Path to the log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    log_path = pathlib.Path(file)
    log_path.parent.mkdir(parents=True, exist_ok=True)  # Pre-commit compliant

    enable_logging: bool = config("ENABLE_LOGGING", cast=bool, default=False)
    if enable_logging:
        formatter = logging.Formatter("{levelname} {asctime} {module} {process:d} {thread:d} {message}", style="{")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = TimedRotatingFileHandler(file, when="midnight", interval=1, backupCount=7)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger
