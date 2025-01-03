import logging
import sys

import colorlog

# Get the root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Formatter
formatter = colorlog.ColoredFormatter(
    "%(bold)s%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s"
    " %(bold)s-%(reset)s %(message)s",
)

# Stream handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return logger with a given name or the root logger."""
    if name:
        return logging.getLogger(__name__ + "." + name)
    return logger
