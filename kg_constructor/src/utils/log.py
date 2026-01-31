from __future__ import annotations

import logging


def get_logger(name: str = "kg_constructor") -> logging.Logger:
    """
    Create or retrieve a configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
