from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "kg_constructor") -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def add_file_handler(log_path: str | Path) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).addHandler(file_handler)