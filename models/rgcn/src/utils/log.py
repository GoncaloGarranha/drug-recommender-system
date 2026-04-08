import logging


def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a simple project logger.

    Parameters
    ----------
    name : str
        Logger name (usually __name__)

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            "%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.setLevel(logging.INFO)

    return logger