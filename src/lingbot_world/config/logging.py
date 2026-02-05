"""Production-grade logging configuration."""

import logging
import sys


def setup_logging(
    level: str = "INFO",
    name: str | None = None,
) -> logging.Logger:
    """Configure production-grade logging.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    name : str, optional
        Logger name. If None, returns root logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Create formatter with structured output
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "lingbot_world") -> logging.Logger:
    """Get a logger instance.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return logging.getLogger(name)
