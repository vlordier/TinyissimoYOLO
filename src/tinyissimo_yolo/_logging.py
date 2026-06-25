"""Lightweight logging for TinyissimoYOLO CLI tools."""

import logging
import sys


def get_logger(name: str = 'tinyissimo_yolo') -> logging.Logger:
    """Return a stdout logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    return logger
