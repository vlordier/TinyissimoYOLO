"""Lightweight logging for TinyissimoYOLO CLI tools."""

import logging
import sys

_LOG: logging.Logger | None = None


def get_logger(name: str = 'tinyissimo_yolo') -> logging.Logger:
    global _LOG
    if _LOG is not None:
        return _LOG
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.handlers.clear()
    logger.addHandler(handler)
    _LOG = logger
    return logger
