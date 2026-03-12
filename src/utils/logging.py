"""Logging utilities: expose a pre-configured loguru logger."""

from loguru import logger as _logger


def get_logger(name: str):
    """Return a loguru logger bound to *name*.

    Using loguru there is only one global logger; binding a name simply
    adds it as a context field so that log messages include the module.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A bound loguru logger instance.
    """
    return _logger.bind(module=name)
