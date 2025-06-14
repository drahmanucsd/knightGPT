import logging
import sys

__all__ = ['configure_logging']

def configure_logging(
    level: str = 'INFO',
    fmt: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt: str = '%Y-%m-%d %H:%M:%S',
    stream: bool = True,
    log_file: str = None
):
    """
    Configure root logger with console and optional file handlers.

    :param level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param fmt: Log message format string
    :param datefmt: Date format string
    :param stream: If True, add a StreamHandler to stderr
    :param log_file: If provided, path to a file to log into
    """
    # Parse level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Clear existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    root.setLevel(numeric_level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Stream handler
    if stream:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(numeric_level)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    logging.debug(f"Logging configured. Level={level}, stream={stream}, log_file={log_file}")
