"""aeon.utils.logging — a small stdout(+file) logger for training runs."""
import logging
import sys

_CONFIGURED = set()


def get_logger(name: str = "aeon", logfile: str = None, level: int = logging.INFO):
    """Return a logger that writes to stdout and optionally to `logfile`.

    Idempotent per (name, logfile): repeated calls don't stack handlers.
    """
    logger = logging.getLogger(name)
    key = (name, logfile)
    if key in _CONFIGURED:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _CONFIGURED.add(key)
    return logger
