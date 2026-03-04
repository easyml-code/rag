import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "app.log")

class ColorFormatter(logging.Formatter):
    # ANSI color codes
    COLORS = {
        logging.DEBUG: "\x1b[38;5;245m",     # grey
        logging.INFO: "\x1b[38;5;39m",       # blue
        logging.WARNING: "\x1b[38;5;214m",   # orange
        logging.ERROR: "\x1b[38;5;196m",     # red
        logging.CRITICAL: "\x1b[48;5;196m\x1b[38;5;15m",  # white on red bg
    }
    RESET = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{base}{self.RESET}" if color else base

def setup_logger(name: str = "app_logger") -> logging.Logger:
    logger = logging.getLogger(name)

    # Prevent duplicate logs if module is imported multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # capture everything; handlers decide what to output
    logger.propagate = False

    fmt = "%(asctime)s %(levelname)s %(name)s %(module)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # ---- File handler (UTF-8, no colors) ----
    file_formatter = logging.Formatter(fmt, datefmt=datefmt)
    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=1 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",   # &lt;-- IMPORTANT: fixes emoji/unicode in file logs
        delay=True,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # ---- Console handler (UTF-8 + colors) ----
    # Ensure stdout is UTF-8 in Python 3.7+ (works well on Windows)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass  # older environments may not support reconfigure

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(ColorFormatter(fmt, datefmt=datefmt))
    logger.addHandler(console_handler)

    return logger

logger = setup_logger("app_logger")