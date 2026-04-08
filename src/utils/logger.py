from __future__ import annotations

import logging
import sys
from pathlib import Path


class _ColorfulFormatter(logging.Formatter):
    _RESET = "\033[0m"
    _RED = "\033[31m"

    def __init__(
        self,
        fmt: str = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt: str | None = None,
        enable_color: bool = True,
        metric_marker: str = "[METRIC]",
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.enable_color = enable_color
        self.metric_marker = metric_marker

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        prefix = f"{self.metric_marker} " if bool(getattr(record, "is_metric", False)) else ""
        if not self.enable_color:
            return f"{prefix}{message}"
        if record.levelno >= logging.WARNING or bool(getattr(record, "is_metric", False)):
            return f"{self._RED}{prefix}{message}{self._RESET}"
        return f"{prefix}{message}"


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: str | None = None,
    use_console: bool = False,
    console_stream: str = "stderr",
    enable_color: bool = True,
    metric_marker: str = "[METRIC]",
    propagate: bool = False,
    reset_handlers: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = propagate

    if reset_handlers:
        logger.handlers.clear()

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(_ColorfulFormatter(enable_color=False, metric_marker=metric_marker))
        logger.addHandler(file_handler)

    if use_console:
        stream = sys.stdout if console_stream == "stdout" else sys.stderr
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(logger.level)
        console_handler.setFormatter(_ColorfulFormatter(enable_color=enable_color, metric_marker=metric_marker))
        logger.addHandler(console_handler)

    return logger
