"""日志系统模块 - 提供统一的日志记录功能。"""

import logging
import os
import sys
from typing import Optional


class Logger:
    """统一的日志记录器，支持控制台和文件输出。"""

    _loggers: dict = {}

    @staticmethod
    def get_logger(
        name: str = "denoising",
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
    ) -> logging.Logger:
        """获取或创建一个日志记录器。

        Args:
            name: 日志记录器名称。
            log_dir: 日志文件目录（可选）。
            log_file: 日志文件名（可选）。
            level: 日志级别。

        Returns:
            配置好的 logging.Logger 实例。
        """
        if name in Logger._loggers:
            return Logger._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        # 避免重复添加handler
        if not logger.handlers:
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # 控制台输出
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # 文件输出（可选）
            if log_dir and log_file:
                os.makedirs(log_dir, exist_ok=True)
                file_path = os.path.join(log_dir, log_file)
                file_handler = logging.FileHandler(file_path, encoding="utf-8")
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

        Logger._loggers[name] = logger
        return logger

    @staticmethod
    def reset():
        """重置所有已创建的日志记录器（主要用于测试）。"""
        for name, logger in Logger._loggers.items():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        Logger._loggers.clear()
