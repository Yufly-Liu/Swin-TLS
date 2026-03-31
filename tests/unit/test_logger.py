"""Logger 单元测试。"""

import logging
import os
import tempfile

import pytest

from utils.logger import Logger


@pytest.fixture(autouse=True)
def reset_loggers():
    """每个测试前后重置日志记录器。"""
    Logger.reset()
    yield
    Logger.reset()


class TestLogger:
    def test_get_logger_returns_logger(self):
        logger = Logger.get_logger("test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test"

    def test_get_logger_same_name_returns_same_instance(self):
        logger1 = Logger.get_logger("test")
        logger2 = Logger.get_logger("test")
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        logger1 = Logger.get_logger("a")
        logger2 = Logger.get_logger("b")
        assert logger1 is not logger2

    def test_logger_has_console_handler(self):
        logger = Logger.get_logger("test")
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_logger_with_file_output(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            logger = Logger.get_logger("file_test", log_dir=tmp_dir, log_file="test.log")
            logger.info("test message")
            # Flush handlers to ensure content is written
            for h in logger.handlers:
                h.flush()
            log_path = os.path.join(tmp_dir, "test.log")
            assert os.path.exists(log_path)
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "test message" in content
        finally:
            # Close handlers before cleanup to avoid Windows file lock
            Logger.reset()

    def test_logger_level(self):
        logger = Logger.get_logger("level_test", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_reset_clears_loggers(self):
        Logger.get_logger("test")
        assert "test" in Logger._loggers
        Logger.reset()
        assert "test" not in Logger._loggers
