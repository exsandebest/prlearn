import logging

import pytest

from prlearn.utils.logger import get_logger
from prlearn.utils.message_utils import (
    queue_receive,
    queue_send,
    try_queue_receive,
    try_queue_send,
)
from prlearn.utils.multiproc_lib import mp


def test_queue_send_and_receive_roundtrip():
    """Sending data through the queue should succeed and be retrievable."""
    q = mp.Queue()
    assert queue_send(q, "payload") == 0
    assert queue_receive(q, timeout=0.1) == "payload"


def test_queue_receive_timeout_returns_none():
    """Receiving from an empty queue with a timeout should return None."""
    q = mp.Queue()
    assert queue_receive(q, timeout=0.01) is None


def test_try_queue_send_on_full_queue_returns_error():
    """Non-blocking send should report a full queue."""
    q = mp.Queue(maxsize=1)
    assert queue_send(q, "first") == 0
    assert try_queue_send(q, "second") == 1


def test_try_queue_receive_empty_returns_none():
    """Non-blocking receive should return None on an empty queue."""
    q = mp.Queue()
    assert try_queue_receive(q) is None


def test_get_logger_respects_env_level(monkeypatch):
    """LOG_LEVEL env var should set the logger level."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    logger = get_logger("test_logger_debug")
    assert logger.level == logging.DEBUG


def test_get_logger_default_level(monkeypatch):
    """Without LOG_LEVEL the logger should default to INFO."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logger = get_logger("test_logger_default")
    assert logger.level == logging.INFO


if __name__ == "__main__":
    pytest.main()
