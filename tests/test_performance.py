import time

import pytest

from prlearn import Experience


def test_experience_add_step_throughput_is_deterministic(monkeypatch):
    """A simple load test that reports deterministic throughput for bulk add_step calls."""
    exp = Experience()
    n_steps = 2000
    fake_clock = iter([0.0, 0.05])
    monkeypatch.setattr(time, "perf_counter", lambda: next(fake_clock))

    start = time.perf_counter()
    for i in range(n_steps):
        exp.add_step(i, i, i, i, False, False, {}, 0, 0, 0)
    duration = time.perf_counter() - start

    assert duration == pytest.approx(0.05)
    throughput = n_steps / duration
    assert throughput == pytest.approx(40000.0)
    assert len(exp) == n_steps
