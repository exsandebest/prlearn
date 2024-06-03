import os
import queue
from typing import Any

import multiprocess as mp


def queue_receive(q: mp.Queue, timeout=None):
    try:
        try:
            return q.get(timeout=timeout)
        except TimeoutError as e:
            return None
    except queue.Empty as e:
        return None


def queue_send(q: mp.Queue, data: Any):
    try:
        q.put(data)
        return 0
    except queue.Full as e:
        return 1


def try_queue_receive(q: mp.Queue):
    if not q.empty():
        try:
            return q.get_nowait()
        except queue.Empty as e:
            return None


def try_queue_send(q: mp.Queue, data: Any) -> int:
    if not q.full():
        try:
            q.put_nowait(data)
            return 0
        except queue.Full as e:
            return 1
