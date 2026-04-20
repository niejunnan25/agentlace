#!/usr/bin/env python3

import socket
import time
from typing import Callable

import numpy as np

from agentlace.data.data_store import QueuedDataStore
from agentlace.internal.trainer_transport import _PushClient
from agentlace.trainer import TrainerClient
from agentlace.trainer import TrainerConfig
from agentlace.trainer import TrainerServer


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 5.0,
    interval_s: float = 0.05,
) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() <= deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return bool(predicate())


class SlowQueuedDataStore(QueuedDataStore):
    def batch_insert(self, batch_data):
        time.sleep(0.25)
        super().batch_insert(batch_data)


def _new_async_config(*, request_types=None, data_queue_capacity=2) -> TrainerConfig:
    return TrainerConfig(
        port_number=_find_free_port(),
        broadcast_port=_find_free_port(),
        data_port=_find_free_port(),
        request_types=list(request_types or []),
        transport_mode="async_commit",
        data_queue_capacity=int(data_queue_capacity),
        data_socket_hwm=4,
        commit_poll_ms=10,
    )


def test_async_commit_transport_commits_without_duplicates():
    config = _new_async_config()
    server = TrainerServer(config)
    learner_store = QueuedDataStore(64)
    server.register_data_store("table1", learner_store)
    server.start(threaded=True)

    actor_store = QueuedDataStore(64)
    client = TrainerClient(
        "table1",
        "127.0.0.1",
        config,
        data_store=actor_store,
        wait_for_server=True,
    )

    try:
        insert_count = 3
        for idx in range(insert_count):
            actor_store.insert(np.array([idx, idx + 1, idx + 2]))

        assert client.update()
        assert client.wait_until_committed("table1", timeout_s=3.0)
        assert _wait_until(lambda: len(learner_store) == insert_count)

        status = client.get_transport_status("table1")
        assert status["transport_mode"] == "async_commit"
        assert status["accepted_update_id"] == actor_store.latest_data_id()
        assert status["committed_update_id"] == actor_store.latest_data_id()

        duplicate_sender = _PushClient(
            server_ip="127.0.0.1",
            port=int(config.data_port),
            hwm=4,
            timeout_ms=200,
        )
        try:
            duplicate_sender.send_msg(
                {
                    "type": "datastore",
                    "store_name": "table1",
                    "payload": {
                        "last_id": int(actor_store.latest_data_id()),
                        "batch_count": insert_count,
                        "data": actor_store.get_latest_data(-1),
                    },
                }
            )
        finally:
            duplicate_sender.close()

        time.sleep(0.2)
        assert len(learner_store) == insert_count
    finally:
        client.stop()
        server.stop()


def test_async_commit_control_plane_stays_responsive_under_backlog():
    config = _new_async_config(request_types=["get-stats"], data_queue_capacity=1)

    def request_callback(request_type: str, payload: dict) -> dict:
        assert request_type == "get-stats"
        del payload
        return {"trainer-status": "ok"}

    server = TrainerServer(config, request_callback=request_callback)
    learner_store = SlowQueuedDataStore(128)
    server.register_data_store("table1", learner_store)
    server.start(threaded=True)

    actor_store = QueuedDataStore(128)
    client = TrainerClient(
        "table1",
        "127.0.0.1",
        config,
        data_store=actor_store,
        wait_for_server=True,
    )

    try:
        for idx in range(24):
            actor_store.insert(np.array([idx] * 8))

        assert client.update()
        assert _wait_until(
            lambda: client.get_transport_status("table1")["transport_backlog"] > 0,
            timeout_s=3.0,
        )

        response = client.request("get-stats", {})
        assert response == {"trainer-status": "ok"}

        assert client.wait_until_committed("table1", timeout_s=5.0)
        final_status = client.get_transport_status("table1")
        assert final_status["transport_backlog"] == 0
        assert final_status["committed_update_id"] == actor_store.latest_data_id()
    finally:
        client.stop()
        server.stop()


def test_async_commit_stop_drains_queue():
    config = _new_async_config(data_queue_capacity=1)
    server = TrainerServer(config)
    learner_store = SlowQueuedDataStore(128)
    server.register_data_store("table1", learner_store)
    server.start(threaded=True)

    actor_store = QueuedDataStore(128)
    client = TrainerClient(
        "table1",
        "127.0.0.1",
        config,
        data_store=actor_store,
        wait_for_server=True,
    )

    try:
        for idx in range(12):
            actor_store.insert(np.array([idx] * 6))

        assert client.update()
        server.stop()
        assert learner_store.latest_data_id() == actor_store.latest_data_id()
        assert server.committed_update_id_map["table1"] == actor_store.latest_data_id()
    finally:
        client.stop()


def test_async_commit_unknown_store_update_fails_cleanly():
    config = _new_async_config()
    server = TrainerServer(config)
    server.register_data_store("table1", QueuedDataStore(32))
    server.start(threaded=True)

    actor_store = QueuedDataStore(32)
    client = TrainerClient(
        "missing_table",
        "127.0.0.1",
        config,
        data_store=actor_store,
        wait_for_server=True,
        timeout_ms=200,
    )

    try:
        actor_store.insert(np.array([1, 2, 3]))
        assert client.update() is False
    finally:
        client.stop()
        server.stop()
