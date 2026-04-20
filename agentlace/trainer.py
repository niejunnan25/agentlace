#!/usr/bin/env python3

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from typing_extensions import Literal
from typing_extensions import Protocol

from agentlace.data.data_store import DataStoreBase
from agentlace.internal.trainer_transport import ASYNC_COMMIT_MODE
from agentlace.internal.trainer_transport import SYNC_COMMIT_MODE
from agentlace.internal.trainer_transport import build_trainer_client
from agentlace.internal.trainer_transport import build_trainer_server
from agentlace.internal.trainer_transport import validate_transport_mode


@dataclass
class TrainerConfig:
    """
    Configuration for the edge server and client.
    NOTE: Client and server should have the same config.
    """

    port_number: int = 5555
    broadcast_port: int = 5556
    request_types: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None
    version: str = "0.0.2"
    experimental_pipeline_port: Optional[str] = None
    transport_mode: Literal["sync_commit", "async_commit"] = SYNC_COMMIT_MODE
    data_port: Optional[int] = None
    control_timeout_ms: int = 800
    data_queue_capacity: int = 8
    data_socket_hwm: int = 8
    commit_poll_ms: int = 20


class DataCallback(Protocol):
    def __call__(self, store_name, payload: dict):
        ...


class RequestCallback(Protocol):
    def __call__(self, type: str, payload: dict) -> dict:
        ...


class TrainerServer:
    def __init__(
        self,
        config: TrainerConfig,
        data_callback: Optional[DataCallback] = None,
        request_callback: Optional[RequestCallback] = None,
        log_level=logging.INFO,
    ):
        transport_mode = validate_transport_mode(config.transport_mode)
        if transport_mode == ASYNC_COMMIT_MODE:
            if config.experimental_pipeline_port is not None:
                raise ValueError(
                    "experimental_pipeline_port is only supported in sync_commit mode"
                )
            if config.data_port is None:
                raise ValueError("async_commit transport requires data_port")

        self.queue = deque()
        self.request_types = set(config.request_types)
        self.data_stores: Dict[str, DataStoreBase] = {}
        self.accepted_update_id_map: Dict[str, int] = {}
        self.committed_update_id_map: Dict[str, int] = {}
        self.last_update_id_map = self.committed_update_id_map
        self.config = config
        self.data_callback = data_callback
        self._transport = build_trainer_server(
            config=config,
            data_stores=self.data_stores,
            accepted_update_id_map=self.accepted_update_id_map,
            committed_update_id_map=self.committed_update_id_map,
            data_callback=data_callback,
            request_callback=request_callback,
            log_level=log_level,
        )
        self.req_rep_server = self._transport.req_rep_server
        self.broadcast_server = self._transport.broadcast_server
        self.consumer = getattr(self._transport, "consumer", None)
        self.pull_server = getattr(self._transport, "pull_server", None)
        self.worker_thread = getattr(self._transport, "worker_thread", None)

        logging.basicConfig(level=log_level)
        logging.debug(
            "Trainer server is listening on port %s", int(config.port_number)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transport, name)

    def register_data_store(self, name, data_store: DataStoreBase):
        self._transport.register_data_store(name, data_store)

    def data_store(self, name) -> Optional[DataStoreBase]:
        return self._transport.data_store(name)

    def store_names(self) -> Set[str]:
        return self._transport.store_names()

    def publish_network(self, payload: dict):
        self._transport.publish_network(payload)

    def start(self, threaded: bool = False):
        self._transport.start(threaded=threaded)
        self.thread = getattr(self.req_rep_server, "thread", None) if threaded else None

    def stop(self):
        self._transport.stop()

    def get_transport_status(self, store_name: str) -> dict[str, Any]:
        return self._transport.get_transport_status(store_name)


class TrainerClient:
    def __init__(
        self,
        name: str,
        server_ip: str,
        config: TrainerConfig,
        data_store: DataStoreBase = None,
        data_stores: Dict[str, DataStoreBase] = {},
        log_level=logging.INFO,
        wait_for_server: bool = False,
        timeout_ms: Optional[float] = None,
    ):
        transport_mode = validate_transport_mode(config.transport_mode)
        if transport_mode == ASYNC_COMMIT_MODE:
            if config.experimental_pipeline_port is not None:
                raise ValueError(
                    "experimental_pipeline_port is only supported in sync_commit mode"
                )
            if config.data_port is None:
                raise ValueError("async_commit transport requires data_port")

        effective_timeout_ms = (
            int(config.control_timeout_ms) if timeout_ms is None else int(timeout_ms)
        )
        self._transport = build_trainer_client(
            name=name,
            server_ip=server_ip,
            config=config,
            data_store=data_store,
            data_stores=dict(data_stores),
            log_level=log_level,
            wait_for_server=wait_for_server,
            timeout_ms=effective_timeout_ms,
        )

        logging.basicConfig(level=log_level)
        logging.debug(
            "Initiated trainer client at %s:%s", server_ip, int(config.port_number)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transport, name)

    def update(self) -> bool:
        return self._transport.update()

    def update_datastore(
        self,
        name: str,
        from_id: int,
        confirm_update: bool = False,
    ) -> bool:
        return self._transport.update_datastore(
            name,
            from_id,
            confirm_update=confirm_update,
        )

    def get_server_last_update_id(self, name: str) -> Optional[int]:
        return self._transport.get_server_last_update_id(name)

    def get_server_accepted_update_id(self, name: str) -> Optional[int]:
        return self._transport.get_server_accepted_update_id(name)

    def get_server_committed_update_id(self, name: str) -> Optional[int]:
        return self._transport.get_server_committed_update_id(name)

    def wait_until_committed(
        self,
        name: Optional[str] = None,
        target_id: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> bool:
        return self._transport.wait_until_committed(
            name=name,
            target_id=target_id,
            timeout_s=timeout_s,
        )

    def get_transport_status(self, name: Optional[str] = None) -> dict[str, Any]:
        return self._transport.get_transport_status(name)

    def request(self, type: str, payload: dict) -> Optional[dict]:
        return self._transport.request(type, payload)

    def recv_network_callback(self, callback):
        self._transport.recv_network_callback(callback)

    def start_async_update(self, interval: int = 10):
        self._transport.start_async_update(interval=interval)

    def stop(self):
        self._transport.stop()


class TrainerSMInterface:
    """
    Utilized shared-memory to recreate the transport layer interface
    of TrainerServer and TrainerClient.
    """

    def __init__(self):
        self._recv_network_fn = None
        self._req_callback_fn = None
        self._data_stores: Dict[str, DataStoreBase] = {}

    def recv_network_callback(self, callback_fn):
        self._recv_network_fn = callback_fn

    def publish_network(self, params: dict):
        if self._recv_network_fn:
            self._recv_network_fn(params)

    def start(self, *args, **kwargs):
        pass

    def stop(self):
        pass

    def update(self):
        pass

    def register_request_callback(self, callback_fn):
        self._req_callback_fn = callback_fn

    def register_data_store(self, name: str, data_store: DataStoreBase):
        self._data_stores[str(name)] = data_store

    def data_store(self, name: str) -> Optional[DataStoreBase]:
        return self._data_stores.get(str(name))

    def store_names(self) -> Set[str]:
        return set(self._data_stores.keys())

    def _store_latest_id(self, name: Optional[str]) -> int:
        if name is None and len(self._data_stores) == 1:
            data_store = next(iter(self._data_stores.values()))
            return int(data_store.latest_data_id())
        if name is None:
            return -1
        data_store = self._data_stores.get(str(name))
        return -1 if data_store is None else int(data_store.latest_data_id())

    def request(self, type: str, payload: dict) -> Optional[dict]:
        if self._req_callback_fn:
            return self._req_callback_fn(type, payload)
        return None

    def start_async_update(self, interval: int = 10):
        pass

    def get_server_last_update_id(self, name: str) -> int:
        return self.get_server_committed_update_id(name)

    def get_server_accepted_update_id(self, name: Optional[str] = None) -> int:
        return self._store_latest_id(name)

    def get_server_committed_update_id(self, name: Optional[str] = None) -> int:
        return self._store_latest_id(name)

    def wait_until_committed(
        self,
        name: Optional[str] = None,
        target_id: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> bool:
        del name, target_id, timeout_s
        return True

    def get_transport_status(self, name: Optional[str] = None) -> dict[str, Any]:
        store_name = "" if name is None else str(name)
        latest_id = int(self._store_latest_id(name))
        return {
            "transport_mode": SYNC_COMMIT_MODE,
            "store_name": store_name,
            "accepted_update_id": latest_id,
            "committed_update_id": latest_id,
            "transport_backlog": 0,
            "data_queue_depth": 0,
            "local_latest_data_id": latest_id,
            "local_acked_update_id": latest_id,
            "local_pending_update_id": None,
            "last_sent_id": latest_id,
        }
