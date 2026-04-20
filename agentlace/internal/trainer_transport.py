from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, Optional, Set

import zmq

from agentlace.data.data_store import DataStoreBase
from agentlace.internal.utils import compute_hash
from agentlace.internal.utils import make_compression_method
from agentlace.zmq_wrapper.broadcast import BroadcastClient
from agentlace.zmq_wrapper.broadcast import BroadcastServer
from agentlace.zmq_wrapper.pipeline import Consumer
from agentlace.zmq_wrapper.pipeline import Producer

SYNC_COMMIT_MODE = "sync_commit"
ASYNC_COMMIT_MODE = "async_commit"
SUPPORTED_TRANSPORT_MODES = (SYNC_COMMIT_MODE, ASYNC_COMMIT_MODE)

_COMPRESS, _DECOMPRESS = make_compression_method("lz4")


def validate_transport_mode(mode: Any) -> str:
    raw_mode = str(mode)
    if raw_mode not in SUPPORTED_TRANSPORT_MODES:
        allowed = ", ".join(repr(name) for name in SUPPORTED_TRANSPORT_MODES)
        raise ValueError(
            f"Unsupported trainer transport mode: {raw_mode!r}. Allowed values: {allowed}"
        )
    return raw_mode


def build_config_json(config: Any) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _serialize_message(message: Any) -> bytes:
    return _COMPRESS(message)


def _deserialize_message(payload: bytes) -> Any:
    return _DECOMPRESS(payload)


def _close_zmq_socket(socket_obj) -> None:
    if socket_obj is None:
        return
    try:
        socket_obj.close(linger=0)
    except Exception:
        try:
            socket_obj.close()
        except Exception:
            pass


def _close_broadcast_endpoint(endpoint: Any) -> None:
    socket_obj = getattr(endpoint, "socket", None)
    context_obj = getattr(endpoint, "context", None)
    _close_zmq_socket(socket_obj)
    if context_obj is not None:
        try:
            context_obj.term()
        except Exception:
            pass


def _store_status_payload(
    *,
    store_name: str,
    transport_mode: str,
    accepted_update_id: int,
    committed_update_id: int,
    data_queue_depth: int,
) -> Dict[str, Any]:
    accepted = int(accepted_update_id)
    committed = int(committed_update_id)
    return {
        "transport_mode": str(transport_mode),
        "store_name": str(store_name),
        "accepted_update_id": accepted,
        "committed_update_id": committed,
        "transport_backlog": int(max(0, accepted - committed)),
        "data_queue_depth": int(max(0, data_queue_depth)),
    }


class _ReqRepServer:
    def __init__(
        self,
        *,
        port: int,
        callback: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        self._callback = callback
        self._stop_event = threading.Event()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{int(port)}")
        self.thread: Optional[threading.Thread] = None

    def _serve(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._stop_event.is_set():
                    break
                continue
            try:
                request = _deserialize_message(payload)
                response = self._callback(dict(request))
            except Exception as exc:  # noqa: BLE001
                response = {"success": False, "message": str(exc)}
            try:
                self._socket.send(_serialize_message(response))
            except zmq.ZMQError:
                if self._stop_event.is_set():
                    break

    def start(self, threaded: bool = False) -> None:
        if threaded:
            self.thread = threading.Thread(target=self._serve, daemon=True)
            self.thread.start()
            return
        self._serve()

    def stop(self) -> None:
        self._stop_event.set()
        _close_zmq_socket(self._socket)
        try:
            self._context.term()
        except Exception:
            pass
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)


class _ReqRepClient:
    def __init__(
        self,
        *,
        server_ip: str,
        port: int,
        timeout_ms: int,
    ) -> None:
        self.ip = str(server_ip)
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self._lock = threading.Lock()
        self.context: Optional[zmq.Context] = None
        self.socket = None
        self.reset_socket()

    def reset_socket(self) -> None:
        if self.socket is not None:
            _close_zmq_socket(self.socket)
        if self.context is not None:
            try:
                self.context.term()
            except Exception:
                pass
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, int(self.timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(self.timeout_ms))
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.ip}:{self.port}")

    def send_msg(
        self,
        request: Dict[str, Any],
        wait_for_response: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if self.socket is None or self.socket.closed:
            return None
        serialized = _serialize_message(request)
        with self._lock:
            try:
                self.socket.send(serialized)
                if not wait_for_response:
                    return None
                response = self.socket.recv()
                return dict(_deserialize_message(response))
            except Exception:
                self.reset_socket()
                return None

    def close(self) -> None:
        with self._lock:
            _close_zmq_socket(self.socket)
            if self.context is not None:
                try:
                    self.context.term()
                except Exception:
                    pass


class _PushClient:
    def __init__(
        self,
        *,
        server_ip: str,
        port: int,
        hwm: int,
        timeout_ms: int,
    ) -> None:
        self.timeout_ms = int(timeout_ms)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, int(hwm))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{str(server_ip)}:{int(port)}")
        self._lock = threading.Lock()

    def send_msg(self, message: Dict[str, Any]) -> bool:
        with self._lock:
            try:
                self._socket.send(_serialize_message(message))
                return True
            except zmq.Again:
                return False
            except zmq.ZMQError:
                return False

    def close(self) -> None:
        with self._lock:
            _close_zmq_socket(self._socket)
            try:
                self._context.term()
            except Exception:
                pass


class _PullServer:
    def __init__(
        self,
        *,
        port: int,
        callback: Callable[[Dict[str, Any]], None],
        hwm: int,
    ) -> None:
        self._callback = callback
        self._stop_event = threading.Event()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.RCVHWM, int(hwm))
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{int(port)}")
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._stop_event.is_set():
                    break
                continue
            try:
                request = _deserialize_message(payload)
                self._callback(dict(request))
            except Exception:
                continue

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        _close_zmq_socket(self._socket)
        try:
            self._context.term()
        except Exception:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


class _TrainerServerTransportBase:
    def __init__(
        self,
        *,
        config: Any,
        data_stores: Dict[str, DataStoreBase],
        accepted_update_id_map: Dict[str, int],
        committed_update_id_map: Dict[str, int],
        data_callback: Optional[Callable[[str, Dict[str, Any]], Any]],
        request_callback: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]],
        log_level: int,
    ) -> None:
        self.config = config
        self.data_stores = data_stores
        self.accepted_update_id_map = accepted_update_id_map
        self.committed_update_id_map = committed_update_id_map
        self.data_callback = data_callback
        self.request_callback = request_callback
        self.request_types = set(config.request_types)
        self.log_level = int(log_level)
        self.broadcast_server = BroadcastServer(
            int(config.broadcast_port),
            log_level=int(log_level),
        )
        self.req_rep_server: Optional[_ReqRepServer] = None

    def register_data_store(self, name: str, data_store: DataStoreBase) -> None:
        store_name = str(name)
        self.data_stores[store_name] = data_store
        self.accepted_update_id_map[store_name] = -1
        self.committed_update_id_map[store_name] = -1

    def data_store(self, name: str) -> Optional[DataStoreBase]:
        return self.data_stores.get(str(name))

    def store_names(self) -> Set[str]:
        return set(self.data_stores.keys())

    def publish_network(self, payload: Dict[str, Any]) -> None:
        self.broadcast_server.broadcast(payload)

    def _custom_request(self, request_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.request_callback(request_type, payload) if self.request_callback else {}


class _SyncCommitTrainerServer(_TrainerServerTransportBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.consumer: Optional[Consumer] = None
        self.req_rep_server = _ReqRepServer(
            port=int(self.config.port_number),
            callback=self._control_callback,
        )
        if self.config.experimental_pipeline_port:
            self.consumer = Consumer(
                self._pipeline_callback,
                int(self.config.experimental_pipeline_port),
            )

    def _insert_payload(self, store_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if store_name not in self.data_stores:
            return {"success": False, "message": "Invalid datastore name"}
        last_update_id = int(payload.get("last_id", -1))
        batch_data = payload.get("data", [])
        self.data_stores[store_name].batch_insert(batch_data)
        if self.data_callback:
            self.data_callback(store_name, payload)
        self.accepted_update_id_map[store_name] = last_update_id
        self.committed_update_id_map[store_name] = last_update_id
        return {"success": True}

    def _pipeline_callback(self, data: Dict[str, Any]) -> None:
        store_name = str(data.get("store_name", ""))
        payload = dict(data.get("payload", {}) or {})
        self._insert_payload(store_name, payload)

    def _control_callback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        request_type = str(data.get("type", ""))
        payload = dict(data.get("payload", {}) or {})
        if request_type == "datastore":
            return self._insert_payload(str(data.get("store_name", "")), payload)
        if request_type == "get_last_update_id":
            store_name = str(payload.get("store_name", ""))
            if store_name not in self.data_stores:
                return {"success": False, "message": "Invalid datastore name"}
            return {
                "success": True,
                "payload": int(self.committed_update_id_map.get(store_name, -1)),
            }
        if request_type == "get_accepted_update_id":
            store_name = str(payload.get("store_name", ""))
            if store_name not in self.data_stores:
                return {"success": False, "message": "Invalid datastore name"}
            return {
                "success": True,
                "payload": int(self.accepted_update_id_map.get(store_name, -1)),
            }
        if request_type == "get_committed_update_id":
            store_name = str(payload.get("store_name", ""))
            if store_name not in self.data_stores:
                return {"success": False, "message": "Invalid datastore name"}
            return {
                "success": True,
                "payload": int(self.committed_update_id_map.get(store_name, -1)),
            }
        if request_type == "get_transport_status":
            store_name = str(payload.get("store_name", ""))
            if store_name not in self.data_stores:
                return {"success": False, "message": "Invalid datastore name"}
            return {"success": True, "payload": self.get_transport_status(store_name)}
        if request_type == "hash":
            return {"success": True, "payload": build_config_json(self.config)}
        if request_type in self.request_types:
            return self._custom_request(request_type, payload)
        return {"success": False, "message": "Invalid type or payload"}

    def start(self, threaded: bool = False) -> None:
        if self.consumer is not None:
            self.consumer.async_start()
        self.req_rep_server.start(threaded=bool(threaded))

    def stop(self) -> None:
        if self.consumer is not None:
            try:
                self.consumer.stop()
            except Exception:
                pass
        self.req_rep_server.stop()
        _close_broadcast_endpoint(self.broadcast_server)

    def get_transport_status(self, store_name: str) -> Dict[str, Any]:
        target_name = str(store_name)
        return _store_status_payload(
            store_name=target_name,
            transport_mode=SYNC_COMMIT_MODE,
            accepted_update_id=int(self.accepted_update_id_map.get(target_name, -1)),
            committed_update_id=int(self.committed_update_id_map.get(target_name, -1)),
            data_queue_depth=0,
        )


class _AsyncCommitTrainerServer(_TrainerServerTransportBase):
    def __init__(
        self,
        *,
        data_queue_capacity: int,
        data_socket_hwm: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._progress_lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, int, list[Any], Dict[str, Any]]] = queue.Queue(
            maxsize=int(data_queue_capacity)
        )
        self._stop_event = threading.Event()
        self.req_rep_server = _ReqRepServer(
            port=int(self.config.port_number),
            callback=self._control_callback,
        )
        self.pull_server = _PullServer(
            port=int(self.config.data_port),
            callback=self._data_callback,
            hwm=int(data_socket_hwm),
        )
        self.worker_thread = threading.Thread(
            target=self._commit_worker,
            daemon=True,
            name="agentlace-async-commit-worker",
        )

    def _control_callback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        request_type = str(data.get("type", ""))
        payload = dict(data.get("payload", {}) or {})
        if request_type == "hash":
            return {"success": True, "payload": build_config_json(self.config)}
        if request_type == "get_last_update_id":
            store_name = str(payload.get("store_name", ""))
            if store_name not in self.data_stores:
                return {"success": False, "message": "Invalid datastore name"}
            return {
                "success": True,
                "payload": int(self.committed_update_id_map.get(store_name, -1)),
            }
        if request_type == "get_accepted_update_id":
            store_name = str(payload.get("store_name", ""))
            if store_name not in self.data_stores:
                return {"success": False, "message": "Invalid datastore name"}
            return {
                "success": True,
                "payload": int(self.accepted_update_id_map.get(store_name, -1)),
            }
        if request_type == "get_committed_update_id":
            store_name = str(payload.get("store_name", ""))
            if store_name not in self.data_stores:
                return {"success": False, "message": "Invalid datastore name"}
            return {
                "success": True,
                "payload": int(self.committed_update_id_map.get(store_name, -1)),
            }
        if request_type == "get_transport_status":
            store_name = str(payload.get("store_name", ""))
            if store_name not in self.data_stores:
                return {"success": False, "message": "Invalid datastore name"}
            return {"success": True, "payload": self.get_transport_status(store_name)}
        if request_type in self.request_types:
            return self._custom_request(request_type, payload)
        return {"success": False, "message": "Invalid type or payload"}

    def _data_callback(self, message: Dict[str, Any]) -> None:
        if str(message.get("type", "")) != "datastore":
            return
        store_name = str(message.get("store_name", ""))
        payload = dict(message.get("payload", {}) or {})
        last_id = int(payload.get("last_id", -1))
        batch_data = list(payload.get("data", []) or [])
        if store_name not in self.data_stores:
            return
        with self._progress_lock:
            if last_id <= int(self.accepted_update_id_map.get(store_name, -1)):
                return
        while not self._stop_event.is_set():
            try:
                self._queue.put((store_name, last_id, batch_data, payload), timeout=0.1)
                with self._progress_lock:
                    self.accepted_update_id_map[store_name] = max(
                        int(self.accepted_update_id_map.get(store_name, -1)),
                        int(last_id),
                    )
                return
            except queue.Full:
                continue

    def _commit_worker(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                store_name, last_id, batch_data, payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.data_stores[store_name].batch_insert(batch_data)
                if self.data_callback:
                    self.data_callback(store_name, payload)
                with self._progress_lock:
                    self.committed_update_id_map[store_name] = max(
                        int(self.committed_update_id_map.get(store_name, -1)),
                        int(last_id),
                    )
            finally:
                self._queue.task_done()

    def start(self, threaded: bool = False) -> None:
        self.worker_thread.start()
        self.pull_server.start()
        self.req_rep_server.start(threaded=bool(threaded))

    def stop(self) -> None:
        self._stop_event.set()
        self.pull_server.stop()
        self.req_rep_server.stop()
        self._queue.join()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        _close_broadcast_endpoint(self.broadcast_server)

    def get_transport_status(self, store_name: str) -> Dict[str, Any]:
        target_name = str(store_name)
        with self._progress_lock:
            return _store_status_payload(
                store_name=target_name,
                transport_mode=ASYNC_COMMIT_MODE,
                accepted_update_id=int(self.accepted_update_id_map.get(target_name, -1)),
                committed_update_id=int(self.committed_update_id_map.get(target_name, -1)),
                data_queue_depth=int(self._queue.qsize()),
            )


class _TrainerClientTransportBase:
    def __init__(
        self,
        *,
        name: str,
        server_ip: str,
        config: Any,
        data_store: Optional[DataStoreBase],
        data_stores: Dict[str, DataStoreBase],
        log_level: int,
        wait_for_server: bool,
        timeout_ms: int,
    ) -> None:
        self.client_name = str(name)
        self.server_ip = str(server_ip)
        self.config = config
        self.request_types = set(config.request_types)
        self.data_store = data_store
        self.data_stores_map = dict(data_stores)
        self.log_level = int(log_level)
        self.last_request_time = 0.0
        self.update_thread: Optional[threading.Thread] = None
        self.stop_update_flag = threading.Event()
        self.broadcast_client: Optional[BroadcastClient] = None
        self.req_rep_client: Optional[_ReqRepClient] = None
        self._last_sent_id_map: Dict[str, int] = {}
        self._default_timeout_ms = int(timeout_ms)
        self._wait_for_server(wait_for_server=bool(wait_for_server), timeout_ms=int(timeout_ms))

    def _wait_for_server(self, *, wait_for_server: bool, timeout_ms: int) -> None:
        raise NotImplementedError

    def _store_map(self) -> Dict[str, DataStoreBase]:
        stores = dict(self.data_stores_map)
        if self.data_store is not None:
            stores.setdefault(self.client_name, self.data_store)
        return stores

    def _resolve_store_name(self, name: Optional[str]) -> str:
        return self.client_name if name is None else str(name)

    def _resolve_store(self, name: str) -> Optional[DataStoreBase]:
        if self.data_store is not None and str(name) == self.client_name:
            return self.data_store
        return self.data_stores_map.get(str(name))

    def _store_latest_id(self, name: str) -> int:
        store = self._resolve_store(name)
        if store is None:
            return -1
        return int(store.latest_data_id())

    def _request_allowed(self) -> bool:
        if self.config.rate_limit and time.time() - self.last_request_time < 1 / self.config.rate_limit:
            logging.warning("Rate limit exceeded")
            return False
        self.last_request_time = time.time()
        return True

    def _current_timeout_ms(self) -> int:
        return int(getattr(self.req_rep_client, "timeout_ms", self._default_timeout_ms))

    def recv_network_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self.broadcast_client = BroadcastClient(
            self.server_ip,
            int(self.config.broadcast_port),
            log_level=self.log_level,
        )
        self.broadcast_client.async_start(callback)

    def start_async_update(self, interval: int = 10) -> None:
        def _periodic_update() -> None:
            while not self.stop_update_flag.is_set():
                self.update()
                time.sleep(interval)

        if self.update_thread is None or not self.update_thread.is_alive():
            self.stop_update_flag.clear()
            self.update_thread = threading.Thread(target=_periodic_update, daemon=True)
            self.update_thread.start()

    def stop(self) -> None:
        if self.broadcast_client:
            self.broadcast_client.stop()
        if self.update_thread and self.update_thread.is_alive():
            self.stop_update_flag.set()
            self.update_thread.join()
        self._close()

    def _close(self) -> None:
        raise NotImplementedError


class _SyncCommitTrainerClient(_TrainerClientTransportBase):
    def __init__(self, **kwargs: Any) -> None:
        self.producer: Optional[Producer] = None
        super().__init__(**kwargs)
        if self.config.experimental_pipeline_port:
            self.producer = Producer(
                ip=self.server_ip,
                port=int(self.config.experimental_pipeline_port),
            )
        res = self.update()
        if not res:
            logging.error("Failed to get res when update server datastore")

    def _wait_for_server(self, *, wait_for_server: bool, timeout_ms: int) -> None:
        self.req_rep_client = _ReqRepClient(
            server_ip=self.server_ip,
            port=int(self.config.port_number),
            timeout_ms=int(timeout_ms),
        )
        response = self.req_rep_client.send_msg({"type": "hash"})
        while wait_for_server and response is None:
            logging.warning("Failed to connect to server, retrying...")
            time.sleep(2.0)
            response = self.req_rep_client.send_msg({"type": "hash"})
        if response is None:
            raise Exception("Failed to connect to server")
        config_json = build_config_json(self.config)
        if compute_hash(config_json) != compute_hash(response.get("payload")):
            raise Exception(
                "Incompatible config with hash with server. Please check the config of the server and client"
            )

    def update(self) -> bool:
        stores = self._store_map()
        if not stores:
            return False
        if self.data_store is not None and not self.data_stores_map:
            from_id = self.get_server_last_update_id(self.client_name)
            if from_id is None:
                return False
            return self.update_datastore(self.client_name, from_id)
        for name in stores:
            from_id = self.get_server_last_update_id(name)
            if from_id is None:
                return False
            if not self.update_datastore(name, from_id):
                return False
        return True

    def update_datastore(
        self,
        name: str,
        from_id: int,
        confirm_update: bool = False,
    ) -> bool:
        data_store = self._resolve_store(name)
        if data_store is None:
            logging.error(f"Datastore {name} not found")
            return False
        client_latest_id = int(data_store.latest_data_id())
        batch_data = data_store.get_latest_data(int(from_id))
        if len(batch_data) == 0:
            return True
        data_dict = {"data": batch_data, "last_id": client_latest_id}
        res = self._update_ds(str(name), data_dict)
        if confirm_update:
            server_last_id = self.get_server_committed_update_id(str(name))
            if server_last_id is None:
                return False
            return bool(int(server_last_id) == int(client_latest_id))
        if res is None or not res.get("success", False):
            logging.warning("Failed to get res when update server datastore")
            return False
        self._last_sent_id_map[str(name)] = int(client_latest_id)
        return True

    def _update_ds(self, name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        msg = {"type": "datastore", "store_name": str(name), "payload": data}
        if self.config.experimental_pipeline_port:
            self.producer.send_msg(msg)
            self._last_sent_id_map[str(name)] = int(data.get("last_id", -1))
            return {"success": True}
        if not self._request_allowed():
            return None
        return self.req_rep_client.send_msg(msg)

    def get_server_last_update_id(self, name: str) -> Optional[int]:
        response = self.req_rep_client.send_msg(
            {"type": "get_last_update_id", "payload": {"store_name": str(name)}}
        )
        if response is None or not response.get("success", False):
            logging.warning("Failed to get last update id")
            return None
        return int(response.get("payload", -1))

    def get_server_accepted_update_id(self, name: str) -> Optional[int]:
        response = self.req_rep_client.send_msg(
            {"type": "get_accepted_update_id", "payload": {"store_name": str(name)}}
        )
        if response is None or not response.get("success", False):
            return None
        return int(response.get("payload", -1))

    def get_server_committed_update_id(self, name: str) -> Optional[int]:
        response = self.req_rep_client.send_msg(
            {"type": "get_committed_update_id", "payload": {"store_name": str(name)}}
        )
        if response is None or not response.get("success", False):
            return None
        return int(response.get("payload", -1))

    def wait_until_committed(
        self,
        name: Optional[str] = None,
        target_id: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> bool:
        target_name = self._resolve_store_name(name)
        target = (
            int(self._last_sent_id_map.get(target_name, self._store_latest_id(target_name)))
            if target_id is None
            else int(target_id)
        )
        deadline = time.monotonic() + (
            float(self._current_timeout_ms()) / 1000.0 if timeout_s is None else float(timeout_s)
        )
        while time.monotonic() <= deadline:
            committed = self.get_server_committed_update_id(target_name)
            if committed is not None and int(committed) >= int(target):
                return True
            time.sleep(float(max(1, int(self.config.commit_poll_ms))) / 1000.0)
        return False

    def get_transport_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        target_name = self._resolve_store_name(name)
        response = self.req_rep_client.send_msg(
            {"type": "get_transport_status", "payload": {"store_name": target_name}}
        )
        payload = (
            dict(response.get("payload", {}))
            if response is not None and response.get("success", False)
            else {}
        )
        payload.update(
            {
                "local_latest_data_id": int(self._store_latest_id(target_name)),
                "local_acked_update_id": int(
                    self._last_sent_id_map.get(target_name, -1)
                ),
                "local_pending_update_id": None,
                "last_sent_id": int(self._last_sent_id_map.get(target_name, -1)),
            }
        )
        return payload

    def request(self, request_type: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if request_type not in self.request_types:
            return None
        if not self._request_allowed():
            return None
        return self.req_rep_client.send_msg({"type": str(request_type), "payload": payload})

    def _close(self) -> None:
        self.req_rep_client.close()
        if self.producer is not None:
            self.producer.close()


class _AsyncCommitTrainerClient(_TrainerClientTransportBase):
    def __init__(
        self,
        *,
        commit_poll_ms: int,
        data_socket_hwm: int,
        timeout_ms: int,
        **kwargs: Any,
    ) -> None:
        self._push_client: Optional[_PushClient] = None
        self._commit_poll_ms = max(1, int(commit_poll_ms))
        self._acked_id_map: Dict[str, int] = {}
        self._pending_message_map: Dict[str, Dict[str, Any]] = {}
        self._pending_last_id_map: Dict[str, int] = {}
        super().__init__(timeout_ms=timeout_ms, **kwargs)
        self._push_client = _PushClient(
            server_ip=self.server_ip,
            port=int(self.config.data_port),
            hwm=int(data_socket_hwm),
            timeout_ms=int(timeout_ms),
        )
        for store_name in self._store_map():
            accepted = self.get_server_accepted_update_id(store_name)
            self._acked_id_map[store_name] = -1 if accepted is None else int(accepted)
        res = self.update()
        if not res:
            logging.error("Failed to get res when update server datastore")

    def _wait_for_server(self, *, wait_for_server: bool, timeout_ms: int) -> None:
        self.req_rep_client = _ReqRepClient(
            server_ip=self.server_ip,
            port=int(self.config.port_number),
            timeout_ms=int(timeout_ms),
        )
        response = self.req_rep_client.send_msg({"type": "hash"})
        while wait_for_server and response is None:
            logging.warning("Failed to connect to server, retrying...")
            time.sleep(2.0)
            response = self.req_rep_client.send_msg({"type": "hash"})
        if response is None:
            raise Exception("Failed to connect to server")
        config_json = build_config_json(self.config)
        if compute_hash(config_json) != compute_hash(response.get("payload")):
            raise Exception(
                "Incompatible config with hash with server. Please check the config of the server and client"
            )

    def _target_last_sent_id(self, name: str) -> int:
        if name in self._pending_last_id_map:
            return int(self._pending_last_id_map[name])
        if name in self._last_sent_id_map:
            return int(self._last_sent_id_map[name])
        return int(self._acked_id_map.get(name, -1))

    def _refresh_accepted_id(self, name: str) -> Optional[int]:
        remote = self.get_server_accepted_update_id(name)
        if remote is None:
            return None
        self._acked_id_map[name] = max(int(self._acked_id_map.get(name, -1)), int(remote))
        if name in self._pending_last_id_map and int(self._acked_id_map[name]) >= int(
            self._pending_last_id_map[name]
        ):
            self._last_sent_id_map[name] = int(self._pending_last_id_map[name])
            self._pending_message_map.pop(name, None)
            self._pending_last_id_map.pop(name, None)
        return int(self._acked_id_map[name])

    def _send_pending_or_new(self, name: str, from_id: int) -> bool:
        if name in self._pending_message_map:
            return self._push_client.send_msg(self._pending_message_map[name])
        data_store = self._resolve_store(name)
        if data_store is None:
            logging.error(f"Datastore {name} not found")
            return False
        acked_id = int(self._acked_id_map.get(name, -1))
        local_latest_id = int(data_store.latest_data_id())
        if local_latest_id <= acked_id:
            return True
        effective_from_id = max(int(from_id), acked_id)
        batch_data = data_store.get_latest_data(effective_from_id)
        if len(batch_data) == 0:
            return True
        self._pending_last_id_map[name] = int(local_latest_id)
        self._pending_message_map[name] = {
            "type": "datastore",
            "store_name": str(name),
            "payload": {
                "last_id": int(local_latest_id),
                "batch_count": int(len(batch_data)),
                "data": list(batch_data),
            },
        }
        return self._push_client.send_msg(self._pending_message_map[name])

    def update(self) -> bool:
        stores = self._store_map()
        if not stores:
            return False
        if self.data_store is not None and not self.data_stores_map:
            acked_id = int(self._acked_id_map.get(self.client_name, -1))
            return self.update_datastore(self.client_name, acked_id)
        for name in stores:
            acked_id = int(self._acked_id_map.get(name, -1))
            if not self.update_datastore(name, acked_id):
                return False
        return True

    def update_datastore(
        self,
        name: str,
        from_id: int,
        confirm_update: bool = False,
    ) -> bool:
        target_name = str(name)
        self._refresh_accepted_id(target_name)
        if not self._send_pending_or_new(target_name, int(from_id)):
            return False
        deadline = time.monotonic() + (float(self._current_timeout_ms()) / 1000.0)
        target_id = int(self._pending_last_id_map.get(target_name, self._acked_id_map.get(target_name, -1)))
        while time.monotonic() <= deadline:
            accepted = self._refresh_accepted_id(target_name)
            if accepted is not None and int(accepted) >= int(target_id):
                if confirm_update:
                    return self.wait_until_committed(target_name, target_id)
                return True
            time.sleep(float(self._commit_poll_ms) / 1000.0)
        return False

    def get_server_last_update_id(self, name: str) -> Optional[int]:
        return self.get_server_committed_update_id(name)

    def get_server_accepted_update_id(self, name: str) -> Optional[int]:
        response = self.req_rep_client.send_msg(
            {"type": "get_accepted_update_id", "payload": {"store_name": str(name)}}
        )
        if response is None or not response.get("success", False):
            return None
        return int(response.get("payload", -1))

    def get_server_committed_update_id(self, name: str) -> Optional[int]:
        response = self.req_rep_client.send_msg(
            {"type": "get_committed_update_id", "payload": {"store_name": str(name)}}
        )
        if response is None or not response.get("success", False):
            return None
        return int(response.get("payload", -1))

    def wait_until_committed(
        self,
        name: Optional[str] = None,
        target_id: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> bool:
        target_name = self._resolve_store_name(name)
        deadline = time.monotonic() + (
            float(self._current_timeout_ms()) / 1000.0 if timeout_s is None else float(timeout_s)
        )
        target = (
            int(self._target_last_sent_id(target_name))
            if target_id is None
            else int(target_id)
        )
        while time.monotonic() <= deadline:
            committed = self.get_server_committed_update_id(target_name)
            if committed is not None and int(committed) >= int(target):
                return True
            time.sleep(float(self._commit_poll_ms) / 1000.0)
        return False

    def get_transport_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        target_name = self._resolve_store_name(name)
        response = self.req_rep_client.send_msg(
            {"type": "get_transport_status", "payload": {"store_name": target_name}}
        )
        payload = (
            dict(response.get("payload", {}))
            if response is not None and response.get("success", False)
            else {}
        )
        payload.update(
            {
                "local_latest_data_id": int(self._store_latest_id(target_name)),
                "local_acked_update_id": int(self._acked_id_map.get(target_name, -1)),
                "local_pending_update_id": (
                    None
                    if target_name not in self._pending_last_id_map
                    else int(self._pending_last_id_map[target_name])
                ),
                "last_sent_id": int(self._target_last_sent_id(target_name)),
            }
        )
        return payload

    def request(self, request_type: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if request_type not in self.request_types:
            return None
        if not self._request_allowed():
            return None
        return self.req_rep_client.send_msg({"type": str(request_type), "payload": payload})

    def _close(self) -> None:
        self.req_rep_client.close()
        self._push_client.close()


def build_trainer_server(
    *,
    config: Any,
    data_stores: Dict[str, DataStoreBase],
    accepted_update_id_map: Dict[str, int],
    committed_update_id_map: Dict[str, int],
    data_callback: Optional[Callable[[str, Dict[str, Any]], Any]],
    request_callback: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]],
    log_level: int,
) -> _TrainerServerTransportBase:
    transport_mode = validate_transport_mode(getattr(config, "transport_mode", SYNC_COMMIT_MODE))
    if transport_mode == ASYNC_COMMIT_MODE:
        return _AsyncCommitTrainerServer(
            config=config,
            data_stores=data_stores,
            accepted_update_id_map=accepted_update_id_map,
            committed_update_id_map=committed_update_id_map,
            data_callback=data_callback,
            request_callback=request_callback,
            data_queue_capacity=int(config.data_queue_capacity),
            data_socket_hwm=int(config.data_socket_hwm),
            log_level=int(log_level),
        )
    return _SyncCommitTrainerServer(
        config=config,
        data_stores=data_stores,
        accepted_update_id_map=accepted_update_id_map,
        committed_update_id_map=committed_update_id_map,
        data_callback=data_callback,
        request_callback=request_callback,
        log_level=int(log_level),
    )


def build_trainer_client(
    *,
    name: str,
    server_ip: str,
    config: Any,
    data_store: Optional[DataStoreBase],
    data_stores: Dict[str, DataStoreBase],
    log_level: int,
    wait_for_server: bool,
    timeout_ms: int,
) -> _TrainerClientTransportBase:
    transport_mode = validate_transport_mode(getattr(config, "transport_mode", SYNC_COMMIT_MODE))
    if transport_mode == ASYNC_COMMIT_MODE:
        return _AsyncCommitTrainerClient(
            name=name,
            server_ip=server_ip,
            config=config,
            data_store=data_store,
            data_stores=data_stores,
            log_level=int(log_level),
            wait_for_server=bool(wait_for_server),
            timeout_ms=int(timeout_ms),
            commit_poll_ms=int(config.commit_poll_ms),
            data_socket_hwm=int(config.data_socket_hwm),
        )
    return _SyncCommitTrainerClient(
        name=name,
        server_ip=server_ip,
        config=config,
        data_store=data_store,
        data_stores=data_stores,
        log_level=int(log_level),
        wait_for_server=bool(wait_for_server),
        timeout_ms=int(timeout_ms),
    )


__all__ = [
    "ASYNC_COMMIT_MODE",
    "SYNC_COMMIT_MODE",
    "SUPPORTED_TRANSPORT_MODES",
    "build_config_json",
    "build_trainer_client",
    "build_trainer_server",
    "validate_transport_mode",
]
