# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
"""
Single Global Router for ServerlessLLM v1-beta.

Design principles (from docs/v1-beta-scalable-router-design.md):
- KISS: Single global router, no per-model processes
- YAGNI: No caching, no circuit breakers until needed
- Explicit: Router reads SQLite directly, no sync loops
- Stateless: Ephemeral state only; crash loses buffered requests (503)
- Separation: Treat router as separate component for future scaling

The Router:
- Reads endpoints from SQLite on every request (no cache)
- Round-robin load balancing across healthy endpoints
- Cold-start buffering (ephemeral, lost on crash)
- Pushes metrics directly to Autoscaler
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import aiohttp

from sllm.database import Database
from sllm.logger import init_logger

if TYPE_CHECKING:
    from sllm.autoscaler import AutoScaler

logger = init_logger(__name__)


@dataclass
class RouterConfig:
    """Router configuration."""

    max_buffer_size: int = 10
    cold_start_timeout: float = 120.0
    request_timeout: float = 300.0
    retry_failed_endpoint: bool = True


@dataclass
class BufferedRequest:
    """A request waiting in the cold-start buffer."""

    model_name: str
    backend: str
    payload: Dict[str, Any]
    path: str
    future: asyncio.Future = field(default_factory=asyncio.Future)


class Router:
    """
    Single global router for all models.

    Reads endpoints from SQLite on every request, provides round-robin
    load balancing, cold-start buffering, and pushes metrics to autoscaler.

    Ephemeral state (lost on restart):
    - Round-robin index per model
    - Cold-start buffer per model
    - In-flight counter per model
    """

    def __init__(
        self,
        database: Database,
        config: Optional[RouterConfig] = None,
        autoscaler: Optional["AutoScaler"] = None,
    ):
        """
        Initialize the Router.

        Args:
            database: Database instance for reading endpoints
            config: Router configuration
            autoscaler: Optional autoscaler for metrics push
        """
        self.database = database
        self.config = config or RouterConfig()

        self._round_robin_idx: Dict[tuple, int] = defaultdict(int)
        self._round_robin_indices = self._round_robin_idx
        self._buffers: Dict[tuple, asyncio.Queue[BufferedRequest]] = {}
        self._in_flight: Dict[tuple, int] = defaultdict(int)

        # Autoscaler reference (set after initialization)
        self._autoscaler: Optional[AutoScaler] = autoscaler

        # HTTP session (created lazily)
        self._session: Optional[aiohttp.ClientSession] = None

        # Background task for buffer draining
        self._drain_task: Optional[asyncio.Task] = None
        self._shutdown = False

        logger.info("Router initialized")

    def set_autoscaler(self, autoscaler: AutoScaler):
        """Set the autoscaler for metrics push."""
        self._autoscaler = autoscaler

    @property
    def autoscaler(self) -> Optional["AutoScaler"]:
        """Get the autoscaler reference."""
        return self._autoscaler

    async def start(self):
        """Start the router background tasks."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(self._buffer_drain_loop())
        logger.info("Router started")

    async def stop(self):
        """Stop the router and clean up resources."""
        self._shutdown = True

        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Router stopped")

    # -------------------------------------------------------------------------
    # Request Handling
    # -------------------------------------------------------------------------

    async def handle_request(
        self,
        payload: Dict[str, Any],
        path: str = "/v1/chat/completions",
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle an inference request for a model.

        Reads endpoints from SQLite on every request, selects one via
        round-robin, and forwards the request. If no endpoints are
        available, buffers the request for cold-start.

        Args:
            payload: Request payload (JSON body)
            path: API path (e.g., "/v1/chat/completions")
            model_name: Model name
            backend: Backend type

        Returns:
            Response from the backend

        Raises:
            Exception: On timeout or forwarding failure
        """
        if model_name is None:
            model_name = payload.get("model")
        if backend is None:
            backend = payload.get("backend", "vllm")

        if not model_name:
            raise ValueError("Request must include a 'model' field")

        key = (model_name, backend)

        if self._session is None:
            self._session = aiohttp.ClientSession()

        endpoints = self.database.get_model_endpoints(model_name, backend)

        if endpoints:
            endpoint = self._select_next_endpoint(key, endpoints)
            return await self._forward_to_endpoint(
                key, endpoint, payload, path
            )
        else:
            return await self._buffer_and_wait(key, payload, path)

    def _select_next_endpoint(self, key: tuple, endpoints: List[str]) -> str:
        """Select the next endpoint using round-robin."""
        idx = self._round_robin_idx[key] % len(endpoints)
        self._round_robin_idx[key] += 1
        return endpoints[idx]

    async def _forward_to_endpoint(
        self,
        key: tuple,
        endpoint: str,
        payload: Dict[str, Any],
        path: str,
    ) -> Dict[str, Any]:
        """Forward request to a specific endpoint."""
        url = f"http://{endpoint}{path}"

        self._in_flight[key] += 1
        self._push_metrics(key)

        try:
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=self.config.request_timeout
                ),
            ) as resp:
                result = await resp.json()

                if resp.status >= 500:
                    logger.warning(
                        f"[{key}] Endpoint {endpoint} returned "
                        f"status {resp.status}"
                    )

                return result

        except aiohttp.ClientError as e:
            logger.error(f"[{key}] Request to {endpoint} failed: {e}")

            if self.config.retry_failed_endpoint:
                endpoints = self.database.get_model_endpoints(key[0], key[1])
                other_endpoints = [ep for ep in endpoints if ep != endpoint]
                if other_endpoints:
                    other_endpoint = self._select_next_endpoint(
                        key, other_endpoints
                    )
                    logger.info(
                        f"[{key}] Retrying on endpoint {other_endpoint}"
                    )
                    self._in_flight[key] -= 1
                    return await self._forward_to_endpoint(
                        key, other_endpoint, payload, path
                    )

            raise Exception(f"Failed to forward request: {e}")

        except asyncio.TimeoutError:
            logger.error(f"[{key}] Request to {endpoint} timed out")
            raise Exception(
                f"Request timeout after {self.config.request_timeout}s"
            )

        finally:
            self._in_flight[key] -= 1
            self._push_metrics(key)

    async def _buffer_and_wait(
        self,
        key: tuple,
        payload: Dict[str, Any],
        path: str,
    ) -> Dict[str, Any]:
        """Buffer request during cold start and wait for result."""
        if key not in self._buffers:
            self._buffers[key] = asyncio.Queue(
                maxsize=self.config.max_buffer_size
            )

        buffer = self._buffers[key]

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request = BufferedRequest(
            model_name=key[0],
            backend=key[1],
            payload=payload,
            path=path,
            future=future,
        )

        try:
            buffer.put_nowait(request)
            logger.info(
                f"[{key}] Buffered request (buffer size: {buffer.qsize()})"
            )
            self._push_metrics(key)
        except asyncio.QueueFull:
            logger.warning(f"[{key}] Buffer full, rejecting request")
            raise Exception("Service overloaded - buffer full")

        try:
            return await asyncio.wait_for(
                future, timeout=self.config.cold_start_timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[{key}] Cold start timeout after "
                f"{self.config.cold_start_timeout}s"
            )
            raise Exception(
                f"Cold start timeout - no instance available after "
                f"{self.config.cold_start_timeout}s"
            )

    async def _buffer_drain_loop(self):
        """Background task to drain buffers when endpoints become available."""
        logger.debug("Buffer drain loop started")

        while not self._shutdown:
            try:
                for key, buffer in list(self._buffers.items()):
                    if not buffer.empty():
                        endpoints = self.database.get_model_endpoints(
                            key[0], key[1]
                        )
                        if endpoints:
                            try:
                                request = buffer.get_nowait()
                                endpoint = self._select_next_endpoint(
                                    key, endpoints
                                )
                                logger.info(
                                    f"[{key}] Draining buffered request "
                                    f"to {endpoint}"
                                )
                                try:
                                    result = await self._forward_to_endpoint(
                                        key,
                                        endpoint,
                                        request.payload,
                                        request.path,
                                    )
                                    if not request.future.done():
                                        request.future.set_result(result)
                                except Exception as e:
                                    if not request.future.done():
                                        request.future.set_exception(e)
                            except asyncio.QueueEmpty:
                                pass

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in drain loop: {e}")
                await asyncio.sleep(1)

        logger.debug("Buffer drain loop stopped")

    def _push_metrics(self, key: tuple):
        """Push metrics immediately to autoscaler."""
        if self._autoscaler:
            buffer = self._buffers.get(key)
            buffer_len = buffer.qsize() if buffer else 0
            in_flight = self._in_flight.get(key, 0)

            self._autoscaler.receive_metrics(
                model_name=key[0],
                backend=key[1],
                buffer_len=buffer_len,
                in_flight=in_flight,
            )

    # -------------------------------------------------------------------------
    # Metrics Access (for status endpoints)
    # -------------------------------------------------------------------------

    def get_buffer_length(self, model_name: str, backend: str) -> int:
        """Get the buffer length for a model."""
        key = (model_name, backend)
        buffer = self._buffers.get(key)
        return buffer.qsize() if buffer else 0

    def get_in_flight(self, model_name: str, backend: str) -> int:
        """Get the in-flight count for a model."""
        key = (model_name, backend)
        return self._in_flight.get(key, 0)

    def get_total_demand(self, model_name: str, backend: str) -> int:
        """Get total demand (buffer + in-flight) for a model."""
        return (
            self.get_buffer_length(model_name, backend)
            + self.get_in_flight(model_name, backend)
        )

    def get_in_flight_count(self, model_name: str, backend: str) -> int:
        """Alias for get_in_flight (for test compatibility)."""
        return self.get_in_flight(model_name, backend)

    def _select_endpoint(self, model_name: str, backend: str) -> Optional[str]:
        """Select an endpoint using round-robin (sync version for tests)."""
        endpoints = self.database.get_model_endpoints(model_name, backend)
        if not endpoints:
            return None
        key = (model_name, backend)
        return self._select_next_endpoint(key, endpoints)

    def get_endpoint_count(self, model_name: str, backend: str) -> int:
        """Get the number of healthy endpoints for a model (from SQLite)."""
        endpoints = self.database.get_model_endpoints(model_name, backend)
        return len(endpoints)

    # -------------------------------------------------------------------------
    # Draining for Shutdown
    # -------------------------------------------------------------------------

    async def drain(self, timeout: float = 30.0):
        """
        Drain all pending requests before shutdown.

        Waits for in-flight requests to complete and buffers to empty.
        """
        logger.info("Draining router...")

        start_time = asyncio.get_event_loop().time()
        all_drained = False
        while asyncio.get_event_loop().time() - start_time < timeout:
            all_keys = set(self._buffers.keys()) | set(self._in_flight.keys())

            all_drained = True
            for key in all_keys:
                if (
                    self.get_buffer_length(key[0], key[1]) > 0
                    or self.get_in_flight(key[0], key[1]) > 0
                ):
                    all_drained = False
                    break

            if all_drained:
                break

            await asyncio.sleep(0.05)

        if not all_drained:
            logger.warning("Router drain timeout")
        else:
            logger.info("Router drain complete")

    def __repr__(self) -> str:
        total_buffer = sum(
            self.get_buffer_length(k[0], k[1]) for k in self._buffers.keys()
        )
        total_inflight = sum(self._in_flight.values())
        return f"Router(buffer={total_buffer}, in_flight={total_inflight})"


# Global router instance
_router: Optional[Router] = None


def get_router() -> Optional[Router]:
    """Get the global Router instance."""
    return _router


def init_router(
    database: Database,
    config: Optional[RouterConfig] = None,
) -> Router:
    """Initialize the global Router instance."""
    global _router
    _router = Router(database=database, config=config)
    return _router
