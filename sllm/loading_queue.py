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
Model Loading Queue for ServerlessLLM.

Provides per-node loading queues with concurrency control and storage level
tracking to enable storage-aware scheduling.

Key Features:
1. Per-node queues with configurable concurrency limits
2. Storage level tracking (DISK, MEMORY, GPU)
3. Request deduplication (same model+level+node)
4. Queue state visibility for scheduler
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

from sllm.logger import init_logger

logger = init_logger(__name__)


class StorageLevel(Enum):
    """Storage level for model loading operations."""

    DISK = "disk"  # Model files on disk (via sllm-store save)
    MEMORY = "memory"  # Model loaded into sllm-store pinned memory pool
    GPU = "gpu"  # Model loaded onto GPU device memory


class LoadingStatus(Enum):
    """Status of a loading request."""

    PENDING = "pending"  # Waiting in queue
    LOADING = "loading"  # Currently being processed
    COMPLETED = "completed"  # Successfully loaded
    FAILED = "failed"  # Failed to load
    CANCELLED = "cancelled"  # Cancelled before completion


@dataclass
class LoadingRequest:
    """A request to load a model to a specific storage level on a node."""

    id: str
    model_name: str
    storage_level: StorageLevel
    node_name: str
    backend: str
    status: LoadingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failure_reason: Optional[str] = None

    @property
    def dedup_key(self) -> str:
        """Key for deduplicating requests.

        For DISK/MEMORY levels: dedup by model+level+node (same download/load once).
        For GPU level: each request is unique (multiple instances allowed).
        """
        if self.storage_level == StorageLevel.GPU:
            return f"{self.node_name}:{self.model_name}:{self.storage_level.value}:{self.id}"
        return f"{self.node_name}:{self.model_name}:{self.storage_level.value}"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        result = {
            "id": self.id,
            "model_name": self.model_name,
            "storage_level": self.storage_level.value,
            "node_name": self.node_name,
            "backend": self.backend,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }
        if self.started_at:
            result["started_at"] = self.started_at.isoformat()
            result["elapsed_seconds"] = (
                datetime.now() - self.started_at
            ).total_seconds()
        if self.completed_at:
            result["completed_at"] = self.completed_at.isoformat()
        if self.failure_reason:
            result["failure_reason"] = self.failure_reason
        return result


@dataclass
class NodeQueue:
    """Queue state for a single node."""

    node_name: str
    max_concurrent: int = 2
    queue: List[LoadingRequest] = field(default_factory=list)
    active: Dict[str, LoadingRequest] = field(default_factory=dict)

    def pending_count(self) -> int:
        """Number of pending requests."""
        return len(self.queue)

    def active_count(self) -> int:
        """Number of active (loading) requests."""
        return len(self.active)

    def can_start_more(self) -> bool:
        """Check if more loads can be started."""
        return len(self.active) < self.max_concurrent

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "node_name": self.node_name,
            "max_concurrent": self.max_concurrent,
            "pending_count": self.pending_count(),
            "active_count": self.active_count(),
            "pending": [r.to_dict() for r in self.queue],
            "active": [r.to_dict() for r in self.active.values()],
        }


# Type alias for load executor callback
LoadExecutor = Callable[[LoadingRequest], "asyncio.Future[bool]"]


class LoadingQueueManager:
    """
    Manages per-node loading queues for model operations.

    Responsibilities:
    1. Queue management (enqueue, dequeue, cancel)
    2. Per-node concurrency control
    3. Request deduplication
    4. State visibility for scheduler
    """

    def __init__(
        self,
        max_concurrent_per_node: int = 2,
        process_interval: float = 1.0,
    ):
        """
        Initialize LoadingQueueManager.

        Args:
            max_concurrent_per_node: Max simultaneous loads per node (default: 2)
            process_interval: Queue processing interval in seconds (default: 1.0)
        """
        self.max_concurrent_per_node = max_concurrent_per_node
        self.process_interval = process_interval

        # Per-node queues
        self._node_queues: Dict[str, NodeQueue] = {}

        # Deduplication index: dedup_key -> request
        self._pending_requests: Dict[str, LoadingRequest] = {}

        # Request lookup by ID
        self._requests_by_id: Dict[str, LoadingRequest] = {}

        # Completed requests (limited history for dedup)
        self._completed: Dict[str, LoadingRequest] = {}
        self._max_completed_history = 100

        # Synchronization
        self._lock = asyncio.Lock()

        # Background processor
        self._processor_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()

        # Track active load tasks for graceful cancellation
        self._active_tasks: Dict[str, asyncio.Task] = {}

        self._load_executor: Optional[LoadExecutor] = None

    def set_load_executor(self, executor: LoadExecutor):
        """Set the callback to execute load operations."""
        self._load_executor = executor

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self):
        """Start the background queue processor."""
        if self._processor_task is not None:
            logger.warning("LoadingQueueManager already started")
            return

        self._shutdown.clear()
        self._processor_task = asyncio.create_task(
            self._process_loop(), name="loading-queue-processor"
        )
        logger.info(
            f"LoadingQueueManager started "
            f"(max_concurrent={self.max_concurrent_per_node})"
        )

    async def stop(self):
        """Stop the background queue processor gracefully."""
        if self._processor_task is None:
            return

        logger.info("Stopping LoadingQueueManager...")
        self._shutdown.set()

        # Wait for processor loop to stop
        try:
            await asyncio.wait_for(self._processor_task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Queue processor did not stop in time, cancelling")
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        self._processor_task = None

        # Cancel any remaining active load tasks
        if self._active_tasks:
            logger.info(
                f"Cancelling {len(self._active_tasks)} active load tasks"
            )
            for task in self._active_tasks.values():
                task.cancel()
            await asyncio.gather(
                *self._active_tasks.values(), return_exceptions=True
            )
            self._active_tasks.clear()

        logger.info("LoadingQueueManager stopped")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def enqueue(
        self,
        model_name: str,
        backend: str,
        storage_level: StorageLevel,
        node_name: str,
    ) -> LoadingRequest:
        """
        Enqueue a loading request.

        For DISK/MEMORY levels: returns existing request if duplicate.
        For GPU level: always creates new request (multiple instances allowed).

        Args:
            model_name: Model to load
            backend: Backend type (vllm, sglang, etc.)
            storage_level: Target storage level
            node_name: Target node

        Returns:
            LoadingRequest (new or existing if duplicate for DISK/MEMORY)
        """
        async with self._lock:
            if storage_level != StorageLevel.GPU:
                dedup_key = f"{node_name}:{model_name}:{storage_level.value}"

                if dedup_key in self._pending_requests:
                    existing = self._pending_requests[dedup_key]
                    logger.debug(
                        f"Duplicate request for {model_name} on {node_name}, "
                        f"returning existing (id={existing.id})"
                    )
                    return existing

                if dedup_key in self._completed:
                    completed = self._completed[dedup_key]
                    if completed.status == LoadingStatus.COMPLETED:
                        logger.debug(
                            f"Model {model_name} already loaded on {node_name}"
                        )
                        return completed

            request = LoadingRequest(
                id=uuid.uuid4().hex[:12],
                model_name=model_name,
                storage_level=storage_level,
                node_name=node_name,
                backend=backend,
                status=LoadingStatus.PENDING,
                created_at=datetime.now(),
            )

            node_queue = self._get_or_create_node_queue(node_name)
            node_queue.queue.append(request)
            self._pending_requests[request.dedup_key] = request
            self._requests_by_id[request.id] = request

            logger.info(
                f"Enqueued load request {request.id}: "
                f"{model_name} -> {storage_level.value} on {node_name}"
            )

            return request

    async def cancel(self, request_id: str) -> bool:
        """
        Cancel a pending request.

        Cannot cancel requests that are already loading.

        Args:
            request_id: Request ID to cancel

        Returns:
            True if cancelled, False if not found or already loading
        """
        async with self._lock:
            request = self._requests_by_id.get(request_id)
            if not request:
                return False

            if request.status != LoadingStatus.PENDING:
                logger.warning(
                    f"Cannot cancel request {request_id}: "
                    f"status is {request.status.value}"
                )
                return False

            node_queue = self._node_queues.get(request.node_name)
            if node_queue and request in node_queue.queue:
                node_queue.queue.remove(request)

            request.status = LoadingStatus.CANCELLED
            request.completed_at = datetime.now()
            self._pending_requests.pop(request.dedup_key, None)
            self._move_to_completed(request)

            logger.info(f"Cancelled request {request_id}")
            return True

    async def get_request(self, request_id: str) -> Optional[LoadingRequest]:
        """Get request by ID."""
        async with self._lock:
            return self._requests_by_id.get(request_id) or self._completed.get(
                request_id
            )

    def get_node_queue_state(self, node_name: str) -> Optional[dict]:
        """Get current queue state for a node, or None if not found."""
        node_queue = self._node_queues.get(node_name)
        return node_queue.to_dict() if node_queue else None

    def get_all_queue_states(self) -> Dict[str, dict]:
        """Get queue states for all nodes."""
        return {
            node_name: nq.to_dict()
            for node_name, nq in self._node_queues.items()
        }

    def get_loading_models(self, node_name: str) -> List[str]:
        """Get models currently loading on a node."""
        node_queue = self._node_queues.get(node_name)
        if not node_queue:
            return []
        return [r.model_name for r in node_queue.active.values()]

    def get_pending_models(self, node_name: str) -> List[str]:
        """Get models pending to load on a node."""
        node_queue = self._node_queues.get(node_name)
        if not node_queue:
            return []
        return [r.model_name for r in node_queue.queue]

    def is_model_loading(
        self,
        model_name: str,
        storage_level: StorageLevel,
        node_name: str,
    ) -> bool:
        """Check if a model is pending or loading at the specified level.

        Note: Only works for DISK/MEMORY levels. For GPU level, use
        count_pending_gpu_requests() instead (GPU requests don't deduplicate).
        """
        return (
            f"{node_name}:{model_name}:{storage_level.value}"
            in self._pending_requests
        )

    def is_model_pending_or_loading(
        self,
        model_name: str,
        node_name: str,
    ) -> bool:
        """Check if a model is pending or loading on a node (any level)."""
        node_queue = self._node_queues.get(node_name)
        if not node_queue:
            return False

        for r in node_queue.queue:
            if r.model_name == model_name:
                return True
        for r in node_queue.active.values():
            if r.model_name == model_name:
                return True
        return False

    def count_pending_gpu_requests(self, model_name: str, backend: str) -> int:
        """Count pending/active GPU requests for a model across all nodes.

        Used by Reconciler to avoid over-enqueuing instance creation requests.
        """
        count = 0
        for node_queue in self._node_queues.values():
            for r in node_queue.queue:
                if (
                    r.model_name == model_name
                    and r.backend == backend
                    and r.storage_level == StorageLevel.GPU
                ):
                    count += 1
            for r in node_queue.active.values():
                if (
                    r.model_name == model_name
                    and r.backend == backend
                    and r.storage_level == StorageLevel.GPU
                ):
                    count += 1
        return count

    # -------------------------------------------------------------------------
    # Background Processor
    # -------------------------------------------------------------------------

    async def _process_loop(self):
        """Background loop to process all node queues."""
        logger.debug("Queue processor loop started")

        while not self._shutdown.is_set():
            try:
                await self._process_all_queues()
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")

            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self.process_interval
                )
                break
            except asyncio.TimeoutError:
                pass

        logger.debug("Queue processor loop stopped")

    async def _process_all_queues(self):
        """Process all node queues, starting loads where possible."""
        # Get node names under lock, then process each separately
        # to reduce lock contention with enqueue() calls
        async with self._lock:
            node_names = list(self._node_queues.keys())

        for node_name in node_names:
            async with self._lock:
                node_queue = self._node_queues.get(node_name)
                if node_queue:
                    await self._process_node_queue(node_queue)

    async def _process_node_queue(self, node_queue: NodeQueue):
        """Start new loads if under concurrency limit."""
        while node_queue.can_start_more() and node_queue.queue:
            request = node_queue.queue.pop(0)
            request.status = LoadingStatus.LOADING
            request.started_at = datetime.now()
            node_queue.active[request.dedup_key] = request

            logger.info(
                f"Starting load {request.id}: "
                f"{request.model_name} -> {request.storage_level.value} "
                f"on {request.node_name}"
            )

            task = asyncio.create_task(
                self._execute_load(request),
                name=f"load-{request.id}",
            )
            self._active_tasks[request.id] = task

    async def _execute_load(self, request: LoadingRequest):
        """Execute a load operation and update request status."""
        try:
            if self._load_executor is None:
                raise RuntimeError("No load executor configured")

            success = await self._load_executor(request)

            async with self._lock:
                if success:
                    request.status = LoadingStatus.COMPLETED
                    logger.info(
                        f"Load completed {request.id}: {request.model_name}"
                    )
                else:
                    request.status = LoadingStatus.FAILED
                    request.failure_reason = "Load executor returned False"
                    logger.error(
                        f"Load failed {request.id}: {request.model_name}"
                    )

        except Exception as e:
            async with self._lock:
                request.status = LoadingStatus.FAILED
                request.failure_reason = str(e)
                logger.error(
                    f"Load failed {request.id}: {request.model_name} - {e}"
                )

        finally:
            async with self._lock:
                request.completed_at = datetime.now()
                node_queue = self._node_queues.get(request.node_name)
                if node_queue:
                    node_queue.active.pop(request.dedup_key, None)
                self._pending_requests.pop(request.dedup_key, None)
                self._active_tasks.pop(request.id, None)
                self._move_to_completed(request)

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _get_or_create_node_queue(self, node_name: str) -> NodeQueue:
        """Get or create queue for a node."""
        if node_name not in self._node_queues:
            self._node_queues[node_name] = NodeQueue(
                node_name=node_name,
                max_concurrent=self.max_concurrent_per_node,
            )
        return self._node_queues[node_name]

    def _move_to_completed(self, request: LoadingRequest):
        """Move request to completed history."""
        self._completed[request.dedup_key] = request
        self._requests_by_id.pop(request.id, None)

        if len(self._completed) > self._max_completed_history:
            oldest_key = min(
                self._completed.keys(),
                key=lambda k: self._completed[k].completed_at or datetime.min,
            )
            del self._completed[oldest_key]

    def __repr__(self) -> str:
        total_pending = sum(
            nq.pending_count() for nq in self._node_queues.values()
        )
        total_active = sum(
            nq.active_count() for nq in self._node_queues.values()
        )
        return (
            f"LoadingQueueManager(nodes={len(self._node_queues)}, "
            f"pending={total_pending}, active={total_active})"
        )
