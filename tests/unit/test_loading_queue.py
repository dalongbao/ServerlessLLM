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
"""Tests for the model loading queue."""

import asyncio
from datetime import datetime

import pytest

from sllm.loading_queue import (
    LoadingQueueManager,
    LoadingRequest,
    LoadingStatus,
    NodeQueue,
    StorageLevel,
)


class TestStorageLevel:
    """Tests for StorageLevel enum."""

    def test_storage_level_values(self):
        """Test storage level values."""
        assert StorageLevel.DISK.value == "disk"
        assert StorageLevel.MEMORY.value == "memory"
        assert StorageLevel.GPU.value == "gpu"


class TestLoadingStatus:
    """Tests for LoadingStatus enum."""

    def test_loading_status_values(self):
        """Test loading status values."""
        assert LoadingStatus.PENDING.value == "pending"
        assert LoadingStatus.LOADING.value == "loading"
        assert LoadingStatus.COMPLETED.value == "completed"
        assert LoadingStatus.FAILED.value == "failed"
        assert LoadingStatus.CANCELLED.value == "cancelled"


class TestLoadingRequest:
    """Tests for LoadingRequest dataclass."""

    def test_loading_request_creation(self):
        """Test creating a loading request."""
        request = LoadingRequest(
            id="abc123",
            model_name="facebook/opt-125m",
            storage_level=StorageLevel.DISK,
            node_name="worker-0",
            backend="vllm",
            status=LoadingStatus.PENDING,
            created_at=datetime.now(),
        )

        assert request.id == "abc123"
        assert request.model_name == "facebook/opt-125m"
        assert request.storage_level == StorageLevel.DISK
        assert request.node_name == "worker-0"
        assert request.backend == "vllm"
        assert request.status == LoadingStatus.PENDING

    def test_dedup_key(self):
        """Test deduplication key generation."""
        request = LoadingRequest(
            id="abc123",
            model_name="facebook/opt-125m",
            storage_level=StorageLevel.DISK,
            node_name="worker-0",
            backend="vllm",
            status=LoadingStatus.PENDING,
            created_at=datetime.now(),
        )

        assert request.dedup_key == "worker-0:facebook/opt-125m:disk"

    def test_to_dict(self):
        """Test converting request to dictionary."""
        created = datetime.now()
        request = LoadingRequest(
            id="abc123",
            model_name="facebook/opt-125m",
            storage_level=StorageLevel.DISK,
            node_name="worker-0",
            backend="vllm",
            status=LoadingStatus.PENDING,
            created_at=created,
        )

        result = request.to_dict()

        assert result["id"] == "abc123"
        assert result["model_name"] == "facebook/opt-125m"
        assert result["storage_level"] == "disk"
        assert result["node_name"] == "worker-0"
        assert result["backend"] == "vllm"
        assert result["status"] == "pending"
        assert result["created_at"] == created.isoformat()

    def test_to_dict_with_elapsed_time(self):
        """Test to_dict includes elapsed time when loading."""
        request = LoadingRequest(
            id="abc123",
            model_name="facebook/opt-125m",
            storage_level=StorageLevel.DISK,
            node_name="worker-0",
            backend="vllm",
            status=LoadingStatus.LOADING,
            created_at=datetime.now(),
            started_at=datetime.now(),
        )

        result = request.to_dict()

        assert "started_at" in result
        assert "elapsed_seconds" in result


class TestNodeQueue:
    """Tests for NodeQueue dataclass."""

    def test_node_queue_creation(self):
        """Test creating a node queue."""
        queue = NodeQueue(node_name="worker-0", max_concurrent=2)

        assert queue.node_name == "worker-0"
        assert queue.max_concurrent == 2
        assert queue.pending_count() == 0
        assert queue.active_count() == 0
        assert queue.can_start_more()

    def test_can_start_more(self):
        """Test can_start_more check."""
        queue = NodeQueue(node_name="worker-0", max_concurrent=2)

        # Empty queue - can start more
        assert queue.can_start_more()

        # Add one active request
        request = LoadingRequest(
            id="abc123",
            model_name="model1",
            storage_level=StorageLevel.DISK,
            node_name="worker-0",
            backend="vllm",
            status=LoadingStatus.LOADING,
            created_at=datetime.now(),
        )
        queue.active[request.dedup_key] = request

        # One active - can still start more
        assert queue.can_start_more()

        # Add second active request
        request2 = LoadingRequest(
            id="def456",
            model_name="model2",
            storage_level=StorageLevel.DISK,
            node_name="worker-0",
            backend="vllm",
            status=LoadingStatus.LOADING,
            created_at=datetime.now(),
        )
        queue.active[request2.dedup_key] = request2

        # Two active at limit - cannot start more
        assert not queue.can_start_more()

    def test_to_dict(self):
        """Test converting node queue to dictionary."""
        queue = NodeQueue(node_name="worker-0", max_concurrent=2)

        result = queue.to_dict()

        assert result["node_name"] == "worker-0"
        assert result["max_concurrent"] == 2
        assert result["pending_count"] == 0
        assert result["active_count"] == 0
        assert result["pending"] == []
        assert result["active"] == []


class TestLoadingQueueManager:
    """Tests for LoadingQueueManager class."""

    @pytest.fixture
    def queue_manager(self):
        """Create a LoadingQueueManager instance."""
        return LoadingQueueManager(
            max_concurrent_per_node=2,
            process_interval=0.1,  # Fast processing for tests
        )

    def test_enqueue_request(self, queue_manager, event_loop):
        """Test enqueuing a load request."""

        async def _test():
            request = await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            assert request.model_name == "facebook/opt-125m"
            assert request.storage_level == StorageLevel.DISK
            assert request.node_name == "worker-0"
            assert request.status == LoadingStatus.PENDING

        event_loop.run_until_complete(_test())

    def test_enqueue_deduplication(self, queue_manager, event_loop):
        """Test that duplicate requests return the existing request."""

        async def _test():
            request1 = await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            request2 = await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            # Should return the same request
            assert request1.id == request2.id

        event_loop.run_until_complete(_test())

    def test_enqueue_different_nodes(self, queue_manager, event_loop):
        """Test that same model on different nodes creates separate requests."""

        async def _test():
            request1 = await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            request2 = await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-1",
            )

            # Should be different requests
            assert request1.id != request2.id

        event_loop.run_until_complete(_test())

    def test_enqueue_different_levels(self, queue_manager, event_loop):
        """Test that same model at different levels creates separate requests."""

        async def _test():
            request1 = await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            request2 = await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.MEMORY,
                node_name="worker-0",
            )

            # Should be different requests
            assert request1.id != request2.id

        event_loop.run_until_complete(_test())

    def test_cancel_pending_request(self, queue_manager, event_loop):
        """Test cancelling a pending request."""

        async def _test():
            request = await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            result = await queue_manager.cancel(request.id)

            assert result is True
            assert request.status == LoadingStatus.CANCELLED

        event_loop.run_until_complete(_test())

    def test_cancel_nonexistent_request(self, queue_manager, event_loop):
        """Test cancelling a non-existent request."""

        async def _test():
            result = await queue_manager.cancel("nonexistent-id")
            assert result is False

        event_loop.run_until_complete(_test())

    def test_is_model_loading(self, queue_manager, event_loop):
        """Test checking if a model is loading."""

        async def _test():
            # Initially not loading
            assert not queue_manager.is_model_loading(
                "facebook/opt-125m", StorageLevel.DISK, "worker-0"
            )

            # Enqueue a request
            await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            # Now it should be loading
            assert queue_manager.is_model_loading(
                "facebook/opt-125m", StorageLevel.DISK, "worker-0"
            )

        event_loop.run_until_complete(_test())

    def test_is_model_pending_or_loading(self, queue_manager, event_loop):
        """Test checking if a model is pending or loading on a node."""

        async def _test():
            # Initially not pending/loading
            assert not queue_manager.is_model_pending_or_loading(
                "facebook/opt-125m", "worker-0"
            )

            # Enqueue a request
            await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            # Now it should be pending/loading
            assert queue_manager.is_model_pending_or_loading(
                "facebook/opt-125m", "worker-0"
            )

        event_loop.run_until_complete(_test())

    def test_get_node_queue_state(self, queue_manager, event_loop):
        """Test getting node queue state."""

        async def _test():
            # Initially no state
            assert queue_manager.get_node_queue_state("worker-0") is None

            # Enqueue a request
            await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            state = queue_manager.get_node_queue_state("worker-0")

            assert state is not None
            assert state["node_name"] == "worker-0"
            assert state["pending_count"] == 1
            assert state["active_count"] == 0

        event_loop.run_until_complete(_test())

    def test_get_all_queue_states(self, queue_manager, event_loop):
        """Test getting all queue states."""

        async def _test():
            await queue_manager.enqueue(
                model_name="model1",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            await queue_manager.enqueue(
                model_name="model2",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-1",
            )

            states = queue_manager.get_all_queue_states()

            assert len(states) == 2
            assert "worker-0" in states
            assert "worker-1" in states

        event_loop.run_until_complete(_test())

    def test_get_pending_models(self, queue_manager, event_loop):
        """Test getting pending models for a node."""

        async def _test():
            await queue_manager.enqueue(
                model_name="model1",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            await queue_manager.enqueue(
                model_name="model2",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            pending = queue_manager.get_pending_models("worker-0")

            assert len(pending) == 2
            assert "model1" in pending
            assert "model2" in pending

        event_loop.run_until_complete(_test())

    def test_queue_processing_with_executor(self, queue_manager, event_loop):
        """Test queue processes requests when executor is set."""

        async def _test():
            # Track executed requests
            executed = []

            async def mock_executor(request):
                executed.append(request.model_name)
                return True

            queue_manager.set_load_executor(mock_executor)

            # Enqueue request
            await queue_manager.enqueue(
                model_name="facebook/opt-125m",
                backend="vllm",
                storage_level=StorageLevel.DISK,
                node_name="worker-0",
            )

            # Start the queue
            await queue_manager.start()

            # Wait for processing
            await asyncio.sleep(0.3)

            # Stop the queue
            await queue_manager.stop()

            # Verify request was executed
            assert "facebook/opt-125m" in executed

        event_loop.run_until_complete(_test())

    def test_queue_concurrency_limit(self, queue_manager, event_loop):
        """Test that queue respects concurrency limit."""

        async def _test():
            # Track active loads
            active_count = 0
            max_active = 0
            load_started = asyncio.Event()
            can_complete = asyncio.Event()

            async def slow_executor(request):
                nonlocal active_count, max_active
                active_count += 1
                max_active = max(max_active, active_count)
                load_started.set()
                await can_complete.wait()
                active_count -= 1
                return True

            queue_manager.set_load_executor(slow_executor)

            # Enqueue 4 requests (more than concurrency limit of 2)
            for i in range(4):
                await queue_manager.enqueue(
                    model_name=f"model{i}",
                    backend="vllm",
                    storage_level=StorageLevel.DISK,
                    node_name="worker-0",
                )

            # Start the queue
            await queue_manager.start()

            # Wait for loads to start
            await asyncio.wait_for(load_started.wait(), timeout=1.0)
            await asyncio.sleep(0.2)  # Allow queue to start more if it would

            # Max active should not exceed concurrency limit
            assert max_active <= 2

            # Allow loads to complete
            can_complete.set()

            # Stop the queue
            await queue_manager.stop()

        event_loop.run_until_complete(_test())

    def test_start_stop_lifecycle(self, queue_manager, event_loop):
        """Test start and stop lifecycle."""

        async def _test():
            # Should not raise
            await queue_manager.start()
            await queue_manager.stop()

            # Can restart
            await queue_manager.start()
            await queue_manager.stop()

        event_loop.run_until_complete(_test())

    def test_repr(self, queue_manager):
        """Test string representation."""
        repr_str = repr(queue_manager)

        assert "LoadingQueueManager" in repr_str
        assert "nodes=" in repr_str
        assert "pending=" in repr_str
        assert "active=" in repr_str
