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
Autoscaler for ServerlessLLM v1-beta.

Receives metrics pushed from Router, calculates desired replicas, writes to SQLite.
The Reconciler then acts on the desired state to create/delete instances.

Design (from docs/v1-beta-scalable-router-design.md):
- Receives metrics via receive_metrics(model_name, backend, buffer_len, in_flight)
- Calculates desired replicas from total demand
- Writes desired state to models.desired_replicas in SQLite

Simplified formula:
    total_demand = buffer_length + in_flight_count
    desired = ceil(total_demand / target_pending_requests)
    desired = clamp(desired, min_replicas, max_replicas)
"""

import asyncio
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from sllm.database import Database, Model
from sllm.logger import init_logger

AUTOSCALER_INTERVAL_SECONDS = 3

logger = init_logger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a model from Router."""

    buffer_len: int = 0
    in_flight: int = 0

    @property
    def total_demand(self) -> int:
        return self.buffer_len + self.in_flight


class AutoScaler:
    """
    Autoscaler that receives metrics from Router and updates desired_replicas.

    The actual scaling is performed by the Reconciler, which reads the
    desired_replicas and creates/deletes instances accordingly.
    """

    def __init__(self, database: Database):
        """
        Initialize the Autoscaler.

        Args:
            database: Database instance for reading/writing model config
        """
        self.database = database
        self.interval = AUTOSCALER_INTERVAL_SECONDS

        self._metrics: Dict[tuple, ModelMetrics] = {}

        self._model_idle_times: Dict[tuple, int] = {}

        # Shutdown flag
        self._shutdown = asyncio.Event()

    def receive_metrics(
        self,
        model_name: str,
        backend: str,
        buffer_len: int,
        in_flight: int,
    ):
        """
        Receive metrics pushed from Router.

        Called by Router on every state change (request start/end, buffer).
        This updates the cached metrics which are used by the scaling loop.

        Args:
            model_name: Model name
            backend: Backend type
            buffer_len: Number of requests in cold-start buffer
            in_flight: Number of active requests being processed
        """
        key = (model_name, backend)
        if key not in self._metrics:
            self._metrics[key] = ModelMetrics()

        metrics = self._metrics[key]
        metrics.buffer_len = buffer_len
        metrics.in_flight = in_flight

        logger.debug(
            f"[{key}] Received metrics: "
            f"buffer={buffer_len}, in_flight={in_flight}"
        )

    async def run(self):
        """Main autoscaler loop."""
        logger.info("Autoscaler started")

        while not self._shutdown.is_set():
            try:
                await self._scale_all_models()
            except asyncio.CancelledError:
                logger.info("Autoscaler cancelled")
                break
            except Exception as e:
                logger.error(f"Error in autoscaler loop: {e}", exc_info=True)

            # Wait for interval or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self.interval
                )
            except asyncio.TimeoutError:
                pass

        logger.info("Autoscaler stopped")

    def shutdown(self):
        """Signal the autoscaler to stop."""
        logger.info("Shutting down Autoscaler")
        self._shutdown.set()

    async def _scale_all_models(self):
        """Check and scale all active models."""
        models = self.database.get_active_models()
        if not models:
            logger.debug("No active models, skipping scaling check")
            return

        logger.debug(f"Scaling check for {len(models)} models")

        for model in models:
            try:
                await self._scale_model(model)
            except Exception as e:
                logger.error(
                    f"Error scaling ({model.model_name}, {model.backend}): {e}",
                    exc_info=True,
                )

    async def _scale_model(self, model: Model):
        """
        Calculate and update desired_replicas for a model.

        Args:
            model: Model to scale
        """
        key = (model.model_name, model.backend)

        metrics = self._metrics.get(key)
        if not metrics:
            total_demand = 0
            buffer_length = 0
            in_flight_count = 0
        else:
            total_demand = metrics.total_demand
            buffer_length = metrics.buffer_len
            in_flight_count = metrics.in_flight

        if total_demand > 0:
            raw_desired = math.ceil(
                total_demand / model.target_pending_requests
            )
        else:
            raw_desired = model.min_replicas

        desired = max(model.min_replicas, min(raw_desired, model.max_replicas))

        current_desired = model.desired_replicas
        if desired < current_desired and model.keep_alive_seconds > 0:
            idle_time = self._model_idle_times.get(key, 0)
            if idle_time < model.keep_alive_seconds:
                self._model_idle_times[key] = idle_time + self.interval
                logger.debug(
                    f"[{key}] Keep alive: idle={idle_time + self.interval}s, "
                    f"keep_alive={model.keep_alive_seconds}s"
                )
                return
            else:
                self._model_idle_times[key] = 0
        else:
            self._model_idle_times[key] = 0

        if desired != current_desired:
            logger.info(
                f"[{key}] Scaling: demand={total_demand} "
                f"(buffer={buffer_length}, in_flight={in_flight_count}), "
                f"desired={current_desired} -> {desired}"
            )
            self.database.update_desired_replicas(
                model.model_name, model.backend, desired
            )


# Global instance
_autoscaler: Optional[AutoScaler] = None


def get_autoscaler() -> Optional[AutoScaler]:
    """Get the global Autoscaler instance."""
    return _autoscaler


def init_autoscaler(database: Database) -> AutoScaler:
    """Initialize the global Autoscaler instance."""
    global _autoscaler
    _autoscaler = AutoScaler(database=database)
    return _autoscaler
