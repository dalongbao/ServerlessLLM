# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import ray

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class SllmRouter(ABC):
    @abstractmethod
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
        router_config: Dict,
    ) -> None:
        pass

    @abstractmethod
    async def start(self, auto_scaling_config: Dict[str, int]):
        pass

    @abstractmethod
    async def shutdown(self):
        pass

    @abstractmethod
    async def update(self, auto_scaling_config: Dict[str, int]):
        pass

    @abstractmethod
    async def inference(self, request_data: dict):
        pass

    @abstractmethod
    async def fine_tuning(self, request_data: dict):
        pass
