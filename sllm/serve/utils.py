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
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ray
from datasets import load_dataset

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


def get_worker_nodes():
    ray_nodes = ray.nodes()
    worker_node_info = {}
    for node in ray_nodes:
        ray_node_id = node.get("NodeID", None)
        assert ray_node_id is not None, "NodeID not found"
        resources = node.get("Resources", {})
        assert resources != {}, "Resources not found"
        node_address = node.get("NodeManagerAddress", None)
        assert (
            node_address is not None and node_address != ""
        ), "NodeManagerAddress not found"
        if resources.get("control_node", 0) > 0:
            continue  # Skip the control node

        for key, value in resources.items():
            if key.startswith("worker_id_"):
                node_id = key.split("_")[-1]
                worker_node_info[node_id] = {
                    "ray_node_id": ray_node_id,
                    "address": node_address,
                    "free_gpu": resources.get("GPU", 0),
                    "total_gpu": resources.get("GPU", 0),
                }

    return worker_node_info


@dataclass
class InstanceStatus:
    instance_id: str
    node_id: str
    num_gpu: int
    concurrency: int

    model_name: Optional[str] = None
    num_current_tokens: Optional[int] = None
    resuming_latency: Optional[float] = None


@dataclass
class InstanceHandle:
    instance_id: str
    max_queue_length: int
    num_gpu: int

    node_id: Optional[str] = None
    backend_instance: Optional[ray.actor.ActorHandle] = None
    ready: bool = False
    concurrency: int = 0

    lock: asyncio.Lock = asyncio.Lock()

    async def add_requests(self, num_requests: int = 1):
        async with self.lock:
            if not self.ready:
                return False
            if (
                self.concurrency + num_requests > self.max_queue_length
                or self.concurrency + num_requests < 0
            ):
                return False
            self.concurrency += num_requests
            return True

    async def check_request_queue(self):
        async with self.lock:
            return self.concurrency + 1 <= self.max_queue_length

    async def get_status(self):
        async with self.lock:
            return InstanceStatus(
                self.instance_id,
                self.node_id,
                self.num_gpu,
                self.concurrency,
            )


def download_and_convert_hf_dataset(
    dataset_name: str,
    output_file: str,
    model: str,
    endpoint: str = "/v1/chat/completions",
    config: Optional[str] = None,
    split: Optional[str] = None,
    format_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download HuggingFace dataset and convert to JSONL format for batch processing.

    Args:
        dataset_name: HuggingFace dataset identifier
        output_file: Path to output JSONL file
        model: Model name to use in requests
        endpoint: API endpoint to use (default: /v1/chat/completions)
        config: Dataset configuration name
        split: Dataset split (train/test/validation)
        format_template: Template for formatting each example (optional)

    Returns:
        Dictionary with dataset metadata
    """
    logger.info(f"Loading dataset {dataset_name} from HuggingFace Hub")

    if config:
        dataset = load_dataset(dataset_name, config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    logger.info(
        f"Converting dataset to JSONL format with {len(dataset)} examples"
    )

    with open(output_file, "w") as f:
        for i, item in enumerate(dataset):
            if format_template:
                formatted_item = format_example(
                    item, format_template, model, endpoint
                )
            else:
                formatted_item = {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": endpoint,
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": str(item)}],
                    },
                }

            json.dump(formatted_item, f)
            f.write("\n")

    metadata = {
        "dataset_name": dataset_name,
        "config": config,
        "split": split,
        "num_examples": len(dataset),
        "columns": (
            list(dataset.column_names)
            if hasattr(dataset, "column_names")
            else []
        ),
        "format_template": format_template,
    }

    logger.info(f"Successfully converted {len(dataset)} examples to JSONL")
    return metadata


def format_example(
    item: Dict[str, Any], template: str, model: str, endpoint: str
) -> Dict[str, Any]:
    """
    Format a single dataset example using a template.

    Template examples:
    - "prompt_completion": Creates {"role": "user", "content": item["prompt"]} + assistant completion
    - "instruction_response": Uses instruction/response fields
    - "custom": Custom JSON template with {field} placeholders
    """
    if template == "prompt_completion":
        return {
            "custom_id": f"request-{hash(str(item))}",
            "method": "POST",
            "url": endpoint,
            "body": {
                "model": model,
                "messages": [
                    {"role": "user", "content": item.get("prompt", str(item))}
                ],
            },
        }
    elif template == "instruction_response":
        return {
            "custom_id": f"request-{hash(str(item))}",
            "method": "POST",
            "url": endpoint,
            "body": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": item.get("instruction", str(item)),
                    }
                ],
            },
        }
    else:
        return {
            "custom_id": f"request-{hash(str(item))}",
            "method": "POST",
            "url": endpoint,
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": str(item)}],
            },
        }
