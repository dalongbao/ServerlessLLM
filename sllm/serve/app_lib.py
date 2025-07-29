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
import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import ray
import ray.exceptions
from datasets import load_dataset
from fastapi import FastAPI, HTTPException, Request
from fastapi.background import BackgroundTask
from fastapi.responses import FileResponse

from sllm.serve.logger import init_logger
from sllm.serve.storage_client import StorageClient
from sllm.serve.utils import download_and_convert_hf_dataset

logger = init_logger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Connect to the Ray cluster
        # ray.init()
        yield
        # Shutdown the Ray cluster
        ray.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.post("/register")
    async def register_handler(request: Request):
        body = await request.json()

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )
        try:
            await controller.register.remote(body)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Cannot register model, please contact the administrator",
            )

        return {"status": "ok"}

    @app.post("/update")
    async def update_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        logger.info(f"Received request to update model {model_name}")
        try:
            await controller.update.remote(model_name, body)
        except ray.exceptions.RayTaskError as e:
            raise HTTPException(status_code=400, detail=str(e.cause))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {"status": f"updated model {model_name}"}

    @app.post("/delete")
    async def delete_model(request: Request):
        body = await request.json()

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )
        lora_adapters = body.get("lora_adapters", None)

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        if lora_adapters is not None:
            logger.info(
                f"Received request to delete LoRA adapters {lora_adapters} on model {model_name}"
            )
            await controller.delete.remote(model_name, lora_adapters)
        else:
            logger.info(f"Received request to delete model {model_name}")
            await controller.delete.remote(model_name)

        return {"status": f"deleted model {model_name}"}

    async def inference_handler(request: Request, action: str):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")

        result = request_router.inference.remote(body, action)
        return await result

    async def fine_tuning_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")

        result = request_router.fine_tuning.remote(body)
        return await result

    @app.post("/v1/chat/completions")
    async def generate_handler(request: Request):
        return await inference_handler(request, "generate")

    @app.post("/v1/embeddings")
    async def embeddings_handler(request: Request):
        return await inference_handler(request, "encode")

    @app.post("/fine-tuning")
    async def fine_tuning(request: Request):
        return await fine_tuning_handler(request)

    @app.get("/v1/models")
    async def get_models():
        logger.info("Attempting to retrieve the controller actor")
        try:
            controller = ray.get_actor("controller")
            if not controller:
                logger.error("Controller not initialized")
                raise HTTPException(
                    status_code=500, detail="Controller not initialized"
                )
            logger.info("Controller actor found")
            result = await controller.status.remote()
            logger.info("Controller status retrieved successfully")
            return result
        except Exception as e:
            logger.error(f"Error retrieving models: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve models"
            )

    @app.post("/v1/batches")
    async def create_batch(request: Request):
        logger.info("Received request to create and execute batch")

        body = await request.json()
        input_file_id = body.get("input_file_id")
        endpoint = body.get("endpoint")
        completion_window = body.get("completion_window", "24h")
        metadata = body.get("metadata", {})

        if not input_file_id:
            raise HTTPException(
                status_code=400, detail="input_file_id is required"
            )
        if not endpoint:
            raise HTTPException(status_code=400, detail="endpoint is required")

        if completion_window not in ["24h"]:
            raise HTTPException(
                status_code=400, detail="completion_window must be '24h'"
            )

        # Validate endpoint
        valid_endpoints = [
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/completions",
        ]
        if endpoint not in valid_endpoints:
            raise HTTPException(
                status_code=400,
                detail=f"endpoint must be one of: {', '.join(valid_endpoints)}",
            )

        # TODO: Implement batch creation and execution functionality
        # - Validate input_file_id exists and is JSONL with proper OpenAI format
        # - Create batch object with status "validating"
        # - Queue for processing

        return {"message": "Batch creation endpoint not implemented"}

    @app.get("/v1/batches")
    async def list_batches(request: Request):
        limit = request.query_params.get("limit", None)
        after = request.query_params.get("after", None)
        logger.info(
            f"Received request to list batches with limit: {limit}, after: {after}"
        )
        # TODO: Implement batch listing functionality
        return {"message": "Batch listing endpoint not implemented"}

    @app.get("/v1/batches/{batch_id}")
    async def get_batch(batch_id: str):
        logger.info(f"Received request to retrieve batch object: {batch_id}")
        # TODO: Implement batch retrieval functionality
        return {
            "message": f"Batch retrieval endpoint not implemented for batch_id: {batch_id}"
        }

    @app.post("/v1/batches/{batch_id}/cancel")
    async def cancel_batch(batch_id: str):
        logger.info(f"Received request to cancel batch: {batch_id}")
        # TODO: Implement batch cancellation functionality
        return {
            "message": f"Batch cancellation endpoint not implemented for batch_id: {batch_id}"
        }

    @app.post("/v1/files")
    async def upload_file(request: Request):
        logger.info("Received file upload request")

        try:
            form = await request.form()
            file_obj = form.get("file")
            purpose = form.get("purpose", None)

            if not file_obj:
                raise HTTPException(status_code=400, detail="No file provided")
            if not purpose:
                raise HTTPException(
                    status_code=400, detail="No purpose provided"
                )

            valid_purposes = [
                "batch",
                "fine-tune",
                "assistants",
                "eval",
                "user_data",
            ]
            if purpose not in valid_purposes:
                if purpose == "vision":
                    raise HTTPException(
                        status_code=400,
                        detail="Vision fine-tuning is not supported yet",
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid purpose '{purpose}'. Must be one of: {', '.join(valid_purposes)}",
                    )

            content = await file_obj.read()
            if file_obj.filename.endswith(".jsonl"):
                try:
                    lines = content.decode("utf-8").strip().split("\n")
                    for i, line in enumerate(lines):
                        if line.strip():
                            json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid JSONL format: {str(e)}",
                    )

            file_id = f"file-{uuid.uuid4().hex[:12]}"

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            metadata = {
                "id": file_id,
                "object": "file",
                "bytes": len(content),
                "created_at": int(datetime.now().timestamp()),
                "filename": file_obj.filename,
                "purpose": purpose,
            }

            storage_client = StorageClient()

            filename = file_obj.filename

            file_key = f"files/{file_id}/{filename}"
            metadata_key = f"files/{file_id}/metadata.json"

            if not storage_client.upload_file(temp_file_path, file_key):
                raise HTTPException(
                    status_code=500, detail="Failed to upload file to storage"
                )

            if not storage_client.upload_json(metadata, metadata_key):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to upload metadata to storage",
                )

            os.unlink(temp_file_path)

            logger.info(
                f"Successfully uploaded JSONL file {file_obj.filename} as {file_id}"
            )
            return metadata

        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to upload file: {str(e)}"
            )

    @app.get("/v1/files")
    async def list_files(request: Request):
        after = request.query_params.get("after", None)
        limit = int(request.query_params.get("limit", 20))
        order = request.query_params.get("order", "desc")
        purpose = request.query_params.get("purpose", None)

        logger.info(
            f"Received request to list file objects with after: {after}, limit: {limit}, order: {order}, purpose: {purpose}"
        )

        try:
            storage_client = StorageClient()
            file_list = []

            try:
                response = storage_client.s3_client.list_objects_v2(
                    Bucket=storage_client.bucket_name,
                    Prefix="files/",
                    Delimiter="/",
                )

                if "CommonPrefixes" in response:
                    for prefix in response["CommonPrefixes"]:
                        file_id = prefix["Prefix"].split("/")[1]
                        metadata = storage_client.get_json(
                            f"files/{file_id}/metadata.json"
                        )
                        if metadata:
                            if (
                                not purpose
                                or metadata.get("purpose") == purpose
                            ):
                                file_list.append(metadata)
            except Exception as e:
                logger.warning(f"Error listing files: {e}")

            file_list.sort(
                key=lambda x: x.get("created_at", 0), reverse=(order == "desc")
            )

            start_idx = 0
            if after:
                for i, file_obj in enumerate(file_list):
                    if file_obj["id"] == after:
                        start_idx = i + 1
                        break

            paginated_files = file_list[start_idx : start_idx + limit]

            return {
                "data": paginated_files,
                "object": "list",
                "has_more": start_idx + limit < len(file_list),
            }

        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to list files: {str(e)}"
            )

    @app.get("/v1/files/{file_id}")
    async def get_file(file_id: str):
        logger.info(f"Received request to retrieve file: {file_id}")

        try:
            storage_client = StorageClient()
            metadata = storage_client.get_json(f"files/{file_id}/metadata.json")

            if not metadata:
                raise HTTPException(
                    status_code=404, detail=f"File {file_id} not found"
                )

            logger.info(f"Successfully retrieved file metadata for {file_id}")
            return metadata

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve file: {str(e)}"
            )

    @app.delete("/v1/files/{file_id}")
    async def delete_file(file_id: str):
        logger.info(f"Received request to delete file: {file_id}")

        try:
            storage_client = StorageClient()

            metadata_key = f"files/{file_id}/metadata.json"

            if not storage_client.file_exists(metadata_key):
                raise HTTPException(
                    status_code=404, detail=f"File {file_id} not found"
                )

            metadata = storage_client.get_json(metadata_key)
            filename = metadata.get("filename", "data") if metadata else "data"
            data_key = f"files/{file_id}/{filename}"

            data_deleted = storage_client.delete_file(data_key)
            metadata_deleted = storage_client.delete_file(metadata_key)

            if not (data_deleted and metadata_deleted):
                raise HTTPException(
                    status_code=500, detail="Failed to delete file from storage"
                )

            logger.info(f"Successfully deleted file {file_id}")
            return {"id": file_id, "object": "file", "deleted": True}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to delete file: {str(e)}"
            )

    @app.get("/v1/files/{file_id}/content")
    async def get_file_content(file_id: str):
        logger.info(f"Received request to retrieve file content: {file_id}")

        try:
            storage_client = StorageClient()

            metadata = storage_client.get_json(f"files/{file_id}/metadata.json")
            if not metadata:
                raise HTTPException(
                    status_code=404, detail=f"File {file_id} not found"
                )

            filename = metadata.get("filename", "data")
            data_key = f"files/{file_id}/{filename}"

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name

            if not storage_client.download_file(data_key, temp_path):
                raise HTTPException(
                    status_code=500, detail="Failed to download file content"
                )

            # Determine content type
            content_type = "application/octet-stream"
            if filename.endswith(".jsonl"):
                content_type = "application/jsonl"
            elif filename.endswith(".json"):
                content_type = "application/json"
            elif filename.endswith(".txt"):
                content_type = "text/plain"
            elif filename.endswith(".csv"):
                content_type = "text/csv"
            elif filename.endswith(".pdf"):
                content_type = "application/pdf"
            elif filename.endswith(".png"):
                content_type = "image/png"
            elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
                content_type = "image/jpeg"
            elif filename.endswith(".gif"):
                content_type = "image/gif"
            elif filename.endswith(".md"):
                content_type = "text/markdown"
            elif filename.endswith(".html"):
                content_type = "text/html"
            elif filename.endswith(".xml"):
                content_type = "application/xml"
            elif filename.endswith(".zip"):
                content_type = "application/zip"

            logger.info(f"Successfully retrieved file content for {file_id}")

            return FileResponse(
                temp_path,
                media_type=content_type,
                filename=filename,
                background=BackgroundTask(os.unlink, temp_path),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve file content {file_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve file content: {str(e)}",
            )

    @app.post("/v1/files/download_hf")
    async def download_hf_dataset(request: Request):
        body = await request.json()
        dataset_name = body.get("dataset_name")
        model = body.get("model")
        endpoint = body.get("endpoint", "/v1/chat/completions")
        config = body.get("config", None)
        split = body.get("split", None)
        format_template = body.get("format_template", None)

        if not dataset_name:
            raise HTTPException(
                status_code=400, detail="dataset_name is required"
            )
        if not model:
            raise HTTPException(status_code=400, detail="model is required")

        # Validate endpoint
        valid_endpoints = [
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/completions",
        ]
        if endpoint not in valid_endpoints:
            raise HTTPException(
                status_code=400,
                detail=f"endpoint must be one of: {', '.join(valid_endpoints)}",
            )

        logger.info(
            f"Received request to download HuggingFace dataset: {dataset_name}, config: {config}, split: {split}"
        )

        try:
            file_id = f"file-{uuid.uuid4().hex[:12]}"

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as temp_file:
                temp_jsonl_path = temp_file.name

            dataset_metadata = download_and_convert_hf_dataset(
                dataset_name=dataset_name,
                output_file=temp_jsonl_path,
                model=model,
                endpoint=endpoint,
                config=config,
                split=split,
                format_template=format_template,
            )

            file_metadata = {
                "id": file_id,
                "object": "file",
                "filename": f"{dataset_name.replace('/', '_')}_{split or 'default'}.jsonl",
                "bytes": os.path.getsize(temp_jsonl_path),
                "created_at": int(datetime.now().timestamp()),
                "purpose": "batch",
                "dataset_info": dataset_metadata,
            }

            storage_client = StorageClient()

            data_key = f"files/{file_id}/data.jsonl"
            metadata_key = f"files/{file_id}/metadata.json"

            if not storage_client.upload_file(temp_jsonl_path, data_key):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to upload dataset to storage",
                )

            if not storage_client.upload_json(file_metadata, metadata_key):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to upload metadata to storage",
                )

            os.unlink(temp_jsonl_path)

            logger.info(
                f"Successfully downloaded and stored dataset {dataset_name} as {file_id}"
            )
            return {
                "id": file_id,
                "object": "file",
                "purpose": "batch",
                "filename": file_metadata["filename"],
                "bytes": file_metadata["bytes"],
                "created_at": file_metadata["created_at"],
                "dataset_info": dataset_metadata,
            }

        except Exception as e:
            logger.error(
                f"Failed to download HuggingFace dataset {dataset_name}: {e}"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to download dataset: {str(e)}"
            )

    return app
