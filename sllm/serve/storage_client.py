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
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class StorageClient:
    def __init__(self):
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        self.access_key = os.getenv("S3_ACCESS_KEY", "admin")
        self.secret_key = os.getenv("S3_SECRET_KEY", "password")
        self.bucket_name = os.getenv("S3_BUCKET", "sllm-datasets")
        self.region = os.getenv("S3_REGION", "us-east-1")
        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )
        
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"Creating bucket {self.bucket_name}")
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                logger.error(f"Error checking bucket: {e}")
                raise
    
    def upload_file(self, file_path: str, object_key: str) -> bool:
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_key)
            logger.info(f"Uploaded {file_path} to {object_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            return False
    
    def upload_json(self, data: Dict[str, Any], object_key: str) -> bool:
        try:
            json_data = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=json_data,
                ContentType='application/json'
            )
            logger.info(f"Uploaded JSON data to {object_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload JSON to {object_key}: {e}")
            return False
    
    def download_file(self, object_key: str, file_path: str) -> bool:
        try:
            self.s3_client.download_file(self.bucket_name, object_key, file_path)
            logger.info(f"Downloaded {object_key} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {object_key}: {e}")
            return False
    
    def get_json(self, object_key: str) -> Optional[Dict[str, Any]]:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Retrieved JSON data from {object_key}")
            return data
        except Exception as e:
            logger.error(f"Failed to get JSON from {object_key}: {e}")
            return None
    
    def file_exists(self, object_key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Error checking if {object_key} exists: {e}")
                return False
    
    def delete_file(self, object_key: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
            logger.info(f"Deleted {object_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {object_key}: {e}")
            return False