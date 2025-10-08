"""
S3 Client for PBF-LB/M Data Pipeline

This module provides S3 operations for data lake storage.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Iterator, Union
from datetime import datetime, timedelta
from pathlib import Path
from src.data_pipeline.config.storage_config import get_s3_config

logger = logging.getLogger(__name__)


class S3Client:
    """
    S3 client for data lake operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize S3 client.
        
        Args:
            config: Optional S3 configuration dictionary
        """
        self.config = config or get_s3_config()
        self.s3_client = None
        self.bucket_name = self.config.get('bucket_name', 'pbf-lbm-data-lake')
        self._initialize_s3_client()
    
    def _initialize_s3_client(self) -> None:
        """Initialize S3 client."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.config.get('aws_access_key_id'),
                aws_secret_access_key=self.config.get('aws_secret_access_key'),
                endpoint_url=self.config.get('endpoint_url'),
                region_name=self.config.get('region_name', 'us-east-1')
            )
            logger.info("S3 client initialized successfully")
        except ImportError:
            logger.error("boto3 not available for S3 operations")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
    
    def upload_file(self, local_path: str, s3_key: str, bucket: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key
            bucket: S3 bucket name (uses default if None)
            metadata: Optional metadata for the object
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        bucket = bucket or self.bucket_name
        
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.upload_file(local_path, bucket, s3_key, ExtraArgs=extra_args)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading file to S3: {e}")
            return False
    
    def download_file(self, s3_key: str, local_path: str, bucket: Optional[str] = None) -> bool:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local file path
            bucket: S3 bucket name (uses default if None)
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        bucket = bucket or self.bucket_name
        
        try:
            self.s3_client.download_file(bucket, s3_key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading file from S3: {e}")
            return False
    
    def upload_data(self, data: Union[str, bytes], s3_key: str, bucket: Optional[str] = None, content_type: Optional[str] = None) -> bool:
        """
        Upload data directly to S3.
        
        Args:
            data: Data to upload (string or bytes)
            s3_key: S3 object key
            bucket: S3 bucket name (uses default if None)
            content_type: Content type of the data
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        bucket = bucket or self.bucket_name
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=data,
                **extra_args
            )
            logger.info(f"Uploaded data to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading data to S3: {e}")
            return False
    
    def download_data(self, s3_key: str, bucket: Optional[str] = None) -> Optional[bytes]:
        """
        Download data directly from S3.
        
        Args:
            s3_key: S3 object key
            bucket: S3 bucket name (uses default if None)
            
        Returns:
            Optional[bytes]: Downloaded data or None if error
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return None
        
        bucket = bucket or self.bucket_name
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
            data = response['Body'].read()
            logger.info(f"Downloaded data from s3://{bucket}/{s3_key}")
            return data
        except Exception as e:
            logger.error(f"Error downloading data from S3: {e}")
            return None
    
    def list_objects(self, prefix: str = '', bucket: Optional[str] = None, max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket.
        
        Args:
            prefix: S3 prefix
            bucket: S3 bucket name (uses default if None)
            max_keys: Maximum number of keys to return
            
        Returns:
            List[Dict[str, Any]]: List of object metadata
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return []
        
        bucket = bucket or self.bucket_name
        objects = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'etag': obj['ETag'],
                            'storage_class': obj.get('StorageClass', 'STANDARD')
                        })
            
            logger.info(f"Listed {len(objects)} objects in s3://{bucket}/{prefix}")
            
        except Exception as e:
            logger.error(f"Error listing S3 objects: {e}")
        
        return objects
    
    def delete_object(self, s3_key: str, bucket: Optional[str] = None) -> bool:
        """
        Delete an object from S3.
        
        Args:
            s3_key: S3 object key
            bucket: S3 bucket name (uses default if None)
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        bucket = bucket or self.bucket_name
        
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=s3_key)
            logger.info(f"Deleted s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting object from S3: {e}")
            return False
    
    def delete_objects(self, s3_keys: List[str], bucket: Optional[str] = None) -> int:
        """
        Delete multiple objects from S3.
        
        Args:
            s3_keys: List of S3 object keys
            bucket: S3 bucket name (uses default if None)
            
        Returns:
            int: Number of objects deleted successfully
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return 0
        
        bucket = bucket or self.bucket_name
        deleted_count = 0
        
        try:
            # Delete in batches of 1000 (S3 limit)
            for i in range(0, len(s3_keys), 1000):
                batch = s3_keys[i:i+1000]
                objects = [{'Key': key} for key in batch]
                
                response = self.s3_client.delete_objects(
                    Bucket=bucket,
                    Delete={'Objects': objects}
                )
                
                deleted_count += len(response.get('Deleted', []))
            
            logger.info(f"Deleted {deleted_count} objects from s3://{bucket}")
            
        except Exception as e:
            logger.error(f"Error deleting objects from S3: {e}")
        
        return deleted_count
    
    def copy_object(self, source_key: str, dest_key: str, source_bucket: Optional[str] = None, dest_bucket: Optional[str] = None) -> bool:
        """
        Copy an object within S3.
        
        Args:
            source_key: Source S3 object key
            dest_key: Destination S3 object key
            source_bucket: Source S3 bucket name (uses default if None)
            dest_bucket: Destination S3 bucket name (uses default if None)
            
        Returns:
            bool: True if copy successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        source_bucket = source_bucket or self.bucket_name
        dest_bucket = dest_bucket or self.bucket_name
        
        try:
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            self.s3_client.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_key)
            logger.info(f"Copied s3://{source_bucket}/{source_key} to s3://{dest_bucket}/{dest_key}")
            return True
        except Exception as e:
            logger.error(f"Error copying object in S3: {e}")
            return False
    
    def get_object_metadata(self, s3_key: str, bucket: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get object metadata from S3.
        
        Args:
            s3_key: S3 object key
            bucket: S3 bucket name (uses default if None)
            
        Returns:
            Optional[Dict[str, Any]]: Object metadata or None if error
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return None
        
        bucket = bucket or self.bucket_name
        
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            metadata = {
                'key': s3_key,
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'etag': response['ETag'],
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {}),
                'storage_class': response.get('StorageClass', 'STANDARD')
            }
            return metadata
        except Exception as e:
            logger.error(f"Error getting object metadata from S3: {e}")
            return None
    
    def generate_presigned_url(self, s3_key: str, expiration: int = 3600, bucket: Optional[str] = None, operation: str = 'get_object') -> Optional[str]:
        """
        Generate a presigned URL for S3 object.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
            bucket: S3 bucket name (uses default if None)
            operation: S3 operation ('get_object', 'put_object', etc.)
            
        Returns:
            Optional[str]: Presigned URL or None if error
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return None
        
        bucket = bucket or self.bucket_name
        
        try:
            url = self.s3_client.generate_presigned_url(
                operation,
                Params={'Bucket': bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            logger.info(f"Generated presigned URL for s3://{bucket}/{s3_key}")
            return url
        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None
    
    def create_bucket(self, bucket_name: str, region: Optional[str] = None) -> bool:
        """
        Create an S3 bucket.
        
        Args:
            bucket_name: Name of the bucket to create
            region: AWS region (uses default if None)
            
        Returns:
            bool: True if creation successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        try:
            if region:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            else:
                self.s3_client.create_bucket(Bucket=bucket_name)
            
            logger.info(f"Created S3 bucket: {bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating S3 bucket: {e}")
            return False
    
    def bucket_exists(self, bucket_name: str) -> bool:
        """
        Check if an S3 bucket exists.
        
        Args:
            bucket_name: Name of the bucket to check
            
        Returns:
            bool: True if bucket exists, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except Exception as e:
            return False
    
    def get_bucket_info(self, bucket_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an S3 bucket.
        
        Args:
            bucket_name: Name of the bucket
            
        Returns:
            Optional[Dict[str, Any]]: Bucket information or None if error
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return None
        
        try:
            response = self.s3_client.head_bucket(Bucket=bucket_name)
            return {
                'bucket': bucket_name,
                'region': response.get('ResponseMetadata', {}).get('HTTPHeaders', {}).get('x-amz-bucket-region'),
                'status': 'accessible'
            }
        except Exception as e:
            logger.error(f"Error getting bucket info for {bucket_name}: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dict[str, Any]: Storage statistics
        """
        return {
            "bucket_name": self.bucket_name,
            "s3_client_initialized": self.s3_client is not None,
            "timestamp": datetime.now().isoformat()
        }
