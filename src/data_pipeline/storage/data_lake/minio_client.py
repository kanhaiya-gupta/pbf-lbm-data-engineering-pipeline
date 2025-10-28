"""
MinIO Client for PBF-LB/M Data Pipeline

This module provides MinIO operations for data lake storage.
MinIO is an S3-compatible object storage server for data lake operations.
"""

import logging
import os
import json
import tempfile
import urllib3
from typing import Dict, Any, Optional, List, Iterator, Union
from datetime import datetime, timedelta
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from src.data_pipeline.config.minio_config import get_minio_config

logger = logging.getLogger(__name__)


class MinIOClient:
    """
    MinIO client for data lake operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MinIO client.
        
        Args:
            config: Optional MinIO configuration dictionary or Pydantic model
        """
        self.config = config or get_minio_config()
        self._initialize_connection()
        self.minio_client = None
        self._initialize_minio_client()
    
    def _initialize_connection(self) -> None:
        """Initialize MinIO connection parameters."""
        try:
            # Handle both dictionary and Pydantic model configurations
            if isinstance(self.config, dict):
                self.host = self.config.get('host', 'localhost')
                self.port = self.config.get('port', 9000)
                self.access_key = self.config.get('access_key', 'minioadmin')
                self.secret_key = self.config.get('secret_key', 'minioadmin123')
                self.secure = self.config.get('secure', False)
                self.ssl_verify = self.config.get('ssl_verify', True)
                self.region = self.config.get('region', 'us-east-1')
                self.bucket_name = self.config.get('bucket_name', 'pbf-lbm-data')
                self.timeout = self.config.get('timeout', 30)
                self.max_retries = self.config.get('max_retries', 3)
                self.retry_delay = self.config.get('retry_delay', 1.0)
                self.multipart_chunksize = self.config.get('multipart_chunksize', 16777216)  # 16MB
                self.multipart_threshold = self.config.get('multipart_threshold', 5242880)  # 5MB
            else:
                # Pydantic model
                self.host = self.config.host
                self.port = self.config.port
                self.access_key = self.config.access_key
                self.secret_key = self.config.secret_key
                self.secure = self.config.secure
                self.ssl_verify = self.config.ssl_verify
                self.region = self.config.region
                self.bucket_name = self.config.bucket_name
                self.timeout = self.config.timeout
                self.max_retries = self.config.max_retries
                self.retry_delay = self.config.retry_delay
                self.multipart_chunksize = self.config.multipart_chunksize
                self.multipart_threshold = self.config.multipart_threshold
            
            logger.info("MinIO connection parameters initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO connection parameters: {e}")
            raise
    
    def _initialize_minio_client(self) -> None:
        """Initialize MinIO client."""
        try:
            
            # Construct endpoint URL
            protocol = "https" if self.secure else "http"
            endpoint = f"{self.host}:{self.port}"
            
            # Custom HTTP client with retries and timeouts for reliability
            http_client = urllib3.PoolManager(
                retries=urllib3.Retry(total=5, backoff_factor=0.2),
                timeout=urllib3.Timeout(connect=10, read=30),
                maxsize=10  # Connection pool size for concurrent uploads
            )
            
            self.minio_client = Minio(
                endpoint=endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                # Remove region parameter to avoid SignatureDoesNotMatch errors
                http_client=http_client
            )
            
            logger.info(f"MinIO client initialized: {protocol}://{endpoint}")
            
        except ImportError:
            logger.error("MinIO library not available. Install with: pip install minio")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            raise
    
    def connect(self) -> bool:
        """
        Test MinIO connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Test connection by listing buckets
            buckets = self.minio_client.list_buckets()
            logger.info(f"✅ Connected to MinIO server. Found {len(buckets)} buckets")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to MinIO: {e}")
            return False
    
    def create_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Create a bucket if it doesn't exist.
        
        Args:
            bucket_name: Optional bucket name (uses default if not provided)
            
        Returns:
            bool: True if bucket created or exists, False otherwise
        """
        try:
            bucket = bucket_name or self.bucket_name
            
            if not self.minio_client.bucket_exists(bucket):
                self.minio_client.make_bucket(bucket)
                logger.info(f"✅ Created bucket: {bucket}")
            else:
                logger.info(f"✅ Bucket already exists: {bucket}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create bucket '{bucket}': {e}")
            return False
    
    def upload_file(self, file_path: Union[str, Path], object_name: Optional[str] = None, 
                   bucket_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Upload a file to MinIO.
        
        Args:
            file_path: Path to file to upload
            object_name: Optional object name in bucket
            bucket_name: Optional bucket name
            metadata: Optional metadata dictionary
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            bucket = bucket_name or self.bucket_name
            object_name = object_name or file_path.name
            
            # Ensure bucket exists
            self.create_bucket(bucket)
            
            # Prepare metadata
            meta = metadata or {}
            meta.update({
                'uploaded_at': datetime.now().isoformat(),
                'file_size': str(file_path.stat().st_size),
                'file_type': file_path.suffix
            })
            
            # Upload file
            self.minio_client.fput_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=str(file_path),
                metadata=meta
            )
            
            logger.info(f"✅ Uploaded {file_path.name} to {bucket}/{object_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload file '{file_path}': {e}")
            return False
    
    def upload_data(self, data: Union[str, bytes], object_name: str, 
                   bucket_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                   compress: bool = False) -> bool:
        """
        Upload data directly to MinIO.
        
        Args:
            data: Data to upload (string or bytes)
            object_name: Object name in bucket
            bucket_name: Optional bucket name
            metadata: Optional metadata dictionary
            compress: Whether to compress JSON/CSV data before upload
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            import io
            import gzip
            
            bucket = bucket_name or self.bucket_name
            
            # Ensure bucket exists
            self.create_bucket(bucket)
            
            # Prepare metadata (don't modify it to avoid signature issues)
            meta = metadata or {}
            # Remove automatic metadata addition that causes SignatureDoesNotMatch
            # meta.update({
            #     'uploaded_at': datetime.now().isoformat(),
            #     'data_type': 'direct_upload'
            # })
            
            # Convert data to bytes if string
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Compress data if requested and file is JSON/CSV
            if compress and (object_name.endswith('.json') or object_name.endswith('.csv')):
                logger.info(f"🗜️ Compressing {object_name} before upload")
                original_size = len(data)
                compressed_data = gzip.compress(data)
                if len(compressed_data) < original_size:
                    data = compressed_data
                    meta['Content-Encoding'] = 'gzip'
                    meta['Original-Size'] = str(original_size)
                    logger.info(f"📊 Compression ratio: {len(compressed_data)/original_size*100:.1f}%")
                else:
                    logger.info("📊 No compression benefit, using original data")
            
            # Use put_object for all files - MinIO handles multipart automatically
            data_stream = io.BytesIO(data)
            self.minio_client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=data_stream,
                length=len(data),
                metadata=meta
            )
            
            logger.info(f"✅ Uploaded data to {bucket}/{object_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload data to '{object_name}': {e}")
            return False
    
    
    def _upload_large_file_direct_path(self, file_path: str, bucket: str, object_name: str, metadata: Dict[str, Any]) -> bool:
        """
        Upload large file using fput_object with direct file path (like working script).
        
        Args:
            file_path: Path to the file on disk
            bucket: Bucket name
            object_name: Object name
            metadata: File metadata
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            
            # Use the exact same approach as debug_minio_direct_fput.py
            http_client = urllib3.PoolManager(
                retries=urllib3.Retry(total=5, backoff_factor=0.2),
                timeout=urllib3.Timeout(connect=10, read=30),
                maxsize=10
            )
            
            # Create fresh client with exact same parameters as working script
            fresh_client = Minio(
                endpoint="localhost:9000",  # Use hardcoded endpoint like working script
                access_key="minioadmin",
                secret_key="minioadmin123",
                secure=False,
                http_client=http_client
                # No region parameter - this is the key fix!
            )
            
            # Use fput_object with exact same parameters as working script
            fresh_client.fput_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=file_path,  # Use direct file path like working script
                part_size=64 * 1024 * 1024,  # 64MiB parts
                num_parallel_uploads=4,  # Parallel threads
                metadata=metadata
            )
            
            logger.info(f"✅ Uploaded large file to {bucket}/{object_name}")
            return True
            
        except S3Error as e:
            logger.error(f"❌ S3 error during large file upload: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to upload large file: {e}")
            return False
    
    def download_file(self, object_name: str, file_path: Union[str, Path], 
                     bucket_name: Optional[str] = None) -> bool:
        """
        Download a file from MinIO.
        
        Args:
            object_name: Object name in bucket
            file_path: Local file path to save to
            bucket_name: Optional bucket name
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            bucket = bucket_name or self.bucket_name
            file_path = Path(file_path)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            self.minio_client.fget_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=str(file_path)
            )
            
            logger.info(f"✅ Downloaded {object_name} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download '{object_name}': {e}")
            return False
    
    def download_data(self, object_name: str, bucket_name: Optional[str] = None) -> Optional[bytes]:
        """
        Download data directly from MinIO.
        
        Args:
            object_name: Object name in bucket
            bucket_name: Optional bucket name
            
        Returns:
            bytes: Downloaded data or None if failed
        """
        try:
            import io
            
            bucket = bucket_name or self.bucket_name
            
            # Download data
            response = self.minio_client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            
            logger.info(f"✅ Downloaded data from {bucket}/{object_name}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to download data from '{object_name}': {e}")
            return None
    
    def list_objects(self, bucket_name: Optional[str] = None, prefix: Optional[str] = None, 
                    recursive: bool = True) -> List[Dict[str, Any]]:
        """
        List objects in a bucket.
        
        Args:
            bucket_name: Optional bucket name
            prefix: Optional object prefix filter
            recursive: Whether to list recursively
            
        Returns:
            List[Dict[str, Any]]: List of object information
        """
        try:
            bucket = bucket_name or self.bucket_name
            objects = []
            
            for obj in self.minio_client.list_objects(bucket, prefix=prefix, recursive=recursive):
                objects.append({
                    'object_name': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag,
                    'content_type': obj.content_type,
                    'metadata': obj.metadata
                })
            
            logger.info(f"✅ Listed {len(objects)} objects from {bucket}")
            return objects
            
        except Exception as e:
            logger.error(f"❌ Failed to list objects from '{bucket}': {e}")
            return []
    
    def delete_object(self, object_name: str, bucket_name: Optional[str] = None) -> bool:
        """
        Delete an object from MinIO.
        
        Args:
            object_name: Object name to delete
            bucket_name: Optional bucket name
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            bucket = bucket_name or self.bucket_name
            
            self.minio_client.remove_object(bucket, object_name)
            
            logger.info(f"✅ Deleted object {bucket}/{object_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete object '{object_name}': {e}")
            return False
    
    def delete_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Delete a bucket (must be empty).
        
        Args:
            bucket_name: Optional bucket name
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            bucket = bucket_name or self.bucket_name
            
            self.minio_client.remove_bucket(bucket)
            
            logger.info(f"✅ Deleted bucket: {bucket}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete bucket '{bucket}': {e}")
            return False
    
    def get_object_info(self, object_name: str, bucket_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about an object.
        
        Args:
            object_name: Object name
            bucket_name: Optional bucket name
            
        Returns:
            Dict[str, Any]: Object information or None if not found
        """
        try:
            bucket = bucket_name or self.bucket_name
            
            stat = self.minio_client.stat_object(bucket, object_name)
            
            return {
                'object_name': object_name,
                'size': stat.size,
                'last_modified': stat.last_modified,
                'etag': stat.etag,
                'content_type': stat.content_type,
                'metadata': stat.metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get object info for '{object_name}': {e}")
            return None
    
    def copy_object(self, source_object: str, dest_object: str, 
                   source_bucket: Optional[str] = None, dest_bucket: Optional[str] = None) -> bool:
        """
        Copy an object within MinIO.
        
        Args:
            source_object: Source object name
            dest_object: Destination object name
            source_bucket: Optional source bucket name
            dest_bucket: Optional destination bucket name
            
        Returns:
            bool: True if copy successful, False otherwise
        """
        try:
            source_bucket = source_bucket or self.bucket_name
            dest_bucket = dest_bucket or self.bucket_name
            
            # Ensure destination bucket exists
            self.create_bucket(dest_bucket)
            
            # Copy object
            from minio.commonconfig import CopySource
            self.minio_client.copy_object(
                bucket_name=dest_bucket,
                object_name=dest_object,
                source=CopySource(source_bucket, source_object)
            )
            
            logger.info(f"✅ Copied {source_bucket}/{source_object} to {dest_bucket}/{dest_object}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy object '{source_object}': {e}")
            return False
    
    def get_presigned_url(self, object_name: str, bucket_name: Optional[str] = None, 
                         expires: timedelta = timedelta(hours=1)) -> Optional[str]:
        """
        Get a presigned URL for an object.
        
        Args:
            object_name: Object name
            bucket_name: Optional bucket name
            expires: URL expiration time
            
        Returns:
            str: Presigned URL or None if failed
        """
        try:
            bucket = bucket_name or self.bucket_name
            
            url = self.minio_client.presigned_get_object(bucket, object_name, expires=expires)
            
            logger.info(f"✅ Generated presigned URL for {bucket}/{object_name}")
            return url
            
        except Exception as e:
            logger.error(f"❌ Failed to generate presigned URL for '{object_name}': {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on MinIO connection.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            start_time = datetime.now()
            
            # Test connection
            connected = self.connect()
            
            # Test bucket operations
            bucket_exists = False
            if connected:
                bucket_exists = self.minio_client.bucket_exists(self.bucket_name)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            return {
                'connected': connected,
                'bucket_exists': bucket_exists,
                'response_time': response_time,
                'endpoint': f"{self.host}:{self.port}",
                'bucket': self.bucket_name,
                'timestamp': end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                'connected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def bucket_exists(self, bucket_name: str) -> bool:
        """
        Check if a bucket exists in MinIO.
        
        Args:
            bucket_name: Name of the bucket to check
            
        Returns:
            bool: True if bucket exists
        """
        try:
            if not self.minio_client:
                self._initialize_minio_client()
            
            return self.minio_client.bucket_exists(bucket_name)
            
        except Exception as e:
            logger.error(f"❌ Failed to check bucket existence {bucket_name}: {e}")
            return False
    
    def disconnect(self):
        """
        Disconnect from MinIO (cleanup method for consistency with other clients).
        """
        try:
            # MinIO client doesn't require explicit disconnection
            # but we can clear the reference for cleanup
            self.minio_client = None
            logger.info("✅ Disconnected from MinIO")
        except Exception as e:
            logger.error(f"❌ Error during disconnect: {e}")
