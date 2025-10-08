"""
S3 Ingester for PBF-LB/M Data Pipeline

This module provides batch ingestion capabilities for data stored in S3.
"""

import logging
import os
import json
import io
from typing import Dict, Any, Optional, List, Iterator
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.data_pipeline.config.storage_config import get_s3_config

logger = logging.getLogger(__name__)


class S3Ingester:
    """
    Batch ingester for data stored in S3.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize S3 ingester.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.s3_config = get_s3_config()
        self.supported_formats = ['.csv', '.json', '.parquet', '.xlsx', '.txt', '.log']
        self.s3_client = None
        self._initialize_s3_client()
    
    def _initialize_s3_client(self) -> None:
        """Initialize S3 client."""
        try:
            import boto3
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.s3_config.get('aws_access_key_id'),
                aws_secret_access_key=self.s3_config.get('aws_secret_access_key'),
                endpoint_url=self.s3_config.get('endpoint_url'),
                region_name=self.s3_config.get('region_name', 'us-east-1')
            )
            logger.info("S3 client initialized successfully")
        except ImportError:
            logger.error("boto3 not available for S3 operations")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
    
    def list_objects(self, bucket: str, prefix: str = '', max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket with given prefix.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix
            max_keys: Maximum number of keys to return
            
        Returns:
            List[Dict[str, Any]]: List of object metadata
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return []
        
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
                            'etag': obj['ETag']
                        })
            
            logger.info(f"Listed {len(objects)} objects in s3://{bucket}/{prefix}")
            
        except Exception as e:
            logger.error(f"Error listing S3 objects: {e}")
        
        return objects
    
    def download_file(self, bucket: str, key: str, local_path: str) -> bool:
        """
        Download a file from S3 to local path.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            local_path: Local file path
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        try:
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading file s3://{bucket}/{key}: {e}")
            return False
    
    def read_file_content(self, bucket: str, key: str) -> Optional[bytes]:
        """
        Read file content from S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            Optional[bytes]: File content or None if error
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return None
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()
            logger.debug(f"Read {len(content)} bytes from s3://{bucket}/{key}")
            return content
        except Exception as e:
            logger.error(f"Error reading file s3://{bucket}/{key}: {e}")
            return None
    
    def ingest_csv_files(self, bucket: str, prefix: str, delimiter: str = ',') -> List[pd.DataFrame]:
        """
        Ingest CSV files from S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix for CSV files
            delimiter: CSV delimiter
            
        Returns:
            List[pd.DataFrame]: List of DataFrames
        """
        logger.info(f"Starting CSV ingestion from S3: s3://{bucket}/{prefix}")
        
        dataframes = []
        
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return dataframes
        
        try:
            objects = self.list_objects(bucket, prefix)
            csv_objects = [obj for obj in objects if obj['key'].lower().endswith('.csv')]
            
            for obj in csv_objects:
                try:
                    content = self.read_file_content(bucket, obj['key'])
                    if content:
                        df = pd.read_csv(io.StringIO(content.decode('utf-8')), delimiter=delimiter)
                        df['_s3_source'] = f"s3://{bucket}/{obj['key']}"
                        dataframes.append(df)
                        logger.debug(f"Processed CSV file: {obj['key']}")
                except Exception as e:
                    logger.error(f"Error processing CSV file {obj['key']}: {e}")
                    continue
            
            logger.info(f"Successfully ingested {len(dataframes)} CSV files from S3")
            
        except Exception as e:
            logger.error(f"Error ingesting CSV files from S3: {e}")
        
        return dataframes
    
    def ingest_json_files(self, bucket: str, prefix: str) -> List[Dict[str, Any]]:
        """
        Ingest JSON files from S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix for JSON files
            
        Returns:
            List[Dict[str, Any]]: List of JSON data
        """
        logger.info(f"Starting JSON ingestion from S3: s3://{bucket}/{prefix}")
        
        json_data = []
        
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return json_data
        
        try:
            objects = self.list_objects(bucket, prefix)
            json_objects = [obj for obj in objects if obj['key'].lower().endswith('.json')]
            
            for obj in json_objects:
                try:
                    content = self.read_file_content(bucket, obj['key'])
                    if content:
                        data = json.loads(content.decode('utf-8'))
                        if isinstance(data, list):
                            for item in data:
                                item['_s3_source'] = f"s3://{bucket}/{obj['key']}"
                                json_data.append(item)
                        else:
                            data['_s3_source'] = f"s3://{bucket}/{obj['key']}"
                            json_data.append(data)
                        logger.debug(f"Processed JSON file: {obj['key']}")
                except Exception as e:
                    logger.error(f"Error processing JSON file {obj['key']}: {e}")
                    continue
            
            logger.info(f"Successfully ingested {len(json_data)} JSON records from S3")
            
        except Exception as e:
            logger.error(f"Error ingesting JSON files from S3: {e}")
        
        return json_data
    
    def ingest_parquet_files(self, bucket: str, prefix: str) -> List[pd.DataFrame]:
        """
        Ingest Parquet files from S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix for Parquet files
            
        Returns:
            List[pd.DataFrame]: List of DataFrames
        """
        logger.info(f"Starting Parquet ingestion from S3: s3://{bucket}/{prefix}")
        
        dataframes = []
        
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return dataframes
        
        try:
            objects = self.list_objects(bucket, prefix)
            parquet_objects = [obj for obj in objects if obj['key'].lower().endswith('.parquet')]
            
            for obj in parquet_objects:
                try:
                    content = self.read_file_content(bucket, obj['key'])
                    if content:
                        df = pd.read_parquet(io.BytesIO(content))
                        df['_s3_source'] = f"s3://{bucket}/{obj['key']}"
                        dataframes.append(df)
                        logger.debug(f"Processed Parquet file: {obj['key']}")
                except Exception as e:
                    logger.error(f"Error processing Parquet file {obj['key']}: {e}")
                    continue
            
            logger.info(f"Successfully ingested {len(dataframes)} Parquet files from S3")
            
        except Exception as e:
            logger.error(f"Error ingesting Parquet files from S3: {e}")
        
        return dataframes
    
    def ingest_log_files(self, bucket: str, prefix: str) -> List[str]:
        """
        Ingest log files from S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix for log files
            
        Returns:
            List[str]: List of log lines
        """
        logger.info(f"Starting log ingestion from S3: s3://{bucket}/{prefix}")
        
        log_lines = []
        
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return log_lines
        
        try:
            objects = self.list_objects(bucket, prefix)
            log_objects = [obj for obj in objects if obj['key'].lower().endswith(('.log', '.txt'))]
            
            for obj in log_objects:
                try:
                    content = self.read_file_content(bucket, obj['key'])
                    if content:
                        lines = content.decode('utf-8').split('\n')
                        for line in lines:
                            if line.strip():
                                log_lines.append(f"{obj['key']}: {line.strip()}")
                        logger.debug(f"Processed log file: {obj['key']}")
                except Exception as e:
                    logger.error(f"Error processing log file {obj['key']}: {e}")
                    continue
            
            logger.info(f"Successfully ingested {len(log_lines)} log lines from S3")
            
        except Exception as e:
            logger.error(f"Error ingesting log files from S3: {e}")
        
        return log_lines
    
    def get_bucket_info(self, bucket: str) -> Dict[str, Any]:
        """
        Get information about S3 bucket.
        
        Args:
            bucket: S3 bucket name
            
        Returns:
            Dict[str, Any]: Bucket information
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return {}
        
        try:
            response = self.s3_client.head_bucket(Bucket=bucket)
            return {
                "bucket": bucket,
                "region": response.get('ResponseMetadata', {}).get('HTTPHeaders', {}).get('x-amz-bucket-region'),
                "status": "accessible"
            }
        except Exception as e:
            logger.error(f"Error getting bucket info for {bucket}: {e}")
            return {"bucket": bucket, "status": "error", "error": str(e)}
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "supported_formats": self.supported_formats,
            "s3_client_initialized": self.s3_client is not None,
            "ingestion_timestamp": datetime.now().isoformat()
        }
