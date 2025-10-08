"""
CT Data Ingester for PBF-LB/M Data Pipeline

This module provides batch ingestion capabilities for CT scan data.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Iterator
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.data_pipeline.config.storage_config import get_s3_config
from src.core.domain.entities.ct_scan import CTScan

logger = logging.getLogger(__name__)


class CTDataIngester:
    """
    Batch ingester for CT scan data from various sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CT data ingester.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.s3_config = get_s3_config()
        self.supported_formats = ['.dcm', '.nii', '.nii.gz', '.mhd', '.raw', '.json']
        self.metadata_cache = {}
    
    def ingest_from_directory(self, directory_path: str, recursive: bool = True) -> List[CTScan]:
        """
        Ingest CT scan data from a directory.
        
        Args:
            directory_path: Path to directory containing CT scan files
            recursive: Whether to search recursively in subdirectories
            
        Returns:
            List[CTScan]: List of CT scan entities
        """
        logger.info(f"Starting CT data ingestion from directory: {directory_path}")
        
        ct_scans = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return ct_scans
        
        # Find all CT scan files
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    ct_scan = self._process_ct_file(file_path)
                    if ct_scan:
                        ct_scans.append(ct_scan)
                except Exception as e:
                    logger.error(f"Error processing CT file {file_path}: {e}")
                    continue
        
        logger.info(f"Successfully ingested {len(ct_scans)} CT scans from {directory_path}")
        return ct_scans
    
    def ingest_from_s3(self, bucket: str, prefix: str) -> List[CTScan]:
        """
        Ingest CT scan data from S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix for CT scan files
            
        Returns:
            List[CTScan]: List of CT scan entities
        """
        logger.info(f"Starting CT data ingestion from S3: s3://{bucket}/{prefix}")
        
        ct_scans = []
        
        try:
            import boto3
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.s3_config.get('aws_access_key_id'),
                aws_secret_access_key=self.s3_config.get('aws_secret_access_key'),
                endpoint_url=self.s3_config.get('endpoint_url')
            )
            
            # List objects in S3
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if any(key.lower().endswith(fmt) for fmt in self.supported_formats):
                            try:
                                ct_scan = self._process_s3_ct_file(bucket, key, s3_client)
                                if ct_scan:
                                    ct_scans.append(ct_scan)
                            except Exception as e:
                                logger.error(f"Error processing S3 CT file s3://{bucket}/{key}: {e}")
                                continue
            
            logger.info(f"Successfully ingested {len(ct_scans)} CT scans from S3")
            
        except ImportError:
            logger.error("boto3 not available for S3 operations")
        except Exception as e:
            logger.error(f"Error ingesting CT data from S3: {e}")
        
        return ct_scans
    
    def ingest_from_api(self, api_endpoint: str, api_key: Optional[str] = None) -> List[CTScan]:
        """
        Ingest CT scan data from an API endpoint.
        
        Args:
            api_endpoint: API endpoint URL
            api_key: Optional API key for authentication
            
        Returns:
            List[CTScan]: List of CT scan entities
        """
        logger.info(f"Starting CT data ingestion from API: {api_endpoint}")
        
        ct_scans = []
        
        try:
            import requests
            
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            response = requests.get(api_endpoint, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Process API response data
            if isinstance(data, list):
                for item in data:
                    try:
                        ct_scan = self._process_api_ct_data(item)
                        if ct_scan:
                            ct_scans.append(ct_scan)
                    except Exception as e:
                        logger.error(f"Error processing API CT data: {e}")
                        continue
            elif isinstance(data, dict):
                ct_scan = self._process_api_ct_data(data)
                if ct_scan:
                    ct_scans.append(ct_scan)
            
            logger.info(f"Successfully ingested {len(ct_scans)} CT scans from API")
            
        except ImportError:
            logger.error("requests not available for API operations")
        except Exception as e:
            logger.error(f"Error ingesting CT data from API: {e}")
        
        return ct_scans
    
    def _process_ct_file(self, file_path: Path) -> Optional[CTScan]:
        """
        Process a single CT scan file.
        
        Args:
            file_path: Path to CT scan file
            
        Returns:
            Optional[CTScan]: CT scan entity or None if processing failed
        """
        try:
            file_size = file_path.stat().st_size
            file_extension = file_path.suffix.lower()
            
            # Extract metadata based on file type
            metadata = self._extract_ct_metadata(file_path, file_extension)
            
            # Generate unique scan ID
            scan_id = f"CT_{file_path.stem}_{int(datetime.now().timestamp())}"
            
            # Create CT scan entity
            ct_scan = CTScan(
                scan_id=scan_id,
                process_id=metadata.get("process_id", "unknown"),
                timestamp=datetime.now(),
                voxel_dimensions=metadata.get("voxel_dimensions", [512, 512, 100]),
                resolution=metadata.get("resolution", 0.1),
                file_path=str(file_path),
                file_size=file_size,
                quality_score=metadata.get("quality_score", 0.8),
                defect_count=metadata.get("defect_count", 0),
                processing_status="ingested",
                metadata=metadata
            )
            
            logger.debug(f"Processed CT file: {file_path}")
            return ct_scan
            
        except Exception as e:
            logger.error(f"Error processing CT file {file_path}: {e}")
            return None
    
    def _process_s3_ct_file(self, bucket: str, key: str, s3_client) -> Optional[CTScan]:
        """
        Process a CT scan file from S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            s3_client: S3 client instance
            
        Returns:
            Optional[CTScan]: CT scan entity or None if processing failed
        """
        try:
            # Get object metadata
            response = s3_client.head_object(Bucket=bucket, Key=key)
            file_size = response['ContentLength']
            
            # Extract metadata
            metadata = self._extract_s3_ct_metadata(bucket, key, response)
            
            # Generate unique scan ID
            scan_id = f"CT_S3_{Path(key).stem}_{int(datetime.now().timestamp())}"
            
            # Create CT scan entity
            ct_scan = CTScan(
                scan_id=scan_id,
                process_id=metadata.get("process_id", "unknown"),
                timestamp=datetime.now(),
                voxel_dimensions=metadata.get("voxel_dimensions", [512, 512, 100]),
                resolution=metadata.get("resolution", 0.1),
                file_path=f"s3://{bucket}/{key}",
                file_size=file_size,
                quality_score=metadata.get("quality_score", 0.8),
                defect_count=metadata.get("defect_count", 0),
                processing_status="ingested",
                metadata=metadata
            )
            
            logger.debug(f"Processed S3 CT file: s3://{bucket}/{key}")
            return ct_scan
            
        except Exception as e:
            logger.error(f"Error processing S3 CT file s3://{bucket}/{key}: {e}")
            return None
    
    def _process_api_ct_data(self, data: Dict[str, Any]) -> Optional[CTScan]:
        """
        Process CT scan data from API response.
        
        Args:
            data: API response data
            
        Returns:
            Optional[CTScan]: CT scan entity or None if processing failed
        """
        try:
            # Extract metadata from API data
            metadata = {
                "process_id": data.get("process_id"),
                "voxel_dimensions": data.get("voxel_dimensions", [512, 512, 100]),
                "resolution": data.get("resolution", 0.1),
                "quality_score": data.get("quality_score", 0.8),
                "defect_count": data.get("defect_count", 0),
                "api_source": True
            }
            
            # Generate unique scan ID
            scan_id = data.get("scan_id", f"CT_API_{int(datetime.now().timestamp())}")
            
            # Create CT scan entity
            ct_scan = CTScan(
                scan_id=scan_id,
                process_id=data.get("process_id", "unknown"),
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                voxel_dimensions=metadata["voxel_dimensions"],
                resolution=metadata["resolution"],
                file_path=data.get("file_path", "api_source"),
                file_size=data.get("file_size", 0),
                quality_score=metadata["quality_score"],
                defect_count=metadata["defect_count"],
                processing_status="ingested",
                metadata=metadata
            )
            
            logger.debug(f"Processed API CT data: {scan_id}")
            return ct_scan
            
        except Exception as e:
            logger.error(f"Error processing API CT data: {e}")
            return None
    
    def _extract_ct_metadata(self, file_path: Path, file_extension: str) -> Dict[str, Any]:
        """
        Extract metadata from CT scan file.
        
        Args:
            file_path: Path to CT scan file
            file_extension: File extension
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        metadata = {
            "file_type": file_extension,
            "ingestion_timestamp": datetime.now().isoformat()
        }
        
        try:
            if file_extension == '.json':
                # Read JSON metadata file
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                    metadata.update(json_data)
            elif file_extension in ['.dcm', '.nii', '.nii.gz']:
                # For medical imaging formats, we would use specialized libraries
                # like pydicom, nibabel, etc. in a real implementation
                metadata.update({
                    "format": "medical_imaging",
                    "voxel_dimensions": [512, 512, 100],  # Default values
                    "resolution": 0.1
                })
            else:
                # Generic metadata extraction
                metadata.update({
                    "format": "generic",
                    "voxel_dimensions": [512, 512, 100],
                    "resolution": 0.1
                })
                
        except Exception as e:
            logger.warning(f"Could not extract metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_s3_ct_metadata(self, bucket: str, key: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from S3 CT scan file.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            response: S3 head_object response
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        metadata = {
            "s3_bucket": bucket,
            "s3_key": key,
            "s3_etag": response.get('ETag', ''),
            "s3_last_modified": response.get('LastModified', datetime.now()).isoformat(),
            "ingestion_timestamp": datetime.now().isoformat()
        }
        
        # Extract custom metadata if available
        if 'Metadata' in response:
            metadata.update(response['Metadata'])
        
        return metadata
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "supported_formats": self.supported_formats,
            "metadata_cache_size": len(self.metadata_cache),
            "ingestion_timestamp": datetime.now().isoformat()
        }
