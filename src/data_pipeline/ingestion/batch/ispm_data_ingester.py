"""
ISPM Data Ingester for PBF-LB/M Data Pipeline

This module provides batch ingestion capabilities for ISPM (In-Situ Process Monitoring) data.
"""

import logging
import os
import json
import csv
from typing import Dict, Any, Optional, List, Iterator
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.data_pipeline.config.storage_config import get_s3_config
from src.core.domain.entities.ispm_monitoring import ISPMMonitoring

logger = logging.getLogger(__name__)


class ISPMDataIngester:
    """
    Batch ingester for ISPM monitoring data from various sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ISPM data ingester.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.s3_config = get_s3_config()
        self.supported_formats = ['.csv', '.json', '.parquet', '.xlsx']
        self.sensor_types = ['temperature', 'pressure', 'laser_power', 'scan_speed', 'vibration', 'acoustic']
    
    def ingest_from_csv(self, file_path: str, delimiter: str = ',') -> List[ISPMMonitoring]:
        """
        Ingest ISPM monitoring data from CSV file.
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter
            
        Returns:
            List[ISPMMonitoring]: List of ISPM monitoring entities
        """
        logger.info(f"Starting ISPM data ingestion from CSV: {file_path}")
        
        ispm_data = []
        
        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            for _, row in df.iterrows():
                try:
                    ispm_monitoring = self._process_csv_row(row, file_path)
                    if ispm_monitoring:
                        ispm_data.append(ispm_monitoring)
                except Exception as e:
                    logger.error(f"Error processing CSV row: {e}")
                    continue
            
            logger.info(f"Successfully ingested {len(ispm_data)} ISPM records from CSV")
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
        
        return ispm_data
    
    def ingest_from_json(self, file_path: str) -> List[ISPMMonitoring]:
        """
        Ingest ISPM monitoring data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List[ISPMMonitoring]: List of ISPM monitoring entities
        """
        logger.info(f"Starting ISPM data ingestion from JSON: {file_path}")
        
        ispm_data = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle both single object and array of objects
            if isinstance(data, list):
                for item in data:
                    try:
                        ispm_monitoring = self._process_json_item(item, file_path)
                        if ispm_monitoring:
                            ispm_data.append(ispm_monitoring)
                    except Exception as e:
                        logger.error(f"Error processing JSON item: {e}")
                        continue
            else:
                ispm_monitoring = self._process_json_item(data, file_path)
                if ispm_monitoring:
                    ispm_data.append(ispm_monitoring)
            
            logger.info(f"Successfully ingested {len(ispm_data)} ISPM records from JSON")
            
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
        
        return ispm_data
    
    def ingest_from_parquet(self, file_path: str) -> List[ISPMMonitoring]:
        """
        Ingest ISPM monitoring data from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            
        Returns:
            List[ISPMMonitoring]: List of ISPM monitoring entities
        """
        logger.info(f"Starting ISPM data ingestion from Parquet: {file_path}")
        
        ispm_data = []
        
        try:
            df = pd.read_parquet(file_path)
            
            for _, row in df.iterrows():
                try:
                    ispm_monitoring = self._process_parquet_row(row, file_path)
                    if ispm_monitoring:
                        ispm_data.append(ispm_monitoring)
                except Exception as e:
                    logger.error(f"Error processing Parquet row: {e}")
                    continue
            
            logger.info(f"Successfully ingested {len(ispm_data)} ISPM records from Parquet")
            
        except Exception as e:
            logger.error(f"Error reading Parquet file {file_path}: {e}")
        
        return ispm_data
    
    def ingest_from_s3(self, bucket: str, prefix: str) -> List[ISPMMonitoring]:
        """
        Ingest ISPM monitoring data from S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix for ISPM data files
            
        Returns:
            List[ISPMMonitoring]: List of ISPM monitoring entities
        """
        logger.info(f"Starting ISPM data ingestion from S3: s3://{bucket}/{prefix}")
        
        ispm_data = []
        
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
                                # Download and process file
                                response = s3_client.get_object(Bucket=bucket, Key=key)
                                content = response['Body'].read()
                                
                                # Process based on file extension
                                if key.lower().endswith('.csv'):
                                    df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                                    for _, row in df.iterrows():
                                        ispm_monitoring = self._process_csv_row(row, f"s3://{bucket}/{key}")
                                        if ispm_monitoring:
                                            ispm_data.append(ispm_monitoring)
                                elif key.lower().endswith('.json'):
                                    data = json.loads(content.decode('utf-8'))
                                    if isinstance(data, list):
                                        for item in data:
                                            ispm_monitoring = self._process_json_item(item, f"s3://{bucket}/{key}")
                                            if ispm_monitoring:
                                                ispm_data.append(ispm_monitoring)
                                    else:
                                        ispm_monitoring = self._process_json_item(data, f"s3://{bucket}/{key}")
                                        if ispm_monitoring:
                                            ispm_data.append(ispm_monitoring)
                                            
                            except Exception as e:
                                logger.error(f"Error processing S3 ISPM file s3://{bucket}/{key}: {e}")
                                continue
            
            logger.info(f"Successfully ingested {len(ispm_data)} ISPM records from S3")
            
        except ImportError:
            logger.error("boto3 not available for S3 operations")
        except Exception as e:
            logger.error(f"Error ingesting ISPM data from S3: {e}")
        
        return ispm_data
    
    def ingest_from_database(self, connection_string: str, query: str) -> List[ISPMMonitoring]:
        """
        Ingest ISPM monitoring data from database.
        
        Args:
            connection_string: Database connection string
            query: SQL query to fetch ISPM data
            
        Returns:
            List[ISPMMonitoring]: List of ISPM monitoring entities
        """
        logger.info("Starting ISPM data ingestion from database")
        
        ispm_data = []
        
        try:
            import sqlalchemy
            
            engine = sqlalchemy.create_engine(connection_string)
            df = pd.read_sql(query, engine)
            
            for _, row in df.iterrows():
                try:
                    ispm_monitoring = self._process_database_row(row, connection_string)
                    if ispm_monitoring:
                        ispm_data.append(ispm_monitoring)
                except Exception as e:
                    logger.error(f"Error processing database row: {e}")
                    continue
            
            logger.info(f"Successfully ingested {len(ispm_data)} ISPM records from database")
            
        except ImportError:
            logger.error("sqlalchemy not available for database operations")
        except Exception as e:
            logger.error(f"Error ingesting ISPM data from database: {e}")
        
        return ispm_data
    
    def _process_csv_row(self, row: pd.Series, source_file: str) -> Optional[ISPMMonitoring]:
        """
        Process a single CSV row into ISPM monitoring entity.
        
        Args:
            row: CSV row data
            source_file: Source file path
            
        Returns:
            Optional[ISPMMonitoring]: ISPM monitoring entity or None if processing failed
        """
        try:
            # Extract required fields
            monitoring_id = str(row.get('monitoring_id', f"ISPM_{int(datetime.now().timestamp())}"))
            process_id = str(row.get('process_id', 'unknown'))
            sensor_id = str(row.get('sensor_id', 'unknown'))
            sensor_type = str(row.get('sensor_type', 'unknown'))
            sensor_value = float(row.get('sensor_value', 0.0))
            unit = str(row.get('unit', ''))
            
            # Parse timestamp
            timestamp_str = row.get('timestamp', datetime.now().isoformat())
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            # Create ISPM monitoring entity
            ispm_monitoring = ISPMMonitoring(
                monitoring_id=monitoring_id,
                process_id=process_id,
                sensor_id=sensor_id,
                timestamp=timestamp,
                sensor_type=sensor_type,
                sensor_value=sensor_value,
                unit=unit,
                metadata={
                    "source_file": source_file,
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "row_index": row.name
                }
            )
            
            return ispm_monitoring
            
        except Exception as e:
            logger.error(f"Error processing CSV row: {e}")
            return None
    
    def _process_json_item(self, item: Dict[str, Any], source_file: str) -> Optional[ISPMMonitoring]:
        """
        Process a single JSON item into ISPM monitoring entity.
        
        Args:
            item: JSON item data
            source_file: Source file path
            
        Returns:
            Optional[ISPMMonitoring]: ISPM monitoring entity or None if processing failed
        """
        try:
            # Extract required fields
            monitoring_id = str(item.get('monitoring_id', f"ISPM_{int(datetime.now().timestamp())}"))
            process_id = str(item.get('process_id', 'unknown'))
            sensor_id = str(item.get('sensor_id', 'unknown'))
            sensor_type = str(item.get('sensor_type', 'unknown'))
            sensor_value = float(item.get('sensor_value', 0.0))
            unit = str(item.get('unit', ''))
            
            # Parse timestamp
            timestamp_str = item.get('timestamp', datetime.now().isoformat())
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            # Create ISPM monitoring entity
            ispm_monitoring = ISPMMonitoring(
                monitoring_id=monitoring_id,
                process_id=process_id,
                sensor_id=sensor_id,
                timestamp=timestamp,
                sensor_type=sensor_type,
                sensor_value=sensor_value,
                unit=unit,
                metadata={
                    "source_file": source_file,
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "raw_data": item
                }
            )
            
            return ispm_monitoring
            
        except Exception as e:
            logger.error(f"Error processing JSON item: {e}")
            return None
    
    def _process_parquet_row(self, row: pd.Series, source_file: str) -> Optional[ISPMMonitoring]:
        """
        Process a single Parquet row into ISPM monitoring entity.
        
        Args:
            row: Parquet row data
            source_file: Source file path
            
        Returns:
            Optional[ISPMMonitoring]: ISPM monitoring entity or None if processing failed
        """
        try:
            # Extract required fields
            monitoring_id = str(row.get('monitoring_id', f"ISPM_{int(datetime.now().timestamp())}"))
            process_id = str(row.get('process_id', 'unknown'))
            sensor_id = str(row.get('sensor_id', 'unknown'))
            sensor_type = str(row.get('sensor_type', 'unknown'))
            sensor_value = float(row.get('sensor_value', 0.0))
            unit = str(row.get('unit', ''))
            
            # Parse timestamp
            timestamp_str = row.get('timestamp', datetime.now().isoformat())
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            # Create ISPM monitoring entity
            ispm_monitoring = ISPMMonitoring(
                monitoring_id=monitoring_id,
                process_id=process_id,
                sensor_id=sensor_id,
                timestamp=timestamp,
                sensor_type=sensor_type,
                sensor_value=sensor_value,
                unit=unit,
                metadata={
                    "source_file": source_file,
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "row_index": row.name
                }
            )
            
            return ispm_monitoring
            
        except Exception as e:
            logger.error(f"Error processing Parquet row: {e}")
            return None
    
    def _process_database_row(self, row: pd.Series, connection_string: str) -> Optional[ISPMMonitoring]:
        """
        Process a single database row into ISPM monitoring entity.
        
        Args:
            row: Database row data
            connection_string: Database connection string
            
        Returns:
            Optional[ISPMMonitoring]: ISPM monitoring entity or None if processing failed
        """
        try:
            # Extract required fields
            monitoring_id = str(row.get('monitoring_id', f"ISPM_{int(datetime.now().timestamp())}"))
            process_id = str(row.get('process_id', 'unknown'))
            sensor_id = str(row.get('sensor_id', 'unknown'))
            sensor_type = str(row.get('sensor_type', 'unknown'))
            sensor_value = float(row.get('sensor_value', 0.0))
            unit = str(row.get('unit', ''))
            
            # Parse timestamp
            timestamp_str = row.get('timestamp', datetime.now().isoformat())
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            # Create ISPM monitoring entity
            ispm_monitoring = ISPMMonitoring(
                monitoring_id=monitoring_id,
                process_id=process_id,
                sensor_id=sensor_id,
                timestamp=timestamp,
                sensor_type=sensor_type,
                sensor_value=sensor_value,
                unit=unit,
                metadata={
                    "source_database": connection_string,
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "row_index": row.name
                }
            )
            
            return ispm_monitoring
            
        except Exception as e:
            logger.error(f"Error processing database row: {e}")
            return None
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "supported_formats": self.supported_formats,
            "sensor_types": self.sensor_types,
            "ingestion_timestamp": datetime.now().isoformat()
        }
