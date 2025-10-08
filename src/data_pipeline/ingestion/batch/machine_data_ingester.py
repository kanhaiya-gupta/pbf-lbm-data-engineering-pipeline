"""
Machine Data Ingester for PBF-LB/M Data Pipeline

This module provides batch ingestion capabilities for PBF-LB/M machine data.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.data_pipeline.config.storage_config import get_s3_config
from src.core.domain.entities.pbf_process import PBFProcess

logger = logging.getLogger(__name__)


class MachineDataIngester:
    """
    Batch ingester for PBF-LB/M machine data from various sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize machine data ingester.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.s3_config = get_s3_config()
        self.supported_formats = ['.csv', '.json', '.parquet', '.xlsx']
        self.machine_types = ['PBF-LB/M', 'PBF-EB/M', 'DED', 'SLM']
    
    def ingest_from_csv(self, file_path: str, delimiter: str = ',') -> List[PBFProcess]:
        """
        Ingest machine data from CSV file.
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter
            
        Returns:
            List[PBFProcess]: List of PBF process entities
        """
        logger.info(f"Starting machine data ingestion from CSV: {file_path}")
        
        machine_data = []
        
        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            for _, row in df.iterrows():
                try:
                    pbf_process = self._process_csv_row(row, file_path)
                    if pbf_process:
                        machine_data.append(pbf_process)
                except Exception as e:
                    logger.error(f"Error processing CSV row: {e}")
                    continue
            
            logger.info(f"Successfully ingested {len(machine_data)} machine records from CSV")
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
        
        return machine_data
    
    def ingest_from_json(self, file_path: str) -> List[PBFProcess]:
        """
        Ingest machine data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List[PBFProcess]: List of PBF process entities
        """
        logger.info(f"Starting machine data ingestion from JSON: {file_path}")
        
        machine_data = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle both single object and array of objects
            if isinstance(data, list):
                for item in data:
                    try:
                        pbf_process = self._process_json_item(item, file_path)
                        if pbf_process:
                            machine_data.append(pbf_process)
                    except Exception as e:
                        logger.error(f"Error processing JSON item: {e}")
                        continue
            else:
                pbf_process = self._process_json_item(data, file_path)
                if pbf_process:
                    machine_data.append(pbf_process)
            
            logger.info(f"Successfully ingested {len(machine_data)} machine records from JSON")
            
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
        
        return machine_data
    
    def ingest_from_s3(self, bucket: str, prefix: str) -> List[PBFProcess]:
        """
        Ingest machine data from S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix for machine data files
            
        Returns:
            List[PBFProcess]: List of PBF process entities
        """
        logger.info(f"Starting machine data ingestion from S3: s3://{bucket}/{prefix}")
        
        machine_data = []
        
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
                                        pbf_process = self._process_csv_row(row, f"s3://{bucket}/{key}")
                                        if pbf_process:
                                            machine_data.append(pbf_process)
                                elif key.lower().endswith('.json'):
                                    data = json.loads(content.decode('utf-8'))
                                    if isinstance(data, list):
                                        for item in data:
                                            pbf_process = self._process_json_item(item, f"s3://{bucket}/{key}")
                                            if pbf_process:
                                                machine_data.append(pbf_process)
                                    else:
                                        pbf_process = self._process_json_item(data, f"s3://{bucket}/{key}")
                                        if pbf_process:
                                            machine_data.append(pbf_process)
                                            
                            except Exception as e:
                                logger.error(f"Error processing S3 machine file s3://{bucket}/{key}: {e}")
                                continue
            
            logger.info(f"Successfully ingested {len(machine_data)} machine records from S3")
            
        except ImportError:
            logger.error("boto3 not available for S3 operations")
        except Exception as e:
            logger.error(f"Error ingesting machine data from S3: {e}")
        
        return machine_data
    
    def _process_csv_row(self, row: pd.Series, source_file: str) -> Optional[PBFProcess]:
        """
        Process a single CSV row into PBF process entity.
        
        Args:
            row: CSV row data
            source_file: Source file path
            
        Returns:
            Optional[PBFProcess]: PBF process entity or None if processing failed
        """
        try:
            # Extract required fields
            process_id = str(row.get('process_id', f"PBF_{int(datetime.now().timestamp())}"))
            machine_id = str(row.get('machine_id', 'unknown'))
            material_type = str(row.get('material_type', 'unknown'))
            process_status = str(row.get('process_status', 'STARTED'))
            
            # Parse timestamps
            start_time_str = row.get('start_time', datetime.now().isoformat())
            if isinstance(start_time_str, str):
                start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            else:
                start_time = datetime.now()
            
            end_time = None
            if 'end_time' in row and pd.notna(row['end_time']):
                end_time_str = str(row['end_time'])
                if end_time_str != 'nan':
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
            
            # Create PBF process entity
            pbf_process = PBFProcess(
                process_id=process_id,
                start_time=start_time,
                end_time=end_time,
                machine_id=machine_id,
                material_type=material_type,
                powder_batch_id=str(row.get('powder_batch_id', 'unknown')),
                process_status=process_status,
                total_layers=int(row.get('total_layers', 0)),
                build_height_mm=float(row.get('build_height_mm', 0.0)),
                laser_power_w=float(row.get('laser_power_w', 0.0)),
                scan_speed_mm_s=float(row.get('scan_speed_mm_s', 0.0)),
                layer_thickness_um=float(row.get('layer_thickness_um', 0.0)),
                atmosphere_type=str(row.get('atmosphere_type', 'Argon')),
                oxygen_content_ppm=float(row['oxygen_content_ppm']) if pd.notna(row.get('oxygen_content_ppm')) else None,
                quality_score=float(row['quality_score']) if pd.notna(row.get('quality_score')) else None,
                notes=str(row.get('notes', '')) if pd.notna(row.get('notes')) else None,
                metadata={
                    "source_file": source_file,
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "row_index": row.name
                }
            )
            
            return pbf_process
            
        except Exception as e:
            logger.error(f"Error processing CSV row: {e}")
            return None
    
    def _process_json_item(self, item: Dict[str, Any], source_file: str) -> Optional[PBFProcess]:
        """
        Process a single JSON item into PBF process entity.
        
        Args:
            item: JSON item data
            source_file: Source file path
            
        Returns:
            Optional[PBFProcess]: PBF process entity or None if processing failed
        """
        try:
            # Extract required fields
            process_id = str(item.get('process_id', f"PBF_{int(datetime.now().timestamp())}"))
            machine_id = str(item.get('machine_id', 'unknown'))
            material_type = str(item.get('material_type', 'unknown'))
            process_status = str(item.get('process_status', 'STARTED'))
            
            # Parse timestamps
            start_time_str = item.get('start_time', datetime.now().isoformat())
            if isinstance(start_time_str, str):
                start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            else:
                start_time = datetime.now()
            
            end_time = None
            if 'end_time' in item and item['end_time']:
                end_time_str = str(item['end_time'])
                end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
            
            # Create PBF process entity
            pbf_process = PBFProcess(
                process_id=process_id,
                start_time=start_time,
                end_time=end_time,
                machine_id=machine_id,
                material_type=material_type,
                powder_batch_id=str(item.get('powder_batch_id', 'unknown')),
                process_status=process_status,
                total_layers=int(item.get('total_layers', 0)),
                build_height_mm=float(item.get('build_height_mm', 0.0)),
                laser_power_w=float(item.get('laser_power_w', 0.0)),
                scan_speed_mm_s=float(item.get('scan_speed_mm_s', 0.0)),
                layer_thickness_um=float(item.get('layer_thickness_um', 0.0)),
                atmosphere_type=str(item.get('atmosphere_type', 'Argon')),
                oxygen_content_ppm=float(item['oxygen_content_ppm']) if item.get('oxygen_content_ppm') is not None else None,
                quality_score=float(item['quality_score']) if item.get('quality_score') is not None else None,
                notes=str(item.get('notes', '')) if item.get('notes') else None,
                metadata={
                    "source_file": source_file,
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "raw_data": item
                }
            )
            
            return pbf_process
            
        except Exception as e:
            logger.error(f"Error processing JSON item: {e}")
            return None
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "supported_formats": self.supported_formats,
            "machine_types": self.machine_types,
            "ingestion_timestamp": datetime.now().isoformat()
        }
