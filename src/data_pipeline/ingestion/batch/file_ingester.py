"""
File Ingester for PBF-LB/M Data Pipeline

This module provides file-based data ingestion capabilities for various
file formats used in PBF-LB/M manufacturing data collection.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional, Union, Iterator
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.data_pipeline.config.storage_config import get_s3_config
from src.data_pipeline.storage.data_lake.s3_client import S3Client

logger = logging.getLogger(__name__)


@dataclass
class FileIngestionConfig:
    """Configuration for file ingestion."""
    supported_formats: List[str] = None
    chunk_size: int = 1000
    encoding: str = 'utf-8'
    delimiter: str = ','
    quote_char: str = '"'
    skip_header: bool = True
    max_file_size_mb: int = 100
    temp_directory: str = '/tmp/pbf_lbm_ingestion'
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.csv', '.json', '.parquet', '.xlsx', '.txt', '.log']


class FileIngester:
    """
    File-based data ingester for PBF-LB/M data pipeline.
    
    This class provides comprehensive file ingestion capabilities for various
    file formats commonly used in PBF-LB/M manufacturing data collection.
    """
    
    def __init__(self, config: Optional[FileIngestionConfig] = None):
        """
        Initialize the file ingester.
        
        Args:
            config: File ingestion configuration
        """
        self.config = config or FileIngestionConfig()
        self.s3_client = None
        self._ensure_temp_directory()
        
        logger.info(f"File Ingester initialized with supported formats: {self.config.supported_formats}")
    
    def _ensure_temp_directory(self):
        """Ensure temporary directory exists."""
        try:
            os.makedirs(self.config.temp_directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating temp directory: {e}")
            self.config.temp_directory = '/tmp'
    
    def _get_s3_client(self) -> S3Client:
        """Get or create S3 client."""
        if self.s3_client is None:
            s3_config = get_s3_config()
            self.s3_client = S3Client(s3_config)
        return self.s3_client
    
    def _validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate file for ingestion.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if file is valid for ingestion
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            # Check file extension
            if file_path.suffix.lower() not in self.config.supported_formats:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return False
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                logger.error(f"File too large: {file_size_mb:.2f}MB > {self.config.max_file_size_mb}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False
    
    def _detect_file_format(self, file_path: Union[str, Path]) -> str:
        """
        Detect file format based on content and extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Detected file format
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # For ambiguous extensions, try to detect by content
        if extension == '.txt':
            try:
                with open(file_path, 'r', encoding=self.config.encoding) as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{') or first_line.startswith('['):
                        return 'json'
                    elif ',' in first_line or '\t' in first_line:
                        return 'csv'
            except Exception:
                pass
        
        return extension
    
    def ingest_csv_file(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas read_csv parameters
            
        Returns:
            List of data records
        """
        try:
            file_path = Path(file_path)
            
            # Configure CSV reading parameters
            csv_params = {
                'delimiter': kwargs.get('delimiter', self.config.delimiter),
                'quotechar': kwargs.get('quotechar', self.config.quote_char),
                'encoding': kwargs.get('encoding', self.config.encoding),
                'skiprows': 1 if self.config.skip_header else 0,
                'chunksize': kwargs.get('chunksize', self.config.chunk_size)
            }
            
            # Read CSV file
            if csv_params['chunksize']:
                # Read in chunks for large files
                records = []
                for chunk in pd.read_csv(file_path, **csv_params):
                    records.extend(chunk.to_dict('records'))
            else:
                df = pd.read_csv(file_path, **{k: v for k, v in csv_params.items() if k != 'chunksize'})
                records = df.to_dict('records')
            
            # Add metadata
            for record in records:
                record['_ingestion_metadata'] = {
                    'source_file': str(file_path),
                    'ingested_at': datetime.now().isoformat(),
                    'file_format': 'csv',
                    'record_count': len(records)
                }
            
            logger.info(f"Ingested {len(records)} records from CSV file: {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Error ingesting CSV file {file_path}: {e}")
            return []
    
    def ingest_json_file(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest data from JSON file.
        
        Args:
            file_path: Path to JSON file
            **kwargs: Additional parameters
            
        Returns:
            List of data records
        """
        try:
            file_path = Path(file_path)
            
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # If it's a single object, wrap in list
                records = [data]
            else:
                logger.error(f"Unexpected JSON structure in {file_path}")
                return []
            
            # Add metadata
            for record in records:
                record['_ingestion_metadata'] = {
                    'source_file': str(file_path),
                    'ingested_at': datetime.now().isoformat(),
                    'file_format': 'json',
                    'record_count': len(records)
                }
            
            logger.info(f"Ingested {len(records)} records from JSON file: {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Error ingesting JSON file {file_path}: {e}")
            return []
    
    def ingest_parquet_file(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest data from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional parameters
            
        Returns:
            List of data records
        """
        try:
            file_path = Path(file_path)
            
            # Read Parquet file
            df = pd.read_parquet(file_path, **kwargs)
            records = df.to_dict('records')
            
            # Add metadata
            for record in records:
                record['_ingestion_metadata'] = {
                    'source_file': str(file_path),
                    'ingested_at': datetime.now().isoformat(),
                    'file_format': 'parquet',
                    'record_count': len(records)
                }
            
            logger.info(f"Ingested {len(records)} records from Parquet file: {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Error ingesting Parquet file {file_path}: {e}")
            return []
    
    def ingest_excel_file(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest data from Excel file.
        
        Args:
            file_path: Path to Excel file
            **kwargs: Additional parameters
            
        Returns:
            List of data records
        """
        try:
            file_path = Path(file_path)
            
            # Read Excel file
            sheet_name = kwargs.get('sheet_name', 0)
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            records = df.to_dict('records')
            
            # Add metadata
            for record in records:
                record['_ingestion_metadata'] = {
                    'source_file': str(file_path),
                    'ingested_at': datetime.now().isoformat(),
                    'file_format': 'excel',
                    'sheet_name': sheet_name,
                    'record_count': len(records)
                }
            
            logger.info(f"Ingested {len(records)} records from Excel file: {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Error ingesting Excel file {file_path}: {e}")
            return []
    
    def ingest_text_file(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest data from text file.
        
        Args:
            file_path: Path to text file
            **kwargs: Additional parameters
            
        Returns:
            List of data records
        """
        try:
            file_path = Path(file_path)
            
            # Read text file
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                lines = f.readlines()
            
            # Process lines based on format
            records = []
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse as JSON first
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text
                    record = {
                        'line_number': i + 1,
                        'content': line,
                        'line_length': len(line)
                    }
                
                # Add metadata
                record['_ingestion_metadata'] = {
                    'source_file': str(file_path),
                    'ingested_at': datetime.now().isoformat(),
                    'file_format': 'text',
                    'line_number': i + 1,
                    'total_lines': len(lines)
                }
                
                records.append(record)
            
            logger.info(f"Ingested {len(records)} records from text file: {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Error ingesting text file {file_path}: {e}")
            return []
    
    def ingest_file(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest data from file with automatic format detection.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional parameters
            
        Returns:
            List of data records
        """
        try:
            file_path = Path(file_path)
            
            # Validate file
            if not self._validate_file(file_path):
                return []
            
            # Detect file format
            file_format = self._detect_file_format(file_path)
            
            # Ingest based on format
            if file_format == '.csv':
                return self.ingest_csv_file(file_path, **kwargs)
            elif file_format == '.json':
                return self.ingest_json_file(file_path, **kwargs)
            elif file_format == '.parquet':
                return self.ingest_parquet_file(file_path, **kwargs)
            elif file_format in ['.xlsx', '.xls']:
                return self.ingest_excel_file(file_path, **kwargs)
            elif file_format in ['.txt', '.log']:
                return self.ingest_text_file(file_path, **kwargs)
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return []
                
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            return []
    
    def ingest_from_s3(self, bucket: str, key: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest data from S3 file.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            **kwargs: Additional parameters
            
        Returns:
            List of data records
        """
        try:
            s3_client = self._get_s3_client()
            
            # Download file from S3 to temp directory
            temp_file_path = Path(self.config.temp_directory) / Path(key).name
            s3_client.download_file(bucket, key, str(temp_file_path))
            
            # Ingest the downloaded file
            records = self.ingest_file(temp_file_path, **kwargs)
            
            # Clean up temp file
            temp_file_path.unlink(missing_ok=True)
            
            # Add S3 metadata
            for record in records:
                record['_ingestion_metadata']['s3_bucket'] = bucket
                record['_ingestion_metadata']['s3_key'] = key
            
            logger.info(f"Ingested {len(records)} records from S3: s3://{bucket}/{key}")
            return records
            
        except Exception as e:
            logger.error(f"Error ingesting from S3 s3://{bucket}/{key}: {e}")
            return []
    
    def ingest_directory(self, directory_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest all supported files from a directory.
        
        Args:
            directory_path: Path to directory
            **kwargs: Additional parameters
            
        Returns:
            List of all data records from directory
        """
        try:
            directory_path = Path(directory_path)
            all_records = []
            
            # Find all supported files
            for file_path in directory_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.config.supported_formats:
                    records = self.ingest_file(file_path, **kwargs)
                    all_records.extend(records)
            
            logger.info(f"Ingested {len(all_records)} total records from directory: {directory_path}")
            return all_records
            
        except Exception as e:
            logger.error(f"Error ingesting directory {directory_path}: {e}")
            return []
    
    def get_file_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get metadata about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File metadata dictionary
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {'error': 'File does not exist'}
            
            stat = file_path.stat()
            
            metadata = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_size_bytes': stat.st_size,
                'file_size_mb': stat.st_size / (1024 * 1024),
                'file_extension': file_path.suffix.lower(),
                'is_supported': file_path.suffix.lower() in self.config.supported_formats,
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'accessed_at': datetime.fromtimestamp(stat.st_atime).isoformat()
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting file metadata for {file_path}: {e}")
            return {'error': str(e)}


# Convenience functions for common operations
def ingest_csv_file(file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to ingest CSV file.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional parameters
        
    Returns:
        List of data records
    """
    ingester = FileIngester()
    return ingester.ingest_csv_file(file_path, **kwargs)


def ingest_json_file(file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to ingest JSON file.
    
    Args:
        file_path: Path to JSON file
        **kwargs: Additional parameters
        
    Returns:
        List of data records
    """
    ingester = FileIngester()
    return ingester.ingest_json_file(file_path, **kwargs)


def ingest_file_auto(file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to ingest file with automatic format detection.
    
    Args:
        file_path: Path to file
        **kwargs: Additional parameters
        
    Returns:
        List of data records
    """
    ingester = FileIngester()
    return ingester.ingest_file(file_path, **kwargs)
