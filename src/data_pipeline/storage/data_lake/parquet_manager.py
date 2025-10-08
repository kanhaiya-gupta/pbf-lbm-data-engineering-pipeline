"""
Parquet Manager for PBF-LB/M Data Pipeline

This module provides Parquet file management capabilities.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class ParquetManager:
    """
    Parquet file manager for data lake operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Parquet manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.default_path = self.config.get('default_path', '/data/parquet')
        self.compression = self.config.get('compression', 'snappy')
        self.engine = self.config.get('engine', 'pyarrow')
        self.spark_session = None
        self._initialize_spark_session()
    
    def _initialize_spark_session(self) -> None:
        """Initialize Spark session for Parquet operations."""
        try:
            from pyspark.sql import SparkSession
            
            self.spark_session = SparkSession.builder \
                .appName("PBF_LB_M_ParquetManager") \
                .getOrCreate()
            
            logger.info("Parquet Spark session initialized successfully")
        except ImportError:
            logger.error("pyspark not available for Parquet operations")
        except Exception as e:
            logger.error(f"Failed to initialize Parquet Spark session: {e}")
    
    def write_parquet(self, data: Any, path: str, mode: str = "overwrite", partition_by: Optional[List[str]] = None) -> bool:
        """
        Write data to Parquet format.
        
        Args:
            data: Data to write (DataFrame or data source)
            path: Output path for Parquet files
            mode: Write mode ("overwrite", "append", "ignore", "error")
            partition_by: List of columns to partition by
            
        Returns:
            bool: True if write successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Write data to Parquet
            writer = data.write \
                .format("parquet") \
                .mode(mode) \
                .option("compression", self.compression)
            
            if partition_by:
                writer = writer.partitionBy(*partition_by)
            
            writer.save(path)
            
            logger.info(f"Wrote data to Parquet: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing data to Parquet {path}: {e}")
            return False
    
    def read_parquet(self, path: str, schema: Optional[Any] = None) -> Optional[Any]:
        """
        Read data from Parquet format.
        
        Args:
            path: Path to Parquet files
            schema: Optional schema for reading
            
        Returns:
            Optional[Any]: DataFrame or None if error
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return None
        
        try:
            # Read data from Parquet
            reader = self.spark_session.read.format("parquet")
            
            if schema:
                reader = reader.schema(schema)
            
            df = reader.load(path)
            
            logger.info(f"Read data from Parquet: {path}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading data from Parquet {path}: {e}")
            return None
    
    def write_pandas_parquet(self, df: pd.DataFrame, path: str, compression: Optional[str] = None) -> bool:
        """
        Write pandas DataFrame to Parquet format.
        
        Args:
            df: Pandas DataFrame to write
            path: Output path for Parquet file
            compression: Compression type (uses default if None)
            
        Returns:
            bool: True if write successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Write DataFrame to Parquet
            compression = compression or self.compression
            df.to_parquet(path, compression=compression, engine=self.engine)
            
            logger.info(f"Wrote pandas DataFrame to Parquet: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing pandas DataFrame to Parquet {path}: {e}")
            return False
    
    def read_pandas_parquet(self, path: str) -> Optional[pd.DataFrame]:
        """
        Read Parquet file into pandas DataFrame.
        
        Args:
            path: Path to Parquet file
            
        Returns:
            Optional[pd.DataFrame]: DataFrame or None if error
        """
        try:
            # Read Parquet file into DataFrame
            df = pd.read_parquet(path, engine=self.engine)
            
            logger.info(f"Read Parquet file into pandas DataFrame: {path}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading Parquet file {path}: {e}")
            return None
    
    def merge_parquet_files(self, input_paths: List[str], output_path: str, schema: Optional[Any] = None) -> bool:
        """
        Merge multiple Parquet files into one.
        
        Args:
            input_paths: List of input Parquet file paths
            output_path: Output path for merged Parquet file
            schema: Optional schema for merging
            
        Returns:
            bool: True if merge successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            # Read all input files
            dfs = []
            for path in input_paths:
                df = self.read_parquet(path, schema)
                if df:
                    dfs.append(df)
            
            if not dfs:
                logger.error("No valid DataFrames to merge")
                return False
            
            # Union all DataFrames
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = merged_df.union(df)
            
            # Write merged DataFrame
            success = self.write_parquet(merged_df, output_path)
            
            if success:
                logger.info(f"Merged {len(input_paths)} Parquet files into: {output_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error merging Parquet files: {e}")
            return False
    
    def get_parquet_info(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about Parquet files.
        
        Args:
            path: Path to Parquet files
            
        Returns:
            Optional[Dict[str, Any]]: Parquet file information or None if error
        """
        try:
            # Get file information
            file_info = {
                "path": path,
                "exists": os.path.exists(path),
                "size_bytes": 0,
                "file_count": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            if os.path.exists(path):
                if os.path.isfile(path):
                    file_info["size_bytes"] = os.path.getsize(path)
                    file_info["file_count"] = 1
                else:
                    # Directory - count files and total size
                    total_size = 0
                    file_count = 0
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            if file.endswith('.parquet'):
                                file_path = os.path.join(root, file)
                                total_size += os.path.getsize(file_path)
                                file_count += 1
                    
                    file_info["size_bytes"] = total_size
                    file_info["file_count"] = file_count
            
            logger.info(f"Retrieved Parquet info for: {path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting Parquet info for {path}: {e}")
            return None
    
    def optimize_parquet(self, path: str, target_file_size: int = 128 * 1024 * 1024) -> bool:
        """
        Optimize Parquet files by repartitioning.
        
        Args:
            path: Path to Parquet files
            target_file_size: Target file size in bytes
            
        Returns:
            bool: True if optimization successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            # Read existing Parquet files
            df = self.read_parquet(path)
            if not df:
                logger.error("No data to optimize")
                return False
            
            # Get current row count and estimate partitions
            row_count = df.count()
            if row_count == 0:
                logger.warning("No rows to optimize")
                return False
            
            # Estimate optimal number of partitions
            # This is a simplified calculation
            optimal_partitions = max(1, row_count // 1000000)  # 1M rows per partition
            
            # Repartition and write back
            optimized_df = df.repartition(optimal_partitions)
            success = self.write_parquet(optimized_df, path, mode="overwrite")
            
            if success:
                logger.info(f"Optimized Parquet files at: {path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error optimizing Parquet files {path}: {e}")
            return False
    
    def validate_parquet(self, path: str, schema: Optional[Any] = None) -> Dict[str, Any]:
        """
        Validate Parquet files.
        
        Args:
            path: Path to Parquet files
            schema: Optional schema to validate against
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            "path": path,
            "valid": False,
            "errors": [],
            "warnings": [],
            "row_count": 0,
            "column_count": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Read Parquet files
            df = self.read_parquet(path, schema)
            if not df:
                validation_results["errors"].append("Failed to read Parquet files")
                return validation_results
            
            # Basic validation
            validation_results["row_count"] = df.count()
            validation_results["column_count"] = len(df.columns)
            
            # Check for null values
            null_counts = {}
            for col in df.columns:
                null_count = df.filter(df[col].isNull()).count()
                if null_count > 0:
                    null_counts[col] = null_count
            
            if null_counts:
                validation_results["warnings"].append(f"Null values found: {null_counts}")
            
            # Check for duplicate rows
            duplicate_count = df.count() - df.distinct().count()
            if duplicate_count > 0:
                validation_results["warnings"].append(f"Duplicate rows found: {duplicate_count}")
            
            # If no errors, mark as valid
            if not validation_results["errors"]:
                validation_results["valid"] = True
            
            logger.info(f"Validated Parquet files at: {path}")
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            logger.error(f"Error validating Parquet files {path}: {e}")
        
        return validation_results
    
    def convert_to_parquet(self, input_path: str, output_path: str, input_format: str = "csv") -> bool:
        """
        Convert data from other formats to Parquet.
        
        Args:
            input_path: Input file path
            output_path: Output Parquet path
            input_format: Input format ("csv", "json", "avro")
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            # Read data based on input format
            if input_format.lower() == "csv":
                df = self.spark_session.read.format("csv").option("header", "true").load(input_path)
            elif input_format.lower() == "json":
                df = self.spark_session.read.format("json").load(input_path)
            elif input_format.lower() == "avro":
                df = self.spark_session.read.format("avro").load(input_path)
            else:
                logger.error(f"Unsupported input format: {input_format}")
                return False
            
            # Write to Parquet
            success = self.write_parquet(df, output_path)
            
            if success:
                logger.info(f"Converted {input_format} to Parquet: {input_path} -> {output_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error converting {input_format} to Parquet: {e}")
            return False
    
    def close_spark_session(self) -> None:
        """Close Spark session."""
        if self.spark_session:
            self.spark_session.stop()
            logger.info("Parquet Spark session closed")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "default_path": self.default_path,
            "compression": self.compression,
            "engine": self.engine,
            "spark_session_initialized": self.spark_session is not None,
            "ingestion_timestamp": datetime.now().isoformat()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_spark_session()
