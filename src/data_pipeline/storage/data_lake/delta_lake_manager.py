"""
Delta Lake Manager for PBF-LB/M Data Pipeline

This module provides Delta Lake operations for data lake storage.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
from src.data_pipeline.config.storage_config import get_delta_lake_config

logger = logging.getLogger(__name__)


class DeltaLakeManager:
    """
    Delta Lake manager for data lake operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Delta Lake manager.
        
        Args:
            config: Optional Delta Lake configuration dictionary or Pydantic model
        """
        self.config = config or get_delta_lake_config()
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """Initialize Delta Lake connection parameters."""
        try:
            # Handle both dictionary and Pydantic model configurations
            if isinstance(self.config, dict):
                self.storage_path = self.config.get('storage_path', 's3a://pbf-lbm-data-lake/delta/')
                self.checkpoint_path = self.config.get('checkpoint_path', 's3a://pbf-lbm-data-lake/checkpoints/')
                self.auto_optimize = self.config.get('auto_optimize', True)
                self.optimize_interval = self.config.get('optimize_interval', 24)
                self.retention_period = self.config.get('retention_period', 30)
                self.vacuum_interval = self.config.get('vacuum_interval', 7)
                self.allow_schema_evolution = self.config.get('allow_schema_evolution', True)
                self.merge_schema = self.config.get('merge_schema', True)
            else:
                # Pydantic model
                self.storage_path = self.config.storage_path
                self.checkpoint_path = self.config.checkpoint_path
                self.auto_optimize = self.config.auto_optimize
                self.optimize_interval = self.config.optimize_interval
                self.retention_period = self.config.retention_period
                self.vacuum_interval = self.config.vacuum_interval
                self.allow_schema_evolution = self.config.allow_schema_evolution
                self.merge_schema = self.config.merge_schema
            
            self.delta_table_path = self.storage_path
            
            logger.info("Delta Lake connection parameters initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Delta Lake connection parameters: {e}")
            raise
        self.history_retention_days = self.config.get('history_retention_days', 30)
        self.optimize_write = self.config.get('optimize_write', True)
        self.spark_session = None
        self._initialize_spark_session()
    
    def _initialize_spark_session(self) -> None:
        """Initialize Spark session with Delta Lake support."""
        try:
            from pyspark.sql import SparkSession
            
            self.spark_session = SparkSession.builder \
                .appName("PBF_LB_M_DeltaLake") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .getOrCreate()
            
            logger.info("Delta Lake Spark session initialized successfully")
        except ImportError:
            logger.error("pyspark not available for Delta Lake operations")
        except Exception as e:
            logger.error(f"Failed to initialize Delta Lake Spark session: {e}")
    
    def create_table(self, table_name: str, schema: Any, location: Optional[str] = None) -> bool:
        """
        Create a Delta Lake table.
        
        Args:
            table_name: Name of the table
            schema: Table schema
            location: Table location (uses default if None)
            
        Returns:
            bool: True if table creation successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Create Delta table
            self.spark_session.sql(f"""
                CREATE TABLE IF NOT EXISTS {table_name}
                USING DELTA
                LOCATION '{location}'
                AS SELECT * FROM (
                    SELECT * FROM VALUES (1, 'dummy') AS t(col1, col2)
                ) WHERE 1=0
            """)
            
            logger.info(f"Created Delta Lake table: {table_name} at {location}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Delta Lake table {table_name}: {e}")
            return False
    
    def write_data(self, data: Any, table_name: str, mode: str = "append", location: Optional[str] = None) -> bool:
        """
        Write data to a Delta Lake table.
        
        Args:
            data: Data to write (DataFrame or data source)
            table_name: Name of the table
            mode: Write mode ("append", "overwrite", "merge")
            location: Table location (uses default if None)
            
        Returns:
            bool: True if write successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Write data to Delta table
            data.write \
                .format("delta") \
                .mode(mode) \
                .option("path", location) \
                .saveAsTable(table_name)
            
            logger.info(f"Wrote data to Delta Lake table: {table_name} in mode {mode}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing data to Delta Lake table {table_name}: {e}")
            return False
    
    def read_data(self, table_name: str, location: Optional[str] = None) -> Optional[Any]:
        """
        Read data from a Delta Lake table.
        
        Args:
            table_name: Name of the table
            location: Table location (uses default if None)
            
        Returns:
            Optional[Any]: DataFrame or None if error
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return None
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Read data from Delta table
            df = self.spark_session.read \
                .format("delta") \
                .option("path", location) \
                .table(table_name)
            
            logger.info(f"Read data from Delta Lake table: {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading data from Delta Lake table {table_name}: {e}")
            return None
    
    def update_data(self, table_name: str, updates: Dict[str, Any], condition: str, location: Optional[str] = None) -> bool:
        """
        Update data in a Delta Lake table.
        
        Args:
            table_name: Name of the table
            updates: Dictionary of column updates
            condition: WHERE condition for updates
            location: Table location (uses default if None)
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Build UPDATE statement
            set_clause = ", ".join([f"{col} = '{val}'" for col, val in updates.items()])
            update_sql = f"""
                UPDATE {table_name}
                SET {set_clause}
                WHERE {condition}
            """
            
            # Execute update
            self.spark_session.sql(update_sql)
            
            logger.info(f"Updated data in Delta Lake table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating data in Delta Lake table {table_name}: {e}")
            return False
    
    def delete_data(self, table_name: str, condition: str, location: Optional[str] = None) -> bool:
        """
        Delete data from a Delta Lake table.
        
        Args:
            table_name: Name of the table
            condition: WHERE condition for deletion
            location: Table location (uses default if None)
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Build DELETE statement
            delete_sql = f"""
                DELETE FROM {table_name}
                WHERE {condition}
            """
            
            # Execute deletion
            self.spark_session.sql(delete_sql)
            
            logger.info(f"Deleted data from Delta Lake table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting data from Delta Lake table {table_name}: {e}")
            return False
    
    def merge_data(self, table_name: str, source_data: Any, merge_condition: str, location: Optional[str] = None) -> bool:
        """
        Merge data into a Delta Lake table.
        
        Args:
            table_name: Name of the table
            source_data: Source data to merge
            merge_condition: Merge condition
            location: Table location (uses default if None)
            
        Returns:
            bool: True if merge successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Create temporary view for source data
            source_data.createOrReplaceTempView("source_data")
            
            # Build MERGE statement
            merge_sql = f"""
                MERGE INTO {table_name} AS target
                USING source_data AS source
                ON {merge_condition}
                WHEN MATCHED THEN UPDATE SET *
                WHEN NOT MATCHED THEN INSERT *
            """
            
            # Execute merge
            self.spark_session.sql(merge_sql)
            
            logger.info(f"Merged data into Delta Lake table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging data into Delta Lake table {table_name}: {e}")
            return False
    
    def get_table_history(self, table_name: str, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get table history from Delta Lake.
        
        Args:
            table_name: Name of the table
            location: Table location (uses default if None)
            
        Returns:
            List[Dict[str, Any]]: Table history
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return []
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Get table history
            history_df = self.spark_session.sql(f"DESCRIBE HISTORY {table_name}")
            history = history_df.collect()
            
            history_list = []
            for row in history:
                history_list.append({
                    "version": row.version,
                    "timestamp": row.timestamp,
                    "user_id": row.userId,
                    "user_name": row.userName,
                    "operation": row.operation,
                    "operation_parameters": row.operationParameters,
                    "job": row.job,
                    "notebook": row.notebook,
                    "cluster_id": row.clusterId,
                    "read_version": row.readVersion,
                    "isolation_level": row.isolationLevel,
                    "is_blind_append": row.isBlindAppend
                })
            
            logger.info(f"Retrieved history for Delta Lake table: {table_name}")
            return history_list
            
        except Exception as e:
            logger.error(f"Error getting history for Delta Lake table {table_name}: {e}")
            return []
    
    def get_table_details(self, table_name: str, location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get table details from Delta Lake.
        
        Args:
            table_name: Name of the table
            location: Table location (uses default if None)
            
        Returns:
            Optional[Dict[str, Any]]: Table details or None if error
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return None
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Get table details
            details_df = self.spark_session.sql(f"DESCRIBE DETAIL {table_name}")
            details = details_df.collect()[0]
            
            table_details = {
                "name": details.name,
                "description": details.description,
                "location": details.location,
                "provider": details.provider,
                "owner": details.owner,
                "create_time": details.createTime,
                "last_access": details.lastAccess,
                "properties": details.properties,
                "partition_columns": details.partitionColumns,
                "num_files": details.numFiles,
                "size_in_bytes": details.sizeInBytes,
                "min_reader_version": details.minReaderVersion,
                "min_writer_version": details.minWriterVersion
            }
            
            logger.info(f"Retrieved details for Delta Lake table: {table_name}")
            return table_details
            
        except Exception as e:
            logger.error(f"Error getting details for Delta Lake table {table_name}: {e}")
            return None
    
    def optimize_table(self, table_name: str, location: Optional[str] = None) -> bool:
        """
        Optimize a Delta Lake table.
        
        Args:
            table_name: Name of the table
            location: Table location (uses default if None)
            
        Returns:
            bool: True if optimization successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Optimize table
            self.spark_session.sql(f"OPTIMIZE {table_name}")
            
            logger.info(f"Optimized Delta Lake table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing Delta Lake table {table_name}: {e}")
            return False
    
    def vacuum_table(self, table_name: str, retention_hours: int = 168, location: Optional[str] = None) -> bool:
        """
        Vacuum a Delta Lake table to remove old files.
        
        Args:
            table_name: Name of the table
            retention_hours: Retention period in hours
            location: Table location (uses default if None)
            
        Returns:
            bool: True if vacuum successful, False otherwise
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return False
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Vacuum table
            self.spark_session.sql(f"VACUUM {table_name} RETAIN {retention_hours} HOURS")
            
            logger.info(f"Vacuumed Delta Lake table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error vacuuming Delta Lake table {table_name}: {e}")
            return False
    
    def get_table_stats(self, table_name: str, location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get table statistics from Delta Lake.
        
        Args:
            table_name: Name of the table
            location: Table location (uses default if None)
            
        Returns:
            Optional[Dict[str, Any]]: Table statistics or None if error
        """
        if not self.spark_session:
            logger.error("Spark session not initialized")
            return None
        
        try:
            location = location or f"{self.delta_table_path}/{table_name}"
            
            # Get table statistics
            stats_df = self.spark_session.sql(f"ANALYZE TABLE {table_name} COMPUTE STATISTICS")
            
            # Get row count
            count_df = self.spark_session.sql(f"SELECT COUNT(*) as row_count FROM {table_name}")
            row_count = count_df.collect()[0].row_count
            
            # Get table details
            details = self.get_table_details(table_name, location)
            
            stats = {
                "table_name": table_name,
                "row_count": row_count,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Retrieved statistics for Delta Lake table: {table_name}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics for Delta Lake table {table_name}: {e}")
            return None
    
    def close_spark_session(self) -> None:
        """Close Spark session."""
        if self.spark_session:
            self.spark_session.stop()
            logger.info("Delta Lake Spark session closed")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "delta_table_path": self.delta_table_path,
            "history_retention_days": self.history_retention_days,
            "optimize_write": self.optimize_write,
            "spark_session_initialized": self.spark_session is not None,
            "ingestion_timestamp": datetime.now().isoformat()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_spark_session()
