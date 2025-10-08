"""
Delta Processor

This module provides Delta Lake incremental processing capabilities for the PBF-LB/M data pipeline.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import json
from delta import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, min as spark_min

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeltaProcessor:
    """
    Delta Lake incremental processor for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.delta_config = self._load_delta_config()
        self.spark = self._initialize_spark()
        self.delta_tables = {}
    
    def _load_delta_config(self) -> Dict[str, Any]:
        """Load Delta Lake configuration."""
        try:
            return self.config.get('delta_lake', {
                'base_path': 's3a://pbf-lbm-data-lake/delta',
                'checkpoint_interval': 10,
                'compaction_interval': 24,  # hours
                'vacuum_retention_hours': 168,  # 7 days
                'tables': {
                    'pbf_process_data': {
                        'path': 'pbf_process',
                        'partition_columns': ['year', 'month', 'day'],
                        'timestamp_column': 'timestamp',
                        'enabled': True
                    },
                    'ispm_monitoring_data': {
                        'path': 'ispm_monitoring',
                        'partition_columns': ['year', 'month', 'day'],
                        'timestamp_column': 'timestamp',
                        'enabled': True
                    },
                    'ct_scan_data': {
                        'path': 'ct_scan',
                        'partition_columns': ['year', 'month', 'day'],
                        'timestamp_column': 'created_at',
                        'enabled': True
                    },
                    'powder_bed_data': {
                        'path': 'powder_bed',
                        'partition_columns': ['year', 'month', 'day'],
                        'timestamp_column': 'timestamp',
                        'enabled': True
                    }
                }
            })
        except Exception as e:
            logger.error(f"Error loading Delta Lake configuration: {e}")
            return {}
    
    def _initialize_spark(self) -> SparkSession:
        """Initialize Spark session for Delta Lake."""
        try:
            spark = SparkSession.builder \
                .appName("PBF-LB/M Delta Processor") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .getOrCreate()
            
            logger.info("Spark session initialized for Delta Lake")
            return spark
            
        except Exception as e:
            logger.error(f"Error initializing Spark session: {e}")
            raise
    
    def process_incremental_data(self, table_name: str, new_data: pd.DataFrame, 
                               operation: str = 'append') -> Dict[str, Any]:
        """Process incremental data for a Delta table."""
        try:
            if not self._is_table_enabled(table_name):
                logger.warning(f"Delta processing disabled for table: {table_name}")
                return {'status': 'disabled', 'processed_count': 0}
            
            table_config = self.delta_config['tables'].get(table_name, {})
            table_path = self._get_table_path(table_name)
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(new_data)
            
            # Add partition columns if needed
            spark_df = self._add_partition_columns(spark_df, table_config)
            
            # Perform operation
            if operation == 'append':
                result = self._append_data(spark_df, table_path, table_name)
            elif operation == 'merge':
                result = self._merge_data(spark_df, table_path, table_name)
            elif operation == 'upsert':
                result = self._upsert_data(spark_df, table_path, table_name)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            # Update table metadata
            self._update_table_metadata(table_name, result)
            
            logger.info(f"Delta incremental processing completed for {table_name}: {result['processed_count']} records")
            return result
            
        except Exception as e:
            logger.error(f"Error processing incremental data for {table_name}: {e}")
            return {'status': 'error', 'error': str(e), 'processed_count': 0}
    
    def _append_data(self, spark_df, table_path: str, table_name: str) -> Dict[str, Any]:
        """Append data to Delta table."""
        try:
            # Write data to Delta table
            spark_df.write \
                .format("delta") \
                .mode("append") \
                .option("mergeSchema", "true") \
                .save(table_path)
            
            # Get record count
            record_count = spark_df.count()
            
            return {
                'status': 'success',
                'operation': 'append',
                'processed_count': record_count,
                'table_path': table_path,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error appending data to Delta table: {e}")
            raise
    
    def _merge_data(self, spark_df, table_path: str, table_name: str) -> Dict[str, Any]:
        """Merge data into Delta table."""
        try:
            table_config = self.delta_config['tables'].get(table_name, {})
            timestamp_column = table_config.get('timestamp_column', 'timestamp')
            
            # Create temporary view
            spark_df.createOrReplaceTempView("new_data")
            
            # Get Delta table
            delta_table = DeltaTable.forPath(self.spark, table_path)
            
            # Perform merge operation
            merge_result = delta_table.alias("target") \
                .merge(spark_df.alias("source"), 
                      f"target.{timestamp_column} = source.{timestamp_column}") \
                .whenMatchedUpdateAll() \
                .whenNotMatchedInsertAll() \
                .execute()
            
            # Get record count
            record_count = spark_df.count()
            
            return {
                'status': 'success',
                'operation': 'merge',
                'processed_count': record_count,
                'table_path': table_path,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error merging data to Delta table: {e}")
            raise
    
    def _upsert_data(self, spark_df, table_path: str, table_name: str) -> Dict[str, Any]:
        """Upsert data into Delta table."""
        try:
            table_config = self.delta_config['tables'].get(table_name, {})
            timestamp_column = table_config.get('timestamp_column', 'timestamp')
            
            # Get Delta table
            delta_table = DeltaTable.forPath(self.spark, table_path)
            
            # Perform upsert operation
            upsert_result = delta_table.alias("target") \
                .merge(spark_df.alias("source"), 
                      f"target.{timestamp_column} = source.{timestamp_column}") \
                .whenMatchedUpdateAll() \
                .whenNotMatchedInsertAll() \
                .execute()
            
            # Get record count
            record_count = spark_df.count()
            
            return {
                'status': 'success',
                'operation': 'upsert',
                'processed_count': record_count,
                'table_path': table_path,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error upserting data to Delta table: {e}")
            raise
    
    def _add_partition_columns(self, spark_df, table_config: Dict[str, Any]):
        """Add partition columns to DataFrame."""
        try:
            partition_columns = table_config.get('partition_columns', [])
            timestamp_column = table_config.get('timestamp_column', 'timestamp')
            
            if partition_columns and timestamp_column in [col.name for col in spark_df.schema.fields]:
                # Add year, month, day columns for partitioning
                spark_df = spark_df.withColumn("year", col(timestamp_column).substr(1, 4)) \
                                 .withColumn("month", col(timestamp_column).substr(6, 2)) \
                                 .withColumn("day", col(timestamp_column).substr(9, 2))
            
            return spark_df
            
        except Exception as e:
            logger.error(f"Error adding partition columns: {e}")
            return spark_df
    
    def _get_table_path(self, table_name: str) -> str:
        """Get Delta table path."""
        try:
            base_path = self.delta_config.get('base_path', 's3a://pbf-lbm-data-lake/delta')
            table_config = self.delta_config['tables'].get(table_name, {})
            table_path = table_config.get('path', table_name)
            
            return f"{base_path}/{table_path}"
            
        except Exception as e:
            logger.error(f"Error getting table path: {e}")
            return f"{base_path}/{table_name}"
    
    def _is_table_enabled(self, table_name: str) -> bool:
        """Check if Delta processing is enabled for a table."""
        try:
            table_config = self.delta_config['tables'].get(table_name, {})
            return table_config.get('enabled', False)
        except Exception as e:
            logger.error(f"Error checking table enablement: {e}")
            return False
    
    def _update_table_metadata(self, table_name: str, result: Dict[str, Any]) -> None:
        """Update table metadata."""
        try:
            if table_name not in self.delta_tables:
                self.delta_tables[table_name] = {}
            
            self.delta_tables[table_name].update({
                'last_processed_at': result.get('processed_at'),
                'last_processed_count': result.get('processed_count'),
                'last_operation': result.get('operation')
            })
            
        except Exception as e:
            logger.error(f"Error updating table metadata: {e}")
    
    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get Delta table statistics."""
        try:
            if not self._is_table_enabled(table_name):
                return {'status': 'disabled'}
            
            table_path = self._get_table_path(table_name)
            
            # Get Delta table
            delta_table = DeltaTable.forPath(self.spark, table_path)
            
            # Get table history
            history = delta_table.history().toPandas()
            
            # Get table details
            table_details = delta_table.detail().toPandas()
            
            # Get record count
            record_count = delta_table.toDF().count()
            
            # Get partition information
            partitions = delta_table.toDF().select("year", "month", "day").distinct().collect()
            
            stats = {
                'status': 'enabled',
                'table_path': table_path,
                'record_count': record_count,
                'partition_count': len(partitions),
                'partitions': [{'year': p.year, 'month': p.month, 'day': p.day} for p in partitions],
                'history_count': len(history),
                'last_operation': history.iloc[-1]['operation'] if len(history) > 0 else None,
                'last_operation_timestamp': history.iloc[-1]['timestamp'] if len(history) > 0 else None,
                'metadata': self.delta_tables.get(table_name, {})
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting table statistics: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def optimize_table(self, table_name: str) -> Dict[str, Any]:
        """Optimize Delta table."""
        try:
            if not self._is_table_enabled(table_name):
                return {'status': 'disabled'}
            
            table_path = self._get_table_path(table_name)
            
            # Get Delta table
            delta_table = DeltaTable.forPath(self.spark, table_path)
            
            # Optimize table
            delta_table.optimize().executeCompaction()
            
            # Get optimization statistics
            stats = self.get_table_statistics(table_name)
            
            result = {
                'status': 'success',
                'table_name': table_name,
                'optimized_at': datetime.now().isoformat(),
                'table_statistics': stats
            }
            
            logger.info(f"Delta table optimized: {table_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing Delta table: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def vacuum_table(self, table_name: str, retention_hours: Optional[int] = None) -> Dict[str, Any]:
        """Vacuum Delta table to remove old files."""
        try:
            if not self._is_table_enabled(table_name):
                return {'status': 'disabled'}
            
            if retention_hours is None:
                retention_hours = self.delta_config.get('vacuum_retention_hours', 168)
            
            table_path = self._get_table_path(table_name)
            
            # Get Delta table
            delta_table = DeltaTable.forPath(self.spark, table_path)
            
            # Vacuum table
            delta_table.vacuum(retentionHours=retention_hours)
            
            result = {
                'status': 'success',
                'table_name': table_name,
                'vacuumed_at': datetime.now().isoformat(),
                'retention_hours': retention_hours
            }
            
            logger.info(f"Delta table vacuumed: {table_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error vacuuming Delta table: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_all_table_statistics(self) -> Dict[str, Any]:
        """Get statistics for all Delta tables."""
        try:
            stats = {}
            
            for table_name in self.delta_config.get('tables', {}):
                if self._is_table_enabled(table_name):
                    stats[table_name] = self.get_table_statistics(table_name)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting all table statistics: {e}")
            return {}
    
    def close(self):
        """Close Spark session."""
        try:
            if self.spark:
                self.spark.stop()
                logger.info("Spark session closed")
        except Exception as e:
            logger.error(f"Error closing Spark session: {e}")
