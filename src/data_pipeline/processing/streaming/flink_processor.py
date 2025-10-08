"""
Flink Processor

This module provides Apache Flink processing capabilities for the PBF-LB/M data pipeline.
"""

import json
from typing import Dict, List, Optional, Any
import logging
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlinkProcessor:
    """
    Apache Flink processor for real-time data processing.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.env = None
        self.table_env = None
        self._initialize_flink()
    
    def _initialize_flink(self):
        """Initialize Flink environment."""
        try:
            # Create StreamExecutionEnvironment
            self.env = StreamExecutionEnvironment.get_execution_environment()
            
            # Create StreamTableEnvironment
            self.table_env = StreamTableEnvironment.create(self.env)
            
            logger.info("Flink processor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Flink processor: {e}")
    
    def process_pbf_process_stream(self, input_topic: str = 'pbf_process_stream', 
                                 output_topic: str = 'pbf_process_processed') -> None:
        """Process PBF process data stream using Flink."""
        try:
            # Define source table
            source_ddl = f"""
                CREATE TABLE pbf_process_source (
                    process_id STRING,
                    timestamp TIMESTAMP(3),
                    temperature DOUBLE,
                    pressure DOUBLE,
                    laser_power DOUBLE,
                    scan_speed DOUBLE,
                    layer_height DOUBLE,
                    WATERMARK FOR timestamp AS timestamp - INTERVAL '5' SECOND
                ) WITH (
                    'connector' = 'kafka',
                    'topic' = '{input_topic}',
                    'properties.bootstrap.servers' = 'localhost:9092',
                    'properties.group.id' = 'flink-pbf-processor',
                    'format' = 'json'
                )
            """
            
            # Define sink table
            sink_ddl = f"""
                CREATE TABLE pbf_process_sink (
                    process_id STRING,
                    timestamp TIMESTAMP(3),
                    temperature DOUBLE,
                    pressure DOUBLE,
                    laser_power DOUBLE,
                    scan_speed DOUBLE,
                    layer_height DOUBLE,
                    processed_at TIMESTAMP(3),
                    processing_status STRING
                ) WITH (
                    'connector' = 'kafka',
                    'topic' = '{output_topic}',
                    'properties.bootstrap.servers' = 'localhost:9092',
                    'format' = 'json'
                )
            """
            
            # Create tables
            self.table_env.execute_sql(source_ddl)
            self.table_env.execute_sql(sink_ddl)
            
            # Process data
            result_table = self.table_env.sql_query("""
                SELECT 
                    process_id,
                    timestamp,
                    temperature,
                    pressure,
                    laser_power,
                    scan_speed,
                    layer_height,
                    CURRENT_TIMESTAMP as processed_at,
                    'completed' as processing_status
                FROM pbf_process_source
            """)
            
            # Insert into sink
            result_table.execute_insert('pbf_process_sink')
            
            logger.info("PBF process stream processing started")
            
        except Exception as e:
            logger.error(f"Error processing PBF process stream: {e}")
    
    def process_ispm_monitoring_stream(self, input_topic: str = 'ispm_monitoring_stream',
                                     output_topic: str = 'ispm_monitoring_processed') -> None:
        """Process ISPM monitoring data stream using Flink."""
        try:
            # Define source table
            source_ddl = f"""
                CREATE TABLE ispm_monitoring_source (
                    monitoring_id STRING,
                    timestamp TIMESTAMP(3),
                    sensor_id STRING,
                    sensor_type STRING,
                    value DOUBLE,
                    unit STRING,
                    WATERMARK FOR timestamp AS timestamp - INTERVAL '5' SECOND
                ) WITH (
                    'connector' = 'kafka',
                    'topic' = '{input_topic}',
                    'properties.bootstrap.servers' = 'localhost:9092',
                    'properties.group.id' = 'flink-ispm-processor',
                    'format' = 'json'
                )
            """
            
            # Define sink table
            sink_ddl = f"""
                CREATE TABLE ispm_monitoring_sink (
                    monitoring_id STRING,
                    timestamp TIMESTAMP(3),
                    sensor_id STRING,
                    sensor_type STRING,
                    value DOUBLE,
                    unit STRING,
                    processed_at TIMESTAMP(3),
                    processing_status STRING
                ) WITH (
                    'connector' = 'kafka',
                    'topic' = '{output_topic}',
                    'properties.bootstrap.servers' = 'localhost:9092',
                    'format' = 'json'
                )
            """
            
            # Create tables
            self.table_env.execute_sql(source_ddl)
            self.table_env.execute_sql(sink_ddl)
            
            # Process data
            result_table = self.table_env.sql_query("""
                SELECT 
                    monitoring_id,
                    timestamp,
                    sensor_id,
                    sensor_type,
                    value,
                    unit,
                    CURRENT_TIMESTAMP as processed_at,
                    'completed' as processing_status
                FROM ispm_monitoring_source
            """)
            
            # Insert into sink
            result_table.execute_insert('ispm_monitoring_sink')
            
            logger.info("ISPM monitoring stream processing started")
            
        except Exception as e:
            logger.error(f"Error processing ISPM monitoring stream: {e}")
    
    def start_processing(self):
        """Start Flink processing."""
        try:
            if self.env:
                self.env.execute("PBF-LB/M Data Pipeline Processing")
                logger.info("Flink processing started")
                
        except Exception as e:
            logger.error(f"Error starting Flink processing: {e}")
    
    def stop_processing(self):
        """Stop Flink processing."""
        try:
            if self.env:
                self.env.cancel()
                logger.info("Flink processing stopped")
                
        except Exception as e:
            logger.error(f"Error stopping Flink processing: {e}")
