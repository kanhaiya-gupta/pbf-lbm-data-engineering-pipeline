"""
ETL Orchestrator

This module provides ETL orchestration for PBF-LB/M data processing.
It coordinates the extract, transform, and load operations for different data types.
"""

from typing import Dict, Any, Optional, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import current_timestamp, lit
import logging
from datetime import datetime

from .extract import (
    extract_pbf_process_data,
    extract_powder_bed_data,
    extract_from_ct_scanner,
    extract_from_ispm_system
)
from .transform import (
    transform_pbf_process_data,
    transform_ispm_monitoring_data,
    transform_ct_scan_data,
    transform_powder_bed_data,
    apply_business_rules
)
from .load import (
    load_pbf_process_data,
    load_ispm_monitoring_data,
    load_to_postgresql,
    load_to_s3,
    load_to_snowflake,
    load_to_delta_lake
)

logger = logging.getLogger(__name__)


class ETLOrchestrator:
    """ETL Orchestrator for PBF-LB/M data processing"""
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        """
        Initialize ETL Orchestrator
        
        Args:
            spark: Spark session
            config: ETL configuration
        """
        self.spark = spark
        self.config = config
        self.job_history: List[Dict[str, Any]] = []
        
    def run_full_etl(
        self,
        data_type: str,
        source_config: Dict[str, Any],
        destination_config: Dict[str, Any],
        transformation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run full ETL pipeline for specified data type
        
        Args:
            data_type: Type of data to process (pbf_process, ispm_monitoring, ct_scan, powder_bed)
            source_config: Source configuration
            destination_config: Destination configuration
            transformation_config: Optional transformation configuration
            
        Returns:
            Dictionary containing job execution results
        """
        job_id = f"{data_type}_etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting full ETL job: {job_id}")
            
            # Extract data
            logger.info(f"Extracting {data_type} data")
            df_extracted = self._extract_data(data_type, source_config)
            
            # Transform data
            logger.info(f"Transforming {data_type} data")
            df_transformed = self._transform_data(data_type, df_extracted, transformation_config)
            
            # Load data
            logger.info(f"Loading {data_type} data")
            self._load_data(data_type, df_transformed, destination_config)
            
            # Record job success
            job_result = {
                "job_id": job_id,
                "data_type": data_type,
                "status": "success",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "records_processed": df_transformed.count(),
                "source_config": source_config,
                "destination_config": destination_config
            }
            
            self.job_history.append(job_result)
            logger.info(f"Successfully completed ETL job: {job_id}")
            
            return job_result
            
        except Exception as e:
            # Record job failure
            job_result = {
                "job_id": job_id,
                "data_type": data_type,
                "status": "failed",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "error": str(e),
                "source_config": source_config,
                "destination_config": destination_config
            }
            
            self.job_history.append(job_result)
            logger.error(f"ETL job failed: {job_id}, Error: {str(e)}")
            raise
    
    def run_incremental_etl(
        self,
        data_type: str,
        source_config: Dict[str, Any],
        destination_config: Dict[str, Any],
        incremental_config: Dict[str, Any],
        transformation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run incremental ETL pipeline for specified data type
        
        Args:
            data_type: Type of data to process
            source_config: Source configuration
            destination_config: Destination configuration
            incremental_config: Incremental processing configuration
            transformation_config: Optional transformation configuration
            
        Returns:
            Dictionary containing job execution results
        """
        job_id = f"{data_type}_incremental_etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting incremental ETL job: {job_id}")
            
            # Get last processed timestamp
            last_processed_timestamp = incremental_config.get("last_processed_timestamp")
            
            # Extract incremental data
            logger.info(f"Extracting incremental {data_type} data since {last_processed_timestamp}")
            df_extracted = self._extract_incremental_data(
                data_type, source_config, last_processed_timestamp
            )
            
            if df_extracted.count() == 0:
                logger.info(f"No new {data_type} data to process")
                return {
                    "job_id": job_id,
                    "data_type": data_type,
                    "status": "success",
                    "records_processed": 0,
                    "message": "No new data to process"
                }
            
            # Transform data
            logger.info(f"Transforming incremental {data_type} data")
            df_transformed = self._transform_data(data_type, df_extracted, transformation_config)
            
            # Load data
            logger.info(f"Loading incremental {data_type} data")
            self._load_data(data_type, df_transformed, destination_config)
            
            # Update last processed timestamp
            new_timestamp = df_transformed.select("timestamp").orderBy("timestamp").collect()[-1]["timestamp"]
            incremental_config["last_processed_timestamp"] = new_timestamp.isoformat()
            
            # Record job success
            job_result = {
                "job_id": job_id,
                "data_type": data_type,
                "status": "success",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "records_processed": df_transformed.count(),
                "last_processed_timestamp": new_timestamp.isoformat(),
                "source_config": source_config,
                "destination_config": destination_config
            }
            
            self.job_history.append(job_result)
            logger.info(f"Successfully completed incremental ETL job: {job_id}")
            
            return job_result
            
        except Exception as e:
            # Record job failure
            job_result = {
                "job_id": job_id,
                "data_type": data_type,
                "status": "failed",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "error": str(e),
                "source_config": source_config,
                "destination_config": destination_config
            }
            
            self.job_history.append(job_result)
            logger.error(f"Incremental ETL job failed: {job_id}, Error: {str(e)}")
            raise
    
    def run_streaming_etl(
        self,
        data_type: str,
        source_config: Dict[str, Any],
        destination_config: Dict[str, Any],
        streaming_config: Dict[str, Any],
        transformation_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Run streaming ETL pipeline for specified data type
        
        Args:
            data_type: Type of data to process
            source_config: Source configuration
            destination_config: Destination configuration
            streaming_config: Streaming configuration
            transformation_config: Optional transformation configuration
        """
        try:
            logger.info(f"Starting streaming ETL for {data_type} data")
            
            # Extract streaming data
            df_stream = self._extract_streaming_data(data_type, source_config)
            
            # Transform streaming data
            df_transformed_stream = self._transform_streaming_data(
                data_type, df_stream, transformation_config
            )
            
            # Load streaming data
            self._load_streaming_data(data_type, df_transformed_stream, destination_config)
            
            logger.info(f"Successfully started streaming ETL for {data_type} data")
            
        except Exception as e:
            logger.error(f"Error starting streaming ETL for {data_type} data: {str(e)}")
            raise
    
    def _extract_data(self, data_type: str, source_config: Dict[str, Any]) -> DataFrame:
        """Extract data based on data type"""
        if data_type == "pbf_process":
            return extract_pbf_process_data(self.spark, source_config)
        elif data_type == "ispm_monitoring":
            return extract_from_ispm_system(self.spark, source_config)
        elif data_type == "ct_scan":
            return extract_from_ct_scanner(self.spark, source_config)
        elif data_type == "powder_bed":
            return extract_powder_bed_data(self.spark, source_config)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _transform_data(
        self, 
        data_type: str, 
        df: DataFrame, 
        transformation_config: Optional[Dict[str, Any]] = None
    ) -> DataFrame:
        """Transform data based on data type"""
        if data_type == "pbf_process":
            df_transformed = transform_pbf_process_data(df, transformation_config)
        elif data_type == "ispm_monitoring":
            df_transformed = transform_ispm_monitoring_data(df, transformation_config)
        elif data_type == "ct_scan":
            df_transformed = transform_ct_scan_data(df, transformation_config)
        elif data_type == "powder_bed":
            df_transformed = transform_powder_bed_data(df, transformation_config)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Apply business rules
        business_rules = transformation_config.get("business_rules", {}) if transformation_config else {}
        df_with_rules = apply_business_rules(df_transformed, business_rules, data_type)
        
        # Add ETL metadata
        return df_with_rules.withColumn(
            "etl_timestamp", current_timestamp()
        ).withColumn(
            "etl_job_id", lit(f"{data_type}_etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
    
    def _load_data(
        self, 
        data_type: str, 
        df: DataFrame, 
        destination_config: Dict[str, Any]
    ) -> None:
        """Load data based on data type"""
        if data_type == "pbf_process":
            load_pbf_process_data(df, destination_config)
        elif data_type == "ispm_monitoring":
            load_ispm_monitoring_data(df, destination_config)
        elif data_type == "ct_scan":
            # CT scan data typically goes to MongoDB or S3
            destination_type = destination_config.get("type", "s3")
            if destination_type == "s3":
                load_to_s3(df, destination_config.get("s3_path", ""))
            elif destination_type == "mongodb":
                from .load import load_to_mongodb
                load_to_mongodb(df, destination_config.get("mongodb_config", {}), 
                              destination_config.get("collection_name", "ct_scan_data"))
        elif data_type == "powder_bed":
            # Powder bed data typically goes to Redis or PostgreSQL
            destination_type = destination_config.get("type", "postgresql")
            if destination_type == "postgresql":
                load_to_postgresql(df, destination_config.get("connection_string", ""), 
                                 destination_config.get("table_name", "powder_bed_data"))
            elif destination_type == "redis":
                from .load import load_to_redis
                load_to_redis(df, destination_config.get("redis_config", {}), 
                            destination_config.get("key_prefix", "powder_bed"))
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _extract_incremental_data(
        self, 
        data_type: str, 
        source_config: Dict[str, Any], 
        last_timestamp: Optional[str]
    ) -> DataFrame:
        """Extract incremental data based on timestamp"""
        # Add timestamp filter to source config
        if last_timestamp:
            source_config["filter_conditions"] = source_config.get("filter_conditions", {})
            source_config["filter_conditions"]["timestamp"] = f"> {last_timestamp}"
        
        return self._extract_data(data_type, source_config)
    
    def _extract_streaming_data(self, data_type: str, source_config: Dict[str, Any]) -> DataFrame:
        """Extract streaming data"""
        # For streaming, we typically use Kafka
        from .extract import extract_from_kafka
        
        return extract_from_kafka(
            spark=self.spark,
            bootstrap_servers=source_config.get("bootstrap_servers", "localhost:9092"),
            topic=source_config.get("topic", f"{data_type}_data")
        )
    
    def _transform_streaming_data(
        self, 
        data_type: str, 
        df_stream: DataFrame, 
        transformation_config: Optional[Dict[str, Any]] = None
    ) -> DataFrame:
        """Transform streaming data"""
        # For streaming, we apply the same transformations but return a streaming DataFrame
        # This is a simplified version - in practice, you'd use structured streaming transformations
        
        # Parse the Kafka message value
        from pyspark.sql.functions import from_json, col
        from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
        
        # Define schema for parsing JSON messages
        message_schema = StructType([
            StructField("id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("value", DoubleType(), True)
        ])
        
        df_parsed = df_stream.select(
            col("key").cast("string"),
            from_json(col("value").cast("string"), message_schema).alias("data")
        ).select("key", "data.*")
        
        return self._transform_data(data_type, df_parsed, transformation_config)
    
    def _load_streaming_data(
        self, 
        data_type: str, 
        df_stream: DataFrame, 
        destination_config: Dict[str, Any]
    ) -> None:
        """Load streaming data"""
        # For streaming, we use writeStream instead of write
        destination_type = destination_config.get("type", "console")
        
        if destination_type == "console":
            df_stream.writeStream.outputMode("append").format("console").start()
        elif destination_type == "kafka":
            df_stream.writeStream.format("kafka").option(
                "kafka.bootstrap.servers", destination_config.get("bootstrap_servers", "localhost:9092")
            ).option("topic", destination_config.get("topic", f"{data_type}_processed")).start()
        elif destination_type == "postgresql":
            df_stream.writeStream.foreachBatch(
                lambda batch_df, batch_id: load_to_postgresql(
                    batch_df, 
                    destination_config.get("connection_string", ""),
                    destination_config.get("table_name", f"{data_type}_data")
                )
            ).start()
        else:
            raise ValueError(f"Unsupported streaming destination type: {destination_type}")
    
    def get_job_history(self) -> List[Dict[str, Any]]:
        """Get ETL job history"""
        return self.job_history.copy()
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific ETL job"""
        for job in self.job_history:
            if job["job_id"] == job_id:
                return job
        return None
    
    def get_data_type_statistics(self, data_type: str) -> Dict[str, Any]:
        """Get statistics for a specific data type"""
        data_type_jobs = [job for job in self.job_history if job["data_type"] == data_type]
        
        if not data_type_jobs:
            return {"data_type": data_type, "total_jobs": 0}
        
        successful_jobs = [job for job in data_type_jobs if job["status"] == "success"]
        failed_jobs = [job for job in data_type_jobs if job["status"] == "failed"]
        
        total_records = sum(job.get("records_processed", 0) for job in successful_jobs)
        
        return {
            "data_type": data_type,
            "total_jobs": len(data_type_jobs),
            "successful_jobs": len(successful_jobs),
            "failed_jobs": len(failed_jobs),
            "success_rate": len(successful_jobs) / len(data_type_jobs) if data_type_jobs else 0,
            "total_records_processed": total_records,
            "last_job": data_type_jobs[-1] if data_type_jobs else None
        }
