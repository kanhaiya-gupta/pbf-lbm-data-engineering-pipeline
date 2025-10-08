# Technical Implementation: Kafka, Airflow & ETL Spark Deep Dive

This document provides detailed technical implementation guidance for integrating Apache Kafka, Apache Airflow, and ETL Spark in the PBF-LB/M data pipeline, focusing on production-ready configurations, troubleshooting, and advanced features.

## 🔧 **Kafka Technical Implementation**

### **1. Kafka Cluster Configuration**

#### **Production Kafka Setup**
```yaml
# kafka-cluster-config.yaml
cluster:
  name: "pbf-lbm-cluster"
  version: "3.5.0"
  
brokers:
  count: 3
  configuration:
    # Core settings
    broker.id: "${BROKER_ID}"
    listeners: "PLAINTEXT://0.0.0.0:9092"
    advertised.listeners: "PLAINTEXT://${HOSTNAME}:9092"
    
    # Log settings
    log.dirs: "/var/lib/kafka-logs"
    log.retention.hours: 168  # 7 days
    log.retention.bytes: 1073741824  # 1GB
    log.segment.bytes: 1073741824  # 1GB
    log.cleanup.policy: "delete"
    
    # Replication settings
    default.replication.factor: 3
    min.insync.replicas: 2
    unclean.leader.election.enable: false
    
    # Performance settings
    num.network.threads: 8
    num.io.threads: 16
    socket.send.buffer.bytes: 102400
    socket.receive.buffer.bytes: 102400
    socket.request.max.bytes: 104857600
    
    # Topic settings
    num.partitions: 12
    default.replication.factor: 3
    min.insync.replicas: 2
    
    # Compression
    compression.type: "snappy"
    
    # Security (if enabled)
    security.inter.broker.protocol: "PLAINTEXT"
    sasl.enabled.mechanisms: "PLAIN"
    sasl.mechanism.inter.broker.protocol: "PLAIN"
```

#### **Topic Configuration for PBF-LB/M**
```bash
# Create topics with specific configurations
kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic ispm-monitoring \
  --partitions 12 \
  --replication-factor 3 \
  --config retention.ms=604800000 \
  --config compression.type=snappy \
  --config cleanup.policy=delete

kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic powder-bed-images \
  --partitions 8 \
  --replication-factor 3 \
  --config retention.ms=2592000000 \
  --config compression.type=lz4 \
  --config cleanup.policy=delete

kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic process-parameters \
  --partitions 6 \
  --replication-factor 3 \
  --config retention.ms=1209600000 \
  --config compression.type=snappy \
  --config cleanup.policy=delete
```

### **2. Kafka Producer Implementation**

#### **High-Performance Producer**
```python
# src/data_pipeline/ingestion/streaming/kafka_producer.py
import json
import logging
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
import time
from dataclasses import asdict

class HighPerformanceKafkaProducer:
    def __init__(self, bootstrap_servers: str, **kwargs):
        self.logger = logging.getLogger(__name__)
        
        # Producer configuration for high throughput
        self.config = {
            'bootstrap_servers': bootstrap_servers,
            'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None,
            'acks': 'all',  # Wait for all replicas
            'retries': 3,
            'retry_backoff_ms': 100,
            'batch_size': 16384,  # 16KB batches
            'linger_ms': 5,  # Wait 5ms for batch
            'compression_type': 'snappy',
            'buffer_memory': 33554432,  # 32MB buffer
            'max_request_size': 1048576,  # 1MB max request
            'request_timeout_ms': 30000,
            'delivery_timeout_ms': 120000,
            'enable_idempotence': True,  # Exactly-once semantics
            'max_in_flight_requests_per_connection': 5,
            **kwargs
        }
        
        self.producer = KafkaProducer(**self.config)
        self.metrics = {
            'messages_sent': 0,
            'messages_failed': 0,
            'bytes_sent': 0,
            'last_send_time': None
        }
    
    def send_message(self, topic: str, message: Dict[str, Any], 
                    key: Optional[str] = None) -> bool:
        """Send a single message to Kafka topic"""
        try:
            # Convert dataclass to dict if needed
            if hasattr(message, '__dataclass_fields__'):
                message = asdict(message)
            
            # Add metadata
            message['_metadata'] = {
                'timestamp': time.time(),
                'producer_id': id(self),
                'message_id': f"{topic}_{self.metrics['messages_sent']}"
            }
            
            # Send message
            future = self.producer.send(topic, value=message, key=key)
            
            # Wait for confirmation (optional for async)
            record_metadata = future.get(timeout=10)
            
            # Update metrics
            self.metrics['messages_sent'] += 1
            self.metrics['bytes_sent'] += len(json.dumps(message))
            self.metrics['last_send_time'] = time.time()
            
            self.logger.debug(f"Message sent to {topic} partition {record_metadata.partition}")
            return True
            
        except KafkaError as e:
            self.logger.error(f"Failed to send message to {topic}: {e}")
            self.metrics['messages_failed'] += 1
            return False
    
    def send_batch(self, topic: str, messages: list, 
                  key_func: Optional[callable] = None) -> Dict[str, int]:
        """Send a batch of messages efficiently"""
        results = {'sent': 0, 'failed': 0}
        
        for message in messages:
            key = key_func(message) if key_func else None
            if self.send_message(topic, message, key):
                results['sent'] += 1
            else:
                results['failed'] += 1
        
        return results
    
    def flush(self):
        """Flush all pending messages"""
        self.producer.flush()
    
    def close(self):
        """Close the producer"""
        self.producer.close()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics"""
        return {
            **self.metrics,
            'producer_metrics': self.producer.metrics()
        }
```

#### **ISPM Data Producer**
```python
# src/data_pipeline/ingestion/streaming/ispm_producer.py
from typing import Dict, Any
import asyncio
import time
from .kafka_producer import HighPerformanceKafkaProducer

class ISPMDataProducer:
    def __init__(self, kafka_servers: str):
        self.producer = HighPerformanceKafkaProducer(kafka_servers)
        self.topic = "ispm-monitoring"
        self.running = False
    
    async def start_producing(self, sensor_config: Dict[str, Any]):
        """Start producing ISPM sensor data"""
        self.running = True
        self.logger.info("Starting ISPM data production")
        
        while self.running:
            try:
                # Simulate sensor data collection
                sensor_data = await self.collect_sensor_data(sensor_config)
                
                # Send to Kafka
                success = self.producer.send_message(
                    self.topic, 
                    sensor_data,
                    key=sensor_data.get('sensor_id')
                )
                
                if not success:
                    self.logger.warning("Failed to send ISPM data")
                
                # Wait for next collection cycle
                await asyncio.sleep(sensor_config.get('collection_interval', 1.0))
                
            except Exception as e:
                self.logger.error(f"Error in ISPM data production: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def collect_sensor_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data from ISPM sensors"""
        # This would interface with actual ISPM hardware
        return {
            'sensor_id': config['sensor_id'],
            'timestamp': time.time(),
            'temperature': self.simulate_temperature(),
            'pressure': self.simulate_pressure(),
            'laser_power': self.simulate_laser_power(),
            'scan_speed': self.simulate_scan_speed(),
            'build_id': config.get('build_id'),
            'layer': config.get('current_layer', 0)
        }
    
    def simulate_temperature(self) -> float:
        """Simulate temperature reading"""
        import random
        return 1600 + random.uniform(-50, 50)
    
    def simulate_pressure(self) -> float:
        """Simulate pressure reading"""
        import random
        return 0.95 + random.uniform(-0.05, 0.05)
    
    def simulate_laser_power(self) -> float:
        """Simulate laser power reading"""
        import random
        return 200 + random.uniform(-10, 10)
    
    def simulate_scan_speed(self) -> float:
        """Simulate scan speed reading"""
        import random
        return 1.2 + random.uniform(-0.1, 0.1)
    
    def stop_producing(self):
        """Stop producing data"""
        self.running = False
        self.producer.close()
```

### **3. Kafka Consumer Implementation**

#### **Reliable Consumer with Error Handling**
```python
# src/data_pipeline/ingestion/streaming/kafka_consumer.py
import json
import logging
from typing import Dict, Any, Callable, Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError, CommitFailedError
import time
from dataclasses import dataclass

@dataclass
class ConsumerConfig:
    bootstrap_servers: str
    group_id: str
    auto_offset_reset: str = 'latest'
    enable_auto_commit: bool = False
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000
    max_poll_interval_ms: int = 300000

class ReliableKafkaConsumer:
    def __init__(self, config: ConsumerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Consumer configuration
        self.consumer_config = {
            'bootstrap_servers': config.bootstrap_servers,
            'group_id': config.group_id,
            'auto_offset_reset': config.auto_offset_reset,
            'enable_auto_commit': config.enable_auto_commit,
            'max_poll_records': config.max_poll_records,
            'session_timeout_ms': config.session_timeout_ms,
            'heartbeat_interval_ms': config.heartbeat_interval_ms,
            'max_poll_interval_ms': config.max_poll_interval_ms,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'key_deserializer': lambda m: m.decode('utf-8') if m else None,
            'consumer_timeout_ms': 1000,  # 1 second timeout
        }
        
        self.consumer = KafkaConsumer(**self.consumer_config)
        self.running = False
        self.metrics = {
            'messages_processed': 0,
            'messages_failed': 0,
            'last_commit_time': None,
            'last_message_time': None
        }
    
    def subscribe(self, topics: list):
        """Subscribe to Kafka topics"""
        self.consumer.subscribe(topics)
        self.logger.info(f"Subscribed to topics: {topics}")
    
    def start_consuming(self, message_handler: Callable[[Dict[str, Any]], bool]):
        """Start consuming messages with error handling"""
        self.running = True
        self.logger.info("Starting message consumption")
        
        try:
            while self.running:
                try:
                    # Poll for messages
                    message_batch = self.consumer.poll(timeout_ms=1000)
                    
                    if not message_batch:
                        continue
                    
                    # Process each message
                    for topic_partition, messages in message_batch.items():
                        self.process_message_batch(
                            topic_partition, 
                            messages, 
                            message_handler
                        )
                
                except KafkaError as e:
                    self.logger.error(f"Kafka error during consumption: {e}")
                    time.sleep(5)  # Back off on error
                
                except Exception as e:
                    self.logger.error(f"Unexpected error during consumption: {e}")
                    time.sleep(5)  # Back off on error
        
        finally:
            self.consumer.close()
            self.logger.info("Consumer stopped")
    
    def process_message_batch(self, topic_partition, messages, message_handler):
        """Process a batch of messages"""
        processed_count = 0
        failed_count = 0
        
        for message in messages:
            try:
                # Process message
                success = message_handler(message.value)
                
                if success:
                    processed_count += 1
                    self.metrics['messages_processed'] += 1
                    self.metrics['last_message_time'] = time.time()
                else:
                    failed_count += 1
                    self.metrics['messages_failed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                failed_count += 1
                self.metrics['messages_failed'] += 1
        
        # Commit offsets if processing was successful
        if processed_count > 0:
            try:
                self.consumer.commit()
                self.metrics['last_commit_time'] = time.time()
                self.logger.debug(f"Committed {processed_count} messages")
            except CommitFailedError as e:
                self.logger.error(f"Failed to commit offsets: {e}")
    
    def stop_consuming(self):
        """Stop consuming messages"""
        self.running = False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics"""
        return {
            **self.metrics,
            'consumer_metrics': self.consumer.metrics()
        }
```

## ⚡ **Spark Technical Implementation**

### **1. Spark Configuration for PBF-LB/M**

#### **Production Spark Configuration**
```python
# src/data_pipeline/config/spark_config.py
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from typing import Dict, Any

class SparkConfigManager:
    def __init__(self):
        self.base_config = {
            # Core settings
            'spark.app.name': 'PBF-LB/M Data Pipeline',
            'spark.master': 'yarn',  # or 'local[*]' for local development
            
            # Memory settings
            'spark.driver.memory': '4g',
            'spark.driver.maxResultSize': '2g',
            'spark.executor.memory': '8g',
            'spark.executor.memoryFraction': '0.8',
            'spark.storage.memoryFraction': '0.2',
            
            # CPU settings
            'spark.executor.cores': '4',
            'spark.executor.instances': '10',
            'spark.default.parallelism': '40',
            
            # Serialization
            'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
            'spark.kryo.registrationRequired': 'false',
            'spark.kryo.unsafe': 'true',
            
            # SQL settings
            'spark.sql.adaptive.enabled': 'true',
            'spark.sql.adaptive.coalescePartitions.enabled': 'true',
            'spark.sql.adaptive.skewJoin.enabled': 'true',
            'spark.sql.adaptive.localShuffleReader.enabled': 'true',
            'spark.sql.adaptive.advisoryPartitionSizeInBytes': '128MB',
            
            # Streaming settings
            'spark.streaming.backpressure.enabled': 'true',
            'spark.streaming.kafka.maxRatePerPartition': '1000',
            'spark.streaming.receiver.maxRate': '1000',
            
            # Performance optimizations
            'spark.sql.execution.arrow.pyspark.enabled': 'true',
            'spark.sql.execution.arrow.maxRecordsPerBatch': '10000',
            'spark.sql.adaptive.skewJoin.skewedPartitionFactor': '5',
            'spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes': '256MB',
            
            # Kafka integration
            'spark.sql.streaming.checkpointLocation': '/tmp/spark-checkpoints',
            'spark.sql.streaming.forceDeleteTempCheckpointLocation': 'true',
            
            # Delta Lake settings
            'spark.sql.extensions': 'io.delta.sql.DeltaSparkSessionExtension',
            'spark.sql.catalog.spark_catalog': 'org.apache.spark.sql.delta.catalog.DeltaCatalog',
            
            # Monitoring
            'spark.sql.execution.metrics.enabled': 'true',
            'spark.sql.execution.metrics.persist': 'true',
            'spark.eventLog.enabled': 'true',
            'spark.eventLog.dir': 'hdfs://namenode:8020/spark-logs',
        }
    
    def get_spark_session(self, app_name: str = None, 
                         additional_config: Dict[str, Any] = None) -> SparkSession:
        """Create a configured Spark session"""
        config = self.base_config.copy()
        
        if app_name:
            config['spark.app.name'] = app_name
        
        if additional_config:
            config.update(additional_config)
        
        # Create SparkConf
        conf = SparkConf()
        for key, value in config.items():
            conf.set(key, value)
        
        # Create SparkSession
        spark = SparkSession.builder \
            .config(conf=conf) \
            .getOrCreate()
        
        # Set log level
        spark.sparkContext.setLogLevel("WARN")
        
        return spark
    
    def get_streaming_config(self) -> Dict[str, Any]:
        """Get configuration optimized for streaming"""
        return {
            'spark.streaming.backpressure.enabled': 'true',
            'spark.streaming.kafka.maxRatePerPartition': '1000',
            'spark.streaming.receiver.maxRate': '1000',
            'spark.streaming.receiver.writeAheadLog.enable': 'true',
            'spark.streaming.driver.writeAheadLog.closeFileAfterWrite': 'true',
            'spark.streaming.receiver.writeAheadLog.rollingIntervalSecs': '60',
            'spark.streaming.receiver.writeAheadLog.maxFailures': '3',
        }
    
    def get_batch_config(self) -> Dict[str, Any]:
        """Get configuration optimized for batch processing"""
        return {
            'spark.sql.adaptive.enabled': 'true',
            'spark.sql.adaptive.coalescePartitions.enabled': 'true',
            'spark.sql.adaptive.skewJoin.enabled': 'true',
            'spark.sql.adaptive.localShuffleReader.enabled': 'true',
            'spark.sql.adaptive.advisoryPartitionSizeInBytes': '128MB',
            'spark.sql.adaptive.skewJoin.skewedPartitionFactor': '5',
            'spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes': '256MB',
        }
```

### **2. Spark Streaming Implementation**

#### **Real-time ISPM Data Processing**
```python
# src/data_pipeline/processing/streaming/ispm_stream_processor.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import json
from typing import Dict, Any

class ISPMStreamProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(__name__)
        
        # Define schema for ISPM data
        self.ispm_schema = StructType([
            StructField("sensor_id", StringType(), True),
            StructField("timestamp", DoubleType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("pressure", DoubleType(), True),
            StructField("laser_power", DoubleType(), True),
            StructField("scan_speed", DoubleType(), True),
            StructField("build_id", StringType(), True),
            StructField("layer", IntegerType(), True),
            StructField("_metadata", StructType([
                StructField("producer_id", StringType(), True),
                StructField("message_id", StringType(), True)
            ]), True)
        ])
    
    def create_streaming_query(self, kafka_servers: str, topics: list) -> Any:
        """Create a streaming query for ISPM data"""
        
        # Read from Kafka
        kafka_df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_servers) \
            .option("subscribe", ",".join(topics)) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .option("kafka.consumer.poll.timeoutMs", "1000") \
            .load()
        
        # Parse JSON data
        parsed_df = kafka_df \
            .select(
                from_json(col("value").cast("string"), self.ispm_schema).alias("data")
            ) \
            .select("data.*")
        
        # Add processing timestamp
        processed_df = parsed_df \
            .withColumn("processing_timestamp", current_timestamp()) \
            .withColumn("date", to_date(col("processing_timestamp")))
        
        # Detect anomalies
        anomaly_df = self.detect_anomalies(processed_df)
        
        # Calculate quality metrics
        quality_df = self.calculate_quality_metrics(processed_df)
        
        # Write to multiple sinks
        self.write_to_sinks(anomaly_df, quality_df)
        
        return processed_df
    
    def detect_anomalies(self, df) -> Any:
        """Detect anomalies in ISPM data"""
        
        # Define anomaly conditions
        anomaly_conditions = (
            (col("temperature") > 1700) |  # Temperature too high
            (col("temperature") < 1500) |  # Temperature too low
            (col("pressure") > 1.0) |     # Pressure too high
            (col("pressure") < 0.9) |     # Pressure too low
            (col("laser_power") > 250) |  # Laser power too high
            (col("laser_power") < 150)    # Laser power too low
        )
        
        # Filter anomalies
        anomaly_df = df \
            .filter(anomaly_conditions) \
            .withColumn("anomaly_type", 
                when(col("temperature") > 1700, "high_temperature")
                .when(col("temperature") < 1500, "low_temperature")
                .when(col("pressure") > 1.0, "high_pressure")
                .when(col("pressure") < 0.9, "low_pressure")
                .when(col("laser_power") > 250, "high_laser_power")
                .when(col("laser_power") < 150, "low_laser_power")
                .otherwise("unknown")
            ) \
            .withColumn("severity", 
                when(col("anomaly_type").isin(["high_temperature", "high_pressure"]), "critical")
                .when(col("anomaly_type").isin(["low_temperature", "low_pressure"]), "warning")
                .otherwise("info")
            )
        
        return anomaly_df
    
    def calculate_quality_metrics(self, df) -> Any:
        """Calculate quality metrics from ISPM data"""
        
        # Window specification for time-based aggregations
        window_spec = Window.partitionBy("build_id", "layer").orderBy("timestamp")
        
        # Calculate metrics
        quality_df = df \
            .withColumn("temperature_avg", avg("temperature").over(window_spec)) \
            .withColumn("temperature_std", stddev("temperature").over(window_spec)) \
            .withColumn("pressure_avg", avg("pressure").over(window_spec)) \
            .withColumn("pressure_std", stddev("pressure").over(window_spec)) \
            .withColumn("laser_power_avg", avg("laser_power").over(window_spec)) \
            .withColumn("laser_power_std", stddev("laser_power").over(window_spec)) \
            .withColumn("scan_speed_avg", avg("scan_speed").over(window_spec)) \
            .withColumn("scan_speed_std", stddev("scan_speed").over(window_spec)) \
            .withColumn("quality_score", 
                when(col("temperature_std") < 10, 1.0)
                .when(col("temperature_std") < 20, 0.8)
                .when(col("temperature_std") < 30, 0.6)
                .otherwise(0.4)
            )
        
        return quality_df
    
    def write_to_sinks(self, anomaly_df, quality_df):
        """Write data to multiple sinks"""
        
        # Write anomalies to PostgreSQL
        anomaly_query = anomaly_df \
            .writeStream \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://localhost:5432/pbf") \
            .option("dbtable", "ispm_anomalies") \
            .option("user", "pbf_user") \
            .option("password", "pbf_password") \
            .option("driver", "org.postgresql.Driver") \
            .option("checkpointLocation", "/tmp/spark-checkpoints/anomalies") \
            .trigger(processingTime='10 seconds') \
            .start()
        
        # Write quality metrics to MongoDB
        quality_query = quality_df \
            .writeStream \
            .format("mongo") \
            .option("uri", "mongodb://localhost:27017/pbf") \
            .option("database", "pbf") \
            .option("collection", "quality_metrics") \
            .option("checkpointLocation", "/tmp/spark-checkpoints/quality") \
            .trigger(processingTime='30 seconds') \
            .start()
        
        return anomaly_query, quality_query
```

### **3. Spark Batch Processing**

#### **Build File Processing with libSLM**
```python
# src/data_pipeline/processing/batch/build_file_processor.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import sys
import os

# Add libSLM to Python path
sys.path.append('/path/to/libSLM/python')

class BuildFileProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(__name__)
        
        # Import libSLM
        try:
            import slm
            self.slm = slm
            self.libslm_available = True
        except ImportError:
            self.logger.warning("libSLM not available")
            self.libslm_available = False
    
    def process_build_files(self, input_path: str, output_path: str):
        """Process build files using Spark and libSLM"""
        
        if not self.libslm_available:
            raise RuntimeError("libSLM not available")
        
        # Read build files
        build_files_df = self.spark \
            .read \
            .format("binaryFile") \
            .load(input_path)
        
        # Process each build file
        processed_df = build_files_df.rdd.mapPartitions(self.process_partition).toDF()
        
        # Write results
        processed_df.write \
            .mode("overwrite") \
            .parquet(output_path)
        
        return processed_df
    
    def process_partition(self, partition):
        """Process a partition of build files"""
        results = []
        
        for row in partition:
            try:
                # Get file path and content
                file_path = row.path
                file_content = row.content
                
                # Parse build file with libSLM
                build_data = self.parse_build_file(file_path, file_content)
                
                if build_data:
                    results.append(build_data)
                
            except Exception as e:
                self.logger.error(f"Error processing file {row.path}: {e}")
                continue
        
        return results
    
    def parse_build_file(self, file_path: str, file_content: bytes) -> Dict[str, Any]:
        """Parse a single build file using libSLM"""
        try:
            # Determine file format
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.mtt':
                return self.parse_mtt_file(file_content)
            elif file_ext in ['.sli', '.cli']:
                return self.parse_eos_file(file_content)
            elif file_ext == '.slm':
                return self.parse_slm_file(file_content)
            else:
                self.logger.warning(f"Unsupported file format: {file_ext}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error parsing build file {file_path}: {e}")
            return None
    
    def parse_mtt_file(self, file_content: bytes) -> Dict[str, Any]:
        """Parse MTT file using libSLM"""
        try:
            # Create MTT reader
            mtt_reader = self.slm.MTTReader()
            
            # Read file
            mtt_reader.readFromMemory(file_content)
            
            # Extract data
            build_data = {
                'file_format': 'mtt',
                'layers': [],
                'metadata': {}
            }
            
            # Get layer count
            layer_count = mtt_reader.getLayerCount()
            
            for layer_id in range(layer_count):
                layer_data = self.extract_layer_data(mtt_reader, layer_id)
                build_data['layers'].append(layer_data)
            
            # Get metadata
            build_data['metadata'] = self.extract_metadata(mtt_reader)
            
            return build_data
            
        except Exception as e:
            self.logger.error(f"Error parsing MTT file: {e}")
            return None
    
    def extract_layer_data(self, reader, layer_id: int) -> Dict[str, Any]:
        """Extract data from a specific layer"""
        try:
            layer_data = {
                'layer_id': layer_id,
                'geometries': [],
                'build_parameters': {}
            }
            
            # Get geometry count
            geom_count = reader.getGeometryCount(layer_id)
            
            for geom_id in range(geom_count):
                geometry_data = self.extract_geometry_data(reader, layer_id, geom_id)
                layer_data['geometries'].append(geometry_data)
            
            # Get build parameters
            layer_data['build_parameters'] = self.extract_build_parameters(reader, layer_id)
            
            return layer_data
            
        except Exception as e:
            self.logger.error(f"Error extracting layer {layer_id} data: {e}")
            return None
    
    def extract_geometry_data(self, reader, layer_id: int, geom_id: int) -> Dict[str, Any]:
        """Extract data from a specific geometry"""
        try:
            geometry_data = {
                'geometry_id': geom_id,
                'geometry_type': reader.getGeometryType(layer_id, geom_id),
                'coordinates': [],
                'build_style': {}
            }
            
            # Get coordinates
            coords = reader.getCoordinates(layer_id, geom_id)
            geometry_data['coordinates'] = coords.tolist()
            
            # Get build style
            build_style_id = reader.getBuildStyleId(layer_id, geom_id)
            build_style = reader.getBuildStyle(build_style_id)
            
            geometry_data['build_style'] = {
                'power': build_style.getPower(),
                'speed': build_style.getSpeed(),
                'exposure_time': build_style.getExposureTime(),
                'point_distance': build_style.getPointDistance(),
                'jump_speed': build_style.getJumpSpeed(),
                'jump_delay': build_style.getJumpDelay()
            }
            
            return geometry_data
            
        except Exception as e:
            self.logger.error(f"Error extracting geometry {geom_id} data: {e}")
            return None
```

## 🗄️ **Data Storage Technical Implementation**

### **1. Multi-Model Storage Configuration**

#### **Local Storage Setup**
```python
# src/data_pipeline/config/storage_config.py
from typing import Dict, Any
import os

class LocalStorageConfig:
    def __init__(self):
        self.postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'pbf_lbm'),
            'user': os.getenv('POSTGRES_USER', 'pbf_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'pbf_password'),
            'pool_size': 20,
            'max_overflow': 30,
            'pool_timeout': 30,
            'pool_recycle': 3600,
        }
        
        self.mongodb_config = {
            'host': os.getenv('MONGODB_HOST', 'localhost'),
            'port': int(os.getenv('MONGODB_PORT', 27017)),
            'database': os.getenv('MONGODB_DB', 'pbf_lbm'),
            'username': os.getenv('MONGODB_USER', 'pbf_user'),
            'password': os.getenv('MONGODB_PASSWORD', 'pbf_password'),
            'max_pool_size': 100,
            'min_pool_size': 10,
            'max_idle_time_ms': 30000,
        }
        
        self.redis_config = {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD', None),
            'db': int(os.getenv('REDIS_DB', 0)),
            'max_connections': 50,
            'socket_timeout': 5,
            'socket_connect_timeout': 5,
        }
        
        self.local_s3_config = {
            'endpoint_url': os.getenv('LOCAL_S3_ENDPOINT', 'http://localhost:9000'),
            'aws_access_key_id': os.getenv('LOCAL_S3_ACCESS_KEY', 'minioadmin'),
            'aws_secret_access_key': os.getenv('LOCAL_S3_SECRET_KEY', 'minioadmin'),
            'region_name': os.getenv('LOCAL_S3_REGION', 'us-east-1'),
            'bucket_name': os.getenv('LOCAL_S3_BUCKET', 'pbf-lbm-data'),
        }

#### **Cloud Storage Setup**
```python
class CloudStorageConfig:
    def __init__(self):
        self.aws_s3_config = {
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'region_name': os.getenv('AWS_REGION', 'us-east-1'),
            'bucket_name': os.getenv('AWS_S3_BUCKET', 'pbf-lbm-data'),
            'storage_class': 'STANDARD_IA',  # For cost optimization
        }
        
        self.snowflake_config = {
            'account': os.getenv('SNOWFLAKE_ACCOUNT'),
            'user': os.getenv('SNOWFLAKE_USER'),
            'password': os.getenv('SNOWFLAKE_PASSWORD'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'PBF_WH'),
            'database': os.getenv('SNOWFLAKE_DATABASE', 'PBF_LBM'),
            'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC'),
            'role': os.getenv('SNOWFLAKE_ROLE', 'PBF_ROLE'),
        }
        
        self.bigquery_config = {
            'project_id': os.getenv('GCP_PROJECT_ID'),
            'credentials_path': os.getenv('GCP_CREDENTIALS_PATH'),
            'dataset_id': os.getenv('BIGQUERY_DATASET', 'pbf_lbm'),
            'location': os.getenv('BIGQUERY_LOCATION', 'US'),
        }
        
        self.mongodb_atlas_config = {
            'connection_string': os.getenv('MONGODB_ATLAS_CONNECTION_STRING'),
            'database': os.getenv('MONGODB_ATLAS_DB', 'pbf_lbm'),
            'max_pool_size': 100,
            'min_pool_size': 10,
        }
```

### **2. Data Storage Implementation**

#### **Unified Storage Manager**
```python
# src/data_pipeline/storage/unified_storage_manager.py
from typing import Dict, Any, Optional, Union
import pandas as pd
import json
from datetime import datetime
import logging

class UnifiedStorageManager:
    def __init__(self, local_config: LocalStorageConfig, cloud_config: CloudStorageConfig):
        self.local_config = local_config
        self.cloud_config = cloud_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage clients
        self.postgres_client = self._init_postgres()
        self.mongodb_client = self._init_mongodb()
        self.redis_client = self._init_redis()
        self.local_s3_client = self._init_local_s3()
        self.aws_s3_client = self._init_aws_s3()
        self.snowflake_client = self._init_snowflake()
        self.bigquery_client = self._init_bigquery()
    
    def store_data(self, data: Any, storage_type: str, 
                   data_type: str, metadata: Dict[str, Any] = None) -> bool:
        """Store data in appropriate storage based on type and requirements"""
        try:
            if storage_type == 'operational':
                return self._store_operational_data(data, data_type, metadata)
            elif storage_type == 'analytics':
                return self._store_analytics_data(data, data_type, metadata)
            elif storage_type == 'ml_training':
                return self._store_ml_data(data, data_type, metadata)
            elif storage_type == 'research':
                return self._store_research_data(data, data_type, metadata)
            else:
                raise ValueError(f"Unknown storage type: {storage_type}")
        except Exception as e:
            self.logger.error(f"Error storing data: {e}")
            return False
    
    def _store_operational_data(self, data: Any, data_type: str, metadata: Dict[str, Any]) -> bool:
        """Store operational data in local storage"""
        if data_type == 'structured':
            return self._store_in_postgres(data, metadata)
        elif data_type == 'document':
            return self._store_in_mongodb(data, metadata)
        elif data_type == 'cache':
            return self._store_in_redis(data, metadata)
        else:
            return self._store_in_local_s3(data, metadata)
    
    def _store_analytics_data(self, data: Any, data_type: str, metadata: Dict[str, Any]) -> bool:
        """Store analytics data in cloud storage"""
        if data_type == 'structured':
            return self._store_in_snowflake(data, metadata)
        elif data_type == 'document':
            return self._store_in_mongodb_atlas(data, metadata)
        else:
            return self._store_in_aws_s3(data, metadata)
    
    def _store_ml_data(self, data: Any, data_type: str, metadata: Dict[str, Any]) -> bool:
        """Store ML training data in both local and cloud"""
        # Store in local for fast access
        local_success = self._store_in_local_s3(data, metadata)
        
        # Store in cloud for scalability
        cloud_success = self._store_in_aws_s3(data, metadata)
        
        return local_success and cloud_success
    
    def _store_research_data(self, data: Any, data_type: str, metadata: Dict[str, Any]) -> bool:
        """Store research data in cloud for collaboration"""
        if data_type == 'structured':
            return self._store_in_bigquery(data, metadata)
        else:
            return self._store_in_aws_s3(data, metadata)
    
    def _store_in_postgres(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
        """Store structured data in PostgreSQL"""
        try:
            table_name = metadata.get('table_name', 'pbf_data')
            if_exists = metadata.get('if_exists', 'append')
            
            data.to_sql(
                table_name,
                self.postgres_client,
                if_exists=if_exists,
                index=False,
                method='multi'
            )
            return True
        except Exception as e:
            self.logger.error(f"Error storing in PostgreSQL: {e}")
            return False
    
    def _store_in_mongodb(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Store document data in MongoDB"""
        try:
            collection_name = metadata.get('collection_name', 'pbf_data')
            collection = self.mongodb_client[collection_name]
            
            # Add metadata
            data['_metadata'] = {
                'stored_at': datetime.now(),
                'source': metadata.get('source', 'unknown'),
                'version': metadata.get('version', '1.0')
            }
            
            collection.insert_one(data)
            return True
        except Exception as e:
            self.logger.error(f"Error storing in MongoDB: {e}")
            return False
    
    def _store_in_redis(self, data: Any, metadata: Dict[str, Any]) -> bool:
        """Store cache data in Redis"""
        try:
            key = metadata.get('key', 'pbf_data')
            ttl = metadata.get('ttl', 3600)  # 1 hour default
            
            if isinstance(data, dict):
                data = json.dumps(data)
            
            self.redis_client.setex(key, ttl, data)
            return True
        except Exception as e:
            self.logger.error(f"Error storing in Redis: {e}")
            return False
    
    def _store_in_local_s3(self, data: Any, metadata: Dict[str, Any]) -> bool:
        """Store data in local S3"""
        try:
            bucket_name = self.local_s3_config['bucket_name']
            key = metadata.get('key', f"pbf_data/{datetime.now().isoformat()}")
            
            if isinstance(data, pd.DataFrame):
                data = data.to_parquet()
            elif isinstance(data, dict):
                data = json.dumps(data).encode()
            
            self.local_s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=data,
                ContentType=metadata.get('content_type', 'application/octet-stream')
            )
            return True
        except Exception as e:
            self.logger.error(f"Error storing in local S3: {e}")
            return False
    
    def _store_in_aws_s3(self, data: Any, metadata: Dict[str, Any]) -> bool:
        """Store data in AWS S3"""
        try:
            bucket_name = self.aws_s3_config['bucket_name']
            key = metadata.get('key', f"pbf_data/{datetime.now().isoformat()}")
            
            if isinstance(data, pd.DataFrame):
                data = data.to_parquet()
            elif isinstance(data, dict):
                data = json.dumps(data).encode()
            
            self.aws_s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=data,
                ContentType=metadata.get('content_type', 'application/octet-stream'),
                StorageClass=metadata.get('storage_class', 'STANDARD_IA')
            )
            return True
        except Exception as e:
            self.logger.error(f"Error storing in AWS S3: {e}")
            return False
    
    def _store_in_snowflake(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
        """Store structured data in Snowflake"""
        try:
            table_name = metadata.get('table_name', 'PBF_DATA')
            schema = metadata.get('schema', 'PUBLIC')
            
            # Convert DataFrame to Snowflake format
            snowflake_df = self._prepare_snowflake_data(data)
            
            # Write to Snowflake
            snowflake_df.write \
                .format("snowflake") \
                .option("sfURL", self.snowflake_config['account']) \
                .option("sfUser", self.snowflake_config['user']) \
                .option("sfPassword", self.snowflake_config['password']) \
                .option("sfDatabase", self.snowflake_config['database']) \
                .option("sfSchema", schema) \
                .option("sfWarehouse", self.snowflake_config['warehouse']) \
                .option("dbtable", table_name) \
                .mode("append") \
                .save()
            
            return True
        except Exception as e:
            self.logger.error(f"Error storing in Snowflake: {e}")
            return False
    
    def _store_in_bigquery(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
        """Store structured data in BigQuery"""
        try:
            table_id = metadata.get('table_id', 'pbf_data')
            dataset_id = self.bigquery_config['dataset_id']
            
            # Write to BigQuery
            data.to_gbq(
                destination_table=f"{dataset_id}.{table_id}",
                project_id=self.bigquery_config['project_id'],
                if_exists='append',
                credentials=self.bigquery_config['credentials_path']
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Error storing in BigQuery: {e}")
            return False
```

### **3. ML Data Pipeline Integration**

#### **ML Training Data Pipeline**
```python
# src/data_pipeline/ml/ml_data_pipeline.py
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
import logging

class MLDataPipeline:
    def __init__(self, storage_manager: UnifiedStorageManager):
        self.storage_manager = storage_manager
        self.logger = logging.getLogger(__name__)
    
    def prepare_training_data(self, data_sources: List[str], 
                            time_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Prepare training data from multiple sources"""
        try:
            training_data = {}
            
            for source in data_sources:
                if source == 'ispm_monitoring':
                    training_data[source] = self._prepare_ispm_data(time_range)
                elif source == 'ct_scans':
                    training_data[source] = self._prepare_ct_data(time_range)
                elif source == 'build_files':
                    training_data[source] = self._prepare_build_data(time_range)
                elif source == 'quality_metrics':
                    training_data[source] = self._prepare_quality_data(time_range)
            
            # Combine and clean data
            combined_data = self._combine_training_data(training_data)
            
            # Store in both local and cloud for ML training
            self._store_training_data(combined_data)
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return {}
    
    def _prepare_ispm_data(self, time_range: Dict[str, datetime]) -> pd.DataFrame:
        """Prepare ISPM monitoring data for ML training"""
        # Query ISPM data from PostgreSQL
        query = """
        SELECT sensor_id, timestamp, temperature, pressure, 
               laser_power, scan_speed, build_id, layer
        FROM ispm_monitoring 
        WHERE timestamp BETWEEN %s AND %s
        """
        
        # This would be implemented with actual database connection
        return pd.DataFrame()  # Placeholder
    
    def _prepare_ct_data(self, time_range: Dict[str, datetime]) -> pd.DataFrame:
        """Prepare CT scan data for ML training"""
        # Query CT data from MongoDB
        # This would be implemented with actual database connection
        return pd.DataFrame()  # Placeholder
    
    def _prepare_build_data(self, time_range: Dict[str, datetime]) -> pd.DataFrame:
        """Prepare build file data for ML training"""
        # Query build data from Snowflake
        # This would be implemented with actual database connection
        return pd.DataFrame()  # Placeholder
    
    def _prepare_quality_data(self, time_range: Dict[str, datetime]) -> pd.DataFrame:
        """Prepare quality metrics data for ML training"""
        # Query quality data from BigQuery
        # This would be implemented with actual database connection
        return pd.DataFrame()  # Placeholder
    
    def _combine_training_data(self, training_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from multiple sources"""
        # Implement data combination logic
        # This would include data cleaning, feature engineering, etc.
        return pd.DataFrame()  # Placeholder
    
    def _store_training_data(self, data: pd.DataFrame):
        """Store training data in both local and cloud storage"""
        # Store in local S3 for fast access
        self.storage_manager.store_data(
            data, 
            storage_type='ml_training',
            data_type='parquet',
            metadata={'key': f"training_data/{datetime.now().isoformat()}"}
        )
        
        # Store in AWS S3 for scalability
        self.storage_manager.store_data(
            data,
            storage_type='ml_training',
            data_type='parquet',
            metadata={'key': f"training_data/{datetime.now().isoformat()}"}
        )
```

## ⏰ **Airflow Technical Implementation**

### **1. Airflow Configuration**

#### **Production Airflow Configuration**
```python
# src/data_pipeline/config/airflow_config.py
from airflow.configuration import conf
from airflow.models import Variable
from typing import Dict, Any

class AirflowConfigManager:
    def __init__(self):
        self.config = {
            # Core settings
            'core.dags_folder': '/opt/airflow/dags',
            'core.base_log_folder': '/opt/airflow/logs',
            'core.executor': 'LocalExecutor',
            'core.parallelism': 32,
            'core.dag_concurrency': 16,
            'core.max_active_runs_per_dag': 1,
            'core.max_active_tasks_per_dag': 16,
            
            # Database settings
            'core.sql_alchemy_conn': 'postgresql://airflow:airflow@localhost:5432/airflow',
            'core.sql_alchemy_pool_size': 5,
            'core.sql_alchemy_max_overflow': 10,
            'core.sql_alchemy_pool_recycle': 1800,
            'core.sql_alchemy_pool_pre_ping': True,
            
            # Scheduler settings
            'scheduler.dag_dir_list_interval': 300,
            'scheduler.max_threads': 2,
            'scheduler.scheduler_heartbeat_sec': 5,
            'scheduler.catchup_by_default': False,
            'scheduler.dag_default_view': 'tree',
            
            # Webserver settings
            'webserver.base_url': 'http://localhost:8080',
            'webserver.web_server_port': 8080,
            'webserver.workers': 4,
            'webserver.worker_timeout': 120,
            'webserver.worker_refresh_batch_size': 1,
            'webserver.worker_refresh_interval': 30,
            
            # Email settings
            'smtp.smtp_host': 'localhost',
            'smtp.smtp_starttls': True,
            'smtp.smtp_ssl': False,
            'smtp.smtp_user': 'airflow',
            'smtp.smtp_password': 'airflow',
            'smtp.smtp_port': 587,
            'smtp.smtp_mail_from': 'airflow@example.com',
            
            # Logging settings
            'logging.logging_level': 'INFO',
            'logging.logging_config_class': 'airflow.config_templates.airflow_local_settings.LOGGING_CONFIG',
            'logging.colored_console_log': True,
            'logging.colored_log_format': '[%%(blue)s%%(asctime)s%%(reset)s] {%%(blue)s%%(filename)s:%%(reset)s%%(lineno)d} %%(log_color)s%%(levelname)s%%(reset)s - %%(log_color)s%%(message)s%%(reset)s',
            'logging.colored_formatter_class': 'airflow.utils.log.colored_log.CustomTTYColoredFormatter',
            
            # Security settings
            'webserver.secret_key': 'your-secret-key-here',
            'webserver.wtf_csrf_enabled': True,
            'webserver.wtf_csrf_time_limit': 3600,
            
            # Performance settings
            'core.worker_precheck': True,
            'core.worker_refresh_batch_size': 1,
            'core.worker_refresh_interval': 30,
            'core.worker_timeout': 120,
            'core.worker_precheck': True,
        }
    
    def apply_config(self):
        """Apply configuration to Airflow"""
        for key, value in self.config.items():
            conf.set(key, value)
    
    def get_spark_config(self) -> Dict[str, Any]:
        """Get Spark-specific configuration"""
        return {
            'spark.app.name': 'PBF-LB/M Airflow Spark',
            'spark.master': 'yarn',
            'spark.executor.memory': '4g',
            'spark.executor.cores': '2',
            'spark.executor.instances': '5',
            'spark.driver.memory': '2g',
            'spark.sql.adaptive.enabled': 'true',
            'spark.sql.adaptive.coalescePartitions.enabled': 'true',
        }
    
    def get_kafka_config(self) -> Dict[str, Any]:
        """Get Kafka-specific configuration"""
        return {
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'airflow-consumer',
            'auto.offset.reset': 'latest',
            'enable.auto.commit': 'false',
            'max.poll.records': '500',
            'session.timeout.ms': '30000',
            'heartbeat.interval.ms': '3000',
        }
```

### **2. Advanced DAG Implementation**

#### **Complex PBF Process Monitoring DAG**
```python
# src/data_pipeline/orchestration/airflow/pbf_process_monitoring_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageSensor
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.sensors.base import PokeReturnValue
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from airflow.utils.dates import days_ago
import json
import logging

# Default arguments
default_args = {
    'owner': 'pbf-data-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}

# DAG definition
dag = DAG(
    'pbf_process_monitoring',
    default_args=default_args,
    description='Comprehensive PBF process monitoring pipeline',
    schedule_interval='@hourly',
    catchup=False,
    max_active_runs=1,
    max_active_tasks=10,
    tags=['pbf', 'monitoring', 'quality', 'real-time'],
)

# Configuration
KAFKA_SERVERS = Variable.get("kafka_servers", "localhost:9092")
SPARK_MASTER = Variable.get("spark_master", "yarn")
POSTGRES_CONN_ID = Variable.get("postgres_conn_id", "postgres_default")
MONGODB_CONN_ID = Variable.get("mongodb_conn_id", "mongodb_default")

# Task Groups
with TaskGroup("data_ingestion", dag=dag) as data_ingestion_group:
    """Data ingestion tasks"""
    
    # Start Kafka consumer for real-time data
    start_kafka_consumer = PythonOperator(
        task_id="start_kafka_consumer",
        python_callable=start_ispm_monitoring,
        op_kwargs={
            'kafka_servers': KAFKA_SERVERS,
            'topics': ['ispm-monitoring', 'powder-bed-images']
        },
        pool='kafka_pool',
        pool_slots=1,
    )
    
    # Check for new CT scan files
    ct_scan_sensor = FileSensor(
        task_id="ct_scan_sensor",
        filepath="/data/ct-scans/",
        fs_conn_id="fs_default",
        poke_interval=60,
        timeout=3600,
        mode='poke',
    )
    
    # Check for new build files
    build_file_sensor = FileSensor(
        task_id="build_file_sensor",
        filepath="/data/build-files/",
        fs_conn_id="fs_default",
        poke_interval=60,
        timeout=3600,
        mode='poke',
    )

with TaskGroup("stream_processing", dag=dag) as stream_processing_group:
    """Stream processing tasks"""
    
    # Process ISPM data
    process_ispm_data = SparkSubmitOperator(
        task_id="process_ispm_data",
        application="/opt/airflow/dags/spark_jobs/ispm_stream_processor.py",
        conn_id="spark_default",
        conf={
            "spark.master": SPARK_MASTER,
            "spark.app.name": "ISPM Stream Processor",
            "spark.streaming.kafka.maxRatePerPartition": "1000",
            "spark.sql.adaptive.enabled": "true",
        },
        application_args=[
            "--kafka-servers", KAFKA_SERVERS,
            "--topics", "ispm-monitoring",
            "--output-path", "s3://pbf-data/processed/ispm/"
        ],
        pool='spark_pool',
        pool_slots=2,
    )
    
    # Process powder bed images
    process_powder_bed = SparkSubmitOperator(
        task_id="process_powder_bed",
        application="/opt/airflow/dags/spark_jobs/powder_bed_processor.py",
        conn_id="spark_default",
        conf={
            "spark.master": SPARK_MASTER,
            "spark.app.name": "Powder Bed Processor",
            "spark.sql.adaptive.enabled": "true",
        },
        application_args=[
            "--kafka-servers", KAFKA_SERVERS,
            "--topics", "powder-bed-images",
            "--output-path", "s3://pbf-data/processed/powder-bed/"
        ],
        pool='spark_pool',
        pool_slots=2,
    )

with TaskGroup("batch_processing", dag=dag) as batch_processing_group:
    """Batch processing tasks"""
    
    # Process CT scan data
    process_ct_data = SparkSubmitOperator(
        task_id="process_ct_data",
        application="/opt/airflow/dags/spark_jobs/ct_scan_processor.py",
        conn_id="spark_default",
        conf={
            "spark.master": SPARK_MASTER,
            "spark.app.name": "CT Scan Processor",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
        },
        application_args=[
            "--input-path", "/data/ct-scans/",
            "--output-path", "s3://pbf-data/processed/ct-scans/"
        ],
        pool='spark_pool',
        pool_slots=3,
    )
    
    # Process build files
    process_build_files = SparkSubmitOperator(
        task_id="process_build_files",
        application="/opt/airflow/dags/spark_jobs/build_file_processor.py",
        conn_id="spark_default",
        conf={
            "spark.master": SPARK_MASTER,
            "spark.app.name": "Build File Processor",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
        },
        application_args=[
            "--input-path", "/data/build-files/",
            "--output-path", "s3://pbf-data/processed/build-files/"
        ],
        pool='spark_pool',
        pool_slots=3,
    )

with TaskGroup("data_fusion", dag=dag) as data_fusion_group:
    """Data fusion tasks"""
    
    # Fuse multimodal data
    fuse_data = SparkSubmitOperator(
        task_id="fuse_data",
        application="/opt/airflow/dags/spark_jobs/data_fusion.py",
        conn_id="spark_default",
        conf={
            "spark.master": SPARK_MASTER,
            "spark.app.name": "Data Fusion",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
        },
        application_args=[
            "--ispm-path", "s3://pbf-data/processed/ispm/",
            "--ct-path", "s3://pbf-data/processed/ct-scans/",
            "--build-path", "s3://pbf-data/processed/build-files/",
            "--output-path", "s3://pbf-data/fused/"
        ],
        pool='spark_pool',
        pool_slots=4,
    )
    
    # Generate quality metrics
    generate_quality_metrics = PythonOperator(
        task_id="generate_quality_metrics",
        python_callable=generate_quality_metrics,
        op_kwargs={
            'fused_data_path': 's3://pbf-data/fused/',
            'output_path': 's3://pbf-data/quality-metrics/'
        },
        pool='python_pool',
        pool_slots=2,
    )

with TaskGroup("quality_assessment", dag=dag) as quality_assessment_group:
    """Quality assessment tasks"""
    
    # Assess quality
    assess_quality = PythonOperator(
        task_id="assess_quality",
        python_callable=assess_quality,
        op_kwargs={
            'quality_metrics_path': 's3://pbf-data/quality-metrics/',
            'thresholds': {
                'temperature_std': 20,
                'pressure_std': 0.05,
                'laser_power_std': 10
            }
        },
        pool='python_pool',
        pool_slots=1,
    )
    
    # Generate quality report
    generate_quality_report = PythonOperator(
        task_id="generate_quality_report",
        python_callable=generate_quality_report,
        op_kwargs={
            'quality_data_path': 's3://pbf-data/quality-metrics/',
            'report_path': 's3://pbf-data/reports/'
        },
        pool='python_pool',
        pool_slots=1,
    )
    
    # Send alerts if needed
    send_alerts = PythonOperator(
        task_id="send_alerts",
        python_callable=send_quality_alerts,
        op_kwargs={
            'quality_data_path': 's3://pbf-data/quality-metrics/',
            'alert_threshold': 0.7
        },
        pool='python_pool',
        pool_slots=1,
    )

# Task dependencies
data_ingestion_group >> stream_processing_group
data_ingestion_group >> batch_processing_group
stream_processing_group >> data_fusion_group
batch_processing_group >> data_fusion_group
data_fusion_group >> quality_assessment_group

# Python functions
def start_ispm_monitoring(kafka_servers: str, topics: list):
    """Start ISPM monitoring consumer"""
    from src.data_pipeline.ingestion.streaming.ispm_producer import ISPMDataProducer
    
    producer = ISPMDataProducer(kafka_servers)
    producer.start_producing({
        'sensor_id': 'ispm_001',
        'build_id': 'build_001',
        'collection_interval': 1.0
    })

def generate_quality_metrics(fused_data_path: str, output_path: str):
    """Generate quality metrics from fused data"""
    from src.data_pipeline.processing.analytics.quality_analyzer import QualityAnalyzer
    
    analyzer = QualityAnalyzer()
    metrics = analyzer.analyze_quality(fused_data_path)
    analyzer.save_metrics(metrics, output_path)

def assess_quality(quality_metrics_path: str, thresholds: dict):
    """Assess quality against thresholds"""
    from src.data_pipeline.quality.assessment import QualityAssessor
    
    assessor = QualityAssessor()
    assessment = assessor.assess_quality(quality_metrics_path, thresholds)
    assessor.save_assessment(assessment, quality_metrics_path)

def generate_quality_report(quality_data_path: str, report_path: str):
    """Generate quality report"""
    from src.data_pipeline.reporting.quality_reporter import QualityReporter
    
    reporter = QualityReporter()
    report = reporter.generate_report(quality_data_path)
    reporter.save_report(report, report_path)

def send_quality_alerts(quality_data_path: str, alert_threshold: float):
    """Send quality alerts if needed"""
    from src.data_pipeline.alerting.quality_alerter import QualityAlerter
    
    alerter = QualityAlerter()
    alerts = alerter.check_alerts(quality_data_path, alert_threshold)
    if alerts:
        alerter.send_alerts(alerts)
```

This technical implementation document provides:

1. **Production-ready configurations** for Kafka, Spark, and Airflow
2. **High-performance implementations** with error handling and monitoring
3. **Real-world examples** specific to PBF-LB/M manufacturing
4. **Advanced features** like streaming, batch processing, and data fusion
5. **Best practices** for production deployment

The document complements the orchestration guide by providing the technical depth needed for actual implementation.
