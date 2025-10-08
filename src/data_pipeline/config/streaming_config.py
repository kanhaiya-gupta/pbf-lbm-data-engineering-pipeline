"""
Streaming Configuration

This module provides streaming configuration for PBF-LB/M real-time data processing.
It handles Kafka, Flink, and streaming job configurations.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class KafkaConfig:
    """Kafka configuration for PBF-LB/M streaming"""
    
    # Connection settings
    bootstrap_servers: str = "localhost:9092"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str = "PLAIN"
    sasl_username: str = ""
    sasl_password: str = ""
    
    # Producer settings
    producer_config: Dict[str, Any] = None
    
    # Consumer settings
    consumer_config: Dict[str, Any] = None
    
    # Topic settings
    topics: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided"""
        if self.producer_config is None:
            self.producer_config = {
                "acks": "all",
                "retries": 3,
                "batch_size": 16384,
                "linger_ms": 5,
                "buffer_memory": 33554432,
                "key_serializer": "org.apache.kafka.common.serialization.StringSerializer",
                "value_serializer": "org.apache.kafka.common.serialization.StringSerializer"
            }
        
        if self.consumer_config is None:
            self.consumer_config = {
                "group_id": "pbf-lbm-consumer-group",
                "auto_offset_reset": "earliest",
                "enable_auto_commit": True,
                "auto_commit_interval_ms": 1000,
                "session_timeout_ms": 30000,
                "key_deserializer": "org.apache.kafka.common.serialization.StringDeserializer",
                "value_deserializer": "org.apache.kafka.common.serialization.StringDeserializer"
            }
        
        if self.topics is None:
            self.topics = {
                "pbf_process": {
                    "name": "pbf_process_data",
                    "partitions": 3,
                    "replication_factor": 1,
                    "retention_ms": 604800000  # 7 days
                },
                "ispm_monitoring": {
                    "name": "ispm_monitoring_data",
                    "partitions": 3,
                    "replication_factor": 1,
                    "retention_ms": 604800000  # 7 days
                },
                "powder_bed": {
                    "name": "powder_bed_data",
                    "partitions": 3,
                    "replication_factor": 1,
                    "retention_ms": 604800000  # 7 days
                },
                "ct_scan": {
                    "name": "ct_scan_data",
                    "partitions": 3,
                    "replication_factor": 1,
                    "retention_ms": 604800000  # 7 days
                }
            }


@dataclass
class FlinkConfig:
    """Apache Flink configuration for PBF-LB/M stream processing"""
    
    # Flink cluster settings
    job_manager_host: str = "localhost"
    job_manager_port: int = 6123
    task_manager_host: str = "localhost"
    task_manager_port: int = 6122
    
    # Job settings
    job_name: str = "PBF-LB/M-Stream-Processing"
    parallelism: int = 4
    checkpoint_interval: int = 60000  # 1 minute
    checkpoint_timeout: int = 600000  # 10 minutes
    
    # Memory settings
    task_manager_memory: str = "2g"
    job_manager_memory: str = "1g"
    
    # State backend settings
    state_backend: str = "rocksdb"
    state_backend_path: str = "file:///tmp/flink/checkpoints"
    
    # Watermark settings
    watermark_interval: int = 1000  # 1 second
    max_out_of_orderness: int = 5000  # 5 seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Flink configuration to dictionary"""
        return {
            "jobmanager.rpc.address": self.job_manager_host,
            "jobmanager.rpc.port": str(self.job_manager_port),
            "taskmanager.host": self.task_manager_host,
            "taskmanager.rpc.port": str(self.task_manager_port),
            "jobmanager.memory.process.size": self.job_manager_memory,
            "taskmanager.memory.process.size": self.task_manager_memory,
            "state.backend": self.state_backend,
            "state.checkpoints.dir": self.state_backend_path,
            "execution.checkpointing.interval": str(self.checkpoint_interval),
            "execution.checkpointing.timeout": str(self.checkpoint_timeout),
            "pipeline.auto-watermark-interval": str(self.watermark_interval)
        }


@dataclass
class StreamingJobConfig:
    """Streaming job configuration for PBF-LB/M processing"""
    
    # Job identification
    job_name: str
    job_type: str  # "kafka_streams", "flink", "spark_streaming"
    enabled: bool = True
    
    # Processing settings
    parallelism: int = 4
    checkpoint_interval: int = 60000
    watermark_interval: int = 1000
    
    # Input settings
    input_topics: List[str] = None
    input_config: Dict[str, Any] = None
    
    # Output settings
    output_topics: List[str] = None
    output_config: Dict[str, Any] = None
    
    # Transformation settings
    transformation_type: str = "standard"  # "standard", "custom", "ml"
    transformation_config: Dict[str, Any] = None
    
    # Window settings
    window_type: str = "tumbling"  # "tumbling", "sliding", "session"
    window_size: int = 60000  # 1 minute
    window_slide: int = 30000  # 30 seconds
    
    def __post_init__(self):
        """Initialize default configurations if not provided"""
        if self.input_topics is None:
            self.input_topics = []
        if self.input_config is None:
            self.input_config = {}
        if self.output_topics is None:
            self.output_topics = []
        if self.output_config is None:
            self.output_config = {}
        if self.transformation_config is None:
            self.transformation_config = {}


class StreamingConfig:
    """Streaming configuration manager for PBF-LB/M data processing"""
    
    def __init__(self):
        self.kafka_config = KafkaConfig()
        self.flink_config = FlinkConfig()
        self.streaming_jobs: Dict[str, StreamingJobConfig] = {}
        self._load_default_configurations()
    
    def _load_default_configurations(self):
        """Load default streaming configurations"""
        # Default Kafka configuration
        self.kafka_config = KafkaConfig()
        
        # Default Flink configuration
        self.flink_config = FlinkConfig()
        
        # Default streaming jobs
        self.streaming_jobs = {
            "ispm_stream_processor": StreamingJobConfig(
                job_name="ISPM Stream Processor",
                job_type="kafka_streams",
                input_topics=["ispm_monitoring"],
                output_topics=["ispm_processed"],
                transformation_type="standard",
                window_type="tumbling",
                window_size=60000
            ),
            "powder_bed_stream_processor": StreamingJobConfig(
                job_name="Powder Bed Stream Processor",
                job_type="kafka_streams",
                input_topics=["powder_bed"],
                output_topics=["powder_bed_processed"],
                transformation_type="standard",
                window_type="tumbling",
                window_size=30000
            ),
            "pbf_process_stream_processor": StreamingJobConfig(
                job_name="PBF Process Stream Processor",
                job_type="flink",
                input_topics=["pbf_process"],
                output_topics=["pbf_process_processed"],
                transformation_type="custom",
                window_type="sliding",
                window_size=60000,
                window_slide=30000
            ),
            "real_time_quality_monitor": StreamingJobConfig(
                job_name="Real-time Quality Monitor",
                job_type="kafka_streams",
                input_topics=["ispm_monitoring", "powder_bed"],
                output_topics=["quality_alerts"],
                transformation_type="ml",
                window_type="tumbling",
                window_size=10000
            )
        }
    
    @classmethod
    def from_environment(cls) -> 'StreamingConfig':
        """Create streaming configuration from environment variables"""
        config = cls()
        
        # Update Kafka configuration from environment
        config.kafka_config.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        config.kafka_config.security_protocol = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
        
        # Update Flink configuration from environment
        config.flink_config.job_manager_host = os.getenv("FLINK_JOB_MANAGER_HOST", "localhost")
        config.flink_config.job_manager_port = int(os.getenv("FLINK_JOB_MANAGER_PORT", "6123"))
        config.flink_config.parallelism = int(os.getenv("FLINK_PARALLELISM", "4"))
        
        return config
    
    def get_kafka_config(self) -> KafkaConfig:
        """Get Kafka configuration"""
        return self.kafka_config
    
    def get_flink_config(self) -> FlinkConfig:
        """Get Flink configuration"""
        return self.flink_config
    
    def get_streaming_settings(self) -> Dict[str, Any]:
        """Get streaming job settings"""
        return {
            "jobs": {name: {
                "job_name": job.job_name,
                "job_type": job.job_type,
                "enabled": job.enabled,
                "parallelism": job.parallelism,
                "checkpoint_interval": job.checkpoint_interval,
                "watermark_interval": job.watermark_interval,
                "input_topics": job.input_topics,
                "output_topics": job.output_topics,
                "transformation_type": job.transformation_type,
                "window_type": job.window_type,
                "window_size": job.window_size,
                "window_slide": job.window_slide
            } for name, job in self.streaming_jobs.items()}
        }
    
    def get_streaming_job(self, job_name: str) -> Optional[StreamingJobConfig]:
        """Get streaming job configuration by name"""
        return self.streaming_jobs.get(job_name)
    
    def get_all_streaming_jobs(self) -> Dict[str, StreamingJobConfig]:
        """Get all streaming job configurations"""
        return self.streaming_jobs.copy()
    
    def add_streaming_job(self, job_name: str, job_config: StreamingJobConfig) -> None:
        """Add a new streaming job configuration"""
        self.streaming_jobs[job_name] = job_config
    
    def update_kafka_config(self, **kwargs) -> None:
        """Update Kafka configuration"""
        for key, value in kwargs.items():
            if hasattr(self.kafka_config, key):
                setattr(self.kafka_config, key, value)
    
    def update_flink_config(self, **kwargs) -> None:
        """Update Flink configuration"""
        for key, value in kwargs.items():
            if hasattr(self.flink_config, key):
                setattr(self.flink_config, key, value)


# Global configuration instance
_streaming_config: Optional[StreamingConfig] = None


def get_streaming_config() -> StreamingConfig:
    """
    Get the global streaming configuration instance.
    
    Returns:
        StreamingConfig: The global streaming configuration
    """
    global _streaming_config
    if _streaming_config is None:
        _streaming_config = StreamingConfig.from_environment()
    return _streaming_config


def set_streaming_config(config: StreamingConfig) -> None:
    """
    Set the global streaming configuration instance.
    
    Args:
        config: The streaming configuration to set
    """
    global _streaming_config
    _streaming_config = config


def reset_streaming_config() -> None:
    """Reset the global streaming configuration to None."""
    global _streaming_config
    _streaming_config = None


def load_streaming_config(config_path: Optional[str] = None) -> StreamingConfig:
    """
    Load streaming configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        StreamingConfig: Loaded streaming configuration
    """
    # For now, just return from environment
    # TODO: Add file-based configuration loading
    return StreamingConfig.from_environment()


def get_kafka_config() -> KafkaConfig:
    """
    Get Kafka configuration from the global streaming config.
    
    Returns:
        KafkaConfig: The Kafka configuration
    """
    return get_streaming_config().get_kafka_config()