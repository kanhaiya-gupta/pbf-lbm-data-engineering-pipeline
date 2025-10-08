"""
Streaming Processor for PBF-LB/M Data Pipeline

This module provides a unified streaming processing interface that orchestrates
real-time data processing for PBF-LB/M manufacturing data.
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.data_pipeline.processing.streaming.kafka_streams_processor import KafkaStreamsProcessor
from src.data_pipeline.processing.streaming.flink_processor import FlinkProcessor
from src.data_pipeline.processing.streaming.real_time_transformer import RealTimeTransformer
from src.data_pipeline.processing.streaming.ispm_stream_joins import ISPMStreamJoins
from src.data_pipeline.processing.streaming.powder_bed_stream_joins import PowderBedStreamJoins
from src.data_pipeline.processing.streaming.stream_sink_manager import StreamSinkManager
from src.data_pipeline.config.streaming_config import get_streaming_config

logger = logging.getLogger(__name__)


@dataclass
class StreamingProcessorConfig:
    """Configuration for streaming processor."""
    processor_type: str = 'kafka_streams'  # kafka_streams, flink, or custom
    input_topics: List[str] = None
    output_topics: List[str] = None
    processing_mode: str = 'real_time'  # real_time, batch, or hybrid
    parallelism: int = 1
    checkpoint_interval: int = 60000  # milliseconds
    state_store_config: Dict[str, Any] = None
    sink_configs: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.input_topics is None:
            self.input_topics = []
        if self.output_topics is None:
            self.output_topics = []
        if self.state_store_config is None:
            self.state_store_config = {}
        if self.sink_configs is None:
            self.sink_configs = []


class StreamingProcessor(ABC):
    """
    Abstract base class for streaming processors.
    
    This class defines the interface that all streaming processors must implement
    for the PBF-LB/M data pipeline.
    """
    
    @abstractmethod
    def start(self) -> bool:
        """Start the streaming processor."""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop the streaming processor."""
        pass
    
    @abstractmethod
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data record."""
        pass
    
    @abstractmethod
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics and statistics."""
        pass


class UnifiedStreamingProcessor:
    """
    Unified streaming processor that orchestrates multiple streaming components.
    
    This class provides a high-level interface for streaming data processing
    in the PBF-LB/M data pipeline, coordinating between different processing
    engines and data sinks.
    """
    
    def __init__(self, config: Optional[StreamingProcessorConfig] = None):
        """
        Initialize the unified streaming processor.
        
        Args:
            config: Streaming processor configuration
        """
        self.config = config or self._load_default_config()
        self.processors = {}
        self.transformers = {}
        self.sink_manager = None
        self.is_running = False
        
        self._initialize_components()
        
        logger.info(f"Unified Streaming Processor initialized with type: {self.config.processor_type}")
    
    def _load_default_config(self) -> StreamingProcessorConfig:
        """Load default configuration from environment."""
        streaming_config = get_streaming_config()
        
        return StreamingProcessorConfig(
            processor_type=streaming_config.processing_engine,
            input_topics=streaming_config.input_topics,
            output_topics=streaming_config.output_topics,
            processing_mode=streaming_config.processing_mode,
            parallelism=streaming_config.parallelism,
            checkpoint_interval=streaming_config.checkpoint_interval,
            state_store_config=streaming_config.state_store_config,
            sink_configs=streaming_config.sink_configs
        )
    
    def _initialize_components(self):
        """Initialize streaming processing components."""
        try:
            # Initialize main processor based on type
            if self.config.processor_type == 'kafka_streams':
                self.processors['main'] = KafkaStreamsProcessor(
                    input_topics=self.config.input_topics,
                    output_topics=self.config.output_topics,
                    parallelism=self.config.parallelism
                )
            elif self.config.processor_type == 'flink':
                self.processors['main'] = FlinkProcessor(
                    input_topics=self.config.input_topics,
                    output_topics=self.config.output_topics,
                    parallelism=self.config.parallelism
                )
            else:
                raise ValueError(f"Unsupported processor type: {self.config.processor_type}")
            
            # Initialize transformers
            self.transformers['real_time'] = RealTimeTransformer()
            self.transformers['ispm_joins'] = ISPMStreamJoins()
            self.transformers['powder_bed_joins'] = PowderBedStreamJoins()
            
            # Initialize sink manager
            self.sink_manager = StreamSinkManager()
            
            logger.info("Streaming processing components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing streaming components: {e}")
            raise
    
    def start(self) -> bool:
        """
        Start the streaming processor.
        
        Returns:
            bool: True if started successfully
        """
        try:
            if self.is_running:
                logger.warning("Streaming processor is already running")
                return True
            
            # Start main processor
            if 'main' in self.processors:
                success = self.processors['main'].start()
                if not success:
                    logger.error("Failed to start main processor")
                    return False
            
            # Start sink manager
            if self.sink_manager:
                self.sink_manager.start()
            
            self.is_running = True
            logger.info("Streaming processor started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting streaming processor: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the streaming processor.
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            if not self.is_running:
                logger.warning("Streaming processor is not running")
                return True
            
            # Stop main processor
            if 'main' in self.processors:
                self.processors['main'].stop()
            
            # Stop sink manager
            if self.sink_manager:
                self.sink_manager.stop()
            
            self.is_running = False
            logger.info("Streaming processor stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping streaming processor: {e}")
            return False
    
    def process_data(self, data: Dict[str, Any], data_type: str = 'generic') -> Dict[str, Any]:
        """
        Process streaming data through the pipeline.
        
        Args:
            data: Input data record
            data_type: Type of data (ispm, powder_bed, pbf_process, ct_scan)
            
        Returns:
            Processed data record
        """
        try:
            if not self.is_running:
                logger.warning("Streaming processor is not running")
                return data
            
            # Apply real-time transformation
            transformed_data = self.transformers['real_time'].transform(data)
            
            # Apply data-type specific joins and transformations
            if data_type == 'ispm':
                processed_data = self.transformers['ispm_joins'].process_ispm_data(transformed_data)
            elif data_type == 'powder_bed':
                processed_data = self.transformers['powder_bed_joins'].process_powder_bed_data(transformed_data)
            else:
                processed_data = transformed_data
            
            # Add processing metadata
            processed_data['processing_metadata'] = {
                'processed_at': datetime.now().isoformat(),
                'processor_type': self.config.processor_type,
                'data_type': data_type,
                'pipeline_version': '1.0'
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return data
    
    def process_ispm_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process ISPM monitoring data.
        
        Args:
            data: ISPM monitoring data
            
        Returns:
            Processed ISPM data
        """
        return self.process_data(data, 'ispm')
    
    def process_powder_bed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process powder bed monitoring data.
        
        Args:
            data: Powder bed monitoring data
            
        Returns:
            Processed powder bed data
        """
        return self.process_data(data, 'powder_bed')
    
    def process_pbf_process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process PBF process data.
        
        Args:
            data: PBF process data
            
        Returns:
            Processed PBF process data
        """
        return self.process_data(data, 'pbf_process')
    
    def process_ct_scan_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process CT scan data.
        
        Args:
            data: CT scan data
            
        Returns:
            Processed CT scan data
        """
        return self.process_data(data, 'ct_scan')
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """
        Get processing metrics and statistics.
        
        Returns:
            Dictionary containing processing metrics
        """
        try:
            metrics = {
                'is_running': self.is_running,
                'processor_type': self.config.processor_type,
                'processing_mode': self.config.processing_mode,
                'parallelism': self.config.parallelism,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get metrics from main processor
            if 'main' in self.processors:
                processor_metrics = self.processors['main'].get_processing_metrics()
                metrics['main_processor'] = processor_metrics
            
            # Get metrics from sink manager
            if self.sink_manager:
                sink_metrics = self.sink_manager.get_sink_metrics()
                metrics['sink_manager'] = sink_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting processing metrics: {e}")
            return {'error': str(e)}
    
    def add_custom_transformer(self, name: str, transformer: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Add a custom data transformer.
        
        Args:
            name: Name of the transformer
            transformer: Transformer function
        """
        self.transformers[name] = transformer
        logger.info(f"Custom transformer '{name}' added")
    
    def configure_sink(self, sink_config: Dict[str, Any]) -> bool:
        """
        Configure a data sink.
        
        Args:
            sink_config: Sink configuration
            
        Returns:
            bool: True if configured successfully
        """
        try:
            if self.sink_manager:
                return self.sink_manager.add_sink(sink_config)
            return False
            
        except Exception as e:
            logger.error(f"Error configuring sink: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the streaming processor.
        
        Returns:
            Health status dictionary
        """
        try:
            status = {
                'status': 'healthy' if self.is_running else 'stopped',
                'is_running': self.is_running,
                'components': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Check main processor health
            if 'main' in self.processors:
                processor_health = self.processors['main'].get_health_status()
                status['components']['main_processor'] = processor_health
            
            # Check sink manager health
            if self.sink_manager:
                sink_health = self.sink_manager.get_health_status()
                status['components']['sink_manager'] = sink_health
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Convenience functions for common operations
def create_streaming_processor(processor_type: str = 'kafka_streams', **kwargs) -> UnifiedStreamingProcessor:
    """
    Create a streaming processor with custom configuration.
    
    Args:
        processor_type: Type of processor (kafka_streams, flink)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured UnifiedStreamingProcessor instance
    """
    config = StreamingProcessorConfig(
        processor_type=processor_type,
        input_topics=kwargs.get('input_topics', []),
        output_topics=kwargs.get('output_topics', []),
        processing_mode=kwargs.get('processing_mode', 'real_time'),
        parallelism=kwargs.get('parallelism', 1),
        checkpoint_interval=kwargs.get('checkpoint_interval', 60000),
        state_store_config=kwargs.get('state_store_config', {}),
        sink_configs=kwargs.get('sink_configs', [])
    )
    
    return UnifiedStreamingProcessor(config)


def process_streaming_data(data: Dict[str, Any], data_type: str = 'generic', **kwargs) -> Dict[str, Any]:
    """
    Process streaming data using a temporary processor.
    
    Args:
        data: Input data record
        data_type: Type of data
        **kwargs: Additional configuration parameters
        
    Returns:
        Processed data record
    """
    with create_streaming_processor(**kwargs) as processor:
        return processor.process_data(data, data_type)
