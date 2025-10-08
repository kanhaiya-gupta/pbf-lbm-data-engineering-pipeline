"""
Dead Letter Queue

This module provides dead letter queue capabilities for handling failed data in the PBF-LB/M data pipeline.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import threading
import time

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeadLetterStatus(Enum):
    """Dead letter status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    RETRYING = "retrying"
    RESOLVED = "resolved"
    FAILED = "failed"
    EXPIRED = "expired"

class DeadLetterPriority(Enum):
    """Dead letter priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DeadLetterAction(Enum):
    """Dead letter action enumeration."""
    RETRY = "retry"
    MANUAL_REVIEW = "manual_review"
    DISCARD = "discard"
    ARCHIVE = "archive"
    ESCALATE = "escalate"

@dataclass
class DeadLetterRecord:
    """Dead letter record data class."""
    id: str
    source_name: str
    data: Dict[str, Any]
    error_message: str
    error_type: str
    status: DeadLetterStatus
    priority: DeadLetterPriority
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_retry_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeadLetterQueueStats:
    """Dead letter queue statistics data class."""
    total_records: int
    pending_records: int
    processing_records: int
    retrying_records: int
    resolved_records: int
    failed_records: int
    expired_records: int
    average_retry_count: float
    oldest_record_age_hours: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class DeadLetterRetryPolicy:
    """Dead letter retry policy data class."""
    max_retries: int
    retry_delay_seconds: int
    exponential_backoff: bool
    max_retry_delay_seconds: int
    retry_conditions: List[str] = field(default_factory=list)

class DeadLetterQueue:
    """
    Dead letter queue service for handling failed data in PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.dead_letter_records: Dict[str, DeadLetterRecord] = {}
        self.retry_policies: Dict[str, DeadLetterRetryPolicy] = {}
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        self.processing_interval = 60  # seconds
        self.retry_lock = threading.Lock()
        
        # Initialize retry policies
        self._initialize_retry_policies()
        
    def add_failed_record(self, source_name: str, data: Dict[str, Any], 
                         error_message: str, error_type: str = "unknown",
                         priority: DeadLetterPriority = DeadLetterPriority.MEDIUM,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a failed record to the dead letter queue.
        
        Args:
            source_name: The data source name
            data: The failed data record
            error_message: The error message
            error_type: The type of error
            priority: The priority level
            metadata: Additional metadata
            
        Returns:
            str: The dead letter record ID
        """
        try:
            record_id = str(uuid.uuid4())
            
            # Create dead letter record
            record = DeadLetterRecord(
                id=record_id,
                source_name=source_name,
                data=data,
                error_message=error_message,
                error_type=error_type,
                status=DeadLetterStatus.PENDING,
                priority=priority,
                metadata=metadata or {}
            )
            
            # Store record
            self.dead_letter_records[record_id] = record
            
            logger.info(f"Added failed record to dead letter queue: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Error adding failed record to dead letter queue: {e}")
            raise
    
    def get_dead_letter_record(self, record_id: str) -> Optional[DeadLetterRecord]:
        """
        Get a dead letter record by ID.
        
        Args:
            record_id: The record ID
            
        Returns:
            DeadLetterRecord: The dead letter record, or None if not found
        """
        return self.dead_letter_records.get(record_id)
    
    def get_pending_records(self, source_name: Optional[str] = None, 
                           priority: Optional[DeadLetterPriority] = None,
                           limit: int = 100) -> List[DeadLetterRecord]:
        """
        Get pending dead letter records.
        
        Args:
            source_name: Filter by source name
            priority: Filter by priority
            limit: Maximum number of records to return
            
        Returns:
            List[DeadLetterRecord]: List of pending records
        """
        try:
            pending_records = []
            
            for record in self.dead_letter_records.values():
                if record.status == DeadLetterStatus.PENDING:
                    # Apply filters
                    if source_name and record.source_name != source_name:
                        continue
                    if priority and record.priority != priority:
                        continue
                    
                    pending_records.append(record)
                    
                    if len(pending_records) >= limit:
                        break
            
            # Sort by priority and creation time
            priority_order = {
                DeadLetterPriority.CRITICAL: 0,
                DeadLetterPriority.HIGH: 1,
                DeadLetterPriority.MEDIUM: 2,
                DeadLetterPriority.LOW: 3
            }
            
            pending_records.sort(key=lambda r: (priority_order.get(r.priority, 4), r.created_at))
            
            return pending_records
            
        except Exception as e:
            logger.error(f"Error getting pending records: {e}")
            return []
    
    def retry_record(self, record_id: str) -> bool:
        """
        Retry a dead letter record.
        
        Args:
            record_id: The record ID
            
        Returns:
            bool: True if retry was successful, False otherwise
        """
        try:
            with self.retry_lock:
                record = self.dead_letter_records.get(record_id)
                if not record:
                    logger.warning(f"Dead letter record not found: {record_id}")
                    return False
                
                # Check if record can be retried
                if record.status not in [DeadLetterStatus.PENDING, DeadLetterStatus.RETRYING]:
                    logger.warning(f"Record {record_id} cannot be retried in status: {record.status}")
                    return False
                
                if record.retry_count >= record.max_retries:
                    logger.warning(f"Record {record_id} has exceeded max retries")
                    record.status = DeadLetterStatus.FAILED
                    return False
                
                # Update record status
                record.status = DeadLetterStatus.RETRYING
                record.retry_count += 1
                record.last_retry_at = datetime.now()
                record.updated_at = datetime.now()
                
                logger.info(f"Retrying dead letter record: {record_id} (attempt {record.retry_count})")
                
                # Simulate retry processing
                # In a real system, this would attempt to reprocess the data
                success = self._process_retry(record)
                
                if success:
                    record.status = DeadLetterStatus.RESOLVED
                    record.resolved_at = datetime.now()
                    logger.info(f"Dead letter record resolved: {record_id}")
                else:
                    record.status = DeadLetterStatus.PENDING
                    logger.warning(f"Dead letter record retry failed: {record_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Error retrying record {record_id}: {e}")
            return False
    
    def resolve_record(self, record_id: str, action: DeadLetterAction, 
                      resolution_notes: Optional[str] = None) -> bool:
        """
        Resolve a dead letter record.
        
        Args:
            record_id: The record ID
            action: The resolution action
            resolution_notes: Notes about the resolution
            
        Returns:
            bool: True if resolution was successful, False otherwise
        """
        try:
            record = self.dead_letter_records.get(record_id)
            if not record:
                logger.warning(f"Dead letter record not found: {record_id}")
                return False
            
            # Update record
            record.status = DeadLetterStatus.RESOLVED
            record.resolved_at = datetime.now()
            record.updated_at = datetime.now()
            record.metadata["resolution_action"] = action.value
            record.metadata["resolution_notes"] = resolution_notes
            record.metadata["resolved_by"] = "system"  # In real system, this would be user ID
            
            logger.info(f"Resolved dead letter record: {record_id} with action: {action.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving record {record_id}: {e}")
            return False
    
    def discard_record(self, record_id: str, reason: str) -> bool:
        """
        Discard a dead letter record.
        
        Args:
            record_id: The record ID
            reason: Reason for discarding
            
        Returns:
            bool: True if discard was successful, False otherwise
        """
        try:
            record = self.dead_letter_records.get(record_id)
            if not record:
                logger.warning(f"Dead letter record not found: {record_id}")
                return False
            
            # Update record
            record.status = DeadLetterStatus.FAILED
            record.updated_at = datetime.now()
            record.metadata["discard_reason"] = reason
            record.metadata["discarded_at"] = datetime.now().isoformat()
            
            logger.info(f"Discarded dead letter record: {record_id} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error discarding record {record_id}: {e}")
            return False
    
    def start_processing(self) -> bool:
        """
        Start the dead letter queue processing thread.
        
        Returns:
            bool: True if processing started successfully, False otherwise
        """
        try:
            if self.is_processing:
                logger.warning("Dead letter queue processing is already running")
                return False
            
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("Dead letter queue processing started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting dead letter queue processing: {e}")
            return False
    
    def stop_processing(self) -> bool:
        """
        Stop the dead letter queue processing thread.
        
        Returns:
            bool: True if processing stopped successfully, False otherwise
        """
        try:
            if not self.is_processing:
                logger.warning("Dead letter queue processing is not running")
                return False
            
            self.is_processing = False
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=10)
            
            logger.info("Dead letter queue processing stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping dead letter queue processing: {e}")
            return False
    
    def get_queue_stats(self) -> DeadLetterQueueStats:
        """
        Get dead letter queue statistics.
        
        Returns:
            DeadLetterQueueStats: Queue statistics
        """
        try:
            total_records = len(self.dead_letter_records)
            pending_records = len([r for r in self.dead_letter_records.values() 
                                 if r.status == DeadLetterStatus.PENDING])
            processing_records = len([r for r in self.dead_letter_records.values() 
                                    if r.status == DeadLetterStatus.PROCESSING])
            retrying_records = len([r for r in self.dead_letter_records.values() 
                                  if r.status == DeadLetterStatus.RETRYING])
            resolved_records = len([r for r in self.dead_letter_records.values() 
                                  if r.status == DeadLetterStatus.RESOLVED])
            failed_records = len([r for r in self.dead_letter_records.values() 
                                if r.status == DeadLetterStatus.FAILED])
            expired_records = len([r for r in self.dead_letter_records.values() 
                                 if r.status == DeadLetterStatus.EXPIRED])
            
            # Calculate average retry count
            if total_records > 0:
                average_retry_count = sum(r.retry_count for r in self.dead_letter_records.values()) / total_records
            else:
                average_retry_count = 0.0
            
            # Calculate oldest record age
            if self.dead_letter_records:
                oldest_record = min(self.dead_letter_records.values(), key=lambda r: r.created_at)
                oldest_record_age = datetime.now() - oldest_record.created_at
                oldest_record_age_hours = oldest_record_age.total_seconds() / 3600
            else:
                oldest_record_age_hours = 0.0
            
            return DeadLetterQueueStats(
                total_records=total_records,
                pending_records=pending_records,
                processing_records=processing_records,
                retrying_records=retrying_records,
                resolved_records=resolved_records,
                failed_records=failed_records,
                expired_records=expired_records,
                average_retry_count=average_retry_count,
                oldest_record_age_hours=oldest_record_age_hours
            )
            
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return DeadLetterQueueStats(
                total_records=0,
                pending_records=0,
                processing_records=0,
                retrying_records=0,
                resolved_records=0,
                failed_records=0,
                expired_records=0,
                average_retry_count=0.0,
                oldest_record_age_hours=0.0
            )
    
    def cleanup_expired_records(self, max_age_hours: int = 168) -> int:
        """
        Clean up expired records from the dead letter queue.
        
        Args:
            max_age_hours: Maximum age in hours before records are considered expired
            
        Returns:
            int: Number of records cleaned up
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            expired_records = []
            
            for record_id, record in self.dead_letter_records.items():
                if (record.status in [DeadLetterStatus.PENDING, DeadLetterStatus.RETRYING] and
                    record.created_at < cutoff_time):
                    record.status = DeadLetterStatus.EXPIRED
                    record.updated_at = datetime.now()
                    expired_records.append(record_id)
            
            logger.info(f"Cleaned up {len(expired_records)} expired records")
            return len(expired_records)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired records: {e}")
            return 0
    
    def export_dead_letter_data(self, record_ids: Optional[List[str]] = None) -> str:
        """
        Export dead letter data as JSON.
        
        Args:
            record_ids: List of record IDs to export, or None for all records
            
        Returns:
            str: JSON representation of the dead letter data
        """
        try:
            if record_ids:
                records_to_export = {rid: self.dead_letter_records[rid] 
                                   for rid in record_ids 
                                   if rid in self.dead_letter_records}
            else:
                records_to_export = self.dead_letter_records
            
            # Convert to dictionary and handle datetime serialization
            export_data = {}
            for record_id, record in records_to_export.items():
                record_dict = asdict(record)
                
                # Convert datetime objects to ISO strings
                for key, value in record_dict.items():
                    if isinstance(value, datetime):
                        record_dict[key] = value.isoformat()
                
                export_data[record_id] = record_dict
            
            return json.dumps(export_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting dead letter data: {e}")
            return json.dumps({"error": str(e)})
    
    def import_dead_letter_data(self, json_data: str) -> int:
        """
        Import dead letter data from JSON.
        
        Args:
            json_data: JSON representation of dead letter data
            
        Returns:
            int: Number of records imported
        """
        try:
            import_data = json.loads(json_data)
            imported_count = 0
            
            for record_id, record_dict in import_data.items():
                try:
                    # Convert ISO strings back to datetime objects
                    for key, value in record_dict.items():
                        if isinstance(value, str) and key.endswith('_at'):
                            try:
                                record_dict[key] = datetime.fromisoformat(value)
                            except ValueError:
                                pass
                    
                    # Create record
                    record = DeadLetterRecord(**record_dict)
                    self.dead_letter_records[record_id] = record
                    imported_count += 1
                    
                except Exception as e:
                    logger.error(f"Error importing record {record_id}: {e}")
            
            logger.info(f"Imported {imported_count} dead letter records")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing dead letter data: {e}")
            return 0
    
    def _processing_loop(self):
        """Main processing loop for dead letter queue."""
        while self.is_processing:
            try:
                # Process pending records
                self._process_pending_records()
                
                # Clean up expired records
                self.cleanup_expired_records()
                
                # Sleep for processing interval
                time.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in dead letter queue processing loop: {e}")
                time.sleep(self.processing_interval)
    
    def _process_pending_records(self):
        """Process pending dead letter records."""
        try:
            # Get pending records
            pending_records = self.get_pending_records(limit=10)
            
            for record in pending_records:
                try:
                    # Check if record should be retried
                    if self._should_retry_record(record):
                        self.retry_record(record.id)
                    else:
                        # Mark as failed if max retries exceeded
                        if record.retry_count >= record.max_retries:
                            record.status = DeadLetterStatus.FAILED
                            record.updated_at = datetime.now()
                            
                except Exception as e:
                    logger.error(f"Error processing record {record.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing pending records: {e}")
    
    def _should_retry_record(self, record: DeadLetterRecord) -> bool:
        """Check if a record should be retried."""
        try:
            # Check if max retries exceeded
            if record.retry_count >= record.max_retries:
                return False
            
            # Check retry delay
            if record.last_retry_at:
                retry_delay = self._calculate_retry_delay(record)
                time_since_last_retry = datetime.now() - record.last_retry_at
                if time_since_last_retry.total_seconds() < retry_delay:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if record should be retried: {e}")
            return False
    
    def _calculate_retry_delay(self, record: DeadLetterRecord) -> int:
        """Calculate retry delay for a record."""
        try:
            # Get retry policy for the source
            policy = self.retry_policies.get(record.source_name)
            if not policy:
                policy = self.retry_policies.get("default")
            
            if not policy:
                return 60  # Default 1 minute
            
            # Calculate delay with exponential backoff
            if policy.exponential_backoff:
                delay = policy.retry_delay_seconds * (2 ** record.retry_count)
                return min(delay, policy.max_retry_delay_seconds)
            else:
                return policy.retry_delay_seconds
                
        except Exception as e:
            logger.error(f"Error calculating retry delay: {e}")
            return 60
    
    def _process_retry(self, record: DeadLetterRecord) -> bool:
        """Process a retry attempt for a record."""
        try:
            # Simulate retry processing
            # In a real system, this would attempt to reprocess the data
            
            # Check if the error is retryable
            retryable_errors = ["timeout", "connection_error", "temporary_failure"]
            if any(error in record.error_message.lower() for error in retryable_errors):
                # Simulate successful retry
                return True
            else:
                # Simulate failed retry
                return False
                
        except Exception as e:
            logger.error(f"Error processing retry: {e}")
            return False
    
    def _initialize_retry_policies(self):
        """Initialize retry policies for different sources."""
        try:
            # Default retry policy
            self.retry_policies["default"] = DeadLetterRetryPolicy(
                max_retries=3,
                retry_delay_seconds=60,
                exponential_backoff=True,
                max_retry_delay_seconds=3600,
                retry_conditions=["timeout", "connection_error", "temporary_failure"]
            )
            
            # PBF Process retry policy
            self.retry_policies["pbf_process"] = DeadLetterRetryPolicy(
                max_retries=5,
                retry_delay_seconds=30,
                exponential_backoff=True,
                max_retry_delay_seconds=1800,
                retry_conditions=["timeout", "connection_error", "temporary_failure", "validation_error"]
            )
            
            # ISPM Monitoring retry policy
            self.retry_policies["ispm_monitoring"] = DeadLetterRetryPolicy(
                max_retries=3,
                retry_delay_seconds=10,
                exponential_backoff=False,
                max_retry_delay_seconds=300,
                retry_conditions=["timeout", "connection_error"]
            )
            
            # CT Scan retry policy
            self.retry_policies["ct_scan"] = DeadLetterRetryPolicy(
                max_retries=2,
                retry_delay_seconds=120,
                exponential_backoff=True,
                max_retry_delay_seconds=7200,
                retry_conditions=["timeout", "connection_error", "temporary_failure"]
            )
            
            # Powder Bed retry policy
            self.retry_policies["powder_bed"] = DeadLetterRetryPolicy(
                max_retries=3,
                retry_delay_seconds=60,
                exponential_backoff=True,
                max_retry_delay_seconds=1800,
                retry_conditions=["timeout", "connection_error", "temporary_failure"]
            )
            
            logger.info(f"Initialized {len(self.retry_policies)} retry policies")
            
        except Exception as e:
            logger.error(f"Error initializing retry policies: {e}")
