"""
Remediation Service for PBF-LB/M Data Pipeline

This module provides a unified remediation service that orchestrates
data quality remediation activities across the PBF-LB/M data pipeline.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.data_pipeline.quality.remediation.remediation_engine import RemediationEngine
from src.data_pipeline.quality.remediation.data_cleanser import DataCleanser
from src.data_pipeline.quality.remediation.quality_router import QualityRouter
from src.data_pipeline.quality.remediation.dead_letter_queue import DeadLetterQueue
from src.data_pipeline.quality.validation.data_quality_service import DataQualityService
from src.data_pipeline.config.quality_config import get_quality_config

logger = logging.getLogger(__name__)


@dataclass
class RemediationConfig:
    """Configuration for remediation service."""
    auto_remediate: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 60
    enable_dead_letter_queue: bool = True
    remediation_timeout_seconds: int = 300
    batch_size: int = 100
    parallel_workers: int = 4
    quality_threshold: float = 0.8
    enable_notifications: bool = True


@dataclass
class RemediationResult:
    """Result of a remediation operation."""
    success: bool
    records_processed: int
    records_remediated: int
    records_failed: int
    remediation_time_seconds: float
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class RemediationService:
    """
    Unified remediation service for PBF-LB/M data pipeline.
    
    This service orchestrates data quality remediation activities,
    including automatic remediation, manual intervention, and
    dead letter queue management.
    """
    
    def __init__(self, config: Optional[RemediationConfig] = None):
        """
        Initialize the remediation service.
        
        Args:
            config: Remediation configuration
        """
        self.config = config or self._load_default_config()
        self.remediation_engine = RemediationEngine()
        self.data_cleanser = DataCleanser()
        self.quality_router = QualityRouter()
        self.dead_letter_queue = DeadLetterQueue() if self.config.enable_dead_letter_queue else None
        self.quality_service = DataQualityService()
        
        logger.info("Remediation Service initialized")
    
    def _load_default_config(self) -> RemediationConfig:
        """Load default configuration from environment."""
        quality_config = get_quality_config()
        
        return RemediationConfig(
            auto_remediate=quality_config.auto_remediate,
            max_retry_attempts=quality_config.max_retry_attempts,
            retry_delay_seconds=quality_config.retry_delay_seconds,
            enable_dead_letter_queue=quality_config.enable_dead_letter_queue,
            remediation_timeout_seconds=quality_config.remediation_timeout_seconds,
            batch_size=quality_config.batch_size,
            parallel_workers=quality_config.parallel_workers,
            quality_threshold=quality_config.quality_threshold,
            enable_notifications=quality_config.enable_notifications
        )
    
    def remediate_data_quality_issues(self, data: List[Dict[str, Any]], 
                                    quality_issues: List[Dict[str, Any]]) -> RemediationResult:
        """
        Remediate data quality issues in a batch of data.
        
        Args:
            data: List of data records
            quality_issues: List of quality issues to remediate
            
        Returns:
            RemediationResult: Result of the remediation operation
        """
        start_time = datetime.now()
        errors = []
        warnings = []
        remediated_count = 0
        failed_count = 0
        
        try:
            logger.info(f"Starting remediation for {len(data)} records with {len(quality_issues)} issues")
            
            # Group issues by type for efficient processing
            issues_by_type = self._group_issues_by_type(quality_issues)
            
            # Process each type of issue
            for issue_type, issues in issues_by_type.items():
                try:
                    result = self._remediate_issue_type(data, issues, issue_type)
                    remediated_count += result['remediated']
                    failed_count += result['failed']
                    errors.extend(result.get('errors', []))
                    warnings.extend(result.get('warnings', []))
                    
                except Exception as e:
                    error_msg = f"Error remediating {issue_type} issues: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    failed_count += len(issues)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = RemediationResult(
                success=len(errors) == 0,
                records_processed=len(data),
                records_remediated=remediated_count,
                records_failed=failed_count,
                remediation_time_seconds=processing_time,
                errors=errors,
                warnings=warnings,
                metadata={
                    'issue_types_processed': list(issues_by_type.keys()),
                    'total_issues': len(quality_issues),
                    'remediation_config': self.config.__dict__
                }
            )
            
            logger.info(f"Remediation completed: {remediated_count} remediated, {failed_count} failed")
            return result
            
        except Exception as e:
            error_msg = f"Critical error in remediation service: {e}"
            logger.error(error_msg)
            return RemediationResult(
                success=False,
                records_processed=len(data),
                records_remediated=0,
                records_failed=len(data),
                remediation_time_seconds=(datetime.now() - start_time).total_seconds(),
                errors=[error_msg],
                warnings=warnings
            )
    
    def _group_issues_by_type(self, quality_issues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group quality issues by their type."""
        issues_by_type = {}
        
        for issue in quality_issues:
            issue_type = issue.get('issue_type', 'unknown')
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        return issues_by_type
    
    def _remediate_issue_type(self, data: List[Dict[str, Any]], 
                            issues: List[Dict[str, Any]], 
                            issue_type: str) -> Dict[str, Any]:
        """Remediate a specific type of quality issue."""
        remediated = 0
        failed = 0
        errors = []
        warnings = []
        
        try:
            if issue_type == 'missing_values':
                result = self.data_cleanser.handle_missing_values(data, issues)
                remediated = result['remediated']
                failed = result['failed']
                warnings.extend(result.get('warnings', []))
                
            elif issue_type == 'data_type_mismatch':
                result = self.data_cleanser.handle_data_type_mismatches(data, issues)
                remediated = result['remediated']
                failed = result['failed']
                warnings.extend(result.get('warnings', []))
                
            elif issue_type == 'out_of_range':
                result = self.data_cleanser.handle_out_of_range_values(data, issues)
                remediated = result['remediated']
                failed = result['failed']
                warnings.extend(result.get('warnings', []))
                
            elif issue_type == 'duplicate_records':
                result = self.data_cleanser.handle_duplicate_records(data, issues)
                remediated = result['remediated']
                failed = result['failed']
                warnings.extend(result.get('warnings', []))
                
            elif issue_type == 'business_rule_violation':
                result = self.remediation_engine.remediate_business_rules(data, issues)
                remediated = result['remediated']
                failed = result['failed']
                errors.extend(result.get('errors', []))
                
            else:
                # Unknown issue type - try generic remediation
                result = self.remediation_engine.generic_remediation(data, issues)
                remediated = result['remediated']
                failed = result['failed']
                warnings.append(f"Used generic remediation for unknown issue type: {issue_type}")
            
            return {
                'remediated': remediated,
                'failed': failed,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            error_msg = f"Error remediating {issue_type} issues: {e}"
            logger.error(error_msg)
            return {
                'remediated': 0,
                'failed': len(issues),
                'errors': [error_msg],
                'warnings': warnings
            }
    
    def auto_remediate(self, data: List[Dict[str, Any]]) -> RemediationResult:
        """
        Automatically detect and remediate quality issues.
        
        Args:
            data: List of data records to remediate
            
        Returns:
            RemediationResult: Result of the auto-remediation
        """
        try:
            if not self.config.auto_remediate:
                logger.info("Auto-remediation is disabled")
                return RemediationResult(
                    success=True,
                    records_processed=len(data),
                    records_remediated=0,
                    records_failed=0,
                    remediation_time_seconds=0.0,
                    warnings=["Auto-remediation is disabled"]
                )
            
            # Detect quality issues
            quality_issues = self.quality_service.detect_quality_issues(data)
            
            if not quality_issues:
                logger.info("No quality issues detected")
                return RemediationResult(
                    success=True,
                    records_processed=len(data),
                    records_remediated=0,
                    records_failed=0,
                    remediation_time_seconds=0.0
                )
            
            # Remediate the issues
            return self.remediate_data_quality_issues(data, quality_issues)
            
        except Exception as e:
            error_msg = f"Error in auto-remediation: {e}"
            logger.error(error_msg)
            return RemediationResult(
                success=False,
                records_processed=len(data),
                records_remediated=0,
                records_failed=len(data),
                remediation_time_seconds=0.0,
                errors=[error_msg]
            )
    
    def route_to_dead_letter_queue(self, data: List[Dict[str, Any]], 
                                 reason: str) -> bool:
        """
        Route data to dead letter queue for manual review.
        
        Args:
            data: Data records to route
            reason: Reason for routing to DLQ
            
        Returns:
            bool: True if successfully routed
        """
        try:
            if not self.dead_letter_queue:
                logger.warning("Dead letter queue is not enabled")
                return False
            
            return self.dead_letter_queue.add_records(data, reason)
            
        except Exception as e:
            logger.error(f"Error routing to dead letter queue: {e}")
            return False
    
    def get_remediation_metrics(self) -> Dict[str, Any]:
        """
        Get remediation service metrics.
        
        Returns:
            Dictionary containing remediation metrics
        """
        try:
            metrics = {
                'service_status': 'active',
                'config': self.config.__dict__,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get metrics from components
            if self.dead_letter_queue:
                dlq_metrics = self.dead_letter_queue.get_metrics()
                metrics['dead_letter_queue'] = dlq_metrics
            
            remediation_metrics = self.remediation_engine.get_metrics()
            metrics['remediation_engine'] = remediation_metrics
            
            cleanser_metrics = self.data_cleanser.get_metrics()
            metrics['data_cleanser'] = cleanser_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting remediation metrics: {e}")
            return {'error': str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the remediation service.
        
        Returns:
            Health status dictionary
        """
        try:
            status = {
                'status': 'healthy',
                'components': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Check component health
            try:
                engine_health = self.remediation_engine.get_health_status()
                status['components']['remediation_engine'] = engine_health
            except Exception as e:
                status['components']['remediation_engine'] = {'status': 'error', 'error': str(e)}
            
            try:
                cleanser_health = self.data_cleanser.get_health_status()
                status['components']['data_cleanser'] = cleanser_health
            except Exception as e:
                status['components']['data_cleanser'] = {'status': 'error', 'error': str(e)}
            
            if self.dead_letter_queue:
                try:
                    dlq_health = self.dead_letter_queue.get_health_status()
                    status['components']['dead_letter_queue'] = dlq_health
                except Exception as e:
                    status['components']['dead_letter_queue'] = {'status': 'error', 'error': str(e)}
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Convenience functions for common operations
def create_remediation_service(**kwargs) -> RemediationService:
    """
    Create a remediation service with custom configuration.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Configured RemediationService instance
    """
    config = RemediationConfig(**kwargs)
    return RemediationService(config)


def auto_remediate_data(data: List[Dict[str, Any]], **kwargs) -> RemediationResult:
    """
    Convenience function for auto-remediation.
    
    Args:
        data: Data records to remediate
        **kwargs: Additional configuration parameters
        
    Returns:
        RemediationResult: Result of the remediation
    """
    service = create_remediation_service(**kwargs)
    return service.auto_remediate(data)
