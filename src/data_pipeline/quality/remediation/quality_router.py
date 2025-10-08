"""
Quality Router

This module provides quality-based routing capabilities for the PBF-LB/M data pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from src.data_pipeline.config.pipeline_config import get_pipeline_config
from src.data_pipeline.quality.validation.data_quality_service import QualityResult, QualityProfile
from src.data_pipeline.quality.monitoring.quality_monitor import QualityAlert, QualityStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoutingDecision(Enum):
    """Routing decision enumeration."""
    ACCEPT = "accept"
    REJECT = "reject"
    ROUTE_TO_CLEANSING = "route_to_cleansing"
    ROUTE_TO_REMEDIATION = "route_to_remediation"
    ROUTE_TO_DEAD_LETTER = "route_to_dead_letter"
    ROUTE_TO_MANUAL_REVIEW = "route_to_manual_review"
    ROUTE_TO_ARCHIVE = "route_to_archive"

class RoutingPriority(Enum):
    """Routing priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RoutingStatus(Enum):
    """Routing status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROUTED = "routed"

@dataclass
class RoutingRule:
    """Quality routing rule data class."""
    id: str
    name: str
    description: str
    condition: str
    decision: RoutingDecision
    priority: RoutingPriority
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_types: List[str] = field(default_factory=list)

@dataclass
class RoutingResult:
    """Quality routing result data class."""
    rule_id: str
    source_name: str
    decision: RoutingDecision
    status: RoutingStatus
    records_processed: int
    records_accepted: int
    records_rejected: int
    records_routed_to_cleansing: int
    records_routed_to_remediation: int
    records_routed_to_dead_letter: int
    records_routed_to_manual_review: int
    records_routed_to_archive: int
    routing_reason: str
    quality_score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RoutingJob:
    """Quality routing job data class."""
    job_id: str
    source_name: str
    data: List[Dict[str, Any]]
    quality_results: List[QualityResult]
    rules: List[RoutingRule]
    status: RoutingStatus
    results: List[RoutingResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_records: int = 0
    processed_records: int = 0
    routed_records: int = 0

class QualityRouter:
    """
    Quality-based routing service for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.routing_rules: Dict[str, RoutingRule] = {}
        self.routing_jobs: Dict[str, RoutingJob] = {}
        self.routing_results: Dict[str, List[RoutingResult]] = {}
        
        # Routing destinations
        self.routing_destinations = {
            RoutingDecision.ACCEPT: "data_warehouse",
            RoutingDecision.REJECT: "rejected_data",
            RoutingDecision.ROUTE_TO_CLEANSING: "data_cleansing",
            RoutingDecision.ROUTE_TO_REMEDIATION: "data_remediation",
            RoutingDecision.ROUTE_TO_DEAD_LETTER: "dead_letter_queue",
            RoutingDecision.ROUTE_TO_MANUAL_REVIEW: "manual_review",
            RoutingDecision.ROUTE_TO_ARCHIVE: "data_archive"
        }
        
        # Initialize routing rules
        self._initialize_routing_rules()
        
    def route_pbf_process_data(self, data: List[Dict[str, Any]], 
                             quality_results: List[QualityResult]) -> RoutingJob:
        """
        Route PBF process data based on quality results.
        
        Args:
            data: List of PBF process data records
            quality_results: List of quality validation results
            
        Returns:
            RoutingJob: The routing job results
        """
        try:
            logger.info(f"Starting PBF process data routing for {len(data)} records")
            
            # Create routing job
            job = RoutingJob(
                job_id=f"pbf_process_routing_{int(datetime.now().timestamp())}",
                source_name="pbf_process",
                data=data,
                quality_results=quality_results,
                rules=self._get_applicable_rules("pbf_process"),
                status=RoutingStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute routing
            job = self._execute_routing_job(job)
            
            # Store job
            self.routing_jobs[job.job_id] = job
            
            logger.info(f"PBF process data routing completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error routing PBF process data: {e}")
            raise
    
    def route_ispm_monitoring_data(self, data: List[Dict[str, Any]], 
                                 quality_results: List[QualityResult]) -> RoutingJob:
        """
        Route ISPM monitoring data based on quality results.
        
        Args:
            data: List of ISPM monitoring data records
            quality_results: List of quality validation results
            
        Returns:
            RoutingJob: The routing job results
        """
        try:
            logger.info(f"Starting ISPM monitoring data routing for {len(data)} records")
            
            # Create routing job
            job = RoutingJob(
                job_id=f"ispm_monitoring_routing_{int(datetime.now().timestamp())}",
                source_name="ispm_monitoring",
                data=data,
                quality_results=quality_results,
                rules=self._get_applicable_rules("ispm_monitoring"),
                status=RoutingStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute routing
            job = self._execute_routing_job(job)
            
            # Store job
            self.routing_jobs[job.job_id] = job
            
            logger.info(f"ISPM monitoring data routing completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error routing ISPM monitoring data: {e}")
            raise
    
    def route_ct_scan_data(self, data: List[Dict[str, Any]], 
                         quality_results: List[QualityResult]) -> RoutingJob:
        """
        Route CT scan data based on quality results.
        
        Args:
            data: List of CT scan data records
            quality_results: List of quality validation results
            
        Returns:
            RoutingJob: The routing job results
        """
        try:
            logger.info(f"Starting CT scan data routing for {len(data)} records")
            
            # Create routing job
            job = RoutingJob(
                job_id=f"ct_scan_routing_{int(datetime.now().timestamp())}",
                source_name="ct_scan",
                data=data,
                quality_results=quality_results,
                rules=self._get_applicable_rules("ct_scan"),
                status=RoutingStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute routing
            job = self._execute_routing_job(job)
            
            # Store job
            self.routing_jobs[job.job_id] = job
            
            logger.info(f"CT scan data routing completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error routing CT scan data: {e}")
            raise
    
    def route_powder_bed_data(self, data: List[Dict[str, Any]], 
                            quality_results: List[QualityResult]) -> RoutingJob:
        """
        Route powder bed data based on quality results.
        
        Args:
            data: List of powder bed data records
            quality_results: List of quality validation results
            
        Returns:
            RoutingJob: The routing job results
        """
        try:
            logger.info(f"Starting powder bed data routing for {len(data)} records")
            
            # Create routing job
            job = RoutingJob(
                job_id=f"powder_bed_routing_{int(datetime.now().timestamp())}",
                source_name="powder_bed",
                data=data,
                quality_results=quality_results,
                rules=self._get_applicable_rules("powder_bed"),
                status=RoutingStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute routing
            job = self._execute_routing_job(job)
            
            # Store job
            self.routing_jobs[job.job_id] = job
            
            logger.info(f"Powder bed data routing completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error routing powder bed data: {e}")
            raise
    
    def get_routing_job(self, job_id: str) -> Optional[RoutingJob]:
        """
        Get a routing job by ID.
        
        Args:
            job_id: The job ID
            
        Returns:
            RoutingJob: The routing job, or None if not found
        """
        return self.routing_jobs.get(job_id)
    
    def get_routing_results(self, source_name: str) -> List[RoutingResult]:
        """
        Get routing results for a specific source.
        
        Args:
            source_name: The data source name
            
        Returns:
            List[RoutingResult]: List of routing results
        """
        return self.routing_results.get(source_name, [])
    
    def add_routing_rule(self, rule: RoutingRule) -> bool:
        """
        Add a new routing rule.
        
        Args:
            rule: The routing rule to add
            
        Returns:
            bool: True if rule was added successfully, False otherwise
        """
        try:
            self.routing_rules[rule.id] = rule
            logger.info(f"Added routing rule: {rule.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding routing rule {rule.id}: {e}")
            return False
    
    def get_routing_rule(self, rule_id: str) -> Optional[RoutingRule]:
        """
        Get a routing rule by ID.
        
        Args:
            rule_id: The rule ID
            
        Returns:
            RoutingRule: The routing rule, or None if not found
        """
        return self.routing_rules.get(rule_id)
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all routing activities.
        
        Returns:
            Dict[str, Any]: Routing summary
        """
        try:
            total_jobs = len(self.routing_jobs)
            completed_jobs = len([job for job in self.routing_jobs.values() 
                                if job.status == RoutingStatus.COMPLETED])
            failed_jobs = len([job for job in self.routing_jobs.values() 
                             if job.status == RoutingStatus.FAILED])
            
            # Calculate total records processed
            total_records_processed = sum(job.processed_records for job in self.routing_jobs.values())
            total_records_routed = sum(job.routed_records for job in self.routing_jobs.values())
            
            # Calculate success rate
            success_rate = completed_jobs / total_jobs if total_jobs > 0 else 0.0
            
            return {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "success_rate": success_rate,
                "total_records_processed": total_records_processed,
                "total_records_routed": total_records_routed,
                "total_rules": len(self.routing_rules),
                "enabled_rules": len([rule for rule in self.routing_rules.values() if rule.enabled]),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting routing summary: {e}")
            return {}
    
    def _execute_routing_job(self, job: RoutingJob) -> RoutingJob:
        """Execute a routing job."""
        try:
            job.status = RoutingStatus.IN_PROGRESS
            job.started_at = datetime.now()
            
            # Process each rule
            for rule in job.rules:
                if not rule.enabled:
                    continue
                
                result = self._execute_routing_rule(job, rule)
                job.results.append(result)
                
                # Update job progress
                job.processed_records += result.records_processed
                job.routed_records += (result.records_accepted + result.records_rejected + 
                                     result.records_routed_to_cleansing + result.records_routed_to_remediation +
                                     result.records_routed_to_dead_letter + result.records_routed_to_manual_review +
                                     result.records_routed_to_archive)
            
            # Mark job as completed
            job.status = RoutingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Store results
            if job.source_name not in self.routing_results:
                self.routing_results[job.source_name] = []
            self.routing_results[job.source_name].extend(job.results)
            
            return job
            
        except Exception as e:
            logger.error(f"Error executing routing job {job.job_id}: {e}")
            job.status = RoutingStatus.FAILED
            job.completed_at = datetime.now()
            return job
    
    def _execute_routing_rule(self, job: RoutingJob, rule: RoutingRule) -> RoutingResult:
        """Execute a specific routing rule."""
        try:
            start_time = datetime.now()
            
            # Initialize result
            result = RoutingResult(
                rule_id=rule.id,
                source_name=job.source_name,
                decision=rule.decision,
                status=RoutingStatus.PENDING,
                routing_reason=rule.description,
                quality_score=0.0
            )
            
            # Calculate overall quality score
            if job.quality_results:
                overall_score = sum(qr.score for qr in job.quality_results) / len(job.quality_results)
                result.quality_score = overall_score
            
            # Apply routing based on rule conditions
            result = self._apply_routing_decision(job, rule, result)
            
            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing routing rule {rule.id}: {e}")
            result.status = RoutingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _apply_routing_decision(self, job: RoutingJob, rule: RoutingRule, 
                               result: RoutingResult) -> RoutingResult:
        """Apply routing decision based on rule conditions."""
        try:
            result.records_processed = len(job.data)
            
            # Evaluate rule conditions
            condition_met = self._evaluate_condition(rule.condition, job.quality_results, result.quality_score)
            
            if condition_met:
                # Apply routing decision
                if rule.decision == RoutingDecision.ACCEPT:
                    result = self._route_to_accept(job, rule, result)
                elif rule.decision == RoutingDecision.REJECT:
                    result = self._route_to_reject(job, rule, result)
                elif rule.decision == RoutingDecision.ROUTE_TO_CLEANSING:
                    result = self._route_to_cleansing(job, rule, result)
                elif rule.decision == RoutingDecision.ROUTE_TO_REMEDIATION:
                    result = self._route_to_remediation(job, rule, result)
                elif rule.decision == RoutingDecision.ROUTE_TO_DEAD_LETTER:
                    result = self._route_to_dead_letter(job, rule, result)
                elif rule.decision == RoutingDecision.ROUTE_TO_MANUAL_REVIEW:
                    result = self._route_to_manual_review(job, rule, result)
                elif rule.decision == RoutingDecision.ROUTE_TO_ARCHIVE:
                    result = self._route_to_archive(job, rule, result)
                else:
                    result.warnings.append(f"Unknown routing decision: {rule.decision}")
            else:
                # Condition not met, accept by default
                result.records_accepted = len(job.data)
                result.routing_reason = "Condition not met, accepted by default"
            
            result.status = RoutingStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error applying routing decision: {e}")
            result.status = RoutingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _evaluate_condition(self, condition: str, quality_results: List[QualityResult], 
                           quality_score: float) -> bool:
        """Evaluate routing condition."""
        try:
            if condition == "high_quality":
                return quality_score >= 0.9
            elif condition == "medium_quality":
                return 0.7 <= quality_score < 0.9
            elif condition == "low_quality":
                return 0.5 <= quality_score < 0.7
            elif condition == "poor_quality":
                return quality_score < 0.5
            elif condition == "has_errors":
                return any(not qr.passed for qr in quality_results)
            elif condition == "has_warnings":
                return any(qr.score < 0.8 for qr in quality_results)
            elif condition == "critical_errors":
                return any(qr.score < 0.3 for qr in quality_results)
            elif condition == "missing_data":
                return any(qr.valid_records < qr.total_records * 0.8 for qr in quality_results)
            else:
                # Default condition
                return True
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    def _route_to_accept(self, job: RoutingJob, rule: RoutingRule, 
                        result: RoutingResult) -> RoutingResult:
        """Route data to acceptance (data warehouse)."""
        try:
            result.records_accepted = len(job.data)
            result.routing_reason = f"High quality data routed to {self.routing_destinations[RoutingDecision.ACCEPT]}"
            
            # Add routing metadata
            for record in job.data:
                record["_routing_decision"] = "accept"
                record["_routing_timestamp"] = datetime.now().isoformat()
                record["_routing_rule"] = rule.id
                record["_routing_destination"] = self.routing_destinations[RoutingDecision.ACCEPT]
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing to accept: {e}")
            result.errors.append(str(e))
            return result
    
    def _route_to_reject(self, job: RoutingJob, rule: RoutingRule, 
                        result: RoutingResult) -> RoutingResult:
        """Route data to rejection."""
        try:
            result.records_rejected = len(job.data)
            result.routing_reason = f"Poor quality data routed to {self.routing_destinations[RoutingDecision.REJECT]}"
            
            # Add routing metadata
            for record in job.data:
                record["_routing_decision"] = "reject"
                record["_routing_timestamp"] = datetime.now().isoformat()
                record["_routing_rule"] = rule.id
                record["_routing_destination"] = self.routing_destinations[RoutingDecision.REJECT]
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing to reject: {e}")
            result.errors.append(str(e))
            return result
    
    def _route_to_cleansing(self, job: RoutingJob, rule: RoutingRule, 
                           result: RoutingResult) -> RoutingResult:
        """Route data to cleansing."""
        try:
            result.records_routed_to_cleansing = len(job.data)
            result.routing_reason = f"Data requiring cleansing routed to {self.routing_destinations[RoutingDecision.ROUTE_TO_CLEANSING]}"
            
            # Add routing metadata
            for record in job.data:
                record["_routing_decision"] = "route_to_cleansing"
                record["_routing_timestamp"] = datetime.now().isoformat()
                record["_routing_rule"] = rule.id
                record["_routing_destination"] = self.routing_destinations[RoutingDecision.ROUTE_TO_CLEANSING]
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing to cleansing: {e}")
            result.errors.append(str(e))
            return result
    
    def _route_to_remediation(self, job: RoutingJob, rule: RoutingRule, 
                             result: RoutingResult) -> RoutingResult:
        """Route data to remediation."""
        try:
            result.records_routed_to_remediation = len(job.data)
            result.routing_reason = f"Data requiring remediation routed to {self.routing_destinations[RoutingDecision.ROUTE_TO_REMEDIATION]}"
            
            # Add routing metadata
            for record in job.data:
                record["_routing_decision"] = "route_to_remediation"
                record["_routing_timestamp"] = datetime.now().isoformat()
                record["_routing_rule"] = rule.id
                record["_routing_destination"] = self.routing_destinations[RoutingDecision.ROUTE_TO_REMEDIATION]
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing to remediation: {e}")
            result.errors.append(str(e))
            return result
    
    def _route_to_dead_letter(self, job: RoutingJob, rule: RoutingRule, 
                             result: RoutingResult) -> RoutingResult:
        """Route data to dead letter queue."""
        try:
            result.records_routed_to_dead_letter = len(job.data)
            result.routing_reason = f"Failed data routed to {self.routing_destinations[RoutingDecision.ROUTE_TO_DEAD_LETTER]}"
            
            # Add routing metadata
            for record in job.data:
                record["_routing_decision"] = "route_to_dead_letter"
                record["_routing_timestamp"] = datetime.now().isoformat()
                record["_routing_rule"] = rule.id
                record["_routing_destination"] = self.routing_destinations[RoutingDecision.ROUTE_TO_DEAD_LETTER]
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing to dead letter: {e}")
            result.errors.append(str(e))
            return result
    
    def _route_to_manual_review(self, job: RoutingJob, rule: RoutingRule, 
                               result: RoutingResult) -> RoutingResult:
        """Route data to manual review."""
        try:
            result.records_routed_to_manual_review = len(job.data)
            result.routing_reason = f"Data requiring manual review routed to {self.routing_destinations[RoutingDecision.ROUTE_TO_MANUAL_REVIEW]}"
            
            # Add routing metadata
            for record in job.data:
                record["_routing_decision"] = "route_to_manual_review"
                record["_routing_timestamp"] = datetime.now().isoformat()
                record["_routing_rule"] = rule.id
                record["_routing_destination"] = self.routing_destinations[RoutingDecision.ROUTE_TO_MANUAL_REVIEW]
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing to manual review: {e}")
            result.errors.append(str(e))
            return result
    
    def _route_to_archive(self, job: RoutingJob, rule: RoutingRule, 
                         result: RoutingResult) -> RoutingResult:
        """Route data to archive."""
        try:
            result.records_routed_to_archive = len(job.data)
            result.routing_reason = f"Data routed to {self.routing_destinations[RoutingDecision.ROUTE_TO_ARCHIVE]}"
            
            # Add routing metadata
            for record in job.data:
                record["_routing_decision"] = "route_to_archive"
                record["_routing_timestamp"] = datetime.now().isoformat()
                record["_routing_rule"] = rule.id
                record["_routing_destination"] = self.routing_destinations[RoutingDecision.ROUTE_TO_ARCHIVE]
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing to archive: {e}")
            result.errors.append(str(e))
            return result
    
    def _get_applicable_rules(self, source_name: str) -> List[RoutingRule]:
        """Get applicable routing rules for a source."""
        try:
            applicable_rules = []
            
            for rule in self.routing_rules.values():
                if rule.enabled and (not rule.source_types or source_name in rule.source_types):
                    applicable_rules.append(rule)
            
            # Sort by priority
            priority_order = {
                RoutingPriority.CRITICAL: 0,
                RoutingPriority.HIGH: 1,
                RoutingPriority.MEDIUM: 2,
                RoutingPriority.LOW: 3
            }
            
            applicable_rules.sort(key=lambda r: priority_order.get(r.priority, 4))
            
            return applicable_rules
            
        except Exception as e:
            logger.error(f"Error getting applicable rules: {e}")
            return []
    
    def _initialize_routing_rules(self):
        """Initialize default routing rules."""
        try:
            # High quality data - accept
            self.routing_rules["high_quality_accept"] = RoutingRule(
                id="high_quality_accept",
                name="High Quality Accept",
                description="Accept high quality data",
                condition="high_quality",
                decision=RoutingDecision.ACCEPT,
                priority=RoutingPriority.HIGH,
                source_types=["pbf_process", "ispm_monitoring", "ct_scan", "powder_bed"]
            )
            
            # Medium quality data - route to cleansing
            self.routing_rules["medium_quality_cleansing"] = RoutingRule(
                id="medium_quality_cleansing",
                name="Medium Quality Cleansing",
                description="Route medium quality data to cleansing",
                condition="medium_quality",
                decision=RoutingDecision.ROUTE_TO_CLEANSING,
                priority=RoutingPriority.MEDIUM,
                source_types=["pbf_process", "ispm_monitoring", "ct_scan", "powder_bed"]
            )
            
            # Low quality data - route to remediation
            self.routing_rules["low_quality_remediation"] = RoutingRule(
                id="low_quality_remediation",
                name="Low Quality Remediation",
                description="Route low quality data to remediation",
                condition="low_quality",
                decision=RoutingDecision.ROUTE_TO_REMEDIATION,
                priority=RoutingPriority.MEDIUM,
                source_types=["pbf_process", "ispm_monitoring", "ct_scan", "powder_bed"]
            )
            
            # Poor quality data - route to dead letter
            self.routing_rules["poor_quality_dead_letter"] = RoutingRule(
                id="poor_quality_dead_letter",
                name="Poor Quality Dead Letter",
                description="Route poor quality data to dead letter queue",
                condition="poor_quality",
                decision=RoutingDecision.ROUTE_TO_DEAD_LETTER,
                priority=RoutingPriority.HIGH,
                source_types=["pbf_process", "ispm_monitoring", "ct_scan", "powder_bed"]
            )
            
            # Critical errors - route to manual review
            self.routing_rules["critical_errors_manual_review"] = RoutingRule(
                id="critical_errors_manual_review",
                name="Critical Errors Manual Review",
                description="Route data with critical errors to manual review",
                condition="critical_errors",
                decision=RoutingDecision.ROUTE_TO_MANUAL_REVIEW,
                priority=RoutingPriority.CRITICAL,
                source_types=["pbf_process", "ispm_monitoring", "ct_scan", "powder_bed"]
            )
            
            # Missing data - route to manual review
            self.routing_rules["missing_data_manual_review"] = RoutingRule(
                id="missing_data_manual_review",
                name="Missing Data Manual Review",
                description="Route data with missing values to manual review",
                condition="missing_data",
                decision=RoutingDecision.ROUTE_TO_MANUAL_REVIEW,
                priority=RoutingPriority.HIGH,
                source_types=["pbf_process", "ispm_monitoring", "ct_scan", "powder_bed"]
            )
            
            logger.info(f"Initialized {len(self.routing_rules)} routing rules")
            
        except Exception as e:
            logger.error(f"Error initializing routing rules: {e}")
