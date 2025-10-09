"""
Drift Alert Manager

This module implements the drift alert manager for PBF-LB/M processes.
It provides centralized alert management, notification systems,
and alert escalation for drift detection services.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import mlflow.tensorflow
from pathlib import Path
from enum import Enum

from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Enums
class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(str, Enum):
    """Notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"


# Pydantic models for API requests and responses
class AlertRule(BaseModel):
    """Model for alert rules."""
    rule_id: str = Field(..., description="Rule ID")
    rule_name: str = Field(..., description="Rule name")
    alert_type: str = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity")
    conditions: Dict[str, Any] = Field(..., description="Alert conditions")
    notification_channels: List[NotificationChannel] = Field(..., description="Notification channels")
    escalation_policy: Optional[Dict[str, Any]] = Field(None, description="Escalation policy")
    enabled: bool = Field(True, description="Whether rule is enabled")


class Alert(BaseModel):
    """Model for alerts."""
    alert_id: str = Field(..., description="Alert ID")
    rule_id: str = Field(..., description="Rule ID")
    alert_type: str = Field(..., description="Alert type")
    severity: AlertSeverity = Field(..., description="Alert severity")
    status: AlertStatus = Field(..., description="Alert status")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    source: str = Field(..., description="Alert source")
    metadata: Dict[str, Any] = Field(..., description="Alert metadata")
    created_at: str = Field(..., description="Alert creation timestamp")
    acknowledged_at: Optional[str] = Field(None, description="Acknowledgment timestamp")
    resolved_at: Optional[str] = Field(None, description="Resolution timestamp")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged")
    resolved_by: Optional[str] = Field(None, description="User who resolved")


class NotificationConfig(BaseModel):
    """Configuration for notifications."""
    email_config: Optional[Dict[str, Any]] = Field(None, description="Email configuration")
    slack_config: Optional[Dict[str, Any]] = Field(None, description="Slack configuration")
    webhook_config: Optional[Dict[str, Any]] = Field(None, description="Webhook configuration")
    sms_config: Optional[Dict[str, Any]] = Field(None, description="SMS configuration")


class AlertManagerRequest(BaseModel):
    """Request model for alert management."""
    alert_type: str = Field(..., description="Alert type")
    severity: AlertSeverity = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    source: str = Field(..., description="Alert source")
    metadata: Dict[str, Any] = Field(..., description="Alert metadata")
    notification_channels: Optional[List[NotificationChannel]] = Field(None, description="Notification channels")


class AlertManagerResponse(BaseModel):
    """Response model for alert management."""
    alert_id: str = Field(..., description="Alert ID")
    status: str = Field(..., description="Alert status")
    notifications_sent: List[str] = Field(..., description="Notification channels used")
    escalation_triggered: bool = Field(False, description="Whether escalation was triggered")
    timestamp: str = Field(..., description="Response timestamp")


class DriftAlertManager:
    """
    Drift alert manager for PBF-LB/M processes.
    
    This manager provides comprehensive alert management capabilities for:
    - Centralized alert management
    - Multi-channel notifications
    - Alert escalation
    - Alert suppression
    - Alert analytics
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the drift alert manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Drift Alert Manager",
            description="Centralized alert management for PBF-LB/M manufacturing",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Alert management
        self.alerts = {}  # Store alerts
        self.alert_rules = {}  # Store alert rules
        self.notification_config = NotificationConfig()  # Notification configuration
        self.alert_counter = 0
        self.rule_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'total_rules': 0,
            'active_rules': 0,
            'notifications_sent': 0,
            'escalations_triggered': 0,
            'last_alert_time': None
        }
        
        # Setup routes
        self._setup_routes()
        
        # Load default alert rules
        self._load_default_rules()
        
        logger.info("Initialized DriftAlertManager")
    
    def _load_default_rules(self):
        """Load default alert rules."""
        # Data drift rule
        data_drift_rule = AlertRule(
            rule_id="data_drift_default",
            rule_name="Data Drift Detection",
            alert_type="data_drift",
            severity=AlertSeverity.WARNING,
            conditions={
                "drift_score_threshold": 0.2,
                "consecutive_detections": 2
            },
            notification_channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD],
            enabled=True
        )
        self.alert_rules[data_drift_rule.rule_id] = data_drift_rule
        
        # Model drift rule
        model_drift_rule = AlertRule(
            rule_id="model_drift_default",
            rule_name="Model Drift Detection",
            alert_type="model_drift",
            severity=AlertSeverity.CRITICAL,
            conditions={
                "performance_degradation_threshold": 0.15,
                "prediction_drift_threshold": 0.2
            },
            notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.DASHBOARD],
            enabled=True
        )
        self.alert_rules[model_drift_rule.rule_id] = model_drift_rule
        
        # Feature drift rule
        feature_drift_rule = AlertRule(
            rule_id="feature_drift_default",
            rule_name="Feature Drift Detection",
            alert_type="feature_drift",
            severity=AlertSeverity.WARNING,
            conditions={
                "feature_drift_threshold": 0.15,
                "high_importance_feature_drift": True
            },
            notification_channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD],
            enabled=True
        )
        self.alert_rules[feature_drift_rule.rule_id] = feature_drift_rule
        
        self.service_metrics['total_rules'] = len(self.alert_rules)
        self.service_metrics['active_rules'] = sum(1 for rule in self.alert_rules.values() if rule.enabled)
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "drift_alert_manager",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/alerts", response_model=AlertManagerResponse)
        async def create_alert(request: AlertManagerRequest):
            """Create a new alert."""
            return await self._create_alert(request)
        
        @self.app.get("/alerts")
        async def list_alerts(
            status: Optional[AlertStatus] = Query(None, description="Filter by status"),
            severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
            limit: int = Query(50, ge=1, le=1000, description="Limit number of results")
        ):
            """List alerts with optional filtering."""
            return await self._list_alerts(status, severity, limit)
        
        @self.app.get("/alerts/{alert_id}")
        async def get_alert(alert_id: str):
            """Get a specific alert."""
            return await self._get_alert(alert_id)
        
        @self.app.post("/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str, acknowledged_by: str = Query(..., description="User who acknowledged")):
            """Acknowledge an alert."""
            return await self._acknowledge_alert(alert_id, acknowledged_by)
        
        @self.app.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str, resolved_by: str = Query(..., description="User who resolved")):
            """Resolve an alert."""
            return await self._resolve_alert(alert_id, resolved_by)
        
        @self.app.post("/alerts/{alert_id}/suppress")
        async def suppress_alert(alert_id: str, suppressed_by: str = Query(..., description="User who suppressed")):
            """Suppress an alert."""
            return await self._suppress_alert(alert_id, suppressed_by)
        
        @self.app.post("/rules", response_model=Dict[str, Any])
        async def create_rule(rule: AlertRule):
            """Create a new alert rule."""
            return await self._create_rule(rule)
        
        @self.app.get("/rules")
        async def list_rules():
            """List all alert rules."""
            return await self._list_rules()
        
        @self.app.get("/rules/{rule_id}")
        async def get_rule(rule_id: str):
            """Get a specific alert rule."""
            return await self._get_rule(rule_id)
        
        @self.app.put("/rules/{rule_id}")
        async def update_rule(rule_id: str, rule: AlertRule):
            """Update an alert rule."""
            return await self._update_rule(rule_id, rule)
        
        @self.app.delete("/rules/{rule_id}")
        async def delete_rule(rule_id: str):
            """Delete an alert rule."""
            return await self._delete_rule(rule_id)
        
        @self.app.post("/notifications/test")
        async def test_notifications(channel: NotificationChannel):
            """Test notification channel."""
            return await self._test_notification(channel)
        
        @self.app.get("/analytics")
        async def get_alert_analytics():
            """Get alert analytics."""
            return await self._get_alert_analytics()
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _create_alert(self, request: AlertManagerRequest) -> AlertManagerResponse:
        """
        Create a new alert.
        
        Args:
            request: Alert creation request
            
        Returns:
            Alert creation response
        """
        # Generate alert ID
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}_{int(time.time())}"
        
        # Find matching alert rules
        matching_rules = []
        for rule in self.alert_rules.values():
            if rule.enabled and rule.alert_type == request.alert_type:
                if await self._evaluate_rule_conditions(rule, request):
                    matching_rules.append(rule)
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            rule_id=matching_rules[0].rule_id if matching_rules else "default",
            alert_type=request.alert_type,
            severity=request.severity,
            status=AlertStatus.ACTIVE,
            title=request.title,
            message=request.message,
            source=request.source,
            metadata=request.metadata,
            created_at=datetime.now().isoformat()
        )
        
        self.alerts[alert_id] = alert
        
        # Send notifications
        notifications_sent = []
        if matching_rules:
            for rule in matching_rules:
                for channel in rule.notification_channels:
                    try:
                        await self._send_notification(alert, channel)
                        notifications_sent.append(channel.value)
                    except Exception as e:
                        logger.error(f"Failed to send notification via {channel}: {e}")
        else:
            # Send to default channels
            default_channels = [NotificationChannel.DASHBOARD]
            for channel in default_channels:
                try:
                    await self._send_notification(alert, channel)
                    notifications_sent.append(channel.value)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel}: {e}")
        
        # Check for escalation
        escalation_triggered = False
        if matching_rules and any(rule.escalation_policy for rule in matching_rules):
            escalation_triggered = await self._check_escalation(alert, matching_rules)
        
        # Update metrics
        self.service_metrics['total_alerts'] += 1
        self.service_metrics['active_alerts'] += 1
        self.service_metrics['notifications_sent'] += len(notifications_sent)
        if escalation_triggered:
            self.service_metrics['escalations_triggered'] += 1
        self.service_metrics['last_alert_time'] = datetime.now().isoformat()
        
        return AlertManagerResponse(
            alert_id=alert_id,
            status="created",
            notifications_sent=notifications_sent,
            escalation_triggered=escalation_triggered,
            timestamp=datetime.now().isoformat()
        )
    
    async def _evaluate_rule_conditions(self, rule: AlertRule, request: AlertManagerRequest) -> bool:
        """Evaluate alert rule conditions."""
        try:
            conditions = rule.conditions
            
            # Check severity condition
            if 'min_severity' in conditions:
                severity_levels = {
                    AlertSeverity.INFO: 0,
                    AlertSeverity.WARNING: 1,
                    AlertSeverity.CRITICAL: 2,
                    AlertSeverity.EMERGENCY: 3
                }
                if severity_levels[request.severity] < severity_levels[conditions['min_severity']]:
                    return False
            
            # Check drift score threshold
            if 'drift_score_threshold' in conditions:
                drift_score = request.metadata.get('drift_score', 0)
                if drift_score < conditions['drift_score_threshold']:
                    return False
            
            # Check performance degradation threshold
            if 'performance_degradation_threshold' in conditions:
                degradation = request.metadata.get('performance_degradation', 0)
                if degradation < conditions['performance_degradation_threshold']:
                    return False
            
            # Check prediction drift threshold
            if 'prediction_drift_threshold' in conditions:
                prediction_drift = request.metadata.get('prediction_drift', 0)
                if prediction_drift < conditions['prediction_drift_threshold']:
                    return False
            
            # Check feature drift threshold
            if 'feature_drift_threshold' in conditions:
                feature_drift = request.metadata.get('feature_drift', 0)
                if feature_drift < conditions['feature_drift_threshold']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating rule conditions: {e}")
            return False
    
    async def _send_notification(self, alert: Alert, channel: NotificationChannel):
        """Send notification via specified channel."""
        try:
            if channel == NotificationChannel.EMAIL:
                await self._send_email_notification(alert)
            elif channel == NotificationChannel.SLACK:
                await self._send_slack_notification(alert)
            elif channel == NotificationChannel.WEBHOOK:
                await self._send_webhook_notification(alert)
            elif channel == NotificationChannel.SMS:
                await self._send_sms_notification(alert)
            elif channel == NotificationChannel.DASHBOARD:
                await self._send_dashboard_notification(alert)
            
            logger.info(f"Notification sent via {channel} for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send notification via {channel}: {e}")
            raise
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        if not self.notification_config.email_config:
            logger.warning("Email configuration not set")
            return
        
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.notification_config.email_config.get('from_email')
            msg['To'] = ', '.join(self.notification_config.email_config.get('to_emails', []))
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
            
            # Create email body
            body = f"""
            Alert Details:
            - ID: {alert.alert_id}
            - Type: {alert.alert_type}
            - Severity: {alert.severity}
            - Source: {alert.source}
            - Time: {alert.created_at}
            
            Message:
            {alert.message}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(
                self.notification_config.email_config.get('smtp_server'),
                self.notification_config.email_config.get('smtp_port')
            )
            server.starttls()
            server.login(
                self.notification_config.email_config.get('username'),
                self.notification_config.email_config.get('password')
            )
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            raise
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification."""
        if not self.notification_config.slack_config:
            logger.warning("Slack configuration not set")
            return
        
        try:
            webhook_url = self.notification_config.slack_config.get('webhook_url')
            if not webhook_url:
                return
            
            # Create Slack message
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }.get(alert.severity, "good")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Alert ID", "value": alert.alert_id, "short": True},
                        {"title": "Type", "value": alert.alert_type, "short": True},
                        {"title": "Severity", "value": alert.severity, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Time", "value": alert.created_at, "short": True}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            raise
    
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification."""
        if not self.notification_config.webhook_config:
            logger.warning("Webhook configuration not set")
            return
        
        try:
            webhook_url = self.notification_config.webhook_config.get('url')
            if not webhook_url:
                return
            
            payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "source": alert.source,
                "metadata": alert.metadata,
                "created_at": alert.created_at
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            raise
    
    async def _send_sms_notification(self, alert: Alert):
        """Send SMS notification."""
        if not self.notification_config.sms_config:
            logger.warning("SMS configuration not set")
            return
        
        # This would implement actual SMS sending
        # For now, just log
        logger.info(f"SMS notification for alert {alert.alert_id}")
    
    async def _send_dashboard_notification(self, alert: Alert):
        """Send dashboard notification."""
        # This would implement dashboard notification
        # For now, just log
        logger.info(f"Dashboard notification for alert {alert.alert_id}")
    
    async def _check_escalation(self, alert: Alert, matching_rules: List[AlertRule]) -> bool:
        """Check if alert should be escalated."""
        try:
            for rule in matching_rules:
                if rule.escalation_policy:
                    escalation_conditions = rule.escalation_policy.get('conditions', {})
                    
                    # Check if alert meets escalation conditions
                    if self._meets_escalation_conditions(alert, escalation_conditions):
                        await self._trigger_escalation(alert, rule)
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking escalation: {e}")
            return False
    
    def _meets_escalation_conditions(self, alert: Alert, conditions: Dict[str, Any]) -> bool:
        """Check if alert meets escalation conditions."""
        try:
            # Check severity threshold
            if 'severity_threshold' in conditions:
                severity_levels = {
                    AlertSeverity.INFO: 0,
                    AlertSeverity.WARNING: 1,
                    AlertSeverity.CRITICAL: 2,
                    AlertSeverity.EMERGENCY: 3
                }
                if severity_levels[alert.severity] < severity_levels[conditions['severity_threshold']]:
                    return False
            
            # Check time threshold
            if 'time_threshold_minutes' in conditions:
                alert_time = datetime.fromisoformat(alert.created_at)
                time_diff = (datetime.now() - alert_time).total_seconds() / 60
                if time_diff < conditions['time_threshold_minutes']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking escalation conditions: {e}")
            return False
    
    async def _trigger_escalation(self, alert: Alert, rule: AlertRule):
        """Trigger alert escalation."""
        try:
            escalation_policy = rule.escalation_policy
            escalation_channels = escalation_policy.get('channels', [])
            
            for channel in escalation_channels:
                await self._send_notification(alert, NotificationChannel(channel))
            
            logger.info(f"Escalation triggered for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error triggering escalation: {e}")
    
    async def _list_alerts(self, status: Optional[AlertStatus], severity: Optional[AlertSeverity], 
                          limit: int) -> Dict[str, Any]:
        """List alerts with optional filtering."""
        alerts = list(self.alerts.values())
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Sort by creation time (newest first)
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply limit
        limited_alerts = alerts[:limit]
        
        return {
            'alerts': [alert.dict() for alert in limited_alerts],
            'total_alerts': len(alerts),
            'filtered_count': len(limited_alerts),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_alert(self, alert_id: str) -> Alert:
        """Get a specific alert."""
        if alert_id not in self.alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        return self.alerts[alert_id]
    
    async def _acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> Dict[str, Any]:
        """Acknowledge an alert."""
        if alert_id not in self.alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now().isoformat()
        alert.acknowledged_by = acknowledged_by
        
        self.service_metrics['active_alerts'] -= 1
        
        return {
            'alert_id': alert_id,
            'status': 'acknowledged',
            'acknowledged_by': acknowledged_by,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _resolve_alert(self, alert_id: str, resolved_by: str) -> Dict[str, Any]:
        """Resolve an alert."""
        if alert_id not in self.alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now().isoformat()
        alert.resolved_by = resolved_by
        
        if alert.status == AlertStatus.ACTIVE:
            self.service_metrics['active_alerts'] -= 1
        self.service_metrics['resolved_alerts'] += 1
        
        return {
            'alert_id': alert_id,
            'status': 'resolved',
            'resolved_by': resolved_by,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _suppress_alert(self, alert_id: str, suppressed_by: str) -> Dict[str, Any]:
        """Suppress an alert."""
        if alert_id not in self.alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        
        if alert.status == AlertStatus.ACTIVE:
            self.service_metrics['active_alerts'] -= 1
        
        return {
            'alert_id': alert_id,
            'status': 'suppressed',
            'suppressed_by': suppressed_by,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _create_rule(self, rule: AlertRule) -> Dict[str, Any]:
        """Create a new alert rule."""
        self.rule_counter += 1
        rule_id = f"rule_{self.rule_counter}_{int(time.time())}"
        rule.rule_id = rule_id
        
        self.alert_rules[rule_id] = rule
        
        self.service_metrics['total_rules'] += 1
        if rule.enabled:
            self.service_metrics['active_rules'] += 1
        
        return {
            'rule_id': rule_id,
            'status': 'created',
            'message': 'Alert rule created successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_rules(self) -> Dict[str, Any]:
        """List all alert rules."""
        rules = [rule.dict() for rule in self.alert_rules.values()]
        
        return {
            'rules': rules,
            'total_rules': len(rules),
            'active_rules': sum(1 for rule in self.alert_rules.values() if rule.enabled),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_rule(self, rule_id: str) -> AlertRule:
        """Get a specific alert rule."""
        if rule_id not in self.alert_rules:
            raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
        
        return self.alert_rules[rule_id]
    
    async def _update_rule(self, rule_id: str, rule: AlertRule) -> Dict[str, Any]:
        """Update an alert rule."""
        if rule_id not in self.alert_rules:
            raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
        
        old_rule = self.alert_rules[rule_id]
        rule.rule_id = rule_id
        
        self.alert_rules[rule_id] = rule
        
        # Update metrics
        if old_rule.enabled and not rule.enabled:
            self.service_metrics['active_rules'] -= 1
        elif not old_rule.enabled and rule.enabled:
            self.service_metrics['active_rules'] += 1
        
        return {
            'rule_id': rule_id,
            'status': 'updated',
            'message': 'Alert rule updated successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _delete_rule(self, rule_id: str) -> Dict[str, Any]:
        """Delete an alert rule."""
        if rule_id not in self.alert_rules:
            raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
        
        rule = self.alert_rules[rule_id]
        del self.alert_rules[rule_id]
        
        self.service_metrics['total_rules'] -= 1
        if rule.enabled:
            self.service_metrics['active_rules'] -= 1
        
        return {
            'rule_id': rule_id,
            'status': 'deleted',
            'message': 'Alert rule deleted successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _test_notification(self, channel: NotificationChannel) -> Dict[str, Any]:
        """Test notification channel."""
        try:
            test_alert = Alert(
                alert_id="test_alert",
                rule_id="test_rule",
                alert_type="test",
                severity=AlertSeverity.INFO,
                status=AlertStatus.ACTIVE,
                title="Test Alert",
                message="This is a test notification",
                source="test_system",
                metadata={"test": True},
                created_at=datetime.now().isoformat()
            )
            
            await self._send_notification(test_alert, channel)
            
            return {
                'channel': channel.value,
                'status': 'success',
                'message': f'Test notification sent via {channel}',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'channel': channel.value,
                'status': 'error',
                'message': f'Failed to send test notification: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_alert_analytics(self) -> Dict[str, Any]:
        """Get alert analytics."""
        alerts = list(self.alerts.values())
        
        # Calculate analytics
        total_alerts = len(alerts)
        active_alerts = sum(1 for a in alerts if a.status == AlertStatus.ACTIVE)
        resolved_alerts = sum(1 for a in alerts if a.status == AlertStatus.RESOLVED)
        
        # Severity distribution
        severity_dist = {}
        for severity in AlertSeverity:
            severity_dist[severity.value] = sum(1 for a in alerts if a.severity == severity)
        
        # Alert type distribution
        type_dist = {}
        for alert in alerts:
            alert_type = alert.alert_type
            type_dist[alert_type] = type_dist.get(alert_type, 0) + 1
        
        # Recent alerts (last 24 hours)
        recent_time = datetime.now() - timedelta(hours=24)
        recent_alerts = sum(1 for a in alerts if datetime.fromisoformat(a.created_at) > recent_time)
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'resolved_alerts': resolved_alerts,
            'severity_distribution': severity_dist,
            'type_distribution': type_dist,
            'recent_alerts_24h': recent_alerts,
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8016):
        """Run the service."""
        logger.info(f"Starting Drift Alert Manager on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = DriftAlertManager()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
