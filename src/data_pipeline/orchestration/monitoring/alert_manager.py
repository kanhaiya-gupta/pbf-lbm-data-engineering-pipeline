"""
Alert Manager

This module manages alerts and notifications for the PBF-LB/M data pipeline.
"""

import smtplib
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import time

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity enumeration."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertChannel(Enum):
    """Alert channel enumeration."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"

@dataclass
class AlertRule:
    """Alert rule data class."""
    id: str
    name: str
    condition: str
    severity: AlertSeverity
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown_minutes: int = 15
    last_triggered: Optional[datetime] = None

@dataclass
class Notification:
    """Notification data class."""
    id: str
    alert_id: str
    channel: AlertChannel
    recipient: str
    message: str
    sent_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, sent, failed

@dataclass
class Alert:
    """Alert data class."""
    id: str
    pipeline_name: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    notifications: List[Notification] = field(default_factory=list)

class AlertManager:
    """
    Manages alerts and notifications for the pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notifications: List[Notification] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.notification_thread: Optional[threading.Thread] = None
        self.is_processing = False
        
        # Initialize alert rules
        self._initialize_alert_rules()
        
    def start_processing(self) -> bool:
        """
        Start alert processing.
        
        Returns:
            bool: True if processing started successfully, False otherwise
        """
        try:
            if self.is_processing:
                logger.warning("Alert processing is already running")
                return False
            
            self.is_processing = True
            self.notification_thread = threading.Thread(target=self._notification_loop, daemon=True)
            self.notification_thread.start()
            
            logger.info("Alert processing started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting alert processing: {e}")
            return False
    
    def stop_processing(self) -> bool:
        """
        Stop alert processing.
        
        Returns:
            bool: True if processing stopped successfully, False otherwise
        """
        try:
            if not self.is_processing:
                logger.warning("Alert processing is not running")
                return False
            
            self.is_processing = False
            
            if self.notification_thread and self.notification_thread.is_alive():
                self.notification_thread.join(timeout=10)
            
            logger.info("Alert processing stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping alert processing: {e}")
            return False
    
    def create_alert(self, pipeline_name: str, alert_type: str, severity: AlertSeverity, message: str) -> Alert:
        """
        Create a new alert.
        
        Args:
            pipeline_name: The pipeline name
            alert_type: The alert type
            severity: The alert severity
            message: The alert message
            
        Returns:
            Alert: The created alert
        """
        try:
            alert_id = f"{pipeline_name}_{alert_type}_{int(time.time())}"
            
            alert = Alert(
                id=alert_id,
                pipeline_name=pipeline_name,
                alert_type=alert_type,
                severity=severity,
                message=message
            )
            
            self.alerts.append(alert)
            
            # Process alert for notifications
            self._process_alert(alert)
            
            logger.info(f"Created alert {alert_id}: {message}")
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            raise
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: The alert ID
            
        Returns:
            bool: True if alert was resolved successfully, False otherwise
        """
        try:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    # Send resolution notification
                    self._send_resolution_notification(alert)
                    
                    logger.info(f"Resolved alert {alert_id}")
                    return True
            
            logger.warning(f"Alert {alert_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get all active (unresolved) alerts.
        
        Returns:
            List[Alert]: List of active alerts
        """
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """
        Get all alerts with a specific severity.
        
        Args:
            severity: The alert severity
            
        Returns:
            List[Alert]: List of alerts with the specified severity
        """
        return [alert for alert in self.alerts if alert.severity == severity]
    
    def get_alerts_by_pipeline(self, pipeline_name: str) -> List[Alert]:
        """
        Get all alerts for a specific pipeline.
        
        Args:
            pipeline_name: The pipeline name
            
        Returns:
            List[Alert]: List of alerts for the specified pipeline
        """
        return [alert for alert in self.alerts if alert.pipeline_name == pipeline_name]
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """
        Add an alert rule.
        
        Args:
            rule: The alert rule to add
            
        Returns:
            bool: True if rule was added successfully, False otherwise
        """
        try:
            self.alert_rules[rule.id] = rule
            logger.info(f"Added alert rule {rule.id}: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding alert rule {rule.id}: {e}")
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_id: The rule ID
            
        Returns:
            bool: True if rule was removed successfully, False otherwise
        """
        try:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"Removed alert rule {rule_id}")
                return True
            else:
                logger.warning(f"Alert rule {rule_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error removing alert rule {rule_id}: {e}")
            return False
    
    def send_anomaly_alert(self, anomaly_data: Dict[str, Any]) -> bool:
        """
        Send an alert for detected anomalies.
        
        Args:
            anomaly_data: The anomaly data
            
        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        try:
            message = f"Anomaly detected in {anomaly_data.get('pipeline', 'unknown pipeline')}: {anomaly_data.get('description', 'Unknown anomaly')}"
            
            alert = self.create_alert(
                pipeline_name=anomaly_data.get('pipeline', 'unknown'),
                alert_type='anomaly_detected',
                severity=AlertSeverity.HIGH,
                message=message
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending anomaly alert: {e}")
            return False
    
    def send_quality_alert(self, quality_data: Dict[str, Any]) -> bool:
        """
        Send an alert for data quality issues.
        
        Args:
            quality_data: The quality data
            
        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        try:
            message = f"Data quality issue in {quality_data.get('pipeline', 'unknown pipeline')}: {quality_data.get('issue', 'Unknown quality issue')}"
            
            alert = self.create_alert(
                pipeline_name=quality_data.get('pipeline', 'unknown'),
                alert_type='data_quality_issue',
                severity=AlertSeverity.MEDIUM,
                message=message
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending quality alert: {e}")
            return False
    
    def send_defect_alert(self, defect_data: Dict[str, Any]) -> bool:
        """
        Send an alert for detected defects.
        
        Args:
            defect_data: The defect data
            
        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        try:
            message = f"Defect detected in {defect_data.get('pipeline', 'unknown pipeline')}: {defect_data.get('defect_type', 'Unknown defect')}"
            
            alert = self.create_alert(
                pipeline_name=defect_data.get('pipeline', 'unknown'),
                alert_type='defect_detected',
                severity=AlertSeverity.HIGH,
                message=message
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending defect alert: {e}")
            return False
    
    def send_surface_quality_alert(self, surface_data: Dict[str, Any]) -> bool:
        """
        Send an alert for surface quality issues.
        
        Args:
            surface_data: The surface quality data
            
        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        try:
            message = f"Surface quality issue in {surface_data.get('pipeline', 'unknown pipeline')}: {surface_data.get('issue', 'Unknown surface quality issue')}"
            
            alert = self.create_alert(
                pipeline_name=surface_data.get('pipeline', 'unknown'),
                alert_type='surface_quality_issue',
                severity=AlertSeverity.MEDIUM,
                message=message
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending surface quality alert: {e}")
            return False
    
    def get_alert_manager_status(self) -> Dict[str, Any]:
        """
        Get the current status of the alert manager.
        
        Returns:
            Dict[str, Any]: Alert manager status information
        """
        try:
            total_alerts = len(self.alerts)
            active_alerts = len(self.get_active_alerts())
            total_rules = len(self.alert_rules)
            enabled_rules = len([rule for rule in self.alert_rules.values() if rule.enabled])
            
            severity_counts = {}
            for severity in AlertSeverity:
                severity_counts[severity.value] = len(self.get_alerts_by_severity(severity))
            
            return {
                "is_processing": self.is_processing,
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "total_rules": total_rules,
                "enabled_rules": enabled_rules,
                "severity_counts": severity_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting alert manager status: {e}")
            return {}
    
    def _initialize_alert_rules(self):
        """Initialize default alert rules."""
        try:
            # Pipeline health rules
            self.add_alert_rule(AlertRule(
                id="pipeline_unhealthy",
                name="Pipeline Unhealthy",
                condition="pipeline_status == 'unhealthy'",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.PAGERDUTY]
            ))
            
            self.add_alert_rule(AlertRule(
                id="pipeline_degraded",
                name="Pipeline Degraded",
                condition="pipeline_status == 'degraded'",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
            ))
            
            # Performance rules
            self.add_alert_rule(AlertRule(
                id="high_error_rate",
                name="High Error Rate",
                condition="error_rate > 0.1",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
            ))
            
            self.add_alert_rule(AlertRule(
                id="high_latency",
                name="High Latency",
                condition="latency > 30",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.EMAIL]
            ))
            
            # Data quality rules
            self.add_alert_rule(AlertRule(
                id="data_quality_issue",
                name="Data Quality Issue",
                condition="quality_score < 0.8",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
            ))
            
        except Exception as e:
            logger.error(f"Error initializing alert rules: {e}")
    
    def _process_alert(self, alert: Alert):
        """Process an alert for notifications."""
        try:
            # Find matching alert rules
            matching_rules = []
            for rule in self.alert_rules.values():
                if rule.enabled and self._evaluate_rule_condition(rule, alert):
                    # Check cooldown
                    if (rule.last_triggered is None or 
                        datetime.now() - rule.last_triggered > timedelta(minutes=rule.cooldown_minutes)):
                        matching_rules.append(rule)
                        rule.last_triggered = datetime.now()
            
            # Send notifications for matching rules
            for rule in matching_rules:
                for channel in rule.channels:
                    self._send_notification(alert, channel, rule.severity)
                    
        except Exception as e:
            logger.error(f"Error processing alert {alert.id}: {e}")
    
    def _evaluate_rule_condition(self, rule: AlertRule, alert: Alert) -> bool:
        """
        Evaluate if an alert matches a rule condition.
        
        Args:
            rule: The alert rule
            alert: The alert
            
        Returns:
            bool: True if condition matches, False otherwise
        """
        try:
            # Simple condition evaluation (in a real system, you'd use a proper expression evaluator)
            if rule.condition == "pipeline_status == 'unhealthy'":
                return alert.alert_type == "pipeline_unhealthy"
            elif rule.condition == "pipeline_status == 'degraded'":
                return alert.alert_type == "pipeline_degraded"
            elif rule.condition == "error_rate > 0.1":
                return alert.alert_type == "high_error_rate"
            elif rule.condition == "latency > 30":
                return alert.alert_type == "high_latency"
            elif rule.condition == "quality_score < 0.8":
                return alert.alert_type == "data_quality_issue"
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating rule condition: {e}")
            return False
    
    def _send_notification(self, alert: Alert, channel: AlertChannel, severity: AlertSeverity):
        """Send a notification through a specific channel."""
        try:
            notification_id = f"{alert.id}_{channel.value}_{int(time.time())}"
            
            notification = Notification(
                id=notification_id,
                alert_id=alert.id,
                channel=channel,
                recipient=self._get_channel_recipient(channel),
                message=self._format_alert_message(alert, severity)
            )
            
            self.notifications.append(notification)
            alert.notifications.append(notification)
            
            # Send notification based on channel
            if channel == AlertChannel.EMAIL:
                self._send_email_notification(notification)
            elif channel == AlertChannel.SLACK:
                self._send_slack_notification(notification)
            elif channel == AlertChannel.PAGERDUTY:
                self._send_pagerduty_notification(notification)
            elif channel == AlertChannel.WEBHOOK:
                self._send_webhook_notification(notification)
            elif channel == AlertChannel.LOG:
                self._send_log_notification(notification)
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def _send_resolution_notification(self, alert: Alert):
        """Send resolution notification for an alert."""
        try:
            message = f"Alert resolved: {alert.message}"
            
            # Send to all channels that received the original alert
            for notification in alert.notifications:
                if notification.status == "sent":
                    resolution_notification = Notification(
                        id=f"resolution_{notification.id}",
                        alert_id=alert.id,
                        channel=notification.channel,
                        recipient=notification.recipient,
                        message=message
                    )
                    
                    self.notifications.append(resolution_notification)
                    
                    # Send resolution notification
                    if notification.channel == AlertChannel.EMAIL:
                        self._send_email_notification(resolution_notification)
                    elif notification.channel == AlertChannel.SLACK:
                        self._send_slack_notification(resolution_notification)
                    elif notification.channel == AlertChannel.PAGERDUTY:
                        self._send_pagerduty_notification(resolution_notification)
                    
        except Exception as e:
            logger.error(f"Error sending resolution notification: {e}")
    
    def _get_channel_recipient(self, channel: AlertChannel) -> str:
        """Get the recipient for a notification channel."""
        try:
            if channel == AlertChannel.EMAIL:
                return self.config.get('email', {}).get('recipients', ['data-team@example.com'])[0]
            elif channel == AlertChannel.SLACK:
                return self.config.get('slack', {}).get('channel', '#alerts')
            elif channel == AlertChannel.PAGERDUTY:
                return self.config.get('pagerduty', {}).get('integration_key', '')
            elif channel == AlertChannel.WEBHOOK:
                return self.config.get('webhook', {}).get('url', '')
            else:
                return 'system'
                
        except Exception as e:
            logger.error(f"Error getting channel recipient: {e}")
            return 'system'
    
    def _format_alert_message(self, alert: Alert, severity: AlertSeverity) -> str:
        """Format an alert message."""
        try:
            message = f"""
            🚨 ALERT: {severity.value.upper()}
            
            Pipeline: {alert.pipeline_name}
            Type: {alert.alert_type}
            Severity: {severity.value}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Message: {alert.message}
            
            Alert ID: {alert.id}
            """
            return message.strip()
            
        except Exception as e:
            logger.error(f"Error formatting alert message: {e}")
            return alert.message
    
    def _send_email_notification(self, notification: Notification):
        """Send email notification."""
        try:
            # This is a placeholder implementation
            # In a real system, you would use proper email sending
            logger.info(f"EMAIL: {notification.recipient} - {notification.message}")
            notification.status = "sent"
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            notification.status = "failed"
    
    def _send_slack_notification(self, notification: Notification):
        """Send Slack notification."""
        try:
            # This is a placeholder implementation
            # In a real system, you would use Slack webhook
            logger.info(f"SLACK: {notification.recipient} - {notification.message}")
            notification.status = "sent"
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            notification.status = "failed"
    
    def _send_pagerduty_notification(self, notification: Notification):
        """Send PagerDuty notification."""
        try:
            # This is a placeholder implementation
            # In a real system, you would use PagerDuty API
            logger.info(f"PAGERDUTY: {notification.recipient} - {notification.message}")
            notification.status = "sent"
            
        except Exception as e:
            logger.error(f"Error sending PagerDuty notification: {e}")
            notification.status = "failed"
    
    def _send_webhook_notification(self, notification: Notification):
        """Send webhook notification."""
        try:
            # This is a placeholder implementation
            # In a real system, you would use HTTP requests
            logger.info(f"WEBHOOK: {notification.recipient} - {notification.message}")
            notification.status = "sent"
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            notification.status = "failed"
    
    def _send_log_notification(self, notification: Notification):
        """Send log notification."""
        try:
            logger.info(f"LOG: {notification.message}")
            notification.status = "sent"
            
        except Exception as e:
            logger.error(f"Error sending log notification: {e}")
            notification.status = "failed"
    
    def _notification_loop(self):
        """Main notification processing loop."""
        while self.is_processing:
            try:
                # Process pending notifications
                pending_notifications = [n for n in self.notifications if n.status == "pending"]
                
                for notification in pending_notifications:
                    # Retry failed notifications
                    if notification.status == "failed":
                        # Implement retry logic here
                        pass
                
                # Sleep for a short interval
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in notification loop: {e}")
                time.sleep(10)
