"""
Email notification utilities for Airflow 3.x DAGs.

This module provides reusable email notification configurations using SmtpNotifier
to replace the deprecated email settings in default_args.
"""

from airflow.providers.smtp.notifications.smtp import SmtpNotifier
from typing import List, Optional


def create_email_notifier(
    dag_name: str,
    recipients: Optional[List[str]] = None,
    from_email: str = 'airflow@example.com'
) -> SmtpNotifier:
    """
    Create a standardized email notifier for DAG task failures.
    
    Args:
        dag_name: Name of the DAG for email subject
        recipients: List of email recipients (defaults to data-team@example.com)
        from_email: Sender email address
        
    Returns:
        SmtpNotifier: Configured email notifier
    """
    if recipients is None:
        recipients = ['data-team@example.com']
    
    return SmtpNotifier(
        from_email=from_email,
        to=recipients,
        subject=f'{dag_name} DAG Task Failed: {{ ti.task_id }}',
        html_content=f'''
        <h3>{dag_name} Data Pipeline Task Failed</h3>
        <p><strong>Task ID:</strong> {{ ti.task_id }}</p>
        <p><strong>DAG ID:</strong> {{ ti.dag_id }}</p>
        <p><strong>Execution Date:</strong> {{ ti.execution_date }}</p>
        <p><strong>Log URL:</strong> <a href="{{ ti.log_url }}">{{ ti.log_url }}</a></p>
        <p>Please check the Airflow logs for more details.</p>
        '''
    )


# Pre-configured notifiers for common DAGs
def get_ct_scan_notifier() -> SmtpNotifier:
    """Get email notifier for CT Scan DAG."""
    return create_email_notifier('CT Scan')


def get_powder_bed_notifier() -> SmtpNotifier:
    """Get email notifier for Powder Bed DAG."""
    return create_email_notifier('Powder Bed')


def get_ispm_monitoring_notifier() -> SmtpNotifier:
    """Get email notifier for ISPM Monitoring DAG."""
    return create_email_notifier('ISPM Monitoring')


def get_pbf_process_notifier() -> SmtpNotifier:
    """Get email notifier for PBF Process DAG."""
    return create_email_notifier('PBF Process')


def get_data_quality_notifier() -> SmtpNotifier:
    """Get email notifier for Data Quality DAG."""
    return create_email_notifier('Data Quality')


def get_dbt_notifier() -> SmtpNotifier:
    """Get email notifier for DBT DAG."""
    return create_email_notifier('DBT')
