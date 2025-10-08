"""
Data Quality Pipeline DAG

This DAG orchestrates the data quality monitoring pipeline including
quality checks, reporting, and remediation.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from .email_notifications import get_data_quality_notifier
from airflow.models import Variable
# days_ago removed in Airflow 3.x - using datetime calculation instead
from airflow.sdk import TaskGroup

# Import our custom modules
from src.data_pipeline.quality.validation.data_quality_service import DataQualityService
from src.data_pipeline.quality.monitoring.quality_monitor import QualityMonitor
from src.data_pipeline.quality.remediation.remediation_service import RemediationService
from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime.now() - timedelta(days=1),
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'catchup': False
}

# Email notification configuration using SmtpNotifier (Airflow 3.x)
email_notifier = get_data_quality_notifier()

# Create the DAG
dag = DAG(
    'data_quality_pipeline',
    default_args=default_args,
    description='Data quality monitoring pipeline DAG',
    schedule='0 1 * * *',  # Daily at 1 AM
    max_active_runs=1,
    max_active_tasks=3,
    tags=['quality', 'monitoring']
)

def check_data_quality(**context):
    """
    Check data quality across all data sources.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize data quality service
        quality_service = DataQualityService()
        
        # Check quality for all data sources
        quality_results = {}
        
        # Check PBF process data quality
        pbf_quality = quality_service.check_pbf_process_quality()
        quality_results['pbf_process'] = pbf_quality
        
        # Check ISPM monitoring data quality
        ispm_quality = quality_service.check_ispm_monitoring_quality()
        quality_results['ispm_monitoring'] = ispm_quality
        
        # Check CT scan data quality
        ct_quality = quality_service.check_ct_scan_quality()
        quality_results['ct_scan'] = ct_quality
        
        # Check powder bed data quality
        powder_quality = quality_service.check_powder_bed_quality()
        quality_results['powder_bed'] = powder_quality
        
        # Store quality results in XCom
        context['task_instance'].xcom_push(key='quality_results', value=quality_results)
        
        print(f"Data quality check completed for {len(quality_results)} data sources")
        return len(quality_results)
        
    except Exception as e:
        print(f"Error checking data quality: {e}")
        raise

def generate_quality_report(**context):
    """
    Generate data quality report.
    """
    try:
        # Get quality results from XCom
        quality_results = context['task_instance'].xcom_pull(key='quality_results')
        
        if not quality_results:
            print("No quality results to report on")
            return 0
        
        # Initialize quality monitor
        quality_monitor = QualityMonitor()
        
        # Generate quality report
        report = quality_monitor.generate_quality_report(quality_results)
        
        # Store report in XCom
        context['task_instance'].xcom_push(key='quality_report', value=report)
        
        print(f"Generated quality report for {len(quality_results)} data sources")
        return len(quality_results)
        
    except Exception as e:
        print(f"Error generating quality report: {e}")
        raise

def identify_quality_issues(**context):
    """
    Identify data quality issues that need remediation.
    """
    try:
        # Get quality results from XCom
        quality_results = context['task_instance'].xcom_pull(key='quality_results')
        
        if not quality_results:
            print("No quality results to analyze for issues")
            return 0
        
        # Initialize quality monitor
        quality_monitor = QualityMonitor()
        
        # Identify quality issues
        quality_issues = quality_monitor.identify_quality_issues(quality_results)
        
        # Store quality issues in XCom
        context['task_instance'].xcom_push(key='quality_issues', value=quality_issues)
        
        print(f"Identified {len(quality_issues)} quality issues")
        return len(quality_issues)
        
    except Exception as e:
        print(f"Error identifying quality issues: {e}")
        raise

def remediate_quality_issues(**context):
    """
    Remediate identified data quality issues.
    """
    try:
        # Get quality issues from XCom
        quality_issues = context['task_instance'].xcom_pull(key='quality_issues')
        
        if not quality_issues:
            print("No quality issues to remediate")
            return 0
        
        # Initialize remediation service
        remediation_service = RemediationService()
        
        # Remediate quality issues
        remediation_results = remediation_service.remediate_quality_issues(quality_issues)
        
        # Store remediation results in XCom
        context['task_instance'].xcom_push(key='remediation_results', value=remediation_results)
        
        print(f"Remediated {len(remediation_results)} quality issues")
        return len(remediation_results)
        
    except Exception as e:
        print(f"Error remediating quality issues: {e}")
        raise

def send_quality_alerts(**context):
    """
    Send alerts for critical quality issues.
    """
    try:
        # Get quality issues from XCom
        quality_issues = context['task_instance'].xcom_pull(key='quality_issues')
        
        if not quality_issues:
            print("No quality issues to alert on")
            return 0
        
        # Initialize alert manager
        from src.data_pipeline.orchestration.monitoring.alert_manager import AlertManager
        alert_manager = AlertManager()
        
        # Send alerts for critical issues
        critical_issues = [issue for issue in quality_issues if issue['severity'] == 'critical']
        
        for issue in critical_issues:
            alert_manager.send_quality_alert(issue)
        
        print(f"Sent alerts for {len(critical_issues)} critical quality issues")
        return len(critical_issues)
        
    except Exception as e:
        print(f"Error sending quality alerts: {e}")
        raise

def store_quality_metrics(**context):
    """
    Store quality metrics in monitoring database.
    """
    try:
        # Get quality results from XCom
        quality_results = context['task_instance'].xcom_pull(key='quality_results')
        
        if not quality_results:
            print("No quality results to store")
            return 0
        
        # Initialize quality monitor
        quality_monitor = QualityMonitor()
        
        # Store quality metrics
        success = quality_monitor.store_quality_metrics(quality_results)
        
        if success:
            print(f"Stored quality metrics for {len(quality_results)} data sources")
            return len(quality_results)
        else:
            raise Exception("Failed to store quality metrics")
            
    except Exception as e:
        print(f"Error storing quality metrics: {e}")
        raise

def send_quality_report(**context):
    """
    Send quality report to stakeholders.
    """
    try:
        # Get quality report from XCom
        quality_report = context['task_instance'].xcom_pull(key='quality_report')
        
        if not quality_report:
            print("No quality report to send")
            return 0
        
        # Initialize quality monitor
        quality_monitor = QualityMonitor()
        
        # Send quality report
        success = quality_monitor.send_quality_report(quality_report)
        
        if success:
            print("Quality report sent to stakeholders")
            return 1
        else:
            raise Exception("Failed to send quality report")
            
    except Exception as e:
        print(f"Error sending quality report: {e}")
        raise

def send_completion_notification(**context):
    """
    Send completion notification.
    """
    try:
        # Get task results
        quality_sources_count = context['task_instance'].xcom_pull(task_ids='check_data_quality')
        quality_issues_count = context['task_instance'].xcom_pull(task_ids='identify_quality_issues')
        remediation_count = context['task_instance'].xcom_pull(task_ids='remediate_quality_issues')
        alerts_count = context['task_instance'].xcom_pull(task_ids='send_quality_alerts')
        
        # Send notification
        message = f"""
        Data Quality Pipeline Completed Successfully!
        
        Quality Check Results:
        - Data Sources Checked: {quality_sources_count}
        - Quality Issues Identified: {quality_issues_count}
        - Issues Remediated: {remediation_count}
        - Alerts Sent: {alerts_count}
        
        Timestamp: {datetime.now().isoformat()}
        """
        
        print(message)
        # Here you would send actual notification (email, Slack, etc.)
        
    except Exception as e:
        print(f"Error sending completion notification: {e}")

# Define tasks
start_task = EmptyOperator(
    task_id='start_data_quality_pipeline',
    dag=dag
)

# Data quality check task
quality_check_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    on_failure_callback=email_notifier,
    dag=dag
)

# Quality report generation task
report_task = PythonOperator(
    task_id='generate_quality_report',
    python_callable=generate_quality_report,
    on_failure_callback=email_notifier,
    dag=dag
)

# Quality issues identification task
issues_task = PythonOperator(
    task_id='identify_quality_issues',
    python_callable=identify_quality_issues,
    on_failure_callback=email_notifier,
    dag=dag
)

# Quality remediation task
remediation_task = PythonOperator(
    task_id='remediate_quality_issues',
    python_callable=remediate_quality_issues,
    on_failure_callback=email_notifier,
    dag=dag
)

# Quality alerting task
alert_task = PythonOperator(
    task_id='send_quality_alerts',
    python_callable=send_quality_alerts,
    on_failure_callback=email_notifier,
    dag=dag
)

# Quality metrics storage task
metrics_task = PythonOperator(
    task_id='store_quality_metrics',
    python_callable=store_quality_metrics,
    on_failure_callback=email_notifier,
    dag=dag
)

# Quality report sending task
send_report_task = PythonOperator(
    task_id='send_quality_report',
    python_callable=send_quality_report,
    on_failure_callback=email_notifier,
    dag=dag
)

# Completion notification task
notification_task = PythonOperator(
    task_id='send_completion_notification',
    python_callable=send_completion_notification,
    on_failure_callback=email_notifier,
    dag=dag
)

end_task = EmptyOperator(
    task_id='end_data_quality_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> quality_check_task
quality_check_task >> [report_task, issues_task, metrics_task]
issues_task >> [remediation_task, alert_task]
report_task >> send_report_task
remediation_task >> notification_task
alert_task >> notification_task
metrics_task >> notification_task
send_report_task >> notification_task
notification_task >> end_task

# Export the DAG
DataQualityDAG = dag
