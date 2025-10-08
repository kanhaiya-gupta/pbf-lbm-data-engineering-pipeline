"""
CT Scan Data Pipeline DAG

This DAG orchestrates the CT scan data pipeline including batch ingestion,
processing, quality checks, and storage.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from .email_notifications import get_ct_scan_notifier
from airflow.models import Variable
# days_ago removed in Airflow 3.x - using datetime calculation instead
from airflow.sdk import TaskGroup

# Import our custom modules
from src.data_pipeline.ingestion.batch.file_ingester import FileIngester
from src.data_pipeline.processing.etl.etl_orchestrator import ETLOrchestrator
from src.data_pipeline.storage.data_warehouse.snowflake_client import SnowflakeClient
from src.data_pipeline.storage.data_lake.s3_client import S3Client
from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime.now() - timedelta(days=1),
    'retries': 2,
    'retry_delay': timedelta(minutes=15),
    'catchup': False
}

# Email notification configuration using SmtpNotifier (Airflow 3.x)
email_notifier = get_ct_scan_notifier()

# Create the DAG
dag = DAG(
    'ct_scan_data_pipeline',
    default_args=default_args,
    description='CT scan data pipeline DAG',
    schedule='0 2 * * *',  # Daily at 2 AM
    max_active_runs=1,
    max_active_tasks=3,
    tags=['ct', 'scan', 'data']
)

def check_s3_files(**context):
    """
    Check if new CT scan files are available in S3.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize S3 client
        s3_client = S3Client()
        
        # Check for new files in S3
        files = s3_client.list_objects(
            bucket_name=config['s3']['ct_scan_bucket'],
            prefix='raw/ct_scan/'
        )
        
        if not files:
            print("No new CT scan files found in S3")
            return 0
        
        print(f"Found {len(files)} CT scan files in S3")
        return len(files)
        
    except Exception as e:
        print(f"Error checking S3 files: {e}")
        raise

def ingest_ct_scan_data(**context):
    """
    Ingest CT scan data from S3 files.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize file ingester
        file_ingester = FileIngester()
        
        # Ingest data from S3
        data = file_ingester.ingest_from_s3(
            bucket_name=config['s3']['ct_scan_bucket'],
            prefix='raw/ct_scan/',
            file_format='parquet'
        )
        
        # Store data in XCom for next task
        context['task_instance'].xcom_push(key='ingested_data', value=data)
        
        print(f"Ingested {len(data)} CT scan records")
        return len(data)
        
    except Exception as e:
        print(f"Error ingesting CT scan data: {e}")
        raise

def process_ct_scan_data(**context):
    """
    Process CT scan data using ETL orchestrator.
    """
    try:
        # Get ingested data from XCom
        ingested_data = context['task_instance'].xcom_pull(key='ingested_data')
        
        if not ingested_data:
            print("No data to process")
            return 0
        
        # Initialize ETL orchestrator
        etl_orchestrator = ETLOrchestrator()
        
        # Process data
        processed_data = etl_orchestrator.process_ct_scan_data(ingested_data)
        
        # Store processed data in XCom
        context['task_instance'].xcom_push(key='processed_data', value=processed_data)
        
        print(f"Processed {len(processed_data)} CT scan records")
        return len(processed_data)
        
    except Exception as e:
        print(f"Error processing CT scan data: {e}")
        raise

def analyze_defects(**context):
    """
    Analyze defects in CT scan data.
    """
    try:
        # Get processed data from XCom
        processed_data = context['task_instance'].xcom_pull(key='processed_data')
        
        if not processed_data:
            print("No data to analyze for defects")
            return 0
        
        # Initialize defect analyzer
        from src.data_pipeline.quality.validation.defect_analyzer import DefectAnalyzer
        defect_analyzer = DefectAnalyzer()
        
        # Analyze defects
        defect_analysis = defect_analyzer.analyze_ct_scan_defects(processed_data)
        
        # Store defect analysis in XCom
        context['task_instance'].xcom_push(key='defect_analysis', value=defect_analysis)
        
        print(f"Analyzed defects in {len(processed_data)} CT scan records")
        return len(defect_analysis)
        
    except Exception as e:
        print(f"Error analyzing defects: {e}")
        raise

def load_ct_scan_data(**context):
    """
    Load processed CT scan data to Snowflake.
    """
    try:
        # Get processed data from XCom
        processed_data = context['task_instance'].xcom_pull(key='processed_data')
        
        if not processed_data:
            print("No data to load")
            return 0
        
        # Initialize Snowflake client
        snowflake_client = SnowflakeClient()
        
        # Load data to Snowflake
        success = snowflake_client.insert_data(
            table_name='fct_ct_scan',
            data=processed_data
        )
        
        if success:
            print(f"Loaded {len(processed_data)} CT scan records to Snowflake")
            return len(processed_data)
        else:
            raise Exception("Failed to load data to Snowflake")
            
    except Exception as e:
        print(f"Error loading CT scan data: {e}")
        raise

def validate_ct_scan_data(**context):
    """
    Validate CT scan data quality.
    """
    try:
        # Get processed data from XCom
        processed_data = context['task_instance'].xcom_pull(key='processed_data')
        
        if not processed_data:
            print("No data to validate")
            return 0
        
        # Initialize data quality validator
        from src.data_pipeline.quality.validation.data_quality_service import DataQualityService
        quality_service = DataQualityService()
        
        # Validate data quality
        validation_results = quality_service.validate_ct_scan_data(processed_data)
        
        # Check if validation passed
        if validation_results['overall_quality_score'] < 0.7:
            raise Exception(f"Data quality validation failed. Score: {validation_results['overall_quality_score']}")
        
        print(f"Data quality validation passed. Score: {validation_results['overall_quality_score']}")
        return validation_results['overall_quality_score']
        
    except Exception as e:
        print(f"Error validating CT scan data: {e}")
        raise

def send_defect_alerts(**context):
    """
    Send alerts for detected defects.
    """
    try:
        # Get defect analysis from XCom
        defect_analysis = context['task_instance'].xcom_pull(key='defect_analysis')
        
        if not defect_analysis:
            print("No defects to alert on")
            return 0
        
        # Initialize alert manager
        from src.data_pipeline.orchestration.monitoring.alert_manager import AlertManager
        alert_manager = AlertManager()
        
        # Send alerts for defects
        for defect in defect_analysis:
            if defect['severity'] == 'high':
                alert_manager.send_defect_alert(defect)
        
        print(f"Sent alerts for {len([d for d in defect_analysis if d['severity'] == 'high'])} high-severity defects")
        return len([d for d in defect_analysis if d['severity'] == 'high'])
        
    except Exception as e:
        print(f"Error sending defect alerts: {e}")
        raise

def archive_ct_scan_data(**context):
    """
    Archive processed CT scan data.
    """
    try:
        # Get processed data from XCom
        processed_data = context['task_instance'].xcom_pull(key='processed_data')
        
        if not processed_data:
            print("No data to archive")
            return 0
        
        # Initialize data archiver
        from src.data_pipeline.storage.data_lake.data_archiver import DataArchiver
        archiver = DataArchiver()
        
        # Archive data to S3
        archive_path = f"archive/ct_scan/{datetime.now().strftime('%Y/%m/%d')}/ct_scan_data.json"
        success = archiver.archive_data(
            data=processed_data,
            archive_path=archive_path,
            content_type="application/json"
        )
        
        if success:
            print(f"Archived {len(processed_data)} CT scan records to S3")
            return len(processed_data)
        else:
            raise Exception("Failed to archive data to S3")
            
    except Exception as e:
        print(f"Error archiving CT scan data: {e}")
        raise

def send_completion_notification(**context):
    """
    Send completion notification.
    """
    try:
        # Get task results
        ingested_count = context['task_instance'].xcom_pull(task_ids='ingest_ct_scan_data')
        processed_count = context['task_instance'].xcom_pull(task_ids='process_ct_scan_data')
        loaded_count = context['task_instance'].xcom_pull(task_ids='load_ct_scan_data')
        defects_count = context['task_instance'].xcom_pull(task_ids='analyze_defects')
        quality_score = context['task_instance'].xcom_pull(task_ids='validate_ct_scan_data')
        
        # Send notification
        message = f"""
        CT Scan Data Pipeline Completed Successfully!
        
        Records Processed:
        - Ingested: {ingested_count}
        - Processed: {processed_count}
        - Loaded: {loaded_count}
        - Defects Analyzed: {defects_count}
        - Quality Score: {quality_score}
        
        Timestamp: {datetime.now().isoformat()}
        """
        
        print(message)
        # Here you would send actual notification (email, Slack, etc.)
        
    except Exception as e:
        print(f"Error sending completion notification: {e}")

# Define tasks
start_task = EmptyOperator(
    task_id='start_ct_scan_pipeline',
    dag=dag
)

# S3 file check task
s3_check_task = PythonOperator(
    task_id='check_s3_files',
    python_callable=check_s3_files,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data ingestion task
ingest_task = PythonOperator(
    task_id='ingest_ct_scan_data',
    python_callable=ingest_ct_scan_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data processing task
process_task = PythonOperator(
    task_id='process_ct_scan_data',
    python_callable=process_ct_scan_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Defect analysis task
defect_task = PythonOperator(
    task_id='analyze_defects',
    python_callable=analyze_defects,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data loading task
load_task = PythonOperator(
    task_id='load_ct_scan_data',
    python_callable=load_ct_scan_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data validation task
validate_task = PythonOperator(
    task_id='validate_ct_scan_data',
    python_callable=validate_ct_scan_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Defect alerting task
alert_task = PythonOperator(
    task_id='send_defect_alerts',
    python_callable=send_defect_alerts,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data archiving task
archive_task = PythonOperator(
    task_id='archive_ct_scan_data',
    python_callable=archive_ct_scan_data,
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
    task_id='end_ct_scan_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> s3_check_task >> ingest_task >> process_task
process_task >> [load_task, validate_task, defect_task]
load_task >> archive_task
validate_task >> archive_task
defect_task >> alert_task >> archive_task
archive_task >> notification_task >> end_task

# Export the DAG
CTScanDAG = dag
