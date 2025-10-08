"""
PBF Process Data Pipeline DAG

This DAG orchestrates the PBF process data pipeline including ingestion,
processing, quality checks, and storage.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from .email_notifications import get_pbf_process_notifier
from airflow.models import Variable
# days_ago removed in Airflow 3.x - using datetime calculation instead
from airflow.sdk import TaskGroup

# Import our custom modules
from src.data_pipeline.ingestion.batch.machine_data_ingester import MachineDataIngester
from src.data_pipeline.processing.etl.etl_orchestrator import ETLOrchestrator
from src.data_pipeline.storage.data_warehouse.snowflake_client import SnowflakeClient
from src.data_pipeline.storage.operational.postgres_client import PostgresClient
from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime.now() - timedelta(days=1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Email notification configuration using SmtpNotifier (Airflow 3.x)
email_notifier = get_pbf_process_notifier()

# Create the DAG
dag = DAG(
    'pbf_process_data_pipeline',
    default_args=default_args,
    description='PBF process data pipeline DAG',
    schedule='0 0 * * *',  # Daily at midnight
    max_active_runs=1,
    max_active_tasks=10,
    tags=['pbf', 'process', 'data']
)

def extract_pbf_process_data(**context):
    """
    Extract PBF process data from operational database.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize data ingester
        ingester = MachineDataIngester()
        
        # Extract data from PostgreSQL
        data = ingester.ingest_from_database(
            db_connection_string=config['postgres']['connection_string'],
            table_name='pbf_process_data',
            query="SELECT * FROM pbf_process_data WHERE updated_at >= '{last_run_timestamp}'"
        )
        
        # Store data in XCom for next task
        context['task_instance'].xcom_push(key='extracted_data', value=data)
        
        print(f"Extracted {len(data)} PBF process records")
        return len(data)
        
    except Exception as e:
        print(f"Error extracting PBF process data: {e}")
        raise

def transform_pbf_process_data(**context):
    """
    Transform PBF process data using Spark.
    """
    try:
        # Get extracted data from XCom
        extracted_data = context['task_instance'].xcom_pull(key='extracted_data')
        
        if not extracted_data:
            print("No data to transform")
            return 0
        
        # Initialize ETL orchestrator
        etl_orchestrator = ETLOrchestrator()
        
        # Transform data
        transformed_data = etl_orchestrator.transform_pbf_process_data(extracted_data)
        
        # Store transformed data in XCom
        context['task_instance'].xcom_push(key='transformed_data', value=transformed_data)
        
        print(f"Transformed {len(transformed_data)} PBF process records")
        return len(transformed_data)
        
    except Exception as e:
        print(f"Error transforming PBF process data: {e}")
        raise

def load_pbf_process_data(**context):
    """
    Load transformed PBF process data to Snowflake.
    """
    try:
        # Get transformed data from XCom
        transformed_data = context['task_instance'].xcom_pull(key='transformed_data')
        
        if not transformed_data:
            print("No data to load")
            return 0
        
        # Initialize Snowflake client
        snowflake_client = SnowflakeClient()
        
        # Load data to Snowflake
        success = snowflake_client.insert_data(
            table_name='fct_pbf_process',
            data=transformed_data
        )
        
        if success:
            print(f"Loaded {len(transformed_data)} PBF process records to Snowflake")
            return len(transformed_data)
        else:
            raise Exception("Failed to load data to Snowflake")
            
    except Exception as e:
        print(f"Error loading PBF process data: {e}")
        raise

def validate_pbf_process_data(**context):
    """
    Validate PBF process data quality.
    """
    try:
        # Get transformed data from XCom
        transformed_data = context['task_instance'].xcom_pull(key='transformed_data')
        
        if not transformed_data:
            print("No data to validate")
            return 0
        
        # Initialize data quality validator
        from src.data_pipeline.quality.validation.data_quality_service import DataQualityService
        quality_service = DataQualityService()
        
        # Validate data quality
        validation_results = quality_service.validate_pbf_process_data(transformed_data)
        
        # Check if validation passed
        if validation_results['overall_quality_score'] < 0.8:
            raise Exception(f"Data quality validation failed. Score: {validation_results['overall_quality_score']}")
        
        print(f"Data quality validation passed. Score: {validation_results['overall_quality_score']}")
        return validation_results['overall_quality_score']
        
    except Exception as e:
        print(f"Error validating PBF process data: {e}")
        raise

def archive_pbf_process_data(**context):
    """
    Archive processed PBF process data.
    """
    try:
        # Get transformed data from XCom
        transformed_data = context['task_instance'].xcom_pull(key='transformed_data')
        
        if not transformed_data:
            print("No data to archive")
            return 0
        
        # Initialize data archiver
        from src.data_pipeline.storage.data_lake.data_archiver import DataArchiver
        archiver = DataArchiver()
        
        # Archive data to S3
        archive_path = f"archive/pbf_process/{datetime.now().strftime('%Y/%m/%d')}/pbf_process_data.json"
        success = archiver.archive_data(
            data=transformed_data,
            archive_path=archive_path,
            content_type="application/json"
        )
        
        if success:
            print(f"Archived {len(transformed_data)} PBF process records to S3")
            return len(transformed_data)
        else:
            raise Exception("Failed to archive data to S3")
            
    except Exception as e:
        print(f"Error archiving PBF process data: {e}")
        raise

def send_completion_notification(**context):
    """
    Send completion notification.
    """
    try:
        # Get task results
        extracted_count = context['task_instance'].xcom_pull(task_ids='extract_pbf_process_data')
        transformed_count = context['task_instance'].xcom_pull(task_ids='transform_pbf_process_data')
        loaded_count = context['task_instance'].xcom_pull(task_ids='load_pbf_process_data')
        quality_score = context['task_instance'].xcom_pull(task_ids='validate_pbf_process_data')
        
        # Send notification
        message = f"""
        PBF Process Data Pipeline Completed Successfully!
        
        Records Processed:
        - Extracted: {extracted_count}
        - Transformed: {transformed_count}
        - Loaded: {loaded_count}
        - Quality Score: {quality_score}
        
        Timestamp: {datetime.now().isoformat()}
        """
        
        print(message)
        # Here you would send actual notification (email, Slack, etc.)
        
    except Exception as e:
        print(f"Error sending completion notification: {e}")

# Define tasks
start_task = EmptyOperator(
    task_id='start_pbf_process_pipeline',
    dag=dag
)

# Data extraction task
extract_task = PythonOperator(
    task_id='extract_pbf_process_data',
    python_callable=extract_pbf_process_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data transformation task
transform_task = PythonOperator(
    task_id='transform_pbf_process_data',
    python_callable=transform_pbf_process_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data loading task
load_task = PythonOperator(
    task_id='load_pbf_process_data',
    python_callable=load_pbf_process_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data validation task
validate_task = PythonOperator(
    task_id='validate_pbf_process_data',
    python_callable=validate_pbf_process_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data archiving task
archive_task = PythonOperator(
    task_id='archive_pbf_process_data',
    python_callable=archive_pbf_process_data,
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
    task_id='end_pbf_process_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> extract_task >> transform_task >> [load_task, validate_task]
load_task >> archive_task
validate_task >> archive_task
archive_task >> notification_task >> end_task

# Export the DAG class for import
PBFProcessDAG = dag