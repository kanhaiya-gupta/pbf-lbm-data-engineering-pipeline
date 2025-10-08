"""
Powder Bed Monitoring Data Pipeline DAG

This DAG orchestrates the powder bed monitoring data pipeline including
streaming ingestion, processing, quality checks, and storage.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageSensor
from .email_notifications import get_powder_bed_notifier
from airflow.models import Variable
# days_ago removed in Airflow 3.x - using datetime calculation instead
from airflow.sdk import TaskGroup

# Import our custom modules
from src.data_pipeline.ingestion.streaming.kafka_ingester import KafkaIngester
from src.data_pipeline.processing.streaming.streaming_processor import UnifiedStreamingProcessor
from src.data_pipeline.storage.data_warehouse.snowflake_client import SnowflakeClient
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
email_notifier = get_powder_bed_notifier()

# Create the DAG
dag = DAG(
    'powder_bed_data_pipeline',
    default_args=default_args,
    description='Powder bed monitoring data pipeline DAG',
    schedule='0 */30 * * *',  # Every 30 minutes
    max_active_runs=1,
    max_active_tasks=5,
    tags=['powder', 'bed', 'data']
)

def check_kafka_topic(**context):
    """
    Check if Kafka topic has new messages.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize Kafka ingester
        kafka_ingester = KafkaIngester()
        
        # Check topic for new messages
        message_count = kafka_ingester.check_topic_message_count(
            topic='powder_bed_monitoring_events',
            consumer_group='powder_bed_etl_consumer'
        )
        
        if message_count == 0:
            print("No new messages in Kafka topic")
            return 0
        
        print(f"Found {message_count} new messages in Kafka topic")
        return message_count
        
    except Exception as e:
        print(f"Error checking Kafka topic: {e}")
        raise

def ingest_powder_bed_data(**context):
    """
    Ingest powder bed monitoring data from Kafka.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize Kafka ingester
        kafka_ingester = KafkaIngester()
        
        # Ingest data from Kafka
        data = kafka_ingester.ingest_from_kafka(
            topic='powder_bed_monitoring_events',
            consumer_group='powder_bed_etl_consumer',
            batch_size=20000
        )
        
        # Store data in XCom for next task
        context['task_instance'].xcom_push(key='ingested_data', value=data)
        
        print(f"Ingested {len(data)} powder bed monitoring records")
        return len(data)
        
    except Exception as e:
        print(f"Error ingesting powder bed data: {e}")
        raise

def process_powder_bed_data(**context):
    """
    Process powder bed monitoring data using streaming processor.
    """
    try:
        # Get ingested data from XCom
        ingested_data = context['task_instance'].xcom_pull(key='ingested_data')
        
        if not ingested_data:
            print("No data to process")
            return 0
        
        # Initialize streaming processor
        streaming_processor = UnifiedStreamingProcessor()
        
        # Process data
        processed_data = streaming_processor.process_powder_bed_data(ingested_data)
        
        # Store processed data in XCom
        context['task_instance'].xcom_push(key='processed_data', value=processed_data)
        
        print(f"Processed {len(processed_data)} powder bed monitoring records")
        return len(processed_data)
        
    except Exception as e:
        print(f"Error processing powder bed data: {e}")
        raise

def analyze_surface_quality(**context):
    """
    Analyze surface quality in powder bed data.
    """
    try:
        # Get processed data from XCom
        processed_data = context['task_instance'].xcom_pull(key='processed_data')
        
        if not processed_data:
            print("No data to analyze for surface quality")
            return 0
        
        # Initialize surface quality analyzer
        from src.data_pipeline.quality.validation.surface_quality_analyzer import SurfaceQualityAnalyzer
        surface_analyzer = SurfaceQualityAnalyzer()
        
        # Analyze surface quality
        surface_analysis = surface_analyzer.analyze_powder_bed_surface_quality(processed_data)
        
        # Store surface analysis in XCom
        context['task_instance'].xcom_push(key='surface_analysis', value=surface_analysis)
        
        print(f"Analyzed surface quality in {len(processed_data)} powder bed records")
        return len(surface_analysis)
        
    except Exception as e:
        print(f"Error analyzing surface quality: {e}")
        raise

def load_powder_bed_data(**context):
    """
    Load processed powder bed monitoring data to Snowflake.
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
            table_name='fct_powder_bed',
            data=processed_data
        )
        
        if success:
            print(f"Loaded {len(processed_data)} powder bed monitoring records to Snowflake")
            return len(processed_data)
        else:
            raise Exception("Failed to load data to Snowflake")
            
    except Exception as e:
        print(f"Error loading powder bed data: {e}")
        raise

def validate_powder_bed_data(**context):
    """
    Validate powder bed monitoring data quality.
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
        validation_results = quality_service.validate_powder_bed_data(processed_data)
        
        # Check if validation passed
        if validation_results['overall_quality_score'] < 0.8:
            raise Exception(f"Data quality validation failed. Score: {validation_results['overall_quality_score']}")
        
        print(f"Data quality validation passed. Score: {validation_results['overall_quality_score']}")
        return validation_results['overall_quality_score']
        
    except Exception as e:
        print(f"Error validating powder bed data: {e}")
        raise

def send_quality_alerts(**context):
    """
    Send alerts for surface quality issues.
    """
    try:
        # Get surface analysis from XCom
        surface_analysis = context['task_instance'].xcom_pull(key='surface_analysis')
        
        if not surface_analysis:
            print("No surface quality issues to alert on")
            return 0
        
        # Initialize alert manager
        from src.data_pipeline.orchestration.monitoring.alert_manager import AlertManager
        alert_manager = AlertManager()
        
        # Send alerts for quality issues
        for analysis in surface_analysis:
            if analysis['quality_score'] < 0.7:
                alert_manager.send_surface_quality_alert(analysis)
        
        print(f"Sent alerts for {len([a for a in surface_analysis if a['quality_score'] < 0.7])} surface quality issues")
        return len([a for a in surface_analysis if a['quality_score'] < 0.7])
        
    except Exception as e:
        print(f"Error sending quality alerts: {e}")
        raise

def archive_powder_bed_data(**context):
    """
    Archive processed powder bed monitoring data.
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
        archive_path = f"archive/powder_bed/{datetime.now().strftime('%Y/%m/%d')}/powder_bed_data.json"
        success = archiver.archive_data(
            data=processed_data,
            archive_path=archive_path,
            content_type="application/json"
        )
        
        if success:
            print(f"Archived {len(processed_data)} powder bed monitoring records to S3")
            return len(processed_data)
        else:
            raise Exception("Failed to archive data to S3")
            
    except Exception as e:
        print(f"Error archiving powder bed data: {e}")
        raise

def send_completion_notification(**context):
    """
    Send completion notification.
    """
    try:
        # Get task results
        ingested_count = context['task_instance'].xcom_pull(task_ids='ingest_powder_bed_data')
        processed_count = context['task_instance'].xcom_pull(task_ids='process_powder_bed_data')
        loaded_count = context['task_instance'].xcom_pull(task_ids='load_powder_bed_data')
        surface_analysis_count = context['task_instance'].xcom_pull(task_ids='analyze_surface_quality')
        quality_score = context['task_instance'].xcom_pull(task_ids='validate_powder_bed_data')
        
        # Send notification
        message = f"""
        Powder Bed Monitoring Data Pipeline Completed Successfully!
        
        Records Processed:
        - Ingested: {ingested_count}
        - Processed: {processed_count}
        - Loaded: {loaded_count}
        - Surface Quality Analyzed: {surface_analysis_count}
        - Quality Score: {quality_score}
        
        Timestamp: {datetime.now().isoformat()}
        """
        
        print(message)
        # Here you would send actual notification (email, Slack, etc.)
        
    except Exception as e:
        print(f"Error sending completion notification: {e}")

# Define tasks
start_task = EmptyOperator(
    task_id='start_powder_bed_pipeline',
    dag=dag
)

# Kafka topic check task
kafka_check_task = PythonOperator(
    task_id='check_kafka_topic',
    python_callable=check_kafka_topic,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data ingestion task
ingest_task = PythonOperator(
    task_id='ingest_powder_bed_data',
    python_callable=ingest_powder_bed_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data processing task
process_task = PythonOperator(
    task_id='process_powder_bed_data',
    python_callable=process_powder_bed_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Surface quality analysis task
surface_task = PythonOperator(
    task_id='analyze_surface_quality',
    python_callable=analyze_surface_quality,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data loading task
load_task = PythonOperator(
    task_id='load_powder_bed_data',
    python_callable=load_powder_bed_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data validation task
validate_task = PythonOperator(
    task_id='validate_powder_bed_data',
    python_callable=validate_powder_bed_data,
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

# Data archiving task
archive_task = PythonOperator(
    task_id='archive_powder_bed_data',
    python_callable=archive_powder_bed_data,
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
    task_id='end_powder_bed_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> kafka_check_task >> ingest_task >> process_task
process_task >> [load_task, validate_task, surface_task]
load_task >> archive_task
validate_task >> archive_task
surface_task >> alert_task >> archive_task
archive_task >> notification_task >> end_task

# Export the DAG
PowderBedDAG = dag
