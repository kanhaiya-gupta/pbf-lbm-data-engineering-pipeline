"""
ISPM Monitoring Data Pipeline DAG

This DAG orchestrates the ISPM monitoring data pipeline including real-time
streaming ingestion, processing, quality checks, and storage.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageSensor
from .email_notifications import get_ispm_monitoring_notifier
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
email_notifier = get_ispm_monitoring_notifier()

# Create the DAG
dag = DAG(
    'ispm_monitoring_data_pipeline',
    default_args=default_args,
    description='ISPM monitoring data pipeline DAG',
    schedule='0 */15 * * *',  # Every 15 minutes
    max_active_runs=1,
    max_active_tasks=5,
    tags=['ispm', 'monitoring', 'data']
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
            topic='ispm_monitoring_events',
            consumer_group='ispm_monitoring_etl_consumer'
        )
        
        if message_count == 0:
            print("No new messages in Kafka topic")
            return 0
        
        print(f"Found {message_count} new messages in Kafka topic")
        return message_count
        
    except Exception as e:
        print(f"Error checking Kafka topic: {e}")
        raise

def ingest_ispm_monitoring_data(**context):
    """
    Ingest ISPM monitoring data from Kafka.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize Kafka ingester
        kafka_ingester = KafkaIngester()
        
        # Ingest data from Kafka
        data = kafka_ingester.ingest_from_kafka(
            topic='ispm_monitoring_events',
            consumer_group='ispm_monitoring_etl_consumer',
            batch_size=10000
        )
        
        # Store data in XCom for next task
        context['task_instance'].xcom_push(key='ingested_data', value=data)
        
        print(f"Ingested {len(data)} ISPM monitoring records")
        return len(data)
        
    except Exception as e:
        print(f"Error ingesting ISPM monitoring data: {e}")
        raise

def process_ispm_monitoring_data(**context):
    """
    Process ISPM monitoring data using streaming processor.
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
        processed_data = streaming_processor.process_ispm_monitoring_data(ingested_data)
        
        # Store processed data in XCom
        context['task_instance'].xcom_push(key='processed_data', value=processed_data)
        
        print(f"Processed {len(processed_data)} ISPM monitoring records")
        return len(processed_data)
        
    except Exception as e:
        print(f"Error processing ISPM monitoring data: {e}")
        raise

def detect_anomalies(**context):
    """
    Detect anomalies in ISPM monitoring data.
    """
    try:
        # Get processed data from XCom
        processed_data = context['task_instance'].xcom_pull(key='processed_data')
        
        if not processed_data:
            print("No data to analyze for anomalies")
            return 0
        
        # Initialize anomaly detector
        from src.data_pipeline.quality.validation.anomaly_detector import AnomalyDetector
        anomaly_detector = AnomalyDetector()
        
        # Detect anomalies
        anomalies = anomaly_detector.detect_ispm_anomalies(processed_data)
        
        # Store anomalies in XCom
        context['task_instance'].xcom_push(key='anomalies', value=anomalies)
        
        print(f"Detected {len(anomalies)} anomalies in ISPM monitoring data")
        return len(anomalies)
        
    except Exception as e:
        print(f"Error detecting anomalies: {e}")
        raise

def load_ispm_monitoring_data(**context):
    """
    Load processed ISPM monitoring data to Snowflake.
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
            table_name='fct_ispm_monitoring',
            data=processed_data
        )
        
        if success:
            print(f"Loaded {len(processed_data)} ISPM monitoring records to Snowflake")
            return len(processed_data)
        else:
            raise Exception("Failed to load data to Snowflake")
            
    except Exception as e:
        print(f"Error loading ISPM monitoring data: {e}")
        raise

def send_anomaly_alerts(**context):
    """
    Send alerts for detected anomalies.
    """
    try:
        # Get anomalies from XCom
        anomalies = context['task_instance'].xcom_pull(key='anomalies')
        
        if not anomalies:
            print("No anomalies to alert on")
            return 0
        
        # Initialize alert manager
        from src.data_pipeline.orchestration.monitoring.alert_manager import AlertManager
        alert_manager = AlertManager()
        
        # Send alerts for anomalies
        for anomaly in anomalies:
            alert_manager.send_anomaly_alert(anomaly)
        
        print(f"Sent alerts for {len(anomalies)} anomalies")
        return len(anomalies)
        
    except Exception as e:
        print(f"Error sending anomaly alerts: {e}")
        raise

def validate_ispm_monitoring_data(**context):
    """
    Validate ISPM monitoring data quality.
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
        validation_results = quality_service.validate_ispm_monitoring_data(processed_data)
        
        # Check if validation passed
        if validation_results['overall_quality_score'] < 0.9:
            raise Exception(f"Data quality validation failed. Score: {validation_results['overall_quality_score']}")
        
        print(f"Data quality validation passed. Score: {validation_results['overall_quality_score']}")
        return validation_results['overall_quality_score']
        
    except Exception as e:
        print(f"Error validating ISPM monitoring data: {e}")
        raise

def send_completion_notification(**context):
    """
    Send completion notification.
    """
    try:
        # Get task results
        ingested_count = context['task_instance'].xcom_pull(task_ids='ingest_ispm_monitoring_data')
        processed_count = context['task_instance'].xcom_pull(task_ids='process_ispm_monitoring_data')
        loaded_count = context['task_instance'].xcom_pull(task_ids='load_ispm_monitoring_data')
        anomalies_count = context['task_instance'].xcom_pull(task_ids='detect_anomalies')
        quality_score = context['task_instance'].xcom_pull(task_ids='validate_ispm_monitoring_data')
        
        # Send notification
        message = f"""
        ISPM Monitoring Data Pipeline Completed Successfully!
        
        Records Processed:
        - Ingested: {ingested_count}
        - Processed: {processed_count}
        - Loaded: {loaded_count}
        - Anomalies Detected: {anomalies_count}
        - Quality Score: {quality_score}
        
        Timestamp: {datetime.now().isoformat()}
        """
        
        print(message)
        # Here you would send actual notification (email, Slack, etc.)
        
    except Exception as e:
        print(f"Error sending completion notification: {e}")

# Define tasks
start_task = EmptyOperator(
    task_id='start_ispm_monitoring_pipeline',
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
    task_id='ingest_ispm_monitoring_data',
    python_callable=ingest_ispm_monitoring_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data processing task
process_task = PythonOperator(
    task_id='process_ispm_monitoring_data',
    python_callable=process_ispm_monitoring_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Anomaly detection task
anomaly_task = PythonOperator(
    task_id='detect_anomalies',
    python_callable=detect_anomalies,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data loading task
load_task = PythonOperator(
    task_id='load_ispm_monitoring_data',
    python_callable=load_ispm_monitoring_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Data validation task
validate_task = PythonOperator(
    task_id='validate_ispm_monitoring_data',
    python_callable=validate_ispm_monitoring_data,
    on_failure_callback=email_notifier,
    dag=dag
)

# Anomaly alerting task
alert_task = PythonOperator(
    task_id='send_anomaly_alerts',
    python_callable=send_anomaly_alerts,
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
    task_id='end_ispm_monitoring_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> kafka_check_task >> ingest_task >> process_task
process_task >> [load_task, validate_task, anomaly_task]
load_task >> notification_task
validate_task >> notification_task
anomaly_task >> alert_task >> notification_task
notification_task >> end_task

# Export the DAG
ISPMMonitoringDAG = dag
