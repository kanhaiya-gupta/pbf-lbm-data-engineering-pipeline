"""
DBT Transformations Pipeline DAG

This DAG orchestrates the DBT transformations pipeline including
staging, intermediate, and mart model transformations.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from .email_notifications import get_dbt_notifier
from airflow.models import Variable
# days_ago removed in Airflow 3.x - using datetime calculation instead
from airflow.sdk import TaskGroup

# Import our custom modules
from src.data_pipeline.processing.dbt.dbt_orchestrator import DBTOrchestrator
from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime.now() - timedelta(days=1),
    'retries': 2,
    'retry_delay': timedelta(minutes=20),
    'catchup': False
}

# Email notification configuration using SmtpNotifier (Airflow 3.x)
email_notifier = get_dbt_notifier()

# Create the DAG
dag = DAG(
    'dbt_transformations_pipeline',
    default_args=default_args,
    description='DBT transformations pipeline DAG',
    schedule='0 3 * * *',  # Daily at 3 AM
    max_active_runs=1,
    max_active_tasks=5,
    tags=['dbt', 'transformations']
)

def run_dbt_staging_models(**context):
    """
    Run DBT staging models.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize DBT orchestrator
        dbt_orchestrator = DBTOrchestrator()
        
        # Run staging models
        staging_results = dbt_orchestrator.run_staging_models()
        
        # Store results in XCom
        context['task_instance'].xcom_push(key='staging_results', value=staging_results)
        
        print(f"DBT staging models completed. Models run: {staging_results['models_run']}")
        return staging_results['models_run']
        
    except Exception as e:
        print(f"Error running DBT staging models: {e}")
        raise

def run_dbt_intermediate_models(**context):
    """
    Run DBT intermediate models.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize DBT orchestrator
        dbt_orchestrator = DBTOrchestrator()
        
        # Run intermediate models
        intermediate_results = dbt_orchestrator.run_intermediate_models()
        
        # Store results in XCom
        context['task_instance'].xcom_push(key='intermediate_results', value=intermediate_results)
        
        print(f"DBT intermediate models completed. Models run: {intermediate_results['models_run']}")
        return intermediate_results['models_run']
        
    except Exception as e:
        print(f"Error running DBT intermediate models: {e}")
        raise

def run_dbt_mart_models(**context):
    """
    Run DBT mart models.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize DBT orchestrator
        dbt_orchestrator = DBTOrchestrator()
        
        # Run mart models
        mart_results = dbt_orchestrator.run_mart_models()
        
        # Store results in XCom
        context['task_instance'].xcom_push(key='mart_results', value=mart_results)
        
        print(f"DBT mart models completed. Models run: {mart_results['models_run']}")
        return mart_results['models_run']
        
    except Exception as e:
        print(f"Error running DBT mart models: {e}")
        raise

def run_dbt_tests(**context):
    """
    Run DBT tests.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize DBT orchestrator
        dbt_orchestrator = DBTOrchestrator()
        
        # Run DBT tests
        test_results = dbt_orchestrator.run_tests()
        
        # Store results in XCom
        context['task_instance'].xcom_push(key='test_results', value=test_results)
        
        print(f"DBT tests completed. Tests run: {test_results['tests_run']}, Passed: {test_results['tests_passed']}")
        return test_results['tests_run']
        
    except Exception as e:
        print(f"Error running DBT tests: {e}")
        raise

def run_dbt_docs(**context):
    """
    Generate DBT documentation.
    """
    try:
        # Get configuration
        config = get_pipeline_config()
        
        # Initialize DBT orchestrator
        dbt_orchestrator = DBTOrchestrator()
        
        # Generate DBT docs
        docs_results = dbt_orchestrator.generate_docs()
        
        # Store results in XCom
        context['task_instance'].xcom_push(key='docs_results', value=docs_results)
        
        print(f"DBT documentation generated successfully")
        return 1
        
    except Exception as e:
        print(f"Error generating DBT documentation: {e}")
        raise

def validate_dbt_models(**context):
    """
    Validate DBT model outputs.
    """
    try:
        # Get all DBT results from XCom
        staging_results = context['task_instance'].xcom_pull(key='staging_results')
        intermediate_results = context['task_instance'].xcom_pull(key='intermediate_results')
        mart_results = context['task_instance'].xcom_pull(key='mart_results')
        test_results = context['task_instance'].xcom_pull(key='test_results')
        
        # Initialize DBT orchestrator
        dbt_orchestrator = DBTOrchestrator()
        
        # Validate model outputs
        validation_results = dbt_orchestrator.validate_model_outputs()
        
        # Store validation results in XCom
        context['task_instance'].xcom_push(key='validation_results', value=validation_results)
        
        print(f"DBT model validation completed. Models validated: {validation_results['models_validated']}")
        return validation_results['models_validated']
        
    except Exception as e:
        print(f"Error validating DBT models: {e}")
        raise

def send_dbt_notification(**context):
    """
    Send DBT completion notification.
    """
    try:
        # Get task results
        staging_models = context['task_instance'].xcom_pull(task_ids='run_dbt_staging_models')
        intermediate_models = context['task_instance'].xcom_pull(task_ids='run_dbt_intermediate_models')
        mart_models = context['task_instance'].xcom_pull(task_ids='run_dbt_mart_models')
        tests_run = context['task_instance'].xcom_pull(task_ids='run_dbt_tests')
        models_validated = context['task_instance'].xcom_pull(task_ids='validate_dbt_models')
        
        # Send notification
        message = f"""
        DBT Transformations Pipeline Completed Successfully!
        
        DBT Results:
        - Staging Models Run: {staging_models}
        - Intermediate Models Run: {intermediate_models}
        - Mart Models Run: {mart_models}
        - Tests Run: {tests_run}
        - Models Validated: {models_validated}
        
        Timestamp: {datetime.now().isoformat()}
        """
        
        print(message)
        # Here you would send actual notification (email, Slack, etc.)
        
    except Exception as e:
        print(f"Error sending DBT notification: {e}")

# Define tasks
start_task = EmptyOperator(
    task_id='start_dbt_pipeline',
    dag=dag
)

# DBT staging models task
staging_task = PythonOperator(
    task_id='run_dbt_staging_models',
    python_callable=run_dbt_staging_models,
    on_failure_callback=email_notifier,
    dag=dag
)

# DBT intermediate models task
intermediate_task = PythonOperator(
    task_id='run_dbt_intermediate_models',
    python_callable=run_dbt_intermediate_models,
    on_failure_callback=email_notifier,
    dag=dag
)

# DBT mart models task
mart_task = PythonOperator(
    task_id='run_dbt_mart_models',
    python_callable=run_dbt_mart_models,
    on_failure_callback=email_notifier,
    dag=dag
)

# DBT tests task
tests_task = PythonOperator(
    task_id='run_dbt_tests',
    python_callable=run_dbt_tests,
    on_failure_callback=email_notifier,
    dag=dag
)

# DBT documentation task
docs_task = PythonOperator(
    task_id='run_dbt_docs',
    python_callable=run_dbt_docs,
    on_failure_callback=email_notifier,
    dag=dag
)

# DBT validation task
validation_task = PythonOperator(
    task_id='validate_dbt_models',
    python_callable=validate_dbt_models,
    on_failure_callback=email_notifier,
    dag=dag
)

# DBT notification task
notification_task = PythonOperator(
    task_id='send_dbt_notification',
    python_callable=send_dbt_notification,
    on_failure_callback=email_notifier,
    dag=dag
)

end_task = EmptyOperator(
    task_id='end_dbt_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> staging_task >> intermediate_task >> mart_task
mart_task >> [tests_task, docs_task, validation_task]
tests_task >> notification_task
docs_task >> notification_task
validation_task >> notification_task
notification_task >> end_task

# Export the DAG
DBTDAG = dag
