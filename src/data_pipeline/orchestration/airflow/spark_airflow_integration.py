"""
Spark-Airflow Integration

This module provides integration between Apache Spark and Apache Airflow for the PBF-LB/M data pipeline.
"""

import os
import subprocess
import json
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import tempfile

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparkAirflowIntegration:
    """
    Integration between Apache Spark and Apache Airflow for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.integration_config = self._load_integration_config()
        self.spark_config = self._load_spark_config()
        self.airflow_config = self._load_airflow_config()
    
    def _load_integration_config(self) -> Dict[str, Any]:
        """Load Spark-Airflow integration configuration."""
        try:
            return self.config.get('spark_airflow_integration', {
                'enabled': True,
                'spark_submit_path': 'spark-submit',
                'spark_apps_path': './spark_apps',
                'spark_logs_path': './spark_logs',
                'airflow_connection_id': 'spark_default',
                'spark_master': 'local[*]',
                'spark_driver_memory': '2g',
                'spark_executor_memory': '2g',
                'spark_executor_cores': '2',
                'spark_max_result_size': '1g',
                'spark_sql_adaptive_enabled': 'true',
                'spark_sql_adaptive_coalesce_partitions_enabled': 'true',
                'spark_dynamic_allocation_enabled': 'true',
                'spark_dynamic_allocation_min_executors': '1',
                'spark_dynamic_allocation_max_executors': '10',
                'spark_dynamic_allocation_initial_executors': '2',
                'spark_apps': {
                    'pbf_process_etl': {
                        'app_name': 'PBFProcessETL',
                        'main_class': 'com.pbflbm.etl.PBFProcessETL',
                        'jar_path': './jars/pbf-process-etl.jar',
                        'enabled': True
                    },
                    'ispm_monitoring_etl': {
                        'app_name': 'ISPMMonitoringETL',
                        'main_class': 'com.pbflbm.etl.ISPMMonitoringETL',
                        'jar_path': './jars/ispm-monitoring-etl.jar',
                        'enabled': True
                    },
                    'ct_scan_etl': {
                        'app_name': 'CTScanETL',
                        'main_class': 'com.pbflbm.etl.CTScanETL',
                        'jar_path': './jars/ct-scan-etl.jar',
                        'enabled': True
                    },
                    'powder_bed_etl': {
                        'app_name': 'PowderBedETL',
                        'main_class': 'com.pbflbm.etl.PowderBedETL',
                        'jar_path': './jars/powder-bed-etl.jar',
                        'enabled': True
                    }
                }
            })
        except Exception as e:
            logger.error(f"Error loading integration configuration: {e}")
            return {}
    
    def _load_spark_config(self) -> Dict[str, Any]:
        """Load Spark configuration."""
        try:
            return self.config.get('spark', {
                'master': 'local[*]',
                'app_name': 'PBF-LB/M Data Pipeline',
                'driver_memory': '2g',
                'executor_memory': '2g',
                'executor_cores': '2',
                'max_result_size': '1g',
                'sql_adaptive_enabled': True,
                'sql_adaptive_coalesce_partitions_enabled': True,
                'dynamic_allocation_enabled': True,
                'dynamic_allocation_min_executors': 1,
                'dynamic_allocation_max_executors': 10,
                'dynamic_allocation_initial_executors': 2
            })
        except Exception as e:
            logger.error(f"Error loading Spark configuration: {e}")
            return {}
    
    def _load_airflow_config(self) -> Dict[str, Any]:
        """Load Airflow configuration."""
        try:
            return self.config.get('airflow', {
                'base_url': 'http://localhost:8080',
                'username': 'admin',
                'password': 'admin',
                'api_version': 'v1',
                'connection_id': 'spark_default'
            })
        except Exception as e:
            logger.error(f"Error loading Airflow configuration: {e}")
            return {}
    
    def submit_spark_job(self, app_name: str, job_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Submit a Spark job."""
        try:
            if not self.integration_config.get('enabled', True):
                return {'status': 'disabled', 'message': 'Spark-Airflow integration is disabled'}
            
            if app_name not in self.integration_config.get('spark_apps', {}):
                return {'status': 'error', 'error': f'Unknown Spark app: {app_name}'}
            
            app_config = self.integration_config['spark_apps'][app_name]
            if not app_config.get('enabled', True):
                return {'status': 'disabled', 'message': f'Spark app {app_name} is disabled'}
            
            # Build Spark submit command
            spark_submit_cmd = self._build_spark_submit_command(app_name, app_config, job_config)
            
            # Submit job
            result = self._execute_spark_submit(spark_submit_cmd, app_name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error submitting Spark job: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _build_spark_submit_command(self, app_name: str, app_config: Dict[str, Any], 
                                  job_config: Optional[Dict[str, Any]]) -> List[str]:
        """Build Spark submit command."""
        try:
            cmd = [self.integration_config.get('spark_submit_path', 'spark-submit')]
            
            # Add Spark configuration
            spark_configs = {
                '--master': self.integration_config.get('spark_master', 'local[*]'),
                '--driver-memory': self.integration_config.get('spark_driver_memory', '2g'),
                '--executor-memory': self.integration_config.get('spark_executor_memory', '2g'),
                '--executor-cores': self.integration_config.get('spark_executor_cores', '2'),
                '--conf': f"spark.maxResultSize={self.integration_config.get('spark_max_result_size', '1g')}",
                '--conf': f"spark.sql.adaptive.enabled={self.integration_config.get('spark_sql_adaptive_enabled', 'true')}",
                '--conf': f"spark.sql.adaptive.coalescePartitions.enabled={self.integration_config.get('spark_sql_adaptive_coalesce_partitions_enabled', 'true')}",
                '--conf': f"spark.dynamicAllocation.enabled={self.integration_config.get('spark_dynamic_allocation_enabled', 'true')}",
                '--conf': f"spark.dynamicAllocation.minExecutors={self.integration_config.get('spark_dynamic_allocation_min_executors', '1')}",
                '--conf': f"spark.dynamicAllocation.maxExecutors={self.integration_config.get('spark_dynamic_allocation_max_executors', '10')}",
                '--conf': f"spark.dynamicAllocation.initialExecutors={self.integration_config.get('spark_dynamic_allocation_initial_executors', '2')}"
            }
            
            for key, value in spark_configs.items():
                cmd.extend([key, value])
            
            # Add application name
            cmd.extend(['--name', app_config.get('app_name', app_name)])
            
            # Add main class
            if 'main_class' in app_config:
                cmd.extend(['--class', app_config['main_class']])
            
            # Add JAR path
            jar_path = app_config.get('jar_path')
            if jar_path and os.path.exists(jar_path):
                cmd.append(jar_path)
            else:
                raise FileNotFoundError(f"JAR file not found: {jar_path}")
            
            # Add job configuration as arguments
            if job_config:
                for key, value in job_config.items():
                    cmd.extend([f'--{key}', str(value)])
            
            return cmd
            
        except Exception as e:
            logger.error(f"Error building Spark submit command: {e}")
            raise
    
    def _execute_spark_submit(self, cmd: List[str], app_name: str) -> Dict[str, Any]:
        """Execute Spark submit command."""
        try:
            # Create logs directory if it doesn't exist
            logs_path = self.integration_config.get('spark_logs_path', './spark_logs')
            os.makedirs(logs_path, exist_ok=True)
            
            # Create log file
            log_file = os.path.join(logs_path, f"{app_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            
            # Execute command
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Wait for process to complete
                return_code = process.wait()
                
                # Read log file
                with open(log_file, 'r') as f:
                    log_content = f.read()
                
                if return_code == 0:
                    return {
                        'status': 'success',
                        'app_name': app_name,
                        'return_code': return_code,
                        'log_file': log_file,
                        'message': 'Spark job completed successfully'
                    }
                else:
                    return {
                        'status': 'failed',
                        'app_name': app_name,
                        'return_code': return_code,
                        'log_file': log_file,
                        'error': f'Spark job failed with return code {return_code}',
                        'logs': log_content
                    }
                    
        except Exception as e:
            logger.error(f"Error executing Spark submit: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def create_spark_airflow_operator(self, app_name: str, task_id: str, 
                                    dag_id: str, job_config: Optional[Dict[str, Any]] = None) -> str:
        """Create Spark Airflow operator code."""
        try:
            if app_name not in self.integration_config.get('spark_apps', {}):
                raise ValueError(f'Unknown Spark app: {app_name}')
            
            app_config = self.integration_config['spark_apps'][app_name]
            
            # Generate operator code
            operator_code = f'''
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

# Default arguments
default_args = {{
    'owner': 'pbf-lbm-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    # Email settings removed - use SmtpNotifier in Airflow 3.x
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}}

# DAG definition
dag = DAG(
    '{dag_id}',
    default_args=default_args,
    description='{app_config.get("app_name", app_name)} Spark Job',
    schedule_interval='@hourly',
    catchup=False,
    max_active_runs=1
)

# Spark submit operator
spark_job = SparkSubmitOperator(
    task_id='{task_id}',
    application='{app_config.get("jar_path", "")}',
    name='{app_config.get("app_name", app_name)}',
    conn_id='{self.integration_config.get("airflow_connection_id", "spark_default")}',
    conf={{
        'spark.master': '{self.integration_config.get("spark_master", "local[*]")}',
        'spark.driver.memory': '{self.integration_config.get("spark_driver_memory", "2g")}',
        'spark.executor.memory': '{self.integration_config.get("spark_executor_memory", "2g")}',
        'spark.executor.cores': '{self.integration_config.get("spark_executor_cores", "2")}',
        'spark.maxResultSize': '{self.integration_config.get("spark_max_result_size", "1g")}',
        'spark.sql.adaptive.enabled': '{self.integration_config.get("spark_sql_adaptive_enabled", "true")}',
        'spark.sql.adaptive.coalescePartitions.enabled': '{self.integration_config.get("spark_sql_adaptive_coalesce_partitions_enabled", "true")}',
        'spark.dynamicAllocation.enabled': '{self.integration_config.get("spark_dynamic_allocation_enabled", "true")}',
        'spark.dynamicAllocation.minExecutors': '{self.integration_config.get("spark_dynamic_allocation_min_executors", "1")}',
        'spark.dynamicAllocation.maxExecutors': '{self.integration_config.get("spark_dynamic_allocation_max_executors", "10")}',
        'spark.dynamicAllocation.initialExecutors': '{self.integration_config.get("spark_dynamic_allocation_initial_executors", "2")}'
    }},
    java_class='{app_config.get("main_class", "")}',
    dag=dag
)
'''
            
            # Add job configuration if provided
            if job_config:
                config_args = ', '.join([f'--{key} {value}' for key, value in job_config.items()])
                operator_code += f'''
# Add job configuration arguments
spark_job.application_args = ['{config_args}']
'''
            
            return operator_code
            
        except Exception as e:
            logger.error(f"Error creating Spark Airflow operator: {e}")
            raise
    
    def get_spark_job_status(self, app_name: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get Spark job status."""
        try:
            if app_name not in self.integration_config.get('spark_apps', {}):
                return {'status': 'error', 'error': f'Unknown Spark app: {app_name}'}
            
            app_config = self.integration_config['spark_apps'][app_name]
            
            # Check if JAR file exists
            jar_path = app_config.get('jar_path')
            if not jar_path or not os.path.exists(jar_path):
                return {
                    'status': 'error',
                    'error': f'JAR file not found: {jar_path}',
                    'app_name': app_name
                }
            
            # Check log files
            logs_path = self.integration_config.get('spark_logs_path', './spark_logs')
            if os.path.exists(logs_path):
                log_files = [f for f in os.listdir(logs_path) if f.startswith(app_name)]
                latest_log = max(log_files) if log_files else None
                
                if latest_log:
                    log_file_path = os.path.join(logs_path, latest_log)
                    with open(log_file_path, 'r') as f:
                        log_content = f.read()
                    
                    # Parse log for job status
                    if 'Job finished' in log_content or 'Job completed' in log_content:
                        status = 'completed'
                    elif 'Job failed' in log_content or 'Exception' in log_content:
                        status = 'failed'
                    else:
                        status = 'running'
                    
                    return {
                        'status': 'success',
                        'app_name': app_name,
                        'job_status': status,
                        'latest_log': latest_log,
                        'log_file': log_file_path
                    }
            
            return {
                'status': 'success',
                'app_name': app_name,
                'job_status': 'unknown',
                'message': 'No log files found'
            }
            
        except Exception as e:
            logger.error(f"Error getting Spark job status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_spark_applications(self) -> Dict[str, Any]:
        """Get all Spark applications."""
        try:
            spark_apps = self.integration_config.get('spark_apps', {})
            
            apps_info = []
            for app_name, app_config in spark_apps.items():
                app_info = {
                    'app_name': app_name,
                    'display_name': app_config.get('app_name', app_name),
                    'main_class': app_config.get('main_class', ''),
                    'jar_path': app_config.get('jar_path', ''),
                    'enabled': app_config.get('enabled', True),
                    'jar_exists': os.path.exists(app_config.get('jar_path', '')) if app_config.get('jar_path') else False
                }
                apps_info.append(app_info)
            
            return {
                'status': 'success',
                'applications': apps_info,
                'total_apps': len(apps_info),
                'enabled_apps': len([app for app in apps_info if app['enabled']]),
                'disabled_apps': len([app for app in apps_info if not app['enabled']])
            }
            
        except Exception as e:
            logger.error(f"Error getting Spark applications: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def validate_spark_setup(self) -> Dict[str, Any]:
        """Validate Spark setup."""
        try:
            validation_results = {
                'spark_submit_available': False,
                'spark_apps_valid': [],
                'spark_apps_invalid': [],
                'overall_status': 'unknown'
            }
            
            # Check if spark-submit is available
            try:
                result = subprocess.run(
                    [self.integration_config.get('spark_submit_path', 'spark-submit'), '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                validation_results['spark_submit_available'] = result.returncode == 0
            except Exception as e:
                logger.warning(f"Spark-submit not available: {e}")
            
            # Validate Spark applications
            spark_apps = self.integration_config.get('spark_apps', {})
            for app_name, app_config in spark_apps.items():
                app_validation = {
                    'app_name': app_name,
                    'jar_exists': False,
                    'jar_valid': False,
                    'main_class_defined': bool(app_config.get('main_class')),
                    'enabled': app_config.get('enabled', True)
                }
                
                jar_path = app_config.get('jar_path')
                if jar_path:
                    app_validation['jar_exists'] = os.path.exists(jar_path)
                    if app_validation['jar_exists']:
                        try:
                            # Check if JAR is valid (basic check)
                            with open(jar_path, 'rb') as f:
                                header = f.read(4)
                                app_validation['jar_valid'] = header == b'PK\x03\x04'  # ZIP/JAR header
                        except Exception:
                            app_validation['jar_valid'] = False
                
                if app_validation['jar_exists'] and app_validation['jar_valid'] and app_validation['main_class_defined']:
                    validation_results['spark_apps_valid'].append(app_validation)
                else:
                    validation_results['spark_apps_invalid'].append(app_validation)
            
            # Determine overall status
            if validation_results['spark_submit_available'] and validation_results['spark_apps_valid']:
                validation_results['overall_status'] = 'healthy'
            elif validation_results['spark_submit_available'] and validation_results['spark_apps_invalid']:
                validation_results['overall_status'] = 'degraded'
            else:
                validation_results['overall_status'] = 'unhealthy'
            
            return {
                'status': 'success',
                'validation_results': validation_results,
                'validated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating Spark setup: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        try:
            stats = {
                'configuration': self.integration_config.copy(),
                'spark_config': self.spark_config.copy(),
                'airflow_config': self.airflow_config.copy(),
                'applications': self.get_spark_applications(),
                'validation': self.validate_spark_setup()
            }
            
            return {
                'status': 'success',
                'statistics': stats,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting integration statistics: {e}")
            return {'status': 'error', 'error': str(e)}
