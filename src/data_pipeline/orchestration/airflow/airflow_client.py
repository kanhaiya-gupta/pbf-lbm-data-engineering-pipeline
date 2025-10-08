"""
Airflow Client

This module provides Airflow client integration capabilities for the PBF-LB/M data pipeline.
"""

import requests
import json
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import base64

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirflowClient:
    """
    Airflow client for managing DAGs and tasks in the PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.airflow_config = self._load_airflow_config()
        self.base_url = self.airflow_config.get('base_url', 'http://localhost:8080')
        self.username = self.airflow_config.get('username', 'admin')
        self.password = self.airflow_config.get('password', 'admin')
        self.api_version = self.airflow_config.get('api_version', 'v1')
        self.session = requests.Session()
        self._authenticate()
    
    def _load_airflow_config(self) -> Dict[str, Any]:
        """Load Airflow configuration."""
        try:
            return self.config.get('airflow', {
                'base_url': 'http://localhost:8080',
                'username': 'admin',
                'password': 'admin',
                'api_version': 'v1',
                'timeout': 30,
                'retry_attempts': 3,
                'retry_delay': 5,
                'dag_sync_interval': 60,
                'task_timeout': 300,
                'max_active_runs': 1,
                'catchup': False,
                'schedule_interval': '@hourly'
            })
        except Exception as e:
            logger.error(f"Error loading Airflow configuration: {e}")
            return {}
    
    def _authenticate(self) -> bool:
        """Authenticate with Airflow API."""
        try:
            auth_url = f"{self.base_url}/api/{self.api_version}/auth/login"
            auth_data = {
                "username": self.username,
                "password": self.password
            }
            
            response = self.session.post(auth_url, json=auth_data, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                logger.info("Successfully authenticated with Airflow")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error authenticating with Airflow: {e}")
            return False
    
    def get_dag_status(self, dag_id: str) -> Dict[str, Any]:
        """Get DAG status."""
        try:
            url = f"{self.base_url}/api/{self.api_version}/dags/{dag_id}"
            response = self.session.get(url, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                dag_info = response.json()
                return {
                    'status': 'success',
                    'dag_id': dag_id,
                    'is_active': dag_info.get('is_active', False),
                    'is_paused': dag_info.get('is_paused', True),
                    'last_run': dag_info.get('last_run'),
                    'next_run': dag_info.get('next_run'),
                    'schedule_interval': dag_info.get('schedule_interval'),
                    'catchup': dag_info.get('catchup', False),
                    'max_active_runs': dag_info.get('max_active_runs', 1)
                }
            else:
                return {'status': 'error', 'error': f'Failed to get DAG status: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error getting DAG status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def trigger_dag(self, dag_id: str, conf: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Trigger a DAG run."""
        try:
            url = f"{self.base_url}/api/{self.api_version}/dags/{dag_id}/dagRuns"
            payload = {
                "conf": conf or {},
                "dag_run_id": f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            response = self.session.post(url, json=payload, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                dag_run = response.json()
                return {
                    'status': 'success',
                    'dag_id': dag_id,
                    'dag_run_id': dag_run.get('dag_run_id'),
                    'state': dag_run.get('state'),
                    'start_date': dag_run.get('start_date'),
                    'conf': dag_run.get('conf', {})
                }
            else:
                return {'status': 'error', 'error': f'Failed to trigger DAG: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error triggering DAG: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_dag_runs(self, dag_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get DAG runs."""
        try:
            url = f"{self.base_url}/api/{self.api_version}/dags/{dag_id}/dagRuns"
            params = {
                'limit': limit,
                'order_by': '-start_date'
            }
            
            response = self.session.get(url, params=params, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                dag_runs = response.json()
                return {
                    'status': 'success',
                    'dag_id': dag_id,
                    'dag_runs': dag_runs.get('dag_runs', []),
                    'total_entries': dag_runs.get('total_entries', 0)
                }
            else:
                return {'status': 'error', 'error': f'Failed to get DAG runs: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error getting DAG runs: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_task_instances(self, dag_id: str, dag_run_id: str) -> Dict[str, Any]:
        """Get task instances for a DAG run."""
        try:
            url = f"{self.base_url}/api/{self.api_version}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
            
            response = self.session.get(url, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                task_instances = response.json()
                return {
                    'status': 'success',
                    'dag_id': dag_id,
                    'dag_run_id': dag_run_id,
                    'task_instances': task_instances.get('task_instances', []),
                    'total_entries': task_instances.get('total_entries', 0)
                }
            else:
                return {'status': 'error', 'error': f'Failed to get task instances: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error getting task instances: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_task_logs(self, dag_id: str, task_id: str, dag_run_id: str, 
                     task_try_number: int = 1) -> Dict[str, Any]:
        """Get task logs."""
        try:
            url = f"{self.base_url}/api/{self.api_version}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{task_try_number}"
            
            response = self.session.get(url, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                logs = response.json()
                return {
                    'status': 'success',
                    'dag_id': dag_id,
                    'task_id': task_id,
                    'dag_run_id': dag_run_id,
                    'try_number': task_try_number,
                    'logs': logs.get('content', ''),
                    'continuation_token': logs.get('continuation_token')
                }
            else:
                return {'status': 'error', 'error': f'Failed to get task logs: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error getting task logs: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def pause_dag(self, dag_id: str) -> Dict[str, Any]:
        """Pause a DAG."""
        try:
            url = f"{self.base_url}/api/{self.api_version}/dags/{dag_id}"
            payload = {"is_paused": True}
            
            response = self.session.patch(url, json=payload, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                return {'status': 'success', 'message': f'DAG {dag_id} paused successfully'}
            else:
                return {'status': 'error', 'error': f'Failed to pause DAG: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error pausing DAG: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def unpause_dag(self, dag_id: str) -> Dict[str, Any]:
        """Unpause a DAG."""
        try:
            url = f"{self.base_url}/api/{self.api_version}/dags/{dag_id}"
            payload = {"is_paused": False}
            
            response = self.session.patch(url, json=payload, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                return {'status': 'success', 'message': f'DAG {dag_id} unpaused successfully'}
            else:
                return {'status': 'error', 'error': f'Failed to unpause DAG: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error unpausing DAG: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_dag_code(self, dag_id: str) -> Dict[str, Any]:
        """Get DAG code."""
        try:
            url = f"{self.base_url}/api/{self.api_version}/dags/{dag_id}/code"
            
            response = self.session.get(url, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                dag_code = response.json()
                return {
                    'status': 'success',
                    'dag_id': dag_id,
                    'code': dag_code.get('content', '')
                }
            else:
                return {'status': 'error', 'error': f'Failed to get DAG code: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error getting DAG code: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_airflow_health(self) -> Dict[str, Any]:
        """Get Airflow health status."""
        try:
            url = f"{self.base_url}/api/{self.api_version}/health"
            
            response = self.session.get(url, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                health_info = response.json()
                return {
                    'status': 'success',
                    'health': health_info.get('health', 'unknown'),
                    'metadatabase': health_info.get('metadatabase', {}),
                    'scheduler': health_info.get('scheduler', {})
                }
            else:
                return {'status': 'error', 'error': f'Failed to get health status: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error getting Airflow health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_all_dags(self, limit: int = 100) -> Dict[str, Any]:
        """Get all DAGs."""
        try:
            url = f"{self.base_url}/api/{self.api_version}/dags"
            params = {
                'limit': limit,
                'order_by': 'dag_id'
            }
            
            response = self.session.get(url, params=params, timeout=self.airflow_config.get('timeout', 30))
            
            if response.status_code == 200:
                dags_info = response.json()
                return {
                    'status': 'success',
                    'dags': dags_info.get('dags', []),
                    'total_entries': dags_info.get('total_entries', 0)
                }
            else:
                return {'status': 'error', 'error': f'Failed to get DAGs: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error getting all DAGs: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_dag_statistics(self) -> Dict[str, Any]:
        """Get DAG statistics."""
        try:
            all_dags = self.get_all_dags()
            if all_dags['status'] != 'success':
                return all_dags
            
            dags = all_dags['dags']
            stats = {
                'total_dags': len(dags),
                'active_dags': len([dag for dag in dags if dag.get('is_active', False)]),
                'paused_dags': len([dag for dag in dags if dag.get('is_paused', True)]),
                'dag_states': {},
                'dag_schedules': {}
            }
            
            # Count DAG states
            for dag in dags:
                state = dag.get('state', 'unknown')
                stats['dag_states'][state] = stats['dag_states'].get(state, 0) + 1
                
                schedule = dag.get('schedule_interval', 'unknown')
                stats['dag_schedules'][schedule] = stats['dag_schedules'].get(schedule, 0) + 1
            
            return {
                'status': 'success',
                'statistics': stats,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting DAG statistics: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def monitor_dag_execution(self, dag_id: str, timeout_minutes: int = 60) -> Dict[str, Any]:
        """Monitor DAG execution until completion or timeout."""
        try:
            start_time = datetime.now()
            timeout = timedelta(minutes=timeout_minutes)
            
            while datetime.now() - start_time < timeout:
                # Get latest DAG run
                dag_runs = self.get_dag_runs(dag_id, limit=1)
                if dag_runs['status'] != 'success':
                    return dag_runs
                
                if not dag_runs['dag_runs']:
                    return {'status': 'error', 'error': 'No DAG runs found'}
                
                latest_run = dag_runs['dag_runs'][0]
                state = latest_run.get('state', 'unknown')
                
                if state in ['success', 'failed', 'upstream_failed']:
                    return {
                        'status': 'completed',
                        'dag_id': dag_id,
                        'dag_run_id': latest_run.get('dag_run_id'),
                        'final_state': state,
                        'start_date': latest_run.get('start_date'),
                        'end_date': latest_run.get('end_date'),
                        'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
                    }
                
                # Wait before checking again
                import time
                time.sleep(30)  # Check every 30 seconds
            
            return {
                'status': 'timeout',
                'dag_id': dag_id,
                'timeout_minutes': timeout_minutes,
                'message': f'DAG execution monitoring timed out after {timeout_minutes} minutes'
            }
            
        except Exception as e:
            logger.error(f"Error monitoring DAG execution: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health based on DAG statuses."""
        try:
            # Get all DAGs
            all_dags = self.get_all_dags()
            if all_dags['status'] != 'success':
                return all_dags
            
            dags = all_dags['dags']
            pipeline_dags = [dag for dag in dags if dag['dag_id'].startswith('pbf_') or dag['dag_id'].startswith('ispm_') or 
                           dag['dag_id'].startswith('ct_') or dag['dag_id'].startswith('powder_')]
            
            health_summary = {
                'total_pipeline_dags': len(pipeline_dags),
                'healthy_dags': 0,
                'unhealthy_dags': 0,
                'dag_details': [],
                'overall_health': 'healthy'
            }
            
            for dag in pipeline_dags:
                dag_id = dag['dag_id']
                is_active = dag.get('is_active', False)
                is_paused = dag.get('is_paused', True)
                
                # Get recent runs
                recent_runs = self.get_dag_runs(dag_id, limit=5)
                recent_success_rate = 0
                
                if recent_runs['status'] == 'success' and recent_runs['dag_runs']:
                    successful_runs = len([run for run in recent_runs['dag_runs'] if run.get('state') == 'success'])
                    recent_success_rate = (successful_runs / len(recent_runs['dag_runs'])) * 100
                
                dag_health = {
                    'dag_id': dag_id,
                    'is_active': is_active,
                    'is_paused': is_paused,
                    'recent_success_rate': recent_success_rate,
                    'health_status': 'healthy' if is_active and not is_paused and recent_success_rate >= 80 else 'unhealthy'
                }
                
                health_summary['dag_details'].append(dag_health)
                
                if dag_health['health_status'] == 'healthy':
                    health_summary['healthy_dags'] += 1
                else:
                    health_summary['unhealthy_dags'] += 1
            
            # Determine overall health
            if health_summary['unhealthy_dags'] > 0:
                health_summary['overall_health'] = 'degraded' if health_summary['unhealthy_dags'] < health_summary['total_pipeline_dags'] / 2 else 'unhealthy'
            
            return {
                'status': 'success',
                'pipeline_health': health_summary,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def close(self):
        """Close the Airflow client session."""
        try:
            if self.session:
                self.session.close()
                logger.info("Airflow client session closed")
        except Exception as e:
            logger.error(f"Error closing Airflow client: {e}")
