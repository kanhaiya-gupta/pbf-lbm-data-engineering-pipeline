"""
DBT Orchestrator for PBF-LB/M Data Pipeline

This module provides orchestration capabilities for DBT (Data Build Tool)
operations in the PBF-LB/M data pipeline.
"""

import logging
import subprocess
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DBTConfig:
    """Configuration for DBT operations."""
    project_dir: str
    profiles_dir: str
    target: str = "dev"
    threads: int = 4
    timeout: int = 300
    debug: bool = False
    full_refresh: bool = False
    select: Optional[str] = None
    exclude: Optional[str] = None
    vars: Optional[Dict[str, Any]] = None


@dataclass
class DBTResult:
    """Result of DBT operation."""
    success: bool
    command: str
    output: str
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
    models_run: int = 0
    models_failed: int = 0
    tests_run: int = 0
    tests_failed: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DBTOrchestrator:
    """
    DBT orchestrator for PBF-LB/M data pipeline.
    
    This orchestrator manages DBT operations including running models,
    tests, seeds, and snapshots for the PBF-LB/M data warehouse.
    """
    
    def __init__(self, config: Optional[DBTConfig] = None):
        """
        Initialize the DBT orchestrator.
        
        Args:
            config: DBT configuration
        """
        self.config = config or DBTConfig(
            project_dir=str(Path(__file__).parent),
            profiles_dir=str(Path(__file__).parent)
        )
        
        # Validate DBT installation
        self._validate_dbt_installation()
        
        logger.info(f"DBT Orchestrator initialized with project dir: {self.config.project_dir}")
    
    def _validate_dbt_installation(self) -> bool:
        """Validate that DBT is installed and accessible."""
        try:
            result = subprocess.run(
                ['dbt', '--version'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info(f"DBT version: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"DBT validation failed: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.error("DBT not found. Please install DBT: pip install dbt-core dbt-postgres")
            return False
        except subprocess.TimeoutExpired:
            logger.error("DBT validation timed out")
            return False
        except Exception as e:
            logger.error(f"Error validating DBT installation: {e}")
            return False
    
    def run_models(self, select: Optional[str] = None, 
                   exclude: Optional[str] = None,
                   full_refresh: bool = False) -> DBTResult:
        """
        Run DBT models.
        
        Args:
            select: Select specific models to run
            exclude: Exclude specific models
            full_refresh: Force full refresh of models
            
        Returns:
            DBTResult: Result of the DBT run operation
        """
        start_time = datetime.now()
        
        try:
            logger.info("Starting DBT models run")
            
            # Build command
            cmd = ['dbt', 'run']
            
            # Add project and profiles directories
            cmd.extend(['--project-dir', self.config.project_dir])
            cmd.extend(['--profiles-dir', self.config.profiles_dir])
            cmd.extend(['--target', self.config.target])
            cmd.extend(['--threads', str(self.config.threads)])
            
            # Add optional parameters
            if full_refresh or self.config.full_refresh:
                cmd.append('--full-refresh')
            
            if select or self.config.select:
                cmd.extend(['--select', select or self.config.select])
            
            if exclude or self.config.exclude:
                cmd.extend(['--exclude', exclude or self.config.exclude])
            
            if self.config.vars:
                vars_str = ' '.join([f"{k}:{v}" for k, v in self.config.vars.items()])
                cmd.extend(['--vars', vars_str])
            
            if self.config.debug:
                cmd.append('--debug')
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=self.config.project_dir
            )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Parse output for statistics
            models_run, models_failed = self._parse_run_output(result.stdout)
            
            # Create result
            dbt_result = DBTResult(
                success=result.returncode == 0,
                command=' '.join(cmd),
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                execution_time_seconds=execution_time,
                models_run=models_run,
                models_failed=models_failed,
                metadata={
                    'select': select or self.config.select,
                    'exclude': exclude or self.config.exclude,
                    'full_refresh': full_refresh or self.config.full_refresh,
                    'target': self.config.target,
                    'threads': self.config.threads
                }
            )
            
            if dbt_result.success:
                logger.info(f"DBT models run completed successfully: {models_run} models run, "
                           f"{models_failed} failed in {execution_time:.2f}s")
            else:
                logger.error(f"DBT models run failed: {result.stderr}")
            
            return dbt_result
            
        except subprocess.TimeoutExpired:
            logger.error("DBT models run timed out")
            return DBTResult(
                success=False,
                command=' '.join(cmd) if 'cmd' in locals() else 'dbt run',
                output="",
                error="Operation timed out",
                execution_time_seconds=(datetime.now() - start_time).total_seconds()
            )
        except Exception as e:
            logger.error(f"Error running DBT models: {e}")
            return DBTResult(
                success=False,
                command='dbt run',
                output="",
                error=str(e),
                execution_time_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    def run_tests(self, select: Optional[str] = None, 
                  exclude: Optional[str] = None) -> DBTResult:
        """
        Run DBT tests.
        
        Args:
            select: Select specific tests to run
            exclude: Exclude specific tests
            
        Returns:
            DBTResult: Result of the DBT test operation
        """
        start_time = datetime.now()
        
        try:
            logger.info("Starting DBT tests run")
            
            # Build command
            cmd = ['dbt', 'test']
            
            # Add project and profiles directories
            cmd.extend(['--project-dir', self.config.project_dir])
            cmd.extend(['--profiles-dir', self.config.profiles_dir])
            cmd.extend(['--target', self.config.target])
            cmd.extend(['--threads', str(self.config.threads)])
            
            # Add optional parameters
            if select or self.config.select:
                cmd.extend(['--select', select or self.config.select])
            
            if exclude or self.config.exclude:
                cmd.extend(['--exclude', exclude or self.config.exclude])
            
            if self.config.vars:
                vars_str = ' '.join([f"{k}:{v}" for k, v in self.config.vars.items()])
                cmd.extend(['--vars', vars_str])
            
            if self.config.debug:
                cmd.append('--debug')
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=self.config.project_dir
            )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Parse output for statistics
            tests_run, tests_failed = self._parse_test_output(result.stdout)
            
            # Create result
            dbt_result = DBTResult(
                success=result.returncode == 0,
                command=' '.join(cmd),
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                execution_time_seconds=execution_time,
                tests_run=tests_run,
                tests_failed=tests_failed,
                metadata={
                    'select': select or self.config.select,
                    'exclude': exclude or self.config.exclude,
                    'target': self.config.target,
                    'threads': self.config.threads
                }
            )
            
            if dbt_result.success:
                logger.info(f"DBT tests completed successfully: {tests_run} tests run, "
                           f"{tests_failed} failed in {execution_time:.2f}s")
            else:
                logger.error(f"DBT tests failed: {result.stderr}")
            
            return dbt_result
            
        except subprocess.TimeoutExpired:
            logger.error("DBT tests timed out")
            return DBTResult(
                success=False,
                command=' '.join(cmd) if 'cmd' in locals() else 'dbt test',
                output="",
                error="Operation timed out",
                execution_time_seconds=(datetime.now() - start_time).total_seconds()
            )
        except Exception as e:
            logger.error(f"Error running DBT tests: {e}")
            return DBTResult(
                success=False,
                command='dbt test',
                output="",
                error=str(e),
                execution_time_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    def run_seeds(self) -> DBTResult:
        """
        Run DBT seeds.
        
        Returns:
            DBTResult: Result of the DBT seed operation
        """
        start_time = datetime.now()
        
        try:
            logger.info("Starting DBT seeds run")
            
            # Build command
            cmd = ['dbt', 'seed']
            
            # Add project and profiles directories
            cmd.extend(['--project-dir', self.config.project_dir])
            cmd.extend(['--profiles-dir', self.config.profiles_dir])
            cmd.extend(['--target', self.config.target])
            cmd.extend(['--threads', str(self.config.threads)])
            
            if self.config.vars:
                vars_str = ' '.join([f"{k}:{v}" for k, v in self.config.vars.items()])
                cmd.extend(['--vars', vars_str])
            
            if self.config.debug:
                cmd.append('--debug')
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=self.config.project_dir
            )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            dbt_result = DBTResult(
                success=result.returncode == 0,
                command=' '.join(cmd),
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                execution_time_seconds=execution_time,
                metadata={
                    'target': self.config.target,
                    'threads': self.config.threads
                }
            )
            
            if dbt_result.success:
                logger.info(f"DBT seeds completed successfully in {execution_time:.2f}s")
            else:
                logger.error(f"DBT seeds failed: {result.stderr}")
            
            return dbt_result
            
        except subprocess.TimeoutExpired:
            logger.error("DBT seeds timed out")
            return DBTResult(
                success=False,
                command=' '.join(cmd) if 'cmd' in locals() else 'dbt seed',
                output="",
                error="Operation timed out",
                execution_time_seconds=(datetime.now() - start_time).total_seconds()
            )
        except Exception as e:
            logger.error(f"Error running DBT seeds: {e}")
            return DBTResult(
                success=False,
                command='dbt seed',
                output="",
                error=str(e),
                execution_time_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    def compile_project(self) -> DBTResult:
        """
        Compile DBT project.
        
        Returns:
            DBTResult: Result of the DBT compile operation
        """
        start_time = datetime.now()
        
        try:
            logger.info("Starting DBT project compilation")
            
            # Build command
            cmd = ['dbt', 'compile']
            
            # Add project and profiles directories
            cmd.extend(['--project-dir', self.config.project_dir])
            cmd.extend(['--profiles-dir', self.config.profiles_dir])
            cmd.extend(['--target', self.config.target])
            
            if self.config.vars:
                vars_str = ' '.join([f"{k}:{v}" for k, v in self.config.vars.items()])
                cmd.extend(['--vars', vars_str])
            
            if self.config.debug:
                cmd.append('--debug')
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=self.config.project_dir
            )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            dbt_result = DBTResult(
                success=result.returncode == 0,
                command=' '.join(cmd),
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                execution_time_seconds=execution_time,
                metadata={
                    'target': self.config.target
                }
            )
            
            if dbt_result.success:
                logger.info(f"DBT project compiled successfully in {execution_time:.2f}s")
            else:
                logger.error(f"DBT project compilation failed: {result.stderr}")
            
            return dbt_result
            
        except subprocess.TimeoutExpired:
            logger.error("DBT compilation timed out")
            return DBTResult(
                success=False,
                command=' '.join(cmd) if 'cmd' in locals() else 'dbt compile',
                output="",
                error="Operation timed out",
                execution_time_seconds=(datetime.now() - start_time).total_seconds()
            )
        except Exception as e:
            logger.error(f"Error compiling DBT project: {e}")
            return DBTResult(
                success=False,
                command='dbt compile',
                output="",
                error=str(e),
                execution_time_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    def _parse_run_output(self, output: str) -> tuple[int, int]:
        """Parse DBT run output to extract model statistics."""
        try:
            models_run = 0
            models_failed = 0
            
            lines = output.split('\n')
            for line in lines:
                if 'Completed successfully' in line and 'model' in line:
                    models_run += 1
                elif 'ERROR' in line and 'model' in line:
                    models_failed += 1
            
            return models_run, models_failed
            
        except Exception as e:
            logger.error(f"Error parsing DBT run output: {e}")
            return 0, 0
    
    def _parse_test_output(self, output: str) -> tuple[int, int]:
        """Parse DBT test output to extract test statistics."""
        try:
            tests_run = 0
            tests_failed = 0
            
            lines = output.split('\n')
            for line in lines:
                if 'PASS' in line:
                    tests_run += 1
                elif 'FAIL' in line:
                    tests_run += 1
                    tests_failed += 1
            
            return tests_run, tests_failed
            
        except Exception as e:
            logger.error(f"Error parsing DBT test output: {e}")
            return 0, 0
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get information about the DBT project."""
        try:
            return {
                'project_dir': self.config.project_dir,
                'profiles_dir': self.config.profiles_dir,
                'target': self.config.target,
                'threads': self.config.threads,
                'timeout': self.config.timeout,
                'debug': self.config.debug,
                'full_refresh': self.config.full_refresh,
                'select': self.config.select,
                'exclude': self.config.exclude,
                'vars': self.config.vars,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting project info: {e}")
            return {'error': str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the DBT orchestrator."""
        try:
            # Check if DBT is accessible
            dbt_available = self._validate_dbt_installation()
            
            # Check if project directory exists
            project_dir_exists = os.path.exists(self.config.project_dir)
            
            # Check if profiles directory exists
            profiles_dir_exists = os.path.exists(self.config.profiles_dir)
            
            return {
                'status': 'healthy' if dbt_available and project_dir_exists and profiles_dir_exists else 'unhealthy',
                'dbt_available': dbt_available,
                'project_dir_exists': project_dir_exists,
                'profiles_dir_exists': profiles_dir_exists,
                'config': self.config.__dict__,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Convenience functions
def create_dbt_orchestrator(**kwargs) -> DBTOrchestrator:
    """Create a DBT orchestrator with custom configuration."""
    config = DBTConfig(**kwargs)
    return DBTOrchestrator(config)


def run_dbt_models(project_dir: str, **kwargs) -> DBTResult:
    """Convenience function for running DBT models."""
    orchestrator = create_dbt_orchestrator(project_dir=project_dir, **kwargs)
    return orchestrator.run_models()


def run_dbt_tests(project_dir: str, **kwargs) -> DBTResult:
    """Convenience function for running DBT tests."""
    orchestrator = create_dbt_orchestrator(project_dir=project_dir, **kwargs)
    return orchestrator.run_tests()
