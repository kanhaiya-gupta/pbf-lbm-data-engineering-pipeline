"""
Query Executor for PBF-LB/M Data Pipeline

This module provides query execution capabilities for data warehouse operations.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from src.data_pipeline.storage.data_warehouse.snowflake_client import SnowflakeClient

logger = logging.getLogger(__name__)


class QueryExecutor:
    """
    Query executor for data warehouse operations.
    """
    
    def __init__(self, snowflake_client: Optional[SnowflakeClient] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize query executor.
        
        Args:
            snowflake_client: Optional Snowflake client instance
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.snowflake_client = snowflake_client or SnowflakeClient()
        self.query_timeout = self.config.get('query_timeout', 3600)  # 1 hour
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 5)  # seconds
        self.execution_stats = {
            "queries_executed": 0,
            "queries_failed": 0,
            "total_execution_time": 0,
            "start_time": datetime.now()
        }
    
    def execute_query(self, query: str, params: Optional[tuple] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a SQL query with retry logic and timeout.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            timeout: Query timeout in seconds (uses default if None)
            
        Returns:
            Dict[str, Any]: Query execution results
        """
        timeout = timeout or self.query_timeout
        start_time = time.time()
        
        result = {
            "success": False,
            "data": [],
            "row_count": 0,
            "execution_time": 0,
            "error": None,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Executing query (attempt {attempt + 1}/{self.max_retries})")
                
                # Execute query
                data = self.snowflake_client.execute_query(query, params)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Check timeout
                if execution_time > timeout:
                    raise TimeoutError(f"Query execution exceeded timeout of {timeout} seconds")
                
                # Update result
                result.update({
                    "success": True,
                    "data": data,
                    "row_count": len(data),
                    "execution_time": execution_time
                })
                
                # Update stats
                self.execution_stats["queries_executed"] += 1
                self.execution_stats["total_execution_time"] += execution_time
                
                logger.info(f"Query executed successfully in {execution_time:.2f} seconds, returned {len(data)} rows")
                break
                
            except Exception as e:
                error_msg = str(e)
                result["error"] = error_msg
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"Query execution failed (attempt {attempt + 1}): {error_msg}, retrying in {self.retry_delay} seconds")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Query execution failed after {self.max_retries} attempts: {error_msg}")
                    self.execution_stats["queries_failed"] += 1
        
        return result
    
    def execute_batch_queries(self, queries: List[str], params_list: Optional[List[tuple]] = None) -> List[Dict[str, Any]]:
        """
        Execute multiple queries in batch.
        
        Args:
            queries: List of SQL query strings
            params_list: Optional list of query parameters
            
        Returns:
            List[Dict[str, Any]]: List of query execution results
        """
        logger.info(f"Executing batch of {len(queries)} queries")
        
        results = []
        params_list = params_list or [None] * len(queries)
        
        for i, (query, params) in enumerate(zip(queries, params_list)):
            try:
                result = self.execute_query(query, params)
                results.append(result)
                
                if not result["success"]:
                    logger.warning(f"Query {i + 1} failed: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error executing query {i + 1}: {e}")
                results.append({
                    "success": False,
                    "data": [],
                    "row_count": 0,
                    "execution_time": 0,
                    "error": str(e),
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                })
        
        successful_queries = sum(1 for r in results if r["success"])
        logger.info(f"Batch execution completed: {successful_queries}/{len(queries)} queries successful")
        
        return results
    
    def execute_query_with_callback(self, query: str, callback: Callable[[Dict[str, Any]], None], params: Optional[tuple] = None) -> bool:
        """
        Execute a query and call a callback function with the results.
        
        Args:
            query: SQL query string
            callback: Callback function to process results
            params: Optional query parameters
            
        Returns:
            bool: True if execution and callback successful, False otherwise
        """
        try:
            result = self.execute_query(query, params)
            
            if result["success"]:
                callback(result)
                return True
            else:
                logger.error(f"Query execution failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing query with callback: {e}")
            return False
    
    def execute_parameterized_query(self, query_template: str, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a parameterized query with multiple parameter sets.
        
        Args:
            query_template: SQL query template with placeholders
            parameters: List of parameter dictionaries
            
        Returns:
            List[Dict[str, Any]]: List of query execution results
        """
        logger.info(f"Executing parameterized query with {len(parameters)} parameter sets")
        
        results = []
        
        for i, params in enumerate(parameters):
            try:
                # Build query with parameters
                query = query_template.format(**params)
                
                result = self.execute_query(query)
                results.append(result)
                
                if not result["success"]:
                    logger.warning(f"Parameterized query {i + 1} failed: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error executing parameterized query {i + 1}: {e}")
                results.append({
                    "success": False,
                    "data": [],
                    "row_count": 0,
                    "execution_time": 0,
                    "error": str(e),
                    "query": query_template,
                    "timestamp": datetime.now().isoformat()
                })
        
        successful_queries = sum(1 for r in results if r["success"])
        logger.info(f"Parameterized query execution completed: {successful_queries}/{len(parameters)} queries successful")
        
        return results
    
    def execute_analytical_query(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Execute an analytical query with enhanced monitoring.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            Dict[str, Any]: Enhanced query execution results
        """
        logger.info("Executing analytical query")
        
        start_time = time.time()
        result = self.execute_query(query, params)
        
        # Add analytical query specific information
        result["query_type"] = "analytical"
        result["monitoring"] = {
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": time.time() - start_time
        }
        
        if result["success"]:
            # Add data quality metrics
            result["data_quality"] = self._analyze_data_quality(result["data"])
            
            # Add performance metrics
            result["performance"] = {
                "rows_per_second": result["row_count"] / result["execution_time"] if result["execution_time"] > 0 else 0,
                "execution_efficiency": "high" if result["execution_time"] < 10 else "medium" if result["execution_time"] < 60 else "low"
            }
        
        return result
    
    def execute_etl_query(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Execute an ETL query with transaction management.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            Dict[str, Any]: ETL query execution results
        """
        logger.info("Executing ETL query")
        
        start_time = time.time()
        
        try:
            # Begin transaction
            self.snowflake_client.cursor.execute("BEGIN TRANSACTION")
            
            # Execute query
            result = self.execute_query(query, params)
            
            if result["success"]:
                # Commit transaction
                self.snowflake_client.cursor.execute("COMMIT")
                logger.info("ETL query transaction committed")
            else:
                # Rollback transaction
                self.snowflake_client.cursor.execute("ROLLBACK")
                logger.warning("ETL query transaction rolled back due to error")
            
            # Add ETL specific information
            result["query_type"] = "etl"
            result["transaction"] = {
                "committed": result["success"],
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Rollback transaction on exception
            try:
                self.snowflake_client.cursor.execute("ROLLBACK")
            except:
                pass
            
            result = {
                "success": False,
                "data": [],
                "row_count": 0,
                "execution_time": time.time() - start_time,
                "error": str(e),
                "query": query,
                "query_type": "etl",
                "transaction": {
                    "committed": False,
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_stats["queries_failed"] += 1
        
        return result
    
    def _analyze_data_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze data quality of query results.
        
        Args:
            data: Query result data
            
        Returns:
            Dict[str, Any]: Data quality metrics
        """
        if not data:
            return {
                "total_rows": 0,
                "null_values": 0,
                "duplicate_rows": 0,
                "data_completeness": 1.0
            }
        
        total_rows = len(data)
        null_values = 0
        duplicate_rows = 0
        
        # Count null values
        for row in data:
            for value in row.values():
                if value is None:
                    null_values += 1
        
        # Count duplicate rows
        unique_rows = len(set(tuple(row.values()) for row in data))
        duplicate_rows = total_rows - unique_rows
        
        # Calculate data completeness
        total_cells = total_rows * len(data[0]) if data else 0
        data_completeness = (total_cells - null_values) / total_cells if total_cells > 0 else 1.0
        
        return {
            "total_rows": total_rows,
            "null_values": null_values,
            "duplicate_rows": duplicate_rows,
            "data_completeness": data_completeness,
            "unique_rows": unique_rows
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get query execution statistics.
        
        Returns:
            Dict[str, Any]: Execution statistics
        """
        current_time = datetime.now()
        total_time = (current_time - self.execution_stats["start_time"]).total_seconds()
        
        stats = self.execution_stats.copy()
        stats.update({
            "total_time": total_time,
            "queries_per_second": stats["queries_executed"] / total_time if total_time > 0 else 0,
            "average_execution_time": stats["total_execution_time"] / stats["queries_executed"] if stats["queries_executed"] > 0 else 0,
            "success_rate": stats["queries_executed"] / (stats["queries_executed"] + stats["queries_failed"]) if (stats["queries_executed"] + stats["queries_failed"]) > 0 else 0,
            "current_time": current_time.isoformat()
        })
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.execution_stats = {
            "queries_executed": 0,
            "queries_failed": 0,
            "total_execution_time": 0,
            "start_time": datetime.now()
        }
        logger.info("Query execution statistics reset")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "query_timeout": self.query_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "execution_stats": self.get_execution_stats(),
            "ingestion_timestamp": datetime.now().isoformat()
        }
