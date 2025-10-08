"""
Connection Pool for PBF-LB/M Data Pipeline

This module provides connection pool management for operational storage.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from queue import Queue, Empty
from contextlib import contextmanager
from src.data_pipeline.config.storage_config import get_postgres_config

logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Connection pool for PostgreSQL operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, min_connections: int = 2, max_connections: int = 10):
        """
        Initialize connection pool.
        
        Args:
            config: Optional PostgreSQL configuration dictionary
            min_connections: Minimum number of connections
            max_connections: Maximum number of connections
        """
        self.config = config or get_postgres_config()
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_queue = Queue(maxsize=max_connections)
        self.active_connections = set()
        self.connection_stats = {
            "connections_created": 0,
            "connections_acquired": 0,
            "connections_released": 0,
            "connections_failed": 0,
            "start_time": datetime.now()
        }
        self.lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool with minimum connections."""
        try:
            for _ in range(self.min_connections):
                connection = self._create_connection()
                if connection:
                    self.connection_queue.put(connection)
                    self.connection_stats["connections_created"] += 1
            
            logger.info(f"Initialized connection pool with {self.min_connections} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
    
    def _create_connection(self) -> Optional[Any]:
        """
        Create a new database connection.
        
        Returns:
            Optional[Any]: Database connection or None if failed
        """
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            
            # Set connection attributes
            connection.autocommit = False
            
            logger.debug("Created new database connection")
            return connection
            
        except ImportError:
            logger.error("psycopg2 not available for PostgreSQL operations")
            return None
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            self.connection_stats["connections_failed"] += 1
            return None
    
    def _validate_connection(self, connection: Any) -> bool:
        """
        Validate if a connection is still active.
        
        Args:
            connection: Database connection to validate
            
        Returns:
            bool: True if connection is valid, False otherwise
        """
        try:
            if connection.closed:
                return False
            
            # Test connection with a simple query
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
            
        except Exception as e:
            logger.debug(f"Connection validation failed: {e}")
            return False
    
    def _cleanup_connection(self, connection: Any) -> None:
        """
        Clean up a database connection.
        
        Args:
            connection: Database connection to clean up
        """
        try:
            if connection and not connection.closed:
                connection.close()
                logger.debug("Cleaned up database connection")
        except Exception as e:
            logger.error(f"Error cleaning up connection: {e}")
    
    def acquire_connection(self, timeout: int = 30) -> Optional[Any]:
        """
        Acquire a connection from the pool.
        
        Args:
            timeout: Timeout in seconds to wait for a connection
            
        Returns:
            Optional[Any]: Database connection or None if failed
        """
        try:
            # Try to get connection from queue
            try:
                connection = self.connection_queue.get(timeout=timeout)
            except Empty:
                logger.warning("No connections available in pool")
                return None
            
            # Validate connection
            if not self._validate_connection(connection):
                logger.warning("Connection validation failed, creating new connection")
                self._cleanup_connection(connection)
                connection = self._create_connection()
                if not connection:
                    return None
            
            # Add to active connections
            with self.lock:
                self.active_connections.add(connection)
                self.connection_stats["connections_acquired"] += 1
            
            logger.debug("Acquired connection from pool")
            return connection
            
        except Exception as e:
            logger.error(f"Error acquiring connection: {e}")
            return None
    
    def release_connection(self, connection: Any) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            connection: Database connection to release
        """
        try:
            if not connection:
                return
            
            # Remove from active connections
            with self.lock:
                self.active_connections.discard(connection)
                self.connection_stats["connections_released"] += 1
            
            # Validate connection before returning to pool
            if self._validate_connection(connection):
                # Return to pool if there's space
                try:
                    self.connection_queue.put_nowait(connection)
                    logger.debug("Released connection back to pool")
                except:
                    # Pool is full, close the connection
                    self._cleanup_connection(connection)
                    logger.debug("Pool full, closed connection")
            else:
                # Connection is invalid, close it
                self._cleanup_connection(connection)
                logger.debug("Invalid connection closed")
                
        except Exception as e:
            logger.error(f"Error releasing connection: {e}")
    
    @contextmanager
    def get_connection(self, timeout: int = 30):
        """
        Context manager for getting a connection from the pool.
        
        Args:
            timeout: Timeout in seconds to wait for a connection
            
        Yields:
            Any: Database connection
        """
        connection = None
        try:
            connection = self.acquire_connection(timeout)
            if not connection:
                raise Exception("Failed to acquire connection from pool")
            yield connection
        finally:
            if connection:
                self.release_connection(connection)
    
    def execute_query(self, query: str, params: Optional[tuple] = None, timeout: int = 30) -> List[Dict[str, Any]]:
        """
        Execute a query using a connection from the pool.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            timeout: Timeout in seconds to wait for a connection
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        with self.get_connection(timeout) as connection:
            try:
                cursor = connection.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Fetch results
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
                
                cursor.close()
                logger.debug(f"Executed query using connection pool, returned {len(results)} rows")
                return results
                
            except Exception as e:
                logger.error(f"Error executing query with connection pool: {e}")
                return []
    
    def execute_transaction(self, queries: List[str], params_list: Optional[List[tuple]] = None, timeout: int = 30) -> bool:
        """
        Execute multiple queries in a transaction using a connection from the pool.
        
        Args:
            queries: List of SQL query strings
            params_list: Optional list of query parameters
            timeout: Timeout in seconds to wait for a connection
            
        Returns:
            bool: True if transaction successful, False otherwise
        """
        with self.get_connection(timeout) as connection:
            try:
                cursor = connection.cursor()
                
                # Begin transaction
                cursor.execute("BEGIN")
                
                # Execute queries
                params_list = params_list or [None] * len(queries)
                for query, params in zip(queries, params_list):
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                
                # Commit transaction
                cursor.execute("COMMIT")
                cursor.close()
                
                logger.info(f"Executed transaction with {len(queries)} queries using connection pool")
                return True
                
            except Exception as e:
                # Rollback transaction
                try:
                    cursor.execute("ROLLBACK")
                    cursor.close()
                except:
                    pass
                
                logger.error(f"Error executing transaction with connection pool: {e}")
                return False
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Dict[str, Any]: Pool statistics
        """
        current_time = datetime.now()
        total_time = (current_time - self.connection_stats["start_time"]).total_seconds()
        
        stats = self.connection_stats.copy()
        stats.update({
            "total_time": total_time,
            "pool_size": self.connection_queue.qsize(),
            "active_connections": len(self.active_connections),
            "available_connections": self.connection_queue.qsize(),
            "max_connections": self.max_connections,
            "min_connections": self.min_connections,
            "connections_per_second": stats["connections_acquired"] / total_time if total_time > 0 else 0,
            "current_time": current_time.isoformat()
        })
        
        return stats
    
    def resize_pool(self, new_min_connections: int, new_max_connections: int) -> bool:
        """
        Resize the connection pool.
        
        Args:
            new_min_connections: New minimum number of connections
            new_max_connections: New maximum number of connections
            
        Returns:
            bool: True if resize successful, False otherwise
        """
        try:
            with self.lock:
                # Update pool size limits
                self.min_connections = new_min_connections
                self.max_connections = new_max_connections
                
                # Create new queue with new max size
                old_queue = self.connection_queue
                self.connection_queue = Queue(maxsize=new_max_connections)
                
                # Transfer existing connections
                transferred = 0
                while not old_queue.empty() and transferred < new_max_connections:
                    try:
                        connection = old_queue.get_nowait()
                        if self._validate_connection(connection):
                            self.connection_queue.put_nowait(connection)
                            transferred += 1
                        else:
                            self._cleanup_connection(connection)
                    except:
                        break
                
                # Create additional connections if needed
                while transferred < new_min_connections:
                    connection = self._create_connection()
                    if connection:
                        self.connection_queue.put(connection)
                        transferred += 1
                        self.connection_stats["connections_created"] += 1
                    else:
                        break
                
                logger.info(f"Resized connection pool to min={new_min_connections}, max={new_max_connections}")
                return True
                
        except Exception as e:
            logger.error(f"Error resizing connection pool: {e}")
            return False
    
    def cleanup_pool(self) -> None:
        """Clean up all connections in the pool."""
        try:
            with self.lock:
                # Close all connections in queue
                while not self.connection_queue.empty():
                    try:
                        connection = self.connection_queue.get_nowait()
                        self._cleanup_connection(connection)
                    except:
                        break
                
                # Close all active connections
                for connection in self.active_connections.copy():
                    self._cleanup_connection(connection)
                
                self.active_connections.clear()
                
                logger.info("Cleaned up connection pool")
                
        except Exception as e:
            logger.error(f"Error cleaning up connection pool: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the connection pool.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            # Test connection acquisition
            connection = self.acquire_connection(timeout=5)
            if not connection:
                return {
                    "healthy": False,
                    "error": "Failed to acquire connection",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Test connection with a simple query
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            
            # Release connection
            self.release_connection(connection)
            
            if result and result[0] == 1:
                return {
                    "healthy": True,
                    "message": "Connection pool is healthy",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "healthy": False,
                    "error": "Connection test query failed",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "pool_stats": self.get_pool_stats(),
            "health_check": self.health_check(),
            "ingestion_timestamp": datetime.now().isoformat()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_pool()
