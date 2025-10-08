"""
Transaction Manager for PBF-LB/M Data Pipeline

This module provides transaction management capabilities for operational storage.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
from src.data_pipeline.storage.operational.connection_pool import ConnectionPool

logger = logging.getLogger(__name__)


class TransactionIsolationLevel(Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class TransactionManager:
    """
    Transaction manager for database operations.
    """
    
    def __init__(self, connection_pool: Optional[ConnectionPool] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize transaction manager.
        
        Args:
            connection_pool: Optional connection pool instance
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.connection_pool = connection_pool or ConnectionPool()
        self.default_isolation_level = TransactionIsolationLevel.READ_COMMITTED
        self.transaction_timeout = self.config.get('transaction_timeout', 300)  # 5 minutes
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 1)  # seconds
        self.transaction_stats = {
            "transactions_started": 0,
            "transactions_committed": 0,
            "transactions_rolled_back": 0,
            "transactions_failed": 0,
            "start_time": datetime.now()
        }
        self.active_transactions = {}
        self.lock = threading.Lock()
    
    def begin_transaction(self, isolation_level: Optional[TransactionIsolationLevel] = None, timeout: int = 30) -> str:
        """
        Begin a new transaction.
        
        Args:
            isolation_level: Transaction isolation level
            timeout: Timeout in seconds to wait for a connection
            
        Returns:
            str: Transaction ID
        """
        try:
            isolation_level = isolation_level or self.default_isolation_level
            
            # Get connection from pool
            connection = self.connection_pool.acquire_connection(timeout)
            if not connection:
                raise Exception("Failed to acquire connection for transaction")
            
            # Generate transaction ID
            transaction_id = f"tx_{int(time.time() * 1000)}_{threading.current_thread().ident}"
            
            # Set isolation level
            cursor = connection.cursor()
            cursor.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation_level.value}")
            
            # Begin transaction
            cursor.execute("BEGIN")
            cursor.close()
            
            # Store transaction info
            with self.lock:
                self.active_transactions[transaction_id] = {
                    "connection": connection,
                    "isolation_level": isolation_level,
                    "start_time": datetime.now(),
                    "status": "active"
                }
                self.transaction_stats["transactions_started"] += 1
            
            logger.info(f"Started transaction {transaction_id} with isolation level {isolation_level.value}")
            return transaction_id
            
        except Exception as e:
            logger.error(f"Error beginning transaction: {e}")
            self.transaction_stats["transactions_failed"] += 1
            raise
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a transaction.
        
        Args:
            transaction_id: Transaction ID to commit
            
        Returns:
            bool: True if commit successful, False otherwise
        """
        try:
            with self.lock:
                if transaction_id not in self.active_transactions:
                    logger.error(f"Transaction {transaction_id} not found")
                    return False
                
                transaction_info = self.active_transactions[transaction_id]
                connection = transaction_info["connection"]
                transaction_info["status"] = "committing"
            
            # Commit transaction
            cursor = connection.cursor()
            cursor.execute("COMMIT")
            cursor.close()
            
            # Release connection
            self.connection_pool.release_connection(connection)
            
            # Update transaction info
            with self.lock:
                transaction_info["status"] = "committed"
                transaction_info["end_time"] = datetime.now()
                del self.active_transactions[transaction_id]
                self.transaction_stats["transactions_committed"] += 1
            
            logger.info(f"Committed transaction {transaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error committing transaction {transaction_id}: {e}")
            self._rollback_transaction(transaction_id)
            return False
    
    def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a transaction.
        
        Args:
            transaction_id: Transaction ID to rollback
            
        Returns:
            bool: True if rollback successful, False otherwise
        """
        return self._rollback_transaction(transaction_id)
    
    def _rollback_transaction(self, transaction_id: str) -> bool:
        """
        Internal method to rollback a transaction.
        
        Args:
            transaction_id: Transaction ID to rollback
            
        Returns:
            bool: True if rollback successful, False otherwise
        """
        try:
            with self.lock:
                if transaction_id not in self.active_transactions:
                    logger.error(f"Transaction {transaction_id} not found")
                    return False
                
                transaction_info = self.active_transactions[transaction_id]
                connection = transaction_info["connection"]
                transaction_info["status"] = "rolling_back"
            
            # Rollback transaction
            cursor = connection.cursor()
            cursor.execute("ROLLBACK")
            cursor.close()
            
            # Release connection
            self.connection_pool.release_connection(connection)
            
            # Update transaction info
            with self.lock:
                transaction_info["status"] = "rolled_back"
                transaction_info["end_time"] = datetime.now()
                del self.active_transactions[transaction_id]
                self.transaction_stats["transactions_rolled_back"] += 1
            
            logger.info(f"Rolled back transaction {transaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back transaction {transaction_id}: {e}")
            self.transaction_stats["transactions_failed"] += 1
            return False
    
    def execute_in_transaction(self, transaction_id: str, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a query within a transaction.
        
        Args:
            transaction_id: Transaction ID
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        try:
            with self.lock:
                if transaction_id not in self.active_transactions:
                    raise Exception(f"Transaction {transaction_id} not found")
                
                transaction_info = self.active_transactions[transaction_id]
                connection = transaction_info["connection"]
            
            # Execute query
            cursor = connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Fetch results
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            
            cursor.close()
            
            logger.debug(f"Executed query in transaction {transaction_id}, returned {len(results)} rows")
            return results
            
        except Exception as e:
            logger.error(f"Error executing query in transaction {transaction_id}: {e}")
            raise
    
    def execute_transaction_with_retry(self, queries: List[str], params_list: Optional[List[tuple]] = None, isolation_level: Optional[TransactionIsolationLevel] = None) -> bool:
        """
        Execute multiple queries in a transaction with retry logic.
        
        Args:
            queries: List of SQL query strings
            params_list: Optional list of query parameters
            isolation_level: Transaction isolation level
            
        Returns:
            bool: True if transaction successful, False otherwise
        """
        for attempt in range(self.retry_attempts):
            try:
                transaction_id = self.begin_transaction(isolation_level)
                
                try:
                    # Execute queries
                    params_list = params_list or [None] * len(queries)
                    for query, params in zip(queries, params_list):
                        self.execute_in_transaction(transaction_id, query, params)
                    
                    # Commit transaction
                    success = self.commit_transaction(transaction_id)
                    if success:
                        logger.info(f"Executed transaction with {len(queries)} queries successfully")
                        return True
                    else:
                        logger.warning(f"Transaction commit failed on attempt {attempt + 1}")
                        
                except Exception as e:
                    logger.warning(f"Transaction execution failed on attempt {attempt + 1}: {e}")
                    self.rollback_transaction(transaction_id)
                
                # Wait before retry
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Error executing transaction with retry: {e}")
                if attempt == self.retry_attempts - 1:
                    return False
        
        return False
    
    @contextmanager
    def transaction(self, isolation_level: Optional[TransactionIsolationLevel] = None, timeout: int = 30):
        """
        Context manager for transaction management.
        
        Args:
            isolation_level: Transaction isolation level
            timeout: Timeout in seconds to wait for a connection
            
        Yields:
            str: Transaction ID
        """
        transaction_id = None
        try:
            transaction_id = self.begin_transaction(isolation_level, timeout)
            yield transaction_id
            self.commit_transaction(transaction_id)
        except Exception as e:
            if transaction_id:
                self.rollback_transaction(transaction_id)
            raise
    
    def execute_with_transaction(self, queries: List[str], params_list: Optional[List[tuple]] = None, isolation_level: Optional[TransactionIsolationLevel] = None) -> bool:
        """
        Execute queries with automatic transaction management.
        
        Args:
            queries: List of SQL query strings
            params_list: Optional list of query parameters
            isolation_level: Transaction isolation level
            
        Returns:
            bool: True if execution successful, False otherwise
        """
        try:
            with self.transaction(isolation_level) as transaction_id:
                params_list = params_list or [None] * len(queries)
                for query, params in zip(queries, params_list):
                    self.execute_in_transaction(transaction_id, query, params)
            
            logger.info(f"Executed {len(queries)} queries with transaction management")
            return True
            
        except Exception as e:
            logger.error(f"Error executing queries with transaction management: {e}")
            return False
    
    def get_transaction_info(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Optional[Dict[str, Any]]: Transaction information or None if not found
        """
        with self.lock:
            if transaction_id in self.active_transactions:
                transaction_info = self.active_transactions[transaction_id].copy()
                transaction_info["transaction_id"] = transaction_id
                return transaction_info
            return None
    
    def get_active_transactions(self) -> List[Dict[str, Any]]:
        """
        Get list of active transactions.
        
        Returns:
            List[Dict[str, Any]]: List of active transaction information
        """
        with self.lock:
            active_transactions = []
            for transaction_id, transaction_info in self.active_transactions.items():
                info = transaction_info.copy()
                info["transaction_id"] = transaction_id
                active_transactions.append(info)
            return active_transactions
    
    def cleanup_stale_transactions(self, max_age_minutes: int = 30) -> int:
        """
        Clean up stale transactions.
        
        Args:
            max_age_minutes: Maximum age of transactions in minutes
            
        Returns:
            int: Number of transactions cleaned up
        """
        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        with self.lock:
            stale_transactions = []
            for transaction_id, transaction_info in self.active_transactions.items():
                if transaction_info["start_time"] < cutoff_time:
                    stale_transactions.append(transaction_id)
            
            for transaction_id in stale_transactions:
                try:
                    self._rollback_transaction(transaction_id)
                    cleaned_count += 1
                    logger.info(f"Cleaned up stale transaction {transaction_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up stale transaction {transaction_id}: {e}")
        
        return cleaned_count
    
    def get_transaction_stats(self) -> Dict[str, Any]:
        """
        Get transaction statistics.
        
        Returns:
            Dict[str, Any]: Transaction statistics
        """
        current_time = datetime.now()
        total_time = (current_time - self.transaction_stats["start_time"]).total_seconds()
        
        stats = self.transaction_stats.copy()
        stats.update({
            "total_time": total_time,
            "active_transactions": len(self.active_transactions),
            "transactions_per_second": stats["transactions_started"] / total_time if total_time > 0 else 0,
            "success_rate": stats["transactions_committed"] / stats["transactions_started"] if stats["transactions_started"] > 0 else 0,
            "current_time": current_time.isoformat()
        })
        
        return stats
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "transaction_timeout": self.transaction_timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "transaction_stats": self.get_transaction_stats(),
            "ingestion_timestamp": datetime.now().isoformat()
        }
