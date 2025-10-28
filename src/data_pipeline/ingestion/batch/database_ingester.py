"""
Database Ingester for PBF-LB/M Data Pipeline

This module provides batch ingestion capabilities for database data.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
from src.data_pipeline.config.postgres_config import get_postgres_config

logger = logging.getLogger(__name__)


class DatabaseIngester:
    """
    Batch ingester for database data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database ingester.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.postgres_config = get_postgres_config()
        self.connection = None
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """Initialize database connection."""
        try:
            import psycopg2
            self.connection = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            logger.info("Database connection initialized successfully")
        except ImportError:
            logger.error("psycopg2 not available for PostgreSQL operations")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        if not self.connection:
            logger.error("Database connection not initialized")
            return []
        
        results = []
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Fetch all results
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                for row in rows:
                    result = dict(zip(columns, row))
                    results.append(result)
                
                logger.info(f"Executed query, returned {len(results)} rows")
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
        
        return results
    
    def ingest_table(self, table_name: str, where_clause: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Ingest data from a database table.
        
        Args:
            table_name: Name of the table to ingest
            where_clause: Optional WHERE clause
            limit: Optional row limit
            
        Returns:
            List[Dict[str, Any]]: Table data
        """
        logger.info(f"Starting table ingestion: {table_name}")
        
        # Build query
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.execute_query(query)
        
        logger.info(f"Successfully ingested {len(results)} rows from table {table_name}")
        return results
    
    def ingest_pbf_process_data(self, where_clause: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Ingest PBF process data from database.
        
        Args:
            where_clause: Optional WHERE clause
            limit: Optional row limit
            
        Returns:
            List[Dict[str, Any]]: PBF process data
        """
        return self.ingest_table("pbf_process_data", where_clause, limit)
    
    def ingest_ispm_monitoring_data(self, where_clause: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Ingest ISPM monitoring data from database.
        
        Args:
            where_clause: Optional WHERE clause
            limit: Optional row limit
            
        Returns:
            List[Dict[str, Any]]: ISPM monitoring data
        """
        return self.ingest_table("ispm_monitoring_data", where_clause, limit)
    
    def ingest_ct_scan_data(self, where_clause: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Ingest CT scan data from database.
        
        Args:
            where_clause: Optional WHERE clause
            limit: Optional row limit
            
        Returns:
            List[Dict[str, Any]]: CT scan data
        """
        return self.ingest_table("ct_scan_data", where_clause, limit)
    
    def ingest_powder_bed_data(self, where_clause: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Ingest powder bed data from database.
        
        Args:
            where_clause: Optional WHERE clause
            limit: Optional row limit
            
        Returns:
            List[Dict[str, Any]]: Powder bed data
        """
        return self.ingest_table("powder_bed_data", where_clause, limit)
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get table schema information.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List[Dict[str, Any]]: Table schema information
        """
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = %s
        ORDER BY ordinal_position
        """
        
        results = self.execute_query(query, (table_name,))
        
        logger.info(f"Retrieved schema for table {table_name}")
        return results
    
    def get_table_count(self, table_name: str) -> int:
        """
        Get row count for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            int: Row count
        """
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        results = self.execute_query(query)
        
        if results:
            count = results[0]['count']
            logger.info(f"Table {table_name} has {count} rows")
            return count
        
        return 0
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive table information.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict[str, Any]: Table information
        """
        info = {
            "table_name": table_name,
            "row_count": self.get_table_count(table_name),
            "schema": self.get_table_schema(table_name),
            "ingestion_timestamp": datetime.now().isoformat()
        }
        
        return info
    
    def close_connection(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "connection_initialized": self.connection is not None,
            "ingestion_timestamp": datetime.now().isoformat()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()
