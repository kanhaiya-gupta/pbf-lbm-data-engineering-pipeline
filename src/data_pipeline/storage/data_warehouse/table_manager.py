"""
Table Manager for PBF-LB/M Data Pipeline

This module provides table management capabilities for data warehouse operations.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from src.data_pipeline.storage.data_warehouse.snowflake_client import SnowflakeClient

logger = logging.getLogger(__name__)


class TableManager:
    """
    Table manager for data warehouse operations.
    """
    
    def __init__(self, snowflake_client: Optional[SnowflakeClient] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize table manager.
        
        Args:
            snowflake_client: Optional Snowflake client instance
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.snowflake_client = snowflake_client or SnowflakeClient()
        self.table_stats = {
            "tables_created": 0,
            "tables_dropped": 0,
            "tables_modified": 0,
            "start_time": datetime.now()
        }
    
    def create_table(self, table_name: str, schema: str, if_not_exists: bool = True, comment: Optional[str] = None) -> bool:
        """
        Create a table with schema and optional comment.
        
        Args:
            table_name: Name of the table
            schema: Table schema definition
            if_not_exists: Whether to use IF NOT EXISTS clause
            comment: Optional table comment
            
        Returns:
            bool: True if table creation successful, False otherwise
        """
        try:
            if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
            comment_clause = f"COMMENT = '{comment}'" if comment else ""
            
            query = f"CREATE TABLE {if_not_exists_clause} {table_name} ({schema}) {comment_clause}"
            
            self.snowflake_client.cursor.execute(query)
            self.table_stats["tables_created"] += 1
            
            logger.info(f"Created table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            return False
    
    def drop_table(self, table_name: str, if_exists: bool = True) -> bool:
        """
        Drop a table.
        
        Args:
            table_name: Name of the table
            if_exists: Whether to use IF EXISTS clause
            
        Returns:
            bool: True if table drop successful, False otherwise
        """
        try:
            if_exists_clause = "IF EXISTS" if if_exists else ""
            query = f"DROP TABLE {if_exists_clause} {table_name}"
            
            self.snowflake_client.cursor.execute(query)
            self.table_stats["tables_dropped"] += 1
            
            logger.info(f"Dropped table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping table {table_name}: {e}")
            return False
    
    def alter_table(self, table_name: str, operation: str, column_name: Optional[str] = None, column_definition: Optional[str] = None) -> bool:
        """
        Alter a table structure.
        
        Args:
            table_name: Name of the table
            operation: Alter operation (ADD, DROP, MODIFY, RENAME)
            column_name: Name of the column (for ADD, DROP, MODIFY, RENAME)
            column_definition: Column definition (for ADD, MODIFY)
            
        Returns:
            bool: True if table alteration successful, False otherwise
        """
        try:
            if operation.upper() == "ADD":
                if not column_name or not column_definition:
                    raise ValueError("Column name and definition required for ADD operation")
                query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
                
            elif operation.upper() == "DROP":
                if not column_name:
                    raise ValueError("Column name required for DROP operation")
                query = f"ALTER TABLE {table_name} DROP COLUMN {column_name}"
                
            elif operation.upper() == "MODIFY":
                if not column_name or not column_definition:
                    raise ValueError("Column name and definition required for MODIFY operation")
                query = f"ALTER TABLE {table_name} MODIFY COLUMN {column_name} {column_definition}"
                
            elif operation.upper() == "RENAME":
                if not column_name or not column_definition:
                    raise ValueError("Column name and new name required for RENAME operation")
                query = f"ALTER TABLE {table_name} RENAME COLUMN {column_name} TO {column_definition}"
                
            else:
                raise ValueError(f"Unsupported alter operation: {operation}")
            
            self.snowflake_client.cursor.execute(query)
            self.table_stats["tables_modified"] += 1
            
            logger.info(f"Altered table {table_name}: {operation} {column_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error altering table {table_name}: {e}")
            return False
    
    def create_index(self, table_name: str, index_name: str, columns: List[str], unique: bool = False) -> bool:
        """
        Create an index on a table.
        
        Args:
            table_name: Name of the table
            index_name: Name of the index
            columns: List of column names for the index
            unique: Whether to create a unique index
            
        Returns:
            bool: True if index creation successful, False otherwise
        """
        try:
            unique_clause = "UNIQUE" if unique else ""
            columns_str = ", ".join(columns)
            query = f"CREATE {unique_clause} INDEX {index_name} ON {table_name} ({columns_str})"
            
            self.snowflake_client.cursor.execute(query)
            
            logger.info(f"Created index {index_name} on table {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index {index_name} on table {table_name}: {e}")
            return False
    
    def drop_index(self, index_name: str) -> bool:
        """
        Drop an index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            bool: True if index drop successful, False otherwise
        """
        try:
            query = f"DROP INDEX {index_name}"
            
            self.snowflake_client.cursor.execute(query)
            
            logger.info(f"Dropped index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping index {index_name}: {e}")
            return False
    
    def create_view(self, view_name: str, query: str, if_not_exists: bool = True) -> bool:
        """
        Create a view.
        
        Args:
            view_name: Name of the view
            query: SQL query for the view
            if_not_exists: Whether to use IF NOT EXISTS clause
            
        Returns:
            bool: True if view creation successful, False otherwise
        """
        try:
            if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
            query_sql = f"CREATE VIEW {if_not_exists_clause} {view_name} AS {query}"
            
            self.snowflake_client.cursor.execute(query_sql)
            
            logger.info(f"Created view: {view_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating view {view_name}: {e}")
            return False
    
    def drop_view(self, view_name: str, if_exists: bool = True) -> bool:
        """
        Drop a view.
        
        Args:
            view_name: Name of the view
            if_exists: Whether to use IF EXISTS clause
            
        Returns:
            bool: True if view drop successful, False otherwise
        """
        try:
            if_exists_clause = "IF EXISTS" if if_exists else ""
            query = f"DROP VIEW {if_exists_clause} {view_name}"
            
            self.snowflake_client.cursor.execute(query)
            
            logger.info(f"Dropped view: {view_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping view {view_name}: {e}")
            return False
    
    def create_sequence(self, sequence_name: str, start_value: int = 1, increment: int = 1, max_value: Optional[int] = None, min_value: Optional[int] = None) -> bool:
        """
        Create a sequence.
        
        Args:
            sequence_name: Name of the sequence
            start_value: Starting value
            increment: Increment value
            max_value: Maximum value
            min_value: Minimum value
            
        Returns:
            bool: True if sequence creation successful, False otherwise
        """
        try:
            query = f"CREATE SEQUENCE {sequence_name} START WITH {start_value} INCREMENT BY {increment}"
            
            if max_value:
                query += f" MAXVALUE {max_value}"
            if min_value:
                query += f" MINVALUE {min_value}"
            
            self.snowflake_client.cursor.execute(query)
            
            logger.info(f"Created sequence: {sequence_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sequence {sequence_name}: {e}")
            return False
    
    def drop_sequence(self, sequence_name: str, if_exists: bool = True) -> bool:
        """
        Drop a sequence.
        
        Args:
            sequence_name: Name of the sequence
            if_exists: Whether to use IF EXISTS clause
            
        Returns:
            bool: True if sequence drop successful, False otherwise
        """
        try:
            if_exists_clause = "IF EXISTS" if if_exists else ""
            query = f"DROP SEQUENCE {if_exists_clause} {sequence_name}"
            
            self.snowflake_client.cursor.execute(query)
            
            logger.info(f"Dropped sequence: {sequence_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping sequence {sequence_name}: {e}")
            return False
    
    def get_table_list(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of tables.
        
        Args:
            schema: Optional schema name to filter by
            
        Returns:
            List[Dict[str, Any]]: List of table information
        """
        try:
            if schema:
                query = f"SHOW TABLES IN SCHEMA {schema}"
            else:
                query = "SHOW TABLES"
            
            results = self.snowflake_client.execute_query(query)
            
            tables = []
            for row in results:
                tables.append({
                    "name": row["name"],
                    "schema": row["schema_name"],
                    "database": row["database_name"],
                    "kind": row["kind"],
                    "comment": row.get("comment", ""),
                    "created_on": row["created_on"]
                })
            
            logger.info(f"Retrieved {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.error(f"Error getting table list: {e}")
            return []
    
    def get_view_list(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of views.
        
        Args:
            schema: Optional schema name to filter by
            
        Returns:
            List[Dict[str, Any]]: List of view information
        """
        try:
            if schema:
                query = f"SHOW VIEWS IN SCHEMA {schema}"
            else:
                query = "SHOW VIEWS"
            
            results = self.snowflake_client.execute_query(query)
            
            views = []
            for row in results:
                views.append({
                    "name": row["name"],
                    "schema": row["schema_name"],
                    "database": row["database_name"],
                    "comment": row.get("comment", ""),
                    "created_on": row["created_on"]
                })
            
            logger.info(f"Retrieved {len(views)} views")
            return views
            
        except Exception as e:
            logger.error(f"Error getting view list: {e}")
            return []
    
    def get_index_list(self, table_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of indexes.
        
        Args:
            table_name: Optional table name to filter by
            
        Returns:
            List[Dict[str, Any]]: List of index information
        """
        try:
            if table_name:
                query = f"SHOW INDEXES ON TABLE {table_name}"
            else:
                query = "SHOW INDEXES"
            
            results = self.snowflake_client.execute_query(query)
            
            indexes = []
            for row in results:
                indexes.append({
                    "name": row["name"],
                    "table": row["table_name"],
                    "schema": row["schema_name"],
                    "database": row["database_name"],
                    "columns": row["column_name"],
                    "unique": row.get("unique", False),
                    "created_on": row["created_on"]
                })
            
            logger.info(f"Retrieved {len(indexes)} indexes")
            return indexes
            
        except Exception as e:
            logger.error(f"Error getting index list: {e}")
            return []
    
    def get_sequence_list(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of sequences.
        
        Args:
            schema: Optional schema name to filter by
            
        Returns:
            List[Dict[str, Any]]: List of sequence information
        """
        try:
            if schema:
                query = f"SHOW SEQUENCES IN SCHEMA {schema}"
            else:
                query = "SHOW SEQUENCES"
            
            results = self.snowflake_client.execute_query(query)
            
            sequences = []
            for row in results:
                sequences.append({
                    "name": row["name"],
                    "schema": row["schema_name"],
                    "database": row["database_name"],
                    "start_value": row["start_value"],
                    "increment": row["increment"],
                    "max_value": row.get("max_value"),
                    "min_value": row.get("min_value"),
                    "created_on": row["created_on"]
                })
            
            logger.info(f"Retrieved {len(sequences)} sequences")
            return sequences
            
        except Exception as e:
            logger.error(f"Error getting sequence list: {e}")
            return []
    
    def get_table_dependencies(self, table_name: str) -> Dict[str, List[str]]:
        """
        Get table dependencies (views, indexes, etc.).
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict[str, List[str]]: Dictionary of dependencies by type
        """
        try:
            dependencies = {
                "views": [],
                "indexes": [],
                "sequences": []
            }
            
            # Get views that depend on this table
            views = self.get_view_list()
            for view in views:
                # This is a simplified check - in practice, you'd need to parse the view definition
                if table_name.lower() in view["name"].lower():
                    dependencies["views"].append(view["name"])
            
            # Get indexes on this table
            indexes = self.get_index_list(table_name)
            for index in indexes:
                dependencies["indexes"].append(index["name"])
            
            logger.info(f"Retrieved dependencies for table {table_name}")
            return dependencies
            
        except Exception as e:
            logger.error(f"Error getting dependencies for table {table_name}: {e}")
            return {"views": [], "indexes": [], "sequences": []}
    
    def backup_table(self, table_name: str, backup_table_name: str) -> bool:
        """
        Create a backup of a table.
        
        Args:
            table_name: Name of the table to backup
            backup_table_name: Name of the backup table
            
        Returns:
            bool: True if backup successful, False otherwise
        """
        try:
            # Create backup table with same structure
            schema_info = self.snowflake_client.get_table_schema(table_name)
            if not schema_info:
                logger.error(f"Could not get schema for table {table_name}")
                return False
            
            # Build schema string
            schema_parts = []
            for col in schema_info:
                schema_parts.append(f"{col['column_name']} {col['data_type']}")
            
            schema_str = ", ".join(schema_parts)
            
            # Create backup table
            if not self.create_table(backup_table_name, schema_str):
                return False
            
            # Copy data
            query = f"INSERT INTO {backup_table_name} SELECT * FROM {table_name}"
            self.snowflake_client.cursor.execute(query)
            
            logger.info(f"Created backup of table {table_name} as {backup_table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up table {table_name}: {e}")
            return False
    
    def restore_table(self, backup_table_name: str, table_name: str) -> bool:
        """
        Restore a table from backup.
        
        Args:
            backup_table_name: Name of the backup table
            table_name: Name of the table to restore
            
        Returns:
            bool: True if restore successful, False otherwise
        """
        try:
            # Drop existing table if it exists
            self.drop_table(table_name, if_exists=True)
            
            # Create table from backup
            schema_info = self.snowflake_client.get_table_schema(backup_table_name)
            if not schema_info:
                logger.error(f"Could not get schema for backup table {backup_table_name}")
                return False
            
            # Build schema string
            schema_parts = []
            for col in schema_info:
                schema_parts.append(f"{col['column_name']} {col['data_type']}")
            
            schema_str = ", ".join(schema_parts)
            
            # Create table
            if not self.create_table(table_name, schema_str):
                return False
            
            # Copy data
            query = f"INSERT INTO {table_name} SELECT * FROM {backup_table_name}"
            self.snowflake_client.cursor.execute(query)
            
            logger.info(f"Restored table {table_name} from backup {backup_table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring table {table_name}: {e}")
            return False
    
    def get_table_stats(self) -> Dict[str, Any]:
        """
        Get table management statistics.
        
        Returns:
            Dict[str, Any]: Table management statistics
        """
        current_time = datetime.now()
        total_time = (current_time - self.table_stats["start_time"]).total_seconds()
        
        stats = self.table_stats.copy()
        stats.update({
            "total_time": total_time,
            "tables_per_second": stats["tables_created"] / total_time if total_time > 0 else 0,
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
            "table_stats": self.get_table_stats(),
            "ingestion_timestamp": datetime.now().isoformat()
        }
