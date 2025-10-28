"""
ClickHouse Client for PBF-LB/M Data Pipeline

This module provides ClickHouse operations for data warehouse analytics.
ClickHouse is a columnar database optimized for analytical queries and reporting.
"""

import logging
import json
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Iterator
from datetime import datetime, timedelta
from pathlib import Path
import clickhouse_connect
from clickhouse_connect.driver.exceptions import DatabaseError, InterfaceError
from src.data_pipeline.config.clickhouse_config import get_clickhouse_config

logger = logging.getLogger(__name__)


class ClickHouseClient:
    """
    ClickHouse client for data warehouse analytics operations.
    
    Handles analytical queries, aggregations, reporting, and time-series
    analytics for PBF-LB/M process data and quality metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ClickHouse client.
        
        Args:
            config: Optional ClickHouse configuration dictionary or Pydantic model
        """
        self.config = config or get_clickhouse_config()
        self._initialize_connection()
        self.client = None
        self._initialize_clickhouse_client()
    
    def _initialize_connection(self) -> None:
        """Initialize ClickHouse connection parameters."""
        try:
            # Handle both dictionary and Pydantic model configurations
            if isinstance(self.config, dict):
                self.host = self.config.get('host', 'localhost')
                self.port = self.config.get('port', 8123)
                self.native_port = self.config.get('native_port', 9000)
                self.database = self.config.get('database', 'pbf_analytics')
                self.username = self.config.get('username', 'analytics_user')
                self.password = self.config.get('password', 'analytics_password')
                self.secure = self.config.get('secure', False)
                self.timeout = self.config.get('timeout', 30)
                self.max_connections = self.config.get('max_connections', 20)
            else:
                # Pydantic model configuration
                self.host = self.config.host
                self.port = self.config.port
                self.native_port = self.config.native_port
                self.database = self.config.database
                self.username = self.config.username
                self.password = self.config.password
                self.secure = self.config.secure
                self.timeout = self.config.timeout
                self.max_connections = self.config.max_connections
            
            logger.info(f"ClickHouse connection parameters initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ClickHouse connection parameters: {e}")
            raise
    
    def _initialize_clickhouse_client(self) -> None:
        """Initialize ClickHouse client connection."""
        try:
            # First, create client without specifying database to check connection
            temp_client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,  # Use HTTP port for HTTP connections
                username=self.username,
                password=self.password,
                secure=self.secure,
                connect_timeout=self.timeout,
                send_receive_timeout=self.timeout,
                compress=True,  # Enable compression for better performance
                query_limit=0,  # No query limit
                settings={
                    'max_memory_usage': self.config.max_memory_usage if hasattr(self.config, 'max_memory_usage') else 10000000000,
                    'max_execution_time': self.config.max_execution_time if hasattr(self.config, 'max_execution_time') else 300,
                    'max_threads': self.config.max_threads if hasattr(self.config, 'max_threads') else 8,
                }
            )
            
            # Create database if it doesn't exist
            try:
                temp_client.command(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                logger.info(f"✅ Database {self.database} created or already exists")
            except Exception as db_e:
                logger.warning(f"Could not create database {self.database}: {db_e}")
            
            # Now create the main client with the database
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,  # Use HTTP port for HTTP connections
                username=self.username,
                password=self.password,
                database=self.database,
                secure=self.secure,
                connect_timeout=self.timeout,
                send_receive_timeout=self.timeout,
                compress=True,  # Enable compression for better performance
                query_limit=0,  # No query limit
                settings={
                    'max_memory_usage': self.config.max_memory_usage if hasattr(self.config, 'max_memory_usage') else 10000000000,
                    'max_execution_time': self.config.max_execution_time if hasattr(self.config, 'max_execution_time') else 300,
                    'max_threads': self.config.max_threads if hasattr(self.config, 'max_threads') else 8,
                }
            )
            
            logger.info(f"ClickHouse client initialized: {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ClickHouse client: {e}")
            raise
    
    def connect(self) -> bool:
        """
        Test ClickHouse connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Test connection with a simple query
            result = self.client.query("SELECT 1 as test")
            if result.result_rows:
                logger.info("✅ Connected to ClickHouse successfully")
                return True
            else:
                logger.error("❌ ClickHouse connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to ClickHouse: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from ClickHouse."""
        try:
            if self.client:
                self.client.close()
                logger.info("✅ Disconnected from ClickHouse")
        except Exception as e:
            logger.error(f"❌ Error during ClickHouse disconnect: {e}")
    
    def create_database(self, database_name: str) -> bool:
        """
        Create a ClickHouse database.
        
        Args:
            database_name: Name of the database to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
            self.client.command(query)
            logger.info(f"✅ Created database: {database_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create database {database_name}: {e}")
            return False
    
    def create_table(self, table_name: str, schema: str, engine: str = "MergeTree") -> bool:
        """
        Create a ClickHouse table.
        
        Args:
            table_name: Name of the table to create
            schema: Table schema definition
            engine: Table engine (default: MergeTree)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {schema}
            ) ENGINE = {engine}
            """
            self.client.command(query)
            logger.info(f"✅ Created table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create table {table_name}: {e}")
            return False
    
    def drop_table(self, table_name: str) -> bool:
        """
        Drop a ClickHouse table.
        
        Args:
            table_name: Name of the table to drop
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            query = f"DROP TABLE IF EXISTS {table_name}"
            self.client.command(query)
            logger.info(f"✅ Dropped table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to drop table {table_name}: {e}")
            return False
    
    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        """
        Insert data into ClickHouse table.
        
        Args:
            table_name: Name of the table to insert into
            data: List of dictionaries containing data to insert
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not data:
                logger.warning("No data to insert")
                return True
            
            # Convert data to DataFrame for better handling
            df = pd.DataFrame(data)
            
            # Ensure DataFrame has proper index for ClickHouse
            df = df.reset_index(drop=True)
            
            # For single row, ensure we have a proper range index
            if len(df) == 1:
                df.index = [0]
            else:
                df.index = pd.RangeIndex(len(df))
            
            # Insert data using ClickHouse client
            self.client.insert_df(table_name, df)
            
            logger.info(f"✅ Inserted {len(data)} records into {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert data into {table_name}: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            logger.error(f"❌ Error details: {str(e)}")
            return False
    
    def insert_batch(self, table_name: str, data: List[Dict[str, Any]], batch_size: int = 1000) -> bool:
        """
        Insert data in batches into ClickHouse table.
        
        Args:
            table_name: Name of the table to insert into
            data: List of dictionaries containing data to insert
            batch_size: Number of records per batch
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not data:
                logger.warning("No data to insert")
                return True
            
            total_records = len(data)
            inserted_count = 0
            
            # Process data in batches
            for i in range(0, total_records, batch_size):
                batch_data = data[i:i + batch_size]
                df = pd.DataFrame(batch_data)
                
                # Insert batch
                self.client.insert_df(table_name, df)
                inserted_count += len(batch_data)
                
                logger.info(f"✅ Inserted batch {i//batch_size + 1}: {len(batch_data)} records")
            
            logger.info(f"✅ Successfully inserted {inserted_count} records into {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert batch data into {table_name}: {e}")
            return False
    
    def bulk_insert(self, table_name: str, data: List[Dict[str, Any]], 
                   batch_size: int = 1000) -> bool:
        """
        Bulk insert data into ClickHouse table.
        
        Args:
            table_name: Name of the table to insert into
            data: List of dictionaries containing data to insert
            batch_size: Number of records per batch
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not data:
                logger.warning("No data to insert")
                return True
            
            total_records = len(data)
            success_count = 0
            
            # Process data in batches
            for i in range(0, total_records, batch_size):
                batch = data[i:i + batch_size]
                df = pd.DataFrame(batch)
                
                self.client.insert_df(table_name, df)
                success_count += len(batch)
                
                logger.debug(f"Inserted batch {i//batch_size + 1}: {len(batch)} records")
            
            logger.info(f"✅ Bulk inserted {success_count} records into {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to bulk insert data into {table_name}: {e}")
            return False
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a ClickHouse query and return results as DataFrame.
        
        Args:
            query: SQL query to execute
            parameters: Optional query parameters
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            if parameters:
                result = self.client.query_df(query, parameters=parameters)
            else:
                result = self.client.query_df(query)
            
            logger.debug(f"Executed query: {query[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to execute query: {e}")
            raise
    
    def execute_command(self, command: str) -> bool:
        """
        Execute a ClickHouse command (DDL operations).
        
        Args:
            command: SQL command to execute
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.command(command)
            logger.debug(f"Executed command: {command[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute command: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a ClickHouse table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict[str, Any]: Table information
        """
        try:
            # Get table structure
            structure_query = f"DESCRIBE TABLE {table_name}"
            structure = self.execute_query(structure_query)
            
            # Get table size and row count
            stats_query = f"""
            SELECT 
                count() as row_count,
                formatReadableSize(sum(bytes)) as size
            FROM system.parts 
            WHERE table = '{table_name}'
            """
            stats = self.execute_query(stats_query)
            
            return {
                'structure': structure.to_dict('records'),
                'row_count': stats.iloc[0]['row_count'] if not stats.empty else 0,
                'size': stats.iloc[0]['size'] if not stats.empty else '0 B'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get table info for {table_name}: {e}")
            return {}
    
    def get_database_tables(self, database_name: Optional[str] = None) -> List[str]:
        """
        Get list of tables in a database.
        
        Args:
            database_name: Name of the database (default: current database)
            
        Returns:
            List[str]: List of table names
        """
        try:
            if database_name:
                query = f"SHOW TABLES FROM {database_name}"
            else:
                query = "SHOW TABLES"
            
            result = self.execute_query(query)
            return result.iloc[:, 0].tolist()
            
        except Exception as e:
            logger.error(f"❌ Failed to get tables: {e}")
            return []
    
    def create_materialized_view(self, view_name: str, query: str, 
                                target_table: str) -> bool:
        """
        Create a materialized view in ClickHouse.
        
        Args:
            view_name: Name of the materialized view
            query: Query definition for the view
            target_table: Target table for the view
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            create_view_query = f"""
            CREATE MATERIALIZED VIEW {view_name}
            TO {target_table}
            AS {query}
            """
            self.client.command(create_view_query)
            logger.info(f"✅ Created materialized view: {view_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create materialized view {view_name}: {e}")
            return False
    
    def create_dictionary(self, dictionary_name: str, source_table: str, 
                        key_column: str, value_columns: List[str]) -> bool:
        """
        Create a dictionary in ClickHouse.
        
        Args:
            dictionary_name: Name of the dictionary
            source_table: Source table for the dictionary
            key_column: Key column name
            value_columns: List of value column names
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            value_columns_str = ', '.join(value_columns)
            create_dict_query = f"""
            CREATE DICTIONARY {dictionary_name} (
                {key_column} UInt64,
                {value_columns_str}
            )
            PRIMARY KEY {key_column}
            SOURCE(CLICKHOUSE(
                TABLE '{source_table}'
            ))
            LAYOUT(HASHED())
            LIFETIME(MIN 0 MAX 3600)
            """
            self.client.command(create_dict_query)
            logger.info(f"✅ Created dictionary: {dictionary_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create dictionary {dictionary_name}: {e}")
            return False
    
    def optimize_table(self, table_name: str) -> bool:
        """
        Optimize a ClickHouse table.
        
        Args:
            table_name: Name of the table to optimize
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.command(f"OPTIMIZE TABLE {table_name}")
            logger.info(f"✅ Optimized table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize table {table_name}: {e}")
            return False
    
    def get_query_stats(self) -> Dict[str, Any]:
        """
        Get ClickHouse query statistics.
        
        Returns:
            Dict[str, Any]: Query statistics
        """
        try:
            stats_query = """
            SELECT 
                count() as total_queries,
                avg(query_duration_ms) as avg_duration_ms,
                max(query_duration_ms) as max_duration_ms,
                min(query_duration_ms) as min_duration_ms
            FROM system.query_log 
            WHERE event_date >= today() - 1
            """
            result = self.execute_query(stats_query)
            
            if not result.empty:
                return result.iloc[0].to_dict()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to get query stats: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform ClickHouse health check.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            # Test basic connectivity
            connectivity = self.connect()
            
            # Get system information
            system_info = self.execute_query("SELECT version() as version, uptime() as uptime")
            
            # Get database information
            db_info = self.execute_query(f"SELECT name FROM system.databases WHERE name = '{self.database}'")
            
            return {
                'status': 'healthy' if connectivity else 'unhealthy',
                'connectivity': connectivity,
                'version': system_info.iloc[0]['version'] if not system_info.empty else 'unknown',
                'uptime': system_info.iloc[0]['uptime'] if not system_info.empty else 0,
                'database_exists': not db_info.empty,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'connectivity': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def execute_sql_file(self, file_path: Union[str, Path]) -> bool:
        """
        Execute SQL file by splitting statements and executing in order.
        ClickHouse doesn't support multi-statement execution like PostgreSQL.
        """
        try:
            file_path = Path(file_path)
            logger.info(f"📄 Executing SQL file: {file_path}")
            
            # Read the entire SQL file
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            logger.info(f"📊 File size: {len(sql_content)} characters")
            
            # Split by semicolon and filter out empty statements
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            logger.info(f"📊 Found {len(statements)} statements")
            
            # Execute each statement separately
            for i, statement in enumerate(statements):
                if statement:
                    try:
                        logger.debug(f"📝 Executing statement {i+1}/{len(statements)}")
                        # Use execute_command for DDL statements (CREATE TABLE, CREATE VIEW, etc.)
                        # Use execute_query for SELECT statements
                        if statement.strip().upper().startswith(('SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN')):
                            self.execute_query(statement)
                        else:
                            self.execute_command(statement)
                        logger.debug(f"✅ Statement {i+1} executed successfully")
                    except Exception as stmt_error:
                        logger.warning(f"⚠️ Statement {i+1} failed: {stmt_error}")
                        # Continue with other statements - some may be materialized views that depend on tables
            
            logger.info("✅ SQL file executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute SQL file {file_path}: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


if __name__ == "__main__":
    """Test the ClickHouse client."""
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test ClickHouse client
    client = ClickHouseClient()
    
    try:
        # Test connection
        if client.connect():
            print("✅ ClickHouse connection successful")
            
            # Test health check
            health = client.health_check()
            print(f"Health check: {health}")
            
            # Test query
            result = client.execute_query("SELECT 1 as test, now() as current_time")
            print(f"Test query result: {result}")
            
        else:
            print("❌ ClickHouse connection failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.disconnect()
