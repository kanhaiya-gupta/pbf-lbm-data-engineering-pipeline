"""
Cassandra Client for Time-Series and High-Volume Data in Operational Layer

This module provides Cassandra integration for high-volume time-series data
and operational metrics in the operational layer. Particularly useful for
PBF-LB/M sensor data, process monitoring metrics, and high-frequency
operational data storage.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import logging
from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy, RetryPolicy, ExponentialReconnectionPolicy
from cassandra.query import SimpleStatement, BatchStatement, ConsistencyLevel
from cassandra import ConsistencyLevel as CL
from cassandra import Unavailable, OperationTimedOut, ReadTimeout, WriteTimeout
import json
import uuid
import time

from src.data_pipeline.config.cassandra_config import CassandraConfig, get_cassandra_config

logger = logging.getLogger(__name__)


class CassandraClient:
    """
    Cassandra client for time-series and high-volume data operations in the operational layer.
    
    Handles time-series data storage, high-volume operational metrics,
    and distributed data management for PBF-LB/M systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, hosts: List[str] = None, 
                 keyspace: str = None, username: Optional[str] = None, 
                 password: Optional[str] = None, port: int = None):
        """
        Initialize Cassandra client.
        
        Args:
            config: Cassandra configuration dictionary (preferred)
            hosts: List of Cassandra host addresses (fallback if no config)
            keyspace: Keyspace name (fallback if no config)
            username: Username for authentication (fallback if no config)
            password: Password for authentication (fallback if no config)
            port: Cassandra port (fallback if no config)
        """
        # Use config if provided, otherwise use individual parameters
        if config:
            self.hosts = config.get('hosts', ["localhost"])
            self.keyspace = config.get('keyspace', "pbf_timeseries")
            self.username = config.get('username')
            self.password = config.get('password')
            self.port = config.get('port', 9042)
            self.batch_size = config.get('batch_size', 100)
            self.max_batch_size = config.get('max_batch_size', 1000)
            self.protocol_version = config.get('protocol_version', 3)
            self.connection_timeout = config.get('connection_timeout', 10)
            self.max_connections = config.get('max_connections', 50)
            self.max_requests_per_connection = config.get('max_requests_per_connection', 32768)
        else:
            # Fallback to individual parameters
            self.hosts = hosts or ["localhost"]
            self.keyspace = keyspace or "pbf_timeseries"
            self.username = username
            self.password = password
            self.port = port or 9042
            self.batch_size = 100
            self.max_batch_size = 1000
            self.protocol_version = 3
            self.connection_timeout = 10
            self.max_connections = 50
            self.max_requests_per_connection = 32768
        
        self._cluster: Optional[Cluster] = None
        self._session: Optional[Session] = None
        self.connected: bool = False
        
        # Connection metrics
        self.connection_attempts: int = 0
        self.last_connection_time: Optional[datetime] = None
        self.total_queries: int = 0
        self.failed_queries: int = 0
        
    def connect(self) -> bool:
        """
        Establish connection to Cassandra.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection_attempts += 1
            logger.info(f"Connecting to Cassandra cluster: {self.hosts}")
            
            # Configure authentication if provided
            auth_provider = None
            if self.username and self.password:
                auth_provider = PlainTextAuthProvider(
                    username=self.username,
                    password=self.password
                )
                logger.info(f"Using authentication for user: {self.username}")
            
            # Configure load balancing policy
            load_balancing_policy = TokenAwarePolicy(
                DCAwareRoundRobinPolicy()
            )
            
            # Configure reconnection policy
            reconnection_policy = ExponentialReconnectionPolicy(
                base_delay=2.0,
                max_delay=60.0
            )
            
            # Use instance variables for configuration
            connect_timeout = self.connection_timeout
            control_timeout = self.connection_timeout
            max_connections = self.max_connections
            max_requests = self.max_requests_per_connection
            protocol_version = self.protocol_version
            
            self._cluster = Cluster(
                self.hosts,
                port=self.port,
                auth_provider=auth_provider,
                load_balancing_policy=load_balancing_policy,
                reconnection_policy=reconnection_policy,
                connect_timeout=connect_timeout,
                control_connection_timeout=control_timeout,
                protocol_version=protocol_version
            )
            
            self._session = self._cluster.connect()
            
            # Create keyspace if it doesn't exist
            create_keyspace_cql = f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH REPLICATION = {{
                'class': 'SimpleStrategy',
                'replication_factor': 1
            }}
            """
            self._session.execute(create_keyspace_cql)
            logger.info(f"✅ Keyspace '{self.keyspace}' created or verified")
            
            # Use keyspace for session (required for some operations)
            self._session.set_keyspace(self.keyspace)
            
            # Set default consistency level
            self._session.default_consistency_level = CL.LOCAL_QUORUM
            
            # Test connection
            self._session.execute("SELECT now() FROM system.local")
            
            self.connected = True
            self.last_connection_time = datetime.utcnow()
            
            logger.info(f"✅ Connected to Cassandra cluster: {self.hosts}")
            logger.info(f"🔑 Keyspace: {self.keyspace}")
            logger.info(f"📊 Cluster metadata: {self._cluster.metadata.cluster_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Cassandra: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Close Cassandra connection.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self._cluster:
                self._cluster.shutdown()
            
            self.connected = False
            logger.info("✅ Disconnected from Cassandra")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting from Cassandra: {e}")
            return False
    
    def close_connection(self):
        """Close Cassandra connection (alias for disconnect)."""
        self.disconnect()
    
    def health_check(self) -> bool:
        """
        Perform health check on Cassandra connection.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if not self.connected or not self._session:
                return False
            
            # Simple query to test connection
            result = self._session.execute("SELECT now() FROM system.local")
            return len(list(result)) > 0
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information and metrics.
        
        Returns:
            Dict[str, Any]: Connection information
        """
        return {
            "connected": self.connected,
            "hosts": self.hosts,
            "keyspace": self.keyspace,
            "connection_attempts": self.connection_attempts,
            "last_connection_time": self.last_connection_time,
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "success_rate": (self.total_queries - self.failed_queries) / max(self.total_queries, 1) * 100
        }
    
    def use_keyspace(self, keyspace: str) -> bool:
        """
        Switch to a different keyspace.
        
        Args:
            keyspace: Keyspace name to switch to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            self._session.set_keyspace(keyspace)
            self.keyspace = keyspace
            logger.info(f"✅ Switched to keyspace: {keyspace}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to switch to keyspace {keyspace}: {e}")
            return False
    
    def create_keyspace(self, keyspace_name: str, replication_factor: int = 3) -> bool:
        """
        Create a keyspace.
        
        Args:
            keyspace_name: Name of the keyspace
            replication_factor: Replication factor
            
        Returns:
            bool: True if keyspace created successfully
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            query = f"""
            CREATE KEYSPACE IF NOT EXISTS {keyspace_name}
            WITH REPLICATION = {{
                'class': 'SimpleStrategy',
                'replication_factor': {replication_factor}
            }}
            """
            
            self._session.execute(query)
            logger.info(f"Created keyspace: {keyspace_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create keyspace {keyspace_name}: {e}")
            raise
    
    def create_table(self, table_name: str, columns: Dict[str, str], 
                    primary_key: List[str], clustering_key: Optional[List[str]] = None) -> bool:
        """
        Create a table.
        
        Args:
            table_name: Name of the table
            columns: Dictionary of column names and types
            primary_key: List of primary key columns
            clustering_key: List of clustering key columns
            
        Returns:
            bool: True if table created successfully
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            # Build column definitions
            column_defs = []
            for col_name, col_type in columns.items():
                column_defs.append(f"{col_name} {col_type}")
            
            # Build primary key definition
            if clustering_key:
                pk_def = f"({', '.join(primary_key)}, {', '.join(clustering_key)})"
            else:
                pk_def = f"({', '.join(primary_key)})"
            
            query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(column_defs)},
                PRIMARY KEY {pk_def}
            )
            """
            
            self._session.execute(query)
            logger.info(f"Created table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise
    
    def create_time_series_table(self, table_name: str, 
                                partition_key: str = 'sensor_id',
                                time_column: str = 'timestamp') -> bool:
        """
        Create a time-series table optimized for sensor data.
        
        Args:
            table_name: Name of the table
            partition_key: Partition key column (usually sensor_id)
            time_column: Time column name
            
        Returns:
            bool: True if table created successfully
        """
        columns = {
            partition_key: 'text',
            time_column: 'timestamp',
            'value': 'double',
            'quality': 'int',
            'metadata': 'text'
        }
        
        return self.create_table(
            table_name=table_name,
            columns=columns,
            primary_key=[partition_key],
            clustering_key=[time_column]
        )
    
    def insert_row(self, table_name: str, data: Dict[str, Any]) -> bool:
        """
        Insert a single row.
        
        Args:
            table_name: Name of the table
            data: Dictionary of column-value pairs
            
        Returns:
            bool: True if insert successful
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            columns = list(data.keys())
            values = list(data.values())
            placeholders = ', '.join(['?' for _ in columns])
            
            query = f"""
            INSERT INTO {self.keyspace}.{table_name} ({', '.join(columns)})
            VALUES ({placeholders})
            """
            
            statement = SimpleStatement(query, consistency_level=CL.ONE)
            self._session.execute(statement, values)
            
            logger.debug(f"Inserted row into {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert row into {table_name}: {e}")
            raise
    
    def insert_batch(self, table_name: str, rows: List[Dict[str, Any]]) -> int:
        """
        Insert multiple rows in a batch with chunking to avoid batch size limits.
        
        Args:
            table_name: Name of the table
            rows: List of dictionaries containing row data
            
        Returns:
            int: Number of rows inserted
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            if not rows:
                return 0
            
            # Use instance variable for batch size
            batch_size = self.batch_size
            
            # Get columns from first row
            columns = list(rows[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            
            query = f"""
            INSERT INTO {self.keyspace}.{table_name} ({', '.join(columns)})
            VALUES ({placeholders})
            """
            
            statement = self._session.prepare(query)
            total_inserted = 0
            
            # Process rows in chunks
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]
                
                batch = BatchStatement(consistency_level=CL.ONE)
                
                for row in chunk:
                    values = [row.get(col) for col in columns]
                    batch.add(statement, values)
                
                self._session.execute(batch)
                total_inserted += len(chunk)
            
            logger.info(f"Inserted {total_inserted} rows into {table_name}")
            return total_inserted
            
        except Exception as e:
            logger.error(f"Failed to insert batch into {table_name}: {e}")
            raise
    
    def select_rows(self, table_name: str, where_clause: Optional[str] = None,
                   columns: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select rows from a table.
        
        Args:
            table_name: Name of the table
            where_clause: WHERE clause (without WHERE keyword)
            columns: List of columns to select
            limit: Maximum number of rows to return
            
        Returns:
            List[Dict]: List of row dictionaries
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            # Build SELECT clause
            if columns:
                select_clause = ', '.join(columns)
            else:
                select_clause = '*'
            
            query = f"SELECT {select_clause} FROM {self.keyspace}.{table_name}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
            
            if limit:
                query += f" LIMIT {limit}"
            
            statement = SimpleStatement(query)
            result = self._session.execute(statement)
            
            rows = []
            for row in result:
                row_dict = {}
                for column in row._fields:
                    value = getattr(row, column)
                    # Convert UUID to string for JSON serialization
                    if isinstance(value, uuid.UUID):
                        value = str(value)
                    row_dict[column] = value
                rows.append(row_dict)
            
            return rows
            
        except Exception as e:
            logger.error(f"Failed to select rows from {table_name}: {e}")
            raise
    
    def select_time_series(self, table_name: str, sensor_id: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select time-series data for a sensor.
        
        Args:
            table_name: Name of the table
            sensor_id: Sensor identifier
            start_time: Start time for data range
            end_time: End time for data range
            limit: Maximum number of rows to return
            
        Returns:
            List[Dict]: List of time-series data points
        """
        try:
            where_conditions = [f"sensor_id = '{sensor_id}'"]
            
            if start_time:
                where_conditions.append(f"timestamp >= '{start_time.isoformat()}'")
            
            if end_time:
                where_conditions.append(f"timestamp <= '{end_time.isoformat()}'")
            
            where_clause = ' AND '.join(where_conditions)
            
            return self.select_rows(
                table_name=table_name,
                where_clause=where_clause,
                limit=limit
            )
            
        except Exception as e:
            logger.error(f"Failed to select time-series data: {e}")
            raise
    
    def update_row(self, table_name: str, data: Dict[str, Any], 
                  where_clause: str) -> bool:
        """
        Update rows in a table.
        
        Args:
            table_name: Name of the table
            data: Dictionary of column-value pairs to update
            where_clause: WHERE clause (without WHERE keyword)
            
        Returns:
            bool: True if update successful
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            set_clause = ', '.join([f"{col} = ?" for col in data.keys()])
            values = list(data.values())
            
            query = f"""
            UPDATE {table_name}
            SET {set_clause}
            WHERE {where_clause}
            """
            
            statement = SimpleStatement(query, consistency_level=CL.ONE)
            self._session.execute(statement, values)
            
            logger.debug(f"Updated rows in {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update rows in {table_name}: {e}")
            raise
    
    def delete_rows(self, table_name: str, where_clause: str) -> bool:
        """
        Delete rows from a table.
        
        Args:
            table_name: Name of the table
            where_clause: WHERE clause (without WHERE keyword)
            
        Returns:
            bool: True if delete successful
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            query = f"DELETE FROM {table_name} WHERE {where_clause}"
            
            statement = SimpleStatement(query, consistency_level=CL.ONE)
            self._session.execute(statement)
            
            logger.debug(f"Deleted rows from {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete rows from {table_name}: {e}")
            raise
    
    def execute_cql(self, cql: str, parameters: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a CQL query.
        
        Args:
            cql: CQL query string
            parameters: Query parameters
            
        Returns:
            List[Dict]: Query results
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            self.total_queries += 1
            
            if parameters:
                # Use prepared statement for parameterized queries
                statement = self._session.prepare(cql)
                result = self._session.execute(statement, parameters)
            else:
                # Use simple statement for non-parameterized queries
                statement = SimpleStatement(cql)
                result = self._session.execute(statement)
            
            rows = []
            for row in result:
                row_dict = {}
                for column in row._fields:
                    value = getattr(row, column)
                    # Convert UUID to string for JSON serialization
                    if isinstance(value, uuid.UUID):
                        value = str(value)
                    row_dict[column] = value
                rows.append(row_dict)
            
            logger.debug(f"Executed CQL query, returned {len(rows)} rows")
            return rows
            
        except Exception as e:
            self.failed_queries += 1
            logger.error(f"Failed to execute CQL: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get table information.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict: Table information
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            # Get table schema
            cql = f"""
            SELECT column_name, type, kind
            FROM system_schema.columns
            WHERE keyspace_name = '{self.keyspace}'
            AND table_name = '{table_name}'
            """
            
            columns = self.execute_cql(cql)
            
            return {
                'table_name': table_name,
                'keyspace': self.keyspace,
                'columns': columns
            }
            
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            raise
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get cluster information.
        
        Returns:
            Dict: Cluster information
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            # Get cluster name
            cluster_name = self._session.cluster.metadata.cluster_name
            
            # Get host information
            hosts = []
            for host in self._session.cluster.metadata.all_hosts():
                hosts.append({
                    'address': str(host.address),
                    'datacenter': host.datacenter,
                    'rack': host.rack,
                    'is_up': host.is_up
                })
            
            return {
                'cluster_name': cluster_name,
                'hosts': hosts,
                'keyspace': self.keyspace
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            raise
    
    def create_index(self, table_name: str, column_name: str) -> bool:
        """
        Create an index on a column.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            
        Returns:
            bool: True if index created successfully
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            index_name = f"{table_name}_{column_name}_idx"
            
            query = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table_name} ({column_name})
            """
            
            self._session.execute(query)
            logger.info(f"Created index {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index on {table_name}.{column_name}: {e}")
            raise
    
    def drop_table(self, table_name: str) -> bool:
        """
        Drop a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            bool: True if table dropped successfully
        """
        try:
            if not self._session:
                raise RuntimeError("Not connected to Cassandra")
            
            query = f"DROP TABLE IF EXISTS {table_name}"
            self._session.execute(query)
            
            logger.info(f"Dropped table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping table {table_name}: {e}")
            raise
    
    def load_data(
        self,
        df: Any,
        table_name: str,
        mode: str = "append",
        batch_size: int = 25,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load Spark DataFrame data into Cassandra table.
        
        This method provides ETL pipeline integration for loading transformed
        Spark DataFrames into Cassandra, following the ETL architecture.
        
        Args:
            df: Spark DataFrame from transform modules
            table_name: Target table name
            mode: Write mode (append, overwrite, ignore, error)
            batch_size: Batch size for processing (Cassandra-optimized)
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Loading results and statistics
        """
        try:
            logger.info(f"Loading Spark DataFrame into Cassandra table: {table_name}")
            
            # Initialize result tracking
            result = {
                "success": False,
                "records_loaded": 0,
                "records_processed": 0,
                "errors": [],
                "warnings": [],
                "table_name": table_name,
                "mode": mode,
                "batch_size": batch_size
            }
            
            # Convert Spark DataFrame to list of dictionaries
            data_list = self._convert_spark_dataframe(df)
            if not data_list:
                result["warnings"].append("No data to load")
                return result
            
            result["records_processed"] = len(data_list)
            logger.info(f"Converted {len(data_list)} records from Spark DataFrame")
            
            # Handle different modes
            if mode == "overwrite":
                # Truncate existing table
                self.execute_query(f"TRUNCATE {table_name}")
                logger.info(f"Truncated existing table: {table_name}")
            
            # Batch processing with Cassandra-optimized batch size
            total_loaded = 0
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                try:
                    # Process batch using existing method
                    batch_result = self._process_batch(table_name, batch)
                    
                    total_loaded += batch_result["loaded"]
                    
                    if batch_result["errors"]:
                        result["errors"].extend(batch_result["errors"])
                    
                    logger.info(f"Processed batch {i//batch_size + 1}: {batch_result['loaded']} records")
                    
                except Exception as e:
                    error_msg = f"Error processing batch {i//batch_size + 1}: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Update result
            result["records_loaded"] = total_loaded
            result["success"] = total_loaded > 0 and len(result["errors"]) == 0
            
            if result["success"]:
                logger.info(f"Successfully loaded {total_loaded} records into {table_name}")
            else:
                logger.error(f"Failed to load data into {table_name}. Errors: {result['errors']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in load_data for table {table_name}: {str(e)}")
            return {
                "success": False,
                "records_loaded": 0,
                "records_processed": 0,
                "errors": [str(e)],
                "warnings": [],
                "table_name": table_name,
                "mode": mode,
                "batch_size": batch_size
            }
    
    def _convert_spark_dataframe(self, df: Any) -> List[Dict[str, Any]]:
        """
        Convert Spark DataFrame to list of dictionaries for Cassandra insertion.
        
        Args:
            df: Spark DataFrame from transform modules
            
        Returns:
            List[Dict[str, Any]]: Converted data list
        """
        try:
            if hasattr(df, 'collect'):
                # Spark DataFrame - convert to list of dicts
                rows = df.collect()
                data_list = []
                
                for row in rows:
                    # Convert Row to dictionary
                    row_dict = row.asDict()
                    
                    # Handle Spark-specific data types
                    processed_dict = self._process_spark_row(row_dict)
                    data_list.append(processed_dict)
                
                return data_list
                
            elif isinstance(df, list):
                # Already a list of dictionaries
                return df
                
            elif isinstance(df, dict):
                # Single dictionary
                return [df]
                
            else:
                logger.warning(f"Unsupported DataFrame type: {type(df)}")
                return []
                
        except Exception as e:
            logger.error(f"Error converting Spark DataFrame: {str(e)}")
            return []
    
    def _process_spark_row(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Spark Row data to Cassandra-compatible format.
        
        Args:
            row_dict: Dictionary from Spark Row
            
        Returns:
            Dict[str, Any]: Processed dictionary
        """
        try:
            processed = {}
            
            for key, value in row_dict.items():
                # Handle None values
                if value is None:
                    processed[key] = None
                # Handle Spark-specific types
                elif hasattr(value, 'isoformat'):
                    # Datetime objects
                    processed[key] = value
                elif isinstance(value, (int, float, str, bool)):
                    # Basic types
                    processed[key] = value
                elif isinstance(value, dict):
                    # Nested dictionaries - convert to JSON string for Cassandra
                    import json
                    processed[key] = json.dumps(value)
                elif isinstance(value, list):
                    # Lists - convert to JSON string for Cassandra
                    import json
                    processed[key] = json.dumps(value)
                elif hasattr(value, '__dict__'):
                    # Complex objects - convert to JSON string
                    import json
                    processed[key] = json.dumps(value.__dict__, default=str)
                else:
                    # Fallback - convert to string
                    processed[key] = str(value)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing Spark row: {str(e)}")
            return row_dict  # Return original if processing fails
    
    
    def _process_batch(self, table_name: str, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of records for Cassandra insertion.
        
        Args:
            table_name: Target table name
            batch: Batch of records to process
            
        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            result = {
                "loaded": 0,
                "errors": []
            }
            
            if batch:
                try:
                    # Use existing insert_batch method
                    inserted_count = self.insert_batch(table_name, batch)
                    result["loaded"] = inserted_count
                    
                except Exception as e:
                    error_msg = f"Error inserting batch: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return {
                "loaded": 0,
                "errors": [str(e)]
            }
