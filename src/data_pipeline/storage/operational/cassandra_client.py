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
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy
from cassandra.query import SimpleStatement, BatchStatement, ConsistencyLevel
from cassandra import ConsistencyLevel as CL
import json
import uuid

logger = logging.getLogger(__name__)


class CassandraClient:
    """
    Cassandra client for time-series and high-volume data operations in the operational layer.
    
    Handles time-series data storage, high-volume operational metrics,
    and distributed data management for PBF-LB/M systems.
    """
    
    def __init__(self, hosts: List[str], keyspace: str, 
                 username: Optional[str] = None, password: Optional[str] = None,
                 port: int = 9042):
        """
        Initialize Cassandra client.
        
        Args:
            hosts: List of Cassandra host addresses
            keyspace: Keyspace name
            username: Username for authentication
            password: Password for authentication
            port: Cassandra port
        """
        self.hosts = hosts
        self.keyspace = keyspace
        self.username = username
        self.password = password
        self.port = port
        self._cluster: Optional[Cluster] = None
        self._session: Optional[Session] = None
        
    def connect(self) -> bool:
        """
        Establish connection to Cassandra.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Configure authentication if provided
            auth_provider = None
            if self.username and self.password:
                auth_provider = PlainTextAuthProvider(
                    username=self.username,
                    password=self.password
                )
            
            # Configure load balancing policy
            load_balancing_policy = TokenAwarePolicy(
                DCAwareRoundRobinPolicy()
            )
            
            self._cluster = Cluster(
                hosts=self.hosts,
                port=self.port,
                auth_provider=auth_provider,
                load_balancing_policy=load_balancing_policy,
                connect_timeout=10,
                control_connection_timeout=10
            )
            
            self._session = self._cluster.connect()
            self._session.set_keyspace(self.keyspace)
            
            logger.info(f"Connected to Cassandra cluster: {self.keyspace}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Cassandra: {e}")
            return False
    
    def disconnect(self):
        """Close Cassandra connection."""
        if self._cluster:
            self._cluster.shutdown()
            logger.info("Disconnected from Cassandra")
    
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
            INSERT INTO {table_name} ({', '.join(columns)})
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
        Insert multiple rows in a batch.
        
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
            
            # Get columns from first row
            columns = list(rows[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            
            query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({placeholders})
            """
            
            batch = BatchStatement(consistency_level=CL.ONE)
            statement = SimpleStatement(query)
            
            for row in rows:
                values = [row.get(col) for col in columns]
                batch.add(statement, values)
            
            self._session.execute(batch)
            logger.info(f"Inserted {len(rows)} rows into {table_name}")
            return len(rows)
            
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
            
            query = f"SELECT {select_clause} FROM {table_name}"
            
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
            
            statement = SimpleStatement(cql)
            result = self._session.execute(statement, parameters or [])
            
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
            logger.error(f"Failed to drop table {table_name}: {e}")
            raise
