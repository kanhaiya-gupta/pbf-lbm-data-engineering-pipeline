"""
PostgreSQL CDC Connector for PBF-LB/M Data Pipeline

This module provides Change Data Capture capabilities for PostgreSQL databases.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from src.data_pipeline.config.postgres_config import get_postgres_config

logger = logging.getLogger(__name__)


class PostgresCDC:
    """
    PostgreSQL Change Data Capture connector.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PostgreSQL CDC connector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.postgres_config = get_postgres_config()
        self.connection = None
        self.replication_slot = None
        self.wal_sender = None
        self.change_handlers = []
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """Initialize database connection for CDC."""
        try:
            import psycopg2
            self.connection = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            logger.info("PostgreSQL CDC connection initialized successfully")
        except ImportError:
            logger.error("psycopg2 not available for PostgreSQL operations")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL CDC connection: {e}")
    
    def add_change_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a change handler for CDC events.
        
        Args:
            handler: Function to handle CDC events
        """
        self.change_handlers.append(handler)
    
    def _emit_change_event(self, event: Dict[str, Any]) -> None:
        """
        Emit a change event to all registered handlers.
        
        Args:
            event: Change event data
        """
        for handler in self.change_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in change handler: {e}")
    
    def create_replication_slot(self, slot_name: str) -> bool:
        """
        Create a replication slot for CDC.
        
        Args:
            slot_name: Name of the replication slot
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connection:
            logger.error("Database connection not initialized")
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Check if slot already exists
                cursor.execute("SELECT * FROM pg_replication_slots WHERE slot_name = %s", (slot_name,))
                if cursor.fetchone():
                    logger.info(f"Replication slot {slot_name} already exists")
                    return True
                
                # Create replication slot
                cursor.execute(f"SELECT pg_create_logical_replication_slot('{slot_name}', 'pgoutput')")
                result = cursor.fetchone()
                if result:
                    self.replication_slot = slot_name
                    logger.info(f"Created replication slot: {slot_name}")
                    return True
                else:
                    logger.error(f"Failed to create replication slot: {slot_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating replication slot: {e}")
            return False
    
    def drop_replication_slot(self, slot_name: str) -> bool:
        """
        Drop a replication slot.
        
        Args:
            slot_name: Name of the replication slot
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connection:
            logger.error("Database connection not initialized")
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"SELECT pg_drop_replication_slot('{slot_name}')")
                logger.info(f"Dropped replication slot: {slot_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error dropping replication slot: {e}")
            return False
    
    def start_cdc_stream(self, slot_name: str, tables: Optional[List[str]] = None) -> None:
        """
        Start CDC stream for specified tables.
        
        Args:
            slot_name: Name of the replication slot
            tables: List of tables to monitor (None for all tables)
        """
        logger.info(f"Starting CDC stream for slot: {slot_name}")
        
        if not self.connection:
            logger.error("Database connection not initialized")
            return
        
        try:
            # Create replication slot if it doesn't exist
            if not self.create_replication_slot(slot_name):
                logger.error("Failed to create replication slot")
                return
            
            # Start logical replication
            with self.connection.cursor() as cursor:
                # Set up logical replication
                cursor.execute("SET wal_level = logical")
                cursor.execute("SET max_replication_slots = 10")
                cursor.execute("SET max_wal_senders = 10")
                
                # Start replication
                if tables:
                    table_list = ','.join(tables)
                    cursor.execute(f"SELECT pg_start_replication('{slot_name}', '{table_list}')")
                else:
                    cursor.execute(f"SELECT pg_start_replication('{slot_name}')")
                
                logger.info("CDC stream started successfully")
                
        except Exception as e:
            logger.error(f"Error starting CDC stream: {e}")
    
    def stop_cdc_stream(self) -> None:
        """Stop CDC stream."""
        logger.info("Stopping CDC stream")
        
        try:
            if self.connection:
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT pg_stop_replication()")
                logger.info("CDC stream stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping CDC stream: {e}")
    
    def process_wal_changes(self, slot_name: str) -> None:
        """
        Process WAL (Write-Ahead Log) changes for CDC.
        
        Args:
            slot_name: Name of the replication slot
        """
        logger.info(f"Processing WAL changes for slot: {slot_name}")
        
        if not self.connection:
            logger.error("Database connection not initialized")
            return
        
        try:
            with self.connection.cursor() as cursor:
                # Get WAL changes
                cursor.execute(f"SELECT * FROM pg_logical_slot_get_changes('{slot_name}', NULL, NULL)")
                changes = cursor.fetchall()
                
                for change in changes:
                    try:
                        # Parse WAL change
                        change_event = self._parse_wal_change(change)
                        if change_event:
                            self._emit_change_event(change_event)
                    except Exception as e:
                        logger.error(f"Error processing WAL change: {e}")
                        continue
                
                logger.info(f"Processed {len(changes)} WAL changes")
                
        except Exception as e:
            logger.error(f"Error processing WAL changes: {e}")
    
    def _parse_wal_change(self, change: tuple) -> Optional[Dict[str, Any]]:
        """
        Parse WAL change into structured event.
        
        Args:
            change: WAL change tuple
            
        Returns:
            Optional[Dict[str, Any]]: Parsed change event
        """
        try:
            # WAL change format: (lsn, xid, data)
            lsn, xid, data = change
            
            # Parse the data (this is a simplified parser)
            # In a real implementation, you would use pgoutput or similar
            change_event = {
                "lsn": lsn,
                "xid": xid,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "event_type": "change"
            }
            
            return change_event
            
        except Exception as e:
            logger.error(f"Error parsing WAL change: {e}")
            return None
    
    def get_replication_slots(self) -> List[Dict[str, Any]]:
        """
        Get list of replication slots.
        
        Returns:
            List[Dict[str, Any]]: List of replication slots
        """
        if not self.connection:
            logger.error("Database connection not initialized")
            return []
        
        slots = []
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        slot_name,
                        plugin,
                        slot_type,
                        datoid,
                        database,
                        active,
                        xmin,
                        catalog_xmin,
                        restart_lsn,
                        confirmed_flush_lsn
                    FROM pg_replication_slots
                """)
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                for row in rows:
                    slot = dict(zip(columns, row))
                    slots.append(slot)
                
                logger.info(f"Retrieved {len(slots)} replication slots")
                
        except Exception as e:
            logger.error(f"Error getting replication slots: {e}")
        
        return slots
    
    def get_replication_status(self) -> Dict[str, Any]:
        """
        Get replication status information.
        
        Returns:
            Dict[str, Any]: Replication status
        """
        if not self.connection:
            logger.error("Database connection not initialized")
            return {}
        
        status = {}
        
        try:
            with self.connection.cursor() as cursor:
                # Get replication statistics
                cursor.execute("""
                    SELECT 
                        client_addr,
                        application_name,
                        state,
                        sent_lsn,
                        write_lsn,
                        flush_lsn,
                        replay_lsn,
                        write_lag,
                        flush_lag,
                        replay_lag
                    FROM pg_stat_replication
                """)
                
                replication_stats = cursor.fetchall()
                status["replication_stats"] = replication_stats
                
                # Get WAL statistics
                cursor.execute("""
                    SELECT 
                        wal_records,
                        wal_fpi,
                        wal_bytes,
                        wal_buffers_full,
                        wal_write,
                        wal_sync,
                        wal_write_time,
                        wal_sync_time
                    FROM pg_stat_wal
                """)
                
                wal_stats = cursor.fetchone()
                status["wal_stats"] = wal_stats
                
                logger.info("Retrieved replication status")
                
        except Exception as e:
            logger.error(f"Error getting replication status: {e}")
        
        return status
    
    def close_connection(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL CDC connection closed")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "connection_initialized": self.connection is not None,
            "replication_slot": self.replication_slot,
            "change_handlers_count": len(self.change_handlers),
            "ingestion_timestamp": datetime.now().isoformat()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()
