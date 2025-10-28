"""
MongoDB Client for Unstructured Data in Operational Layer

This module provides MongoDB integration for unstructured data storage
in the operational layer. Particularly useful for PBF-LB/M document data,
images, files, and unstructured content storage.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError, OperationFailure
from pymongo.database import Database
from pymongo.collection import Collection
from gridfs import GridFS
from bson import ObjectId
import json

logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    MongoDB client for unstructured data operations in the operational layer.
    
    Handles document storage, GridFS file operations, and unstructured data
    management for PBF-LB/M systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 host: str = None, port: int = None, 
                 username: str = None, password: str = None,
                 database: str = None, auth_source: str = None):
        """
        Initialize MongoDB client.
        
        Args:
            config: MongoDB configuration dictionary (preferred)
            host: MongoDB host (fallback if no config)
            port: MongoDB port (fallback if no config)
            username: Username for authentication (fallback if no config)
            password: Password for authentication (fallback if no config)
            database: Database name (fallback if no config)
            auth_source: Authentication database (fallback if no config)
        """
        # Use config if provided, otherwise use individual parameters
        if config:
            self.host = config.get('host', 'localhost')
            self.port = config.get('port', 27017)
            self.username = config.get('username')
            self.password = config.get('password')
            self.database = config.get('database', 'pbf_data_lake')
            self.auth_source = config.get('auth_source', 'admin')
        else:
            # Fallback to individual parameters
            self.host = host or 'localhost'
            self.port = port or 27017
            self.username = username
            self.password = password
            self.database = database or 'pbf_data_lake'
            self.auth_source = auth_source or 'admin'
        
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._fs: Optional[GridFS] = None
        self.connected: bool = False
        
        # Connection metrics
        self.connection_attempts: int = 0
        self.last_connection_time: Optional[datetime] = None
        self.total_operations: int = 0
        self.failed_operations: int = 0
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection_attempts += 1
            logger.info(f"Connecting to MongoDB: {self.host}:{self.port}")
            
            # Construct connection URI
            if self.username and self.password:
                uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.auth_source}"
                logger.info(f"Using authentication for user: {self.username}")
            else:
                uri = f"mongodb://{self.host}:{self.port}"
                logger.info("Using no authentication")
            
            # Create MongoDB client
            self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self._client.admin.command('ping')
            
            # Get database and GridFS
            self._db = self._client[self.database]
            self._fs = GridFS(self._db)
            
            self.connected = True
            self.last_connection_time = datetime.utcnow()
            
            logger.info(f"✅ Connected to MongoDB: {self.database}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"❌ MongoDB connection error: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self._client:
            self._client.close()
            self.connected = False
            logger.info("🔌 Disconnected from MongoDB")
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a MongoDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection: MongoDB collection object
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        return self._db[collection_name]
    
    def find_documents(self, collection_name: str, query: Dict[str, Any] = None, 
                      limit: int = None, skip: int = None, 
                      sort: List[tuple] = None) -> List[Dict[str, Any]]:
        """
        Find documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query: MongoDB query dictionary
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            sort: List of (field, direction) tuples for sorting
            
        Returns:
            List[Dict[str, Any]]: List of documents
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(query or {})
            
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            documents = list(cursor)
            self.total_operations += 1
            
            logger.debug(f"Found {len(documents)} documents in {collection_name}")
            return documents
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"❌ Failed to find documents in {collection_name}: {e}")
            return []
    
    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> Optional[ObjectId]:
        """
        Insert a single document.
        
        Args:
            collection_name: Name of the collection
            document: Document to insert
            
        Returns:
            ObjectId: ID of the inserted document, or None if failed
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_one(document)
            self.total_operations += 1
            
            logger.debug(f"Inserted document with ID: {result.inserted_id}")
            return result.inserted_id
            
        except DuplicateKeyError as e:
            self.failed_operations += 1
            logger.warning(f"⚠️ Duplicate key error in {collection_name}: {e}")
            return None
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"❌ Failed to insert document in {collection_name}: {e}")
            return None
    
    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[ObjectId]:
        """
        Insert multiple documents.
        
        Args:
            collection_name: Name of the collection
            documents: List of documents to insert
            
        Returns:
            List[ObjectId]: List of inserted document IDs
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_many(documents)
            self.total_operations += 1
            
            logger.debug(f"Inserted {len(result.inserted_ids)} documents")
            return result.inserted_ids
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"❌ Failed to insert documents in {collection_name}: {e}")
            return []
    
    def update_document(self, collection_name: str, query: Dict[str, Any], 
                       update: Dict[str, Any], upsert: bool = False) -> bool:
        """
        Update a single document.
        
        Args:
            collection_name: Name of the collection
            query: Query to find the document
            update: Update operations
            upsert: Create document if it doesn't exist
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one(query, update, upsert=upsert)
            self.total_operations += 1
            
            logger.debug(f"Updated {result.modified_count} document(s)")
            return result.modified_count > 0 or result.upserted_id is not None
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"❌ Failed to update document in {collection_name}: {e}")
            return False
    
    def delete_document(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """
        Delete a single document.
        
        Args:
            collection_name: Name of the collection
            query: Query to find the document to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one(query)
            self.total_operations += 1
            
            logger.debug(f"Deleted {result.deleted_count} document(s)")
            return result.deleted_count > 0
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"❌ Failed to delete document in {collection_name}: {e}")
            return False
    
    def store_file(self, file_data: bytes, filename: str, 
                   metadata: Dict[str, Any] = None) -> Optional[ObjectId]:
        """
        Store a file in GridFS.
        
        Args:
            file_data: File content as bytes
            filename: Name of the file
            metadata: Optional metadata for the file
            
        Returns:
            ObjectId: ID of the stored file, or None if failed
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            file_id = self._fs.put(file_data, filename=filename, metadata=metadata or {})
            self.total_operations += 1
            
            logger.debug(f"Stored file {filename} with ID: {file_id}")
            return file_id
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"❌ Failed to store file {filename}: {e}")
            return None
    
    def upload_file(self, file_path: str, file_data: bytes, 
                   metadata: Dict[str, Any] = None) -> Optional[ObjectId]:
        """
        Upload a file to GridFS (alias for store_file for compatibility).
        
        Args:
            file_path: Path/name of the file
            file_data: File content as bytes
            metadata: Optional metadata for the file
            
        Returns:
            ObjectId: ID of the stored file, or None if failed
        """
        return self.store_file(file_data, file_path, metadata)
    
    def get_file(self, file_id: ObjectId) -> Optional[bytes]:
        """
        Retrieve a file from GridFS.
        
        Args:
            file_id: ID of the file to retrieve
            
        Returns:
            bytes: File content, or None if not found
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            file_data = self._fs.get(file_id).read()
            self.total_operations += 1
            
            logger.debug(f"Retrieved file with ID: {file_id}")
            return file_data
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"❌ Failed to retrieve file {file_id}: {e}")
            return None
    
    def delete_file(self, file_id: ObjectId) -> bool:
        """
        Delete a file from GridFS.
        
        Args:
            file_id: ID of the file to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            self._fs.delete(file_id)
            self.total_operations += 1
            
            logger.debug(f"Deleted file with ID: {file_id}")
            return True
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"❌ Failed to delete file {file_id}: {e}")
            return False
    
    def create_index(self, collection_name: str, index_spec: List[tuple], 
                    unique: bool = False, background: bool = True) -> bool:
        """
        Create an index on a collection.
        
        Args:
            collection_name: Name of the collection
            index_spec: List of (field, direction) tuples for the index
            unique: Whether the index should be unique
            background: Whether to create the index in the background
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            collection = self.get_collection(collection_name)
            collection.create_index(index_spec, unique=unique, background=background)
            self.total_operations += 1
            
            logger.debug(f"Created index on {collection_name}: {index_spec}")
            return True
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"❌ Failed to create index on {collection_name}: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict[str, Any]: Collection statistics
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            collection = self.get_collection(collection_name)
            stats = self._db.command("collStats", collection_name)
            self.total_operations += 1
            
            return {
                'count': stats.get('count', 0),
                'size': stats.get('size', 0),
                'avgObjSize': stats.get('avgObjSize', 0),
                'storageSize': stats.get('storageSize', 0),
                'indexes': stats.get('nindexes', 0),
                'totalIndexSize': stats.get('totalIndexSize', 0)
            }
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"❌ Failed to get stats for {collection_name}: {e}")
            return {}
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection and operation statistics.
        
        Returns:
            Dict[str, Any]: Connection statistics
        """
        return {
            'connected': self.connected,
            'connection_attempts': self.connection_attempts,
            'last_connection_time': self.last_connection_time,
            'total_operations': self.total_operations,
            'failed_operations': self.failed_operations,
            'success_rate': (self.total_operations - self.failed_operations) / max(self.total_operations, 1) * 100
        }
    
    def load_data(
        self,
        df: Any,
        collection_name: str,
        mode: str = "append",
        batch_size: int = 1000,
        use_gridfs: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load Spark DataFrame data into MongoDB collection.
        
        This method provides ETL pipeline integration for loading transformed
        Spark DataFrames into MongoDB, following the ETL architecture.
        
        Args:
            df: Spark DataFrame from transform modules
            collection_name: Target collection name
            mode: Write mode (append, overwrite, ignore, error)
            batch_size: Batch size for processing
            use_gridfs: Whether to use GridFS for large files
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Loading results and statistics
        """
        try:
            logger.info(f"Loading Spark DataFrame into MongoDB collection: {collection_name}")
            
            # Initialize result tracking
            result = {
                "success": False,
                "documents_loaded": 0,
                "documents_processed": 0,
                "errors": [],
                "warnings": [],
                "collection_name": collection_name,
                "mode": mode,
                "batch_size": batch_size,
                "gridfs_files": 0
            }
            
            # Convert Spark DataFrame to list of dictionaries
            data_list = self._convert_spark_dataframe(df)
            if not data_list:
                result["warnings"].append("No data to load")
                return result
            
            result["documents_processed"] = len(data_list)
            logger.info(f"Converted {len(data_list)} documents from Spark DataFrame")
            
            # Handle different modes
            if mode == "overwrite":
                # Clear existing collection
                collection = self.get_collection(collection_name)
                collection.drop()
                logger.info(f"Cleared existing collection: {collection_name}")
            
            # Batch processing
            total_loaded = 0
            gridfs_count = 0
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                try:
                    # Process batch with GridFS support
                    batch_result = self._process_batch(
                        collection_name, batch, use_gridfs
                    )
                    
                    total_loaded += batch_result["loaded"]
                    gridfs_count += batch_result["gridfs_files"]
                    
                    if batch_result["errors"]:
                        result["errors"].extend(batch_result["errors"])
                    
                    logger.info(f"Processed batch {i//batch_size + 1}: {batch_result['loaded']} documents")
                    
                except Exception as e:
                    error_msg = f"Error processing batch {i//batch_size + 1}: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Update result
            result["documents_loaded"] = total_loaded
            result["gridfs_files"] = gridfs_count
            result["success"] = total_loaded > 0 and len(result["errors"]) == 0
            
            if result["success"]:
                logger.info(f"Successfully loaded {total_loaded} documents into {collection_name}")
                if gridfs_count > 0:
                    logger.info(f"Stored {gridfs_count} files in GridFS")
            else:
                logger.error(f"Failed to load data into {collection_name}. Errors: {result['errors']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in load_data for collection {collection_name}: {str(e)}")
            return {
                "success": False,
                "documents_loaded": 0,
                "documents_processed": 0,
                "errors": [str(e)],
                "warnings": [],
                "collection_name": collection_name,
                "mode": mode,
                "batch_size": batch_size,
                "gridfs_files": 0
            }
    
    def _convert_spark_dataframe(self, df: Any) -> List[Dict[str, Any]]:
        """
        Convert Spark DataFrame to list of dictionaries for MongoDB insertion.
        
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
        Process Spark Row data to MongoDB-compatible format.
        
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
                    # Nested dictionaries
                    processed[key] = value
                elif isinstance(value, list):
                    # Lists
                    processed[key] = value
                elif hasattr(value, '__dict__'):
                    # Complex objects - convert to dict
                    processed[key] = value.__dict__
                else:
                    # Fallback - convert to string
                    processed[key] = str(value)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing Spark row: {str(e)}")
            return row_dict  # Return original if processing fails
    
    def _process_batch(
        self, 
        collection_name: str, 
        batch: List[Dict[str, Any]], 
        use_gridfs: bool = False
    ) -> Dict[str, Any]:
        """
        Process a batch of documents with optional GridFS support.
        
        Args:
            collection_name: Target collection name
            batch: Batch of documents to process
            use_gridfs: Whether to use GridFS for large files
            
        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            result = {
                "loaded": 0,
                "errors": [],
                "gridfs_files": 0
            }
            
            processed_documents = []
            
            for document in batch:
                try:
                    processed_doc = document.copy()
                    
                    # Handle GridFS for large files
                    if use_gridfs and 'file_path' in processed_doc and 'file_size' in processed_doc:
                        if processed_doc['file_size'] is not None and processed_doc['file_size'] > 1024 * 1024:  # Files larger than 1MB
                            # Store in GridFS and update document
                            file_content = processed_doc.get('file_content', '')
                            if file_content:
                                file_id = self.upload_file(
                                    processed_doc['file_path'], 
                                    file_content.encode() if isinstance(file_content, str) else file_content,
                                    metadata={"source": "load_data"}
                                )
                                processed_doc['gridfs_file_id'] = str(file_id)
                                result["gridfs_files"] += 1
                            
                            # Remove the large content from document
                            processed_doc.pop('file_content', None)
                    
                    processed_documents.append(processed_doc)
                    
                except Exception as e:
                    error_msg = f"Error processing document: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.warning(error_msg)
            
            # Insert processed documents
            if processed_documents:
                inserted_ids = self.insert_documents(collection_name, processed_documents)
                result["loaded"] = len(inserted_ids)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return {
                "loaded": 0,
                "errors": [str(e)],
                "gridfs_files": 0
            }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def get_all_process_images(self) -> List[Dict[str, Any]]:
        """Get all process images from MongoDB."""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self._db.get_collection('process_images')
            return list(collection.find({}))
        except Exception as e:
            logger.error(f"Failed to get process images: {e}")
            return []
    
    def get_all_ct_scan_images(self) -> List[Dict[str, Any]]:
        """Get all CT scan images from MongoDB."""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self._db.get_collection('ct_scan_images')
            return list(collection.find({}))
        except Exception as e:
            logger.error(f"Failed to get CT scan images: {e}")
            return []
    
    def get_all_powder_bed_images(self) -> List[Dict[str, Any]]:
        """Get all powder bed images from MongoDB."""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self._db.get_collection('powder_bed_images')
            return list(collection.find({}))
        except Exception as e:
            logger.error(f"Failed to get powder bed images: {e}")
            return []
    
    def get_all_machine_build_files(self) -> List[Dict[str, Any]]:
        """Get all machine build files from MongoDB."""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self._db.get_collection('machine_build_files')
            return list(collection.find({}))
        except Exception as e:
            logger.error(f"Failed to get machine build files: {e}")
            return []
    
    def get_all_3d_model_files(self) -> List[Dict[str, Any]]:
        """Get all 3D model files from MongoDB."""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self._db.get_collection('model_3d_files')
            return list(collection.find({}))
        except Exception as e:
            logger.error(f"Failed to get 3D model files: {e}")
            return []
    
    def get_all_raw_sensor_data(self) -> List[Dict[str, Any]]:
        """Get all raw sensor data from MongoDB."""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self._db.get_collection('raw_sensor_data')
            return list(collection.find({}))
        except Exception as e:
            logger.error(f"Failed to get raw sensor data: {e}")
            return []
    
    def get_all_process_logs(self) -> List[Dict[str, Any]]:
        """Get all process logs from MongoDB."""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self._db.get_collection('process_logs')
            return list(collection.find({}))
        except Exception as e:
            logger.error(f"Failed to get process logs: {e}")
            return []
    
    def get_all_machine_configurations(self) -> List[Dict[str, Any]]:
        """Get all machine configurations from MongoDB."""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self._db.get_collection('machine_configurations')
            return list(collection.find({}))
        except Exception as e:
            logger.error(f"Failed to get machine configurations: {e}")
            return []
    
    def close_connection(self):
        """Close MongoDB connection."""
        try:
            if self._client:
                self._client.close()
                self.connected = False
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")