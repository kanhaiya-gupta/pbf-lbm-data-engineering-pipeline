"""
MongoDB Client for Document Storage in Data Lake

This module provides MongoDB integration for storing unstructured document data
in the data lake layer, particularly useful for PBF-LB/M process metadata,
sensor configurations, and experimental results.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure, DuplicateKeyError
from bson import ObjectId
import json

logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    MongoDB client for document storage operations in the data lake.
    
    Handles connection management, document operations, indexing,
    and aggregation for PBF-LB/M research data.
    """
    
    def __init__(self, connection_string: str, database_name: str):
        """
        Initialize MongoDB client.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database to use
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
        
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self._client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000
            )
            
            # Test connection
            self._client.admin.command('ping')
            self._database = self._client[self.database_name]
            
            logger.info(f"Connected to MongoDB database: {self.database_name}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("Disconnected from MongoDB")
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a collection reference.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection: MongoDB collection object
        """
        if not self._database:
            raise RuntimeError("Not connected to MongoDB")
        return self._database[collection_name]
    
    def create_index(self, collection_name: str, index_spec: Union[str, List[tuple]], 
                    unique: bool = False, background: bool = True) -> str:
        """
        Create an index on a collection.
        
        Args:
            collection_name: Name of the collection
            index_spec: Index specification (field name or list of tuples)
            unique: Whether the index should be unique
            background: Whether to create index in background
            
        Returns:
            str: Name of the created index
        """
        try:
            collection = self.get_collection(collection_name)
            index_name = collection.create_index(
                index_spec,
                unique=unique,
                background=background
            )
            logger.info(f"Created index {index_name} on collection {collection_name}")
            return index_name
            
        except OperationFailure as e:
            logger.error(f"Failed to create index on {collection_name}: {e}")
            raise
    
    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """
        Insert a single document.
        
        Args:
            collection_name: Name of the collection
            document: Document to insert
            
        Returns:
            str: Inserted document ID
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Add metadata
            document['created_at'] = datetime.utcnow()
            document['updated_at'] = datetime.utcnow()
            
            result = collection.insert_one(document)
            logger.debug(f"Inserted document with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except DuplicateKeyError as e:
            logger.error(f"Duplicate key error inserting document: {e}")
            raise
        except OperationFailure as e:
            logger.error(f"Failed to insert document: {e}")
            raise
    
    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple documents.
        
        Args:
            collection_name: Name of the collection
            documents: List of documents to insert
            
        Returns:
            List[str]: List of inserted document IDs
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Add metadata to all documents
            current_time = datetime.utcnow()
            for doc in documents:
                doc['created_at'] = current_time
                doc['updated_at'] = current_time
            
            result = collection.insert_many(documents)
            logger.info(f"Inserted {len(result.inserted_ids)} documents")
            return [str(doc_id) for doc_id in result.inserted_ids]
            
        except OperationFailure as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
    
    def find_document(self, collection_name: str, query: Dict[str, Any], 
                     projection: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Find a single document.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            projection: Fields to include/exclude
            
        Returns:
            Optional[Dict]: Found document or None
        """
        try:
            collection = self.get_collection(collection_name)
            document = collection.find_one(query, projection)
            
            if document and '_id' in document:
                document['_id'] = str(document['_id'])
            
            return document
            
        except OperationFailure as e:
            logger.error(f"Failed to find document: {e}")
            raise
    
    def find_documents(self, collection_name: str, query: Dict[str, Any] = None,
                      projection: Optional[Dict[str, Any]] = None,
                      sort: Optional[List[tuple]] = None,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find multiple documents.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            projection: Fields to include/exclude
            sort: Sort specification
            limit: Maximum number of documents to return
            
        Returns:
            List[Dict]: List of found documents
        """
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(query or {}, projection)
            
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
            
            documents = list(cursor)
            
            # Convert ObjectId to string
            for doc in documents:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            return documents
            
        except OperationFailure as e:
            logger.error(f"Failed to find documents: {e}")
            raise
    
    def update_document(self, collection_name: str, query: Dict[str, Any],
                       update: Dict[str, Any], upsert: bool = False) -> bool:
        """
        Update a single document.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            update: Update operations
            upsert: Whether to insert if document doesn't exist
            
        Returns:
            bool: True if document was updated/inserted
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Add update timestamp
            update['$set'] = update.get('$set', {})
            update['$set']['updated_at'] = datetime.utcnow()
            
            result = collection.update_one(query, update, upsert=upsert)
            
            if result.modified_count > 0 or result.upserted_id:
                logger.debug(f"Updated document in {collection_name}")
                return True
            else:
                logger.debug(f"No document updated in {collection_name}")
                return False
                
        except OperationFailure as e:
            logger.error(f"Failed to update document: {e}")
            raise
    
    def update_documents(self, collection_name: str, query: Dict[str, Any],
                        update: Dict[str, Any]) -> int:
        """
        Update multiple documents.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            update: Update operations
            
        Returns:
            int: Number of documents updated
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Add update timestamp
            update['$set'] = update.get('$set', {})
            update['$set']['updated_at'] = datetime.utcnow()
            
            result = collection.update_many(query, update)
            logger.info(f"Updated {result.modified_count} documents in {collection_name}")
            return result.modified_count
            
        except OperationFailure as e:
            logger.error(f"Failed to update documents: {e}")
            raise
    
    def delete_document(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """
        Delete a single document.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            
        Returns:
            bool: True if document was deleted
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one(query)
            
            if result.deleted_count > 0:
                logger.debug(f"Deleted document from {collection_name}")
                return True
            else:
                logger.debug(f"No document deleted from {collection_name}")
                return False
                
        except OperationFailure as e:
            logger.error(f"Failed to delete document: {e}")
            raise
    
    def delete_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Delete multiple documents.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            
        Returns:
            int: Number of documents deleted
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} documents from {collection_name}")
            return result.deleted_count
            
        except OperationFailure as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform aggregation on a collection.
        
        Args:
            collection_name: Name of the collection
            pipeline: Aggregation pipeline
            
        Returns:
            List[Dict]: Aggregation results
        """
        try:
            collection = self.get_collection(collection_name)
            results = list(collection.aggregate(pipeline))
            
            # Convert ObjectId to string
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            return results
            
        except OperationFailure as e:
            logger.error(f"Failed to perform aggregation: {e}")
            raise
    
    def count_documents(self, collection_name: str, query: Dict[str, Any] = None) -> int:
        """
        Count documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            
        Returns:
            int: Number of documents
        """
        try:
            collection = self.get_collection(collection_name)
            return collection.count_documents(query or {})
            
        except OperationFailure as e:
            logger.error(f"Failed to count documents: {e}")
            raise
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict: Collection statistics
        """
        try:
            collection = self.get_collection(collection_name)
            stats = self._database.command("collStats", collection_name)
            
            return {
                'name': collection_name,
                'count': stats.get('count', 0),
                'size': stats.get('size', 0),
                'avgObjSize': stats.get('avgObjSize', 0),
                'storageSize': stats.get('storageSize', 0),
                'indexes': stats.get('nindexes', 0),
                'totalIndexSize': stats.get('totalIndexSize', 0)
            }
            
        except OperationFailure as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
    
    def create_text_index(self, collection_name: str, fields: List[str], 
                         language: str = 'english') -> str:
        """
        Create a text search index.
        
        Args:
            collection_name: Name of the collection
            fields: Fields to include in text index
            language: Language for text analysis
            
        Returns:
            str: Name of the created index
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Create text index specification
            text_spec = {field: "text" for field in fields}
            text_spec['weights'] = {field: 1 for field in fields}
            
            index_name = collection.create_index(
                list(text_spec.items()),
                default_language=language
            )
            
            logger.info(f"Created text index {index_name} on {collection_name}")
            return index_name
            
        except OperationFailure as e:
            logger.error(f"Failed to create text index: {e}")
            raise
    
    def text_search(self, collection_name: str, search_text: str, 
                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform text search on a collection.
        
        Args:
            collection_name: Name of the collection
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List[Dict]: Search results with scores
        """
        try:
            collection = self.get_collection(collection_name)
            
            results = collection.find(
                {"$text": {"$search": search_text}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            documents = list(results)
            
            # Convert ObjectId to string
            for doc in documents:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            return documents
            
        except OperationFailure as e:
            logger.error(f"Failed to perform text search: {e}")
            raise
