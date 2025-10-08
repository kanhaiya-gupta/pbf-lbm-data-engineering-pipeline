"""
Elasticsearch Client for Search and Analytics in Data Warehouse

This module provides Elasticsearch integration for advanced search,
analytics, and real-time data exploration in the data warehouse layer.
Particularly useful for PBF-LB/M process data analysis, quality metrics
search, and research data discovery.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError
from elasticsearch.helpers import bulk, scan
import json

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """
    Elasticsearch client for search and analytics operations in the data warehouse.
    
    Handles indexing, searching, aggregations, and analytics for PBF-LB/M
    research data and quality metrics.
    """
    
    def __init__(self, hosts: List[str], username: Optional[str] = None, 
                 password: Optional[str] = None, verify_certs: bool = True):
        """
        Initialize Elasticsearch client.
        
        Args:
            hosts: List of Elasticsearch host URLs
            username: Username for authentication
            password: Password for authentication
            verify_certs: Whether to verify SSL certificates
        """
        self.hosts = hosts
        self.username = username
        self.password = password
        self.verify_certs = verify_certs
        self._client: Optional[Elasticsearch] = None
        
    def connect(self) -> bool:
        """
        Establish connection to Elasticsearch.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)
            
            self._client = Elasticsearch(
                hosts=self.hosts,
                http_auth=auth,
                verify_certs=self.verify_certs,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
            
            # Test connection
            if not self._client.ping():
                raise ConnectionError("Failed to ping Elasticsearch")
            
            logger.info(f"Connected to Elasticsearch cluster: {self._client.info()['cluster_name']}")
            return True
            
        except ConnectionError as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False
    
    def disconnect(self):
        """Close Elasticsearch connection."""
        if self._client:
            self._client.close()
            logger.info("Disconnected from Elasticsearch")
    
    def create_index(self, index_name: str, mapping: Dict[str, Any], 
                    settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create an Elasticsearch index.
        
        Args:
            index_name: Name of the index
            mapping: Index mapping definition
            settings: Index settings
            
        Returns:
            bool: True if index created successfully
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            body = {"mappings": mapping}
            if settings:
                body["settings"] = settings
            
            self._client.indices.create(index=index_name, body=body)
            logger.info(f"Created index: {index_name}")
            return True
            
        except RequestError as e:
            if e.error == 'resource_already_exists_exception':
                logger.warning(f"Index {index_name} already exists")
                return True
            else:
                logger.error(f"Failed to create index {index_name}: {e}")
                raise
    
    def delete_index(self, index_name: str) -> bool:
        """
        Delete an Elasticsearch index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            bool: True if index deleted successfully
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            self._client.indices.delete(index=index_name)
            logger.info(f"Deleted index: {index_name}")
            return True
            
        except NotFoundError:
            logger.warning(f"Index {index_name} not found")
            return True
        except RequestError as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            raise
    
    def index_document(self, index_name: str, document: Dict[str, Any], 
                      doc_id: Optional[str] = None) -> str:
        """
        Index a single document.
        
        Args:
            index_name: Name of the index
            document: Document to index
            doc_id: Optional document ID
            
        Returns:
            str: Document ID
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            # Add metadata
            document['indexed_at'] = datetime.utcnow().isoformat()
            
            response = self._client.index(
                index=index_name,
                body=document,
                id=doc_id
            )
            
            doc_id = response['_id']
            logger.debug(f"Indexed document with ID: {doc_id}")
            return doc_id
            
        except RequestError as e:
            logger.error(f"Failed to index document: {e}")
            raise
    
    def bulk_index(self, index_name: str, documents: List[Dict[str, Any]]) -> int:
        """
        Bulk index multiple documents.
        
        Args:
            index_name: Name of the index
            documents: List of documents to index
            
        Returns:
            int: Number of successfully indexed documents
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            # Prepare bulk actions
            actions = []
            current_time = datetime.utcnow().isoformat()
            
            for doc in documents:
                action = {
                    "_index": index_name,
                    "_source": doc
                }
                action["_source"]["indexed_at"] = current_time
                actions.append(action)
            
            # Perform bulk indexing
            success_count, failed_items = bulk(
                self._client,
                actions,
                chunk_size=1000,
                request_timeout=60
            )
            
            if failed_items:
                logger.warning(f"Failed to index {len(failed_items)} documents")
            
            logger.info(f"Successfully indexed {success_count} documents")
            return success_count
            
        except RequestError as e:
            logger.error(f"Failed to bulk index documents: {e}")
            raise
    
    def get_document(self, index_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            index_name: Name of the index
            doc_id: Document ID
            
        Returns:
            Optional[Dict]: Document if found, None otherwise
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            response = self._client.get(index=index_name, id=doc_id)
            return response['_source']
            
        except NotFoundError:
            logger.debug(f"Document {doc_id} not found in index {index_name}")
            return None
        except RequestError as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            raise
    
    def update_document(self, index_name: str, doc_id: str, 
                       update_data: Dict[str, Any]) -> bool:
        """
        Update a document.
        
        Args:
            index_name: Name of the index
            doc_id: Document ID
            update_data: Data to update
            
        Returns:
            bool: True if document updated successfully
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            # Add update timestamp
            update_data['updated_at'] = datetime.utcnow().isoformat()
            
            self._client.update(
                index=index_name,
                id=doc_id,
                body={"doc": update_data}
            )
            
            logger.debug(f"Updated document {doc_id}")
            return True
            
        except NotFoundError:
            logger.warning(f"Document {doc_id} not found for update")
            return False
        except RequestError as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            raise
    
    def delete_document(self, index_name: str, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            index_name: Name of the index
            doc_id: Document ID
            
        Returns:
            bool: True if document deleted successfully
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            self._client.delete(index=index_name, id=doc_id)
            logger.debug(f"Deleted document {doc_id}")
            return True
            
        except NotFoundError:
            logger.warning(f"Document {doc_id} not found for deletion")
            return False
        except RequestError as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise
    
    def search(self, index_name: str, query: Dict[str, Any], 
               size: int = 10, from_: int = 0) -> Dict[str, Any]:
        """
        Search documents in an index.
        
        Args:
            index_name: Name of the index
            query: Search query
            size: Number of results to return
            from_: Starting offset
            
        Returns:
            Dict: Search results
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            response = self._client.search(
                index=index_name,
                body=query,
                size=size,
                from_=from_
            )
            
            return {
                'hits': response['hits']['hits'],
                'total': response['hits']['total']['value'],
                'max_score': response['hits']['max_score'],
                'took': response['took']
            }
            
        except RequestError as e:
            logger.error(f"Failed to search index {index_name}: {e}")
            raise
    
    def multi_search(self, searches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform multiple searches in a single request.
        
        Args:
            searches: List of search requests
            
        Returns:
            List[Dict]: List of search results
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            response = self._client.msearch(body=searches)
            return response['responses']
            
        except RequestError as e:
            logger.error(f"Failed to perform multi-search: {e}")
            raise
    
    def aggregate(self, index_name: str, aggregation: Dict[str, Any],
                 query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform aggregation on an index.
        
        Args:
            index_name: Name of the index
            aggregation: Aggregation definition
            query: Optional query filter
            
        Returns:
            Dict: Aggregation results
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            search_body = {"aggs": aggregation}
            if query:
                search_body["query"] = query
            
            response = self._client.search(
                index=index_name,
                body=search_body,
                size=0  # We only want aggregations
            )
            
            return response['aggregations']
            
        except RequestError as e:
            logger.error(f"Failed to perform aggregation: {e}")
            raise
    
    def scroll_search(self, index_name: str, query: Dict[str, Any],
                     scroll: str = "5m", size: int = 1000) -> List[Dict[str, Any]]:
        """
        Perform scroll search for large result sets.
        
        Args:
            index_name: Name of the index
            query: Search query
            scroll: Scroll timeout
            size: Number of results per scroll
            
        Returns:
            List[Dict]: All matching documents
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            documents = []
            for doc in scan(
                self._client,
                query=query,
                index=index_name,
                scroll=scroll,
                size=size
            ):
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents via scroll")
            return documents
            
        except RequestError as e:
            logger.error(f"Failed to perform scroll search: {e}")
            raise
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Dict: Index statistics
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            stats = self._client.indices.stats(index=index_name)
            return stats['indices'][index_name]
            
        except RequestError as e:
            logger.error(f"Failed to get index stats: {e}")
            raise
    
    def create_alias(self, index_name: str, alias_name: str) -> bool:
        """
        Create an alias for an index.
        
        Args:
            index_name: Name of the index
            alias_name: Name of the alias
            
        Returns:
            bool: True if alias created successfully
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            self._client.indices.put_alias(index=index_name, name=alias_name)
            logger.info(f"Created alias {alias_name} for index {index_name}")
            return True
            
        except RequestError as e:
            logger.error(f"Failed to create alias: {e}")
            raise
    
    def reindex(self, source_index: str, dest_index: str, 
                query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Reindex data from one index to another.
        
        Args:
            source_index: Source index name
            dest_index: Destination index name
            query: Optional query to filter source data
            
        Returns:
            Dict: Reindex task information
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            reindex_body = {
                "source": {"index": source_index},
                "dest": {"index": dest_index}
            }
            
            if query:
                reindex_body["source"]["query"] = query
            
            response = self._client.reindex(body=reindex_body, wait_for_completion=False)
            logger.info(f"Started reindexing from {source_index} to {dest_index}")
            return response
            
        except RequestError as e:
            logger.error(f"Failed to reindex: {e}")
            raise
    
    def text_search(self, index_name: str, search_text: str, 
                   fields: List[str], size: int = 10) -> Dict[str, Any]:
        """
        Perform full-text search on specified fields.
        
        Args:
            index_name: Name of the index
            search_text: Text to search for
            fields: Fields to search in
            size: Number of results to return
            
        Returns:
            Dict: Search results
        """
        try:
            query = {
                "multi_match": {
                    "query": search_text,
                    "fields": fields,
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
            
            return self.search(index_name, {"query": query}, size=size)
            
        except RequestError as e:
            logger.error(f"Failed to perform text search: {e}")
            raise
    
    def range_search(self, index_name: str, field: str, 
                    gte: Optional[Any] = None, lte: Optional[Any] = None,
                    size: int = 10) -> Dict[str, Any]:
        """
        Perform range search on a field.
        
        Args:
            index_name: Name of the index
            field: Field to search on
            gte: Greater than or equal value
            lte: Less than or equal value
            size: Number of results to return
            
        Returns:
            Dict: Search results
        """
        try:
            range_query = {"range": {field: {}}}
            
            if gte is not None:
                range_query["range"][field]["gte"] = gte
            if lte is not None:
                range_query["range"][field]["lte"] = lte
            
            return self.search(index_name, {"query": range_query}, size=size)
            
        except RequestError as e:
            logger.error(f"Failed to perform range search: {e}")
            raise
