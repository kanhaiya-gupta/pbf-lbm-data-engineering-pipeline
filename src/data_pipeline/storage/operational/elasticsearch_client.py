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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, hosts: List[str] = None, 
                 username: Optional[str] = None, password: Optional[str] = None, 
                 verify_certs: bool = True):
        """
        Initialize Elasticsearch client.
        
        Args:
            config: Elasticsearch configuration dictionary (preferred)
            hosts: List of Elasticsearch host URLs (fallback if no config)
            username: Username for authentication (fallback if no config)
            password: Password for authentication (fallback if no config)
            verify_certs: Whether to verify SSL certificates (fallback if no config)
        """
        # Use config if provided, otherwise use individual parameters
        if config:
            self.hosts = config.get('hosts', ['localhost:9200'])
            self.username = config.get('username')
            self.password = config.get('password')
            self.verify_certs = config.get('verify_certs', True)
        else:
            self.hosts = hosts or ['localhost:9200']
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
    
    def count_documents(self, index_name: str) -> int:
        """
        Count documents in an index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            int: Number of documents in the index
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Elasticsearch")
            
            response = self._client.count(index=index_name)
            return response['count']
            
        except RequestError as e:
            logger.error(f"Failed to count documents in {index_name}: {e}")
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
    
    def get_all_process_logs(self) -> List[Dict[str, Any]]:
        """Get all process logs from Elasticsearch."""
        try:
            result = self.search('process_logs', {"query": {"match_all": {}}}, size=10000)
            return [hit['_source'] for hit in result.get('hits', {}).get('hits', [])]
        except Exception as e:
            logger.error(f"Failed to get process logs: {e}")
            return []
    
    def get_all_machine_logs(self) -> List[Dict[str, Any]]:
        """Get all machine logs from Elasticsearch."""
        try:
            result = self.search('machine_logs', {"query": {"match_all": {}}}, size=10000)
            return [hit['_source'] for hit in result.get('hits', {}).get('hits', [])]
        except Exception as e:
            logger.error(f"Failed to get machine logs: {e}")
            return []
    
    def get_all_sensor_logs(self) -> List[Dict[str, Any]]:
        """Get all sensor logs from Elasticsearch."""
        try:
            result = self.search('sensor_logs', {"query": {"match_all": {}}}, size=10000)
            return [hit['_source'] for hit in result.get('hits', {}).get('hits', [])]
        except Exception as e:
            logger.error(f"Failed to get sensor logs: {e}")
            return []
    
    def get_all_error_logs(self) -> List[Dict[str, Any]]:
        """Get all error logs from Elasticsearch."""
        try:
            result = self.search('error_logs', {"query": {"match_all": {}}}, size=10000)
            return [hit['_source'] for hit in result.get('hits', {}).get('hits', [])]
        except Exception as e:
            logger.error(f"Failed to get error logs: {e}")
            return []
    
    def get_all_performance_metrics(self) -> List[Dict[str, Any]]:
        """Get all performance metrics from Elasticsearch."""
        try:
            result = self.search('performance_metrics', {"query": {"match_all": {}}}, size=10000)
            return [hit['_source'] for hit in result.get('hits', {}).get('hits', [])]
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return []
    
    def load_data(
        self,
        df: Any,
        index_name: str,
        mode: str = "append",
        batch_size: int = 1000,
        doc_type: str = "_doc",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load Spark DataFrame data into Elasticsearch index.
        
        This method provides ETL pipeline integration for loading transformed
        Spark DataFrames into Elasticsearch, following the ETL architecture.
        
        Args:
            df: Spark DataFrame from transform modules
            index_name: Target Elasticsearch index name
            mode: Write mode (append, overwrite, ignore, error)
            batch_size: Batch size for processing
            doc_type: Document type (default: "_doc")
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Loading results and statistics
        """
        try:
            logger.info(f"Loading Spark DataFrame into Elasticsearch index: {index_name}")
            
            # Initialize result tracking
            result = {
                "success": False,
                "documents_loaded": 0,
                "documents_processed": 0,
                "errors": [],
                "warnings": [],
                "index_name": index_name,
                "mode": mode,
                "batch_size": batch_size,
                "doc_type": doc_type
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
                # Delete existing index and recreate
                try:
                    if self.index_exists(index_name):
                        self.delete_index(index_name)
                        logger.info(f"Deleted existing index: {index_name}")
                except Exception as e:
                    logger.warning(f"Could not delete existing index {index_name}: {e}")
            
            # Ensure index exists
            if not self.index_exists(index_name):
                self.create_index(index_name)
                logger.info(f"Created index: {index_name}")
            
            # Batch processing using Elasticsearch bulk API
            total_loaded = 0
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                try:
                    # Process batch using bulk API
                    batch_result = self._process_elasticsearch_batch(index_name, batch, doc_type)
                    
                    total_loaded += batch_result["loaded"]
                    
                    if batch_result["errors"]:
                        result["errors"].extend(batch_result["errors"])
                    
                    logger.info(f"Processed batch {i//batch_size + 1}: {batch_result['loaded']} documents")
                    
                except Exception as e:
                    error_msg = f"Error processing batch {i//batch_size + 1}: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Update result
            result["documents_loaded"] = total_loaded
            result["success"] = total_loaded > 0 and len(result["errors"]) == 0
            
            if result["success"]:
                logger.info(f"Successfully loaded {total_loaded} documents into index: {index_name}")
            else:
                logger.error(f"Failed to load data into Elasticsearch. Errors: {result['errors']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in load_data for index {index_name}: {str(e)}")
            return {
                "success": False,
                "documents_loaded": 0,
                "documents_processed": 0,
                "errors": [str(e)],
                "warnings": [],
                "index_name": index_name,
                "mode": mode,
                "batch_size": batch_size,
                "doc_type": doc_type
            }
    
    def _convert_spark_dataframe(self, df: Any) -> List[Dict[str, Any]]:
        """
        Convert Spark DataFrame to list of dictionaries for Elasticsearch insertion.
        
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
        Process Spark Row data to Elasticsearch-compatible format.
        
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
                    # Datetime objects - convert to ISO string
                    processed[key] = value.isoformat()
                elif isinstance(value, (int, float, str, bool)):
                    # Basic types
                    processed[key] = value
                elif isinstance(value, dict):
                    # Nested dictionaries - preserve structure
                    processed[key] = value
                elif isinstance(value, list):
                    # Lists - preserve structure
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
    
    def _process_elasticsearch_batch(
        self, 
        index_name: str, 
        batch: List[Dict[str, Any]], 
        doc_type: str = "_doc"
    ) -> Dict[str, Any]:
        """
        Process a batch of records for Elasticsearch insertion using bulk API.
        
        Args:
            index_name: Target Elasticsearch index
            batch: List of data dictionaries
            doc_type: Document type
            
        Returns:
            Dict[str, Any]: Batch processing results
        """
        try:
            # Prepare bulk actions
            actions = []
            
            for i, data in enumerate(batch):
                try:
                    # Generate document ID if not present
                    doc_id = data.get('id') or data.get('_id') or f"doc_{i}"
                    
                    # Remove ID fields from document body
                    doc_body = {k: v for k, v in data.items() if k not in ['id', '_id']}
                    
                    # Create bulk action
                    action = {
                        "_index": index_name,
                        "_type": doc_type,
                        "_id": doc_id,
                        "_source": doc_body
                    }
                    
                    actions.append(action)
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare document {i}: {str(e)}")
                    continue
            
            # Execute bulk operation
            if actions:
                success_count, failed_items = bulk(
                    self._client,
                    actions,
                    chunk_size=len(actions),
                    request_timeout=60
                )
                
                errors = []
                if failed_items:
                    for item in failed_items:
                        if 'index' in item and 'error' in item['index']:
                            errors.append(f"Document {item['index']['_id']}: {item['index']['error']}")
                
                return {
                    "loaded": success_count,
                    "errors": errors
                }
            else:
                return {
                    "loaded": 0,
                    "errors": ["No valid documents to process"]
                }
            
        except Exception as e:
            logger.error(f"Error processing Elasticsearch batch: {str(e)}")
            return {
                "loaded": 0,
                "errors": [str(e)]
            }