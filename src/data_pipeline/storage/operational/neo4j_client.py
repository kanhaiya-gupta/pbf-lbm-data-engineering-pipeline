"""
Neo4j Client for Graph Relationships in Operational Layer

This module provides Neo4j integration for managing complex relationships
and graph data in the operational layer. Particularly useful for PBF-LB/M
process relationships, equipment hierarchies, material flow tracking,
and complex operational dependencies.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError, TransientError
import json

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j client for graph relationship operations in the operational layer.
    
    Handles node and relationship management, graph queries, and complex
    relationship analysis for PBF-LB/M operational systems.
    """
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j server URI
            username: Username for authentication
            password: Password for authentication
            database: Database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None
        
    def connect(self) -> bool:
        """
        Establish connection to Neo4j.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=30
            )
            
            # Test connection
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            
            logger.info(f"Connected to Neo4j database: {self.database}")
            return True
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def disconnect(self):
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            logger.info("Disconnected from Neo4j")
    
    def create_node(self, label: str, properties: Dict[str, Any]) -> int:
        """
        Create a single node.
        
        Args:
            label: Node label
            properties: Node properties
            
        Returns:
            int: Node ID
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                # Add metadata
                properties['created_at'] = datetime.utcnow().isoformat()
                properties['updated_at'] = datetime.utcnow().isoformat()
                
                result = session.run(
                    f"CREATE (n:{label} $props) RETURN id(n) as node_id",
                    props=properties
                )
                
                node_id = result.single()["node_id"]
                logger.debug(f"Created node with ID: {node_id}")
                return node_id
                
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            raise
    
    def create_nodes(self, label: str, nodes_data: List[Dict[str, Any]]) -> List[int]:
        """
        Create multiple nodes.
        
        Args:
            label: Node label
            nodes_data: List of node properties
            
        Returns:
            List[int]: List of node IDs
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                # Add metadata to all nodes
                current_time = datetime.utcnow().isoformat()
                for node_data in nodes_data:
                    node_data['created_at'] = current_time
                    node_data['updated_at'] = current_time
                
                result = session.run(
                    f"UNWIND $nodes AS node CREATE (n:{label}) SET n = node RETURN id(n) as node_id",
                    nodes=nodes_data
                )
                
                node_ids = [record["node_id"] for record in result]
                logger.info(f"Created {len(node_ids)} nodes")
                return node_ids
                
        except Exception as e:
            logger.error(f"Failed to create nodes: {e}")
            raise
    
    def create_relationship(self, from_node_id: int, to_node_id: int,
                           relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a relationship between two nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Type of relationship
            properties: Relationship properties
            
        Returns:
            int: Relationship ID
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                # Add metadata
                if properties is None:
                    properties = {}
                properties['created_at'] = datetime.utcnow().isoformat()
                
                result = session.run(
                    """
                    MATCH (a), (b)
                    WHERE id(a) = $from_id AND id(b) = $to_id
                    CREATE (a)-[r:{} $props]->(b)
                    RETURN id(r) as rel_id
                    """.format(relationship_type),
                    from_id=from_node_id,
                    to_id=to_node_id,
                    props=properties
                )
                
                rel_id = result.single()["rel_id"]
                logger.debug(f"Created relationship with ID: {rel_id}")
                return rel_id
                
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            raise
    
    def find_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        Find a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Optional[Dict]: Node data if found, None otherwise
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    "MATCH (n) WHERE id(n) = $node_id RETURN n",
                    node_id=node_id
                )
                
                record = result.single()
                if record:
                    node = dict(record["n"])
                    node["id"] = node_id
                    return node
                return None
                
        except Exception as e:
            logger.error(f"Failed to find node {node_id}: {e}")
            raise
    
    def find_nodes_by_label(self, label: str, properties: Optional[Dict[str, Any]] = None,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find nodes by label and optional properties.
        
        Args:
            label: Node label
            properties: Optional property filters
            limit: Maximum number of nodes to return
            
        Returns:
            List[Dict]: List of matching nodes
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                query = f"MATCH (n:{label})"
                params = {}
                
                if properties:
                    conditions = []
                    for key, value in properties.items():
                        conditions.append(f"n.{key} = ${key}")
                        params[key] = value
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                query += " RETURN n, id(n) as node_id"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                result = session.run(query, params)
                
                nodes = []
                for record in result:
                    node = dict(record["n"])
                    node["id"] = record["node_id"]
                    nodes.append(node)
                
                return nodes
                
        except Exception as e:
            logger.error(f"Failed to find nodes by label {label}: {e}")
            raise
    
    def find_relationships(self, from_node_id: Optional[int] = None,
                          to_node_id: Optional[int] = None,
                          relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find relationships between nodes.
        
        Args:
            from_node_id: Optional source node ID
            to_node_id: Optional target node ID
            relationship_type: Optional relationship type
            
        Returns:
            List[Dict]: List of relationships
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                query = "MATCH (a)-[r]->(b)"
                conditions = []
                params = {}
                
                if from_node_id is not None:
                    conditions.append("id(a) = $from_id")
                    params["from_id"] = from_node_id
                
                if to_node_id is not None:
                    conditions.append("id(b) = $to_id")
                    params["to_id"] = to_node_id
                
                if relationship_type:
                    conditions.append("type(r) = $rel_type")
                    params["rel_type"] = relationship_type
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " RETURN a, r, b, id(r) as rel_id, id(a) as from_id, id(b) as to_id"
                
                result = session.run(query, params)
                
                relationships = []
                for record in result:
                    rel_data = {
                        "id": record["rel_id"],
                        "type": type(record["r"]).__name__,
                        "from_node": dict(record["a"]),
                        "to_node": dict(record["b"]),
                        "from_id": record["from_id"],
                        "to_id": record["to_id"],
                        "properties": dict(record["r"])
                    }
                    relationships.append(rel_data)
                
                return relationships
                
        except Exception as e:
            logger.error(f"Failed to find relationships: {e}")
            raise
    
    def update_node(self, node_id: int, properties: Dict[str, Any]) -> bool:
        """
        Update node properties.
        
        Args:
            node_id: Node ID
            properties: Properties to update
            
        Returns:
            bool: True if update successful
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                # Add update timestamp
                properties['updated_at'] = datetime.utcnow().isoformat()
                
                result = session.run(
                    "MATCH (n) WHERE id(n) = $node_id SET n += $props RETURN n",
                    node_id=node_id,
                    props=properties
                )
                
                if result.single():
                    logger.debug(f"Updated node {node_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}")
            raise
    
    def delete_node(self, node_id: int, delete_relationships: bool = True) -> bool:
        """
        Delete a node.
        
        Args:
            node_id: Node ID
            delete_relationships: Whether to delete relationships
            
        Returns:
            bool: True if delete successful
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                if delete_relationships:
                    query = "MATCH (n) WHERE id(n) = $node_id DETACH DELETE n"
                else:
                    query = "MATCH (n) WHERE id(n) = $node_id DELETE n"
                
                result = session.run(query, node_id=node_id)
                
                logger.debug(f"Deleted node {node_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            raise
    
    def execute_cypher(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query.
        
        Args:
            cypher: Cypher query string
            parameters: Query parameters
            
        Returns:
            List[Dict]: Query results
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                result = session.run(cypher, parameters or {})
                
                records = []
                for record in result:
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Convert Neo4j types to Python types
                        if hasattr(value, '__dict__'):
                            record_dict[key] = dict(value)
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                return records
                
        except Exception as e:
            logger.error(f"Failed to execute Cypher query: {e}")
            raise
    
    def get_shortest_path(self, from_node_id: int, to_node_id: int,
                         relationship_types: Optional[List[str]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Find shortest path between two nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_types: Optional relationship types to follow
            
        Returns:
            Optional[List[Dict]]: Path if found, None otherwise
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                if relationship_types:
                    rel_types = "|".join(relationship_types)
                    query = f"""
                    MATCH (a), (b), p = shortestPath((a)-[:{rel_types}*]-(b))
                    WHERE id(a) = $from_id AND id(b) = $to_id
                    RETURN p
                    """
                else:
                    query = """
                    MATCH (a), (b), p = shortestPath((a)-[*]-(b))
                    WHERE id(a) = $from_id AND id(b) = $to_id
                    RETURN p
                    """
                
                result = session.run(query, from_id=from_node_id, to_id=to_node_id)
                record = result.single()
                
                if record:
                    path = record["p"]
                    path_data = []
                    
                    for node in path.nodes:
                        path_data.append({
                            "type": "node",
                            "id": node.id,
                            "labels": list(node.labels),
                            "properties": dict(node)
                        })
                    
                    for rel in path.relationships:
                        path_data.append({
                            "type": "relationship",
                            "id": rel.id,
                            "type_name": rel.type,
                            "properties": dict(rel)
                        })
                    
                    return path_data
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to find shortest path: {e}")
            raise
    
    def get_node_degree(self, node_id: int, direction: str = "both") -> Dict[str, int]:
        """
        Get node degree (number of relationships).
        
        Args:
            node_id: Node ID
            direction: "in", "out", or "both"
            
        Returns:
            Dict: Degree information
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                if direction == "in":
                    query = """
                    MATCH (n)<-[r]-(m)
                    WHERE id(n) = $node_id
                    RETURN count(r) as degree
                    """
                elif direction == "out":
                    query = """
                    MATCH (n)-[r]->(m)
                    WHERE id(n) = $node_id
                    RETURN count(r) as degree
                    """
                else:  # both
                    query = """
                    MATCH (n)-[r]-(m)
                    WHERE id(n) = $node_id
                    RETURN count(r) as degree
                    """
                
                result = session.run(query, node_id=node_id)
                degree = result.single()["degree"]
                
                return {
                    "node_id": node_id,
                    "direction": direction,
                    "degree": degree
                }
                
        except Exception as e:
            logger.error(f"Failed to get node degree: {e}")
            raise
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dict: Graph statistics
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                # Get node count
                node_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = node_result.single()["node_count"]
                
                # Get relationship count
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = rel_result.single()["rel_count"]
                
                # Get label counts
                label_result = session.run("""
                    CALL db.labels() YIELD label
                    CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value
                    RETURN label, value.count as count
                """)
                
                label_counts = {}
                for record in label_result:
                    label_counts[record["label"]] = record["count"]
                
                return {
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "label_counts": label_counts,
                    "database": self.database
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            raise
