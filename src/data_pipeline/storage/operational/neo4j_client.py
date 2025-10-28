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
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.data_pipeline.config.neo4j_config import get_neo4j_config, Neo4jConfig

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j client for graph relationship operations in the operational layer.
    
    Handles node and relationship management, graph queries, and complex
    relationship analysis for PBF-LB/M operational systems.
    """
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        """
        Initialize Neo4j client.
        
        Args:
            config: Neo4j configuration object. If None, loads from environment.
        """
        if config is None:
            config = get_neo4j_config()
        
        self.config = config
        self.uri = config.uri
        self.username = config.username
        self.password = config.password
        self.database = config.default_database
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
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout,
                connection_timeout=self.config.connection_timeout,
                max_transaction_retry_time=self.config.max_transaction_retry_time,
                encrypted=self.config.encrypted,
                trust=self.config.trust
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
    
    # =============================================================================
    # KNOWLEDGE GRAPH SPECIFIC METHODS
    # =============================================================================
    
    def create_process_node(self, process_id: str, properties: Dict[str, Any]) -> int:
        """
        Create a process node with PBF-LB/M specific properties.
        
        Args:
            process_id: Unique process identifier
            properties: Process properties
            
        Returns:
            int: Node ID
        """
        properties['process_id'] = process_id
        properties['node_type'] = 'Process'
        return self.create_node(self.config.process_node_label, properties)
    
    def create_machine_node(self, machine_id: str, properties: Dict[str, Any]) -> int:
        """
        Create a machine node with PBF-LB/M specific properties.
        
        Args:
            machine_id: Unique machine identifier
            properties: Machine properties
            
        Returns:
            int: Node ID
        """
        properties['machine_id'] = machine_id
        properties['node_type'] = 'Machine'
        return self.create_node(self.config.machine_node_label, properties)
    
    def create_part_node(self, part_id: str, properties: Dict[str, Any]) -> int:
        """
        Create a part node with PBF-LB/M specific properties.
        
        Args:
            part_id: Unique part identifier
            properties: Part properties
            
        Returns:
            int: Node ID
        """
        properties['part_id'] = part_id
        properties['node_type'] = 'Part'
        return self.create_node(self.config.part_node_label, properties)
    
    def create_sensor_node(self, sensor_id: str, properties: Dict[str, Any]) -> int:
        """
        Create a sensor node with PBF-LB/M specific properties.
        
        Args:
            sensor_id: Unique sensor identifier
            properties: Sensor properties
            
        Returns:
            int: Node ID
        """
        properties['sensor_id'] = sensor_id
        properties['node_type'] = 'Sensor'
        return self.create_node(self.config.sensor_node_label, properties)
    
    def create_build_node(self, build_id: str, properties: Dict[str, Any]) -> int:
        """
        Create a build node with PBF-LB/M specific properties.
        
        Args:
            build_id: Unique build identifier
            properties: Build properties
            
        Returns:
            int: Node ID
        """
        properties['build_id'] = build_id
        properties['node_type'] = 'Build'
        return self.create_node(self.config.build_node_label, properties)
    
    def create_process_machine_relationship(self, process_node_id: int, machine_node_id: int, 
                                          properties: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a relationship between process and machine.
        
        Args:
            process_node_id: Process node ID
            machine_node_id: Machine node ID
            properties: Relationship properties
            
        Returns:
            int: Relationship ID
        """
        return self.create_relationship(
            process_node_id, 
            machine_node_id, 
            self.config.process_uses_machine, 
            properties
        )
    
    def create_process_part_relationship(self, process_node_id: int, part_node_id: int, 
                                       properties: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a relationship between process and part.
        
        Args:
            process_node_id: Process node ID
            part_node_id: Part node ID
            properties: Relationship properties
            
        Returns:
            int: Relationship ID
        """
        return self.create_relationship(
            process_node_id, 
            part_node_id, 
            self.config.process_creates_part, 
            properties
        )
    
    def create_machine_sensor_relationship(self, machine_node_id: int, sensor_node_id: int, 
                                         properties: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a relationship between machine and sensor.
        
        Args:
            machine_node_id: Machine node ID
            sensor_node_id: Sensor node ID
            properties: Relationship properties
            
        Returns:
            int: Relationship ID
        """
        return self.create_relationship(
            machine_node_id, 
            sensor_node_id, 
            self.config.machine_has_sensor, 
            properties
        )
    
    def create_part_build_relationship(self, part_node_id: int, build_node_id: int, 
                                     properties: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a relationship between part and build.
        
        Args:
            part_node_id: Part node ID
            build_node_id: Build node ID
            properties: Relationship properties
            
        Returns:
            int: Relationship ID
        """
        return self.create_relationship(
            part_node_id, 
            build_node_id, 
            self.config.part_belongs_to_build, 
            properties
        )
    
    def create_sensor_process_relationship(self, sensor_node_id: int, process_node_id: int, 
                                        properties: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a relationship between sensor and process.
        
        Args:
            sensor_node_id: Sensor node ID
            process_node_id: Process node ID
            properties: Relationship properties
            
        Returns:
            int: Relationship ID
        """
        return self.create_relationship(
            sensor_node_id, 
            process_node_id, 
            self.config.sensor_monitors_process, 
            properties
        )
    
    def find_process_by_id(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a process node by process ID.
        
        Args:
            process_id: Process identifier
            
        Returns:
            Optional[Dict[str, Any]]: Process node data or None
        """
        nodes = self.find_nodes_by_label(
            self.config.process_node_label, 
            {"process_id": process_id}
        )
        return nodes[0] if nodes else None
    
    def find_machine_by_id(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a machine node by machine ID.
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            Optional[Dict[str, Any]]: Machine node data or None
        """
        nodes = self.find_nodes_by_label(
            self.config.machine_node_label, 
            {"machine_id": machine_id}
        )
        return nodes[0] if nodes else None
    
    def find_part_by_id(self, part_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a part node by part ID.
        
        Args:
            part_id: Part identifier
            
        Returns:
            Optional[Dict[str, Any]]: Part node data or None
        """
        nodes = self.find_nodes_by_label(
            self.config.part_node_label, 
            {"part_id": part_id}
        )
        return nodes[0] if nodes else None
    
    def find_sensor_by_id(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a sensor node by sensor ID.
        
        Args:
            sensor_id: Sensor identifier
            
        Returns:
            Optional[Dict[str, Any]]: Sensor node data or None
        """
        nodes = self.find_nodes_by_label(
            self.config.sensor_node_label, 
            {"sensor_id": sensor_id}
        )
        return nodes[0] if nodes else None
    
    def find_build_by_id(self, build_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a build node by build ID.
        
        Args:
            build_id: Build identifier
            
        Returns:
            Optional[Dict[str, Any]]: Build node data or None
        """
        nodes = self.find_nodes_by_label(
            self.config.build_node_label, 
            {"build_id": build_id}
        )
        return nodes[0] if nodes else None
    
    def get_process_relationships(self, process_id: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for a process.
        
        Args:
            process_id: Process identifier
            
        Returns:
            List[Dict[str, Any]]: List of relationships
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (p:Process {process_id: $process_id})-[r]-(n)
                    RETURN p, r, n, type(r) as relationship_type
                """, process_id=process_id)
                
                relationships = []
                for record in result:
                    relationships.append({
                        'process': dict(record['p']),
                        'relationship': dict(record['r']),
                        'related_node': dict(record['n']),
                        'relationship_type': record['relationship_type']
                    })
                
                return relationships
                
        except Exception as e:
            logger.error(f"Failed to get process relationships: {e}")
            raise
    
    def get_machine_sensors(self, machine_id: str) -> List[Dict[str, Any]]:
        """
        Get all sensors for a machine.
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            List[Dict[str, Any]]: List of sensors
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (m:Machine {machine_id: $machine_id})-[:HAS_SENSOR]->(s:Sensor)
                    RETURN s
                """, machine_id=machine_id)
                
                sensors = []
                for record in result:
                    sensors.append(dict(record['s']))
                
                return sensors
                
        except Exception as e:
            logger.error(f"Failed to get machine sensors: {e}")
            raise
    
    def get_process_parts(self, process_id: str) -> List[Dict[str, Any]]:
        """
        Get all parts created by a process.
        
        Args:
            process_id: Process identifier
            
        Returns:
            List[Dict[str, Any]]: List of parts
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (p:Process {process_id: $process_id})-[:CREATES_PART]->(part:Part)
                    RETURN part
                """, process_id=process_id)
                
                parts = []
                for record in result:
                    parts.append(dict(record['part']))
                
                return parts
                
        except Exception as e:
            logger.error(f"Failed to get process parts: {e}")
            raise
    
    def get_build_parts(self, build_id: str) -> List[Dict[str, Any]]:
        """
        Get all parts in a build.
        
        Args:
            build_id: Build identifier
            
        Returns:
            List[Dict[str, Any]]: List of parts
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (b:Build {build_id: $build_id})<-[:BELONGS_TO_BUILD]-(part:Part)
                    RETURN part
                """, build_id=build_id)
                
                parts = []
                for record in result:
                    parts.append(dict(record['part']))
                
                return parts
                
        except Exception as e:
            logger.error(f"Failed to get build parts: {e}")
            raise
    
    def get_process_flow(self, process_id: str) -> Dict[str, Any]:
        """
        Get the complete process flow for a process.
        
        Args:
            process_id: Process identifier
            
        Returns:
            Dict[str, Any]: Process flow data
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                # Get process details
                process_result = session.run("""
                    MATCH (p:Process {process_id: $process_id})
                    RETURN p
                """, process_id=process_id)
                
                process_data = None
                for record in process_result:
                    process_data = dict(record['p'])
                    break
                
                if not process_data:
                    return {}
                
                # Get related machines
                machines = self.get_process_relationships(process_id)
                machine_data = [rel['related_node'] for rel in machines 
                              if rel['relationship_type'] == self.config.process_uses_machine]
                
                # Get created parts
                parts = self.get_process_parts(process_id)
                
                return {
                    'process': process_data,
                    'machines': machine_data,
                    'parts': parts,
                    'relationships': machines
                }
                
        except Exception as e:
            logger.error(f"Failed to get process flow: {e}")
            raise
    
    def create_knowledge_graph_indexes(self) -> bool:
        """
        Create indexes for knowledge graph optimization.
        
        Returns:
            bool: True if successful
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                # Create indexes for node identifiers
                indexes = [
                    f"CREATE INDEX process_id_index IF NOT EXISTS FOR (n:{self.config.process_node_label}) ON (n.process_id)",
                    f"CREATE INDEX machine_id_index IF NOT EXISTS FOR (n:{self.config.machine_node_label}) ON (n.machine_id)",
                    f"CREATE INDEX part_id_index IF NOT EXISTS FOR (n:{self.config.part_node_label}) ON (n.part_id)",
                    f"CREATE INDEX sensor_id_index IF NOT EXISTS FOR (n:{self.config.sensor_node_label}) ON (n.sensor_id)",
                    f"CREATE INDEX build_id_index IF NOT EXISTS FOR (n:{self.config.build_node_label}) ON (n.build_id)"
                ]
                
                for index_query in indexes:
                    session.run(index_query)
                
                logger.info("Knowledge graph indexes created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create knowledge graph indexes: {e}")
            return False
    
    def clear_knowledge_graph(self) -> bool:
        """
        Clear all knowledge graph data.
        
        Returns:
            bool: True if successful
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")
                
                logger.info("Knowledge graph cleared successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear knowledge graph: {e}")
            return False
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes from Neo4j."""
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                result = session.run("MATCH (n) RETURN n")
                return [record["n"] for record in result]
        except Exception as e:
            logger.error(f"Failed to get all nodes: {e}")
            return []
    
    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships from Neo4j."""
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self._driver.session(database=self.database) as session:
                result = session.run("MATCH (a)-[r]->(b) RETURN a, r, b")
                return [{"from": record["a"], "relationship": record["r"], "to": record["b"]} for record in result]
        except Exception as e:
            logger.error(f"Failed to get all relationships: {e}")
            return []
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type."""
        try:
            return self.find_nodes_by_label(node_type)
        except Exception as e:
            logger.error(f"Failed to get nodes by type {node_type}: {e}")
            return []
    
    def load_data(
        self,
        df: Any,
        graph_type: str,
        mode: str = "append",
        batch_size: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load Spark DataFrame data into Neo4j graph.
        
        This method provides ETL pipeline integration for loading transformed
        Spark DataFrames into Neo4j, following the ETL architecture.
        
        Args:
            df: Spark DataFrame from transform modules
            graph_type: Type of graph data ("nodes", "relationships", "mixed")
            mode: Write mode (append, overwrite, ignore, error)
            batch_size: Batch size for processing
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Loading results and statistics
        """
        try:
            logger.info(f"Loading Spark DataFrame into Neo4j graph: {graph_type}")
            
            # Initialize result tracking
            result = {
                "success": False,
                "nodes_loaded": 0,
                "relationships_loaded": 0,
                "records_processed": 0,
                "errors": [],
                "warnings": [],
                "graph_type": graph_type,
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
                # Clear existing graph data
                self.clear_knowledge_graph()
                logger.info("Cleared existing graph data")
            
            # Process data based on graph type
            if graph_type == "nodes":
                total_nodes = self._process_nodes_batch(data_list, batch_size)
                result["nodes_loaded"] = total_nodes
            elif graph_type == "relationships":
                total_relationships = self._process_relationships_batch(data_list, batch_size)
                result["relationships_loaded"] = total_relationships
            elif graph_type == "mixed":
                nodes_result = self._process_nodes_batch(data_list, batch_size)
                relationships_result = self._process_relationships_batch(data_list, batch_size)
                result["nodes_loaded"] = nodes_result
                result["relationships_loaded"] = relationships_result
            else:
                result["errors"].append(f"Unsupported graph type: {graph_type}")
                return result
            
            # Update result
            total_loaded = result["nodes_loaded"] + result["relationships_loaded"]
            result["success"] = total_loaded > 0 and len(result["errors"]) == 0
            
            if result["success"]:
                logger.info(f"Successfully loaded {result['nodes_loaded']} nodes and {result['relationships_loaded']} relationships")
            else:
                logger.error(f"Failed to load data into Neo4j. Errors: {result['errors']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in load_data for graph type {graph_type}: {str(e)}")
            return {
                "success": False,
                "nodes_loaded": 0,
                "relationships_loaded": 0,
                "records_processed": 0,
                "errors": [str(e)],
                "warnings": [],
                "graph_type": graph_type,
                "mode": mode,
                "batch_size": batch_size
            }
    
    def _convert_spark_dataframe(self, df: Any) -> List[Dict[str, Any]]:
        """
        Convert Spark DataFrame to list of dictionaries for Neo4j insertion.
        
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
        Process Spark Row data to Neo4j-compatible format.
        
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
    
    def _process_nodes_batch(self, data_list: List[Dict[str, Any]], batch_size: int) -> int:
        """
        Process a batch of node data for Neo4j insertion.
        
        Args:
            data_list: List of data dictionaries
            batch_size: Batch size for processing
            
        Returns:
            int: Number of nodes loaded
        """
        try:
            total_loaded = 0
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                try:
                    # Process batch of nodes
                    batch_result = self._create_nodes_batch(batch)
                    total_loaded += batch_result
                    
                    logger.info(f"Processed node batch {i//batch_size + 1}: {batch_result} nodes")
                    
                except Exception as e:
                    logger.error(f"Error processing node batch {i//batch_size + 1}: {str(e)}")
                    continue
            
            return total_loaded
            
        except Exception as e:
            logger.error(f"Error processing nodes batch: {str(e)}")
            return 0
    
    def _process_relationships_batch(self, data_list: List[Dict[str, Any]], batch_size: int) -> int:
        """
        Process a batch of relationship data for Neo4j insertion.
        
        Args:
            data_list: List of data dictionaries
            batch_size: Batch size for processing
            
        Returns:
            int: Number of relationships loaded
        """
        try:
            total_loaded = 0
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                try:
                    # Process batch of relationships
                    batch_result = self._create_relationships_batch(batch)
                    total_loaded += batch_result
                    
                    logger.info(f"Processed relationship batch {i//batch_size + 1}: {batch_result} relationships")
                    
                except Exception as e:
                    logger.error(f"Error processing relationship batch {i//batch_size + 1}: {str(e)}")
                    continue
            
            return total_loaded
            
        except Exception as e:
            logger.error(f"Error processing relationships batch: {str(e)}")
            return 0
    
    def _create_nodes_batch(self, batch: List[Dict[str, Any]]) -> int:
        """
        Create a batch of nodes in Neo4j.
        
        Args:
            batch: List of node data dictionaries
            
        Returns:
            int: Number of nodes created
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            loaded = 0
            
            with self._driver.session(database=self.database) as session:
                for data in batch:
                    try:
                        # Extract node information
                        node_id = data.get('id') or data.get('node_id')
                        node_type = data.get('type') or data.get('node_type') or 'Node'
                        properties = {k: v for k, v in data.items() if k not in ['id', 'node_id', 'type', 'node_type']}
                        
                        if not node_id:
                            logger.warning("Skipping node without ID")
                            continue
                        
                        # Create node
                        query = f"""
                        MERGE (n:{node_type} {{id: $id}})
                        SET n += $properties
                        RETURN n
                        """
                        
                        session.run(query, id=node_id, properties=properties)
                        loaded += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to create node: {str(e)}")
                        continue
            
            return loaded
            
        except Exception as e:
            logger.error(f"Error creating nodes batch: {str(e)}")
            return 0
    
    def _create_relationships_batch(self, batch: List[Dict[str, Any]]) -> int:
        """
        Create a batch of relationships in Neo4j.
        
        Args:
            batch: List of relationship data dictionaries
            
        Returns:
            int: Number of relationships created
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            loaded = 0
            
            with self._driver.session(database=self.database) as session:
                for data in batch:
                    try:
                        # Extract relationship information
                        from_id = data.get('from_id') or data.get('source_id')
                        to_id = data.get('to_id') or data.get('target_id')
                        rel_type = data.get('relationship_type') or data.get('rel_type') or 'RELATES_TO'
                        properties = {k: v for k, v in data.items() if k not in ['from_id', 'source_id', 'to_id', 'target_id', 'relationship_type', 'rel_type']}
                        
                        if not from_id or not to_id:
                            logger.warning("Skipping relationship without source or target ID")
                            continue
                        
                        # Create relationship
                        query = f"""
                        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += $properties
                        RETURN r
                        """
                        
                        session.run(query, from_id=from_id, to_id=to_id, properties=properties)
                        loaded += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to create relationship: {str(e)}")
                        continue
            
            return loaded
            
        except Exception as e:
            logger.error(f"Error creating relationships batch: {str(e)}")
            return 0