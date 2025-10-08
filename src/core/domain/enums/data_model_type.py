"""
Data model type enumeration for multi-model data operations.
"""

from enum import Enum


class DataModelType(Enum):
    """
    Enumeration for different data model types supported in the system.
    
    This enum defines the various data storage and processing models
    that can be used for PBF-LB/M data pipeline operations.
    """
    
    # Relational/SQL models
    RELATIONAL = "relational"
    SQL = "sql"
    
    # Document models
    DOCUMENT = "document"
    MONGODB = "mongodb"
    
    # Key-Value models
    KEY_VALUE = "key_value"
    REDIS = "redis"
    
    # Columnar models
    COLUMNAR = "columnar"
    CASSANDRA = "cassandra"
    
    # Graph models
    GRAPH = "graph"
    NEO4J = "neo4j"
    
    # Search models
    SEARCH = "search"
    ELASTICSEARCH = "elasticsearch"
    
    # Time-series models
    TIME_SERIES = "time_series"
    
    # Hybrid models
    HYBRID = "hybrid"
    MULTI_MODEL = "multi_model"
    
    @classmethod
    def get_sql_models(cls):
        """Get SQL-based model types."""
        return [cls.RELATIONAL, cls.SQL]
    
    @classmethod
    def get_nosql_models(cls):
        """Get NoSQL model types."""
        return [
            cls.DOCUMENT, cls.MONGODB,
            cls.KEY_VALUE, cls.REDIS,
            cls.COLUMNAR, cls.CASSANDRA,
            cls.GRAPH, cls.NEO4J,
            cls.SEARCH, cls.ELASTICSEARCH,
            cls.TIME_SERIES
        ]
    
    @classmethod
    def get_hybrid_models(cls):
        """Get hybrid model types."""
        return [cls.HYBRID, cls.MULTI_MODEL]
    
    def is_sql_model(self):
        """Check if this is a SQL-based model."""
        return self in self.get_sql_models()
    
    def is_nosql_model(self):
        """Check if this is a NoSQL model."""
        return self in self.get_nosql_models()
    
    def is_hybrid_model(self):
        """Check if this is a hybrid model."""
        return self in self.get_hybrid_models()
    
    def get_compatible_models(self):
        """Get models that are compatible with this model type."""
        compatibility_map = {
            cls.RELATIONAL: [cls.SQL, cls.HYBRID],
            cls.SQL: [cls.RELATIONAL, cls.HYBRID],
            cls.DOCUMENT: [cls.MONGODB, cls.HYBRID],
            cls.MONGODB: [cls.DOCUMENT, cls.HYBRID],
            cls.KEY_VALUE: [cls.REDIS, cls.HYBRID],
            cls.REDIS: [cls.KEY_VALUE, cls.HYBRID],
            cls.COLUMNAR: [cls.CASSANDRA, cls.TIME_SERIES, cls.HYBRID],
            cls.CASSANDRA: [cls.COLUMNAR, cls.TIME_SERIES, cls.HYBRID],
            cls.GRAPH: [cls.NEO4J, cls.HYBRID],
            cls.NEO4J: [cls.GRAPH, cls.HYBRID],
            cls.SEARCH: [cls.ELASTICSEARCH, cls.HYBRID],
            cls.ELASTICSEARCH: [cls.SEARCH, cls.HYBRID],
            cls.TIME_SERIES: [cls.COLUMNAR, cls.CASSANDRA, cls.HYBRID],
            cls.HYBRID: list(cls),
            cls.MULTI_MODEL: list(cls),
        }
        return compatibility_map.get(self, [])
