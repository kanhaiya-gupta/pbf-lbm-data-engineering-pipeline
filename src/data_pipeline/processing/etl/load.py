"""
Data Loading Module

This module provides Spark-based data loading functions for PBF-LB/M data processing.
It supports loading to various destinations including PostgreSQL, S3, Snowflake, and Delta Lake.
"""

from typing import Dict, Any, Optional, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import current_timestamp, lit
import logging

logger = logging.getLogger(__name__)


def load_to_postgresql(
    df: DataFrame,
    connection_string: str,
    table_name: str,
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to PostgreSQL using Spark JDBC
    
    Args:
        df: DataFrame to load
        connection_string: PostgreSQL connection string
        table_name: Target table name
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional JDBC options
    """
    try:
        logger.info(f"Loading data to PostgreSQL table: {table_name}")
        
        # Default options for JDBC writing
        default_options = {
            "driver": "org.postgresql.Driver",
            "batchsize": "1000",
            "isolationLevel": "READ_COMMITTED"
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to PostgreSQL
        df_with_metadata.write.format("jdbc").options(
            url=connection_string,
            dbtable=table_name,
            **default_options
        ).mode(mode).save()
        
        logger.info(f"Successfully loaded {df.count()} records to PostgreSQL table: {table_name}")
        
    except Exception as e:
        logger.error(f"Error loading data to PostgreSQL table {table_name}: {str(e)}")
        raise


def load_to_s3(
    df: DataFrame,
    s3_path: str,
    format: str = "parquet",
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to S3 using Spark
    
    Args:
        df: DataFrame to load
        s3_path: S3 path for the data
        format: File format (parquet, json, csv, delta)
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional write options
    """
    try:
        logger.info(f"Loading data to S3: {s3_path}")
        
        # Default options for S3 writing
        default_options = {
            "compression": "snappy",
            "parquet.enable.summary-metadata": "false"
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        ).withColumn(
            "file_format", lit(format)
        )
        
        # Write to S3
        writer = df_with_metadata.write.options(**default_options).mode(mode)
        
        if format == "parquet":
            writer.parquet(s3_path)
        elif format == "json":
            writer.json(s3_path)
        elif format == "csv":
            writer.option("header", "true").csv(s3_path)
        elif format == "delta":
            writer.format("delta").save(s3_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Successfully loaded {df.count()} records to S3: {s3_path}")
        
    except Exception as e:
        logger.error(f"Error loading data to S3 {s3_path}: {str(e)}")
        raise


def load_to_snowflake(
    df: DataFrame,
    snowflake_config: Dict[str, str],
    table_name: str,
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to Snowflake using Spark
    
    Args:
        df: DataFrame to load
        snowflake_config: Snowflake configuration
        table_name: Target table name
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional Snowflake options
    """
    try:
        logger.info(f"Loading data to Snowflake table: {table_name}")
        
        # Default options for Snowflake writing
        default_options = {
            "sfURL": snowflake_config.get("account", ""),
            "sfUser": snowflake_config.get("user", ""),
            "sfPassword": snowflake_config.get("password", ""),
            "sfDatabase": snowflake_config.get("database", ""),
            "sfSchema": snowflake_config.get("schema", ""),
            "sfWarehouse": snowflake_config.get("warehouse", ""),
            "sfRole": snowflake_config.get("role", ""),
            "batchsize": "1000"
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to Snowflake
        df_with_metadata.write.format("snowflake").options(
            **default_options
        ).option("dbtable", table_name).mode(mode).save()
        
        logger.info(f"Successfully loaded {df.count()} records to Snowflake table: {table_name}")
        
    except Exception as e:
        logger.error(f"Error loading data to Snowflake table {table_name}: {str(e)}")
        raise


def load_to_delta_lake(
    df: DataFrame,
    delta_path: str,
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to Delta Lake using Spark
    
    Args:
        df: DataFrame to load
        delta_path: Delta Lake path
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional Delta Lake options
    """
    try:
        logger.info(f"Loading data to Delta Lake: {delta_path}")
        
        # Default options for Delta Lake writing
        default_options = {
            "compression": "snappy",
            "autoOptimize": "true",
            "optimizeWrite": "true"
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to Delta Lake
        df_with_metadata.write.format("delta").options(
            **default_options
        ).mode(mode).save(delta_path)
        
        logger.info(f"Successfully loaded {df.count()} records to Delta Lake: {delta_path}")
        
    except Exception as e:
        logger.error(f"Error loading data to Delta Lake {delta_path}: {str(e)}")
        raise


def load_to_mongodb(
    df: DataFrame,
    mongodb_config: Dict[str, str],
    collection_name: str,
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to MongoDB using Spark
    
    Args:
        df: DataFrame to load
        mongodb_config: MongoDB configuration
        collection_name: Target collection name
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional MongoDB options
    """
    try:
        logger.info(f"Loading data to MongoDB collection: {collection_name}")
        
        # Default options for MongoDB writing
        default_options = {
            "uri": mongodb_config.get("uri", ""),
            "database": mongodb_config.get("database", ""),
            "collection": collection_name,
            "batchsize": "1000"
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to MongoDB
        df_with_metadata.write.format("mongo").options(
            **default_options
        ).mode(mode).save()
        
        logger.info(f"Successfully loaded {df.count()} records to MongoDB collection: {collection_name}")
        
    except Exception as e:
        logger.error(f"Error loading data to MongoDB collection {collection_name}: {str(e)}")
        raise


def load_to_redis(
    df: DataFrame,
    redis_config: Dict[str, str],
    key_prefix: str,
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to Redis using Spark
    
    Args:
        df: DataFrame to load
        redis_config: Redis configuration
        key_prefix: Key prefix for Redis keys
        options: Optional Redis options
    """
    try:
        logger.info(f"Loading data to Redis with key prefix: {key_prefix}")
        
        # Default options for Redis writing
        default_options = {
            "host": redis_config.get("host", "localhost"),
            "port": redis_config.get("port", "6379"),
            "password": redis_config.get("password", ""),
            "database": redis_config.get("database", "0"),
            "key.prefix": key_prefix,
            "ttl": redis_config.get("ttl", "3600")
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "key_prefix", lit(key_prefix)
        )
        
        # Write to Redis
        df_with_metadata.write.format("redis").options(
            **default_options
        ).save()
        
        logger.info(f"Successfully loaded {df.count()} records to Redis with key prefix: {key_prefix}")
        
    except Exception as e:
        logger.error(f"Error loading data to Redis with key prefix {key_prefix}: {str(e)}")
        raise


def load_to_cassandra(
    df: DataFrame,
    cassandra_config: Dict[str, str],
    keyspace: str,
    table_name: str,
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to Cassandra using Spark
    
    Args:
        df: DataFrame to load
        cassandra_config: Cassandra configuration
        keyspace: Target keyspace
        table_name: Target table name
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional Cassandra options
    """
    try:
        logger.info(f"Loading data to Cassandra table: {keyspace}.{table_name}")
        
        # Default options for Cassandra writing
        default_options = {
            "spark.cassandra.connection.host": cassandra_config.get("host", "localhost"),
            "spark.cassandra.connection.port": cassandra_config.get("port", "9042"),
            "spark.cassandra.auth.username": cassandra_config.get("username", ""),
            "spark.cassandra.auth.password": cassandra_config.get("password", ""),
            "keyspace": keyspace,
            "table": table_name,
            "batch.size.rows": "1000"
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to Cassandra
        df_with_metadata.write.format("org.apache.spark.sql.cassandra").options(
            **default_options
        ).mode(mode).save()
        
        logger.info(f"Successfully loaded {df.count()} records to Cassandra table: {keyspace}.{table_name}")
        
    except Exception as e:
        logger.error(f"Error loading data to Cassandra table {keyspace}.{table_name}: {str(e)}")
        raise


def load_to_elasticsearch(
    df: DataFrame,
    elasticsearch_config: Dict[str, str],
    index_name: str,
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to Elasticsearch using Spark
    
    Args:
        df: DataFrame to load
        elasticsearch_config: Elasticsearch configuration
        index_name: Target index name
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional Elasticsearch options
    """
    try:
        logger.info(f"Loading data to Elasticsearch index: {index_name}")
        
        # Default options for Elasticsearch writing
        default_options = {
            "es.nodes": elasticsearch_config.get("nodes", "localhost"),
            "es.port": elasticsearch_config.get("port", "9200"),
            "es.net.http.auth.user": elasticsearch_config.get("username", ""),
            "es.net.http.auth.pass": elasticsearch_config.get("password", ""),
            "es.resource": index_name,
            "es.batch.size.entries": "1000"
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to Elasticsearch
        df_with_metadata.write.format("es").options(
            **default_options
        ).mode(mode).save()
        
        logger.info(f"Successfully loaded {df.count()} records to Elasticsearch index: {index_name}")
        
    except Exception as e:
        logger.error(f"Error loading data to Elasticsearch index {index_name}: {str(e)}")
        raise


def load_pbf_process_data(
    df: DataFrame,
    destination_config: Dict[str, Any]
) -> None:
    """
    Load PBF process data to specified destination
    
    Args:
        df: DataFrame containing PBF process data
        destination_config: Destination configuration
    """
    try:
        logger.info("Loading PBF process data")
        
        destination_type = destination_config.get("type", "postgresql")
        
        if destination_type == "postgresql":
            load_to_postgresql(
                df=df,
                connection_string=destination_config.get("connection_string", ""),
                table_name=destination_config.get("table_name", "pbf_process_data"),
                mode=destination_config.get("mode", "append")
            )
        elif destination_type == "s3":
            load_to_s3(
                df=df,
                s3_path=destination_config.get("s3_path", ""),
                format=destination_config.get("format", "parquet"),
                mode=destination_config.get("mode", "append")
            )
        elif destination_type == "snowflake":
            load_to_snowflake(
                df=df,
                snowflake_config=destination_config.get("snowflake_config", {}),
                table_name=destination_config.get("table_name", "pbf_process_data"),
                mode=destination_config.get("mode", "append")
            )
        elif destination_type == "delta_lake":
            load_to_delta_lake(
                df=df,
                delta_path=destination_config.get("delta_path", ""),
                mode=destination_config.get("mode", "append")
            )
        elif destination_type == "mongodb":
            load_to_mongodb(
                df=df,
                mongodb_config=destination_config.get("mongodb_config", {}),
                collection_name=destination_config.get("collection_name", "pbf_process_data"),
                mode=destination_config.get("mode", "append")
            )
        else:
            raise ValueError(f"Unsupported destination type: {destination_type}")
            
    except Exception as e:
        logger.error(f"Error loading PBF process data: {str(e)}")
        raise


def load_ispm_monitoring_data(
    df: DataFrame,
    destination_config: Dict[str, Any]
) -> None:
    """
    Load ISPM monitoring data to specified destination
    
    Args:
        df: DataFrame containing ISPM monitoring data
        destination_config: Destination configuration
    """
    try:
        logger.info("Loading ISPM monitoring data")
        
        destination_type = destination_config.get("type", "cassandra")
        
        if destination_type == "cassandra":
            load_to_cassandra(
                df=df,
                cassandra_config=destination_config.get("cassandra_config", {}),
                keyspace=destination_config.get("keyspace", "lpbf_research"),
                table_name=destination_config.get("table_name", "ispm_monitoring_data"),
                mode=destination_config.get("mode", "append")
            )
        elif destination_type == "redis":
            load_to_redis(
                df=df,
                redis_config=destination_config.get("redis_config", {}),
                key_prefix=destination_config.get("key_prefix", "ispm_monitoring"),
                options=destination_config.get("options", {})
            )
        elif destination_type == "postgresql":
            load_to_postgresql(
                df=df,
                connection_string=destination_config.get("connection_string", ""),
                table_name=destination_config.get("table_name", "ispm_monitoring_data"),
                mode=destination_config.get("mode", "append")
            )
        else:
            raise ValueError(f"Unsupported destination type: {destination_type}")
            
    except Exception as e:
        logger.error(f"Error loading ISPM monitoring data: {str(e)}")
        raise


# =============================================================================
# NoSQL Data Loading Functions
# =============================================================================

def load_to_mongodb(
    df: DataFrame,
    connection_string: str,
    database_name: str,
    collection_name: str,
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to MongoDB using Spark MongoDB connector
    
    Args:
        df: DataFrame to load
        connection_string: MongoDB connection string
        database_name: MongoDB database name
        collection_name: MongoDB collection name
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional MongoDB connector options
    """
    try:
        logger.info(f"Loading data to MongoDB collection: {collection_name}")
        
        # Build MongoDB URI
        mongo_uri = f"{connection_string}/{database_name}.{collection_name}"
        
        # Default options for MongoDB writing
        default_options = {
            "uri": mongo_uri,
            "partitioner": "MongoSamplePartitioner",
            "partitionerOptions.partitionSizeMB": "64"
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to MongoDB
        df_with_metadata.write \
            .format("mongo") \
            .options(**default_options) \
            .mode(mode) \
            .save()
        
        logger.info(f"Successfully loaded {df.count()} records to MongoDB")
        
    except Exception as e:
        logger.error(f"Error loading data to MongoDB: {str(e)}")
        raise


def load_to_redis(
    df: DataFrame,
    host: str,
    port: int = 6379,
    password: Optional[str] = None,
    db: int = 0,
    key_column: str = "key",
    value_column: str = "value",
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to Redis using Spark Redis connector
    
    Args:
        df: DataFrame to load
        host: Redis host
        port: Redis port
        password: Optional Redis password
        db: Redis database number
        key_column: Column name containing keys
        value_column: Column name containing values
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional Redis connector options
    """
    try:
        logger.info(f"Loading data to Redis database: {db}")
        
        # Configure Redis connector options
        default_options = {
            "host": host,
            "port": str(port),
            "db": str(db),
            "keyColumn": key_column,
            "valueColumn": value_column
        }
        
        if password:
            default_options["auth"] = password
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to Redis
        df_with_metadata.write \
            .format("org.apache.spark.sql.redis") \
            .options(**default_options) \
            .mode(mode) \
            .save()
        
        logger.info(f"Successfully loaded {df.count()} records to Redis")
        
    except Exception as e:
        logger.error(f"Error loading data to Redis: {str(e)}")
        raise


def load_to_cassandra(
    df: DataFrame,
    hosts: List[str],
    keyspace: str,
    table_name: str,
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to Cassandra using Spark Cassandra connector
    
    Args:
        df: DataFrame to load
        hosts: List of Cassandra host addresses
        keyspace: Cassandra keyspace name
        table_name: Cassandra table name
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional Cassandra connector options
    """
    try:
        logger.info(f"Loading data to Cassandra table: {keyspace}.{table_name}")
        
        # Configure Cassandra connector options
        default_options = {
            "keyspace": keyspace,
            "table": table_name,
            "spark.cassandra.connection.host": ",".join(hosts),
            "spark.cassandra.connection.port": "9042",
            "spark.cassandra.output.batch.size.rows": "1000",
            "spark.cassandra.output.concurrent.writes": "10"
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to Cassandra
        df_with_metadata.write \
            .format("org.apache.spark.sql.cassandra") \
            .options(**default_options) \
            .mode(mode) \
            .save()
        
        logger.info(f"Successfully loaded {df.count()} records to Cassandra")
        
    except Exception as e:
        logger.error(f"Error loading data to Cassandra: {str(e)}")
        raise


def load_to_elasticsearch(
    df: DataFrame,
    hosts: List[str],
    index_name: str,
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to Elasticsearch using Spark Elasticsearch connector
    
    Args:
        df: DataFrame to load
        hosts: List of Elasticsearch host addresses
        index_name: Elasticsearch index name
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional Elasticsearch connector options
    """
    try:
        logger.info(f"Loading data to Elasticsearch index: {index_name}")
        
        # Configure Elasticsearch connector options
        default_options = {
            "es.nodes": ",".join(hosts),
            "es.port": "9200",
            "es.resource": index_name,
            "es.batch.size.entries": "1000",
            "es.batch.size.bytes": "1mb"
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to Elasticsearch
        df_with_metadata.write \
            .format("org.elasticsearch.spark.sql") \
            .options(**default_options) \
            .mode(mode) \
            .save()
        
        logger.info(f"Successfully loaded {df.count()} documents to Elasticsearch")
        
    except Exception as e:
        logger.error(f"Error loading data to Elasticsearch: {str(e)}")
        raise


def load_to_neo4j(
    df: DataFrame,
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
    node_label: Optional[str] = None,
    relationship_type: Optional[str] = None,
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to Neo4j using Spark Neo4j connector
    
    Args:
        df: DataFrame to load
        uri: Neo4j server URI
        username: Username for authentication
        password: Password for authentication
        database: Neo4j database name
        node_label: Optional node label for node creation
        relationship_type: Optional relationship type for relationship creation
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional Neo4j connector options
    """
    try:
        logger.info(f"Loading data to Neo4j database: {database}")
        
        # Configure Neo4j connector options
        default_options = {
            "url": uri,
            "authentication.type": "basic",
            "authentication.basic.username": username,
            "authentication.basic.password": password,
            "database": database
        }
        
        if node_label:
            default_options["labels"] = node_label
        
        if relationship_type:
            default_options["relationship"] = relationship_type
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to Neo4j
        df_with_metadata.write \
            .format("org.neo4j.spark.DataSource") \
            .options(**default_options) \
            .mode(mode) \
            .save()
        
        logger.info(f"Successfully loaded {df.count()} records to Neo4j")
        
    except Exception as e:
        logger.error(f"Error loading data to Neo4j: {str(e)}")
        raise


def load_to_nosql_destination(
    df: DataFrame,
    destination_type: str,
    connection_config: Dict[str, Any],
    load_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Generic function to load data to various NoSQL destinations
    
    Args:
        df: DataFrame to load
        destination_type: Type of NoSQL destination (mongodb, redis, cassandra, elasticsearch, neo4j)
        connection_config: Connection configuration for the destination
        load_config: Optional load-specific configuration
    """
    try:
        logger.info(f"Loading data to NoSQL destination: {destination_type}")
        
        if load_config is None:
            load_config = {}
        
        mode = load_config.get("mode", "append")
        options = load_config.get("options")
        
        if destination_type.lower() == "mongodb":
            load_to_mongodb(
                df=df,
                connection_string=connection_config["connection_string"],
                database_name=connection_config["database_name"],
                collection_name=connection_config["collection_name"],
                mode=mode,
                options=options
            )
        
        elif destination_type.lower() == "redis":
            load_to_redis(
                df=df,
                host=connection_config["host"],
                port=connection_config.get("port", 6379),
                password=connection_config.get("password"),
                db=connection_config.get("db", 0),
                key_column=load_config.get("key_column", "key"),
                value_column=load_config.get("value_column", "value"),
                mode=mode,
                options=options
            )
        
        elif destination_type.lower() == "cassandra":
            load_to_cassandra(
                df=df,
                hosts=connection_config["hosts"],
                keyspace=connection_config["keyspace"],
                table_name=connection_config["table_name"],
                mode=mode,
                options=options
            )
        
        elif destination_type.lower() == "elasticsearch":
            load_to_elasticsearch(
                df=df,
                hosts=connection_config["hosts"],
                index_name=connection_config["index_name"],
                mode=mode,
                options=options
            )
        
        elif destination_type.lower() == "neo4j":
            load_to_neo4j(
                df=df,
                uri=connection_config["uri"],
                username=connection_config["username"],
                password=connection_config["password"],
                database=connection_config.get("database", "neo4j"),
                node_label=load_config.get("node_label"),
                relationship_type=load_config.get("relationship_type"),
                mode=mode,
                options=options
            )
        
        else:
            raise ValueError(f"Unsupported NoSQL destination type: {destination_type}")
            
    except Exception as e:
        logger.error(f"Error loading data to NoSQL destination {destination_type}: {str(e)}")
        raise
