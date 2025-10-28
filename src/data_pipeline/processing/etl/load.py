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


def load_to_clickhouse(
    df: DataFrame,
    clickhouse_config: Dict[str, str],
    table_name: str,
    database: str = "default",
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to ClickHouse using Spark JDBC
    
    Args:
        df: DataFrame to load
        clickhouse_config: ClickHouse configuration
        table_name: Target table name
        database: Target database name
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional ClickHouse options
    """
    try:
        logger.info(f"Loading data to ClickHouse table: {database}.{table_name}")
        
        # Build ClickHouse connection string
        host = clickhouse_config.get("host", "localhost")
        port = clickhouse_config.get("port", "8123")
        username = clickhouse_config.get("username", "default")
        password = clickhouse_config.get("password", "")
        
        connection_string = f"jdbc:clickhouse://{host}:{port}/{database}"
        
        # Default options for ClickHouse JDBC writing
        default_options = {
            "driver": "ru.yandex.clickhouse.ClickHouseDriver",
            "batchsize": "1000",
            "isolationLevel": "READ_COMMITTED",
            "user": username,
            "password": password
        }
        
        if options:
            default_options.update(options)
        
        # Add metadata columns
        df_with_metadata = df.withColumn(
            "load_timestamp", current_timestamp()
        ).withColumn(
            "load_mode", lit(mode)
        )
        
        # Write to ClickHouse
        df_with_metadata.write.format("jdbc").options(
            url=connection_string,
            dbtable=table_name,
            **default_options
        ).mode(mode).save()
        
        logger.info(f"Successfully loaded {df.count()} records to ClickHouse table: {database}.{table_name}")
        
    except Exception as e:
        logger.error(f"Error loading data to ClickHouse table {database}.{table_name}: {str(e)}")
        raise


def load_to_minio(
    df: DataFrame,
    minio_config: Dict[str, str],
    bucket_name: str,
    object_path: str,
    format: str = "parquet",
    mode: str = "append",
    options: Optional[Dict[str, str]] = None
) -> None:
    """
    Load data to MinIO (S3-compatible) using Spark
    
    Args:
        df: DataFrame to load
        minio_config: MinIO configuration
        bucket_name: MinIO bucket name
        object_path: Object path within bucket
        format: File format (parquet, json, csv, delta)
        mode: Write mode (append, overwrite, ignore, error)
        options: Optional MinIO options
    """
    try:
        logger.info(f"Loading data to MinIO: s3a://{bucket_name}/{object_path}")
        
        # Build S3 path for MinIO
        s3_path = f"s3a://{bucket_name}/{object_path}"
        
        # Default options for MinIO writing
        default_options = {
            "fs.s3a.endpoint": minio_config.get("endpoint", "http://localhost:9000"),
            "fs.s3a.access.key": minio_config.get("access_key", "minioadmin"),
            "fs.s3a.secret.key": minio_config.get("secret_key", "minioadmin"),
            "fs.s3a.path.style.access": "true",
            "fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
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
        
        # Write to MinIO
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
        
        logger.info(f"Successfully loaded {df.count()} records to MinIO: s3a://{bucket_name}/{object_path}")
        
    except Exception as e:
        logger.error(f"Error loading data to MinIO s3a://{bucket_name}/{object_path}: {str(e)}")
        raise






# =============================================================================
# Generic NoSQL Destination Loading Function
# =============================================================================


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
