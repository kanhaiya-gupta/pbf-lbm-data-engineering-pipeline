"""
Data Extraction Module

This module provides Spark-based data extraction functions for PBF-LB/M data processing.
It supports extraction from various sources including CSV, JSON, databases, Kafka, 
specialized systems, and NoSQL databases (MongoDB, Redis, Cassandra, Elasticsearch).
"""

from typing import Dict, Any, Optional, List, Union
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from pyspark.sql.functions import col, from_json, explode, lit
import logging
import json

logger = logging.getLogger(__name__)


def extract_from_data_lake(
    spark: SparkSession,
    source_type: str,  # "minio", "s3", "postgresql", "mongodb", "kafka", "csv", "json", etc.
    source_config: Dict[str, Any],
    **kwargs
) -> DataFrame:
    """
    Universal data extraction method - handles ALL data sources.
    
    This is the main extraction method that handles all data sources directly.
    All extraction logic is consolidated in this single method.
    
    Args:
        spark: Spark session
        source_type: Type of data source (minio, s3, postgresql, mongodb, kafka, csv, json, etc.)
        source_config: Source configuration
        **kwargs: Additional parameters specific to source type
        
    Returns:
        DataFrame containing extracted data
    """
    try:
        logger.info(f"Extracting data from {source_type.upper()} source")
        
        # Object Storage (S3/MinIO)
        if source_type.lower() in ["minio", "s3"]:
            bucket_name = kwargs.get("bucket_name", "")
            object_path = kwargs.get("object_path", "")
            file_format = kwargs.get("file_format", "json")
            schema = kwargs.get("schema")
            
            if source_type.lower() == "s3":
                # AWS S3 Configuration
                spark.conf.set("spark.hadoop.fs.s3a.access.key", source_config["access_key"])
                spark.conf.set("spark.hadoop.fs.s3a.secret.key", source_config["secret_key"])
                spark.conf.set("spark.hadoop.fs.s3a.endpoint", source_config.get("endpoint", "s3.amazonaws.com"))
                spark.conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
                path = f"s3a://{bucket_name}/{object_path}"
            else:  # minio
                spark.conf.set("spark.hadoop.fs.s3a.access.key", source_config["access_key"])
                spark.conf.set("spark.hadoop.fs.s3a.secret.key", source_config["secret_key"])
                spark.conf.set("spark.hadoop.fs.s3a.endpoint", source_config["endpoint"])
                spark.conf.set("spark.hadoop.fs.s3a.path.style.access", "true")
                spark.conf.set("spark.hadoop.fs.s3a.connection.ssl.enabled", str(source_config.get("ssl_enabled", False)).lower())
                path = f"s3a://{bucket_name}/{object_path}"
            
            # Read data based on format
            if file_format.lower() == "json":
                if schema:
                    df = spark.read.schema(schema).json(path)
                else:
                    df = spark.read.json(path)
            elif file_format.lower() == "csv":
                df = spark.read.csv(path, header=True, inferSchema=True)
            elif file_format.lower() == "parquet":
                df = spark.read.parquet(path)
            elif file_format.lower() == "txt":
                df = spark.read.text(path)
            elif file_format.lower() == "avro":
                df = spark.read.format("avro").load(path)
            elif file_format.lower() == "orc":
                df = spark.read.orc(path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Successfully extracted {df.count()} records from {source_type.upper()} object storage")
            return df
        
        # File Systems (CSV, JSON, Parquet)
        elif source_type.lower() == "csv":
            file_path = kwargs.get("file_path", "")
            schema = kwargs.get("schema")
            options = kwargs.get("options", {})
            
            default_options = {
                "header": "true",
                "inferSchema": "true",
                "multiline": "true"
            }
            default_options.update(options)
            
            if schema:
                df = spark.read.schema(schema).options(**default_options).csv(file_path)
            else:
                df = spark.read.options(**default_options).csv(file_path)
            
            logger.info(f"Successfully extracted {df.count()} records from CSV: {file_path}")
            return df
            
        elif source_type.lower() == "json":
            file_path = kwargs.get("file_path", "")
            schema = kwargs.get("schema")
            options = kwargs.get("options", {})
            
            default_options = {
                "multiline": "true",
                "allowUnquotedFieldNames": "true"
            }
            default_options.update(options)
            
            if schema:
                df = spark.read.schema(schema).options(**default_options).json(file_path)
            else:
                df = spark.read.options(**default_options).json(file_path)
            
            logger.info(f"Successfully extracted {df.count()} records from JSON: {file_path}")
            return df
        
        elif source_type.lower() == "parquet":
            file_path = kwargs.get("file_path", "")
            schema = kwargs.get("schema")
            options = kwargs.get("options", {})
            
            if schema:
                df = spark.read.schema(schema).options(**options).parquet(file_path)
            else:
                df = spark.read.options(**options).parquet(file_path)
            
            logger.info(f"Successfully extracted {df.count()} records from Parquet: {file_path}")
            return df
        
        elif source_type.lower() in ["txt", "text"]:
            file_path = kwargs.get("file_path", "")
            options = kwargs.get("options", {})
            
            df = spark.read.options(**options).text(file_path)
            
            logger.info(f"Successfully extracted {df.count()} records from text file: {file_path}")
            return df
        
        elif source_type.lower() == "avro":
            file_path = kwargs.get("file_path", "")
            options = kwargs.get("options", {})
            
            df = spark.read.format("avro").options(**options).load(file_path)
            
            logger.info(f"Successfully extracted {df.count()} records from Avro: {file_path}")
            return df
        
        elif source_type.lower() == "orc":
            file_path = kwargs.get("file_path", "")
            options = kwargs.get("options", {})
            
            df = spark.read.options(**options).orc(file_path)
            
            logger.info(f"Successfully extracted {df.count()} records from ORC: {file_path}")
            return df
        
        # Relational Databases
        elif source_type.lower() in ["postgresql", "mysql", "oracle"]:
            connection_string = source_config.get("connection_string", "")
            table_name = kwargs.get("table_name", "")
            query = kwargs.get("query")
            options = kwargs.get("options", {})
            
            default_options = {
                "driver": "org.postgresql.Driver",
                "fetchsize": "1000",
                "batchsize": "1000"
            }
            default_options.update(options)
            
            if query:
                df = spark.read.format("jdbc").options(
                    url=connection_string,
                    query=query,
                    **default_options
                ).load()
            else:
                df = spark.read.format("jdbc").options(
                    url=connection_string,
                    dbtable=table_name,
                    **default_options
                ).load()
            
            logger.info(f"Successfully extracted {df.count()} records from database table: {table_name}")
            return df
        
        # NoSQL Databases - delegate to existing methods
        elif source_type.lower() == "mongodb":
            return extract_from_mongodb(spark, source_config, **kwargs)
        elif source_type.lower() == "cassandra":
            return extract_from_cassandra(spark, source_config, **kwargs)
        elif source_type.lower() == "redis":
            return extract_from_redis(spark, source_config, **kwargs)
        elif source_type.lower() == "elasticsearch":
            return extract_from_elasticsearch(spark, source_config, **kwargs)
        elif source_type.lower() == "kafka":
            return extract_from_kafka(spark, source_config, **kwargs)
        
        # Specialized Systems
        elif source_type.lower() == "ct_scanner":
            return extract_from_ct_scanner(spark, source_config, **kwargs)
        elif source_type.lower() == "ispm_system":
            return extract_from_ispm_system(spark, source_config, **kwargs)
        
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
            
    except Exception as e:
        logger.error(f"Error extracting data from {source_type}: {str(e)}")
        raise


def extract_from_csv(
    spark: SparkSession,
    file_path: str,
    schema: Optional[StructType] = None,
    options: Optional[Dict[str, str]] = None
) -> DataFrame:
    """
    Extract data from CSV files using Spark
    
    Args:
        spark: Spark session
        file_path: Path to CSV file
        schema: Optional schema for the data
        options: Optional options for CSV reading
        
    Returns:
        DataFrame containing the extracted data
    """
    try:
        logger.info(f"Extracting data from CSV: {file_path}")
        
        # Default options for CSV reading
        default_options = {
            "header": "true",
            "inferSchema": "true",
            "multiline": "true"
        }
        
        if options:
            default_options.update(options)
        
        # Read CSV file
        if schema:
            df = spark.read.schema(schema).options(**default_options).csv(file_path)
        else:
            df = spark.read.options(**default_options).csv(file_path)
        
        logger.info(f"Successfully extracted {df.count()} records from CSV: {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from CSV {file_path}: {str(e)}")
        raise


def extract_from_json(
    spark: SparkSession,
    file_path: str,
    schema: Optional[StructType] = None,
    options: Optional[Dict[str, str]] = None
) -> DataFrame:
    """
    Extract data from JSON files using Spark
    
    Args:
        spark: Spark session
        file_path: Path to JSON file
        schema: Optional schema for the data
        options: Optional options for JSON reading
        
    Returns:
        DataFrame containing the extracted data
    """
    try:
        logger.info(f"Extracting data from JSON: {file_path}")
        
        # Default options for JSON reading
        default_options = {
            "multiline": "true",
            "allowUnquotedFieldNames": "true"
        }
        
        if options:
            default_options.update(options)
        
        # Read JSON file
        if schema:
            df = spark.read.schema(schema).options(**default_options).json(file_path)
        else:
            df = spark.read.options(**default_options).json(file_path)
        
        logger.info(f"Successfully extracted {df.count()} records from JSON: {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from JSON {file_path}: {str(e)}")
        raise


def extract_from_database(
    spark: SparkSession,
    connection_string: str,
    table_name: str,
    query: Optional[str] = None,
    options: Optional[Dict[str, str]] = None
) -> DataFrame:
    """
    Extract data from database using Spark JDBC
    
    Args:
        spark: Spark session
        connection_string: Database connection string
        table_name: Name of the table to extract from
        query: Optional custom SQL query
        options: Optional options for JDBC reading
        
    Returns:
        DataFrame containing the extracted data
    """
    try:
        logger.info(f"Extracting data from database table: {table_name}")
        
        # Default options for JDBC reading
        default_options = {
            "driver": "org.postgresql.Driver",
            "fetchsize": "1000",
            "batchsize": "1000"
        }
        
        if options:
            default_options.update(options)
        
        # Read from database
        if query:
            df = spark.read.format("jdbc").options(
                url=connection_string,
                query=query,
                **default_options
            ).load()
        else:
            df = spark.read.format("jdbc").options(
                url=connection_string,
                dbtable=table_name,
                **default_options
            ).load()
        
        logger.info(f"Successfully extracted {df.count()} records from database table: {table_name}")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from database table {table_name}: {str(e)}")
        raise


def extract_from_kafka(
    spark: SparkSession,
    kafka_config: Dict[str, Any],
    **kwargs
) -> DataFrame:
    """
    Extract data from Kafka using Spark Structured Streaming
    
    Args:
        spark: Spark session
        kafka_config: Kafka configuration dictionary
        **kwargs: Additional parameters (bootstrap_servers, topic, etc.)
        
    Returns:
        DataFrame containing the extracted data
    """
    try:
        bootstrap_servers = kwargs.get("bootstrap_servers", kafka_config.get("bootstrap_servers", "localhost:9092"))
        topic = kwargs.get("topic", kafka_config.get("topic", "pbf_process_data"))
        starting_offsets = kwargs.get("starting_offsets", kafka_config.get("starting_offsets", "earliest"))
        options = kwargs.get("options", {})
        
        logger.info(f"Extracting data from Kafka topic: {topic}")
        
        # Default options for Kafka reading
        default_options = {
            "kafka.bootstrap.servers": bootstrap_servers,
            "subscribe": topic,
            "startingOffsets": starting_offsets,
            "failOnDataLoss": "false"
        }
        
        if options:
            default_options.update(options)
        
        # Read from Kafka
        df = spark.readStream.format("kafka").options(**default_options).load()
        
        logger.info(f"Successfully set up Kafka stream for topic: {topic}")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from Kafka topic {topic}: {str(e)}")
        raise


def extract_from_ct_scanner(
    spark: SparkSession,
    scanner_config: Dict[str, Any],
    scan_parameters: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Extract data from CT scanner system
    
    Args:
        spark: Spark session
        scanner_config: CT scanner configuration
        scan_parameters: Optional scan parameters
        
    Returns:
        DataFrame containing the extracted CT scan data
    """
    try:
        logger.info("Extracting data from CT scanner system")
        
        # CT scan data schema
        ct_scan_schema = StructType([
            StructField("scan_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("voxel_dimensions", StringType(), True),  # JSON string
            StructField("resolution", DoubleType(), True),
            StructField("scan_parameters", StringType(), True),  # JSON string
            StructField("file_path", StringType(), True),
            StructField("file_size", IntegerType(), True),
            StructField("quality_score", DoubleType(), True),
            StructField("defect_count", IntegerType(), True),
            StructField("processing_status", StringType(), True)
        ])
        
        # For now, return an empty DataFrame with the schema
        # In a real implementation, this would connect to the CT scanner system
        df = spark.createDataFrame([], ct_scan_schema)
        
        logger.info("Successfully set up CT scanner data extraction")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from CT scanner: {str(e)}")
        raise


def extract_from_ispm_system(
    spark: SparkSession,
    ispm_config: Dict[str, Any],
    monitoring_parameters: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Extract data from ISPM (In-Situ Process Monitoring) system
    
    Args:
        spark: Spark session
        ispm_config: ISPM system configuration
        monitoring_parameters: Optional monitoring parameters
        
    Returns:
        DataFrame containing the extracted ISPM data
    """
    try:
        logger.info("Extracting data from ISPM system")
        
        # ISPM monitoring data schema
        ispm_schema = StructType([
            StructField("sensor_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("sensor_type", StringType(), True),
            StructField("measurement_value", DoubleType(), True),
            StructField("measurement_unit", StringType(), True),
            StructField("quality_flag", StringType(), True),
            StructField("process_id", StringType(), True),
            StructField("layer_number", IntegerType(), True),
            StructField("x_coordinate", DoubleType(), True),
            StructField("y_coordinate", DoubleType(), True),
            StructField("z_coordinate", DoubleType(), True)
        ])
        
        # For now, return an empty DataFrame with the schema
        # In a real implementation, this would connect to the ISPM system
        df = spark.createDataFrame([], ispm_schema)
        
        logger.info("Successfully set up ISPM system data extraction")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from ISPM system: {str(e)}")
        raise


def extract_pbf_process_data(
    spark: SparkSession,
    source_config: Dict[str, Any]
) -> DataFrame:
    """
    Extract PBF process data from various sources
    
    Args:
        spark: Spark session
        source_config: Source configuration
        
    Returns:
        DataFrame containing the extracted PBF process data
    """
    try:
        logger.info("Extracting PBF process data")
        
        source_type = source_config.get("type", "kafka")
        
        if source_type == "kafka":
            return extract_from_kafka(
                spark=spark,
                bootstrap_servers=source_config.get("bootstrap_servers", "localhost:9092"),
                topic=source_config.get("topic", "pbf_process_data")
            )
        elif source_type == "database":
            return extract_from_database(
                spark=spark,
                connection_string=source_config.get("connection_string", ""),
                table_name=source_config.get("table_name", "pbf_process_data")
            )
        elif source_type == "csv":
            return extract_from_csv(
                spark=spark,
                file_path=source_config.get("file_path", "")
            )
        elif source_type == "json":
            return extract_from_json(
                spark=spark,
                file_path=source_config.get("file_path", "")
            )
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
            
    except Exception as e:
        logger.error(f"Error extracting PBF process data: {str(e)}")
        raise


def extract_powder_bed_data(
    spark: SparkSession,
    source_config: Dict[str, Any]
) -> DataFrame:
    """
    Extract powder bed monitoring data from various sources
    
    Args:
        spark: Spark session
        source_config: Source configuration
        
    Returns:
        DataFrame containing the extracted powder bed data
    """
    try:
        logger.info("Extracting powder bed data")
        
        source_type = source_config.get("type", "kafka")
        
        if source_type == "kafka":
            return extract_from_kafka(
                spark=spark,
                bootstrap_servers=source_config.get("bootstrap_servers", "localhost:9092"),
                topic=source_config.get("topic", "powder_bed_data")
            )
        elif source_type == "database":
            return extract_from_database(
                spark=spark,
                connection_string=source_config.get("connection_string", ""),
                table_name=source_config.get("table_name", "powder_bed_data")
            )
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
            
    except Exception as e:
        logger.error(f"Error extracting powder bed data: {str(e)}")
        raise


# =============================================================================
# NoSQL Data Extraction Functions
# =============================================================================

def extract_from_mongodb(
    spark: SparkSession,
    mongo_config: Dict[str, Any],
    **kwargs
) -> DataFrame:
    """
    Extract data from MongoDB using Spark MongoDB connector
    
    Args:
        spark: Spark session
        mongo_config: MongoDB configuration dictionary
        **kwargs: Additional parameters (connection_string, database_name, etc.)
        
    Returns:
        DataFrame containing the extracted data
    """
    try:
        connection_string = kwargs.get("connection_string", mongo_config.get("connection_string", ""))
        database_name = kwargs.get("database_name", mongo_config.get("database_name", ""))
        collection_name = kwargs.get("collection_name", mongo_config.get("collection_name", ""))
        query = kwargs.get("query", mongo_config.get("query"))
        projection = kwargs.get("projection", mongo_config.get("projection"))
        limit = kwargs.get("limit", mongo_config.get("limit"))
        
        logger.info(f"Extracting data from MongoDB collection: {collection_name}")
        
        # Build MongoDB URI
        mongo_uri = f"{connection_string}/{database_name}.{collection_name}"
        
        # Configure MongoDB connector options
        options = {
            "uri": mongo_uri,
            "partitioner": "MongoSamplePartitioner",
            "partitionerOptions.partitionSizeMB": "64"
        }
        
        if query:
            options["query"] = json.dumps(query)
        
        if projection:
            options["projection"] = json.dumps(projection)
        
        if limit:
            options["limit"] = str(limit)
        
        # Read from MongoDB
        df = spark.read.format("mongo").options(**options).load()
        
        logger.info(f"Successfully extracted {df.count()} documents from MongoDB")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from MongoDB: {str(e)}")
        raise


def extract_from_redis(
    spark: SparkSession,
    redis_config: Dict[str, Any],
    **kwargs
) -> DataFrame:
    """
    Extract data from Redis using Spark Redis connector
    
    Args:
        spark: Spark session
        redis_config: Redis configuration dictionary
        **kwargs: Additional parameters (host, port, password, etc.)
        
    Returns:
        DataFrame containing the extracted data
    """
    try:
        host = kwargs.get("host", redis_config.get("host", "localhost"))
        port = kwargs.get("port", redis_config.get("port", 6379))
        password = kwargs.get("password", redis_config.get("password"))
        db = kwargs.get("db", redis_config.get("db", 0))
        key_pattern = kwargs.get("key_pattern", redis_config.get("key_pattern", "*"))
        data_type = kwargs.get("data_type", redis_config.get("data_type", "string"))
        
        logger.info(f"Extracting {data_type} data from Redis with pattern: {key_pattern}")
        
        # Configure Redis connector options
        options = {
            "host": host,
            "port": str(port),
            "db": str(db),
            "keyPattern": key_pattern,
            "dataType": data_type
        }
        
        if password:
            options["auth"] = password
        
        # Read from Redis
        df = spark.read.format("org.apache.spark.sql.redis").options(**options).load()
        
        logger.info(f"Successfully extracted {df.count()} records from Redis")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from Redis: {str(e)}")
        raise


def extract_from_cassandra(
    spark: SparkSession,
    cassandra_config: Dict[str, Any],
    **kwargs
) -> DataFrame:
    """
    Extract data from Cassandra using Spark Cassandra connector
    
    Args:
        spark: Spark session
        cassandra_config: Cassandra configuration dictionary
        **kwargs: Additional parameters (hosts, keyspace, table_name, etc.)
        
    Returns:
        DataFrame containing the extracted data
    """
    try:
        hosts = kwargs.get("hosts", cassandra_config.get("hosts", ["localhost"]))
        keyspace = kwargs.get("keyspace", cassandra_config.get("keyspace", ""))
        table_name = kwargs.get("table_name", cassandra_config.get("table_name", ""))
        query = kwargs.get("query", cassandra_config.get("query"))
        columns = kwargs.get("columns", cassandra_config.get("columns"))
        
        logger.info(f"Extracting data from Cassandra table: {keyspace}.{table_name}")
        
        # Configure Cassandra connector options
        options = {
            "keyspace": keyspace,
            "table": table_name,
            "spark.cassandra.connection.host": ",".join(hosts),
            "spark.cassandra.connection.port": "9042"
        }
        
        if columns:
            options["spark.cassandra.input.columns"] = ",".join(columns)
        
        # Read from Cassandra
        if query:
            df = spark.read.format("org.apache.spark.sql.cassandra").options(**options).load()
            # Apply additional filtering if query provided
            if "WHERE" in query.upper():
                # Parse and apply WHERE conditions
                where_clause = query.split("WHERE")[1].strip()
                df = df.filter(where_clause)
        else:
            df = spark.read.format("org.apache.spark.sql.cassandra").options(**options).load()
        
        logger.info(f"Successfully extracted {df.count()} records from Cassandra")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from Cassandra: {str(e)}")
        raise


def extract_from_elasticsearch(
    spark: SparkSession,
    elasticsearch_config: Dict[str, Any],
    **kwargs
) -> DataFrame:
    """
    Extract data from Elasticsearch using Spark Elasticsearch connector
    
    Args:
        spark: Spark session
        elasticsearch_config: Elasticsearch configuration dictionary
        **kwargs: Additional parameters (hosts, index_name, query, etc.)
        
    Returns:
        DataFrame containing the extracted data
    """
    try:
        hosts = kwargs.get("hosts", elasticsearch_config.get("hosts", ["localhost"]))
        index_name = kwargs.get("index_name", elasticsearch_config.get("index_name", ""))
        query = kwargs.get("query", elasticsearch_config.get("query"))
        size = kwargs.get("size", elasticsearch_config.get("size", 10000))
        scroll = kwargs.get("scroll", elasticsearch_config.get("scroll", "5m"))
        
        logger.info(f"Extracting data from Elasticsearch index: {index_name}")
        
        # Configure Elasticsearch connector options
        options = {
            "es.nodes": ",".join(hosts),
            "es.port": "9200",
            "es.resource": index_name,
            "es.scroll.size": str(size),
            "es.scroll.keepalive": scroll
        }
        
        if query:
            options["es.query"] = json.dumps(query)
        
        # Read from Elasticsearch
        df = spark.read.format("org.elasticsearch.spark.sql").options(**options).load()
        
        logger.info(f"Successfully extracted {df.count()} documents from Elasticsearch")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from Elasticsearch: {str(e)}")
        raise

