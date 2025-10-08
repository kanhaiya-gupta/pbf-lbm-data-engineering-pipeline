"""
Data Transformation Module

This module provides Spark-based data transformation functions for PBF-LB/M data processing.
It includes specialized transformations for different data types and business rules.
"""

from typing import Dict, Any, Optional, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, isnan, isnull, coalesce, lit, current_timestamp,
    regexp_replace, trim, upper, lower, split, explode,
    sum as spark_sum, avg, min as spark_min, max as spark_max,
    count, countDistinct, row_number, rank, dense_rank,
    lag, lead, window, to_timestamp, date_format, year, month, dayofmonth
)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, BooleanType
import logging

logger = logging.getLogger(__name__)


def transform_pbf_process_data(
    df: DataFrame,
    transformation_config: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Transform PBF process data using Spark
    
    Args:
        df: Input DataFrame containing PBF process data
        transformation_config: Optional transformation configuration
        
    Returns:
        Transformed DataFrame
    """
    try:
        logger.info("Transforming PBF process data")
        
        if transformation_config is None:
            transformation_config = {}
        
        # Apply data cleaning transformations
        df_cleaned = _clean_pbf_process_data(df)
        
        # Apply data enrichment transformations
        df_enriched = _enrich_pbf_process_data(df_cleaned, transformation_config)
        
        # Apply data validation transformations
        df_validated = _validate_pbf_process_data(df_enriched, transformation_config)
        
        # Apply business rule transformations
        df_transformed = _apply_pbf_business_rules(df_validated, transformation_config)
        
        logger.info(f"Successfully transformed PBF process data: {df_transformed.count()} records")
        return df_transformed
        
    except Exception as e:
        logger.error(f"Error transforming PBF process data: {str(e)}")
        raise


def transform_ispm_monitoring_data(
    df: DataFrame,
    transformation_config: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Transform ISPM monitoring data using Spark
    
    Args:
        df: Input DataFrame containing ISPM monitoring data
        transformation_config: Optional transformation configuration
        
    Returns:
        Transformed DataFrame
    """
    try:
        logger.info("Transforming ISPM monitoring data")
        
        if transformation_config is None:
            transformation_config = {}
        
        # Apply data cleaning transformations
        df_cleaned = _clean_ispm_monitoring_data(df)
        
        # Apply data aggregation transformations
        df_aggregated = _aggregate_ispm_monitoring_data(df_cleaned, transformation_config)
        
        # Apply anomaly detection transformations
        df_anomaly_detected = _detect_ispm_anomalies(df_aggregated, transformation_config)
        
        # Apply quality scoring transformations
        df_quality_scored = _score_ispm_quality(df_anomaly_detected, transformation_config)
        
        logger.info(f"Successfully transformed ISPM monitoring data: {df_quality_scored.count()} records")
        return df_quality_scored
        
    except Exception as e:
        logger.error(f"Error transforming ISPM monitoring data: {str(e)}")
        raise


def transform_ct_scan_data(
    df: DataFrame,
    transformation_config: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Transform CT scan data using Spark
    
    Args:
        df: Input DataFrame containing CT scan data
        transformation_config: Optional transformation configuration
        
    Returns:
        Transformed DataFrame
    """
    try:
        logger.info("Transforming CT scan data")
        
        if transformation_config is None:
            transformation_config = {}
        
        # Apply data cleaning transformations
        df_cleaned = _clean_ct_scan_data(df)
        
        # Apply metadata extraction transformations
        df_metadata_extracted = _extract_ct_scan_metadata(df_cleaned, transformation_config)
        
        # Apply defect detection transformations
        df_defect_detected = _detect_ct_scan_defects(df_metadata_extracted, transformation_config)
        
        # Apply quality assessment transformations
        df_quality_assessed = _assess_ct_scan_quality(df_defect_detected, transformation_config)
        
        logger.info(f"Successfully transformed CT scan data: {df_quality_assessed.count()} records")
        return df_quality_assessed
        
    except Exception as e:
        logger.error(f"Error transforming CT scan data: {str(e)}")
        raise


def transform_powder_bed_data(
    df: DataFrame,
    transformation_config: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Transform powder bed data using Spark
    
    Args:
        df: Input DataFrame containing powder bed data
        transformation_config: Optional transformation configuration
        
    Returns:
        Transformed DataFrame
    """
    try:
        logger.info("Transforming powder bed data")
        
        if transformation_config is None:
            transformation_config = {}
        
        # Apply data cleaning transformations
        df_cleaned = _clean_powder_bed_data(df)
        
        # Apply image analysis transformations
        df_image_analyzed = _analyze_powder_bed_images(df_cleaned, transformation_config)
        
        # Apply surface quality transformations
        df_surface_quality = _assess_powder_bed_surface_quality(df_image_analyzed, transformation_config)
        
        # Apply layer consistency transformations
        df_layer_consistent = _assess_layer_consistency(df_surface_quality, transformation_config)
        
        logger.info(f"Successfully transformed powder bed data: {df_layer_consistent.count()} records")
        return df_layer_consistent
        
    except Exception as e:
        logger.error(f"Error transforming powder bed data: {str(e)}")
        raise


def apply_business_rules(
    df: DataFrame,
    business_rules: Dict[str, Any],
    data_type: str
) -> DataFrame:
    """
    Apply business rules to transformed data
    
    Args:
        df: Input DataFrame
        business_rules: Business rules configuration
        data_type: Type of data being processed
        
    Returns:
        DataFrame with business rules applied
    """
    try:
        logger.info(f"Applying business rules for {data_type} data")
        
        # Apply common business rules
        df_with_rules = _apply_common_business_rules(df, business_rules)
        
        # Apply data-type specific business rules
        if data_type == "pbf_process":
            df_with_rules = _apply_pbf_process_business_rules(df_with_rules, business_rules)
        elif data_type == "ispm_monitoring":
            df_with_rules = _apply_ispm_monitoring_business_rules(df_with_rules, business_rules)
        elif data_type == "ct_scan":
            df_with_rules = _apply_ct_scan_business_rules(df_with_rules, business_rules)
        elif data_type == "powder_bed":
            df_with_rules = _apply_powder_bed_business_rules(df_with_rules, business_rules)
        
        logger.info(f"Successfully applied business rules for {data_type} data")
        return df_with_rules
        
    except Exception as e:
        logger.error(f"Error applying business rules for {data_type} data: {str(e)}")
        raise


# Helper functions for PBF process data transformations
def _clean_pbf_process_data(df: DataFrame) -> DataFrame:
    """Clean PBF process data"""
    return df.filter(
        col("process_id").isNotNull() &
        col("timestamp").isNotNull() &
        col("laser_power").isNotNull() &
        col("scan_speed").isNotNull()
    ).withColumn(
        "laser_power", 
        when(col("laser_power") < 0, 0).otherwise(col("laser_power"))
    ).withColumn(
        "scan_speed",
        when(col("scan_speed") < 0, 0).otherwise(col("scan_speed"))
    )


def _enrich_pbf_process_data(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Enrich PBF process data with additional fields"""
    return df.withColumn(
        "processing_timestamp", current_timestamp()
    ).withColumn(
        "data_source", lit("pbf_process")
    ).withColumn(
        "quality_tier",
        when(col("laser_power") > 200, "high")
        .when(col("laser_power") > 100, "medium")
        .otherwise("low")
    )


def _validate_pbf_process_data(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Validate PBF process data"""
    return df.withColumn(
        "is_valid",
        when(
            (col("laser_power") >= 0) &
            (col("laser_power") <= 1000) &
            (col("scan_speed") >= 0) &
            (col("scan_speed") <= 10000),
            True
        ).otherwise(False)
    )


def _apply_pbf_business_rules(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Apply PBF-specific business rules"""
    return df.withColumn(
        "process_efficiency",
        col("laser_power") / (col("scan_speed") + 1)
    ).withColumn(
        "energy_density",
        col("laser_power") * col("scan_speed")
    )


# Helper functions for ISPM monitoring data transformations
def _clean_ispm_monitoring_data(df: DataFrame) -> DataFrame:
    """Clean ISPM monitoring data"""
    return df.filter(
        col("sensor_id").isNotNull() &
        col("timestamp").isNotNull() &
        col("measurement_value").isNotNull()
    ).withColumn(
        "measurement_value",
        when(isnan(col("measurement_value")), 0).otherwise(col("measurement_value"))
    )


def _aggregate_ispm_monitoring_data(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Aggregate ISPM monitoring data"""
    window_spec = window.partitionBy("sensor_id").orderBy("timestamp")
    
    return df.withColumn(
        "measurement_avg_5min",
        avg("measurement_value").over(window_spec.rowsBetween(-4, 0))
    ).withColumn(
        "measurement_std_5min",
        spark_sum((col("measurement_value") - col("measurement_avg_5min")) ** 2)
        .over(window_spec.rowsBetween(-4, 0))
    )


def _detect_ispm_anomalies(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Detect anomalies in ISPM monitoring data"""
    return df.withColumn(
        "is_anomaly",
        when(
            abs(col("measurement_value") - col("measurement_avg_5min")) > 
            (2 * col("measurement_std_5min")),
            True
        ).otherwise(False)
    )


def _score_ispm_quality(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Score ISPM data quality"""
    return df.withColumn(
        "quality_score",
        when(col("is_anomaly"), 0.5)
        .when(col("measurement_std_5min") > 10, 0.7)
        .otherwise(1.0)
    )


# Helper functions for CT scan data transformations
def _clean_ct_scan_data(df: DataFrame) -> DataFrame:
    """Clean CT scan data"""
    return df.filter(
        col("scan_id").isNotNull() &
        col("timestamp").isNotNull() &
        col("file_path").isNotNull()
    )


def _extract_ct_scan_metadata(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Extract metadata from CT scan data"""
    return df.withColumn(
        "file_extension",
        split(col("file_path"), "\\.")[-1]
    ).withColumn(
        "scan_date",
        date_format(col("timestamp"), "yyyy-MM-dd")
    )


def _detect_ct_scan_defects(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Detect defects in CT scan data"""
    return df.withColumn(
        "has_defects",
        when(col("defect_count") > 0, True).otherwise(False)
    ).withColumn(
        "defect_density",
        col("defect_count") / (col("resolution") ** 3)
    )


def _assess_ct_scan_quality(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Assess CT scan data quality"""
    return df.withColumn(
        "quality_assessment",
        when(col("quality_score") > 0.9, "excellent")
        .when(col("quality_score") > 0.7, "good")
        .when(col("quality_score") > 0.5, "fair")
        .otherwise("poor")
    )


# Helper functions for powder bed data transformations
def _clean_powder_bed_data(df: DataFrame) -> DataFrame:
    """Clean powder bed data"""
    return df.filter(
        col("camera_id").isNotNull() &
        col("timestamp").isNotNull() &
        col("image_path").isNotNull()
    )


def _analyze_powder_bed_images(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Analyze powder bed images"""
    return df.withColumn(
        "image_analysis_timestamp", current_timestamp()
    ).withColumn(
        "surface_roughness", lit(0.0)  # Placeholder for actual image analysis
    )


def _assess_powder_bed_surface_quality(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Assess powder bed surface quality"""
    return df.withColumn(
        "surface_quality",
        when(col("surface_roughness") < 0.1, "excellent")
        .when(col("surface_roughness") < 0.3, "good")
        .when(col("surface_roughness") < 0.5, "fair")
        .otherwise("poor")
    )


def _assess_layer_consistency(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Assess layer consistency in powder bed data"""
    return df.withColumn(
        "layer_consistency_score",
        when(col("surface_quality") == "excellent", 1.0)
        .when(col("surface_quality") == "good", 0.8)
        .when(col("surface_quality") == "fair", 0.6)
        .otherwise(0.4)
    )


# Business rules helper functions
def _apply_common_business_rules(df: DataFrame, rules: Dict[str, Any]) -> DataFrame:
    """Apply common business rules"""
    return df.withColumn(
        "business_rule_version", lit(rules.get("version", "1.0"))
    ).withColumn(
        "rule_application_timestamp", current_timestamp()
    )


def _apply_pbf_process_business_rules(df: DataFrame, rules: Dict[str, Any]) -> DataFrame:
    """Apply PBF process-specific business rules"""
    return df.withColumn(
        "process_category",
        when(col("laser_power") > 300, "high_power")
        .when(col("laser_power") > 150, "medium_power")
        .otherwise("low_power")
    )


def _apply_ispm_monitoring_business_rules(df: DataFrame, rules: Dict[str, Any]) -> DataFrame:
    """Apply ISPM monitoring-specific business rules"""
    return df.withColumn(
        "monitoring_alert_level",
        when(col("is_anomaly"), "high")
        .when(col("quality_score") < 0.8, "medium")
        .otherwise("low")
    )


def _apply_ct_scan_business_rules(df: DataFrame, rules: Dict[str, Any]) -> DataFrame:
    """Apply CT scan-specific business rules"""
    return df.withColumn(
        "scan_priority",
        when(col("defect_count") > 10, "high")
        .when(col("defect_count") > 5, "medium")
        .otherwise("low")
    )


def _apply_powder_bed_business_rules(df: DataFrame, rules: Dict[str, Any]) -> DataFrame:
    """Apply powder bed-specific business rules"""
    return df.withColumn(
        "layer_quality_grade",
        when(col("layer_consistency_score") > 0.9, "A")
        .when(col("layer_consistency_score") > 0.7, "B")
        .when(col("layer_consistency_score") > 0.5, "C")
        .otherwise("D")
    )


# =============================================================================
# NoSQL Data Transformation Functions
# =============================================================================

def transform_document_data(
    df: DataFrame,
    document_config: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Transform document-based data (MongoDB) for PBF-LB/M processing
    
    Args:
        df: Input DataFrame containing document data
        document_config: Optional document transformation configuration
        
    Returns:
        Transformed DataFrame with flattened document structure
    """
    try:
        logger.info("Transforming document data for PBF-LB/M processing")
        
        if document_config is None:
            document_config = {}
        
        # Flatten nested document structures
        df_flattened = _flatten_document_structure(df, document_config)
        
        # Apply document-specific transformations
        df_transformed = _apply_document_transformations(df_flattened, document_config)
        
        # Validate document data
        df_validated = _validate_document_data(df_transformed, document_config)
        
        logger.info("Successfully transformed document data")
        return df_validated
        
    except Exception as e:
        logger.error(f"Error transforming document data: {str(e)}")
        raise


def transform_key_value_data(
    df: DataFrame,
    kv_config: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Transform key-value data (Redis) for PBF-LB/M processing
    
    Args:
        df: Input DataFrame containing key-value data
        kv_config: Optional key-value transformation configuration
        
    Returns:
        Transformed DataFrame with structured key-value data
    """
    try:
        logger.info("Transforming key-value data for PBF-LB/M processing")
        
        if kv_config is None:
            kv_config = {}
        
        # Parse key-value structures
        df_parsed = _parse_key_value_structure(df, kv_config)
        
        # Apply key-value specific transformations
        df_transformed = _apply_key_value_transformations(df_parsed, kv_config)
        
        # Validate key-value data
        df_validated = _validate_key_value_data(df_transformed, kv_config)
        
        logger.info("Successfully transformed key-value data")
        return df_validated
        
    except Exception as e:
        logger.error(f"Error transforming key-value data: {str(e)}")
        raise


def transform_columnar_data(
    df: DataFrame,
    columnar_config: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Transform columnar data (Cassandra) for PBF-LB/M processing
    
    Args:
        df: Input DataFrame containing columnar data
        columnar_config: Optional columnar transformation configuration
        
    Returns:
        Transformed DataFrame with optimized columnar structure
    """
    try:
        logger.info("Transforming columnar data for PBF-LB/M processing")
        
        if columnar_config is None:
            columnar_config = {}
        
        # Optimize columnar structure
        df_optimized = _optimize_columnar_structure(df, columnar_config)
        
        # Apply columnar-specific transformations
        df_transformed = _apply_columnar_transformations(df_optimized, columnar_config)
        
        # Validate columnar data
        df_validated = _validate_columnar_data(df_transformed, columnar_config)
        
        logger.info("Successfully transformed columnar data")
        return df_validated
        
    except Exception as e:
        logger.error(f"Error transforming columnar data: {str(e)}")
        raise


def transform_graph_data(
    df: DataFrame,
    graph_config: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Transform graph data (Neo4j) for PBF-LB/M processing
    
    Args:
        df: Input DataFrame containing graph data
        graph_config: Optional graph transformation configuration
        
    Returns:
        Transformed DataFrame with graph relationships
    """
    try:
        logger.info("Transforming graph data for PBF-LB/M processing")
        
        if graph_config is None:
            graph_config = {}
        
        # Extract graph relationships
        df_relationships = _extract_graph_relationships(df, graph_config)
        
        # Apply graph-specific transformations
        df_transformed = _apply_graph_transformations(df_relationships, graph_config)
        
        # Validate graph data
        df_validated = _validate_graph_data(df_transformed, graph_config)
        
        logger.info("Successfully transformed graph data")
        return df_validated
        
    except Exception as e:
        logger.error(f"Error transforming graph data: {str(e)}")
        raise


def transform_multi_model_data(
    df: DataFrame,
    source_type: str,
    transformation_config: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Generic function to transform data from various NoSQL sources
    
    Args:
        df: Input DataFrame containing NoSQL data
        source_type: Type of NoSQL source (mongodb, redis, cassandra, elasticsearch, neo4j)
        transformation_config: Optional transformation configuration
        
    Returns:
        Transformed DataFrame
    """
    try:
        logger.info(f"Transforming multi-model data from source: {source_type}")
        
        if transformation_config is None:
            transformation_config = {}
        
        if source_type.lower() == "mongodb":
            return transform_document_data(df, transformation_config)
        
        elif source_type.lower() == "redis":
            return transform_key_value_data(df, transformation_config)
        
        elif source_type.lower() == "cassandra":
            return transform_columnar_data(df, transformation_config)
        
        elif source_type.lower() == "elasticsearch":
            # Elasticsearch data is similar to document data
            return transform_document_data(df, transformation_config)
        
        elif source_type.lower() == "neo4j":
            return transform_graph_data(df, transformation_config)
        
        else:
            raise ValueError(f"Unsupported NoSQL source type: {source_type}")
            
    except Exception as e:
        logger.error(f"Error transforming multi-model data from {source_type}: {str(e)}")
        raise


# =============================================================================
# Helper Functions for NoSQL Transformations
# =============================================================================

def _flatten_document_structure(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Flatten nested document structures"""
    try:
        # Get columns that contain nested structures
        nested_columns = config.get("nested_columns", [])
        
        df_flattened = df
        for col_name in nested_columns:
            if col_name in df.columns:
                # Explode nested arrays
                df_flattened = df_flattened.withColumn(
                    f"{col_name}_exploded",
                    explode(col(col_name))
                ).drop(col_name)
        
        return df_flattened
        
    except Exception as e:
        logger.error(f"Error flattening document structure: {str(e)}")
        raise


def _apply_document_transformations(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Apply document-specific transformations"""
    try:
        # Apply field mappings
        field_mappings = config.get("field_mappings", {})
        for old_field, new_field in field_mappings.items():
            if old_field in df.columns:
                df = df.withColumnRenamed(old_field, new_field)
        
        # Apply data type conversions
        type_conversions = config.get("type_conversions", {})
        for field, target_type in type_conversions.items():
            if field in df.columns:
                df = df.withColumn(field, col(field).cast(target_type))
        
        return df
        
    except Exception as e:
        logger.error(f"Error applying document transformations: {str(e)}")
        raise


def _validate_document_data(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Validate document data"""
    try:
        # Apply validation rules
        validation_rules = config.get("validation_rules", {})
        
        for field, rules in validation_rules.items():
            if field in df.columns:
                # Apply field-specific validation
                if "required" in rules and rules["required"]:
                    df = df.filter(col(field).isNotNull())
                
                if "min_length" in rules:
                    df = df.filter(length(col(field)) >= rules["min_length"])
                
                if "max_length" in rules:
                    df = df.filter(length(col(field)) <= rules["max_length"])
        
        return df
        
    except Exception as e:
        logger.error(f"Error validating document data: {str(e)}")
        raise


def _parse_key_value_structure(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Parse key-value structures"""
    try:
        # Parse JSON values if they exist
        json_columns = config.get("json_columns", [])
        
        df_parsed = df
        for col_name in json_columns:
            if col_name in df.columns:
                # Parse JSON strings
                df_parsed = df_parsed.withColumn(
                    f"{col_name}_parsed",
                    from_json(col(col_name), "MAP<STRING,STRING>")
                )
        
        return df_parsed
        
    except Exception as e:
        logger.error(f"Error parsing key-value structure: {str(e)}")
        raise


def _apply_key_value_transformations(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Apply key-value specific transformations"""
    try:
        # Apply key transformations
        key_transformations = config.get("key_transformations", {})
        
        for key_pattern, transformation in key_transformations.items():
            # Apply pattern-based transformations
            if transformation == "uppercase":
                df = df.withColumn("key", upper(col("key")))
            elif transformation == "lowercase":
                df = df.withColumn("key", lower(col("key")))
        
        return df
        
    except Exception as e:
        logger.error(f"Error applying key-value transformations: {str(e)}")
        raise


def _validate_key_value_data(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Validate key-value data"""
    try:
        # Apply key validation
        key_validation = config.get("key_validation", {})
        
        if "required_pattern" in key_validation:
            pattern = key_validation["required_pattern"]
            df = df.filter(col("key").rlike(pattern))
        
        return df
        
    except Exception as e:
        logger.error(f"Error validating key-value data: {str(e)}")
        raise


def _optimize_columnar_structure(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Optimize columnar structure for Cassandra data"""
    try:
        # Apply partitioning optimizations
        partition_columns = config.get("partition_columns", [])
        
        if partition_columns:
            # Ensure partition columns are properly typed
            for col_name in partition_columns:
                if col_name in df.columns:
                    df = df.withColumn(col_name, col(col_name).cast("string"))
        
        return df
        
    except Exception as e:
        logger.error(f"Error optimizing columnar structure: {str(e)}")
        raise


def _apply_columnar_transformations(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Apply columnar-specific transformations"""
    try:
        # Apply clustering key optimizations
        clustering_keys = config.get("clustering_keys", [])
        
        for key in clustering_keys:
            if key in df.columns:
                # Ensure proper ordering for clustering keys
                df = df.orderBy(col(key))
        
        return df
        
    except Exception as e:
        logger.error(f"Error applying columnar transformations: {str(e)}")
        raise


def _validate_columnar_data(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Validate columnar data"""
    try:
        # Apply columnar validation rules
        validation_rules = config.get("validation_rules", {})
        
        for field, rules in validation_rules.items():
            if field in df.columns:
                # Apply field-specific validation
                if "not_null" in rules and rules["not_null"]:
                    df = df.filter(col(field).isNotNull())
        
        return df
        
    except Exception as e:
        logger.error(f"Error validating columnar data: {str(e)}")
        raise


def _extract_graph_relationships(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Extract graph relationships from Neo4j data"""
    try:
        # Extract node and relationship data
        node_columns = config.get("node_columns", [])
        relationship_columns = config.get("relationship_columns", [])
        
        # Create separate DataFrames for nodes and relationships
        if node_columns:
            node_df = df.select(*node_columns)
        
        if relationship_columns:
            rel_df = df.select(*relationship_columns)
        
        return df
        
    except Exception as e:
        logger.error(f"Error extracting graph relationships: {str(e)}")
        raise


def _apply_graph_transformations(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Apply graph-specific transformations"""
    try:
        # Apply relationship transformations
        relationship_mappings = config.get("relationship_mappings", {})
        
        for old_rel, new_rel in relationship_mappings.items():
            if old_rel in df.columns:
                df = df.withColumnRenamed(old_rel, new_rel)
        
        return df
        
    except Exception as e:
        logger.error(f"Error applying graph transformations: {str(e)}")
        raise


def _validate_graph_data(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Validate graph data"""
    try:
        # Apply graph validation rules
        validation_rules = config.get("validation_rules", {})
        
        # Validate node properties
        if "node_properties" in validation_rules:
            node_props = validation_rules["node_properties"]
            for prop in node_props:
                if prop in df.columns:
                    df = df.filter(col(prop).isNotNull())
        
        return df
        
    except Exception as e:
        logger.error(f"Error validating graph data: {str(e)}")
        raise
