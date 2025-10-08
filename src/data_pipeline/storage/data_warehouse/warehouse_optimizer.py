"""
Warehouse Optimizer for PBF-LB/M Data Pipeline

This module provides warehouse optimization capabilities for data warehouse operations.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from src.data_pipeline.storage.data_warehouse.snowflake_client import SnowflakeClient

logger = logging.getLogger(__name__)


class WarehouseOptimizer:
    """
    Warehouse optimizer for data warehouse operations.
    """
    
    def __init__(self, snowflake_client: Optional[SnowflakeClient] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize warehouse optimizer.
        
        Args:
            snowflake_client: Optional Snowflake client instance
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.snowflake_client = snowflake_client or SnowflakeClient()
        self.optimization_stats = {
            "optimizations_performed": 0,
            "tables_optimized": 0,
            "queries_optimized": 0,
            "start_time": datetime.now()
        }
    
    def analyze_query_performance(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Analyze query performance and provide optimization recommendations.
        
        Args:
            query: SQL query to analyze
            params: Optional query parameters
            
        Returns:
            Dict[str, Any]: Query performance analysis
        """
        try:
            # Execute query with EXPLAIN PLAN
            explain_query = f"EXPLAIN {query}"
            explain_results = self.snowflake_client.execute_query(explain_query, params)
            
            # Analyze execution plan
            analysis = {
                "query": query,
                "execution_plan": explain_results,
                "recommendations": [],
                "performance_score": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Basic performance analysis
            if explain_results:
                # Check for table scans
                for row in explain_results:
                    if "TableScan" in str(row):
                        analysis["recommendations"].append("Consider adding indexes to avoid table scans")
                        analysis["performance_score"] -= 10
                    
                    if "Sort" in str(row):
                        analysis["recommendations"].append("Consider adding ORDER BY indexes")
                        analysis["performance_score"] -= 5
                    
                    if "HashJoin" in str(row):
                        analysis["recommendations"].append("Consider optimizing join conditions")
                        analysis["performance_score"] -= 3
            
            # Add positive score for good practices
            if "WHERE" in query.upper():
                analysis["performance_score"] += 5
            if "LIMIT" in query.upper():
                analysis["performance_score"] += 3
            if "INDEX" in query.upper():
                analysis["performance_score"] += 2
            
            # Normalize score to 0-100
            analysis["performance_score"] = max(0, min(100, analysis["performance_score"] + 50))
            
            logger.info(f"Analyzed query performance, score: {analysis['performance_score']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query performance: {e}")
            return {
                "query": query,
                "execution_plan": [],
                "recommendations": [f"Error analyzing query: {str(e)}"],
                "performance_score": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def optimize_table_storage(self, table_name: str) -> Dict[str, Any]:
        """
        Optimize table storage and provide recommendations.
        
        Args:
            table_name: Name of the table to optimize
            
        Returns:
            Dict[str, Any]: Table optimization results
        """
        try:
            # Get table information
            table_info = self.snowflake_client.get_table_info(table_name)
            if not table_info:
                logger.error(f"Could not get information for table {table_name}")
                return {"success": False, "error": "Table not found"}
            
            # Analyze table structure
            optimization = {
                "table_name": table_name,
                "current_info": table_info,
                "recommendations": [],
                "optimizations_applied": [],
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Check for optimization opportunities
            schema = table_info.get("schema", [])
            
            # Check for missing indexes
            if table_info["row_count"] > 10000:  # Only for larger tables
                optimization["recommendations"].append("Consider adding indexes for frequently queried columns")
            
            # Check for data types
            for column in schema:
                if column["data_type"].upper() == "VARCHAR" and "max" in column["data_type"].lower():
                    optimization["recommendations"].append(f"Consider specifying length for column {column['column_name']}")
                
                if column["data_type"].upper() == "NUMBER" and not column["data_type"].upper().endswith(")"):
                    optimization["recommendations"].append(f"Consider specifying precision for column {column['column_name']}")
            
            # Check for partitioning opportunities
            if table_info["row_count"] > 1000000:  # 1M+ rows
                optimization["recommendations"].append("Consider partitioning for large tables")
            
            # Apply automatic optimizations
            if optimization["recommendations"]:
                optimization["optimizations_applied"].append("Generated optimization recommendations")
                self.optimization_stats["tables_optimized"] += 1
            
            logger.info(f"Optimized table storage for {table_name}")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing table storage for {table_name}: {e}")
            return {
                "table_name": table_name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def optimize_warehouse_size(self, current_size: str, usage_pattern: str = "mixed") -> Dict[str, Any]:
        """
        Optimize warehouse size based on usage patterns.
        
        Args:
            current_size: Current warehouse size
            usage_pattern: Usage pattern ("cpu_intensive", "memory_intensive", "mixed")
            
        Returns:
            Dict[str, Any]: Warehouse size optimization recommendations
        """
        try:
            # Size mapping
            size_hierarchy = ["XSMALL", "SMALL", "MEDIUM", "LARGE", "XLARGE", "XXLARGE", "XXXLARGE"]
            
            current_index = size_hierarchy.index(current_size.upper()) if current_size.upper() in size_hierarchy else 2
            
            optimization = {
                "current_size": current_size,
                "usage_pattern": usage_pattern,
                "recommendations": [],
                "suggested_size": current_size,
                "reasoning": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Size recommendations based on usage pattern
            if usage_pattern == "cpu_intensive":
                if current_index < 4:  # Below LARGE
                    optimization["suggested_size"] = size_hierarchy[min(current_index + 1, len(size_hierarchy) - 1)]
                    optimization["reasoning"].append("CPU-intensive workloads benefit from larger warehouses")
                else:
                    optimization["reasoning"].append("Current warehouse size is appropriate for CPU-intensive workloads")
            
            elif usage_pattern == "memory_intensive":
                if current_index < 3:  # Below MEDIUM
                    optimization["suggested_size"] = size_hierarchy[min(current_index + 2, len(size_hierarchy) - 1)]
                    optimization["reasoning"].append("Memory-intensive workloads require larger warehouses")
                else:
                    optimization["reasoning"].append("Current warehouse size is appropriate for memory-intensive workloads")
            
            else:  # mixed
                if current_index < 2:  # Below MEDIUM
                    optimization["suggested_size"] = size_hierarchy[min(current_index + 1, len(size_hierarchy) - 1)]
                    optimization["reasoning"].append("Mixed workloads benefit from medium-sized warehouses")
                else:
                    optimization["reasoning"].append("Current warehouse size is appropriate for mixed workloads")
            
            # Add general recommendations
            optimization["recommendations"].extend([
                "Monitor warehouse usage and adjust size based on actual performance",
                "Consider auto-suspend settings to optimize costs",
                "Use query result caching to reduce warehouse usage"
            ])
            
            logger.info(f"Optimized warehouse size recommendation: {current_size} -> {optimization['suggested_size']}")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing warehouse size: {e}")
            return {
                "current_size": current_size,
                "usage_pattern": usage_pattern,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def optimize_query_cache(self, query: str) -> Dict[str, Any]:
        """
        Optimize query for better caching.
        
        Args:
            query: SQL query to optimize
            
        Returns:
            Dict[str, Any]: Query cache optimization results
        """
        try:
            optimization = {
                "query": query,
                "cache_optimizations": [],
                "recommendations": [],
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Check for cache-friendly patterns
            query_upper = query.upper()
            
            # Check for deterministic functions
            if "NOW()" in query_upper or "CURRENT_TIMESTAMP" in query_upper:
                optimization["cache_optimizations"].append("Replace NOW() with specific timestamp for better caching")
                optimization["recommendations"].append("Use parameterized queries with specific timestamps")
            
            # Check for random functions
            if "RANDOM()" in query_upper or "RAND()" in query_upper:
                optimization["cache_optimizations"].append("Random functions prevent query caching")
                optimization["recommendations"].append("Consider using deterministic alternatives")
            
            # Check for user-specific functions
            if "CURRENT_USER" in query_upper or "USER" in query_upper:
                optimization["cache_optimizations"].append("User-specific functions may limit caching")
                optimization["recommendations"].append("Consider using role-based access instead")
            
            # Check for proper WHERE clauses
            if "WHERE" not in query_upper:
                optimization["recommendations"].append("Add WHERE clauses to limit result sets and improve caching")
            
            # Check for LIMIT clauses
            if "LIMIT" not in query_upper and "SELECT" in query_upper:
                optimization["recommendations"].append("Add LIMIT clauses to prevent large result sets")
            
            logger.info(f"Optimized query for caching: {len(optimization['cache_optimizations'])} optimizations found")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing query cache: {e}")
            return {
                "query": query,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def optimize_data_compression(self, table_name: str) -> Dict[str, Any]:
        """
        Optimize data compression for a table.
        
        Args:
            table_name: Name of the table to optimize
            
        Returns:
            Dict[str, Any]: Data compression optimization results
        """
        try:
            # Get table information
            table_info = self.snowflake_client.get_table_info(table_name)
            if not table_info:
                logger.error(f"Could not get information for table {table_name}")
                return {"success": False, "error": "Table not found"}
            
            optimization = {
                "table_name": table_name,
                "compression_optimizations": [],
                "recommendations": [],
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Analyze table structure for compression opportunities
            schema = table_info.get("schema", [])
            row_count = table_info.get("row_count", 0)
            
            # Check for compression opportunities
            if row_count > 100000:  # Only for larger tables
                optimization["compression_optimizations"].append("Table is large enough to benefit from compression")
                optimization["recommendations"].append("Consider using columnar storage formats")
            
            # Check for data types that benefit from compression
            for column in schema:
                if column["data_type"].upper() in ["VARCHAR", "TEXT", "CHAR"]:
                    optimization["compression_optimizations"].append(f"Column {column['column_name']} uses text data type - good for compression")
                
                if column["data_type"].upper() in ["NUMBER", "INTEGER", "BIGINT"]:
                    optimization["compression_optimizations"].append(f"Column {column['column_name']} uses numeric data type - good for compression")
            
            # Add general compression recommendations
            optimization["recommendations"].extend([
                "Use appropriate data types to maximize compression",
                "Consider partitioning for better compression ratios",
                "Regularly analyze and optimize table compression"
            ])
            
            logger.info(f"Optimized data compression for {table_name}")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing data compression for {table_name}: {e}")
            return {
                "table_name": table_name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_optimization_recommendations(self, table_name: str, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive optimization recommendations for a table and optional query.
        
        Args:
            table_name: Name of the table
            query: Optional SQL query to analyze
            
        Returns:
            Dict[str, Any]: Comprehensive optimization recommendations
        """
        try:
            recommendations = {
                "table_name": table_name,
                "table_optimization": {},
                "query_optimization": {},
                "warehouse_optimization": {},
                "compression_optimization": {},
                "overall_score": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Table optimization
            table_opt = self.optimize_table_storage(table_name)
            recommendations["table_optimization"] = table_opt
            
            # Query optimization
            if query:
                query_opt = self.analyze_query_performance(query)
                recommendations["query_optimization"] = query_opt
            
            # Warehouse optimization
            warehouse_opt = self.optimize_warehouse_size("MEDIUM", "mixed")
            recommendations["warehouse_optimization"] = warehouse_opt
            
            # Compression optimization
            compression_opt = self.optimize_data_compression(table_name)
            recommendations["compression_optimization"] = compression_opt
            
            # Calculate overall score
            scores = []
            if table_opt.get("success"):
                scores.append(75)  # Base score for table
            if query and query_opt.get("performance_score"):
                scores.append(query_opt["performance_score"])
            if compression_opt.get("success"):
                scores.append(80)  # Base score for compression
            
            recommendations["overall_score"] = sum(scores) / len(scores) if scores else 0
            
            # Update stats
            self.optimization_stats["optimizations_performed"] += 1
            if query:
                self.optimization_stats["queries_optimized"] += 1
            
            logger.info(f"Generated optimization recommendations for {table_name}, overall score: {recommendations['overall_score']}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting optimization recommendations for {table_name}: {e}")
            return {
                "table_name": table_name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Dict[str, Any]: Optimization statistics
        """
        current_time = datetime.now()
        total_time = (current_time - self.optimization_stats["start_time"]).total_seconds()
        
        stats = self.optimization_stats.copy()
        stats.update({
            "total_time": total_time,
            "optimizations_per_second": stats["optimizations_performed"] / total_time if total_time > 0 else 0,
            "current_time": current_time.isoformat()
        })
        
        return stats
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "optimization_stats": self.get_optimization_stats(),
            "ingestion_timestamp": datetime.now().isoformat()
        }
