"""
Process Data Loader

This module implements utilities for loading process parameter data
from PBF-LB/M manufacturing systems.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import yaml
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ProcessDataLoader:
    """
    Utility class for loading process parameter data from various sources.
    
    This class handles:
    - Loading process parameters from CSV, JSON, YAML files
    - Loading from databases (MongoDB, PostgreSQL)
    - Loading from time-series databases (InfluxDB)
    - Data validation and type conversion
    - Time-based filtering and aggregation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the process data loader.
        
        Args:
            config: Configuration dictionary with data source settings
        """
        self.config = config or {}
        self.supported_formats = ['csv', 'json', 'yaml', 'parquet']
        self.process_parameters = [
            'laser_power', 'scan_speed', 'hatch_spacing', 'layer_thickness',
            'preheating_temperature', 'build_chamber_temperature', 'atmosphere_pressure',
            'laser_spot_size', 'scan_pattern', 'support_structure_type'
        ]
        
        logger.info("Initialized ProcessDataLoader")
    
    def load_from_file(self, file_path: Union[str, Path], 
                      file_format: Optional[str] = None) -> pd.DataFrame:
        """
        Load process data from a file.
        
        Args:
            file_path: Path to the data file
            file_format: File format (csv, json, yaml, parquet). If None, inferred from extension
            
        Returns:
            DataFrame with process data
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file format
        if file_format is None:
            file_format = file_path.suffix.lower().lstrip('.')
        
        if file_format not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        try:
            if file_format == 'csv':
                return self._load_csv(file_path)
            elif file_format == 'json':
                return self._load_json(file_path)
            elif file_format == 'yaml':
                return self._load_yaml(file_path)
            elif file_format == 'parquet':
                return self._load_parquet(file_path)
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            raise
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        return self._validate_and_convert_process_data(df)
    
    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'process_data' in data:
                df = pd.DataFrame(data['process_data'])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("Invalid JSON structure")
        
        return self._validate_and_convert_process_data(df)
    
    def _load_yaml(self, file_path: Path) -> pd.DataFrame:
        """Load data from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle different YAML structures
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'process_data' in data:
                df = pd.DataFrame(data['process_data'])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("Invalid YAML structure")
        
        return self._validate_and_convert_process_data(df)
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load data from Parquet file."""
        df = pd.read_parquet(file_path)
        return self._validate_and_convert_process_data(df)
    
    def _validate_and_convert_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and convert process data to appropriate types.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Validated and converted DataFrame
        """
        # Ensure required columns exist
        missing_columns = []
        for param in self.process_parameters:
            if param not in df.columns:
                missing_columns.append(param)
        
        if missing_columns:
            logger.warning(f"Missing process parameters: {missing_columns}")
        
        # Convert data types
        type_conversions = {
            'laser_power': 'float64',
            'scan_speed': 'float64',
            'hatch_spacing': 'float64',
            'layer_thickness': 'float64',
            'preheating_temperature': 'float64',
            'build_chamber_temperature': 'float64',
            'atmosphere_pressure': 'float64',
            'laser_spot_size': 'float64',
            'scan_pattern': 'string',
            'support_structure_type': 'string'
        }
        
        for column, dtype in type_conversions.items():
            if column in df.columns:
                try:
                    df[column] = df[column].astype(dtype)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert {column} to {dtype}: {e}")
        
        # Handle timestamp columns
        timestamp_columns = ['timestamp', 'created_at', 'updated_at', 'build_time']
        for col in timestamp_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert {col} to datetime: {e}")
        
        # Validate parameter ranges
        self._validate_parameter_ranges(df)
        
        return df
    
    def _validate_parameter_ranges(self, df: pd.DataFrame):
        """
        Validate that process parameters are within acceptable ranges.
        
        Args:
            df: DataFrame to validate
        """
        parameter_ranges = {
            'laser_power': (0, 1000),  # Watts
            'scan_speed': (0, 5000),   # mm/s
            'hatch_spacing': (0, 1),   # mm
            'layer_thickness': (0, 0.5),  # mm
            'preheating_temperature': (0, 500),  # °C
            'build_chamber_temperature': (0, 200),  # °C
            'atmosphere_pressure': (0, 2),  # bar
            'laser_spot_size': (0, 1),  # mm
        }
        
        for param, (min_val, max_val) in parameter_ranges.items():
            if param in df.columns:
                invalid_count = ((df[param] < min_val) | (df[param] > max_val)).sum()
                if invalid_count > 0:
                    logger.warning(f"{invalid_count} values of {param} are outside range [{min_val}, {max_val}]")
    
    def load_from_database(self, connection_string: str, 
                          query: str, 
                          database_type: str = 'mongodb') -> pd.DataFrame:
        """
        Load process data from a database.
        
        Args:
            connection_string: Database connection string
            query: Query to execute
            database_type: Type of database (mongodb, postgresql, influxdb)
            
        Returns:
            DataFrame with process data
        """
        try:
            if database_type.lower() == 'mongodb':
                return self._load_from_mongodb(connection_string, query)
            elif database_type.lower() == 'postgresql':
                return self._load_from_postgresql(connection_string, query)
            elif database_type.lower() == 'influxdb':
                return self._load_from_influxdb(connection_string, query)
            else:
                raise ValueError(f"Unsupported database type: {database_type}")
        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            raise
    
    def _load_from_mongodb(self, connection_string: str, query: str) -> pd.DataFrame:
        """Load data from MongoDB."""
        try:
            from pymongo import MongoClient
            import json
            
            client = MongoClient(connection_string)
            db_name = connection_string.split('/')[-1]
            db = client[db_name]
            
            # Parse query (assuming it's a JSON string)
            query_dict = json.loads(query) if isinstance(query, str) else query
            
            collection_name = query_dict.get('collection', 'process_data')
            collection = db[collection_name]
            
            # Execute query
            cursor = collection.find(query_dict.get('filter', {}))
            data = list(cursor)
            
            if not data:
                logger.warning("No data found in MongoDB collection")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Remove MongoDB _id field if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            return self._validate_and_convert_process_data(df)
            
        except ImportError:
            raise ImportError("pymongo is required for MongoDB support. Install with: pip install pymongo")
    
    def _load_from_postgresql(self, connection_string: str, query: str) -> pd.DataFrame:
        """Load data from PostgreSQL."""
        try:
            import psycopg2
            from sqlalchemy import create_engine
            
            engine = create_engine(connection_string)
            df = pd.read_sql_query(query, engine)
            
            return self._validate_and_convert_process_data(df)
            
        except ImportError:
            raise ImportError("psycopg2 and sqlalchemy are required for PostgreSQL support. Install with: pip install psycopg2 sqlalchemy")
    
    def _load_from_influxdb(self, connection_string: str, query: str) -> pd.DataFrame:
        """Load data from InfluxDB."""
        try:
            from influxdb import InfluxDBClient
            
            # Parse connection string
            parts = connection_string.split('://')
            if len(parts) != 2:
                raise ValueError("Invalid InfluxDB connection string format")
            
            protocol, rest = parts
            if '@' in rest:
                auth, host_port = rest.split('@')
                username, password = auth.split(':')
            else:
                username, password = None, None
                host_port = rest
            
            if ':' in host_port:
                host, port = host_port.split(':')
                port = int(port)
            else:
                host = host_port
                port = 8086
            
            client = InfluxDBClient(host=host, port=port, username=username, password=password)
            
            # Execute query
            result = client.query(query)
            
            if not result:
                logger.warning("No data found in InfluxDB")
                return pd.DataFrame()
            
            # Convert to DataFrame
            points = list(result.get_points())
            df = pd.DataFrame(points)
            
            return self._validate_and_convert_process_data(df)
            
        except ImportError:
            raise ImportError("influxdb is required for InfluxDB support. Install with: pip install influxdb")
    
    def load_time_series_data(self, start_time: datetime, 
                             end_time: datetime,
                             parameters: Optional[List[str]] = None,
                             aggregation: str = 'mean',
                             interval: str = '1min') -> pd.DataFrame:
        """
        Load time-series process data.
        
        Args:
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            parameters: List of parameters to retrieve (None for all)
            aggregation: Aggregation method (mean, sum, min, max, count)
            interval: Time interval for aggregation
            
        Returns:
            DataFrame with time-series process data
        """
        if parameters is None:
            parameters = self.process_parameters
        
        # This would typically connect to a time-series database
        # For now, return a placeholder DataFrame
        time_range = pd.date_range(start=start_time, end=end_time, freq=interval)
        
        data = {
            'timestamp': time_range,
        }
        
        for param in parameters:
            # Generate sample data (in real implementation, this would come from database)
            data[param] = np.random.normal(100, 20, len(time_range))
        
        df = pd.DataFrame(data)
        return self._validate_and_convert_process_data(df)
    
    def load_batch_data(self, batch_ids: List[str]) -> pd.DataFrame:
        """
        Load process data for specific batches.
        
        Args:
            batch_ids: List of batch identifiers
            
        Returns:
            DataFrame with batch process data
        """
        # This would typically query a database for batch-specific data
        # For now, return a placeholder DataFrame
        data = []
        
        for batch_id in batch_ids:
            batch_data = {
                'batch_id': batch_id,
                'laser_power': np.random.uniform(200, 600),
                'scan_speed': np.random.uniform(500, 2000),
                'hatch_spacing': np.random.uniform(0.05, 0.2),
                'layer_thickness': np.random.uniform(0.02, 0.1),
                'preheating_temperature': np.random.uniform(50, 200),
                'build_chamber_temperature': np.random.uniform(20, 100),
                'atmosphere_pressure': np.random.uniform(0.5, 1.5),
                'laser_spot_size': np.random.uniform(0.05, 0.2),
                'scan_pattern': np.random.choice(['zigzag', 'contour', 'spiral']),
                'support_structure_type': np.random.choice(['tree', 'block', 'none']),
                'created_at': datetime.now()
            }
            data.append(batch_data)
        
        df = pd.DataFrame(data)
        return self._validate_and_convert_process_data(df)
    
    def load_material_specific_data(self, material_type: str) -> pd.DataFrame:
        """
        Load process data for a specific material type.
        
        Args:
            material_type: Type of material (e.g., 'Ti-6Al-4V', 'Inconel 718')
            
        Returns:
            DataFrame with material-specific process data
        """
        # This would typically query a database for material-specific data
        # For now, return a placeholder DataFrame with material-specific ranges
        
        material_ranges = {
            'Ti-6Al-4V': {
                'laser_power': (300, 500),
                'scan_speed': (800, 1500),
                'hatch_spacing': (0.08, 0.15),
                'layer_thickness': (0.03, 0.08),
                'preheating_temperature': (100, 200)
            },
            'Inconel 718': {
                'laser_power': (400, 600),
                'scan_speed': (600, 1200),
                'hatch_spacing': (0.06, 0.12),
                'layer_thickness': (0.02, 0.06),
                'preheating_temperature': (150, 250)
            },
            '316L': {
                'laser_power': (350, 550),
                'scan_speed': (700, 1400),
                'hatch_spacing': (0.07, 0.14),
                'layer_thickness': (0.025, 0.07),
                'preheating_temperature': (80, 180)
            }
        }
        
        if material_type not in material_ranges:
            logger.warning(f"Unknown material type: {material_type}")
            material_type = 'Ti-6Al-4V'  # Default
        
        ranges = material_ranges[material_type]
        
        # Generate sample data within material-specific ranges
        data = {
            'material_type': [material_type] * 100,
            'laser_power': np.random.uniform(ranges['laser_power'][0], ranges['laser_power'][1], 100),
            'scan_speed': np.random.uniform(ranges['scan_speed'][0], ranges['scan_speed'][1], 100),
            'hatch_spacing': np.random.uniform(ranges['hatch_spacing'][0], ranges['hatch_spacing'][1], 100),
            'layer_thickness': np.random.uniform(ranges['layer_thickness'][0], ranges['layer_thickness'][1], 100),
            'preheating_temperature': np.random.uniform(ranges['preheating_temperature'][0], ranges['preheating_temperature'][1], 100),
            'build_chamber_temperature': np.random.uniform(20, 100, 100),
            'atmosphere_pressure': np.random.uniform(0.5, 1.5, 100),
            'laser_spot_size': np.random.uniform(0.05, 0.2, 100),
            'scan_pattern': np.random.choice(['zigzag', 'contour', 'spiral'], 100),
            'support_structure_type': np.random.choice(['tree', 'block', 'none'], 100),
            'created_at': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(100)]
        }
        
        df = pd.DataFrame(data)
        return self._validate_and_convert_process_data(df)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for process data.
        
        Args:
            df: DataFrame with process data
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns summary
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            summary['numeric_summary'] = df[numeric_columns].describe().to_dict()
        
        # Categorical columns summary
        categorical_columns = df.select_dtypes(include=['object', 'string']).columns
        for col in categorical_columns:
            summary['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].value_counts().head().to_dict()
            }
        
        # Time range if timestamp column exists
        timestamp_columns = ['timestamp', 'created_at', 'updated_at', 'build_time']
        for col in timestamp_columns:
            if col in df.columns:
                summary['time_range'] = {
                    'start': df[col].min(),
                    'end': df[col].max(),
                    'duration': df[col].max() - df[col].min()
                }
                break
        
        return summary
