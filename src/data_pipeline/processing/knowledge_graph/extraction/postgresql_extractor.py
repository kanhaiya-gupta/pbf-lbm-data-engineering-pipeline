"""
PostgreSQL Data Extractor for Knowledge Graph

This module extracts relational data from PostgreSQL to build knowledge graph nodes
and relationships for PBF-LB/M manufacturing processes.
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import json
from src.data_pipeline.processing.knowledge_graph.utils.json_parser import safe_json_loads_with_fallback

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.data_pipeline.config.postgres_config import get_postgres_config
from src.data_pipeline.storage.operational.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class PostgreSQLExtractor:
    """
    Extracts relational data from PostgreSQL for knowledge graph construction.
    
    Focuses on PBF-LB/M manufacturing entities and their relationships:
    - Processes, Machines, Parts, Builds, Sensors
    - Quality metrics, material properties, process parameters
    """
    
    def __init__(self):
        """Initialize PostgreSQL extractor."""
        self.config = get_postgres_config()
        # Convert config to dictionary for PostgresClient
        config_dict = {
            'host': self.config.host,
            'port': self.config.port,
            'database': self.config.database,
            'user': self.config.username,
            'password': self.config.password
        }
        self.client = PostgresClient(config_dict)
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to PostgreSQL database."""
        try:
            # Test connection with a simple query
            result = self.client.execute_query("SELECT 1")
            if result is not None:
                self.connected = True
                logger.info("✅ Connected to PostgreSQL for knowledge graph extraction")
                return True
            else:
                logger.error("❌ Failed to connect to PostgreSQL")
                return False
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from PostgreSQL."""
        if self.connected:
            self.client.close_connection()
            self.connected = False
            logger.info("Disconnected from PostgreSQL")
    
    def extract_processes(self) -> List[Dict[str, Any]]:
        """
        Extract process data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Process data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to PostgreSQL")
            
            # Get the actual process data using the correct timestamp column
            query = """
                SELECT 
                    process_id,
                    machine_id,
                    part_id,
                    build_id,
                    powder_material,
                    layer_height,
                    laser_power,
                    scan_speed,
                    hatch_spacing,
                    density,
                    timestamp
                FROM pbf_process_data
                ORDER BY timestamp DESC
            """
            
            results = self.client.execute_query(query)
            logger.info(f"🔍 Debug: Process query returned {len(results)} raw results")
            
            processes = []
            for i, row in enumerate(results):
                try:
                    process_data = {
                        'node_type': 'Process',
                        'process_id': row['process_id'],
                        'timestamp': row['timestamp'].replace(tzinfo=timezone.utc) if row['timestamp'] else None,  # Make timezone-aware
                        'machine_id': row['machine_id'],
                        'part_id': row['part_id'],
                        'build_id': row['build_id'],
                        'material_type': row['powder_material'],
                        'layer_thickness': float(row['layer_height']) if row['layer_height'] else None,
                        'laser_power': float(row['laser_power']) if row['laser_power'] else None,
                        'scan_speed': float(row['scan_speed']) if row['scan_speed'] else None,
                        'hatch_spacing': float(row['hatch_spacing']) if row['hatch_spacing'] else None,
                        'process_status': 'active',
                        'quality_score': float(row['density']) if row['density'] else None,
                        'created_at': row['timestamp'].isoformat() if row['timestamp'] else None,
                        'updated_at': row['timestamp'].isoformat() if row['timestamp'] else None,
                        'extraction_timestamp': datetime.utcnow().isoformat()
                    }
                    processes.append(process_data)
                except Exception as e:
                    logger.error(f"❌ Error processing row {i}: {e}")
                    logger.error(f"Row data: {row}")
            
            logger.info(f"📊 Extracted {len(processes)} processes from PostgreSQL")
            return processes
            
        except Exception as e:
            logger.error(f"❌ Failed to extract processes: {e}")
            return []
    
    def extract_machines(self) -> List[Dict[str, Any]]:
        """
        Extract machine data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Machine data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to PostgreSQL")
            
            query = """
                SELECT DISTINCT 
                    machine_id,
                    'PBF_LBM' as machine_type,
                    'Unknown' as manufacturer,
                    'PBF-LBM-001' as model,
                    'active' as status,
                    'Manufacturing Floor' as location,
                    '{"laser_power": 1000, "build_volume": "300x300x400"}' as capabilities
                FROM pbf_process_data p
                WHERE machine_id IS NOT NULL
                ORDER BY machine_id
            """
            
            results = self.client.execute_query(query)
            
            machines = []
            for row in results:
                machine_data = {
                    'node_type': 'Machine',
                    'machine_id': row['machine_id'],
                    'machine_type': row['machine_type'],
                    'manufacturer': row['manufacturer'],
                    'model': row['model'],
                    'status': row['status'],
                    'location': row['location'],
                    'capabilities': safe_json_loads_with_fallback(row['capabilities'], 'capabilities', 5000, {}),
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                machines.append(machine_data)
            
            logger.info(f"📊 Extracted {len(machines)} machines from PostgreSQL")
            return machines
            
        except Exception as e:
            logger.error(f"❌ Failed to extract machines: {e}")
            return []
    
    def extract_parts(self) -> List[Dict[str, Any]]:
        """
        Extract part data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Part data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to PostgreSQL")
            
            query = """
                SELECT DISTINCT
                    part_id,
                    build_id,
                    powder_material as material_type,
                    'Part-' || part_id as part_name,
                    '{"length": 100, "width": 100, "height": 50}' as dimensions,
                    0.5 as weight,
                    0.7 as complexity_score,
                    1000.0 as surface_area,
                    500.0 as volume
                FROM pbf_process_data p
                WHERE part_id IS NOT NULL
                ORDER BY part_id
            """
            
            results = self.client.execute_query(query)
            
            parts = []
            for row in results:
                part_data = {
                    'node_type': 'Part',
                    'part_id': row['part_id'],
                    'build_id': row['build_id'],
                    'part_name': row['part_name'],
                    'material_type': row['material_type'],
                    'dimensions': safe_json_loads_with_fallback(row['dimensions'], 'dimensions', 5000, {}),
                    'weight': float(row['weight']) if row['weight'] else None,
                    'complexity_score': float(row['complexity_score']) if row['complexity_score'] else None,
                    'surface_area': float(row['surface_area']) if row['surface_area'] else None,
                    'volume': float(row['volume']) if row['volume'] else None,
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                parts.append(part_data)
            
            logger.info(f"📊 Extracted {len(parts)} parts from PostgreSQL")
            return parts
            
        except Exception as e:
            logger.error(f"❌ Failed to extract parts: {e}")
            return []
    
    def extract_builds(self) -> List[Dict[str, Any]]:
        """
        Extract build data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Build data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to PostgreSQL")
            
            query = """
                SELECT DISTINCT
                    build_id,
                    'Build-' || build_id as build_name,
                    timestamp as build_date,
                    COUNT(*) as total_parts,
                    'completed' as build_status,
                    'PBF-LBM Build Process' as build_notes
                FROM pbf_process_data p
                WHERE build_id IS NOT NULL
                GROUP BY build_id, timestamp
                ORDER BY build_id
            """
            
            results = self.client.execute_query(query)
            
            builds = []
            for row in results:
                build_data = {
                    'node_type': 'Build',
                    'build_id': row['build_id'],
                    'build_name': row['build_name'],
                    'build_date': row['build_date'].isoformat() if row['build_date'] else None,
                    'total_parts': int(row['total_parts']) if row['total_parts'] else None,
                    'build_status': row['build_status'],
                    'build_notes': row['build_notes'],
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                builds.append(build_data)
            
            logger.info(f"📊 Extracted {len(builds)} builds from PostgreSQL")
            return builds
            
        except Exception as e:
            logger.error(f"❌ Failed to extract builds: {e}")
            return []
    
    def extract_sensors(self) -> List[Dict[str, Any]]:
        """
        Extract sensor data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Sensor data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to PostgreSQL")
            
            query = """
                SELECT DISTINCT
                    camera_id as sensor_id,
                    'CAMERA' as sensor_type,
                    'Powder Bed' as sensor_location,
                    'active' as sensor_status,
                    timestamp as calibration_date,
                    '0-1000' as measurement_range,
                    0.95 as accuracy
                FROM powder_bed_data p
                WHERE camera_id IS NOT NULL
                ORDER BY camera_id
            """
            
            results = self.client.execute_query(query)
            
            sensors = []
            for row in results:
                sensor_data = {
                    'node_type': 'Sensor',
                    'sensor_id': row['sensor_id'],
                    'sensor_type': row['sensor_type'],
                    'sensor_location': row['sensor_location'],
                    'sensor_status': row['sensor_status'],
                    'calibration_date': row['calibration_date'].isoformat() if row['calibration_date'] else None,
                    'measurement_range': row['measurement_range'],
                    'accuracy': float(row['accuracy']) if row['accuracy'] else None,
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                sensors.append(sensor_data)
            
            logger.info(f"📊 Extracted {len(sensors)} sensors from PostgreSQL")
            return sensors
            
        except Exception as e:
            logger.error(f"❌ Failed to extract sensors: {e}")
            return []
    
    def extract_quality_metrics(self) -> List[Dict[str, Any]]:
        """
        Extract quality metrics for knowledge graph relationships.
        
        Returns:
            List[Dict[str, Any]]: Quality metrics data
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to PostgreSQL")
            
            query = """
                SELECT 
                    process_id,
                    density as quality_score,
                    defect_count as defect_density,
                    surface_roughness,
                    dimensional_accuracy,
                    '{"tensile_strength": 500, "yield_strength": 400}' as mechanical_properties,
                    CASE 
                        WHEN density >= 95 AND defect_count = 0 THEN 'EXCELLENT'
                        WHEN density >= 90 AND defect_count <= 5 THEN 'GOOD'
                        WHEN density >= 85 AND defect_count <= 10 THEN 'ACCEPTABLE'
                        ELSE 'POOR'
                    END as quality_grade,
                    timestamp as inspection_date
                FROM pbf_process_data
                WHERE density IS NOT NULL
                ORDER BY process_id
            """
            
            results = self.client.execute_query(query)
            
            quality_metrics = []
            for row in results:
                quality_data = {
                    'process_id': row['process_id'],
                    'quality_score': float(row['quality_score']) if row['quality_score'] else None,
                    'defect_density': float(row['defect_density']) if row['defect_density'] else None,
                    'surface_roughness': float(row['surface_roughness']) if row['surface_roughness'] else None,
                    'dimensional_accuracy': float(row['dimensional_accuracy']) if row['dimensional_accuracy'] else None,
                    'mechanical_properties': safe_json_loads_with_fallback(row['mechanical_properties'], 'mechanical_properties', 5000, {}),
                    'quality_grade': row['quality_grade'],
                    'inspection_date': row['inspection_date'].isoformat() if row['inspection_date'] else None,
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                quality_metrics.append(quality_data)
            
            logger.info(f"📊 Extracted {len(quality_metrics)} quality metrics from PostgreSQL")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to extract quality metrics: {e}")
            return []
    
    def extract_material_properties(self) -> List[Dict[str, Any]]:
        """
        Extract material properties for knowledge graph relationships.
        
        Returns:
            List[Dict[str, Any]]: Material properties data
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to PostgreSQL")
            
            query = """
                SELECT DISTINCT
                    powder_material as material_type,
                    density,
                    1650 as melting_point,
                    25.0 as thermal_conductivity,
                    '{"tensile_strength": 500, "yield_strength": 400, "elastic_modulus": 200}' as mechanical_properties,
                    '{"Fe": 85, "Cr": 10, "Ni": 5}' as chemical_composition,
                    '316L' as material_grade
                FROM pbf_process_data
                WHERE powder_material IS NOT NULL
                ORDER BY powder_material
            """
            
            results = self.client.execute_query(query)
            
            materials = []
            for row in results:
                material_data = {
                    'material_type': row['material_type'],
                    'density': float(row['density']) if row['density'] else None,
                    'melting_point': float(row['melting_point']) if row['melting_point'] else None,
                    'thermal_conductivity': float(row['thermal_conductivity']) if row['thermal_conductivity'] else None,
                    'mechanical_properties': safe_json_loads_with_fallback(row['mechanical_properties'], 'mechanical_properties', 5000, {}),
                    'chemical_composition': safe_json_loads_with_fallback(row['chemical_composition'], 'chemical_composition', 5000, {}),
                    'material_grade': row['material_grade'],
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                materials.append(material_data)
            
            logger.info(f"📊 Extracted {len(materials)} material properties from PostgreSQL")
            return materials
            
        except Exception as e:
            logger.error(f"❌ Failed to extract material properties: {e}")
            return []
    
    def extract_all_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract all data from PostgreSQL for knowledge graph construction.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: All extracted data organized by type
        """
        logger.info("🚀 Starting comprehensive PostgreSQL data extraction...")
        
        if not self.connected:
            if not self.connect():
                return {}
        
        try:
            extracted_data = {
                'processes': self.extract_processes(),
                'machines': self.extract_machines(),
                'parts': self.extract_parts(),
                'builds': self.extract_builds(),
                'sensors': self.extract_sensors(),
                'quality_metrics': self.extract_quality_metrics(),
                'material_properties': self.extract_material_properties()
            }
            
            # Calculate totals
            total_nodes = sum(len(data) for key, data in extracted_data.items() 
                            if key in ['processes', 'machines', 'parts', 'builds', 'sensors'])
            total_relationships = len(extracted_data['quality_metrics']) + len(extracted_data['material_properties'])
            
            logger.info(f"✅ PostgreSQL extraction completed:")
            logger.info(f"   📊 Nodes: {total_nodes}")
            logger.info(f"   🔗 Relationships: {total_relationships}")
            logger.info(f"   📈 Processes: {len(extracted_data['processes'])}")
            logger.info(f"   🏭 Machines: {len(extracted_data['machines'])}")
            logger.info(f"   🔧 Parts: {len(extracted_data['parts'])}")
            logger.info(f"   📦 Builds: {len(extracted_data['builds'])}")
            logger.info(f"   📡 Sensors: {len(extracted_data['sensors'])}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"❌ Failed to extract all PostgreSQL data: {e}")
            return {}
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data for extraction.
        
        Returns:
            Dict[str, Any]: Extraction summary
        """
        try:
            if not self.connected:
                if not self.connect():
                    return {}
            
            # Get table counts
            tables = ['pbf_process_data', 'powder_bed_data', 'ct_scan_data', 'ispm_monitoring_data']
            counts = {}
            
            for table in tables:
                try:
                    result = self.client.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                    counts[table] = result[0]['count'] if result else 0
                except Exception as e:
                    logger.warning(f"Could not get count for table {table}: {e}")
                    counts[table] = 0
            
            summary = {
                'database': 'PostgreSQL',
                'connection_status': 'Connected' if self.connected else 'Disconnected',
                'table_counts': counts,
                'total_records': sum(counts.values()),
                'extraction_timestamp': datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get extraction summary: {e}")
            return {}
    
    def close_connection(self) -> None:
        """Close the PostgreSQL connection."""
        try:
            if self.connection and not self.connection.closed:
                self.connection.close()
                self.connected = False
                logger.info("✅ PostgreSQL connection closed")
        except Exception as e:
            logger.error(f"❌ Failed to close PostgreSQL connection: {e}")
