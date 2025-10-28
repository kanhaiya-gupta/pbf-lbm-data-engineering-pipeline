"""
MongoDB Data Extractor for Knowledge Graph

This module extracts unstructured data from MongoDB to build knowledge graph nodes
and relationships for PBF-LB/M manufacturing processes.
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.data_pipeline.config.mongodb_config import get_mongodb_config
from src.data_pipeline.storage.operational.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class MongoDbExtractor:
    """
    Extracts unstructured data from MongoDB for knowledge graph construction.
    
    Focuses on PBF-LB/M manufacturing unstructured entities:
    - Images, 3D models, build files, sensor data, logs
    - Metadata, relationships, file references
    """
    
    def __init__(self):
        """Initialize MongoDB extractor."""
        self.config = get_mongodb_config()
        # Convert config to dictionary for MongoDBClient
        config_dict = {
            'host': self.config.host,
            'port': self.config.port,
            'database': self.config.database,
            'username': self.config.username,
            'password': self.config.password,
            'auth_source': self.config.auth_source
        }
        self.client = MongoDBClient(config_dict)
        
    def connect(self) -> bool:
        """Connect to MongoDB database."""
        try:
            if self.client.connect():
                logger.info("✅ Connected to MongoDB for knowledge graph extraction")
                return True
            else:
                logger.error("❌ Failed to connect to MongoDB")
                return False
        except Exception as e:
            logger.error(f"❌ MongoDB connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client.connected:
            self.client.disconnect()
            logger.info("Disconnected from MongoDB")
    
    def extract_process_images(self) -> List[Dict[str, Any]]:
        """
        Extract process images for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Process image data for graph nodes
        """
        try:
            if not self.client.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self.client.get_collection('process_images')
            cursor = collection.find({}, {
                '_id': 1,
                'process_id': 1,
                'image_type': 1,
                'image_format': 1,
                'image_dimensions': 1,
                'file_size': 1,
                'metadata': 1,
                'tags': 1,
                'created_at': 1
            })
            
            images = []
            for doc in cursor:
                image_data = {
                    'node_type': 'ProcessImage',
                    'document_id': str(doc['_id']),
                    'process_id': doc.get('process_id'),
                    'image_type': doc.get('image_type'),
                    'image_format': doc.get('image_format'),
                    'image_dimensions': doc.get('image_dimensions'),
                    'file_size': doc.get('file_size'),
                    'metadata': doc.get('metadata', {}),
                    'tags': doc.get('tags', []),
                    'created_at': doc.get('created_at'),
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                images.append(image_data)
            
            logger.info(f"📊 Extracted {len(images)} process images from MongoDB")
            return images
            
        except Exception as e:
            logger.error(f"❌ Failed to extract process images: {e}")
            return []
    
    def extract_ct_scan_images(self) -> List[Dict[str, Any]]:
        """
        Extract CT scan images for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: CT scan image data for graph nodes
        """
        try:
            if not self.client.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self.client.get_collection('ct_scan_images')
            cursor = collection.find({}, {
                '_id': 1,
                'part_id': 1,
                'scan_type': 1,
                'image_format': 1,
                'scan_parameters': 1,
                'file_size': 1,
                'metadata': 1,
                'created_at': 1
            })
            
            ct_scans = []
            for doc in cursor:
                ct_data = {
                    'node_type': 'CTScanImage',
                    'document_id': str(doc['_id']),
                    'part_id': doc.get('part_id'),
                    'scan_type': doc.get('scan_type'),
                    'image_format': doc.get('image_format'),
                    'scan_parameters': doc.get('scan_parameters', {}),
                    'file_size': doc.get('file_size'),
                    'metadata': doc.get('metadata', {}),
                    'created_at': doc.get('created_at'),
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                ct_scans.append(ct_data)
            
            logger.info(f"📊 Extracted {len(ct_scans)} CT scan images from MongoDB")
            return ct_scans
            
        except Exception as e:
            logger.error(f"❌ Failed to extract CT scan images: {e}")
            return []
    
    def extract_powder_bed_images(self) -> List[Dict[str, Any]]:
        """
        Extract powder bed images for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Powder bed image data for graph nodes
        """
        try:
            if not self.client.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self.client.get_collection('powder_bed_images')
            cursor = collection.find({}, {
                '_id': 1,
                'process_id': 1,
                'layer_number': 1,
                'image_type': 1,
                'image_format': 1,
                'image_dimensions': 1,
                'file_size': 1,
                'metadata': 1,
                'created_at': 1
            })
            
            powder_bed_images = []
            for doc in cursor:
                image_data = {
                    'node_type': 'PowderBedImage',
                    'document_id': str(doc['_id']),
                    'process_id': doc.get('process_id'),
                    'layer_number': doc.get('layer_number'),
                    'image_type': doc.get('image_type'),
                    'image_format': doc.get('image_format'),
                    'image_dimensions': doc.get('image_dimensions'),
                    'file_size': doc.get('file_size'),
                    'metadata': doc.get('metadata', {}),
                    'created_at': doc.get('created_at'),
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                powder_bed_images.append(image_data)
            
            logger.info(f"📊 Extracted {len(powder_bed_images)} powder bed images from MongoDB")
            return powder_bed_images
            
        except Exception as e:
            logger.error(f"❌ Failed to extract powder bed images: {e}")
            return []
    
    def extract_machine_build_files(self) -> List[Dict[str, Any]]:
        """
        Extract machine build files for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Machine build file data for graph nodes
        """
        try:
            if not self.client.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self.client.get_collection('machine_build_files')
            cursor = collection.find({}, {
                '_id': 1,
                'build_id': 1,
                'machine_id': 1,
                'file_type': 1,
                'file_format': 1,
                'file_size': 1,
                'metadata': 1,
                'created_at': 1
            })
            
            build_files = []
            for doc in cursor:
                file_data = {
                    'node_type': 'MachineBuildFile',
                    'document_id': str(doc['_id']),
                    'build_id': doc.get('build_id'),
                    'machine_id': doc.get('machine_id'),
                    'file_type': doc.get('file_type'),
                    'file_format': doc.get('file_format'),
                    'file_size': doc.get('file_size'),
                    'metadata': doc.get('metadata', {}),
                    'created_at': doc.get('created_at'),
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                build_files.append(file_data)
            
            logger.info(f"📊 Extracted {len(build_files)} machine build files from MongoDB")
            return build_files
            
        except Exception as e:
            logger.error(f"❌ Failed to extract machine build files: {e}")
            return []
    
    def extract_3d_model_files(self) -> List[Dict[str, Any]]:
        """
        Extract 3D model files for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: 3D model file data for graph nodes
        """
        try:
            if not self.client.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self.client.get_collection('model_3d_files')
            cursor = collection.find({}, {
                '_id': 1,
                'part_id': 1,
                'build_id': 1,
                'machine_id': 1,
                'file_type': 1,
                'file_format': 1,
                'file_size': 1,
                'metadata': 1,
                'created_at': 1
            })
            
            model_files = []
            for doc in cursor:
                file_data = {
                    'node_type': 'Model3DFile',
                    'document_id': str(doc['_id']),
                    'part_id': doc.get('part_id'),
                    'build_id': doc.get('build_id'),
                    'machine_id': doc.get('machine_id'),
                    'file_type': doc.get('file_type'),
                    'file_format': doc.get('file_format'),
                    'file_size': doc.get('file_size'),
                    'metadata': doc.get('metadata', {}),
                    'created_at': doc.get('created_at'),
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                model_files.append(file_data)
            
            logger.info(f"📊 Extracted {len(model_files)} 3D model files from MongoDB")
            return model_files
            
        except Exception as e:
            logger.error(f"❌ Failed to extract 3D model files: {e}")
            return []
    
    def extract_raw_sensor_data(self) -> List[Dict[str, Any]]:
        """
        Extract raw sensor data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Raw sensor data for graph nodes
        """
        try:
            if not self.client.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self.client.get_collection('raw_sensor_data')
            cursor = collection.find({}, {
                '_id': 1,
                'sensor_id': 1,
                'process_id': 1,
                'sensor_type': 1,
                'data_format': 1,
                'file_size': 1,
                'metadata': 1,
                'created_at': 1
            })
            
            sensor_data = []
            for doc in cursor:
                data = {
                    'node_type': 'RawSensorData',
                    'document_id': str(doc['_id']),
                    'sensor_id': doc.get('sensor_id'),
                    'process_id': doc.get('process_id'),
                    'sensor_type': doc.get('sensor_type'),
                    'data_format': doc.get('data_format'),
                    'file_size': doc.get('file_size'),
                    'metadata': doc.get('metadata', {}),
                    'created_at': doc.get('created_at'),
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                sensor_data.append(data)
            
            logger.info(f"📊 Extracted {len(sensor_data)} raw sensor data files from MongoDB")
            return sensor_data
            
        except Exception as e:
            logger.error(f"❌ Failed to extract raw sensor data: {e}")
            return []
    
    def extract_process_logs(self) -> List[Dict[str, Any]]:
        """
        Extract process logs for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Process log data for graph nodes
        """
        try:
            if not self.client.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self.client.get_collection('process_logs')
            cursor = collection.find({}, {
                '_id': 1,
                'process_id': 1,
                'log_type': 1,
                'log_level': 1,
                'message': 1,
                'metadata': 1,
                'created_at': 1
            })
            
            logs = []
            for doc in cursor:
                log_data = {
                    'node_type': 'ProcessLog',
                    'document_id': str(doc['_id']),
                    'process_id': doc.get('process_id'),
                    'log_type': doc.get('log_type'),
                    'log_level': doc.get('log_level'),
                    'message': doc.get('message'),
                    'metadata': doc.get('metadata', {}),
                    'created_at': doc.get('created_at'),
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                logs.append(log_data)
            
            logger.info(f"📊 Extracted {len(logs)} process logs from MongoDB")
            return logs
            
        except Exception as e:
            logger.error(f"❌ Failed to extract process logs: {e}")
            return []
    
    def extract_machine_configurations(self) -> List[Dict[str, Any]]:
        """
        Extract machine configurations for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Machine configuration data for graph nodes
        """
        try:
            if not self.client.connected:
                raise RuntimeError("Not connected to MongoDB")
            
            collection = self.client.get_collection('machine_configurations')
            cursor = collection.find({}, {
                '_id': 1,
                'machine_id': 1,
                'process_id': 1,
                'build_id': 1,
                'config_type': 1,
                'config_data': 1,
                'file_size': 1,
                'created_at': 1
            })
            
            configs = []
            for doc in cursor:
                config_data = {
                    'node_type': 'MachineConfig',
                    'document_id': str(doc['_id']),
                    'machine_id': doc.get('machine_id'),
                    'process_id': doc.get('process_id'),
                    'build_id': doc.get('build_id'),
                    'config_type': doc.get('config_type'),
                    'config_data': doc.get('config_data', {}),
                    'file_size': doc.get('file_size'),
                    'created_at': doc.get('created_at'),
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                configs.append(config_data)
            
            logger.info(f"📊 Extracted {len(configs)} machine configurations from MongoDB")
            return configs
            
        except Exception as e:
            logger.error(f"❌ Failed to extract machine configurations: {e}")
            return []
    
    def extract_all_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract all data from MongoDB for knowledge graph construction.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: All extracted data organized by type
        """
        logger.info("🚀 Starting comprehensive MongoDB data extraction...")
        
        if not self.client.connected:
            if not self.connect():
                return {}
        
        try:
            extracted_data = {
                'process_images': self.extract_process_images(),
                'ct_scan_images': self.extract_ct_scan_images(),
                'powder_bed_images': self.extract_powder_bed_images(),
                'machine_build_files': self.extract_machine_build_files(),
                'model_3d_files': self.extract_3d_model_files(),
                'raw_sensor_data': self.extract_raw_sensor_data(),
                'process_logs': self.extract_process_logs(),
                'machine_configurations': self.extract_machine_configurations()
            }
            
            # Calculate totals
            total_documents = sum(len(data) for data in extracted_data.values())
            
            logger.info(f"✅ MongoDB extraction completed:")
            logger.info(f"   📊 Total Documents: {total_documents}")
            logger.info(f"   🖼️ Process Images: {len(extracted_data['process_images'])}")
            logger.info(f"   🔬 CT Scan Images: {len(extracted_data['ct_scan_images'])}")
            logger.info(f"   🏭 Powder Bed Images: {len(extracted_data['powder_bed_images'])}")
            logger.info(f"   📁 Machine Build Files: {len(extracted_data['machine_build_files'])}")
            logger.info(f"   🎯 3D Model Files: {len(extracted_data['model_3d_files'])}")
            logger.info(f"   📡 Raw Sensor Data: {len(extracted_data['raw_sensor_data'])}")
            logger.info(f"   📝 Process Logs: {len(extracted_data['process_logs'])}")
            logger.info(f"   ⚙️ Machine Configurations: {len(extracted_data['machine_configurations'])}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"❌ Failed to extract all MongoDB data: {e}")
            return {}
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data for extraction.
        
        Returns:
            Dict[str, Any]: Extraction summary
        """
        try:
            if not self.client.connected:
                if not self.connect():
                    return {}
            
            # Get collection counts
            collections = [
                'process_images', 'ct_scan_images', 'powder_bed_images',
                'machine_build_files', 'model_3d_files', 'raw_sensor_data',
                'process_logs', 'machine_configurations'
            ]
            counts = {}
            
            for collection_name in collections:
                try:
                    collection = self.client.get_collection(collection_name)
                    count = collection.count_documents({})
                    counts[collection_name] = count
                except Exception as e:
                    logger.warning(f"Could not get count for collection {collection_name}: {e}")
                    counts[collection_name] = 0
            
            summary = {
                'database': 'MongoDB',
                'connection_status': 'Connected' if self.client.connected else 'Disconnected',
                'collection_counts': counts,
                'total_documents': sum(counts.values()),
                'extraction_timestamp': datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get extraction summary: {e}")
            return {}
    
    def close_connection(self) -> None:
        """Close the MongoDB connection."""
        try:
            if self.client and self.client.connected:
                self.client.close_connection()
                self.connected = False
                logger.info("✅ MongoDB connection closed")
        except Exception as e:
            logger.error(f"❌ Failed to close MongoDB connection: {e}")
