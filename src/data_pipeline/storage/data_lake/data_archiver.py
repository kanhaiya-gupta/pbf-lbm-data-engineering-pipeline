"""
Data Archiver for PBF-LB/M Data Pipeline

This module provides data archiving capabilities to S3.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from src.data_pipeline.storage.data_lake.s3_client import S3Client

logger = logging.getLogger(__name__)


class DataArchiver:
    """
    Data archiver for moving data to S3 for long-term storage.
    """
    
    def __init__(self, s3_client: Optional[S3Client] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data archiver.
        
        Args:
            s3_client: Optional S3 client instance
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.s3_client = s3_client or S3Client()
        self.archive_bucket = self.config.get('archive_bucket', 'pbf-lbm-archive')
        self.retention_days = self.config.get('retention_days', 365)
        self.compression_enabled = self.config.get('compression_enabled', True)
        self.archive_stats = {
            "files_archived": 0,
            "bytes_archived": 0,
            "archives_created": 0,
            "start_time": datetime.now()
        }
    
    def archive_file(self, local_path: str, archive_path: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Archive a single file to S3.
        
        Args:
            local_path: Local file path to archive
            archive_path: Archive path in S3
            metadata: Optional metadata for the archived file
            
        Returns:
            bool: True if archiving successful, False otherwise
        """
        try:
            if not os.path.exists(local_path):
                logger.error(f"File not found: {local_path}")
                return False
            
            # Get file size
            file_size = os.path.getsize(local_path)
            
            # Prepare metadata
            archive_metadata = {
                "original_path": local_path,
                "archive_timestamp": datetime.now().isoformat(),
                "file_size": str(file_size),
                "retention_days": str(self.retention_days)
            }
            
            if metadata:
                archive_metadata.update(metadata)
            
            # Upload to S3
            success = self.s3_client.upload_file(
                local_path=local_path,
                s3_key=archive_path,
                bucket=self.archive_bucket,
                metadata=archive_metadata
            )
            
            if success:
                self.archive_stats["files_archived"] += 1
                self.archive_stats["bytes_archived"] += file_size
                logger.info(f"Archived file: {local_path} -> s3://{self.archive_bucket}/{archive_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error archiving file {local_path}: {e}")
            return False
    
    def archive_directory(self, local_dir: str, archive_prefix: str, file_pattern: str = "*", recursive: bool = True) -> int:
        """
        Archive all files in a directory to S3.
        
        Args:
            local_dir: Local directory path to archive
            archive_prefix: Archive prefix in S3
            file_pattern: File pattern to match
            recursive: Whether to archive recursively
            
        Returns:
            int: Number of files archived successfully
        """
        logger.info(f"Starting directory archive: {local_dir} -> s3://{self.archive_bucket}/{archive_prefix}")
        
        archived_count = 0
        directory = Path(local_dir)
        
        if not directory.exists():
            logger.error(f"Directory not found: {local_dir}")
            return 0
        
        try:
            # Find files to archive
            if recursive:
                pattern = f"**/{file_pattern}"
            else:
                pattern = file_pattern
            
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    # Create archive path
                    relative_path = file_path.relative_to(directory)
                    archive_path = f"{archive_prefix}/{relative_path}"
                    
                    # Archive file
                    if self.archive_file(str(file_path), archive_path):
                        archived_count += 1
            
            logger.info(f"Archived {archived_count} files from directory {local_dir}")
            
        except Exception as e:
            logger.error(f"Error archiving directory {local_dir}: {e}")
        
        return archived_count
    
    def archive_data(self, data: Union[str, bytes], archive_path: str, content_type: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Archive data directly to S3.
        
        Args:
            data: Data to archive (string or bytes)
            archive_path: Archive path in S3
            content_type: Content type of the data
            metadata: Optional metadata for the archived data
            
        Returns:
            bool: True if archiving successful, False otherwise
        """
        try:
            # Prepare metadata
            archive_metadata = {
                "archive_timestamp": datetime.now().isoformat(),
                "retention_days": str(self.retention_days),
                "data_type": "direct_data"
            }
            
            if metadata:
                archive_metadata.update(metadata)
            
            # Upload to S3
            success = self.s3_client.upload_data(
                data=data,
                s3_key=archive_path,
                bucket=self.archive_bucket,
                content_type=content_type
            )
            
            if success:
                self.archive_stats["files_archived"] += 1
                if isinstance(data, bytes):
                    self.archive_stats["bytes_archived"] += len(data)
                else:
                    self.archive_stats["bytes_archived"] += len(data.encode('utf-8'))
                logger.info(f"Archived data to s3://{self.archive_bucket}/{archive_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error archiving data to {archive_path}: {e}")
            return False
    
    def create_archive_manifest(self, archive_paths: List[str], manifest_path: str) -> bool:
        """
        Create an archive manifest file.
        
        Args:
            archive_paths: List of archive paths
            manifest_path: Path for the manifest file
            
        Returns:
            bool: True if manifest creation successful, False otherwise
        """
        try:
            manifest = {
                "manifest_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "archive_count": len(archive_paths),
                "archives": []
            }
            
            for archive_path in archive_paths:
                # Get metadata for each archive
                metadata = self.s3_client.get_object_metadata(archive_path, self.archive_bucket)
                if metadata:
                    manifest["archives"].append({
                        "path": archive_path,
                        "size": metadata["size"],
                        "last_modified": metadata["last_modified"].isoformat(),
                        "etag": metadata["etag"]
                    })
            
            # Upload manifest to S3
            manifest_data = json.dumps(manifest, indent=2)
            success = self.s3_client.upload_data(
                data=manifest_data,
                s3_key=manifest_path,
                bucket=self.archive_bucket,
                content_type="application/json"
            )
            
            if success:
                logger.info(f"Created archive manifest: s3://{self.archive_bucket}/{manifest_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error creating archive manifest: {e}")
            return False
    
    def restore_file(self, archive_path: str, local_path: str) -> bool:
        """
        Restore a file from archive.
        
        Args:
            archive_path: Archive path in S3
            local_path: Local path to restore to
            
        Returns:
            bool: True if restoration successful, False otherwise
        """
        try:
            # Create local directory if it doesn't exist
            local_dir = Path(local_path).parent
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Download from S3
            success = self.s3_client.download_file(
                s3_key=archive_path,
                local_path=local_path,
                bucket=self.archive_bucket
            )
            
            if success:
                logger.info(f"Restored file: s3://{self.archive_bucket}/{archive_path} -> {local_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error restoring file {archive_path}: {e}")
            return False
    
    def restore_directory(self, archive_prefix: str, local_dir: str) -> int:
        """
        Restore all files in an archive prefix to a local directory.
        
        Args:
            archive_prefix: Archive prefix in S3
            local_dir: Local directory to restore to
            
        Returns:
            int: Number of files restored successfully
        """
        logger.info(f"Starting directory restore: s3://{self.archive_bucket}/{archive_prefix} -> {local_dir}")
        
        restored_count = 0
        
        try:
            # List objects in archive prefix
            objects = self.s3_client.list_objects(prefix=archive_prefix, bucket=self.archive_bucket)
            
            for obj in objects:
                archive_path = obj["key"]
                relative_path = archive_path[len(archive_prefix):].lstrip("/")
                local_path = os.path.join(local_dir, relative_path)
                
                # Restore file
                if self.restore_file(archive_path, local_path):
                    restored_count += 1
            
            logger.info(f"Restored {restored_count} files from archive prefix {archive_prefix}")
            
        except Exception as e:
            logger.error(f"Error restoring directory {archive_prefix}: {e}")
        
        return restored_count
    
    def cleanup_expired_archives(self, days_old: Optional[int] = None) -> int:
        """
        Clean up expired archives.
        
        Args:
            days_old: Number of days old to consider expired (uses retention_days if None)
            
        Returns:
            int: Number of archives cleaned up
        """
        days_old = days_old or self.retention_days
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        logger.info(f"Starting cleanup of archives older than {days_old} days")
        
        cleaned_count = 0
        
        try:
            # List all objects in archive bucket
            objects = self.s3_client.list_objects(bucket=self.archive_bucket)
            
            for obj in objects:
                if obj["last_modified"] < cutoff_date:
                    # Delete expired archive
                    if self.s3_client.delete_object(obj["key"], self.archive_bucket):
                        cleaned_count += 1
                        logger.debug(f"Deleted expired archive: {obj['key']}")
            
            logger.info(f"Cleaned up {cleaned_count} expired archives")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired archives: {e}")
        
        return cleaned_count
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """
        Get archive statistics.
        
        Returns:
            Dict[str, Any]: Archive statistics
        """
        current_time = datetime.now()
        processing_time = (current_time - self.archive_stats["start_time"]).total_seconds()
        
        stats = self.archive_stats.copy()
        stats["processing_time_seconds"] = processing_time
        stats["files_per_second"] = stats["files_archived"] / processing_time if processing_time > 0 else 0
        stats["bytes_per_second"] = stats["bytes_archived"] / processing_time if processing_time > 0 else 0
        stats["current_time"] = current_time.isoformat()
        
        return stats
    
    def get_archive_info(self, archive_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an archived file.
        
        Args:
            archive_path: Archive path in S3
            
        Returns:
            Optional[Dict[str, Any]]: Archive information or None if not found
        """
        try:
            metadata = self.s3_client.get_object_metadata(archive_path, self.archive_bucket)
            if metadata:
                # Add archive-specific information
                metadata["archive_bucket"] = self.archive_bucket
                metadata["retention_days"] = self.retention_days
                return metadata
            return None
        except Exception as e:
            logger.error(f"Error getting archive info for {archive_path}: {e}")
            return None
    
    def list_archives(self, prefix: str = '') -> List[Dict[str, Any]]:
        """
        List archived files.
        
        Args:
            prefix: Archive prefix to filter by
            
        Returns:
            List[Dict[str, Any]]: List of archive information
        """
        try:
            objects = self.s3_client.list_objects(prefix=prefix, bucket=self.archive_bucket)
            archives = []
            
            for obj in objects:
                archive_info = {
                    "path": obj["key"],
                    "size": obj["size"],
                    "last_modified": obj["last_modified"],
                    "etag": obj["etag"],
                    "storage_class": obj.get("storage_class", "STANDARD")
                }
                archives.append(archive_info)
            
            return archives
            
        except Exception as e:
            logger.error(f"Error listing archives: {e}")
            return []
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "archive_bucket": self.archive_bucket,
            "retention_days": self.retention_days,
            "compression_enabled": self.compression_enabled,
            "archive_stats": self.get_archive_stats(),
            "ingestion_timestamp": datetime.now().isoformat()
        }
