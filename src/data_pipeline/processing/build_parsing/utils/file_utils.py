"""
File Utilities for PBF-LB/M Build Files.

This module provides utility functions for file operations, validation,
and common file handling tasks for build file processing.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import hashlib
import os
import shutil

logger = logging.getLogger(__name__)


class FileUtils:
    """
    Utility class for file operations related to PBF-LB/M build files.
    
    This class provides common file handling functionality including
    validation, copying, hashing, and metadata extraction.
    """
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path]) -> bool:
        """
        Validate that a file path exists and is accessible.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid and accessible, False otherwise
        """
        try:
            path = Path(file_path)
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except Exception as e:
            logger.warning(f"Error validating file path {file_path}: {e}")
            return False
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file information
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            
            return {
                'file_name': path.name,
                'file_path': str(path.absolute()),
                'file_size': stat.st_size,
                'file_extension': path.suffix.lower(),
                'creation_time': stat.st_ctime,
                'modification_time': stat.st_mtime,
                'access_time': stat.st_atime,
                'is_readable': os.access(path, os.R_OK),
                'is_writable': os.access(path, os.W_OK),
                'file_hash': FileUtils.calculate_file_hash(path)
            }
        
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> Optional[str]:
        """
        Calculate hash of a file.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
            
        Returns:
            Hash string or None if error
        """
        try:
            path = Path(file_path)
            
            if algorithm == 'md5':
                hash_obj = hashlib.md5()
            elif algorithm == 'sha1':
                hash_obj = hashlib.sha1()
            elif algorithm == 'sha256':
                hash_obj = hashlib.sha256()
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
        
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    @staticmethod
    def copy_file(source_path: Union[str, Path], dest_path: Union[str, Path], 
                  preserve_metadata: bool = True) -> bool:
        """
        Copy a file to a new location.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            preserve_metadata: Whether to preserve file metadata
            
        Returns:
            True if copy successful, False otherwise
        """
        try:
            source = Path(source_path)
            dest = Path(dest_path)
            
            # Create destination directory if it doesn't exist
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            if preserve_metadata:
                shutil.copy2(source, dest)
            else:
                shutil.copy(source, dest)
            
            logger.info(f"File copied from {source} to {dest}")
            return True
        
        except Exception as e:
            logger.error(f"Error copying file from {source_path} to {dest_path}: {e}")
            return False
    
    @staticmethod
    def move_file(source_path: Union[str, Path], dest_path: Union[str, Path]) -> bool:
        """
        Move a file to a new location.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            
        Returns:
            True if move successful, False otherwise
        """
        try:
            source = Path(source_path)
            dest = Path(dest_path)
            
            # Create destination directory if it doesn't exist
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source), str(dest))
            logger.info(f"File moved from {source} to {dest}")
            return True
        
        except Exception as e:
            logger.error(f"Error moving file from {source_path} to {dest_path}: {e}")
            return False
    
    @staticmethod
    def delete_file(file_path: Union[str, Path]) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"File deleted: {path}")
                return True
            else:
                logger.warning(f"File does not exist: {path}")
                return False
        
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    @staticmethod
    def create_backup(file_path: Union[str, Path], backup_suffix: str = '.backup') -> Optional[str]:
        """
        Create a backup of a file.
        
        Args:
            file_path: Path to the file to backup
            backup_suffix: Suffix to add to backup file
            
        Returns:
            Path to backup file or None if error
        """
        try:
            path = Path(file_path)
            backup_path = path.with_suffix(path.suffix + backup_suffix)
            
            if FileUtils.copy_file(path, backup_path):
                return str(backup_path)
            else:
                return None
        
        except Exception as e:
            logger.error(f"Error creating backup for {file_path}: {e}")
            return None
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """
        Get list of supported build file extensions.
        
        Returns:
            List of supported file extensions
        """
        return ['.mtt', '.sli', '.cli', '.slm', '.rea', '.txt', '.json', '.xml', '.csv', '.dat']
    
    @staticmethod
    def is_supported_format(file_path: Union[str, Path]) -> bool:
        """
        Check if file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if format is supported, False otherwise
        """
        try:
            path = Path(file_path)
            return path.suffix.lower() in FileUtils.get_supported_extensions()
        except Exception:
            return False
    
    @staticmethod
    def get_file_size_mb(file_path: Union[str, Path]) -> float:
        """
        Get file size in megabytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in MB
        """
        try:
            path = Path(file_path)
            return path.stat().st_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    @staticmethod
    def ensure_directory_exists(directory_path: Union[str, Path]) -> bool:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            True if directory exists or was created, False otherwise
        """
        try:
            path = Path(directory_path)
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error ensuring directory exists {directory_path}: {e}")
            return False
    
    @staticmethod
    def list_files_in_directory(directory_path: Union[str, Path], 
                               pattern: str = "*", 
                               recursive: bool = False) -> List[Path]:
        """
        List files in a directory matching a pattern.
        
        Args:
            directory_path: Path to the directory
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        try:
            path = Path(directory_path)
            if recursive:
                return list(path.rglob(pattern))
            else:
                return list(path.glob(pattern))
        except Exception as e:
            logger.error(f"Error listing files in {directory_path}: {e}")
            return []
    
    @staticmethod
    def get_temp_file_path(prefix: str = "build_file", suffix: str = ".tmp") -> str:
        """
        Get a temporary file path.
        
        Args:
            prefix: File prefix
            suffix: File suffix
            
        Returns:
            Temporary file path
        """
        import tempfile
        temp_dir = tempfile.gettempdir()
        return str(Path(temp_dir) / f"{prefix}_{hashlib.md5().hexdigest()[:8]}{suffix}")
    
    @staticmethod
    def cleanup_temp_files(temp_paths: List[Union[str, Path]]) -> int:
        """
        Clean up temporary files.
        
        Args:
            temp_paths: List of temporary file paths
            
        Returns:
            Number of files successfully deleted
        """
        deleted_count = 0
        for temp_path in temp_paths:
            if FileUtils.delete_file(temp_path):
                deleted_count += 1
        return deleted_count
