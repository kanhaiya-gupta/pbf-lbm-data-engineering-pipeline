"""
Image Data Loader

This module implements utilities for loading image data from various sources
in PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import json
import yaml
from datetime import datetime
import cv2
from PIL import Image
import h5py

logger = logging.getLogger(__name__)


class ImageDataLoader:
    """
    Utility class for loading image data from various sources.
    
    This class handles:
    - Loading images from various file formats (PNG, JPEG, TIFF, HDF5)
    - Loading from image databases and directories
    - Image preprocessing and validation
    - Batch loading and memory management
    - Image metadata extraction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the image data loader.
        
        Args:
            config: Configuration dictionary with image loading settings
        """
        self.config = config or {}
        self.supported_formats = ['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'hdf5', 'npy']
        
        # Define image types and their expected properties
        self.image_types = {
            'ct_scan': {
                'expected_channels': 1,  # Grayscale
                'expected_dtype': np.uint16,
                'typical_size': (512, 512),
                'description': 'CT scan images for internal structure analysis'
            },
            'powder_bed': {
                'expected_channels': 1,  # Grayscale
                'expected_dtype': np.uint8,
                'typical_size': (1024, 1024),
                'description': 'Powder bed images for surface analysis'
            },
            'defect_image': {
                'expected_channels': 3,  # RGB
                'expected_dtype': np.uint8,
                'typical_size': (256, 256),
                'description': 'Defect images for classification'
            },
            'surface_texture': {
                'expected_channels': 1,  # Grayscale
                'expected_dtype': np.uint8,
                'typical_size': (512, 512),
                'description': 'Surface texture images for roughness analysis'
            },
            'thermal_image': {
                'expected_channels': 1,  # Grayscale
                'expected_dtype': np.uint16,
                'typical_size': (640, 480),
                'description': 'Thermal images for temperature analysis'
            }
        }
        
        logger.info("Initialized ImageDataLoader")
    
    def load_image(self, image_path: Union[str, Path], 
                  image_type: str,
                  target_size: Optional[Tuple[int, int]] = None,
                  normalize: bool = True) -> np.ndarray:
        """
        Load a single image from file.
        
        Args:
            image_path: Path to the image file
            image_type: Type of image (ct_scan, powder_bed, defect_image, etc.)
            target_size: Target size (width, height) for resizing
            normalize: Whether to normalize pixel values to [0, 1]
            
        Returns:
            Image as numpy array
            
        Raises:
            ValueError: If image type is not supported
            FileNotFoundError: If image file does not exist
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if image_type not in self.image_types:
            raise ValueError(f"Unsupported image type: {image_type}")
        
        # Determine file format
        file_format = image_path.suffix.lower().lstrip('.')
        
        if file_format not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {file_format}")
        
        try:
            if file_format == 'hdf5':
                image = self._load_hdf5_image(image_path)
            elif file_format == 'npy':
                image = self._load_npy_image(image_path)
            else:
                image = self._load_standard_image(image_path)
            
            # Validate image
            image = self._validate_image(image, image_type)
            
            # Resize if target size specified
            if target_size is not None:
                image = self._resize_image(image, target_size)
            
            # Normalize if requested
            if normalize:
                image = self._normalize_image(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def _load_standard_image(self, image_path: Path) -> np.ndarray:
        """Load image using OpenCV."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except ImportError:
            # Fallback to PIL if OpenCV is not available
            try:
                with Image.open(image_path) as img:
                    image = np.array(img)
                    return image
            except ImportError:
                raise ImportError("OpenCV or PIL is required for image loading. Install with: pip install opencv-python pillow")
    
    def _load_hdf5_image(self, image_path: Path) -> np.ndarray:
        """Load image from HDF5 file."""
        try:
            with h5py.File(image_path, 'r') as f:
                # Try to find image data in common locations
                data_paths = ['/image', '/data', '/image_data', '/']
                
                for path in data_paths:
                    if path in f:
                        image = f[path][:]
                        break
                else:
                    # If no specific path found, try to load the first dataset
                    keys = list(f.keys())
                    if keys:
                        image = f[keys[0]][:]
                    else:
                        raise ValueError("No data found in HDF5 file")
                
                return image
                
        except ImportError:
            raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    
    def _load_npy_image(self, image_path: Path) -> np.ndarray:
        """Load image from NumPy file."""
        return np.load(image_path)
    
    def _validate_image(self, image: np.ndarray, image_type: str) -> np.ndarray:
        """
        Validate image properties and convert if necessary.
        
        Args:
            image: Image array
            image_type: Type of image
            
        Returns:
            Validated image array
        """
        image_config = self.image_types[image_type]
        
        # Check image dimensions
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Invalid image dimensions: {image.shape}")
        
        # Check number of channels
        if len(image.shape) == 3:
            actual_channels = image.shape[2]
            expected_channels = image_config['expected_channels']
            
            if actual_channels != expected_channels:
                logger.warning(f"Expected {expected_channels} channels, got {actual_channels}")
                
                # Convert if possible
                if expected_channels == 1 and actual_channels == 3:
                    # Convert RGB to grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif expected_channels == 3 and actual_channels == 1:
                    # Convert grayscale to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Check data type
        expected_dtype = image_config['expected_dtype']
        if image.dtype != expected_dtype:
            logger.warning(f"Expected dtype {expected_dtype}, got {image.dtype}")
            # Convert to expected dtype
            if expected_dtype == np.uint8:
                image = image.astype(np.uint8)
            elif expected_dtype == np.uint16:
                image = image.astype(np.uint16)
            elif expected_dtype == np.float32:
                image = image.astype(np.float32)
        
        return image
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Image array
            target_size: Target size (width, height)
            
        Returns:
            Resized image array
        """
        width, height = target_size
        
        if len(image.shape) == 3:
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1].
        
        Args:
            image: Image array
            
        Returns:
            Normalized image array
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            # Assume already in [0, 1] range
            return image.astype(np.float32)
    
    def load_image_batch(self, image_paths: List[Union[str, Path]], 
                        image_type: str,
                        target_size: Optional[Tuple[int, int]] = None,
                        normalize: bool = True,
                        batch_size: int = 32) -> List[np.ndarray]:
        """
        Load a batch of images.
        
        Args:
            image_paths: List of image file paths
            image_type: Type of images
            target_size: Target size for resizing
            normalize: Whether to normalize pixel values
            batch_size: Number of images to load at once (for memory management)
            
        Returns:
            List of image arrays
        """
        images = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image = self.load_image(path, image_type, target_size, normalize)
                    batch_images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
                    continue
            
            images.extend(batch_images)
            
            # Log progress
            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Loaded {min(i + batch_size, len(image_paths))} / {len(image_paths)} images")
        
        return images
    
    def load_from_directory(self, directory_path: Union[str, Path], 
                           image_type: str,
                           file_pattern: str = "*",
                           recursive: bool = True,
                           target_size: Optional[Tuple[int, int]] = None,
                           normalize: bool = True) -> Tuple[List[np.ndarray], List[Path]]:
        """
        Load all images from a directory.
        
        Args:
            directory_path: Path to directory containing images
            image_type: Type of images
            file_pattern: File pattern to match (e.g., "*.png", "*.jpg")
            recursive: Whether to search recursively
            target_size: Target size for resizing
            normalize: Whether to normalize pixel values
            
        Returns:
            Tuple of (images, file_paths)
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        if recursive:
            image_files = list(directory_path.rglob(file_pattern))
        else:
            image_files = list(directory_path.glob(file_pattern))
        
        # Filter by supported formats
        supported_extensions = [f".{fmt}" for fmt in self.supported_formats if fmt != 'hdf5' and fmt != 'npy']
        image_files = [f for f in image_files if f.suffix.lower() in supported_extensions]
        
        if not image_files:
            logger.warning(f"No image files found in {directory_path}")
            return [], []
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Load images
        images = []
        valid_paths = []
        
        for image_file in image_files:
            try:
                image = self.load_image(image_file, image_type, target_size, normalize)
                images.append(image)
                valid_paths.append(image_file)
            except Exception as e:
                logger.warning(f"Failed to load image {image_file}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(images)} images")
        
        return images, valid_paths
    
    def load_image_metadata(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load metadata from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image metadata
        """
        image_path = Path(image_path)
        
        metadata = {
            'file_path': str(image_path),
            'file_name': image_path.name,
            'file_size': image_path.stat().st_size,
            'file_format': image_path.suffix.lower().lstrip('.'),
            'created_time': datetime.fromtimestamp(image_path.stat().st_ctime),
            'modified_time': datetime.fromtimestamp(image_path.stat().st_mtime)
        }
        
        try:
            # Load image to get additional metadata
            image = self._load_standard_image(image_path)
            
            metadata.update({
                'width': image.shape[1],
                'height': image.shape[0],
                'channels': image.shape[2] if len(image.shape) == 3 else 1,
                'dtype': str(image.dtype),
                'min_value': float(image.min()),
                'max_value': float(image.max()),
                'mean_value': float(image.mean()),
                'std_value': float(image.std())
            })
            
            # Try to load EXIF data if available
            try:
                with Image.open(image_path) as img:
                    if hasattr(img, '_getexif') and img._getexif() is not None:
                        exif_data = img._getexif()
                        metadata['exif_data'] = exif_data
            except Exception:
                pass  # EXIF data not available or not readable
            
        except Exception as e:
            logger.warning(f"Failed to load image metadata for {image_path}: {e}")
        
        return metadata
    
    def load_image_dataset(self, dataset_path: Union[str, Path], 
                          image_type: str,
                          target_size: Optional[Tuple[int, int]] = None,
                          normalize: bool = True) -> Dict[str, Any]:
        """
        Load a complete image dataset with labels and metadata.
        
        Args:
            dataset_path: Path to dataset directory
            image_type: Type of images
            target_size: Target size for resizing
            normalize: Whether to normalize pixel values
            
        Returns:
            Dictionary with dataset information
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        dataset_info = {
            'dataset_path': str(dataset_path),
            'image_type': image_type,
            'images': [],
            'labels': [],
            'metadata': [],
            'class_distribution': {},
            'total_images': 0
        }
        
        # Look for labels file
        labels_file = dataset_path / 'labels.json'
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                labels_data = json.load(f)
        else:
            labels_data = {}
        
        # Look for metadata file
        metadata_file = dataset_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata_data = json.load(f)
        else:
            metadata_data = {}
        
        # Load images from subdirectories (assuming class-based organization)
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        if not class_dirs:
            # No subdirectories, load all images from root
            images, paths = self.load_from_directory(dataset_path, image_type, target_size=target_size, normalize=normalize)
            dataset_info['images'] = images
            dataset_info['labels'] = [labels_data.get(p.name, 'unknown') for p in paths]
            dataset_info['metadata'] = [metadata_data.get(p.name, {}) for p in paths]
        else:
            # Load images from each class directory
            for class_dir in class_dirs:
                class_name = class_dir.name
                images, paths = self.load_from_directory(class_dir, image_type, target_size=target_size, normalize=normalize)
                
                dataset_info['images'].extend(images)
                dataset_info['labels'].extend([class_name] * len(images))
                dataset_info['metadata'].extend([metadata_data.get(p.name, {}) for p in paths])
                dataset_info['class_distribution'][class_name] = len(images)
        
        dataset_info['total_images'] = len(dataset_info['images'])
        
        # Calculate class distribution if not already done
        if not dataset_info['class_distribution']:
            from collections import Counter
            dataset_info['class_distribution'] = dict(Counter(dataset_info['labels']))
        
        return dataset_info
    
    def preprocess_image(self, image: np.ndarray, 
                        preprocessing_steps: List[str]) -> np.ndarray:
        """
        Apply preprocessing steps to an image.
        
        Args:
            image: Input image array
            preprocessing_steps: List of preprocessing steps to apply
            
        Returns:
            Preprocessed image array
        """
        processed_image = image.copy()
        
        for step in preprocessing_steps:
            if step == 'grayscale':
                if len(processed_image.shape) == 3:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
            elif step == 'resize':
                # Resize to standard size
                processed_image = cv2.resize(processed_image, (224, 224))
            elif step == 'normalize':
                processed_image = self._normalize_image(processed_image)
            elif step == 'histogram_equalization':
                if len(processed_image.shape) == 2:
                    processed_image = cv2.equalizeHist(processed_image)
                else:
                    # Apply to each channel
                    for i in range(processed_image.shape[2]):
                        processed_image[:, :, i] = cv2.equalizeHist(processed_image[:, :, i])
            elif step == 'gaussian_blur':
                processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)
            elif step == 'median_filter':
                processed_image = cv2.medianBlur(processed_image, 5)
            elif step == 'edge_detection':
                processed_image = cv2.Canny(processed_image, 50, 150)
            else:
                logger.warning(f"Unknown preprocessing step: {step}")
        
        return processed_image
    
    def augment_image(self, image: np.ndarray, 
                     augmentation_config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Apply data augmentation to an image.
        
        Args:
            image: Input image array
            augmentation_config: Configuration for augmentation
            
        Returns:
            List of augmented images
        """
        augmented_images = [image]
        
        # Rotation
        if augmentation_config.get('rotation', False):
            angles = augmentation_config.get('rotation_angles', [90, 180, 270])
            for angle in angles:
                rotated = self._rotate_image(image, angle)
                augmented_images.append(rotated)
        
        # Flipping
        if augmentation_config.get('flip_horizontal', False):
            flipped = cv2.flip(image, 1)
            augmented_images.append(flipped)
        
        if augmentation_config.get('flip_vertical', False):
            flipped = cv2.flip(image, 0)
            augmented_images.append(flipped)
        
        # Brightness adjustment
        if augmentation_config.get('brightness', False):
            brightness_range = augmentation_config.get('brightness_range', [0.8, 1.2])
            for factor in brightness_range:
                brightened = self._adjust_brightness(image, factor)
                augmented_images.append(brightened)
        
        # Contrast adjustment
        if augmentation_config.get('contrast', False):
            contrast_range = augmentation_config.get('contrast_range', [0.8, 1.2])
            for factor in contrast_range:
                contrasted = self._adjust_contrast(image, factor)
                augmented_images.append(contrasted)
        
        return augmented_images
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        if image.dtype == np.uint8:
            adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=int(255 * (factor - 1.0)))
        else:
            adjusted = image * factor
            adjusted = np.clip(adjusted, 0, 1)
        
        return adjusted
    
    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        if image.dtype == np.uint8:
            adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        else:
            adjusted = image * factor
            adjusted = np.clip(adjusted, 0, 1)
        
        return adjusted
    
    def get_image_statistics(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of images.
        
        Args:
            images: List of image arrays
            
        Returns:
            Dictionary with image statistics
        """
        if not images:
            return {}
        
        # Stack all images
        all_images = np.stack(images)
        
        statistics = {
            'total_images': len(images),
            'image_shape': images[0].shape,
            'dtype': str(images[0].dtype),
            'min_value': float(all_images.min()),
            'max_value': float(all_images.max()),
            'mean_value': float(all_images.mean()),
            'std_value': float(all_images.std()),
            'median_value': float(np.median(all_images))
        }
        
        # Per-image statistics
        per_image_stats = []
        for i, image in enumerate(images):
            img_stats = {
                'image_index': i,
                'min_value': float(image.min()),
                'max_value': float(image.max()),
                'mean_value': float(image.mean()),
                'std_value': float(image.std())
            }
            per_image_stats.append(img_stats)
        
        statistics['per_image_statistics'] = per_image_stats
        
        return statistics
