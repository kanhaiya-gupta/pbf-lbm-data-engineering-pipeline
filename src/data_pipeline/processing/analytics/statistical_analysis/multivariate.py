"""
Multivariate Analysis for PBF-LB/M Process Data

This module provides comprehensive multivariate analysis capabilities including
principal component analysis (PCA), clustering analysis, and multivariate
statistical methods for PBF-LB/M process data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

logger = logging.getLogger(__name__)


@dataclass
class MultivariateConfig:
    """Configuration for multivariate analysis."""
    
    # PCA parameters
    pca_components: Optional[int] = None  # None for automatic selection
    pca_variance_threshold: float = 0.95  # Variance explained threshold
    
    # Clustering parameters
    clustering_method: str = "kmeans"  # "kmeans", "dbscan", "hierarchical"
    n_clusters: Optional[int] = None  # None for automatic selection
    clustering_metric: str = "euclidean"
    
    # Scaling parameters
    scaling_method: str = "standard"  # "standard", "minmax", "none"
    
    # Analysis parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05
    random_seed: Optional[int] = None


@dataclass
class MultivariateResult:
    """Result of multivariate analysis."""
    
    success: bool
    method: str
    feature_names: List[str]
    analysis_results: Dict[str, Any]
    explained_variance: Dict[str, float]
    component_loadings: Optional[pd.DataFrame] = None
    cluster_labels: Optional[np.ndarray] = None
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class MultivariateAnalyzer:
    """
    Multivariate analyzer for PBF-LB/M process data.
    
    This class provides comprehensive multivariate analysis capabilities including
    principal component analysis, clustering analysis, and multivariate statistical
    methods for understanding complex relationships in PBF-LB/M data.
    """
    
    def __init__(self, config: MultivariateConfig = None):
        """Initialize the multivariate analyzer."""
        self.config = config or MultivariateConfig()
        self.analysis_cache = {}
        
        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info("Multivariate Analyzer initialized")
    
    def analyze_pca(
        self,
        data: pd.DataFrame,
        feature_names: List[str] = None,
        n_components: int = None
    ) -> MultivariateResult:
        """
        Perform principal component analysis.
        
        Args:
            data: Input data as DataFrame
            feature_names: List of feature names (optional)
            n_components: Number of components (optional)
            
        Returns:
            MultivariateResult: PCA analysis results
        """
        try:
            start_time = datetime.now()
            
            if feature_names is None:
                feature_names = list(data.columns)
            
            if n_components is None:
                n_components = self.config.pca_components
            
            # Prepare data
            X = data[feature_names].values
            
            # Handle missing values
            X = self._handle_missing_values(X)
            
            # Scale data
            X_scaled = self._scale_data(X)
            
            # Perform PCA
            pca = PCA(n_components=n_components, random_state=self.config.random_seed)
            pca_result = pca.fit_transform(X_scaled)
            
            # Calculate explained variance
            explained_variance = {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
                'total_variance_explained': np.sum(pca.explained_variance_ratio_)
            }
            
            # Create component loadings DataFrame
            component_loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=feature_names
            )
            
            # Calculate analysis results
            analysis_results = {
                'n_components': pca.n_components_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'singular_values': pca.singular_values_,
                'mean': pca.mean_,
                'components': pca.components_,
                'transformed_data': pca_result
            }
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = MultivariateResult(
                success=True,
                method="PCA",
                feature_names=feature_names,
                analysis_results=analysis_results,
                explained_variance=explained_variance,
                component_loadings=component_loadings,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("pca", result)
            
            logger.info(f"PCA analysis completed: {analysis_time:.2f}s, {pca.n_components_} components")
            return result
            
        except Exception as e:
            logger.error(f"Error in PCA analysis: {e}")
            return MultivariateResult(
                success=False,
                method="PCA",
                feature_names=feature_names or [],
                analysis_results={},
                explained_variance={},
                error_message=str(e)
            )
    
    def analyze_clustering(
        self,
        data: pd.DataFrame,
        feature_names: List[str] = None,
        method: str = None,
        n_clusters: int = None
    ) -> MultivariateResult:
        """
        Perform clustering analysis.
        
        Args:
            data: Input data as DataFrame
            feature_names: List of feature names (optional)
            method: Clustering method (optional)
            n_clusters: Number of clusters (optional)
            
        Returns:
            MultivariateResult: Clustering analysis results
        """
        try:
            start_time = datetime.now()
            
            if feature_names is None:
                feature_names = list(data.columns)
            
            if method is None:
                method = self.config.clustering_method
            
            if n_clusters is None:
                n_clusters = self.config.n_clusters
            
            # Prepare data
            X = data[feature_names].values
            
            # Handle missing values
            X = self._handle_missing_values(X)
            
            # Scale data
            X_scaled = self._scale_data(X)
            
            # Perform clustering
            if method == "kmeans":
                cluster_result = self._perform_kmeans_clustering(X_scaled, n_clusters)
            elif method == "dbscan":
                cluster_result = self._perform_dbscan_clustering(X_scaled)
            elif method == "hierarchical":
                cluster_result = self._perform_hierarchical_clustering(X_scaled, n_clusters)
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            cluster_labels = cluster_result['labels']
            n_clusters_found = len(np.unique(cluster_labels[cluster_labels != -1]))
            
            # Calculate clustering metrics
            clustering_metrics = self._calculate_clustering_metrics(X_scaled, cluster_labels)
            
            # Calculate analysis results
            analysis_results = {
                'n_clusters': n_clusters_found,
                'cluster_labels': cluster_labels,
                'cluster_centers': cluster_result.get('centers', None),
                'clustering_metrics': clustering_metrics,
                'method': method
            }
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = MultivariateResult(
                success=True,
                method=f"Clustering_{method}",
                feature_names=feature_names,
                analysis_results=analysis_results,
                explained_variance={},
                cluster_labels=cluster_labels,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("clustering", result)
            
            logger.info(f"Clustering analysis completed: {analysis_time:.2f}s, {n_clusters_found} clusters")
            return result
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {e}")
            return MultivariateResult(
                success=False,
                method=f"Clustering_{method}",
                feature_names=feature_names or [],
                analysis_results={},
                explained_variance={},
                error_message=str(e)
            )
    
    def analyze_correlation(
        self,
        data: pd.DataFrame,
        feature_names: List[str] = None
    ) -> MultivariateResult:
        """
        Perform correlation analysis.
        
        Args:
            data: Input data as DataFrame
            feature_names: List of feature names (optional)
            
        Returns:
            MultivariateResult: Correlation analysis results
        """
        try:
            start_time = datetime.now()
            
            if feature_names is None:
                feature_names = list(data.columns)
            
            # Prepare data
            X = data[feature_names].values
            
            # Handle missing values
            X = self._handle_missing_values(X)
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(X.T)
            
            # Calculate significance tests
            n_samples = X.shape[0]
            significance_matrix = self._calculate_correlation_significance(correlation_matrix, n_samples)
            
            # Calculate analysis results
            analysis_results = {
                'correlation_matrix': correlation_matrix,
                'significance_matrix': significance_matrix,
                'n_samples': n_samples,
                'feature_names': feature_names
            }
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = MultivariateResult(
                success=True,
                method="Correlation",
                feature_names=feature_names,
                analysis_results=analysis_results,
                explained_variance={},
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("correlation", result)
            
            logger.info(f"Correlation analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return MultivariateResult(
                success=False,
                method="Correlation",
                feature_names=feature_names or [],
                analysis_results={},
                explained_variance={},
                error_message=str(e)
            )
    
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values in data."""
        # Simple imputation with mean
        if np.isnan(X).any():
            logger.warning("Missing values detected, imputing with mean")
            for i in range(X.shape[1]):
                col = X[:, i]
                col[np.isnan(col)] = np.nanmean(col)
        return X
    
    def _scale_data(self, X: np.ndarray) -> np.ndarray:
        """Scale data according to configuration."""
        if self.config.scaling_method == "standard":
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        elif self.config.scaling_method == "minmax":
            scaler = MinMaxScaler()
            return scaler.fit_transform(X)
        else:
            return X
    
    def _perform_kmeans_clustering(self, X: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Perform K-means clustering."""
        if n_clusters is None:
            # Determine optimal number of clusters using elbow method
            n_clusters = self._determine_optimal_clusters(X, method="kmeans")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_seed)
        labels = kmeans.fit_predict(X)
        
        return {
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }
    
    def _perform_dbscan_clustering(self, X: np.ndarray) -> Dict[str, Any]:
        """Perform DBSCAN clustering."""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        return {
            'labels': labels,
            'centers': None,
            'eps': dbscan.eps,
            'min_samples': dbscan.min_samples
        }
    
    def _perform_hierarchical_clustering(self, X: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Perform hierarchical clustering."""
        if n_clusters is None:
            # Determine optimal number of clusters
            n_clusters = self._determine_optimal_clusters(X, method="hierarchical")
        
        # Calculate linkage matrix
        linkage_matrix = linkage(X, method='ward')
        
        # Get cluster labels
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        return {
            'labels': labels,
            'centers': None,
            'linkage_matrix': linkage_matrix
        }
    
    def _determine_optimal_clusters(self, X: np.ndarray, method: str) -> int:
        """Determine optimal number of clusters."""
        max_clusters = min(10, X.shape[0] // 2)
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            if method == "kmeans":
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_seed)
                labels = kmeans.fit_predict(X)
                inertias.append(kmeans.inertia_)
            elif method == "hierarchical":
                linkage_matrix = linkage(X, method='ward')
                labels = fcluster(linkage_matrix, k, criterion='maxclust')
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(X, labels))
            else:
                silhouette_scores.append(0)
        
        # Find optimal number of clusters using silhouette score
        optimal_k = np.argmax(silhouette_scores) + 2
        
        return optimal_k
    
    def _calculate_clustering_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}
        
        # Remove noise points for metric calculation
        valid_labels = labels[labels != -1]
        valid_X = X[labels != -1]
        
        if len(np.unique(valid_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(valid_X, valid_labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(valid_X, valid_labels)
        else:
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz_score'] = 0.0
        
        # Calculate inertia (for K-means)
        if len(np.unique(valid_labels)) > 1:
            centers = np.array([valid_X[valid_labels == i].mean(axis=0) for i in np.unique(valid_labels)])
            inertia = 0.0
            for i, center in enumerate(centers):
                cluster_points = valid_X[valid_labels == i]
                inertia += np.sum((cluster_points - center) ** 2)
            metrics['inertia'] = inertia
        else:
            metrics['inertia'] = 0.0
        
        return metrics
    
    def _calculate_correlation_significance(self, correlation_matrix: np.ndarray, n_samples: int) -> np.ndarray:
        """Calculate correlation significance matrix."""
        significance_matrix = np.zeros_like(correlation_matrix)
        
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                if i != j:
                    r = correlation_matrix[i, j]
                    # Calculate t-statistic
                    t_stat = r * np.sqrt((n_samples - 2) / (1 - r**2))
                    # Calculate p-value
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 2))
                    significance_matrix[i, j] = p_value
        
        return significance_matrix
    
    def _cache_result(self, method: str, result: MultivariateResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.feature_names))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str, feature_names: List[str]) -> Optional[MultivariateResult]:
        """Get cached analysis result."""
        cache_key = f"{method}_{hash(str(feature_names))}"
        return self.analysis_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            'cache_size': len(self.analysis_cache),
            'config': {
                'pca_components': self.config.pca_components,
                'pca_variance_threshold': self.config.pca_variance_threshold,
                'clustering_method': self.config.clustering_method,
                'n_clusters': self.config.n_clusters,
                'scaling_method': self.config.scaling_method
            }
        }


class PCAAnalyzer(MultivariateAnalyzer):
    """Specialized PCA analyzer."""
    
    def __init__(self, config: MultivariateConfig = None):
        super().__init__(config)
        self.method_name = "PCA"
    
    def analyze(self, data: pd.DataFrame, feature_names: List[str] = None, 
                n_components: int = None) -> MultivariateResult:
        """Perform PCA analysis."""
        return self.analyze_pca(data, feature_names, n_components)


class ClusterAnalyzer(MultivariateAnalyzer):
    """Specialized clustering analyzer."""
    
    def __init__(self, config: MultivariateConfig = None):
        super().__init__(config)
        self.method_name = "Clustering"
    
    def analyze(self, data: pd.DataFrame, feature_names: List[str] = None, 
                method: str = None, n_clusters: int = None) -> MultivariateResult:
        """Perform clustering analysis."""
        return self.analyze_clustering(data, feature_names, method, n_clusters)
