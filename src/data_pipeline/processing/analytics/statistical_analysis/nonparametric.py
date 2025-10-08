"""
Nonparametric Analysis for PBF-LB/M Process Data

This module provides comprehensive nonparametric analysis capabilities including
kernel density estimation, nonparametric tests, and nonparametric regression
for PBF-LB/M process data analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu, kruskal, spearmanr, kendalltau
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import warnings

logger = logging.getLogger(__name__)


@dataclass
class NonparametricConfig:
    """Configuration for nonparametric analysis."""
    
    # Kernel density parameters
    kde_bandwidth: str = "scott"  # "scott", "silverman", or float
    kde_kernel: str = "gaussian"  # "gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"
    
    # Nonparametric test parameters
    test_alpha: float = 0.05
    alternative: str = "two-sided"  # "two-sided", "less", "greater"
    
    # Analysis parameters
    confidence_level: float = 0.95
    random_seed: Optional[int] = None


@dataclass
class NonparametricResult:
    """Result of nonparametric analysis."""
    
    success: bool
    method: str
    data_names: List[str]
    analysis_results: Dict[str, Any]
    test_statistics: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class NonparametricAnalyzer:
    """
    Nonparametric analyzer for PBF-LB/M process data.
    
    This class provides comprehensive nonparametric analysis capabilities including
    kernel density estimation, nonparametric tests, and nonparametric regression
    for understanding distributions and relationships in PBF-LB/M process data.
    """
    
    def __init__(self, config: NonparametricConfig = None):
        """Initialize the nonparametric analyzer."""
        self.config = config or NonparametricConfig()
        self.analysis_cache = {}
        
        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info("Nonparametric Analyzer initialized")
    
    def analyze_kernel_density(
        self,
        data: pd.Series,
        data_name: str = None
    ) -> NonparametricResult:
        """
        Perform kernel density estimation.
        
        Args:
            data: Input data as Series
            data_name: Name of the data (optional)
            
        Returns:
            NonparametricResult: Kernel density analysis results
        """
        try:
            start_time = datetime.now()
            
            if data_name is None:
                data_name = data.name if hasattr(data, 'name') else 'data'
            
            # Prepare data
            y = data.values
            
            # Handle missing values
            y = y[~np.isnan(y)]
            
            if len(y) == 0:
                raise ValueError("No valid data points for kernel density estimation")
            
            # Create kernel density estimator
            kde = KernelDensity(
                bandwidth=self.config.kde_bandwidth,
                kernel=self.config.kde_kernel
            )
            
            # Fit kernel density
            kde.fit(y.reshape(-1, 1))
            
            # Generate density evaluation points
            x_min, x_max = y.min(), y.max()
            x_range = x_max - x_min
            x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
            
            # Calculate density
            log_density = kde.score_samples(x_eval.reshape(-1, 1))
            density = np.exp(log_density)
            
            # Calculate statistics
            mean_density = np.mean(density)
            max_density = np.max(density)
            density_entropy = -np.sum(density * np.log(density + 1e-10))
            
            # Calculate analysis results
            analysis_results = {
                'kde_model': kde,
                'evaluation_points': x_eval,
                'density_values': density,
                'log_density_values': log_density,
                'bandwidth': kde.bandwidth,
                'kernel': self.config.kde_kernel,
                'mean_density': mean_density,
                'max_density': max_density,
                'density_entropy': density_entropy
            }
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = NonparametricResult(
                success=True,
                method="KernelDensity",
                data_names=[data_name],
                analysis_results=analysis_results,
                test_statistics={},
                p_values={},
                confidence_intervals={},
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("kernel_density", result)
            
            logger.info(f"Kernel density analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in kernel density analysis: {e}")
            return NonparametricResult(
                success=False,
                method="KernelDensity",
                data_names=[data_name],
                analysis_results={},
                test_statistics={},
                p_values={},
                confidence_intervals={},
                error_message=str(e)
            )
    
    def analyze_nonparametric_tests(
        self,
        data1: pd.Series,
        data2: pd.Series = None,
        data_name1: str = None,
        data_name2: str = None
    ) -> NonparametricResult:
        """
        Perform nonparametric statistical tests.
        
        Args:
            data1: First data series
            data2: Second data series (optional, for two-sample tests)
            data_name1: Name of first data (optional)
            data_name2: Name of second data (optional)
            
        Returns:
            NonparametricResult: Nonparametric test results
        """
        try:
            start_time = datetime.now()
            
            if data_name1 is None:
                data_name1 = data1.name if hasattr(data1, 'name') else 'data1'
            
            if data_name2 is None and data2 is not None:
                data_name2 = data2.name if hasattr(data2, 'name') else 'data2'
            
            # Prepare data
            y1 = data1.values
            y1 = y1[~np.isnan(y1)]
            
            if data2 is not None:
                y2 = data2.values
                y2 = y2[~np.isnan(y2)]
            else:
                y2 = None
            
            # Perform tests
            test_statistics = {}
            p_values = {}
            analysis_results = {}
            
            if y2 is not None:
                # Two-sample tests
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = ks_2samp(y1, y2)
                test_statistics['ks_statistic'] = ks_stat
                p_values['ks_p_value'] = ks_p
                
                # Mann-Whitney U test
                mw_stat, mw_p = mannwhitneyu(y1, y2, alternative=self.config.alternative)
                test_statistics['mann_whitney_u'] = mw_stat
                p_values['mann_whitney_p_value'] = mw_p
                
                analysis_results['two_sample_tests'] = {
                    'ks_test': {'statistic': ks_stat, 'p_value': ks_p},
                    'mann_whitney_test': {'statistic': mw_stat, 'p_value': mw_p}
                }
            else:
                # One-sample tests
                # Kolmogorov-Smirnov test against normal distribution
                ks_stat, ks_p = stats.kstest(y1, 'norm', args=(np.mean(y1), np.std(y1)))
                test_statistics['ks_normality'] = ks_stat
                p_values['ks_normality_p_value'] = ks_p
                
                # Shapiro-Wilk test for normality
                shapiro_stat, shapiro_p = stats.shapiro(y1)
                test_statistics['shapiro_wilk'] = shapiro_stat
                p_values['shapiro_wilk_p_value'] = shapiro_p
                
                analysis_results['one_sample_tests'] = {
                    'ks_normality_test': {'statistic': ks_stat, 'p_value': ks_p},
                    'shapiro_wilk_test': {'statistic': shapiro_stat, 'p_value': shapiro_p}
                }
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(y1, y2)
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = NonparametricResult(
                success=True,
                method="NonparametricTests",
                data_names=[data_name1] + ([data_name2] if data_name2 else []),
                analysis_results=analysis_results,
                test_statistics=test_statistics,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("nonparametric_tests", result)
            
            logger.info(f"Nonparametric tests completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in nonparametric tests: {e}")
            return NonparametricResult(
                success=False,
                method="NonparametricTests",
                data_names=[data_name1] + ([data_name2] if data_name2 else []),
                analysis_results={},
                test_statistics={},
                p_values={},
                confidence_intervals={},
                error_message=str(e)
            )
    
    def analyze_correlation(
        self,
        data1: pd.Series,
        data2: pd.Series,
        data_name1: str = None,
        data_name2: str = None
    ) -> NonparametricResult:
        """
        Perform nonparametric correlation analysis.
        
        Args:
            data1: First data series
            data2: Second data series
            data_name1: Name of first data (optional)
            data_name2: Name of second data (optional)
            
        Returns:
            NonparametricResult: Correlation analysis results
        """
        try:
            start_time = datetime.now()
            
            if data_name1 is None:
                data_name1 = data1.name if hasattr(data1, 'name') else 'data1'
            
            if data_name2 is None:
                data_name2 = data2.name if hasattr(data2, 'name') else 'data2'
            
            # Prepare data
            y1 = data1.values
            y2 = data2.values
            
            # Handle missing values
            valid_mask = ~(np.isnan(y1) | np.isnan(y2))
            y1 = y1[valid_mask]
            y2 = y2[valid_mask]
            
            if len(y1) == 0:
                raise ValueError("No valid data points for correlation analysis")
            
            # Calculate Spearman correlation
            spearman_corr, spearman_p = spearmanr(y1, y2)
            
            # Calculate Kendall's tau
            kendall_tau, kendall_p = kendalltau(y1, y2)
            
            # Calculate Pearson correlation for comparison
            pearson_corr, pearson_p = stats.pearsonr(y1, y2)
            
            # Calculate test statistics
            test_statistics = {
                'spearman_correlation': spearman_corr,
                'kendall_tau': kendall_tau,
                'pearson_correlation': pearson_corr
            }
            
            # Calculate p-values
            p_values = {
                'spearman_p_value': spearman_p,
                'kendall_p_value': kendall_p,
                'pearson_p_value': pearson_p
            }
            
            # Calculate analysis results
            analysis_results = {
                'correlation_analysis': {
                    'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
                    'kendall': {'tau': kendall_tau, 'p_value': kendall_p},
                    'pearson': {'correlation': pearson_corr, 'p_value': pearson_p}
                },
                'n_samples': len(y1)
            }
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_correlation_confidence_intervals(
                spearman_corr, kendall_tau, len(y1)
            )
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = NonparametricResult(
                success=True,
                method="NonparametricCorrelation",
                data_names=[data_name1, data_name2],
                analysis_results=analysis_results,
                test_statistics=test_statistics,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("nonparametric_correlation", result)
            
            logger.info(f"Nonparametric correlation analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in nonparametric correlation analysis: {e}")
            return NonparametricResult(
                success=False,
                method="NonparametricCorrelation",
                data_names=[data_name1, data_name2],
                analysis_results={},
                test_statistics={},
                p_values={},
                confidence_intervals={},
                error_message=str(e)
            )
    
    def _calculate_confidence_intervals(
        self, 
        y1: np.ndarray, 
        y2: np.ndarray = None
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for nonparametric statistics."""
        confidence_intervals = {}
        
        # Calculate confidence interval for median of y1
        median_ci = self._calculate_median_confidence_interval(y1)
        confidence_intervals['median_confidence_interval'] = median_ci
        
        if y2 is not None:
            # Calculate confidence interval for median of y2
            median_ci2 = self._calculate_median_confidence_interval(y2)
            confidence_intervals['median_confidence_interval_2'] = median_ci2
        
        return confidence_intervals
    
    def _calculate_median_confidence_interval(self, y: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for median using bootstrap."""
        n = len(y)
        n_bootstrap = 1000
        
        # Bootstrap samples
        bootstrap_medians = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(y, size=n, replace=True)
            bootstrap_medians.append(np.median(bootstrap_sample))
        
        # Calculate confidence interval
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_medians, lower_percentile)
        ci_upper = np.percentile(bootstrap_medians, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def _calculate_correlation_confidence_intervals(
        self, 
        spearman_corr: float, 
        kendall_tau: float, 
        n: int
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for correlation coefficients."""
        confidence_intervals = {}
        
        # Fisher's z-transformation for Spearman correlation
        z_spearman = 0.5 * np.log((1 + spearman_corr) / (1 - spearman_corr))
        se_spearman = 1 / np.sqrt(n - 3)
        
        alpha = 1 - self.config.confidence_level
        z_critical = stats.norm.ppf(1 - alpha / 2)
        
        ci_lower_z = z_spearman - z_critical * se_spearman
        ci_upper_z = z_spearman + z_critical * se_spearman
        
        ci_lower_spearman = (np.exp(2 * ci_lower_z) - 1) / (np.exp(2 * ci_lower_z) + 1)
        ci_upper_spearman = (np.exp(2 * ci_upper_z) - 1) / (np.exp(2 * ci_upper_z) + 1)
        
        confidence_intervals['spearman_confidence_interval'] = (ci_lower_spearman, ci_upper_spearman)
        
        # Confidence interval for Kendall's tau
        se_kendall = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
        ci_lower_kendall = kendall_tau - z_critical * se_kendall
        ci_upper_kendall = kendall_tau + z_critical * se_kendall
        
        confidence_intervals['kendall_confidence_interval'] = (ci_lower_kendall, ci_upper_kendall)
        
        return confidence_intervals
    
    def _cache_result(self, method: str, result: NonparametricResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.data_names))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str, data_names: List[str]) -> Optional[NonparametricResult]:
        """Get cached analysis result."""
        cache_key = f"{method}_{hash(str(data_names))}"
        return self.analysis_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            'cache_size': len(self.analysis_cache),
            'config': {
                'kde_bandwidth': self.config.kde_bandwidth,
                'kde_kernel': self.config.kde_kernel,
                'test_alpha': self.config.test_alpha,
                'alternative': self.config.alternative,
                'confidence_level': self.config.confidence_level
            }
        }


class KernelDensityAnalyzer(NonparametricAnalyzer):
    """Specialized kernel density analyzer."""
    
    def __init__(self, config: NonparametricConfig = None):
        super().__init__(config)
        self.method_name = "KernelDensity"
    
    def analyze(self, data: pd.Series, data_name: str = None) -> NonparametricResult:
        """Perform kernel density analysis."""
        return self.analyze_kernel_density(data, data_name)
