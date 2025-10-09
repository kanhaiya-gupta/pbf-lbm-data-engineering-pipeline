"""
Classification Metrics

This module implements evaluation metrics for classification models
in PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
    cohen_kappa_score, matthews_corrcoef, log_loss
)
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """
    Utility class for classification model evaluation metrics.
    
    This class handles:
    - Standard classification metrics (accuracy, precision, recall, F1, etc.)
    - Manufacturing-specific metrics
    - Multi-class and multi-label classification
    - Confusion matrix analysis
    - ROC and PR curve analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the classification metrics calculator.
        
        Args:
            config: Configuration dictionary with metric settings
        """
        self.config = config or {}
        
        # Standard classification metrics
        self.standard_metrics = {
            'accuracy': self._calculate_accuracy,
            'precision': self._calculate_precision,
            'recall': self._calculate_recall,
            'f1': self._calculate_f1,
            'f1_macro': self._calculate_f1_macro,
            'f1_weighted': self._calculate_f1_weighted,
            'auc': self._calculate_auc,
            'average_precision': self._calculate_average_precision,
            'cohen_kappa': self._calculate_cohen_kappa,
            'matthews_corrcoef': self._calculate_matthews_corrcoef,
            'log_loss': self._calculate_log_loss
        }
        
        # Manufacturing-specific metrics
        self.manufacturing_metrics = {
            'defect_detection_accuracy': self._calculate_defect_detection_accuracy,
            'false_alarm_rate': self._calculate_false_alarm_rate,
            'miss_rate': self._calculate_miss_rate,
            'quality_control_score': self._calculate_quality_control_score,
            'production_efficiency': self._calculate_production_efficiency,
            'cost_effectiveness': self._calculate_cost_effectiveness
        }
        
        logger.info("Initialized ClassificationMetrics")
    
    def calculate_metrics(self, y_true: Union[np.ndarray, pd.Series, List], 
                         y_pred: Union[np.ndarray, pd.Series, List],
                         y_prob: Optional[Union[np.ndarray, pd.Series, List]] = None,
                         metrics: Optional[List[str]] = None,
                         manufacturing_metrics: bool = False,
                         cost_matrix: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for AUC, log loss, etc.)
            metrics: List of metrics to calculate (None for all standard metrics)
            manufacturing_metrics: Whether to include manufacturing-specific metrics
            cost_matrix: Cost matrix for cost-sensitive metrics
            
        Returns:
            Dictionary with calculated metrics
        """
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_prob is not None:
            y_prob = np.array(y_prob)
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if y_prob is not None:
            y_prob = y_prob[valid_mask]
        
        if len(y_true) == 0:
            raise ValueError("No valid values found after removing NaN")
        
        results = {}
        
        # Calculate standard metrics
        if metrics is None:
            metrics = list(self.standard_metrics.keys())
        
        for metric in metrics:
            if metric in self.standard_metrics:
                try:
                    if metric in ['auc', 'average_precision', 'log_loss'] and y_prob is None:
                        logger.warning(f"Probability scores required for {metric}")
                        results[metric] = np.nan
                        continue
                    
                    results[metric] = self.standard_metrics[metric](y_true, y_pred, y_prob)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric}: {e}")
                    results[metric] = np.nan
        
        # Calculate manufacturing-specific metrics
        if manufacturing_metrics:
            for metric_name, metric_func in self.manufacturing_metrics.items():
                try:
                    if metric_name in ['cost_effectiveness'] and cost_matrix is None:
                        logger.warning(f"Cost matrix required for {metric_name}")
                        continue
                    
                    results[metric_name] = metric_func(y_true, y_pred, y_prob, cost_matrix)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
                    results[metric_name] = np.nan
        
        return results
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> float:
        """Calculate accuracy score."""
        return float(accuracy_score(y_true, y_pred))
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> float:
        """Calculate precision score."""
        return float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> float:
        """Calculate recall score."""
        return float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    
    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> float:
        """Calculate F1 score."""
        return float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    def _calculate_f1_macro(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> float:
        """Calculate macro-averaged F1 score."""
        return float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    def _calculate_f1_weighted(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> float:
        """Calculate weighted F1 score."""
        return float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    def _calculate_auc(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate ROC AUC score."""
        if len(np.unique(y_true)) == 2:
            # Binary classification
            return float(roc_auc_score(y_true, y_prob))
        else:
            # Multi-class classification
            return float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'))
    
    def _calculate_average_precision(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate average precision score."""
        if len(np.unique(y_true)) == 2:
            # Binary classification
            return float(average_precision_score(y_true, y_prob))
        else:
            # Multi-class classification
            return float(average_precision_score(y_true, y_prob, average='weighted'))
    
    def _calculate_cohen_kappa(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> float:
        """Calculate Cohen's kappa score."""
        return float(cohen_kappa_score(y_true, y_pred))
    
    def _calculate_matthews_corrcoef(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> float:
        """Calculate Matthews correlation coefficient."""
        return float(matthews_corrcoef(y_true, y_pred))
    
    def _calculate_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate log loss."""
        return float(log_loss(y_true, y_prob))
    
    def _calculate_defect_detection_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                          y_prob: Optional[np.ndarray] = None, 
                                          cost_matrix: Optional[Dict[str, Any]] = None) -> float:
        """Calculate defect detection accuracy for manufacturing."""
        # Assuming 1 = defect, 0 = no defect
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            return float(accuracy)
        else:
            # Multi-class case
            return float(accuracy_score(y_true, y_pred))
    
    def _calculate_false_alarm_rate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: Optional[np.ndarray] = None, 
                                  cost_matrix: Optional[Dict[str, Any]] = None) -> float:
        """Calculate false alarm rate (Type I error)."""
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            return float(false_alarm_rate)
        else:
            # Multi-class case - calculate average false alarm rate
            false_alarm_rates = []
            for i in range(cm.shape[0]):
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                false_alarm_rates.append(false_alarm_rate)
            return float(np.mean(false_alarm_rates))
    
    def _calculate_miss_rate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None, 
                           cost_matrix: Optional[Dict[str, Any]] = None) -> float:
        """Calculate miss rate (Type II error)."""
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            return float(miss_rate)
        else:
            # Multi-class case - calculate average miss rate
            miss_rates = []
            for i in range(cm.shape[0]):
                fn = np.sum(cm[i, :]) - cm[i, i]
                tp = cm[i, i]
                miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                miss_rates.append(miss_rate)
            return float(np.mean(miss_rates))
    
    def _calculate_quality_control_score(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_prob: Optional[np.ndarray] = None, 
                                       cost_matrix: Optional[Dict[str, Any]] = None) -> float:
        """Calculate quality control score for manufacturing."""
        # Combine accuracy, precision, and recall into a quality score
        accuracy = self._calculate_accuracy(y_true, y_pred)
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)
        
        # Weighted combination
        quality_score = (accuracy * 0.4 + precision * 0.3 + recall * 0.3) * 100
        return float(quality_score)
    
    def _calculate_production_efficiency(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_prob: Optional[np.ndarray] = None, 
                                       cost_matrix: Optional[Dict[str, Any]] = None) -> float:
        """Calculate production efficiency based on classification performance."""
        # High accuracy and low false alarm rate indicate good efficiency
        accuracy = self._calculate_accuracy(y_true, y_pred)
        false_alarm_rate = self._calculate_false_alarm_rate(y_true, y_pred)
        
        # Efficiency score (0-100)
        efficiency = (accuracy * 0.7 + (1 - false_alarm_rate) * 0.3) * 100
        return float(max(0, min(100, efficiency)))
    
    def _calculate_cost_effectiveness(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_prob: Optional[np.ndarray] = None, 
                                    cost_matrix: Optional[Dict[str, Any]] = None) -> float:
        """Calculate cost effectiveness based on cost matrix."""
        if cost_matrix is None:
            return np.nan
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate total cost
        total_cost = 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cost_key = f"{i}_{j}"
                if cost_key in cost_matrix:
                    total_cost += cm[i, j] * cost_matrix[cost_key]
        
        # Normalize by number of samples
        cost_per_sample = total_cost / len(y_true)
        
        # Convert to effectiveness score (lower cost = higher effectiveness)
        max_acceptable_cost = cost_matrix.get('max_cost', 1.0)
        effectiveness = max(0, (max_acceptable_cost - cost_per_sample) / max_acceptable_cost * 100)
        
        return float(effectiveness)
    
    def calculate_confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Perform detailed confusion matrix analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with confusion matrix analysis
        """
        cm = confusion_matrix(y_true, y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        
        analysis = {
            'confusion_matrix': cm.tolist(),
            'labels': labels.tolist(),
            'class_metrics': {},
            'overall_metrics': {}
        }
        
        # Calculate per-class metrics
        for i, label in enumerate(labels):
            if i < cm.shape[0]:
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                tn = np.sum(cm) - (tp + fp + fn)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                analysis['class_metrics'][str(label)] = {
                    'true_positive': int(tp),
                    'false_positive': int(fp),
                    'true_negative': int(tn),
                    'false_negative': int(fn),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'specificity': float(specificity),
                    'support': int(tp + fn)
                }
        
        # Calculate overall metrics
        accuracy = np.trace(cm) / np.sum(cm)
        macro_precision = np.mean([analysis['class_metrics'][str(label)]['precision'] for label in labels])
        macro_recall = np.mean([analysis['class_metrics'][str(label)]['recall'] for label in labels])
        macro_f1 = np.mean([analysis['class_metrics'][str(label)]['f1_score'] for label in labels])
        
        analysis['overall_metrics'] = {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1)
        }
        
        return analysis
    
    def visualize_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 title: str = "Confusion Matrix",
                                 normalize: bool = False,
                                 save_path: Optional[str] = None) -> None:
        """
        Visualize confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=np.unique(y_pred), 
                   yticklabels=np.unique(y_true))
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      title: str = "ROC Curve",
                      save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
        """
        if len(np.unique(y_true)) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            n_classes = y_true_bin.shape[1]
            
            plt.figure(figsize=(10, 8))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                  title: str = "Precision-Recall Curve",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
        """
        if len(np.unique(y_true)) == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2, 
                    label=f'PR curve (AP = {avg_precision:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            n_classes = y_true_bin.shape[1]
            
            plt.figure(figsize=(10, 8))
            
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                plt.plot(recall, precision, lw=2, 
                        label=f'Class {i} (AP = {avg_precision:.2f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]], 
                      metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Compare multiple models based on a specific metric.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
            metric: Metric to use for comparison
            
        Returns:
            Dictionary with comparison results
        """
        if not model_results:
            return {}
        
        # Extract metric values
        metric_values = {}
        for model_name, metrics in model_results.items():
            if metric in metrics:
                metric_values[model_name] = metrics[metric]
            else:
                logger.warning(f"Metric {metric} not found for model {model_name}")
        
        if not metric_values:
            return {}
        
        # Find best and worst models
        best_model = max(metric_values, key=metric_values.get)
        worst_model = min(metric_values, key=metric_values.get)
        
        comparison = {
            'metric_used': metric,
            'best_model': best_model,
            'best_score': metric_values[best_model],
            'worst_model': worst_model,
            'worst_score': metric_values[worst_model],
            'model_rankings': sorted(metric_values.items(), key=lambda x: x[1], reverse=True),
            'score_range': float(max(metric_values.values()) - min(metric_values.values())),
            'score_std': float(np.std(list(metric_values.values())))
        }
        
        return comparison
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_prob: Optional[np.ndarray] = None,
                       model_name: str = "Model",
                       manufacturing_metrics: bool = False,
                       cost_matrix: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            manufacturing_metrics: Whether to include manufacturing-specific metrics
            cost_matrix: Cost matrix for cost-sensitive metrics
            
        Returns:
            Formatted evaluation report
        """
        # Calculate all metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob, manufacturing_metrics=manufacturing_metrics, cost_matrix=cost_matrix)
        
        # Perform confusion matrix analysis
        cm_analysis = self.calculate_confusion_matrix_analysis(y_true, y_pred)
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append(f"CLASSIFICATION MODEL EVALUATION REPORT")
        report.append(f"Model: {model_name}")
        report.append("=" * 60)
        report.append("")
        
        # Basic metrics
        report.append("BASIC METRICS:")
        report.append(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.6f}")
        report.append(f"  Precision (Weighted): {metrics.get('precision', 'N/A'):.6f}")
        report.append(f"  Recall (Weighted): {metrics.get('recall', 'N/A'):.6f}")
        report.append(f"  F1 Score (Weighted): {metrics.get('f1', 'N/A'):.6f}")
        report.append(f"  F1 Score (Macro): {metrics.get('f1_macro', 'N/A'):.6f}")
        report.append(f"  Cohen's Kappa: {metrics.get('cohen_kappa', 'N/A'):.6f}")
        report.append(f"  Matthews Correlation Coefficient: {metrics.get('matthews_corrcoef', 'N/A'):.6f}")
        
        if y_prob is not None:
            report.append(f"  ROC AUC: {metrics.get('auc', 'N/A'):.6f}")
            report.append(f"  Average Precision: {metrics.get('average_precision', 'N/A'):.6f}")
            report.append(f"  Log Loss: {metrics.get('log_loss', 'N/A'):.6f}")
        
        report.append("")
        
        # Manufacturing metrics
        if manufacturing_metrics:
            report.append("MANUFACTURING METRICS:")
            if 'defect_detection_accuracy' in metrics:
                report.append(f"  Defect Detection Accuracy: {metrics['defect_detection_accuracy']:.2f}%")
            if 'false_alarm_rate' in metrics:
                report.append(f"  False Alarm Rate: {metrics['false_alarm_rate']:.2f}%")
            if 'miss_rate' in metrics:
                report.append(f"  Miss Rate: {metrics['miss_rate']:.2f}%")
            if 'quality_control_score' in metrics:
                report.append(f"  Quality Control Score: {metrics['quality_control_score']:.2f}")
            if 'production_efficiency' in metrics:
                report.append(f"  Production Efficiency: {metrics['production_efficiency']:.2f}%")
            if 'cost_effectiveness' in metrics:
                report.append(f"  Cost Effectiveness: {metrics['cost_effectiveness']:.2f}%")
            report.append("")
        
        # Confusion matrix analysis
        report.append("CONFUSION MATRIX ANALYSIS:")
        overall_metrics = cm_analysis['overall_metrics']
        report.append(f"  Overall Accuracy: {overall_metrics['accuracy']:.6f}")
        report.append(f"  Macro Precision: {overall_metrics['macro_precision']:.6f}")
        report.append(f"  Macro Recall: {overall_metrics['macro_recall']:.6f}")
        report.append(f"  Macro F1 Score: {overall_metrics['macro_f1']:.6f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS METRICS:")
        for label, class_metrics in cm_analysis['class_metrics'].items():
            report.append(f"  Class {label}:")
            report.append(f"    Precision: {class_metrics['precision']:.6f}")
            report.append(f"    Recall: {class_metrics['recall']:.6f}")
            report.append(f"    F1 Score: {class_metrics['f1_score']:.6f}")
            report.append(f"    Specificity: {class_metrics['specificity']:.6f}")
            report.append(f"    Support: {class_metrics['support']}")
        report.append("")
        
        # Model performance assessment
        report.append("MODEL PERFORMANCE ASSESSMENT:")
        accuracy = metrics.get('accuracy', 0)
        f1 = metrics.get('f1', 0)
        
        if accuracy > 0.95:
            report.append("  Accuracy: EXCELLENT (>0.95)")
        elif accuracy > 0.90:
            report.append("  Accuracy: GOOD (0.90-0.95)")
        elif accuracy > 0.80:
            report.append("  Accuracy: FAIR (0.80-0.90)")
        else:
            report.append("  Accuracy: POOR (<0.80)")
        
        if f1 > 0.90:
            report.append("  F1 Score: EXCELLENT (>0.90)")
        elif f1 > 0.80:
            report.append("  F1 Score: GOOD (0.80-0.90)")
        elif f1 > 0.70:
            report.append("  F1 Score: FAIR (0.70-0.80)")
        else:
            report.append("  F1 Score: POOR (<0.70)")
        
        if manufacturing_metrics and 'quality_control_score' in metrics:
            qc_score = metrics['quality_control_score']
            if qc_score > 90:
                report.append("  Quality Control Score: EXCELLENT (>90)")
            elif qc_score > 80:
                report.append("  Quality Control Score: GOOD (80-90)")
            elif qc_score > 70:
                report.append("  Quality Control Score: FAIR (70-80)")
            else:
                report.append("  Quality Control Score: POOR (<70)")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
