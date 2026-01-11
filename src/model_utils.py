"""
Model Training and Evaluation Utilities
Home Credit Default Risk Project

This module contains helper functions for:
- Model evaluation metrics calculation
- Visualization (ROC curve, PR curve, confusion matrix)
- Model calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    f1_score, precision_score, recall_score, 
    accuracy_score, brier_score_loss,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)


def calculate_all_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Tính toán tất cả các metrics đánh giá model
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    threshold : float, default=0.5
        Classification threshold
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all calculated metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_pred_proba)
    }
    
    return metrics


def print_metrics(metrics, title="Model Metrics"):
    """
    In ra các metrics một cách đẹp mắt
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics from calculate_all_metrics()
    title : str, default="Model Metrics"
        Title to display above metrics
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"ROC AUC Score:     {metrics['roc_auc']:.4f}")
    print(f"PR AUC Score:      {metrics['pr_auc']:.4f}")
    print(f"F1 Score:          {metrics['f1']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Brier Score:       {metrics['brier']:.4f}")
    print(f"{'='*60}")


def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Vẽ ROC curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str, default="Model"
        Name of the model for plot title
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Vẽ Precision-Recall curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str, default="Model"
        Name of the model for plot title
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'{model_name} (PR AUC = {pr_auc:.4f})')
    plt.axhline(y=y_true.sum()/len(y_true), color='k', linestyle='--', linewidth=1, 
                label=f'Baseline ({y_true.sum()/len(y_true):.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Vẽ confusion matrix với cả số lượng và phần trăm
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels (not probabilities)
    model_name : str, default="Model"
        Name of the model for plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create labels with both count and percentage
    labels = np.asarray([[f'{count}\n({percent:.1f}%)' 
                         for count, percent in zip(row_counts, row_percents)]
                        for row_counts, row_percents in zip(cm, cm_percent)])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True,
                xticklabels=['No Default (0)', 'Default (1)'],
                yticklabels=['No Default (0)', 'Default (1)'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # In thêm metrics từ confusion matrix
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    print(f"\n{'='*60}")
    print(f"Confusion Matrix Details:")
    print(f"{'='*60}")
    print(f"True Negatives:  {tn:>7} ({tn/total*100:>5.2f}%)")
    print(f"False Positives: {fp:>7} ({fp/total*100:>5.2f}%)")
    print(f"False Negatives: {fn:>7} ({fn/total*100:>5.2f}%)")
    print(f"True Positives:  {tp:>7} ({tp/total*100:>5.2f}%)")
    print(f"{'='*60}")


def save_model_results(model_name, model, y_true, y_pred_proba, results_dict, save_path=None, show_charts=True):
    """
    Lưu model, tính toán và in metrics, classification report, vẽ charts
    
    Parameters:
    -----------
    model_name : str
        Tên của model
    model : model object
        Trained model
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    results_dict : dict
        Dictionary để lưu kết quả metrics
    save_path : str, optional
        Path to save model file
    show_charts : bool, default=True
        Whether to show ROC, PR, and Confusion Matrix charts
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all calculated metrics
    """
    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred_proba)
    
    # Classification report
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    print(f"\n📊 Classification Report:")
    print(classification_report(y_true, y_pred_class, target_names=['No Default (0)', 'Default (1)']))
    
    # Print metrics
    print_metrics(metrics, f"{model_name} - Test Set")
    
    # Vẽ các charts nếu show_charts=True
    if show_charts:
        print(f"\n📈 Generating visualizations for {model_name}...")
        
        # ROC Curve
        plot_roc_curve(y_true, y_pred_proba, model_name)
        
        # Precision-Recall Curve
        plot_precision_recall_curve(y_true, y_pred_proba, model_name)
        
        # Confusion Matrix
        plot_confusion_matrix(y_true, y_pred_class, model_name)
    
    # Save metrics
    results_dict[model_name] = metrics
    
    # Save model
    if save_path:
        joblib.dump(model, save_path)
        print(f"\n✓ Model saved to '{save_path}'")
    
    return metrics
