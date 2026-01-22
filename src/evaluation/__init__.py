"""
This module provides comprehensive evaluation metrics including:
- Accuracy
- Macro F1 (main metric)
- Weighted F1
- Confusion matrix
- Per-class F1 scores

Main functions:
    evaluate_all_metrics: Compute all metrics at once
    get_per_class_f1_table: Get per-class F1 scores as a table
    print_evaluation_summary: Print formatted summary of all metrics

Individual metric functions:
    calculate_accuracy: Calculate accuracy score
    calculate_macro_f1_score: Calculate macro-averaged F1 (main metric)
    calculate_weighted_f1_score: Calculate weighted-averaged F1
    calculate_confusion_matrix: Calculate confusion matrix
    calculate_f1_scores_per_class: Calculate per-class F1 scores
"""

# Accuracy
from .accuracy import calculate_accuracy

# F1 Scores
from .f1_score import (
    calculate_f1_score,
    calculate_macro_f1_score,
    calculate_weighted_f1_score,
    calculate_f1_scores_per_class
)

# Confusion Matrix
from .confusion_matrix import (
    calculate_confusion_matrix,
    calculate_confusion_matrix_normalized,
    get_confusion_matrix_metrics
)

# Precision and Recall (available but not explicitly requested)
from .precision_recall import (
    calculate_precision,
    calculate_recall,
    calculate_precision_per_class,
    calculate_recall_per_class,
    calculate_macro_precision,
    calculate_macro_recall
)

# Comprehensive evaluation module
from .metrics import (
    evaluate_all_metrics,
    get_per_class_f1_table,
    print_evaluation_summary
)

__all__ = [
    # Main evaluation functions
    'evaluate_all_metrics',
    'get_per_class_f1_table',
    'print_evaluation_summary',
    
    # Accuracy
    'calculate_accuracy',
    
    # F1 Scores
    'calculate_f1_score',
    'calculate_macro_f1_score',  # Main metric
    'calculate_weighted_f1_score',
    'calculate_f1_scores_per_class',
    
    # Confusion Matrix
    'calculate_confusion_matrix',
    'calculate_confusion_matrix_normalized',
    'get_confusion_matrix_metrics',
    
    # Precision and Recall 
    'calculate_precision',
    'calculate_recall',
    'calculate_precision_per_class',
    'calculate_recall_per_class',
    'calculate_macro_precision',
    'calculate_macro_recall',
]
