import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from src.models.model import build_model
from src.data_prep import test_dataset, test_transform
from src.evaluation import (
    calculate_macro_f1_score,
    print_evaluation_summary,
    evaluate_all_metrics,
    get_per_class_f1_table
)

# Configuration
MODEL_PATH = Path('experiments_finetuned/best_model.pt')  # Best fine-tuned model
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if test dataset has data
if len(test_dataset) == 0:
    raise ValueError("No test data found! Make sure the pipeline has been run.")

# Get number of classes and class names
num_classes = len(test_dataset.class_names)
class_names = test_dataset.class_names
print(f"\nTest Dataset Info:")
print(f"  Number of classes: {num_classes}")
print(f"  Classes: {class_names}")
print(f"  Test samples: {len(test_dataset)}")

# Load the model checkpoint
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train a model first!")

print(f"\nLoading model from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Build model and load weights
model = build_model("resnet18", num_classes=num_classes, input_channels=1, small_input=True)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()  # Set to evaluation mode

print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
if 'val_f1' in checkpoint:
    print(f"Validation F1 (from training): {checkpoint.get('val_f1', 0.0):.4f}")

# Create test data loader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

print("\n" + "="*60)
print("Evaluating on Test Set...")
print("="*60)
print()

# Evaluate on test set
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Processed {batch_idx + 1} batches...")

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate metrics
print("\n" + "="*60)
print("TEST SET EVALUATION RESULTS")
print("="*60)
print()

# Print detailed evaluation summary
print_evaluation_summary(
    y_true=all_labels,
    y_pred=all_preds,
    class_names=class_names
)

# Get additional metrics
results = evaluate_all_metrics(
    y_true=all_labels,
    y_pred=all_preds,
    class_names=class_names
)

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Test Accuracy:        {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
print(f"Test Macro F1:        {results['macro_f1']:.4f} ({results['macro_f1']*100:.2f}%)")
print(f"Test Weighted F1:     {results['weighted_f1']:.4f} ({results['weighted_f1']*100:.2f}%)")
print()

# Show per-class performance sorted by F1 score
print("Per-Class Performance (sorted by F1 score):")
print("-" * 60)
f1_table = get_per_class_f1_table(
    y_true=all_labels,
    y_pred=all_preds,
    class_names=class_names
)
f1_table_sorted = f1_table.sort_values('F1 Score', ascending=False)
print(f1_table_sorted.to_string(index=False))
print()

# Identify best and worst performing classes
best_class = f1_table_sorted.iloc[0]
worst_class = f1_table_sorted.iloc[-1]
print(f"Best performing class:  {best_class['Class']} (F1: {best_class['F1 Score']:.4f})")
print(f"Worst performing class: {worst_class['Class']} (F1: {worst_class['F1 Score']:.4f})")
print()

# Save results to file
results_dir = Path('evaluation_results')
results_dir.mkdir(exist_ok=True)
results_file = results_dir / 'test_evaluation_results.txt'

with open(results_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("TEST SET EVALUATION RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test samples: {len(test_dataset)}\n")
    f.write(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
    f.write(f"Test Macro F1: {results['macro_f1']:.4f} ({results['macro_f1']*100:.2f}%)\n")
    f.write(f"Test Weighted F1: {results['weighted_f1']:.4f} ({results['weighted_f1']*100:.2f}%)\n\n")
    f.write("Per-Class F1 Scores:\n")
    f.write("-" * 60 + "\n")
    f.write(f1_table_sorted.to_string(index=False))
    f.write("\n\n")
    f.write("Confusion Matrix:\n")
    f.write("-" * 60 + "\n")
    f.write(str(results['confusion_matrix']))
    f.write("\n")

print(f"Results saved to: {results_file}")
print("="*60)
