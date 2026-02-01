import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from src.models.model import build_model
from src.data_prep import train_dataset, val_dataset, train_transform, test_transform
from src.evaluation import calculate_macro_f1_score, print_evaluation_summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU cache to free memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check if datasets have data
if len(train_dataset) == 0:
    raise ValueError("No training data found! Make sure the pipeline has been run.")
if len(val_dataset) == 0:
    raise ValueError("No validation data found! Make sure the pipeline has been run.")

# Get number of classes and class names
num_classes = len(train_dataset.class_names)
class_names = train_dataset.class_names
print(f"Number of classes: {num_classes}")
print(f"Classes: {class_names}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Build model 
model = build_model("resnet18", num_classes=num_classes, input_channels=1, small_input=True)
model = model.to(device)

# Standard loss (no class weights - they seemed to hurt performance)
criterion = nn.CrossEntropyLoss()

# Improved: Lower learning rate and add weight decay for regularization
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

# Improved: Learning rate scheduler - reduce LR when plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
)

# Batch size - reduced to 16 for shared GPU memory
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

print("\nStarting improved training v2 (without class weights)...\n")
print("Improvements:")
print("  - Lower learning rate (5e-4)")
print("  - Weight decay regularization (1e-4)")
print("  - Learning rate scheduler (ReduceLROnPlateau)")
print("  - Standard loss (no class weights)")
print("  - Gradient clipping")
print(f"  - Batch size ({batch_size})")
print("  - Memory management")
print()

# Create save directory
save_dir = Path('experiments_improved_v2')
save_dir.mkdir(exist_ok=True)
checkpoints_dir = save_dir / 'checkpoints'
checkpoints_dir.mkdir(exist_ok=True)

num_epochs = 150 
best_f1 = 0.0
best_epoch = 0
patience = 15  
patience_counter = 0

for epoch in range(num_epochs):
    
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    train_preds = []
    train_labels = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        # Improved: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Clear cache periodically to free memory
        if (total + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        train_preds.extend(pred.cpu().numpy())
        train_labels.extend(y.cpu().numpy())

    train_loss = total_loss / total
    train_acc = correct / total
    train_f1 = calculate_macro_f1_score(train_labels, train_preds)

    # Validation 
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            val_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)
            
            val_preds.extend(pred.cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_f1 = calculate_macro_f1_score(val_labels, val_preds)
    
    # Improved: Update learning rate based on validation F1
    scheduler.step(val_f1)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro F1: {train_f1:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro F1: {val_f1:.4f}")
    print(f"Learning Rate: {current_lr:.6f}")

    # Print detailed evaluation
    print_detailed_summary = (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1
    
    if print_detailed_summary:
        print("\n" + "="*60)
        print(f"Validation Set Evaluation - Epoch {epoch+1}")
        print("="*60)
        print_evaluation_summary(
            y_true=np.array(val_labels),
            y_pred=np.array(val_preds),
            class_names=class_names
        )
        print()

    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'num_classes': num_classes,
        'class_names': class_names,
    }
    
    checkpoint_path = checkpoints_dir / f'checkpoint_epoch_{epoch+1}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately if this is the best F1 score
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch + 1
        patience_counter = 0 
        best_checkpoint_path = save_dir / 'best_model.pt'
        torch.save(checkpoint, best_checkpoint_path)
        print(f"  → Saved best model (Macro F1: {best_f1:.4f}) to {best_checkpoint_path}")
    else:
        patience_counter += 1
        print(f"  → Saved checkpoint to {checkpoint_path}")
    
    # Early stopping: stop if no improvement for 'patience' epochs
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
        print(f"Best validation Macro F1 was {best_f1:.4f} at epoch {best_epoch}")
        break

print(f"\nTraining complete!")
print(f"Best Validation Macro F1: {best_f1:.4f}")
print(f"Best model saved to: {save_dir / 'best_model.pt'}")
print(f"All checkpoints saved to: {checkpoints_dir}")
