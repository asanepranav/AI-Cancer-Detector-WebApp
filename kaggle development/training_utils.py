# training_utils.py

import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Utility script 'training_utils.py' loaded.")

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one full training epoch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Use tqdm for a progress bar
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()

        # --- NEW: Gradient Clipping ---
        # Clip gradients to a maximum norm of 1.0 to prevent them from exploding,
        # which helps stabilize training for complex models.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # --- END OF NEW CODE ---

        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        
        # Convert outputs to predictions (0 or 1)
        preds = torch.sigmoid(outputs).round()
        # Detach tensors from the computation graph before converting to numpy
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs one full validation epoch.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradients needed for validation
        progress_bar = tqdm(loader, desc="Validating", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            
            preds = torch.sigmoid(outputs).round()
            # It's good practice to detach here as well, though not strictly necessary inside no_grad()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    
    # Calculate all metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }
    
    return epoch_loss, metrics

