import os
from datetime import datetime
import torch
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import inspect

from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

from src.utils import PHEMEDataset
from src.models.gcn import KMGCN

def train_gcn(dataset, 
              hidden_dim=64, 
              epochs=100, 
              lr=0.001, 
              pool='max',
              batch_size=32, 
              test_split=0.2, 
              random_seed=42,
              early_stopping=10,
              output_dir='./outputs'):
    """
    Train the GCN with enhanced experiment tracking and unique folder creation
    """
    # Create unique experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Determine input dimension from the first graph
    input_dim = dataset[0].x.shape[1]
    
    # Determine number of classes (assuming binary classification)
    num_classes = 2
    
    # Split dataset into train and test
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_split, 
        random_state=random_seed, 
        stratify=[dataset[i].y.item() for i in indices]
    )
    
    # Create data loaders
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = KMGCN(input_dim, hidden_dim, num_classes, pool)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Tracking metrics
    train_losses, val_losses, accuracies = [], [], []
    precisions, recalls, f1_scores = [], [], []
    
    # Tracking best model
    best_accuracy = 0
    best_epoch = 0
    
    # Create a main progress bar for epochs
    epochs_progress = tqdm(
        range(epochs), 
        desc="Training Progress", 
        position=0
    )
    
    # Training loop
    for epoch in epochs_progress:
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                loss = criterion(out, batch.y.squeeze())
                total_val_loss += loss.item()
                
                pred = out.argmax(dim=1)
                correct += (pred == batch.y.squeeze()).sum().item()
                total += batch.y.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.squeeze().cpu().numpy())
        
        # Calculate metrics
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(test_loader)
        accuracy = correct / total
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Update epochs progress bar
        epochs_progress.set_postfix({
            'Acc': f'{accuracy:.4f}', 
            'TLoss': f'{train_loss:.4f}', 
            'VLoss': f'{val_loss:.4f}',
            'F1': f'{f1:.4f}'
        })
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_model = model.state_dict().copy()
        
        # Early stopping
        if early_stopping and epoch - best_epoch > early_stopping:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break
    
    # Close progress bar
    epochs_progress.close()
    
    # Create unique experiment folder
    exp_folder_name = f"{timestamp}_acc_{best_accuracy:.4f}"
    exp_output_dir = os.path.join(output_dir, exp_folder_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Prepare experiment log
    experiment_log = {
        'configuration': {
            'hidden_dim': hidden_dim,
            'epochs': epochs,
            'learning_rate': lr,
            'pool_method': pool,
            'batch_size': batch_size,
            'test_split': test_split,
            'random_seed': random_seed
        },
        'results': {
            'best_accuracy': best_accuracy,
            'best_epoch': best_epoch,
            'final_precision': float(precision),
            'final_recall': float(recall),
            'final_f1_score': float(f1)
        },
        'training_curves': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'accuracies': accuracies,
            'precisions': precisions,
            'recalls': recalls,
            'f1_scores': f1_scores
        }
    }
    
    # Save experiment log as JSON
    log_path = os.path.join(exp_output_dir, f'{exp_folder_name}_experiment_log.json')
    with open(log_path, 'w') as f:
        json.dump(experiment_log, f, indent=4)
    
    # Save best model
    best_model_path = os.path.join(exp_output_dir, f'{exp_folder_name}_best_model.pth')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model,
        'accuracy': best_accuracy
    }, best_model_path)
    
    # Plotting (accuracy)
    plt.figure(figsize=(15, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(accuracies, label='Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Train vs Val Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(exp_output_dir, f'{exp_folder_name}_training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    
    return {
        'model': model,
        'best_epoch': best_epoch,
        'best_accuracy': best_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'accuracies': accuracies
    }

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Train Graph Convolutional Network")

    # Add attributes with more descriptive help text
    parser.add_argument('--hidden_dim', type=int, default=64, 
                        help='Number of hidden units in the GCN')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate for optimization')
    parser.add_argument('--pool', type=str, default='max', 
                        help='Either use max or average pooling')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--test_split', type=float, default=0.4, 
                        help='Proportion of dataset to use for testing')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--early_stopping', type=int, default=None, 
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--output_dir', type=str, default='./outputs', 
                        help='Directory to save output files')
    parser.add_argument('--embedding',type=str,default='bert',
                        help='The word embedding to use : either bert or w2v')
    parser.add_argument('--multimodality',type=bool, default=True,
                        help='Either to use multimodality or not')

    # Parse the arguments
    args = parser.parse_args()

    # Set up logging
    print("Training Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    #Get parameters of each function
    train_specific_args = {k: v for k, v in vars(args).items() if k in list(inspect.signature(train_gcn).parameters)}
    PHEMEDataset_args = {k: v for k, v in vars(args).items() if k in list(inspect.signature(PHEMEDataset).parameters)}

    # Create the dataset
    print("\nCreating PHEME dataset ...")
    pheme_dataset = PHEMEDataset(**PHEMEDataset_args)
    
    # Training
    print("\nStarting training ...")
    train_results = train_gcn(
        dataset=pheme_dataset,
        **train_specific_args
    )

    # Print final results
    print("\nTraining Complete!")
    print(f"Best Accuracy: {train_results['best_accuracy']:.4f}")
    print(f"Precision: {train_results['precision']:.4f}")
    print(f"Recall: {train_results['recall']:.4f}")
    print(f"F1 Score: {train_results['f1_score']:.4f}")
    print(f"Best Epoch: {train_results['best_epoch']}")

if __name__ == '__main__':
    main()