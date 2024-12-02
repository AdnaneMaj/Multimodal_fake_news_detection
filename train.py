import os
import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt

from src.utils import PHEMEDataset
from src.models import KMGCN

def train_gcn(dataset, 
              hidden_dim=64, 
              epochs=100, 
              lr=0.001, 
              batch_size=32, 
              test_split=0.2, 
              random_seed=42,
              early_stopping=10,
              output_dir='./outputs'):
    """
    Train the GCN for graph-level classification with improved tracking and logging
    
    Args:
    dataset (PHEMEDataset): PHEME dataset
    hidden_dim (int): Dimension of hidden layers
    epochs (int): Number of training epochs
    lr (float): Learning rate
    batch_size (int): Batch size for training
    test_split (float): Proportion of dataset to use for testing
    random_seed (int): Random seed for reproducibility
    early_stopping (int, optional): Number of epochs to wait for improvement
    output_dir (str): Directory to save outputs
    
    Returns:
    dict: Training results including model, best epoch, and metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    model = KMGCN(input_dim, hidden_dim, num_classes)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Tracking metrics
    train_losses = []
    val_losses = []
    accuracies = []
    
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
        
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                loss = criterion(out, batch.y.squeeze())
                total_val_loss += loss.item()
                
                pred = out.argmax(dim=1)
                correct += (pred == batch.y.squeeze()).sum().item()
                total += batch.y.size(0)
        
        # Calculate metrics
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(test_loader)
        accuracy = correct / total
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Update epochs progress bar with metrics
        epochs_progress.set_postfix({
            'Acc': f'{accuracy:.4f}', 
            'TLoss': f'{train_loss:.4f}', 
            'VLoss': f'{val_loss:.4f}'
        })
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            
            # Save best model checkpoint
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': accuracy
            }, best_model_path)
            
            # Also save the current best model in memory
            best_model = model.state_dict().copy()
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': accuracy
            }, checkpoint_path)
        
        # Early stopping
        if early_stopping and epoch - best_epoch > early_stopping:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break
    
    # Close progress bar
    epochs_progress.close()
    
    # Plotting training metrics
    plt.figure(figsize=(12, 4))
    
    # Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    # Restore best model
    model.load_state_dict(best_model)
    
    return {
        'model': model,
        'best_epoch': best_epoch,
        'best_accuracy': best_accuracy,
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
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--test_split', type=float, default=0.2, 
                        help='Proportion of dataset to use for testing')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--early_stopping', type=int, default=None, 
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--output_dir', type=str, default='./outputs', 
                        help='Directory to save output files')

    # Parse the arguments
    args = parser.parse_args()

    # Set up logging
    print("Training Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Create the dataset
    print("\nCreating PHEME dataset ...")
    pheme_dataset = PHEMEDataset()
    
    # Training
    print("\nStarting training ...")
    train_results = train_gcn(
        dataset=pheme_dataset,
        **vars(args)
    )

    # Print final results
    print("\nTraining Complete!")
    print(f"Best Accuracy: {train_results['best_accuracy']:.4f}")
    print(f"Best Epoch: {train_results['best_epoch']}")

if __name__ == '__main__':
    main()