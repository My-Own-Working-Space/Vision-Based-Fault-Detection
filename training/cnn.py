import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
import helper_utils as helper_utils

TRAIN_PATH = "../dataset/train"
VAL_PATH = "../dataset/valid"
MODEL_SAVE_PATH = "../models/insulator_cnn.pth"
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} ")

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def define_transformation(mean, std):
    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transformations, val_transformations

print("--- Verifying define_transformations: ---\n ")
train_transform_verify, val_transform_verify = define_transformation(mean, std)

print("Training Transformations:")
print(train_transform_verify)
print("-" * 30)
print("\nValidation Transformations:")
print(val_transform_verify)

train_transform, val_transform = define_transformation(mean, std)

all_target_classes = ['Clean-Insulator', 'Dirt-Insulator']

train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VAL_PATH, transform=val_transform)

batch_size = BATCH_SIZE

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

helper_utils.visualise_images(train_loader, grid=(3, 5))

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.block(x)

print("\n --- Verifying CNNBlock: ---\n")

verify_cnn_block = CNNBlock(in_channels=3, out_channels=16)
print("Block Structure:\n")
print(verify_cnn_block)

dummy_input = torch.randn(1, 3, 32, 32)
print(f"\nInput tensor shape:  {dummy_input.shape}")

output = verify_cnn_block(dummy_input)
print(f"Output tensor shape: {output.shape}")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128*4*4, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

print("--- Verifying SimpleCNN ---\n")

verify_simple_cnn = SimpleCNN(num_classes=15)
print("Model Structure:\n")
print(verify_simple_cnn)

dummy_input = torch.randn(64, 3, 32, 32)
print(f"\nInput tensor shape:  {dummy_input.shape}")

output = verify_simple_cnn(dummy_input)
print(f"Output tensor shape: {output.shape}")

num_classes = len(train_dataset.classes)
model=SimpleCNN(num_classes=num_classes)

def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss= 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0) 

    return running_loss / len(train_loader)

helper_utils.verify_training_process(SimpleCNN, train_loader, loss_function, optimizer, device)

def validate_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss= 0.0
    correct= 0 
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(val_loader), correct / total

helper_utils.verify_validation_process(SimpleCNN, val_loader, loss_function, device)

def training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    """
    Trains and validates a PyTorch neural network model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        loss_function (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        num_epochs (int): The total number of epochs to train for.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run training on.

    Returns:
        tuple: A tuple containing the best trained model and a list of metrics
               (train_losses, val_losses, val_accuracies).
    """
    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    
    # Initialize variables to track the best performing model
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Initialize lists to store training and validation metrics
    train_losses, val_losses, val_accuracies = [], [], []
    
    print("--- Training Started ---")
    
    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        # Perform one epoch of training
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        train_losses.append(epoch_loss)
        
        # Perform one epoch of validation
        epoch_val_loss, epoch_accuracy = validate_epoch(model, val_loader, loss_function, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)
        
        # Print the metrics for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")
        
        # Check if the current model is the best one so far
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            # Save the state of the best model in memory
            best_model_state = copy.deepcopy(model.state_dict())
            
    print("--- Finished Training ---")
    
    # Load the best model weights before returning
    if best_model_state:
        print(f"\n--- Returning best model with {best_val_accuracy:.2f}% validation accuracy, achieved at epoch {best_epoch} ---")
        model.load_state_dict(best_model_state)
    
    # Consolidate all metrics into a single list
    metrics = [train_losses, val_losses, val_accuracies]
    
    # Return the trained model and the collected metrics
    return model, metrics

# Start the training process by calling the training loop function
trained_model, training_metrics = training_loop(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    loss_function=loss_function, 
    optimizer=optimizer, 
    num_epochs=50, 
    device=device
)

# Visualize the training metrics (loss and accuracy)
print("\n--- Training Plots ---\n")
helper_utils.plot_training_metrics(training_metrics)

# Import the preview function that demonstrates concepts from the next course
from c2_preview.c2_preview import course_2_preview

# This helper function runs a training loop using a powerful strategy that will be taught
# in the next course. Run this cell to see the improved results in action.
trained_model = course_2_preview(
    train_dataset, 
    val_dataset, 
    loss_function,
    device,
    num_epochs=5
    )