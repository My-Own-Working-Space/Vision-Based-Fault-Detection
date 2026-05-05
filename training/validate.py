import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.models as tv_models
from torchvision import transforms
from tqdm import tqdm

def train_and_validate(train_dataset, val_dataset, loss_function, device, num_epochs):
    """
    Trains and validates a machine learning model on a given dataset using Transfer Learning.
    """
    # Define the mean and standard deviation for ImageNet normalization.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Create a composition of transforms for the training data, including augmentation.
    # We use 224x224 as required by MobileNetV3
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # Create a composition of transforms for the validation data.
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # Apply the defined transformations to the datasets.
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform

    # Set the batch size for the DataLoaders.
    batch_size = 32
    
    # Create a DataLoader for the training dataset.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create a DataLoader for the validation dataset.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize MobileNetV3 Small with pre-trained weights from torchvision
    print("--- Loading Pre-trained MobileNetV3 Small ---")
    weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT
    model = tv_models.mobilenet_v3_small(weights=weights)
    
    # Determine the number of classes from the training dataset.
    num_classes = len(train_dataset.classes)
    
    # Replace the final classifier layer to match the number of classes (2: Clean vs Dirt)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    # Move the model to the specified compute device.
    model.to(device)

    # Use AdamW with a slightly lower learning rate for transfer learning
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Cosine Annealing scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    
    print(f"--- Starting Transfer Learning (MobileNetV3) for {num_epochs} epochs ---")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = loss_function(outputs, labels)
                running_val_loss += val_loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_accuracy = 100.0 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")
        
        scheduler.step()
        
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state:
        print(f"\n--- Best model: {best_val_accuracy:.2f}% accuracy at epoch {best_epoch} ---")
        model.load_state_dict(best_model_state)
    
    return model