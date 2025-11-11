import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import SubsetRandomSampler
import json

# --- 1. Configuration ---
DATA_DIR = '/home/yeling/dog_project/data'
NUM_EPOCHS = 40  # Increased epochs for better convergence
BATCH_SIZE = 32
LEARNING_RATE = 0.0003  # Lowered for stability
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = '/home/yeling/dog_project/dog_pose_efficientnet_b0.pth'
CLASSES_FILE = '/home/yeling/dog_project/dog_pose_classes.txt'
WEIGHTS_PATH = '/home/yeling/.cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth'

# --- 2. Validate Data Directory ---
def validate_data_dir(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Error: Directory {data_dir} does not exist")
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not classes:
        raise ValueError(f"Error: No class subdirectories found in {data_dir}")
    print(f"Data directory contains classes: {classes}")
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        num_files = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"data/{cls}: {num_files} images")
    return classes

class_names = validate_data_dir(DATA_DIR)

# --- 3. Data Preprocessing ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),  # Increased rotation
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # Enhanced augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load and split dataset
try:
    dataset = datasets.ImageFolder(DATA_DIR, data_transforms['train'])
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(42)  # Fixed seed for reproducibility
    np.random.shuffle(indices)
    train_size = int(0.8 * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4),
        'val': torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4)
    }
    dataset_sizes = {'train': len(train_indices), 'val': len(val_indices)}
except Exception as e:
    print(f"Error: Failed to load dataset: {e}")
    exit(1)

# Check class distribution
class_counts = {class_name: len(os.listdir(os.path.join(DATA_DIR, class_name))) for class_name in class_names}
print("Dataset class distribution:", class_counts)
print(f"Training set: {dataset_sizes['train']} images, Validation set: {dataset_sizes['val']} images")

# --- 4. Initialize EfficientNet-B0 ---
try:
    if os.path.exists(WEIGHTS_PATH):
        model = EfficientNet.from_name('efficientnet-b0')
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"Loaded local weights from {WEIGHTS_PATH}")
    else:
        print(f"Error: Local weights not found at {WEIGHTS_PATH}, please download manually")
        exit(1)
except Exception as e:
    print(f"Error: Failed to load EfficientNet-B0: {e}")
    exit(1)

num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(DEVICE)

# Unfreeze more layers for better performance
for name, param in model.named_parameters():
    if '_conv_head' in name or '_fc' in name or '_block' in name:  # Unfreeze some conv layers
        param.requires_grad = True
    else:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Adjusted decay

# --- 5. Training Function ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_accs = []
    val_accs = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

            if phase == 'train':
                train_accs.append(epoch_acc.item())
            else:
                val_accs.append(epoch_acc.item())
                val_f1_scores.append(epoch_f1)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} F1: {val_f1_scores[-1]:.4f}')

    model.load_state_dict(best_model_wts)
    
    # Save accuracy plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(num_epochs), train_accs, label='Training Accuracy')
    plt.plot(range(num_epochs), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('EfficientNet-B0 Training and Validation Accuracy')
    plt.legend()
    plt.savefig('/home/yeling/dog_project/accuracy_efficientnet_b0.png')
    plt.close()
    
    return model, best_acc, val_f1_scores[-1], time_elapsed

# --- 6. Train and Save ---
try:
    model, best_acc, best_f1, train_time = train_model(model, criterion, optimizer, scheduler)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(CLASSES_FILE, 'w', encoding='utf-8') as f:
        json.dump({name: idx for idx, name in enumerate(class_names)}, f, ensure_ascii=False, indent=4)
    print(f"Model saved to {MODEL_PATH}, Classes saved to {CLASSES_FILE}")
    print(f"Validation Accuracy: {best_acc:.4f}, F1-score: {best_f1:.4f}, Training Time: {train_time:.2f} seconds")
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

    