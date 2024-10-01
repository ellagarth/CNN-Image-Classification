import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------
# CONFIGURATION VARIABLES FOR CNN IMAGE CLASSIFICATION
# -----------------------
image_size = 10       # number of pixels (e.g. 10x10 pixels)
output_neurons = 4    # 4 output classes (circle, square, triangle, cross)

# Activation functions for hidden layer (choose between 'relu', 'leaky_relu', 'sigmoid', 'tanh')
activation_function = 'leaky_relu'
# output layer is softmax to convert to probabilities for multi-class classification

# Maximum epochs for training
num_epochs = 1000
learning_rate = 0.00001
batch_size = 1
use_early_stopping = False
# Patience for early stopping (only used if early stopping is enabled)
patience = 1000
use_lr_scheduler = False
# Dropout rate (set to 0.0 if no dropout is required)
dropout_rate = 0.0
# set constant random seed to give consistent results
torch.manual_seed(42)
np.random.seed(42)

# -----------------------
# END CONFIGURATION
# -----------------------

##############################################
### LAYERS ###

# 1. Convolutional layer (Conv2d)
# 2. Activation layer (ReLU, etc.)
# 3. Pooling layer (MaxPool2d, AvgPool2d)
# 4. Fully connected (dense) layer (Linear)
# 5. Flattening layer
# 6. Droppout layer
# 7. Output layer (softmax for classification, linear activation for regression)

# summary: CNNs use the convolutional and pooling layers to automatically extract relevant 
# features and the fully connected layers to classify those features.
#############################################

# Set device (GPU preferable for parallel processing like deep learning models)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Activation function choices
def get_activation_function(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Invalid activation function name: {name}")

# Custom dataset class
class ShapeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Load image as grayscale
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# Image transformations (resize image to pixel values set at top of script and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# Load the dataset
train_dataset = ShapeDataset(
    csv_file='/Users/ellagarth/Desktop/Portfolio3/Training_Images/training.csv',
    img_dir='/Users/ellagarth/Desktop/Portfolio3/Training_Images/',
    transform=transform
)

# Split the data into training (80%) and validation (20%) sets
train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)

# Use PyTorch DataLoaders with adjustable batch size
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Convolutional Neural Network (CNN) class
class CNN(nn.Module):
    def __init__(self, output_neurons, dropout_rate, activation_function):
        super(CNN, self).__init__()

        # First Convolutional Layer: 1 input channel (grayscale image), 16 output channels, 3x3 filter size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)

        # Second Convolutional Layer: 16 input channels, 32 output channels, 3x3 filter size
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        # Pooling Layer: Reduces the size of the feature map by 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout (if required)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else None

        # Activation function
        self.activation_function = get_activation_function(activation_function)

        # Fully connected layer (we will calculate input size dynamically)
        self.fc1 = None  # To be initialized later dynamically
        self.fc2 = nn.Linear(128, output_neurons)

    def forward(self, x):
        # Apply first convolution, activation, and pooling
        x = self.pool(self.activation_function(self.conv1(x)))

        # Apply second convolution, activation, and pooling
        x = self.pool(self.activation_function(self.conv2(x)))

        # Dynamically calculate the size of the flattened feature maps
        if self.fc1 is None:
            # Calculate the flattened size from the feature map
            flattened_size = x.view(x.size(0), -1).shape[1]
            # Define the fully connected layer with the correct flattened size
            self.fc1 = nn.Linear(flattened_size, 128)

        # Flatten the feature maps into a vector for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten

        # Apply fully connected layer and activation
        x = self.activation_function(self.fc1(x))

        # Apply dropout (if required)
        if self.dropout:
            x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        return x


# Initialise the CNN model, loss function, and optimiser
model = CNN(output_neurons=output_neurons, dropout_rate=dropout_rate, activation_function=activation_function).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
if use_lr_scheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Early stopping parameters
best_val_loss = float('inf')
epochs_no_improve = 0

# Track training loss for visualisation
loss_history = []

# Training loop with loss printed every 100 epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

    # Save the loss history for plotting
    loss_history.append(running_loss / len(train_loader))

    # Learning rate scheduler step
    if use_lr_scheduler:
        scheduler.step(val_loss)

    # Early stopping
    if use_early_stopping:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Plot the training loss over time
plt.figure()
plt.plot(loss_history)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Load the test dataset
test_dataset = ShapeDataset(
    csv_file='/Users/ellagarth/Desktop/Portfolio3/Test_Images/test.csv',
    img_dir='/Users/ellagarth/Desktop/Portfolio3/Test_Images',
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the network on test data
model.eval()
all_labels = []
all_preds = []
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Print the accuracy
print(f'Accuracy on test images: {100 * correct / total:.2f}%')

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Circle', 'Square', 'Triangle', 'Cross']))

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Circle', 'Square', 'Triangle', 'Cross'], yticklabels=['Circle', 'Square', 'Triangle', 'Cross'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
