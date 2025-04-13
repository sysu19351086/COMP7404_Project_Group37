import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np  # Add support for NumPy

class LocalClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LocalClassifier, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, 64)
        # Second fully connected layer
        self.fc2 = nn.Linear(64, 32)
        # Output layer
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Apply ReLU activation to the first layer
        x = torch.relu(self.fc1(x))
        # Apply ReLU activation to the second layer
        x = torch.relu(self.fc2(x))
        # Output layer (no activation function as CrossEntropyLoss includes softmax)
        x = self.fc3(x)
        return x

def train_local_classifier(inputs, labels, num_classes, n_epochs, device):
    # Initialize local classifier
    model = LocalClassifier(inputs.shape[1], num_classes).to(device)
    # Define loss function (CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()
    # Define optimizer (Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the local classifier
    for epoch in range(n_epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

    return model

def LCkNNModel(x_train, y_train, x_test, k=5, n_epochs=100, device=None):
    # Automatically select device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if inputs are NumPy arrays and convert to PyTorch tensors if necessary
    if isinstance(x_train, np.ndarray):
        x_train = torch.from_numpy(x_train).float()  # Convert to float32 type
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train).long()   # Convert to long type (classification labels)
    if isinstance(x_test, np.ndarray):
        x_test = torch.from_numpy(x_test).float()    # Convert to float32 type

    # Move data to the selected device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)

    # Determine the number of classes
    num_classes = len(torch.unique(y_train))
    y_pred = []  # List to store predictions
    start_time = time.time()  # Start timing

    # Iterate over each test sample
    for i in tqdm(range(x_test.shape[0]), desc="LC-kNN"):
        # Calculate distances from the test sample to all training samples
        distances = torch.norm(x_train - x_test[i], dim=1)
        # Find the indices of the k nearest neighbors
        knn_indices = torch.topk(distances, k, largest=False).indices
        knn_x = x_train[knn_indices]  # Features of the k nearest neighbors
        knn_y = y_train[knn_indices]  # Labels of the k nearest neighbors

        # Train a local classifier on the k nearest neighbors
        local_classifier = train_local_classifier(knn_x, knn_y, num_classes, n_epochs, device)
        local_classifier.eval()  # Set model to evaluation mode

        # Make prediction using the local classifier
        with torch.no_grad():  # Disable gradient calculation during prediction
            output = local_classifier(x_test[i].unsqueeze(0))
            _, predicted = torch.max(output, 1)
            y_pred.append(predicted.item())  # Store the predicted label

    end_time = time.time()  # End timing
    # Calculate average iteration time
    avg_iteration_time = (end_time - start_time) / x_test.shape[0]

    return y_pred, avg_iteration_time