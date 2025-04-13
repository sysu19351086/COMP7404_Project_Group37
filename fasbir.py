import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time

class SimpleNN(nn.Module):
    def __init__(self, input_size=4, hidden_size1=128, hidden_size2=64, output_size=2):
        super(SimpleNN, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # Output layer
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # Apply ReLU activation to the first layer
        x = torch.relu(self.fc1(x))
        # Apply ReLU activation to the second layer
        x = torch.relu(self.fc2(x))
        # Output layer (no activation function as CrossEntropyLoss includes softmax)
        x = self.fc3(x)
        return x

class FasbirModel:
    def __init__(self, input_size=4, hidden_size1=128, hidden_size2=64, output_size=2, lr=0.001, n_epochs=20, batch_size=32):
        # Initialize the neural network model
        self.model = SimpleNN(input_size, hidden_size1, hidden_size2, output_size)
        # Automatically select device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move model to the selected device
        self.model.to(self.device)
        # Define loss function (CrossEntropyLoss for classification)
        self.criterion = nn.CrossEntropyLoss()
        # Define optimizer (Adam optimizer)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Number of training epochs
        self.n_epochs = n_epochs
        # Batch size for training and evaluation
        self.batch_size = batch_size
        # Average iteration time
        self.avg_iteration_time = 0

    def fit(self, X_train, y_train):
        # Convert input data to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)

        # Create a data loader for training data
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        total_time = 0
        pbar = tqdm(total=self.n_epochs, desc="FASBIR")

        for epoch in range(self.n_epochs):
            start_time = time.time()
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            correct = 0
            total = 0

            # Iterate over batches of training data
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(inputs)
                # Calculate loss
                loss = self.criterion(outputs, labels)
                # Backward pass
                loss.backward()
                # Update model parameters
                self.optimizer.step()

                running_loss += loss.item()
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'time': f'{epoch_time:.2f} s'})

        pbar.close()
        # Calculate average iteration time
        self.avg_iteration_time = total_time / self.n_epochs

    def evaluate(self, X_test, y_test):
        # Convert input data to PyTorch tensors
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create a data loader for test data
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient calculation during evaluation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate accuracy
        accuracy = correct / total
        return accuracy

    def predict(self, X):
        # Convert input data to PyTorch tensor
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation during prediction
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
        # Return predictions as numpy array
        return predicted.cpu().numpy()