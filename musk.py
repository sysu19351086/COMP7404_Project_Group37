import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

# --------------------------------------------------------------------------------
#                                 Data Loading
# --------------------------------------------------------------------------------
# Data loading and preprocessing
def load_clean1_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path, header=None)
    # Assume the last column is the target variable
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

# Load data
file_path = "datasets/feature_number/musk/clean1.data"
X, y = load_clean1_data(file_path)

# Ensure all feature columns are numeric
for col in range(X.shape[1]):
    X[:, col] = pd.to_numeric(X[:, col], errors='coerce')

# Fill missing values (if any)
X = pd.DataFrame(X).fillna(0).values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors and adjust dimensions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.tensor(X_train.T, dtype=torch.float32).to(device)  # [d, n]
X_test_tensor = torch.tensor(X_test.T, dtype=torch.float32).to(device)    # [d, m]
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# --------------------------------------------------------------------------------
#                                     KTree
# --------------------------------------------------------------------------------
from ktree import KTreeModel

# Configuration
ktree_config = {
    'k_features': 100,
    'sigma': 1.0,
    'rho1': 1e-4,
    'rho2': 1e-4,
    'n_epochs': 24000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Initialize Model
ktree_model = KTreeModel(**ktree_config)
ktree_model.fit(X_train, y_train)

end_time.record()
torch.cuda.synchronize()

# Running Cost
running_cost_ktree = start_time.elapsed_time(end_time) / ktree_config['n_epochs']
print(f"KTree Running Cost: {running_cost_ktree:.4f}")

# Prediction Phase
accuracy_ktree = ktree_model.evaluate(X_test, y_test)
print(f"KTree Test Accuracy: {accuracy_ktree:.4f}")

# --------------------------------------------------------------------------------
#                                  KStarTree
# --------------------------------------------------------------------------------
from kstartree import KStarTreeModel

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Initialize and train the model
kstartree_model = KStarTreeModel(**ktree_config)
kstartree_model.fit(X_train, y_train)

end_time.record()
torch.cuda.synchronize()

# Running Cost
running_cost_kstartree = start_time.elapsed_time(end_time) / (ktree_config['n_epochs'])
print(f"KStarTree Running Cost: {running_cost_kstartree:.4f}")

# Accuracy
accuracy_kstartree = kstartree_model.evaluate(X_test, y_test)
print(f"KStarTree Test Accuracy: {accuracy_kstartree:.4f}")

# --------------------------------------------------------------------------------
#                                  Traditional KNN
# --------------------------------------------------------------------------------
X_train_tensor = torch.tensor(X_train.T, dtype=torch.float32).to(device)  # [d, n]
X_test_tensor = torch.tensor(X_test.T, dtype=torch.float32).to(device)    # [d, m]
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

from knn import kNN

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Initialize KNN Model
knn_model = kNN(
    k=10,
    device=device)
knn_model.fit(X_train_tensor, y_train_tensor)

y_pred_knn = knn_model.predict(X_test_tensor)

end_time.record()
torch.cuda.synchronize()

# Running Cost
running_cost_knn = start_time.elapsed_time(end_time) / len(X_test)
print(f"Traditional kNN Running Cost: {running_cost_knn:.4f}")

# Compute Accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn.cpu().numpy())
print(f'Traditional kNN Test Accuracy: {accuracy_knn:.4f}')

# --------------------------------------------------------------------------------
#                                   AD-kNN
# --------------------------------------------------------------------------------
from adknn import ADkNNModel

# Transform input size
X_train_tensor_adknn = torch.tensor(X_train, dtype=torch.float32).to(device)    # [n, d]
X_test_tensor_adknn = torch.tensor(X_test, dtype=torch.float32).to(device)      # [m, d]
y_train_tensor_adknn = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor_adknn = torch.tensor(y_test, dtype=torch.long).to(device)

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Initial AD-kNN model
adknn_model = ADkNNModel(base_k=10, alpha=0.1, device=device)

# Train model
adknn_model.fit(X_train_tensor_adknn, y_train_tensor_adknn)

# Prediction
y_pred = adknn_model.predict(X_test_tensor_adknn, show_progress=True)

end_time.record()
torch.cuda.synchronize()

# Running Cost
running_cost_adknn = start_time.elapsed_time(end_time) / len(X_test)
print(f"AD-kNN Running Cost: {running_cost_adknn:.4f}")

# Compute Accuracy
accuracy_adknn = accuracy_score(y_test_tensor_adknn.cpu().numpy(), y_pred)
print(f"AD-kNN Test Accuracy: {accuracy_adknn:.4f}")

# --------------------------------------------------------------------------------
#                                    S-kNN
# --------------------------------------------------------------------------------
from sknn import SkNNModel

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Initial S-kNN model
sknn_model = SkNNModel(**ktree_config)
sknn_model.fit(X_train, y_train)

end_time.record()
torch.cuda.synchronize()

# Running Cost
running_cost_sknn = start_time.elapsed_time(end_time) / ktree_config['n_epochs']
print(f"S-kNN Running Cost: {running_cost_sknn:.4f}")

# Prediction
y_pred_sknn = sknn_model.predict(X_test)

# Compute Accuracy
accuracy_sknn = accuracy_score(y_test, y_pred_sknn)
print(f"S-kNN Test Accuracy: {accuracy_sknn:.4f}")

# --------------------------------------------------------------------------------
#                                   GS-kNN
# --------------------------------------------------------------------------------
from gsknn import GSkNNModel

# Adjust inputs
X_train_gsknn = torch.tensor(X_train, dtype=torch.float32).to(device)  # [n_samples, n_features]
X_test_gsknn = torch.tensor(X_test, dtype=torch.float32).to(device)     # [n_test_samples, n_features]

# GS-kNN specific configurations
gsknn_config = {
    'k_neighbors': 10,
    'rho': 0.6,
    'delta': 0.001,
    'n_epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Initial GS-kNN model
gsknn_model = GSkNNModel(**gsknn_config)

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Training
gsknn_model.fit(X_train_gsknn.cpu().numpy(), y_train)

end_time.record()
torch.cuda.synchronize()

# Running Cost
running_cost_gsknn = start_time.elapsed_time(end_time) / gsknn_config['n_epochs']
print(f"GS-kNN Running Cost: {running_cost_gsknn:.4f}")

# Prediction
y_pred_gsknn = gsknn_model.predict(X_test_gsknn.cpu().numpy())

# Compute Accuracy
accuracy_gsknn = accuracy_score(y_test, y_pred_gsknn)
print(f"GS-kNN Test Accuracy: {accuracy_gsknn:.4f}")

# --------------------------------------------------------------------------------
#                                   FASBIR
# --------------------------------------------------------------------------------
from fasbir import FasbirModel

# Instantiate FasbirModel, adjust input_size based on the number of input features
input_dim = X_train.shape[1]  # Input feature dimension
output_dim = len(np.unique(y_train))  # Number of classes

# Configuration
fasbir_config = {
    'input_size': input_dim,
    'hidden_size1': 128,
    'hidden_size2': 64,
    'output_size': output_dim,
    'lr': 0.05,
    'n_epochs': 200,
    'batch_size': 32
}

fasbir_model = FasbirModel(**fasbir_config)

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Train the model (input is a numpy array)
fasbir_model.fit(X_train, y_train)

end_time.record()
torch.cuda.synchronize()

# Running Cost
running_cost_fasbir = start_time.elapsed_time(end_time) / fasbir_config['n_epochs']
print(f"FASBIR Running Cost: {running_cost_fasbir:.4f}")

# Evaluate the test set and get accuracy
accuracy_fasbir = fasbir_model.evaluate(X_test, y_test)
print(f"FASBIR Test Accuracy: {accuracy_fasbir:.4f}")

# --------------------------------------------------------------------------------
#                                   LC-kNN
# --------------------------------------------------------------------------------
from lcknn import LCkNNModel

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Call the LC-kNN model for prediction
y_pred_lcknn, _ = LCkNNModel(
    x_train=X_train,       # Training features (numpy array or Tensor)
    y_train=y_train,       # Training labels (numpy array or Tensor)
    x_test=X_test,         # Test features (numpy array or Tensor)
    k=5,                   # Number of neighbors
    n_epochs=50,           # Number of training epochs for local classifiers
    device=device          # Device (automatically selects GPU or CPU)
)

# Compute accuracy
accuracy_lcknn = accuracy_score(y_test, y_pred_lcknn)

end_time.record()
torch.cuda.synchronize()

running_cost_lcknn = start_time.elapsed_time(end_time) / len(X_test)
print(f"LC-kNN Running Cost: {running_cost_lcknn:.4f}")

# Print Results
print(f"LC-kNN Test Accuracy: {accuracy_lcknn:.4f}")

# --------------------------------------------------------------------------------
#                                 Comparison
# --------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# Hypothetical Algorithm Names
algorithms = ["KTree", "KStarTree", "kNN", "AD-kNN", "S-kNN", "GS-kNN", "FASBIR", "LC-kNN"]

# Hypothetical Accuracy and Average Iteration Time
accuracy = [accuracy_ktree, accuracy_kstartree, accuracy_knn, accuracy_adknn, accuracy_sknn, accuracy_gsknn, accuracy_fasbir, accuracy_lcknn]
running_cost = [running_cost_ktree, running_cost_kstartree, running_cost_knn, running_cost_adknn, running_cost_sknn, running_cost_gsknn, running_cost_fasbir, running_cost_lcknn]

# Create Directory for Saving Files (if not exists)
output_dir = "outputs/feature_number/musk"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save Data to CSV File
csv_filename = os.path.join(output_dir, "musk.csv")
with open(csv_filename, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Algorithm", "Accuracy", "Average Iteration Time"])
    for i, algo in enumerate(algorithms):
        writer.writerow([algo, accuracy[i], running_cost[i]])
print(f"Data saved to: {csv_filename}")

# Create Figure and Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Use Rainbow Colors
rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, len(algorithms)))

# Plot Accuracy Bar Chart
x = np.arange(len(algorithms))
bars1 = ax1.bar(x, accuracy, color=rainbow_colors)
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms)
ax1.set_title('Accuracy Comparison')
ax1.set_xlabel('Algorithms')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add Labels to Accuracy Bar Chart
ax1.bar_label(bars1, padding=3)

# Plot Average Iteration Time Bar Chart
bars2 = ax2.bar(x, running_cost, color=rainbow_colors)
ax2.set_xticks(x)
ax2.set_xticklabels(algorithms)
ax2.set_title('Running Cost Comparison')
ax2.set_xlabel('Algorithms')
ax2.set_ylabel('Running cost')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add Labels to Average Iteration Time Bar Chart
ax2.bar_label(bars2, padding=3)

# Adjust Layout
plt.tight_layout()

# Save Chart to File
image_filename = os.path.join(output_dir, "musk.png")
plt.savefig(image_filename, dpi=300)
print(f"Chart saved to: {image_filename}")

# Display Chart (Optional)
plt.show()