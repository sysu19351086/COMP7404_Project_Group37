import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

# --------------------------------------------------------------------------------
#                                 Data Loading
# --------------------------------------------------------------------------------
# Data loading and preprocessing
url = "datasets/sample_size/german/german.data"
columns = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'existing_credits', 'job', 'num_people', 'own_telephone', 'foreign_worker', 'class'
]
data = pd.read_csv(url, names=columns, sep='\s+')

# Separate numerical and categorical features
numeric_cols = ['duration', 'credit_amount', 'installment_rate', 'residence_since',
                'age', 'existing_credits', 'num_people']
categorical_cols = list(set(columns) - set(numeric_cols) - {'class'})

# Label encode categorical features
for col in categorical_cols + ['class']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Standardize numerical features
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Prepare features and target
X = data.drop('class', axis=1).values
y = data['class'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors and adjust dimensions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.tensor(X_train.T, dtype=torch.float32).to(device)  # [d, n]
X_test_tensor = torch.tensor(X_test.T, dtype=torch.float32).to(device)     # [d, m]
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# --------------------------------------------------------------------------------
#                                     KTree
# --------------------------------------------------------------------------------
from ktree import KTreeModel

# Configuration
ktree_config = {
    'k_features': 10,
    'sigma': 1.0,
    'rho1': 1e-4,
    'rho2': 1e-4,
    'n_epochs': 20000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Initialize Model
model = KTreeModel(**ktree_config)
model.fit(X_train, y_train)

end_time.record()
torch.cuda.synchronize()

running_cost_ktree = start_time.elapsed_time(end_time) / ktree_config['n_epochs']
print(f"KTree Running Cost: {running_cost_ktree:.4f}")

# Prediction Phase
y_pred = model.predict(X_test)

# Evaluation
accuracy_ktree = accuracy_score(y_test, y_pred)

# Print Results
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
model = KStarTreeModel(**ktree_config)
model.fit(X_train, y_train)

end_time.record()
torch.cuda.synchronize()

running_cost_kstartree = start_time.elapsed_time(end_time) / (ktree_config['n_epochs'])
print(f"KStarTree Running Cost: {running_cost_kstartree:.4f}")

# Evaluate the model
accuracy_kstartree = model.evaluate(X_test, y_test)
avg_iter_time_kstartree = model.avg_iteration_time

print(f"KStarTree Test Accuracy: {accuracy_kstartree:.4f}")

# --------------------------------------------------------------------------------
#                                  Traditional KNN
# --------------------------------------------------------------------------------
from knn import kNN

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Initialize KNN Model
knn_model = kNN(
    k=3,
    device=device)
knn_model.fit(X_train_tensor, y_train_tensor)

y_pred_knn = knn_model.predict(X_test_tensor)

end_time.record()
torch.cuda.synchronize()

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
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)    # [n, d]
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)      # [m, d]

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Initial AD-kNN model
model = ADkNNModel(base_k=5, alpha=0.1, device=device)

# Train model
model.fit(X_train_tensor, y_train_tensor)

# Prediction
y_pred = model.predict(X_test_tensor, show_progress=True)

end_time.record()
torch.cuda.synchronize()

running_cost_adknn = start_time.elapsed_time(end_time) / len(X_test)
print(f"AD-kNN Running Cost: {running_cost_adknn:.4f}")

# Compute Accuracy
accuracy_adknn = accuracy_score(y_test_tensor.cpu().numpy(), y_pred)

# Print Results
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
model = SkNNModel(**ktree_config)
model.fit(X_train, y_train)

end_time.record()
torch.cuda.synchronize()

running_cost_sknn = start_time.elapsed_time(end_time) / ktree_config['n_epochs']
print(f"S-kNN Running Cost: {running_cost_sknn:.4f}")

# Prediction
y_pred_sknn = model.predict(X_test)
accuracy_sknn = accuracy_score(y_test, y_pred_sknn)

print(f"S-kNN Test Accuracy: {accuracy_sknn:.4f}")

# --------------------------------------------------------------------------------
#                                   GS-kNN
# --------------------------------------------------------------------------------
from gsknn import GSkNNModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Data Preprocessing
scaler = StandardScaler()
X_gsknn = scaler.fit_transform(X)

# Split Dataset
X_train_gsknn, X_test_gsknn, y_train_gsknn, y_test_gsknn = train_test_split(
    X_gsknn, y,
    test_size=0.3,
    random_state=42,
)

# Add GS-kNN specific configurations after the original configurations
gs_config = {
    'k_neighbors': 3,
    'rho': 0.6,
    'delta': 0.001,
    'n_epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Initialize Model
model = GSkNNModel(**gs_config)

# Training Phase
model.fit(X_train_gsknn, y_train_gsknn)

end_time.record()
torch.cuda.synchronize()

running_cost_gsknn = start_time.elapsed_time(end_time) / (gs_config['n_epochs'])
print(f"GS-kNN Running Cost: {running_cost_gsknn:.4f}")

# Prediction Phase
y_pred_gsknn = model.predict(X_test_gsknn)

# Evaluation
accuracy_gsknn = accuracy_score(y_test_gsknn, y_pred_gsknn)

# Output Results
print(f"GS-kNN Test Accuracy: {accuracy_gsknn:.4f}")

# --------------------------------------------------------------------------------
#                                   FASBIR
# --------------------------------------------------------------------------------
from fasbir import FasbirModel
import numpy as np

# Determine input dimension based on your data
input_dim = X_train.shape[1]  # Number of features after preprocessing
output_dim = len(np.unique(y))  # Number of classes

# Configuration
fasbir_config = {
    'input_size': input_dim,
    'hidden_size1': 128,
    'hidden_size2': 64,
    'output_size': output_dim,
    'lr': 0.001,
    'n_epochs': 20,
    'batch_size': 32
}

# Initialize Model
model = FasbirModel(**fasbir_config)

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Training Phase
model.fit(X_train, y_train)

end_time.record()
torch.cuda.synchronize()

running_cost_fasbir = start_time.elapsed_time(end_time) / fasbir_config['n_epochs']
print(f"FASBIR Running Cost: {running_cost_fasbir:.4f}")

# Evaluation
accuracy_fasbir = model.evaluate(X_test, y_test)

# Output Results
print(f"FASBIR Test Accuracy: {accuracy_fasbir:.4f}")

# --------------------------------------------------------------------------------
#                                   LC-kNN
# --------------------------------------------------------------------------------
from lcknn import LCkNNModel

# GPU Timing
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()

# Run LC-kNN
y_pred, avg_iter_time_lcknn = LCkNNModel(X_train, y_train, X_test, k=5, n_epochs=100, device=device)
accuracy_lcknn = accuracy_score(y_test, y_pred)

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
output_dir = "outputs/sample_size/german"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save Data to CSV File
csv_filename = os.path.join(output_dir, "german.csv")
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
image_filename = os.path.join(output_dir, "german.png")
plt.savefig(image_filename, dpi=300)
print(f"Chart saved to: {image_filename}")

# Display Chart (Optional)
plt.show()