import torch
import torch.optim as optim
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

class KStarTreeModel:
    def __init__(self, k_features, sigma, rho1, rho2, n_epochs, device='cuda'):
        # Initialize model parameters
        self.k_features = k_features  # Number of nearest features for similarity
        self.sigma = sigma            # Sigma for Gaussian kernel
        self.rho1 = rho1              # Regularization parameter for L1 loss
        self.rho2 = rho2              # Regularization parameter for Laplacian loss
        self.n_epochs = n_epochs      # Number of training epochs
        self.device = device          # Device for computation (GPU or CPU)
        self.S = None                 # Similarity matrix
        self.L = None                 # Laplacian matrix
        self.W = None                 # Sparse representation matrix
        self.k_tree = None            # Decision tree for predicting k values
        self.X_train = None           # Training data
        self.y_train = None           # Training labels

    def compute_S(self, X_features):
        # Compute similarity matrix using Gaussian kernel
        d, n = X_features.shape
        S = torch.zeros(d, d, device=self.device)  # Initialize similarity matrix
        for i in range(d):
            # Compute pairwise distances for feature i
            diff = X_features[i].unsqueeze(0) - X_features
            distances = torch.norm(diff, dim=1)
            # Compute similarities using Gaussian kernel
            similarities = torch.exp(-distances**2 / (2 * self.sigma**2))
            # Find k nearest features (excluding itself)
            _, indices = torch.topk(distances, k=self.k_features+1, largest=False)
            valid_indices = indices[indices != i][:self.k_features]
            # Assign similarities to the top-k features
            S[i, valid_indices] = similarities[valid_indices]
            S[valid_indices, i] = similarities[valid_indices]
        return S

    def compute_L(self, S):
        # Compute Laplacian matrix from similarity matrix
        D = torch.diag(torch.sum(S, dim=1))  # Degree matrix
        return D - S  # Laplacian matrix = Degree matrix - Similarity matrix

    def train_sparse_model(self, X_train_tensor, L):
        # Train sparse representation matrix W using optimization
        n = X_train_tensor.shape[1]
        W = torch.rand(n, n, device=self.device, requires_grad=True)  # Initialize W
        optimizer = optim.Adam([W], lr=0.001)  # Adam optimizer for training
        total_time = 0.0
        iteration_times = []
        progress_bar = tqdm(range(self.n_epochs), desc="KStarTree")  # Progress bar for training

        for _ in progress_bar:
            # Record start time for iteration timing
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            optimizer.zero_grad()  # Clear gradients
            XW = torch.mm(X_train_tensor, W)  # Compute X * W
            # Reconstruction loss: ||XW - X||_F^2
            loss_recon = torch.norm(XW - X_train_tensor, p='fro')**2
            # L1 regularization: rho1 * ||W||_1
            loss_l1 = self.rho1 * torch.norm(W, p=1)
            # Laplacian regularization: rho2 * trace(W^T X^T L X W)
            XT_L_X = torch.mm(torch.mm(X_train_tensor.T, L), X_train_tensor)
            loss_laplacian = self.rho2 * torch.trace(torch.mm(W.T, torch.mm(XT_L_X, W)))
            # Total loss = Reconstruction + L1 + Laplacian
            total_loss = loss_recon + loss_l1 + loss_laplacian
            total_loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Ensure W is non-negative
            with torch.no_grad():
                W.data = torch.clamp(W.data, min=0)

            # Record end time and compute iteration time
            end_time.record()
            torch.cuda.synchronize()
            iteration_time = start_time.elapsed_time(end_time)
            iteration_times.append(iteration_time)
            total_time += iteration_time

            # Update progress bar with loss and time
            progress_bar.set_postfix({'loss': total_loss.item(), 'time': f'{iteration_time:.2f} ms'})

        # Compute average iteration time
        avg_iteration_time = total_time / self.n_epochs
        return W, avg_iteration_time

    def fit(self, X_train, y_train):
        # Fit the model to the training data
        X_tensor = torch.tensor(X_train.T, dtype=torch.float32).to(self.device)  # Convert to tensor
        self.S = self.compute_S(X_tensor)  # Compute similarity matrix
        self.L = self.compute_L(self.S)    # Compute Laplacian matrix
        self.W, self.avg_iteration_time = self.train_sparse_model(X_tensor, self.L)  # Train sparse model
        # Compute k values based on non-zero entries in W
        k_values = (self.W > 1e-4).sum(dim=0).cpu().numpy().astype(int)
        k_values = np.clip(k_values, 1, None)  # Ensure k is at least 1
        # Train a decision tree to predict k values
        self.k_tree = DecisionTreeClassifier()
        self.k_tree.fit(X_train, k_values)
        self.X_train = X_train  # Store training data
        self.y_train = y_train  # Store training labels

    def predict(self, X_test):
        # Predict labels for test data using the trained model
        pred_k = self.k_tree.predict(X_test)  # Predict k values using decision tree
        pred_k = np.clip(pred_k, 1, None)     # Ensure k is at least 1
        predictions = []
        for i, k in enumerate(pred_k):
            # Use KNN with predicted k to classify each test sample
            effective_k = min(k, len(self.X_train))  # Ensure k does not exceed training samples
            knn = KNeighborsClassifier(n_neighbors=effective_k)
            knn.fit(self.X_train, self.y_train)
            pred = knn.predict(X_test[i:i+1])
            predictions.append(pred[0])
        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        # Evaluate the model on test data
        preds = self.predict(X_test)  # Get predictions
        return np.mean(preds == y_test)  # Compute accuracy
