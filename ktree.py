import torch
import torch.optim as optim
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

class KTreeModel:
    def __init__(self,k_features,sigma,rho1,rho2,n_epochs,device='cuda'):
        """
        Parameters:
        k_features: Number of nearest neighbors for feature similarity
        sigma: RBF kernel parameter
        rho1: L1 regularization coefficient
        rho2: Laplacian regularization coefficient
        n_epochs: Training epochs for sparse model
        device: Computation device ('cuda' or 'cpu')
        """
        self.k_features = k_features
        self.sigma = sigma
        self.rho1 = rho1
        self.rho2 = rho2
        self.n_epochs = n_epochs
        self.device = device
        self.S = None          # Feature similarity matrix
        self.L = None          # Laplacian matrix
        self.W = None          # Sparse reconstruction matrix
        self.k_tree = None     # Decision tree for k prediction
        self.X_train = None    # Stored training data
        self.y_train = None    # Stored training labels

    def compute_S(self, X_features):
        """Compute feature similarity matrix using RBF kernel"""
        d, n = X_features.shape
        S = torch.zeros(d, d, device=self.device)

        for i in range(d):
            # Calculate pairwise distances
            diff = X_features[i].unsqueeze(0) - X_features
            distances = torch.norm(diff, dim=1)

            # Compute RBF similarities
            similarities = torch.exp(-distances**2 / (2 * self.sigma**2))

            # Select top-k neighbors (excluding self)
            _, indices = torch.topk(distances, k=self.k_features+1, largest=False)
            valid_indices = indices[indices != i][:self.k_features]

            # Update similarity matrix
            S[i, valid_indices] = similarities[valid_indices]
            S[valid_indices, i] = similarities[valid_indices]

        return S

    def compute_L(self, S):
        """Compute Laplacian matrix from similarity matrix"""
        D = torch.diag(torch.sum(S, dim=1))
        return D - S

    def train_sparse_model(self, X_train_tensor, L):
        """Optimize sparse reconstruction matrix W"""
        n = X_train_tensor.shape[1]
        W = torch.rand(n, n, device=self.device, requires_grad=True)
        optimizer = optim.Adam([W], lr=0.001)

        # Initialize timing variables
        total_time = 0.0
        iteration_times = []

        # Training loop
        progress_bar = tqdm(range(self.n_epochs), desc="KTree")
        for _ in progress_bar:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()  # Record start time

            optimizer.zero_grad()

            # Reconstruction loss
            XW = torch.mm(X_train_tensor, W)
            loss_recon = torch.norm(XW - X_train_tensor, p='fro')**2

            # Regularization terms
            loss_l1 = self.rho1 * torch.norm(W, p=1)
            XT_L_X = torch.mm(torch.mm(X_train_tensor.T, L), X_train_tensor)
            loss_laplacian = self.rho2 * torch.trace(torch.mm(W.T, torch.mm(XT_L_X, W)))

            # Total loss
            total_loss = loss_recon + loss_l1 + loss_laplacian
            total_loss.backward()
            optimizer.step()

            # Apply non-negative constraint
            with torch.no_grad():
                W.data = torch.clamp(W.data, min=0)

            end_time.record()  # Record end time
            torch.cuda.synchronize()  # Ensure all operations are completed

            # Calculate iteration time
            iteration_time = start_time.elapsed_time(end_time)  # Time in milliseconds
            iteration_times.append(iteration_time)
            total_time += iteration_time

            progress_bar.set_postfix({'loss': total_loss.item(), 'time': f'{iteration_time:.2f} ms'})

        # Calculate average iteration time
        avg_iteration_time = total_time / self.n_epochs

        return W, avg_iteration_time

    def fit(self, X_train, y_train):
        """Full training pipeline"""
        # Convert to tensor and transpose for [d, n] shape
        X_tensor = torch.tensor(X_train.T, dtype=torch.float32).to(self.device)

        # Step 1: Learn optimal k-values
        self.S = self.compute_S(X_tensor)
        self.L = self.compute_L(self.S)
        self.W, self.avg_iteration_time = self.train_sparse_model(X_tensor, self.L)  # Capture average iteration time

        # Extract k-values (number of non-zero elements per column)
        k_values = (self.W > 1e-4).sum(dim=0).cpu().numpy().astype(int)
        k_values = np.clip(k_values, 1, None)  # Ensure k ≥ 1

        # Step 2: Build kTree
        self.k_tree = DecisionTreeClassifier()
        self.k_tree.fit(X_train, k_values)

        # Store training data for kNN classification
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Full prediction pipeline"""
        # Step 1: Predict k-values using kTree
        pred_k = self.k_tree.predict(X_test)
        pred_k = np.clip(pred_k, 1, None)

        # Step 2: Dynamic kNN classification
        predictions = []
        for i, k in enumerate(pred_k):
            # Handle cases where k exceeds number of training samples
            effective_k = min(k, len(self.X_train))

            knn = KNeighborsClassifier(n_neighbors=effective_k)
            knn.fit(self.X_train, self.y_train)
            pred = knn.predict(X_test[i:i+1])
            predictions.append(pred[0])

        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        """Evaluate accuracy"""
        preds = self.predict(X_test)
        return np.mean(preds == y_test)