import torch
import torch.optim as optim
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

class GSkNNModel:
    def __init__(self, k_neighbors, rho=0.5, delta=0.001, n_epochs=50, device='cuda'):
        """
        Parameters:
        k_neighbors: Target number of neighbors
        rho: Neighbor sampling rate (0-1)
        delta: Early stopping threshold
        n_epochs: Maximum number of iterations
        device: Computation device
        """
        self.k = k_neighbors
        self.rho = rho
        self.delta = delta
        self.n_epochs = n_epochs
        self.device = device
        self.S = None          # Sparse similarity matrix
        self.W = None          # Reconstruction weight matrix
        self.X_train = None    # Training data
        self.y_train = None    # Training labels
        self.avg_iteration_time = 0.0  # Average iteration time

    def _nn_descent(self, X):
        """NN-Descent algorithm to construct an approximate K-nearest neighbor graph"""
        # Move data to CPU and convert to numpy
        X_cpu = X.cpu().numpy() if X.is_cuda else X.numpy()

        n = X_cpu.shape[0]

        # Initialize random K-nearest neighbors
        knn = NearestNeighbors(n_neighbors=self.k+1).fit(X_cpu)
        _, indices = knn.kneighbors(X_cpu)

        # Build initial sparse similarity matrix
        S = torch.zeros(n, n, device=self.device)

        # Initialize adjacency lists and populate initial similarities
        B = {i: set() for i in range(n)}
        for i in range(n):
            for j in indices[i][1:]:  # Exclude self
                B[i].add(j)
                # Calculate initial similarity
                sim = torch.exp(-torch.norm(X[i]-X[j], p=2)**2)
                S[i, j] = sim

        R = {i: set() for i in range(n)}
        for i in range(n):
            for j in B[i]:
                R[j].add(i)

        # Iterative optimization
        prev_update = float('inf')
        total_time = 0.0
        iteration_times = []

        progress_bar = tqdm(range(self.n_epochs), desc="NN Descent")
        for epoch in progress_bar:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()  # Record start time

            update_count = 0
            new_B = {i: set() for i in range(n)}

            # Sample neighbors for local connections
            for i in range(n):
                candidates = list(B[i].union(R[i]))
                if not candidates:
                    continue

                # Sampling ratio
                sample_size = max(1, int(len(candidates)*self.rho))
                sampled = np.random.choice(candidates, size=sample_size, replace=False)

                # Local connections (using original tensor X for distance calculation)
                for u in sampled:
                    for v in B.get(u, set()):
                        if v == i:
                            continue
                        # Calculate current similarity
                        current_sim = torch.exp(-torch.norm(X[i]-X[v], p=2)**2)

                        # Key fix: Use dynamically updated S matrix
                        if current_sim > S[i, v]:
                            update_count += 1
                            new_B[i].add(v)
                            new_B[v].add(i)
                            # Immediately update similarity matrix
                            S[i, v] = current_sim
                            S[v, i] = current_sim

            # Merge updates (keep index operations on CPU)
            for i in range(n):
                B[i].update(new_B[i])
                if len(B[i]) > self.k:
                    # Sort using the updated S matrix
                    neighbors = list(B[i])
                    similarities = [S[i, j].item() for j in neighbors]
                    sorted_neighbors = [x for _, x in sorted(zip(similarities, neighbors), reverse=True)]
                    B[i] = set(sorted_neighbors[:self.k])

            end_time.record()  # Record end time
            torch.cuda.synchronize()  # Ensure all operations complete

            # Calculate iteration time
            iteration_time = start_time.elapsed_time(end_time)  # Time in milliseconds
            iteration_times.append(iteration_time)
            total_time += iteration_time

            progress_bar.set_postfix({'updates': update_count, 'time': f'{iteration_time:.2f} ms'})

            # Early stopping condition
            if update_count < self.delta * self.k * n:
                break

        # Calculate average iteration time
        self.avg_iteration_time = total_time / (epoch + 1) if (epoch + 1) > 0 else 0

        return S

    def _train_reconstruction(self, X):
        """Sparse reconstruction optimization"""
        n = X.shape[0]
        W = torch.zeros(n, n, device=self.device, requires_grad=True)
        optimizer = optim.Adam([W], lr=0.01)

        total_time = 0.0
        iteration_times = []

        progress_bar = tqdm(range(self.n_epochs), desc="GS-kNN")
        for epoch in range(self.n_epochs):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()  # Record start time

            optimizer.zero_grad()
            # Reconstruction loss
            loss = torch.norm(X - torch.mm(W, X)) + 0.5*torch.norm(W, p=1)
            loss.backward()
            optimizer.step()
            # Non-negative constraint
            with torch.no_grad():
                W.data = torch.clamp(W.data, min=0)

            end_time.record()  # Record end time
            torch.cuda.synchronize()  # Ensure all operations complete

            # Calculate iteration time
            iteration_time = start_time.elapsed_time(end_time)
            iteration_times.append(iteration_time)
            total_time += iteration_time

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'time': f'{iteration_time:.2f} ms'})

        # Calculate average iteration time
        self.avg_iteration_time = total_time / self.n_epochs

        return W

    def fit(self, X_train, y_train):
        """Training process"""
        # Convert to tensor and store
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)

        # Subsequent calculations remain unchanged
        self.S = self._nn_descent(self.X_train)
        self.W = self._train_reconstruction(self.X_train)

    def predict(self, X_test):
        """Dynamic kNN prediction"""
        # Convert input to PyTorch tensor
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        # Ensure training data is a tensor
        assert isinstance(self.X_train, torch.Tensor), "Training data must be Tensor"

        # Calculate similarity matrix
        sims = torch.mm(X_test_tensor, self.X_train.T)  # Now both are Tensors

        k_values = (self.W > 1e-4).sum(dim=1).cpu().numpy()
        k_values = np.clip(k_values, 1, self.k)

        preds = []
        for i in range(len(X_test_tensor)):
            k = int(k_values[i])
            _, indices = torch.topk(sims[i], k=k)
            votes = self.y_train[indices.cpu().numpy()]  # Ensure labels are Tensors
            pred = np.bincount(votes.cpu().numpy()).argmax()
            preds.append(pred)
        return np.array(preds)