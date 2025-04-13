import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class SkNNModel:
    def __init__(self, k_features, sigma, rho1, rho2, n_epochs, device='cuda'):
        self.k_features = k_features
        self.sigma = sigma
        self.rho1 = rho1
        self.rho2 = rho2
        self.n_epochs = n_epochs
        self.device = device
        self.S = None
        self.L = None
        self.W = None
        self.k_values = None
        self.X_train = None
        self.y_train = None
        self.avg_iteration_time = None

    def compute_S(self, X_features):
        d, n = X_features.shape
        S = torch.zeros(d, d, device=self.device)
        for i in range(d):
            diff = X_features[i].unsqueeze(0) - X_features
            distances = torch.norm(diff, dim=1)
            similarities = torch.exp(-distances**2 / (2 * self.sigma**2))
            _, indices = torch.topk(distances, k=self.k_features+1, largest=False)
            valid_indices = indices[indices != i][:self.k_features]
            S[i, valid_indices] = similarities[valid_indices]
            S[valid_indices, i] = similarities[valid_indices]
        return S

    def compute_L(self, S):
        D = torch.diag(torch.sum(S, dim=1))
        return D - S

    def train_sparse_model(self, X_train_tensor, L):
        n = X_train_tensor.shape[1]
        W = torch.rand(n, n, device=self.device, requires_grad=True)
        optimizer = optim.Adam([W], lr=0.001)
        total_time = 0.0
        iteration_times = []
        progress_bar = tqdm(range(self.n_epochs), desc="S-kNN")
        for _ in progress_bar:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            optimizer.zero_grad()
            XW = torch.mm(X_train_tensor, W)
            loss_recon = torch.norm(XW - X_train_tensor, p='fro')**2
            loss_l1 = self.rho1 * torch.norm(W, p=1)
            XT_L_X = torch.mm(torch.mm(X_train_tensor.T, L), X_train_tensor)
            loss_laplacian = self.rho2 * torch.trace(torch.mm(W.T, torch.mm(XT_L_X, W)))
            total_loss = loss_recon + loss_l1 + loss_laplacian
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                W.data = torch.clamp(W.data, min=0)
            end_time.record()
            torch.cuda.synchronize()
            iteration_time = start_time.elapsed_time(end_time)
            iteration_times.append(iteration_time)
            total_time += iteration_time
            progress_bar.set_postfix({'loss': total_loss.item(), 'time': f'{iteration_time:.2f}ms'})
        avg_iteration_time = total_time / self.n_epochs
        return W, avg_iteration_time

    def fit(self, X_train, y_train):
        X_tensor = torch.tensor(X_train.T, dtype=torch.float32).to(self.device)
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        self.S = self.compute_S(X_tensor)
        self.L = self.compute_L(self.S)
        self.W, self.avg_iteration_time = self.train_sparse_model(X_tensor, self.L)
        k_values = (self.W > 1e-4).sum(dim=0).cpu().numpy().astype(int)
        k_values = np.clip(k_values, 1, None)
        self.k_values = torch.tensor(k_values, dtype=torch.long).to(self.device)

    def predict(self, X_test):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        dist_mat = torch.cdist(X_test_tensor, self.X_train, p=2)
        nearest_indices = torch.argmin(dist_mat, dim=1)
        k_test = self.k_values[nearest_indices]
        sorted_indices = torch.argsort(dist_mat, dim=1)
        predictions = []
        for i in range(X_test_tensor.shape[0]):
            k = k_test[i].item()
            effective_k = min(k, self.X_train.shape[0])
            neighbors_indices = sorted_indices[i, :effective_k]
            neighbors_labels = self.y_train[neighbors_indices]
            counts = torch.bincount(neighbors_labels, minlength=2)
            pred = torch.argmax(counts).item()
            predictions.append(pred)
        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return np.mean(preds == y_test)