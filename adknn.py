import torch
import numpy as np
from tqdm import tqdm

class ADkNNModel:
    def __init__(self, base_k=5, alpha=0.1, device='cuda'):
        self.base_k = base_k
        self.alpha = alpha
        self.device = device
        self.X_train = None
        self.y_train = None
        self.avg_iteration_time = 0.0

    def fit(self, X_train, y_train):
        # Ensure input data is on GPU and transposed to [n, d]
        self.X_train = X_train.to(self.device)  # [n, d]
        self.y_train = y_train.to(self.device)

    def predict(self, X_test, show_progress=True):
        X_test = X_test.to(self.device)  # [m, d]
        dists = torch.cdist(X_test, self.X_train)  # [m, n]

        y_pred = []
        total_time = 0.0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Add progress bar
        iterator = range(len(X_test))

        for i in tqdm(iterator, desc='AD-kNN'):
            torch.cuda.synchronize()
            start_event.record()

            # Dynamically calculate k value
            mean_dist = torch.mean(dists[i])
            k = max(1, int(self.base_k + self.alpha * mean_dist.item()))
            k = min(k, len(dists[i]))  # Ensure k does not exceed the length of dists[i]
            _, indices = torch.topk(dists[i], k=k, largest=False)
            labels = self.y_train[indices]
            counts = torch.bincount(labels)
            pred = torch.argmax(counts).item()
            y_pred.append(pred)

            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event)

        # Calculate average iteration time
        self.avg_iteration_time = total_time / len(X_test) if len(X_test) > 0 else 0

        return np.array(y_pred)