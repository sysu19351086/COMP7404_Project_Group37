import torch
from tqdm import tqdm

class kNN:
    def __init__(self, k=5, device='cuda'):
        self.k = k
        self.device = device
        self.avg_iteration_time = 0.0

    def fit(self, X_train, y_train):
        # 转置输入数据，确保形状为 [n, d]
        self.X_train = X_train.T  # [n, d]
        self.y_train = y_train   # [n]

    def predict(self, X_test):
        # X_test形状应为 [d, m]，在内部转置为 [m, d]
        X_test = X_test.T        # [m, d]
        num_test_samples = X_test.shape[0]

        y_pred = torch.zeros(num_test_samples, dtype=torch.long, device=self.device)
        total_time = 0.0
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        for i in tqdm(range(num_test_samples), desc='Traditional kNN'):
            x = X_test[i].unsqueeze(0)  # [1, d]

            start_time.record()
            dists = torch.cdist(x, self.X_train)  # [1, n]
            _, indices = torch.topk(dists, k=self.k, largest=False, dim=1)
            nearest_labels = self.y_train[indices].squeeze()
            counts = torch.bincount(nearest_labels)
            y_pred[i] = torch.argmax(counts)

            end_time.record()
            torch.cuda.synchronize()
            total_time += start_time.elapsed_time(end_time)

        self.avg_iteration_time = total_time / num_test_samples if num_test_samples > 0 else 0.0
        return y_pred