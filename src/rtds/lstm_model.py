"""The LSTM behind the SurvivalModel interface, so SurvSHAP(t) and Run 2 use it unchanged.

It accepts the same flat columns as the forest (visit_1, visit_2, visit_3), rearranges
them into a sequence in visit order inside, and returns sksurv StepFunctions on the grid.
"""

import copy

import numpy as np
import torch
from sksurv.functions import StepFunction

from rtds import seeds
from rtds.lstm_net import LSTMHazard, interval_index, neg_log_likelihood


class LSTMModel:
    def __init__(self, cfg, dataset_index, grid):
        self.p = cfg["lstm"]
        self.grid = np.asarray(grid, dtype=float)
        self.master_seed = cfg["master_seed"]
        self.dataset_index = dataset_index
        self.history = []  # (epoch, training loss, validation loss), for the pilot report

    def _sequence(self, X):
        # Order by the visit number in the column name, never by the DataFrame's column order.
        cols = sorted(X.columns, key=lambda c: int(c.removeprefix("visit_")))
        return torch.tensor(X[cols].to_numpy(dtype=np.float32)).unsqueeze(-1)  # (patients, 3, 1)

    def fit(self, X, y):
        # One thread per model: parallelism is across datasets, and one thread keeps results reproducible.
        torch.set_num_threads(1)
        torch.manual_seed(seeds.int_seed(self.master_seed, "lstm", self.dataset_index))
        gen = seeds.rng(self.master_seed, "lstm", self.dataset_index)

        seq = self._sequence(X)
        k = torch.tensor(interval_index(y["time"], self.grid), dtype=torch.long)
        event = torch.tensor(y["event"], dtype=torch.bool)
        order = gen.permutation(len(seq))
        n_val = int(round(self.p["val_fraction"] * len(seq)))
        val, train = order[:n_val], order[n_val:]  # validation patients come from the training set

        self.net = LSTMHazard(self.p["hidden_size"], len(self.grid))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.p["learning_rate"])
        best, best_state, waited = np.inf, None, 0
        for epoch in range(self.p["max_epochs"]):
            self.net.train()
            batches = gen.permutation(train)
            train_loss = []
            for start in range(0, len(batches), self.p["batch_size"]):
                b = batches[start:start + self.p["batch_size"]]
                opt.zero_grad()
                loss = neg_log_likelihood(self.net(seq[b]), k[b], event[b])
                loss.backward()
                opt.step()
                train_loss.append(loss.item())
            self.net.eval()
            with torch.no_grad():
                val_loss = neg_log_likelihood(self.net(seq[val]), k[val], event[val]).item()
            self.history.append((epoch + 1, float(np.mean(train_loss)), val_loss))
            # Early stopping: keep the weights with the lowest validation loss.
            if val_loss < best:
                best, best_state, waited = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                waited += 1
                if waited >= self.p["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.best_epoch = int(np.argmin([h[2] for h in self.history])) + 1
        return self

    def _survival(self, X):
        with torch.no_grad():
            h = torch.sigmoid(self.net(self._sequence(X)))
        return torch.cumprod(1 - h, dim=1).numpy().astype(float)  # S at each grid point

    def predict_survival_function(self, X):
        # Defined at the grid points, where it is exact; the explainer always evaluates on the grid.
        return np.array([StepFunction(self.grid, s) for s in self._survival(X)])

    def predict_risk(self, X):
        """Risk score for Harrell's C-index: less area under the survival curve = higher risk."""
        return -self._survival(X).sum(axis=1)

    def assert_grid_inside(self, X, grid):
        lo, hi = self.predict_survival_function(X.iloc[[0]])[0].domain
        assert lo <= grid[0] and grid[-1] <= hi, f"grid [{grid[0]}, {grid[-1]}] outside LSTM range [{lo}, {hi}]"
        return lo, hi
