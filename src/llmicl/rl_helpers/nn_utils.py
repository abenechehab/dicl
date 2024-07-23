from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim



class NeuralNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_layers: int = 2,
        layer_size: int = 128
    ):
        """
        Base class for the NN

        Args:
            input_size: Number of dimension in the input
                (n_observations + n_actions * prediction_horizon)
            output_size: Number of dimension in the output
                (n_observations * prediction_horizon)
        """
        super(NeuralNet, self).__init__()

        self.linear0 = nn.Linear(input_size, layer_size)
        self.act0 = nn.ReLU()

        self.common_block = nn.Sequential()
        for i in range(n_layers):
            self.common_block.add_module(
                f"layer{i+1}-lin", nn.Linear(layer_size, layer_size)
            )
            self.common_block.add_module(f"layer{i+1}-act", nn.ReLU())

        self.mu = nn.Sequential(
            nn.Linear(layer_size, output_size),
        )

    def forward(self, x):
        x = self.linear0(x)
        x = self.act0(x)
        x = self.common_block(x)
        return self.mu(x)

def train_mlp(model, X_train, y_train):
    learning_rate = 0.001

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    n_epochs = 100

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).type(torch.FloatTensor),
        torch.from_numpy(y_train).type(torch.FloatTensor),
    )

    validation_fraction = 0.1
    n_samples = len(dataset)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    ind_split = int(np.floor((1 - validation_fraction) * n_samples))
    train_indices, val_indices = indices[:ind_split], indices[ind_split:]

    dataset_valid = torch.utils.data.TensorDataset(*dataset[val_indices])
    dataset_train = torch.utils.data.TensorDataset(*dataset[train_indices])

    dataset_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        drop_last=False,
    )
    dataset_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=len(dataset_valid), shuffle=False
    )

    tracked_losses = []
    val_tracked_losses = []

    n_train = len(dataset_train.dataset)

    for _ in tqdm(range(1, 1 + n_epochs), desc="training epochs"):
        model.train()

        train_loss = 0
        for _, (x, y) in enumerate(dataset_train):
            model.zero_grad()

            out = model(x)
            loss = loss_fn(y, out)
            train_loss += len(x) * loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= n_train
        tracked_losses.append(train_loss)

        # evaluation
        model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(dataset_valid):
                out = model(x)
                loss = loss_fn(y, out)
        val_tracked_losses.append(loss)
    return tracked_losses, val_tracked_losses, model
