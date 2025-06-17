import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        epoch_time = time.time() - start_time
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(
            f'Epoch {epoch + 1}/{num_epochs}.. Train loss: {train_loss:.4f}.. Val loss: {val_loss:.4f}.. Time: {epoch_time:.2f}s')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model


def evaluate_model(model, test_loader, scaler_y, device):
    model.to(device)
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            predictions.extend(scaler_y.inverse_transform(outputs))
            actuals.extend(scaler_y.inverse_transform(targets.numpy()))

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    return predictions, actuals


def train_multitask_model(model, train_loader, val_loader, num_epochs, learning_rate, device, loss_weights):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for (x_short, x_long), targets in train_loader:
            x_short, x_long, targets = x_short.to(device), x_long.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(x_short, x_long)

            # 计算加权总损失
            loss = 0
            for i, output in enumerate(outputs):
                loss += criterion(output, targets[:, i].unsqueeze(1)) * loss_weights[i]

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x_short.size(0)

        print(f'Epoch {epoch + 1}/{num_epochs}.. Train loss: {total_train_loss / len(train_loader.dataset):.4f}')

    return model