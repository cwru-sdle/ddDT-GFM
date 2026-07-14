# %%
''' load libraries '''

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected


# %%
''' model definition '''

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, stride), (0, padding))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, stride), (0, padding))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, stride), (0, padding))

    def forward(self, X):
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        return H


class STConvEncoder(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels,
                 kernel_size, stride, padding, K, normalization='sym', bias=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias

        self._temporal_conv1 = TemporalConv(in_channels, hidden_channels, kernel_size, stride, padding)
        self._graph_conv = GCNConv(hidden_channels, hidden_channels, bias=bias)
        self._temporal_conv2 = TemporalConv(hidden_channels, out_channels, kernel_size, stride, padding)

    def forward(self, X, edge_index, edge_weight=None):
        T_0 = self._temporal_conv1(X)
        T = torch.zeros_like(T_0).to(T_0.device)
        for b in range(T_0.size(0)):
            for t in range(T_0.size(1)):
                T[b][t] = self._graph_conv(T_0[b][t], edge_index)
        T = F.relu(T)
        T = self._temporal_conv2(T)
        return T


# %%
''' load data '''

REPO_ROOT = Path(__file__).resolve().parents[3]
PARQUET_PATH = REPO_ROOT / 'data' / '111417_width_measure_interval_lpbf.parquet'

table = pq.read_table(PARQUET_PATH)
df = table.to_pandas().dropna(subset=['distance'])


# %%
''' extract node features and labels '''

node_features = []
node_labels_1 = []
node_labels_2 = []

for sanu, group in df.groupby('sanu'):
    distances = group['distance'].values
    sasp_value = group['sasp'].iloc[0]
    sapw_value = group['sapw'].iloc[0]
    labels = group['complete'].iloc[0]
    labels_2 = group['pore_obs'].iloc[0]
    if len(distances) == 40:
        trimmed_distances = distances[3:-3]
        node_feature_vector = [sanu] + list(trimmed_distances) + [sasp_value, sapw_value]
        node_features.append(node_feature_vector)
        node_labels_1.append(labels)
        node_labels_2.append(labels_2)


# %%
''' split train/valid/test with a fixed seed '''

def split_array(data):
    arr = copy.deepcopy(data)
    random.seed(42)
    random.shuffle(arr)
    cut = int(len(arr) * 0.8)
    return arr[:cut], arr[cut:]


train_features, test_features = split_array(node_features)
train_features, valid_features = split_array(train_features)

train_labels_1, test_labels_1 = split_array(node_labels_1)
train_labels_1, valid_label_1 = split_array(train_labels_1)

train_labels_2, test_labels_2 = split_array(node_labels_2)
train_labels_2, valid_label_2 = split_array(train_labels_2)


# %%
''' build edges by energy-density similarity '''

RATIO_THRESHOLD = 0.3


def build_edge_index(features, ratio_threshold=RATIO_THRESHOLD):
    edges = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            sasp1, sapw1 = features[i][-2], features[i][-1]
            sasp2, sapw2 = features[j][-2], features[j][-1]
            egy_den1 = sapw1 / sasp1
            egy_den2 = sapw2 / sasp2
            if abs(egy_den1 - egy_den2) < ratio_threshold:
                edges.append([i, j])
    return to_undirected(torch.tensor(edges).t())


train_edge_index = build_edge_index(train_features)
valid_edge_index = build_edge_index(valid_features)
test_edge_index = build_edge_index(test_features)


# %%
''' reshape into (batch, time_steps, num_nodes, num_features) '''

NUM_TIME_STEPS = 34


def features_to_tensor(features):
    for row in features:
        del row[0]
        density = row[-1] / row[-2]
        row.append(density)
        row.append(0)

    temp = []
    for row in features:
        last_four = row[-4:]
        split_data = [[row[i]] for i in range(NUM_TIME_STEPS)]
        for sublist in split_data:
            sublist.extend(last_four)
        temp.append(split_data)

    x = torch.tensor(temp, dtype=torch.float)
    x = x.permute(1, 0, 2)
    x = x.unsqueeze(0)
    return x


train_features = features_to_tensor(train_features)
valid_features = features_to_tensor(valid_features)
test_features = features_to_tensor(test_features)


# %%
''' assemble PyG Data objects '''

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_labels_1 = torch.tensor(train_labels_1, dtype=torch.long)
valid_label_1 = torch.tensor(valid_label_1, dtype=torch.long)
test_labels_1 = torch.tensor(test_labels_1, dtype=torch.long)

train_labels_2 = torch.tensor(train_labels_2, dtype=torch.long)
valid_label_2 = torch.tensor(valid_label_2, dtype=torch.long)
test_labels_2 = torch.tensor(test_labels_2, dtype=torch.long)

train_data = Data(x=train_features, edge_index=train_edge_index, y=train_labels_1)
valid_data = Data(x=valid_features, edge_index=valid_edge_index, y=valid_label_1)
test_data = Data(x=test_features, edge_index=test_edge_index, y=test_labels_1)


# %%
''' train the model '''

EPOCHS = 200
LEARNING_RATE = 1e-4
NUM_FEATURES = 5
NUM_NODES = train_features.shape[2]

model = STConvEncoder(
    num_nodes=NUM_NODES,
    in_channels=NUM_FEATURES,
    hidden_channels=16,
    out_channels=2,
    kernel_size=3,
    stride=7,
    padding=1,
    K=2,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(model, data, val_data, optimizer, criterion, epochs):
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
    }
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        pred = torch.argmax(output.squeeze(), dim=1)
        correct = (pred == data.y).sum().item()
        epoch_accuracy = 100 * correct / len(data.y)
        history['train_accuracies'].append(epoch_accuracy)

        loss = criterion(output.squeeze(), data.y.squeeze())
        history['train_losses'].append(loss.item())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(val_data.x, val_data.edge_index)
            val_pred = torch.argmax(val_output.squeeze(), dim=1)
            val_correct = (val_pred == val_data.y).sum().item()
            val_accuracy = 100 * val_correct / len(val_data.y)
            history['val_accuracies'].append(val_accuracy)
            val_loss = criterion(val_output.squeeze(), val_data.y.squeeze())
            history['val_losses'].append(val_loss.item())

        print(f'Epoch {epoch + 1}/{epochs}, '
              f'Train Loss: {loss.item():.4f}, Train Acc: {epoch_accuracy:.2f}%, '
              f'Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.2f}%')

    return history


history = train(model, train_data, valid_data, optimizer, criterion, epochs=EPOCHS)


# %%
''' evaluate on test set '''

model.eval()
with torch.no_grad():
    output = model(test_data.x, test_data.edge_index)
    pred = torch.argmax(output.squeeze(), dim=1)
    correct = (pred == test_data.y).sum().item()
    accuracy = 100 * correct / len(test_data.y)
    test_loss = criterion(output.squeeze(), test_data.y.squeeze())
    print(f'Test set loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
