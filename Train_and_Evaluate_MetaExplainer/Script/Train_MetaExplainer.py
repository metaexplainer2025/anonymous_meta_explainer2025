import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
# from torch_geometric.data import DataLoader
import argparse
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from math import sqrt
from statistics import mean
import torch_geometric
from torch_geometric.datasets import TUDataset
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from sklearn import metrics
from scipy.spatial.distance import hamming
import statistics
import pandas
import csv
from time import perf_counter
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch_geometric.nn as gnn
from torch.autograd import graph
from typing import Any, Dict, Optional, Union
from IPython.core.display import deepcopy
from torch_geometric.nn import MessagePassing
import copy
from importlib import reload
import pickle
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm
from torch_geometric.data import Data, Batch, Dataset


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score, f1_score
#
# directory_x = ("/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Creation_for_MetaExplainer/Experimental Results/" +
#                "X.pt")
# directory_y = ("/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Creation_for_MetaExplainer/Experimental Results/" +
#                "Y.pt")
# X_list = torch.load(directory_x)
# Y_list = torch.load(directory_y)
#
# X_data = torch.stack(X_list).float()
# Y_data = torch.stack(Y_list).float()
#
# if Y_data.dim() > 1 and Y_data.size(1) > 1:
#     Y_data = torch.argmax(Y_data, dim=1)
#
# X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42, shuffle=True)
#
#
# import torch
# from collections import Counter
#
#
# class_counts = Counter(Y_data.tolist())
#
# print("Class distribution:")
# for class_label, count in class_counts.items():
#     print(f"Class {class_label}: {count} samples")
#
# class MetaExplainer(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MetaExplainer, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# input_size = X_train.shape[1]
# hidden_size = 64
# output_size = len(torch.unique(Y_data))
#
# meta_explainer = MetaExplainer(input_size, hidden_size, output_size)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(meta_explainer.parameters(), lr=0.001)
#
# num_epochs = 500
# batch_size = 32
#
# train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# epoch_loss_list = []
# epoch_accuracy_list = []
# test_loss_list = []
# test_accuracy_list = []
#
# for epoch in range(num_epochs):
#     meta_explainer.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#
#     for batch_X, batch_Y in train_loader:
#         outputs = meta_explainer(batch_X)
#         loss = criterion(outputs, batch_Y)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         correct += (predicted == batch_Y).sum().item()
#         total += batch_Y.size(0)
#
#     avg_loss = total_loss / len(train_loader)
#     accuracy = 100 * correct / total
#
#     epoch_loss_list.append(avg_loss)
#     epoch_accuracy_list.append(accuracy)
#
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
#
# meta_explainer.eval()
# test_loss = 0
# test_correct = 0
# test_total = 0
# all_test_outputs = []
# all_test_labels = []
# precision_list = []
# recall_list = []
# f1_list = []
#
# with torch.no_grad():
#     for test_X, test_Y in test_loader:
#         test_outputs = meta_explainer(test_X)
#         test_loss += criterion(test_outputs, test_Y).item()
#
#         _, test_predicted = torch.max(test_outputs, 1)
#         test_correct += (test_predicted == test_Y).sum().item()
#         test_total += test_Y.size(0)
#         all_test_outputs.extend(test_predicted.cpu().numpy())
#         all_test_labels.extend(test_Y.cpu().numpy())
#
# avg_test_loss = test_loss / len(test_loader)
# test_accuracy = 100 * test_correct / test_total
#
# test_loss_list.append(avg_test_loss)
# test_accuracy_list.append(test_accuracy)
#
# print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
#
# epochs = range(1, num_epochs + 1)
#
# plt.figure(figsize=(12, 5))
#
# plt.subplot(1, 2, 1)
# plt.plot(epochs, epoch_loss_list, 'b', label='Training Loss')
# plt.title('Training Loss', fontsize=22)
# plt.xlabel('Epochs', fontsize=20)
# plt.ylabel('Loss', fontsize=18, labelpad=5)
# plt.legend(fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=14)
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs, epoch_accuracy_list, 'r', label='Training Accuracy')
# plt.title('Training Accuracy', fontsize=22)
# plt.xlabel('Epochs', fontsize=20)
# plt.ylabel('Accuracy (%)', fontsize=18, labelpad=2)
# plt.legend(fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=14)
#
# directory_plot = ("/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Representation_Learning/Experimental Results/MetaExplaienr_Accuracy.pdf")
# plt.savefig(directory_plot)
# # plt.show()
# plt.close()
#
#
#
#
# precision = precision_score(all_test_labels, all_test_outputs, average='weighted')
# recall = recall_score(all_test_labels, all_test_outputs, average='weighted')
# f1 = f1_score(all_test_labels, all_test_outputs, average='weighted')
# precision_list.append(precision)
# recall_list.append(recall)
# f1_list.append(f1)
#
# print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
# print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')




import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict, Counter
import pickle

# ========== Config (edit as you like) ==========
RECALL_AVERAGING = "macro"   # <-- change to "macro" (was "weighted")
PRINT_CLASS_DISTRIBUTION = True
# ===============================================

# 1) Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

weights_distribution = "UniformDistribution"#"GridSearch"
num_metrics = 6
weight_levels = [1, 2, 3, 4]
if weights_distribution == "UniformDistribution":
    num_samples = 4000
    file_names = weights_distribution + "_" + str(num_samples) + "_Samples" + "_Weights_" + '_'.join(str(i) for i in weight_levels)
elif weights_distribution == "GridSearch":
    file_names = weights_distribution + "_Weights_" + '_'.join(str(i) for i in weight_levels)

directory_x = (
    "/data/cs.aau.dk/ey33jw/Explainability_Methods/"
    "Dataset_Creation_for_MetaExplainer/Experimental Results/"
    f"X_{file_names}.pt"
)
directory_y = (
    "/data/cs.aau.dk/ey33jw/Explainability_Methods/"
    "Dataset_Creation_for_MetaExplainer/Experimental Results/"
    f"Y_{file_names}.pt"
)
X_list = torch.load(directory_x)
Y_list = torch.load(directory_y)

X_data = torch.stack(X_list).float()
Y_data = torch.stack(Y_list).float()
if Y_data.dim() > 1 and Y_data.size(1) > 1:
    Y_data = torch.argmax(Y_data, dim=1)

# Optional: inspect overall class distribution
if PRINT_CLASS_DISTRIBUTION:
    counts = Counter(Y_data.tolist())
    total = len(Y_data)
    print("Overall class distribution:")
    for k in sorted(counts.keys()):
        print(f"  class {k}: {counts[k]} ({counts[k]/total:.2%})")

class MetaExplainer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MetaExplainer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_and_evaluate(train_loader, test_loader, input_size, hidden_size, output_size, num_epochs):
    meta_explainer = MetaExplainer(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(meta_explainer.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        meta_explainer.train()
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            outputs = meta_explainer(batch_X)
            loss = criterion(outputs, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    meta_explainer.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_test_outputs = []
    all_test_labels = []

    with torch.no_grad():
        for test_X, test_Y in test_loader:
            test_X = test_X.to(device)
            test_Y = test_Y.to(device)

            test_outputs = meta_explainer(test_X)
            test_loss += criterion(test_outputs, test_Y).item()
            _, test_predicted = torch.max(test_outputs, 1)
            test_correct += (test_predicted == test_Y).sum().item()
            test_total += test_Y.size(0)

            all_test_outputs.extend(test_predicted.cpu().numpy())
            all_test_labels.extend(test_Y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / test_total

    # --- Metrics ---
    precision_weighted = precision_score(all_test_labels, all_test_outputs, average='weighted')
    recall_macro       = recall_score(all_test_labels, all_test_outputs, average=RECALL_AVERAGING)  # "macro"
    f1_weighted        = f1_score(all_test_labels, all_test_outputs, average='weighted')

    recall_weighted_dbg = recall_score(all_test_labels, all_test_outputs, average='weighted')

    # Debug note if recall_weighted equals accuracy (expected for single-label)
    if abs(recall_weighted_dbg - test_accuracy) < 1e-12:
        pass  # expected; no print to keep logs clean

    return avg_test_loss, test_accuracy, precision_weighted, recall_macro, f1_weighted

hidden_size = 64
output_size = len(torch.unique(Y_data))
num_epochs = 500

scenarios = [0.2, 0.4, 0.6, 0.8]
test_size = 0.2
results = defaultdict(list)

for train_size in scenarios:
    print(f"\nTraining with {int(train_size * 100)}% of the data...")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_data, Y_data,
        test_size=test_size,
        train_size=train_size,
        random_state=42,
        shuffle=True
    )

    if PRINT_CLASS_DISTRIBUTION:
        counts_test = Counter(Y_test.tolist())
        total_test = len(Y_test)
        print("  Test set distribution:")
        for k in sorted(counts_test.keys()):
            print(f"    class {k}: {counts_test[k]} ({counts_test[k]/total_test:.2%})")

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset  = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    avg_test_loss, test_accuracy, precision, recall, f1 = train_and_evaluate(
        train_loader,
        test_loader,
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        output_size=output_size,
        num_epochs=num_epochs
    )

    results['train_size'].append(train_size * 100)
    results['test_loss'].append(avg_test_loss)
    results['test_accuracy'].append(test_accuracy)
    results['precision_weighted'].append(precision)
    results['recall_macro'].append(recall)
    results['f1_weighted'].append(f1)

    print(f"Test Loss: {avg_test_loss:.4f}, "
          f"Acc: {test_accuracy:.2%}, "
          f"Prec(w): {precision:.4f}, "
          f"Rec(macro): {recall:.4f}, "
          f"F1(w): {f1:.4f}")

results_file = (
    "/data/cs.aau.dk/ey33jw/Explainability_Methods/"
    "Dataset_Representation_Learning/Experimental Results/"
    f"meta_explainer_results_{file_names}.pkl"
)
with open(results_file, "wb") as f:
    pickle.dump(results, f)

print("\nSaved results keys:", list(results.keys()))


