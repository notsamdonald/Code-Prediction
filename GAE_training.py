import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.nn import GAE
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from tqdm import tqdm

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def main():
    with open('GAE_graph_tensors.pkl', 'rb') as fp:
        train_graph_tensors, val_graph_tensors = pickle.load(fp)

    # Model config
    out_channels = 8
    num_features = train_graph_tensors[0].num_features  # 5
    model = GAE(GCNEncoder(num_features, out_channels))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training config
    training_size = 500
    epochs = 1
    val_size = 50

    # Tracking arrays
    loss_tracker = []
    val_loss_tracker = []
    precision_tracker = []
    recall_tracker = []
    acc_tracker = []
    f1_tracker = []
    TP_tracker = []
    FP_tracker = []
    TN_tracker = []
    FN_tracker = []
    import copy
    for epoch in range(1, epochs + 1):

        # Training
        model.train()
        optimizer.zero_grad()
        for graph_n, graph in tqdm(enumerate(train_graph_tensors)):
            graph_2 = copy.deepcopy(graph)

            # Preparing data
            x = graph_2.x.to(device)
            data = train_test_split_edges(graph_2, test_ratio=1, val_ratio=0)
            #data = data.to(device)
            #train_pos_edge_index = data.train_pos_edge_index.to(device)

            # Forward pass
            z = model.encode(x, data.test_pos_edge_index.to(device))
            loss = model.recon_loss(z, data.test_pos_edge_index.to(device), data.test_neg_edge_index.to(device))

            # Tracking loss
            loss_tracker.append(loss.item())

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (graph_n + 1) % 16 == 0:
                optimizer.step()

        # Validation
        model.eval()
        TP, FN, TN, FP, n_pos, n_neg = 0, 0, 0, 0, 0, 0
        for graph_n, graph in tqdm(enumerate(val_graph_tensors)):
            # Preparing data
            graph_2 = copy.deepcopy(graph)
            x = graph_2.x.to(device)
            data = train_test_split_edges(graph_2, test_ratio=1, val_ratio=0)
            train_pos_edge_index = data.train_pos_edge_index.to(device)

            # Forward pass
            z = model.encode(x, data.test_pos_edge_index.to(device))
            val_loss = model.recon_loss(z, data.test_pos_edge_index.to(device))

            # Tracking metrics
            val_loss_tracker.append(val_loss.item())
            neg_preds = model.decode(z, data.test_neg_edge_index.to(device)).detach().cpu().numpy() < 0.5
            pos_preds = model.decode(z, data.test_pos_edge_index.to(device)).detach().cpu().numpy() > 0.5
            n_pos += len(pos_preds)
            n_neg += len(neg_preds)
            TP += np.sum(pos_preds)
            FN += (len(pos_preds) - np.sum(pos_preds))
            TN += np.sum(neg_preds)
            FP += (len(neg_preds) - np.sum(neg_preds))

        # Total validation set metrics
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)
        acc = (TP + TN) / (TP + FP + TN + FN)

        precision_tracker.append(precision)
        recall_tracker.append(recall)
        acc_tracker.append(acc)
        f1_tracker.append(F1)
        TP_tracker.append(TP / n_pos)
        FP_tracker.append(FP / n_neg)
        TN_tracker.append(TN / n_neg)
        FN_tracker.append(FN / n_pos)

        # Printing loss and other accuracy metrics
        print('Epoch:{:03d}, Loss:{:.4f}, Val Loss:{:.4f}'.format(epoch, np.mean(loss_tracker[-training_size:]),
                                                                  np.mean(val_loss_tracker[-val_size:])))

        print("Acc:{:.3f}, Precision:{:.3f}, Recall:{:.3f}, F1{:.3f}".format(acc, precision, recall, F1))
        print("TP:{}, FN:{}, TN:{}, FP{}".format(TP, FN, TN, FP))
        print("\n")

    # Saving model:
    # TODO!
    torch.save(model, "GAE")

    # Plotting results of training
    plt.plot(np.linspace(0, epochs, len(running_mean(loss_tracker, training_size))),
             running_mean(loss_tracker, training_size))
    plt.plot(np.linspace(0, epochs, len(running_mean(val_loss_tracker, val_size))),
             running_mean(val_loss_tracker, val_size))

    plt.show()

    plt.plot(acc_tracker, label="accuracy")
    plt.plot(precision_tracker, label="precision")
    plt.plot(recall_tracker, label="recall")
    plt.plot(f1_tracker, label="F1")
    plt.legend()
    plt.show()

    plt.plot(TP_tracker, label="TPR")
    plt.plot(FN_tracker, label="FNR")
    plt.plot(TN_tracker, label="TNR")
    plt.plot(FP_tracker, label="FPR")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
