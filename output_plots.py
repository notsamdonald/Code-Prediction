import pickle

import matplotlib.pyplot as plt
import numpy as np

epochs = 10


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


with open('GAE_graph_training_info.pkl', 'rb') as handle:
    (loss_tracker, val_loss_tracker, acc_tracker,
     precision_tracker, recall_tracker, f1_tracker,
     TP_tracker, FN_tracker, TN_tracker, FP_tracker) = pickle.load(handle)


def avg_plotter(data, window=100, label=None, alpha=1, c=None):
    data_avg = running_mean(data, window)
    plt.plot(np.linspace(0, epochs, len(data_avg)), data_avg, label=label, alpha=alpha, c=c)


output_dir = "output_graphs/combine_8_{}.png"

epochs = 10
file = "GAE_training_loss"
plt.figure(figsize=(5, 4))
avg_plotter(val_loss_tracker, window=20, alpha=0.4, c='tab:orange')
avg_plotter(loss_tracker, window=20, alpha=0.4, c='tab:blue')
avg_plotter(loss_tracker, window=1000, alpha=1, label="Train", c='tab:blue')
avg_plotter(val_loss_tracker, window=1000, label="Validation", c='tab:orange')
plt.xlim(0, epochs)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend(loc='upper right')
# plt.show()
plt.savefig(output_dir.format(file))

file = "GAE_confusion_matrix"
plt.figure(figsize=(5, 4))
avg_plotter(acc_tracker, window=3,label="Accuracy", c='tab:blue')
avg_plotter(precision_tracker, window=3,label="Precision", c='tab:orange')
avg_plotter(TP_tracker, window=3,label="TPR/Recall", c='tab:red')
avg_plotter(TN_tracker, window=3, label="TNR/Specificity", c='tab:green')
plt.xlim(0, epochs)
plt.xlabel('Epoch')
plt.ylabel('Metric value')
plt.grid()
plt.legend(loc='lower right')
# plt.show()
plt.savefig(output_dir.format(file))