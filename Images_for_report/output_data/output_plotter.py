import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

data_files = glob.glob("*.pkl")
epochs = 5

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def avg_plotter(data, window=100, label=None, alpha=1, c=None):
    data_avg = running_mean(data, window)
    if c is None:
        plt.plot(np.linspace(0, epochs, len(data_avg)), data_avg, label=label, alpha=alpha)
    else:
        plt.plot(np.linspace(0, epochs, len(data_avg)), data_avg, label=label, alpha=alpha, c=c)


for file in data_files:
    with open(file, 'rb') as handle:
        (train_loss, val_loss, validate_per) = pickle.load(handle)

    file_name = file.split('.')[0]
    output_dir = "{}.png".format(file)

    plt.figure(figsize=(5, 4))
    avg_plotter(val_loss, window=1, alpha=0.4, c='tab:orange')
    avg_plotter(train_loss, window=1, alpha=0.4, c='tab:blue')
    avg_plotter(train_loss, window=1000, alpha=1, label="Train", c='tab:blue')
    avg_plotter(val_loss, window=1000, label="Validation", c='tab:orange')
    plt.xlim(0, epochs)
    plt.ylim(2,6)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(output_dir.format(file_name))

plt.figure(figsize=(5, 4))
for file in data_files:
    with open(file, 'rb') as handle:
        (train_loss, val_loss, validate_per) = pickle.load(handle)

    file_name = file.split(')')[-1].split(".")[0]

    avg_plotter(val_loss, window=2000, label="{}".format(file_name))
    plt.xlim(0, epochs)
    plt.ylim(3.9,5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend(loc='upper right')
    # plt.show()

plt.grid()
output_dir = "collate.png".format(file)
plt.savefig(output_dir.format(file_name))