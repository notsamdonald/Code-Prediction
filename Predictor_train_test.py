"""
Training of the LSTM based code token predictor model, uses data from Predictor_preprocessor

To train without the embedded AST information, do not pass a value to GAE_embedding within the Predictor model

Currently, configured for training, yet can be switched to testing
"""


import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW

from Predictor_model import LSTM_predictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def running_mean(x, N):
    # Supporting function for averaging
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def test(test_input_tokens, test_embedded_graphs, model):
    # Testing trained model

    val_ids = random.sample(range(len(test_input_tokens)), len(test_input_tokens))

    no_gae_acc_tracker = []

    for val_id in val_ids:
        input_token = test_input_tokens[val_id]
        z = test_embedded_graphs[val_id]

        input_tensor = torch.tensor(input_token).reshape(1, -1).to(device)
        z = z.to(device)

        output = model(input_tensor, GAE_embedding=z, labels=input_tensor)
        output_GAE = model(input_tensor, labels=input_tensor)

        input_decoded = input_tensor.detach().cpu().numpy()[0][1:]
        output_GAE_decoded = torch.argmax(output_GAE[1], axis=-1).detach().cpu().numpy()[0][:-1]
        output_decoded = torch.argmax(output[1], axis=-1).detach().cpu().numpy()[0][:-1]

        GAE_acc = sum(input_decoded == output_GAE_decoded) / len(input_decoded)
        no_GAE_acc = sum(input_decoded == output_decoded) / len(input_decoded)
        diff = GAE_acc - no_GAE_acc

        no_gae_acc_tracker.append(no_GAE_acc)

        print("Base acc: {:.4f}, GAE acc: {:.4f}, Diff: {:.4f}".format(no_GAE_acc, GAE_acc, diff))

    print(np.average(no_gae_acc_tracker))


def train():
    # Loading datasets with encoded AST information (embedded graphs)
    with open('train_data(input_target_z).pkl', 'rb') as handle:
        input_tokens, target_tokens, embedded_graphs = pickle.load(handle)

    with open('validation_data(input_target_z).pkl', 'rb') as handle:
        val_input_tokens, val_target_tokens, val_embedded_graphs = pickle.load(handle)

    # Model configurations
    model = LSTM_predictor(50001, 768, 768, 1)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    epochs = 5
    batch_size = 32

    # Dataset configurations
    train_size = len(input_tokens)
    val_size = int(len(val_input_tokens) / 4)
    validate_per = 500

    # Tacking arrays
    epoch_loss = []
    val_epoch_loss = []

    for epoch in range(epochs):

        # Randomizing training data
        test_ids = random.sample(range(train_size), train_size)

        for count, id in tqdm(enumerate(test_ids)):

            model.train()

            # Extracting input tokens and AST embedding Z
            input_token = input_tokens[id]
            z = embedded_graphs[id]

            # Reshaping and moving to GPU
            input_tensor = torch.tensor(input_token).reshape(1, -1).to(device)
            z = z.to(device)

            # Forward pass with GAE_embedding
            output = model(input_tensor,GAE_embedding=z, labels=input_tensor)

            # Backward pass
            loss = output[0]
            loss.backward()

            # Gradient accumulation
            if (count + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Tracking loss
            epoch_loss.append(loss.item())

            # Validation
            if (count + 1) % validate_per == 0:

                model.eval()
                val_ids = random.sample(range(val_size), val_size)

                for val_id in val_ids:
                    val_input_token = val_input_tokens[val_id]

                    input_tensor = torch.tensor(val_input_token).reshape(1, -1).to(device)
                    z = z.to(device)
                    output = model(input_tensor, labels=input_tensor)

                    loss = output[0]
                    val_epoch_loss.append(loss.item())

                print(np.average(epoch_loss[-200:]), np.average(val_epoch_loss[-200:]))

        # Plotting training metrics
        train_loss = running_mean(epoch_loss, 100)
        val_loss = running_mean(val_epoch_loss, 100)
        plt.plot(np.linspace(0, epoch + 1, len(train_loss)), train_loss, label="train loss")
        plt.plot(np.linspace(0, epoch + 1, len(val_loss)), val_loss, label="val loss")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    train()
    # test()
