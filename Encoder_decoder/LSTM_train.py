from model import Code_GPT_modified
import torch
import pickle
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('train_data_100.pkl', 'rb') as handle:
    input_tokens, target_tokens = pickle.load(handle)

model = Code_GPT_modified(50001, 768, 768, 1)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=0.001)

for epoch in range(10):
    epoch_loss = []
    for id in tqdm(range(len(input_tokens))):
        test_input = input_tokens[id]
        input_tensor = torch.tensor(test_input).reshape(1, -1).to(device)

        target_token = target_tokens[id]
        target = torch.nn.functional.one_hot(torch.tensor(target_token), num_classes=50001)

        output = model(input_tensor,labels=input_tensor)

        loss = output[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss.append(loss.item())

        if (id + 1) % 100 == 0:
            print(np.average(epoch_loss[-200:]))