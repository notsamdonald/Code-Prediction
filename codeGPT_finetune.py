from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
import itertools
from transformers import GPT2TokenizerFast
import numpy as np
from tqdm import tqdm
from anytree import Node
import ast
from Graph_generator import ast_visit,graph_to_dict,make_graph_tensors
import pickle
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn


class COMB_Arch(nn.Module):

    def __init__(self, code_gpt):
        super(COMB_Arch, self).__init__()

        self.code_gpt = code_gpt

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(50001, 50001)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(50001, 50001)

        self.output_1 = nn.Linear(768, 768*2) # We can add the hidden layer into this!
        self.output_2 = nn.Linear(768*2, 50001) # We can add the hidden layer into this!

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

        self.criterion = nn.CrossEntropyLoss()

    # define the forward pass
    def forward(self, input_tensor,labels):
        # pass the inputs to the model
        code_gpt_out = self.code_gpt(input_tensor,labels=input_tensor)

        hidden = code_gpt_out.hidden_states[12]

        x = self.output_1(hidden)
        x = self.relu(x)
        x = self.output_2(x)
        x = self.softmax(x)

        loss = self.criterion(torch.softmax(x, dim=-1).squeeze()[1:,:],input_tensor.squeeze()[:-1])

        return x, loss


        final_token = code_gpt_out.logits[:, :, -1, :]

        x = self.fc1(final_token)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
with open('train_data_100.pkl', 'rb') as handle:
    input_tokens, target_tokens = pickle.load(handle)



gpt = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-py")
tokenizer = GPT2TokenizerFast.from_pretrained("microsoft/CodeGPT-small-py", add_prefix_space=True)
gpt_model = gpt
gpt_model.config.output_hidden_states = True

# freeze all the parameters
for param in gpt_model.parameters():
    param.requires_grad = False


model = COMB_Arch(gpt_model)

optimizer = AdamW(model.parameters(), lr=0.01)

for epoch in range(10):
    epoch_loss = []
    for id in tqdm(range(len(input_tokens))):
        input = input_tokens[id]
        input_tensor = torch.tensor(input).reshape(1, 1, -1).to(device)

        target = input[1:]
        target.append(target_tokens[id])

        target_tensor = torch.tensor(target).reshape(1, 1, -1).to(device)

        # test_target.append(target_tokens[id])

        # output = model(torch.tensor(test_input).reshape(-1,1), labels=torch.tensor(test_target).reshape(-1,1), attention_mask=torch.ones(len(test_input)).reshape(-1,1))
        output = model(input_tensor, labels=input_tensor)
        # output_str = tokenizer.decode(torch.argmax(torch.softmax(output.logits, dim=2), dim=-1).reshape(-1).numpy())
        loss = output[1]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss.append(loss.item())

        #print(np.average(epoch_loss[-100:]))
        print(epoch_loss[-1])