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
from GAE_generate_graph_tensors import ast_visit,graph_to_dict,make_graph_tensors
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

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        code_gpt_out = self.code_gpt(sent_id, attention_mask=mask)

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
model = gpt
optimizer = AdamW(model.parameters(), lr=0.001)

id = 0
for id in range(len(input_tokens)):
    test_input = input_tokens[id]
    input_tensor = torch.tensor(test_input).reshape(1, 1, -1).to(device)
    # test_target = test_input[:]
    # test_target.append(target_tokens[id])
    target_token = target_tokens[id]
    target = torch.nn.functional.one_hot(torch.tensor(target_token), num_classes=50001)

    #output = model(torch.tensor(test_input).reshape(-1,1), labels=torch.tensor(test_target).reshape(-1,1), attention_mask=torch.ones(len(test_input)).reshape(-1,1))
    output = model(input_tensor, hidden=torch.tensor([1,2,3]).reshape(1,1,-1),labels=input_tensor)
    #output_str = tokenizer.decode(torch.argmax(torch.softmax(output.logits, dim=2), dim=-1).reshape(-1).numpy())
    loss = output.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss.item())
print("")