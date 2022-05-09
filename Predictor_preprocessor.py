"""
Preprocessing function to tokenize functions and encode ASTs in accordance with the specified GAE
"""

import ast
import copy
import pickle

import torch
from anytree import Node
from datasets import load_dataset
from torch_geometric.utils import train_test_split_edges
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from Graph_generator import ast_visit, graph_to_dict, make_graph_tensors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

tokenizer = GPT2TokenizerFast.from_pretrained("microsoft/CodeGPT-small-py", add_prefix_space=True)

dataset = load_dataset("code_search_net", "python")

input_tokens = []
target_tokens = []
embedded_graphs = []

# Loading trained model
model = torch.load('GAE_models/GAE')
model.eval()
model.to(device)

split_type = "validation"
dataset_size = 5000  # Number of functions to process

for i, function in tqdm(enumerate(dataset[split_type])):

    if i > dataset_size:
        break

    # Extracting strings and tokens from dataset
    function_string = function['func_code_string']
    function_tokens = function['func_code_tokens']

    pre_tokens = tokenizer(function_tokens, is_split_into_words=True)

    # Discarding tokens over 784 in length (not needed for current LSTM downstream application)
    if len(pre_tokens.data['input_ids']) < 784:

        split_ratio = 0.9  # This is not needed for the current configuration, yet is splitting the function at 90%
        tokens = pre_tokens.data['input_ids']
        split_id = int(len(tokens) * split_ratio)

        target_token = tokens[split_id]
        input_token = tokens[:split_id]

        # AST Generation
        ast_dicts = []
        try:
            # Generating and walking AST graph
            ast_graph = ast_visit(ast.parse(function_string), parent_node=Node("Root"))
            ast_dict = graph_to_dict(ast_graph)
            ast_dicts.append(ast_dict)
            ast_graph_tensor = make_graph_tensors(ast_dicts)[0]
            graph_2 = copy.deepcopy(ast_graph_tensor)

            # Preparing data
            x = graph_2.x.to(device)
            data = train_test_split_edges(graph_2, test_ratio=1, val_ratio=0)

            # Forward pass
            z = model.encode(x, data.test_pos_edge_index.to(device))

            # Appending only on successful AST generation
            input_tokens.append(input_token)
            target_tokens.append(target_token)
            embedded_graphs.append(z)

        except SyntaxError or TypeError:
            # Occasionally the functions cannot be compiled and will throw syntax or type errors
            pass

# Collating and saving data for downstream training of the predictor model
data = (input_tokens, target_tokens, embedded_graphs)
with open('{}_data(input_target_z).pkl'.format(split_type), 'wb') as handle:
    pickle.dump(data, handle)