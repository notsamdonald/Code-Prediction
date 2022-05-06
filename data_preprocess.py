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
from GAE_generate_graph_tensors import generate_AST_graph_tensor
from torch_geometric.nn import GAE
from GAE_training import GCNEncoder
from torch_geometric.utils import train_test_split_edges
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

tokenizer = GPT2TokenizerFast.from_pretrained("microsoft/CodeGPT-small-py", add_prefix_space=True)

dataset = load_dataset("code_search_net", "python")

input_tokens = []
target_tokens = []
embedded_graphs = []

model = torch.load('GAE')
model.eval()
model.to(device)


for i, function in tqdm(enumerate(dataset['train'])):

    if i > 1000:
        break

    function_string = function['func_code_string']
    function_tokens = function['func_code_tokens']

    pre_tokens = tokenizer(function_tokens, is_split_into_words=True)

    if len(pre_tokens.data['input_ids']) < 784:
        split_ratio = 0.9
        tokens = pre_tokens.data['input_ids']
        split_id = int(len(tokens)*split_ratio)

        target_token = tokens[split_id]
        input_token = tokens[:split_id]

        input_tokens.append(input_token)
        target_tokens.append(target_token)

        if True:
            # AST Generation
            ast_dicts = []

            try:
                ast_graph = ast_visit(ast.parse(function_string), parent_node=Node("Root"))
                ast_dict = graph_to_dict(ast_graph)
                ast_dicts.append(ast_dict)
                ast_graph_tensor = make_graph_tensors(ast_dicts)[0]

                graph_2 = copy.deepcopy(ast_graph_tensor)

                # Preparing data
                x = graph_2.x.to(device)
                data = train_test_split_edges(graph_2, test_ratio=1, val_ratio=0)
                # data = data.to(device)
                # train_pos_edge_index = data.train_pos_edge_index.to(device)

                # Forward pass
                z = model.encode(x, data.test_pos_edge_index.to(device))


                # Appending on sucessful AST generation
                input_tokens.append(input_token)
                target_tokens.append(target_token)
                embedded_graphs.append(z)
            except SyntaxError or TypeError:
                pass


data = (input_tokens, target_tokens, embedded_graphs)
with open('train_data(input_target_z).pkl', 'wb') as handle:
    pickle.dump(data, handle)

#with open('train_data(input_target_z).pkl', 'rb') as handle:
#    b = pickle.load(handle)

print("")



