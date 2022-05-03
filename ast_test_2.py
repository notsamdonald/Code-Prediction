import ast

import anytree
from anytree import Node
from anytree.exporter import DictExporter
from datasets import load_dataset
from graphviz import Source
from graphviz import render
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
from torch_geometric.data import Data
import torch

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE
import torch.nn as nn
import ctree





def ast_visit(node, parent_node, level=0):
    name = type(node).__name__

    output = name
    sub_node = Node(output, parent=parent_node)

    sub_val = None
    if name == 'Name':
        sub_val = node.id
    if name == 'Constant':
        sub_val = str(node.value)
    if name == 'arg':
        sub_val = node.arg

    if sub_val is not None and len(sub_val) < 20:
        sub_sub_node = Node(sub_val, parent=sub_node)

    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    ast_visit(item, level=level + 1, parent_node=sub_node)
        elif isinstance(value, ast.AST):
            if not (isinstance(value, ast.Store) or isinstance(value, ast.Load)):
                ast_visit(value, level=level + 1, parent_node=sub_node)

    return sub_node


def get_all_keys(d):
    for key, value in d.items():

        if isinstance(value, str):
            yield value
        elif key != "children":
            yield key

        if isinstance(value, dict):
            yield from get_all_keys(value)
        elif isinstance(value, list):
            for item in value:
                yield from get_all_keys(item)


def save_ast_graph(ast_graph, output_name):

    output_file = '{}.dot'.format(output_name)

    anytree.exporter.dotexporter.UniqueDotExporter(ast_graph).to_dotfile(output_file)
    Source.from_file(output_file)

    render('dot', 'png', output_file)


def get_graph_words(ast_dict):
    words = []
    for x in get_all_keys(ast_dict):
        words.append(x)
    return words


def graph_to_dict(ast_graph):
    exporter = DictExporter()
    ast_dict = exporter.export(ast_graph)
    return ast_dict


def train_tokenizer(files):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()

    # files = ['output_test.txt']
    tokenizer.train(files, trainer)
    tokenizer.save("test.json")
    return tokenizer

def save_ast_words(ast_words, output_file='ast_graph_words.txt'):
    # TODO - fix to work with list of list (?)
    # Currently assuming its a 2D list (containing words for each function)
    with open(output_file, 'w', encoding="utf-8") as f:
        for ast_data in ast_words:
            f.write(" ".join(ast_data))
            f.write("\n")

def make_graph_tensors(ast_dict_list):

    def make_graph_tensor(ast_dict, parent_id=0):

        node_name = ast_dict['name']

        # Quick fix to pad tokens
        tokenized = tokenizer.encode(node_name).ids
        max_len = 5
        base = [0]*max_len  # TODO rename
        for i, token in enumerate(tokenized):
            if i == max_len:
                break
            base[i] = token


        graph_nodes.append(base)

        if len(graph_nodes) > 1:
            graph_edges.append([parent_id, len(graph_nodes)-1])

        parent_id = len(graph_nodes)-1

        if 'children' in list(ast_dict.keys()):
            child_nodes = ast_dict['children']
            for child_node in child_nodes:
                make_graph_tensor(child_node, parent_id)

    encoded_graph_list = []
    for ast_dict in ast_dict_list:
        graph_nodes = []
        graph_edges = []
        make_graph_tensor(ast_dict)

        # make graph
        edge_index = torch.tensor(graph_edges, dtype=torch.long)
        x = torch.tensor(graph_nodes, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index.t().contiguous())
        encoded_graph_list.append(data)

    return encoded_graph_list


dataset = load_dataset("code_search_net", "python")

functions = dataset['train'][:100]['func_code_string']

# functions = [test_func]
word_collection = []
output_file = 'ast_graph_words.txt'

ast_dicts = []
for i, function in tqdm(enumerate(functions)):
    ast_graph = ast_visit(ast.parse(function), parent_node=Node("Root"))
    ast_dict = graph_to_dict(ast_graph)
    ast_dicts.append(ast_dict)
    word_collection.append(get_graph_words(ast_dict))

    #save_ast_graph(ast_graph, "ast_graphs/output_test_{}".format(i))

tokenizer = train_tokenizer(files=[output_file])
save_ast_words(word_collection, output_file=output_file)

graph_tensors = make_graph_tensors(ast_dicts)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.in_channels = in_channels

        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)  # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)  # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    #if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


# model config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
out_channels = 1
num_features = graph_tensors[0].num_features  # 5
epochs = 100
model = GAE(GCNEncoder(num_features, out_channels))
model = model.to(device)

""""
data = graph_tensors[0]
data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
"""
# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import copy

graph_2 = copy.deepcopy(graph_tensors[50])
x = graph_2.x.to(device)
data = train_test_split_edges(graph_2)
train_pos_edge_index = data.train_pos_edge_index.to(device)

for epoch in tqdm(range(1, epochs + 1)):

    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)

    print(loss.item())
    loss.backward()
    optimizer.step()

auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

Z = model.encode(x, train_pos_edge_index)