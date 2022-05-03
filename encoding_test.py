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
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE
import torch.nn as nn
import ctree
from torch_geometric.transforms import RandomLinkSplit


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


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
        max_len = 1
        base = [0]*max_len  # TODO rename
        for i, token in enumerate(tokenized):
            if i == max_len:
                break
            base[i] = ((token/1579)-0.5)  # bounding between -1 and 1
            base[i] = token # bounding between -1 and 1
        graph_node_names.append(node_name)


        graph_nodes.append(base)

        if len(graph_nodes) > 1:
            graph_edges.append([parent_id, len(graph_nodes)-1])
            graph_edges.append([len(graph_nodes)-1,parent_id])

        parent_id = len(graph_nodes)-1

        if 'children' in list(ast_dict.keys()):
            child_nodes = ast_dict['children']
            for child_node in child_nodes:
                make_graph_tensor(child_node, parent_id)

    encoded_graph_list = []
    for ast_dict in ast_dict_list:
        graph_nodes = []
        graph_edges = []
        graph_node_names = []
        make_graph_tensor(ast_dict)

        # make graph
        edge_index = torch.tensor(graph_edges, dtype=torch.long)
        x = torch.tensor(graph_nodes, dtype=torch.float)

        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                              split_labels=True, add_negative_train_samples=False),
        ])

        data = Data(x=x, edge_index=edge_index.t().contiguous(),transform=transform)
        encoded_graph_list.append(data)

    return encoded_graph_list


dataset = load_dataset("code_search_net", "python")

functions = dataset['train'][:1000]['func_code_string']

# functions = [test_func]
word_collection = []
output_file = 'ast_graph_words.txt'

ast_dicts = []
for i, function in tqdm(enumerate(functions)):
    ast_graph = ast_visit(ast.parse(function), parent_node=Node("Root"))
    ast_dict = graph_to_dict(ast_graph)
    ast_dicts.append(ast_dict)
    word_collection.append(get_graph_words(ast_dict))

save_ast_graph(ast_graph, "ast_graphs/_output_test_{}".format(i))

tokenizer = train_tokenizer(files=[output_file])
save_ast_words(word_collection, output_file=output_file)

graph_tensors = make_graph_tensors(ast_dicts)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 3 * out_channels)
        self.conv1a = GCNConv(3 * out_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv1a(x, edge_index).relu()

        return self.conv2(x, edge_index)

class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

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
    #model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)

    a = model.decoder(z, edge_index=neg_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

"""
# model config

out_channels = 6
num_features = graph_tensors[0].num_features  # 5
model = GAE(GCNEncoder(num_features, out_channels))
#model = GAE(LinearEncoder(num_features, out_channels))

model = model.to(device)


# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import copy

graph_2 = copy.deepcopy(graph_tensors[50])
x = graph_2.x.to(device)
data = train_test_split_edges(graph_2)
train_pos_edge_index = data.train_pos_edge_index.to(device)

import warnings
warnings.filterwarnings("ignore")

loss_tracker = []
auc_tracker =[]
ap_tracker = []

training_size = 100
epochs = 100

for epoch in range(1, epochs + 1):

    for graph in graph_tensors[:training_size]:
        graph_2 = copy.deepcopy(graph)
        x = graph_2.x.to(device)
        data = train_test_split_edges(graph_2, test_ratio=0.2, val_ratio=0)
        #data = pyg_utils.train_test_split_edges(data)
        train_pos_edge_index = data.train_pos_edge_index.to(device)

        model.train()
        optimizer.zero_grad()

        z = model.encode(x, train_pos_edge_index)
        loss = model.recon_loss(z, train_pos_edge_index)
        #z = model.encode(graph_2.x, graph_2.edge_index)
        #loss = model.recon_loss(z, graph_2.pos_edge_label_index)

        loss_tracker.append(loss.item())
        #print(loss.item())
        loss.backward()
        optimizer.step()


        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        auc_tracker.append(auc)
        ap_tracker.append(ap)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, Loss{:.4f}'.format(epoch, np.mean(auc_tracker[-training_size:]),
                                                                          np.mean(ap_tracker[-training_size:]),
                                                                          np.mean(loss_tracker[-training_size:])))

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

plt.plot(running_mean(auc_tracker,training_size))
plt.plot(running_mean(ap_tracker, training_size))
plt.plot(running_mean(loss_tracker, training_size))
plt.show()
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler as RawNeighborSampler

import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'Cora'
path = './data'
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
#data = dataset[0]
data = graph_tensors[-1]

class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(),),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(NeighborSampler, self).sample(batch)


train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256,
                               shuffle=True, num_nodes=data.num_nodes)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(data.num_node_features, hidden_channels=64, num_layers=2)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
x, edge_index = data.x.to(device), data.edge_index.to(device)


def train():
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

        x[n_id] = 1  # such that only the adjs are used
        out = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test():
    model.eval()
    out = model.full_forward(x, edge_index).cpu()

    clf = LogisticRegression()
    clf.fit(out, data.y)

    val_acc = clf.score(out, data.y)
    test_acc = clf.score(out, data.y)

    return 1, 1


for epoch in range(1, 100):
    loss = train()
    #val_acc, test_acc = test()
    val_acc, test_acc = 1,1
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

with torch.no_grad():
    model.eval()
    out = model.full_forward(x, edge_index).cpu()

palette = {}

for n, y in enumerate(set(data.x.numpy().reshape(-1))):
    palette[y] = f'C{n}'

embd = umap.UMAP().fit_transform(out.cpu().numpy())


embd = umap.UMAP().fit_transform(out.cpu().numpy())

import matplotlib

items = len(embd.T[1])
colors = matplotlib.cm.rainbow(np.linspace(0, 1, items))
cs = [colors[i//items] for i in range(items)] #c

"""
plt.scatter(x=embd.T[0], y=embd.T[1], label=data.x.numpy().reshape(-1), color=cs)
plt.legend()
plt.show()
"""

"""
fig = plt.figure()
ax = fig.add_subplot(111)
for x,y,lab in zip(embd.T[0],embd.T[1],data.x.numpy().reshape(-1)):
        ax.scatter(x,y,text=lab)
plt.show()
"""


plt.scatter(embd.T[0],embd.T[1])

# zip joins x and y coordinates in pairs
for x,y,lab in zip(embd.T[0],embd.T[1],data.x.numpy().reshape(-1)):

    label = "{}".format(lab)

    plt.annotate(tokenizer.decode([int(float(label))]), # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()
#plt.figure(figsize=(10, 10))

sns.scatterplot(x=embd.T[0], y=embd.T[1], hue=data.x.numpy().reshape(-1), palette=palette)
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.savefig("umap_embd_sage.png", dpi=120)