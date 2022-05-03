import ast

import anytree
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.transforms as T
from anytree import Node
from anytree.exporter import DictExporter
from datasets import load_dataset
from graphviz import Source
from graphviz import render
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch_geometric.data import Data
from torch_geometric.nn import GAE
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from tqdm import tqdm

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
        if node_name in vocab:
            node_id = vocab[node_name]
        else:
            node_id = len(list(vocab.keys()))
            vocab[node_name] = node_id

        base = [0] * 7000  # TODO this 7000 needs to be dynamic somehow, could use an unk token maybe
        base[node_id] = 1

        """
        # Quick fix to pad tokens
        # TODO - I think this is the problem, should be simple one hot per token here
        tokenized = tokenizer.encode(node_name).ids
        max_len = 10
        base = [0]*max_len  # TODO rename
        for i, token in enumerate(tokenized):
            if i == max_len:
                break
            base[i] = 2*((token/1579)-0.5)  # bounding between -1 and 1
            #base[i] = token # bounding between -1 and 1
        """

        graph_node_names.append(node_name)

        graph_nodes.append(base)

        if len(graph_nodes) > 1:
            graph_edges.append([parent_id, len(graph_nodes) - 1])
            # graph_edges.append([len(graph_nodes)-1,parent_id])

        parent_id = len(graph_nodes) - 1

        if 'children' in list(ast_dict.keys()):
            child_nodes = ast_dict['children']
            for child_node in child_nodes:
                make_graph_tensor(child_node, parent_id)

    vocab = {}
    vocab_id = 0

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

        data = Data(x=x, edge_index=edge_index.t().contiguous(), transform=transform)
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
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index).tanh()


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
    # if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    # model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)

    a = model.decoder(z, edge_index=neg_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


# model config

out_channels = 1
num_features = graph_tensors[0].num_features  # 5
model = GAE(GCNEncoder(num_features, out_channels))
# model = GAE(LinearEncoder(num_features, out_channels))

model = model.to(device)

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
import copy

graph_2 = copy.deepcopy(graph_tensors[50])
x = graph_2.x.to(device)
data = train_test_split_edges(graph_2)
train_pos_edge_index = data.train_pos_edge_index.to(device)

import warnings

warnings.filterwarnings("ignore")

loss_tracker = []
val_loss_tracker = []
auc_tracker = []
ap_tracker = []
precision_tracker = []
recall_tracker = []
acc_tracker = []
f1_tracker = []

TP_tracker = []
FP_tracker = []
TN_tracker = []
FN_tracker = []



training_size = 500
epochs = 100
val_size = 10
for epoch in range(1, epochs + 1):

    model.train()
    optimizer.zero_grad()
    for graph_n, graph in enumerate(graph_tensors[:training_size]):
        graph_2 = copy.deepcopy(graph)
        x = graph_2.x.to(device)
        data = train_test_split_edges(graph_2, test_ratio=1, val_ratio=0)
        # data = pyg_utils.train_test_split_edges(data)
        train_pos_edge_index = data.train_pos_edge_index.to(device)


        z = model.encode(x, data.test_pos_edge_index)
        loss = model.recon_loss(z, data.test_pos_edge_index, data.test_neg_edge_index)


        # z = model.encode(x, train_pos_edge_index)
        # val_loss = model.recon_loss(z, data.test_pos_edge_index)

        # z = model.encode(graph_2.x, graph_2.edge_index)
        # loss = model.recon_loss(z, graph_2.pos_edge_label_index)

        loss_tracker.append(loss.item())
        # val_loss_tracker.append(val_loss.item())

        # print(loss.item())
        loss.backward()
        if (graph_n+1) % 16 == 0:
            optimizer.step()


    TP, FN, TN, FP, n_pos, n_neg =  0,0,0,0,0,0

    for graph in graph_tensors[training_size:training_size + val_size]:
        model.eval()

        graph_2 = copy.deepcopy(graph)
        x = graph_2.x.to(device)
        data = train_test_split_edges(graph_2, test_ratio=1, val_ratio=0)
        # data = pyg_utils.train_test_split_edges(data)
        train_pos_edge_index = data.train_pos_edge_index.to(device)

        z = model.encode(x, data.test_pos_edge_index)
        val_loss = model.recon_loss(z, data.test_pos_edge_index)

        # z = model.encode(graph_2.x, graph_2.edge_index)
        # loss = model.recon_loss(z, graph_2.pos_edge_label_index)

        val_loss_tracker.append(val_loss.item())


        neg_preds = model.decode(z, data.test_neg_edge_index).detach().cpu().numpy() < 0.5
        pos_preds = model.decode(z, data.test_pos_edge_index).detach().cpu().numpy() > 0.5
        n_pos += len(pos_preds)
        n_neg += len(neg_preds)
        TP += np.sum(pos_preds)
        FN += (len(pos_preds) - np.sum(pos_preds))
        TN += np.sum(neg_preds)
        FP += (len(neg_preds) - np.sum(neg_preds))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*(precision*recall)/(precision+recall)
    acc = (TP+TN)/(TP+FP+TN+FN)


    precision_tracker.append(precision)
    recall_tracker.append(recall)
    acc_tracker.append(acc)
    f1_tracker.append(F1)
    TP_tracker.append(TP/(n_pos))
    FP_tracker.append(FP/(n_neg))
    TN_tracker.append(TN/(n_neg))
    FN_tracker.append(FN/(n_pos))

    """
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    auc_tracker.append(auc)
    ap_tracker.append(ap)
    """
    """
    print('Epoch:{:03d}, AUC:{:.4f}, AP:{:.4f}, Loss:{:.4f}, Val Loss:{:.4f}'.format(epoch,
                                                                                     np.mean(auc_tracker[-val_size:]),
                                                                                     np.mean(ap_tracker[-val_size:]),
                                                                                     np.mean(
                                                                                         loss_tracker[-training_size:]),
                                                                                     np.mean(
                                                                                         val_loss_tracker[-val_size:])))
    """
    print('Epoch:{:03d}, Loss:{:.4f}, Val Loss:{:.4f}'.format(epoch,np.mean(loss_tracker[-training_size:]),
                                                              np.mean(val_loss_tracker[-val_size:])))

    print("Acc:{:.3f}, Precision:{:.3f}, Recall:{:.3f}, F1{:.3f}".format(acc, precision, recall, F1))
    print("TP:{}, FN:{}, TN:{}, FP{}".format(TP, FN, TN, FP))
    print("\n")


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


plt.plot(np.linspace(0, epochs, len(running_mean(loss_tracker, training_size))),
         running_mean(loss_tracker, training_size))
plt.plot(np.linspace(0, epochs, len(running_mean(val_loss_tracker, val_size))),
         running_mean(val_loss_tracker, val_size))

plt.show()

plt.plot(acc_tracker, label="accuracy")
plt.plot(precision_tracker, label="precision")
plt.plot(recall_tracker, label="recall")
plt.plot(f1_tracker, label="F1")
plt.legend()
plt.show()


plt.plot(TP_tracker, label="TPR")
plt.plot(FN_tracker, label="FNR")
plt.plot(TN_tracker, label="TNR")
plt.plot(FP_tracker, label="FPR")
plt.legend()
plt.show()