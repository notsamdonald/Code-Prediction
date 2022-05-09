"""
A collation of support functions used to generate and process AST graphs based on input Python functions
"""

import ast

import anytree
import torch
from anytree import Node
from anytree.exporter import DictExporter
from graphviz import Source
from graphviz import render
from torch_geometric.data import Data
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    # Get all keys within the AST dictionary

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
    # Rendering and saving AST graph

    output_file = '{}.dot'.format(output_name)
    anytree.exporter.dotexporter.UniqueDotExporter(ast_graph).to_dotfile(output_file)
    Source.from_file(output_file)
    render('dot', 'png', output_file)


def get_graph_words(ast_dict):
    # Get all words within AST graph (nodes)

    words = []
    for x in get_all_keys(ast_dict):
        words.append(x)
    return words


def graph_to_dict(ast_graph):
    # Converting AST graph to dictionary

    exporter = DictExporter()
    ast_dict = exporter.export(ast_graph)
    return ast_dict


def save_ast_words(ast_words, output_file='ast_graph_words.txt'):
    # Save AST words/node names (unused currently)

    with open(output_file, 'w', encoding="utf-8") as f:
        for ast_data in ast_words:
            f.write(" ".join(ast_data))
            f.write("\n")


def make_graph_tensors(ast_dict_list):
    # Converts a given AST dictionary into a tensor by walking it and extracting edge/node information

    def make_graph_tensor(ast_dict, parent_id=0):

        node_name = ast_dict['name']
        if node_name in vocab:
            node_id = vocab[node_name]
        else:
            node_id = len(list(vocab.keys()))
            if node_id >= feature_len:
                node_id = 0  # UNK (Overflowing)
            else:
                vocab[node_name] = node_id
        base = [0] * feature_len
        base[node_id] = 1
        graph_node_names.append(node_name)
        graph_nodes.append(base)

        # TODO - Currently configured to Bi-directional encoding, make this dynamic
        # Extracting edge information
        if len(graph_nodes) > 1:
            graph_edges.append([parent_id, len(graph_nodes) - 1])
            graph_edges.append([len(graph_nodes) - 1, parent_id])

        parent_id = len(graph_nodes) - 1

        if 'children' in list(ast_dict.keys()):
            child_nodes = ast_dict['children']
            for child_node in child_nodes:
                make_graph_tensor(child_node, parent_id)

    vocab = {"UNK": 0}
    feature_len = 10000
    encoded_graph_list = []
    for ast_dict in ast_dict_list:
        graph_nodes = []
        graph_edges = []
        graph_node_names = []
        make_graph_tensor(ast_dict)

        # make graph
        edge_index = torch.tensor(graph_edges, dtype=torch.long)
        x = torch.tensor(graph_nodes, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index.t().contiguous())
        encoded_graph_list.append(data)

    return encoded_graph_list


def split_index(length, ratio=0.001):
    # Support funtion to split datasets
    return (int(length * (1 - ratio)))


def generate_AST_graph_tensor(data):
    # Master function calling subroutines to generate an AST graph

    ast_dicts = []
    print("Generating AST Dictionaries...")

    for i, function in tqdm(enumerate(data['func_code_string'])):
        try:
            ast_graph = ast_visit(ast.parse(function), parent_node=Node("Root"))
            ast_dict = graph_to_dict(ast_graph)
            ast_dicts.append(ast_dict)

        except SyntaxError:
            # Occasionally some code will not compile into an AST
            pass

    return make_graph_tensors(ast_dicts)
