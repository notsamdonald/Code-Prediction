"""
Generating and saving AST graph tensors for training of the GAE

Can also be used to generate rending of AST graphs.

Note:   the bidrectional/unidirectional edge encoding is determined on the
        make_graph_tensors function within Graph_generator.py
"""

import pickle

from datasets import load_dataset

from Graph_generator import split_index, generate_AST_graph_tensor

if __name__ == "__main__":

    dataset = load_dataset("code_search_net", "python")

    # Using the final 5% of data to train and validate the GAE
    ratio = 0.05
    train_len, val_len, test_len = len(dataset['train']), len(dataset['validation']), len(dataset['test'])
    train_split, val_split, test_split = split_index(train_len, ratio=ratio), \
                                         split_index(val_len, ratio=ratio), \
                                         split_index(test_len, ratio=ratio)

    train_data = dataset["train"][train_split:]
    val_data = dataset["validation"][val_split:]
    test_data = dataset["test"][test_split:]

    output_graph_tensors = []
    for split_data in [train_data, val_data]:
        output_graph_tensors.append(generate_AST_graph_tensor(split_data))

    # save_ast_graph(ast_graph, "ast_graphs/_output_test_{}".format(i)) - for generating rengered AST graphs

    # Saving GAE Graphs
    with open('GAE_graph_tensors_bidirectional.pkl', 'wb') as handle:
        pickle.dump(output_graph_tensors, handle)
