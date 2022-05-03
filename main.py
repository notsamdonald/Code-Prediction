from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import re
from tqdm import tqdm
import pandas as pd
import ast
from collections import defaultdict

class AstGraphGenerator(object):

    def __init__(self, source):
        self.graph = defaultdict(lambda: [])
        self.source = source  # lines of the source code

    def __str__(self):
        return str(self.graph)

    def _getid(self, node):
        try:
            lineno = node.lineno - 1
            return "%s: %s" % (type(node), self.source[lineno].strip())

        except AttributeError:
            return type(node)

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)

            elif isinstance(value, ast.AST):
                node_source = self._getid(node)
                value_source = self._getid(value)
                self.graph[node_source].append(value_source)
                # self.graph[type(node)].append(type(value))
                self.visit(value)





df = pd. read_pickle('pkl_test.pkl')

dataset = load_dataset("code_search_net", "python")


test_data = dataset['train'][1]['func_code_string']
test_ast_gen = AstGraphGenerator(test_data)
test_ast_gen.generic_visit(ast.parse(test_data).body[0])
print("")


model_name = 'codebert-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)

data = [" ".join(x) for x in dataset['train'][-1000:]['func_code_tokens'] if "#" not in " ".join(x)]




inputs = tokenizer(" ".join(dataset['train']), truncation=False)

print("Done!")
print("Done!")

"""
token_list = []
for item in tqdm(dataset['train']):
    code = item['func_code_string']

    code = [x for x in code if "#" not in x]  # Removing comment tokens
    tokens = tokenizer(" ".join(code), padding='max_length', max_length=512)

    token_list.append(tokens)
"""