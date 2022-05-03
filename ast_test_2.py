import ast
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import anytree
def str_node(node):
    if isinstance(node, ast.AST):
        fields = [(name, str_node(val)) for name, val in ast.iter_fields(node) if name not in ('left', 'right')]
        rv = '%s(%s' % (node.__class__.__name__, ', '.join('%s=%s' % field for field in fields))
        return rv + ')'
    else:
        return repr(node)

def ast_visit(node, parent_node, level=0):
    name = type(node).__name__



    output = name
    print('  ' * level + str(level) + name)
    sub_node = Node(output, parent=parent_node)

    sub_val = None
    if name == 'Name':
        sub_val = node.id
    if name == 'Constant':
        sub_val = str(node.value)
    if name == 'arg':
        sub_val = node.arg

    if sub_val is not None:
        sub_sub_node = Node(sub_val, parent=sub_node)

    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    ast_visit(item, level=level+1, parent_node=sub_node)
        elif isinstance(value, ast.AST):
            if not (isinstance(value, ast.Store) or isinstance(value, ast.Load)):
                ast_visit(value, level=level+1, parent_node=sub_node)

    return sub_node
func = """class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.in_channels = in_channels

        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)  # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)  # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

"""

root = Node("Root")
a = ast_visit(ast.parse(func), parent_node=root)

for pre, fill, node in RenderTree(a):
    print("%s%s" % (pre, node.name))

anytree.exporter.dotexporter.UniqueDotExporter(a).to_dotfile('udo.dot')
from graphviz import Source
Source.from_file('udo.dot')
from graphviz import render
render('dot', 'png', 'udo.dot')