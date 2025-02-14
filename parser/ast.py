# parser/ast.py
# ASTNode class for representing nodes in the Abstract Syntax Tree (AST)
class ASTNode:
    def __init__(self, node_type, **kwargs):
        self.node_type = node_type
        self.children = kwargs

    def __repr__(self):
        return f"{self.node_type}({self.children})"
