# interpreter.py
import sys
from parser.parser import parser
from transpiler.ann_generator import generate_ann_code
from transpiler.snn_generator import generate_snn_code

def find_network_def(ast):
    """
    Searches for the 'network_def' node in the AST.
    """
    if ast.node_type == "program":
        for stmt in ast.children.get("statements", []):
            if stmt.node_type == "network_def":
                return stmt
    if ast.node_type == "network_def":
        return ast
    return None

def interpret_neuroscript(code):
    """
    Interprets NeuroScript DSL code on the fly.
    Parses the code, generates Python code, and executes it in the current global context.
    
    Args:
      code (str): The NeuroScript DSL code.
    """
    parsed_ast = parser.parse(code)
    if not parsed_ast:
        print("Parsing failed.")
        return
    network_ast = find_network_def(parsed_ast)
    if not network_ast:
        print("No network definition found.")
        return
    net_type = network_ast.children.get("net_type")
    if net_type.upper() == "ANN":
        generated_code = generate_ann_code(network_ast)
    elif net_type.upper() == "SNN":
        generated_code = generate_snn_code(network_ast)
    else:
        print("Unknown network type.")
        return
    # Execute the generated code in the current global context.
    exec(generated_code, globals())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        code = sys.argv[1]
        interpret_neuroscript(code)
    else:
        print("Please provide NeuroScript code as an argument.")
