#!/usr/bin/env python
# main.py
import argparse
import sys
import os
from parser.parser import parser
from transpiler.ann_generator import generate_ann_code
from transpiler.snn_generator import generate_snn_code

def find_network_def(ast):
    """
    Searches for the 'network_def' node in the AST.
    If the AST is a 'program' node, it iterates over its statements.
    """
    if ast.node_type == "program":
        for stmt in ast.children.get("statements", []):
            if stmt.node_type == "network_def":
                return stmt
    if ast.node_type == "network_def":
        return ast
    return None

def main():
    # Define command-line arguments
    arg_parser = argparse.ArgumentParser(
        description="NeuroScript DSL CLI. Provide a path to a DSL (.ns) file or a code string, and optionally an output Python file name."
    )
    arg_parser.add_argument(
        "code",
        help="Path to the NeuroScript DSL source file or a DSL code string."
    )
    arg_parser.add_argument(
        "-s", "--string",
        action="store_true",
        help="Indicates that the provided code argument is a DSL code string rather than a file path."
    )
    arg_parser.add_argument(
        "-o", "--output",
        help="Path to the output Python file (default: <network_name>_generated.py)",
        default=None
    )
    args = arg_parser.parse_args()

    # Read the DSL source code (from file or directly from string)
    if args.string:
        data = args.code
    else:
        try:
            with open(args.code, "r", encoding="utf-8") as f:
                data = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.code}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)

    # Parse the DSL code
    parsed_ast = parser.parse(data)
    if not parsed_ast:
        print("Parsing failed.")
        sys.exit(1)

    # Find the network definition in the AST
    network_ast = find_network_def(parsed_ast)
    if not network_ast:
        print("Error: No network definition ('network_def') found in the code.")
        sys.exit(1)

    # Determine network type: ANN or SNN
    net_type = network_ast.children.get("net_type")
    if not net_type:
        print("Error: Network type ('net_type') is not specified in the network definition.")
        sys.exit(1)

    # Generate Python code based on the network type
    if net_type.upper() == "ANN":
        generated_code = generate_ann_code(network_ast)
    elif net_type.upper() == "SNN":
        generated_code = generate_snn_code(network_ast)
    else:
        print(f"Error: Unknown network type '{net_type}'. Allowed types are 'ANN' or 'SNN'.")
        sys.exit(1)

    # Determine output file name
    network_name = network_ast.children.get("name", "GeneratedNetwork")
    if args.output:
        output_filename = args.output
    else:
        output_filename = f"{network_name}_generated.py"

    # Write the generated code to the output file
    try:
        with open(output_filename, "w", encoding="utf-8") as out_file:
            out_file.write(generated_code)
        print(f"Generated Python code successfully written to '{output_filename}'.")
    except Exception as e:
        print(f"Error writing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
