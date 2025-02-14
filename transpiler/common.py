# transpiler/common.py
"""
This module contains common utility functions used by the code generation modules.
"""

def indent(code, level=1, indent_str="    "):
    """
    Returns the code with indentation applied to each line.
    
    Args:
        code (str): The code to be indented.
        level (int): Number of indentation levels.
        indent_str (str): The string used for one level of indentation.
    
    Returns:
        str: The indented code.
    """
    indented_code = "\n".join(indent_str * level + line if line.strip() != "" else line
                              for line in code.splitlines())
    return indented_code
